#!/usr/bin/env python3
"""Convert a CRAFT PyTorch .pth checkpoint to ONNX.

One-time setup: downloads CRAFT's canonical model class from
clovaai/CRAFT-pytorch into a caller-provided working directory, loads
the .pth weights, and exports to ONNX with dynamic H/W axes. If
onnxruntime and a verification image are available, also runs a forward
pass and renders a heatmap overlay so a human can sanity-check the
conversion visually.

This script is invoked from scripts/fetch-model.sh inside a throw-away
virtualenv. Don't add dependencies beyond torch / onnx / onnxruntime /
pillow / numpy without updating the shell script's pip install line.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.onnx
from PIL import Image


CRAFT_REPO = "https://github.com/clovaai/CRAFT-pytorch.git"


def clone_craft(dest_dir: str) -> None:
    """Shallow-clone clovaai/CRAFT-pytorch so we can import its model class."""
    subprocess.check_call(
        ["git", "clone", "--depth=1", CRAFT_REPO, dest_dir],
        stderr=subprocess.STDOUT,
    )


def patch_craft_source(craft_repo_dir: str) -> None:
    """Strip `torchvision.models.vgg.model_urls` references (removed in torchvision 0.13)."""
    vgg_file = os.path.join(craft_repo_dir, "basenet", "vgg16_bn.py")
    with open(vgg_file, "r", encoding="utf-8") as f:
        src = f.read()
    replacements = [
        ("from torchvision.models.vgg import model_urls\n", ""),
        (
            "        model_urls['vgg16_bn'] = model_urls['vgg16_bn']"
            ".replace('https://', 'http://')\n",
            "",
        ),
    ]
    changed = False
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
            changed = True
    if changed:
        with open(vgg_file, "w", encoding="utf-8") as f:
            f.write(src)


def load_craft(pth_path: str, craft_repo_dir: str):
    """Instantiate CRAFT and load a .pth checkpoint (DataParallel prefix stripped)."""
    patch_craft_source(craft_repo_dir)
    sys.path.insert(0, craft_repo_dir)
    from craft import CRAFT  # type: ignore

    # pretrained=False skips torchvision's VGG weight download — the .pth we
    # load already contains the full CRAFT weights (VGG backbone included).
    model = CRAFT(pretrained=False, freeze=False)
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    stripped = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(stripped)
    model.eval()
    return model


class CraftWrapper(torch.nn.Module):
    """CRAFT returns (y, feature). ONNX export only needs the heatmap tensor `y`."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.inner(x)
        return y


def export_onnx(model: torch.nn.Module, onnx_path: str) -> None:
    """Export CRAFT to ONNX with dynamic spatial axes."""
    wrapped = CraftWrapper(model)
    wrapped.eval()
    dummy = torch.zeros(1, 3, 640, 640)
    # Pin the legacy TorchScript-based exporter — it's mature, stable, and
    # produces ONNX graphs that wonnx's op set handles well. The new dynamo
    # exporter is not worth the extra risk for a frozen checkpoint.
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "H", 3: "W"},
            "output": {1: "H2", 2: "W2"},
        },
        opset_version=11,
        do_constant_folding=True,
        dynamo=False,
    )


def _preprocess_for_verify(image_path: str, target_long: int = 1280):
    """Letterbox + ImageNet-normalize — mirrors the Rust preprocess in Task 5."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = target_long / max(w, h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    nw = ((nw + 31) // 32) * 32
    nh = ((nh + 31) // 32) * 32
    img_r = img.resize((nw, nh), Image.BILINEAR)
    arr = np.asarray(img_r, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None].astype(np.float32)
    return img_r, arr


def _extract_region_channel(out: np.ndarray) -> np.ndarray:
    """Return the region heatmap regardless of channels-first vs channels-last layout."""
    if out.ndim == 4 and out.shape[-1] == 2:  # [1, H, W, 2]
        return out[0, :, :, 0]
    if out.ndim == 4 and out.shape[1] == 2:  # [1, 2, H, W]
        return out[0, 0, :, :]
    raise RuntimeError(f"unexpected CRAFT output shape {out.shape}")


def run_verification(onnx_path: str, image_path: str, overlay_path: str) -> None:
    """Run the ONNX on `image_path` and save a red-tinted region-heatmap overlay."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        print("  (onnxruntime not installed; skipping visual verification)")
        return

    img_r, tensor = _preprocess_for_verify(image_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    out = sess.run([out_name], {in_name: tensor})[0]
    print(f"  ONNX output shape: {out.shape} ({out.dtype})")

    region = _extract_region_channel(out)
    print(
        f"  Region score: min={region.min():.3f} max={region.max():.3f} "
        f"mean={region.mean():.3f} frac>0.4={(region > 0.4).mean():.3f}"
    )

    nw, nh = img_r.size
    region_u8 = (np.clip(region, 0, 1) * 255).astype(np.uint8)
    mask = np.asarray(Image.fromarray(region_u8).resize((nw, nh), Image.BILINEAR))
    base = np.asarray(img_r).astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.55
    blended = base * (1.0 - alpha) + red * alpha
    Image.fromarray(blended.astype(np.uint8)).save(overlay_path)
    print(f"  Saved verification overlay: {overlay_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pth", required=True, help="CRAFT .pth checkpoint path")
    ap.add_argument("--out-onnx", required=True, help="output ONNX path")
    ap.add_argument("--verify-image", default=None, help="optional image for visual sanity check")
    ap.add_argument("--verify-overlay", default=None, help="output PNG for the sanity-check overlay")
    ap.add_argument(
        "--tmpdir",
        default=None,
        help="working directory (script clones CRAFT-pytorch into <tmpdir>/CRAFT-pytorch)",
    )
    args = ap.parse_args()

    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="craft-convert-")
    craft_dir = os.path.join(tmpdir, "CRAFT-pytorch")
    if not os.path.isdir(craft_dir):
        clone_craft(craft_dir)

    model = load_craft(args.pth, craft_dir)
    export_onnx(model, args.out_onnx)
    print(f"Exported ONNX: {args.out_onnx}")

    if args.verify_image and args.verify_overlay:
        run_verification(args.out_onnx, args.verify_image, args.verify_overlay)

    return 0


if __name__ == "__main__":
    sys.exit(main())
