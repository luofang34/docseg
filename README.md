# docseg

Rust → WASM → WebGPU demo that segments a page image of dense handwritten
script (the reference input is a Yi / classical Chinese manuscript) into
per-character oriented bounding boxes, with an interactive tuning UI,
auto/manual reading-order overlay, a glyph-ribbon sidebar, and one-click
zip export.

## What it does

- Fetches [CRAFT](https://arxiv.org/abs/1904.01941) (`craft_mlt_25k`) as
  an ONNX blob (exported with fixed 640×640 input, nearest-mode Resize —
  see `scripts/convert-model.py` for why).
- Loads CRAFT into
  [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) on
  the WebGPU execution provider.
- Rust-WASM runs preprocess (letterbox + ImageNet normalize), postprocess
  (thresholded region ∧ ¬affinity → 4-connected components → axis-aligned
  or rotated quads → area/aspect/pad-ghost filters), reading-order
  inference (column-cluster + vertical sort for Yi/CJK, or horizontal
  line-cluster for Latin), canvas painting with numbered labels and
  reading-order arrows, crop PNG encode, and zip export.
- Interactive UI: region/affinity/erosion/min-area sliders trigger
  postprocess re-runs (no re-inference, <30ms); direction dropdown
  switches reading order in place; "Draw order" lets you click boxes in
  sequence to override the auto-computed reading order; glyph ribbon on
  the right shows every character in order and hover-highlights its box
  on the main canvas.

## Requirements

- Browser: Chrome 113+, Safari 17.4+, or Firefox Nightly with WebGPU
  enabled. `navigator.gpu` missing → blocking banner, no WebGL2 fallback.
- Rust: stable 1.74+ (`rust-toolchain.toml` pins this plus the
  `wasm32-unknown-unknown` target).
- Tools: `wasm-pack` (`cargo install wasm-pack`), `python3` + `git` for
  the one-time model conversion, `curl` and `shasum`.

## Build + run locally

```bash
# 1. Fetch + convert CRAFT weights. First run downloads the PyTorch .pth
#    (~80 MB) into a throw-away venv and exports ONNX to models/. ~2 min.
./scripts/fetch-model.sh

# 2. Pin the printed hash on first run:
#    echo "<sha>" > models/EXPECTED_SHA256

# 3. Build the wasm bundle into web/pkg/.
./scripts/build-web.sh

# 4. Copy the model and sample image into web/ for serving.
mkdir -p web/models
cp models/craft_mlt_25k.onnx web/models/
cp ../OCR_Yi/test_images/test_case1.png web/   # or any PNG/JPEG

# 5. Serve.
python3 -m http.server 8787 --directory web
#    open http://localhost:8787/ in a WebGPU-capable browser.
```

All of `web/test_case1.png`, `web/models/`, and `web/pkg/` are gitignored —
the dev workflow rebuilds them locally.

## Local CI

```bash
./scripts/ci-local.sh
```

Runs `fmt --check → clippy -D warnings → test --all-targets → doc
-D missing_docs → wasm check → wasm build`. This is the same gate the
GitHub Actions workflow runs.

## Deploying to GitHub Pages

The repo ships a `.github/workflows/pages.yml` workflow. It builds the
wasm bundle + static assets and publishes them to GitHub Pages. The 83 MB
CRAFT ONNX is **not** bundled into the deploy artifact; the browser
fetches it from a URL you set once.

### 1. Host the ONNX as a GitHub Release asset

GH Release assets have effectively unmetered bandwidth, are CDN-backed,
and serve `Access-Control-Allow-Origin: *` — enough for this demo, and
nothing extra to sign up for.

Option A: one-click via the supplied workflow.

- Go to **Actions → "Publish CRAFT ONNX to GitHub Release" → Run workflow**.
  It sets up a throw-away Python venv, runs `scripts/convert-model.py`,
  and attaches `craft_mlt_25k.onnx` to a release tagged `model-v1`
  (creating the release if it doesn't exist). Takes ~3 minutes.

Option B: manual upload.

```bash
./scripts/fetch-model.sh   # produces models/craft_mlt_25k.onnx
gh release create model-v1 models/craft_mlt_25k.onnx \
    --title "CRAFT ONNX (craft_mlt_25k, 640×640)" \
    --notes "Converted from clovaai/CRAFT-pytorch .pth. SHA-256 pinned in models/EXPECTED_SHA256."
```

Either way, the browser-facing URL will be:

```
https://github.com/<your-user>/<your-repo>/releases/download/model-v1/craft_mlt_25k.onnx
```

### 2. Point the Pages workflow at it

- In your GitHub repo settings, add a **repository variable** named
  `DOCSEG_MODEL_URL` with the URL above.
- In **Settings → Pages**, set "Source" to "GitHub Actions".

Push to `main`; the workflow will build and deploy.

### Alternative hosting

- **Hugging Face Hub.** If you already have an HF account, `hf-cli
  upload <user>/craft_mlt_25k_onnx_640 models/craft_mlt_25k.onnx` also
  works and is CDN-backed. `DOCSEG_MODEL_URL` becomes
  `https://huggingface.co/<user>/craft_mlt_25k_onnx_640/resolve/main/craft_mlt_25k.onnx`.
- **`gh-pages` branch bundle.** Remove the `rm -rf deploy/models` line
  in `pages.yml` and `cp` the ONNX into `deploy/models/` before upload.
  Simple, but every deploy re-commits 83 MB.
- **GH LFS.** Not recommended — 1 GB/month free bandwidth ceiling,
  each demo visit is 83 MB.

## Crates

- `docseg-core` — pure Rust: preprocess, wonnx session (native), reading
  order, postprocess, geometry, error types. `cargo test -p docseg-core`.
  Tests that need the model file are `#[ignore]`d — opt in with
  `-- --ignored`.
- `docseg-web` — `wasm-bindgen` adapter: image decode, canvas overlay,
  hit-test, cropPng, exportZip, reading-order API. Exposes `DocsegApp`
  (`new`, `preprocessImage`, `postprocess`, `computeReadingOrder`,
  `setCustomOrder`, `paint`, `hit`, `cropPng`, `exportZip`).

## Known limitations

- **Resolution.** The ONNX is exported at 640×640 fixed input to fit
  wgpu's default `max_storage_buffer_binding_size` (128 MiB). On a
  979×1450 manuscript this letterboxes to 432×640 — enough to detect
  ~100 characters on the reference page but small characters on heavily
  distressed paper can be missed. Pre-approved escalation paths: tile-
  based inference at 1280, pivot to `burn + wgpu` with raised limits.
- **Upsampling mode.** `Resize(mode=linear)` is unsupported by
  `wonnx` 0.5 and — at the time of writing — also by `onnxruntime-web`'s
  WebGPU EP on some CRAFT-shaped graphs. We post-process the exported
  ONNX to rewrite those Resize nodes to `mode=nearest`; region scores
  degrade by ≲1% on this distribution. See `scripts/convert-model.py`.
- **Model training distribution.** CRAFT is trained on mostly-horizontal
  MLT text. Its affinity channel marks inter-character space effectively
  on horizontal scripts but only partially on vertical CJK stacks; the
  segmentation quality comes from the region channel + connected
  components with an axis-aligned rectangle fit.

## License

Code: MIT OR Apache-2.0 (see each crate's `Cargo.toml`).
Model weights (CRAFT by Clova AI): [MIT](https://github.com/clovaai/CRAFT-pytorch/blob/master/LICENSE).
