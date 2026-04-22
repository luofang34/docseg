#!/usr/bin/env bash
# One-time: fetch the CRAFT PyTorch .pth checkpoint, convert it to ONNX,
# and pin its SHA-256. Runtime (browser) never uses Python — this script
# only exists because no anonymously-fetchable craft_mlt_25k.onnx mirror
# is public.
#
# Everything except the final ONNX and a verification-overlay PNG lives
# in a temp dir and is deleted on exit (successful or not).
#
# Usage:
#   scripts/fetch-model.sh
#
# Prereqs: curl, python3 (>=3.9), git, shasum.

set -euo pipefail

cd "$(dirname "$0")/.."

PTH_URL="https://huggingface.co/boomb0om/CRAFT-text-detector/resolve/main/craft_mlt_25k.pth"
ONNX_PATH="models/craft_mlt_25k.onnx"
OVERLAY_PATH="models/verification_overlay.png"
VERIFY_IMAGE_CANDIDATES=(
  "../OCR_Yi/test_images/test_case1.png"
  "test_case1.png"
  "web/test_case1.png"
)
EXPECTED_FILE="models/EXPECTED_SHA256"

if [[ -f "$ONNX_PATH" ]]; then
  echo "ONNX already exists at $ONNX_PATH (skipping conversion)"
else
  if ! command -v python3 >/dev/null; then
    echo "python3 is required to convert the .pth to .onnx" >&2
    exit 1
  fi

  TMP=$(mktemp -d -t docseg-fetch-model-XXXXXX)
  trap 'rm -rf "$TMP"' EXIT

  echo "==> mktemp -d: $TMP"

  echo "==> python3 -m venv $TMP/venv"
  python3 -m venv "$TMP/venv"
  # Pin lightweight CPU-only deps into the venv — never touches user site-packages.
  "$TMP/venv/bin/pip" install --quiet --upgrade pip
  "$TMP/venv/bin/pip" install --quiet \
    "torch>=2.1,<3" "torchvision>=0.16,<1" \
    "onnx>=1.14" "onnxscript>=0.1" "onnxruntime>=1.16" \
    "onnxsim>=0.4" \
    "pillow>=10" "numpy>=1.26"

  echo "==> curl $PTH_URL"
  curl -fL --retry 3 -o "$TMP/craft_mlt_25k.pth" "$PTH_URL"

  PTH_SHA=$(shasum -a 256 "$TMP/craft_mlt_25k.pth" | awk '{print $1}')
  echo "    .pth SHA-256: $PTH_SHA"

  VERIFY_ARGS=()
  for cand in "${VERIFY_IMAGE_CANDIDATES[@]}"; do
    if [[ -f "$cand" ]]; then
      VERIFY_ARGS=(--verify-image "$cand" --verify-overlay "$OVERLAY_PATH")
      echo "    verification image: $cand"
      break
    fi
  done
  if [[ ${#VERIFY_ARGS[@]} -eq 0 ]]; then
    echo "    (no verification image found; skipping overlay)"
  fi

  echo "==> convert-model.py"
  "$TMP/venv/bin/python" scripts/convert-model.py \
    --pth "$TMP/craft_mlt_25k.pth" \
    --out-onnx "$ONNX_PATH" \
    --tmpdir "$TMP" \
    "${VERIFY_ARGS[@]}"

  # trap cleans up $TMP (venv, .pth, CRAFT-pytorch clone) on exit.
fi

ACTUAL=$(shasum -a 256 "$ONNX_PATH" | awk '{print $1}')
EXPECTED=$(tr -d '[:space:]' < "$EXPECTED_FILE" || true)

if [[ -z "$EXPECTED" ]]; then
  echo "No expected hash pinned. Computed hash:"
  echo "  $ACTUAL"
  echo "To pin, write that value to $EXPECTED_FILE and commit."
  exit 0
fi

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "SHA-256 mismatch for $ONNX_PATH" >&2
  echo "  expected: $EXPECTED" >&2
  echo "  actual:   $ACTUAL" >&2
  exit 1
fi

echo "OK $ONNX_PATH ($ACTUAL)"
