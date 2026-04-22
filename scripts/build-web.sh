#!/usr/bin/env bash
# Builds the docseg-web wasm bundle into web/pkg/.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v wasm-pack >/dev/null; then
  echo "wasm-pack not found. Install with:" >&2
  echo "  cargo install wasm-pack" >&2
  exit 1
fi

wasm-pack build crates/docseg-web --target web --release --out-dir ../../web/pkg
