#!/usr/bin/env bash
# Runs every gate required before pushing. Mirrors the rules in
# docs/superpowers/specs/2026-04-21-rust-wasm-wgpu-character-segmentation-design.md §9.
set -euo pipefail
export RUST_BACKTRACE=1

cd "$(dirname "$0")/.."

echo "==> fmt --check"
cargo fmt --all -- --check

echo "==> clippy -D warnings"
cargo clippy --all-targets --workspace -- -D warnings

echo "==> test --workspace --all-targets"
cargo test --workspace --all-targets

echo "==> doc -D missing_docs"
RUSTDOCFLAGS="-D missing_docs -D rustdoc::broken_intra_doc_links" \
  cargo doc --workspace --no-deps

echo "==> check -p docseg-core --target wasm32-unknown-unknown"
cargo check -p docseg-core --target wasm32-unknown-unknown --all-targets

echo "==> build --release --target wasm32-unknown-unknown (docseg-web if present)"
if [ -f crates/docseg-web/Cargo.toml ]; then
  cargo build --release --target wasm32-unknown-unknown -p docseg-web
else
  echo "   (docseg-web not yet added; skipping wasm build)"
fi

echo "OK"
