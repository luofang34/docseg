# `docseg` — Rust/WASM/WebGPU Character Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-only Rust→WASM demo that takes a handwritten-manuscript image and returns per-character oriented bounding boxes, with CRAFT ONNX inference running on WebGPU via `wonnx`, and exportable per-glyph PNG crops.

**Architecture:** Cargo workspace with two crates — `docseg-core` (pure Rust, native-testable: preprocess, model session, postprocess) and `docseg-web` (wasm-bindgen adapter: image decode, canvas overlay, export). Plain-JS bootstrap drives the WASM entry point. Model file (CRAFT ONNX, ~20 MB) is fetched at runtime and cached by the browser. Postprocess uses CRAFT's **region score only** (no affinity merging) so each connected blob = one glyph.

**Tech Stack:** Rust 1.74+, `wonnx` (ONNX on `wgpu`), `image`, `imageproc`, `geo`, `thiserror`, `tracing`, `wasm-bindgen`, `web-sys`, `js-sys`, `wasm-bindgen-futures`, `zip`, plain ES modules.

**Design reference:** `docs/superpowers/specs/2026-04-21-rust-wasm-wgpu-character-segmentation-design.md`.

**Execution guardrails (from user global rules):**

- Every committed task MUST leave `cargo fmt --check`, `cargo clippy --all-targets --workspace -- -D warnings`, and `cargo test --workspace --all-targets` passing (tests that require the model file are `#[ignore]`d and opt-in).
- No `unwrap` / `expect` / `panic` outside tests. No `mod.rs`. No generic `utils.rs` modules. Public items have doc comments (`missing_docs = "deny"` at the workspace level). No `eprintln!`/`println!` — use `tracing`.
- Commits happen at the end of every task (and sometimes mid-task if a step says so). Each commit compiles and passes the lint/test gates for the workspace state as of that commit.
- Work runs inside `/Users/fangluo/Desktop/docseg` on branch `main`. No pushes unless the user requests them.

---

## File Structure

Files created (relative to repo root, `/Users/fangluo/Desktop/docseg`):

```
Cargo.toml                                   # workspace + lints table
clippy.toml                                  # disallowed_types
rust-toolchain.toml                          # pins stable + wasm target
.gitignore                                   # ignores /target, /web/pkg, /models/*.onnx
scripts/fetch-model.sh                       # downloads CRAFT ONNX + hash check
scripts/ci-local.sh                          # runs full local CI gate
scripts/build-web.sh                         # wasm-pack build wrapper
crates/docseg-core/Cargo.toml
crates/docseg-core/src/lib.rs                # crate doc, re-exports only
crates/docseg-core/src/error.rs              # CoreError enum
crates/docseg-core/src/preprocess.rs         # letterbox + normalize
crates/docseg-core/src/preprocess/tests.rs   # unit tests
crates/docseg-core/src/geometry.rs           # Point/Quad/min-area-rect
crates/docseg-core/src/geometry/tests.rs     # unit tests
crates/docseg-core/src/postprocess.rs        # heatmap → components → CharBox
crates/docseg-core/src/postprocess/tests.rs  # unit tests
crates/docseg-core/src/model.rs              # wonnx session wrapper
crates/docseg-core/tests/forward_pass.rs     # CRAFT ONNX verification (ignored unless model present)
crates/docseg-core/tests/end_to_end.rs       # full pipeline on fixture (ignored unless model present)
crates/docseg-core/tests/fixtures/synthetic_glyphs.png   # generated in Task 10
crates/docseg-web/Cargo.toml
crates/docseg-web/src/lib.rs                 # crate doc, re-exports only
crates/docseg-web/src/entry.rs               # #[wasm_bindgen] init/run/export
crates/docseg-web/src/canvas.rs              # image decode from bytes
crates/docseg-web/src/render.rs              # paint overlays, hit-test
crates/docseg-web/src/export.rs              # zip of crops + boxes.json
web/index.html
web/main.js
web/style.css
README.md
```

Files NOT committed:

- `models/craft_mlt_25k.onnx` — fetched by `scripts/fetch-model.sh`, ignored.
- `web/pkg/` — wasm-pack output, ignored.
- `target/` — ignored.

File-size discipline (per user rules): every `.rs` stays ≤ 500 LOC, every fn ≤ 80 LOC, `lib.rs` ≤ 100 LOC with only a crate `//!` and module declarations / re-exports.

---

## Task 1: Workspace scaffolding, lints, and local CI script

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/Cargo.toml`
- Create: `/Users/fangluo/Desktop/docseg/clippy.toml`
- Create: `/Users/fangluo/Desktop/docseg/rust-toolchain.toml`
- Create: `/Users/fangluo/Desktop/docseg/.gitignore`
- Create: `/Users/fangluo/Desktop/docseg/scripts/ci-local.sh`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Write `.gitignore`**

```
/target
/web/pkg
/models/*.onnx
/models/*.bin
.DS_Store
```

- [ ] **Step 2: Write `rust-toolchain.toml`**

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["wasm32-unknown-unknown"]
```

- [ ] **Step 3: Write workspace `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = ["crates/docseg-core", "crates/docseg-web"]

[workspace.package]
edition = "2021"
rust-version = "1.74"
license = "MIT OR Apache-2.0"
publish = false

[workspace.dependencies]
thiserror = "1"
tracing = { version = "0.1", default-features = false, features = ["std"] }
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
imageproc = { version = "0.25", default-features = false }
geo = "0.28"
bytemuck = { version = "1", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[workspace.lints.rust]
unsafe_code = "forbid"
missing_docs = "deny"

[workspace.lints.clippy]
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
let_underscore_must_use = "deny"
let_underscore_future = "deny"
await_holding_lock = "deny"
disallowed_types = "deny"
```

- [ ] **Step 4: Write `clippy.toml`**

```toml
disallowed-types = [
    { path = "anyhow::Error", reason = "use thiserror in library crates" },
]
```

- [ ] **Step 5: Write `crates/docseg-core/Cargo.toml`**

```toml
[package]
name = "docseg-core"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true
description = "Per-character document segmentation (preprocess, CRAFT inference, postprocess). No web deps."

[lib]
crate-type = ["rlib"]

[dependencies]
thiserror.workspace = true
tracing.workspace = true
image.workspace = true
imageproc.workspace = true
geo.workspace = true
bytemuck.workspace = true
serde = { workspace = true, optional = true }

[features]
default = ["serde"]
serde = ["dep:serde"]

[lints]
workspace = true
```

- [ ] **Step 6: Write `crates/docseg-core/src/lib.rs`**

```rust
//! `docseg-core` — per-character document segmentation.
//!
//! Takes a decoded page image, returns oriented quadrilaterals around each
//! detected glyph. Target deployment is a browser via WASM, but this crate
//! is pure Rust with no web dependencies so it is natively testable.

#![deny(missing_docs)]

pub mod error;

pub use error::CoreError;
```

- [ ] **Step 7: Write minimal `error.rs` (just so the crate compiles)**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/error.rs`

```rust
//! Typed errors surfaced by the core crate.

use thiserror::Error;

/// Errors produced anywhere in the `docseg-core` pipeline.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// Image bytes could not be decoded into a supported pixel format.
    #[error("image decode failed")]
    Decode(#[source] image::ImageError),
}
```

- [ ] **Step 8: Write `scripts/ci-local.sh`**

```bash
#!/usr/bin/env bash
# Runs every gate required before pushing. Mirrors the rules in
# docs/superpowers/specs/2026-04-21-rust-wasm-wgpu-character-segmentation-design.md §9.
set -euo pipefail

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

echo "==> build --release --target wasm32-unknown-unknown (docseg-web if present)"
if cargo metadata --no-deps --format-version 1 | grep -q '"docseg-web"'; then
  cargo build --release --target wasm32-unknown-unknown -p docseg-web
else
  echo "   (docseg-web not yet added; skipping wasm build)"
fi

echo "OK"
```

Then `chmod +x scripts/ci-local.sh`.

- [ ] **Step 9: Run the scaffolding to prove it compiles and lints**

```bash
cd /Users/fangluo/Desktop/docseg
cargo fmt --all
./scripts/ci-local.sh
```

Expected: `OK` at the end. `cargo test` reports "0 passed" for `docseg-core`. wasm build is skipped with the "not yet added" message.

- [ ] **Step 10: Commit**

```bash
git add Cargo.toml clippy.toml rust-toolchain.toml .gitignore scripts/ci-local.sh crates/docseg-core
git commit -m "$(cat <<'EOF'
chore: scaffold Cargo workspace with lint gates and local CI script

Adds docseg-core crate skeleton, workspace-wide deny on unsafe/unwrap/
expect/panic/missing_docs/disallowed_types, and scripts/ci-local.sh
that runs fmt/clippy/test/doc/wasm-build.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Flesh out `CoreError` with all variants

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/error.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/error/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Write a failing test for each variant's `Display` output**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/error/tests.rs`

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::CoreError;

#[test]
fn decode_error_displays_context() {
    let io = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad png");
    let err = CoreError::Decode(image::ImageError::IoError(io));
    assert!(format!("{err}").contains("image decode"));
}

#[test]
fn model_fetch_error_carries_url() {
    let err = CoreError::ModelFetch {
        url: "https://example.com/model.onnx".into(),
        source: Box::new(std::io::Error::other("timeout")),
    };
    assert!(format!("{err}").contains("model fetch"));
    assert!(format!("{err}").contains("https://example.com/model.onnx"));
}

#[test]
fn model_load_error_carries_op_hint() {
    let err = CoreError::ModelLoad {
        hint: "op Resize".into(),
        source: Box::new(std::io::Error::other("unsupported")),
    };
    assert!(format!("{err}").contains("model load"));
    assert!(format!("{err}").contains("op Resize"));
}

#[test]
fn inference_error_displays_context() {
    let err = CoreError::Inference {
        source: Box::new(std::io::Error::other("oom")),
    };
    assert!(format!("{err}").contains("inference"));
}

#[test]
fn preprocess_error_carries_dims() {
    let err = CoreError::Preprocess {
        width: 0,
        height: 10,
        reason: "zero width".into(),
    };
    let s = format!("{err}");
    assert!(s.contains("preprocess"));
    assert!(s.contains("0"));
    assert!(s.contains("zero width"));
}

#[test]
fn postprocess_error_carries_reason() {
    let err = CoreError::Postprocess {
        reason: "heatmap shape mismatch".into(),
    };
    assert!(format!("{err}").contains("postprocess"));
    assert!(format!("{err}").contains("heatmap shape mismatch"));
}
```

- [ ] **Step 2: Register the test module in `error.rs`**

At the bottom of `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/error.rs` add:

```rust

#[cfg(test)]
mod tests;
```

- [ ] **Step 3: Run the tests — expect failure**

```bash
cargo test -p docseg-core --lib
```

Expected: compilation errors — `CoreError::ModelFetch` etc. don't exist yet.

- [ ] **Step 4: Implement the full `CoreError` enum**

Replace the body of `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/error.rs` with:

```rust
//! Typed errors surfaced by the core crate.

use thiserror::Error;

/// Errors produced anywhere in the `docseg-core` pipeline.
///
/// Variants carry every piece of context a caller needs to produce a useful
/// user-facing message (URLs, image dimensions, op hints) — no string-concat
/// at the error site.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// Image bytes could not be decoded into a supported pixel format.
    #[error("image decode failed")]
    Decode(#[source] image::ImageError),

    /// HTTP / network failure while fetching the model bundle.
    #[error("model fetch failed from {url}")]
    ModelFetch {
        /// URL that was attempted.
        url: String,
        /// Underlying transport error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Model bundle parsed but could not be initialized (unsupported op,
    /// malformed graph, etc.).
    #[error("model load failed ({hint})")]
    ModelLoad {
        /// Short diagnostic hint, e.g. the offending op name.
        hint: String,
        /// Underlying loader error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Forward pass failed (OOM, adapter lost, shape mismatch).
    #[error("inference failed")]
    Inference {
        /// Underlying runtime error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Preprocess rejected the input (zero-size, absurd aspect, etc.).
    #[error("preprocess failed ({width}x{height}): {reason}")]
    Preprocess {
        /// Offending image width in pixels.
        width: u32,
        /// Offending image height in pixels.
        height: u32,
        /// Human-readable reason.
        reason: String,
    },

    /// Postprocess could not interpret the inference output.
    #[error("postprocess failed: {reason}")]
    Postprocess {
        /// Human-readable reason.
        reason: String,
    },
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 5: Run tests — expect pass**

```bash
cargo test -p docseg-core --lib
```

Expected: 6 passed.

- [ ] **Step 6: Run the full local CI gate**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): typed CoreError with context-carrying variants

Every variant carries the context needed for its Display message (URL,
op hint, dims, reason) so callers can render useful errors without
string-concat at the error site.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Model fetch script + checked-in hash pinning scaffold

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/scripts/fetch-model.sh`
- Create: `/Users/fangluo/Desktop/docseg/models/.gitkeep`
- Create: `/Users/fangluo/Desktop/docseg/models/EXPECTED_SHA256`

- [ ] **Step 1: Create `models/` and `.gitkeep`**

```bash
mkdir -p /Users/fangluo/Desktop/docseg/models
touch /Users/fangluo/Desktop/docseg/models/.gitkeep
```

- [ ] **Step 2: Create an empty `EXPECTED_SHA256` file**

`/Users/fangluo/Desktop/docseg/models/EXPECTED_SHA256` (empty string — will be populated after the first verified fetch):

```
```

- [ ] **Step 3: Write `scripts/fetch-model.sh`**

```bash
#!/usr/bin/env bash
# Fetches the CRAFT ONNX checkpoint used by docseg.
#
# Usage:
#   scripts/fetch-model.sh
#
# Behavior:
#   - Skips download if models/craft_mlt_25k.onnx already exists.
#   - Verifies SHA-256 against models/EXPECTED_SHA256 if non-empty.
#   - On first run with empty EXPECTED_SHA256, prints the computed hash
#     and exits 0 so the user can commit the hash to pin it.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_URL="https://huggingface.co/Bingsu/craft-onnx/resolve/main/craft_mlt_25k.onnx"
MODEL_PATH="models/craft_mlt_25k.onnx"
EXPECTED_FILE="models/EXPECTED_SHA256"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Downloading $MODEL_URL ..."
  curl -fL --retry 3 -o "$MODEL_PATH" "$MODEL_URL"
fi

ACTUAL=$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')
EXPECTED=$(tr -d '[:space:]' < "$EXPECTED_FILE" || true)

if [[ -z "$EXPECTED" ]]; then
  echo "No expected hash pinned. Computed hash:"
  echo "  $ACTUAL"
  echo "To pin, write that value to $EXPECTED_FILE and commit."
  exit 0
fi

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "SHA-256 mismatch for $MODEL_PATH" >&2
  echo "  expected: $EXPECTED" >&2
  echo "  actual:   $ACTUAL" >&2
  exit 1
fi

echo "OK $MODEL_PATH ($ACTUAL)"
```

Then `chmod +x scripts/fetch-model.sh`.

- [ ] **Step 4: Run the fetch script locally**

```bash
./scripts/fetch-model.sh
```

Expected: downloads ~20 MB to `models/craft_mlt_25k.onnx`, prints "No expected hash pinned." + the computed SHA-256.

- [ ] **Step 5: Pin the observed hash**

Copy the SHA-256 printed in Step 4 into `models/EXPECTED_SHA256`:

```bash
echo "<paste-the-hash-here>" > models/EXPECTED_SHA256
./scripts/fetch-model.sh
```

Expected: second run prints `OK models/craft_mlt_25k.onnx (<hash>)`.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch-model.sh models/.gitkeep models/EXPECTED_SHA256
git commit -m "$(cat <<'EOF'
chore: fetch-model.sh for pinned CRAFT ONNX download

Downloads craft_mlt_25k.onnx from Bingsu/craft-onnx on HuggingFace and
verifies SHA-256 against models/EXPECTED_SHA256. The .onnx file itself
is gitignored; the hash is committed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CRAFT forward-pass verification (the gate)

**Goal of this task:** run a single forward pass through the CRAFT ONNX on `wonnx` (WebGPU/Vulkan/Metal via `wgpu`) and assert the output has the expected shape. If `wonnx` rejects an op, record the deviation and pivot to `burn` before any further work.

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/tests/forward_pass.rs`

- [ ] **Step 1: Add `wonnx` + test-only deps to `docseg-core/Cargo.toml`**

Append to `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml`:

```toml
[dependencies.wonnx]
version = "0.5"
default-features = false

[dev-dependencies]
pollster = "0.3"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
```

- [ ] **Step 2: Write the verification test**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/tests/forward_pass.rs`

```rust
//! Gate test: CRAFT ONNX must load and run a single forward pass via wonnx.
//!
//! Requires `models/craft_mlt_25k.onnx`. Run with:
//!
//!   cargo test -p docseg-core --test forward_pass -- --ignored
//!
//! This test is the decision point for the inference backend. If wonnx
//! rejects the model, the deviation path (burn) is pre-approved.

#![allow(clippy::expect_used, clippy::panic)]

use std::collections::HashMap;
use std::path::PathBuf;
use wonnx::utils::InputTensor;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/craft_mlt_25k.onnx")
}

#[test]
#[ignore = "requires models/craft_mlt_25k.onnx (run scripts/fetch-model.sh first)"]
fn craft_forward_pass_produces_two_heatmaps() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,wonnx=warn")
        .try_init();

    let path = model_path();
    assert!(path.exists(), "model file not found at {}", path.display());

    let session = pollster::block_on(wonnx::Session::from_path(&path))
        .expect("CRAFT ONNX failed to load in wonnx. If the op name is logged, \
                 record it and pivot to the burn+wgpu fallback per the spec.");

    // CRAFT accepts any HxW that is a multiple of 32. Use 640x640 for this smoke test.
    let (h, w) = (640_usize, 640_usize);
    let input: Vec<f32> = vec![0.0; 1 * 3 * h * w];
    let mut inputs: HashMap<String, InputTensor> = HashMap::new();
    inputs.insert("input".to_string(), input.as_slice().into());

    let outputs = pollster::block_on(session.run(&inputs))
        .expect("forward pass failed");

    // CRAFT output is a single tensor shaped [1, H/2, W/2, 2] — channels-last
    // for the Clova export — or [1, 2, H/2, W/2] for the ONNX-optimized export.
    // Either way, total element count = 1 * 2 * (H/2) * (W/2).
    let expected_elems = 2 * (h / 2) * (w / 2);

    let output_name = outputs.keys().next().cloned().expect("no output");
    let tensor: Vec<f32> = (&outputs[&output_name])
        .try_into()
        .expect("output should be f32");
    assert_eq!(
        tensor.len(),
        expected_elems,
        "CRAFT output must contain 2 heatmaps of size H/2 x W/2"
    );
}
```

- [ ] **Step 3: Ensure `models/craft_mlt_25k.onnx` is present**

```bash
ls -la models/craft_mlt_25k.onnx
```

If missing, run `./scripts/fetch-model.sh`.

- [ ] **Step 4: Run the test — expect PASS**

```bash
cargo test -p docseg-core --test forward_pass -- --ignored --nocapture
```

Expected: test passes. The first run downloads wonnx (takes a minute).

**Contingency if wonnx fails (unsupported op):**

1. Note the failing op in the test output.
2. Replace the `[dependencies.wonnx]` block with:

   ```toml
   [dependencies.burn]
   version = "0.14"
   default-features = false
   features = ["wgpu", "ndarray"]

   [dependencies.burn-import]
   version = "0.14"
   ```

3. Rewrite the test to load via `burn-import::onnx::ModelGen` and run with the `Wgpu` backend. Keep the same assertion on output element count.
4. Add a short commit `chore(core): pivot inference backend from wonnx to burn (op X unsupported)` and update the design spec's §4 to reflect the chosen backend.

Do NOT skip to later tasks if this step fails — subsequent tasks assume a working inference backend.

- [ ] **Step 5: Add a `cargo test` guardrail so `./scripts/ci-local.sh` doesn't break on missing model**

Nothing to change: the test is `#[ignore]`d. Default `cargo test` skips it. CI runs pass without the model.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
test(core): CRAFT forward-pass gate verifies wonnx can run the model

Adds an ignored integration test that loads craft_mlt_25k.onnx via wonnx
and asserts the forward pass produces a [1, 2, H/2, W/2] output. Run with
`cargo test --test forward_pass -- --ignored` after fetching the model.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Preprocess — letterbox + normalize

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/preprocess.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/preprocess/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

**Contract:** takes an `image::DynamicImage`, letterboxes so the long side = `target_long` (default 1280) and both dims are multiples of 32, normalizes with ImageNet mean/std in NCHW `f32`, returns a `PreprocessOutput { tensor, padded_size, scale, pad_offset }` describing the transform so postprocess can map coordinates back.

- [ ] **Step 1: Write the failing tests**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/preprocess/tests.rs`

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{preprocess, PreprocessOptions};
use image::{DynamicImage, RgbImage};

fn solid(color: [u8; 3], w: u32, h: u32) -> DynamicImage {
    let img = RgbImage::from_fn(w, h, |_, _| image::Rgb(color));
    DynamicImage::ImageRgb8(img)
}

#[test]
fn letterbox_pads_to_multiple_of_32() {
    // 1000 x 1500 landscape image, target long-side 1280 -> scale 1280/1500.
    let img = solid([128, 128, 128], 1000, 1500);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    let (pw, ph) = out.padded_size;
    assert_eq!(pw % 32, 0, "padded width {pw} not multiple of 32");
    assert_eq!(ph % 32, 0, "padded height {ph} not multiple of 32");
    assert!(ph <= 1280 + 31);
    // long side == 1280 after scaling
    assert_eq!(ph.max(pw), 1280);
}

#[test]
fn scale_and_offset_preserve_aspect() {
    let img = solid([255, 255, 255], 500, 1000);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    // long side is height -> scaled height = 1280, scaled width = 640.
    // Padded width is next multiple of 32 ≥ 640 = 640.
    assert_eq!(out.scale, 1280.0 / 1000.0);
    // zero pad offset on height, zero on width (both already multiples).
    assert_eq!(out.pad_offset, (0, 0));
}

#[test]
fn nchw_tensor_length_matches_padded_size() {
    let img = solid([0, 0, 0], 256, 256);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    let (pw, ph) = out.padded_size;
    assert_eq!(out.tensor.len(), 3 * (pw as usize) * (ph as usize));
}

#[test]
fn normalization_matches_imagenet_constants() {
    // A pure-white image becomes (1.0 - mean)/std per channel.
    let img = solid([255, 255, 255], 32, 32);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    let (pw, ph) = out.padded_size;
    let plane = (pw as usize) * (ph as usize);
    let r0 = out.tensor[0];
    let g0 = out.tensor[plane];
    let b0 = out.tensor[2 * plane];
    let expected_r = (1.0 - 0.485) / 0.229;
    let expected_g = (1.0 - 0.456) / 0.224;
    let expected_b = (1.0 - 0.406) / 0.225;
    assert!((r0 - expected_r).abs() < 1e-4, "R got {r0}, want {expected_r}");
    assert!((g0 - expected_g).abs() < 1e-4);
    assert!((b0 - expected_b).abs() < 1e-4);
}

#[test]
fn zero_size_image_is_rejected() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(0, 10));
    let err = preprocess(&img, PreprocessOptions::default()).expect_err("should fail");
    let s = format!("{err}");
    assert!(s.contains("preprocess"), "got {s}");
}
```

- [ ] **Step 2: Create the module file stub so tests compile-fail for the right reason**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/preprocess.rs`

```rust
//! Letterbox + ImageNet-normalize an `image::DynamicImage` into the NCHW
//! f32 tensor CRAFT expects, plus the inverse transform for mapping model-
//! space coordinates back to the original image.

// deliberately minimal — tests below will drive the real API.

#[cfg(test)]
mod tests;
```

Register in `lib.rs`:

```rust
//! `docseg-core` — per-character document segmentation.
//!
//! Takes a decoded page image, returns oriented quadrilaterals around each
//! detected glyph. Target deployment is a browser via WASM, but this crate
//! is pure Rust with no web dependencies so it is natively testable.

#![deny(missing_docs)]

pub mod error;
pub mod preprocess;

pub use error::CoreError;
pub use preprocess::{preprocess, PreprocessOptions, PreprocessOutput};
```

Run: `cargo test -p docseg-core --lib` — expect compile error: `preprocess` / `PreprocessOptions` / `PreprocessOutput` don't exist.

- [ ] **Step 3: Implement `preprocess`**

Replace `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/preprocess.rs`:

```rust
//! Letterbox + ImageNet-normalize an `image::DynamicImage` into the NCHW
//! f32 tensor CRAFT expects, plus the inverse transform for mapping model-
//! space coordinates back to the original image.

use image::{imageops::FilterType, DynamicImage, GenericImageView};

use crate::CoreError;

/// ImageNet mean (RGB order).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet std (RGB order).
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Knobs for the preprocess stage.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessOptions {
    /// Longest side of the scaled (pre-pad) image, in pixels.
    pub target_long_side: u32,
    /// Both padded dimensions are rounded up to a multiple of this value.
    pub size_multiple: u32,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            target_long_side: 1280,
            size_multiple: 32,
        }
    }
}

/// Everything downstream stages need to consume the preprocess output.
#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    /// NCHW f32 tensor, length `3 * padded_w * padded_h`.
    pub tensor: Vec<f32>,
    /// Padded model-input dimensions `(width, height)`.
    pub padded_size: (u32, u32),
    /// Multiply an original-image coordinate by `scale` to reach model space.
    pub scale: f32,
    /// Padding added before the scaled image in model space `(x, y)`.
    /// Subtract, then divide by `scale`, to invert.
    pub pad_offset: (u32, u32),
}

/// Run the full preprocess pipeline.
pub fn preprocess(
    img: &DynamicImage,
    opts: PreprocessOptions,
) -> Result<PreprocessOutput, CoreError> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(CoreError::Preprocess {
            width: w,
            height: h,
            reason: "zero-size image".into(),
        });
    }

    let long_side = w.max(h) as f32;
    let target = opts.target_long_side as f32;
    let scale = target / long_side;
    let scaled_w = ((w as f32) * scale).round() as u32;
    let scaled_h = ((h as f32) * scale).round() as u32;

    let m = opts.size_multiple;
    let padded_w = scaled_w.div_ceil(m) * m;
    let padded_h = scaled_h.div_ceil(m) * m;

    let resized = img
        .resize_exact(scaled_w, scaled_h, FilterType::Triangle)
        .to_rgb8();

    let plane = (padded_w as usize) * (padded_h as usize);
    let mut tensor = vec![0.0_f32; 3 * plane];

    for y in 0..scaled_h {
        for x in 0..scaled_w {
            let px = resized.get_pixel(x, y);
            let idx = (y as usize) * (padded_w as usize) + (x as usize);
            let r = (px[0] as f32) / 255.0;
            let g = (px[1] as f32) / 255.0;
            let b = (px[2] as f32) / 255.0;
            tensor[idx] = (r - MEAN[0]) / STD[0];
            tensor[plane + idx] = (g - MEAN[1]) / STD[1];
            tensor[2 * plane + idx] = (b - MEAN[2]) / STD[2];
        }
    }
    // Pad region stays zero — i.e. (0 - mean) / std per channel, which is
    // already what the tensor is initialized to if we pre-fill with the
    // normalized zero. Re-fill to match:
    let zero_r = (0.0 - MEAN[0]) / STD[0];
    let zero_g = (0.0 - MEAN[1]) / STD[1];
    let zero_b = (0.0 - MEAN[2]) / STD[2];
    for y in 0..padded_h {
        for x in 0..padded_w {
            if x < scaled_w && y < scaled_h {
                continue;
            }
            let idx = (y as usize) * (padded_w as usize) + (x as usize);
            tensor[idx] = zero_r;
            tensor[plane + idx] = zero_g;
            tensor[2 * plane + idx] = zero_b;
        }
    }

    Ok(PreprocessOutput {
        tensor,
        padded_size: (padded_w, padded_h),
        scale,
        pad_offset: (0, 0),
    })
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test -p docseg-core --lib preprocess
```

Expected: 5 passed.

- [ ] **Step 5: Run full local CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): preprocess module — letterbox + ImageNet normalize

Takes a DynamicImage, scales so the long side matches target_long_side
(default 1280), pads both dims to a multiple of 32, and emits an NCHW
f32 tensor normalized with ImageNet constants. Returns the inverse
transform (scale, pad offset) so postprocess can map boxes back to the
original image space.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Geometry primitives (Point, Quad, min-area rect)

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/geometry.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/geometry/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/geometry/tests.rs`

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{min_area_quad, Point, Quad};

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn axis_aligned_square_returns_same_square() {
    let pts = vec![
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        Point::new(10.0, 10.0),
        Point::new(0.0, 10.0),
        Point::new(5.0, 5.0),
    ];
    let q = min_area_quad(&pts).expect("quad");
    // All four corners of a 10x10 square should appear (in some order).
    let mut xs: Vec<f32> = q.points.iter().map(|p| p.x).collect();
    let mut ys: Vec<f32> = q.points.iter().map(|p| p.y).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(approx_eq(xs[0], 0.0, 1e-3));
    assert!(approx_eq(xs[3], 10.0, 1e-3));
    assert!(approx_eq(ys[0], 0.0, 1e-3));
    assert!(approx_eq(ys[3], 10.0, 1e-3));
}

#[test]
fn rotated_square_returns_rotated_quad_with_same_area() {
    // 45° rotated square, side sqrt(2), corners at (1,0)(0,1)(-1,0)(0,-1).
    let pts = vec![
        Point::new(1.0, 0.0),
        Point::new(0.0, 1.0),
        Point::new(-1.0, 0.0),
        Point::new(0.0, -1.0),
    ];
    let q = min_area_quad(&pts).expect("quad");
    assert!(approx_eq(q.area(), 2.0, 1e-3), "area {}", q.area());
}

#[test]
fn quad_area_for_unit_square_is_one() {
    let q = Quad::new([
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 1.0),
    ]);
    assert!(approx_eq(q.area(), 1.0, 1e-6));
}

#[test]
fn empty_input_returns_none() {
    let pts: Vec<Point> = vec![];
    assert!(min_area_quad(&pts).is_none());
}
```

- [ ] **Step 2: Create module stub + register in `lib.rs`**

`/Users/fangluo/Desktop/docseg/crates/docseg-core/src/geometry.rs`:

```rust
//! Planar geometry primitives used by postprocess.

#[cfg(test)]
mod tests;
```

Update `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`:

```rust
//! `docseg-core` — per-character document segmentation.
//!
//! Takes a decoded page image, returns oriented quadrilaterals around each
//! detected glyph. Target deployment is a browser via WASM, but this crate
//! is pure Rust with no web dependencies so it is natively testable.

#![deny(missing_docs)]

pub mod error;
pub mod geometry;
pub mod preprocess;

pub use error::CoreError;
pub use geometry::{min_area_quad, Point, Quad};
pub use preprocess::{preprocess, PreprocessOptions, PreprocessOutput};
```

Run: `cargo test -p docseg-core --lib geometry` — expect compile failure.

- [ ] **Step 3: Implement the primitives (uses `geo::algorithm::minimum_rotated_rect`)**

Replace `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/geometry.rs`:

```rust
//! Planar geometry primitives used by postprocess.
//!
//! We lean on `geo`'s `MinimumRotatedRect` for the rotating-calipers
//! algorithm rather than rolling our own — it is ~40 KB of extra WASM and
//! saves a known-tricky implementation.

use geo::algorithm::minimum_rotated_rect::MinimumRotatedRect;
use geo::{Coord, MultiPoint, Polygon};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 2D point with `f32` coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point {
    /// x coordinate.
    pub x: f32,
    /// y coordinate.
    pub y: f32,
}

impl Point {
    /// Construct a point from components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Oriented quadrilateral. Points are stored in an arbitrary but consistent
/// order — downstream consumers may normalize to CW from top-left if needed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Quad {
    /// Four corner points.
    pub points: [Point; 4],
}

impl Quad {
    /// Construct a quad from four points.
    #[must_use]
    pub const fn new(points: [Point; 4]) -> Self {
        Self { points }
    }

    /// Shoelace-formula area of the quad.
    #[must_use]
    pub fn area(&self) -> f32 {
        let p = &self.points;
        let sum = p[0].x * p[1].y - p[1].x * p[0].y
            + p[1].x * p[2].y - p[2].x * p[1].y
            + p[2].x * p[3].y - p[3].x * p[2].y
            + p[3].x * p[0].y - p[0].x * p[3].y;
        (sum * 0.5).abs()
    }
}

/// Compute the minimum-area rotated rectangle enclosing `points`.
///
/// Returns `None` when the input is empty or degenerate (all collinear).
#[must_use]
pub fn min_area_quad(points: &[Point]) -> Option<Quad> {
    if points.is_empty() {
        return None;
    }
    let multipoint: MultiPoint<f64> = points
        .iter()
        .map(|p| Coord {
            x: f64::from(p.x),
            y: f64::from(p.y),
        })
        .collect::<Vec<_>>()
        .into();
    let poly: Polygon<f64> = multipoint.minimum_rotated_rect()?;
    let ring = poly.exterior();
    if ring.0.len() < 5 {
        // geo returns a closed ring (first == last), so 5 entries = 4 distinct corners.
        return None;
    }
    let mut corners = [Point::new(0.0, 0.0); 4];
    for (i, c) in ring.0.iter().take(4).enumerate() {
        corners[i] = Point::new(c.x as f32, c.y as f32);
    }
    Some(Quad::new(corners))
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test -p docseg-core --lib geometry
```

Expected: 4 passed.

- [ ] **Step 5: Run full local CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): Point, Quad, and min-area-rect via geo crate

Postprocess needs an oriented bounding quad per connected component; the
`geo` crate provides a rotating-calipers `MinimumRotatedRect` that we
wrap in a small Rust-native API to avoid a custom implementation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Postprocess — heatmap threshold + connected components

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

**Contract:** given a flat `f32` region-score map of size `heatmap_w × heatmap_h` and a threshold, return a `Vec<Vec<(u32, u32)>>` — one point list per connected component (4-connectivity).

- [ ] **Step 1: Write the failing test**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess/tests.rs`

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{components_from_heatmap, PostprocessOptions};

#[test]
fn two_separate_blobs_produce_two_components() {
    // 8x4 heatmap with two 2x2 blobs separated by a gap.
    // Indices: (x,y).
    //
    // . . . . . . . .
    // . X X . . X X .
    // . X X . . X X .
    // . . . . . . . .
    let mut map = vec![0.0_f32; 8 * 4];
    let set = |m: &mut Vec<f32>, x: usize, y: usize| m[y * 8 + x] = 1.0;
    for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2), (5, 1), (6, 1), (5, 2), (6, 2)] {
        set(&mut map, x, y);
    }
    let comps =
        components_from_heatmap(&map, 8, 4, PostprocessOptions::default());
    assert_eq!(comps.len(), 2, "expected 2 components, got {}", comps.len());
    for c in &comps {
        assert_eq!(c.len(), 4, "each blob has 4 pixels");
    }
}

#[test]
fn subthreshold_noise_is_excluded() {
    let map = vec![0.1_f32; 16 * 16]; // below default threshold
    let comps =
        components_from_heatmap(&map, 16, 16, PostprocessOptions::default());
    assert!(comps.is_empty());
}

#[test]
fn one_diagonal_touch_does_not_merge_components_under_4_connectivity() {
    // Two 1-pixel blobs at (1,1) and (2,2) — 8-connected neighbours, but
    // 4-connected they stay separate.
    let mut map = vec![0.0_f32; 4 * 4];
    map[1 * 4 + 1] = 1.0;
    map[2 * 4 + 2] = 1.0;
    let opts = PostprocessOptions {
        region_threshold: 0.5,
        ..Default::default()
    };
    let comps = components_from_heatmap(&map, 4, 4, opts);
    assert_eq!(comps.len(), 2);
}
```

- [ ] **Step 2: Create stub + register**

`/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess.rs`:

```rust
//! Convert CRAFT region-score heatmaps into per-character oriented boxes.

#[cfg(test)]
mod tests;
```

Update `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`:

```rust
//! `docseg-core` — per-character document segmentation.
//!
//! Takes a decoded page image, returns oriented quadrilaterals around each
//! detected glyph. Target deployment is a browser via WASM, but this crate
//! is pure Rust with no web dependencies so it is natively testable.

#![deny(missing_docs)]

pub mod error;
pub mod geometry;
pub mod postprocess;
pub mod preprocess;

pub use error::CoreError;
pub use geometry::{min_area_quad, Point, Quad};
pub use postprocess::{
    components_from_heatmap, PostprocessOptions,
};
pub use preprocess::{preprocess, PreprocessOptions, PreprocessOutput};
```

Run: `cargo test -p docseg-core --lib postprocess` — expect compile failure.

- [ ] **Step 3: Implement `components_from_heatmap` using `imageproc::region_labelling::connected_components`**

Replace `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess.rs`:

```rust
//! Convert CRAFT region-score heatmaps into per-character oriented boxes.

use image::{GrayImage, Luma};
use imageproc::region_labelling::{connected_components, Connectivity};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Tuning knobs for the heatmap → boxes conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PostprocessOptions {
    /// Pixels with region score ≥ this are "inside a character."
    pub region_threshold: f32,
    /// Drop components with fewer pixels than this (removes red seal noise).
    pub min_component_area_px: u32,
    /// Drop components whose aspect ratio (long/short of the oriented rect)
    /// exceeds this — filters streaks from gutter ink.
    pub max_aspect_ratio: f32,
}

impl Default for PostprocessOptions {
    fn default() -> Self {
        Self {
            region_threshold: 0.4,
            min_component_area_px: 12,
            max_aspect_ratio: 8.0,
        }
    }
}

/// Binarize the heatmap with `opts.region_threshold` and return each
/// 4-connected component as a point list in heatmap coordinates.
///
/// Components are unfiltered; callers apply area/aspect filters later.
#[must_use]
pub fn components_from_heatmap(
    map: &[f32],
    width: u32,
    height: u32,
    opts: PostprocessOptions,
) -> Vec<Vec<(u32, u32)>> {
    if width == 0 || height == 0 || map.len() != (width as usize) * (height as usize) {
        return Vec::new();
    }
    let mut binary = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y as usize) * (width as usize) + (x as usize);
            let v = if map[idx] >= opts.region_threshold { 255 } else { 0 };
            binary.put_pixel(x, y, Luma([v]));
        }
    }
    let labels = connected_components(&binary, Connectivity::Four, Luma([0]));
    let mut buckets: Vec<Vec<(u32, u32)>> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let label = labels.get_pixel(x, y).0[0];
            if label == 0 {
                continue;
            }
            let idx = (label as usize).saturating_sub(1);
            while buckets.len() <= idx {
                buckets.push(Vec::new());
            }
            buckets[idx].push((x, y));
        }
    }
    buckets.retain(|v| !v.is_empty());
    buckets
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test -p docseg-core --lib postprocess
```

Expected: 3 passed.

- [ ] **Step 5: Run full local CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): postprocess connected-components from CRAFT region heatmap

Binarizes the region score at PostprocessOptions::region_threshold and
extracts 4-connected components via imageproc. Returns per-component
pixel lists in heatmap coordinates; area/aspect filters land in the
next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Postprocess — components → `CharBox` with filters

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

**Contract:** takes the component lists + heatmap + the `PreprocessOutput` transform, maps each component's pixels back to original-image coordinates (accounting for heatmap being H/2 × W/2 of the padded input), fits a min-area quad, drops by area/aspect, attaches a score (the max region-score within the component), returns `Vec<CharBox>`.

- [ ] **Step 1: Add failing tests for `CharBox` extraction**

Append to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess/tests.rs`:

```rust

use super::{charboxes_from_heatmap, CharBox};
use crate::preprocess::PreprocessOutput;

fn fake_preproc(padded_w: u32, padded_h: u32, scale: f32) -> PreprocessOutput {
    PreprocessOutput {
        tensor: Vec::new(),
        padded_size: (padded_w, padded_h),
        scale,
        pad_offset: (0, 0),
    }
}

#[test]
fn charbox_maps_back_to_original_image_coords() {
    // Padded input 64x64, heatmap 32x32. Scale 0.5 (original 128x128).
    // Put a 4x4 blob starting at (8, 8) in heatmap space. In padded-image
    // space that's (16, 16)..(24, 24). In original space (divide by 0.5),
    // that's (32, 32)..(48, 48). Box should cover ~that range.
    let mut map = vec![0.0_f32; 32 * 32];
    for y in 8..12 {
        for x in 8..12 {
            map[y * 32 + x] = 0.9;
        }
    }
    let boxes = charboxes_from_heatmap(
        &map,
        32,
        32,
        &fake_preproc(64, 64, 0.5),
        PostprocessOptions {
            region_threshold: 0.5,
            min_component_area_px: 1,
            max_aspect_ratio: 8.0,
        },
    );
    assert_eq!(boxes.len(), 1);
    let b = &boxes[0];
    let xs: Vec<f32> = b.quad.points.iter().map(|p| p.x).collect();
    let ys: Vec<f32> = b.quad.points.iter().map(|p| p.y).collect();
    let xmin = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let xmax = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ymin = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let ymax = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!((xmin - 32.0).abs() < 4.0, "xmin={xmin}");
    assert!((xmax - 48.0).abs() < 4.0, "xmax={xmax}");
    assert!((ymin - 32.0).abs() < 4.0);
    assert!((ymax - 48.0).abs() < 4.0);
    assert!(b.score > 0.5);
}

#[test]
fn charbox_area_filter_drops_small_blobs() {
    // 1-pixel blob shouldn't survive min_component_area_px = 5.
    let mut map = vec![0.0_f32; 32 * 32];
    map[10 * 32 + 10] = 1.0;
    let boxes = charboxes_from_heatmap(
        &map,
        32,
        32,
        &fake_preproc(64, 64, 1.0),
        PostprocessOptions {
            region_threshold: 0.5,
            min_component_area_px: 5,
            max_aspect_ratio: 8.0,
        },
    );
    assert!(boxes.is_empty());
}
```

- [ ] **Step 2: Add `CharBox` + `charboxes_from_heatmap` (tests compile-fail until both exist)**

Append to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess.rs`:

```rust

use crate::geometry::{min_area_quad, Point, Quad};
use crate::preprocess::PreprocessOutput;

/// One detected glyph: its oriented box in original-image coordinates and a
/// confidence score.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CharBox {
    /// Oriented quadrilateral in original-image pixel coordinates.
    pub quad: Quad,
    /// Max region-score within the component (0..1).
    pub score: f32,
}

/// Full heatmap → `CharBox` pipeline with area/aspect filtering and
/// coordinate mapping back to the original image.
#[must_use]
pub fn charboxes_from_heatmap(
    map: &[f32],
    heatmap_w: u32,
    heatmap_h: u32,
    preproc: &PreprocessOutput,
    opts: PostprocessOptions,
) -> Vec<CharBox> {
    let comps = components_from_heatmap(map, heatmap_w, heatmap_h, opts);
    let (padded_w, padded_h) = preproc.padded_size;
    // Scale from heatmap space → padded-input space.
    let map_to_padded_x = f32::from(padded_w as u16) / (heatmap_w as f32);
    let map_to_padded_y = f32::from(padded_h as u16) / (heatmap_h as f32);
    let inv_scale = 1.0 / preproc.scale;
    let (pad_ox, pad_oy) = preproc.pad_offset;

    let mut out = Vec::with_capacity(comps.len());
    for comp in comps {
        if (comp.len() as u32) < opts.min_component_area_px {
            continue;
        }
        let mut pts = Vec::with_capacity(comp.len());
        let mut score = 0.0_f32;
        for (x, y) in &comp {
            let idx = (*y as usize) * (heatmap_w as usize) + (*x as usize);
            score = score.max(map[idx]);
            // Each heatmap pixel covers a (map_to_padded_x × map_to_padded_y) cell
            // — use the cell centroid in padded-input space.
            let px_in_padded = ((*x as f32) + 0.5) * map_to_padded_x;
            let py_in_padded = ((*y as f32) + 0.5) * map_to_padded_y;
            let ox = (px_in_padded - pad_ox as f32) * inv_scale;
            let oy = (py_in_padded - pad_oy as f32) * inv_scale;
            pts.push(Point::new(ox, oy));
        }
        let Some(quad) = min_area_quad(&pts) else {
            continue;
        };
        if quad_aspect(&quad) > opts.max_aspect_ratio {
            continue;
        }
        out.push(CharBox { quad, score });
    }
    out
}

fn quad_aspect(q: &Quad) -> f32 {
    let p = &q.points;
    let e0 = dist(p[0], p[1]);
    let e1 = dist(p[1], p[2]);
    let long = e0.max(e1);
    let short = e0.min(e1);
    if short < 1e-6 {
        f32::INFINITY
    } else {
        long / short
    }
}

fn dist(a: Point, b: Point) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}
```

Update `lib.rs` re-exports:

```rust
pub use postprocess::{
    charboxes_from_heatmap, components_from_heatmap, CharBox, PostprocessOptions,
};
```

- [ ] **Step 3: Run tests — expect PASS**

```bash
cargo test -p docseg-core --lib postprocess
```

Expected: 5 passed.

- [ ] **Step 4: Run full local CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): components → CharBox with area/aspect filters and coord mapping

Extends postprocess to emit oriented per-character boxes in original-
image coordinates. Each component is reduced to a min-area rotated
rectangle (via `geometry::min_area_quad`), filtered by minimum area and
maximum aspect ratio, and annotated with its peak region score.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Model module — wonnx session wrapper (production path)

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/model.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

**Contract:** an async `CraftSession` type with `from_bytes(&[u8]) -> Result<Self, CoreError>` and `run(&self, PreprocessOutput) -> Result<RegionMap, CoreError>` where `RegionMap` is `{ data: Vec<f32>, width: u32, height: u32 }` representing just the region-score channel (we discard affinity).

- [ ] **Step 1: Move `wonnx` from `[dev-dependencies]` note to a full runtime dependency**

Edit `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml` — the `[dependencies.wonnx]` block from Task 4 already has it as a runtime dep; confirm so and keep `pollster` only under `[dev-dependencies]`.

- [ ] **Step 2: Write a failing unit test asserting channel extraction**

Append to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess/tests.rs`:

```rust

#[test]
fn region_channel_extracted_from_channels_last_output() {
    // 2x2 heatmap, channels-last layout: [r00,a00,r01,a01,r10,a10,r11,a11].
    let raw = vec![
        0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6,
    ];
    let region = crate::model::region_channel_channels_last(&raw, 2, 2);
    assert_eq!(region, vec![0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn region_channel_extracted_from_channels_first_output() {
    // 2x2 heatmap, channels-first layout: [r00,r01,r10,r11, a00,a01,a10,a11].
    let raw = vec![
        0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6,
    ];
    let region = crate::model::region_channel_channels_first(&raw, 2, 2);
    assert_eq!(region, vec![0.1, 0.2, 0.3, 0.4]);
}
```

- [ ] **Step 3: Create `model.rs` with the extraction helpers + session stub**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/model.rs`

```rust
//! CRAFT model session: load an ONNX bundle, run a forward pass, return the
//! region-score heatmap (we discard the affinity channel because this demo
//! segments at the character level, not the word level).

use std::collections::HashMap;

use wonnx::utils::InputTensor;
use wonnx::Session;

use crate::preprocess::PreprocessOutput;
use crate::CoreError;

/// Region-score heatmap at CRAFT's native output resolution (H/2 × W/2 of
/// the padded input).
#[derive(Debug, Clone)]
pub struct RegionMap {
    /// Row-major f32 values, length `width * height`.
    pub data: Vec<f32>,
    /// Heatmap width.
    pub width: u32,
    /// Heatmap height.
    pub height: u32,
}

/// Loaded CRAFT session.
pub struct CraftSession {
    inner: Session,
    input_name: String,
    output_name: String,
    /// `true` if the observed output layout was `[1, H/2, W/2, 2]` — we pick
    /// this at first-run time by inspecting the output shape relative to the
    /// known padded input size.
    channels_last: bool,
}

impl CraftSession {
    /// Load an ONNX bundle from bytes (works in native and WASM).
    pub async fn from_bytes(bytes: &[u8]) -> Result<Self, CoreError> {
        let inner = Session::from_bytes(bytes)
            .await
            .map_err(|e| CoreError::ModelLoad {
                hint: "wonnx Session::from_bytes".into(),
                source: Box::new(e),
            })?;
        let input_name = inner
            .inputs()
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| CoreError::ModelLoad {
                hint: "no inputs".into(),
                source: Box::new(std::io::Error::other("no inputs")),
            })?;
        let output_name = inner
            .outputs()
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| CoreError::ModelLoad {
                hint: "no outputs".into(),
                source: Box::new(std::io::Error::other("no outputs")),
            })?;
        Ok(Self {
            inner,
            input_name,
            output_name,
            channels_last: true, // assumption; refined in `run` on first call
        })
    }

    /// Run a single forward pass and return the region-score heatmap only.
    pub async fn run(
        &self,
        preproc: &PreprocessOutput,
    ) -> Result<RegionMap, CoreError> {
        let mut inputs: HashMap<String, InputTensor> = HashMap::new();
        inputs.insert(self.input_name.clone(), preproc.tensor.as_slice().into());
        let outputs = self
            .inner
            .run(&inputs)
            .await
            .map_err(|e| CoreError::Inference {
                source: Box::new(e),
            })?;
        let out_tensor = outputs.get(&self.output_name).ok_or_else(|| {
            CoreError::Postprocess {
                reason: format!("expected output {}", self.output_name),
            }
        })?;
        let raw: Vec<f32> =
            out_tensor
                .try_into()
                .map_err(|e: wonnx::utils::TensorConversionError| {
                    CoreError::Postprocess {
                        reason: format!("output not f32: {e}"),
                    }
                })?;

        let (padded_w, padded_h) = preproc.padded_size;
        let hm_w = padded_w / 2;
        let hm_h = padded_h / 2;
        let expected = 2 * (hm_w as usize) * (hm_h as usize);
        if raw.len() != expected {
            return Err(CoreError::Postprocess {
                reason: format!(
                    "CRAFT output length {} != expected {expected}",
                    raw.len()
                ),
            });
        }

        let region = if self.channels_last {
            region_channel_channels_last(&raw, hm_w, hm_h)
        } else {
            region_channel_channels_first(&raw, hm_w, hm_h)
        };
        Ok(RegionMap {
            data: region,
            width: hm_w,
            height: hm_h,
        })
    }
}

/// Extract the `region` channel assuming a channels-last layout
/// `[1, H, W, 2]` where `[..][..][0]` is region and `[..][..][1]` is affinity.
#[must_use]
pub fn region_channel_channels_last(raw: &[f32], w: u32, h: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w as usize) * (h as usize));
    for i in 0..((w as usize) * (h as usize)) {
        out.push(raw[i * 2]);
    }
    out
}

/// Extract the `region` channel assuming a channels-first layout
/// `[1, 2, H, W]` where channel 0 is region, channel 1 is affinity.
#[must_use]
pub fn region_channel_channels_first(raw: &[f32], w: u32, h: u32) -> Vec<f32> {
    let plane = (w as usize) * (h as usize);
    raw[..plane].to_vec()
}
```

Update `lib.rs`:

```rust
pub mod model;
pub use model::{CraftSession, RegionMap};
```

- [ ] **Step 4: Run the unit tests — expect PASS**

```bash
cargo test -p docseg-core --lib postprocess
```

Expected: 7 passed.

- [ ] **Step 5: Update `tests/forward_pass.rs` to detect layout at runtime**

Replace the body of `/Users/fangluo/Desktop/docseg/crates/docseg-core/tests/forward_pass.rs` with:

```rust
//! Gate test: CRAFT ONNX must load and run a single forward pass via wonnx,
//! producing a correctly-shaped output, and we must be able to extract the
//! region channel under whichever layout the checkpoint uses.

#![allow(clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use docseg_core::preprocess::{preprocess, PreprocessOptions};
use docseg_core::CraftSession;
use image::{DynamicImage, RgbImage};

fn model_bytes() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/craft_mlt_25k.onnx");
    std::fs::read(&path).expect("read model")
}

#[test]
#[ignore = "requires models/craft_mlt_25k.onnx (run scripts/fetch-model.sh first)"]
fn craft_forward_pass_produces_region_heatmap_of_expected_shape() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,wonnx=warn")
        .try_init();

    let bytes = model_bytes();
    let session = pollster::block_on(CraftSession::from_bytes(&bytes))
        .expect("CRAFT ONNX failed to load in wonnx. If the op name is logged, \
                 record it and pivot to the burn+wgpu fallback per the spec.");

    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(
        640,
        640,
        image::Rgb([255, 255, 255]),
    ));
    let pre = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    let region = pollster::block_on(session.run(&pre)).expect("run");
    let (w, h) = pre.padded_size;
    assert_eq!(region.width, w / 2);
    assert_eq!(region.height, h / 2);
    assert_eq!(
        region.data.len(),
        (w / 2) as usize * (h / 2) as usize
    );
}
```

- [ ] **Step 6: Run the gate test** (still `--ignored`, opt-in)

```bash
cargo test -p docseg-core --test forward_pass -- --ignored --nocapture
```

Expected: PASS.

**If the test shows the heatmap "looks flipped" or noisy on the visual end-to-end test in Task 10**, toggle `channels_last = false` in `CraftSession::from_bytes` to try the other layout. (This is a real 50/50 between the two common CRAFT exports; we either get it right the first try or flip the flag once.)

- [ ] **Step 7: Run full local CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 8: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): CraftSession wonnx wrapper emitting a region-score heatmap

Wraps wonnx::Session with from_bytes/run that accepts a PreprocessOutput
and returns a RegionMap at the CRAFT native H/2 x W/2 resolution. The
affinity channel is discarded — this demo segments at the character
level. A static flag picks between channels-first and channels-last ONNX
export layouts; covered by unit tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Core end-to-end test on a synthetic fixture

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/tests/end_to_end.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/tests/fixtures/gen_synthetic.rs` (small binary that the test runs at build time? — no, simpler: generate the fixture inside the test)

- [ ] **Step 1: Write an end-to-end test that generates a synthetic 3-blob image and expects 3 boxes**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-core/tests/end_to_end.rs`

```rust
//! End-to-end pipeline test: synthetic image with 3 well-separated filled
//! rectangles → CRAFT → postprocess → 3 CharBoxes.
//!
//! Gated on `models/craft_mlt_25k.onnx` being present. Run with
//!   cargo test -p docseg-core --test end_to_end -- --ignored

#![allow(clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use docseg_core::postprocess::{charboxes_from_heatmap, PostprocessOptions};
use docseg_core::preprocess::{preprocess, PreprocessOptions};
use docseg_core::CraftSession;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;

fn model_bytes() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/craft_mlt_25k.onnx");
    std::fs::read(&path).expect("read model")
}

fn synthetic_three_blobs() -> DynamicImage {
    let mut img = RgbImage::from_pixel(320, 320, Rgb([255, 255, 255]));
    let ink = Rgb([20, 20, 20]);
    // Three 60x60 dark blocks along a row.
    for (cx, cy) in [(60, 160), (160, 160), (260, 160)] {
        draw_filled_rect_mut(
            &mut img,
            Rect::at(cx - 30, cy - 30).of_size(60, 60),
            ink,
        );
    }
    DynamicImage::ImageRgb8(img)
}

#[test]
#[ignore = "requires models/craft_mlt_25k.onnx"]
fn end_to_end_detects_three_blobs() {
    let img = synthetic_three_blobs();
    let pre = preprocess(&img, PreprocessOptions::default()).expect("preprocess");

    let bytes = model_bytes();
    let session =
        pollster::block_on(CraftSession::from_bytes(&bytes)).expect("load");
    let region =
        pollster::block_on(session.run(&pre)).expect("inference");

    let boxes = charboxes_from_heatmap(
        &region.data,
        region.width,
        region.height,
        &pre,
        PostprocessOptions::default(),
    );
    // Three blobs — tolerate off-by-one if CRAFT accidentally merges/splits.
    assert!(
        (2..=4).contains(&boxes.len()),
        "expected ~3 boxes, got {}: {:?}",
        boxes.len(),
        boxes
    );
    for b in &boxes {
        assert!(b.score > 0.1, "score {} too low", b.score);
        for p in b.quad.points {
            assert!(p.x >= 0.0 && p.x <= 320.0);
            assert!(p.y >= 0.0 && p.y <= 320.0);
        }
    }
}
```

- [ ] **Step 2: Run the test** (with `--ignored`)

```bash
cargo test -p docseg-core --test end_to_end -- --ignored --nocapture
```

Expected: PASS, with 2–4 boxes.

**If 0 boxes are returned:** the channels layout flag in `CraftSession` is wrong — flip `channels_last` in `from_bytes` (task 9, step 6 note) and re-run. If the flip gives reasonable boxes on the real manuscript later but this synthetic still fails, lower `region_threshold` on the `PostprocessOptions` used here (CRAFT's confidence on synthetic shapes is lower than on natural text).

- [ ] **Step 3: Run full local CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK` (the ignored tests do not run by default).

- [ ] **Step 4: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
test(core): end-to-end pipeline asserts three CharBoxes on synthetic blobs

Generates a 320x320 canvas with three well-separated dark rectangles,
runs preprocess → CraftSession → postprocess, and asserts the detector
returns roughly three in-bounds boxes with nonzero scores. Gated on the
CRAFT ONNX being present; runs opt-in under `--ignored`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `docseg-web` crate scaffolding + wasm-pack build wiring

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-web/Cargo.toml`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/lib.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`
- Create: `/Users/fangluo/Desktop/docseg/scripts/build-web.sh`

- [ ] **Step 1: Write `crates/docseg-web/Cargo.toml`**

```toml
[package]
name = "docseg-web"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true
description = "WASM/browser adapter for docseg-core: image decode, canvas overlay, export zip."

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
docseg-core = { path = "../docseg-core" }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
serde.workspace = true
serde_json.workspace = true
serde-wasm-bindgen = "0.6"
thiserror.workspace = true
tracing.workspace = true
tracing-wasm = "0.2"
console_error_panic_hook = "0.1"
image.workspace = true
zip = { version = "0.6", default-features = false, features = ["deflate"] }

[dependencies.web-sys]
version = "0.3"
features = [
  "Window", "Document", "Element", "HtmlCanvasElement", "HtmlImageElement",
  "CanvasRenderingContext2d", "ImageData", "Blob", "Url", "Event",
  "MouseEvent", "Response", "Request", "RequestInit", "RequestMode",
  "console",
]

[lints]
workspace = true
```

- [ ] **Step 2: Write `crates/docseg-web/src/lib.rs`**

```rust
//! `docseg-web` — thin wasm-bindgen adapter over `docseg-core`.
//!
//! Responsibilities: decode image bytes, fetch the model bundle, drive the
//! core pipeline, render overlays on an HTMLCanvasElement, and produce a
//! zip of per-glyph crops on demand. No model logic lives here.

#![deny(missing_docs)]

mod canvas;
mod entry;
mod export;
mod render;

pub use entry::DocsegApp;
```

- [ ] **Step 3: Write a minimal `entry.rs` with a `DocsegApp::new()` stub**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

```rust
//! Top-level wasm-bindgen surface: construct a `DocsegApp`, load the model,
//! run detection, and expose overlay / export handles.

use wasm_bindgen::prelude::*;

/// Top-level handle returned to JS.
#[wasm_bindgen]
pub struct DocsegApp {
    _private: (),
}

#[wasm_bindgen]
impl DocsegApp {
    /// Construct a fresh app. Installs a tracing subscriber and a panic hook.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let _ = tracing_wasm::try_set_as_global_default();
        Self { _private: () }
    }
}

impl Default for DocsegApp {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Write placeholder `canvas.rs`, `render.rs`, `export.rs` so the crate compiles**

`/Users/fangluo/Desktop/docseg/crates/docseg-web/src/canvas.rs`:

```rust
//! Image decoding from bytes the browser hands us.
```

`/Users/fangluo/Desktop/docseg/crates/docseg-web/src/render.rs`:

```rust
//! Overlay painting and click hit-testing.
```

`/Users/fangluo/Desktop/docseg/crates/docseg-web/src/export.rs`:

```rust
//! Per-glyph crop + zip export.
```

- [ ] **Step 5: Write `scripts/build-web.sh`**

```bash
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
```

Then `chmod +x scripts/build-web.sh`.

- [ ] **Step 6: Run the wasm build**

```bash
./scripts/build-web.sh
```

Expected: `web/pkg/docseg_web.js`, `web/pkg/docseg_web_bg.wasm` exist.

- [ ] **Step 7: Update `scripts/ci-local.sh`**

No change needed — its `cargo metadata` check will now include `docseg-web` and run the wasm build path.

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 8: Commit**

```bash
git add crates/docseg-web scripts/build-web.sh Cargo.toml
git commit -m "$(cat <<'EOF'
feat(web): scaffold docseg-web wasm-bindgen crate + build-web.sh

Adds a cdylib+rlib crate that re-exports DocsegApp as the JS surface,
with web-sys features wired for canvas / fetch / image, plus
tracing-wasm + console_error_panic_hook for browser diagnostics. No
pipeline logic yet — that lands in the next tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: WASM entry — image decode, model load, run, return boxes

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/canvas.rs`

- [ ] **Step 1: Fill in `canvas.rs` (image decode from `&[u8]`)**

```rust
//! Image decoding from bytes the browser hands us.

use docseg_core::CoreError;
use image::DynamicImage;

/// Decode a PNG or JPEG blob into a `DynamicImage`.
pub fn decode(bytes: &[u8]) -> Result<DynamicImage, CoreError> {
    image::load_from_memory(bytes).map_err(CoreError::Decode)
}
```

- [ ] **Step 2: Replace `entry.rs` with the load/run surface**

```rust
//! Top-level wasm-bindgen surface: construct a `DocsegApp`, load the model,
//! run detection, and return serialized boxes to JS.

use std::cell::RefCell;

use docseg_core::postprocess::{charboxes_from_heatmap, CharBox, PostprocessOptions};
use docseg_core::preprocess::{preprocess, PreprocessOptions};
use docseg_core::CraftSession;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::canvas::decode;

#[derive(Serialize)]
struct BoxOut {
    id: u32,
    quad: [[f32; 2]; 4],
    score: f32,
}

#[derive(Serialize)]
struct DetectionOut {
    image: ImageMeta,
    model: &'static str,
    boxes: Vec<BoxOut>,
}

#[derive(Serialize)]
struct ImageMeta {
    width: u32,
    height: u32,
}

/// Top-level handle returned to JS.
#[wasm_bindgen]
pub struct DocsegApp {
    session: RefCell<Option<CraftSession>>,
}

#[wasm_bindgen]
impl DocsegApp {
    /// Construct a fresh app. Installs a tracing subscriber and a panic hook.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let _ = tracing_wasm::try_set_as_global_default();
        Self {
            session: RefCell::new(None),
        }
    }

    /// Load the CRAFT ONNX bundle. Must be called once before `detect`.
    #[wasm_bindgen(js_name = loadModel)]
    pub async fn load_model(&self, model_bytes: Vec<u8>) -> Result<(), JsError> {
        let s = CraftSession::from_bytes(&model_bytes)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        *self.session.borrow_mut() = Some(s);
        Ok(())
    }

    /// Run detection on encoded PNG / JPEG bytes. Returns a JSON-shape value:
    /// `{ image: {width, height}, model, boxes: [{ id, quad: [[x,y]x4], score }] }`.
    pub async fn detect(&self, image_bytes: Vec<u8>) -> Result<JsValue, JsError> {
        let img = decode(&image_bytes).map_err(|e| JsError::new(&format!("{e:#}")))?;
        let (iw, ih) = (img.width(), img.height());
        let pre = preprocess(&img, PreprocessOptions::default())
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        let session_ref = self.session.borrow();
        let session = session_ref
            .as_ref()
            .ok_or_else(|| JsError::new("loadModel must be called before detect"))?;
        let region = session
            .run(&pre)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        let boxes = charboxes_from_heatmap(
            &region.data,
            region.width,
            region.height,
            &pre,
            PostprocessOptions::default(),
        );
        let out = DetectionOut {
            image: ImageMeta {
                width: iw,
                height: ih,
            },
            model: "craft_mlt_25k",
            boxes: boxes.iter().enumerate().map(|(i, b)| box_out(i, b)).collect(),
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&format!("{e}")))
    }
}

impl Default for DocsegApp {
    fn default() -> Self {
        Self::new()
    }
}

fn box_out(i: usize, b: &CharBox) -> BoxOut {
    let quad = [
        [b.quad.points[0].x, b.quad.points[0].y],
        [b.quad.points[1].x, b.quad.points[1].y],
        [b.quad.points[2].x, b.quad.points[2].y],
        [b.quad.points[3].x, b.quad.points[3].y],
    ];
    BoxOut {
        id: i as u32,
        quad,
        score: b.score,
    }
}
```

- [ ] **Step 3: Build the wasm bundle**

```bash
./scripts/build-web.sh
```

Expected: builds clean. No tests yet for the wasm side — integration test is a manual smoke test in Task 16.

- [ ] **Step 4: Run the lint gate on the whole workspace**

```bash
cargo clippy --all-targets --workspace -- -D warnings
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): DocsegApp.loadModel + detect wiring end-to-end

Adds the async JS surface: loadModel takes ONNX bytes, detect takes
encoded image bytes and returns a DetectionOut JSON with per-character
quads + scores mapped back to original image coordinates. Canvas
rendering and export land in the next tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Canvas rendering — draw image + box overlays

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/render.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

- [ ] **Step 1: Flesh out `render.rs` with a stateless drawing helper**

```rust
//! Overlay painting and click hit-testing.

use docseg_core::postprocess::CharBox;
use wasm_bindgen::JsValue;
use web_sys::{CanvasRenderingContext2d, HtmlImageElement};

/// Paint the source image at native size followed by a yellow polygon per
/// detected char box.
pub fn paint(
    ctx: &CanvasRenderingContext2d,
    image: &HtmlImageElement,
    boxes: &[CharBox],
) -> Result<(), JsValue> {
    ctx.draw_image_with_html_image_element(image, 0.0, 0.0)?;
    ctx.set_stroke_style(&JsValue::from_str("rgba(255, 196, 0, 0.9)"));
    ctx.set_line_width(2.0);
    for b in boxes {
        let p = &b.quad.points;
        ctx.begin_path();
        ctx.move_to(p[0].x.into(), p[0].y.into());
        for pt in &p[1..] {
            ctx.line_to(pt.x.into(), pt.y.into());
        }
        ctx.close_path();
        ctx.stroke();
    }
    Ok(())
}

/// Return the index of the first `boxes` entry whose axis-aligned bounding
/// rectangle contains `(x, y)`, or `None`.
#[must_use]
pub fn hit_test(boxes: &[CharBox], x: f32, y: f32) -> Option<usize> {
    boxes.iter().position(|b| contains(b, x, y))
}

fn contains(b: &CharBox, x: f32, y: f32) -> bool {
    let p = &b.quad.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    let xmin = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let xmax = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ymin = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let ymax = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    x >= xmin && x <= xmax && y >= ymin && y <= ymax
}
```

- [ ] **Step 2: Store the last detection on `DocsegApp` and expose `paint` + `hit`**

Append to `crates/docseg-web/src/entry.rs` (inside the existing `impl DocsegApp` block and supporting state — the simplest change is to add a `RefCell<Vec<CharBox>>` + `RefCell<Option<HtmlImageElement>>` and paint/hit methods):

Replace the entire file with the version below (the surface now includes `paint` and `hit`):

```rust
//! Top-level wasm-bindgen surface: construct a `DocsegApp`, load the model,
//! run detection, paint overlays, and expose hit-test + export handles.

use std::cell::RefCell;

use docseg_core::postprocess::{charboxes_from_heatmap, CharBox, PostprocessOptions};
use docseg_core::preprocess::{preprocess, PreprocessOptions};
use docseg_core::CraftSession;
use serde::Serialize;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlImageElement};

use crate::canvas::decode;
use crate::render::{hit_test, paint};

#[derive(Serialize)]
struct BoxOut {
    id: u32,
    quad: [[f32; 2]; 4],
    score: f32,
}

#[derive(Serialize)]
struct DetectionOut {
    image: ImageMeta,
    model: &'static str,
    boxes: Vec<BoxOut>,
}

#[derive(Serialize)]
struct ImageMeta {
    width: u32,
    height: u32,
}

/// Top-level handle returned to JS.
#[wasm_bindgen]
pub struct DocsegApp {
    session: RefCell<Option<CraftSession>>,
    last_boxes: RefCell<Vec<CharBox>>,
    last_image_bytes: RefCell<Vec<u8>>,
}

#[wasm_bindgen]
impl DocsegApp {
    /// Construct a fresh app. Installs a tracing subscriber and a panic hook.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let _ = tracing_wasm::try_set_as_global_default();
        Self {
            session: RefCell::new(None),
            last_boxes: RefCell::new(Vec::new()),
            last_image_bytes: RefCell::new(Vec::new()),
        }
    }

    /// Load the CRAFT ONNX bundle. Must be called once before `detect`.
    #[wasm_bindgen(js_name = loadModel)]
    pub async fn load_model(&self, model_bytes: Vec<u8>) -> Result<(), JsError> {
        let s = CraftSession::from_bytes(&model_bytes)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        *self.session.borrow_mut() = Some(s);
        Ok(())
    }

    /// Run detection on encoded PNG / JPEG bytes. Returns a JSON-shape value.
    pub async fn detect(&self, image_bytes: Vec<u8>) -> Result<JsValue, JsError> {
        let img = decode(&image_bytes).map_err(|e| JsError::new(&format!("{e:#}")))?;
        let (iw, ih) = (img.width(), img.height());
        let pre = preprocess(&img, PreprocessOptions::default())
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        let session_ref = self.session.borrow();
        let session = session_ref
            .as_ref()
            .ok_or_else(|| JsError::new("loadModel must be called before detect"))?;
        let region = session
            .run(&pre)
            .await
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        let boxes = charboxes_from_heatmap(
            &region.data,
            region.width,
            region.height,
            &pre,
            PostprocessOptions::default(),
        );
        *self.last_boxes.borrow_mut() = boxes.clone();
        *self.last_image_bytes.borrow_mut() = image_bytes;
        let out = DetectionOut {
            image: ImageMeta {
                width: iw,
                height: ih,
            },
            model: "craft_mlt_25k",
            boxes: boxes.iter().enumerate().map(|(i, b)| box_out(i, b)).collect(),
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Paint the source image + per-character overlay onto a canvas context.
    pub fn paint(
        &self,
        ctx: &CanvasRenderingContext2d,
        img: &HtmlImageElement,
    ) -> Result<(), JsError> {
        paint(ctx, img, &self.last_boxes.borrow())
            .map_err(|e| JsError::new(&format!("paint failed: {e:?}")))
    }

    /// Hit-test the last detection. Returns the 0-based id of the first box
    /// whose bounding rect contains (x, y), or `-1`.
    pub fn hit(&self, x: f32, y: f32) -> i32 {
        hit_test(&self.last_boxes.borrow(), x, y)
            .map_or(-1, |i| i as i32)
    }
}

impl Default for DocsegApp {
    fn default() -> Self {
        Self::new()
    }
}

fn box_out(i: usize, b: &CharBox) -> BoxOut {
    let quad = [
        [b.quad.points[0].x, b.quad.points[0].y],
        [b.quad.points[1].x, b.quad.points[1].y],
        [b.quad.points[2].x, b.quad.points[2].y],
        [b.quad.points[3].x, b.quad.points[3].y],
    ];
    BoxOut {
        id: i as u32,
        quad,
        score: b.score,
    }
}
```

Also derive `Clone` on `CharBox` in `crates/docseg-core/src/postprocess.rs` (the `clone()` in Step 2 requires it):

```rust
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CharBox { /* ... */ }
```

Already `Clone` from Task 8 — no change needed.

- [ ] **Step 3: Rebuild + clippy**

```bash
./scripts/build-web.sh
cargo clippy --all-targets --workspace -- -D warnings
```

Expected: both clean.

- [ ] **Step 4: Commit**

```bash
git add crates/docseg-web crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(web): render overlays + hit-test on the last detection

Stores the most recent CharBox list on DocsegApp so the JS side can call
paint(ctx, img) and hit(x, y) without re-running inference. paint draws
the source image at native size then strokes each quad in translucent
yellow; hit returns the box index whose AABB contains a click point.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Single-glyph crop export (click → download PNG)

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/export.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

**Contract:** `cropPng(id)` returns `Uint8Array` of a PNG containing the axis-aligned bounding-rect crop of box `id`. (We crop the AABB, not the rotated rect, so the output is a rectangular PNG. For a demo this is the right trade-off; a perspective-unwarp step is a future extension.)

- [ ] **Step 1: Implement `export::crop_png`**

File: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/export.rs`

```rust
//! Per-glyph crop + zip export.

use docseg_core::postprocess::CharBox;
use docseg_core::CoreError;
use image::codecs::png::PngEncoder;
use image::{ColorType, GenericImageView, ImageEncoder};

/// Crop the axis-aligned bounding rectangle of `boxes[id]` from `image_bytes`
/// and return a fresh PNG-encoded byte vector.
pub fn crop_png(
    image_bytes: &[u8],
    boxes: &[CharBox],
    id: usize,
) -> Result<Vec<u8>, CoreError> {
    let img = image::load_from_memory(image_bytes).map_err(CoreError::Decode)?;
    let (w, h) = img.dimensions();
    let b = boxes.get(id).ok_or_else(|| CoreError::Postprocess {
        reason: format!("no box with id {id}"),
    })?;
    let (xmin, ymin, xmax, ymax) = aabb(b);
    let x0 = xmin.max(0.0).floor() as u32;
    let y0 = ymin.max(0.0).floor() as u32;
    let x1 = xmax.min(w as f32 - 1.0).ceil() as u32;
    let y1 = ymax.min(h as f32 - 1.0).ceil() as u32;
    if x1 <= x0 || y1 <= y0 {
        return Err(CoreError::Postprocess {
            reason: format!("degenerate crop for id {id}"),
        });
    }
    let crop = img.crop_imm(x0, y0, x1 - x0, y1 - y0).to_rgba8();
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(
            crop.as_raw(),
            crop.width(),
            crop.height(),
            ColorType::Rgba8.into(),
        )
        .map_err(|e| CoreError::Postprocess {
            reason: format!("png encode: {e}"),
        })?;
    Ok(out)
}

fn aabb(b: &CharBox) -> (f32, f32, f32, f32) {
    let p = &b.quad.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    (
        xs.iter().cloned().fold(f32::INFINITY, f32::min),
        ys.iter().cloned().fold(f32::INFINITY, f32::min),
        xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    )
}
```

- [ ] **Step 2: Add `cropPng(id)` to `DocsegApp`**

Inside the `#[wasm_bindgen] impl DocsegApp` block in `entry.rs`, append:

```rust
    /// Return a PNG of the axis-aligned crop of box `id` from the last image
    /// passed to `detect`. Throws if detect hasn't run or id is out of range.
    #[wasm_bindgen(js_name = cropPng)]
    pub fn crop_png(&self, id: u32) -> Result<Vec<u8>, JsError> {
        crate::export::crop_png(
            &self.last_image_bytes.borrow(),
            &self.last_boxes.borrow(),
            id as usize,
        )
        .map_err(|e| JsError::new(&format!("{e:#}")))
    }
```

- [ ] **Step 3: Rebuild + clippy**

```bash
./scripts/build-web.sh
cargo clippy --all-targets --workspace -- -D warnings
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): cropPng(id) returns a PNG of a single glyph's AABB

The demo's click handler gets a one-shot path to a downloadable PNG for
the clicked character. We crop the axis-aligned bounding rect rather
than the rotated quad — a perspective-unwarp would be future work, but
for vertical-column manuscripts the AABB is already the usable unit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: "Export all" zip — per-glyph PNGs + `boxes.json`

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/export.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

- [ ] **Step 1: Append `export_zip` to `export.rs`**

```rust

use std::io::Write;
use zip::write::FileOptions;
use zip::CompressionMethod;

#[derive(serde::Serialize)]
struct BoxesJson<'a> {
    image: ImageMeta,
    model: &'a str,
    boxes: Vec<BoxEntry>,
}

#[derive(serde::Serialize)]
struct ImageMeta {
    width: u32,
    height: u32,
}

#[derive(serde::Serialize)]
struct BoxEntry {
    id: u32,
    quad: [[f32; 2]; 4],
    score: f32,
}

/// Bundle every box's AABB crop + a `boxes.json` manifest into a single
/// uncompressed-but-DEFLATE'd zip.
pub fn export_zip(
    image_bytes: &[u8],
    boxes: &[CharBox],
    model_name: &str,
) -> Result<Vec<u8>, CoreError> {
    let img = image::load_from_memory(image_bytes).map_err(CoreError::Decode)?;
    let (w, h) = img.dimensions();

    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut zipw = zip::ZipWriter::new(&mut buf);
        let opts = FileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);

        let manifest = BoxesJson {
            image: ImageMeta { width: w, height: h },
            model: model_name,
            boxes: boxes
                .iter()
                .enumerate()
                .map(|(i, b)| BoxEntry {
                    id: i as u32,
                    quad: [
                        [b.quad.points[0].x, b.quad.points[0].y],
                        [b.quad.points[1].x, b.quad.points[1].y],
                        [b.quad.points[2].x, b.quad.points[2].y],
                        [b.quad.points[3].x, b.quad.points[3].y],
                    ],
                    score: b.score,
                })
                .collect(),
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("json: {e}"),
            })?;
        zipw.start_file("boxes.json", opts)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip start boxes.json: {e}"),
            })?;
        zipw.write_all(&manifest_bytes)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip write boxes.json: {e}"),
            })?;

        for i in 0..boxes.len() {
            let png = crop_png(image_bytes, boxes, i)?;
            let name = format!("crops/{i:05}.png");
            zipw.start_file(&name, opts)
                .map_err(|e| CoreError::Postprocess {
                    reason: format!("zip start {name}: {e}"),
                })?;
            zipw.write_all(&png).map_err(|e| CoreError::Postprocess {
                reason: format!("zip write {name}: {e}"),
            })?;
        }

        zipw.finish().map_err(|e| CoreError::Postprocess {
            reason: format!("zip finish: {e}"),
        })?;
    }

    Ok(buf.into_inner())
}
```

- [ ] **Step 2: Expose `exportZip()` on `DocsegApp`**

Inside the `#[wasm_bindgen] impl DocsegApp`, append:

```rust
    /// Export a zip containing every crop plus a `boxes.json` manifest.
    #[wasm_bindgen(js_name = exportZip)]
    pub fn export_zip(&self) -> Result<Vec<u8>, JsError> {
        crate::export::export_zip(
            &self.last_image_bytes.borrow(),
            &self.last_boxes.borrow(),
            "craft_mlt_25k",
        )
        .map_err(|e| JsError::new(&format!("{e:#}")))
    }
```

- [ ] **Step 3: Rebuild + clippy**

```bash
./scripts/build-web.sh
cargo clippy --all-targets --workspace -- -D warnings
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): exportZip() bundles crops/*.png + boxes.json

One-click 'export all' from the demo UI. Zip is produced entirely inside
the wasm module using the `zip` crate — the JS side only handles the
resulting Uint8Array → Blob → anchor download.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: HTML + JS bootstrap + WebGPU detection

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/web/index.html`
- Create: `/Users/fangluo/Desktop/docseg/web/main.js`
- Create: `/Users/fangluo/Desktop/docseg/web/style.css`

- [ ] **Step 1: Write `web/style.css`**

```css
body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }
header { padding: 12px 16px; border-bottom: 1px solid #333; display: flex; gap: 12px; align-items: center; }
header h1 { font-size: 16px; margin: 0; font-weight: 600; }
main { padding: 16px; }
#canvas-wrap { overflow: auto; max-width: 100%; }
canvas { display: block; background: #222; image-rendering: pixelated; }
#banner { padding: 16px; background: #822; color: #fff; margin: 16px; border-radius: 6px; }
#status { font-family: ui-monospace, monospace; font-size: 12px; color: #aaa; }
button, label { font-size: 13px; padding: 6px 10px; border-radius: 4px; background: #333; color: #eee; border: 1px solid #555; cursor: pointer; }
button:disabled { opacity: 0.5; cursor: default; }
```

- [ ] **Step 2: Write `web/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>docseg — character segmentation demo</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <header>
    <h1>docseg</h1>
    <label>
      <input id="file" type="file" accept="image/png,image/jpeg" hidden />
      <span>Open image</span>
    </label>
    <button id="export" disabled>Export all (zip)</button>
    <span id="status">initializing…</span>
  </header>
  <div id="banner" hidden></div>
  <main>
    <div id="canvas-wrap">
      <canvas id="canvas" width="0" height="0"></canvas>
    </div>
  </main>
  <script type="module" src="main.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write `web/main.js`**

```js
import init, { DocsegApp } from "./pkg/docseg_web.js";

const MODEL_URL =
  "https://huggingface.co/Bingsu/craft-onnx/resolve/main/craft_mlt_25k.onnx";

const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };

async function main() {
  if (!("gpu" in navigator)) {
    const banner = $("banner");
    banner.hidden = false;
    banner.textContent =
      "WebGPU is required. This demo needs Chrome 113+, Safari 17.4+, or Firefox Nightly with WebGPU enabled.";
    status("");
    return;
  }

  status("loading wasm…");
  await init();
  const app = new DocsegApp();

  status("fetching model (~20 MB)…");
  const modelRes = await fetch(MODEL_URL);
  if (!modelRes.ok) throw new Error(`model fetch: ${modelRes.status}`);
  const modelBytes = new Uint8Array(await modelRes.arrayBuffer());

  status("loading model onto GPU…");
  await app.loadModel(modelBytes);
  status("ready. open an image.");

  $("file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await runDetection(app, file);
  });

  $("export").addEventListener("click", () => {
    const bytes = app.exportZip();
    downloadBlob(new Blob([bytes], { type: "application/zip" }), "docseg.zip");
  });

  // Auto-load the sample image if we're served a neighbor file.
  try {
    const sample = await fetch("test_case1.png");
    if (sample.ok) {
      const blob = await sample.blob();
      await runDetection(app, blob);
    }
  } catch { /* no sample present, skip */ }
}

async function runDetection(app, blob) {
  status("decoding…");
  const imgBytes = new Uint8Array(await blob.arrayBuffer());

  status("running inference…");
  const t0 = performance.now();
  const detection = await app.detect(imgBytes);
  const t1 = performance.now();

  const htmlImg = await blobToHtmlImage(blob);
  const canvas = $("canvas");
  canvas.width = detection.image.width;
  canvas.height = detection.image.height;
  const ctx = canvas.getContext("2d");
  app.paint(ctx, htmlImg);

  canvas.onclick = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
    const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
    const id = app.hit(x, y);
    if (id < 0) return;
    const png = app.cropPng(id);
    downloadBlob(new Blob([png], { type: "image/png" }), `glyph_${id}.png`);
  };

  $("export").disabled = false;
  status(`detected ${detection.boxes.length} characters in ${(t1 - t0).toFixed(0)} ms`);
}

function blobToHtmlImage(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { resolve(img); };
    img.onerror = (e) => reject(e);
    img.src = url;
  });
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

main().catch((e) => {
  console.error(e);
  status(`error: ${e.message}`);
});
```

- [ ] **Step 4: Copy the sample image into `web/` so it auto-loads**

```bash
cp /Users/fangluo/Desktop/OCR_Yi/test_images/test_case1.png web/test_case1.png
```

- [ ] **Step 5: Build + serve + manual smoke test**

```bash
./scripts/build-web.sh
python3 -m http.server 8787 --directory web
```

Then in Chrome 113+:

1. Navigate to `http://localhost:8787/`.
2. Wait for `status: detected N characters in M ms` (will auto-load `test_case1.png`).
3. Verify yellow polygons cover most of the visible glyphs.
4. Click a character — verify a `glyph_*.png` downloads showing just that character.
5. Click "Export all (zip)" — verify `docseg.zip` downloads and (unzipped on disk) contains `boxes.json` + `crops/*.png`.

Record the observed character count and timing in the commit message.

**If WebGPU banner appears even in a WebGPU-capable Chrome:** ensure `chrome://flags/#unsafe-webgpu` is enabled if you're on Linux, or upgrade Chrome. The banner is correct behavior for non-WebGPU browsers.

**If the overlay is clearly misaligned (boxes don't land on glyphs):** the `channels_last` flag in `CraftSession::from_bytes` is wrong for this export. Flip it (Task 9 Step 6), rebuild, reload.

- [ ] **Step 6: Commit**

```bash
git add web
git commit -m "$(cat <<'EOF'
feat(web): HTML+JS bootstrap with WebGPU gate and auto-load sample

Plain ES-module main.js fetches the CRAFT ONNX, initializes DocsegApp,
auto-loads test_case1.png if present, paints the canvas, and wires click-
to-download-crop + Export All. WebGPU is gated up-front with a visible
banner on unsupported browsers — there is no WebGL2 fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: README and final full-gate CI pass

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# docseg

Rust → WASM → WebGPU demo that takes a handwritten-manuscript image and
returns per-character oriented bounding boxes. All inference runs client-
side in the browser via `wonnx`; no server.

## What it does

- Loads a page image (PNG / JPEG).
- Runs [CRAFT](https://arxiv.org/abs/1904.01941) (pre-trained
  `craft_mlt_25k`) through `wonnx` on WebGPU.
- Thresholds the region-score heatmap, extracts connected components,
  fits a min-area oriented rectangle per component, filters by area and
  aspect ratio, maps back to original-image coordinates.
- Overlays each detected character and supports click-to-download crop
  + "Export all (zip)" of `crops/*.png` + `boxes.json`.

## Requirements

- **Browser:** Chrome 113+, Safari 17.4+, or Firefox Nightly with WebGPU
  enabled. The app gates on `navigator.gpu` and stops with a visible
  banner on unsupported browsers.
- **Rust:** stable 1.74+ (pinned in `rust-toolchain.toml`) with the
  `wasm32-unknown-unknown` target.
- **Tools:** `wasm-pack` (`cargo install wasm-pack`), `python3`, `curl`,
  `shasum`.

## Build + run

```bash
# 1. Fetch the model (~20 MB). First run prints the SHA-256 for pinning.
./scripts/fetch-model.sh

# 2. Build the wasm bundle.
./scripts/build-web.sh

# 3. Serve the demo.
python3 -m http.server 8787 --directory web

# 4. Open http://localhost:8787/ in a WebGPU-capable browser.
```

## Local CI

Run the full gate (fmt → clippy → test → doc → wasm build) before
pushing:

```bash
./scripts/ci-local.sh
```

## Crates

- `docseg-core` — pure Rust: preprocess, wonnx session, postprocess,
  geometry, error types. Natively testable with `cargo test -p
  docseg-core`. Tests that require the model file are `#[ignore]`d;
  opt in with `-- --ignored`.
- `docseg-web` — wasm-bindgen adapter: image decode, canvas overlay,
  zip export. Exposes `DocsegApp` (constructor, `loadModel`, `detect`,
  `paint`, `hit`, `cropPng`, `exportZip`) to JS.

## Known limitations

- The browser's built-in HTTP cache handles model caching; no IndexedDB
  persistence. First load after a cache clear re-downloads ~20 MB.
- Crops are axis-aligned bounding rectangles of the oriented quad, not
  perspective-unwarped.
- WebGPU adapter limits vary across browsers; very large pages may hit
  a buffer-size cap. Reduce `PreprocessOptions::target_long_side` in
  `docseg-core` if so.
```

- [ ] **Step 2: Run the full local CI gate one more time**

```bash
./scripts/ci-local.sh
```

Expected: `OK`. All of fmt/clippy/test/doc/wasm-build clean.

- [ ] **Step 3: Run the ignored integration tests if the model is present**

```bash
cargo test -p docseg-core -- --ignored
```

Expected: forward-pass test and end-to-end test both pass.

- [ ] **Step 4: Final smoke test in browser**

Repeat Task 16 Step 5 and verify:

- sample loads
- boxes align to glyphs
- individual click download works
- export-zip works and contains `boxes.json` + `crops/*.png`

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: README with build/run/browser-support instructions

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review (completed before saving)

**Spec coverage:** every section of the design spec maps to at least one task.

| Spec section | Task(s) |
|---|---|
| §2 Non-goals (no OCR, no training, no CPU/WebGL2 fallback, no mobile, one image) | Enforced by scope of tasks 11–16; WebGPU gate in Task 16 |
| §3 Chosen model: CRAFT, region-only post-process | Tasks 3, 4, 7, 8, 9 |
| §4 Inference stack: wonnx primary, burn fallback | Task 4 (gate + contingency) |
| §5 Architecture (workspace, crate boundaries, file list) | Tasks 1, 5, 6, 7, 8, 9, 11 |
| §6 Data flow | End-to-end realized by Tasks 12–16 |
| §7 Error handling | Task 2 (CoreError) + JsError mapping in Task 12; WebGPU banner in Task 16 |
| §8 Testing strategy (unit + opt-in model-requiring integration) | Tasks 2, 5, 6, 7, 8, 9 (unit); 4, 10 (integration, ignored) |
| §9 Build / dev workflow + CI gates | Tasks 1 (ci-local.sh), 11 (build-web.sh), 17 (final gate) |
| §10 Workspace-level lints | Task 1 |
| §11 Open risks (wonnx gap, cursive merging, seal noise, wasm size, adapter limits) | Task 4 gate covers the first; tuning knobs exposed in `PostprocessOptions`; wasm size is implicit in build config; adapter limits noted in README Known Limitations |
| §12 Deliverables | 1 = Tasks 1–11, 2 = Task 4, 3 = Tasks 12–16, 4 = Task 17 |

**Placeholder scan:** every code-changing step contains complete code. No "TBD", no "similar to Task N" without repetition, no bare "add error handling" instructions.

**Type / name consistency:**

- `PreprocessOutput { tensor, padded_size, scale, pad_offset }` defined in Task 5 and consumed in Tasks 8, 9, 12. ✓
- `CoreError` variants defined in Task 2 and used in Tasks 5, 8, 9, 14, 15. All match. ✓
- `CharBox { quad, score }` defined in Task 8, stored on `DocsegApp` in Task 13, cropped in Task 14, zipped in Task 15. ✓
- `RegionMap { data, width, height }` defined in Task 9 and consumed in Task 10 + 12 + 13. ✓
- `DocsegApp` methods (`loadModel`, `detect`, `paint`, `hit`, `cropPng`, `exportZip`) declared progressively in Tasks 11–15 and all called from `web/main.js` in Task 16. Names match on both sides. ✓

No issues found.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-21-rust-wasm-wgpu-character-segmentation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — I execute tasks in this session using the executing-plans skill, batch execution with checkpoints for review.

Which approach?
