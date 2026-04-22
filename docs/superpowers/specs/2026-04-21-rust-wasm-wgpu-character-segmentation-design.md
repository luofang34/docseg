# `docseg` — Rust / WASM / WebGPU per-character segmentation demo

Status: design approved 2026-04-21. Implementation plan to follow.

## 1. Goal

Ship a single-page, client-side web demo that takes a page image of dense handwritten script (e.g. the attached Yi-script manuscript, `test_case1.png`) and returns a per-character segmentation: an oriented box (quadrilateral) around each individual glyph, visualized over the original image, with one-click crop export.

The entire inference pipeline runs in the browser via Rust compiled to WASM, with neural-network inference dispatched through WebGPU. No server, no JS-side inference runtime.

## 2. Non-goals

- **No character recognition / OCR.** Output is geometry only. Recognition stays in the sibling `OCR_Yi` Python project.
- **No training.** We consume a pre-trained open-source checkpoint.
- **No CPU / WebGL2 fallback.** If `navigator.gpu` is absent, the app renders a clear "WebGPU required" message and stops. Rationale: the brief is "rust-wasm-wgpu"; a second code path doubles scope.
- **No mobile-specific UI polish.** Desktop Chrome / Safari TP / Firefox Nightly with WebGPU is the target.
- **No multi-page / PDF ingest.** One image at a time.

## 3. Chosen model: CRAFT

We use **CRAFT** (Character Region Awareness for Text Detection, Baek et al. 2019) — specifically a pre-converted ONNX export of the `craft_mlt_25k` checkpoint.

Why CRAFT over alternatives:

| Candidate | Why not |
|---|---|
| DBNet / PP-OCRv4 detector | Outputs text-line regions, not per-character regions. Splitting lines back into characters on dense vertical cursive returns us to projection-profile failure modes. |
| Classical binarization + connected components | Identical failure class to the existing `OCR_Yi/segmentation.py`: touching strokes merge, red seal dots split, column boundaries blur. |
| SAM / MobileSAM | Not text-specialized; too large for a demo; requires prompts. |
| Train a custom U-Net on Yi manuscripts | Out of scope for a demo (requires labeled data). Kept as a future direction. |

CRAFT predicts two per-pixel heatmaps:

- **Region score** — probability each pixel is inside a character.
- **Affinity score** — probability each pixel is *between* two adjacent characters of the same word.

For a script without word-level grouping (each glyph is its own unit), we post-process using **region score alone**: threshold → connected components → minimum-area oriented bounding box per component. Affinity-based merging is intentionally skipped; each component = one character.

Input: RGB, letterboxed to a multiple of 32 on both sides (typical demo size: 1280×960 or whatever keeps aspect). Output: two H/2 × W/2 heatmaps.

## 4. Inference stack

Primary: **`wonnx`** — pure-Rust ONNX runtime on `wgpu`. Purpose-built for "ONNX on WebGPU from Rust, compile to WASM." Smallest dep surface, no JS bridge.

Fallback (if `wonnx` is missing a CRAFT op after verification): **`burn` with the `wgpu` backend** plus `burn-import` ONNX. Heavier binary, but broader op coverage.

Last-resort fallback (spec deviation): `ort` behind a wasm-bindgen shim to `onnxruntime-web`. Recorded here so that if *both* Rust paths fail, the deviation is visible; choosing it requires user sign-off.

Verification step (part of the plan's first task): load the chosen CRAFT ONNX and run a forward pass on a known input in a native Rust test before investing in the web harness. If `wonnx` rejects an op, switch to `burn` before building any UI.

## 5. Architecture

Cargo workspace at repo root:

```
docseg/
├── Cargo.toml                        # workspace
├── crates/
│   ├── docseg-core/                  # pure Rust, no web deps
│   │   └── src/
│   │       ├── lib.rs                # re-exports only
│   │       ├── preprocess.rs         # image -> normalized tensor
│   │       ├── model.rs              # load ONNX, run inference (wonnx)
│   │       ├── postprocess.rs        # heatmap -> polygons
│   │       ├── geometry.rs           # quad / polygon primitives
│   │       └── error.rs              # thiserror types
│   └── docseg-web/                   # wasm-bindgen glue
│       └── src/
│           ├── lib.rs                # re-exports only
│           ├── entry.rs              # #[wasm_bindgen] init / run / export
│           ├── canvas.rs             # image loading from HTMLImageElement / File
│           ├── render.rs             # draw overlays, handle clicks
│           └── export.rs             # zip of crops + boxes.json
├── web/
│   ├── index.html
│   ├── main.js                       # minimal bootstrap only (no inference, no build step)
│   └── style.css
├── models/
│   └── craft_mlt_25k.onnx            # git-lfs or .gitignore'd, fetched by script
└── docs/superpowers/specs/…
```

Boundaries:

- **`docseg-core`** knows nothing about the browser. Inputs: `image::DynamicImage` or raw `&[u8]` RGB. Outputs: `Vec<CharBox>` where `CharBox` is an oriented quadrilateral plus confidence. All heavy logic lives here and is covered by native `cargo test`.
- **`docseg-web`** is a thin adapter: decode image from `File` / `HTMLImageElement`, call `docseg-core`, paint canvas, hook click handlers, build export zip. No model logic.
- **`web/`** is bootstrap JS only: load the wasm bundle, wire file picker, call `docseg_web::run(imageBytes)`.

## 6. Data flow

```
[File input] ─▶ [decode in Rust/WASM]
             ─▶ [preprocess: letterbox + normalize → f32 NCHW tensor]   (docseg-core)
             ─▶ [wonnx forward on WebGPU → region_map, affinity_map]    (docseg-core)
             ─▶ [threshold region_map → connected components
                  → min-area oriented rect per component → CharBox[]]   (docseg-core)
             ─▶ [paint original + overlay polygons on <canvas>]         (docseg-web)
             ─▶ [click box → crop glyph → download PNG]                 (docseg-web)
             ─▶ ['Export all' → zip(crops/*.png, boxes.json)]           (docseg-web)
```

`boxes.json` schema:

```json
{
  "image": { "width": 979, "height": 1450 },
  "model": "craft_mlt_25k",
  "boxes": [
    {
      "id": 0,
      "quad": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
      "score": 0.87
    }
  ]
}
```

## 7. Error handling

Surfaces and expected failures:

| Surface | Failure | Behavior |
|---|---|---|
| `navigator.gpu` absent | No WebGPU | JS bootstrap detects before loading wasm; shows a blocking banner with the list of supported browsers. |
| Model fetch | Network / 404 | `docseg-core` returns `CoreError::ModelFetch { url, source }`; UI shows retry button. |
| ONNX parse / unsupported op | Invalid bundle | `CoreError::ModelLoad { source }`; surface the op name to the console. |
| Preprocess | Zero-size image, unreadable format | `CoreError::Decode { source }`. |
| Inference | OOM, adapter lost | `CoreError::Inference { source }`; suggest reducing input size. |
| Postprocess | No components above threshold | Not an error — returns empty `Vec<CharBox>`; UI shows "0 characters detected." |

`docseg-core` uses `thiserror`-typed errors (per workspace rule: no `anyhow` in libraries). `docseg-web` maps them to `JsError` at the wasm boundary. No `unwrap` / `expect` / `panic` in non-test code (enforced by workspace `deny`).

## 8. Testing strategy

- **Unit tests** in `src/<module>/tests.rs` with access to internals:
  - `preprocess`: letterbox math (pad values, aspect preservation) on fixed inputs.
  - `postprocess`: synthetic heatmap with known blobs → known polygon count / positions.
  - `geometry`: min-area-rect for simple point sets (square, rotated rectangle) with numeric tolerance.
- **Integration tests** in `crates/docseg-core/tests/`, public API only:
  - `end_to_end_on_fixture`: runs the full pipeline on a small checked-in fixture image (not the full manuscript — a 256×256 synthetic or cropped region) and asserts a stable glyph count within a tolerance band.
- **Web harness**: manual smoke test page loads in Chrome with WebGPU enabled; `docseg-web` exposes a `run_for_test` that returns the `boxes.json` string so we can diff it against a golden file in an optional Playwright job (not a CI blocker for the demo).
- **No model-quality regression tests.** The model is pre-trained and frozen.

## 9. Build / dev workflow

- Native: `cargo test -p docseg-core --all-targets`
- WASM build: `wasm-pack build crates/docseg-web --target web --release` → `web/pkg/`
- Dev server: `python3 -m http.server` in `web/` (no npm, no JS build step — `main.js` is plain ES modules loaded by the browser directly)
- Model fetch: `scripts/fetch-model.sh` downloads `craft_mlt_25k.onnx` (SHA-256 pinned) to `models/`. The file is not committed.
- Lints / CI gates (per workspace rules): `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --all-targets`, `cargo doc -D missing_docs`, `cargo build --release --target wasm32-unknown-unknown -p docseg-web`.

## 10. Workspace-level lints (from user global rules)

`Cargo.toml` `[workspace.lints.rust]` / `[workspace.lints.clippy]`:

- `forbid`: `unsafe_code`
- `deny`: `unwrap_used`, `expect_used`, `panic`, `let_underscore_must_use`, `let_underscore_future`, `await_holding_lock`, `missing_docs`
- `deny` on `disallowed_types` for `anyhow::Error` in library crates; `anyhow` is only allowed in any future binary `main()` — we currently have none.
- Tests allow `clippy::expect_used` / `clippy::panic`.

## 11. Open risks and their mitigations

| Risk | Mitigation |
|---|---|
| `wonnx` lacks an op CRAFT uses | Verified in the first plan task (native forward pass) before any UI is built. Fallback to `burn` is pre-approved. |
| CRAFT under-segments cursive vertical glyphs where affinity bridges are strong | We use region-only thresholding, not region+affinity. A threshold knob is exposed in the UI for manual tuning. |
| Red seal dots are detected as characters | Filter by a minimum component area and an aspect-ratio sanity check (both knobs exposed in UI). |
| WASM binary too large | Target `wasm-opt -Oz`; accept up to ~5 MB for the wasm itself, plus the ~20 MB model. |
| WebGPU adapter limits differ across browsers | Read `adapter.limits` and bail early with a clear message if the model's required buffer sizes exceed them. |

## 12. Deliverables

1. Rust workspace per §5 with the two crates, passing all CI gates in §9.
2. CRAFT forward-pass verification test (native) — passes before UI work starts.
3. Browser demo at `web/index.html` that loads `test_case1.png` by default, detects characters, renders overlays, and exports `crops/*.png` + `boxes.json`.
4. `README.md` with build / run / browser-support instructions.
