//! `docseg-web` — thin wasm-bindgen adapter over `docseg-core`.
//!
//! Responsibilities: decode image bytes, fetch the model bundle, drive the
//! core pipeline, render overlays on an HTMLCanvasElement, and produce a
//! zip of per-glyph crops on demand. No model logic lives here — inference
//! wiring lands in Task 12.

mod canvas;
mod entry;
mod export;
mod render;

pub use entry::DocsegApp;
