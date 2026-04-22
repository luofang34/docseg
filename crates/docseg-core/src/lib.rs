//! `docseg-core` — per-character document segmentation.
//!
//! Takes a decoded page image, returns oriented quadrilaterals around each
//! detected glyph. Target deployment is a browser via WASM, but this crate
//! is pure Rust with no web dependencies so it is natively testable.

#![deny(missing_docs)]

pub mod error;

pub use error::CoreError;
