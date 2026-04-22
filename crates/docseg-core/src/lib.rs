//! `docseg-core` — per-character document segmentation.
//!
//! Takes a decoded page image, returns oriented quadrilaterals around each
//! detected glyph. Target deployment is a browser via WASM, but this crate
//! is pure Rust with no web dependencies so it is natively testable.
//!
//! The `model` module (wonnx-based native inference) is compiled out on
//! `wasm32` targets — the browser inference path lives in `docseg-web`.

pub mod batch;
pub mod diff;
pub mod edit_log;
pub mod error;
pub mod geometry;

#[cfg(not(target_arch = "wasm32"))]
pub mod model;

pub mod postprocess;
pub mod postprocess_merge;
pub mod preprocess;
pub mod reading_order;
pub mod regions;

pub use batch::{Batch, Page, PageStatus, SliderDefaults, SliderValues, CURRENT_SCHEMA_VERSION};
pub use diff::{compute_diff, DiffEntry};
pub use edit_log::{EditEvent, EditLog};
pub use error::CoreError;
pub use geometry::{min_area_quad, Point, Quad};
#[cfg(not(target_arch = "wasm32"))]
pub use model::{CraftSession, RegionMap};
pub use postprocess::{
    charboxes_from_heatmap, components_from_heatmap, CharBox, PostprocessOptions,
};
pub use postprocess_merge::{iou_aabb, merge_manual_with_auto};
pub use preprocess::{preprocess, PreprocessOptions, PreprocessOutput};
pub use reading_order::{
    compute_reading_order, compute_reading_order_with_regions, ReadingDirection,
};
pub use regions::{region_for_box, Region, RegionRole, RegionShape};
