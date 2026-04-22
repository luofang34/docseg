//! Batch + Page state model. Pure data types; persistence lives in
//! `batch_persist`.

use sha2::{Digest, Sha256};
use ulid::Ulid;

use crate::edit_log::EditLog;
use crate::postprocess::CharBox;
use crate::regions::Region;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Bump whenever Batch / Page / SliderValues shape changes in a way
/// older deserializers couldn't auto-default. Migration chain in
/// `batch_persist::migrate`.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Effective slider values for a page (or the session defaults on a
/// Batch).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SliderValues {
    /// CRAFT region-score threshold.
    pub region_threshold: f32,
    /// CRAFT affinity-score threshold.
    pub affinity_threshold: f32,
    /// Morphological erosion radius in heatmap pixels.
    pub erosion_px: u8,
    /// Minimum connected-component area in heatmap pixels.
    pub min_component_area_px: u32,
    /// `true` for axis-aligned box fit; `false` for min-area rotated rect.
    pub axis_aligned: bool,
}

impl Default for SliderValues {
    fn default() -> Self {
        Self {
            region_threshold: 0.4,
            affinity_threshold: 0.3,
            erosion_px: 0,
            min_component_area_px: 8,
            axis_aligned: true,
        }
    }
}

/// Typed alias for the Batch's "sticky" slider defaults. Same shape as
/// a page's `SliderValues`; kept distinct so methods that mutate it
/// don't get called on the wrong struct.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SliderDefaults {
    /// The sticky slider snapshot. `None` = use `SliderValues::default()`.
    pub values: Option<SliderValues>,
}

/// Review status of a page in a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PageStatus {
    /// Never opened.
    #[default]
    Untouched,
    /// Opened, has edits, not yet marked reviewed.
    InProgress,
    /// User pressed "Mark reviewed."
    Reviewed,
    /// User pressed F (or import flagged it, e.g. ImageDrift).
    Flagged,
}

/// One page in a batch. Owns its image bytes (kept until explicit
/// save, then offloaded to sidecar on disk).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Page {
    /// Stable id across the batch session (ULID string).
    pub id: String,
    /// Full original image bytes (PNG/JPEG). Not serialized — lives in
    /// sidecar `images/page_NNNN.bin`. See `batch_persist`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub image_bytes: Vec<u8>,
    /// SHA-256 of `image_bytes`, used for drift detection.
    pub image_sha256: [u8; 32],
    /// `(width, height)` once the image has been decoded. (0, 0) before
    /// first preprocess.
    pub image_dims: (u32, u32),
    /// Current review status.
    pub status: PageStatus,
    /// Effective slider values (either inherited from batch defaults or
    /// page-specific).
    pub sliders: SliderValues,
    /// User's (and auto-detected) boxes.
    pub boxes: Vec<CharBox>,
    /// Drawn regions.
    pub regions: Vec<Region>,
    /// Reading order (indices into `boxes`).
    pub order: Vec<u32>,
    /// Undo / redo stack.
    pub edit_log: EditLog,
    /// Unix epoch seconds when the page was last marked Reviewed.
    pub reviewed_at: Option<i64>,
}

impl Page {
    /// Build a fresh Untouched page from raw image bytes. `image_dims`
    /// is set lazily by the caller after first decode.
    #[must_use]
    pub fn new(image_bytes: &[u8]) -> Self {
        let digest = Sha256::digest(image_bytes);
        let mut sha = [0u8; 32];
        sha.copy_from_slice(&digest);
        Self {
            id: Ulid::new().to_string(),
            image_bytes: image_bytes.to_vec(),
            image_sha256: sha,
            image_dims: (0, 0),
            status: PageStatus::Untouched,
            sliders: SliderValues::default(),
            boxes: Vec::new(),
            regions: Vec::new(),
            order: Vec::new(),
            edit_log: EditLog::new(),
            reviewed_at: None,
        }
    }

    /// Transition the page's status in response to a user edit:
    /// Untouched → InProgress, Reviewed → InProgress (edits re-open
    /// review), everything else unchanged.
    pub fn mark_edited(&mut self) {
        match self.status {
            PageStatus::Untouched | PageStatus::Reviewed => {
                self.status = PageStatus::InProgress;
            }
            PageStatus::InProgress | PageStatus::Flagged => {}
        }
    }
}

/// Top-level batch object.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Batch {
    /// Batch ULID.
    pub id: String,
    /// Schema version. See `CURRENT_SCHEMA_VERSION`.
    pub schema_version: u32,
    /// Pages in user-visible order.
    pub pages: Vec<Page>,
    /// Sticky slider values — populated whenever the user moves a
    /// slider, consumed when opening an Untouched page.
    pub session_defaults: SliderDefaults,
    /// Unix epoch seconds.
    pub created_at: i64,
    /// Unix epoch seconds.
    pub updated_at: i64,
}

impl Batch {
    /// Create an empty batch with `CURRENT_SCHEMA_VERSION`.
    #[must_use]
    pub fn new() -> Self {
        let now = now_epoch_seconds();
        Self {
            id: Ulid::new().to_string(),
            schema_version: CURRENT_SCHEMA_VERSION,
            pages: Vec::new(),
            session_defaults: SliderDefaults::default(),
            created_at: now,
            updated_at: now,
        }
    }

    /// `true` if no slider defaults have been set yet.
    #[must_use]
    pub fn session_defaults_empty(&self) -> bool {
        self.session_defaults.values.is_none()
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn now_epoch_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn now_epoch_seconds() -> i64 {
    // SystemTime isn't available on wasm32-unknown-unknown; the web
    // crate passes the JS-side Date.now() in if needed, but core just
    // returns 0 so core-level tests don't need to stub a clock.
    0
}

#[cfg(test)]
mod tests;
