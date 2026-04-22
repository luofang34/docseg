//! Convert CRAFT region-score heatmaps into per-character oriented boxes.
//!
//! This task covers the heatmap → connected-components step only; Task 8
//! extends it to produce `CharBox` entries with area/aspect filters and
//! original-image coordinate mapping.

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
    /// Drop components with fewer pixels than this (removes red-seal and
    /// speckle noise). Consumed by Task 8's `CharBox` extraction.
    pub min_component_area_px: u32,
    /// Drop components whose oriented-rect aspect ratio (long/short) exceeds
    /// this — filters streaks from gutter ink. Consumed by Task 8.
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
/// Returns an empty `Vec` for zero-size maps or when `map.len()` doesn't
/// match `width * height` (caller misuse is quiet rather than panicking —
/// the workspace bans panics outside tests).
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
            let v = if map[idx] >= opts.region_threshold {
                255
            } else {
                0
            };
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
