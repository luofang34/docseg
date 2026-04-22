//! Convert CRAFT region-score heatmaps into per-character oriented boxes.
//!
//! This task covers the heatmap → connected-components step only; Task 8
//! extends it to produce `CharBox` entries with area/aspect filters and
//! original-image coordinate mapping.

use image::{GrayImage, Luma};
use imageproc::region_labelling::{connected_components, Connectivity};

use crate::geometry::{min_area_quad, Point, Quad};
use crate::preprocess::PreprocessOutput;

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

/// One detected glyph: its oriented box in original-image coordinates plus a
/// confidence score (the peak region-score value within the component).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CharBox {
    /// Oriented quadrilateral in original-image pixel coordinates.
    pub quad: Quad,
    /// Max region-score within the component (0..1).
    pub score: f32,
}

/// Full heatmap → `CharBox` pipeline: threshold, connected-components, area
/// filter, min-area quad fit, aspect filter, map back to original-image
/// coordinates using the `PreprocessOutput` transform.
#[must_use]
pub fn charboxes_from_heatmap(
    map: &[f32],
    heatmap_w: u32,
    heatmap_h: u32,
    preproc: &PreprocessOutput,
    opts: PostprocessOptions,
) -> Vec<CharBox> {
    let comps = components_from_heatmap(map, heatmap_w, heatmap_h, opts);
    if comps.is_empty() || heatmap_w == 0 || heatmap_h == 0 || preproc.scale <= 0.0 {
        return Vec::new();
    }

    let (padded_w, padded_h) = preproc.padded_size;
    // Scale from heatmap space → padded-input space.
    let map_to_padded_x = padded_w as f32 / heatmap_w as f32;
    let map_to_padded_y = padded_h as f32 / heatmap_h as f32;
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
            if let Some(v) = map.get(idx) {
                if *v > score {
                    score = *v;
                }
            }
            // Each heatmap pixel covers a (map_to_padded_x × map_to_padded_y)
            // cell in padded-input space; use the cell centroid.
            let px_padded = ((*x as f32) + 0.5) * map_to_padded_x;
            let py_padded = ((*y as f32) + 0.5) * map_to_padded_y;
            let ox = (px_padded - pad_ox as f32) * inv_scale;
            let oy = (py_padded - pad_oy as f32) * inv_scale;
            pts.push(Point::new(ox, oy));
        }
        let Some(quad) = min_area_quad(&pts) else {
            continue;
        };
        if quad_aspect(&quad) > opts.max_aspect_ratio {
            continue;
        }
        if !quad_centroid_inside(&quad, preproc.original_size) {
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

fn quad_centroid_inside(q: &Quad, (w, h): (u32, u32)) -> bool {
    let cx = (q.points[0].x + q.points[1].x + q.points[2].x + q.points[3].x) / 4.0;
    let cy = (q.points[0].y + q.points[1].y + q.points[2].y + q.points[3].y) / 4.0;
    cx >= 0.0 && cx < w as f32 && cy >= 0.0 && cy < h as f32
}

#[cfg(test)]
mod tests;
