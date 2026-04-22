//! Convert CRAFT region-score heatmaps into per-character oriented boxes.
//!
//! This task covers the heatmap → connected-components step only; Task 8
//! extends it to produce `CharBox` entries with area/aspect filters and
//! original-image coordinate mapping.

use image::{GrayImage, Luma};
use imageproc::distance_transform::Norm;
use imageproc::morphology::erode;
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
    /// Pixels with affinity score ≥ this are "between two characters" and
    /// are masked out of the character region. Word-level CRAFT trained on
    /// horizontal Latin/MLT text produces strong affinity between
    /// horizontally adjacent glyphs; on vertical Chinese stacks this
    /// channel rarely fires where we need splits, so it's a soft signal
    /// here — the main vertical-split lever is `erosion_px`.
    pub affinity_threshold: f32,
    /// Binary-mask erosion radius in heatmap pixels applied before
    /// connected-component labelling. 1 is enough to break the thin
    /// activation bridges that otherwise merge vertically-touching glyphs
    /// in dense cursive columns; 0 disables. The erosion uses an L∞
    /// (chessboard) norm so both 4- and 8-connected bridges break at the
    /// same radius.
    pub erosion_px: u8,
    /// Drop components with fewer pixels than this (removes red-seal and
    /// speckle noise). Applied AFTER erosion.
    pub min_component_area_px: u32,
    /// Drop components whose oriented-rect aspect ratio (long/short) exceeds
    /// this — filters streaks from gutter ink.
    pub max_aspect_ratio: f32,
    /// If `true`, emit axis-aligned rectangles instead of min-area rotated
    /// quads. CRAFT's min-area rect tends to spuriously rotate around
    /// roughly-upright glyphs (producing visibly tilted overlays on vertical
    /// Chinese / Yi manuscripts); AABBs track the character shape more
    /// predictably. Set to `false` to opt back into rotated rects for scripts
    /// with genuinely slanted glyphs.
    pub axis_aligned: bool,
}

impl Default for PostprocessOptions {
    fn default() -> Self {
        Self {
            region_threshold: 0.4,
            affinity_threshold: 0.3,
            erosion_px: 0,
            min_component_area_px: 8,
            max_aspect_ratio: 8.0,
            axis_aligned: true,
        }
    }
}

/// Binarize the region heatmap with `opts.region_threshold`, optionally
/// subtract the affinity heatmap (pixels above `opts.affinity_threshold`
/// are masked out), and return each 4-connected component as a point list
/// in heatmap coordinates.
///
/// `affinity` is optional: pass `None` to disable inter-character masking
/// (the behavior from the original word-level CRAFT workflow). Pass
/// `Some(&affinity_map)` for character-level splitting — required to
/// prevent vertically-touching Chinese/Yi glyphs from merging into one
/// vertical blob.
///
/// Components are unfiltered; callers apply area/aspect filters later.
/// Returns an empty `Vec` for zero-size maps or when any slice's length
/// doesn't match `width * height`.
#[must_use]
pub fn components_from_heatmap(
    region: &[f32],
    affinity: Option<&[f32]>,
    width: u32,
    height: u32,
    opts: PostprocessOptions,
) -> Vec<Vec<(u32, u32)>> {
    let plane = (width as usize) * (height as usize);
    if width == 0 || height == 0 || region.len() != plane {
        return Vec::new();
    }
    if let Some(aff) = affinity {
        if aff.len() != plane {
            return Vec::new();
        }
    }
    let mut binary = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y as usize) * (width as usize) + (x as usize);
            let in_region = region[idx] >= opts.region_threshold;
            let is_link = affinity
                .map(|a| a[idx] >= opts.affinity_threshold)
                .unwrap_or(false);
            let v = if in_region && !is_link { 255 } else { 0 };
            binary.put_pixel(x, y, Luma([v]));
        }
    }
    if opts.erosion_px > 0 {
        binary = erode(&binary, Norm::LInf, opts.erosion_px);
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

/// Full heatmap → `CharBox` pipeline: threshold (region ∧ ¬affinity),
/// connected-components, area filter, min-area quad fit, aspect filter,
/// map back to original-image coordinates using the `PreprocessOutput`
/// transform.
///
/// `affinity` is optional — see [`components_from_heatmap`] for details.
/// For Chinese / Yi / other vertically-stacked cursive scripts, pass
/// `Some(&affinity_map)` to split touching characters.
#[must_use]
pub fn charboxes_from_heatmap(
    region: &[f32],
    affinity: Option<&[f32]>,
    heatmap_w: u32,
    heatmap_h: u32,
    preproc: &PreprocessOutput,
    opts: PostprocessOptions,
) -> Vec<CharBox> {
    let comps = components_from_heatmap(region, affinity, heatmap_w, heatmap_h, opts);
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
            if let Some(v) = region.get(idx) {
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
        let quad = if opts.axis_aligned {
            match aabb_quad(&pts) {
                Some(q) => q,
                None => continue,
            }
        } else {
            match min_area_quad(&pts) {
                Some(q) => q,
                None => continue,
            }
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

fn aabb_quad(pts: &[Point]) -> Option<Quad> {
    if pts.is_empty() {
        return None;
    }
    let (mut xmin, mut ymin) = (f32::INFINITY, f32::INFINITY);
    let (mut xmax, mut ymax) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in pts {
        if p.x < xmin {
            xmin = p.x;
        }
        if p.x > xmax {
            xmax = p.x;
        }
        if p.y < ymin {
            ymin = p.y;
        }
        if p.y > ymax {
            ymax = p.y;
        }
    }
    if xmax <= xmin || ymax <= ymin {
        return None;
    }
    Some(Quad::new([
        Point::new(xmin, ymin),
        Point::new(xmax, ymin),
        Point::new(xmax, ymax),
        Point::new(xmin, ymax),
    ]))
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
