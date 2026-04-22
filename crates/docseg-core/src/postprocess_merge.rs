//! Merge user-edited (manual) CharBoxes with the auto-detected set,
//! using an axis-aligned IoU > 0.5 rule so a user correction cannot be
//! silently duplicated by a slider-driven re-detection.

use crate::geometry::Quad;
use crate::postprocess::CharBox;

/// Axis-aligned IoU on the bounding rect of two `CharBox`es. 0.0 when
/// either box has zero area (degenerate).
#[must_use]
pub fn iou_aabb(a: &CharBox, b: &CharBox) -> f32 {
    let (ax0, ay0, ax1, ay1) = aabb(&a.quad);
    let (bx0, by0, bx1, by1) = aabb(&b.quad);
    let iw = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let ih = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let inter = iw * ih;
    let a_area = (ax1 - ax0).max(0.0) * (ay1 - ay0).max(0.0);
    let b_area = (bx1 - bx0).max(0.0) * (by1 - by0).max(0.0);
    let union = a_area + b_area - inter;
    if union <= 0.0 {
        return 0.0;
    }
    inter / union
}

/// Build the final CharBox list: every manual box survives, and every
/// auto-detected box passes through UNLESS its IoU with some manual
/// box exceeds 0.5.
#[must_use]
pub fn merge_manual_with_auto(auto: &[CharBox], manual: &[CharBox]) -> Vec<CharBox> {
    let mut out: Vec<CharBox> = Vec::with_capacity(manual.len() + auto.len());
    out.extend(manual.iter().cloned());
    for a in auto {
        if manual.iter().any(|m| iou_aabb(m, a) > 0.5) {
            continue;
        }
        out.push(a.clone());
    }
    out
}

fn aabb(q: &Quad) -> (f32, f32, f32, f32) {
    let p = &q.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    (
        xs.iter().copied().fold(f32::INFINITY, f32::min),
        ys.iter().copied().fold(f32::INFINITY, f32::min),
        xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    )
}

#[cfg(test)]
mod tests;
