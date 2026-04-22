//! Infer a linear reading order over a list of `CharBox`es.
//!
//! For vertical-RTL (default on Yi / classical Chinese manuscripts) we
//! cluster boxes by x-center into columns, order columns right-to-left,
//! and sort boxes within each column top-to-bottom. The column clustering
//! is a 1-D single-pass agglomeration that walks x-centers in sorted order
//! and starts a new cluster whenever the gap to the previous center exceeds
//! a fraction of the median box width.

use crate::postprocess::CharBox;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Reading direction. Determines both how boxes cluster into rows/columns
/// and how those groups are ordered.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum ReadingDirection {
    /// Columns read top-to-bottom; columns ordered right-to-left (classical
    /// Chinese, Yi, traditional Japanese).
    #[default]
    VerticalRtl,
    /// Columns read top-to-bottom; columns ordered left-to-right (rarely
    /// used; included for completeness).
    VerticalLtr,
    /// Lines read left-to-right; lines ordered top-to-bottom (Latin, modern
    /// horizontal CJK).
    HorizontalLtr,
}

/// Return the indices into `boxes` in reading order under `direction`.
///
/// Empty input returns an empty vec. Output always has the same length as
/// the input (every box appears exactly once).
#[must_use]
pub fn compute_reading_order(boxes: &[CharBox], direction: ReadingDirection) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    match direction {
        ReadingDirection::VerticalRtl | ReadingDirection::VerticalLtr => {
            vertical_order(boxes, direction == ReadingDirection::VerticalRtl)
        }
        ReadingDirection::HorizontalLtr => horizontal_order(boxes),
    }
}

fn vertical_order(boxes: &[CharBox], rtl: bool) -> Vec<usize> {
    let centers: Vec<(f32, f32)> = boxes.iter().map(box_center).collect();
    let widths: Vec<f32> = boxes.iter().map(box_width).collect();
    let gap_threshold = median_positive(&widths) * 0.6;

    // Sort original indices by x, so the cluster walk encounters centers in ascending x.
    let mut by_x: Vec<usize> = (0..boxes.len()).collect();
    by_x.sort_by(|&a, &b| {
        centers[a]
            .0
            .partial_cmp(&centers[b].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Agglomerate into columns using the gap threshold.
    let mut columns: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut prev_x: f32 = f32::NEG_INFINITY;
    for &idx in &by_x {
        let cx = centers[idx].0;
        if !current.is_empty() && cx - prev_x > gap_threshold {
            columns.push(std::mem::take(&mut current));
        }
        current.push(idx);
        prev_x = cx;
    }
    if !current.is_empty() {
        columns.push(current);
    }

    if rtl {
        columns.reverse();
    }

    let mut order = Vec::with_capacity(boxes.len());
    for col in &mut columns {
        col.sort_by(|&a, &b| {
            centers[a]
                .1
                .partial_cmp(&centers[b].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.extend(col.iter().copied());
    }
    order
}

fn horizontal_order(boxes: &[CharBox]) -> Vec<usize> {
    let centers: Vec<(f32, f32)> = boxes.iter().map(box_center).collect();
    let heights: Vec<f32> = boxes.iter().map(box_height).collect();
    let gap_threshold = median_positive(&heights) * 0.6;

    let mut by_y: Vec<usize> = (0..boxes.len()).collect();
    by_y.sort_by(|&a, &b| {
        centers[a]
            .1
            .partial_cmp(&centers[b].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut prev_y: f32 = f32::NEG_INFINITY;
    for &idx in &by_y {
        let cy = centers[idx].1;
        if !current.is_empty() && cy - prev_y > gap_threshold {
            lines.push(std::mem::take(&mut current));
        }
        current.push(idx);
        prev_y = cy;
    }
    if !current.is_empty() {
        lines.push(current);
    }

    let mut order = Vec::with_capacity(boxes.len());
    for line in &mut lines {
        line.sort_by(|&a, &b| {
            centers[a]
                .0
                .partial_cmp(&centers[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.extend(line.iter().copied());
    }
    order
}

fn box_center(b: &CharBox) -> (f32, f32) {
    let p = &b.quad.points;
    (
        (p[0].x + p[1].x + p[2].x + p[3].x) * 0.25,
        (p[0].y + p[1].y + p[2].y + p[3].y) * 0.25,
    )
}

fn box_width(b: &CharBox) -> f32 {
    let p = &b.quad.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    let xmin = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let xmax = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (xmax - xmin).abs()
}

fn box_height(b: &CharBox) -> f32 {
    let p = &b.quad.points;
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    let ymin = ys.iter().copied().fold(f32::INFINITY, f32::min);
    let ymax = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (ymax - ymin).abs()
}

/// Median of the strictly-positive values. Returns 0 if `vals` has no
/// positive entries.
fn median_positive(vals: &[f32]) -> f32 {
    let mut positive: Vec<f32> = vals.iter().copied().filter(|v| *v > 0.0).collect();
    if positive.is_empty() {
        return 0.0;
    }
    positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    positive[positive.len() / 2]
}

use crate::regions::{region_for_box, Region};

/// Compute reading order taking layout regions into account. Boxes are
/// grouped by region (boxes outside every region fall into an implicit
/// "Body" group at rank 2). Groups are ordered by region rank; within
/// each group the existing direction-dependent column/line clustering
/// is applied.
#[must_use]
pub fn compute_reading_order_with_regions(
    boxes: &[CharBox],
    regions: &[Region],
    direction: ReadingDirection,
) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    if regions.is_empty() {
        return compute_reading_order(boxes, direction);
    }

    // Group indices by (rank, region_id). Implicit Body uses rank 2 and
    // a sentinel region id of u32::MAX so it sorts after same-rank drawn
    // regions — acceptable ambiguity for v1.
    let mut grouped: std::collections::BTreeMap<(u32, u32), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, b) in boxes.iter().enumerate() {
        let key = match region_for_box(b, regions) {
            Some(rid) => {
                let rank = regions
                    .iter()
                    .find(|r| r.id == rid)
                    .map(|r| r.rank)
                    .unwrap_or(2);
                (rank, rid)
            }
            None => (2, u32::MAX),
        };
        grouped.entry(key).or_default().push(i);
    }

    let mut out = Vec::with_capacity(boxes.len());
    for ((_rank, _rid), idxs) in grouped {
        let subset: Vec<CharBox> = idxs.iter().map(|&i| boxes[i].clone()).collect();
        let sub_order = compute_reading_order(&subset, direction);
        for rank_in_sub in sub_order {
            out.push(idxs[rank_in_sub]);
        }
    }
    out
}

#[cfg(test)]
mod tests;
