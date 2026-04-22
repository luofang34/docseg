//! Layout regions on a page. v1 uses axis-aligned rectangles; the
//! `RegionShape::Polygon` variant is API-ready so a future auto-layout
//! detector or the v2 polygon tool can emit polygons without breaking
//! callers.

use crate::geometry::Point;
use crate::postprocess::CharBox;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Semantic role of a layout region. Determines overlay color and the
/// default rank (Header < Body < Footer < Notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RegionRole {
    /// Header / rubric / title — read first.
    Header,
    /// Main body text.
    Body,
    /// Footer / colophon.
    Footer,
    /// Marginal notes / annotations.
    Notes,
}

/// Region geometry.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "lowercase"))]
pub enum RegionShape {
    /// Axis-aligned rectangle in original-image coordinates.
    Rect {
        /// Left edge.
        xmin: f32,
        /// Top edge.
        ymin: f32,
        /// Right edge (exclusive).
        xmax: f32,
        /// Bottom edge (exclusive).
        ymax: f32,
    },
    /// Closed polygon in original-image coordinates. Not exposed in v1
    /// UI; the API accepts it so callers that build regions outside the
    /// interactive tool (e.g. a future auto-detector) don't need an API
    /// bump.
    Polygon(Vec<Point>),
}

impl RegionShape {
    /// `true` iff `p` is inside the shape. Half-open on the right/bottom
    /// edges for Rect; standard even-odd fill for Polygon.
    #[must_use]
    pub fn contains(&self, p: Point) -> bool {
        match self {
            Self::Rect {
                xmin,
                ymin,
                xmax,
                ymax,
            } => p.x >= *xmin && p.x < *xmax && p.y >= *ymin && p.y < *ymax,
            Self::Polygon(pts) => polygon_contains(pts, p),
        }
    }
}

/// A layout region as stored on a `Page`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Region {
    /// Stable id within a page.
    pub id: u32,
    /// Geometry.
    pub shape: RegionShape,
    /// Semantic role (drives overlay color + default rank).
    pub role: RegionRole,
    /// Reading-order priority; lower ranks read earlier.
    pub rank: u32,
}

/// Return the id of the region whose shape contains the centroid of
/// `b.quad`. If multiple regions contain the centroid, the one with
/// the lowest `rank` wins; ties broken by lowest `id`. `None` when no
/// region matches.
#[must_use]
pub fn region_for_box(b: &CharBox, regions: &[Region]) -> Option<u32> {
    let p = &b.quad.points;
    let cx = (p[0].x + p[1].x + p[2].x + p[3].x) * 0.25;
    let cy = (p[0].y + p[1].y + p[2].y + p[3].y) * 0.25;
    let centroid = Point::new(cx, cy);
    regions
        .iter()
        .filter(|r| r.shape.contains(centroid))
        .min_by(|a, b| a.rank.cmp(&b.rank).then_with(|| a.id.cmp(&b.id)))
        .map(|r| r.id)
}

/// Even-odd fill test for a closed polygon. `pts` is treated as a
/// closed loop (last point implicitly connects to first).
fn polygon_contains(pts: &[Point], p: Point) -> bool {
    if pts.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = pts.len() - 1;
    for i in 0..pts.len() {
        let pi = pts[i];
        let pj = pts[j];
        if (pi.y > p.y) != (pj.y > p.y) && p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests;
