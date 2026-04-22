//! Planar geometry primitives used by postprocess.
//!
//! We lean on `geo`'s `MinimumRotatedRect` for the rotating-calipers
//! algorithm rather than rolling our own — it is a small additional
//! dependency and saves a known-tricky implementation.

use geo::algorithm::minimum_rotated_rect::MinimumRotatedRect;
use geo::{Coord, MultiPoint, Polygon};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 2D point with `f32` coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point {
    /// x coordinate.
    pub x: f32,
    /// y coordinate.
    pub y: f32,
}

impl Point {
    /// Construct a point from components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Oriented quadrilateral. Points are stored in an arbitrary but consistent
/// order — downstream consumers may normalize to CW from top-left if needed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Quad {
    /// Four corner points.
    pub points: [Point; 4],
}

impl Quad {
    /// Construct a quad from four points.
    #[must_use]
    pub const fn new(points: [Point; 4]) -> Self {
        Self { points }
    }

    /// Shoelace-formula area of the quad.
    #[must_use]
    pub fn area(&self) -> f32 {
        let p = &self.points;
        let sum = p[0].x * p[1].y - p[1].x * p[0].y + p[1].x * p[2].y - p[2].x * p[1].y
            + p[2].x * p[3].y
            - p[3].x * p[2].y
            + p[3].x * p[0].y
            - p[0].x * p[3].y;
        (sum * 0.5).abs()
    }
}

/// Compute the minimum-area rotated rectangle enclosing `points`.
///
/// Returns `None` when the input is empty or degenerate (all collinear /
/// single point).
#[must_use]
pub fn min_area_quad(points: &[Point]) -> Option<Quad> {
    if points.is_empty() {
        return None;
    }
    let multipoint: MultiPoint<f64> = points
        .iter()
        .map(|p| Coord {
            x: f64::from(p.x),
            y: f64::from(p.y),
        })
        .collect::<Vec<_>>()
        .into();
    let poly: Polygon<f64> = multipoint.minimum_rotated_rect()?;
    let ring = poly.exterior();
    // `geo` returns a closed ring (first == last), so 5 entries = 4 distinct corners.
    if ring.0.len() < 5 {
        return None;
    }
    let mut corners = [Point::new(0.0, 0.0); 4];
    for (i, c) in ring.0.iter().take(4).enumerate() {
        corners[i] = Point::new(c.x as f32, c.y as f32);
    }
    Some(Quad::new(corners))
}

#[cfg(test)]
mod tests;
