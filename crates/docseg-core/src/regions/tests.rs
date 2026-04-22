#![allow(clippy::expect_used, clippy::panic)]

use super::{region_for_box, Region, RegionRole, RegionShape};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;

fn rect_box(cx: f32, cy: f32) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(cx - 5.0, cy - 5.0),
            Point::new(cx + 5.0, cy - 5.0),
            Point::new(cx + 5.0, cy + 5.0),
            Point::new(cx - 5.0, cy + 5.0),
        ]),
        score: 1.0,
        manual: false,
    }
}

fn header_rect(x0: f32, y0: f32, x1: f32, y1: f32, rank: u32) -> Region {
    Region {
        id: 1,
        shape: RegionShape::Rect {
            xmin: x0,
            ymin: y0,
            xmax: x1,
            ymax: y1,
        },
        role: RegionRole::Header,
        rank,
    }
}

#[test]
fn rect_contains_interior_point() {
    let shape = RegionShape::Rect {
        xmin: 0.0,
        ymin: 0.0,
        xmax: 100.0,
        ymax: 100.0,
    };
    assert!(shape.contains(Point::new(50.0, 50.0)));
}

#[test]
fn rect_excludes_exterior_point() {
    let shape = RegionShape::Rect {
        xmin: 0.0,
        ymin: 0.0,
        xmax: 100.0,
        ymax: 100.0,
    };
    assert!(!shape.contains(Point::new(150.0, 50.0)));
}

#[test]
fn rect_edge_counts_as_inside() {
    // Half-open interval [xmin, xmax); top-left edge inside, bottom-right outside.
    let shape = RegionShape::Rect {
        xmin: 0.0,
        ymin: 0.0,
        xmax: 10.0,
        ymax: 10.0,
    };
    assert!(shape.contains(Point::new(0.0, 0.0)));
    assert!(!shape.contains(Point::new(10.0, 10.0)));
}

#[test]
fn region_for_box_returns_lowest_rank_on_overlap() {
    let inner = Region {
        id: 1,
        shape: RegionShape::Rect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 100.0,
            ymax: 100.0,
        },
        role: RegionRole::Header,
        rank: 1,
    };
    let outer = Region {
        id: 2,
        shape: RegionShape::Rect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 200.0,
            ymax: 200.0,
        },
        role: RegionRole::Body,
        rank: 2,
    };
    let regions = vec![outer, inner]; // order in slice doesn't matter
    let b = rect_box(50.0, 50.0);
    assert_eq!(region_for_box(&b, &regions), Some(1));
}

#[test]
fn region_for_box_returns_none_outside_all_regions() {
    let r = header_rect(0.0, 0.0, 50.0, 50.0, 1);
    let b = rect_box(100.0, 100.0);
    assert_eq!(region_for_box(&b, &[r]), None);
}

#[test]
fn region_for_box_with_empty_regions_returns_none() {
    let b = rect_box(10.0, 10.0);
    assert_eq!(region_for_box(&b, &[]), None);
}

#[test]
fn polygon_contains_concave_hook() {
    // An L-shaped polygon: (0,0)->(10,0)->(10,4)->(4,4)->(4,10)->(0,10)->(0,0).
    let poly = RegionShape::Polygon(vec![
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        Point::new(10.0, 4.0),
        Point::new(4.0, 4.0),
        Point::new(4.0, 10.0),
        Point::new(0.0, 10.0),
    ]);
    assert!(poly.contains(Point::new(2.0, 2.0))); // in the thick part
    assert!(poly.contains(Point::new(6.0, 2.0))); // in the top-bar right arm
    assert!(!poly.contains(Point::new(6.0, 6.0))); // in the L's interior notch (outside)
    assert!(!poly.contains(Point::new(12.0, 2.0))); // outside
}
