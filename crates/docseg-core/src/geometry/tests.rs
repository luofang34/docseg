#![allow(clippy::expect_used, clippy::panic)]

use super::{min_area_quad, Point, Quad};

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn axis_aligned_square_returns_same_square() {
    let pts = vec![
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        Point::new(10.0, 10.0),
        Point::new(0.0, 10.0),
        Point::new(5.0, 5.0),
    ];
    let q = min_area_quad(&pts).expect("quad");
    // All four corners of a 10x10 square should appear (in some order).
    let mut xs: Vec<f32> = q.points.iter().map(|p| p.x).collect();
    let mut ys: Vec<f32> = q.points.iter().map(|p| p.y).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    ys.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    assert!(approx_eq(xs[0], 0.0, 1e-3));
    assert!(approx_eq(xs[3], 10.0, 1e-3));
    assert!(approx_eq(ys[0], 0.0, 1e-3));
    assert!(approx_eq(ys[3], 10.0, 1e-3));
}

#[test]
fn rotated_square_returns_rotated_quad_with_same_area() {
    // 45° rotated square, corners at (1,0)(0,1)(-1,0)(0,-1); area = 2.
    let pts = vec![
        Point::new(1.0, 0.0),
        Point::new(0.0, 1.0),
        Point::new(-1.0, 0.0),
        Point::new(0.0, -1.0),
    ];
    let q = min_area_quad(&pts).expect("quad");
    assert!(approx_eq(q.area(), 2.0, 1e-3), "area {}", q.area());
}

#[test]
fn quad_area_for_unit_square_is_one() {
    let q = Quad::new([
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 1.0),
    ]);
    assert!(approx_eq(q.area(), 1.0, 1e-6));
}

#[test]
fn empty_input_returns_none() {
    let pts: Vec<Point> = vec![];
    assert!(min_area_quad(&pts).is_none());
}

#[test]
fn single_point_returns_none_or_zero_area_quad() {
    // `geo`'s MinimumRotatedRect may return None for degenerate cases — the
    // caller already treats None as "skip this component". Either outcome is
    // acceptable as long as it doesn't panic and doesn't report spurious area.
    let pts = vec![Point::new(5.0, 5.0)];
    match min_area_quad(&pts) {
        Some(q) => assert!(q.area() < 1e-6, "degenerate quad area {}", q.area()),
        None => { /* acceptable */ }
    }
}
