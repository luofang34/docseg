#![allow(clippy::expect_used, clippy::panic)]

use super::{iou_aabb, merge_manual_with_auto};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;

fn rect(x: f32, y: f32, w: f32, h: f32, manual: bool) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ]),
        score: 1.0,
        manual,
    }
}

#[test]
fn iou_identical_rects_is_one() {
    let a = rect(0.0, 0.0, 10.0, 10.0, false);
    let b = rect(0.0, 0.0, 10.0, 10.0, false);
    assert!((iou_aabb(&a, &b) - 1.0).abs() < 1e-4);
}

#[test]
fn iou_disjoint_rects_is_zero() {
    let a = rect(0.0, 0.0, 10.0, 10.0, false);
    let b = rect(100.0, 100.0, 10.0, 10.0, false);
    assert!(iou_aabb(&a, &b).abs() < 1e-6);
}

#[test]
fn iou_half_overlap() {
    // 10x10 at (0,0) and 10x10 at (5,0): overlap 5x10 = 50, union 10x10 + 10x10 - 50 = 150.
    let a = rect(0.0, 0.0, 10.0, 10.0, false);
    let b = rect(5.0, 0.0, 10.0, 10.0, false);
    let v = iou_aabb(&a, &b);
    assert!((v - 50.0 / 150.0).abs() < 1e-4, "iou={v}");
}

#[test]
fn merge_drops_auto_box_overlapping_manual_box() {
    let manual = vec![rect(0.0, 0.0, 10.0, 10.0, true)];
    let auto = vec![
        rect(1.0, 1.0, 9.0, 9.0, false), // IoU with manual > 0.5 → dropped
        rect(100.0, 100.0, 10.0, 10.0, false), // disjoint → kept
    ];
    let merged = merge_manual_with_auto(&auto, &manual);
    assert_eq!(merged.len(), 2, "expected manual + one survivor auto");
    assert!(merged[0].manual, "manual box first");
    assert!(!merged[1].manual, "auto-survivor second");
    let xs: Vec<f32> = merged.iter().map(|b| b.quad.points[0].x).collect();
    assert!(xs.contains(&0.0));
    assert!(xs.contains(&100.0));
}

#[test]
fn merge_preserves_manual_flag_through_merge() {
    let manual = vec![rect(0.0, 0.0, 10.0, 10.0, true)];
    let auto: Vec<CharBox> = vec![];
    let merged = merge_manual_with_auto(&auto, &manual);
    assert_eq!(merged.len(), 1);
    assert!(merged[0].manual);
}

#[test]
fn merge_with_empty_manual_passes_all_auto_through() {
    let manual: Vec<CharBox> = vec![];
    let auto = vec![
        rect(0.0, 0.0, 10.0, 10.0, false),
        rect(50.0, 0.0, 10.0, 10.0, false),
    ];
    let merged = merge_manual_with_auto(&auto, &manual);
    assert_eq!(merged.len(), 2);
    for b in &merged {
        assert!(!b.manual);
    }
}
