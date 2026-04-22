#![allow(clippy::expect_used, clippy::panic)]

use super::{compute_diff, DiffEntry};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;

fn b(x: f32, manual: bool) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, 0.0),
            Point::new(x + 10.0, 0.0),
            Point::new(x + 10.0, 10.0),
            Point::new(x, 10.0),
        ]),
        score: 1.0,
        manual,
    }
}

#[test]
fn unchanged_boxes_produce_unchanged_entries() {
    let auto = vec![b(10.0, false), b(50.0, false)];
    let current = auto.clone();
    let diff = compute_diff(&auto, &current);
    assert_eq!(diff.len(), 2);
    for e in &diff {
        assert!(matches!(e, DiffEntry::Unchanged { .. }));
    }
}

#[test]
fn auto_box_removed_by_user_produces_dropped_entry() {
    let auto = vec![b(10.0, false), b(50.0, false)];
    let current = vec![b(10.0, false)]; // user deleted the x=50 box
    let diff = compute_diff(&auto, &current);
    // One Unchanged + one Dropped.
    assert_eq!(diff.len(), 2);
    assert!(diff.iter().any(|e| matches!(e, DiffEntry::Dropped { .. })));
}

#[test]
fn user_added_manual_box_produces_added_entry() {
    let auto = vec![b(10.0, false)];
    let current = vec![b(10.0, false), b(200.0, true)];
    let diff = compute_diff(&auto, &current);
    assert!(diff.iter().any(|e| matches!(e, DiffEntry::Added { .. })));
}

#[test]
fn user_moved_box_produces_moved_entry() {
    let auto = vec![b(10.0, false)];
    let current = vec![b(12.0, true)]; // user nudged +2px and the edit set manual=true
    let diff = compute_diff(&auto, &current);
    assert!(
        diff.iter().any(|e| matches!(e, DiffEntry::Moved { .. })),
        "expected a Moved entry, got {:?}",
        diff
    );
}

#[test]
fn empty_inputs_produce_empty_diff() {
    assert!(compute_diff(&[], &[]).is_empty());
}
