#![allow(clippy::expect_used, clippy::panic)]

use super::{EditEvent, EditLog};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;
use crate::regions::{Region, RegionRole, RegionShape};

fn box_at(x: f32) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, 0.0),
            Point::new(x + 10.0, 0.0),
            Point::new(x + 10.0, 10.0),
            Point::new(x, 10.0),
        ]),
        score: 1.0,
        manual: true,
    }
}

#[test]
fn empty_log_undo_returns_none() {
    let mut log = EditLog::new();
    assert!(log.undo().is_none());
}

#[test]
fn push_then_undo_returns_the_event() {
    let mut log = EditLog::new();
    let ev = EditEvent::AddBox(box_at(10.0));
    log.push(ev.clone());
    let popped = log.undo().expect("should have an event");
    assert!(matches!(popped, EditEvent::AddBox(_)));
}

#[test]
fn undo_then_redo_restores_the_event() {
    let mut log = EditLog::new();
    log.push(EditEvent::AddBox(box_at(10.0)));
    let _ = log.undo();
    let redone = log.redo().expect("redo should have something");
    assert!(matches!(redone, EditEvent::AddBox(_)));
}

#[test]
fn new_push_clears_redo_stack() {
    let mut log = EditLog::new();
    log.push(EditEvent::AddBox(box_at(10.0)));
    let _ = log.undo();
    log.push(EditEvent::AddBox(box_at(20.0)));
    assert!(log.redo().is_none());
}

#[test]
fn log_caps_at_50_entries() {
    let mut log = EditLog::new();
    for i in 0..60 {
        log.push(EditEvent::AddBox(box_at(i as f32)));
    }
    // Oldest 10 should be evicted.
    let mut undone = 0;
    while log.undo().is_some() {
        undone += 1;
    }
    assert_eq!(undone, 50);
}

#[test]
fn region_event_variants_roundtrip() {
    let r = Region {
        id: 1,
        shape: RegionShape::Rect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 10.0,
            ymax: 10.0,
        },
        role: RegionRole::Header,
        rank: 1,
    };
    let mut log = EditLog::new();
    log.push(EditEvent::AddRegion(r.clone()));
    assert!(matches!(log.undo(), Some(EditEvent::AddRegion(_))));
}
