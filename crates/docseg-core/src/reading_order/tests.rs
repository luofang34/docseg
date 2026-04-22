#![allow(clippy::expect_used, clippy::panic)]

use super::{compute_reading_order, compute_reading_order_with_regions, ReadingDirection};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;

fn rect(x: f32, y: f32, w: f32, h: f32) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ]),
        score: 1.0,
        manual: false,
    }
}

#[test]
fn vertical_rtl_orders_right_column_first_top_to_bottom() {
    // Three columns at x = 10, 60, 110. Each has three glyphs at y = 10, 60, 110.
    // Under Vertical-RTL, the rightmost (x=110) column reads first, top-to-bottom,
    // then middle, then left.
    let boxes = vec![
        rect(10.0, 10.0, 20.0, 20.0),   // 0: col-left, top
        rect(10.0, 60.0, 20.0, 20.0),   // 1: col-left, mid
        rect(10.0, 110.0, 20.0, 20.0),  // 2: col-left, bot
        rect(60.0, 10.0, 20.0, 20.0),   // 3: col-mid, top
        rect(60.0, 60.0, 20.0, 20.0),   // 4: col-mid, mid
        rect(60.0, 110.0, 20.0, 20.0),  // 5: col-mid, bot
        rect(110.0, 10.0, 20.0, 20.0),  // 6: col-right, top
        rect(110.0, 60.0, 20.0, 20.0),  // 7: col-right, mid
        rect(110.0, 110.0, 20.0, 20.0), // 8: col-right, bot
    ];
    let order = compute_reading_order(&boxes, ReadingDirection::VerticalRtl);
    assert_eq!(order, vec![6, 7, 8, 3, 4, 5, 0, 1, 2]);
}

#[test]
fn vertical_ltr_orders_left_column_first() {
    let boxes = vec![
        rect(10.0, 10.0, 20.0, 20.0),
        rect(110.0, 10.0, 20.0, 20.0),
        rect(10.0, 60.0, 20.0, 20.0),
        rect(110.0, 60.0, 20.0, 20.0),
    ];
    let order = compute_reading_order(&boxes, ReadingDirection::VerticalLtr);
    // Left column (x=10): 0 then 2; right column (x=110): 1 then 3.
    assert_eq!(order, vec![0, 2, 1, 3]);
}

#[test]
fn horizontal_ltr_orders_top_line_first_left_to_right() {
    let boxes = vec![
        rect(100.0, 10.0, 20.0, 20.0), // 0: top-right
        rect(10.0, 10.0, 20.0, 20.0),  // 1: top-left
        rect(100.0, 60.0, 20.0, 20.0), // 2: bot-right
        rect(10.0, 60.0, 20.0, 20.0),  // 3: bot-left
    ];
    let order = compute_reading_order(&boxes, ReadingDirection::HorizontalLtr);
    assert_eq!(order, vec![1, 0, 3, 2]);
}

#[test]
fn empty_input_returns_empty_order() {
    assert!(compute_reading_order(&[], ReadingDirection::VerticalRtl).is_empty());
}

#[test]
fn single_box_returns_single_index() {
    let boxes = vec![rect(10.0, 10.0, 20.0, 20.0)];
    assert_eq!(
        compute_reading_order(&boxes, ReadingDirection::VerticalRtl),
        vec![0]
    );
}

#[test]
fn order_is_a_permutation() {
    // Random-ish layout: 12 boxes spread over 3 columns with varying y.
    let boxes = vec![
        rect(10.0, 50.0, 20.0, 20.0),
        rect(70.0, 10.0, 20.0, 20.0),
        rect(130.0, 90.0, 20.0, 20.0),
        rect(10.0, 10.0, 20.0, 20.0),
        rect(70.0, 50.0, 20.0, 20.0),
        rect(130.0, 50.0, 20.0, 20.0),
        rect(10.0, 90.0, 20.0, 20.0),
        rect(70.0, 90.0, 20.0, 20.0),
        rect(130.0, 10.0, 20.0, 20.0),
        rect(10.0, 130.0, 20.0, 20.0),
        rect(70.0, 130.0, 20.0, 20.0),
        rect(130.0, 130.0, 20.0, 20.0),
    ];
    let order = compute_reading_order(&boxes, ReadingDirection::VerticalRtl);
    assert_eq!(order.len(), boxes.len());
    let mut sorted = order.clone();
    sorted.sort_unstable();
    let expected: Vec<usize> = (0..boxes.len()).collect();
    assert_eq!(sorted, expected);
}

use crate::regions::{Region, RegionRole, RegionShape};

#[test]
fn regions_order_by_rank_then_within_each() {
    // Layout: header rect covers top 50px, body is the rest.
    let boxes = vec![
        rect(10.0, 100.0, 20.0, 20.0), // 0: body, top
        rect(10.0, 200.0, 20.0, 20.0), // 1: body, bot
        rect(10.0, 10.0, 20.0, 20.0),  // 2: header
        rect(40.0, 10.0, 20.0, 20.0),  // 3: header (different col, same row)
    ];
    let regions = vec![Region {
        id: 1,
        shape: RegionShape::Rect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 100.0,
            ymax: 50.0,
        },
        role: RegionRole::Header,
        rank: 1,
    }];
    let order = compute_reading_order_with_regions(&boxes, &regions, ReadingDirection::VerticalRtl);
    // Header first (rank 1): right col header reads first under RTL.
    // Within header: col at x=40 before col at x=10.
    // Body last: col at x=10 only, top-to-bottom.
    assert_eq!(order, vec![3, 2, 0, 1]);
}

#[test]
fn box_outside_every_region_falls_into_implicit_body_rank_2() {
    // No regions drawn; behavior should match the non-region call.
    let boxes = vec![rect(10.0, 10.0, 20.0, 20.0), rect(10.0, 60.0, 20.0, 20.0)];
    let without = compute_reading_order(&boxes, ReadingDirection::VerticalRtl);
    let with = compute_reading_order_with_regions(&boxes, &[], ReadingDirection::VerticalRtl);
    assert_eq!(without, with);
}

#[test]
fn box_outside_drawn_region_is_treated_as_implicit_body() {
    // Header drawn over top-left only; the rest are implicit body (rank 2).
    // Body box should come AFTER the header box.
    let boxes = vec![
        rect(200.0, 200.0, 20.0, 20.0), // 0: body (outside header)
        rect(10.0, 10.0, 20.0, 20.0),   // 1: header
    ];
    let regions = vec![Region {
        id: 1,
        shape: RegionShape::Rect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 50.0,
            ymax: 50.0,
        },
        role: RegionRole::Header,
        rank: 1,
    }];
    let order = compute_reading_order_with_regions(&boxes, &regions, ReadingDirection::VerticalRtl);
    assert_eq!(order, vec![1, 0]);
}
