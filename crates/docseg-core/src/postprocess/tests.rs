#![allow(clippy::expect_used, clippy::panic)]

use super::{components_from_heatmap, PostprocessOptions};

#[test]
fn two_separate_blobs_produce_two_components() {
    // 8x4 heatmap with two 2x2 blobs separated by a gap:
    //   . . . . . . . .
    //   . X X . . X X .
    //   . X X . . X X .
    //   . . . . . . . .
    let mut map = vec![0.0_f32; 8 * 4];
    let set = |m: &mut Vec<f32>, x: usize, y: usize| m[y * 8 + x] = 1.0;
    for (x, y) in [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (5, 1),
        (6, 1),
        (5, 2),
        (6, 2),
    ] {
        set(&mut map, x, y);
    }
    let comps = components_from_heatmap(&map, 8, 4, PostprocessOptions::default());
    assert_eq!(comps.len(), 2, "expected 2 components, got {}", comps.len());
    for c in &comps {
        assert_eq!(c.len(), 4, "each blob has 4 pixels");
    }
}

#[test]
fn subthreshold_noise_is_excluded() {
    let map = vec![0.1_f32; 16 * 16]; // below default threshold (0.4)
    let comps = components_from_heatmap(&map, 16, 16, PostprocessOptions::default());
    assert!(comps.is_empty());
}

#[test]
fn diagonal_touch_does_not_merge_components_under_4_connectivity() {
    // Two 1-pixel blobs at (1,1) and (2,2) — 8-connected neighbours, but
    // 4-connected they stay separate.
    let mut map = vec![0.0_f32; 4 * 4];
    map[4 + 1] = 1.0;
    map[2 * 4 + 2] = 1.0;
    let opts = PostprocessOptions {
        region_threshold: 0.5,
        ..Default::default()
    };
    let comps = components_from_heatmap(&map, 4, 4, opts);
    assert_eq!(comps.len(), 2);
}

#[test]
fn empty_or_zero_size_heatmap_returns_no_components() {
    assert!(components_from_heatmap(&[], 0, 0, PostprocessOptions::default()).is_empty());
    // Mismatched length is a misuse; returning empty rather than panicking matches
    // the workspace rule of no panics outside tests.
    let map = vec![1.0_f32; 5];
    assert!(components_from_heatmap(&map, 4, 4, PostprocessOptions::default()).is_empty());
}
