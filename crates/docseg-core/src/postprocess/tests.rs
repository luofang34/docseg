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

use super::charboxes_from_heatmap;
use crate::preprocess::PreprocessOutput;

fn fake_preproc(padded_w: u32, padded_h: u32, scale: f32) -> PreprocessOutput {
    // original_size chosen large enough that none of the tests' synthetic
    // blobs are ever rejected as out-of-bounds. Each test with a small
    // padded size sets its scale so mapped coordinates stay within this.
    let orig_w = ((padded_w as f32) / scale).ceil() as u32;
    let orig_h = ((padded_h as f32) / scale).ceil() as u32;
    PreprocessOutput {
        tensor: Vec::new(),
        padded_size: (padded_w, padded_h),
        scale,
        pad_offset: (0, 0),
        original_size: (orig_w, orig_h),
    }
}

#[test]
fn charbox_maps_back_to_original_image_coords() {
    // Padded input 64x64, heatmap 32x32. Scale 0.5 — so original image was 128x128.
    // Put a 4x4 blob starting at (8, 8) in heatmap space. That cell maps to
    // padded-input region (16, 16)..(24, 24) and original region (32, 32)..(48, 48).
    let mut map = vec![0.0_f32; 32 * 32];
    for y in 8..12 {
        for x in 8..12 {
            map[y * 32 + x] = 0.9;
        }
    }
    let boxes = charboxes_from_heatmap(
        &map,
        32,
        32,
        &fake_preproc(64, 64, 0.5),
        PostprocessOptions {
            region_threshold: 0.5,
            min_component_area_px: 1,
            max_aspect_ratio: 8.0,
        },
    );
    assert_eq!(boxes.len(), 1);
    let b = &boxes[0];
    let xs: Vec<f32> = b.quad.points.iter().map(|p| p.x).collect();
    let ys: Vec<f32> = b.quad.points.iter().map(|p| p.y).collect();
    let xmin = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let xmax = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ymin = ys.iter().copied().fold(f32::INFINITY, f32::min);
    let ymax = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!((xmin - 32.0).abs() < 4.0, "xmin={xmin}");
    assert!((xmax - 48.0).abs() < 4.0, "xmax={xmax}");
    assert!((ymin - 32.0).abs() < 4.0);
    assert!((ymax - 48.0).abs() < 4.0);
    assert!(b.score > 0.5);
}

#[test]
fn charbox_area_filter_drops_small_blobs() {
    // 1-pixel blob shouldn't survive min_component_area_px = 5.
    let mut map = vec![0.0_f32; 32 * 32];
    map[10 * 32 + 10] = 1.0;
    let boxes = charboxes_from_heatmap(
        &map,
        32,
        32,
        &fake_preproc(64, 64, 1.0),
        PostprocessOptions {
            region_threshold: 0.5,
            min_component_area_px: 5,
            max_aspect_ratio: 8.0,
        },
    );
    assert!(boxes.is_empty());
}

#[test]
fn charbox_aspect_filter_drops_long_thin_streaks() {
    // 1×20 horizontal streak in a 32x32 map: aspect ratio 20, should be
    // dropped with max_aspect_ratio = 8.
    let mut map = vec![0.0_f32; 32 * 32];
    for x in 5..25 {
        map[10 * 32 + x] = 1.0;
    }
    let boxes = charboxes_from_heatmap(
        &map,
        32,
        32,
        &fake_preproc(64, 64, 1.0),
        PostprocessOptions {
            region_threshold: 0.5,
            min_component_area_px: 1,
            max_aspect_ratio: 8.0,
        },
    );
    assert!(boxes.is_empty(), "got {} boxes; expected 0", boxes.len());
}

#[test]
fn pad_region_ghost_component_is_dropped() {
    // Heatmap 32x32, padded 64x64, scale 1.0 — so original 64x64. A blob at
    // heatmap (20, 20) maps to padded-input centroid (40, 40). If we say
    // original was only 30x30, that centroid lands at (40, 40) in original
    // space — outside the 30x30 image, i.e. a pad-region ghost.
    let mut map = vec![0.0_f32; 32 * 32];
    for y in 19..21 {
        for x in 19..21 {
            map[y * 32 + x] = 0.9;
        }
    }
    let preproc = PreprocessOutput {
        tensor: Vec::new(),
        padded_size: (64, 64),
        scale: 1.0,
        pad_offset: (0, 0),
        original_size: (30, 30),
    };
    let boxes = charboxes_from_heatmap(
        &map,
        32,
        32,
        &preproc,
        PostprocessOptions {
            region_threshold: 0.5,
            min_component_area_px: 1,
            max_aspect_ratio: 8.0,
        },
    );
    assert!(
        boxes.is_empty(),
        "pad-region ghost should have been dropped, got {} boxes",
        boxes.len()
    );
}
