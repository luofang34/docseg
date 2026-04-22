//! End-to-end pipeline test: synthetic image with 3 well-separated filled
//! rectangles → preprocess → CraftSession → charboxes_from_heatmap → 3 boxes.
//!
//! Gated on `models/craft_mlt_25k.onnx` being present. Run with:
//!   cargo test -p docseg-core --test end_to_end -- --ignored

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use docseg_core::postprocess::{charboxes_from_heatmap, PostprocessOptions};
use docseg_core::preprocess::{preprocess, PreprocessOptions};
use docseg_core::CraftSession;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;

fn model_bytes() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/craft_mlt_25k.onnx");
    std::fs::read(&path).expect("read model")
}

fn synthetic_three_blobs() -> DynamicImage {
    // 320x320 white canvas, three ~60x60 dark ink blocks spaced apart.
    let mut img = RgbImage::from_pixel(320, 320, Rgb([255, 255, 255]));
    let ink = Rgb([20, 20, 20]);
    for (cx, cy) in [(60, 160), (160, 160), (260, 160)] {
        draw_filled_rect_mut(&mut img, Rect::at(cx - 30, cy - 30).of_size(60, 60), ink);
    }
    DynamicImage::ImageRgb8(img)
}

#[test]
#[ignore = "requires models/craft_mlt_25k.onnx"]
fn end_to_end_detects_three_blobs() {
    let img = synthetic_three_blobs();
    let pre = preprocess(&img, PreprocessOptions::default()).expect("preprocess");

    let bytes = model_bytes();
    let session = pollster::block_on(CraftSession::from_bytes(&bytes)).expect("load");
    let region = pollster::block_on(session.run(&pre)).expect("inference");

    // Looser threshold than the default (synthetic solid-fill shapes score
    // lower than natural text under CRAFT) and a modest area filter to keep
    // CRAFT's scattered single-pixel activations from becoming "boxes."
    // Pad-region ghosts are filtered inside `charboxes_from_heatmap` via the
    // `PreprocessOutput::original_size` centroid check.
    let opts = PostprocessOptions {
        region_threshold: 0.2,
        min_component_area_px: 8,
        max_aspect_ratio: 8.0,
    };
    let boxes = charboxes_from_heatmap(&region.data, region.width, region.height, &pre, opts);

    // Three blobs — tolerate ±1 in case CRAFT accidentally merges or splits.
    println!(
        "end-to-end: {} boxes, scores {:?}",
        boxes.len(),
        boxes.iter().map(|b| b.score).collect::<Vec<_>>()
    );
    assert!(
        (2..=4).contains(&boxes.len()),
        "expected ~3 boxes, got {}: {:?}",
        boxes.len(),
        boxes
    );
    for b in &boxes {
        assert!(b.score > 0.05, "score {} suspiciously low", b.score);
        for p in b.quad.points {
            assert!(p.x >= -8.0 && p.x <= 328.0, "x {} out of bounds", p.x);
            assert!(p.y >= -8.0 && p.y <= 328.0, "y {} out of bounds", p.y);
        }
    }
}
