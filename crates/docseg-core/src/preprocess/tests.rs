#![allow(clippy::expect_used, clippy::panic)]

use super::{preprocess, PreprocessOptions};
use image::{DynamicImage, RgbImage};

fn solid(color: [u8; 3], w: u32, h: u32) -> DynamicImage {
    let img = RgbImage::from_fn(w, h, |_, _| image::Rgb(color));
    DynamicImage::ImageRgb8(img)
}

#[test]
fn letterbox_padded_size_equals_fixed_target() {
    // 1000 x 1500 portrait, target side 640. Long side scales to 640;
    // short side scales to round(1000 * 640/1500) = 427; padded canvas is 640x640.
    let img = solid([128, 128, 128], 1000, 1500);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    assert_eq!(out.padded_size, (640, 640));
}

#[test]
fn scale_matches_long_side_ratio() {
    let img = solid([255, 255, 255], 500, 1000);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    assert!(
        (out.scale - (640.0 / 1000.0)).abs() < 1e-6,
        "scale {} != expected {}",
        out.scale,
        640.0 / 1000.0
    );
}

#[test]
fn pad_offset_is_top_left_origin() {
    // The scaled image is pasted at (0, 0); padding goes on the right and bottom.
    let img = solid([255, 255, 255], 500, 1000);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    assert_eq!(out.pad_offset, (0, 0));
}

#[test]
fn nchw_tensor_length_matches_fixed_target() {
    // Regardless of input size, output is 3 x 640 x 640.
    let img = solid([0, 0, 0], 256, 256);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    assert_eq!(out.tensor.len(), 3 * 640 * 640);
}

#[test]
fn normalization_matches_imagenet_constants() {
    // A pure-white 640x640 input: (1.0 - mean)/std per channel at every content pixel.
    let img = solid([255, 255, 255], 640, 640);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    let plane = 640 * 640;
    let r0 = out.tensor[0];
    let g0 = out.tensor[plane];
    let b0 = out.tensor[2 * plane];
    let expected_r = (1.0 - 0.485) / 0.229;
    let expected_g = (1.0 - 0.456) / 0.224;
    let expected_b = (1.0 - 0.406) / 0.225;
    assert!(
        (r0 - expected_r).abs() < 1e-4,
        "R got {r0}, want {expected_r}"
    );
    assert!((g0 - expected_g).abs() < 1e-4);
    assert!((b0 - expected_b).abs() < 1e-4);
}

#[test]
fn padding_region_is_imagenet_normalized_zero() {
    // A 10x10 white input leaves a huge pad region. Sample a pixel deep in the pad
    // (x=500, y=500) on channel R: value should be (0 - 0.485) / 0.229.
    let img = solid([255, 255, 255], 10, 10);
    let out = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    let plane = 640 * 640;
    let idx = 500 * 640 + 500;
    let r = out.tensor[idx];
    let g = out.tensor[plane + idx];
    let b = out.tensor[2 * plane + idx];
    let expected_r = (0.0 - 0.485) / 0.229;
    let expected_g = (0.0 - 0.456) / 0.224;
    let expected_b = (0.0 - 0.406) / 0.225;
    assert!((r - expected_r).abs() < 1e-4);
    assert!((g - expected_g).abs() < 1e-4);
    assert!((b - expected_b).abs() < 1e-4);
}

#[test]
fn zero_size_image_is_rejected() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(0, 10));
    let err = preprocess(&img, PreprocessOptions::default()).expect_err("should fail");
    let s = format!("{err}");
    assert!(s.contains("preprocess"), "got {s}");
}
