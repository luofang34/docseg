//! Gate test: CRAFT ONNX must load and run a single forward pass via the
//! production CraftSession wrapper.
//!
//! Requires `models/craft_mlt_25k.onnx`. Run with:
//!
//!   cargo test -p docseg-core --test forward_pass -- --ignored

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use docseg_core::preprocess::{preprocess, PreprocessOptions};
use docseg_core::CraftSession;
use image::{DynamicImage, RgbImage};

fn model_bytes() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/craft_mlt_25k.onnx");
    std::fs::read(&path).expect("read model")
}

#[test]
#[ignore = "requires models/craft_mlt_25k.onnx (run scripts/fetch-model.sh first)"]
fn craft_session_returns_heatmap_with_expected_shape() {
    tracing_subscriber::fmt()
        .with_env_filter("info,wonnx=warn")
        .try_init()
        .ok();

    let bytes = model_bytes();
    let session = pollster::block_on(CraftSession::from_bytes(&bytes)).expect("load session");

    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(640, 640, image::Rgb([255, 255, 255])));
    let pre = preprocess(&img, PreprocessOptions::default()).expect("preprocess");
    let region = pollster::block_on(session.run(&pre)).expect("run");

    let (pw, ph) = pre.padded_size;
    assert_eq!(region.width, pw / 2);
    assert_eq!(region.height, ph / 2);
    assert_eq!(region.data.len(), (pw / 2) as usize * (ph / 2) as usize);
}
