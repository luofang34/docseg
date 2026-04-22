//! Gate test: CRAFT ONNX must load and run a single forward pass via wonnx.
//!
//! Requires `models/craft_mlt_25k.onnx`. Run with:
//!
//!   cargo test -p docseg-core --test forward_pass -- --ignored
//!
//! This test is the decision point for the inference backend. If wonnx
//! rejects the model, the deviation path (burn) is pre-approved.

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::expect_used, clippy::panic)]

use std::collections::HashMap;
use std::path::PathBuf;
use wonnx::utils::InputTensor;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/craft_mlt_25k.onnx")
}

#[test]
#[ignore = "requires models/craft_mlt_25k.onnx (run scripts/fetch-model.sh first)"]
fn craft_forward_pass_produces_two_heatmaps() {
    tracing_subscriber::fmt()
        .with_env_filter("info,wonnx=warn")
        .try_init()
        .ok();

    let path = model_path();
    assert!(path.exists(), "model file not found at {}", path.display());

    let session = pollster::block_on(wonnx::Session::from_path(&path)).expect(
        "CRAFT ONNX failed to load in wonnx. If the op name is logged, \
                 record it and pivot to the burn+wgpu fallback per the spec.",
    );

    // CRAFT is exported at a fixed 640x640 input (see scripts/convert-model.py).
    // 640 fits every default WebGPU limit with margin — see that file's comment
    // for why this is tighter than the design's original 1280 target.
    // The test input shape MUST match exactly — wonnx rejects mismatched shapes.
    let (h, w) = (640_usize, 640_usize);
    let input: Vec<f32> = vec![0.0; 3 * h * w];
    let mut inputs: HashMap<String, InputTensor> = HashMap::new();
    inputs.insert("input".to_string(), input.as_slice().into());

    let outputs = pollster::block_on(session.run(&inputs)).expect("forward pass failed");

    // CRAFT output is a single tensor shaped [1, H/2, W/2, 2] (channels-last
    // for our export). Total element count = 1 * 2 * (H/2) * (W/2).
    let expected_elems = 2 * (h / 2) * (w / 2);

    let output_name = outputs.keys().next().cloned().expect("no output");
    let tensor: &[f32] = (&outputs[&output_name])
        .try_into()
        .expect("output should be f32");
    assert_eq!(
        tensor.len(),
        expected_elems,
        "CRAFT output must contain 2 heatmaps of size H/2 x W/2"
    );
}
