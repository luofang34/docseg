#![allow(clippy::expect_used, clippy::panic)]

use super::CoreError;

#[test]
fn decode_error_displays_context() {
    let io = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad png");
    let err = CoreError::Decode(image::ImageError::IoError(io));
    assert!(format!("{err}").contains("image decode"));
}

#[test]
fn model_fetch_error_carries_url() {
    let err = CoreError::ModelFetch {
        url: "https://example.com/model.onnx".into(),
        source: Box::new(std::io::Error::other("timeout")),
    };
    assert!(format!("{err}").contains("model fetch"));
    assert!(format!("{err}").contains("https://example.com/model.onnx"));
}

#[test]
fn model_load_error_carries_op_hint() {
    let err = CoreError::ModelLoad {
        hint: "op Resize".into(),
        source: Box::new(std::io::Error::other("unsupported")),
    };
    assert!(format!("{err}").contains("model load"));
    assert!(format!("{err}").contains("op Resize"));
}

#[test]
fn inference_error_displays_context() {
    let err = CoreError::Inference {
        source: Box::new(std::io::Error::other("oom")),
    };
    assert!(format!("{err}").contains("inference"));
}

#[test]
fn preprocess_error_carries_dims() {
    let err = CoreError::Preprocess {
        width: 0,
        height: 10,
        reason: "zero width".into(),
    };
    let s = format!("{err}");
    assert!(s.contains("preprocess"));
    assert!(s.contains("0"));
    assert!(s.contains("zero width"));
}

#[test]
fn postprocess_error_carries_reason() {
    let err = CoreError::Postprocess {
        reason: "heatmap shape mismatch".into(),
    };
    assert!(format!("{err}").contains("postprocess"));
    assert!(format!("{err}").contains("heatmap shape mismatch"));
}

#[test]
fn model_fetch_preserves_source_chain() {
    let err = CoreError::ModelFetch {
        url: "x".into(),
        source: Box::new(std::io::Error::other("boom")),
    };
    let src = std::error::Error::source(&err).expect("has source");
    assert!(src.to_string().contains("boom"));
}
