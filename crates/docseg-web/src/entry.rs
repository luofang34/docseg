//! Top-level wasm-bindgen surface. Rust owns preprocess + postprocess +
//! downstream geometry; JS owns image/model fetch and the inference call
//! into `onnxruntime-web` (see `web/main.js`).

use std::cell::RefCell;

use docseg_core::postprocess::{charboxes_from_heatmap, CharBox, PostprocessOptions};
use docseg_core::preprocess::{preprocess, PreprocessOptions, PreprocessOutput};
use serde::Serialize;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlImageElement};

use crate::canvas::decode;
use crate::render::{hit_test, paint};

/// One detected glyph as it crosses the JS boundary.
#[derive(Serialize)]
struct BoxOut {
    id: u32,
    quad: [[f32; 2]; 4],
    score: f32,
}

/// Result of a full detection run.
#[derive(Serialize)]
struct DetectionOut {
    image: ImageMeta,
    model: &'static str,
    boxes: Vec<BoxOut>,
}

#[derive(Serialize)]
struct ImageMeta {
    width: u32,
    height: u32,
}

/// Subset of `PreprocessOutput` JS needs to round-trip back into postprocess.
#[derive(Serialize)]
struct PreprocessJs {
    /// NCHW f32 tensor, length `3 * padded_w * padded_h`.
    tensor: Vec<f32>,
    scale: f32,
    padded_w: u32,
    padded_h: u32,
    original_w: u32,
    original_h: u32,
    pad_x: u32,
    pad_y: u32,
}

/// Top-level handle returned to JS.
#[wasm_bindgen]
pub struct DocsegApp {
    /// Image bytes of the most recent preprocess — retained so
    /// `cropPng` / `exportZip` (later tasks) can re-decode without a
    /// round-trip through JS.
    last_image_bytes: RefCell<Vec<u8>>,
    /// Most recent CharBox list from postprocess.
    last_boxes: RefCell<Vec<CharBox>>,
}

#[wasm_bindgen]
impl DocsegApp {
    /// Construct a fresh app. Installs a tracing subscriber and a panic hook.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        tracing_wasm::try_set_as_global_default().ok();
        Self {
            last_image_bytes: RefCell::new(Vec::new()),
            last_boxes: RefCell::new(Vec::new()),
        }
    }

    /// Decode + preprocess an encoded image. Returns a JSON-shape value JS
    /// hands to onnxruntime-web.
    #[wasm_bindgen(js_name = preprocessImage)]
    pub fn preprocess_image(&self, image_bytes: Vec<u8>) -> Result<JsValue, JsError> {
        let img = decode(&image_bytes).map_err(|e| JsError::new(&format!("{e:#}")))?;
        let pre = preprocess(&img, PreprocessOptions::default())
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        *self.last_image_bytes.borrow_mut() = image_bytes;
        let out = PreprocessJs {
            tensor: pre.tensor,
            scale: pre.scale,
            padded_w: pre.padded_size.0,
            padded_h: pre.padded_size.1,
            original_w: pre.original_size.0,
            original_h: pre.original_size.1,
            pad_x: pre.pad_offset.0,
            pad_y: pre.pad_offset.1,
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Convert the region-score heatmap (post-inference) into per-character
    /// boxes. `region_data` is the flat float32 region channel extracted by
    /// JS from the onnxruntime-web output tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn postprocess(
        &self,
        region_data: Vec<f32>,
        heatmap_w: u32,
        heatmap_h: u32,
        scale: f32,
        padded_w: u32,
        padded_h: u32,
        original_w: u32,
        original_h: u32,
    ) -> Result<JsValue, JsError> {
        let pre = PreprocessOutput {
            tensor: Vec::new(),
            padded_size: (padded_w, padded_h),
            scale,
            pad_offset: (0, 0),
            original_size: (original_w, original_h),
        };
        let boxes = charboxes_from_heatmap(
            &region_data,
            heatmap_w,
            heatmap_h,
            &pre,
            PostprocessOptions::default(),
        );
        *self.last_boxes.borrow_mut() = boxes.clone();
        let out = DetectionOut {
            image: ImageMeta {
                width: original_w,
                height: original_h,
            },
            model: "craft_mlt_25k",
            boxes: boxes
                .iter()
                .enumerate()
                .map(|(i, b)| BoxOut {
                    id: i as u32,
                    quad: [
                        [b.quad.points[0].x, b.quad.points[0].y],
                        [b.quad.points[1].x, b.quad.points[1].y],
                        [b.quad.points[2].x, b.quad.points[2].y],
                        [b.quad.points[3].x, b.quad.points[3].y],
                    ],
                    score: b.score,
                })
                .collect(),
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Paint the source image + per-character overlay onto a canvas
    /// context, using the boxes from the most recent `postprocess` call.
    pub fn paint(
        &self,
        ctx: &CanvasRenderingContext2d,
        img: &HtmlImageElement,
    ) -> Result<(), JsError> {
        paint(ctx, img, &self.last_boxes.borrow())
            .map_err(|e| JsError::new(&format!("paint failed: {e:?}")))
    }

    /// Hit-test the last postprocess result. Returns the 0-based id of the
    /// first box whose axis-aligned bounding rect contains `(x, y)`, or
    /// `-1` if nothing matches.
    pub fn hit(&self, x: f32, y: f32) -> i32 {
        hit_test(&self.last_boxes.borrow(), x, y).map_or(-1, |i| i as i32)
    }

    /// Return a PNG of the axis-aligned crop of box `id` from the last
    /// image passed to `preprocessImage`. Errors if that image hasn't been
    /// stored, or if the id is out of range.
    #[wasm_bindgen(js_name = cropPng)]
    pub fn crop_png(&self, id: u32) -> Result<Vec<u8>, JsError> {
        crate::export::crop_png(
            &self.last_image_bytes.borrow(),
            &self.last_boxes.borrow(),
            id as usize,
        )
        .map_err(|e| JsError::new(&format!("{e:#}")))
    }

    /// Export a zip containing every crop + a `boxes.json` manifest,
    /// using the state retained from the last `preprocessImage` +
    /// `postprocess` calls.
    #[wasm_bindgen(js_name = exportZip)]
    pub fn export_zip(&self) -> Result<Vec<u8>, JsError> {
        crate::export::export_zip(
            &self.last_image_bytes.borrow(),
            &self.last_boxes.borrow(),
            "craft_mlt_25k",
        )
        .map_err(|e| JsError::new(&format!("{e:#}")))
    }
}

impl Default for DocsegApp {
    fn default() -> Self {
        Self::new()
    }
}
