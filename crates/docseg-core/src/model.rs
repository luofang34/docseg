//! CRAFT model session: load an ONNX bundle, run a forward pass, return the
//! region-score heatmap (affinity channel is discarded because this demo
//! segments at the character level, not the word level).
//!
//! Native-only: wonnx depends on wgpu, which does not build cleanly for
//! bare `wasm32-unknown-unknown`. The browser inference path lives in the
//! `docseg-web` crate and is wired up separately there.

use std::collections::HashMap;

use protobuf::Message;
use wonnx::onnx::ModelProto;
use wonnx::utils::InputTensor;
use wonnx::Session;

use crate::preprocess::PreprocessOutput;
use crate::CoreError;

/// Region-score heatmap at CRAFT's native output resolution (H/2 × W/2 of
/// the padded input).
#[derive(Debug, Clone)]
pub struct RegionMap {
    /// Row-major f32 values, length `width * height`.
    pub data: Vec<f32>,
    /// Heatmap width.
    pub width: u32,
    /// Heatmap height.
    pub height: u32,
}

/// Loaded CRAFT session.
pub struct CraftSession {
    inner: Session,
    input_name: String,
    output_name: String,
    /// `true` if the ONNX output is laid out `[1, H/2, W/2, 2]` (channels-last,
    /// which the current export emits). If a future export switches to
    /// `[1, 2, H/2, W/2]`, flip this flag.
    channels_last: bool,
}

impl CraftSession {
    /// Load an ONNX bundle from bytes. Async because wonnx's device init is async.
    pub async fn from_bytes(bytes: &[u8]) -> Result<Self, CoreError> {
        // Parse the protobuf once so we can pull the graph input/output names
        // before handing it off — `wonnx::Session` does not expose these.
        let model = ModelProto::parse_from_bytes(bytes).map_err(|e| CoreError::ModelLoad {
            hint: "parse ModelProto".into(),
            source: Box::new(e),
        })?;
        let graph = model.get_graph();
        let input_name = graph
            .get_input()
            .iter()
            .next()
            .map(|v| v.get_name().to_string())
            .ok_or_else(|| CoreError::ModelLoad {
                hint: "no graph inputs".into(),
                source: Box::new(std::io::Error::other("no inputs")),
            })?;
        let output_name = graph
            .get_output()
            .iter()
            .next()
            .map(|v| v.get_name().to_string())
            .ok_or_else(|| CoreError::ModelLoad {
                hint: "no graph outputs".into(),
                source: Box::new(std::io::Error::other("no outputs")),
            })?;
        let inner = Session::from_model(model)
            .await
            .map_err(|e| CoreError::ModelLoad {
                hint: "wonnx Session::from_model".into(),
                source: Box::new(e),
            })?;
        Ok(Self {
            inner,
            input_name,
            output_name,
            channels_last: true,
        })
    }

    /// Run a single forward pass and return the region-score heatmap only.
    pub async fn run(&self, preproc: &PreprocessOutput) -> Result<RegionMap, CoreError> {
        let mut inputs: HashMap<String, InputTensor> = HashMap::new();
        inputs.insert(self.input_name.clone(), preproc.tensor.as_slice().into());
        let outputs = self
            .inner
            .run(&inputs)
            .await
            .map_err(|e| CoreError::Inference {
                source: Box::new(e),
            })?;
        let out_tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| CoreError::Postprocess {
                reason: format!("expected output {}", self.output_name),
            })?;
        let raw_slice: &[f32] =
            out_tensor
                .try_into()
                .map_err(
                    |e: wonnx::utils::TensorConversionError| CoreError::Postprocess {
                        reason: format!("output not f32: {e}"),
                    },
                )?;

        let (padded_w, padded_h) = preproc.padded_size;
        let hm_w = padded_w / 2;
        let hm_h = padded_h / 2;
        let expected = 2 * (hm_w as usize) * (hm_h as usize);
        if raw_slice.len() != expected {
            return Err(CoreError::Postprocess {
                reason: format!(
                    "CRAFT output length {} != expected {expected}",
                    raw_slice.len()
                ),
            });
        }

        let region = if self.channels_last {
            region_channel_channels_last(raw_slice, hm_w, hm_h)
        } else {
            region_channel_channels_first(raw_slice, hm_w, hm_h)
        };
        Ok(RegionMap {
            data: region,
            width: hm_w,
            height: hm_h,
        })
    }
}

/// Extract the `region` channel assuming a channels-last layout
/// `[1, H, W, 2]` where `[..][..][0]` is region and `[..][..][1]` is affinity.
#[must_use]
pub fn region_channel_channels_last(raw: &[f32], w: u32, h: u32) -> Vec<f32> {
    let plane = (w as usize) * (h as usize);
    let mut out = Vec::with_capacity(plane);
    for i in 0..plane {
        if let Some(v) = raw.get(i * 2) {
            out.push(*v);
        }
    }
    out
}

/// Extract the `region` channel assuming a channels-first layout
/// `[1, 2, H, W]` where channel 0 is region, channel 1 is affinity.
#[must_use]
pub fn region_channel_channels_first(raw: &[f32], w: u32, h: u32) -> Vec<f32> {
    let plane = (w as usize) * (h as usize);
    raw.iter().take(plane).copied().collect()
}

#[cfg(test)]
mod tests;
