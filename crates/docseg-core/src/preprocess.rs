//! Letterbox + ImageNet-normalize an `image::DynamicImage` into the NCHW
//! f32 tensor CRAFT expects, plus the inverse transform for mapping model-
//! space coordinates back to the original image.
//!
//! The CRAFT ONNX (see `scripts/convert-model.py`) is exported at a fixed
//! (1, 3, 640, 640) input shape, so preprocess always produces a 640×640
//! padded tensor regardless of the input aspect ratio.

use image::{imageops::FilterType, DynamicImage, GenericImageView};

use crate::CoreError;

/// ImageNet mean (RGB order).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet std (RGB order).
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Knobs for the preprocess stage.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessOptions {
    /// Side length of the square padded canvas. Must match the CRAFT ONNX
    /// input dimension — both are currently 640.
    pub target_side: u32,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self { target_side: 640 }
    }
}

/// Everything downstream stages need to consume the preprocess output.
#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    /// NCHW f32 tensor, length `3 * padded_w * padded_h`.
    pub tensor: Vec<f32>,
    /// Padded model-input dimensions `(width, height)`. Always `(target_side, target_side)`.
    pub padded_size: (u32, u32),
    /// Multiply an original-image coordinate by `scale` to reach model space.
    pub scale: f32,
    /// Padding added before the scaled image in model space `(x, y)`. Always `(0, 0)`
    /// because we paste the scaled image at the top-left and pad right+bottom.
    pub pad_offset: (u32, u32),
    /// Original-image dimensions `(width, height)` before any scaling.
    /// Downstream uses this to drop components that fall in the letterbox
    /// pad region — CRAFT can hallucinate mid-confidence activations over
    /// the zero-filled pad area, and those ghosts otherwise map to out-of-
    /// image coordinates.
    pub original_size: (u32, u32),
}

/// Run the full preprocess pipeline.
pub fn preprocess(
    img: &DynamicImage,
    opts: PreprocessOptions,
) -> Result<PreprocessOutput, CoreError> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(CoreError::Preprocess {
            width: w,
            height: h,
            reason: "zero-size image".into(),
        });
    }

    let side = opts.target_side;
    let long_side = w.max(h) as f32;
    // Clamp to 1.0 so small inputs aren't upscaled — they stay at native
    // resolution with the remainder of the canvas filled as pad.
    let scale = ((side as f32) / long_side).min(1.0);
    let scaled_w = ((w as f32) * scale).round().min(side as f32) as u32;
    let scaled_h = ((h as f32) * scale).round().min(side as f32) as u32;

    let resized = img
        .resize_exact(scaled_w, scaled_h, FilterType::Triangle)
        .to_rgb8();

    let plane = (side as usize) * (side as usize);
    let mut tensor = vec![0.0_f32; 3 * plane];

    // Normalized-zero values fill the pad region (zero in pixel space → normalized below).
    let zero_r = (0.0 - MEAN[0]) / STD[0];
    let zero_g = (0.0 - MEAN[1]) / STD[1];
    let zero_b = (0.0 - MEAN[2]) / STD[2];
    for r in tensor[0..plane].iter_mut() {
        *r = zero_r;
    }
    for g in tensor[plane..2 * plane].iter_mut() {
        *g = zero_g;
    }
    for b in tensor[2 * plane..3 * plane].iter_mut() {
        *b = zero_b;
    }

    // Paste the scaled image into the top-left of the canvas, overwriting the
    // pad fill. Normalization happens per-pixel in the same loop.
    for y in 0..scaled_h {
        for x in 0..scaled_w {
            let px = resized.get_pixel(x, y);
            let idx = (y as usize) * (side as usize) + (x as usize);
            let r = (px[0] as f32) / 255.0;
            let g = (px[1] as f32) / 255.0;
            let b = (px[2] as f32) / 255.0;
            tensor[idx] = (r - MEAN[0]) / STD[0];
            tensor[plane + idx] = (g - MEAN[1]) / STD[1];
            tensor[2 * plane + idx] = (b - MEAN[2]) / STD[2];
        }
    }

    Ok(PreprocessOutput {
        tensor,
        padded_size: (side, side),
        scale,
        pad_offset: (0, 0),
        original_size: (w, h),
    })
}

#[cfg(test)]
mod tests;
