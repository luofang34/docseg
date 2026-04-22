//! Per-glyph crop + zip export.

use docseg_core::postprocess::CharBox;
use docseg_core::CoreError;
use image::codecs::png::PngEncoder;
use image::{ColorType, GenericImageView, ImageEncoder};

/// Crop the axis-aligned bounding rectangle of `boxes[id]` from
/// `image_bytes` and return a fresh PNG-encoded byte vector.
pub fn crop_png(image_bytes: &[u8], boxes: &[CharBox], id: usize) -> Result<Vec<u8>, CoreError> {
    let img = image::load_from_memory(image_bytes).map_err(CoreError::Decode)?;
    let (w, h) = img.dimensions();
    let b = boxes.get(id).ok_or_else(|| CoreError::Postprocess {
        reason: format!("no box with id {id}"),
    })?;
    let (xmin, ymin, xmax, ymax) = aabb(b);
    let x0 = xmin.max(0.0).floor() as u32;
    let y0 = ymin.max(0.0).floor() as u32;
    let x1 = xmax.min((w as f32) - 1.0).ceil() as u32;
    let y1 = ymax.min((h as f32) - 1.0).ceil() as u32;
    if x1 <= x0 || y1 <= y0 {
        return Err(CoreError::Postprocess {
            reason: format!("degenerate crop for id {id}"),
        });
    }
    let crop = img.crop_imm(x0, y0, x1 - x0, y1 - y0).to_rgba8();
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(
            crop.as_raw(),
            crop.width(),
            crop.height(),
            ColorType::Rgba8.into(),
        )
        .map_err(|e| CoreError::Postprocess {
            reason: format!("png encode: {e}"),
        })?;
    Ok(out)
}

fn aabb(b: &CharBox) -> (f32, f32, f32, f32) {
    let p = &b.quad.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    (
        xs.iter().copied().fold(f32::INFINITY, f32::min),
        ys.iter().copied().fold(f32::INFINITY, f32::min),
        xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    )
}
