//! Per-glyph crop + zip export.

use std::io::Write;

use docseg_core::postprocess::CharBox;
use docseg_core::CoreError;
use image::codecs::png::PngEncoder;
use image::{ColorType, GenericImageView, ImageEncoder};
use zip::write::FileOptions;
use zip::CompressionMethod;

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

#[derive(serde::Serialize)]
struct BoxesJson<'a> {
    image: ImageMetaJson,
    model: &'a str,
    boxes: Vec<BoxEntry>,
}

#[derive(serde::Serialize)]
struct ImageMetaJson {
    width: u32,
    height: u32,
}

#[derive(serde::Serialize)]
struct BoxEntry {
    id: u32,
    quad: [[f32; 2]; 4],
    score: f32,
}

/// Bundle every box's AABB crop + a `boxes.json` manifest into a single
/// DEFLATE-compressed zip.
pub fn export_zip(
    image_bytes: &[u8],
    boxes: &[CharBox],
    model_name: &str,
) -> Result<Vec<u8>, CoreError> {
    let img = image::load_from_memory(image_bytes).map_err(CoreError::Decode)?;
    let (w, h) = img.dimensions();

    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut zipw = zip::ZipWriter::new(&mut buf);
        let opts = FileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);

        let manifest = BoxesJson {
            image: ImageMetaJson {
                width: w,
                height: h,
            },
            model: model_name,
            boxes: boxes
                .iter()
                .enumerate()
                .map(|(i, b)| BoxEntry {
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
        let manifest_bytes =
            serde_json::to_vec_pretty(&manifest).map_err(|e| CoreError::Postprocess {
                reason: format!("json: {e}"),
            })?;
        zipw.start_file("boxes.json", opts)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip start boxes.json: {e}"),
            })?;
        zipw.write_all(&manifest_bytes)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip write boxes.json: {e}"),
            })?;

        for i in 0..boxes.len() {
            let png = crop_png(image_bytes, boxes, i)?;
            let name = format!("crops/{i:05}.png");
            zipw.start_file(&name, opts)
                .map_err(|e| CoreError::Postprocess {
                    reason: format!("zip start {name}: {e}"),
                })?;
            zipw.write_all(&png).map_err(|e| CoreError::Postprocess {
                reason: format!("zip write {name}: {e}"),
            })?;
        }

        zipw.finish().map_err(|e| CoreError::Postprocess {
            reason: format!("zip finish: {e}"),
        })?;
    }

    Ok(buf.into_inner())
}
