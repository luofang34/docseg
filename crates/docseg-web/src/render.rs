//! Overlay painting and click hit-testing. Stateless helpers — state lives
//! on `DocsegApp::last_boxes`.

use docseg_core::postprocess::CharBox;
use wasm_bindgen::JsValue;
use web_sys::{CanvasRenderingContext2d, HtmlImageElement};

/// Paint the source image at native size followed by a translucent-yellow
/// polygon for each detected char box.
pub fn paint(
    ctx: &CanvasRenderingContext2d,
    image: &HtmlImageElement,
    boxes: &[CharBox],
) -> Result<(), JsValue> {
    ctx.draw_image_with_html_image_element(image, 0.0, 0.0)?;
    ctx.set_stroke_style_str("rgba(255, 196, 0, 0.9)");
    ctx.set_line_width(2.0);
    for b in boxes {
        let p = &b.quad.points;
        ctx.begin_path();
        ctx.move_to(p[0].x.into(), p[0].y.into());
        for pt in &p[1..] {
            ctx.line_to(pt.x.into(), pt.y.into());
        }
        ctx.close_path();
        ctx.stroke();
    }
    Ok(())
}

/// Return the index of the first `boxes` entry whose axis-aligned bounding
/// rectangle contains `(x, y)`, or `None`.
#[must_use]
pub fn hit_test(boxes: &[CharBox], x: f32, y: f32) -> Option<usize> {
    boxes.iter().position(|b| contains(b, x, y))
}

fn contains(b: &CharBox, x: f32, y: f32) -> bool {
    let p = &b.quad.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    let xmin = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let xmax = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ymin = ys.iter().copied().fold(f32::INFINITY, f32::min);
    let ymax = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    x >= xmin && x <= xmax && y >= ymin && y <= ymax
}
