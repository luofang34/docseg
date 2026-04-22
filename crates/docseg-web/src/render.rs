//! Overlay painting and click hit-testing. Stateless helpers — state lives
//! on `DocsegApp::last_boxes`.

use docseg_core::postprocess::CharBox;
use wasm_bindgen::JsValue;
use web_sys::{CanvasRenderingContext2d, HtmlImageElement};

/// Box outline color.
const BOX_STROKE: &str = "rgba(255, 196, 0, 0.9)";
/// Highlight outline color (used when `highlight_id.is_some()`).
const HIGHLIGHT_STROKE: &str = "rgba(80, 220, 255, 1.0)";
/// Reading-order arrow color.
const ARROW_STROKE: &str = "rgba(80, 220, 255, 0.65)";
/// Reading-order index label color.
const LABEL_FILL: &str = "rgba(255, 255, 255, 0.95)";
/// Reading-order index label background.
const LABEL_BG: &str = "rgba(0, 0, 0, 0.6)";

/// Paint the source image at native size followed by per-box overlays.
/// `order` is the reading-order index list; arrows and numeric labels are
/// drawn when it is non-empty. `highlight_id`, when `Some(id)`, strokes that
/// box in a distinct color so the ribbon-to-canvas hover affordance has a
/// visible target.
pub fn paint_with_order(
    ctx: &CanvasRenderingContext2d,
    image: &HtmlImageElement,
    boxes: &[CharBox],
    order: &[usize],
    show_order: bool,
    highlight_id: Option<usize>,
) -> Result<(), JsValue> {
    ctx.draw_image_with_html_image_element(image, 0.0, 0.0)?;
    ctx.set_line_width(2.0);

    // Pass 1: box outlines.
    ctx.set_stroke_style_str(BOX_STROKE);
    for b in boxes {
        stroke_quad(ctx, b);
    }

    // Pass 2: arrows between consecutive reading-order centers.
    if show_order && order.len() > 1 {
        ctx.set_stroke_style_str(ARROW_STROKE);
        ctx.set_line_width(1.5);
        for w in order.windows(2) {
            let (Some(a), Some(b)) = (boxes.get(w[0]), boxes.get(w[1])) else {
                continue;
            };
            let (ax, ay) = center(a);
            let (bx, by) = center(b);
            draw_arrow(ctx, ax, ay, bx, by);
        }
    }

    // Pass 3: numeric labels.
    if show_order && !order.is_empty() {
        ctx.set_font("bold 13px ui-monospace, monospace");
        ctx.set_text_baseline("top");
        for (rank, &idx) in order.iter().enumerate() {
            let Some(b) = boxes.get(idx) else { continue };
            let (cx, cy) = top_left(b);
            let label = format!("{}", rank + 1);
            // Approximate label width: 8px per digit + 6px padding. Works for
            // the monospace font set above without needing web-sys's TextMetrics.
            let tw = (label.len() as f32) * 8.0;
            let pad: f32 = 3.0;
            let bw = tw + pad * 2.0;
            let bx = cx - bw;
            let by = cy - 16.0;
            ctx.set_fill_style_str(LABEL_BG);
            ctx.fill_rect(bx.into(), by.into(), bw.into(), 16.0);
            ctx.set_fill_style_str(LABEL_FILL);
            ctx.fill_text(&label, (bx + pad).into(), (by + 1.0).into())
                .ok();
        }
    }

    // Pass 4: highlight stroke over a single box, if any.
    if let Some(hid) = highlight_id {
        if let Some(b) = boxes.get(hid) {
            ctx.set_stroke_style_str(HIGHLIGHT_STROKE);
            ctx.set_line_width(4.0);
            stroke_quad(ctx, b);
        }
    }

    Ok(())
}

/// Return the index of the first `boxes` entry whose axis-aligned bounding
/// rectangle contains `(x, y)`, or `None`.
#[must_use]
pub fn hit_test(boxes: &[CharBox], x: f32, y: f32) -> Option<usize> {
    boxes.iter().position(|b| contains(b, x, y))
}

fn stroke_quad(ctx: &CanvasRenderingContext2d, b: &CharBox) {
    let p = &b.quad.points;
    ctx.begin_path();
    ctx.move_to(p[0].x.into(), p[0].y.into());
    for pt in &p[1..] {
        ctx.line_to(pt.x.into(), pt.y.into());
    }
    ctx.close_path();
    ctx.stroke();
}

fn center(b: &CharBox) -> (f32, f32) {
    let p = &b.quad.points;
    (
        (p[0].x + p[1].x + p[2].x + p[3].x) * 0.25,
        (p[0].y + p[1].y + p[2].y + p[3].y) * 0.25,
    )
}

fn top_left(b: &CharBox) -> (f32, f32) {
    let p = &b.quad.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    let xmin = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let ymin = ys.iter().copied().fold(f32::INFINITY, f32::min);
    (xmin, ymin)
}

fn draw_arrow(ctx: &CanvasRenderingContext2d, ax: f32, ay: f32, bx: f32, by: f32) {
    ctx.begin_path();
    ctx.move_to(ax.into(), ay.into());
    ctx.line_to(bx.into(), by.into());
    ctx.stroke();
    // Arrowhead: 6px barbs at 25°.
    let dx = bx - ax;
    let dy = by - ay;
    let len = (dx * dx + dy * dy).sqrt().max(1e-3);
    let ux = dx / len;
    let uy = dy / len;
    let size: f32 = 6.0;
    let angle: f32 = 0.436_33; // ~25° in radians
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    // Rotate -ux,-uy by ±angle and scale.
    let lx = -ux * cos_a - -uy * sin_a;
    let ly = -ux * sin_a + -uy * cos_a;
    let rx = -ux * cos_a + -uy * sin_a;
    let ry = ux * sin_a + -uy * cos_a;
    ctx.begin_path();
    ctx.move_to(bx.into(), by.into());
    ctx.line_to((bx + lx * size).into(), (by + ly * size).into());
    ctx.move_to(bx.into(), by.into());
    ctx.line_to((bx + rx * size).into(), (by + ry * size).into());
    ctx.stroke();
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
