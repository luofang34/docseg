//! Overlay painting and click hit-testing. Stateless helpers — state lives
//! on `DocsegApp::last_boxes`.

use docseg_core::postprocess::CharBox;
use wasm_bindgen::JsValue;
use web_sys::{CanvasRenderingContext2d, HtmlImageElement};

/// Box outline color.
const BOX_STROKE: &str = "rgba(255, 196, 0, 0.9)";
/// Highlight outline color (used when `highlight_id.is_some()`).
const HIGHLIGHT_STROKE: &str = "rgba(80, 220, 255, 1.0)";
/// Reading-order index label color.
const LABEL_FILL: &str = "rgba(255, 255, 255, 0.95)";
/// Reading-order index label background.
const LABEL_BG: &str = "rgba(0, 0, 0, 0.6)";

/// Paint the source image at native size followed by per-box overlays.
/// `order` is the reading-order index list; arrows and numeric labels are
/// drawn when it is non-empty. `highlight_id`, when `Some(id)`, strokes that
/// box in a distinct color so the ribbon-to-canvas hover affordance has a
/// visible target. `regions` and `reading_direction_orthogonal_is_x` control
/// the three-style arrow classifier: Continue, CarriageReturn, RegionBreak.
/// `selected_id`, when `Some(id)`, renders 8 resize handles around that box.
#[allow(clippy::too_many_arguments)]
pub fn paint_with_order(
    ctx: &CanvasRenderingContext2d,
    image: &HtmlImageElement,
    boxes: &[CharBox],
    order: &[usize],
    regions: &[docseg_core::regions::Region],
    reading_direction_orthogonal_is_x: bool,
    show_order: bool,
    highlight_id: Option<usize>,
    selected_id: Option<usize>,
) -> Result<(), JsValue> {
    ctx.draw_image_with_html_image_element(image, 0.0, 0.0)?;

    // Pass 0: region overlays (drawn behind boxes).
    for r in regions {
        if let docseg_core::regions::RegionShape::Rect {
            xmin,
            ymin,
            xmax,
            ymax,
        } = r.shape
        {
            let (stroke, fill) = region_colors(r.role);
            ctx.set_fill_style_str(fill);
            ctx.fill_rect(
                xmin.into(),
                ymin.into(),
                (xmax - xmin).into(),
                (ymax - ymin).into(),
            );
            ctx.set_stroke_style_str(stroke);
            ctx.set_line_width(1.0);
            ctx.stroke_rect(
                xmin.into(),
                ymin.into(),
                (xmax - xmin).into(),
                (ymax - ymin).into(),
            );
        }
    }

    // Pass 1: box outlines (low-confidence gets dashed; manual gets cyan inner ring).
    let score_thr = low_confidence_threshold(boxes);
    ctx.set_line_width(2.0);
    for b in boxes {
        let low_conf = b.score < score_thr;
        if low_conf {
            let dash = js_sys::Array::new();
            dash.push(&JsValue::from(4.0));
            dash.push(&JsValue::from(3.0));
            ctx.set_line_dash(&dash).ok();
        } else {
            ctx.set_line_dash(&js_sys::Array::new()).ok();
        }
        ctx.set_stroke_style_str(BOX_STROKE);
        stroke_quad(ctx, b);
        if b.manual {
            ctx.set_line_dash(&js_sys::Array::new()).ok();
            ctx.set_stroke_style_str("rgba(80, 220, 255, 0.9)");
            ctx.set_line_width(1.0);
            stroke_quad_inset(ctx, b, 2.0);
            ctx.set_line_width(2.0);
        }
    }
    ctx.set_line_dash(&js_sys::Array::new()).ok();

    // Pass 2: arrows — three styles:
    //   continue        = solid thin line, no arrowhead (same line/column of same region)
    //   carriage-return = dashed chevron (column/line break within a region)
    //   region-break    = thicker orange arrow (crossing between regions)
    if show_order && order.len() > 1 {
        // Compute median box dimensions for the orthogonal-jump heuristic.
        let (median_w, median_h) = median_box_dims(boxes);
        for w in order.windows(2) {
            let (Some(a), Some(b)) = (boxes.get(w[0]), boxes.get(w[1])) else {
                continue;
            };
            let (ax, ay) = center(a);
            let (bx, by) = center(b);
            let style = classify_transition(
                a,
                b,
                regions,
                reading_direction_orthogonal_is_x,
                median_w,
                median_h,
            );
            match style {
                ArrowStyle::Continue => {
                    ctx.set_stroke_style_str("rgba(80, 220, 255, 0.55)");
                    ctx.set_line_width(1.0);
                    ctx.set_line_dash(&js_sys::Array::new()).ok();
                    ctx.begin_path();
                    ctx.move_to(ax.into(), ay.into());
                    ctx.line_to(bx.into(), by.into());
                    ctx.stroke();
                }
                ArrowStyle::CarriageReturn => {
                    ctx.set_stroke_style_str("rgba(120, 240, 255, 0.9)");
                    ctx.set_line_width(2.0);
                    let dash = js_sys::Array::new();
                    dash.push(&JsValue::from(6.0));
                    dash.push(&JsValue::from(4.0));
                    ctx.set_line_dash(&dash).ok();
                    ctx.begin_path();
                    ctx.move_to(ax.into(), ay.into());
                    ctx.line_to(bx.into(), by.into());
                    ctx.stroke();
                    ctx.set_line_dash(&js_sys::Array::new()).ok();
                    draw_arrow_head(ctx, ax, ay, bx, by, 7.0);
                }
                ArrowStyle::RegionBreak => {
                    ctx.set_stroke_style_str("rgba(255, 160, 60, 0.9)");
                    ctx.set_line_width(3.0);
                    ctx.set_line_dash(&js_sys::Array::new()).ok();
                    ctx.begin_path();
                    ctx.move_to(ax.into(), ay.into());
                    ctx.line_to(bx.into(), by.into());
                    ctx.stroke();
                    draw_arrow_head(ctx, ax, ay, bx, by, 10.0);
                }
            }
        }
        // Restore defaults.
        ctx.set_line_dash(&js_sys::Array::new()).ok();
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

    // Pass 5: selection handles on the currently-selected box.
    if let Some(sid) = selected_id {
        if let Some(b) = boxes.get(sid) {
            let p = &b.quad.points;
            let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
            let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
            let xmin = xs.iter().copied().fold(f32::INFINITY, f32::min);
            let xmax = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ymin = ys.iter().copied().fold(f32::INFINITY, f32::min);
            let ymax = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let midx = (xmin + xmax) * 0.5;
            let midy = (ymin + ymax) * 0.5;
            let handles = [
                (xmin, ymin),
                (midx, ymin),
                (xmax, ymin),
                (xmax, midy),
                (xmax, ymax),
                (midx, ymax),
                (xmin, ymax),
                (xmin, midy),
            ];
            ctx.set_fill_style_str("rgba(80, 220, 255, 0.95)");
            ctx.set_stroke_style_str("rgba(0, 0, 0, 0.8)");
            ctx.set_line_width(1.0);
            for (hx, hy) in handles {
                ctx.fill_rect((hx - 3.0).into(), (hy - 3.0).into(), 6.0, 6.0);
                ctx.stroke_rect((hx - 3.0).into(), (hy - 3.0).into(), 6.0, 6.0);
            }
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

enum ArrowStyle {
    Continue,
    CarriageReturn,
    RegionBreak,
}

fn classify_transition(
    a: &CharBox,
    b: &CharBox,
    regions: &[docseg_core::regions::Region],
    orthogonal_is_x: bool,
    median_w: f32,
    median_h: f32,
) -> ArrowStyle {
    use docseg_core::regions::region_for_box;
    let ra = region_for_box(a, regions);
    let rb = region_for_box(b, regions);
    if ra != rb {
        return ArrowStyle::RegionBreak;
    }
    let (ax, ay) = center(a);
    let (bx, by) = center(b);
    let orthogonal_jump = if orthogonal_is_x {
        (bx - ax).abs()
    } else {
        (by - ay).abs()
    };
    let threshold = if orthogonal_is_x { median_w } else { median_h };
    if orthogonal_jump > threshold {
        ArrowStyle::CarriageReturn
    } else {
        ArrowStyle::Continue
    }
}

fn median_box_dims(boxes: &[CharBox]) -> (f32, f32) {
    if boxes.is_empty() {
        return (1.0, 1.0);
    }
    let mut ws: Vec<f32> = boxes.iter().map(box_width).collect();
    let mut hs: Vec<f32> = boxes.iter().map(box_height).collect();
    ws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (ws[ws.len() / 2], hs[hs.len() / 2])
}

fn box_width(b: &CharBox) -> f32 {
    let p = &b.quad.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - xs.iter().copied().fold(f32::INFINITY, f32::min)
}

fn box_height(b: &CharBox) -> f32 {
    let p = &b.quad.points;
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    ys.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - ys.iter().copied().fold(f32::INFINITY, f32::min)
}

fn draw_arrow_head(ctx: &CanvasRenderingContext2d, ax: f32, ay: f32, bx: f32, by: f32, size: f32) {
    let dx = bx - ax;
    let dy = by - ay;
    let len = (dx * dx + dy * dy).sqrt().max(1e-3);
    let ux = dx / len;
    let uy = dy / len;
    let angle: f32 = 0.436_33;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
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

fn stroke_quad_inset(ctx: &CanvasRenderingContext2d, b: &CharBox, inset: f32) {
    let p = &b.quad.points;
    // Move each corner inward along its diagonal toward the centroid by `inset` pixels.
    let cx = (p[0].x + p[1].x + p[2].x + p[3].x) * 0.25;
    let cy = (p[0].y + p[1].y + p[2].y + p[3].y) * 0.25;
    ctx.begin_path();
    let mut first = true;
    for pt in p {
        let dx = pt.x - cx;
        let dy = pt.y - cy;
        let len = (dx * dx + dy * dy).sqrt().max(1e-3);
        let x = pt.x - dx / len * inset;
        let y = pt.y - dy / len * inset;
        if first {
            ctx.move_to(x.into(), y.into());
            first = false;
        } else {
            ctx.line_to(x.into(), y.into());
        }
    }
    ctx.close_path();
    ctx.stroke();
}

/// Dashed-outline threshold: score < median − MAD. Returns 0 when boxes is empty.
fn low_confidence_threshold(boxes: &[CharBox]) -> f32 {
    if boxes.is_empty() {
        return 0.0;
    }
    let mut scores: Vec<f32> = boxes.iter().map(|b| b.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = scores[scores.len() / 2];
    let mut devs: Vec<f32> = scores.iter().map(|s| (s - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = devs[devs.len() / 2];
    (median - mad).max(0.0)
}

fn region_colors(role: docseg_core::regions::RegionRole) -> (&'static str, &'static str) {
    use docseg_core::regions::RegionRole;
    match role {
        RegionRole::Header => ("rgba(80, 140, 255, 0.9)", "rgba(80, 140, 255, 0.10)"),
        RegionRole::Body => ("rgba(180, 180, 180, 0.0)", "rgba(180, 180, 180, 0.0)"),
        RegionRole::Footer => ("rgba(80, 220, 120, 0.9)", "rgba(80, 220, 120, 0.10)"),
        RegionRole::Notes => ("rgba(200, 120, 240, 0.9)", "rgba(200, 120, 240, 0.10)"),
    }
}
