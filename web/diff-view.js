// Diff-view overlay renderer. Call drawDiff(ctx, entries) to draw
// Dropped / Added / Moved entries on top of the normal paint pass.

export function drawDiff(ctx, entries) {
  ctx.save();
  ctx.lineWidth = 2;
  for (const e of entries) {
    switch (e.type) {
      case "unchanged":
        // skip — the normal paint pass already drew it
        break;
      case "dropped":
        drawDashed(ctx, e.box, "rgba(200,200,200,0.8)");
        strike(ctx, e.box);
        break;
      case "added":
        drawStroked(ctx, e.box, "rgba(80,220,255,0.95)");
        drawPlus(ctx, e.box);
        break;
      case "moved":
        drawDashed(ctx, e.from, "rgba(200,200,200,0.6)");
        drawStroked(ctx, e.to, "rgba(255,196,0,0.95)");
        connect(ctx, e.from, e.to);
        break;
    }
  }
  ctx.restore();
}

function aabb(box) {
  if (box.quad) {
    const xs = box.quad.points
      ? box.quad.points.map((p) => p.x)
      : box.quad.map((p) => p[0]);
    const ys = box.quad.points
      ? box.quad.points.map((p) => p.y)
      : box.quad.map((p) => p[1]);
    return {
      x0: Math.min(...xs),
      y0: Math.min(...ys),
      x1: Math.max(...xs),
      y1: Math.max(...ys),
    };
  }
  // Fallback: assume {points: [{x,y}, ...]}
  const xs = box.points.map((p) => p.x);
  const ys = box.points.map((p) => p.y);
  return {
    x0: Math.min(...xs),
    y0: Math.min(...ys),
    x1: Math.max(...xs),
    y1: Math.max(...ys),
  };
}

function drawDashed(ctx, b, style) {
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = style;
  const r = aabb(b);
  ctx.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
  ctx.setLineDash([]);
}
function drawStroked(ctx, b, style) {
  ctx.strokeStyle = style;
  const r = aabb(b);
  ctx.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
}
function strike(ctx, b) {
  const r = aabb(b);
  ctx.beginPath();
  ctx.moveTo(r.x0, r.y0);
  ctx.lineTo(r.x1, r.y1);
  ctx.stroke();
}
function drawPlus(ctx, b) {
  const r = aabb(b);
  ctx.fillStyle = "rgba(80,220,255,0.9)";
  ctx.font = "bold 14px ui-monospace, monospace";
  ctx.fillText("+", r.x1 + 2, r.y0 + 12);
}
function connect(ctx, a, b) {
  const ra = aabb(a);
  const rb = aabb(b);
  const ax = (ra.x0 + ra.x1) / 2;
  const ay = (ra.y0 + ra.y1) / 2;
  const bx = (rb.x0 + rb.x1) / 2;
  const by = (rb.y0 + rb.y1) / 2;
  ctx.strokeStyle = "rgba(200,200,200,0.7)";
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
}
