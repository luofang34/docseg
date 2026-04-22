// Tool palette + drag/keyboard handlers. Works in canvas-pixel
// coordinates (original image space), not CSS pixels.
//
// Public interface:
//   initTools({ canvas, app, state, onChange })
//     state    — { mode, selectedId, drawSequence, lastDetection, ... }
//     onChange — called after every structural edit so main.js repaints
//                + updates the ribbon.

const TOOLS = ["select", "add", "delete", "order", "region"];

export function initTools({ canvas, app, state, onChange }) {
  const modeButtons = {
    select: document.getElementById("tool-select"),
    add: document.getElementById("tool-add"),
    delete: document.getElementById("tool-delete"),
    order: document.getElementById("tool-order"),
    region: document.getElementById("tool-region"),
  };

  function setTool(name) {
    state.mode = name;
    for (const [key, btn] of Object.entries(modeButtons)) {
      btn?.classList.toggle("mode-active", key === name);
    }
    canvas.style.cursor = {
      select: "pointer",
      add: "crosshair",
      delete: "crosshair",
      order: "crosshair",
      region: "crosshair",
    }[name] || "default";
  }
  for (const [key, btn] of Object.entries(modeButtons)) {
    btn?.addEventListener("click", () => setTool(key));
  }
  setTool("select");

  // Keyboard shortcuts (module-level handler; main.js already has one —
  // this one is additive).
  window.addEventListener("keydown", (ev) => {
    if (ev.target?.tagName === "INPUT" || ev.target?.tagName === "SELECT") return;
    if (ev.metaKey || ev.ctrlKey) {
      if (ev.key === "z" && !ev.shiftKey) {
        ev.preventDefault();
        if (app.undo()) onChange();
        return;
      }
      if (ev.key === "z" && ev.shiftKey) {
        ev.preventDefault();
        if (app.redo()) onChange();
        return;
      }
      return;
    }
    const key = ev.key.toLowerCase();
    const map = { v: "select", a: "add", d: "delete", o: "order", r: "region" };
    if (map[key]) {
      setTool(map[key]);
      return;
    }
    if (key === "escape") setTool("select");
    if (key === "delete" || key === "backspace") {
      if ((state.selectedId ?? -1) >= 0) {
        app.removeBox(state.selectedId);
        state.selectedId = -1;
        app.setSelected(-1);
        onChange();
        ev.preventDefault();
      }
    }
  });

  // Mouse events.
  let drag = null;

  function toImage(ev) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (ev.clientX - rect.left) * (canvas.width / rect.width),
      y: (ev.clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  canvas.addEventListener("mousedown", (ev) => {
    const p = toImage(ev);
    if (state.mode === "select") {
      const id = app.hit(p.x, p.y);
      state.selectedId = id;
      app.setSelected(id);
      if (id >= 0) {
        drag = { kind: "move", startId: id, startX: p.x, startY: p.y };
      }
      onChange();
    } else if (state.mode === "add") {
      drag = { kind: "add", startX: p.x, startY: p.y, curX: p.x, curY: p.y };
    } else if (state.mode === "region") {
      drag = { kind: "region", startX: p.x, startY: p.y, curX: p.x, curY: p.y };
    } else if (state.mode === "delete") {
      const id = app.hit(p.x, p.y);
      if (id >= 0) {
        app.removeBox(id);
        onChange();
      }
    } else if (state.mode === "order") {
      const id = app.hit(p.x, p.y);
      if (id >= 0 && !(state.drawSequence ?? []).includes(id)) {
        state.drawSequence = state.drawSequence ?? [];
        state.drawSequence.push(id);
        app.setCustomOrder(new Uint32Array(state.drawSequence));
        onChange();
      }
    }
  });

  canvas.addEventListener("mousemove", (ev) => {
    if (!drag) return;
    const p = toImage(ev);
    if (drag.kind === "add" || drag.kind === "region") {
      drag.curX = p.x;
      drag.curY = p.y;
      onChange();
    } else if (drag.kind === "move" && drag.startId >= 0) {
      const dx = p.x - drag.startX;
      const dy = p.y - drag.startY;
      const b = state.lastDetection?.boxes?.[drag.startId];
      if (!b) return;
      const xs = b.quad.map((pt) => pt[0]);
      const ys = b.quad.map((pt) => pt[1]);
      const xmin = Math.min(...xs) + dx;
      const ymin = Math.min(...ys) + dy;
      const xmax = Math.max(...xs) + dx;
      const ymax = Math.max(...ys) + dy;
      app.updateBox(drag.startId, xmin, ymin, xmax, ymax);
      onChange();
    }
  });

  canvas.addEventListener("mouseup", (ev) => {
    if (!drag) return;
    const p = toImage(ev);
    if (drag.kind === "add") {
      const x0 = Math.min(drag.startX, p.x);
      const y0 = Math.min(drag.startY, p.y);
      const x1 = Math.max(drag.startX, p.x);
      const y1 = Math.max(drag.startY, p.y);
      if (x1 - x0 > 2 && y1 - y0 > 2) {
        app.addBoxManual(x0, y0, x1, y1);
        onChange();
      }
    } else if (drag.kind === "region") {
      const x0 = Math.min(drag.startX, p.x);
      const y0 = Math.min(drag.startY, p.y);
      const x1 = Math.max(drag.startX, p.x);
      const y1 = Math.max(drag.startY, p.y);
      if (x1 - x0 > 4 && y1 - y0 > 4) {
        app.addRegion(x0, y0, x1, y1, "header", 0);
        onChange();
      }
    }
    drag = null;
  });

  return { setTool };
}
