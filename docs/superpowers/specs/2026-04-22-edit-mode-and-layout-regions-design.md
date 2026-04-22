# `docseg` — edit mode + layout regions

Status: design approved 2026-04-22. Implementation plan to follow.

## 1. Goal

Replace the current "tune + read-only overlay" model with "tune, then edit."
After detection fills in CharBoxes, the user can:

- Move, resize, delete individual boxes.
- Draw new boxes for missed glyphs.
- Group boxes into layout regions (e.g. the top-left rubric on the
  reference Yi manuscript is a Header distinct from the Body).
- Override reading order by clicking boxes in sequence (kept from the
  previous iteration).

Reading-order overlay is visually restructured to distinguish three kinds
of transition: within-line (continue), line-break / carriage return, and
region-break.

Manual edits (box moves, creations, deletions, region assignments) are
**preserved** across live-slider postprocess re-runs. Only auto-detected
boxes refresh when the region/affinity/erosion sliders change.

## 2. Non-goals

- **No automatic layout detection.** The user explicitly flagged this as
  future work. This spec's contribution is that regions are first-class
  objects in the data model, so a later auto-detector can emit them
  without any API change.
- **No polygon regions in v1.** Rectangles are sufficient for the target
  manuscripts. The `Region` type is designed so a polygon variant can be
  added later without breaking the UI/API contract — see §5.
- **No per-box confidence editor.** Score stays detector-assigned.
- **No undo/redo stack.** Reset-to-detection is the only broad revert.
  A redo/undo history is worth adding later but doubles the state-model
  scope.
- **No multi-select.** v1 edits one box at a time.

## 3. Problem recap (from the current demo)

Screenshot of the current state on `test_case1.png`, 2026-04-22 review:

- Reading-order arrows treat "label 1 → label 2 within the same column"
  and "label 8 → label 9 across a column break" identically. Both are a
  thin solid line with no arrowhead.
- The top-left rubric (4 small boxes above the page rule) is detected
  but ends up ordered as the LAST column under the vertical-RTL
  algorithm (labels 74/66/65/57) rather than FIRST (as a header).
- Users have no tool to nudge a drifted box, add a missed character,
  or remove a spurious detection. With ~100 boxes per page, 95% recall
  is not usable without manual correction.

## 4. User-facing changes

### 4.1 Tool palette

A small left-side tool palette or keyboard shortcuts toggle the edit
tool. Default tool is Select.

| Tool | Shortcut | Cursor | Effect |
|---|---|---|---|
| Select | `V` | pointer | Click = highlight. Drag body = move. Drag a handle = resize. Delete key deletes selected. |
| Add box | `A` | crosshair | Drag a rect on empty canvas → new `CharBox` with `manual = true`. |
| Delete | `D` | crosshair | Click a box → remove. |
| Order-draw | `O` | numbered crosshair | Click boxes in sequence → custom reading order. (Current "Draw order" behavior.) |
| Region | `R` | dashed crosshair | Drag a rect → new Region. Role defaults to Header and rank defaults to `(min existing rank) − 1` (new regions naturally land before existing ones). A small role/rank dropdown sits in the region's corner badge for editing afterwards. |

The three mode buttons (Auto order / Draw order / Reset) collapse into
this palette: Auto order = Select tool with show-labels on;
Draw order = Order-draw tool; Reset = clears the custom-order list.

### 4.2 Reading-order arrow styles

Three transition kinds, each visually distinct:

- **continue** (same line/column of same region, adjacent rank) —
  solid thin line, **no arrowhead**, color `rgba(80, 220, 255, 0.55)`.
  Implicit direction (the rank labels give it away).
- **carriage-return** (column-break or line-break within the same
  region) — **dashed line with a chevron arrowhead**, brighter
  `rgba(120, 240, 255, 0.9)`, width 2 px. The dash communicates "this
  is a break, not a continuation."
- **region-break** (last box of region N → first box of region N+1) —
  thicker (3 px) solid orange line with a larger arrowhead,
  `rgba(255, 160, 60, 0.9)`. A "big jump."

Heuristic for classifying an arrow:
- If boxes belong to different regions → region-break.
- Else if the axis orthogonal to the reading direction jumps by more
  than the median box dimension on that axis → carriage-return.
- Else → continue.

### 4.3 Region overlays

Each region renders as a translucent colored rectangle (role determines
the color) with a corner badge showing `rank · role` (e.g. `1 · Header`).
Roles and their defaults:

| Role | Default rank | Color |
|---|---|---|
| Header | 1 | blue |
| Body | 2 | no overlay (default background, so the body doesn't feel boxed-in) |
| Footer | 3 | green |
| Notes | 4 | violet |

If the user never creates any regions, the whole page is implicitly
Body — matches current behavior.

### 4.4 Slider behavior with manual edits

Moving a detection slider triggers a postprocess re-run. Manual state
that persists across a re-run:

- Boxes with `manual = true` (user-added or user-edited).
- All regions.
- Custom reading order (if set via Order-draw).

What gets regenerated:
- Boxes with `manual = false` (from CRAFT + postprocess).

Collision rule: if a regenerated auto-box has IoU > 0.5 with any manual
box, it is dropped (the manual box wins). Prevents drifted auto-boxes
from duplicating a user's correction.

## 5. Data model

### 5.1 `Region`

```rust
#[derive(Debug, Clone)]
pub struct Region {
    pub id: u32,
    pub shape: RegionShape,
    pub role: RegionRole,
    pub rank: u32,
}

#[derive(Debug, Clone)]
pub enum RegionShape {
    /// v1 — axis-aligned rectangle in original-image coordinates.
    Rect { xmin: f32, ymin: f32, xmax: f32, ymax: f32 },
    /// v2 — arbitrary polygon. Not exposed in v1 UI; API accepts it
    /// so callers that build regions outside the interactive tool
    /// (e.g. a future auto-detector) don't need an API bump.
    Polygon(Vec<Point>),
}

#[derive(Debug, Clone, Copy)]
pub enum RegionRole { Header, Body, Footer, Notes }
```

The enum leaves the v2 polygon path open while the v1 UI emits only
`Rect` shapes. `contains(point)` is defined for both variants (AABB
for Rect, even-odd fill for Polygon).

### 5.2 `CharBox` additions

```rust
pub struct CharBox {
    pub quad: Quad,
    pub score: f32,
    pub manual: bool,     // NEW — true if user-added or user-edited.
    // region assignment is derived from geometry at query time; no
    // region_id field avoids a stale-pointer invariant.
}
```

Region assignment = the first (lowest `rank`) region whose shape
contains the box centroid. Boxes outside every region fall into the
implicit Body region.

### 5.3 Reading-order algorithm update

```
compute_reading_order(boxes, regions, direction) -> Vec<usize>:
  1. Assign each box to a region (by centroid-inside, tie broken by rank).
  2. Group boxes by region, ordered by rank.
  3. Within each group, run the existing column-cluster + orthogonal-sort
     algorithm with the region's direction (v1: always same as page-wide
     direction — role-specific direction is a v2 nicety).
  4. Concatenate region-ordered groups.
```

A box not inside any region falls into the implicit Body group at rank 2
(so a user can create only a Header region and the rest flows as Body
without having to explicitly draw a body rect).

## 6. Architecture

- `docseg-core/src/regions.rs` — new module: `Region`, `RegionShape`,
  `RegionRole`, `Region::contains`, `assign_region`.
- `docseg-core/src/reading_order.rs` — extended with
  `compute_reading_order_with_regions(&[CharBox], &[Region], direction)`.
  Existing `compute_reading_order` keeps its current signature
  (calls the new function with empty regions).
- `docseg-core/src/postprocess.rs` — `CharBox` gains `manual: bool`.
  `charboxes_from_heatmap` sets `manual = false`. New
  `merge_manual_with_auto(auto: &[CharBox], manual: &[CharBox]) -> Vec<CharBox>`
  applies the IoU-0.5 collision rule.
- `docseg-web/src/entry.rs` — new methods on `DocsegApp`:
  - `addRegion(xmin, ymin, xmax, ymax, role, rank) -> u32` (returns id)
  - `removeRegion(id)`
  - `listRegions() -> JsValue` (serialized `Vec<RegionJs>`)
  - `addBox(xmin, ymin, xmax, ymax) -> u32`
  - `updateBox(id, xmin, ymin, xmax, ymax)`
  - `removeBox(id)`
  - `postprocess` signature unchanged externally, but internally preserves
    manual state across re-runs.
- `docseg-web/src/render.rs` — `paint_with_order` extended to classify
  each arrow as continue/carriage-return/region-break and style
  accordingly. Region overlays rendered in a new pass before box
  outlines.
- `web/main.js` — tool-palette UI, keyboard shortcuts, drag handlers
  for add/resize/move, region-role inline picker. Structured as a
  small `tools.js` helper module to keep `main.js` under the file-size
  limit.
- `web/index.html`, `web/style.css` — palette buttons + region badge
  styling.

## 7. Interaction detail: resize handles

Each selected axis-aligned box renders 8 handles (4 corner + 4 edge,
6×6 px squares). Dragging:

- Corner handle → scales both dimensions from the opposite corner.
- Edge handle → scales the one dimension it lies on.
- Box body → translates both dimensions.

Shift-drag constrains to square (for corner handles). Alt-drag
anchors to the center (scale uniformly around centroid). Both are
nice-to-haves; cut if they complicate the implementation.

Rotated quads (opted into via the existing `axis_aligned = false`
toggle) fall back to **move-only** in v1 — resizing a rotated quad
needs rotation-aware handles which is a v2.

## 8. Persistence

Export zip gets a `regions.json` alongside `boxes.json`:

```json
{
  "regions": [
    {
      "id": 0,
      "shape": { "type": "rect", "xmin": ..., "ymin": ..., "xmax": ..., "ymax": ... },
      "role": "Header",
      "rank": 1
    }
  ]
}
```

`boxes.json` gets the `manual` flag per entry so re-imports (future
feature) can distinguish user work from detector output.

No import path in v1 — the data is for external tooling.

## 9. Error handling

- Out-of-bounds coordinates on `addBox` / `addRegion` are clamped to
  `[0, original_size)`. No error.
- Zero-area rects on `addBox` / `addRegion` are rejected with
  `CoreError::Postprocess { reason: "zero-area region/box" }`.
- `updateBox(id, ...)` on an unknown id → silently no-op (return false
  from the boolean variant in JS). No panic.
- Region shape variant mismatch (v1 UI handing a polygon to the v1
  code path) is not possible because v1 UI only emits Rect.

## 10. Testing

Unit tests added in the relevant modules:

- `regions::tests`:
  - `rect_contains_returns_true_for_interior_point`
  - `rect_contains_returns_false_for_exterior_point`
  - `polygon_contains_handles_concave_shapes` (v2 plumbing, tested now)
- `reading_order::tests`:
  - `regions_order_per_rank_then_within`
  - `box_outside_every_region_falls_into_implicit_body`
  - `box_inside_overlapping_regions_goes_to_lowest_rank`
- `postprocess::tests`:
  - `merge_drops_auto_box_overlapping_manual_box` (IoU-0.5 rule)
  - `manual_box_flag_is_not_cleared_by_merge`

Integration-level verification happens in the browser as before;
wasm-bindgen tests remain out of scope.

## 11. File-size budget

The UI JS work is the only place file size is a real concern.
`web/main.js` is currently ~220 lines; the edit-mode logic (tool
state, handle geometry, hit-test, drag handlers) will add ~250.
Split along the natural seam:

```
web/main.js            # orchestration: session init, detection, paint
web/tools.js           # tool palette, selection state, drag handlers
web/regions-ui.js      # region rect draw + role picker
```

All under 300 lines each.

## 12. Flexibility for v2 (noted, not implemented)

- Polygon regions: API ready; UI needs a polygon tool.
- Auto-detected layout: emits `Region` objects via an async call; hooks
  into the same rendering path.
- Multi-select: needs a `selection: Vec<usize>` on app state; all edit
  ops become `for id in selection`.
- Undo/redo: ring buffer of app snapshots. Snapshots are small.
- Rotation-aware resize handles.
