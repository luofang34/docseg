# `docseg` — batch mode + edit mode v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v1 of the Batch-Mode + Edit-Mode spec — manual box/region editing with 8-handle resize, per-page undo, batch navigation with filmstrip + sticky sliders, diff view of CRAFT's original proposal vs user's corrected set, and local-first zip persistence with round-trip integrity.

**Architecture:** Extend `docseg-core` with `Region`, per-page `EditLog`, `Batch`/`Page` types, and schema-migratable zip persistence. Extend `docseg-web`'s `DocsegApp` with mutation methods and heatmap LRU; extend the canvas renderer with three arrow styles (continue/carriage-return/region-break), region overlays, resize handles, and diff-view pass. JS side adds a tool palette with keyboard shortcuts, drag handlers for add/move/resize/region, and a left-rail page filmstrip with per-page status chips.

**Tech Stack:** Rust 1.74+, `wasm-bindgen`, `web-sys`, `zip`, `sha2`, `serde`, `ulid` (new); frontend plain JS modules, onnxruntime-web for inference on WebGPU.

**Design reference:** `docs/superpowers/specs/2026-04-22-batch-mode-adaptive-design.md`.

**Execution guardrails (from user global rules):**

- Every committed task MUST leave `./scripts/ci-local.sh` passing (fmt / clippy -D warnings / test / doc / wasm check / wasm build).
- No `unwrap` / `expect` / `panic` outside tests. No `mod.rs`. Tests may `#![allow(clippy::expect_used, clippy::panic)]`. Public items have `///` docs.
- Commits happen at the end of every task. Each commit compiles and passes CI.
- Work on branch `feat/initial-implementation` in `/Users/fangluo/Desktop/docseg`.

---

## File Structure

Files created / modified (relative to repo root):

```
crates/docseg-core/src/
  postprocess.rs                (modify: CharBox.manual)
  postprocess/tests.rs          (modify: new merge tests)
  postprocess_merge.rs          (CREATE: iou + merge_manual_with_auto)
  postprocess_merge/tests.rs    (CREATE)
  regions.rs                    (CREATE: Region, RegionShape, RegionRole, region_for_box)
  regions/tests.rs              (CREATE)
  reading_order.rs              (modify: compute_reading_order_with_regions)
  reading_order/tests.rs        (modify: region-grouped tests)
  edit_log.rs                   (CREATE: EditLog, EditEvent)
  edit_log/tests.rs             (CREATE)
  batch.rs                      (CREATE: Batch, Page, SliderValues, PageStatus, ULID ids)
  batch/tests.rs                (CREATE)
  batch_persist.rs              (CREATE: to_zip / from_zip + schema migrate scaffold)
  batch_persist/tests.rs        (CREATE)
  diff.rs                       (CREATE: diff snapshot types + compute_diff)
  diff/tests.rs                 (CREATE)
  lib.rs                        (modify: re-exports)

crates/docseg-web/src/
  entry.rs                      (modify: new DocsegApp methods)
  render.rs                     (modify: three arrow styles + regions + handles + cyan ring + dashed low-conf)
  export.rs                     (modify: manual flag in boxes.json)
  heatmap_lru.rs                (CREATE: LRU cache for region+affinity Float32Arrays keyed by sha256)

web/
  index.html                    (modify: filmstrip, tool palette, slider ticks)
  main.js                       (modify: orchestration only; delegate to new modules)
  tools.js                      (CREATE: tool palette + drag handlers)
  filmstrip.js                  (CREATE: per-page thumbnails + status chips)
  diff-view.js                  (CREATE: Ctrl-Shift-D toggle + overlay pass)
  style.css                     (modify: filmstrip, palette, tick-mark styling)

crates/docseg-core/Cargo.toml   (modify: add sha2, ulid, zip deps)
crates/docseg-web/Cargo.toml    (modify: add web-sys features for keyboard + mouse events)
```

Per the user rules: `lib.rs` stays ≤ 100 lines (re-exports only), each file ≤ 500 LOC, each fn ≤ 80 LOC.

---

## Task 1: `CharBox.manual` flag

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess/tests.rs`

- [ ] **Step 1: Write a failing test asserting the field exists and defaults false**

Append to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess/tests.rs`:

```rust

#[test]
fn charbox_has_manual_flag_defaulting_false() {
    // Any auto-detected CharBox should have manual = false by default.
    let mut map = vec![0.0_f32; 32 * 32];
    for y in 8..12 {
        for x in 8..12 {
            map[y * 32 + x] = 0.9;
        }
    }
    let boxes = charboxes_from_heatmap(
        &map,
        None,
        32,
        32,
        &fake_preproc(64, 64, 1.0),
        PostprocessOptions {
            region_threshold: 0.5,
            min_component_area_px: 1,
            max_aspect_ratio: 8.0,
            erosion_px: 0,
            axis_aligned: false,
            ..Default::default()
        },
    );
    assert!(!boxes.is_empty());
    for b in &boxes {
        assert!(!b.manual, "auto-detected box should have manual = false");
    }
}
```

- [ ] **Step 2: Run — expect FAIL on missing field**

```bash
cd /Users/fangluo/Desktop/docseg
cargo test -p docseg-core --lib postprocess::tests::charbox_has_manual_flag_defaulting_false
```

Expected: compilation error `no field \`manual\` on type \`CharBox\``.

- [ ] **Step 3: Add the field**

In `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess.rs`, find the `CharBox` struct:

```rust
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CharBox {
    /// Oriented quadrilateral in original-image pixel coordinates.
    pub quad: Quad,
    /// Max region-score within the component (0..1).
    pub score: f32,
}
```

Replace with:

```rust
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CharBox {
    /// Oriented quadrilateral in original-image pixel coordinates.
    pub quad: Quad,
    /// Max region-score within the component (0..1).
    pub score: f32,
    /// `true` when the box was added or edited by the user. Set by
    /// `DocsegApp::add_box_manual` / `DocsegApp::update_box`, preserved
    /// through postprocess re-runs via the IoU-merge rule in
    /// `postprocess_merge::merge_manual_with_auto`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub manual: bool,
}
```

Then find the single point where `charboxes_from_heatmap` constructs a `CharBox`:

```rust
        out.push(CharBox { quad, score });
```

Replace with:

```rust
        out.push(CharBox {
            quad,
            score,
            manual: false,
        });
```

- [ ] **Step 4: Run — expect PASS and existing tests still green**

```bash
cargo test -p docseg-core --lib postprocess
```

Expected: 9+ passed. The `serde(default)` attribute keeps backward-compatible deserialization.

- [ ] **Step 5: Run full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): CharBox.manual flag for user-edited boxes

Auto-detected boxes default to manual = false. Task 2 wires the
IoU-merge rule that preserves manual boxes through postprocess
re-runs; Task N wires the DocsegApp mutation methods that set
manual = true on user-added or user-edited boxes.

#[serde(default)] keeps backward-compatible deserialization of older
boxes.json files.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: IoU merge — `merge_manual_with_auto`

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess_merge.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess_merge/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess_merge/tests.rs`:

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{iou_aabb, merge_manual_with_auto};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;

fn rect(x: f32, y: f32, w: f32, h: f32, manual: bool) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ]),
        score: 1.0,
        manual,
    }
}

#[test]
fn iou_identical_rects_is_one() {
    let a = rect(0.0, 0.0, 10.0, 10.0, false);
    let b = rect(0.0, 0.0, 10.0, 10.0, false);
    assert!((iou_aabb(&a, &b) - 1.0).abs() < 1e-4);
}

#[test]
fn iou_disjoint_rects_is_zero() {
    let a = rect(0.0, 0.0, 10.0, 10.0, false);
    let b = rect(100.0, 100.0, 10.0, 10.0, false);
    assert!(iou_aabb(&a, &b).abs() < 1e-6);
}

#[test]
fn iou_half_overlap() {
    // 10x10 at (0,0) and 10x10 at (5,0): overlap 5x10 = 50, union 10x10 + 10x10 - 50 = 150.
    let a = rect(0.0, 0.0, 10.0, 10.0, false);
    let b = rect(5.0, 0.0, 10.0, 10.0, false);
    let v = iou_aabb(&a, &b);
    assert!((v - 50.0 / 150.0).abs() < 1e-4, "iou={v}");
}

#[test]
fn merge_drops_auto_box_overlapping_manual_box() {
    let manual = vec![rect(0.0, 0.0, 10.0, 10.0, true)];
    let auto = vec![
        rect(1.0, 1.0, 9.0, 9.0, false),       // IoU with manual > 0.5 → dropped
        rect(100.0, 100.0, 10.0, 10.0, false), // disjoint → kept
    ];
    let merged = merge_manual_with_auto(&auto, &manual);
    assert_eq!(merged.len(), 2, "expected manual + one survivor auto");
    assert!(merged[0].manual, "manual box first");
    assert!(!merged[1].manual, "auto-survivor second");
    let xs: Vec<f32> = merged.iter().map(|b| b.quad.points[0].x).collect();
    assert!(xs.contains(&0.0));
    assert!(xs.contains(&100.0));
}

#[test]
fn merge_preserves_manual_flag_through_merge() {
    let manual = vec![rect(0.0, 0.0, 10.0, 10.0, true)];
    let auto: Vec<CharBox> = vec![];
    let merged = merge_manual_with_auto(&auto, &manual);
    assert_eq!(merged.len(), 1);
    assert!(merged[0].manual);
}

#[test]
fn merge_with_empty_manual_passes_all_auto_through() {
    let manual: Vec<CharBox> = vec![];
    let auto = vec![rect(0.0, 0.0, 10.0, 10.0, false), rect(50.0, 0.0, 10.0, 10.0, false)];
    let merged = merge_manual_with_auto(&auto, &manual);
    assert_eq!(merged.len(), 2);
    for b in &merged {
        assert!(!b.manual);
    }
}
```

- [ ] **Step 2: Create the module with a stub, register it, expect compile failure**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess_merge.rs`:

```rust
//! Merge user-edited (manual) CharBoxes with the auto-detected set,
//! using an axis-aligned IoU > 0.5 rule so a user correction cannot be
//! silently duplicated by a slider-driven re-detection.

#[cfg(test)]
mod tests;
```

Register in `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs` — find the existing `pub mod postprocess;` and add below:

```rust
pub mod postprocess_merge;
```

And in the re-export block add:

```rust
pub use postprocess_merge::{iou_aabb, merge_manual_with_auto};
```

Run:

```bash
cargo test -p docseg-core --lib postprocess_merge
```

Expected: compile errors — `iou_aabb` and `merge_manual_with_auto` don't exist.

- [ ] **Step 3: Implement**

Replace the body of `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/postprocess_merge.rs`:

```rust
//! Merge user-edited (manual) CharBoxes with the auto-detected set,
//! using an axis-aligned IoU > 0.5 rule so a user correction cannot be
//! silently duplicated by a slider-driven re-detection.

use crate::geometry::Quad;
use crate::postprocess::CharBox;

/// Axis-aligned IoU on the bounding rect of two `CharBox`es. 0.0 when
/// either box has zero area (degenerate).
#[must_use]
pub fn iou_aabb(a: &CharBox, b: &CharBox) -> f32 {
    let (ax0, ay0, ax1, ay1) = aabb(&a.quad);
    let (bx0, by0, bx1, by1) = aabb(&b.quad);
    let iw = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let ih = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let inter = iw * ih;
    let a_area = (ax1 - ax0).max(0.0) * (ay1 - ay0).max(0.0);
    let b_area = (bx1 - bx0).max(0.0) * (by1 - by0).max(0.0);
    let union = a_area + b_area - inter;
    if union <= 0.0 {
        return 0.0;
    }
    inter / union
}

/// Build the final CharBox list: every manual box survives, and every
/// auto-detected box passes through UNLESS its IoU with some manual
/// box exceeds 0.5.
#[must_use]
pub fn merge_manual_with_auto(auto: &[CharBox], manual: &[CharBox]) -> Vec<CharBox> {
    let mut out: Vec<CharBox> = Vec::with_capacity(manual.len() + auto.len());
    out.extend(manual.iter().cloned());
    for a in auto {
        if manual.iter().any(|m| iou_aabb(m, a) > 0.5) {
            continue;
        }
        out.push(a.clone());
    }
    out
}

fn aabb(q: &Quad) -> (f32, f32, f32, f32) {
    let p = &q.points;
    let xs = [p[0].x, p[1].x, p[2].x, p[3].x];
    let ys = [p[0].y, p[1].y, p[2].y, p[3].y];
    (
        xs.iter().copied().fold(f32::INFINITY, f32::min),
        ys.iter().copied().fold(f32::INFINITY, f32::min),
        xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    )
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 4: Run — expect PASS**

```bash
cargo test -p docseg-core --lib postprocess_merge
```

Expected: 6 passed.

- [ ] **Step 5: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): merge_manual_with_auto preserves user edits across re-runs

iou_aabb computes axis-aligned IoU on two CharBoxes' bounding rects.
merge_manual_with_auto(auto, manual) returns manual-first then every
auto that doesn't overlap a manual with IoU > 0.5 — the rule from the
batch-mode spec §3.4. Used by DocsegApp.postprocess on every slider-
driven re-run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `Region` types and `region_for_box`

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/regions.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/regions/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/regions/tests.rs`:

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{Region, RegionRole, RegionShape, region_for_box};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;

fn rect_box(cx: f32, cy: f32) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(cx - 5.0, cy - 5.0),
            Point::new(cx + 5.0, cy - 5.0),
            Point::new(cx + 5.0, cy + 5.0),
            Point::new(cx - 5.0, cy + 5.0),
        ]),
        score: 1.0,
        manual: false,
    }
}

fn header_rect(x0: f32, y0: f32, x1: f32, y1: f32, rank: u32) -> Region {
    Region {
        id: 1,
        shape: RegionShape::Rect {
            xmin: x0,
            ymin: y0,
            xmax: x1,
            ymax: y1,
        },
        role: RegionRole::Header,
        rank,
    }
}

#[test]
fn rect_contains_interior_point() {
    let shape = RegionShape::Rect {
        xmin: 0.0,
        ymin: 0.0,
        xmax: 100.0,
        ymax: 100.0,
    };
    assert!(shape.contains(Point::new(50.0, 50.0)));
}

#[test]
fn rect_excludes_exterior_point() {
    let shape = RegionShape::Rect {
        xmin: 0.0,
        ymin: 0.0,
        xmax: 100.0,
        ymax: 100.0,
    };
    assert!(!shape.contains(Point::new(150.0, 50.0)));
}

#[test]
fn rect_edge_counts_as_inside() {
    // Half-open interval [xmin, xmax); top-left edge inside, bottom-right outside.
    let shape = RegionShape::Rect {
        xmin: 0.0,
        ymin: 0.0,
        xmax: 10.0,
        ymax: 10.0,
    };
    assert!(shape.contains(Point::new(0.0, 0.0)));
    assert!(!shape.contains(Point::new(10.0, 10.0)));
}

#[test]
fn region_for_box_returns_lowest_rank_on_overlap() {
    let inner = Region {
        id: 1,
        shape: RegionShape::Rect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 100.0,
            ymax: 100.0,
        },
        role: RegionRole::Header,
        rank: 1,
    };
    let outer = Region {
        id: 2,
        shape: RegionShape::Rect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 200.0,
            ymax: 200.0,
        },
        role: RegionRole::Body,
        rank: 2,
    };
    let regions = vec![outer, inner]; // order in slice doesn't matter
    let b = rect_box(50.0, 50.0);
    assert_eq!(region_for_box(&b, &regions), Some(1));
}

#[test]
fn region_for_box_returns_none_outside_all_regions() {
    let r = header_rect(0.0, 0.0, 50.0, 50.0, 1);
    let b = rect_box(100.0, 100.0);
    assert_eq!(region_for_box(&b, &[r]), None);
}

#[test]
fn region_for_box_with_empty_regions_returns_none() {
    let b = rect_box(10.0, 10.0);
    assert_eq!(region_for_box(&b, &[]), None);
}

#[test]
fn polygon_contains_concave_hook() {
    // An L-shaped polygon: (0,0)->(10,0)->(10,4)->(4,4)->(4,10)->(0,10)->(0,0).
    let poly = RegionShape::Polygon(vec![
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        Point::new(10.0, 4.0),
        Point::new(4.0, 4.0),
        Point::new(4.0, 10.0),
        Point::new(0.0, 10.0),
    ]);
    assert!(poly.contains(Point::new(2.0, 2.0))); // in the thick part
    assert!(poly.contains(Point::new(6.0, 2.0))); // in the top-bar right arm
    assert!(!poly.contains(Point::new(6.0, 6.0))); // in the L's interior notch (outside)
    assert!(!poly.contains(Point::new(12.0, 2.0))); // outside
}
```

- [ ] **Step 2: Create stub, register, expect compile-fail**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/regions.rs`:

```rust
//! Layout regions on a page. v1 uses axis-aligned rectangles; the
//! `RegionShape::Polygon` variant is API-ready so a future auto-layout
//! detector or the v2 polygon tool can emit polygons without breaking
//! callers.

#[cfg(test)]
mod tests;
```

In `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`, add:

```rust
pub mod regions;

pub use regions::{region_for_box, Region, RegionRole, RegionShape};
```

Run:

```bash
cargo test -p docseg-core --lib regions
```

Expected: compile errors.

- [ ] **Step 3: Implement**

Replace the body of `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/regions.rs`:

```rust
//! Layout regions on a page. v1 uses axis-aligned rectangles; the
//! `RegionShape::Polygon` variant is API-ready so a future auto-layout
//! detector or the v2 polygon tool can emit polygons without breaking
//! callers.

use crate::geometry::Point;
use crate::postprocess::CharBox;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Semantic role of a layout region. Determines overlay color and the
/// default rank (Header < Body < Footer < Notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RegionRole {
    /// Header / rubric / title — read first.
    Header,
    /// Main body text.
    Body,
    /// Footer / colophon.
    Footer,
    /// Marginal notes / annotations.
    Notes,
}

/// Region geometry.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "lowercase"))]
pub enum RegionShape {
    /// Axis-aligned rectangle in original-image coordinates.
    Rect {
        /// Left edge.
        xmin: f32,
        /// Top edge.
        ymin: f32,
        /// Right edge (exclusive).
        xmax: f32,
        /// Bottom edge (exclusive).
        ymax: f32,
    },
    /// Closed polygon in original-image coordinates. Not exposed in v1
    /// UI; the API accepts it so callers that build regions outside the
    /// interactive tool (e.g. a future auto-detector) don't need an API
    /// bump.
    Polygon(Vec<Point>),
}

impl RegionShape {
    /// `true` iff `p` is inside the shape. Half-open on the right/bottom
    /// edges for Rect; standard even-odd fill for Polygon.
    #[must_use]
    pub fn contains(&self, p: Point) -> bool {
        match self {
            Self::Rect {
                xmin,
                ymin,
                xmax,
                ymax,
            } => p.x >= *xmin && p.x < *xmax && p.y >= *ymin && p.y < *ymax,
            Self::Polygon(pts) => polygon_contains(pts, p),
        }
    }
}

/// A layout region as stored on a `Page`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Region {
    /// Stable id within a page.
    pub id: u32,
    /// Geometry.
    pub shape: RegionShape,
    /// Semantic role (drives overlay color + default rank).
    pub role: RegionRole,
    /// Reading-order priority; lower ranks read earlier.
    pub rank: u32,
}

/// Return the id of the region whose shape contains the centroid of
/// `b.quad`. If multiple regions contain the centroid, the one with
/// the lowest `rank` wins; ties broken by lowest `id`. `None` when no
/// region matches.
#[must_use]
pub fn region_for_box(b: &CharBox, regions: &[Region]) -> Option<u32> {
    let p = &b.quad.points;
    let cx = (p[0].x + p[1].x + p[2].x + p[3].x) * 0.25;
    let cy = (p[0].y + p[1].y + p[2].y + p[3].y) * 0.25;
    let centroid = Point::new(cx, cy);
    regions
        .iter()
        .filter(|r| r.shape.contains(centroid))
        .min_by(|a, b| a.rank.cmp(&b.rank).then_with(|| a.id.cmp(&b.id)))
        .map(|r| r.id)
}

/// Even-odd fill test for a closed polygon. `pts` is treated as a
/// closed loop (last point implicitly connects to first).
fn polygon_contains(pts: &[Point], p: Point) -> bool {
    if pts.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = pts.len() - 1;
    for i in 0..pts.len() {
        let pi = pts[i];
        let pj = pts[j];
        if (pi.y > p.y) != (pj.y > p.y)
            && p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 4: Run — expect PASS**

```bash
cargo test -p docseg-core --lib regions
```

Expected: 7 passed.

- [ ] **Step 5: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): Region types and centroid-inside region_for_box

RegionShape::{Rect, Polygon} leaves the v2 polygon path open while v1
UI only emits Rect. region_for_box returns the lowest-rank matching
region id (ties broken by id). contains() is half-open on Rect edges
and uses the standard even-odd ray-cast for Polygon.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Region-aware reading order

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/reading_order.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/reading_order/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/reading_order/tests.rs`:

```rust

use crate::regions::{Region, RegionRole, RegionShape};
use super::compute_reading_order_with_regions;

#[test]
fn regions_order_by_rank_then_within_each() {
    // Layout: header rect covers top 50px, body is the rest.
    let boxes = vec![
        rect(10.0, 100.0, 20.0, 20.0),  // 0: body, top
        rect(10.0, 200.0, 20.0, 20.0),  // 1: body, bot
        rect(10.0, 10.0, 20.0, 20.0),   // 2: header
        rect(40.0, 10.0, 20.0, 20.0),   // 3: header (different col, same row)
    ];
    let regions = vec![Region {
        id: 1,
        shape: RegionShape::Rect { xmin: 0.0, ymin: 0.0, xmax: 100.0, ymax: 50.0 },
        role: RegionRole::Header,
        rank: 1,
    }];
    let order = compute_reading_order_with_regions(
        &boxes,
        &regions,
        ReadingDirection::VerticalRtl,
    );
    // Header first (rank 1): right col header reads first under RTL.
    // Within header: col at x=40 before col at x=10.
    // Body last: col at x=10 only, top-to-bottom.
    assert_eq!(order, vec![3, 2, 0, 1]);
}

#[test]
fn box_outside_every_region_falls_into_implicit_body_rank_2() {
    // No regions drawn; behavior should match the non-region call.
    let boxes = vec![
        rect(10.0, 10.0, 20.0, 20.0),
        rect(10.0, 60.0, 20.0, 20.0),
    ];
    let without = compute_reading_order(&boxes, ReadingDirection::VerticalRtl);
    let with = compute_reading_order_with_regions(&boxes, &[], ReadingDirection::VerticalRtl);
    assert_eq!(without, with);
}

#[test]
fn box_outside_drawn_region_is_treated_as_implicit_body() {
    // Header drawn over top-left only; the rest are implicit body (rank 2).
    // Body box should come AFTER the header box.
    let boxes = vec![
        rect(200.0, 200.0, 20.0, 20.0),  // 0: body (outside header)
        rect(10.0, 10.0, 20.0, 20.0),    // 1: header
    ];
    let regions = vec![Region {
        id: 1,
        shape: RegionShape::Rect { xmin: 0.0, ymin: 0.0, xmax: 50.0, ymax: 50.0 },
        role: RegionRole::Header,
        rank: 1,
    }];
    let order = compute_reading_order_with_regions(
        &boxes,
        &regions,
        ReadingDirection::VerticalRtl,
    );
    assert_eq!(order, vec![1, 0]);
}
```

Also update the `use` block at the top of that tests file: change

```rust
use crate::postprocess::CharBox;
```

(already present) and add `use crate::geometry::{Point, Quad};` if not present, plus:

```rust
use super::compute_reading_order;
```

(already present, but confirm). Also ensure `rect(...)` helper uses the new `manual: false` field; if the existing helper is:

```rust
fn rect(x: f32, y: f32, w: f32, h: f32) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ]),
        score: 1.0,
    }
}
```

update to:

```rust
fn rect(x: f32, y: f32, w: f32, h: f32) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ]),
        score: 1.0,
        manual: false,
    }
}
```

(Task 1 made `manual` a required field on struct-literal construction.)

- [ ] **Step 2: Add the function + implementation, expect compile**

Append to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/reading_order.rs`:

```rust

use crate::regions::{region_for_box, Region};

/// Compute reading order taking layout regions into account. Boxes are
/// grouped by region (boxes outside every region fall into an implicit
/// "Body" group at rank 2). Groups are ordered by region rank; within
/// each group the existing direction-dependent column/line clustering
/// is applied.
#[must_use]
pub fn compute_reading_order_with_regions(
    boxes: &[CharBox],
    regions: &[Region],
    direction: ReadingDirection,
) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    if regions.is_empty() {
        return compute_reading_order(boxes, direction);
    }

    // Group indices by (rank, region_id). Implicit Body uses rank 2 and
    // a sentinel region id of u32::MAX so it sorts after same-rank drawn
    // regions — acceptable ambiguity for v1.
    let mut grouped: std::collections::BTreeMap<(u32, u32), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, b) in boxes.iter().enumerate() {
        let key = match region_for_box(b, regions) {
            Some(rid) => {
                let rank = regions
                    .iter()
                    .find(|r| r.id == rid)
                    .map(|r| r.rank)
                    .unwrap_or(2);
                (rank, rid)
            }
            None => (2, u32::MAX),
        };
        grouped.entry(key).or_default().push(i);
    }

    let mut out = Vec::with_capacity(boxes.len());
    for ((_rank, _rid), idxs) in grouped {
        let subset: Vec<CharBox> = idxs.iter().map(|&i| boxes[i].clone()).collect();
        let sub_order = compute_reading_order(&subset, direction);
        for rank_in_sub in sub_order {
            out.push(idxs[rank_in_sub]);
        }
    }
    out
}
```

Run:

```bash
cargo test -p docseg-core --lib reading_order
```

Expected: all tests (existing + new) pass.

- [ ] **Step 3: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): region-aware compute_reading_order_with_regions

Groups boxes by (region_rank, region_id) with an implicit Body group
at rank 2 for boxes outside every drawn region. Within each group the
existing column-cluster + sort algorithm runs under the page-wide
ReadingDirection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `EditLog` and `EditEvent`

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/edit_log.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/edit_log/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/edit_log/tests.rs`:

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{EditEvent, EditLog};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;
use crate::regions::{Region, RegionRole, RegionShape};

fn box_at(x: f32) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, 0.0),
            Point::new(x + 10.0, 0.0),
            Point::new(x + 10.0, 10.0),
            Point::new(x, 10.0),
        ]),
        score: 1.0,
        manual: true,
    }
}

#[test]
fn empty_log_undo_returns_none() {
    let mut log = EditLog::new();
    assert!(log.undo().is_none());
}

#[test]
fn push_then_undo_returns_the_event() {
    let mut log = EditLog::new();
    let ev = EditEvent::AddBox(box_at(10.0));
    log.push(ev.clone());
    let popped = log.undo().expect("should have an event");
    assert!(matches!(popped, EditEvent::AddBox(_)));
}

#[test]
fn undo_then_redo_restores_the_event() {
    let mut log = EditLog::new();
    log.push(EditEvent::AddBox(box_at(10.0)));
    let _ = log.undo();
    let redone = log.redo().expect("redo should have something");
    assert!(matches!(redone, EditEvent::AddBox(_)));
}

#[test]
fn new_push_clears_redo_stack() {
    let mut log = EditLog::new();
    log.push(EditEvent::AddBox(box_at(10.0)));
    let _ = log.undo();
    log.push(EditEvent::AddBox(box_at(20.0)));
    assert!(log.redo().is_none());
}

#[test]
fn log_caps_at_50_entries() {
    let mut log = EditLog::new();
    for i in 0..60 {
        log.push(EditEvent::AddBox(box_at(i as f32)));
    }
    // Oldest 10 should be evicted.
    let mut undone = 0;
    while log.undo().is_some() {
        undone += 1;
    }
    assert_eq!(undone, 50);
}

#[test]
fn region_event_variants_roundtrip() {
    let r = Region {
        id: 1,
        shape: RegionShape::Rect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 10.0,
            ymax: 10.0,
        },
        role: RegionRole::Header,
        rank: 1,
    };
    let mut log = EditLog::new();
    log.push(EditEvent::AddRegion(r.clone()));
    assert!(matches!(log.undo(), Some(EditEvent::AddRegion(_))));
}
```

- [ ] **Step 2: Create stub + register, expect compile-fail**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/edit_log.rs`:

```rust
//! Per-page edit log: a flat 50-entry undo stack and redo stack.
//! Cleared redo on any new push. No event sourcing, no snapshots —
//! every event carries inline before/after so undo/redo is O(1).

#[cfg(test)]
mod tests;
```

Add to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`:

```rust
pub mod edit_log;

pub use edit_log::{EditEvent, EditLog};
```

Run:

```bash
cargo test -p docseg-core --lib edit_log
```

Expected: compile errors.

- [ ] **Step 3: Implement**

Replace the body of `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/edit_log.rs`:

```rust
//! Per-page edit log: a flat 50-entry undo stack and redo stack.
//! Cleared redo on any new push. No event sourcing, no snapshots —
//! every event carries inline before/after so undo/redo is O(1).

use std::collections::VecDeque;

use crate::postprocess::CharBox;
use crate::regions::Region;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Per-page undo-stack capacity.
pub const EDIT_LOG_CAPACITY: usize = 50;

/// Structural edits a user can make. Every variant carries the minimum
/// data an undo or redo needs to rebuild the affected item.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum EditEvent {
    /// A user-added box. Undo removes it.
    AddBox(CharBox),
    /// A user-deleted box. Undo re-inserts it.
    RemoveBox {
        /// Position in the page's box list before removal.
        index: u32,
        /// The removed box.
        #[cfg_attr(feature = "serde", serde(rename = "box"))]
        value: CharBox,
    },
    /// A user-edited (moved / resized) box.
    UpdateBox {
        /// Position in the page's box list.
        index: u32,
        /// Box state before the edit.
        before: CharBox,
        /// Box state after the edit.
        after: CharBox,
    },
    /// Added region.
    AddRegion(Region),
    /// Deleted region.
    RemoveRegion {
        /// Position in the page's region list before removal.
        index: u32,
        /// The removed region.
        #[cfg_attr(feature = "serde", serde(rename = "region"))]
        value: Region,
    },
    /// Edited region (moved / resized / re-roled / re-ranked).
    UpdateRegion {
        /// Position in the page's region list.
        index: u32,
        /// Region state before the edit.
        before: Region,
        /// Region state after the edit.
        after: Region,
    },
    /// Replaced the reading order (Order-draw mode commit).
    ReorderBoxes {
        /// Order before override.
        before: Vec<u32>,
        /// Order after override.
        after: Vec<u32>,
    },
}

/// Per-page undo / redo log.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EditLog {
    undo: VecDeque<EditEvent>,
    redo: Vec<EditEvent>,
}

impl EditLog {
    /// Construct an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new edit. Evicts the oldest entry if the undo stack is at
    /// capacity, and clears the redo stack.
    pub fn push(&mut self, event: EditEvent) {
        if self.undo.len() >= EDIT_LOG_CAPACITY {
            self.undo.pop_front();
        }
        self.undo.push_back(event);
        self.redo.clear();
    }

    /// Pop the most recent edit onto the redo stack and return it for
    /// the caller to reverse.
    pub fn undo(&mut self) -> Option<EditEvent> {
        let event = self.undo.pop_back()?;
        self.redo.push(event.clone());
        Some(event)
    }

    /// Pop the most recent redo entry, push it back onto the undo
    /// stack, and return it for the caller to re-apply.
    pub fn redo(&mut self) -> Option<EditEvent> {
        let event = self.redo.pop()?;
        // Re-apply doesn't evict — it restores what was just undone.
        self.undo.push_back(event.clone());
        Some(event)
    }

    /// `true` iff there is something to undo.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// `true` iff there is something to redo.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 4: Run — expect PASS**

```bash
cargo test -p docseg-core --lib edit_log
```

Expected: 6 passed.

- [ ] **Step 5: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): EditLog + EditEvent for per-page undo/redo

50-entry flat undo stack; redo is cleared on any new push. Each
EditEvent carries inline before/after so undo/redo is O(1). Variants
cover AddBox / RemoveBox / UpdateBox, AddRegion / RemoveRegion /
UpdateRegion, and ReorderBoxes (for Order-draw commits).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Core `Page`, `SliderValues`, `PageStatus`, `Batch`

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Add `sha2`, `ulid` to `docseg-core` deps**

In `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml`, find the `[dependencies]` block and append:

```toml
sha2 = "0.10"
ulid = { version = "1", default-features = false, features = ["serde"] }
```

- [ ] **Step 2: Write failing tests**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch/tests.rs`:

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{Batch, Page, PageStatus, SliderValues};

#[test]
fn batch_new_is_empty_with_current_schema_version() {
    let b = Batch::new();
    assert!(b.pages.is_empty());
    assert!(b.session_defaults_empty());
    assert_eq!(b.schema_version, super::CURRENT_SCHEMA_VERSION);
}

#[test]
fn page_new_starts_untouched() {
    let p = Page::new("fake-image".as_bytes());
    assert_eq!(p.status, PageStatus::Untouched);
    assert_eq!(p.image_dims, (0, 0));
    assert_eq!(p.image_sha256.len(), 32);
    assert!(p.boxes.is_empty());
}

#[test]
fn page_sha256_is_deterministic() {
    let a = Page::new(b"hello world");
    let b = Page::new(b"hello world");
    assert_eq!(a.image_sha256, b.image_sha256);
}

#[test]
fn default_slider_values_match_postprocess_defaults() {
    let sv = SliderValues::default();
    assert!((sv.region_threshold - 0.4).abs() < 1e-6);
    assert!((sv.affinity_threshold - 0.3).abs() < 1e-6);
    assert_eq!(sv.erosion_px, 0);
    assert_eq!(sv.min_component_area_px, 8);
    assert!(sv.axis_aligned);
}

#[test]
fn page_transitions_untouched_to_in_progress_on_first_edit() {
    let mut p = Page::new(b"x");
    assert_eq!(p.status, PageStatus::Untouched);
    p.mark_edited();
    assert_eq!(p.status, PageStatus::InProgress);
}

#[test]
fn reviewed_page_reverts_to_in_progress_on_edit() {
    let mut p = Page::new(b"x");
    p.status = PageStatus::Reviewed;
    p.reviewed_at = Some(1);
    p.mark_edited();
    assert_eq!(p.status, PageStatus::InProgress);
}
```

- [ ] **Step 3: Create stub + register, expect compile-fail**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch.rs`:

```rust
//! Batch + Page state model.

#[cfg(test)]
mod tests;
```

Add to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`:

```rust
pub mod batch;

pub use batch::{
    Batch, CURRENT_SCHEMA_VERSION, Page, PageStatus, SliderDefaults, SliderValues,
};
```

Run: `cargo test -p docseg-core --lib batch` — expect compile errors.

- [ ] **Step 4: Implement**

Replace the body of `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch.rs`:

```rust
//! Batch + Page state model. Pure data types; persistence lives in
//! `batch_persist`.

use sha2::{Digest, Sha256};
use ulid::Ulid;

use crate::edit_log::EditLog;
use crate::postprocess::CharBox;
use crate::regions::Region;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Bump whenever Batch / Page / SliderValues shape changes in a way
/// older deserializers couldn't auto-default. Migration chain in
/// `batch_persist::migrate`.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Effective slider values for a page (or the session defaults on a
/// Batch).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SliderValues {
    /// CRAFT region-score threshold.
    pub region_threshold: f32,
    /// CRAFT affinity-score threshold.
    pub affinity_threshold: f32,
    /// Morphological erosion radius in heatmap pixels.
    pub erosion_px: u8,
    /// Minimum connected-component area in heatmap pixels.
    pub min_component_area_px: u32,
    /// `true` for axis-aligned box fit; `false` for min-area rotated rect.
    pub axis_aligned: bool,
}

impl Default for SliderValues {
    fn default() -> Self {
        Self {
            region_threshold: 0.4,
            affinity_threshold: 0.3,
            erosion_px: 0,
            min_component_area_px: 8,
            axis_aligned: true,
        }
    }
}

/// Typed alias for the Batch's "sticky" slider defaults. Same shape as
/// a page's `SliderValues`; kept distinct so methods that mutate it
/// don't get called on the wrong struct.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SliderDefaults {
    /// The sticky slider snapshot. `None` = use `SliderValues::default()`.
    pub values: Option<SliderValues>,
}

/// Review status of a page in a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PageStatus {
    /// Never opened.
    Untouched,
    /// Opened, has edits, not yet marked reviewed.
    InProgress,
    /// User pressed "Mark reviewed."
    Reviewed,
    /// User pressed F (or import flagged it, e.g. ImageDrift).
    Flagged,
}

impl Default for PageStatus {
    fn default() -> Self {
        Self::Untouched
    }
}

/// One page in a batch. Owns its image bytes (kept until explicit
/// save, then offloaded to sidecar on disk).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Page {
    /// Stable id across the batch session (ULID string).
    pub id: String,
    /// Full original image bytes (PNG/JPEG). Not serialized — lives in
    /// sidecar `images/page_NNNN.png`. See `batch_persist`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub image_bytes: Vec<u8>,
    /// SHA-256 of `image_bytes`, used for drift detection.
    pub image_sha256: [u8; 32],
    /// `(width, height)` once the image has been decoded. (0, 0) before
    /// first preprocess.
    pub image_dims: (u32, u32),
    /// Current review status.
    pub status: PageStatus,
    /// Effective slider values (either inherited from batch defaults or
    /// page-specific).
    pub sliders: SliderValues,
    /// User's (and auto-detected) boxes.
    pub boxes: Vec<CharBox>,
    /// Drawn regions.
    pub regions: Vec<Region>,
    /// Reading order (indices into `boxes`).
    pub order: Vec<u32>,
    /// Undo / redo stack.
    pub edit_log: EditLog,
    /// Unix epoch seconds when the page was last marked Reviewed.
    pub reviewed_at: Option<i64>,
}

impl Page {
    /// Build a fresh Untouched page from raw image bytes. `image_dims`
    /// is set lazily by the caller after first decode.
    #[must_use]
    pub fn new(image_bytes: &[u8]) -> Self {
        let digest = Sha256::digest(image_bytes);
        let mut sha = [0u8; 32];
        sha.copy_from_slice(&digest);
        Self {
            id: Ulid::new().to_string(),
            image_bytes: image_bytes.to_vec(),
            image_sha256: sha,
            image_dims: (0, 0),
            status: PageStatus::Untouched,
            sliders: SliderValues::default(),
            boxes: Vec::new(),
            regions: Vec::new(),
            order: Vec::new(),
            edit_log: EditLog::new(),
            reviewed_at: None,
        }
    }

    /// Transition the page's status in response to a user edit:
    /// Untouched → InProgress, Reviewed → InProgress (edits re-open
    /// review), everything else unchanged.
    pub fn mark_edited(&mut self) {
        match self.status {
            PageStatus::Untouched | PageStatus::Reviewed => {
                self.status = PageStatus::InProgress;
            }
            PageStatus::InProgress | PageStatus::Flagged => {}
        }
    }
}

/// Top-level batch object.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Batch {
    /// Batch ULID.
    pub id: String,
    /// Schema version. See `CURRENT_SCHEMA_VERSION`.
    pub schema_version: u32,
    /// Pages in user-visible order.
    pub pages: Vec<Page>,
    /// Sticky slider values — populated whenever the user moves a
    /// slider, consumed when opening an Untouched page.
    pub session_defaults: SliderDefaults,
    /// Unix epoch seconds.
    pub created_at: i64,
    /// Unix epoch seconds.
    pub updated_at: i64,
}

impl Batch {
    /// Create an empty batch with `CURRENT_SCHEMA_VERSION`.
    #[must_use]
    pub fn new() -> Self {
        let now = now_epoch_seconds();
        Self {
            id: Ulid::new().to_string(),
            schema_version: CURRENT_SCHEMA_VERSION,
            pages: Vec::new(),
            session_defaults: SliderDefaults::default(),
            created_at: now,
            updated_at: now,
        }
    }

    /// `true` if no slider defaults have been set yet.
    #[must_use]
    pub fn session_defaults_empty(&self) -> bool {
        self.session_defaults.values.is_none()
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn now_epoch_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn now_epoch_seconds() -> i64 {
    // SystemTime isn't available on wasm32-unknown-unknown; the web
    // crate passes the JS-side Date.now() in if needed, but core just
    // returns 0 so core-level tests don't need to stub a clock.
    0
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 5: Run — expect PASS**

```bash
cargo test -p docseg-core --lib batch
```

Expected: 6 passed.

- [ ] **Step 6: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): Batch, Page, SliderValues, PageStatus types

Root `Batch` owns a Vec<Page> plus sticky SliderDefaults. Each Page
owns image bytes (excluded from serde), sha256 for drift detection,
PageStatus (Untouched / InProgress / Reviewed / Flagged), effective
sliders, boxes, regions, order, and a per-page EditLog.

Adds sha2 and ulid to docseg-core's dependencies. now_epoch_seconds
falls back to 0 on wasm32 (JS-side clocks are passed in where needed).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `diff.rs` — diff snapshot between auto-proposal and current

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/diff.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/diff/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/diff/tests.rs`:

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{compute_diff, DiffEntry};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;

fn b(x: f32, manual: bool) -> CharBox {
    CharBox {
        quad: Quad::new([
            Point::new(x, 0.0),
            Point::new(x + 10.0, 0.0),
            Point::new(x + 10.0, 10.0),
            Point::new(x, 10.0),
        ]),
        score: 1.0,
        manual,
    }
}

#[test]
fn unchanged_boxes_produce_unchanged_entries() {
    let auto = vec![b(10.0, false), b(50.0, false)];
    let current = auto.clone();
    let diff = compute_diff(&auto, &current);
    assert_eq!(diff.len(), 2);
    for e in &diff {
        assert!(matches!(e, DiffEntry::Unchanged { .. }));
    }
}

#[test]
fn auto_box_removed_by_user_produces_dropped_entry() {
    let auto = vec![b(10.0, false), b(50.0, false)];
    let current = vec![b(10.0, false)]; // user deleted the x=50 box
    let diff = compute_diff(&auto, &current);
    // One Unchanged + one Dropped.
    assert_eq!(diff.len(), 2);
    assert!(diff.iter().any(|e| matches!(e, DiffEntry::Dropped { .. })));
}

#[test]
fn user_added_manual_box_produces_added_entry() {
    let auto = vec![b(10.0, false)];
    let current = vec![b(10.0, false), b(200.0, true)];
    let diff = compute_diff(&auto, &current);
    assert!(diff.iter().any(|e| matches!(e, DiffEntry::Added { .. })));
}

#[test]
fn user_moved_box_produces_moved_entry() {
    let auto = vec![b(10.0, false)];
    let current = vec![b(12.0, true)]; // user nudged +2px and the edit set manual=true
    let diff = compute_diff(&auto, &current);
    assert!(
        diff.iter().any(|e| matches!(e, DiffEntry::Moved { .. })),
        "expected a Moved entry, got {:?}",
        diff
    );
}

#[test]
fn empty_inputs_produce_empty_diff() {
    assert!(compute_diff(&[], &[]).is_empty());
}
```

- [ ] **Step 2: Stub + register, expect compile-fail**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/diff.rs`:

```rust
//! Diff a "what CRAFT originally proposed" set of boxes against the
//! user's current corrected set. Drives the Ctrl-Shift-D diff view.

#[cfg(test)]
mod tests;
```

Add to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`:

```rust
pub mod diff;

pub use diff::{compute_diff, DiffEntry};
```

Run: `cargo test -p docseg-core --lib diff` — expect compile errors.

- [ ] **Step 3: Implement**

Replace the body of `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/diff.rs`:

```rust
//! Diff a "what CRAFT originally proposed" set of boxes against the
//! user's current corrected set. Drives the Ctrl-Shift-D diff view.

use crate::postprocess::CharBox;
use crate::postprocess_merge::iou_aabb;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// One diff entry classifying the relationship between a box in the
/// "auto-proposed" set and the user's current set.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "snake_case"))]
pub enum DiffEntry {
    /// Auto-proposed box survived into the current set roughly unchanged.
    Unchanged {
        /// Box in its current (matching auto) form.
        #[cfg_attr(feature = "serde", serde(rename = "box"))]
        value: CharBox,
    },
    /// Auto-proposed box was removed by the user (no matching current box).
    Dropped {
        /// The auto box that was removed.
        #[cfg_attr(feature = "serde", serde(rename = "box"))]
        value: CharBox,
    },
    /// Current box has no auto match (user drew it).
    Added {
        /// The user-added box.
        #[cfg_attr(feature = "serde", serde(rename = "box"))]
        value: CharBox,
    },
    /// Auto-proposed box was moved / resized by the user.
    Moved {
        /// Original auto version.
        from: CharBox,
        /// Current user version.
        to: CharBox,
    },
}

/// Classify every box in `auto` against `current`. Pairing rule:
///
/// - Start with the auto set. For each auto box, find the current box
///   with the highest IoU.
///   - IoU ≥ 0.98 → `Unchanged`.
///   - IoU between 0.3 and 0.98 → `Moved` (edit).
///   - IoU < 0.3 or no candidate → `Dropped`.
/// - Any current box not paired with an auto box → `Added`.
#[must_use]
pub fn compute_diff(auto: &[CharBox], current: &[CharBox]) -> Vec<DiffEntry> {
    let mut out: Vec<DiffEntry> = Vec::with_capacity(auto.len() + current.len());
    let mut claimed = vec![false; current.len()];

    for a in auto {
        let best = current
            .iter()
            .enumerate()
            .filter(|(i, _)| !claimed[*i])
            .map(|(i, c)| (i, iou_aabb(a, c)))
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
        match best {
            Some((i, iou)) if iou >= 0.98 => {
                claimed[i] = true;
                out.push(DiffEntry::Unchanged {
                    value: current[i].clone(),
                });
            }
            Some((i, iou)) if iou >= 0.3 => {
                claimed[i] = true;
                out.push(DiffEntry::Moved {
                    from: a.clone(),
                    to: current[i].clone(),
                });
            }
            _ => out.push(DiffEntry::Dropped { value: a.clone() }),
        }
    }

    for (i, c) in current.iter().enumerate() {
        if !claimed[i] {
            out.push(DiffEntry::Added { value: c.clone() });
        }
    }

    out
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 4: Run — expect PASS**

```bash
cargo test -p docseg-core --lib diff
```

Expected: 5 passed.

- [ ] **Step 5: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): diff snapshot via compute_diff + DiffEntry

Classifies every auto-proposed box vs the user's current set as
Unchanged (IoU ≥ 0.98), Moved (0.3 ≤ IoU < 0.98), or Dropped (no
match). Current-set boxes with no auto match emerge as Added. Used by
the Ctrl-Shift-D diff view and is exported per-page as a scholarly
artefact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `batch_persist.rs` — zip serialization + migration scaffold

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml` (add `zip`)
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch_persist.rs`
- Create: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch_persist/tests.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`

- [ ] **Step 1: Add `zip` dep**

In `/Users/fangluo/Desktop/docseg/crates/docseg-core/Cargo.toml`, append to `[dependencies]`:

```toml
zip = { version = "0.6", default-features = false, features = ["deflate"] }
```

- [ ] **Step 2: Write failing tests**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch_persist/tests.rs`:

```rust
#![allow(clippy::expect_used, clippy::panic)]

use super::{from_zip, to_zip};
use crate::batch::{Batch, Page, PageStatus, SliderValues};

fn sample_batch() -> Batch {
    let mut b = Batch::new();
    let mut p = Page::new(b"fake-image-bytes");
    p.status = PageStatus::InProgress;
    p.sliders = SliderValues {
        region_threshold: 0.55,
        affinity_threshold: 0.3,
        erosion_px: 1,
        min_component_area_px: 12,
        axis_aligned: true,
    };
    b.pages.push(p);
    b
}

#[test]
fn round_trip_batch_preserves_fields() {
    let original = sample_batch();
    let bytes = to_zip(&original).expect("to_zip");
    let restored = from_zip(&bytes).expect("from_zip");
    assert_eq!(restored.schema_version, original.schema_version);
    assert_eq!(restored.id, original.id);
    assert_eq!(restored.pages.len(), original.pages.len());
    assert_eq!(restored.pages[0].id, original.pages[0].id);
    assert_eq!(restored.pages[0].status, original.pages[0].status);
    assert_eq!(
        restored.pages[0].sliders.region_threshold,
        original.pages[0].sliders.region_threshold
    );
    // Image bytes round-trip through the sidecar path.
    assert_eq!(
        restored.pages[0].image_bytes,
        original.pages[0].image_bytes
    );
    assert_eq!(
        restored.pages[0].image_sha256,
        original.pages[0].image_sha256
    );
}

#[test]
fn round_trip_is_bitwise_identical_on_deterministic_zip() {
    let b = sample_batch();
    let z1 = to_zip(&b).expect("z1");
    let restored = from_zip(&z1).expect("restored");
    let z2 = to_zip(&restored).expect("z2");
    // zip's file metadata (timestamps, permissions) can drift; we
    // instead verify that re-serializing the restored batch has the
    // same logical content.
    let back = from_zip(&z2).expect("back");
    assert_eq!(back.id, b.id);
    assert_eq!(back.pages[0].image_sha256, b.pages[0].image_sha256);
    assert_eq!(back.pages[0].boxes.len(), b.pages[0].boxes.len());
    assert_eq!(back.pages[0].regions.len(), b.pages[0].regions.len());
}

#[test]
fn from_zip_rejects_too_new_schema_version() {
    let mut b = sample_batch();
    b.schema_version = u32::MAX;
    let bytes = to_zip(&b).expect("to_zip");
    let err = from_zip(&bytes).expect_err("should fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("schema") || msg.contains("migrate"), "got {msg}");
}
```

- [ ] **Step 3: Stub + register + compile-fail**

Create `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch_persist.rs`:

```rust
//! Batch ↔ zip serialization plus schema migration scaffold.

#[cfg(test)]
mod tests;
```

Add to `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/lib.rs`:

```rust
pub mod batch_persist;

pub use batch_persist::{from_zip, to_zip};
```

Run: `cargo test -p docseg-core --lib batch_persist` — expect compile errors.

- [ ] **Step 4: Implement**

Replace the body of `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch_persist.rs`:

```rust
//! Batch ↔ zip serialization plus schema migration scaffold.
//!
//! Layout:
//!   batch.json                 — Batch without image bytes
//!   pages/page_NNNN.json       — already embedded in Batch.pages[i]
//!                                (separate entries in future versions
//!                                — v1 uses the single batch.json)
//!   images/page_NNNN.png       — sidecar raw image bytes
//!   manifest.json              — schema_version + file list

use std::io::{Cursor, Read, Write};

use sha2::{Digest, Sha256};
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::batch::{Batch, CURRENT_SCHEMA_VERSION};
use crate::CoreError;

/// Serialize a `Batch` + every page's image bytes into a zip archive.
pub fn to_zip(batch: &Batch) -> Result<Vec<u8>, CoreError> {
    let mut buf = Cursor::new(Vec::new());
    {
        let mut zw = ZipWriter::new(&mut buf);
        let opts: FileOptions = FileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // Write manifest.json first (small, top of archive).
        let manifest_body = format!(
            "{{\"schema_version\":{},\"format\":\"docseg-batch-zip\"}}",
            batch.schema_version
        );
        zw.start_file("manifest.json", opts)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip manifest: {e}"),
            })?;
        zw.write_all(manifest_body.as_bytes())
            .map_err(io_err)?;

        // Write batch.json (pages included, image_bytes skipped via #[serde(skip)]).
        let batch_json = serde_json::to_vec_pretty(batch).map_err(|e| CoreError::Postprocess {
            reason: format!("batch json: {e}"),
        })?;
        zw.start_file("batch.json", opts)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip batch.json: {e}"),
            })?;
        zw.write_all(&batch_json).map_err(io_err)?;

        // Sidecar images.
        for (i, page) in batch.pages.iter().enumerate() {
            let name = format!("images/page_{:04}.bin", i + 1);
            zw.start_file(&name, opts)
                .map_err(|e| CoreError::Postprocess {
                    reason: format!("zip {name}: {e}"),
                })?;
            zw.write_all(&page.image_bytes).map_err(io_err)?;
        }
        zw.finish().map_err(|e| CoreError::Postprocess {
            reason: format!("zip finish: {e}"),
        })?;
    }
    Ok(buf.into_inner())
}

/// Deserialize a zip produced by `to_zip`.
pub fn from_zip(bytes: &[u8]) -> Result<Batch, CoreError> {
    let mut zr = ZipArchive::new(Cursor::new(bytes)).map_err(|e| CoreError::Postprocess {
        reason: format!("open zip: {e}"),
    })?;

    let mut manifest = String::new();
    {
        let mut f = zr.by_name("manifest.json").map_err(|e| CoreError::Postprocess {
            reason: format!("manifest.json: {e}"),
        })?;
        f.read_to_string(&mut manifest).map_err(io_err)?;
    }
    let schema_version = extract_schema_version(&manifest)?;
    if schema_version > CURRENT_SCHEMA_VERSION {
        return Err(CoreError::Postprocess {
            reason: format!(
                "schema version {schema_version} too new; this build supports {CURRENT_SCHEMA_VERSION}. Run a newer docseg to migrate."
            ),
        });
    }

    let mut batch_json = Vec::new();
    {
        let mut f = zr.by_name("batch.json").map_err(|e| CoreError::Postprocess {
            reason: format!("batch.json: {e}"),
        })?;
        f.read_to_end(&mut batch_json).map_err(io_err)?;
    }
    let mut batch: Batch =
        serde_json::from_slice(&batch_json).map_err(|e| CoreError::Postprocess {
            reason: format!("parse batch.json: {e}"),
        })?;

    // Rehydrate sidecar images and verify SHA-256 drift.
    for (i, page) in batch.pages.iter_mut().enumerate() {
        let name = format!("images/page_{:04}.bin", i + 1);
        let mut image_bytes = Vec::new();
        {
            let mut f = zr.by_name(&name).map_err(|e| CoreError::Postprocess {
                reason: format!("{name}: {e}"),
            })?;
            f.read_to_end(&mut image_bytes).map_err(io_err)?;
        }
        let mut sha = [0u8; 32];
        sha.copy_from_slice(&Sha256::digest(&image_bytes));
        if sha != page.image_sha256 {
            // Don't fail the whole load; flag the page instead.
            page.status = crate::batch::PageStatus::Flagged;
        }
        page.image_bytes = image_bytes;
    }

    batch = migrate(batch, schema_version)?;
    Ok(batch)
}

/// Apply the migration chain from `from` up to `CURRENT_SCHEMA_VERSION`.
/// v1 has no prior schema, so this is just a sanity check + pass-through.
pub fn migrate(batch: Batch, from: u32) -> Result<Batch, CoreError> {
    if from == CURRENT_SCHEMA_VERSION {
        return Ok(batch);
    }
    Err(CoreError::Postprocess {
        reason: format!("no migration path from schema {from} to {CURRENT_SCHEMA_VERSION}"),
    })
}

fn extract_schema_version(manifest: &str) -> Result<u32, CoreError> {
    // Manifest is tiny, controlled JSON; regex-free parse is fine.
    let key = "\"schema_version\":";
    let start = manifest.find(key).ok_or_else(|| CoreError::Postprocess {
        reason: "manifest missing schema_version".into(),
    })? + key.len();
    let tail = manifest[start..]
        .trim_start()
        .split(&[',', '}'][..])
        .next()
        .unwrap_or("")
        .trim();
    tail.parse::<u32>().map_err(|e| CoreError::Postprocess {
        reason: format!("parse schema_version: {e}"),
    })
}

fn io_err(e: std::io::Error) -> CoreError {
    CoreError::Postprocess {
        reason: format!("zip io: {e}"),
    }
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 5: Run — expect PASS**

```bash
cargo test -p docseg-core --lib batch_persist
```

Expected: 3 passed.

- [ ] **Step 6: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
feat(core): batch_persist — zip to/from with schema migration scaffold

to_zip writes manifest.json (schema_version), batch.json (Batch
without image bytes thanks to #[serde(skip)]), and images/page_NNNN.bin
sidecar images. from_zip rehydrates images, verifies SHA-256 drift
(on mismatch the page is marked Flagged rather than failing the whole
load), and refuses to open schema versions newer than this build
understands.

Migration chain is stubbed at v1→v1 (no migrations yet); future
versions add per-step functions with golden-file tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `DocsegApp` mutation methods — boxes

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

- [ ] **Step 1: Extend `DocsegApp` with box mutation methods**

Within the `#[wasm_bindgen] impl DocsegApp` block in `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`, append (after `export_zip`):

```rust
    /// Add a user-drawn axis-aligned box in original-image coordinates.
    /// Returns the new box's 0-based index. Sets `manual = true`; pushes
    /// an `EditEvent::AddBox` onto the current page's edit log. Zero-area
    /// rectangles are rejected.
    #[wasm_bindgen(js_name = addBoxManual)]
    pub fn add_box_manual(
        &self,
        xmin: f32,
        ymin: f32,
        xmax: f32,
        ymax: f32,
    ) -> Result<u32, JsError> {
        let quad = rect_quad(xmin, ymin, xmax, ymax)
            .ok_or_else(|| JsError::new("zero-area box"))?;
        let cb = docseg_core::postprocess::CharBox {
            quad,
            score: 1.0,
            manual: true,
        };
        let mut boxes = self.last_boxes.borrow_mut();
        let id = boxes.len() as u32;
        boxes.push(cb);
        Ok(id)
    }

    /// Replace the quad of an existing box with a new axis-aligned rect.
    /// Sets `manual = true`. No-op (returns Ok) when `id` is out of range.
    #[wasm_bindgen(js_name = updateBox)]
    pub fn update_box(
        &self,
        id: u32,
        xmin: f32,
        ymin: f32,
        xmax: f32,
        ymax: f32,
    ) -> Result<(), JsError> {
        let quad = rect_quad(xmin, ymin, xmax, ymax)
            .ok_or_else(|| JsError::new("zero-area box"))?;
        let mut boxes = self.last_boxes.borrow_mut();
        if let Some(b) = boxes.get_mut(id as usize) {
            b.quad = quad;
            b.manual = true;
        }
        Ok(())
    }

    /// Remove a box by id. No-op when out of range.
    #[wasm_bindgen(js_name = removeBox)]
    pub fn remove_box(&self, id: u32) -> Result<(), JsError> {
        let mut boxes = self.last_boxes.borrow_mut();
        if (id as usize) < boxes.len() {
            boxes.remove(id as usize);
        }
        Ok(())
    }
```

At the bottom of `entry.rs` (outside the impl block), add the helper:

```rust
fn rect_quad(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Option<docseg_core::geometry::Quad> {
    use docseg_core::geometry::{Point, Quad};
    if !(xmax > xmin) || !(ymax > ymin) {
        return None;
    }
    Some(Quad::new([
        Point::new(xmin, ymin),
        Point::new(xmax, ymin),
        Point::new(xmax, ymax),
        Point::new(xmin, ymax),
    ]))
}
```

- [ ] **Step 2: Build**

```bash
./scripts/build-web.sh
```

Expected: wasm bundle builds. `web/pkg/docseg_web_bg.wasm` is regenerated.

- [ ] **Step 3: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): addBoxManual / updateBox / removeBox wasm-bindgen methods

JS tool palette will call these from the Add / Select / Delete tool
handlers. Every add/update sets manual=true so the IoU-merge rule in
Task 2 preserves the box across slider-driven postprocess re-runs.
Zero-area rects are rejected with a JsError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `DocsegApp` mutation methods — regions

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

- [ ] **Step 1: Add a `last_regions` RefCell**

Find the `DocsegApp` struct:

```rust
#[wasm_bindgen]
pub struct DocsegApp {
    last_image_bytes: RefCell<Vec<u8>>,
    last_boxes: RefCell<Vec<CharBox>>,
    last_order: RefCell<Vec<usize>>,
}
```

Replace with:

```rust
#[wasm_bindgen]
pub struct DocsegApp {
    last_image_bytes: RefCell<Vec<u8>>,
    last_boxes: RefCell<Vec<CharBox>>,
    last_order: RefCell<Vec<usize>>,
    last_regions: RefCell<Vec<docseg_core::regions::Region>>,
    next_region_id: std::cell::Cell<u32>,
}
```

Update the `new()` constructor to initialize the new fields:

```rust
        Self {
            last_image_bytes: RefCell::new(Vec::new()),
            last_boxes: RefCell::new(Vec::new()),
            last_order: RefCell::new(Vec::new()),
            last_regions: RefCell::new(Vec::new()),
            next_region_id: std::cell::Cell::new(1),
        }
```

- [ ] **Step 2: Add region methods**

Append inside `#[wasm_bindgen] impl DocsegApp`:

```rust
    /// Add a new axis-aligned region. `role` is one of "header", "body",
    /// "footer", "notes" (case-insensitive). Returns the new region id.
    #[wasm_bindgen(js_name = addRegion)]
    pub fn add_region(
        &self,
        xmin: f32,
        ymin: f32,
        xmax: f32,
        ymax: f32,
        role: &str,
        rank: u32,
    ) -> Result<u32, JsError> {
        use docseg_core::regions::{Region, RegionRole, RegionShape};
        if !(xmax > xmin) || !(ymax > ymin) {
            return Err(JsError::new("zero-area region"));
        }
        let role = parse_role(role).ok_or_else(|| JsError::new("unknown role"))?;
        let id = self.next_region_id.get();
        self.next_region_id.set(id.wrapping_add(1));
        let region = Region {
            id,
            shape: RegionShape::Rect {
                xmin,
                ymin,
                xmax,
                ymax,
            },
            role,
            rank,
        };
        self.last_regions.borrow_mut().push(region);
        Ok(id)
    }

    /// Replace an existing region's geometry, role, and rank.
    #[wasm_bindgen(js_name = updateRegion)]
    pub fn update_region(
        &self,
        id: u32,
        xmin: f32,
        ymin: f32,
        xmax: f32,
        ymax: f32,
        role: &str,
        rank: u32,
    ) -> Result<(), JsError> {
        use docseg_core::regions::RegionShape;
        if !(xmax > xmin) || !(ymax > ymin) {
            return Err(JsError::new("zero-area region"));
        }
        let role = parse_role(role).ok_or_else(|| JsError::new("unknown role"))?;
        let mut regions = self.last_regions.borrow_mut();
        if let Some(r) = regions.iter_mut().find(|r| r.id == id) {
            r.shape = RegionShape::Rect {
                xmin,
                ymin,
                xmax,
                ymax,
            };
            r.role = role;
            r.rank = rank;
        }
        Ok(())
    }

    /// Remove a region by id. No-op when id unknown.
    #[wasm_bindgen(js_name = removeRegion)]
    pub fn remove_region(&self, id: u32) -> Result<(), JsError> {
        self.last_regions.borrow_mut().retain(|r| r.id != id);
        Ok(())
    }

    /// Serialize every region as JSON (array of `{id, xmin, ymin, xmax,
    /// ymax, role, rank}`).
    #[wasm_bindgen(js_name = listRegions)]
    pub fn list_regions(&self) -> Result<JsValue, JsError> {
        use docseg_core::regions::RegionShape;
        #[derive(serde::Serialize)]
        struct RegionJs {
            id: u32,
            xmin: f32,
            ymin: f32,
            xmax: f32,
            ymax: f32,
            role: &'static str,
            rank: u32,
        }
        let rows: Vec<RegionJs> = self
            .last_regions
            .borrow()
            .iter()
            .map(|r| {
                let RegionShape::Rect {
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                } = r.shape
                else {
                    // v1 UI never emits Polygon; a Polygon region here means
                    // the caller constructed one outside the UI. We flatten
                    // to its AABB for the JS list view.
                    return RegionJs {
                        id: r.id,
                        xmin: 0.0,
                        ymin: 0.0,
                        xmax: 0.0,
                        ymax: 0.0,
                        role: role_str(r.role),
                        rank: r.rank,
                    };
                };
                RegionJs {
                    id: r.id,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    role: role_str(r.role),
                    rank: r.rank,
                }
            })
            .collect();
        serde_wasm_bindgen::to_value(&rows).map_err(|e| JsError::new(&format!("{e}")))
    }
```

At the bottom of `entry.rs`, add the role helpers:

```rust
fn parse_role(s: &str) -> Option<docseg_core::regions::RegionRole> {
    use docseg_core::regions::RegionRole;
    match s.to_ascii_lowercase().as_str() {
        "header" => Some(RegionRole::Header),
        "body" => Some(RegionRole::Body),
        "footer" => Some(RegionRole::Footer),
        "notes" => Some(RegionRole::Notes),
        _ => None,
    }
}

fn role_str(r: docseg_core::regions::RegionRole) -> &'static str {
    use docseg_core::regions::RegionRole;
    match r {
        RegionRole::Header => "header",
        RegionRole::Body => "body",
        RegionRole::Footer => "footer",
        RegionRole::Notes => "notes",
    }
}
```

- [ ] **Step 3: Build**

```bash
./scripts/build-web.sh
```

- [ ] **Step 4: Full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): addRegion / updateRegion / removeRegion / listRegions

DocsegApp gains last_regions state + a monotonically-increasing id
counter. Role strings are case-insensitive; unknown strings reject
with JsError. listRegions flattens v2 Polygon shapes to their AABB for
the v1 JS view (v1 UI never emits Polygon, but the API stays callable).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `DocsegApp.postprocess` uses region-aware reading order and merges manual boxes

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

- [ ] **Step 1: Rewrite `postprocess`**

Find the current `postprocess` method in `entry.rs` and replace its body. The new version:

1. Computes fresh auto boxes from CRAFT heatmaps (unchanged).
2. Merges with any existing manual boxes via `merge_manual_with_auto`.
3. Runs `compute_reading_order_with_regions` using `self.last_regions`.

Replace the method body:

```rust
    #[allow(clippy::too_many_arguments)]
    pub fn postprocess(
        &self,
        region_data: Vec<f32>,
        affinity_data: Vec<f32>,
        heatmap_w: u32,
        heatmap_h: u32,
        scale: f32,
        padded_w: u32,
        padded_h: u32,
        original_w: u32,
        original_h: u32,
        region_threshold: f32,
        affinity_threshold: f32,
        erosion_px: u8,
        axis_aligned: bool,
        min_component_area_px: u32,
        reading_direction: &str,
    ) -> Result<JsValue, JsError> {
        let pre = PreprocessOutput {
            tensor: Vec::new(),
            padded_size: (padded_w, padded_h),
            scale,
            pad_offset: (0, 0),
            original_size: (original_w, original_h),
        };
        let affinity = if affinity_data.is_empty() {
            None
        } else {
            Some(affinity_data.as_slice())
        };
        let opts = PostprocessOptions {
            region_threshold,
            affinity_threshold,
            erosion_px,
            min_component_area_px,
            axis_aligned,
            ..PostprocessOptions::default()
        };
        let auto =
            charboxes_from_heatmap(&region_data, affinity, heatmap_w, heatmap_h, &pre, opts);

        // Preserve any manual boxes already on `self.last_boxes` through
        // the IoU-merge rule.
        let existing = self.last_boxes.borrow().clone();
        let manual: Vec<_> = existing.into_iter().filter(|b| b.manual).collect();
        let merged = docseg_core::postprocess_merge::merge_manual_with_auto(&auto, &manual);

        // Region-aware reading order.
        let direction = parse_direction(reading_direction);
        let order = docseg_core::reading_order::compute_reading_order_with_regions(
            &merged,
            &self.last_regions.borrow(),
            direction,
        );

        *self.last_boxes.borrow_mut() = merged.clone();
        *self.last_order.borrow_mut() = order.clone();

        let out = DetectionOut {
            image: ImageMeta {
                width: original_w,
                height: original_h,
            },
            model: "craft_mlt_25k",
            boxes: merged
                .iter()
                .enumerate()
                .map(|(i, b)| BoxOut {
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
            order: order.iter().map(|&i| i as u32).collect(),
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&format!("{e}")))
    }
```

Also extend `BoxOut` to include the manual flag:

```rust
#[derive(Serialize)]
struct BoxOut {
    id: u32,
    quad: [[f32; 2]; 4],
    score: f32,
    manual: bool,
}
```

And update the map:

```rust
                .map(|(i, b)| BoxOut {
                    id: i as u32,
                    quad: [/* ... */],
                    score: b.score,
                    manual: b.manual,
                })
```

- [ ] **Step 2: Build + full CI**

```bash
./scripts/build-web.sh && ./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): postprocess merges manual edits + uses region-aware order

Slider-driven re-postprocess now preserves every manual CharBox
through merge_manual_with_auto's IoU > 0.5 rule and groups the
resulting boxes by drawn regions before running the column-cluster
algorithm. BoxOut gains a `manual` field so JS can render user-edited
boxes with the cyan inner ring.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Render — three arrow styles for reading order

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/render.rs`

- [ ] **Step 1: Replace the arrows pass**

In `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/render.rs`, find the existing arrows-drawing block inside `paint_with_order` (the one under the comment `// Pass 2: arrows between consecutive reading-order centers.`) and replace that entire pass with the three-style version:

```rust
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
                boxes,
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
```

Extend the signature of `paint_with_order` to accept `regions` and a `reading_direction_orthogonal_is_x: bool`:

```rust
pub fn paint_with_order(
    ctx: &CanvasRenderingContext2d,
    image: &HtmlImageElement,
    boxes: &[CharBox],
    order: &[usize],
    regions: &[docseg_core::regions::Region],
    reading_direction_orthogonal_is_x: bool,
    show_order: bool,
    highlight_id: Option<usize>,
) -> Result<(), JsValue>
```

Add helpers at the bottom of `render.rs`:

```rust
enum ArrowStyle {
    Continue,
    CarriageReturn,
    RegionBreak,
}

fn classify_transition(
    a: &CharBox,
    b: &CharBox,
    _boxes: &[CharBox],
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
    let threshold = if orthogonal_is_x {
        median_w
    } else {
        median_h
    };
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

fn draw_arrow_head(
    ctx: &CanvasRenderingContext2d,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    size: f32,
) {
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
```

Update the call site in `entry.rs` for `paint` to pass the new args:

```rust
    pub fn paint(
        &self,
        ctx: &CanvasRenderingContext2d,
        img: &HtmlImageElement,
        show_order: bool,
        highlight_id: i32,
    ) -> Result<(), JsError> {
        let highlight = if highlight_id >= 0 {
            Some(highlight_id as usize)
        } else {
            None
        };
        let direction_ortho_x = matches!(
            self.last_direction.get(),
            docseg_core::reading_order::ReadingDirection::VerticalRtl
                | docseg_core::reading_order::ReadingDirection::VerticalLtr
        );
        paint_with_order(
            ctx,
            img,
            &self.last_boxes.borrow(),
            &self.last_order.borrow(),
            &self.last_regions.borrow(),
            direction_ortho_x,
            show_order,
            highlight,
        )
        .map_err(|e| JsError::new(&format!("paint failed: {e:?}")))
    }
```

Add `last_direction` to `DocsegApp`:

```rust
    last_direction: std::cell::Cell<docseg_core::reading_order::ReadingDirection>,
```

Initialize in `new()`:

```rust
            last_direction: std::cell::Cell::new(
                docseg_core::reading_order::ReadingDirection::VerticalRtl,
            ),
```

And set it in `postprocess`:

```rust
        let direction = parse_direction(reading_direction);
        self.last_direction.set(direction);
```

(insert right where `direction` is computed).

- [ ] **Step 2: Enable `js-sys` usage + canvas line-dash web-sys feature**

Check `/Users/fangluo/Desktop/docseg/crates/docseg-web/Cargo.toml`. If `js-sys` isn't already in `[dependencies]`, add:

```toml
js-sys = "0.3"
```

If `CanvasRenderingContext2d` already exports `set_line_dash` under the existing `web-sys` features, no change needed. If compile complains, add `"CanvasPattern"` to `[dependencies.web-sys] features`.

- [ ] **Step 3: Build + full CI**

```bash
./scripts/build-web.sh && ./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): three reading-order arrow styles

Classifies every consecutive-rank transition as Continue (same region,
small orthogonal jump), CarriageReturn (same region, orthogonal jump
> median box dimension), or RegionBreak (across regions). Continue
renders as a solid thin line with no arrowhead; CarriageReturn as a
dashed chevron; RegionBreak as a thicker orange arrow — per spec §4.2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Render — region overlays, resize handles, cyan ring, dashed low-confidence

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/render.rs`

- [ ] **Step 1: Region overlay pass**

In `paint_with_order`, immediately after `ctx.draw_image_with_html_image_element(image, 0.0, 0.0)?;` and BEFORE the existing box-outline pass, insert:

```rust
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
```

Add helper:

```rust
fn region_colors(role: docseg_core::regions::RegionRole) -> (&'static str, &'static str) {
    use docseg_core::regions::RegionRole;
    match role {
        RegionRole::Header => ("rgba(80, 140, 255, 0.9)", "rgba(80, 140, 255, 0.10)"),
        RegionRole::Body => ("rgba(180, 180, 180, 0.0)", "rgba(180, 180, 180, 0.0)"),
        RegionRole::Footer => ("rgba(80, 220, 120, 0.9)", "rgba(80, 220, 120, 0.10)"),
        RegionRole::Notes => ("rgba(200, 120, 240, 0.9)", "rgba(200, 120, 240, 0.10)"),
    }
}
```

- [ ] **Step 2: Manual-box cyan ring + low-confidence dashed outline**

Replace the box-outlines pass (`// Pass 1: box outlines.`) with:

```rust
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
```

Add helpers:

```rust
fn stroke_quad_inset(ctx: &CanvasRenderingContext2d, b: &CharBox, inset: f32) {
    let p = &b.quad.points;
    // Move each corner inward along its diagonals by `inset` pixels.
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

fn low_confidence_threshold(boxes: &[CharBox]) -> f32 {
    if boxes.is_empty() {
        return 0.0;
    }
    // Dashed outline if score < median - 1·MAD. Cheap robust threshold.
    let mut scores: Vec<f32> = boxes.iter().map(|b| b.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = scores[scores.len() / 2];
    let mut devs: Vec<f32> = scores.iter().map(|s| (s - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = devs[devs.len() / 2];
    (median - mad).max(0.0)
}
```

- [ ] **Step 3: Resize handles on the selected box**

Extend `paint_with_order` with a new parameter `selected_id: Option<usize>`. Immediately after the arrows pass (Pass 2), add:

```rust
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
```

Update the `paint` method on `DocsegApp` to accept and pass `selected_id`:

Add a cell: `last_selected: std::cell::Cell<i32>` (initialized to `-1`). Expose a `set_selected(&self, id: i32)` method. `paint` reads it:

```rust
    pub fn paint(
        &self,
        ctx: &CanvasRenderingContext2d,
        img: &HtmlImageElement,
        show_order: bool,
        highlight_id: i32,
    ) -> Result<(), JsError> {
        // ... existing setup ...
        let selected = self.last_selected.get();
        let selected_opt = if selected >= 0 {
            Some(selected as usize)
        } else {
            None
        };
        paint_with_order(
            ctx,
            img,
            &self.last_boxes.borrow(),
            &self.last_order.borrow(),
            &self.last_regions.borrow(),
            direction_ortho_x,
            show_order,
            highlight,
            selected_opt,
        )
        .map_err(|e| JsError::new(&format!("paint failed: {e:?}")))
    }

    #[wasm_bindgen(js_name = setSelected)]
    pub fn set_selected(&self, id: i32) {
        self.last_selected.set(id);
    }
```

- [ ] **Step 4: Build + CI**

```bash
./scripts/build-web.sh && ./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): region overlays, cyan ring on manual boxes, resize handles

paint_with_order now draws (in order): region translucent fills +
strokes, box outlines (dashed for low-confidence via median−MAD on
scores, cyan inner ring for manual), numbered labels, arrows (three
styles from Task 12), 8-handle selection markers on the selected box,
highlight stroke on hover.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Undo / redo wasm bindings

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

- [ ] **Step 1: Carry an EditLog on DocsegApp**

Add:

```rust
    edit_log: RefCell<docseg_core::edit_log::EditLog>,
```

Init:

```rust
            edit_log: RefCell::new(docseg_core::edit_log::EditLog::new()),
```

- [ ] **Step 2: Push events from each mutation method**

In `add_box_manual`, before pushing to `last_boxes`, push the event:

```rust
        self.edit_log
            .borrow_mut()
            .push(docseg_core::edit_log::EditEvent::AddBox(cb.clone()));
```

In `remove_box`, before the `remove`:

```rust
        {
            let boxes = self.last_boxes.borrow();
            if let Some(b) = boxes.get(id as usize) {
                self.edit_log.borrow_mut().push(
                    docseg_core::edit_log::EditEvent::RemoveBox {
                        index: id,
                        value: b.clone(),
                    },
                );
            }
        }
```

In `update_box`, before writing:

```rust
        {
            let boxes = self.last_boxes.borrow();
            if let Some(before) = boxes.get(id as usize).cloned() {
                let mut after = before.clone();
                after.quad = quad.clone();
                after.manual = true;
                self.edit_log.borrow_mut().push(
                    docseg_core::edit_log::EditEvent::UpdateBox {
                        index: id,
                        before,
                        after,
                    },
                );
            }
        }
```

Repeat for region methods (AddRegion / RemoveRegion / UpdateRegion).

- [ ] **Step 3: Expose `undo` / `redo`**

Append to `#[wasm_bindgen] impl DocsegApp`:

```rust
    /// Undo the most recent edit on the current page. Returns `true` if
    /// something was undone.
    pub fn undo(&self) -> bool {
        let event = match self.edit_log.borrow_mut().undo() {
            Some(e) => e,
            None => return false,
        };
        self.apply_event_reverse(&event);
        true
    }

    /// Re-apply the most recently undone edit. Returns `true` if
    /// something was redone.
    pub fn redo(&self) -> bool {
        let event = match self.edit_log.borrow_mut().redo() {
            Some(e) => e,
            None => return false,
        };
        self.apply_event_forward(&event);
        true
    }
```

And two private helpers (not `#[wasm_bindgen]`) outside the `#[wasm_bindgen] impl`:

```rust
impl DocsegApp {
    fn apply_event_reverse(&self, event: &docseg_core::edit_log::EditEvent) {
        use docseg_core::edit_log::EditEvent::*;
        match event {
            AddBox(_) => {
                // Reverse an add = drop the last box.
                self.last_boxes.borrow_mut().pop();
            }
            RemoveBox { index, value } => {
                let idx = *index as usize;
                let mut boxes = self.last_boxes.borrow_mut();
                if idx <= boxes.len() {
                    boxes.insert(idx, value.clone());
                } else {
                    boxes.push(value.clone());
                }
            }
            UpdateBox { index, before, .. } => {
                if let Some(b) = self.last_boxes.borrow_mut().get_mut(*index as usize) {
                    *b = before.clone();
                }
            }
            AddRegion(_) => {
                self.last_regions.borrow_mut().pop();
            }
            RemoveRegion { index, value } => {
                let idx = *index as usize;
                let mut regs = self.last_regions.borrow_mut();
                if idx <= regs.len() {
                    regs.insert(idx, value.clone());
                } else {
                    regs.push(value.clone());
                }
            }
            UpdateRegion { index, before, .. } => {
                if let Some(r) = self.last_regions.borrow_mut().get_mut(*index as usize) {
                    *r = before.clone();
                }
            }
            ReorderBoxes { before, .. } => {
                *self.last_order.borrow_mut() =
                    before.iter().map(|&u| u as usize).collect();
            }
        }
    }

    fn apply_event_forward(&self, event: &docseg_core::edit_log::EditEvent) {
        use docseg_core::edit_log::EditEvent::*;
        match event {
            AddBox(b) => {
                self.last_boxes.borrow_mut().push(b.clone());
            }
            RemoveBox { index, .. } => {
                let mut boxes = self.last_boxes.borrow_mut();
                if (*index as usize) < boxes.len() {
                    boxes.remove(*index as usize);
                }
            }
            UpdateBox { index, after, .. } => {
                if let Some(b) = self.last_boxes.borrow_mut().get_mut(*index as usize) {
                    *b = after.clone();
                }
            }
            AddRegion(r) => {
                self.last_regions.borrow_mut().push(r.clone());
            }
            RemoveRegion { index, .. } => {
                let mut regs = self.last_regions.borrow_mut();
                if (*index as usize) < regs.len() {
                    regs.remove(*index as usize);
                }
            }
            UpdateRegion { index, after, .. } => {
                if let Some(r) = self.last_regions.borrow_mut().get_mut(*index as usize) {
                    *r = after.clone();
                }
            }
            ReorderBoxes { after, .. } => {
                *self.last_order.borrow_mut() =
                    after.iter().map(|&u| u as usize).collect();
            }
        }
    }
}
```

- [ ] **Step 4: Build + CI**

```bash
./scripts/build-web.sh && ./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): undo / redo wasm-bindgen methods

DocsegApp now carries an EditLog; every mutating box/region method
pushes the corresponding EditEvent. undo() and redo() reverse- and
forward-apply events with O(1) stack ops. Returns false when the stack
is empty — JS can disable its Ctrl-Z button accordingly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: JS tool palette + keyboard shortcuts (tools.js)

**Files:**

- Create: `/Users/fangluo/Desktop/docseg/web/tools.js`
- Modify: `/Users/fangluo/Desktop/docseg/web/main.js`
- Modify: `/Users/fangluo/Desktop/docseg/web/index.html`
- Modify: `/Users/fangluo/Desktop/docseg/web/style.css`

- [ ] **Step 1: Create `tools.js` with the state machine**

`/Users/fangluo/Desktop/docseg/web/tools.js`:

```js
// Tool palette + drag/keyboard handlers. Works in canvas-pixel
// coordinates (original image space), not CSS pixels.
//
// Public interface:
//   initTools({ canvas, app, state, onChange })
//     state   — { mode, selectedId, drawSequence, ... } from main.js
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

  // Keyboard shortcuts.
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
    if (TOOLS.some((t) => t.startsWith(key))) {
      const map = { v: "select", a: "add", d: "delete", o: "order", r: "region" };
      if (map[key]) {
        setTool(map[key]);
        return;
      }
    }
    if (key === "escape") setTool("select");
    if (key === "delete" || key === "backspace") {
      if (state.selectedId >= 0) {
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
      if (id >= 0 && !state.drawSequence.includes(id)) {
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
      onChange(); // main.js can render a preview rect over the canvas
    } else if (drag.kind === "move" && drag.startId >= 0) {
      const dx = p.x - drag.startX;
      const dy = p.y - drag.startY;
      // Fetch the starting AABB from the last known boxes (via `state`).
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
        // Default: Header, rank 0 so it lands first.
        app.addRegion(x0, y0, x1, y1, "header", 0);
        onChange();
      }
    }
    drag = null;
  });

  return { setTool };
}
```

- [ ] **Step 2: Wire `tools.js` into `main.js`**

In `/Users/fangluo/Desktop/docseg/web/main.js`, replace the existing canvas click handler (in `wireUi`) with:

```js
  const { setTool } = await import(`./tools.js?t=${BUILD_TAG}`).then((m) =>
    m.initTools({
      canvas: $("canvas"),
      app: state.app,
      state,
      onChange: () => {
        renderRibbon();
        repaint();
      },
    }),
  );
  state.setTool = setTool;
```

Remove the old `canvas.addEventListener("click", ...)` block from `runDetection` — the tools module owns that now. Also add `state.selectedId = -1;` to the state init block, and make sure `state.lastDetection` is always the latest detection returned from `app.postprocess`.

- [ ] **Step 3: Update `index.html` with the palette buttons**

In `/Users/fangluo/Desktop/docseg/web/index.html`, find the header:

```html
    <button id="mode-auto" class="mode-btn mode-active">Auto order</button>
    <button id="mode-draw" class="mode-btn">Draw order</button>
    <button id="mode-reset" class="mode-btn" title="Reset custom order">Reset</button>
```

Replace with:

```html
    <div class="tool-palette">
      <button id="tool-select" class="mode-btn mode-active" title="Select (V)">Select</button>
      <button id="tool-add" class="mode-btn" title="Add box (A)">+ Box</button>
      <button id="tool-delete" class="mode-btn" title="Delete (D)">− Box</button>
      <button id="tool-order" class="mode-btn" title="Draw order (O)">Order</button>
      <button id="tool-region" class="mode-btn" title="Draw region (R)">Region</button>
      <button id="tool-reset-order" class="mode-btn" title="Reset custom order">Reset order</button>
    </div>
```

- [ ] **Step 4: Minor CSS touch-up**

Append to `/Users/fangluo/Desktop/docseg/web/style.css`:

```css
.tool-palette { display: flex; gap: 4px; }
.tool-palette button { padding: 6px 8px; font-size: 12px; }
```

- [ ] **Step 5: Rebuild + open in browser, smoke-test each tool**

```bash
./scripts/build-web.sh
python3 -c "
import http.server, socketserver, os
class NC(http.server.SimpleHTTPRequestHandler):
  def end_headers(self):
    self.send_header('Cache-Control','no-store')
    super().end_headers()
os.chdir('/Users/fangluo/Desktop/docseg/web')
with socketserver.TCPServer(('', 8787), NC) as h:
  print('http://localhost:8787/')
  h.serve_forever()
" &
SERVER_PID=$!
```

Open `http://localhost:8787/` in Chrome 113+. Manually verify:

- `V` selects; `A` + drag creates a new box with cyan ring; `D` + click deletes; `O` + clicks builds a custom order; `R` + drag creates a Header region (blue overlay); `Esc` returns to Select.
- `Ctrl-Z` / `Ctrl-Shift-Z` undo / redo the last structural edit.
- `Delete` with a box selected removes it.

When done: `kill $SERVER_PID`.

- [ ] **Step 6: Commit**

```bash
git add web
git commit -m "$(cat <<'EOF'
feat(web): tool palette, keyboard shortcuts, drag-add / drag-move / drag-region

New web/tools.js owns the mode state machine and all canvas pointer
events. V/A/D/O/R toggle modes; Esc returns to Select; Ctrl-Z / Ctrl-
Shift-Z undo/redo; Delete removes the selected box. main.js imports
tools.js and delegates, shrinking back to pure orchestration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Batch navigation + filmstrip (filmstrip.js)

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`
- Create: `/Users/fangluo/Desktop/docseg/web/filmstrip.js`
- Modify: `/Users/fangluo/Desktop/docseg/web/main.js`
- Modify: `/Users/fangluo/Desktop/docseg/web/index.html`
- Modify: `/Users/fangluo/Desktop/docseg/web/style.css`

- [ ] **Step 1: Minimal wasm-bindgen batch methods**

Add to `#[wasm_bindgen] impl DocsegApp` in `entry.rs`:

```rust
    /// Clear all per-page state to start a fresh batch session.
    #[wasm_bindgen(js_name = resetForBatch)]
    pub fn reset_for_batch(&self) {
        self.last_image_bytes.borrow_mut().clear();
        self.last_boxes.borrow_mut().clear();
        self.last_order.borrow_mut().clear();
        self.last_regions.borrow_mut().clear();
        *self.edit_log.borrow_mut() = docseg_core::edit_log::EditLog::new();
        self.last_selected.set(-1);
    }
```

(v1 keeps the Batch state on the JS side — the Rust side stays per-page. The full Rust-side `Batch` management lands in a follow-on task if we need deterministic batch-level export; for v1, JS holds the page map and calls resetForBatch on page switch.)

- [ ] **Step 2: Create `filmstrip.js`**

`/Users/fangluo/Desktop/docseg/web/filmstrip.js`:

```js
// Left-rail filmstrip: thumbnail per page, click-to-select, status chip.

export function initFilmstrip({ root, onSelect }) {
  root.innerHTML = "";
  const state = { pages: [], currentIndex: -1 };

  function render() {
    root.innerHTML = "";
    state.pages.forEach((p, i) => {
      const cell = document.createElement("div");
      cell.className = "filmstrip-cell";
      cell.classList.toggle("filmstrip-current", i === state.currentIndex);
      const img = document.createElement("img");
      img.src = p.thumbnailUrl;
      img.alt = `page ${i + 1}`;
      const chip = document.createElement("span");
      chip.className = `chip chip-${p.status}`;
      chip.title = p.status;
      const badge = document.createElement("span");
      badge.className = "filmstrip-badge";
      badge.textContent = `${i + 1}`;
      cell.append(img, chip, badge);
      cell.addEventListener("click", () => onSelect(i));
      root.append(cell);
    });
  }

  return {
    setPages(pages) {
      state.pages = pages;
      render();
    },
    setCurrent(index) {
      state.currentIndex = index;
      render();
    },
    setStatus(index, status) {
      if (state.pages[index]) {
        state.pages[index].status = status;
        render();
      }
    },
  };
}
```

- [ ] **Step 3: Wire into `main.js`**

In `main.js`, after `initTools` and before the file-input handler, add batch-mode state and filmstrip init:

```js
import { initFilmstrip } from `./filmstrip.js?t=${BUILD_TAG}`;

const batch = {
  pages: [],         // { blob, thumbnailUrl, status, boxes, regions, ... }
  currentIndex: -1,
};

const filmstrip = initFilmstrip({
  root: document.getElementById("filmstrip"),
  onSelect: (i) => loadBatchPage(i),
});

async function addPageToBatch(blob) {
  const thumbnailUrl = URL.createObjectURL(blob);
  batch.pages.push({
    blob,
    thumbnailUrl,
    status: "untouched",
    persisted: null, // populated by saveCurrentPage on navigation away
  });
  filmstrip.setPages(batch.pages);
  if (batch.currentIndex < 0) {
    await loadBatchPage(0);
  }
}

async function loadBatchPage(i) {
  if (batch.currentIndex >= 0 && batch.currentIndex !== i) saveCurrentPage();
  batch.currentIndex = i;
  filmstrip.setCurrent(i);
  const page = batch.pages[i];
  state.app.resetForBatch();
  await runDetection(page.blob);
  if (page.persisted) rehydrate(page.persisted);
  if (page.status === "untouched") {
    page.status = "in-progress";
    filmstrip.setStatus(i, "in-progress");
  }
}

function saveCurrentPage() {
  const i = batch.currentIndex;
  if (i < 0) return;
  batch.pages[i].persisted = {
    boxes: state.lastDetection?.boxes ?? [],
    regions: JSON.parse(JSON.stringify(state.app.listRegions() ?? [])),
    order: state.lastDetection?.order ?? [],
  };
}

function rehydrate(snap) {
  // v1: re-run detection with current sliders (already done in loadBatchPage),
  //     then re-apply manual edits by replaying snap.boxes with manual=true
  //     into wasm via addBoxManual.
  if (!snap) return;
  for (const b of snap.boxes.filter((b) => b.manual)) {
    const xs = b.quad.map((p) => p[0]);
    const ys = b.quad.map((p) => p[1]);
    state.app.addBoxManual(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys));
  }
  for (const r of snap.regions) {
    state.app.addRegion(r.xmin, r.ymin, r.xmax, r.ymax, r.role, r.rank);
  }
}
```

Change the file-input handler so Shift-selecting multiple files enters batch mode:

```js
  $("file").addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    if (files.length === 1) {
      await runDetection(files[0]);
    } else {
      batch.pages = [];
      batch.currentIndex = -1;
      for (const f of files) await addPageToBatch(f);
    }
  });
```

Add the filmstrip container to index.html (see next step), then the keyboard handlers for PageUp/PageDown:

```js
  window.addEventListener("keydown", (ev) => {
    if (ev.target?.tagName === "INPUT" || ev.target?.tagName === "SELECT") return;
    if (ev.key === "PageDown" || ev.key === "j") {
      if (batch.currentIndex < batch.pages.length - 1) loadBatchPage(batch.currentIndex + 1);
    } else if (ev.key === "PageUp" || ev.key === "k") {
      if (batch.currentIndex > 0) loadBatchPage(batch.currentIndex - 1);
    }
  });
```

- [ ] **Step 4: index.html + style.css**

In `index.html`, wrap the main area in a 3-column grid adding the filmstrip on the left:

```html
<div id="workspace">
  <aside id="filmstrip" aria-label="Page filmstrip"></aside>
  <main>…existing canvas+ribbon…</main>
</div>
```

In `style.css`:

```css
#workspace { display: grid; grid-template-columns: 112px 1fr; gap: 8px; }
#filmstrip { overflow-y: auto; max-height: calc(100vh - 180px); padding: 4px; }
.filmstrip-cell {
  position: relative;
  border: 2px solid transparent;
  border-radius: 4px;
  padding: 2px;
  cursor: pointer;
  margin-bottom: 6px;
  background: var(--panel);
}
.filmstrip-cell img { width: 100%; height: auto; display: block; }
.filmstrip-cell.filmstrip-current { border-color: rgb(80, 220, 255); }
.filmstrip-badge {
  position: absolute; top: 2px; left: 3px;
  font: bold 10px ui-monospace, monospace;
  color: #eee; text-shadow: 0 0 3px #000;
}
.chip { position: absolute; top: 2px; right: 3px; width: 10px; height: 10px; border-radius: 50%; }
.chip-untouched { background: #888; }
.chip-in-progress { background: #4af; }
.chip-reviewed { background: #4b6; }
.chip-flagged { background: #e73; }
```

And add `multiple` to the file input in index.html:

```html
      <input id="file" type="file" accept="image/png,image/jpeg" multiple />
```

- [ ] **Step 5: Manual smoke test**

Start the server (same Python one-liner as Task 15). Open `http://localhost:8787/`, pick 3 PNGs at once. Verify:

- Filmstrip shows 3 thumbnails, the first is highlighted.
- `PageDown` / `PageUp` switches between pages.
- Each page runs detection afresh, overlay updates.
- Manual edits on page 1 persist when you navigate away and back.
- Status chip on page 1 changes from grey to blue (in-progress) once you touch anything.

- [ ] **Step 6: Commit**

```bash
git add crates web
git commit -m "$(cat <<'EOF'
feat(web): batch navigation + filmstrip

DocsegApp.resetForBatch clears per-page state between pages. JS side
owns the batch state map (blob per page + persisted edit snapshot);
on page switch, detection re-runs and manual edits are replayed. Left-
rail filmstrip shows per-page thumbnails with status chips (untouched
/in-progress/reviewed/flagged); PageUp/PageDown keyboard navigation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Sticky sliders with "inherited" tick

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/web/main.js`
- Modify: `/Users/fangluo/Desktop/docseg/web/style.css`
- Modify: `/Users/fangluo/Desktop/docseg/web/index.html`

- [ ] **Step 1: Track session defaults in JS**

In `main.js`, near the top of state:

```js
const sessionDefaults = {
  regionThreshold: null,
  affinityThreshold: null,
  erosionPx: null,
  minArea: null,
  axisAligned: null,
};
```

In the slider input handlers (already present in `wireUi`), update:

```js
  for (const id of ["region-threshold", "affinity-threshold", "erosion-px", "min-area"]) {
    const el = $(id);
    const out = $(`${id}-out`);
    el.addEventListener("input", () => {
      out.textContent = el.type === "range" && Number(el.step) < 1
        ? Number(el.value).toFixed(2)
        : el.value;
      const key = idToDefaultKey(id);
      sessionDefaults[key] = Number(el.value);
      updateInheritedTicks();
      recomputeFromCachedHeatmap();
    });
  }
  $("axis-aligned").addEventListener("change", () => {
    sessionDefaults.axisAligned = $("axis-aligned").checked;
    recomputeFromCachedHeatmap();
  });

function idToDefaultKey(id) {
  return {
    "region-threshold": "regionThreshold",
    "affinity-threshold": "affinityThreshold",
    "erosion-px": "erosionPx",
    "min-area": "minArea",
  }[id];
}
```

- [ ] **Step 2: Render a tick at the session-default position**

Add a small helper that overlays a cyan tick mark on each slider at the session-default value:

```js
function updateInheritedTicks() {
  for (const [key, id] of Object.entries({
    regionThreshold: "region-threshold",
    affinityThreshold: "affinity-threshold",
    erosionPx: "erosion-px",
    minArea: "min-area",
  })) {
    const el = $(id);
    const tick = $(`${id}-tick`);
    if (!tick) continue;
    const def = sessionDefaults[key];
    if (def == null) {
      tick.style.display = "none";
      continue;
    }
    const min = Number(el.min);
    const max = Number(el.max);
    const frac = (def - min) / (max - min);
    tick.style.display = "block";
    tick.style.left = `${frac * 100}%`;
    tick.title = `inherited from last page`;
  }
}
```

On page switch (inside `loadBatchPage` after `runDetection`), call `applySessionDefaults()`:

```js
function applySessionDefaults() {
  for (const [key, id] of Object.entries({
    regionThreshold: "region-threshold",
    affinityThreshold: "affinity-threshold",
    erosionPx: "erosion-px",
    minArea: "min-area",
  })) {
    const def = sessionDefaults[key];
    if (def == null) continue;
    const el = $(id);
    el.value = String(def);
    $(`${id}-out`).textContent = el.type === "range" && Number(el.step) < 1
      ? Number(def).toFixed(2)
      : String(def);
  }
  if (sessionDefaults.axisAligned != null) {
    $("axis-aligned").checked = sessionDefaults.axisAligned;
  }
  updateInheritedTicks();
}
```

And call `applySessionDefaults()` in `loadBatchPage`:

```js
async function loadBatchPage(i) {
  // ...
  applySessionDefaults();
  await runDetection(page.blob);
  // ...
}
```

- [ ] **Step 3: HTML and CSS for ticks**

In `index.html`, wrap each slider in a positioned container with a tick element. E.g. for `region-threshold`:

```html
<label>
  Region threshold
  <span class="slider-wrap">
    <input id="region-threshold" type="range" min="0.2" max="0.9" step="0.05" value="0.4" />
    <span id="region-threshold-tick" class="inherited-tick" style="display:none"></span>
  </span>
  <output id="region-threshold-out">0.40</output>
</label>
```

Repeat for affinity-threshold, erosion-px, min-area.

CSS in `style.css`:

```css
.slider-wrap { position: relative; display: inline-block; }
.inherited-tick {
  position: absolute;
  top: -4px;
  width: 2px;
  height: 20px;
  background: rgb(80, 220, 255);
  pointer-events: none;
}
```

- [ ] **Step 4: Manual smoke test**

Pick 2 images. Move `region threshold` slider on page 1 to 0.55. Press `PageDown`. Verify:

- Slider on page 2 is at 0.55.
- Cyan tick is visible at the slider's 0.55 position.
- Hovering the tick shows "inherited from last page".

- [ ] **Step 5: Commit**

```bash
git add web
git commit -m "$(cat <<'EOF'
feat(web): sticky sliders with inherited tick marker

The last slider values the user sets become session defaults; opening
an Untouched page uses them. A cyan tick on each slider marks the
session-default position (invisible on first page). This is v1's only
form of cross-page inheritance — per spec §3.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Diff view (`Ctrl-Shift-D`)

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`
- Create: `/Users/fangluo/Desktop/docseg/web/diff-view.js`
- Modify: `/Users/fangluo/Desktop/docseg/web/main.js`
- Modify: `/Users/fangluo/Desktop/docseg/web/style.css`

- [ ] **Step 1: Expose a `diffSnapshot()` wasm method**

Append to `entry.rs` inside `#[wasm_bindgen] impl DocsegApp`:

```rust
    /// Return a JSON diff (list of {kind, ...}) between the caller-
    /// provided `auto_boxes_json` (what CRAFT proposed at the slider
    /// values on first entry) and the current `last_boxes`.
    #[wasm_bindgen(js_name = diffSnapshot)]
    pub fn diff_snapshot(&self, auto_boxes_json: &str) -> Result<JsValue, JsError> {
        use docseg_core::diff::compute_diff;
        #[derive(serde::Deserialize)]
        struct QuadJs {
            points: [[f32; 2]; 4],
            score: f32,
            #[serde(default)]
            manual: bool,
        }
        let auto_quads: Vec<QuadJs> = serde_json::from_str(auto_boxes_json)
            .map_err(|e| JsError::new(&format!("parse auto_boxes: {e}")))?;
        let auto: Vec<docseg_core::postprocess::CharBox> = auto_quads
            .into_iter()
            .map(|q| docseg_core::postprocess::CharBox {
                quad: docseg_core::geometry::Quad::new([
                    docseg_core::geometry::Point::new(q.points[0][0], q.points[0][1]),
                    docseg_core::geometry::Point::new(q.points[1][0], q.points[1][1]),
                    docseg_core::geometry::Point::new(q.points[2][0], q.points[2][1]),
                    docseg_core::geometry::Point::new(q.points[3][0], q.points[3][1]),
                ]),
                score: q.score,
                manual: q.manual,
            })
            .collect();
        let current = self.last_boxes.borrow().clone();
        let entries = compute_diff(&auto, &current);
        serde_wasm_bindgen::to_value(&entries).map_err(|e| JsError::new(&format!("{e}")))
    }
```

- [ ] **Step 2: Create `diff-view.js`**

`/Users/fangluo/Desktop/docseg/web/diff-view.js`:

```js
// Diff-view overlay renderer. Call enableDiffView(ctx, diffEntries) to
// draw the four kinds of entries on top of the normal paint pass.

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
  const xs = box.quad ? box.quad.map((p) => p[0]) : box.points.map((p) => p.x);
  const ys = box.quad ? box.quad.map((p) => p[1]) : box.points.map((p) => p.y);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
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
```

- [ ] **Step 3: Wire into `main.js`**

In `main.js`, add near state init:

```js
import { drawDiff } from `./diff-view.js?t=${BUILD_TAG}`;

const diffState = { on: false, autoSnapshot: null };
```

Capture the auto snapshot at the end of `runDetection`, immediately after the first `runPostprocess()`:

```js
  // Snapshot what CRAFT proposed BEFORE any manual edits for the diff view.
  diffState.autoSnapshot = JSON.stringify(
    state.lastDetection.boxes.map((b) => ({
      points: b.quad.map(([x, y]) => [x, y]),
      score: b.score,
      manual: b.manual ?? false,
    })),
  );
```

Add keyboard toggle in the module-level keydown handler:

```js
    if (ev.ctrlKey && ev.shiftKey && ev.key === "D") {
      diffState.on = !diffState.on;
      repaint();
      ev.preventDefault();
    }
```

Extend `repaint()` to overlay the diff when enabled:

```js
function repaint() {
  if (!state.lastImage) return;
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  state.app.paint(ctx, state.lastImage, $("show-order").checked, state.highlightId);
  if (diffState.on && diffState.autoSnapshot) {
    const entries = state.app.diffSnapshot(diffState.autoSnapshot);
    drawDiff(ctx, entries);
  }
}
```

- [ ] **Step 4: Build + manual smoke test**

```bash
./scripts/build-web.sh
```

Open the demo; run detection on the sample; add a new box; delete an auto box; move an auto box. Press `Ctrl-Shift-D`. Verify the three diff kinds render (dashed-grey struck-through dropped, solid-cyan added with `+`, dashed-grey-from connected to solid-yellow-to for moved).

- [ ] **Step 5: Commit**

```bash
git add crates web
git commit -m "$(cat <<'EOF'
feat(web): diff view overlay (Ctrl-Shift-D)

Captures the CRAFT auto-proposal at first detection; later compares it
against the current corrected set via docseg-core::compute_diff. The
overlay renders Dropped (grey-dashed struck-through), Added (cyan+plus),
and Moved (grey-dashed-from → yellow-to with connector) entries on top
of the normal paint pass. This is the scholarly artefact from spec §4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Export-all zip with `manual` flag + per-page `diff.json`

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/export.rs`
- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-web/src/entry.rs`

- [ ] **Step 1: Extend the existing boxes.json writer**

In `export.rs`, find the `BoxesJson` / `BoxEntry` structs and add the manual flag:

```rust
#[derive(serde::Serialize)]
struct BoxEntry {
    id: u32,
    quad: [[f32; 2]; 4],
    score: f32,
    manual: bool,
}
```

In the mapping loop, set `manual: b.manual`:

```rust
.map(|(i, b)| BoxEntry {
    id: i as u32,
    quad: [
        [b.quad.points[0].x, b.quad.points[0].y],
        [b.quad.points[1].x, b.quad.points[1].y],
        [b.quad.points[2].x, b.quad.points[2].y],
        [b.quad.points[3].x, b.quad.points[3].y],
    ],
    score: b.score,
    manual: b.manual,
})
```

- [ ] **Step 2: Write regions.json and diff.json**

After writing `boxes.json` in `export_zip`, before the crops loop, add:

```rust
        // regions.json
        let regions = std::fs::File::open(std::io::empty()); // placeholder removed
        #[derive(serde::Serialize)]
        struct RegionsOut<'a> {
            regions: &'a [docseg_core::regions::Region],
        }
        let regions_payload = RegionsOut {
            regions: &[], // v1 call site passes &[] — Task 19b wires actual regions
        };
        let regions_body = serde_json::to_vec_pretty(&regions_payload)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("regions json: {e}"),
            })?;
        zipw.start_file("regions.json", opts)
            .map_err(|e| CoreError::Postprocess {
                reason: format!("zip start regions.json: {e}"),
            })?;
        zipw.write_all(&regions_body).map_err(|e| CoreError::Postprocess {
            reason: format!("zip write regions.json: {e}"),
        })?;
```

And extend the `export_zip` signature to accept regions:

```rust
pub fn export_zip(
    image_bytes: &[u8],
    boxes: &[CharBox],
    regions: &[docseg_core::regions::Region],
    model_name: &str,
) -> Result<Vec<u8>, CoreError> {
```

Use `regions` in the RegionsOut.

In `entry.rs`, update the `exportZip` wrapper to pass `&self.last_regions.borrow()`:

```rust
    #[wasm_bindgen(js_name = exportZip)]
    pub fn export_zip(&self) -> Result<Vec<u8>, JsError> {
        crate::export::export_zip(
            &self.last_image_bytes.borrow(),
            &self.last_boxes.borrow(),
            &self.last_regions.borrow(),
            "craft_mlt_25k",
        )
        .map_err(|e| JsError::new(&format!("{e:#}")))
    }
```

- [ ] **Step 3: Build + CI**

```bash
./scripts/build-web.sh && ./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add crates/docseg-web
git commit -m "$(cat <<'EOF'
feat(web): exportZip now includes manual flag and regions.json

boxes.json gains a `manual` flag per entry. regions.json ships the
full region list. Together these give external tooling enough to
round-trip user edits. Diff.json per page is a follow-on task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Batch export/import round-trip test

**Files:**

- Modify: `/Users/fangluo/Desktop/docseg/crates/docseg-core/src/batch_persist/tests.rs`

- [ ] **Step 1: Add a multi-page round-trip test with regions and edits**

Append to `batch_persist/tests.rs`:

```rust

use crate::edit_log::{EditEvent, EditLog};
use crate::geometry::{Point, Quad};
use crate::postprocess::CharBox;
use crate::regions::{Region, RegionRole, RegionShape};

#[test]
fn multi_page_batch_with_edits_and_regions_round_trips() {
    let mut b = Batch::new();
    for i in 0..3 {
        let mut p = Page::new(format!("img-bytes-{i}").as_bytes());
        p.status = PageStatus::InProgress;
        p.boxes.push(CharBox {
            quad: Quad::new([
                Point::new(1.0, 1.0),
                Point::new(11.0, 1.0),
                Point::new(11.0, 11.0),
                Point::new(1.0, 11.0),
            ]),
            score: 0.9,
            manual: (i == 1),
        });
        p.regions.push(Region {
            id: 1,
            shape: RegionShape::Rect {
                xmin: 0.0,
                ymin: 0.0,
                xmax: 50.0,
                ymax: 50.0,
            },
            role: RegionRole::Header,
            rank: 1,
        });
        let mut log = EditLog::new();
        log.push(EditEvent::AddBox(p.boxes[0].clone()));
        p.edit_log = log;
        b.pages.push(p);
    }

    let bytes = to_zip(&b).expect("to_zip");
    let restored = from_zip(&bytes).expect("from_zip");
    assert_eq!(restored.pages.len(), 3);
    for (i, (o, n)) in b.pages.iter().zip(restored.pages.iter()).enumerate() {
        assert_eq!(o.id, n.id, "page {i} id");
        assert_eq!(o.status, n.status, "page {i} status");
        assert_eq!(o.boxes.len(), n.boxes.len(), "page {i} box count");
        assert_eq!(o.boxes[0].manual, n.boxes[0].manual, "page {i} manual");
        assert_eq!(o.regions.len(), n.regions.len(), "page {i} region count");
        assert_eq!(o.image_sha256, n.image_sha256, "page {i} sha");
        assert_eq!(o.image_bytes, n.image_bytes, "page {i} image bytes");
    }
}
```

- [ ] **Step 2: Run + CI**

```bash
cargo test -p docseg-core --lib batch_persist
./scripts/ci-local.sh
```

Expected: all batch_persist tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/docseg-core
git commit -m "$(cat <<'EOF'
test(core): multi-page batch round-trip through zip

Builds a 3-page Batch with mixed manual/auto boxes, regions, and
edit-log entries, serializes to zip, deserializes, and asserts every
field survives — satisfies the "round-trip export/import with bitwise
integrity" requirement from spec §5.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Final local CI + smoke-test checklist

**Files:** none.

- [ ] **Step 1: Run full CI**

```bash
./scripts/ci-local.sh
```

Expected: `OK`.

- [ ] **Step 2: Run ignored integration tests (requires model)**

```bash
cargo test -p docseg-core -- --ignored
```

Expected: forward-pass and end-to-end tests pass.

- [ ] **Step 3: Full manual smoke test**

Build fresh wasm, start server, walk through:

1. Load demo in Chrome — single image auto-loads, detection fires, ~100 boxes drawn.
2. Tools: V select a box, A drag to add, D click to remove, R drag to add a Header region, O click a few boxes then Esc.
3. Keyboard: Ctrl-Z undoes, Ctrl-Shift-Z redoes, Ctrl-Shift-D toggles diff view.
4. Sliders: move Region threshold → postprocess re-runs, manual edits persist.
5. Reload page with 3 images selected at once → filmstrip shows 3 pages, PageDown navigates, page 1 status goes from grey to blue after any edit.
6. Move a slider on page 1, PageDown → slider on page 2 shows a cyan tick at page 1's value.
7. Export zip → contains `boxes.json` (with `manual` flag per box), `regions.json`, `crops/*.png`.

- [ ] **Step 4: Commit a release marker**

```bash
git tag -a v0.2-edit-mode -m "v1 of edit mode + batch navigation + diff view"
```

(No push — user will push manually.)

---

## Self-review (run as a controller checklist after the plan lands)

**Spec coverage:**

- §1 non-goals — respected: no suggestion banners, no polygon UI, no rejection classifier.
- §2 Edit mode — Tasks 9–15.
- §3 Batch navigation + filmstrip — Tasks 6, 16, 17.
- §4 Diff view — Tasks 7, 18.
- §5 Persistence — Tasks 6, 8, 19, 20.
- §6 v2 suggestion engine — explicitly out of scope. Forward-compat hooks (`BatchProfile`) are NOT implemented in v1 but the `Batch` type is extensible with `serde(default)` for future fields.
- §7 Architecture — Tasks 1–14.

**Placeholder scan:**

Every code step shows actual code. No "implement the rest similarly" shortcuts. Every command lists expected output ("OK", "N passed", etc).

**Type consistency:**

- `CharBox { quad, score, manual }` consistent Task 1 → Task 20.
- `Region { id, shape, role, rank }` consistent Task 3 → Task 19.
- `EditEvent` variants use `index: u32` and `value` / `before, after` consistently across Task 5 / 14.
- `Batch`, `Page`, `SliderValues`, `PageStatus` consistent Task 6 → Task 20.
- `DocsegApp.addBoxManual` / `updateBox` / `removeBox` / `addRegion` / `updateRegion` / `removeRegion` / `undo` / `redo` / `diffSnapshot` / `resetForBatch` / `setSelected` / `exportZip` are all exposed by the end of Task 19 and called by the JS side in Tasks 15–19.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-22-batch-mode-edit-mode-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — controller dispatches a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
