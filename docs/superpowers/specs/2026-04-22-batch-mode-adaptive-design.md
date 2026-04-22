# `docseg` — batch mode, edit mode, and adaptive assistance

Status: design under review 2026-04-22. Synthesized from four parallel
Opus-4.7 design-agent angles (UX / adaptive-algorithms / architecture /
critic). Supersedes and subsumes
`docs/superpowers/specs/2026-04-22-edit-mode-and-layout-regions-design.md`
(the "v1 edit-mode" spec), keeping its CharBox.manual/Region/IoU-merge
bones but reworking scope and adaptation.

## 0. Design philosophy

The ONNX CRAFT weights are frozen. No postprocess trick, threshold
sweep, or rejection classifier makes the model smarter on later pages.
The tool's adaptation is the propagation of the **user's deterministic
preferences** (slider values, region layout, direction, size priors)
across pages — with every propagation visible, reversible, and explicit.

Two consequences we enforce:

- **No silent inheritance.** Any parameter that differs from the
  slider's default on the page the user is looking at must show a
  visible "inherited from page k" tick at the control itself, not in
  a footer. If it's load-bearing enough to change the output, it's
  load-bearing enough to show at the control.
- **No pretend intelligence.** A tooltip next to the adaptation UI
  reads: *"Your corrections tune the post-processing, not the neural
  model. Rerunning page 1 after 50 corrections will apply a better
  threshold, not a smarter network."* Domain experts prefer honesty
  about limits to euphemism.

## 1. Scope

### v1 (this spec — ships)

1. **Edit mode** on a single page: Select / Add / Delete / Resize /
   Order-draw / Region tools + 50-entry flat undo stack.
2. **Batch navigation**: filmstrip, keyboard nav, per-page status,
   slider values carry forward as sticky defaults.
3. **Visible inheritance only**: no auto-computed profile, no
   suggestion banners. The slider just remembers the last position
   and shows "inherited" at the control, one-click to reset-to-default.
4. **Diff view**: per-page toggle showing original CRAFT proposal
   (greyed) vs user-corrected set (full color). Primary scholarly
   artefact.
5. **Persistence**: local-first zip/directory project file with
   sidecar PNGs (not base64), schema-versioned JSON, save-on-nav
   + explicit Save. No background autosave timer.
6. **Round-trip export/import** with a unit test proving bitwise
   idempotence.

### v2 (deferred — scoped here for forward compatibility)

- **Opt-in suggestion engine** (`BatchProfile` running stats, delta
  preview, accept/reject per field) — only if the user explicitly
  wants it after trying v1. Guardrails in §6.
- **Region template propagation** across pages (proportional rect
  with whitespace snap).
- **Named page ranges** (folio / recto-verso / front-matter).
- **Flagged-for-review hints** (e.g. heatmap entropy outliers).
- **Multi-select / batch edit** (promote one box's change to all
  selected).

### Explicitly cut (from both prior spec and agents' proposals)

| Feature | Proposed by | Why cut |
|---|---|---|
| Rejection classifier on color/shape features | adaptive-algorithms | Second ML system bolted on the first. Scholar can delete a seal dot in one click. Defer to v3 or never. |
| Every-10th-page mandatory sanity-review banner | UX | Paternalistic to a domain expert. They know when to review. |
| 200-entry undo + 2-second delete ghost + Recently Removed tray + event-sourced edit log | UX + architecture | Four overlapping reversal systems for "adjust a box." Flat 50-entry undo is enough. |
| Sweep-and-pick-best-F1 threshold calibration | adaptive-algorithms | Optimizes F1 against a biased tiny sample ("errors the user noticed most recently"). Show raw FP/FN counts instead; let the expert decide. |
| IndexedDB autosave every 15 s | UX | Arbitrary cadence. Save-on-nav is clearer. |
| JSON project file with embedded base64 images | (naive default) | 100 MB base64-inflated JSON defeats diffability + git-friendliness. Sidecar PNGs + SHA256 is standard. |
| "Parameters inherited" subtle footer note | UX | Worst of both worlds: present enough to imply disclosure, quiet enough to miss. Show at the control or don't inherit. |
| Low-confidence amber fill | UX | Loud. Dashed outline alone is enough. |

## 2. v1 — edit mode (single page)

Inherits the tool palette from the prior v1 edit-mode spec; the changes
below are the ones this synthesis tightens.

### 2.1 Tool palette

| Tool | Shortcut | Notes |
|---|---|---|
| Select | `V` | Click = highlight. Tab / Shift-Tab walks reading order with a 30 % zoom-in preview. Drag body = move. Drag 8 handles = resize. Delete key or `D` removes. |
| Add box | `A` | Drag-rect on empty canvas. Shift-locks to median neighbor height. 6 px edge snap to disambiguate "nudge existing" vs "create new". |
| Delete (standalone) | — | Folded into Select+Delete-key. No separate tool. |
| Order-draw | `O` | Click boxes in sequence. `Esc` commits. |
| Region | `R` | Drag-rect. Role defaults to Header; rank defaults to (min existing rank) − 1. Role/rank dropdown in corner badge. |

Every tool has `Esc` → return to Select. Spacebar toggles overlay
visibility (quick raw-image check). No context menus.

### 2.2 Visual language

- Auto-detected box: solid yellow stroke, 2 px.
- User-edited box: same stroke + 1 px cyan inner ring. The ring
  persists in exports so a later reader can see what was human-touched.
- Low-confidence box (region score < page median − 1σ): dashed 1.5 px
  stroke. No fill. No amber. Cyan ring wins over dashed if both apply.
- Region: translucent stroke in role color, corner badge
  `rank · role`.
- Reading-order arrows: three styles per the prior spec — continue
  (solid, no arrowhead), carriage-return (dashed chevron), region-break
  (thicker orange arrow).

### 2.3 Undo / redo

One flat `VecDeque<EditEvent>` per page, capacity 50. Redo stack
cleared on every new edit. `Ctrl-Z` / `Ctrl-Shift-Z`. Switching pages
does not clear (per-page).

```rust
pub enum EditEvent {
    AddBox(CharBox),
    RemoveBox(BoxId, CharBox),
    UpdateBox { id: BoxId, before: CharBox, after: CharBox },
    AddRegion(Region),
    RemoveRegion(RegionId, Region),
    UpdateRegion { id: RegionId, before: Region, after: Region },
    ReorderBoxes { before: Vec<BoxId>, after: Vec<BoxId> },
}
```

Inline before/after gives O(1) undo without snapshots. Slider
overrides are not in the edit log — they're page state, and the user
can just move the slider back.

## 3. v1 — batch navigation

### 3.1 Data model

`Batch` becomes the root object; `DocsegApp` is the view onto the
currently selected page.

```rust
// docseg-core
pub struct Batch {
    pub id: BatchId,              // ULID
    pub schema_version: u32,      // bump on migration
    pub pages: Vec<Page>,
    pub session_defaults: SliderDefaults,  // "sticky" sliders, §3.3
    pub created_at: i64,
    pub updated_at: i64,
}

pub struct Page {
    pub id: PageId,
    pub image_sha256: [u8; 32],
    pub image_dims: (u32, u32),
    pub status: PageStatus,       // Untouched | InProgress | Reviewed | Flagged
    pub boxes: Vec<CharBox>,
    pub regions: Vec<Region>,
    pub order: Vec<BoxId>,
    pub sliders: SliderValues,    // effective values for this page
    pub edit_log: EditLog,
    pub reviewed_at: Option<i64>,
}
```

Not stored on disk: `preprocess_cache`, `heatmap_cache` (region +
affinity f32 at 320×320). LRU-cached in `docseg-web`, keyed by
`(image_sha256, preprocess_params_hash)`, window of last 3 pages
(~2.4 MB × 3 = 7.2 MB RAM). Recomputing a non-cached page on
navigation is ~400 ms, acceptable.

### 3.2 UI layout

```
┌─filmstrip ─┬─────────── canvas ─────────┬── ribbon ──┐
│ [thumb 1] │                             │ [crop 1]   │
│ [thumb 2] │  image + overlay + arrows   │ [crop 2]   │
│ [thumb 3] │                             │ ...        │
│ ...       │                             │            │
└───────────┴─────────────────────────────┴────────────┘
```

Filmstrip thumbnails show the image + low-opacity detection overlay.
Status chip per thumbnail:

- grey dot — Untouched (never opened)
- blue half-circle — InProgress (opened, has edits)
- green check — Reviewed (user marked complete)
- amber triangle — Flagged (user pressed `F` to revisit later)

Navigation: `PageDown` / `PageUp` or `J` / `K`, `Ctrl-G` jump prompt,
click thumbnail. Current page has 2 px accent border.

Top bar: `Page 7 of 42 · 6 reviewed · 2 flagged · ~34 remaining`.
Estimated time remaining is only shown after ≥ 3 pages reviewed
(median time × remaining).

### 3.3 Sticky sliders (the only inheritance in v1)

When the user moves a slider, that value becomes the `session_defaults`
for the batch. Opening a new `Untouched` page uses those defaults.
Opening an `InProgress` / `Reviewed` page restores its own saved values.

**Visible inheritance indicator**: each slider shows a small tick mark
at the session-default value, in cyan, with a tooltip `inherited from
page 3`. The current handle is the normal color. Clicking the tick
resets the slider to the session default. This makes the inheritance
unmissable at the control.

No other inheritance. Region template, reading direction, order
overrides — all are per-page fresh by default. If the user wants to
copy a region across pages, v2 will offer "apply to all pages"; v1 is
manual.

### 3.4 Page re-detection policy

When the user opens a page for the first time (Untouched → InProgress),
we run preprocess + inference + postprocess with the current
`session_defaults`. When they adjust a slider, we re-run postprocess
only (heatmap cached), merging with manual edits via the IoU-merge
rule below.

**IoU-merge rule** (inherited from the prior edit-mode spec and restated
here for self-containment):

```
merge(auto_boxes, manual_boxes) -> final:
  final = manual_boxes.clone()
  for a in auto_boxes:
    if !manual_boxes.any(|m| iou(m.aabb, a.aabb) > 0.5):
      final.push(a)   // auto box passes through; no manual overlap
  # manual boxes always win in collisions.
```

`CharBox { quad, score, manual: bool }` carries the `manual` flag
(true for user-added or user-edited boxes). The flag persists through
exports so a later reader can see what was human-touched.

When the user reopens a Reviewed page and edits a box, status
transitions back to InProgress. Explicit `Mark reviewed` button
re-promotes. (This is the "safer" answer to the architecture agent's
open question.)

## 4. v1 — diff view

Toggle `Ctrl-Shift-D` or a toolbar button: Diff On / Diff Off.

When on, the canvas renders the original CRAFT proposal (what the
detector returned for `session_defaults` with zero manual edits)
overlaid with the current corrected set:

- **Dropped auto-boxes** (present in original, removed by user):
  grey dashed outline, strike-through label.
- **Manual adds** (present in current, absent in original): solid
  cyan outline, `+` marker in corner.
- **Moved/resized boxes**: grey dashed outline at original position,
  solid yellow at current position, thin connector between centroids.
- **Unchanged auto-boxes**: solid yellow (normal).

This is the **scholarly artefact**: a visible record of what the
machine said vs what the expert decided. Every page's diff is also
captured to the export zip as `pages/NNNN/diff.json`.

## 5. v1 — persistence

### 5.1 Project format

Zip container by default; direct directory when the File System
Access API is available (Chromium). Same logical layout either way:

```
batch.json                 # Batch minus pages[].image / heavy fields
manifest.json              # schema_version + sha256 of every file
pages/
  page_0001.json           # Page: boxes, regions, order, status, sliders, edit_log
  page_0001.diff.json      # diff view snapshot (§4)
  page_0002.json
  ...
images/
  page_0001.png            # original bytes, unmodified
  page_0002.png
  ...
```

- JSON, not msgpack. Diffability > 2× compression. A 50-page batch is
  ~5 MB of JSON (boxes dominate, ~1 KB / 100 boxes) + original images.
- `image_sha256` stored in each `page_NNNN.json`; on load, SHA256 of
  the on-disk `page_NNNN.png` is verified. On drift, the page loads
  with status = `Flagged(ImageDrift)` and is excluded from anything
  dependent on image content until re-reviewed.
- `schema_version: u32` at the top of `batch.json`; `docseg-core`
  owns `migrate(value, from) -> Batch` as a chain of per-version
  migrations, each covered by a golden-file test.

### 5.2 Save triggers

- **On navigate away** from a page with unsaved edits.
- **On explicit Save** (`Ctrl-S` or toolbar).
- **On tab-close** via `beforeunload`, best-effort.

No 15-second timer. No IndexedDB mirror. The FSA handle or zip blob
is the source of truth.

### 5.3 Round-trip test

`docseg-core` ships a unit test:

```rust
#[test]
fn export_then_import_is_bitwise_identical() { ... }
```

Serialize a `Batch` fixture → parse → serialize again; the two byte
sequences must be identical. This is the primary guarantee behind
"canonical segmentation."

## 6. v2 (deferred, but API-ready) — suggestion engine

When the user asks for adaptation — **never on by default** — we
expose a `BatchProfile` as running statistics and a `propose(next_page)
→ ProfileDelta` entry point. All math lives in `docseg-core` as pure
folds.

### 6.1 Signals

| Signal | Formula | Notes |
|---|---|---|
| Region threshold | EMA on signed error: `theta_r ← clamp(theta_r + eta·(FP_rate − FN_rate), 0.2, 0.9)`, `eta = 0.05·alpha_k` | Cheapest, explainable. No sweep-and-pick-F1 (biased). |
| Affinity threshold | Same EMA on affinity domain |  |
| Size prior | `median ± 3·MAD` over confirmed-kept boxes, P5/P95 clamps. Updated with `max(slider_default, adapted)` guard on first 2 pages | Robust stats only. Shown as visual hint on the Add tool, never enforced as a filter in v1. |
| Direction prior | Count disagreements between auto-order and user-override; flip direction only when disagreement ratio > 0.7 and pattern explains ≥ 0.7 of reversed pairs | Always confirm before flipping (answer to UX agent's Q1). |
| Region layout | Proportional rect `(x/W, y/H, w/W, h/H)` with ±5% whitespace-snap on edges. No SIFT/ORB. | Only if the user has drawn a region. |

Explicitly rejected: rejection classifier on color/shape features.

### 6.2 Guardrails

- **≥ 2 reviewed-non-isolated pages required** before any suggestion
  appears.
- **Per-step caps**: `|Δtheta| ≤ 0.05`, `|Δmin_area| ≤ 20 %` per page.
- **Regret check**: if `FP + FN` rises for two consecutive pages after
  an adaptation, revert to the pre-adaptation values and freeze for
  one page. UI: "adaptation paused — corrections increased."
- **Profile is always recomputable** from the reviewed-not-isolated
  set. The running stats are a cache; `recompute(batch) ==
  batch.profile` is an invariant checked in debug builds.

### 6.3 UI

Banner on navigate-forward to an Untouched page:

> Suggestion: region threshold 0.48 → 0.52, min-area 180 → 240.
> Based on corrections from pages 1-5.
> **Apply to this page** · **Apply to rest of batch** · **Dismiss**

Never auto-applied on the page the user is looking at. The suggestion
also exposes the per-signal breakdown with individual accept toggles
(per the adaptive-algorithms agent's alignment requirement).

### 6.4 What to isolate

Per-page `isolated: bool` (aka `excluded_from_stats`). Toggling
recomputes the profile from scratch — cheap at N ≤ 50 pages. Filmstrip
shows a dashed border for isolated pages.

## 7. Architecture

### 7.1 Crate split

- `docseg-core`: `Batch`, `Page`, `BatchProfile`, `ProfileDelta`,
  `EditLog`, `EditEvent`, `CharBox`, `Region`, all serde-safe. All
  math (profile folds, proposal generation, migrations). Native-
  testable.
- `docseg-web`: `DocsegApp` facade with additive methods (§7.2). Owns
  the ORT session and the heatmap LRU.

### 7.2 `DocsegApp` additions (additive, old methods unchanged)

```rust
#[wasm_bindgen]
impl DocsegApp {
    pub fn new_batch(&mut self, id: &str) -> Result<(), JsError>;
    pub fn add_page(&mut self, image_bytes: &[u8]) -> Result<u32, JsError>;
    pub fn select_page(&mut self, index: u32) -> Result<(), JsError>;
    pub fn mark_reviewed(&mut self, index: u32) -> Result<(), JsError>;
    pub fn set_flagged(&mut self, index: u32, flagged: bool) -> Result<(), JsError>;
    // v2 only — needed by the suggestion engine to exclude outlier pages
    // from BatchProfile. v1 Batch does not yet use it.
    pub fn set_excluded(&mut self, index: u32, excluded: bool) -> Result<(), JsError>;
    pub fn add_box_manual(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) -> Result<u32, JsError>;
    pub fn update_box(&mut self, id: u32, x0: f32, y0: f32, x1: f32, y1: f32) -> Result<(), JsError>;
    pub fn remove_box(&mut self, id: u32) -> Result<(), JsError>;
    pub fn add_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, role: &str, rank: u32) -> Result<u32, JsError>;
    pub fn update_region(&mut self, id: u32, x0: f32, y0: f32, x1: f32, y1: f32, role: &str, rank: u32) -> Result<(), JsError>;
    pub fn remove_region(&mut self, id: u32) -> Result<(), JsError>;
    pub fn undo(&mut self) -> Result<bool, JsError>;    // returns true if something was undone
    pub fn redo(&mut self) -> Result<bool, JsError>;
    pub fn export_batch_zip(&self) -> Result<Vec<u8>, JsError>;
    pub fn import_batch_zip(&mut self, bytes: &[u8]) -> Result<(), JsError>;
    pub fn diff_snapshot(&self, index: u32) -> Result<JsValue, JsError>;
    // existing single-page API still works; if no batch exists, a one-page
    // batch is synthesized on first preprocess_image call.
}
```

### 7.3 Heatmap LRU

```rust
// docseg-web
struct HeatmapCache {
    entries: VecDeque<(CacheKey, Heatmaps)>, // max 3 entries
}
```

Preprocess-params-hash is the hash of `(target_side, mean, std)`.
Changing the side blows the cache (rare); changing only the
postprocess sliders does not (they don't affect inference input).

### 7.4 Failure modes and structural mitigations

| Failure | Mitigation |
|---|---|
| Profile poisoning from one outlier page | `BatchProfile` is a pure fold over reviewed-non-excluded. `recompute(batch) == batch.profile` invariant checked in debug. |
| Image drift between save and load | `image_sha256` per page, verified on load. On drift → `Flagged(ImageDrift)`; never silently re-run on a different image under old boxes. |
| Schema-version skew, older build opening newer file | `schema_version` read first; `MigrationError::TooNew { file, supported }` with numbers in message. No partial parse. |
| v1 user opens a v1-vintage file in a v2 build | Chain of `v1_to_v2` migrations, golden-file tests per step. |

## 8. Open questions for the user

1. **(From UX agent)** When corrections imply the reading direction
   was wrong on page 3, auto-flip on future pages with a banner, or
   always require explicit confirmation? **Proposed answer in this
   spec: always require confirmation.** (Respects expert; avoids
   disorientation.)

2. **(From adaptive-algorithms agent)** Within one batch, is the
   scribe / hand assumed constant? If yes, size priors and direction
   priors keep accumulating. If no, adaptation should reset at a
   detected hand-boundary. **Proposed answer: assume constant in v1;
   address in v2 with an explicit "new scribe from here" divider.**

3. **(From architecture agent)** Re-editing a box on a Reviewed page:
   auto-revert to InProgress, or silently update profile? **Proposed
   answer: auto-revert to InProgress.** (Matches deferred-adaptation
   model; prevents retroactive profile mutation.)

4. **(From critic agent)** The v1 fork:
   **(a)** deterministic sliders + better visibility into where the
   model disagrees with you (what this spec proposes), or
   **(b)** the tool proposes parameter changes you accept/reject.
   **Proposed answer: (a) in v1, (b) opt-in in v2 (§6).**

## 9. Implementation phases

- **Phase A** — edit mode on single page (existing v1 edit-mode spec
  features): tool palette, 8-handle resize, regions, IoU-merge. The
  foundation.
- **Phase B** — batch navigation + persistence: `Batch` / `Page`
  types, filmstrip, sticky sliders, zip export/import, migration
  scaffold. Ships as the stated v1.
- **Phase C** — diff view + round-trip test. Completes v1.
- **Phase D** (future) — suggestion engine (§6). Not part of v1.

Phases A and B can overlap in implementation if convenient. Phase C
can't start before both.
