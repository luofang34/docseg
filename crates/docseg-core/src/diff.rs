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
