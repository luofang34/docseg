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
