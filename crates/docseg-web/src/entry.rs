//! Top-level wasm-bindgen surface: construct a `DocsegApp` and wire up
//! tracing + panic forwarding. Model load / detect / paint / export all
//! land in Task 12 onwards.

use wasm_bindgen::prelude::*;

/// Top-level handle returned to JS.
#[wasm_bindgen]
pub struct DocsegApp {
    _private: (),
}

#[wasm_bindgen]
impl DocsegApp {
    /// Construct a fresh app. Installs a tracing subscriber and a panic
    /// hook so Rust panics and `tracing::error!` calls show up in the
    /// browser console.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        tracing_wasm::try_set_as_global_default().ok();
        Self { _private: () }
    }
}

impl Default for DocsegApp {
    fn default() -> Self {
        Self::new()
    }
}
