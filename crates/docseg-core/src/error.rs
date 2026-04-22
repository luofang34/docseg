//! Typed errors surfaced by the core crate.

use thiserror::Error;

/// Errors produced anywhere in the `docseg-core` pipeline.
///
/// Variants carry every piece of context a caller needs to produce a useful
/// user-facing message (URLs, image dimensions, op hints) — no string-concat
/// at the error site.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// Image bytes could not be decoded into a supported pixel format.
    #[error("image decode failed")]
    Decode(#[source] image::ImageError),

    /// HTTP / network failure while fetching the model bundle.
    #[error("model fetch failed from {url}")]
    ModelFetch {
        /// URL that was attempted.
        url: String,
        /// Underlying transport error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Model bundle parsed but could not be initialized (unsupported op,
    /// malformed graph, etc.).
    #[error("model load failed ({hint})")]
    ModelLoad {
        /// Short diagnostic hint, e.g. the offending op name.
        hint: String,
        /// Underlying loader error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Forward pass failed (OOM, adapter lost, shape mismatch).
    #[error("inference failed")]
    Inference {
        /// Underlying runtime error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Preprocess rejected the input (zero-size, absurd aspect, etc.).
    #[error("preprocess failed ({width}x{height}): {reason}")]
    Preprocess {
        /// Offending image width in pixels.
        width: u32,
        /// Offending image height in pixels.
        height: u32,
        /// Human-readable reason.
        reason: String,
    },

    /// Postprocess could not interpret the inference output.
    #[error("postprocess failed: {reason}")]
    Postprocess {
        /// Human-readable reason.
        reason: String,
    },
}

#[cfg(test)]
mod tests;
