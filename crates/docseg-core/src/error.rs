//! Typed errors surfaced by the core crate.

use thiserror::Error;

/// Errors produced anywhere in the `docseg-core` pipeline.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// Image bytes could not be decoded into a supported pixel format.
    #[error("image decode failed")]
    Decode(#[source] image::ImageError),
}
