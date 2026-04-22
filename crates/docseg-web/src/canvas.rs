//! Image decoding from the bytes the browser hands us.

use docseg_core::CoreError;
use image::DynamicImage;

/// Decode a PNG / JPEG byte blob into a `DynamicImage`.
pub fn decode(bytes: &[u8]) -> Result<DynamicImage, CoreError> {
    image::load_from_memory(bytes).map_err(CoreError::Decode)
}
