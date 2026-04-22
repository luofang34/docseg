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
    assert_eq!(restored.pages[0].image_bytes, original.pages[0].image_bytes);
    assert_eq!(
        restored.pages[0].image_sha256,
        original.pages[0].image_sha256
    );
}

#[test]
fn round_trip_is_logically_identical() {
    let b = sample_batch();
    let z1 = to_zip(&b).expect("z1");
    let restored = from_zip(&z1).expect("restored");
    let z2 = to_zip(&restored).expect("z2");
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
    assert!(
        msg.contains("schema") || msg.contains("migrate"),
        "got {msg}"
    );
}
