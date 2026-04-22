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
