#![allow(clippy::expect_used, clippy::panic)]

use super::{region_channel_channels_first, region_channel_channels_last};

#[test]
fn region_channel_extracted_from_channels_last_output() {
    // 2x2 heatmap, channels-last layout: [r00,a00, r01,a01, r10,a10, r11,a11].
    let raw = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6];
    let region = region_channel_channels_last(&raw, 2, 2);
    assert_eq!(region, vec![0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn region_channel_extracted_from_channels_first_output() {
    // 2x2 heatmap, channels-first layout: [r00,r01,r10,r11, a00,a01,a10,a11].
    let raw = vec![0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6];
    let region = region_channel_channels_first(&raw, 2, 2);
    assert_eq!(region, vec![0.1, 0.2, 0.3, 0.4]);
}
