"""Detector-box and geometric-flow unit checks, not real-frame masking acceptance."""

from types import SimpleNamespace

import numpy as np

from drishti.stages.s3_masking import detection_mask, motion_mask


def motion_config():
    return SimpleNamespace(motion_features=1000, motion_quality=0.01, motion_spacing_px=4,
                           flow_window_px=21, flow_levels=3, flow_fb_threshold_px=1.5,
                           min_motion_matches=16, motion_threshold_px=2., motion_ransac_iterations=500,
                           motion_ransac_confidence=0.99, motion_min_inlier_fraction=0.5,
                           flow_pyr_scale=0.5, flow_iterations=3, flow_poly_n=5, flow_poly_sigma=1.2)


def test_unknown_textureless_frame_is_not_confident_all_static():
    image = np.zeros((50, 50, 3), np.uint8)
    mask, metrics = motion_mask(image, image, motion_config())
    assert mask is None
    assert metrics["status"] == "unknown"


def test_supported_homography_motion_residual_runs_on_actual_optical_flow():
    import cv2

    rng = np.random.default_rng(12)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (160, 160), np.uint8), (3, 3), 0)
    image = np.repeat(gray[..., None], 3, axis=2)
    shifted = cv2.warpAffine(image, np.array([[1., 0, 4], [0, 1., 2]]), (160, 160))
    mask, metrics = motion_mask(image, shifted, motion_config(), seed=123)
    assert mask is not None
    assert metrics["status"] == "ok"
    assert metrics["method"] == "farneback_homography_residual"
    assert metrics["inlier_matches"] >= 16
    assert np.mean(mask[20:-20, 20:-20]) < 0.05


def test_conservative_dynamic_boxes_clipped_and_dilated():
    mask, selected = detection_mask((20, 20), [
        {"class": "person", "box": [-1, 2, 5, 8], "score": 0.8},
        {"class": "building", "box": [10, 10, 20, 20], "score": 0.9}
    ], ["person"], 2)
    assert len(selected) == 1
    assert np.all(mask[2:8, :5])
    assert mask[8, 4]
    assert not mask[19, 19]
