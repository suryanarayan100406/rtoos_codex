"""Constructed geometry tests for S1 admission; these are not flight validation."""

from types import SimpleNamespace

import numpy as np
import pytest

from drishti.bundle.core import Bundle
from drishti.config.models import FrameQAConfig
from drishti.stages.s1_frameqa import STAGE as S1_STAGE
from drishti.stages.s1_frameqa import _load_optional_calibration, geometric_parallax_signal

cv2 = pytest.importorskip("cv2")

K = np.asarray([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    camera_points = (rotation @ points.T + translation[:, None]).T
    image_points = (K @ camera_points.T).T
    return image_points[:, :2] / image_points[:, 2, None]


def _correspondences(first: np.ndarray, second: np.ndarray):
    first_points = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in first]
    second_points = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in second]
    descriptors = np.zeros((len(first_points), 32), dtype=np.uint8)
    matcher = SimpleNamespace(
        match=lambda left, right: [cv2.DMatch(index, index, 0.0) for index in range(len(left))]
    )
    return (first_points, descriptors, 640, 480), (second_points, descriptors.copy(), 640, 480), matcher


def _scene() -> np.ndarray:
    return np.asarray(
        [[x, y, z] for z in (4.0, 6.0, 9.0, 13.0)
         for x in (-2.0, -1.0, 0.0, 1.0, 2.0)
         for y in (-1.0, 0.0, 1.0)],
        dtype=np.float64,
    )


def _settings() -> FrameQAConfig:
    return FrameQAConfig(
        min_parallax_px=2.0,
        min_geometric_matches=12,
        min_match_coverage_fraction=0.125,
        min_angular_parallax_deg=0.1,
    )


def test_rotation_is_removed_by_homography_despite_large_pixel_motion():
    points = _scene()
    rotation, _ = cv2.Rodrigues(np.asarray([0.0, 0.06, 0.0]))
    first, second, matcher = _correspondences(
        _project(points, np.eye(3), np.zeros(3)),
        _project(points, rotation, np.zeros(3)),
    )

    signal = geometric_parallax_signal(first, second, matcher, _settings())

    assert signal["feature_displacement_px"] > 10.0
    assert signal["homography"]["supported"]
    assert signal["homography"]["residual_quantile_px"] < 0.01
    assert signal["ambiguity"] == "rotation_or_planar_homography_dominant"
    assert not signal["admissible"]


def test_translation_with_depth_variation_survives_homography_compensation():
    points = _scene()
    first, second, matcher = _correspondences(
        _project(points, np.eye(3), np.zeros(3)),
        _project(points, np.eye(3), np.asarray([0.3, 0.0, 0.0])),
    )

    signal = geometric_parallax_signal(first, second, matcher, _settings())

    assert signal["model"] == "homography_rotation_planar_compensated_proxy"
    assert signal["homography"]["inlier_count"] >= _settings().min_geometric_matches
    assert signal["homography"]["residual_quantile_px"] > _settings().min_parallax_px
    assert signal["admissible"]


def test_validated_intrinsics_add_triangulated_angular_parallax_signal(tmp_path):
    points = _scene()
    first, second, matcher = _correspondences(
        _project(points, np.eye(3), np.zeros(3)),
        _project(points, np.eye(3), np.asarray([0.3, 0.0, 0.0])),
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        '{"width": 640, "height": 480, "camera_model": "PINHOLE", '
        '"params": [800.0, 800.0, 320.0, 240.0]}'
    )
    cfg = SimpleNamespace(optional=SimpleNamespace(intrinsics=calibration_path))
    calibration, status = _load_optional_calibration(SimpleNamespace(dataset=cfg), {
        "width": 640,
        "height": 480,
    })

    signal = geometric_parallax_signal(first, second, matcher, _settings(), calibration=calibration)

    assert status["status"] == "validated"
    assert signal["model"] == "essential_recover_pose_triangulated_angular_parallax"
    assert signal["essential"]["supported"]
    assert signal["essential"]["positive_depth_fraction"] == pytest.approx(1.0)
    assert signal["angular_parallax_deg"] > _settings().min_angular_parallax_deg


def test_no_model_support_is_a_cadence_fallback_and_due_frame_is_kept(tmp_path):
    blank = np.zeros((32, 32), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "first.png"), blank)
    cv2.imwrite(str(tmp_path / "second.png"), blank)
    bundle = Bundle(tmp_path, {
        "bundle_version": "1",
        "stages": {"s0_ingest": {"outputs": {"frames": "s0_ingest/frames.json"}}},
        "inputs": [],
        "config_snapshot": {},
    })
    bundle.write_json("s0_ingest/frames.json", {
        "frames": [
            {"id": "f0", "path": "first.png", "time_s": 0.0, "width": 32, "height": 32,
             "gps": {"latitude": 1.0}},
            {"id": "f1", "path": "second.png", "time_s": 2.0, "width": 32, "height": 32,
             "gps": {"latitude": 1.0}},
        ],
        "capabilities": {},
    })
    cfg = SimpleNamespace(
        run=SimpleNamespace(seed=1234),
        frameqa=FrameQAConfig(min_sharpness=0.0, max_dark_fraction=1.0, max_interval_s=1.0),
        dataset=SimpleNamespace(optional=SimpleNamespace(intrinsics=None)),
    )

    S1_STAGE.run(bundle, cfg, None)

    qa = bundle.read_json("s1_frameqa/qa.json")["frames"]
    assert qa[1]["selected"]
    assert qa[1]["qa"]["selection_basis"] == "coverage_interval"
    assert qa[1]["qa"]["geometric"]["status"] == "no_matches"
