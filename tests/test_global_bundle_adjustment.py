"""Constructed S6/anchor contracts; no recorded-flight acceptance is implied."""

import copy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from drishti.bundle import Bundle
from drishti.config import Config, load_config
from drishti.config.models import BundleAdjustmentConfig, DatasetDescriptor, TelemetryConfig
from drishti.errors import InputError
from drishti.recon.depth import sparse_anchors
from drishti.report.builder import fit_metrics
from drishti.runtime import Runtime
from drishti.stages.runner import STAGES, StageRunner
from drishti.stages.s4_depth import pose_sparse_source
from drishti.stages.s6_global import STAGE

ROOT = Path(__file__).resolve().parents[1]


def scene_bundle(tmp_path):
    """Measured-feature-shaped math inputs, never substitutes for real footage."""
    centers = np.array([[0., 0., 0.], [1., .2, .03], [2., -.3, .04], [3., .4, -.1]])
    points = np.array([[x, y, 8. + (x + y) / 4] for x in (-1., 0., 1., 2.)
                       for y in (-.7, .1, .8)], dtype=np.float64)
    K = np.array([[160., 0., 120.], [0., 160., 100.], [0., 0., 1.]])
    cameras = []
    for index, center in enumerate(centers):
        xyz = points - center
        pixels = xyz[:, :2] / xyz[:, 2, None] * 160. + [120., 100.]
        initial_center = center + [.01 * index, -.015 * index, .005 * index]
        cameras.append({
            "frame_id": f"math{index}", "image_path": f"unused{index}.png",
            "time_s": float(index), "width": 240, "height": 200, "K": K.tolist(),
            "distortion": [], "projection_domain": "undistorted_pinhole",
            "R_cw": np.eye(3).tolist(), "center": initial_center.tolist(),
            "visual_reference": {"R_cw": np.eye(3).tolist(), "center": initial_center.tolist()},
            "gps": (center + [.12, .02, .03]).tolist(), "gps_inlier": True,
            "gps_sample": {"sigma_h_m": .2, "sigma_v_m": .3},
            "observations": [{"point_index": i, "uv": uv.tolist()} for i, uv in enumerate(pixels)],
            "anchors": [],
        })
    bundle = Bundle(tmp_path, {"bundle_version": "1", "stages": {}, "inputs": [],
                               "config_snapshot": {}})
    bundle.stage_dir("s2_poses")
    poses = {"frame": "projected_local", "origin": [500000., 3000000., 100.],
             "crs": {"epsg": 32643, "vertical": "ellipsoidal"}, "cameras": cameras,
             "pose_revision": "s2_metric_spine", "sparse_path": "s2_poses/sparse.npz"}
    bundle.write_json("s2_poses/poses.json", poses)
    sparse = {"points": points + [.015, -.02, .04],
              "colors": np.full((len(points), 3), 128, dtype=np.uint8),
              "confidence": np.full(len(points), .6), "inferred": np.zeros(len(points), bool),
              "reprojection_error_px": np.full(len(points), 1.),
              "track_length": np.full(len(points), len(cameras), dtype=np.int64)}
    np.savez_compressed(tmp_path / "s2_poses/sparse.npz", **sparse)
    bundle.manifest["stages"]["s2_poses"] = {
        "outputs": {"poses": "s2_poses/poses.json", "sparse": "s2_poses/sparse.npz"},
    }
    return bundle, Config(), poses, sparse


def test_s6_joint_adjustment_publishes_coherent_final_poses_structure_and_anchors(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("gtsam")
    pytest.importorskip("cv2")
    bundle, cfg, source_poses, source_sparse = scene_bundle(tmp_path)
    result = STAGE.run(bundle, cfg, None)
    assert result.degraded_reason is None, result.degraded_reason
    bundle.manifest["stages"]["s6_global"] = {"outputs": result.outputs}
    poses = bundle.read_json(result.outputs["poses"])
    optimization = bundle.read_json(result.outputs["optimization"])
    adjustment = optimization["bundle_adjustment"]
    assert adjustment["status"] == "optimized"
    assert adjustment["optimizer_invoked"]
    assert optimization["optimizer_invoked"]  # GTSAM also actually executed.
    assert adjustment["final_robust_cost"] < adjustment["initial_robust_cost"]
    assert poses["pose_revision"] == "s6_joint_reprojection_GNSS_bundle_adjusted"
    assert optimization["final_pose_revision"] == poses["pose_revision"]
    assert poses["bundle_adjustment"] == adjustment
    assert adjustment["calibrated_intrinsics_fixed"]
    assert not adjustment["fit_is_independent_accuracy"]
    assert fit_metrics(result.metrics)["bundle_adjustment"]["gnss_fit_rmse_m"] == adjustment["gnss_fit_rmse_m"]
    sparse_path, source, warning = pose_sparse_source(bundle, poses)
    assert sparse_path == bundle.artifact("s6_global", "sparse")
    assert source == "final_pose_declared_sparse" and warning is None
    with np.load(sparse_path, allow_pickle=False) as data:
        points, confidence = data["points"], data["confidence"]
        assert points.dtype == np.float64
        assert not np.array_equal(points, source_sparse["points"])
        assert np.all(confidence > 0)
        np.testing.assert_array_equal(data["inferred"], source_sparse["inferred"])
        np.testing.assert_array_equal(data["colors"], source_sparse["colors"])
    pixel_errors = []
    for camera, original in zip(poses["cameras"], source_poses["cameras"], strict=True):
        assert camera["K"] == original["K"] and camera["image_path"] == original["image_path"]
        assert camera["observations"] == original["observations"]
        assert "pose_covariance_tangent" not in camera
        assert "covariance_domain" not in camera
        xyz = (points - camera["center"]) @ np.asarray(camera["R_cw"]).T
        uv = xyz[:, :2] / xyz[:, 2, None] * 160. + [120., 100.]
        observed = np.array([observation["uv"] for observation in camera["observations"]])
        errors = np.linalg.norm(uv - observed, axis=1)
        pixel_errors.extend(errors.tolist())
        for index, anchor in enumerate(camera["anchors"]):
            np.testing.assert_allclose(anchor, [*observed[index], xyz[index, 2], errors[index]], atol=1e-12)
        # The depth consumer must get new Z, not the original S2 anchors/points.
        anchors = sparse_anchors(points, camera, np.zeros((200, 240), bool), confidence,
                                 cfg.depth.max_anchors, cfg.depth.max_anchor_reprojection_error_px)
        assert len(anchors) == len(points)
        expected = np.column_stack((uv, xyz[:, 2]))
        np.testing.assert_allclose(anchors[np.argsort(anchors[:, 0])], expected[np.argsort(expected[:, 0])])
    assert adjustment["reprojection_rmse_px"] == pytest.approx(np.sqrt(np.mean(np.square(pixel_errors))))
    assert bundle.read_json("s2_poses/poses.json") == source_poses
    with np.load(bundle.artifact("s2_poses", "sparse"), allow_pickle=False) as source_data:
        np.testing.assert_array_equal(source_data["points"], source_sparse["points"])


def test_s6_nonconvergence_keeps_pose_graph_products_and_records_reason(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("gtsam")
    bundle, cfg, _, _ = scene_bundle(tmp_path)
    cfg.bundle_adjustment.enabled = False
    baseline = STAGE.run(bundle, cfg, None)
    baseline_poses = bundle.read_json(baseline.outputs["poses"])
    with np.load(bundle.path(baseline.outputs["sparse"]), allow_pickle=False) as data:
        baseline_sparse = {key: data[key].copy() for key in data.files}
    cfg.bundle_adjustment.enabled = True
    cfg.bundle_adjustment.max_nfev = 1
    failed = STAGE.run(bundle, cfg, None)
    final = bundle.read_json(failed.outputs["poses"])
    adjustment = final["bundle_adjustment"]
    assert adjustment["status"] == "failed_retained_input"
    assert adjustment["optimizer_invoked"] and adjustment["poses_retained"]
    assert adjustment["points_retained"] and "converge" in failed.degraded_reason
    assert final["cameras"] == baseline_poses["cameras"]
    assert final["pose_revision"] == baseline_poses["pose_revision"] == "s6_global_visual_GNSS_refined"
    with np.load(bundle.path(failed.outputs["sparse"]), allow_pickle=False) as data:
        for key, original in baseline_sparse.items():
            np.testing.assert_array_equal(data[key], original)


def test_new_adjusted_revision_cannot_use_legacy_s2_sparse(tmp_path):
    bundle, _, poses, _ = scene_bundle(tmp_path)
    final = copy.deepcopy(poses)
    final.pop("sparse_path")
    final["pose_revision"] = "s6_joint_reprojection_GNSS_bundle_adjusted"
    with pytest.raises(InputError, match="matching sparse"):
        pose_sparse_source(bundle, final)


@pytest.mark.parametrize("field,value", [
    ("enabled", 1), ("max_points", 2), ("max_observations", 8), ("min_track_length", 1),
    ("max_nfev", 0), ("reprojection_sigma_px", 0.), ("min_depth_m", float("nan")),
    ("ftol", 1e-20), ("xtol", 0.), ("gtol", 1.),
])
def test_adjustment_config_rejects_invalid_solver_parameters(field, value):
    with pytest.raises(ValidationError):
        BundleAdjustmentConfig.model_validate({field: value})


def test_saved_legacy_config_gains_defaults_and_s6_freshness_covers_adjustment(tmp_path):
    cfg = load_config(config_dir=ROOT / "configs", environ={})
    assert cfg.bundle_adjustment == BundleAdjustmentConfig()
    legacy = cfg.model_dump(mode="python")
    legacy.pop("bundle_adjustment")
    assert Config.model_validate(legacy).bundle_adjustment == BundleAdjustmentConfig()
    for name in ("fast", "balanced", "max"):
        assert load_config(profile=name, config_dir=ROOT / "configs", environ={}).bundle_adjustment.enabled
    s6 = next(stage for stage in STAGES if stage.name == "s6_global")
    assert s6.config_sections == STAGE.config_sections
    bundle, _, _, _ = scene_bundle(tmp_path)
    cfg.dataset = DatasetDescriptor(
        name="math-contract", video=tmp_path / "unused-video.unit",
        telemetry=TelemetryConfig(format="csv", path=tmp_path / "unused-gps.unit",
                                  altitude_datum="unknown", time_origin_s=0.),
    )
    cfg.dataset.video.write_bytes(b"unit source hash specimen, not footage")
    cfg.dataset.telemetry.path.write_bytes(b"unit source hash specimen, not telemetry")
    runner = StageRunner(bundle, cfg, Runtime("cpu", {"versions": {}}))
    assert "bundle_adjustment" in runner.params(s6)
    first = runner.params(s6)
    fingerprint = runner.fingerprint(s6)
    cfg.bundle_adjustment.reprojection_sigma_px = 2.
    assert runner.params(s6) != first
    assert runner.fingerprint(s6) != fingerprint
