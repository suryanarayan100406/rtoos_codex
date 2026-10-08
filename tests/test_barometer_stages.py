"""Constructed stage/provenance contracts, not recorded-flight or accuracy acceptance."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from drishti.bundle import Bundle, content_hash
from drishti.config import Config
from drishti.stages import s2_poses, s6_global

MODEL = "relative_height_with_constant_unknown_bias"


def observation(index):
    return {"source_time_s": 1000. + index, "time_s": float(index),
            "height_m": 50. + .03 * index, "sigma_m": .4,
            "sigma_source": "csv", "source_index": index, "source_line": index + 2,
            "time_error_s": -.01, "frame_time_s": index + .01}


def settings(tmp_path, enabled=True):
    return Config.model_validate({
        "dataset": {"name": "barometer-stage-contract", "video": tmp_path / "unused.unit",
                    "telemetry": {"format": "csv", "path": tmp_path / "unused-gps.unit",
                                  "altitude_datum": "ellipsoidal", "time_origin_s": 0.}},
        "barometer": {"enabled": enabled, "time_origin_s": 1000. if enabled else None},
    })


def metadata(method, status="optimized", used=False, *, factors=None):
    accepted = status == "optimized"
    factors = (4 if used else 0) if factors is None else factors
    barometer = {
        "status": "optimized" if accepted and used else "not_used",
        "reason": None if accepted and used else "Constructed optional support not accepted",
        "factors_added": factors, "optimizer_invoked": factors > 0,
        "model": MODEL, "absolute_datum": "not_interpreted",
        "selected_source_indices": list(range(factors)), "excluded": [],
        "uncertainty_sources": ["csv"] if factors else [],
        "bias_m": 50. if accepted and used else None,
        "fit_rmse_m": .02 if accepted and used else None,
        "fit_is_independent_accuracy": False,
    }
    result = {"method": method, "status": status, "optimizer_invoked": True,
              "baro_used": used, "barometer": barometer,
              "reason": None if accepted else "Constructed solver retained input"}
    if method == "contract_joint_BA":
        result.update(optimized_points=12, retained_points=0,
                      point_confidence_method="constructed_observed_support")
    return result


def input_ledger():
    return {"baro": {"provided": True, "available": True, "parsed": True,
                     "aligned_frames": 4, "used": False, "status": "aligned_not_fused",
                     "reason": None},
            "imu": {"provided": False, "available": False, "used": False}}


def stage_bundle(tmp_path):
    return Bundle(tmp_path, {"bundle_version": "1", "stages": {}, "inputs": [],
                             "config_snapshot": {}, "crs": {
                                 "epsg": 32643, "vertical": "ellipsoidal",
                                 "source_altitude_datum": "ellipsoidal"}})


def sparse_scene():
    points = np.array([[x, y, 8. + (x + y) / 4] for x in (-1., 0., 1., 2.)
                       for y in (-.7, .1, .8)], dtype=np.float64)
    cameras = []
    for index in range(4):
        center = np.array([float(index), .15 * (index % 2), .03 * index])
        pixels = (points - center)[:, :2] / (points - center)[:, 2, None] * 160. + [120., 100.]
        cameras.append({
            "frame_id": f"math{index}", "image_path": f"unused{index}.png",
            "time_s": index + .01, "width": 240, "height": 200,
            "K": [[160., 0., 120.], [0., 160., 100.], [0., 0., 1.]],
            "distortion": [], "projection_domain": "undistorted_pinhole",
            "R_cw": np.eye(3).tolist(), "center": center.tolist(),
            "visual_reference": {"R_cw": np.eye(3).tolist(), "center": center.tolist()},
            "gps": (center + [.12, .02, .03]).tolist(), "gps_inlier": True,
            "gps_sample": {"sigma_h_m": .2, "sigma_v_m": .3},
            "observations": [{"point_index": i, "uv": uv.tolist()} for i, uv in enumerate(pixels)],
            "barometer": observation(index), "barometer_reason": None, "anchors": [],
        })
    sparse = {"points": points + [.015, -.02, .04],
              "colors": np.full((len(points), 3), 128, dtype=np.uint8),
              "confidence": np.full(len(points), .6), "inferred": np.zeros(len(points), bool),
              "reprojection_error_px": np.full(len(points), 1.),
              "track_length": np.full(len(points), len(cameras), dtype=np.int64)}
    return cameras, sparse


def s2_fixture(tmp_path, monkeypatch):
    bundle = stage_bundle(tmp_path)
    cameras, sparse = sparse_scene()
    frames = [{"id": camera["frame_id"], "path": camera["image_path"],
               "time_s": camera["time_s"], "gps": camera["gps_sample"],
               "barometer": camera["barometer"], "barometer_reason": None} for camera in cameras]
    # Camera 1 is unregistered; extraction order differs from the QA frame list.
    frames[1]["barometer"] = None
    frames[1]["barometer_reason"] = "no_nearest_sample"
    extracted = [copy.deepcopy(cameras[index]) for index in (3, 0, 2)]
    for camera in extracted:
        camera.pop("barometer")
        camera.pop("barometer_reason")
    bundle.write_json("s1_frameqa/keyframes.json", {"frames": frames, "capabilities": input_ledger()})
    bundle.manifest["stages"]["s1_frameqa"] = {"outputs": {"keyframes": "s1_frameqa/keyframes.json"}}
    monkeypatch.setattr(s2_poses, "run_sfm", lambda *args, **kwargs:
                        (None, tmp_path / "undistorted", {"intrinsics": "self_calibrated"}))
    monkeypatch.setattr(s2_poses, "extract_sparse", lambda *args: (extracted, sparse))
    monkeypatch.setattr(s2_poses, "project_gps", lambda *args, **kwargs:
                        (np.array([c["gps"] for c in extracted]), np.array([500000., 3000000., 100.])))
    transform = SimpleNamespace(rotation=np.eye(3), apply=lambda values: np.asarray(values, dtype=np.float64))
    monkeypatch.setattr(s2_poses, "robust_similarity", lambda *args, **kwargs:
                        (transform, {"warnings": [], "inlier_mask": [True] * 3}))
    return bundle, frames


def test_s2_transfers_only_registered_frames_by_id_and_preserves_recorded_provenance(tmp_path, monkeypatch):
    bundle, frames = s2_fixture(tmp_path, monkeypatch)
    cfg = settings(tmp_path)
    captured = []
    expected = {frame["id"]: frame for frame in frames}

    def graph(cameras, spine, *, barometer_settings):
        assert barometer_settings is cfg.barometer and spine is cfg.spine
        assert [c["frame_id"] for c in cameras] == ["math3", "math0", "math2"]
        for camera in cameras:
            assert camera["barometer"] == expected[camera["frame_id"]]["barometer"]
            assert camera["time_s"] == expected[camera["frame_id"]]["time_s"]
        captured.extend(copy.deepcopy(cameras))
        return copy.deepcopy(cameras), metadata("contract_graph", used=True, factors=3)

    monkeypatch.setattr(s2_poses, "optimize_pose_graph", graph)
    result = s2_poses.STAGE.run(bundle, cfg, None)
    poses = bundle.read_json(result.outputs["poses"])
    spine = bundle.read_json(result.outputs["spine"])
    assert result.metrics["unregistered_frame_ids"] == ["math1"]
    assert poses["pose_revision"] == "s2_metric_spine"
    fusion = poses["barometer_fusion"]
    assert fusion == result.metrics["barometer_fusion"] == spine["barometer_fusion"]
    assert fusion["used"] and fusion["stage"] == "s2_poses"
    assert fusion["final_source"] == "pose_graph" and fusion["barometer"]["model"] == MODEL
    assert fusion["attempts"]["bundle_adjustment"] is None
    ledger = poses["optional_inputs"]["barometer"]
    assert ledger["used"] and ledger["available"] and ledger["parsed"]
    assert ledger["aligned_frames"] == 4 and ledger["input_status"] == "aligned_not_fused"
    assert poses["optional_inputs"]["baro"] == ledger
    assert result.metrics["optional_inputs"] == spine["optional_inputs"] == poses["optional_inputs"]
    for camera in poses["cameras"]:
        assert camera["barometer"] == expected[camera["frame_id"]]["barometer"]
    captured[0]["barometer"]["height_m"] = -900.
    assert frames[3]["barometer"]["height_m"] == observation(3)["height_m"]
    assert bundle.read_json("s1_frameqa/keyframes.json")["frames"] == frames


def test_s2_missing_optional_intrinsics_falls_back_to_self_calibration(tmp_path, monkeypatch):
    bundle, _ = s2_fixture(tmp_path, monkeypatch)
    cfg = settings(tmp_path)
    cfg.dataset.optional.intrinsics = tmp_path / "missing-intrinsics.json"
    calls = []

    def sfm(*args, **kwargs):
        calls.append(kwargs.get("intrinsics_path"))
        return None, tmp_path / "undistorted", {"intrinsics": "self_calibrated"}

    monkeypatch.setattr(s2_poses, "run_sfm", sfm)
    result = s2_poses.STAGE.run(bundle, cfg, None)
    poses = bundle.read_json(result.outputs["poses"])
    assert calls == [None]
    assert poses["optional_inputs"]["intrinsics"]["status"] == "provided_file_missing"
    assert not poses["optional_inputs"]["intrinsics"]["used"]
    assert "missing" in (result.degraded_reason or "")


@pytest.mark.parametrize("status,claimed,factors", [
    ("optimized", False, 0), ("failed_retained_input", False, 3),
    ("failed_retained_input", True, 3), ("optimized", True, 0),
])
def test_s2_input_alignment_or_rejected_factors_never_claim_fusion(tmp_path, monkeypatch, status, claimed, factors):
    bundle, _ = s2_fixture(tmp_path, monkeypatch)
    optimizer = metadata("contract_graph", status, claimed, factors=factors)
    monkeypatch.setattr(s2_poses, "optimize_pose_graph", lambda cameras, spine, **kwargs:
                        (copy.deepcopy(cameras), optimizer))
    result = s2_poses.STAGE.run(bundle, settings(tmp_path), None)
    poses = bundle.read_json(result.outputs["poses"])
    assert not poses["barometer_fusion"]["used"]
    assert poses["barometer_fusion"]["barometer"] == optimizer["barometer"]
    assert poses["barometer_fusion"]["reason"]
    assert not poses["optional_inputs"]["barometer"]["used"]
    assert poses["optional_inputs"]["barometer"]["available"]


@pytest.mark.parametrize("legacy_config", [False, True])
def test_s2_disabled_legacy_callers_require_no_sensor_or_keyword(tmp_path, monkeypatch, legacy_config):
    bundle, _ = s2_fixture(tmp_path, monkeypatch)
    keyframes = bundle.read_json("s1_frameqa/keyframes.json")
    keyframes.pop("capabilities")
    for frame in keyframes["frames"]:
        frame.pop("barometer")
        frame.pop("barometer_reason")
    bundle.write_json("s1_frameqa/keyframes.json", keyframes)
    cfg = settings(tmp_path, enabled=False)
    if legacy_config:
        cfg = SimpleNamespace(dataset=cfg.dataset, poses=cfg.poses, spine=cfg.spine, crs=cfg.crs, run=cfg.run)

    def graph(cameras, spine):
        assert all("barometer" not in c for c in cameras)
        return cameras, {"status": "optimized", "method": "legacy_graph"}

    monkeypatch.setattr(s2_poses, "optimize_pose_graph", graph)
    result = s2_poses.STAGE.run(bundle, cfg, None)
    assert result.metrics["barometer_fusion"]["status"] == "disabled"
    assert result.metrics["barometer_fusion"]["barometer"] is None
    assert not result.metrics["optional_inputs"]["barometer"]["used"]


def s6_fixture(tmp_path, inherited=True):
    bundle = stage_bundle(tmp_path)
    cameras, sparse = sparse_scene()
    optional = input_ledger()
    poses = {"frame": "projected_local", "origin": [500000., 3000000., 100.],
             "crs": bundle.manifest["crs"], "cameras": cameras,
             "pose_revision": "s2_metric_spine", "sparse_path": "s2_poses/sparse.npz",
             "optional_inputs": optional}
    if inherited:
        prior = metadata("contract_s2_graph", used=True)
        poses["barometer_fusion"] = {
            "stage": "s2_poses", "method": prior["method"], "status": "used", "used": True,
            "reason": None, "final_source": "pose_graph", "barometer": prior["barometer"],
            "attempts": {"pose_graph": prior, "bundle_adjustment": None},
        }
        optional["baro"]["used"] = True
    bundle.write_json("s2_poses/poses.json", poses)
    np.savez_compressed(bundle.root / "s2_poses/sparse.npz", **sparse)
    bundle.manifest["stages"]["s2_poses"] = {
        "outputs": {"poses": "s2_poses/poses.json", "sparse": "s2_poses/sparse.npz"}}
    return bundle, poses, sparse


def assert_final_anchors(bundle, result, poses):
    assert poses["sparse_path"] == result.outputs["sparse"] == "s6_global/sparse.npz"
    with np.load(bundle.path(poses["sparse_path"]), allow_pickle=False) as data:
        points = data["points"]
        assert points.dtype == np.float64
    for camera in poses["cameras"]:
        for feature, anchor in zip(camera["observations"], camera["anchors"], strict=True):
            xyz = np.asarray(camera["R_cw"]) @ (points[feature["point_index"]] - camera["center"])
            projected = np.asarray(camera["K"]) @ xyz
            error = np.linalg.norm(projected[:2] / projected[2] - feature["uv"])
            np.testing.assert_allclose(anchor, [*feature["uv"], xyz[2], error])


@pytest.mark.parametrize("graph_status,graph_used,ba_status,ba_used,source,used", [
    ("optimized", True, "optimized", True, "bundle_adjustment", True),
    ("optimized", True, "optimized", False, "bundle_adjustment", False),
    ("optimized", False, "optimized", True, "bundle_adjustment", True),
    ("failed_retained_input", False, "optimized", True, "bundle_adjustment", True),
    ("optimized", True, "failed_retained_input", False, "pose_graph", True),
    ("optimized", True, "disabled_retained_input", False, "pose_graph", True),
    ("optimized", False, "failed_retained_input", False, "pose_graph", False),
    ("failed_retained_input", False, "failed_retained_input", False, "retained_s2", True),
])
def test_s6_final_use_selects_accepted_geometry_and_keeps_attempts(
    tmp_path, monkeypatch, graph_status, graph_used, ba_status, ba_used, source, used,
):
    bundle, original, source_sparse = s6_fixture(tmp_path)
    cfg = settings(tmp_path)
    graph_meta = metadata("contract_graph", graph_status, graph_used)
    ba_meta = metadata("contract_joint_BA", ba_status, ba_used)
    source_by_id = {c["frame_id"]: c for c in original["cameras"]}

    def graph(cameras, spine, *, global_refinement, barometer_settings):
        assert global_refinement and spine is cfg.spine and barometer_settings is cfg.barometer
        output = copy.deepcopy(cameras)
        for camera in output:
            if graph_status == "optimized":
                camera["center"][2] += .03
            # A legacy camera rebuilder can drop arbitrary sensor fields.
            camera.pop("barometer")
            camera.pop("barometer_reason")
        return output, graph_meta

    def adjust(cameras, sparse, ba, spine, *, barometer_settings):
        assert ba is cfg.bundle_adjustment and spine is cfg.spine and barometer_settings is cfg.barometer
        for camera in cameras:
            assert camera["barometer"] == source_by_id[camera["frame_id"]]["barometer"]
        output, points = copy.deepcopy(cameras), copy.deepcopy(sparse)
        if ba_status == "optimized":
            for camera in output:
                camera["center"][2] += .07
            points["points"][:, 2] += .02
        for camera in output:
            camera.pop("barometer")
            camera.pop("barometer_reason")
        return output, points, ba_meta

    monkeypatch.setattr(s6_global, "optimize_pose_graph", graph)
    monkeypatch.setattr(s6_global, "optimize_bundle", adjust)
    result = s6_global.STAGE.run(bundle, cfg, None)
    poses = bundle.read_json(result.outputs["poses"])
    optimization = bundle.read_json(result.outputs["optimization"])
    fusion = poses["barometer_fusion"]
    assert fusion == optimization["barometer_fusion"] == result.metrics["barometer_fusion"]
    assert fusion["stage"] == "s6_global" and fusion["final_source"] == source
    assert fusion["used"] is used
    assert fusion["attempts"]["pose_graph"]["used"] is graph_used
    assert fusion["attempts"]["bundle_adjustment"]["used"] is ba_used
    assert fusion["inherited"] == original["barometer_fusion"]
    selected = {"bundle_adjustment": ba_meta["barometer"], "pose_graph": graph_meta["barometer"],
                "retained_s2": original["barometer_fusion"]["barometer"]}[source]
    assert fusion["barometer"] == selected
    assert not fusion["barometer"]["fit_is_independent_accuracy"]
    assert fusion["barometer"]["absolute_datum"] == "not_interpreted"
    ledger = poses["optional_inputs"]["barometer"]
    assert ledger["used"] is used and ledger["final_source"] == source
    assert ledger["available"] and ledger["parsed"] and ledger["aligned_frames"] == 4
    assert ledger["input_status"] == "aligned_not_fused"
    assert poses["optional_inputs"]["baro"] == ledger
    assert optimization["optional_inputs"] == result.metrics["optional_inputs"] == poses["optional_inputs"]
    assert content_hash(fusion)  # Report/manifest JSON is finite and serializable.
    for camera in poses["cameras"]:
        original_camera = source_by_id[camera["frame_id"]]
        assert camera["barometer"] == original_camera["barometer"]
        assert camera["barometer_reason"] == original_camera["barometer_reason"]
        assert camera["time_s"] == original_camera["time_s"]
    assert_final_anchors(bundle, result, poses)
    assert bundle.read_json("s2_poses/poses.json") == original
    with np.load(bundle.artifact("s2_poses", "sparse"), allow_pickle=False) as data:
        np.testing.assert_array_equal(data["points"], source_sparse["points"])


def test_s6_no_prior_solver_provenance_cannot_infer_use_from_aligned_input(tmp_path, monkeypatch):
    bundle, original, _ = s6_fixture(tmp_path, inherited=False)
    original["optional_inputs"]["baro"]["used"] = True  # Unsupported old ledger claim.
    bundle.write_json("s2_poses/poses.json", original)
    monkeypatch.setattr(s6_global, "optimize_pose_graph", lambda cameras, spine, **kwargs:
                        (cameras, metadata("contract_graph", "unavailable_retained_input")))
    monkeypatch.setattr(s6_global, "optimize_bundle", lambda cameras, sparse, ba, spine, **kwargs:
                        (cameras, sparse, metadata("contract_joint_BA", "unsupported_retained_input")))
    result = s6_global.STAGE.run(bundle, settings(tmp_path), None)
    fusion = result.metrics["barometer_fusion"]
    assert fusion["final_source"] == "retained_s2" and not fusion["used"]
    assert fusion["barometer"] is None and "no recorded" in fusion["reason"]
    assert "inherited" not in fusion
    assert not result.metrics["optional_inputs"]["barometer"]["used"]
    assert not result.metrics["optional_inputs"]["baro"]["used"]


def test_s6_disabled_legacy_signatures_and_camera_schema_remain_compatible(tmp_path, monkeypatch):
    bundle, original, _ = s6_fixture(tmp_path, inherited=False)
    original.pop("optional_inputs")
    for camera in original["cameras"]:
        camera.pop("barometer")
        camera.pop("barometer_reason")
    bundle.write_json("s2_poses/poses.json", original)

    def graph(cameras, spine, *, global_refinement=False):
        assert global_refinement
        return cameras, {"status": "optimized", "method": "legacy_graph"}

    def adjust(cameras, sparse, ba, spine):
        return cameras, sparse, {"status": "disabled_retained_input", "reason": "disabled"}

    monkeypatch.setattr(s6_global, "optimize_pose_graph", graph)
    monkeypatch.setattr(s6_global, "optimize_bundle", adjust)
    cfg = settings(tmp_path, enabled=False)
    cfg.bundle_adjustment.enabled = False
    legacy_cfg = SimpleNamespace(spine=cfg.spine, bundle_adjustment=cfg.bundle_adjustment)
    result = s6_global.STAGE.run(bundle, legacy_cfg, None)
    poses = bundle.read_json(result.outputs["poses"])
    assert poses["pose_revision"] == "s6_global_visual_GNSS_refined"
    assert poses["barometer_fusion"]["status"] == "disabled"
    assert poses["barometer_fusion"]["barometer"] is None
    assert all("barometer" not in c for c in poses["cameras"])
    assert result.degraded_reason is None
    assert_final_anchors(bundle, result, poses)


def test_s6_disabling_new_fusion_retains_real_s2_factor_provenance(tmp_path):
    bundle, original, _ = s6_fixture(tmp_path)
    cfg = settings(tmp_path, enabled=False)
    cfg.spine.optimize = False
    cfg.bundle_adjustment.enabled = False
    result = s6_global.STAGE.run(bundle, cfg, None)
    poses = bundle.read_json(result.outputs["poses"])
    fusion = poses["barometer_fusion"]
    assert fusion["used"] and fusion["status"] == "used"
    assert fusion["final_source"] == "retained_s2"
    assert fusion["barometer"] == original["barometer_fusion"]["barometer"]
    assert not fusion["attempts"]["pose_graph"]["used"]
    assert not fusion["attempts"]["bundle_adjustment"]["used"]
    assert poses["optional_inputs"]["barometer"]["used"]
    assert_final_anchors(bundle, result, poses)


@pytest.mark.parametrize("kind", ["absent", "malformed", "reused_single_sample"])
def test_s6_real_solves_keep_baseline_with_unsupported_optional_sensor(tmp_path, kind):
    pytest.importorskip("gtsam")
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    bundle, original, _ = s6_fixture(tmp_path, inherited=False)
    for camera in original["cameras"]:
        if kind == "absent":
            camera["barometer"] = None
            camera["barometer_reason"] = "optional_sensor_not_provided"
        elif kind == "malformed":
            camera["barometer"]["sigma_m"] = 0.
        else:
            camera["barometer"]["source_index"] = 0
    if kind == "absent":
        original["optional_inputs"]["baro"].update(provided=False, available=False, parsed=False,
                                                  aligned_frames=0, status="not_provided")
    bundle.write_json("s2_poses/poses.json", original)
    result = s6_global.STAGE.run(bundle, settings(tmp_path), None)
    poses = bundle.read_json(result.outputs["poses"])
    fusion = poses["barometer_fusion"]
    assert poses["global_optimization"]["status"] == "optimized"
    assert poses["bundle_adjustment"]["status"] == "optimized"
    assert fusion["final_source"] == "bundle_adjustment" and not fusion["used"]
    assert fusion["reason"] and fusion["barometer"]["factors_added"] == 0
    assert fusion["barometer"]["bias_m"] is None and fusion["barometer"]["fit_rmse_m"] is None
    assert not poses["optional_inputs"]["barometer"]["used"]
    assert result.degraded_reason is None
    for camera, source in zip(poses["cameras"], original["cameras"], strict=True):
        assert camera["barometer"] == source["barometer"]
    assert_final_anchors(bundle, result, poses)


def test_s6_real_barometer_factors_publish_accepted_ba_provenance_and_final_anchors(tmp_path):
    pytest.importorskip("gtsam")
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    bundle, original, _ = s6_fixture(tmp_path, inherited=False)
    # Nonuniform initial pose error forces actual joint refinement, rather than
    # the already-consistent scene's uniform graph translation remaining optimal.
    for index, camera in enumerate(original["cameras"]):
        camera["center"] = (np.asarray(camera["center"]) + [.01 * index, -.015 * index, .005 * index]).tolist()
        camera["visual_reference"]["center"] = camera["center"][:]
    bundle.write_json("s2_poses/poses.json", original)
    cfg = settings(tmp_path)
    result = s6_global.STAGE.run(bundle, cfg, None)
    poses = bundle.read_json(result.outputs["poses"])
    fusion = poses["barometer_fusion"]
    assert poses["bundle_adjustment"]["status"] == "optimized", poses["bundle_adjustment"]
    assert fusion["used"] and fusion["final_source"] == "bundle_adjustment"
    assert fusion["attempts"]["pose_graph"]["used"]
    assert fusion["barometer"]["factors_added"] == 4
    assert fusion["barometer"]["optimizer_invoked"]
    assert fusion["barometer"]["model"] == MODEL
    assert np.isfinite(fusion["barometer"]["bias_m"])
    assert np.isfinite(fusion["barometer"]["fit_rmse_m"])
    assert not fusion["barometer"]["fit_is_independent_accuracy"]
    assert result.metrics["optional_inputs"]["barometer"]["used"]
    assert poses["pose_revision"] == "s6_joint_reprojection_GNSS_bundle_adjusted"
    assert poses["bundle_adjustment"]["poses_changed"]
    for camera, source in zip(poses["cameras"], original["cameras"], strict=True):
        assert camera["barometer"] == source["barometer"]
        assert "pose_covariance_tangent" not in camera
    assert_final_anchors(bundle, result, poses)
