"""Constructed BA geometry tests; these are not mission-acceptance tests."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from drishti.recon.bundle_adjustment import optimize_bundle

pytest.importorskip("scipy")


def _settings(**overrides):
    values = {
        "enabled": True,
        "reprojection_sigma_px": 1.0,
        "min_track_length": 3,
        "max_points": 10000,
        "max_observations": 100000,
        "max_nfev": 100,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "gtol": 1e-10,
        "min_depth_m": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _spine(**overrides):
    values = {
        "optimize": True,
        "gnss_sigma_h_m": 0.15,
        "gnss_sigma_v_m": 0.2,
        "rotation_sigma_rad": 0.05,
        "robust_delta": 1.345,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _geometry(*, gps_sigma=0.15, distorted=False, inferred=False):
    K = np.array([[120.0, 0.0, 100.0], [0.0, 120.0, 80.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, -0.1, 0.0]], dtype=np.float64)
    points = np.array([
        [-1.0, -0.5, 8.0], [0.5, -0.3, 9.0], [1.5, 0.1, 10.0], [2.5, 0.6, 11.0],
        [-0.4, 0.8, 12.0], [1.0, 1.0, 13.0],
    ], dtype=np.float64)
    cameras = []
    for index, center in enumerate(centers):
        observations = []
        for point_index, point in enumerate(points):
            camera_point = point - center
            observations.append({
                "point_index": point_index,
                "uv": [float(K[0, 0] * camera_point[0] / camera_point[2] + K[0, 2]),
                       float(K[1, 1] * camera_point[1] / camera_point[2] + K[1, 2])],
            })
        cameras.append({
            "frame_id": f"f{index}", "width": 200, "height": 160, "K": K.tolist(),
            "distortion": [0.1] if distorted else [],
            "projection_domain": "distorted" if distorted else "undistorted_pinhole",
            "R_cw": np.eye(3).tolist(), "center": center.tolist(),
            "visual_reference": {"R_cw": np.eye(3).tolist(), "center": center.tolist()},
            "gps": (center + np.array([0.25, 0.05, 0.08])).tolist(), "gps_inlier": True,
            "gps_sample": {"sigma_h_m": gps_sigma, "sigma_v_m": gps_sigma},
            "observations": observations,
        })
    sparse = {
        "points": points.copy(), "colors": np.zeros((len(points), 3), dtype=np.uint8),
        "confidence": np.full(len(points), 0.8, dtype=np.float64),
        "inferred": np.full(len(points), inferred, dtype=bool),
        "reprojection_error_px": np.full(len(points), 0.5, dtype=np.float64),
        "track_length": np.full(len(points), len(cameras), dtype=np.int64),
    }
    return cameras, sparse


def _project(camera, point):
    K = np.asarray(camera["K"], dtype=np.float64)
    camera_point = np.asarray(camera["R_cw"], dtype=np.float64) @ (
        np.asarray(point, dtype=np.float64) - np.asarray(camera["center"], dtype=np.float64))
    return np.array([K[0, 0] * camera_point[0] / camera_point[2] + K[0, 2],
                     K[1, 1] * camera_point[1] / camera_point[2] + K[1, 2]])


def _sequence_geometry():
    """Nine constructed views and local tracks across four overlapping windows."""
    templates, base_sparse = _geometry()
    cameras = []
    for index in range(9):
        camera = copy.deepcopy(templates[0])
        camera["frame_id"] = f"sequence_{index}"
        camera["center"] = [float(index), 0.1 * (index % 2), 0.0]
        camera["gps"] = copy.deepcopy(camera["center"])
        camera["visual_reference"]["center"] = copy.deepcopy(camera["center"])
        camera["observations"] = []
        cameras.append(camera)
    points = []
    window_points = np.vstack((base_sparse["points"], base_sparse["points"][:2] + [0.01, 0.02, 0.03]))
    for start in (0, 2, 4, 6):
        for point in window_points + [float(start), 0.0, 0.0]:
            point_index = len(points)
            points.append(point)
            for camera in cameras[start:start + 3]:
                camera["observations"].append({"point_index": point_index,
                                               "uv": _project(camera, point).tolist()})
    count = len(points)
    sparse = {"points": np.asarray(points, dtype=np.float64),
              "colors": np.zeros((count, 3), dtype=np.uint8),
              "confidence": np.full(count, 0.8), "inferred": np.zeros(count, dtype=bool),
              "track_length": np.full(count, 3, dtype=np.int64)}
    return cameras, sparse


def _assert_retained(cameras, sparse, output_cameras, output_sparse):
    # A serialized camera contract and every array's raw bytes survive fallback.
    assert json.dumps(output_cameras, sort_keys=True) == json.dumps(cameras, sort_keys=True)
    assert output_sparse.keys() == sparse.keys()
    for key, original in sparse.items():
        result = output_sparse[key]
        assert result.dtype == original.dtype and result.shape == original.shape
        assert result.tobytes() == original.tobytes()
        assert not np.shares_memory(result, original)


def test_joint_optimizer_reduces_objective_and_moves_cameras_and_points():
    cameras, sparse = _geometry()
    input_cameras = copy.deepcopy(cameras)
    input_sparse = {key: value.copy() for key, value in sparse.items()}
    result_cameras, result_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())

    assert metadata["status"] == "optimized"
    assert metadata["selection_order_point_indices"] == list(range(6))
    assert metadata["optimizer_invoked"]
    assert metadata["final_robust_cost"] < metadata["initial_robust_cost"]
    assert metadata["reprojection_rmse_px"] < 1e-5
    assert metadata["gnss_fit_rmse_m"] < metadata["initial_gnss_fit_rmse_m"]
    assert any(not np.allclose(result["center"], original["center"])
               for result, original in zip(result_cameras, input_cameras, strict=True))
    assert not np.array_equal(result_sparse["points"], input_sparse["points"])
    assert metadata["calibrated_intrinsics_fixed"]
    assert metadata["loss"] == "huber"
    assert metadata["jacobian"] == "sparse_jacobian_pattern"
    assert metadata["fit_is_independent_accuracy"] is False
    assert metadata["imu_used"] is False and metadata["baro_used"] is False
    assert metadata["covariance_status"] == "unavailable"
    assert cameras == input_cameras
    np.testing.assert_array_equal(sparse["points"], input_sparse["points"])


def test_world_to_camera_transformation_and_confidence_use_final_observations():
    cameras, sparse = _geometry()
    for camera in cameras:
        camera["pose_covariance_tangent"] = np.eye(6).tolist()
        camera["covariance_domain"] = "pre_BA_pose_graph"
    result_cameras, result_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())

    assert metadata["status"] == "optimized"
    for camera in result_cameras:
        for point_index, observation in enumerate(camera["observations"]):
            np.testing.assert_allclose(_project(camera, result_sparse["points"][point_index]), observation["uv"], atol=1e-4)
    assert result_sparse["points"].dtype == np.float64
    assert np.all(result_sparse["confidence"] > 0.0)
    assert all("pose_covariance_tangent" not in camera for camera in result_cameras)
    assert all("covariance_domain" not in camera for camera in result_cameras)
    for camera, original in zip(result_cameras, cameras, strict=True):
        assert camera["visual_reference"] == original["visual_reference"]
        assert camera["K"] == original["K"]
    for point_index in range(len(sparse["points"])):
        errors = [_project(camera, result_sparse["points"][point_index]) - camera["observations"][point_index]["uv"]
                  for camera in result_cameras]
        rmse = np.sqrt(np.mean(np.sum(np.square(errors), axis=1)))
        assert result_sparse["confidence"][point_index] == pytest.approx((1 - 1 / 3) / (1 + rmse))


def test_logged_gnss_sigmas_are_consumed():
    small_cameras, sparse = _geometry(gps_sigma=0.03)
    _, _, small_metadata = optimize_bundle(small_cameras, sparse, _settings(), _spine())

    assert small_metadata["status"] == "optimized"
    assert all(item["horizontal_source"] == "gps_sample" for item in small_metadata["gnss_sigma_used"])
    assert small_metadata["gnss_sigma_used"][0]["horizontal_m"] == 0.03
    spine_cameras, spine_sparse = _geometry()
    for camera in spine_cameras:
        camera["gps_sample"] = {}
    _, _, spine_metadata = optimize_bundle(spine_cameras, spine_sparse, _settings(), _spine())
    assert spine_metadata["status"] == "optimized"
    assert all(item["horizontal_source"] == "spine.gnss_sigma_h_m"
               for item in spine_metadata["gnss_sigma_used"])


def test_inferred_tracks_are_excluded_without_false_claim():
    cameras, sparse = _geometry(inferred=True)
    original = {key: value.copy() for key, value in sparse.items()}
    result_cameras, result_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())

    assert metadata["status"] == "unsupported_retained_input"
    assert not metadata["optimizer_invoked"]
    assert "inferred" in metadata["reason"] or "tracks" in metadata["reason"]
    assert result_cameras == cameras
    np.testing.assert_array_equal(result_sparse["points"], original["points"])
    np.testing.assert_array_equal(result_sparse["confidence"], original["confidence"])


def test_disabled_dependency_and_distorted_domain_retain_input(monkeypatch):
    cameras, sparse = _geometry()
    _, _, disabled = optimize_bundle(cameras, sparse, _settings(enabled=False), _spine())
    assert disabled["status"] == "disabled_retained_input"
    assert not disabled["optimizer_invoked"]
    _, _, spine_disabled = optimize_bundle(cameras, sparse, _settings(), _spine(optimize=False))
    assert spine_disabled["status"] == "disabled_retained_input"
    assert not spine_disabled["optimizer_invoked"]

    for module_name in ("scipy", "scipy.optimize", "scipy.sparse", "scipy.spatial",
                        "scipy.spatial.transform"):
        monkeypatch.setitem(sys.modules, module_name, None)
    _, _, unavailable = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert unavailable["status"] == "unavailable_retained_input"
    assert not unavailable["optimizer_invoked"]

    distorted_cameras, distorted_sparse = _geometry(distorted=True)
    # Restore SciPy to inspect the domain guard independently.
    monkeypatch.undo()
    result_cameras, result_sparse, unsupported = optimize_bundle(
        distorted_cameras, distorted_sparse, _settings(), _spine())
    assert unsupported["status"] == "unsupported_retained_input"
    assert result_cameras == distorted_cameras
    np.testing.assert_array_equal(result_sparse["points"], distorted_sparse["points"])


def test_nonconvergence_and_invalid_index_retain_exact_values():
    cameras, sparse = _geometry()
    original_cameras = copy.deepcopy(cameras)
    original_sparse = {key: value.copy() for key, value in sparse.items()}
    result_cameras, result_sparse, metadata = optimize_bundle(
        cameras, sparse, _settings(max_nfev=1), _spine())
    assert metadata["status"] == "failed_retained_input"
    assert metadata["optimizer_invoked"]
    assert result_cameras == original_cameras
    for key in sparse:
        np.testing.assert_array_equal(result_sparse[key], original_sparse[key])
    _assert_retained(cameras, sparse, result_cameras, result_sparse)

    invalid_cameras, invalid_sparse = _geometry()
    invalid_cameras[0]["observations"][0]["point_index"] = 999
    result_cameras, result_sparse, invalid = optimize_bundle(
        invalid_cameras, invalid_sparse, _settings(), _spine())
    assert invalid["status"] == "unsupported_retained_input"
    assert result_cameras == invalid_cameras
    np.testing.assert_array_equal(result_sparse["points"], invalid_sparse["points"])


def test_deterministic_whole_track_budget_and_disconnected_guard():
    cameras, sparse = _geometry()
    result_cameras, result_sparse, metadata = optimize_bundle(
        cameras, sparse, _settings(max_points=3, max_observations=9), _spine())
    assert metadata["status"] == "optimized"
    assert metadata["selected_points"] == 3
    assert metadata["selection_order_point_indices"] == [0, 1, 2]
    assert metadata["selected_observations"] == 9
    assert metadata["exclusions"]["budget_points"] == len(sparse["points"]) - 3
    assert result_sparse["points"].shape == sparse["points"].shape
    assert np.all(result_sparse["confidence"][3:] == 0.0)
    np.testing.assert_array_equal(result_sparse["points"][3:], sparse["points"][3:])
    for key in ("colors", "inferred", "track_length"):
        np.testing.assert_array_equal(result_sparse[key], sparse[key])
    repeat_cameras, repeat_sparse, repeat_metadata = optimize_bundle(
        cameras, sparse, _settings(max_points=3, max_observations=9), _spine())
    assert metadata == repeat_metadata
    assert result_cameras == repeat_cameras
    for key in sparse:
        assert result_sparse[key].tobytes() == repeat_sparse[key].tobytes()
    _, observation_limited_sparse, observation_limited = optimize_bundle(
        cameras, sparse, _settings(max_observations=10), _spine())
    assert observation_limited["status"] == "optimized"
    assert observation_limited["selected_point_indices"] == [0, 1, 2]
    assert observation_limited["selected_observations"] == 9
    assert observation_limited["exclusions"]["budget_observations"] == 3
    assert observation_limited["retained_points"] == 3
    assert observation_limited_sparse["points"].tobytes() == result_sparse["points"].tobytes()

    _, retained_sparse, observation_budget = optimize_bundle(
        cameras, sparse, _settings(max_observations=8), _spine())
    assert observation_budget["status"] == "unsupported_retained_input"
    assert "visual features" in observation_budget["reason"]
    np.testing.assert_array_equal(retained_sparse["points"], sparse["points"])

    disconnected_cameras, disconnected_sparse = _geometry()
    disconnected_cameras[-1]["observations"] = []
    _, _, disconnected = optimize_bundle(
        disconnected_cameras, disconnected_sparse, _settings(), _spine())
    assert disconnected["status"] == "unsupported_retained_input"
    assert "visual support" in disconnected["reason"] or "tracks" in disconnected["reason"]


def test_invalid_gnss_sigma_and_nonpositive_initial_depth_are_honestly_retained():
    cameras, sparse = _geometry()
    cameras[1]["gps_sample"]["sigma_h_m"] = 0.0
    result_cameras, result_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "unsupported_retained_input"
    assert "sigma" in metadata["reason"]
    assert result_cameras == cameras
    np.testing.assert_array_equal(result_sparse["points"], sparse["points"])

    cameras, sparse = _geometry()
    sparse["points"][:, 2] = -1.0
    _, _, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "unsupported_retained_input"
    assert "tracks" in metadata["reason"] or "support" in metadata["reason"]


def test_nonidentity_world_to_camera_rotations_and_joint_visual_motion():
    from scipy.spatial.transform import Rotation

    cameras, sparse = _geometry()
    # Re-express the same constructed scene in a nontrivial local coordinate
    # frame; this catches accidental R_wc, transposition and center conventions.
    Q = Rotation.from_rotvec([0.2, -0.35, 0.1]).as_matrix()
    offset = np.array([20.0, -9.0, 2.0])
    sparse["points"] = sparse["points"] @ Q.T + offset
    for camera in cameras:
        camera["R_cw"] = (np.asarray(camera["R_cw"]) @ Q.T).tolist()
        camera["center"] = (Q @ np.asarray(camera["center"]) + offset).tolist()
        camera["gps"] = (Q @ np.asarray(camera["gps"]) + offset).tolist()
        camera["visual_reference"] = {"R_cw": copy.deepcopy(camera["R_cw"]),
                                      "center": copy.deepcopy(camera["center"])}
    for index, camera in enumerate(cameras[1:], start=1):
        perturbation = Rotation.from_rotvec([0.001 * index, -0.002, 0.001]).as_matrix()
        camera["R_cw"] = (perturbation @ np.asarray(camera["R_cw"])).tolist()
    sparse["points"] += np.array([0.02, -0.01, 0.04])
    old_cameras, old_points = copy.deepcopy(cameras), sparse["points"].copy()
    output_cameras, output_sparse, metadata = optimize_bundle(
        cameras, sparse, _settings(ftol=1e-6, xtol=1e-6, gtol=1e-6), _spine())
    assert metadata["status"] == "optimized", metadata
    assert metadata["reprojection_rmse_px"] < metadata["initial_reprojection_rmse_px"]
    assert metadata["final_robust_cost"] < metadata["initial_robust_cost"]
    assert not np.array_equal(output_sparse["points"], old_points)
    assert any(not np.allclose(camera["R_cw"], old["R_cw"])
               for camera, old in zip(output_cameras, old_cameras, strict=True))
    for camera in output_cameras:
        rotation = np.asarray(camera["R_cw"])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        for observation in camera["observations"]:
            np.testing.assert_allclose(_project(camera, output_sparse["points"][observation["point_index"]]),
                                       observation["uv"], atol=1e-3)


def test_actual_solver_receives_normalized_residuals_and_sparse_pattern(monkeypatch):
    import scipy.optimize
    from scipy.sparse import issparse
    from scipy.spatial.transform import Rotation

    actual_solver = scipy.optimize.least_squares
    cameras, sparse = _geometry()
    cameras[0]["gps_sample"] = {"sigma_h_m": 0.2, "sigma_v_m": 0.4}
    cameras[1]["gps_sample"] = {"sigma_h_m": 0.3}  # vertical setting fallback
    cameras[2]["gps_sample"] = None
    cameras[0]["visual_reference"]["R_cw"] = Rotation.from_rotvec([0.0, 0.0, 0.001]).as_matrix().tolist()
    cameras[0]["observations"][0]["uv"][0] += 0.4
    settings = _settings(reprojection_sigma_px=2.0, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    spine = _spine(rotation_sigma_rad=0.03)
    inspected = {}

    def inspect_solver(fun, initial, **kwargs):
        residual = fun(initial)
        observation_count = 18
        assert residual[0] == pytest.approx(-0.4 / settings.reprojection_sigma_px)
        expected_sigmas = [[0.2, 0.2, 0.4], [0.3, 0.3, spine.gnss_sigma_v_m],
                           [spine.gnss_sigma_h_m] * 2 + [spine.gnss_sigma_v_m]]
        for index, sigma in enumerate(expected_sigmas):
            start = 2 * observation_count + 3 * index
            np.testing.assert_allclose(residual[start:start + 3],
                                       (np.asarray(cameras[index]["center"]) - cameras[index]["gps"]) / sigma)
        np.testing.assert_allclose(residual[-3:], [0, 0, -0.001 / spine.rotation_sigma_rad], atol=1e-12)
        pattern = kwargs["jac_sparsity"]
        assert issparse(pattern) and pattern.shape == (48, 36)
        assert kwargs["loss"] == "huber" and kwargs["f_scale"] == spine.robust_delta
        assert kwargs["max_nfev"] == settings.max_nfev
        # Point 0 depends on camera 0 and point 0, but not on camera 1 or point 1.
        assert pattern[0, 0] and pattern[0, 18]
        assert not pattern[0, 6] and not pattern[0, 21]
        assert pattern[36, 3] and not pattern[36, 0]
        inspected["residuals_and_pattern"] = True
        result = actual_solver(fun, initial, **kwargs)
        inspected["cost"] = result.cost
        return result

    monkeypatch.setattr(scipy.optimize, "least_squares", inspect_solver)
    _, _, metadata = optimize_bundle(cameras, sparse, settings, spine)
    assert inspected.get("residuals_and_pattern"), metadata
    assert metadata["optimizer_invoked"]
    assert metadata["final_robust_cost"] == pytest.approx(inspected["cost"])
    assert metadata["gnss_sigma_used"][1]["vertical_source"] == "spine.gnss_sigma_v_m"


@pytest.mark.parametrize("case", ["exception", "over_limit", "nan", "increased", "behind", "near_plane"])
def test_solver_failure_acceptance_gates_return_input_bytes(monkeypatch, case):
    import scipy.optimize

    cameras, sparse = _geometry()
    cameras[0]["pose_covariance_tangent"] = np.eye(6).tolist()
    cameras[0]["covariance_domain"] = "input_pose_graph"
    settings = _settings()

    def faulty_solver(fun, initial, **kwargs):
        if case == "exception":
            raise RuntimeError("constructed solver exception")
        state = initial.copy()
        if case == "nan":
            state[0] = np.nan
        elif case == "increased":
            state[3] += 1.0
        elif case in ("behind", "near_plane"):
            # A coherent similarity preserves image rays and matches GPS, but
            # reflects signed Z (or moves the whole scene below min_depth).
            scale = -1.0 if case == "behind" else 0.0001
            shift = np.asarray([0.25, 0.05, 0.08])
            poses = state[:18].reshape(3, 6)
            poses[:, 3:] = poses[:, 3:] * scale + shift
            state[18:] = (state[18:].reshape(-1, 3) * scale + shift).ravel()
        result_fun = fun(state) if case != "nan" else np.zeros(48)
        return SimpleNamespace(x=state, fun=result_fun, cost=0.5 * np.dot(result_fun, result_fun),
                               nfev=settings.max_nfev + 1 if case == "over_limit" else 1,
                               success=True, status=1, message="constructed invalid solver output")

    # Make initial GNSS fit large enough that reflected geometry has smaller
    # robust cost; the cheirality gate, not the cost gate, must reject it.
    if case in ("behind", "near_plane"):
        scale = -1.0 if case == "behind" else 0.0001
        for camera in cameras:
            camera["gps_sample"] = {"sigma_h_m": 0.001, "sigma_v_m": 0.001}
            camera["gps"] = (np.asarray(camera["center"]) * scale + [0.25, 0.05, 0.08]).tolist()
    before_cameras = copy.deepcopy(cameras)
    before_sparse = copy.deepcopy(sparse)
    monkeypatch.setattr(scipy.optimize, "least_squares", faulty_solver)
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, settings, _spine())
    assert metadata["status"] == "failed_retained_input", metadata
    assert metadata["optimizer_invoked"] and metadata["poses_retained"] and metadata["points_retained"]
    _assert_retained(before_cameras, before_sparse, output_cameras, output_sparse)
    expected_reason = {"exception": "constructed solver exception", "over_limit": "exceeded max_nfev",
                       "nan": "non-finite", "increased": "objective increased",
                       "behind": "behind/too near", "near_plane": "behind/too near"}
    assert expected_reason[case] in metadata["reason"]


@pytest.mark.parametrize("case", ["float_index", "negative_index", "bool_index", "invalid_R", "negative_K",
                                  "skew_K", "float32", "nan_point", "provenance_length",
                                  "inferred_length", "gnss_sigma_nan", "gnss_sigma_negative", "no_prior",
                                  "fewer_gps", "no_gps_baseline", "too_few_tracks", "collinear_features"])
def test_invalid_and_insufficient_inputs_are_retained(monkeypatch, case):
    import scipy.optimize

    cameras, sparse = _geometry()
    if case == "float_index":
        cameras[0]["observations"][0]["point_index"] = 0.0
    elif case == "negative_index":
        cameras[0]["observations"][0]["point_index"] = -1
    elif case == "bool_index":
        cameras[0]["observations"][0]["point_index"] = True
    elif case == "invalid_R":
        cameras[0]["R_cw"][0][0] = -1.0
    elif case == "negative_K":
        cameras[0]["K"][0][0] = -120.0
    elif case == "skew_K":
        cameras[0]["K"][0][1] = 1.0
    elif case == "float32":
        sparse["points"] = sparse["points"].astype(np.float32)
    elif case == "nan_point":
        sparse["points"][0, 0] = np.nan
    elif case == "provenance_length":
        sparse["track_length"] = sparse["track_length"][:-1]
    elif case == "inferred_length":
        sparse["inferred"] = sparse["inferred"][:-1]
    elif case == "gnss_sigma_nan":
        cameras[0]["gps_sample"]["sigma_h_m"] = np.nan
    elif case == "gnss_sigma_negative":
        cameras[0]["gps_sample"]["sigma_v_m"] = -1.0
    elif case == "no_prior":
        cameras[0].pop("visual_reference")
    elif case == "fewer_gps":
        cameras[1]["gps_inlier"] = False
    elif case == "no_gps_baseline":
        for camera in cameras:
            camera["gps"] = [0.0, 0.0, 0.0]
    elif case == "too_few_tracks":
        for camera in cameras:
            camera["observations"] = camera["observations"][:1]
    elif case == "collinear_features":
        for camera in cameras:
            for observation in camera["observations"]:
                observation["uv"][1] = 80.0

    def never_called(*args, **kwargs):
        pytest.fail("unsupported input invoked optimizer")

    monkeypatch.setattr(scipy.optimize, "least_squares", never_called)
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "unsupported_retained_input", metadata
    assert not metadata["optimizer_invoked"]
    _assert_retained(cameras, sparse, output_cameras, output_sparse)


def test_disconnected_supported_components_are_not_optimized():
    cameras, sparse = _geometry()
    second_cameras = copy.deepcopy(cameras)
    for camera in second_cameras:
        camera["frame_id"] += "_second"
        for observation in camera["observations"]:
            observation["point_index"] += 6
    for key, value in sparse.items():
        sparse[key] = np.concatenate((value, value))
    all_cameras = cameras + second_cameras
    output_cameras, output_sparse, metadata = optimize_bundle(all_cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "unsupported_retained_input"
    assert "disconnected" in metadata["reason"]
    assert metadata["selected_points"] == 12 and metadata["selected_observations"] == 36
    assert metadata["retained_points"] == 12 and metadata["retained_observations"] == 36
    assert not metadata["optimizer_invoked"]
    _assert_retained(all_cameras, sparse, output_cameras, output_sparse)


def test_excluded_tracks_keep_real_points_provenance_and_lose_stale_confidence():
    cameras, sparse = _geometry()
    # Leave three full tracks so this remains a supported solve.  The other
    # three are inferred, single-view, and invalid-observation whole tracks.
    sparse["inferred"][3] = True
    for camera in cameras[1:]:
        camera["observations"] = [observation for observation in camera["observations"]
                                  if observation["point_index"] != 4]
    cameras[0]["observations"][5]["uv"] = [-1.0, 50.0]
    old_points = sparse["points"].copy()
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "optimized", metadata
    assert metadata["selected_point_indices"] == [0, 1, 2]
    assert metadata["exclusions"]["inferred_points"] == 1
    assert metadata["exclusions"]["single_view_tracks"] == 1
    assert metadata["exclusions"]["invalid_tracks"] == 1
    assert metadata["poses_changed"]
    np.testing.assert_array_equal(output_sparse["points"][3:], old_points[3:])
    np.testing.assert_array_equal(output_sparse["confidence"][3:], 0)
    for key in ("inferred", "colors", "track_length"):
        np.testing.assert_array_equal(output_sparse[key], sparse[key])
    for camera, original in zip(output_cameras, cameras, strict=True):
        assert camera["observations"] == original["observations"]


@pytest.mark.parametrize("case", ["nonfinite_uv", "out_of_bounds_uv", "duplicate"])
def test_bad_observations_exclude_whole_tracks_without_changing_indexing(case):
    cameras, sparse = _geometry()
    if case == "nonfinite_uv":
        cameras[0]["observations"][0]["uv"][0] = np.nan
    elif case == "out_of_bounds_uv":
        cameras[0]["observations"][0]["uv"][0] = cameras[0]["width"]
    else:
        cameras[0]["observations"].append(copy.deepcopy(cameras[0]["observations"][0]))
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "optimized", metadata
    assert metadata["selected_point_indices"] == [1, 2, 3, 4, 5]
    assert metadata["exclusions"]["invalid_tracks"] == 1
    np.testing.assert_array_equal(output_sparse["points"][0], sparse["points"][0])
    assert output_sparse["confidence"][0] == 0
    assert len(output_cameras[0]["observations"]) == len(cameras[0]["observations"])


def test_near_plane_guard_is_finite_and_signed_but_cannot_qualify_result(monkeypatch):
    import scipy.optimize

    cameras, sparse = _geometry()
    inspected = []

    def inspect_guard(fun, initial, **kwargs):
        state = initial.copy()
        # Place an observed point exactly on, then just behind, camera 0's
        # plane.  The finite residual guard is separate from acceptance.
        for z in (0.0, -1e-12, 1e-12):
            state[20] = z
            inspected.append(bool(np.isfinite(fun(state)).all()))
        state[20] = 0.0
        residual = fun(state)
        return SimpleNamespace(x=state, fun=residual, cost=0.5 * np.dot(residual, residual),
                               nfev=1, success=True, status=1, message="constructed near plane")

    monkeypatch.setattr(scipy.optimize, "least_squares", inspect_guard)
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert inspected == [True, True, True], metadata
    assert metadata["status"] == "failed_retained_input"
    assert metadata["optimizer_invoked"]
    _assert_retained(cameras, sparse, output_cameras, output_sparse)


@pytest.mark.parametrize("mode", ["enabled_false", "spine_false", "unavailable"])
def test_disabled_and_dependency_fallback_preserve_all_input_bytes(monkeypatch, mode):
    cameras, sparse = _geometry()
    settings, spine = _settings(), _spine()
    if mode == "enabled_false":
        settings.enabled = False
    elif mode == "spine_false":
        spine.optimize = False
    else:
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, settings, spine)
    assert metadata["status"] == ("unavailable_retained_input" if mode == "unavailable"
                                  else "disabled_retained_input")
    assert not metadata["optimizer_invoked"]
    _assert_retained(cameras, sparse, output_cameras, output_sparse)


def test_success_with_unchanged_poses_keeps_unsupported_confidence_and_prior():
    cameras, sparse = _geometry()
    for camera in cameras:
        camera["gps"] = copy.deepcopy(camera["center"])
        camera["pose_covariance_tangent"] = np.eye(6).tolist()
        camera["covariance_domain"] = "input_conditional_covariance"
    output_cameras, output_sparse, metadata = optimize_bundle(
        cameras, sparse, _settings(max_points=3), _spine())
    assert metadata["status"] == "optimized"
    assert metadata["poses_retained"] and metadata["points_retained"]
    assert not metadata["poses_changed"]
    assert metadata["initial_robust_cost"] == metadata["final_robust_cost"] == 0
    assert output_cameras == cameras
    np.testing.assert_array_equal(output_sparse["confidence"][3:], sparse["confidence"][3:])


@pytest.mark.parametrize("max_points,max_observations", [(12, 36), (32, 36), (12, 96), (14, 43)])
def test_sequence_coverage_reaches_late_cameras_with_bounded_deterministic_selection(max_points, max_observations):
    cameras, sparse = _sequence_geometry()
    # Ascending-only selection under these budgets misses the final cameras.
    ascending_limit = min(max_points, max_observations // 3)
    assert not any(observation["point_index"] < ascending_limit for observation in cameras[-1]["observations"])
    settings = _settings(max_points=max_points, max_observations=max_observations)
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, settings, _spine())
    assert metadata["status"] == "optimized", metadata
    assert metadata["optimizer_invoked"] and metadata["nfev"] == 1
    assert metadata["selection_coverage_complete"] and metadata["selection_network_connected"]
    assert metadata["selection_is_exhaustive_feasibility_search"] is False
    assert metadata["coverage_selected_points"] == 12
    assert metadata["coverage_selected_observations"] == 36
    assert all(support >= 3 for support in metadata["camera_visual_support"])
    assert metadata["selected_points"] <= max_points
    assert metadata["selected_observations"] <= max_observations
    selected = set(metadata["selected_point_indices"])
    assert selected == set(metadata["selection_order_point_indices"])
    for camera in cameras:
        features = [observation["uv"] for observation in camera["observations"]
                    if observation["point_index"] in selected]
        assert len(features) >= 3
        assert np.linalg.matrix_rank(np.asarray(features) - features[0]) == 2
    # Each chosen point still contributes its complete three-view track.
    assert metadata["selected_observations"] == 3 * len(selected)
    repeat_cameras, repeat_sparse, repeat_metadata = optimize_bundle(cameras, sparse, settings, _spine())
    assert repeat_metadata == metadata
    assert repeat_cameras == output_cameras
    for key in sparse:
        assert repeat_sparse[key].tobytes() == output_sparse[key].tobytes()


@pytest.mark.parametrize("max_points,max_observations", [(11, 96), (32, 35), (3, 9)])
def test_sequence_genuinely_insufficient_budgets_retain_input(max_points, max_observations):
    cameras, sparse = _sequence_geometry()
    # Cameras 0,3,5,8 each need three different tracks from disjoint local
    # windows.  Twelve whole tracks (36 observations) are a lower bound here.
    output_cameras, output_sparse, metadata = optimize_bundle(
        cameras, sparse, _settings(max_points=max_points, max_observations=max_observations), _spine())
    assert metadata["status"] == "unsupported_retained_input", metadata
    assert not metadata["optimizer_invoked"]
    assert metadata["selected_points"] <= max_points
    assert metadata["selected_observations"] <= max_observations
    assert not metadata["selection_coverage_complete"]
    _assert_retained(cameras, sparse, output_cameras, output_sparse)


def test_coverage_selection_adds_component_bridge_before_ascending_fill():
    cameras, sparse = _geometry()
    second_cameras = copy.deepcopy(cameras)
    for index, camera in enumerate(second_cameras):
        camera["center"] = [3.0 + index, 0.1 * index, 0.0]
        for observation in camera["observations"]:
            observation["point_index"] += 6
    for key, value in sparse.items():
        sparse[key] = np.concatenate((value, value), axis=0)
    # Put a single track bridging the two sets after the early redundant ones.
    for key, value in sparse.items():
        sparse[key] = np.concatenate((value, value[:1]), axis=0)
    cameras += second_cameras
    for camera in cameras:
        camera["gps"] = copy.deepcopy(camera["center"])
        for observation in camera["observations"]:
            observation["uv"] = _project(camera, sparse["points"][observation["point_index"]]).tolist()
    for camera in cameras[1:4]:
        camera["observations"].append({"point_index": 12, "uv": _project(camera, sparse["points"][12]).tolist()})
    _, _, metadata = optimize_bundle(cameras, sparse, _settings(max_points=7, max_observations=21), _spine())
    assert metadata["status"] == "optimized", metadata
    assert metadata["optimizer_invoked"]
    assert metadata["coverage_selected_points"] == 6
    assert metadata["connectivity_selected_points"] == 1
    assert metadata["connectivity_selected_observations"] == 3
    assert metadata["selection_order_point_indices"][-1] == 12
    assert metadata["selection_network_connected"]


@pytest.mark.parametrize("case", ["confidence_below_zero", "confidence_above_one", "colors_float", "colors_int16"])
def test_confidence_range_and_contracted_color_type_are_guarded(case):
    cameras, sparse = _geometry()
    if case == "confidence_below_zero":
        sparse["confidence"][0] = -0.001
    elif case == "confidence_above_one":
        sparse["confidence"][0] = 1.001
    elif case == "colors_float":
        sparse["colors"] = sparse["colors"].astype(np.float64)
    else:
        sparse["colors"] = sparse["colors"].astype(np.int16)
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "unsupported_retained_input"
    assert not metadata["optimizer_invoked"]
    assert "confidence" in metadata["reason"] or "uint8" in metadata["reason"]
    _assert_retained(cameras, sparse, output_cameras, output_sparse)


@pytest.mark.parametrize("case", ["projection_distorted", "image_distorted", "missing_projection", "both_distorted",
                                  "image_unknown", "image_projection_spelling", "image_null"])
def test_distorted_unknown_or_undeclared_projection_domain_is_rejected(case):
    cameras, sparse = _geometry()
    if case in ("projection_distorted", "both_distorted"):
        cameras[0]["projection_domain"] = "distorted"
    if case in ("image_distorted", "both_distorted"):
        cameras[0]["image_domain"] = "distorted"
    if case == "missing_projection":
        cameras[0].pop("projection_domain")
    if case == "image_unknown":
        cameras[0]["image_domain"] = "unknown"
    if case == "image_projection_spelling":
        cameras[0]["image_domain"] = "undistorted_pinhole"
    if case == "image_null":
        cameras[0]["image_domain"] = None
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "unsupported_retained_input"
    assert not metadata["optimizer_invoked"]
    assert "undistorted pinhole required" in metadata["reason"]
    _assert_retained(cameras, sparse, output_cameras, output_sparse)


def test_explicit_undistorted_image_domain_matches_absent_image_domain():
    cameras, sparse = _geometry()
    absent_cameras, absent_sparse, absent_metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    for camera in cameras:
        camera["image_domain"] = "undistorted"
    output_cameras, output_sparse, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["status"] == "optimized" and metadata["optimizer_invoked"]
    assert metadata == absent_metadata
    for output_camera, absent_camera in zip(output_cameras, absent_cameras, strict=True):
        assert output_camera["image_domain"] == "undistorted"
        assert output_camera["projection_domain"] == "undistorted_pinhole"
        assert output_camera == {**absent_camera, "image_domain": "undistorted"}
    for key in sparse:
        np.testing.assert_array_equal(output_sparse[key], absent_sparse[key])


def test_typed_settings_are_required_and_used_directly():
    from drishti.config.models import BundleAdjustmentConfig, SpineConfig

    cameras, sparse = _geometry()
    _, _, metadata = optimize_bundle(cameras, sparse, BundleAdjustmentConfig(max_points=3), SpineConfig())
    assert metadata["status"] == "optimized", metadata
    assert metadata["settings"]["max_points"] == 3
    assert metadata["settings"]["gnss_sigma_h_m"] == SpineConfig().gnss_sigma_h_m
    settings = _settings()
    del settings.gtol
    output_cameras, output_sparse, missing = optimize_bundle(cameras, sparse, settings, _spine())
    assert missing["status"] == "failed_retained_input"
    assert not missing["optimizer_invoked"] and "gtol" in missing["reason"]
    _assert_retained(cameras, sparse, output_cameras, output_sparse)


def test_scipy_version_is_measured_or_explicitly_unavailable(monkeypatch):
    from importlib.metadata import PackageNotFoundError, version

    import drishti.recon.bundle_adjustment as module

    cameras, sparse = _geometry()
    _, _, metadata = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert metadata["scipy_version"] == version("scipy")

    def absent_distribution(name):
        assert name == "scipy"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(module, "version", absent_distribution)
    _, _, no_distribution = optimize_bundle(cameras, sparse, _settings(), _spine())
    assert no_distribution["status"] == "optimized"
    assert no_distribution["scipy_version"] is None
