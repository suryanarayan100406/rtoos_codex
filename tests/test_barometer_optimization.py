"""Constructed real-solver checks, not flight-quality or accuracy acceptance."""

from __future__ import annotations

import copy
import json
import math
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from drishti.config.models import BundleAdjustmentConfig, SpineConfig
from drishti.recon.barometer import make_height_factor, pose_height_error, select_barometer
from drishti.recon.bundle_adjustment import optimize_bundle
from drishti.recon.spine import optimize_pose_graph


def _barometer(**overrides):
    return SimpleNamespace(**{"enabled": True, "min_measurements": 3, "sigma_m": 0.02,
                              "max_time_error_s": 0.1, "time_origin_s": 1000.,
                              "time_offset_s": 0., **overrides})


def _spine(**overrides):
    return SpineConfig(gnss_sigma_h_m=0.05, gnss_sigma_v_m=0.2, visual_sigma_m=0.7,
                       max_iterations=200, **overrides)


def _ba(**overrides):
    return BundleAdjustmentConfig.model_validate({"max_nfev": 500, "ftol": 1e-8,
                                                  "xtol": 1e-8, "gtol": 1e-8,
                                                  "reprojection_sigma_px": 4.0, **overrides})


def _observation(index, height, *, sigma=0.02, error=0.0, frame_time=None, source=None):
    time = float(index)
    return {"source_index": index if source is None else source, "source_line": index + 2,
            "source_time_s": 1000.0 + time, "time_s": time, "frame_time_s": time if frame_time is None else frame_time,
            "time_error_s": error, "height_m": float(height), "sigma_m": sigma,
            "sigma_source": "csv"}


def _scene(sigma=0.02):
    """Positive-depth pinhole observations and GNSS with vertical perturbations."""
    true_centers = np.array([[0., 0., 0.], [1., .2, .2], [2., -.2, -.15], [3., .1, .1]])
    points = np.array([[x, y, 8. + .3 * x + .2 * y]
                       for x in (-.5, .5, 1.5, 2.5) for y in (-.7, .1, .8)], dtype=np.float64)
    errors = np.array([.15, -.12, .09, -.12])  # zero mean; no imposed absolute baro datum
    K = [[140., 0., 120.], [0., 140., 100.], [0., 0., 1.]]
    cameras = []
    for index, center in enumerate(true_centers):
        current = center + [0., 0., errors[index]]
        xyz = points - center
        uv = xyz[:, :2] / xyz[:, 2, None] * 140. + [120., 100.]
        cameras.append({
            "frame_id": f"math_{index}", "time_s": float(index), "width": 240, "height": 200,
            "K": copy.deepcopy(K), "R_cw": np.eye(3).tolist(), "center": current.tolist(),
            "visual_reference": {"R_cw": np.eye(3).tolist(), "center": current.tolist()},
            "projection_domain": "undistorted_pinhole", "distortion": [],
            "gps": current.tolist(), "gps_inlier": True,
            "barometer": _observation(index, center[2] + 41.7, sigma=sigma),
            "observations": [{"point_index": p, "uv": pixel.tolist()} for p, pixel in enumerate(uv)],
            "pose_covariance_tangent": np.eye(6).tolist(), "covariance_domain": "prior_graph",
        })
    sparse = {"points": points.copy(), "colors": np.full((len(points), 3), 128, np.uint8),
              "inferred": np.zeros(len(points), bool), "confidence": np.full(len(points), .8),
              "track_length": np.full(len(points), 4, np.int64)}
    return cameras, sparse, true_centers


def _solve(kind, cameras, sparse, barometer, **overrides):
    if kind == "graph":
        pytest.importorskip("gtsam")
        output, metadata = optimize_pose_graph(cameras, _spine(**overrides),
                                               global_refinement=True, barometer_settings=barometer)
        return output, sparse, metadata
    pytest.importorskip("scipy")
    return optimize_bundle(cameras, sparse, _ba(**overrides), _spine(), barometer_settings=barometer)


def _relative_z_error(cameras, truth):
    errors = np.array([camera["center"][2] for camera in cameras]) - truth[:, 2]
    return np.sqrt(np.mean(np.square(errors - np.mean(errors))))


def _assert_unused(metadata):
    barometer = metadata["barometer"]
    assert metadata["baro_used"] is False
    assert barometer["factors_accepted"] == 0
    assert barometer["bias_m"] is None and barometer["fit_rmse_m"] is None
    assert barometer["fit_is_independent_accuracy"] is False


def _assert_baseline_preserved(kind, cameras, sparse, settings):
    before_cameras, before_sparse = copy.deepcopy(cameras), copy.deepcopy(sparse)
    baseline, baseline_sparse, baseline_metadata = _solve(kind, cameras, sparse, None)
    output, refined, metadata = _solve(kind, cameras, sparse, settings)
    assert baseline_metadata["status"] == metadata["status"] == "optimized", metadata
    assert metadata["optimizer_invoked"] and output == baseline
    assert metadata["method"] == baseline_metadata["method"]
    _assert_unused(metadata)
    assert metadata["barometer"]["status"] == "unavailable"
    assert metadata["barometer"]["factors_added"] == 0
    assert not metadata["barometer"]["optimizer_invoked"]
    if kind == "ba":
        assert metadata["jacobian_shape"] == baseline_metadata["jacobian_shape"]
    for key in sparse:
        np.testing.assert_array_equal(refined[key], baseline_sparse[key])
        assert sparse[key].tobytes() == before_sparse[key].tobytes()
    assert cameras == before_cameras
    json.dumps(metadata, allow_nan=False)
    return metadata


def _reuse_first_source(cameras):
    cameras[1]["time_s"] = .02
    cameras[1]["barometer"] = copy.deepcopy(cameras[0]["barometer"])
    cameras[1]["barometer"].update(frame_time_s=.02, time_error_s=-.02)


@pytest.mark.parametrize("kind", ["graph", "ba"])
def test_real_solver_corrects_relative_vertical_perturbation_with_free_unknown_offset(kind):
    cameras, sparse, truth = _scene()
    before_cameras, before_sparse = copy.deepcopy(cameras), copy.deepcopy(sparse)
    baseline, _, baseline_metadata = _solve(kind, cameras, sparse, None)
    output, refined, metadata = _solve(kind, cameras, sparse, _barometer())
    assert baseline_metadata["status"] == metadata["status"] == "optimized", metadata
    assert _relative_z_error(output, truth) < _relative_z_error(baseline, truth) * .15
    assert _relative_z_error(output, truth) < .006
    barometer = metadata["barometer"]
    assert metadata["baro_used"] and barometer["optimizer_invoked"]
    assert barometer["status"] == "optimized" and barometer["reason"] is None
    assert barometer["factors_added"] == barometer["factors_accepted"] == 4
    assert barometer["bias_m"] == pytest.approx(41.7, abs=.03)
    raw = [camera["center"][2] + barometer["bias_m"] - camera["barometer"]["height_m"] for camera in output]
    assert barometer["fit_rmse_m"] == pytest.approx(np.sqrt(np.mean(np.square(raw))))
    assert not barometer["fit_is_independent_accuracy"] and barometer["bias_prior"] is None
    assert barometer["absolute_datum"] == "not_interpreted"
    assert cameras == before_cameras
    for key in sparse:
        np.testing.assert_array_equal(sparse[key], before_sparse[key])
    if kind == "ba":
        assert metadata["poses_changed"] and metadata["covariance_cleared_cameras"] == 4
        assert all("pose_covariance_tangent" not in camera for camera in output)
        assert np.all(refined["confidence"] > 0) and refined["points"].dtype == np.float64
    else:
        assert metadata["covariance_status"] == "computed_conditional_on_factor_noise"
        assert all(np.isfinite(camera["pose_covariance_tangent"]).all() for camera in output)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("transform", ["local_translation", "sensor_offset"])
def test_free_bias_preserves_translation_and_sensor_offset_invariance(kind, transform):
    cameras, sparse, _ = _scene()
    output, points, metadata = _solve(kind, cameras, sparse, _barometer())
    transformed, transformed_sparse = copy.deepcopy(cameras), copy.deepcopy(sparse)
    shift = 12.5
    if transform == "local_translation":
        transformed_sparse["points"][:, 2] += shift
        for camera in transformed:
            for field in ("center", "gps"):
                camera[field][2] += shift
            camera["visual_reference"]["center"][2] += shift
    else:
        for camera in transformed:
            camera["barometer"]["height_m"] += shift
    shifted_output, shifted_points, shifted = _solve(kind, transformed, transformed_sparse, _barometer())
    assert metadata["status"] == shifted["status"] == "optimized", shifted
    offset = [0., 0., shift] if transform == "local_translation" else [0., 0., 0.]
    np.testing.assert_allclose([c["center"] for c in shifted_output], np.array([c["center"] for c in output]) + offset,
                               atol=1e-4)
    np.testing.assert_allclose(shifted_points["points"], points["points"] + offset, atol=1e-4)
    expected_bias_shift = -shift if transform == "local_translation" else shift
    assert shifted["barometer"]["bias_m"] == pytest.approx(metadata["barometer"]["bias_m"] + expected_bias_shift, abs=1e-4)
    assert shifted["barometer"]["fit_rmse_m"] == pytest.approx(metadata["barometer"]["fit_rmse_m"], abs=1e-5)


@pytest.mark.parametrize("kind", ["graph", "ba"])
def test_actual_measurement_sigma_controls_relative_vertical_weight(kind):
    strong_cameras, sparse, truth = _scene(sigma=.005)
    weak_cameras, _, _ = _scene(sigma=10.)
    strong, _, strong_metadata = _solve(kind, strong_cameras, sparse, _barometer())
    weak, _, weak_metadata = _solve(kind, weak_cameras, sparse, _barometer())
    assert strong_metadata["status"] == weak_metadata["status"] == "optimized"
    assert _relative_z_error(strong, truth) < _relative_z_error(weak, truth) * .05
    assert {item["sigma_m"] for item in strong_metadata["barometer"]["sigma_used"]} == {.005}
    assert {item["sigma_m"] for item in weak_metadata["barometer"]["sigma_used"]} == {10.}


def test_shared_selection_deduplicates_by_time_then_camera_index_and_records_provenance():
    cameras, _, _ = _scene()
    for index, error in enumerate((.09, -.02, .02)):
        frame_time = 1. - error
        cameras[index]["time_s"] = frame_time
        cameras[index]["barometer"] = _observation(1, 41.7, error=error, frame_time=frame_time, source=0)
    selection = select_barometer(cameras, _barometer(min_measurements=2))
    assert [item.camera_index for item in selection.observations] == [1, 3]
    assert [item.source_index for item in selection.observations] == [0, 3]
    assert selection.metadata["distinct_measurements"] == 2
    assert {item["camera_index"] for item in selection.metadata["excluded"]} == {0, 2}
    assert selection.metadata["sigma_used"][0]["sigma_source"] == "csv"
    assert selection.metadata["timing_approximation"] == "bounded_nearest_recorded_sample_no_interpolation"
    assert selection.metadata["factors_added"] == 0 and not selection.metadata["optimizer_invoked"]
    insufficient = select_barometer(cameras, _barometer())
    assert insufficient.observations == ()
    assert insufficient.metadata["distinct_measurements"] == 2
    assert "insufficient distinct" in insufficient.metadata["reason"]

    # A replacement winner must also update earlier duplicate exclusions.
    cameras[2]["barometer"]["time_error_s"] = .01
    cameras[2]["time_s"] = cameras[2]["barometer"]["frame_time_s"] = .99
    changed = select_barometer(cameras, _barometer(min_measurements=2))
    assert changed.observations[0].camera_index == 2
    assert all(item["selected_camera_index"] == 2 for item in changed.metadata["excluded"])


@pytest.mark.parametrize("field,value", [
    ("height_m", float("nan")), ("height_m", "41.7"), ("height_m", True),
    ("sigma_m", 0.), ("sigma_m", -1.), ("sigma_m", float("inf")),
    ("source_index", True), ("source_index", -1), ("source_index", 1.0),
    ("source_line", 0), ("source_time_s", float("inf")), ("time_s", None),
    ("frame_time_s", 999.), ("time_error_s", .1001), ("time_error_s", float("nan")),
    ("sigma_source", "calibrated_accuracy"), ("unknown_field", 1),
])
def test_exact_finite_observation_contract_excludes_malformed_associations(field, value):
    cameras, _, _ = _scene()
    cameras[0]["barometer"][field] = value
    selection = select_barometer(cameras, _barometer())
    assert [item.camera_index for item in selection.observations] == [1, 2, 3]
    assert len(selection.metadata["excluded"]) == 1
    assert selection.metadata["excluded"][0]["reason"].startswith("invalid optional observation")


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("case", ["malformed", "duplicates", "disabled", "missing"])
def test_bad_or_inadequate_optional_support_still_executes_unchanged_baseline(kind, case):
    cameras, sparse, _ = _scene()
    settings = _barometer()
    if case == "malformed":
        cameras[0]["barometer"] = {"height_m": 41.7}
        cameras[1]["barometer"]["sigma_m"] = -1.
    elif case == "duplicates":
        for index, camera in enumerate(cameras):
            frame_time = .02 * index
            camera["time_s"] = frame_time
            camera["barometer"] = _observation(0, 41.7, error=-frame_time, frame_time=frame_time)
    elif case == "disabled":
        settings.enabled = False
    else:
        for camera in cameras:
            camera.pop("barometer")
    before = copy.deepcopy(cameras)
    baseline, baseline_points, baseline_metadata = _solve(kind, cameras, sparse, None)
    output, output_points, metadata = _solve(kind, cameras, sparse, settings)
    assert metadata["optimizer_invoked"] and metadata["status"] == "optimized"
    assert output == baseline and metadata["method"] == baseline_metadata["method"]
    for key in sparse:
        np.testing.assert_array_equal(output_points[key], baseline_points[key])
    _assert_unused(metadata)
    assert metadata["barometer"]["factors_added"] == 0
    assert not metadata["barometer"]["optimizer_invoked"]
    assert cameras == before
    if kind == "ba":
        assert metadata["jacobian_shape"] == baseline_metadata["jacobian_shape"]


@pytest.mark.parametrize("kind", ["graph", "ba"])
def test_accepted_factors_are_deduplicated_even_with_adequate_support(kind):
    cameras, sparse, _ = _scene()
    cameras[1]["time_s"] = .02
    cameras[1]["barometer"] = _observation(0, 41.7, error=-.02, frame_time=.02)
    output, _, metadata = _solve(kind, cameras, sparse, _barometer())
    assert metadata["status"] == "optimized", metadata
    assert len(output) == 4
    assert metadata["barometer"]["factors_added"] == metadata["barometer"]["factors_accepted"] == 3
    assert [item["camera_index"] for item in metadata["barometer"]["selected"]] == [0, 2, 3]


@pytest.mark.parametrize("kind", ["graph", "ba"])
def test_assumed_sigma_retains_explicit_provenance(kind):
    cameras, sparse, _ = _scene()
    for camera in cameras:
        camera["barometer"]["sigma_source"] = "barometer.sigma_m"
    _, _, metadata = _solve(kind, cameras, sparse, _barometer())
    assert metadata["baro_used"]
    assert {item["sigma_source"] for item in metadata["barometer"]["sigma_used"]} == {"barometer.sigma_m"}
    cameras[0]["barometer"]["sigma_m"] = .3
    selected = select_barometer(cameras, _barometer())
    assert "assumed sigma_m differs" in selected.metadata["excluded"][0]["reason"]


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("failure", ["exception", "disabled_solver", "missing_dependency"])
def test_fallback_distinguishes_attempted_and_rejected_factors_without_publishing_bias(kind, failure, monkeypatch):
    cameras, sparse, _ = _scene()
    before = copy.deepcopy(cameras)
    overrides = {}
    if failure == "exception":
        def fail(*args, **kwargs):
            raise RuntimeError("constructed optimizer failure")
        if kind == "graph":
            gtsam = pytest.importorskip("gtsam")
            monkeypatch.setattr(gtsam, "LevenbergMarquardtOptimizer", fail)
        else:
            scipy = pytest.importorskip("scipy.optimize")
            monkeypatch.setattr(scipy, "least_squares", fail)
    elif failure == "disabled_solver":
        overrides["optimize" if kind == "graph" else "enabled"] = False
    else:
        monkeypatch.setitem(sys.modules, "gtsam" if kind == "graph" else "scipy.optimize", None)
    # Call directly: the wrapper's importorskip must not skip dependency fallback.
    if kind == "graph":
        output, metadata = optimize_pose_graph(cameras, _spine(**overrides), barometer_settings=_barometer())
        output_sparse = sparse
    else:
        output, output_sparse, metadata = optimize_bundle(cameras, sparse, _ba(**overrides), _spine(),
                                                          barometer_settings=_barometer())
    _assert_unused(metadata)
    assert metadata["poses_retained"]
    assert [c["center"] for c in output] == [c["center"] for c in before]
    assert [c["R_cw"] for c in output] == [c["R_cw"] for c in before]
    for key in sparse:
        np.testing.assert_array_equal(output_sparse[key], sparse[key])
    barometer = metadata["barometer"]
    expected = 4 if failure == "exception" else 0
    assert barometer["factors_added"] == barometer["factors_rejected"] == expected
    assert barometer["factors_attempted"] == expected
    assert barometer["optimizer_invoked"] == (failure == "exception")
    assert cameras == before


def test_ba_nonconvergence_rejects_actual_attempt_and_preserves_sparse_provenance():
    cameras, sparse, _ = _scene()
    output, refined, metadata = _solve("ba", cameras, sparse, _barometer(), max_nfev=1)
    assert metadata["status"] == "failed_retained_input" and metadata["optimizer_invoked"]
    _assert_unused(metadata)
    assert metadata["barometer"]["factors_rejected"] == 4
    assert output == cameras
    for key in sparse:
        assert refined[key].tobytes() == sparse[key].tobytes()
        assert not np.shares_memory(refined[key], sparse[key])


def test_pose3_double_custom_factor_tangent_jacobian_matches_real_retract_finite_differences():
    gtsam = pytest.importorskip("gtsam")
    pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(.4, -.7, .2), np.array([2., -3., 5.]))
    bias, height = 37., 41.7
    jacobians: list[Any] = [None, None]
    residual = pose_height_error(pose, bias, height, jacobians)
    numeric = np.empty((1, 6))
    eps = 1e-6
    for index in range(6):
        delta = np.zeros(6)
        delta[index] = eps
        numeric[:, index] = (pose_height_error(pose.retract(delta), bias, height)
                             - pose_height_error(pose.retract(-delta), bias, height)) / (2 * eps)
    np.testing.assert_allclose(jacobians[0], numeric, atol=3e-9)
    numeric_bias = (pose_height_error(pose, bias + eps, height)
                    - pose_height_error(pose, bias - eps, height)) / (2 * eps)
    np.testing.assert_allclose(jacobians[1], numeric_bias.reshape(1, 1), atol=3e-9)
    # Exercise the actual CustomFactor callback and GTSAM whitening/linearization.
    cameras, _, _ = _scene(sigma=2.)
    observation = select_barometer(cameras, _barometer()).observations[0]
    x, b = gtsam.symbol("x", 0), gtsam.symbol("b", 0)
    values = gtsam.Values()
    values.insert(x, pose)
    values.insert(b, bias)
    factor = make_height_factor(gtsam, x, b, observation, gtsam.noiseModel.mEstimator.Huber.Create(1.345))
    np.testing.assert_allclose(factor.unwhitenedError(values), residual)
    linearized = factor.linearize(values)
    A, rhs = linearized.jacobian()
    # key order is [pose, bias]; the low residual is inside the Huber quadratic.
    np.testing.assert_allclose(A, np.column_stack((numeric, jacobians[1])) / 2., atol=2e-9)
    np.testing.assert_allclose(rhs, -residual / 2.)
    assert factor.keys() == [x, b]

    # Outside the quadratic region the real factor uses Huber, not a Gaussian.
    values.update(b, bias + 20.)
    normalized = abs(float(factor.unwhitenedError(values)[0])) / observation.sigma_m
    assert factor.error(values) == pytest.approx(1.345 * (normalized - .5 * 1.345))


def test_ba_appends_only_one_bias_and_correct_sparse_dependencies_to_actual_scipy_call(monkeypatch):
    scipy = pytest.importorskip("scipy.optimize")
    actual_solver = scipy.least_squares
    cameras, sparse, _ = _scene()
    inspected = []

    def inspect(fun, initial, **kwargs):
        baseline_variables = 6 * 4 + 3 * 12
        baseline_rows = 2 * 4 * 12 + 3 * 4 + 3
        pattern = kwargs["jac_sparsity"].toarray()
        assert initial.shape == (baseline_variables + 1,)
        assert pattern.shape == (baseline_rows + 4, baseline_variables + 1)
        initial_bias = np.median([c["barometer"]["height_m"] - c["center"][2] for c in cameras])
        assert initial[-1] == initial_bias
        assert not pattern[:baseline_rows, -1].any()
        for index, camera in enumerate(cameras):
            row = baseline_rows + index
            dependencies = np.flatnonzero(pattern[row])
            np.testing.assert_array_equal(dependencies, [6 * index + 5, baseline_variables])
            assert fun(initial)[row] == pytest.approx((camera["center"][2] + initial_bias - camera["barometer"]["height_m"]) / .02)
            # Independent perturbations verify normalized actual residual slopes.
            for column in dependencies:
                modified = initial.copy()
                modified[column] += 1e-5
                difference = (fun(modified) - fun(initial)) / 1e-5
                assert difference[row] == pytest.approx(1 / .02, rel=1e-8)
                np.testing.assert_allclose(difference[baseline_rows:][np.arange(4) != index],
                                           1 / .02 if column == baseline_variables else 0., atol=1e-7)
        assert kwargs["loss"] == "huber" and kwargs["f_scale"] == _spine().robust_delta
        assert kwargs["jac"] == "3-point"
        inspected.append(True)
        return actual_solver(fun, initial, **kwargs)

    monkeypatch.setattr(scipy, "least_squares", inspect)
    _, _, metadata = _solve("ba", cameras, sparse, _barometer())
    assert inspected == [True] and metadata["status"] == "optimized", metadata


@pytest.mark.parametrize("kind", ["graph", "ba"])
def test_optional_height_does_not_replace_mandatory_gnss_support(kind):
    cameras, sparse, _ = _scene()
    cameras[0]["gps_inlier"] = cameras[1]["gps_inlier"] = False
    output, refined, metadata = _solve(kind, cameras, sparse, _barometer())
    assert metadata["status"] in ("unsupported_retained_input", "failed_retained_input")
    assert not metadata["optimizer_invoked"]
    assert metadata["barometer"]["factors_added"] == 0
    assert "GNSS" in metadata["reason"]
    _assert_unused(metadata)
    assert [c["center"] for c in output] == [c["center"] for c in cameras]
    for key in sparse:
        np.testing.assert_array_equal(refined[key], sparse[key])


@pytest.mark.parametrize("scale", [-1., .0001])
def test_ba_cheirality_rejects_even_a_cost_reducing_result_with_active_barometer(scale, monkeypatch):
    scipy = pytest.importorskip("scipy.optimize")
    cameras, sparse, _ = _scene()
    shift = np.array([.25, .05, .08])
    for camera in cameras:
        camera["gps"] = (np.asarray(camera["center"]) * scale + shift).tolist()
        camera["gps_sample"] = {"sigma_h_m": .001, "sigma_v_m": .001}
        camera["barometer"]["height_m"] = camera["gps"][2] + 41.7

    def invalid_solver(fun, initial, **kwargs):
        state = initial.copy()
        poses = state[:24].reshape(-1, 6)
        poses[:, 3:] = poses[:, 3:] * scale + shift
        # The appended bias must not be interpreted as a point coordinate.
        state[24:-1] = (state[24:-1].reshape(-1, 3) * scale + shift).ravel()
        state[-1] = 41.7
        residual = fun(state)
        return SimpleNamespace(x=state, fun=residual, cost=.5 * np.dot(residual, residual),
                               nfev=1, success=True, status=1, message="constructed invalid cheirality")

    monkeypatch.setattr(scipy, "least_squares", invalid_solver)
    output, refined, metadata = _solve("ba", cameras, sparse, _barometer())
    assert metadata["status"] == "failed_retained_input", metadata
    assert metadata["final_robust_cost"] < metadata["initial_robust_cost"]
    assert "behind/too near" in metadata["reason"]
    assert output == cameras
    for key in sparse:
        assert refined[key].tobytes() == sparse[key].tobytes()
    _assert_unused(metadata)
    assert metadata["barometer"]["factors_rejected"] == 4


def test_ba_active_fusion_preserves_support_confidence_and_clears_obsolete_covariance():
    cameras, sparse, _ = _scene()
    output, refined, metadata = _solve("ba", cameras, sparse, _barometer(), max_points=3)
    assert metadata["status"] == "optimized", metadata
    indices = metadata["selected_point_indices"]
    assert len(indices) == 3 and metadata["baro_used"]
    unsupported = sorted(set(range(len(sparse["points"]))) - set(indices))
    np.testing.assert_array_equal(refined["confidence"][unsupported], 0.)
    np.testing.assert_array_equal(refined["points"][unsupported], sparse["points"][unsupported])
    for point_index in indices:
        errors = []
        for camera in output:
            xyz = np.asarray(camera["R_cw"]) @ (refined["points"][point_index] - camera["center"])
            uv = xyz[:2] / xyz[2] * 140. + [120., 100.]
            errors.append(np.linalg.norm(uv - camera["observations"][point_index]["uv"]))
        rmse = np.sqrt(np.mean(np.square(errors)))
        assert refined["confidence"][point_index] == pytest.approx((1. - 1. / 4) / (1. + rmse))
    assert metadata["covariance_cleared_cameras"] == 4
    for camera in output:
        assert "pose_covariance_tangent" not in camera and "covariance_domain" not in camera
    for key in ("colors", "track_length", "inferred"):
        np.testing.assert_array_equal(refined[key], sparse[key])


@pytest.mark.parametrize("settings", [
    _barometer(enabled=1), _barometer(min_measurements=1), _barometer(min_measurements=True),
    _barometer(max_time_error_s=float("nan")), _barometer(sigma_m=-1.),
])
def test_malformed_optional_settings_skip_fusion_but_run_baseline(settings):
    cameras, sparse, _ = _scene()
    output, refined, metadata = _solve("ba", cameras, sparse, settings)
    baseline, baseline_sparse, _ = _solve("ba", cameras, sparse, None)
    assert metadata["status"] == "optimized" and output == baseline
    assert metadata["barometer"]["status"] == "unavailable"
    assert "invalid optional barometer settings" in metadata["barometer"]["reason"]
    _assert_unused(metadata)
    for key in sparse:
        np.testing.assert_array_equal(refined[key], baseline_sparse[key])


@pytest.mark.parametrize("kind", ["graph", "ba"])
def test_typed_ingest_alignment_fields_are_consumable_by_both_real_solvers(kind, tmp_path):
    from drishti.config.models import BarometerConfig
    from drishti.io.barometer import BarometerAligner, parse_barometer_csv

    cameras, sparse, truth = _scene()
    path = tmp_path / "constructed-height-records.csv"
    path.write_text("timestamp,height_m\n" + "".join(f"{1000 + i},{center[2] + 41.7}\n"
                                                    for i, center in enumerate(truth)))
    settings = BarometerConfig(enabled=True, time_origin_s=1000., sigma_m=.02)
    samples = parse_barometer_csv(path, settings)
    aligner = BarometerAligner(samples, settings)
    for camera in cameras:
        observation, reason = aligner.at(camera["time_s"])
        assert reason is None and observation is not None
        camera["barometer"] = observation
    _, _, metadata = _solve(kind, cameras, sparse, settings)
    assert metadata["status"] == "optimized" and metadata["baro_used"], metadata
    assert metadata["barometer"]["factors_accepted"] == 4
    assert {item["sigma_source"] for item in metadata["barometer"]["sigma_used"]} == {"barometer.sigma_m"}


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("case,reason", [
    ("normalized_time", "time_s inconsistent"),
    ("source_time", "time_s inconsistent"),
    ("signed_error", "time_error_s inconsistent"),
    ("wrong_error_sign", "time_error_s inconsistent"),
    ("changed_origin", "time_s inconsistent"),
    ("changed_offset", "time_error_s inconsistent"),
    ("changed_assumed_sigma", "assumed sigma_m differs"),
    ("changed_time_bound", "exceeds barometer.max_time_error_s"),
])
def test_tampered_clocks_and_changed_settings_on_saved_observations_preserve_baseline(kind, case, reason):
    cameras, sparse, _ = _scene()
    settings = _barometer()
    assert len(select_barometer(cameras, settings).observations) == 4
    if case == "changed_origin":
        settings.time_origin_s += .01
    elif case == "changed_offset":
        settings.time_offset_s += .01
    elif case == "changed_assumed_sigma":
        for camera in cameras:
            camera["barometer"]["sigma_source"] = "barometer.sigma_m"
        settings.sigma_m = .03
    else:
        for camera in cameras:
            observation = camera["barometer"]
            if case == "normalized_time":
                observation["time_s"] += .01
                observation["time_error_s"] = .01  # internally aligned, but wrong raw normalization
            elif case == "source_time":
                observation["source_time_s"] += 1e-9  # far below the user gap, above arithmetic roundoff
            elif case == "signed_error":
                observation["time_error_s"] = 1e-9
            elif case == "wrong_error_sign":
                camera["time_s"] += .01
                observation.update(frame_time_s=camera["time_s"], time_error_s=.01)
            elif case == "changed_time_bound":
                camera["time_s"] += .02
                observation.update(frame_time_s=camera["time_s"], time_error_s=-.02)
        if case == "changed_time_bound":
            assert len(select_barometer(cameras, settings).observations) == 4
            settings.max_time_error_s = .01
    metadata = _assert_baseline_preserved(kind, cameras, sparse, settings)
    excluded = metadata["barometer"]["excluded"]
    assert len(excluded) == 4 and all(reason in item["reason"] for item in excluded)


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("field,value", [
    ("time_origin_s", None), ("time_origin_s", float("nan")), ("time_origin_s", float("inf")),
    ("time_origin_s", True), ("time_origin_s", "1000"),
    ("time_offset_s", None), ("time_offset_s", float("nan")), ("time_offset_s", float("inf")),
    ("time_offset_s", False), ("time_offset_s", "0"),
])
def test_active_settings_require_finite_explicit_clock_without_blocking_baseline(kind, field, value):
    cameras, sparse, _ = _scene()
    metadata = _assert_baseline_preserved(kind, cameras, sparse, _barometer(**{field: value}))
    assert "invalid optional barometer settings" in metadata["barometer"]["reason"]
    assert f"barometer.{field}" in metadata["barometer"]["reason"]


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("field", ["time_origin_s", "time_offset_s"])
def test_missing_active_clock_fields_preserve_baseline(kind, field):
    cameras, sparse, _ = _scene()
    settings = _barometer()
    delattr(settings, field)
    metadata = _assert_baseline_preserved(kind, cameras, sparse, settings)
    assert field in metadata["barometer"]["reason"]


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("offset", [0., -.125])
def test_literal_signed_error_roundoff_is_accepted_with_current_clock_and_explicit_offset(kind, offset):
    cameras, sparse, _ = _scene()
    for index, camera in enumerate(cameras):
        frame = index + .01 - offset
        camera["time_s"] = frame
        camera["barometer"].update(frame_time_s=frame, time_error_s=-.01)
    before = copy.deepcopy(cameras)
    # For later frames the recomputed difference is not exactly the stored literal.
    assert cameras[-1]["barometer"]["time_s"] - (cameras[-1]["time_s"] + offset) != -.01
    settings = _barometer(time_offset_s=offset)
    selection = select_barometer(cameras, settings)
    assert len(selection.observations) == 4 and not selection.metadata["excluded"]
    _, _, metadata = _solve(kind, cameras, sparse, settings)
    assert metadata["status"] == "optimized" and metadata["baro_used"], metadata
    assert metadata["barometer"]["factors_accepted"] == 4
    clock = metadata["barometer"]["clock_validation"]
    assert clock["time_origin_s"] == 1000. and clock["time_offset_s"] == offset
    assert clock["time_error_bound"] == "strict_on_recorded_and_recomputed_error_without_tolerance"
    assert "sum_binary64_ulps" in clock["comparison"]
    assert cameras == before  # Validation never repairs or rewrites recorded values.


@pytest.mark.parametrize("origin", [1000., 1700000000.])
def test_normalized_clock_allows_only_binary64_roundoff_not_epoch_scaled_tolerance(origin):
    cameras, _, _ = _scene()
    for index, camera in enumerate(cameras):
        normalized = index + .01
        camera["time_s"] = normalized
        camera["barometer"].update(source_time_s=origin + normalized, time_s=normalized,
                                   frame_time_s=normalized, time_error_s=0.)
    settings = _barometer(time_origin_s=origin)
    assert len(select_barometer(cameras, settings).observations) == 4
    for camera in cameras:
        camera["barometer"]["time_s"] += .001
        camera["barometer"]["time_error_s"] = .001
    selection = select_barometer(cameras, settings)
    assert not selection.observations and selection.metadata["distinct_measurements"] == 0
    assert all("time_s inconsistent" in item["reason"] for item in selection.metadata["excluded"])


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("which", ["recorded", "recomputed", "source_clock"])
def test_clock_roundoff_tolerance_never_extends_configured_timing_bound(kind, which):
    cameras, sparse, _ = _scene()
    for index, camera in enumerate(cameras):
        time = index + .1 if which == "source_clock" else 2. + index
        frame = float(index) if which == "source_clock" else time - .1
        camera["time_s"] = frame
        camera["barometer"].update(source_time_s=1000. + time, time_s=time, frame_time_s=frame,
                                   time_error_s=math.nextafter(.1, math.inf) if which == "recorded" else .1)
    metadata = _assert_baseline_preserved(kind, cameras, sparse, _barometer())
    exclusions = metadata["barometer"]["excluded"]
    assert len(exclusions) >= 3
    assert all("exceeds barometer.max_time_error_s" in item["reason"] for item in exclusions)
    if which == "source_clock":
        assert len(exclusions) == 4
        assert "source-clock recomputed" in exclusions[0]["reason"]
        # The stored normalized time and error are exactly .1 for camera 0,
        # but its raw timestamp yields .10000000000002274 after normalization.
        assert cameras[0]["barometer"]["time_error_s"] == .1
        assert cameras[0]["barometer"]["source_time_s"] - 1000. > .1


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("field", ["height_m", "sigma_m", "sigma_source", "source_time_s", "source_line"])
def test_contradictory_raw_record_under_reused_source_id_excludes_every_association(kind, field):
    cameras, sparse, _ = _scene()
    _reuse_first_source(cameras)
    observation = cameras[1]["barometer"]
    if field == "sigma_source":
        observation[field] = "barometer.sigma_m"  # Both sources individually valid; provenance conflicts.
    elif field == "source_time_s":
        observation[field] += .01
        observation["time_s"] += .01
        observation["time_error_s"] = observation["time_s"] - observation["frame_time_s"]
    else:
        observation[field] += 1 if field != "sigma_m" else .01
    metadata = _assert_baseline_preserved(kind, cameras, sparse, _barometer())
    barometer = metadata["barometer"]
    assert barometer["conflicting_source_indices"] == [0]
    assert {item["source_index"] for item in barometer["selected"]} == {2, 3}
    assert barometer["distinct_measurements"] == 2
    assert len(barometer["excluded"]) == 2
    assert all("conflicting raw provenance" in item["reason"] and field in item["reason"]
               for item in barometer["excluded"])
    assert {item["camera_index"] for item in barometer["excluded"]} == {0, 1}
    # Quarantining a contradictory record must not depend on traversal order.
    reversed_selection = select_barometer(list(reversed(cameras)), _barometer())
    assert reversed_selection.metadata["conflicting_source_indices"] == [0]
    assert {item["source_index"] for item in reversed_selection.metadata["selected"]} == {2, 3}


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("case", ["identical_raw", "different_height", "bad_alignment", "nonfinite_alignment"])
def test_same_raw_timestamp_with_conflicting_ids_cannot_become_independent_factors(kind, case):
    cameras, sparse, _ = _scene()
    _reuse_first_source(cameras)
    cameras[1]["barometer"]["source_index"] = 1
    if case == "different_height":
        cameras[1]["barometer"]["height_m"] += 1.
    elif case == "bad_alignment":
        cameras[1]["barometer"]["time_error_s"] = .02
    elif case == "nonfinite_alignment":
        cameras[1]["barometer"]["time_error_s"] = None
    metadata = _assert_baseline_preserved(kind, cameras, sparse, _barometer())
    barometer = metadata["barometer"]
    assert barometer["conflicting_source_indices"] == [0, 1]
    assert barometer["duplicate_source_times_s"] == [1000.]
    assert barometer["distinct_measurements"] == 2
    assert {item["source_index"] for item in barometer["selected"]} == {2, 3}
    assert any("source_time_s reused under different" in item["reason"] for item in barometer["excluded"])


@pytest.mark.parametrize("kind", ["graph", "ba"])
@pytest.mark.parametrize("bad_clock", ["inconsistent_normalization", "nonfinite_signed_error"])
def test_raw_identity_conflict_is_not_hidden_by_invalid_clock_on_the_contradictory_association(kind, bad_clock):
    cameras, sparse, _ = _scene()
    _reuse_first_source(cameras)
    cameras[1]["barometer"]["source_time_s"] += 1.
    # Deliberately leave normalized time unmodified, so this association also
    # has an invalid clock. Its contradictory identity still invalidates ID 0.
    if bad_clock == "nonfinite_signed_error":
        cameras[1]["barometer"]["time_error_s"] = None
    metadata = _assert_baseline_preserved(kind, cameras, sparse, _barometer())["barometer"]
    assert metadata["conflicting_source_indices"] == [0]
    assert metadata["distinct_measurements"] == 2
    assert {item["source_index"] for item in metadata["selected"]} == {2, 3}
    reason = "time_s inconsistent" if bad_clock == "inconsistent_normalization" else "must be a finite number"
    assert any(reason in item["reason"] for item in metadata["excluded"])
    assert any("conflicting raw provenance" in item["reason"] for item in metadata["excluded"])


@pytest.mark.parametrize("kind", ["graph", "ba"])
def test_sufficient_clean_distinct_support_can_fuse_after_conflicting_ids_are_excluded(kind):
    cameras, sparse, _ = _scene()
    _reuse_first_source(cameras)
    cameras[1]["barometer"]["source_index"] = 1
    _, _, metadata = _solve(kind, cameras, sparse, _barometer(min_measurements=2))
    assert metadata["status"] == "optimized" and metadata["baro_used"], metadata
    barometer = metadata["barometer"]
    assert barometer["factors_added"] == barometer["factors_accepted"] == 2
    assert {item["source_index"] for item in barometer["selected"]} == {2, 3}
    assert barometer["conflicting_source_indices"] == [0, 1]
