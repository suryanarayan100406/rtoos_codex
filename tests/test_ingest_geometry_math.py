"""Constructed geometry units validate conventions, not reconstruction acceptance."""

import copy
import json
import sys

import numpy as np
import pytest

from drishti.config.models import CRSConfig, SpineConfig
from drishti.errors import InputError, ReconstructionError
from drishti.geo.alignment import robust_similarity, umeyama
from drishti.geo.crs import derive_epsg, project_gps, utm_epsg
from drishti.recon.sfm import load_intrinsics, recompute_anchors
from drishti.recon.spine import optimize_pose_graph, refine_sparse_points


@pytest.mark.parametrize("latitude,longitude,epsg", [
    (0, -180, 32601), (0, 180, 32660), (-1, 0, 32731), (0, 6, 32632),
    (12, 77, 32643), (60, 6, 32632), (75, 10, 32633), (84, 40, 32637), (-80, 0, 32731),
])
def test_utm_boundaries_hemispheres_and_official_exceptions(latitude, longitude, epsg):
    assert utm_epsg(latitude, longitude) == epsg


@pytest.mark.parametrize("latitude", [-80.001, 84.001])
def test_high_latitude_demands_explicit_override(latitude):
    with pytest.raises(InputError, match="explicit"):
        utm_epsg(latitude, 0)


def test_crs_median_projection_local_float64_and_source_height():
    pytest.importorskip("pyproj")
    samples = [{"latitude": -12.0, "longitude": 77.0 + i * 0.001, "altitude_m": 100.0 + i} for i in range(3)]
    epsg = derive_epsg(samples, CRSConfig())
    assert epsg == 32743
    local, origin = project_gps(samples, epsg)
    assert local.dtype == np.float64 and origin.dtype == np.float64
    assert origin[2] == 101 and np.array_equal(local[:, 2], [-1, 0, 1])
    repeated, repeated_origin = project_gps(samples, epsg, origin=origin)
    np.testing.assert_allclose(repeated, local)
    np.testing.assert_allclose(repeated_origin, origin)


def test_explicit_crs_must_be_projected_metres_and_dateline_is_ambiguous():
    pytest.importorskip("pyproj")
    samples = [{"latitude": 0, "longitude": 179, "altitude_m": 0},
               {"latitude": 0, "longitude": -179, "altitude_m": 0}]
    with pytest.raises(InputError, match="antimeridian"):
        derive_epsg(samples, CRSConfig())
    assert derive_epsg(samples, CRSConfig(mode="epsg", epsg=3857)) == 3857
    with pytest.raises(InputError, match="projected"):
        derive_epsg(samples, CRSConfig(mode="epsg", epsg=4326))
    with pytest.raises(InputError, match="geoid"):
        project_gps(samples, 3857, settings=CRSConfig(vertical="orthometric"), source_datum="ellipsoidal")


def test_umeyama_transform_and_camera_rotation_direction():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [1., 1., 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([12., -8., 3.])
    target = 2.5 * source @ rotation.T + translation
    transform = umeyama(source, target)
    np.testing.assert_allclose(transform.apply(source), target, atol=1e-12)
    assert transform.scale == pytest.approx(2.5)
    assert np.linalg.det(transform.rotation) == pytest.approx(1)
    point, center = np.array([1., 2., 5.]), np.array([0., 0., 1.])
    camera_rotation = np.eye(3)
    new_rotation = camera_rotation @ transform.rotation.T
    np.testing.assert_allclose(new_rotation @ (transform.apply(point) - transform.apply(center)),
                               transform.scale * camera_rotation @ (point - center))


def test_ransac_rejects_outlier_with_seeded_repeatable_result():
    source = np.random.default_rng(42).normal(size=(30, 3))
    target = source * 4 + [10, 20, 30]
    target[-1] += [100, -100, 80]
    settings = SpineConfig(alignment_threshold_m=0.01, ransac_iterations=100)
    transform, metrics = robust_similarity(source, target, settings, seed=19)
    repeat, repeat_metrics = robust_similarity(source, target, settings, seed=19)
    assert metrics["inliers"] == 29 and not metrics["inlier_mask"][-1]
    assert metrics["fit_is_independent_accuracy"] is False
    assert metrics == repeat_metrics
    np.testing.assert_allclose(transform.rotation, repeat.rotation)
    assert transform.scale == pytest.approx(4)


def test_similarity_rejects_unobservable_line_and_flags_low_rank_plane():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    with pytest.raises(ReconstructionError, match="unobservable"):
        umeyama(source, source * 2)
    plane = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [1., 1., 0.]])
    _, metrics = robust_similarity(plane, plane * 4, SpineConfig(), seed=123)
    assert any("low_3d_rank" in warning for warning in metrics["warnings"])
    with pytest.raises(ReconstructionError, match="baseline"):
        robust_similarity(plane, plane * 0.01, SpineConfig(), seed=123)


def camera(center, uv=None):
    result = {"frame_id": "unit", "R_cw": np.eye(3).tolist(), "center": list(center),
              "K": [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]], "width": 100,
              "height": 100, "gps": list(center), "observations": []}
    if uv is not None:
        result["observations"] = [{"point_index": 0, "uv": uv}]
    return result


def test_anchor_uses_final_metric_camera_z_and_observed_uv_residual():
    cams = [camera([0., 0., 0.], [72., 50.])]
    metrics = recompute_anchors(cams, [[2., 0., 10.]])
    assert cams[0]["anchors"] == [[72., 50., 10., 2.]]
    assert metrics["anchor_reprojection_rmse_px"] == 2
    cams[0]["center"][2] = 2
    recompute_anchors(cams, [[2., 0., 10.]])
    assert cams[0]["anchors"][0][2] == 8


def test_sparse_retriangulation_uses_actual_observation_rays():
    cams = [camera([0., 0., 0.], [60., 50.]), camera([2., 0., 0.], [40., 50.])]
    sparse = {"points": np.array([[7., 2., 5.]]), "colors": np.array([[1, 2, 3]], dtype=np.uint8),
              "confidence": np.array([0.2]), "inferred": np.array([False])}
    result, metrics = refine_sparse_points(cams, sparse)
    np.testing.assert_allclose(result["points"], [[1., 0., 10.]], atol=1e-12)
    np.testing.assert_allclose(sparse["points"], [[7., 2., 5.]])
    assert metrics["retriangulated_points"] == 1
    assert result["confidence"][0] == pytest.approx(0.5)
    assert not result["inferred"][0]


def test_sparse_retriangulation_invalidates_unsupported_confidence():
    cams = [camera([0., 0., 0.], [60., 50.])]
    sparse = {"points": np.array([[7., 2., 5.]]), "colors": np.array([[1, 2, 3]], dtype=np.uint8),
              "confidence": np.array([0.7]), "inferred": np.array([False])}
    result, metrics = refine_sparse_points(cams, sparse)
    assert result["confidence"][0] == 0
    if "reprojection_error_px" in result:
        assert np.isinf(result["reprojection_error_px"][0])
    assert metrics["confidence_invalidated_points"] == 1


def graph_cameras():
    cams = [camera([0., 0., 0.]), camera([10., 0., 0.]), camera([10., 10., 0.])]
    cams[1]["gps"][1] += 0.5
    cams[2]["gps"][1] += 0.5
    for item in cams:
        item["observations"] = [{"point_index": 0, "uv": [50., 50.]}]
    return cams


def test_real_gtsam_optimizer_executes_without_imu_and_reduces_objective():
    pytest.importorskip("gtsam")
    inputs = graph_cameras()
    unchanged = copy.deepcopy(inputs)
    result, metadata = optimize_pose_graph(inputs, SpineConfig(), global_refinement=True)
    assert inputs == unchanged
    assert metadata["status"] == "optimized" and metadata["optimizer_invoked"]
    assert metadata["final_error"] < metadata["initial_error"]
    assert metadata["visual_factors"] == 3 and metadata["gnss_factors"] == 3
    assert not metadata["imu_used"] and not metadata["baro_used"]
    assert result[0]["center"] != inputs[0]["center"]
    assert metadata["fit_is_independent_accuracy"] is False
    assert len(result[0]["pose_covariance_tangent"]) == 6


def test_optimizer_failure_retains_real_input_and_labels_it(monkeypatch):
    gtsam = pytest.importorskip("gtsam")
    def fail(*args):
        raise RuntimeError("isolated optimizer failure")
    monkeypatch.setattr(gtsam, "LevenbergMarquardtOptimizer", fail)
    inputs = graph_cameras()
    result, metadata = optimize_pose_graph(inputs, SpineConfig())
    assert metadata["status"] == "failed_retained_input"
    assert metadata["optimizer_invoked"] and metadata["poses_retained"]
    assert [c["center"] for c in result] == [c["center"] for c in inputs]
    assert result[0]["pose_method"] == "retained_input_after_optimizer_failure"


def test_unavailable_gtsam_and_disabled_graph_are_never_claimed(monkeypatch):
    monkeypatch.setitem(sys.modules, "gtsam", None)
    _, metadata = optimize_pose_graph(graph_cameras(), SpineConfig())
    assert metadata["status"] == "unavailable_retained_input" and not metadata["optimizer_invoked"]
    _, metadata = optimize_pose_graph(graph_cameras(), SpineConfig(optimize=False))
    assert metadata["status"] == "disabled_retained_input" and not metadata["optimizer_invoked"]


def test_intrinsics_validation_and_model_parameter_mapping(tmp_path):
    path = tmp_path / "intrinsics.json"
    data = {"width": 100, "height": 100, "K": [[80., 0., 50.], [0., 81., 50.], [0., 0., 1.]],
            "distortion": [0.1, 0.2, 0.01, 0.02, 0.3]}
    path.write_text(json.dumps(data))
    result = load_intrinsics(path, 100, 100)
    assert result["camera_model"] == "FULL_OPENCV"
    assert result["params"] == [80, 81, 50, 50, 0.1, 0.2, 0.01, 0.02, 0.3, 0, 0, 0]
    with pytest.raises(InputError, match="size"):
        load_intrinsics(path, 200, 100)
    data["K"][0][0] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(InputError, match="positive"):
        load_intrinsics(path, 100, 100)
