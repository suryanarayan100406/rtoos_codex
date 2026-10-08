"""Executed visual+GNSS GTSAM pose graphs; honest retained-pose degradation."""

from __future__ import annotations

import copy
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from typing import Any

from drishti.errors import ReconstructionError
from drishti.recon.barometer import make_height_factor, select_barometer


def _edges(cameras: list[dict[str, Any]], global_refinement: bool) -> list[tuple[int, int]]:
    edges = {(index - 1, index) for index in range(1, len(cameras))}
    if global_refinement:
        observers: dict[int, list[int]] = {}
        for index, camera in enumerate(cameras):
            for observation in camera.get("observations", []):
                observers.setdefault(int(observation["point_index"]), []).append(index)
        for indices in observers.values():
            edges.update(combinations(sorted(set(indices)), 2))
    return sorted(edges)


def optimize_pose_graph(cameras: list[dict[str, Any]], settings: Any, *,
                        global_refinement: bool = False,
                        barometer_settings: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """SE3 visual/GNSS graph with optional recorded heights and one free bias.

    Visual measurements remain the original Sim3-transformed SfM poses, rather
    than becoming last iteration's output. IMU/baro are not prerequisites.
    """
    import numpy as np

    retained = copy.deepcopy(cameras)
    barometer = select_barometer(cameras, barometer_settings)
    mode = "global_covisibility_visual_GNSS_pose_graph" if global_refinement else "sequential_visual_GNSS_pose_graph"
    metadata: dict[str, Any] = {"method": mode, "optimizer_invoked": False,
                               "status": "disabled_retained_input" if not settings.optimize else "pending",
                               "poses_retained": True, "imu_used": False, "baro_used": False,
                               "barometer": barometer.metadata,
                              "accuracy_validated": False, "measured_gnss_camera_lever_arm_m": None,
                              "gnss_reference": "logged_position_attached_to_camera_centre_without_measured_lever_arm"}
    if not settings.optimize:
        metadata["reason"] = "spine.optimize=false"
        barometer.reject(metadata["reason"])
        return retained, metadata
    try:
        import gtsam
    except (ImportError, OSError) as exc:
        metadata.update(status="unavailable_retained_input", reason=f"GTSAM unavailable: {exc}")
        barometer.reject(metadata["reason"])
        return retained, metadata
    try:
        metadata["gtsam_version"] = version("gtsam")
    except PackageNotFoundError:
        metadata["gtsam_version"] = None
    if len(cameras) < 3:
        raise ReconstructionError("Metric pose graph requires at least three cameras")
    try:
        keys = [gtsam.symbol("x", index) for index in range(len(cameras))]
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        reference_poses = []
        for key, camera in zip(keys, cameras, strict=True):
            pose = gtsam.Pose3(gtsam.Rot3(np.asarray(camera["R_cw"], dtype=np.float64).T),
                               np.asarray(camera["center"], dtype=np.float64))
            reference = camera.get("visual_reference", camera)
            measured = gtsam.Pose3(gtsam.Rot3(np.asarray(reference["R_cw"], dtype=np.float64).T),
                                   np.asarray(reference["center"], dtype=np.float64))
            initial.insert(key, pose)
            reference_poses.append(measured)
        robust = gtsam.noiseModel.mEstimator.Huber.Create(settings.robust_delta)
        visual_diagonal = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [settings.rotation_sigma_rad] * 3 + [settings.visual_sigma_m] * 3, dtype=np.float64))
        visual_noise = gtsam.noiseModel.Robust.Create(robust, visual_diagonal)
        edges = _edges(cameras, global_refinement)
        for left, right in edges:
            graph.add(gtsam.BetweenFactorPose3(keys[left], keys[right],
                                              reference_poses[left].between(reference_poses[right]), visual_noise))
        gnss_factors = 0
        for index, camera in enumerate(cameras):
            if camera.get("gps") is None or not camera.get("gps_inlier", True):
                continue
            sample = camera.get("gps_sample", {})
            horizontal = sample.get("sigma_h_m", settings.gnss_sigma_h_m)
            vertical = sample.get("sigma_v_m", settings.gnss_sigma_v_m)
            noise = gtsam.noiseModel.Robust.Create(robust, gtsam.noiseModel.Diagonal.Sigmas(
                np.array([horizontal, horizontal, vertical], dtype=np.float64)))
            graph.add(gtsam.GPSFactor(keys[index], np.asarray(camera["gps"], dtype=np.float64), noise))
            gnss_factors += 1
        if gnss_factors < 3:
            raise ReconstructionError("Metric pose graph lacks three supported GNSS factors")
        # Relative visual poses leave a global rotation gauge. GNSS positions
        # plus this explicitly recorded visual-alignment rotation prior close it.
        rotation_noise = gtsam.noiseModel.Isotropic.Sigma(3, settings.rotation_sigma_rad)
        graph.add(gtsam.PoseRotationPrior3D(keys[0], reference_poses[0], rotation_noise))
        bias_key = gtsam.symbol("b", 0)
        if barometer.observations:
            initial.insert(bias_key, barometer.initial_bias([camera["center"] for camera in cameras]))
            barometer.metadata["factors_attempted"] = len(barometer.observations)
            for observation in barometer.observations:
                graph.add(make_height_factor(gtsam, keys[observation.camera_index], bias_key,
                                             observation, robust))
                barometer.metadata["factors_added"] += 1
        parameters = gtsam.LevenbergMarquardtParams()
        parameters.setMaxIterations(settings.max_iterations)
        initial_error = float(graph.error(initial))
        metadata.update(optimizer_invoked=True, visual_factors=len(edges), gnss_factors=gnss_factors,
                        rotation_prior="first_Sim3_visual_rotation", initial_error=initial_error)
        barometer.metadata["optimizer_invoked"] = bool(barometer.metadata["factors_added"])
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, parameters)
        result = optimizer.optimize()
        final_error = float(graph.error(result))
        if not np.isfinite(final_error) or final_error > initial_error + np.finfo(float).eps * max(1, initial_error) * 100:
            raise RuntimeError("Optimizer objective is non-finite or increased")
        output = copy.deepcopy(cameras)
        for index, camera in enumerate(output):
            pose = result.atPose3(keys[index])
            center = np.asarray(pose.translation(), dtype=np.float64)
            rotation = np.asarray(pose.rotation().matrix(), dtype=np.float64).T
            if not np.isfinite(center).all() or not np.isfinite(rotation).all():
                raise RuntimeError("Optimizer produced non-finite camera poses")
            camera["center"], camera["R_cw"] = center.tolist(), rotation.tolist()
            camera["pose_method"] = mode
        covariance_status = "unavailable"
        try:
            marginals = gtsam.Marginals(graph, result)
            for index, camera in enumerate(output):
                covariance = np.asarray(marginals.marginalCovariance(keys[index]), dtype=np.float64)
                if not np.isfinite(covariance).all():
                    raise RuntimeError("non-finite marginal covariance")
                camera["pose_covariance_tangent"] = covariance.tolist()
                camera["covariance_domain"] = "conditional_Pose3_rotation_translation_tangent_assumed_factor_noise"
            covariance_status = "computed_conditional_on_factor_noise"
        except (RuntimeError, ValueError) as exc:
            metadata["covariance_reason"] = str(exc)
        residuals = [np.linalg.norm(np.asarray(c["center"]) - np.asarray(c["gps"]))
                     for c in output if c.get("gps") is not None and c.get("gps_inlier", True)]
        if barometer.observations:
            barometer.accept([camera["center"] for camera in output], float(result.atDouble(bias_key)))
        metadata.update(status="optimized", poses_retained=False, final_error=final_error,
                        iterations=optimizer.iterations(), covariance_status=covariance_status,
                        gnss_fit_rmse_m=float(np.sqrt(np.mean(np.square(residuals)))),
                        fit_is_independent_accuracy=False)
        metadata["baro_used"] = bool(barometer.observations)
        return output, metadata
    except (RuntimeError, ValueError, TypeError, AttributeError, OverflowError, ReconstructionError) as exc:
        metadata.update(status="failed_retained_input", reason=f"GTSAM optimization failed: {exc}",
                        poses_retained=True, baro_used=False)
        barometer.reject(metadata["reason"])
        for camera in retained:
            camera["pose_method"] = "retained_input_after_optimizer_failure"
        return retained, metadata


def refine_sparse_points(cameras: list[dict[str, Any]], sparse: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Triangulate observed sparse tracks using the final metric pinhole cameras."""
    import numpy as np

    output = {key: np.array(value, copy=True) for key, value in sparse.items()}
    observations: dict[int, list[tuple[Any, Any, Any]]] = {}
    for camera in cameras:
        rotation = np.asarray(camera["R_cw"], dtype=np.float64)
        center = np.asarray(camera["center"], dtype=np.float64)
        matrix = np.asarray(camera["K"], dtype=np.float64)
        projection = matrix @ np.column_stack((rotation, -rotation @ center))
        for observation in camera.get("observations", []):
            observations.setdefault(int(observation["point_index"]), []).append(
                (projection, np.asarray(observation["uv"], dtype=np.float64), camera))
    updated = 0
    retained = 0
    invalidated = 0

    def invalidate(index: int) -> None:
        nonlocal invalidated
        if "confidence" in output:
            output["confidence"][index] = 0
        if "reprojection_error_px" in output:
            output["reprojection_error_px"][index] = np.inf
        invalidated += 1

    for index, track in observations.items():
        if len(track) < 2:
            retained += 1
            invalidate(index)
            continue
        rows = []
        for projection, uv, _ in track:
            rows.extend([uv[0] * projection[2] - projection[0], uv[1] * projection[2] - projection[1]])
        _, singular, vt = np.linalg.svd(np.asarray(rows))
        homogeneous = vt[-1]
        if singular[-2] <= np.finfo(float).eps * singular[0] or abs(homogeneous[3]) <= np.finfo(float).eps:
            retained += 1
            invalidate(index)
            continue
        point = homogeneous[:3] / homogeneous[3]
        errors: list[float] = []
        for projection, uv, camera in track:
            projected = projection @ np.append(point, 1)
            depth = (np.asarray(camera["R_cw"]) @ (point - np.asarray(camera["center"])))[2]
            if depth <= 0 or not np.isfinite(projected).all():
                errors = []
                break
            errors.append(float(np.linalg.norm(projected[:2] / projected[2] - uv)))
        if not errors or not np.isfinite(point).all():
            retained += 1
            invalidate(index)
            continue
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        output["points"][index] = point
        output["confidence"][index] = (1 - 1 / len(track)) / (1 + rmse)
        if "reprojection_error_px" in output:
            output["reprojection_error_px"][index] = rmse
        updated += 1
    return output, {"sparse_refinement_method": "linear_multiview_triangulation_final_poses",
                     "retriangulated_points": updated, "retained_sparse_points": len(output["points"]) - updated,
                     "failed_or_unsupported_tracks": retained,
                     "confidence_invalidated_points": invalidated,
                     "point_confidence_method": "observed_support_times_inverse_one_plus_reprojection_rmse"}
