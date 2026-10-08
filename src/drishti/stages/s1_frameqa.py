"""Measured frame QA and geometry/coverage keyframe admission.

The uncalibrated selector fits a robust homography and measures the residual
left after rotation/planar motion compensation.  That residual is explicitly
only a geometric parallax proxy: a planar scene is not made metric by this
stage.  When a validated, user-provided calibration is available, an essential
matrix/recoverPose path adds a triangulated positive-depth angular-parallax
signal.  Neither path invents camera intrinsics.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from drishti.errors import DependencyError, ReconstructionError
from drishti.stages.base import StageResult


def _matched_points(previous: Any, current: Any, matcher: Any) -> tuple[Any, Any, list[Any]] | None:
    previous_points, previous_descriptors, _, _ = previous
    points, descriptors, _, _ = current
    if previous_descriptors is None or descriptors is None:
        return None
    raw_matches = matcher.match(previous_descriptors, descriptors)
    if raw_matches is None:
        return None
    matches = list(raw_matches)
    if not matches:
        return None
    import numpy as np

    source = np.asarray([previous_points[match.queryIdx].pt for match in matches], dtype=np.float64)
    target = np.asarray([points[match.trainIdx].pt for match in matches], dtype=np.float64)
    return source, target, matches


def _coverage(points: Any, width: int, height: int) -> float:
    cells = set()
    for point in points:
        x, y = point
        cells.add((max(0, min(3, int(x / width * 4))),
                   max(0, min(3, int(y / height * 4)))))
    return len(cells) / 16


def feature_displacement(previous: Any, current: Any, matcher: Any) -> tuple[float | None, int, float]:
    """Return the legacy image-displacement/coverage measurement.

    This function remains API-compatible for callers that need the old
    measurement.  S1 records it for diagnostics but does not use it as its
    geometric admission signal.
    """
    import numpy as np

    observations = _matched_points(previous, current, matcher)
    if observations is None:
        return None, 0, 0.0
    source, target, matches = observations
    _, _, width, height = current
    displacement = np.linalg.norm(source - target, axis=1)
    return float(np.median(displacement)), len(matches), _coverage(target, width, height)


def _finite_quantile(values: Any, quantile: float) -> float | None:
    import numpy as np

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return float(np.quantile(finite, quantile))


def _homography_measurement(source: Any, target: Any, settings: Any, cv2: Any) -> dict[str, Any]:
    import numpy as np

    result: dict[str, Any] = {
        "status": "insufficient_matches",
        "inlier_count": 0,
        "inlier_fraction": 0.0,
        "residual_median_px": None,
        "residual_quantile_px": None,
        "inlier_residual_median_px": None,
        "supported": False,
    }
    if len(source) < settings.min_geometric_matches:
        return result
    try:
        homography, mask = cv2.findHomography(
            source.astype(np.float32), target.astype(np.float32), cv2.RANSAC,
            settings.homography_ransac_threshold_px,
            maxIters=settings.homography_ransac_iterations,
            confidence=settings.homography_ransac_confidence,
        )
        if homography is None or mask is None or not np.isfinite(homography).all():
            result["status"] = "estimation_failed"
            return result
        homogeneous = np.concatenate([source, np.ones((len(source), 1))], axis=1)
        projected_h = homogeneous @ homography.T
        denominator = projected_h[:, 2]
        valid = np.isfinite(projected_h).all(axis=1) & (np.abs(denominator) > 1e-12)
        projected = np.full_like(source, np.nan)
        projected[valid] = projected_h[valid, :2] / denominator[valid, None]
        residuals = np.linalg.norm(projected - target, axis=1)
        residuals[~valid] = np.nan
        inliers = np.asarray(mask, dtype=np.uint8).reshape(-1).astype(bool) & np.isfinite(residuals)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_fraction = inlier_count / len(source)
        result.update(
            status="supported" if (
                inlier_count >= settings.min_geometric_matches
                and inlier_fraction >= settings.homography_min_inlier_fraction
            ) else "weak_support",
            inlier_count=inlier_count,
            inlier_fraction=float(inlier_fraction),
            residual_median_px=_finite_quantile(residuals, 0.5),
            residual_quantile_px=_finite_quantile(residuals, settings.geometric_residual_quantile),
            inlier_residual_median_px=_finite_quantile(residuals[inliers], 0.5),
            supported=bool(
                inlier_count >= settings.min_geometric_matches
                and inlier_fraction >= settings.homography_min_inlier_fraction
            ),
        )
    except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError):
        result["status"] = "estimation_failed"
    return result


def _essential_measurement(
    source: Any,
    target: Any,
    calibration: dict[str, Any] | None,
    settings: Any,
    cv2: Any,
    homography: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    result: dict[str, Any] = {
        "status": "not_available" if calibration is None else "insufficient_matches",
        "inlier_count": 0,
        "inlier_fraction": 0.0,
        "positive_depth_count": 0,
        "positive_depth_fraction": 0.0,
        "angular_parallax_deg": None,
        "supported": False,
        "homography_dominates": False,
    }
    if calibration is None or len(source) < settings.min_geometric_matches:
        return result
    try:
        K = np.asarray(calibration["K"], dtype=np.float64)
        distortion = np.asarray(calibration["distortion"], dtype=np.float64)
        if calibration["camera_model"] == "OPENCV_FISHEYE":
            normalized_source = cv2.fisheye.undistortPoints(
                source.reshape(-1, 1, 2), K, distortion.reshape(-1, 1)
            ).reshape(-1, 2)
            normalized_target = cv2.fisheye.undistortPoints(
                target.reshape(-1, 1, 2), K, distortion.reshape(-1, 1)
            ).reshape(-1, 2)
        else:
            normalized_source = cv2.undistortPoints(
                source.reshape(-1, 1, 2), K, distortion
            ).reshape(-1, 2)
            normalized_target = cv2.undistortPoints(
                target.reshape(-1, 1, 2), K, distortion
            ).reshape(-1, 2)
        focal = float(min(K[0, 0], K[1, 1]))
        essential, estimate_mask = cv2.findEssentialMat(
            normalized_source, normalized_target, np.eye(3), cv2.RANSAC,
            settings.essential_ransac_confidence,
            settings.essential_ransac_threshold_px / focal,
            settings.essential_ransac_iterations,
        )
        if essential is None or estimate_mask is None:
            result["status"] = "estimation_failed"
            return result
        essential_array = np.asarray(essential, dtype=np.float64)
        if essential_array.shape == (3, 3):
            candidates = [essential_array]
        elif essential_array.ndim == 2 and essential_array.shape[1] == 3 and essential_array.shape[0] % 3 == 0:
            candidates = list(essential_array.reshape(-1, 3, 3))
        else:
            result["status"] = "estimation_failed"
            return result
        best: dict[str, Any] | None = None
        identity = np.eye(3, dtype=np.float64)
        for candidate in candidates:
            try:
                _, rotation, translation, pose_mask = cv2.recoverPose(
                    candidate, normalized_source, normalized_target, identity, mask=estimate_mask
                )
                inliers = np.asarray(pose_mask, dtype=np.uint8).reshape(-1).astype(bool)
                if np.count_nonzero(inliers) < 2:
                    continue
                points_h = cv2.triangulatePoints(
                    np.hstack([identity, np.zeros((3, 1))]),
                    np.hstack([rotation, np.asarray(translation, dtype=np.float64).reshape(3, 1)]),
                    normalized_source[inliers].T,
                    normalized_target[inliers].T,
                )
                weights = points_h[3]
                valid_weights = np.isfinite(weights) & (np.abs(weights) > 1e-12)
                points = np.full((3, len(weights)), np.nan, dtype=np.float64)
                points[:, valid_weights] = points_h[:3, valid_weights] / weights[valid_weights]
                second_camera_points = rotation @ points + np.asarray(translation).reshape(3, 1)
                positive = (
                    np.isfinite(points).all(axis=0)
                    & np.isfinite(second_camera_points).all(axis=0)
                    & (points[2] > 1e-8)
                    & (second_camera_points[2] > 1e-8)
                )
                rays_one = np.column_stack([normalized_source[inliers], np.ones(np.count_nonzero(inliers))])
                rays_two = np.column_stack([normalized_target[inliers], np.ones(np.count_nonzero(inliers))])
                rays_two_in_first = (rotation.T @ rays_two.T).T
                rays_one /= np.linalg.norm(rays_one, axis=1, keepdims=True)
                rays_two_in_first /= np.linalg.norm(rays_two_in_first, axis=1, keepdims=True)
                dots = np.clip(np.sum(rays_one[positive] * rays_two_in_first[positive], axis=1), -1.0, 1.0)
                angles = np.rad2deg(np.arccos(dots))
                candidate_result = {
                    "inlier_count": int(np.count_nonzero(inliers)),
                    "inlier_fraction": float(np.count_nonzero(inliers) / len(source)),
                    "positive_depth_count": int(np.count_nonzero(positive)),
                    "positive_depth_fraction": float(np.count_nonzero(positive) / np.count_nonzero(inliers)),
                    "angular_parallax_deg": _finite_quantile(angles, 0.5),
                }
                if best is None or candidate_result["positive_depth_count"] > best["positive_depth_count"]:
                    best = candidate_result
            except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
                continue
        if best is None:
            result["status"] = "pose_or_triangulation_failed"
            return result
        homography_dominates = bool(
            homography["supported"]
            and homography["residual_quantile_px"] is not None
            and homography["residual_quantile_px"] < settings.min_parallax_px
            and homography["inlier_fraction"] + settings.essential_homography_inlier_margin
            >= best["inlier_fraction"]
        )
        result.update(
            status="supported" if (
                best["inlier_count"] >= settings.min_geometric_matches
                and best["inlier_fraction"] >= settings.essential_min_inlier_fraction
                and best["positive_depth_fraction"] >= settings.essential_min_positive_depth_fraction
            ) else "weak_support",
            homography_dominates=homography_dominates,
            supported=bool(
                best["inlier_count"] >= settings.min_geometric_matches
                and best["inlier_fraction"] >= settings.essential_min_inlier_fraction
                and best["positive_depth_fraction"] >= settings.essential_min_positive_depth_fraction
            ),
        )
        result.update(best)
    except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
        result["status"] = "estimation_failed"
    return result


def geometric_parallax_signal(
    previous: Any,
    current: Any,
    matcher: Any,
    settings: Any,
    *,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure robust geometric support without treating pixel flow as parallax."""
    import cv2
    import numpy as np

    observations = _matched_points(previous, current, matcher)
    if observations is None:
        return {
            "model": "coverage_cadence_fallback",
            "status": "no_matches",
            "support": "none",
            "ambiguity": "no_correspondence_support",
            "matched_features": 0,
            "matched_coverage_fraction": 0.0,
            "feature_displacement_px": None,
            "homography": {"status": "not_available"},
            "essential": {"status": "not_available"},
            "admissible": False,
            "admission_basis": "coverage_cadence_fallback",
        }
    source, target, matches = observations
    _, _, width, height = current
    displacement = float(np.median(np.linalg.norm(source - target, axis=1)))
    coverage = _coverage(target, width, height)
    homography = _homography_measurement(source, target, settings, cv2)
    essential = _essential_measurement(source, target, calibration, settings, cv2, homography)
    coverage_ok = coverage >= settings.min_match_coverage_fraction
    homography_signal = bool(
        homography["inlier_count"] >= settings.min_geometric_matches
        and homography["residual_quantile_px"] is not None
        and homography["residual_quantile_px"] >= settings.min_parallax_px
        and coverage_ok
    )
    essential_signal = bool(
        essential["supported"]
        and not essential["homography_dominates"]
        and essential["angular_parallax_deg"] is not None
        and essential["angular_parallax_deg"] >= settings.min_angular_parallax_deg
        and coverage_ok
    )
    if essential_signal:
        model = "essential_recover_pose_triangulated_angular_parallax"
        support = "strong"
        admission_basis = model
        admissible = True
    elif homography_signal:
        model = "homography_rotation_planar_compensated_proxy"
        support = "strong" if homography["supported"] else "weak_homography_residual_support"
        admission_basis = model
        admissible = True
    else:
        model = "coverage_cadence_fallback"
        admissible = False
        if not coverage_ok:
            ambiguity = "insufficient_spatial_coverage"
        elif essential.get("homography_dominates"):
            ambiguity = "rotation_or_planar_homography_dominant"
        elif (
            homography["supported"]
            and homography["residual_quantile_px"] is not None
            and homography["residual_quantile_px"] < settings.min_parallax_px
        ):
            ambiguity = "rotation_or_planar_homography_dominant"
        elif homography["supported"] or essential["supported"]:
            ambiguity = "supported_motion_below_geometric_threshold"
        else:
            ambiguity = "insufficient_geometric_model_support"
        support = "supported_no_signal" if homography["supported"] or essential["supported"] else "weak"
        admission_basis = "coverage_cadence_fallback"
    if model == "essential_recover_pose_triangulated_angular_parallax":
        ambiguity = "model_supported"
    elif model == "homography_rotation_planar_compensated_proxy":
        ambiguity = "model_supported" if homography["supported"] else "weak_homography_residual_support"
    return {
        "model": model,
        "status": "supported" if admissible else "fallback",
        "support": support,
        "ambiguity": ambiguity,
        "matched_features": len(matches),
        "matched_coverage_fraction": coverage,
        "coverage_ok": coverage_ok,
        "feature_displacement_px": displacement,
        "homography": homography,
        "essential": essential,
        "geometric_parallax_proxy_px": homography.get("residual_quantile_px"),
        "angular_parallax_deg": essential.get("angular_parallax_deg"),
        "admissible": admissible,
        "admission_basis": admission_basis,
    }


def _calibration_from_validated(validated: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    model = validated["camera_model"]
    params = np.asarray(validated["params"], dtype=np.float64)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        K = np.asarray([[params[0], 0.0, params[1]], [0.0, params[0], params[2]], [0.0, 0.0, 1.0]])
    else:
        K = np.asarray([[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]])
    if model == "SIMPLE_RADIAL":
        distortion = params[3:4]
    elif model == "RADIAL":
        distortion = params[3:6]
    elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        distortion = params[4:]
    else:
        distortion = np.asarray([], dtype=np.float64)
    return {
        "camera_model": model,
        "K": K,
        "distortion": distortion,
        "width": validated["width"],
        "height": validated["height"],
    }


def _load_optional_calibration(cfg: Any, frame: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if frame is None:
        return None, {"status": "not_used", "reason": "no_frames"}
    dataset = getattr(cfg, "dataset", None)
    optional = getattr(dataset, "optional", None)
    path_value = getattr(optional, "intrinsics", None)
    if path_value is None:
        return None, {"status": "absent", "reason": None}
    path = Path(path_value)
    if not path.is_file():
        return None, {"status": "provided_file_missing", "reason": f"optional calibration file is missing: {path}"}
    try:
        from drishti.recon.sfm import load_intrinsics

        validated = load_intrinsics(path, int(frame["width"]), int(frame["height"]))
        return _calibration_from_validated(validated), {
            "status": "validated",
            "reason": None,
            "camera_model": validated["camera_model"],
        }
    except Exception as exc:
        # Intrinsics are optional. Keep S1 usable and make the fallback visible.
        return None, {"status": "invalid_not_used", "reason": str(exc)}


class S1FrameQA:
    name = "s1_frameqa"
    requires = ("s0_ingest",)
    config_sections = ("frameqa", "run")

    def run(self, bundle: Any, cfg: Any, runtime: Any) -> StageResult:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise DependencyError("Frame QA requires opencv-python-headless and numpy") from exc
        cv2.setNumThreads(1)
        cv2.setRNGSeed(cfg.run.seed)
        settings = cfg.frameqa
        cv_api: Any = cv2  # The installed stubs omit this runtime factory.
        orb = cv_api.ORB_create(nfeatures=settings.orb_features)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        document = bundle.read_json(str(bundle.artifact("s0_ingest", "frames").relative_to(bundle.root)))
        frames = document["frames"]
        calibration, calibration_status = _load_optional_calibration(cfg, frames[0] if frames else None)
        selected: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        previous = None
        last_time = None
        last_shape = None
        for frame in frames:
            reasons: list[str] = []
            gray = cv2.imread(str(bundle.root / frame["path"]), cv2.IMREAD_GRAYSCALE)
            qa: dict[str, Any] = {"intrinsics": dict(calibration_status)}
            features = None
            if gray is None:
                reasons.append("image_decode_failed")
            else:
                qa.update(
                    sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    dark_fraction=float(np.mean(gray <= settings.dark_threshold)),
                    bright_fraction=float(np.mean(gray >= settings.bright_threshold)),
                )
                if qa["sharpness"] < settings.min_sharpness:
                    reasons.append("blur_below_sharpness_threshold")
                if qa["dark_fraction"] > settings.max_dark_fraction:
                    reasons.append("underexposed")
                if qa["bright_fraction"] > settings.max_bright_fraction:
                    reasons.append("overexposed")
            if frame["gps"] is None:
                reasons.extend(frame.get("flags") or ["telemetry_unsupported"])
            if not reasons:
                points, descriptors = orb.detectAndCompute(gray, None)
                features = (points, descriptors, gray.shape[1], gray.shape[0])
                qa["orb_features"] = len(points)
                if previous is not None and gray.shape == last_shape:
                    geometric = geometric_parallax_signal(
                        previous, features, matcher, settings, calibration=calibration
                    )
                    qa.update(
                        feature_displacement_px=geometric["feature_displacement_px"],
                        matched_features=geometric["matched_features"],
                        matched_coverage_fraction=geometric["matched_coverage_fraction"],
                        displacement_domain="cross_checked_ORB_image_displacement_not_geometric_parallax",
                        geometric=geometric,
                    )
                    due = frame["time_s"] - last_time >= settings.max_interval_s
                    if not due and not geometric["admissible"]:
                        reasons.append(
                            "insufficient_geometric_coverage"
                            if geometric["ambiguity"] == "insufficient_spatial_coverage"
                            else "insufficient_geometric_parallax"
                        )
                    qa["selection_basis"] = (
                        "coverage_interval" if due else geometric["admission_basis"]
                    )
                else:
                    qa["selection_basis"] = "first_frame_or_changed_image_shape"
            decision = {"frame_id": frame["id"], "selected": not reasons, "reasons": reasons, "qa": qa}
            decisions.append(decision)
            if not reasons:
                selected.append({**frame, "qa": qa})
                previous, last_time, last_shape = features, frame["time_s"], gray.shape
        # Preserve coverage across the entire clip instead of truncating its tail.
        if settings.max_keyframes and len(selected) > settings.max_keyframes:
            keep_indices = set(np.linspace(0, len(selected) - 1, settings.max_keyframes, dtype=int).tolist())
            keep_ids = {frame["id"] for index, frame in enumerate(selected) if index in keep_indices}
            for decision in decisions:
                if decision["selected"] and decision["frame_id"] not in keep_ids:
                    decision["selected"] = False
                    decision["reasons"] = ["max_keyframes_coverage_thinning"]
            selected = [frame for frame in selected if frame["id"] in keep_ids]
        bundle.stage_dir(self.name)
        keyframes_path, qa_path = f"{self.name}/keyframes.json", f"{self.name}/qa.json"
        bundle.write_json(keyframes_path, {"frames": selected, "capabilities": document["capabilities"]})
        bundle.write_json(qa_path, {"frames": decisions, "drop_counts": dict(Counter(
            reason for decision in decisions for reason in decision["reasons"]))})
        if not selected:
            raise ReconstructionError("All frames failed QA/telemetry gates; inspect s1_frameqa/qa.json")
        return StageResult(outputs={"keyframes": keyframes_path, "qa": qa_path},
                           metrics={"input_frames": len(frames), "keyframes": len(selected),
                                    "dropped_frames": len(frames) - len(selected)}, confidence=None)


STAGE = S1FrameQA()
