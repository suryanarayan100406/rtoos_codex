"""Joint, fixed-calibration reprojection/GNSS BA with unchanged-input fallback.

All geometry is local-origin float64.  Camera rotations are world/local to
camera (R_cw); measurements are actual undistorted feature pixels.  SciPy is
loaded only on execution.  No solver failure publishes partially updated data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from typing import Any

from drishti.recon.barometer import select_barometer


class _Unsupported(ValueError):
    """Malformed or insufficiently supported input cannot safely be optimized."""


def _length(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _proper_rotation(value: Any, name: str, np: Any) -> Any:
    rotation = np.asarray(value, dtype=np.float64)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6)):
        raise _Unsupported(f"{name} must be a finite proper orthonormal 3x3 rotation")
    return rotation


def _positive(value: Any, name: str, np: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise _Unsupported(f"{name} must be finite and positive")
    return float(value)


def _parameters(settings: Any, spine: Any, np: Any) -> dict[str, Any]:
    parameters = {
        "reprojection_sigma_px": settings.reprojection_sigma_px,
        "min_track_length": settings.min_track_length,
        "max_points": settings.max_points,
        "max_observations": settings.max_observations,
        "max_nfev": settings.max_nfev,
        "ftol": settings.ftol, "xtol": settings.xtol, "gtol": settings.gtol,
        "min_depth_m": settings.min_depth_m,
        "gnss_sigma_h_m": spine.gnss_sigma_h_m,
        "gnss_sigma_v_m": spine.gnss_sigma_v_m,
        "rotation_sigma_rad": spine.rotation_sigma_rad,
        "robust_delta": spine.robust_delta,
    }
    for key in ("min_track_length", "max_points", "max_observations", "max_nfev"):
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise _Unsupported(f"{key} must be a positive integer")
        parameters[key] = int(value)
    if parameters["min_track_length"] < 2:
        raise _Unsupported("min_track_length must be at least two distinct views")
    for key in ("reprojection_sigma_px", "ftol", "xtol", "gtol", "min_depth_m"):
        parameters[key] = _positive(parameters[key], key, np)
    for key in ("gnss_sigma_h_m", "gnss_sigma_v_m", "rotation_sigma_rad", "robust_delta"):
        parameters[key] = _positive(parameters[key], f"spine.{key}", np)
    return parameters


def _sparse_arrays(sparse: dict[str, Any], np: Any) -> tuple[Any, Any]:
    points = np.asarray(sparse.get("points"))
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.dtype("float64")
            or not np.isfinite(points).all()):
        raise _Unsupported("sparse points must be finite float64 Nx3 local coordinates")
    count = len(points)
    inferred = np.asarray(sparse.get("inferred"))
    confidence = np.asarray(sparse.get("confidence"))
    if inferred.shape != (count,) or inferred.dtype != np.dtype(bool):
        raise _Unsupported("inferred must be an N-element bool provenance array")
    if (confidence.shape != (count,) or not np.issubdtype(confidence.dtype, np.floating)
            or not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1)):
        raise _Unsupported("confidence must be an N-element finite floating array in [0,1]")
    for key, value in sparse.items():
        array = np.asarray(value)
        if array.ndim and len(array) != count:
            raise _Unsupported(f"sparse provenance {key!r} length does not match points")
    if "colors" in sparse:
        colors = np.asarray(sparse["colors"])
        if colors.shape != (count, 3) or colors.dtype != np.dtype(np.uint8):
            raise _Unsupported("colors must be an Nx3 uint8 array")
    for key in ("reprojection_error_px", "track_length"):
        if key in sparse and np.asarray(sparse[key]).shape != (count,):
            raise _Unsupported(f"{key} must be an N-element provenance array")
    return points, inferred


def _connected(support: list[list[int]], camera_count: int) -> bool:
    # A star per track suffices for connectivity, with linear storage rather
    # than a clique (quadratic in the number of observing cameras).
    neighbours: list[set[int]] = [set() for _ in range(camera_count)]
    for indices in support:
        for index in indices[1:]:
            neighbours[indices[0]].add(index)
            neighbours[index].add(indices[0])
    seen, pending = {0}, [0]
    while pending:
        for index in neighbours[pending.pop()]:
            if index not in seen:
                seen.add(index)
                pending.append(index)
    return len(seen) == camera_count


def _feature_progress(features: list[Any], uv: Any, np: Any) -> int:
    """Affine feature-span progress: empty, one, two distinct, non-collinear."""
    if len(features) == 3:
        return 3
    if not features:
        return 1
    augmented = np.asarray([*features, uv], dtype=np.float64)
    return 1 + int(np.linalg.matrix_rank(augmented - augmented[0]))


def _select_tracks(eligible: list[tuple[int, dict[int, Any]]], camera_count: int,
                   parameters: dict[str, Any], metadata: dict[str, Any], np: Any) -> list[tuple[int, dict[int, Any]]]:
    """Bounded greedy coverage/connectivity first, then ascending-index fill.

    Cover the least-supported cameras first.  Prefer whole tracks which advance
    other cameras' non-collinear feature support per observation, then merge
    visual components per observation; break ties by cost and original index.
    This deterministic heuristic is not an exhaustive feasibility search.  The
    final support/connectedness checks remain mandatory even if selection stalls.
    """
    remaining = dict(eligible)
    observers: list[list[int]] = [[] for _ in range(camera_count)]
    for point_index, track in eligible:
        for camera_index in track:
            observers[camera_index].append(point_index)
    # At most three affinely independent witnesses per camera.  Collinear or
    # coincident pixels cannot consume a coverage slot and masquerade as PnP
    # support.  Actual selected-view counts are computed separately below.
    witnesses: list[list[Any]] = [[] for _ in range(camera_count)]
    parents = list(range(camera_count))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merges(track: dict[int, Any]) -> int:
        return len({root(index) for index in track}) - 1

    selected: list[tuple[int, dict[int, Any]]] = []
    selected_observations = 0

    def fits(track: dict[int, Any]) -> bool:
        return bool(len(selected) < parameters["max_points"]
                    and selected_observations + len(track) <= parameters["max_observations"])

    def add(point_index: int) -> None:
        nonlocal selected_observations
        track = remaining.pop(point_index)
        selected.append((point_index, track))
        selected_observations += len(track)
        for index, uv in track.items():
            if _feature_progress(witnesses[index], uv, np) > len(witnesses[index]):
                witnesses[index].append(uv)
        indices = list(track)
        for index in indices[1:]:
            parents[root(index)] = root(indices[0])

    camera_order = sorted(range(camera_count), key=lambda index: (len(observers[index]), index))
    for camera_index in camera_order:
        while len(witnesses[camera_index]) < 3:
            best_index: int | None = None
            best_score: tuple[float, float, int, int] | None = None
            for point_index in observers[camera_index]:
                candidate = remaining.get(point_index)
                if candidate is None or not fits(candidate):
                    continue
                if _feature_progress(witnesses[camera_index], candidate[camera_index], np) <= len(witnesses[camera_index]):
                    continue
                gain = sum(_feature_progress(witnesses[index], uv, np) - len(witnesses[index])
                           for index, uv in candidate.items())
                score = (gain / len(candidate), merges(candidate) / len(candidate), -len(candidate), -point_index)
                if best_score is None or score > best_score:
                    best_index, best_score = point_index, score
            if best_index is None:
                break
            add(best_index)
    coverage_points = len(selected)
    coverage_observations = selected_observations
    while len({root(index) for index in range(camera_count)}) > 1:
        best_bridge: int | None = None
        bridge_score: tuple[float, int, int] | None = None
        for point_index, track in remaining.items():
            merge_count = merges(track)
            if merge_count == 0 or not fits(track):
                continue
            candidate_score = (merge_count / len(track), -len(track), -point_index)
            if bridge_score is None or candidate_score > bridge_score:
                best_bridge, bridge_score = point_index, candidate_score
        if best_bridge is None:
            break
        add(best_bridge)
    connectivity_points = len(selected) - coverage_points
    connectivity_observations = selected_observations - coverage_observations
    for point_index in sorted(remaining):
        if fits(remaining[point_index]):
            add(point_index)
    exclusions = metadata["exclusions"]
    exclusions["budget_tracks"] = len(remaining)
    if len(selected) >= parameters["max_points"]:
        exclusions["budget_points"] = len(remaining)
    else:
        exclusions["budget_observations"] = len(remaining)
    metadata.update(
        selection_method="least_supported_camera_coverage_then_component_bridges_then_ascending_whole_tracks",
        selection_is_exhaustive_feasibility_search=False,
        selection_camera_order=camera_order,
        selection_order_point_indices=[index for index, _ in selected],
        coverage_selected_points=coverage_points,
        coverage_selected_observations=coverage_observations,
        connectivity_selected_points=connectivity_points,
        connectivity_selected_observations=connectivity_observations,
        fill_selected_points=len(selected) - coverage_points - connectivity_points,
        fill_selected_observations=selected_observations - coverage_observations - connectivity_observations,
        selection_coverage_complete=all(len(features) == 3 for features in witnesses),
        selection_network_connected=len({root(index) for index in range(camera_count)}) == 1,
    )
    # Solver variable/observation order stays ascending for existing all-view
    # inputs.  The actual coverage/bridge/fill selection order is recorded above.
    return sorted(selected, key=lambda item: item[0])


@dataclass
class _Problem:
    rotations: Any
    centers: Any
    intrinsics: Any
    prior_rotation: Any
    point_indices: Any
    points: Any
    observation_cameras: Any
    observation_tracks: Any
    observed_uv: Any
    support: Any
    gps_indices: Any
    gps: Any
    gps_sigmas: Any
    barometer_indices: Any = None
    barometer_heights: Any = None
    barometer_sigmas: Any = None
    barometer_bias_initial: float | None = None

    def initial(self, np: Any) -> Any:
        poses = np.zeros((len(self.centers), 6), dtype=np.float64)
        poses[:, 3:] = self.centers
        state = np.concatenate((poses.ravel(), self.points.ravel()))
        if self.barometer_bias_initial is not None:
            state = np.append(state, self.barometer_bias_initial)
        return state

    def decode(self, state: Any, rotation_class: Any, np: Any) -> tuple[Any, Any, Any]:
        poses = state[:6 * len(self.centers)].reshape(-1, 6)
        rotations = rotation_class.from_rotvec(poses[:, :3]).as_matrix() @ self.rotations
        point_start = 6 * len(self.centers)
        point_end = point_start + 3 * len(self.points)
        return rotations, poses[:, 3:], state[point_start:point_end].reshape(-1, 3)

    def pixels(self, state: Any, rotation_class: Any, np: Any,
               depth_floor: float | None) -> tuple[Any, Any]:
        rotations, centers, points = self.decode(state, rotation_class, np)
        indices = self.observation_cameras
        camera_xyz = np.einsum("nij,nj->ni", rotations[indices],
                               points[self.observation_tracks] - centers[indices])
        depths = camera_xyz[:, 2]
        denominator = depths
        if depth_floor is not None:
            # Preserve the side of the image plane while avoiding division at
            # zero.  These guarded pixels never bypass final cheirality checks.
            denominator = np.copysign(np.maximum(np.abs(depths), depth_floor), depths)
        K = self.intrinsics[indices]
        projected = camera_xyz[:, :2] / denominator[:, None]
        projected[:, 0] = projected[:, 0] * K[:, 0, 0] + K[:, 0, 2]
        projected[:, 1] = projected[:, 1] * K[:, 1, 1] + K[:, 1, 2]
        if not np.isfinite(projected).all() or not np.isfinite(depths).all():
            raise RuntimeError("non-finite BA projection")
        return projected - self.observed_uv, depths

    def residual(self, state: Any, parameters: dict[str, Any], rotation_class: Any, np: Any) -> Any:
        if not np.isfinite(state).all():
            raise RuntimeError("non-finite BA state")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            pixel_errors, _ = self.pixels(state, rotation_class, np, parameters["min_depth_m"])
            rotations, centers, _ = self.decode(state, rotation_class, np)
            gps_errors = (centers[self.gps_indices] - self.gps) / self.gps_sigmas
            prior = rotation_class.from_matrix(rotations[0] @ self.prior_rotation.T).as_rotvec()
            result = np.concatenate((
                (pixel_errors / parameters["reprojection_sigma_px"]).ravel(),
                gps_errors.ravel(), prior / parameters["rotation_sigma_rad"],
            ))
            if self.barometer_bias_initial is not None:
                heights = (centers[self.barometer_indices, 2] + state[-1] - self.barometer_heights)
                result = np.concatenate((result, heights / self.barometer_sigmas))
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite normalized BA residual")
        return result

    def jacobian_pattern(self, sparse_class: Any, np: Any) -> Any:
        observation_count = len(self.observed_uv)
        gps_count = len(self.gps_indices)
        residual_count = 2 * observation_count + 3 * gps_count + 3
        variable_count = 6 * len(self.centers) + 3 * len(self.points)
        barometer_count = (len(self.barometer_indices) if self.barometer_bias_initial is not None else 0)
        rows: list[int] = []
        columns: list[int] = []
        for index, (camera, track) in enumerate(zip(self.observation_cameras,
                                                   self.observation_tracks, strict=True)):
            for row in (2 * index, 2 * index + 1):
                rows.extend([row] * 9)
                columns.extend(range(6 * int(camera), 6 * int(camera) + 6))
                start = 6 * len(self.centers) + 3 * int(track)
                columns.extend(range(start, start + 3))
        for index, camera in enumerate(self.gps_indices):
            for component in range(3):
                rows.append(2 * observation_count + 3 * index + component)
                columns.append(6 * int(camera) + 3 + component)
        for row in range(residual_count - 3, residual_count):
            rows.extend([row] * 3)
            columns.extend(range(3))
        for index in range(barometer_count):
            rows.extend([residual_count + index] * 2)
            columns.extend([6 * int(self.barometer_indices[index]) + 5, variable_count])
        residual_count += barometer_count
        variable_count += int(bool(barometer_count))
        return sparse_class((np.ones(len(rows), dtype=bool), (rows, columns)),
                            shape=(residual_count, variable_count)).tocsr()


def _prepare(cameras: list[dict[str, Any]], sparse: dict[str, Any], parameters: dict[str, Any],
             metadata: dict[str, Any], np: Any) -> _Problem:
    points, inferred = _sparse_arrays(sparse, np)
    metadata["input_points"] = len(points)
    if len(cameras) < 3:
        raise _Unsupported("joint BA requires at least three cameras and valid GNSS factors")
    rotations, centers, intrinsics = [], [], []
    gps_indices, gps, sigmas, sigma_sources = [], [], [], []
    tracks: dict[int, dict[int, Any]] = {}
    invalid_tracks: set[int] = set()
    exclusions = {
        "inferred_points": int(np.count_nonzero(inferred)), "inferred_observations": 0,
        "nonfinite_or_out_of_bounds_observations": 0, "duplicate_observations": 0,
        "initial_nonpositive_depth_observations": 0, "single_view_tracks": 0,
        "short_tracks": 0, "invalid_tracks": 0, "unobserved_points": 0,
        "degenerate_tracks": 0, "budget_tracks": 0, "budget_points": 0,
        "budget_observations": 0,
    }
    metadata["exclusions"] = exclusions
    observation_count = 0
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            raise _Unsupported(f"camera {index} must be a mapping")
        rotation = _proper_rotation(camera.get("R_cw"), f"camera {index} R_cw", np)
        center = np.asarray(camera.get("center"), dtype=np.float64)
        K = np.asarray(camera.get("K"), dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise _Unsupported(f"camera {index} center must be finite XYZ")
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.array_equal(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0):
            raise _Unsupported(f"camera {index} K must have standard positive zero-skew pinhole form")
        distortion = camera.get("distortion", [])
        if distortion is None:
            distortion = []
        if (camera.get("projection_domain") != "undistorted_pinhole"
                or camera.get("image_domain", "undistorted") != "undistorted"
                or len(distortion)):
            raise _Unsupported(f"camera {index} has distorted/unsupported pixels; undistorted pinhole required")
        for key in ("width", "height"):
            size = camera.get(key)
            if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
                raise _Unsupported(f"camera {index} {key} must be a positive integer")
        rotations.append(rotation)
        centers.append(center)
        intrinsics.append(K)
        sample = camera.get("gps_sample")
        if sample is None:
            sample = {}
        if not isinstance(sample, dict):
            raise _Unsupported(f"camera {index} gps_sample must be a mapping or null")
        sigma_h = _positive(sample.get("sigma_h_m", parameters["gnss_sigma_h_m"]),
                            f"camera {index} sigma_h_m", np)
        sigma_v = _positive(sample.get("sigma_v_m", parameters["gnss_sigma_v_m"]),
                            f"camera {index} sigma_v_m", np)
        inlier = camera.get("gps_inlier", True)
        if not isinstance(inlier, bool | np.bool_):
            raise _Unsupported(f"camera {index} gps_inlier must be boolean")
        if camera.get("gps") is not None:
            position = np.asarray(camera["gps"], dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise _Unsupported(f"camera {index} GPS must be finite local XYZ")
            if inlier:
                gps_indices.append(index)
                gps.append(position)
                sigmas.append([sigma_h, sigma_h, sigma_v])
                sigma_sources.append({
                    "camera_index": index, "horizontal_m": sigma_h, "vertical_m": sigma_v,
                    "horizontal_source": "gps_sample" if "sigma_h_m" in sample else "spine.gnss_sigma_h_m",
                    "vertical_source": "gps_sample" if "sigma_v_m" in sample else "spine.gnss_sigma_v_m",
                })
        observations = camera.get("observations", [])
        if not isinstance(observations, list):
            raise _Unsupported(f"camera {index} observations must be a list")
        observation_count += len(observations)
        seen: set[int] = set()
        for observation in observations:
            if not isinstance(observation, dict):
                raise _Unsupported("every observation must be a mapping")
            point_index = observation.get("point_index")
            if isinstance(point_index, bool) or not isinstance(point_index, Integral):
                raise _Unsupported("observation point_index must be an integer confined to sparse indexing")
            point_index = int(point_index)
            if not 0 <= point_index < len(points):
                raise _Unsupported("observation point_index must be an integer confined to sparse indexing")
            uv = np.asarray(observation.get("uv"), dtype=np.float64)
            if uv.shape != (2,):
                raise _Unsupported("observation uv must have two pixel coordinates")
            if inferred[point_index]:
                exclusions["inferred_observations"] += 1
                continue
            if point_index in seen:
                exclusions["duplicate_observations"] += 1
                invalid_tracks.add(point_index)
            seen.add(point_index)
            if (not np.isfinite(uv).all() or not 0 <= uv[0] < camera["width"]
                    or not 0 <= uv[1] < camera["height"]):
                exclusions["nonfinite_or_out_of_bounds_observations"] += 1
                invalid_tracks.add(point_index)
            depth = (rotation @ (points[point_index] - center))[2]
            if not np.isfinite(depth) or depth <= 0:
                exclusions["initial_nonpositive_depth_observations"] += 1
                invalid_tracks.add(point_index)
            tracks.setdefault(point_index, {})[index] = uv
    metadata.update(input_observations=observation_count, gnss_factors=len(gps_indices),
                    gnss_sigma_used=sigma_sources)
    if len(gps_indices) < 3:
        raise _Unsupported("joint BA requires at least three valid logged GNSS factors")
    if np.linalg.matrix_rank(np.asarray(gps) - gps[0]) == 0:
        raise _Unsupported("GNSS factors have no baseline to constrain metric scale")
    reference = cameras[0].get("visual_reference")
    if not isinstance(reference, dict) or "R_cw" not in reference:
        raise _Unsupported("first camera lacks recorded first Sim3 visual rotation prior")
    prior_rotation = _proper_rotation(reference["R_cw"], "first visual_reference R_cw", np)

    eligible: list[tuple[int, dict[int, Any]]] = []
    for point_index, track in sorted(tracks.items()):
        if point_index in invalid_tracks:
            exclusions["invalid_tracks"] += 1
        elif len(track) == 1:
            exclusions["single_view_tracks"] += 1
        elif len(track) < parameters["min_track_length"]:
            exclusions["short_tracks"] += 1
        else:
            rays = np.asarray([points[point_index] - centers[index] for index in track])
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            if np.linalg.matrix_rank(rays) < 2:
                exclusions["degenerate_tracks"] += 1
            else:
                eligible.append((point_index, track))
    exclusions["unobserved_points"] = len(points) - len(tracks) - int(np.count_nonzero(inferred))
    selected = _select_tracks(eligible, len(cameras), parameters, metadata, np)
    selected_observations = sum(len(track) for _, track in selected)
    metadata.update(
        eligible_points=len(eligible), selected_points=len(selected),
        selected_observations=selected_observations,
        retained_points=len(points) - len(selected),
        retained_observations=observation_count - selected_observations,
        selected_point_indices=[index for index, _ in selected],
    )
    support = [list(track) for _, track in selected]
    if not selected:
        raise _Unsupported("no supported multi-view non-inferred tracks remain after exclusions")
    if not _connected(support, len(cameras)):
        raise _Unsupported("coverage-first selection left an empty or disconnected visual network within budgets; every camera needs visual support")
    per_camera: list[list[Any]] = [[] for _ in cameras]
    for _, track in selected:
        for index, uv in track.items():
            per_camera[index].append(uv)
    metadata["camera_visual_support"] = [len(features) for features in per_camera]
    for index, features in enumerate(per_camera):
        # Three distinct non-collinear features are a necessary conservative
        # PnP support check, including for cameras without GNSS factors.
        if len(features) < 3 or np.linalg.matrix_rank(np.asarray(features) - features[0]) < 2:
            raise _Unsupported(f"camera {index} lacks three non-collinear visual features")
    variable_count = 6 * len(cameras) + 3 * len(selected)
    residual_count = 2 * selected_observations + 3 * len(gps_indices) + 3
    if residual_count < variable_count:
        raise _Unsupported("joint BA is underdetermined: fewer residuals than variables")
    observation_cameras, observation_tracks, observed_uv = [], [], []
    for track_index, (_, track) in enumerate(selected):
        for camera_index, uv in track.items():
            observation_cameras.append(camera_index)
            observation_tracks.append(track_index)
            observed_uv.append(uv)
    point_indices = np.asarray([index for index, _ in selected], dtype=np.int64)
    return _Problem(
        rotations=np.asarray(rotations, dtype=np.float64), centers=np.asarray(centers, dtype=np.float64),
        intrinsics=np.asarray(intrinsics, dtype=np.float64), prior_rotation=prior_rotation,
        point_indices=point_indices, points=points[point_indices].copy(),
        observation_cameras=np.asarray(observation_cameras, dtype=np.int64),
        observation_tracks=np.asarray(observation_tracks, dtype=np.int64),
        observed_uv=np.asarray(observed_uv, dtype=np.float64),
        support=np.asarray([len(track) for _, track in selected], dtype=np.int64),
        gps_indices=np.asarray(gps_indices, dtype=np.int64), gps=np.asarray(gps, dtype=np.float64),
        gps_sigmas=np.asarray(sigmas, dtype=np.float64),
    )


def _robust_cost(residual: Any, delta: float, np: Any) -> float:
    # SciPy's Huber is component-wise, after sigma normalization.  Compute the
    # same objective without squaring already-large outlier residuals.
    absolute = np.abs(residual)
    small = absolute <= delta
    with np.errstate(over="raise", invalid="raise"):
        cost = 0.5 * np.sum(np.square(absolute[small]))
        cost += np.sum(delta * (absolute[~small] - 0.5 * delta))
    if not np.isfinite(cost):
        raise RuntimeError("non-finite robust objective")
    return float(cost)


def _vector_squared_errors(errors: Any, np: Any) -> Any:
    with np.errstate(over="raise", invalid="raise"):
        squared = np.sum(np.square(errors), axis=1)
    if not np.isfinite(squared).all():
        raise RuntimeError("non-finite geometric fit errors")
    return squared


def _rmse(errors: Any, np: Any) -> float:
    with np.errstate(over="raise", invalid="raise"):
        result = float(np.sqrt(np.mean(_vector_squared_errors(errors, np))))
    if not np.isfinite(result):
        raise RuntimeError("non-finite geometric fit RMSE")
    return result


def optimize_bundle(cameras: list[dict[str, Any]], sparse: dict[str, Any], settings: Any,
                    spine_settings: Any, *, barometer_settings: Any = None
                    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return ``(cameras, full_indexed_sparse, metadata)`` without mutating input.

    Vary SO3 rotvec increments, centers and selected multi-view measured points;
    keep K fixed.  The first recorded Sim3 rotation closes the weak rotation
    gauge.  Input survives every unsupported/dependency/solver failure intact.
    Optional recorded sensor heights add one free constant bias, without a prior.
    """
    retained_cameras, retained_sparse = copy.deepcopy(cameras), copy.deepcopy(sparse)
    barometer = select_barometer(cameras, barometer_settings)
    input_point_count = _length(sparse.get("points", []))
    camera_observation_counts = [_length(camera.get("observations", [])) if isinstance(camera, dict) else None
                                 for camera in cameras]
    input_observation_count = (sum(count for count in camera_observation_counts if count is not None)
                               if all(count is not None for count in camera_observation_counts) else None)
    metadata: dict[str, Any] = {
        "optimizer_invoked": False, "status": "pending",
        "poses_retained": True, "points_retained": True,
        "initial_robust_cost": None, "final_robust_cost": None,
        "initial_reprojection_rmse_px": None, "reprojection_rmse_px": None,
        "initial_gnss_fit_rmse_m": None, "gnss_fit_rmse_m": None,
        "nfev": None, "termination": None, "method": None,
        "scipy_version": None,
        "calibrated_intrinsics_fixed": True, "fit_is_independent_accuracy": False,
        "imu_used": False, "baro_used": False, "covariance_status": "unavailable",
        "barometer": barometer.metadata,
        "covariance_reason": "BA covariance is not estimated",
        "coordinate_domain": "projected_local_float64",
        "projection_domain": "undistorted_pinhole",
        "rotation_prior": "first_Sim3_visual_rotation",
        "gnss_reference": "logged_position_attached_to_camera_centre_without_measured_lever_arm",
        "selected_points": 0, "selected_observations": 0,
        "retained_points": input_point_count, "retained_observations": input_observation_count,
        "poses_changed": False, "points_changed": False,
    }

    def fallback(status: str, reason: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        metadata.update(status=status, reason=reason, fallback_reason=reason,
                        retained_points=input_point_count, retained_observations=input_observation_count)
        metadata["baro_used"] = False
        barometer.reject(reason)
        return retained_cameras, retained_sparse, metadata

    if not settings.enabled:
        return fallback("disabled_retained_input", "bundle_adjustment.enabled=false")
    if not spine_settings.optimize:
        return fallback("disabled_retained_input", "spine.optimize=false")
    try:
        import numpy as np
        from scipy.optimize import least_squares
        from scipy.sparse import coo_matrix
        from scipy.spatial.transform import Rotation
    except (ImportError, OSError) as exc:
        return fallback("unavailable_retained_input", f"SciPy unavailable: {exc}")
    try:
        metadata["scipy_version"] = version("scipy")
    except PackageNotFoundError:
        metadata["scipy_version"] = None
    try:
        parameters = _parameters(settings, spine_settings, np)
        metadata["settings"] = parameters
        try:
            problem = _prepare(cameras, sparse, parameters, metadata, np)
        except _Unsupported:
            raise
        except (TypeError, ValueError) as exc:
            raise _Unsupported(f"invalid BA input: {exc}") from exc
        if barometer.observations:
            problem.barometer_indices = np.asarray([item.camera_index for item in barometer.observations], dtype=np.int64)
            problem.barometer_heights = np.asarray([item.height_m for item in barometer.observations], dtype=np.float64)
            problem.barometer_sigmas = np.asarray([item.sigma_m for item in barometer.observations], dtype=np.float64)
            problem.barometer_bias_initial = barometer.initial_bias(problem.centers)
            barometer.metadata.update(factors_attempted=len(barometer.observations),
                                      factors_added=len(barometer.observations))
        initial = problem.initial(np)
        pattern = problem.jacobian_pattern(coo_matrix, np)

        def residual(state: Any) -> Any:
            return problem.residual(state, parameters, Rotation, np)

        initial_residual = residual(initial)
        initial_cost = _robust_cost(initial_residual, parameters["robust_delta"], np)
        initial_pixels, _ = problem.pixels(initial, Rotation, np, None)
        metadata.update(
            initial_robust_cost=initial_cost,
            initial_reprojection_rmse_px=_rmse(initial_pixels, np),
            initial_gnss_fit_rmse_m=_rmse(problem.centers[problem.gps_indices] - problem.gps, np),
            jacobian="sparse_jacobian_pattern", jacobian_nnz=int(pattern.nnz),
            jacobian_shape=list(pattern.shape),
            depth_guard={"handling": "signed_max_abs_Z_min_depth_in_residual_only",
                         "min_depth_m": parameters["min_depth_m"],
                         "acceptance": "unclipped_Z_at_least_min_depth_m_for_all_selected_observations"},
        )
        metadata.update(optimizer_invoked=True, method="scipy.optimize.least_squares_joint_reprojection_GNSS",
                         optimizer_algorithm="trf", linear_solver="lsmr", x_scale="jac",
                         loss="huber", robust_delta=parameters["robust_delta"],
                         rotation_sigma_rad=parameters["rotation_sigma_rad"])
        barometer.metadata["optimizer_invoked"] = bool(barometer.metadata["factors_added"])
        # Central differences avoid subtractive cancellation in the offset/Z
        # columns under robust weighting.  Keep the baseline numerical scheme
        # exactly as before when no optional factors are present.
        numerical_jacobian = "3-point" if barometer.observations else "2-point"
        if barometer.observations:
            barometer.metadata["numerical_jacobian"] = numerical_jacobian
        result = least_squares(
            residual, initial, jac=numerical_jacobian, jac_sparsity=pattern, method="trf", tr_solver="lsmr",
            x_scale="jac", loss="huber", f_scale=parameters["robust_delta"],
            max_nfev=parameters["max_nfev"], ftol=parameters["ftol"],
            xtol=parameters["xtol"], gtol=parameters["gtol"],
        )
        metadata.update(nfev=int(result.nfev),
                        termination={"status": int(result.status), "message": str(result.message)})
        if (result.x.shape != initial.shape or not np.isfinite(result.x).all()
                or not np.isfinite(result.fun).all() or not np.isfinite(result.cost)):
            raise RuntimeError("solver returned non-finite or malformed result")
        final_cost = _robust_cost(residual(result.x), parameters["robust_delta"], np)
        metadata["final_robust_cost"] = final_cost
        if result.nfev > parameters["max_nfev"]:
            raise RuntimeError("solver exceeded max_nfev")
        if not result.success or result.status not in (1, 2, 3, 4):
            raise RuntimeError(f"solver did not converge: {result.message}")
        if final_cost > initial_cost:
            raise RuntimeError("robust objective increased")
        rotations, centers, points = problem.decode(result.x, Rotation, np)
        final_pixels, final_depths = problem.pixels(result.x, Rotation, np, parameters["min_depth_m"])
        if np.any(final_depths < parameters["min_depth_m"]):
            raise RuntimeError("selected observation has final Z behind/too near the camera")
        # Recompute unguarded pixels for reported fit and point confidence.
        final_pixels, _ = problem.pixels(result.x, Rotation, np, None)
        if not np.isfinite(centers).all() or not np.isfinite(points).all():
            raise RuntimeError("non-finite decoded geometry")
        for index, rotation in enumerate(rotations):
            _proper_rotation(rotation, f"BA camera {index}", np)
        pose_changes = [not np.array_equal(rotation, original_rotation)
                        or not np.array_equal(center, original_center)
                        for rotation, original_rotation, center, original_center in zip(
                            rotations, problem.rotations, centers, problem.centers, strict=True)]
        points_changed = not np.array_equal(points, problem.points)
        cameras_changed = any(pose_changes)
        output_cameras, output_sparse = copy.deepcopy(cameras), copy.deepcopy(sparse)
        for index, camera in enumerate(output_cameras):
            if pose_changes[index]:
                camera["R_cw"], camera["center"] = rotations[index].tolist(), centers[index].tolist()
                camera["pose_method"] = metadata["method"]
        cleared_covariances = 0
        if cameras_changed:
            # Existing marginals describe the pre-BA result.  They are cleared
            # from every camera when any accepted pose changed.
            for output_camera in output_cameras:
                cleared_covariances += int("pose_covariance_tangent" in output_camera
                                           or "covariance_domain" in output_camera)
                output_camera.pop("pose_covariance_tangent", None)
                output_camera.pop("covariance_domain", None)
        output_sparse["points"] = np.asarray(output_sparse["points"], dtype=np.float64).copy()
        output_sparse["points"][problem.point_indices] = points
        confidence = np.asarray(output_sparse["confidence"]).copy()
        if cameras_changed:
            confidence[:] = 0
        squared_errors = _vector_squared_errors(final_pixels, np)
        with np.errstate(over="raise", invalid="raise"):
            per_track_rmse = np.sqrt(np.bincount(problem.observation_tracks, weights=squared_errors,
                                                minlength=len(points)) / problem.support)
        if not np.isfinite(per_track_rmse).all():
            raise RuntimeError("non-finite per-track reprojection RMSE")
        confidence[problem.point_indices] = (1 - 1 / problem.support) / (1 + per_track_rmse)
        output_sparse["confidence"] = confidence
        if "reprojection_error_px" in output_sparse:
            errors = np.asarray(output_sparse["reprojection_error_px"], dtype=np.float64).copy()
            errors[problem.point_indices] = per_track_rmse
            output_sparse["reprojection_error_px"] = errors
        if barometer.observations:
            barometer.accept(centers, float(result.x[-1]))
        metadata.update(
            status="optimized", poses_retained=not cameras_changed, points_retained=not points_changed,
            poses_changed=cameras_changed, points_changed=points_changed,
            reprojection_rmse_px=_rmse(final_pixels, np),
            gnss_fit_rmse_m=_rmse(centers[problem.gps_indices] - problem.gps, np),
            minimum_final_depth_m=float(np.min(final_depths)),
            optimized_points=len(points), retained_points=len(sparse["points"]) - len(points),
            point_confidence_method="(1-1/actual_selected_view_support)/(1+final_track_reprojection_rmse_px)",
            unsupported_confidence="zeroed_after_pose_change" if cameras_changed else "retained_input",
            covariance_cleared_cameras=cleared_covariances,
        )
        metadata["baro_used"] = bool(barometer.observations)
        return output_cameras, output_sparse, metadata
    except _Unsupported as exc:
        return fallback("unsupported_retained_input", str(exc))
    except Exception as exc:
        return fallback("failed_retained_input", f"BA failed: {type(exc).__name__}: {exc}")
