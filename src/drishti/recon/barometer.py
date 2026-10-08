"""Optional recorded-height support shared by the actual graph and BA solvers.

Sensor height has no interpreted absolute datum.  A single free offset is an
optimization nuisance variable, never a measured calibration or accuracy claim.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class BarometerObservation:
    camera_index: int
    source_index: int
    source_line: int
    source_time_s: float
    time_s: float
    frame_time_s: float
    time_error_s: float
    height_m: float
    sigma_m: float
    sigma_source: str


@dataclass
class BarometerSelection:
    observations: tuple[BarometerObservation, ...]
    metadata: dict[str, Any]

    def initial_bias(self, centers: Any) -> float:
        import numpy as np

        offsets = [item.height_m - float(centers[item.camera_index][2])
                   for item in self.observations]
        # This median is only a starting state; there is deliberately no prior.
        bias = float(np.median(offsets))
        if not math.isfinite(bias):
            raise RuntimeError("non-finite barometer bias initialization")
        return bias

    def accept(self, centers: Any, bias: float) -> None:
        errors = [float(centers[item.camera_index][2]) + bias - item.height_m
                  for item in self.observations]
        fit = math.hypot(*errors) / math.sqrt(len(errors))
        if not math.isfinite(bias) or not math.isfinite(fit):
            raise RuntimeError("non-finite accepted barometer bias/fit")
        self.metadata.update(status="optimized", reason=None, bias_m=float(bias),
                             fit_rmse_m=fit, factors_accepted=len(errors), factors_rejected=0)

    def reject(self, reason: str) -> None:
        """Retain attempt provenance, but never publish an unaccepted bias/fit."""
        if self.metadata["status"] in ("ready", "optimized"):
            added = self.metadata["factors_added"]
            self.metadata.update(status="failed_retained_input" if added else "not_invoked",
                                 reason=reason, bias_m=None, fit_rmse_m=None,
                                 factors_accepted=0, factors_rejected=added)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _arithmetic_equal(recorded: float, computed: float, *operands: float) -> bool:
    """Allow binary64 arithmetic roundoff only, with no timing-gap tolerance.

    One ULP for each operand, intermediate, computed result and recorded result
    bounds input representation and elementary-operation rounding.  In particular
    this is not a relative tolerance scaled by an epoch timestamp.  Callers apply
    the configured timing bound strictly to both recorded and recomputed errors.
    """
    if not math.isfinite(computed):
        return False
    tolerance = math.fsum(math.ulp(value) for value in (recorded, computed, *operands))
    return abs(recorded - computed) <= tolerance


def select_barometer(cameras: list[dict[str, Any]], settings: Any = None) -> BarometerSelection:
    """Validate exact recorded fields; deduplicate before checking minimum support.

    Malformed associations are excluded; contradictory raw identities invalidate
    every association of the affected source IDs before timing deduplication.
    These failures cannot suppress mandatory visual/GNSS baseline optimization.
    """
    metadata: dict[str, Any] = {
        "status": "disabled", "reason": "barometer.enabled=false or settings absent",
        "factors_added": 0, "factors_attempted": 0, "factors_accepted": 0,
        "factors_rejected": 0, "optimizer_invoked": False,
        "selected": [], "excluded": [], "distinct_measurements": 0,
        "conflicting_source_indices": [], "duplicate_source_times_s": [],
        "sigma_used": [], "bias_m": None, "fit_rmse_m": None,
        "fit_is_independent_accuracy": False,
        "model": "relative_height_with_constant_unknown_bias",
        "absolute_datum": "not_interpreted",
        "method": "camera_local_z_plus_free_constant_bias_minus_recorded_height",
        "timing_approximation": "bounded_nearest_recorded_sample_no_interpolation",
        "bias_prior": None,
        "bias_initialization": "median_recorded_height_minus_initial_camera_local_z_not_calibration",
    }
    selection = BarometerSelection((), metadata)
    if settings is None or getattr(settings, "enabled", False) is False:
        return selection
    try:
        if settings.enabled is not True:
            raise ValueError("barometer.enabled must be boolean")
        minimum = _integer(settings.min_measurements, "barometer.min_measurements", 2)
        max_time_error = _finite(settings.max_time_error_s, "barometer.max_time_error_s")
        assumed_sigma = _finite(settings.sigma_m, "barometer.sigma_m")
        time_origin = _finite(settings.time_origin_s, "barometer.time_origin_s")
        time_offset = _finite(settings.time_offset_s, "barometer.time_offset_s")
        if max_time_error <= 0 or assumed_sigma <= 0:
            raise ValueError("barometer time bound and assumed sigma must be positive")
    except (AttributeError, TypeError, ValueError) as exc:
        metadata.update(status="unavailable", reason=f"invalid optional barometer settings: {exc}")
        return selection
    metadata["min_measurements"] = minimum
    metadata["clock_validation"] = {
        "time_origin_s": time_origin, "time_offset_s": time_offset,
        "normalized_time": "source_time_s - barometer.time_origin_s",
        "signed_time_error": "time_s - (frame_time_s + barometer.time_offset_s)",
        "comparison": "absolute_difference_at_most_sum_binary64_ulps_of_operands_intermediates_and_both_results",
        "time_error_bound": "strict_on_recorded_and_recomputed_error_without_tolerance",
        "source_clock_time_error_bound": "strict_on_(source_time_s-time_origin_s)-query_without_tolerance",
        "max_time_error_s": max_time_error,
        "raw_provenance_comparison": "exact_no_roundoff_tolerance",
        "conflict_handling": "exclude_all_associations_of_every_conflicting_source_index",
    }
    fields = {"source_time_s", "time_s", "height_m", "sigma_m", "sigma_source",
              "source_index", "source_line", "time_error_s", "frame_time_s"}
    candidates: dict[int, BarometerObservation] = {}
    valid: list[BarometerObservation] = []
    raw_fields = ("source_time_s", "source_line", "height_m", "sigma_m", "sigma_source")
    raw_signatures: dict[int, tuple[Any, ...]] = {}
    conflicts: dict[int, set[str]] = {}
    timestamp_sources: dict[float, set[int]] = {}
    excluded: list[dict[str, Any]] = metadata["excluded"]
    for index, camera in enumerate(cameras):
        observation = camera.get("barometer") if isinstance(camera, dict) else None
        if observation is None:
            excluded.append({"camera_index": index, "source_index": None,
                             "reason": "no aligned barometer observation"})
            continue
        source_index: int | None = None
        try:
            if not isinstance(observation, dict) or set(observation) != fields:
                raise ValueError("barometer observation must contain exactly the recorded contract fields")
            source_index = _integer(observation["source_index"], "source_index", 0)
            source_line = _integer(observation["source_line"], "source_line", 1)
            numeric = {name: _finite(observation[name], name) for name in
                       ("source_time_s", "height_m", "sigma_m")}
            if numeric["sigma_m"] <= 0:
                raise ValueError("sigma_m must be positive")
            sigma_source = observation["sigma_source"]
            if sigma_source not in ("csv", "barometer.sigma_m"):
                raise ValueError("sigma_source must be csv or barometer.sigma_m")
            # Check raw identity even if this association subsequently fails its
            # clock fields/bounds checks; malformed alignment cannot conceal a
            # contradictory raw record or let another ID count the same timestamp.
            signature = (numeric["source_time_s"], source_line, numeric["height_m"],
                         numeric["sigma_m"], sigma_source)
            previous_signature = raw_signatures.setdefault(source_index, signature)
            differences = {name for name, before, after in zip(raw_fields, previous_signature, signature, strict=True)
                           if before != after}
            if differences:
                conflicts.setdefault(source_index, set()).add(
                    "conflicting raw provenance for source_index: " + ", ".join(sorted(differences)))
            timestamp_sources.setdefault(numeric["source_time_s"], set()).add(source_index)
            numeric.update({name: _finite(observation[name], name) for name in
                            ("time_s", "time_error_s", "frame_time_s")})
            item = BarometerObservation(index, source_index, source_line,
                                       sigma_source=sigma_source, **numeric)
            if sigma_source == "barometer.sigma_m" and numeric["sigma_m"] != assumed_sigma:
                raise ValueError("assumed sigma_m differs from barometer.sigma_m")
            normalized = item.source_time_s - time_origin
            if not _arithmetic_equal(item.time_s, normalized, item.source_time_s, time_origin):
                raise ValueError("time_s inconsistent with source_time_s - barometer.time_origin_s")
            query = item.frame_time_s + time_offset
            if not math.isfinite(query):
                raise ValueError("frame_time_s + barometer.time_offset_s must be finite")
            computed_error = item.time_s - query
            if not _arithmetic_equal(item.time_error_s, computed_error,
                                     item.time_s, item.frame_time_s, time_offset, query):
                raise ValueError("time_error_s inconsistent with time_s - (frame_time_s + barometer.time_offset_s)")
            if abs(item.time_error_s) > max_time_error or abs(computed_error) > max_time_error:
                raise ValueError("recorded or recomputed time_error_s exceeds barometer.max_time_error_s")
            # A roundoff-sized normalization discrepancy must not move a raw
            # sensor timestamp from outside the user bound to inside it either.
            if abs(normalized - query) > max_time_error:
                raise ValueError("source-clock recomputed time_error_s exceeds barometer.max_time_error_s")
            if _finite(camera.get("time_s"), "camera.time_s") != numeric["frame_time_s"]:
                raise ValueError("frame_time_s differs from camera.time_s")
            center = camera.get("center")
            if center is None or len(center) != 3:
                raise ValueError("camera center must be finite XYZ")
            for component in center:
                _finite(component, "camera.center")
        except (KeyError, TypeError, ValueError) as exc:
            excluded.append({"camera_index": index, "source_index": source_index,
                             "reason": f"invalid optional observation: {exc}"})
            continue
        valid.append(item)
    duplicate_times = sorted(time for time, indices in timestamp_sources.items() if len(indices) > 1)
    for time in duplicate_times:
        for source_index in timestamp_sources[time]:
            conflicts.setdefault(source_index, set()).add("source_time_s reused under different source_index values")
    metadata.update(conflicting_source_indices=sorted(conflicts), duplicate_source_times_s=duplicate_times)
    for item in valid:
        source_index = item.source_index
        if source_index in conflicts:
            excluded.append({"camera_index": item.camera_index, "source_index": source_index,
                             "reason": "; ".join(sorted(conflicts[source_index]))})
            continue
        previous = candidates.get(source_index)
        if previous is not None:
            if (abs(item.time_error_s), item.camera_index) < (abs(previous.time_error_s), previous.camera_index):
                loser, winner = previous, item
            else:
                loser, winner = item, previous
            excluded.append({"camera_index": loser.camera_index, "source_index": source_index,
                             "reason": "duplicate source_index; closest timing then camera index retained",
                             "selected_camera_index": winner.camera_index})
            candidates[source_index] = winner
        else:
            candidates[source_index] = item
    observations = tuple(sorted(candidates.values(), key=lambda item: item.camera_index))
    for exclusion in excluded:
        if "selected_camera_index" in exclusion:
            exclusion["selected_camera_index"] = candidates[exclusion["source_index"]].camera_index
    metadata.update(selected=[asdict(item) for item in observations],
                    distinct_measurements=len(observations),
                    sigma_used=[{"camera_index": item.camera_index, "source_index": item.source_index,
                                 "sigma_m": item.sigma_m, "sigma_source": item.sigma_source}
                                for item in observations])
    if len(observations) < minimum:
        metadata.update(status="unavailable", reason=f"insufficient distinct barometer measurements: {len(observations)} < {minimum}")
        return selection
    selection.observations = observations
    metadata.update(status="ready", reason=None)
    return selection


def pose_height_error(pose: Any, bias: float, height: float, jacobians: Any = None) -> Any:
    """Unwhitened height error and GTSAM's right-local Pose3/Double derivatives.

    Pose3.retract([omega, v]) translates by R_wc @ v to first order.  Height
    depends on the last row of R_wc, not world-Z in a local translation slot;
    pure local rotation does not move the camera centre.
    """
    import numpy as np

    if jacobians is not None:
        pose_jacobian = np.zeros((1, 6), dtype=np.float64, order="F")
        pose_jacobian[0, 3:] = np.asarray(pose.rotation().matrix(), dtype=np.float64)[2]
        jacobians[0] = pose_jacobian
        jacobians[1] = np.ones((1, 1), dtype=np.float64, order="F")
    return np.array([float(pose.translation()[2]) + bias - height], dtype=np.float64)


def make_height_factor(gtsam: Any, pose_key: int, bias_key: int,
                       observation: BarometerObservation, robust: Any) -> Any:
    """Actual Pose3+Double CustomFactor, sigma-normalized by robust scalar noise."""
    noise = gtsam.noiseModel.Robust.Create(
        robust, gtsam.noiseModel.Isotropic.Sigma(1, observation.sigma_m))

    def error(factor: Any, values: Any, jacobians: Any) -> Any:
        return pose_height_error(values.atPose3(pose_key), values.atDouble(bias_key),
                                 observation.height_m, jacobians)

    return gtsam.CustomFactor(noise, [pose_key, bias_key], error)
