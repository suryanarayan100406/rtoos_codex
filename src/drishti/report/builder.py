"""Partial-safe HTML/JSON reports from measured bundle provenance."""

import html
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def stage_records(manifest):
    stages = manifest.get("stages") or {}
    if isinstance(stages, dict):
        return [{**record, "name": name} for name, record in stages.items() if isinstance(record, dict)]
    return [record for record in stages if isinstance(record, dict)]


def confidence_regions(cloud: dict, region_size_m: float, *, chunk_size: int | None = None) -> dict:
    import numpy as np

    from drishti.geo.products import validate_cloud

    cloud = validate_cloud(cloud, chunk_size=chunk_size)
    if not math.isfinite(region_size_m) or region_size_m <= 0:
        raise ValueError("report.region_size_m must be positive")
    points, inferred = cloud["points"], cloud["inferred"]
    step = chunk_size if chunk_size is not None else len(points)
    totals: dict[tuple[int, int], tuple[int, int, float]] = {}
    for start in range(0, len(points), step):
        cells = np.floor(points[start:start + step, :2] / region_size_m).astype(np.int64)
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        observed = ~inferred[start:start + step]
        observed_counts = np.bincount(inverse[observed], minlength=len(unique))
        inferred_counts = np.bincount(inverse[~observed], minlength=len(unique))
        sums = np.bincount(inverse[observed],
                           weights=cloud["confidence"][start:start + step][observed],
                           minlength=len(unique))
        for i, cell in enumerate(unique):
            key = (int(cell[0]), int(cell[1]))
            prior_observed, prior_inferred, prior_sum = totals.get(key, (0, 0, 0.))
            totals[key] = (prior_observed + int(observed_counts[i]),
                           prior_inferred + int(inferred_counts[i]),
                           prior_sum + float(sums[i]))
    regions = []
    total_observed = total_inferred = 0
    total_confidence = 0.
    for (x, y), (observed_count, inferred_count, confidence_sum) in sorted(totals.items()):
        bounds = [float(x * region_size_m), float(y * region_size_m),
                  float((x + 1) * region_size_m), float((y + 1) * region_size_m)]
        regions.append({"local_xy_bounds": bounds, "observed_points": observed_count,
                        "inferred_points": inferred_count,
                        "observed_point_density_per_m2": float(observed_count / region_size_m**2),
                        "mean_observed_confidence": confidence_sum / observed_count
                        if observed_count else None,
                        "absolute_accuracy_m": None})
        total_observed += observed_count
        total_inferred += inferred_count
        total_confidence += confidence_sum
    return {"status": "MEASURED", "method": "source_point_support_and_mean_observed_confidence",
            "confidence_is_accuracy": False, "frame": "projected_local",
            "region_size_m": region_size_m, "regions": regions,
            "observed_points": total_observed, "inferred_points": total_inferred,
            "mean_observed_confidence": total_confidence / total_observed if total_observed else None}


def fit_metrics(metrics: dict) -> dict:
    """Retain nested S2 alignment/optimization and flat S6 GNSS measurements."""
    selected = {}
    for key, value in metrics.items():
        if any(term in key.lower() for term in ("gps", "gnss", "alignment")):
            selected[key] = value
        elif isinstance(value, dict):
            nested = fit_metrics(value)
            if nested:
                selected[key] = nested
    return selected


def _barometer_documents(bundle, record: dict, specs: tuple, issues: list[str]) -> list:
    """Read only logical outputs or contract filenames in a recorded partial dir."""
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        return []
    documents = []
    for key, filename in specs:
        reference = outputs.get(key)
        if reference is None and "partial" not in outputs:
            continue
        try:
            value = reference if reference is not None else outputs["partial"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Barometer reference must be a nonblank bundle-relative path")
            path = bundle.path(value)
            if reference is None:
                path = bundle.path((path / filename).relative_to(bundle.root))
                if not path.exists():
                    continue  # A partial directory need not contain every product.
            relative = path.relative_to(bundle.root).as_posix()
            document = bundle.read_json(relative)
            if not isinstance(document, dict):
                raise ValueError("Barometer artifact metadata must be a JSON object")
            documents.append((relative, document))
        except Exception as exc:
            issues.append(f"Cannot read barometer {record.get('name')}/{key}: {type(exc).__name__}: {exc}")
    return documents


def _barometer_integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _barometer_number(value: Any, name: str, *, positive: bool = False,
                      nonnegative: bool = False) -> float:
    try:
        number = finite_number(value)
    except OverflowError:
        number = None
    if number is None or (positive and number <= 0) or (nonnegative and number < 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else
                                                  " and nonnegative" if nonnegative else ""))
    return number


def _validate_barometer_diagnostics(diagnostics: Any, *, used: bool) -> None:
    if diagnostics is None and not used:
        return  # Legacy disabled/unrecorded solves have no sensor diagnostics.
    if not isinstance(diagnostics, dict):
        raise ValueError("Barometer diagnostics must be an object")
    if (diagnostics.get("model") != "relative_height_with_constant_unknown_bias"
            or diagnostics.get("absolute_datum") != "not_interpreted"
            or diagnostics.get("fit_is_independent_accuracy") is not False):
        raise ValueError("Barometer diagnostics must declare relative height, unknown bias and non-independent fit")
    if diagnostics.get("status") not in ("disabled", "unavailable", "ready", "not_invoked",
                                         "failed_retained_input", "optimized"):
        raise ValueError("Invalid barometer diagnostic status")
    for key in ("factors_added", "factors_attempted", "factors_accepted", "factors_rejected",
                "distinct_measurements", "min_measurements"):
        if key in diagnostics:
            _barometer_integer(diagnostics[key], key, 2 if key == "min_measurements" else 0)
    for key in ("bias_m", "fit_rmse_m"):
        if diagnostics.get(key) is not None:
            _barometer_number(diagnostics[key], key, nonnegative=key == "fit_rmse_m")
    if type(diagnostics.get("optimizer_invoked")) is not bool:
        raise ValueError("Barometer optimizer_invoked must be boolean")
    if diagnostics.get("bias_prior") is not None:
        raise ValueError("Relative barometer bias must remain a free nuisance variable")
    excluded = diagnostics.get("excluded", [])
    if not isinstance(excluded, list):
        raise ValueError("Barometer excluded support must be a list")
    for item in excluded:
        if not isinstance(item, dict):
            raise ValueError("Barometer exclusion must be an object")
        for key in ("camera_index", "source_index", "selected_camera_index"):
            if item.get(key) is not None:
                _barometer_integer(item[key], key)
    selected = diagnostics.get("selected", [])
    sigmas = diagnostics.get("sigma_used", [])
    if not isinstance(selected, list) or not isinstance(sigmas, list):
        raise ValueError("Barometer selected/sigma_used must be lists")
    identities = []
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("Barometer selected observation must be an object")
        identities.append((_barometer_integer(item.get("camera_index"), "camera_index"),
                           _barometer_integer(item.get("source_index"), "source_index")))
        _barometer_integer(item.get("source_line"), "source_line", 1)
        for key in ("source_time_s", "time_s", "frame_time_s", "time_error_s", "height_m", "sigma_m"):
            _barometer_number(item.get(key), key, positive=key == "sigma_m")
        if item.get("sigma_source") not in ("csv", "barometer.sigma_m"):
            raise ValueError("Barometer uncertainty source is not recorded CSV or assumed noise")
    if (len({camera for camera, _ in identities}) != len(identities)
            or len({source for _, source in identities}) != len(identities)):
        raise ValueError("Barometer factors must use distinct cameras and recorded samples")
    for item in sigmas:
        if not isinstance(item, dict):
            raise ValueError("Barometer sigma_used entry must be an object")
        _barometer_integer(item.get("camera_index"), "camera_index")
        _barometer_integer(item.get("source_index"), "source_index")
        _barometer_number(item.get("sigma_m"), "sigma_m", positive=True)
        if item.get("sigma_source") not in ("csv", "barometer.sigma_m"):
            raise ValueError("Invalid barometer sigma_source")
    if "distinct_measurements" in diagnostics and diagnostics["distinct_measurements"] != len(selected):
        raise ValueError("Barometer distinct_measurements differs from selected support")
    added = diagnostics.get("factors_added", 0)
    accepted = diagnostics.get("factors_accepted", 0)
    rejected = diagnostics.get("factors_rejected", 0)
    if accepted + rejected > added:
        raise ValueError("Barometer accepted/rejected counts exceed added factors")
    if used:
        if (diagnostics.get("status") != "optimized" or not diagnostics["optimizer_invoked"]
                or not added or accepted != added or rejected
                or diagnostics.get("factors_attempted") != added
                or len(selected) != added or added < diagnostics.get("min_measurements", 2)
                or diagnostics.get("bias_m") is None or diagnostics.get("fit_rmse_m") is None):
            raise ValueError("Barometer used claim lacks accepted optimized factors, bias or fit")
        expected_sigmas = [{key: item[key] for key in ("camera_index", "source_index", "sigma_m", "sigma_source")}
                          for item in selected]
        if sigmas != expected_sigmas:
            raise ValueError("Barometer accepted factors lack matching actual uncertainty provenance")
    elif accepted or diagnostics.get("bias_m") is not None or diagnostics.get("fit_rmse_m") is not None:
        raise ValueError("Unused barometer attempt cannot declare accepted factors, bias or fit")


def _validate_barometer_fusion(fusion: Any, stage: str) -> bool:
    """Return whether the selected final geometry has a recorded accepted solve."""
    if not isinstance(fusion, dict) or fusion.get("stage") != stage:
        raise ValueError("Invalid barometer fusion stage identity")
    if fusion.get("status") not in ("used", "not_used", "disabled") or type(fusion.get("used")) is not bool:
        raise ValueError("Invalid barometer fusion status/use")
    attempts = fusion.get("attempts")
    if not isinstance(attempts, dict):
        raise ValueError("Barometer fusion must retain solver attempts")
    for key in ("pose_graph", "bundle_adjustment"):
        attempt = attempts.get(key)
        if attempt is None and key == "bundle_adjustment":
            continue
        if (not isinstance(attempt, dict) or not isinstance(attempt.get("status"), str)
                or type(attempt.get("used")) is not bool
                or type(attempt.get("optimizer_invoked")) is not bool):
            raise ValueError("Malformed barometer solver attempt")
        if attempt["used"] and (attempt["status"] != "optimized" or not attempt["optimizer_invoked"]):
            raise ValueError("Barometer used claim requires an invoked accepted solver")
        _validate_barometer_diagnostics(attempt.get("barometer"), used=attempt["used"])
    graph, ba = attempts["pose_graph"], attempts.get("bundle_adjustment")
    expected_source = ("bundle_adjustment" if ba and ba["status"] == "optimized" else
                       "pose_graph" if stage == "s2_poses" or graph["status"] == "optimized" else "retained_s2")
    if fusion.get("final_source") != expected_source:
        raise ValueError("Barometer final_source does not match accepted final geometry")
    accepted_solve = False
    if expected_source == "retained_s2":
        selected = fusion.get("inherited")
        if selected is None:
            if fusion["used"] or fusion.get("barometer") is not None:
                raise ValueError("Retained S2 use requires recorded inherited solver provenance")
            return False
        accepted_solve = _validate_barometer_fusion(selected, "s2_poses")
    else:
        selected = attempts[expected_source]
        accepted_solve = selected["status"] == "optimized"
    if (fusion["used"] != selected["used"] or fusion.get("barometer") != selected.get("barometer")
            or fusion.get("method") != selected.get("method")):
        raise ValueError("Barometer final summary disagrees with selected solver provenance")
    if (fusion["status"] == "used") != fusion["used"]:
        raise ValueError("Barometer status and final use disagree")
    return accepted_solve


def _barometer_input(bundle, record: dict, issues: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": "s0_ingest", "stage_status": record.get("status"), "current": False,
        "status": "unavailable", "reason": "No recorded S0 barometer input metadata",
        "provided": None, "available": None, "parsed": None, "aligned_frames": None,
        "sample_count": None, "used": None, "clock": None, "clock_available": None,
        "uncertainty": None, "source_descriptor": None, "ledger": None, "descriptor": None,
        "descriptor_valid": False,
    }
    metrics = record.get("metrics")
    optional = metrics.get("optional_inputs") if isinstance(metrics, dict) else None
    documents = _barometer_documents(bundle, record, (("barometer", "barometer.json"),
                                                     ("frames", "frames.json")), issues)
    for path, document in documents:
        if "measurement" in document or "samples" in document:
            result.update(source_descriptor=path,
                          descriptor={key: value for key, value in document.items() if key != "samples"})
            try:
                if (document.get("status") not in ("ok", "disabled", "not_provided", "unavailable")
                        or document.get("measurement") != "relative_sensor_height_m"
                        or document.get("absolute_datum") != "not_interpreted"):
                    raise ValueError("Invalid S0 barometer status/measurement/datum")
                samples = document.get("samples")
                if not isinstance(samples, list):
                    raise ValueError("Barometer input samples must be a list")
                for sample in samples:
                    if not isinstance(sample, dict):
                        raise ValueError("Barometer input sample must be an object")
                    for key in ("source_time_s", "time_s", "height_m", "sigma_m"):
                        _barometer_number(sample.get(key), key, positive=key == "sigma_m")
                    _barometer_integer(sample.get("source_index"), "source_index")
                    _barometer_integer(sample.get("source_line"), "source_line", 1)
                    if sample.get("sigma_source") not in ("csv", "barometer.sigma_m"):
                        raise ValueError("Invalid input uncertainty source")
                aligned = _barometer_integer(document.get("aligned_frames"), "aligned_frames")
                clock = document.get("clock")
                if not isinstance(clock, dict):
                    raise ValueError("Barometer input clock must be an object")
                if clock.get("time_origin_s") is not None:
                    _barometer_number(clock["time_origin_s"], "time_origin_s")
                for key in ("time_offset_s", "max_time_error_s"):
                    _barometer_number(clock.get(key), key, positive=key == "max_time_error_s")
                uncertainty = document.get("uncertainty")
                if uncertainty is not None:
                    if not isinstance(uncertainty, dict) or uncertainty.get("calibrated_accuracy") is not False:
                        raise ValueError("Input uncertainty must not claim calibrated accuracy")
                    if uncertainty.get("assumed_sigma_m") is not None:
                        _barometer_number(uncertainty["assumed_sigma_m"], "assumed_sigma_m", positive=True)
                result.update(status=document["status"], reason=document.get("reason"), clock=clock,
                              clock_available=clock.get("time_origin_s") is not None,
                              uncertainty=uncertainty, sample_count=len(samples), descriptor_valid=True,
                              aligned_frames=aligned, parsed=bool(samples), used=False,
                              current=record.get("status") in ("done", "degraded"))
            except Exception as exc:
                issues.append(f"Cannot read barometer S0 descriptor: {type(exc).__name__}: {exc}")
        elif optional is None:
            optional = document.get("capabilities")
    ledger = optional.get("baro", optional.get("barometer")) if isinstance(optional, dict) else None
    if ledger is not None:
        result["ledger"] = ledger
        try:
            if not isinstance(ledger, dict):
                raise ValueError("Barometer input ledger must be an object")
            for key in ("provided", "available", "parsed"):
                if key in ledger:
                    if type(ledger[key]) is not bool:
                        raise ValueError(f"Barometer ledger {key} must be boolean")
                    result[key] = ledger[key]
            for key in ("sample_count", "aligned_frames"):
                if key in ledger:
                    result[key] = _barometer_integer(ledger[key], key)
            if ledger.get("used") is not False:
                raise ValueError("S0 parsing/alignment cannot claim executed factors")
            if result["source_descriptor"] is None:
                result.update(status=ledger.get("status", "unavailable"), reason=ledger.get("reason"),
                              used=False, current=record.get("status") in ("done", "degraded"))
        except Exception as exc:
            issues.append(f"Cannot read barometer S0 ledger: {type(exc).__name__}: {exc}")
    return result


def barometer_summary(bundle, cfg, issues: list[str]) -> dict[str, Any]:
    """Newest recorded final stage wins, regardless of success or freshness.

    Annotated raw summaries preserve attempts and retained diagnostics. Only
    validated current final-factor diagnostics populate the measurement fields.
    Configuration and S0 availability are deliberately separate evidence.
    """
    records = {item["name"]: item for item in stage_records(bundle.manifest)}
    enabled = getattr(getattr(cfg, "barometer", None), "enabled", None)
    input_summary = _barometer_input(bundle, records.get("s0_ingest", {}), issues)
    stage = "s6_global" if "s6_global" in records else "s2_poses" if "s2_poses" in records else None
    result: dict[str, Any] = {
        "configuration": {"enabled": enabled if type(enabled) is bool else None},
        "input": input_summary, "stage": stage, "stage_status": None, "current": False,
        "status": "unavailable", "recorded_status": None,
        "reason": "No recorded final barometer solve summary", "availability_reason": None,
        "source_descriptor": None, "source_field": None, "final_source": None, "method": None,
        "used": None, "bias_m": None, "fit_rmse_m": None,
        "factors_added": None, "factors_attempted": None, "factors_accepted": None, "factors_rejected": None,
        "uncertainty_provenance": None, "attempts": None, "inherited": None, "provenance": [],
        "independent_accuracy": False, "fit_is_independent_accuracy": False,
        "model": "relative_height_with_constant_unknown_bias", "absolute_datum": "not_interpreted",
        "interpretation": "Relative sensor height plus a free constant nuisance bias; GNSS remains the absolute anchor. Fit and assumed noise are not independent accuracy or measured calibration.",
    }
    final_entry = None
    for name in ("s2_poses", "s6_global"):
        if name not in records:
            continue
        record = records[name]
        metrics = record.get("metrics")
        optional_inputs = metrics.get("optional_inputs") if isinstance(metrics, dict) else None
        candidates = []
        if isinstance(metrics, dict) and "barometer_fusion" in metrics:
            candidates.append((None, "metrics.barometer_fusion", metrics["barometer_fusion"]))
        specs = (("spine", "spine.json"), ("poses", "poses.json")) if name == "s2_poses" else (
            ("optimization", "optimization.json"), ("poses", "poses.json"))
        for path, document in _barometer_documents(bundle, record, specs, issues):
            if "barometer_fusion" in document:
                candidates.append((path, "barometer_fusion", document["barometer_fusion"]))
            if optional_inputs is None:
                optional_inputs = document.get("optional_inputs")
        entry: dict[str, Any] = {
            "stage": name, "stage_status": record.get("status"), "current": False,
            "selected_final": name == stage, "source_descriptor": None, "source_field": None,
            "summary": None, "accepted_solve": False, "validation_status": "unavailable",
            "validation_reason": "No recorded barometer fusion summary",
            "optional_inputs": optional_inputs,
        }
        for path, field, fusion in candidates:
            if entry["summary"] is None:
                entry.update(summary=fusion, source_descriptor=path, source_field=field)
            try:
                accepted = _validate_barometer_fusion(fusion, name)
                if entry["validation_status"] == "valid" and fusion != entry["summary"]:
                    raise ValueError("Recorded barometer summaries disagree between metrics/artifacts")
                entry.update(summary=fusion, source_descriptor=path, source_field=field,
                             accepted_solve=accepted, validation_status="valid", validation_reason=None,
                             current=record.get("status") in ("done", "degraded"))
            except Exception as exc:
                reason = f"Cannot validate barometer {name}/{field}: {type(exc).__name__}: {exc}"
                issues.append(reason)
                if entry["validation_status"] == "valid":
                    # A conflicting valid record cannot be resolved by source precedence.
                    if "summaries disagree" in str(exc):
                        entry.update(current=False, validation_status="invalid", validation_reason=reason)
                        break
                else:
                    entry.update(validation_status="invalid", validation_reason=reason)
        result["provenance"].append(entry)
        if name == stage:
            final_entry = entry
    if final_entry is None:
        if (input_summary["current"] and input_summary["descriptor_valid"]
                and input_summary["status"] == "disabled"):
            result.update(status="disabled", reason=input_summary["reason"])
        return result
    result["stage_status"] = final_entry["stage_status"]
    fusion = final_entry["summary"]
    if isinstance(fusion, dict):
        result.update(recorded_status=fusion.get("status"), reason=fusion.get("reason"),
                      final_source=fusion.get("final_source"), method=fusion.get("method"),
                      attempts=fusion.get("attempts"), inherited=fusion.get("inherited"),
                      source_descriptor=final_entry["source_descriptor"], source_field=final_entry["source_field"])
    if final_entry["validation_status"] != "valid":
        result["availability_reason"] = final_entry["validation_reason"]
        return result
    if not final_entry["current"]:
        result["availability_reason"] = f"{stage} stage is {final_entry['stage_status'] or 'unrecorded'}; retained barometer metadata is not current"
        return result
    result["current"] = True
    if not final_entry["accepted_solve"] and fusion["status"] != "disabled":
        result["availability_reason"] = "Selected final geometry has no recorded accepted barometer-capable solve"
        return result
    if fusion.get("barometer") is None and fusion["status"] != "disabled":
        result["availability_reason"] = "Selected final solve has no recorded barometer diagnostics"
        return result
    result.update(status=fusion["status"], used=fusion["used"])
    diagnostics = fusion.get("barometer")
    if isinstance(diagnostics, dict):
        for key in ("factors_added", "factors_attempted", "factors_accepted", "factors_rejected"):
            result[key] = diagnostics.get(key)
        if fusion["used"]:
            result.update(bias_m=diagnostics["bias_m"], fit_rmse_m=diagnostics["fit_rmse_m"])
        result["uncertainty_provenance"] = {key: diagnostics[key] for key in
                                            ("sigma_used", "timing_approximation", "bias_prior", "bias_initialization")
                                            if key in diagnostics}
    return result


def semantic_summary(bundle, geo: dict, issues: list[str]) -> dict[str, Any]:
    """Retained semantic provenance is inspectable, never a current measurement.

    Only recorded S9 outputs (including a recorded partial directory) are read.
    Validation here concerns metadata/file references, not raster contents or
    model accuracy. No configured/detected model is promoted to executed provenance.
    """
    record: dict[str, Any] = next((item for item in stage_records(bundle.manifest)
                                  if item.get("name") == "s9_geo"), {})
    stage_status = record.get("status")
    stage_current = stage_status in ("done", "degraded")
    summary: dict[str, Any] = {
        "stage": "s9_geo", "stage_status": stage_status, "current": False,
        "status": "unavailable", "recorded_status": None,
        "source_descriptor": None, "source_field": None,
        "reason": "No recorded S9 semantic descriptor",
        "availability_reason": None, "classes": [], "nodata": 0,
        "provenance": None, "confidence_method": None,
        "confidence_interpretation": "Winning source-class softmax probability; uncalibrated; not geometry confidence or absolute accuracy",
        "confidence_is_geometry_confidence": False, "confidence_is_accuracy": False,
        "accuracy_status": "UNVALIDATED", "supported_cells": None,
        "raster_path": None, "confidence_path": None, "preview": None,
    }
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}

    def confined(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Semantic file reference must be a nonblank bundle-relative path")
        return bundle.path(value)

    for key, filename, field in (("semantic_metadata", "semantics.json", None),
                                  ("metadata", "geo.json", "semantics")):
        reference = outputs.get(key)
        if reference is None and "partial" not in outputs:
            continue
        try:
            path = (confined(reference) if reference is not None
                    else confined(outputs["partial"]) / filename)
            path = bundle.path(path.relative_to(bundle.root))
            relative = path.relative_to(bundle.root).as_posix()
            # Current geo metadata was already read by the ordinary report reader.
            # Reuse it to avoid duplicate issues for malformed mandatory metadata.
            if field and (stage_current or stage_status is None):
                descriptor = geo.get(field)
            else:
                if not path.is_file():
                    raise ValueError("Recorded semantic metadata file is missing")
                descriptor = bundle.read_json(relative)
                if field and isinstance(descriptor, dict):
                    descriptor = descriptor.get(field)
            if descriptor is None:
                continue
            if not isinstance(descriptor, dict):
                raise ValueError("Semantic metadata must be a JSON object")

            from drishti.geo.semantics import CONFIDENCE_METHOD, validate_descriptor

            status, provenance = descriptor.get("status"), descriptor.get("provenance")
            if not isinstance(status, str) or status not in {"ok", "partial", "disabled", "unavailable"}:
                raise ValueError("Invalid semantic metadata status")
            if descriptor.get("id") != "semantic" or descriptor.get("title") != "Semantic classes":
                raise ValueError("Invalid semantic metadata identity")
            if provenance is not None and not isinstance(provenance, dict):
                raise ValueError("Semantic provenance must be an object or null")
            # Reuse the shared class-mapping validation without inventing S3 frames.
            validate_descriptor({"status": "unavailable", "reason": descriptor.get("reason"),
                                 "classes": descriptor.get("classes"), "nodata": descriptor.get("nodata"),
                                 "frames": [], "model": (provenance or {}).get("model"),
                                 "confidence_method": descriptor.get("confidence_method")})
            if status in ("disabled", "unavailable") and not (descriptor.get("reason") or "").strip():
                raise ValueError("Unavailable/disabled semantics must explain their reason")
            model = (provenance or {}).get("model")
            if model and "inference_frames" in model and (
                type(model["inference_frames"]) is not int or model["inference_frames"] <= 0
            ):
                raise ValueError("Semantic model provenance must record actual inference")
            supported = descriptor.get("supported_cells")
            if type(supported) is not int or supported < 0:
                raise ValueError("Semantic supported_cells must be a nonnegative integer")
            if descriptor.get("accuracy_status") != "UNVALIDATED":
                raise ValueError("Semantic accuracy must remain UNVALIDATED")
            if provenance and provenance.get("source_descriptor") is not None:
                confined(provenance["source_descriptor"])
            if status in ("ok", "partial") and (
                not descriptor["classes"] or not supported or not provenance
                or not provenance.get("model") or descriptor.get("confidence_method") != CONFIDENCE_METHOD
                or descriptor.get("raster_path") is None
            ):
                raise ValueError("Available semantics require recorded inference, classes, probability method and raster support")
            for path_key in ("raster_path", "confidence_path"):
                if descriptor.get(path_key) is not None:
                    confined(descriptor[path_key])
            preview = descriptor.get("preview")
            if preview is not None:
                if not isinstance(preview, dict) or preview.get("crs") != "EPSG:4326":
                    raise ValueError("Semantic preview must declare EPSG:4326")
                confined(preview.get("path"))
                bounds = preview.get("bounds")
                if (not isinstance(bounds, list) or len(bounds) != 4
                        or any(finite_number(value) is None for value in bounds)
                        or not -180 <= bounds[0] < bounds[2] <= 180
                        or not -90 <= bounds[1] < bounds[3] <= 90
                        or any(type(preview.get(dim)) is not int or preview[dim] <= 0
                               for dim in ("width", "height"))):
                    raise ValueError("Semantic preview bounds/dimensions are malformed")

            summary.update(current=stage_current, status=status, recorded_status=status,
                           source_descriptor=relative, source_field=field,
                           reason=descriptor.get("reason"), classes=descriptor["classes"],
                           provenance=provenance, confidence_method=descriptor.get("confidence_method"))
            if not stage_current:
                summary["availability_reason"] = f"S9 stage is {stage_status or 'unrecorded'}; retained semantic metadata is not current"
                if status in ("ok", "partial"):
                    summary["status"] = "unavailable"
                return summary
            if status not in ("ok", "partial"):
                return summary
            missing = []
            for path_key in ("raster_path", "confidence_path", "preview"):
                value = preview.get("path") if path_key == "preview" and preview else descriptor.get(path_key)
                if value is not None:
                    asset = confined(value)
                    if asset.is_file() and asset.stat().st_size:
                        summary[path_key] = preview if path_key == "preview" else value
                        continue
                    issues.append(f"Semantic {path_key}: recorded output is missing or empty")
                missing.append(path_key)
            if summary["raster_path"] is None:
                summary["status"] = "unavailable"
            else:
                summary["supported_cells"] = supported
                if missing or descriptor.get("reason") or (provenance or {}).get("source_status") == "partial":
                    summary["status"] = "partial"
            if missing:
                summary["availability_reason"] = "Semantic products unavailable: " + ", ".join(missing)
            return summary
        except Exception as exc:
            reason = f"Cannot read semantic {key}: {type(exc).__name__}: {exc}"
            issues.append(reason)
            summary.update(current=False, status="unavailable", reason=reason,
                           supported_cells=None, raster_path=None, confidence_path=None, preview=None)
    return summary


def render_html(report: dict) -> str:
    """No untrusted content is interpolated as HTML, script or a URL."""
    def escape(value):
        return html.escape(str(value), quote=True)
    rows = []
    for stage in report["stages"]:
        rows.append("<tr>" + "".join(f"<td>{escape(stage.get(key))}</td>" for key in
                    ("name", "status", "wall_seconds", "confidence", "confidence_status", "confidence_reason", "degraded_reason")) + "</tr>")
    export_rows = []
    for name, record in report.get("exports", {}).get("formats", {}).items():
        export_rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in
                          (name, record.get("status"), record.get("path"), record.get("reason"))) + "</tr>")
    accuracy = report["accuracy"]
    barometer = report.get("barometer_fusion", {})
    barometer_input = barometer.get("input", {})
    barometer_provenance = escape(json.dumps(barometer.get("provenance"), indent=2,
                                            ensure_ascii=False, allow_nan=False))
    cloud = report.get("cloud_execution")
    cloud_section = ""
    if isinstance(cloud, dict):
        snapshot = cloud.get("report_cloud_snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        limits = cloud.get("limits")
        limits = limits if isinstance(limits, dict) else {}
        cloud_section = f"""<h2>Measured cloud execution snapshot</h2>
<p>Status: {escape(cloud.get('status'))}; scope: {escape(snapshot.get('scope'))}.
Orchestration wall seconds: {escape(cloud.get('orchestration_wall_seconds'))}.</p>
<p>Sampled process-tree RAM peak: {escape(cloud.get('rss_peak_bytes'))} bytes;
selected-GPU VRAM peak: {escape(cloud.get('vram_peak_bytes'))} bytes.
RAM/VRAM stop budgets: {escape(limits.get('ram_bytes'))} / {escape(limits.get('vram_bytes'))} bytes.</p>
<p>Sampled peaks are lower bounds. Torch allocator limits do not cover every native
CUDA allocation; this is not evidence of a hard instantaneous memory ceiling.</p>
<p>Snapshot excludes: {escape(snapshot.get('excludes'))}.
Authoritative final record: {escape(snapshot.get('authoritative_final_record'))}.</p>
<p>{escape(snapshot.get('interpretation'))}</p>
<pre>{escape(json.dumps(cloud, indent=2, ensure_ascii=False, allow_nan=False))}</pre>"""
    semantics = report.get("semantics", {})
    semantic_rows = []
    for item in semantics.get("classes", []):
        semantic_rows.append("<tr>" + "".join(f"<td>{escape(item.get(key))}</td>" for key in
                             ("id", "name", "color", "source_ids")) + "</tr>")
    semantic_provenance = escape(json.dumps(semantics.get("provenance"), indent=2,
                                           ensure_ascii=False, allow_nan=False))
    title = escape(report.get("dataset_name") or report.get("run_id") or "DRISHTI")
    # A preformatted escaped JSON payload exposes all measurements without unsafe
    # script embedding (including strings containing </script> or event handlers).
    detail = escape(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title} — DRISHTI report</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px;color:#172033}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd3df;padding:.5rem;text-align:left;overflow-wrap:anywhere}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f8;padding:1rem}}
.verdict{{padding:1rem;background:#fff3cc}}h2{{margin-top:2rem}}
</style></head><body><h1>{title} — accuracy &amp; confidence report</h1>
<p>Run status: <strong>{escape(report.get('status'))}</strong></p>
<div class="verdict">Full 3D absolute accuracy: <strong>{escape(accuracy['status'])}</strong>.
Design target: {escape(accuracy['design_target_m'])} m. GPS alignment/fit residuals are not independent accuracy.
DSM elevation checks establish elevation-only error; they do not certify horizontal or full 3D accuracy.</div>
<h2>Coverage and confidence</h2><p>Visible-scene completeness: {escape(report['coverage'].get('scene_completeness'))}.
{escape(report['coverage'].get('reason'))} Point confidence is a support/residual proxy, not an absolute accuracy measurement.</p>
<h2>Optional semantic layer</h2>
<p>Status: <strong>{escape(semantics.get('status'))}</strong>.
Recorded descriptor status: {escape(semantics.get('recorded_status'))}.
S9 stage: {escape(semantics.get('stage_status'))}. Current metadata: {escape(semantics.get('current'))}.</p>
<p>{escape(semantics.get('reason'))} {escape(semantics.get('availability_reason'))}</p>
<p>Current supported semantic cells: {escape(semantics.get('supported_cells'))}.
Semantic accuracy: <strong>{escape(semantics.get('accuracy_status'))}</strong>.
Semantic probabilities are uncalibrated, distinct from geometry confidence and absolute accuracy.</p>
<p>Recorded confidence method: {escape(semantics.get('confidence_method'))}.
Source descriptor: {escape(semantics.get('source_descriptor'))} {escape(semantics.get('source_field'))}.</p>
<table><thead><tr><th>Class ID</th><th>Class name</th><th>RGB</th><th>Source channels</th></tr></thead>
<tbody>{''.join(semantic_rows)}</tbody></table>
<h3>Recorded semantic provenance (may be retained)</h3><pre>{semantic_provenance}</pre>
<h2>Optional barometer fusion</h2>
<p>Status: <strong>{escape(barometer.get('status'))}</strong>. Recorded final status: {escape(barometer.get('recorded_status'))}.
Final stage: {escape(barometer.get('stage'))}; stage status: {escape(barometer.get('stage_status'))}; current: {escape(barometer.get('current'))}.</p>
<p>{escape(barometer.get('reason'))} {escape(barometer.get('availability_reason'))}</p>
<p>Configuration enabled: {escape(barometer.get('configuration', {}).get('enabled'))}.
S0 input stage status: {escape(barometer_input.get('stage_status'))}; current input metadata: {escape(barometer_input.get('current'))}.
Input status: {escape(barometer_input.get('status'))}; provided: {escape(barometer_input.get('provided'))}; available: {escape(barometer_input.get('available'))};
parsed: {escape(barometer_input.get('parsed'))}; samples: {escape(barometer_input.get('sample_count'))}; aligned frames: {escape(barometer_input.get('aligned_frames'))}.
Parsing/alignment does not establish accepted factor use.</p>
<p>Input reason: {escape(barometer_input.get('reason'))}. Recorded clock available: {escape(barometer_input.get('clock_available'))}.
Clock: {escape(barometer_input.get('clock'))}. Input descriptor: {escape(barometer_input.get('source_descriptor'))}.</p>
<p>Selected final source: {escape(barometer.get('final_source'))}; method: {escape(barometer.get('method'))}; current final use: {escape(barometer.get('used'))}.
Source descriptor: {escape(barometer.get('source_descriptor'))} {escape(barometer.get('source_field'))}.</p>
<p>Current factors attempted / added / accepted / rejected: {escape(barometer.get('factors_attempted'))} / {escape(barometer.get('factors_added'))} /
{escape(barometer.get('factors_accepted'))} / {escape(barometer.get('factors_rejected'))}.
Current nuisance bias: {escape(barometer.get('bias_m'))} m; sensor fit RMSE: {escape(barometer.get('fit_rmse_m'))} m.</p>
<p>{escape(barometer.get('interpretation'))} Absolute sensor datum: {escape(barometer.get('absolute_datum'))}.</p>
<p>Actual selected uncertainty provenance: {escape(barometer.get('uncertainty_provenance'))}.</p>
<h3>Recorded barometer attempts and inherited provenance (may be retained)</h3><pre>{barometer_provenance}</pre>
{cloud_section}
 <h2>Measured stage timings</h2><table><thead><tr><th>Stage</th><th>Status</th><th>Wall seconds</th><th>Confidence</th><th>Confidence status</th><th>Confidence reason</th><th>Degradation</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>Actual stage environments are recorded in the detailed provenance below. Missing measurements remain null.</p>
<h2>Actual exports</h2><table><thead><tr><th>Format</th><th>Status</th><th>Bundle path</th><th>Reason</th></tr></thead>
<tbody>{''.join(export_rows)}</tbody></table><h2>Measurements and provenance</h2><pre>{detail}</pre>
</body></html>"""


def build_report(bundle, cfg) -> dict[str, str]:
    """Build an inspectable report even before any stage/CRS/export exists.

    Return logical-name → bundle-relative paths, suitable for StageResult.outputs.
    Optional analysis failures are captured as report issues, not invented values.
    """
    manifest = bundle.manifest
    issues: list[str] = []
    directory = bundle.stage_dir("report")

    def relative(path):
        return path.relative_to(bundle.root).as_posix()

    def optional_artifact(stage, key):
        record = (manifest.get("stages") or {}).get(stage, {}) if isinstance(manifest.get("stages"), dict) else {}
        if not isinstance(record, dict):
            return None
        if record.get("status") is not None and record["status"] not in ("done", "degraded"):
            return None
        try:
            return bundle.artifact(stage, key)
        except Exception:
            return None

    def optional_json(stage, key):
        path = optional_artifact(stage, key)
        if path is None:
            return None
        try:
            data = bundle.read_json(relative(path))
            if not isinstance(data, dict):
                raise ValueError("Artifact metadata must be a JSON object")
            return data
        except Exception as exc:
            issues.append(f"Cannot read {stage}/{key}: {exc}")
            return None

    report_cfg = getattr(cfg, "report", None)
    target = finite_number(getattr(report_cfg, "accuracy_target_m", None))
    stages = []
    for record in stage_records(manifest):
        confidence = record.get("confidence")
        if isinstance(confidence, dict):
            confidence = confidence.get("summary")
        confidence = finite_number(confidence)
        if confidence is not None and not 0 <= confidence <= 1:
            confidence = None
        if record.get("status") is not None and record["status"] not in ("done", "degraded"):
            confidence = None
        confidence_status = record.get("confidence_status")
        confidence_reason = record.get("confidence_reason")
        if confidence_status not in ("measured_proxy", "unavailable"):
            confidence_status = "measured_proxy" if confidence is not None else "unavailable"
            confidence_reason = None if confidence is not None else (
                record.get("degraded_reason") or "Stage did not publish a numeric confidence summary")
        wall = finite_number(record.get("wall_seconds"))
        if wall is not None and wall < 0:
            wall = None
        stages.append({"name": record.get("name"), "status": record.get("status"),
                       "wall_seconds": wall, "environment": record.get("environment"),
                       "confidence": confidence, "confidence_status": confidence_status,
                       "confidence_reason": confidence_reason,
                       "degraded_reason": record.get("degraded_reason"),
                       "metrics": record.get("metrics") if isinstance(record.get("metrics"), dict) else {},
                       "outputs": record.get("outputs") or {}, "error": record.get("error"),
                       "models": record.get("models") or []})
    geo = optional_json("s9_geo", "metadata") or {}
    semantics = semantic_summary(bundle, geo, issues)
    barometer = barometer_summary(bundle, cfg, issues)
    if "semantics" in geo:
        geo["semantics"] = semantics
    exports = optional_json("s10_export", "exports") or {"formats": {}, "georeference": None}
    if not isinstance(exports.get("formats"), dict):
        issues.append("Export format metadata is missing or malformed")
        exports["formats"] = {}
    for kind, record in exports.get("formats", {}).items():
        if not isinstance(record, dict):
            exports["formats"][kind] = {"status": "failed", "path": None,
                                         "reason": "Malformed format metadata"}
            issues.append(f"Export {kind}: malformed format metadata")
            continue
        if record.get("status") == "ok":
            try:
                path = (bundle.root / record["path"]).resolve()
                path.relative_to(bundle.root.resolve())
                if not path.exists() or (path.is_file() and not path.stat().st_size):
                    raise ValueError("recorded successful output is missing or empty")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                record.update(status="failed", path=None, reason=str(exc))
                issues.append(f"Export {kind}: {exc}")
    regional = {"status": "UNAVAILABLE", "regions": [], "mean_observed_confidence": None}
    cloud = optional_artifact("s7_dense", "cloud")
    if cloud is not None:
        try:
            from drishti.geo.products import read_cloud

            if report_cfg is None:
                raise ValueError("report.region_size_m is not configured")
            chunk = getattr(getattr(cfg, "dense", None), "max_points_per_chunk", None)
            regional = confidence_regions(read_cloud(bundle), report_cfg.region_size_m, chunk_size=chunk)
        except Exception as exc:
            issues.append(f"Regional confidence unavailable: {exc}")
    elevation: dict[str, Any] = {"status": "UNVALIDATED", "rmse_z_m": None,
                 "reason": "No independent checkpoint data supplied"}
    check_path = getattr(report_cfg, "check_points", None)
    if check_path is None:
        dataset_report = getattr(getattr(cfg, "dataset", None), "report", None)
        check_path = getattr(dataset_report, "check_points", None)
    if check_path is not None:
        dsm = optional_artifact("s9_geo", "dsm")
        if dsm is None:
            elevation["reason"] = "DSM unavailable for supplied checkpoint comparison"
        else:
            try:
                from drishti.report.accuracy import validate_checkpoints

                check_path = Path(check_path)
                if not check_path.is_absolute():
                    check_path = bundle.root / check_path
                elevation = validate_checkpoints(check_path, dsm, target_m=target)
            except Exception as exc:
                checkpoint_error = f"Checkpoint validation unavailable: {exc}"
                elevation["reason"] = checkpoint_error
                issues.append(checkpoint_error)
    fit = [{"stage": stage["name"], "status": stage["status"],
            "metrics": fit_metrics(stage["metrics"])}
           for stage in stages if stage["name"] in ("s2_poses", "s6_global")]
    dataset = getattr(cfg, "dataset", None)
    measured_times = [stage["wall_seconds"] for stage in stages]
    report = {
        "schema_version": "1", "generated_utc": datetime.now(UTC).isoformat(),
        "run_id": manifest.get("run_id"), "content_id": manifest.get("content_id"),
        "dataset_name": getattr(dataset, "name", None), "status": manifest.get("status", "partial"),
        "accuracy": {"status": "UNVALIDATED", "design_target_m": target,
                     "absolute_spatial_rmse_m": None, "horizontal_rmse_m": None,
                     "reason": "No independent corresponding 3D reference measurement is available",
                     "elevation_validation": elevation},
        "gps_fit": {"independent_accuracy": False, "stages": fit,
                    "interpretation": "Residuals against the GNSS used to fit the solution"},
        "coverage": {"scene_completeness": None,
                     "reason": "Visible-scene target surface denominator is not supplied",
                     "supported_area_m2": geo.get("supported_area_m2"),
                     "bounding_grid_coverage_fraction": geo.get("bounding_grid_coverage_fraction"),
                     "bounding_grid_fraction_is_scene_completeness": False,
                     "supported_cells": geo.get("supported_cells"),
                     "total_grid_cells": geo.get("total_cells")},
        "confidence": regional, "semantics": semantics, "barometer_fusion": barometer, "stages": stages,
        "timing": {"recorded_stage_wall_seconds_sum": sum(measured_times)
                   if measured_times and all(time is not None for time in measured_times) else None,
                    "end_to_end_wall_seconds": finite_number(manifest.get("wall_seconds")),
                    "last_invocation_wall_seconds": finite_number(manifest.get("last_invocation_wall_seconds")),
                    "executions": manifest.get("executions", []),
                   "environment": manifest.get("environment"),
                   "interpretation": "Recorded execution measurements; cached work is not fresh wall time"},
        "georeference": geo.get("georeference") or (
            exports["georeference"].get("reference")
            if isinstance(exports.get("georeference"), dict) else None
        ) or manifest.get("crs") or None,
        "geospatial_products": geo, "exports": exports,
        "inputs": manifest.get("inputs", []), "models": manifest.get("models", []),
        "failure": manifest.get("error"), "issues": issues,
    }
    if isinstance(manifest.get("cloud_execution"), dict):
        # This is recorded orchestration evidence, separate from mission accuracy
        # and core end-to-end timing. The final runner refresh freezes its scope.
        report["cloud_execution"] = manifest["cloud_execution"]
    report = json_safe(report)
    json_path, html_path = directory / "report.json", directory / "report.html"
    bundle.write_json(relative(json_path), report)
    fd, temporary = tempfile.mkstemp(prefix=".report-", suffix=".html", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(render_html(report))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, html_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"json": relative(json_path), "html": relative(html_path)}
