"""Constructed numerical/report contracts, not recorded-flight or mission acceptance.

The final test executes real S6 GTSAM/SciPy solvers on a small mathematical scene.
Neither its fitted sensor residual nor the metadata specimens establish accuracy.
"""

import copy
import html
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from drishti.bundle import Bundle
from drishti.config import Config
from drishti.report.builder import build_report
from drishti.stages import s6_global

MODEL = "relative_height_with_constant_unknown_bias"
CURRENT_FIELDS = ("used", "bias_m", "fit_rmse_m", "factors_added", "factors_attempted",
                  "factors_accepted", "factors_rejected")


@pytest.fixture
def report_fixture(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    bundle = Bundle(root, {"bundle_version": "1", "status": "partial", "stages": {},
                           "inputs": [], "config_snapshot": {}, "models": []})
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(name="constructed barometer report boundary"),
        barometer=SimpleNamespace(enabled=True, sigma_m=.4),
        report=SimpleNamespace(check_points=None, accuracy_target_m=1., region_size_m=1.),
        dense=SimpleNamespace(max_points_per_chunk=1),
    )
    return bundle, cfg


def observation(index):
    return {"source_time_s": 1000. + index, "time_s": float(index),
            "height_m": 50. + .03 * index, "sigma_m": .4,
            "sigma_source": "csv", "source_index": index, "source_line": index + 2,
            "time_error_s": -.01, "frame_time_s": index + .01}


def diagnostics(used=True, *, rejected=False, disabled=False, bias=-50.):
    selected = [] if disabled else [{"camera_index": i, **observation(i)} for i in range(4)]
    added = 4 if used or rejected else 0
    return {
        "status": "optimized" if used else "disabled" if disabled else
                  "failed_retained_input" if rejected else "unavailable",
        "reason": None if used else "Recorded sensor support not accepted",
        "factors_added": added, "factors_attempted": added,
        "factors_accepted": 4 if used else 0, "factors_rejected": 4 if rejected else 0,
        "optimizer_invoked": bool(added), "selected": selected, "excluded": [],
        "distinct_measurements": len(selected), "min_measurements": 3,
        "sigma_used": [{key: item[key] for key in ("camera_index", "source_index", "sigma_m", "sigma_source")}
                       for item in selected],
        "bias_m": bias if used else None,
        "fit_rmse_m": math.sqrt(sum(value**2 for value in (.1, -.2, .3, -.4)) / 4) if used else None,
        "fit_is_independent_accuracy": False, "model": MODEL, "absolute_datum": "not_interpreted",
        "timing_approximation": "bounded_nearest_recorded_sample_no_interpolation",
        "bias_prior": None,
        "bias_initialization": "median_recorded_height_minus_initial_camera_local_z_not_calibration",
    }


def attempt(used=True, *, status="optimized", disabled=False):
    return {"status": status, "method": "constructed_solver_metadata_not_execution",
            "optimizer_invoked": status == "optimized" or status == "failed_retained_input",
            "used": used, "reason": None if used else "Actual recorded attempt reason",
            "barometer": diagnostics(used, rejected=status == "failed_retained_input", disabled=disabled)}


def fusion(stage="s6_global", *, source="bundle_adjustment", used=True, disabled=False):
    graph = attempt(True)
    ba = attempt(used, disabled=disabled)
    inherited = fusion("s2_poses") if stage == "s6_global" else None
    if stage == "s2_poses":
        source, graph, ba = "pose_graph", attempt(used, disabled=disabled), None
        selected = graph
    elif source == "pose_graph":
        graph = attempt(used, disabled=disabled)
        ba = attempt(False, status="failed_retained_input")
        selected = graph
    elif source == "retained_s2":
        graph, ba = (attempt(False, status="failed_retained_input") for _ in range(2))
        inherited = fusion("s2_poses", used=used, disabled=disabled)
        selected = inherited
    else:
        selected = ba
    result = {"stage": stage, "method": selected["method"],
              "status": "used" if used else "disabled" if disabled else "not_used",
              "used": used, "reason": selected["reason"], "final_source": source,
              "barometer": copy.deepcopy(selected["barometer"]),
              "attempts": {"pose_graph": graph, "bundle_adjustment": ba}}
    if inherited is not None:
        result["inherited"] = inherited
    return result


def record_fusion(bundle, summary, *, stage_status="done", location="all", partial=False):
    name = summary["stage"]
    filename = "spine.json" if name == "s2_poses" else "optimization.json"
    key = "spine" if name == "s2_poses" else "optimization"
    document = {"barometer_fusion": summary}
    bundle.write_json(f"{name}/{filename}", document)
    bundle.write_json(f"{name}/poses.json", document)
    outputs = {key: f"{name}/{filename}", "poses": f"{name}/poses.json"}
    if partial:
        outputs = {"partial": name}
    elif location == "metrics":
        outputs = {}
    elif location == "poses":
        outputs = {"poses": outputs["poses"]}
    elif location == "diagnostics":
        outputs.pop("poses")
    record = {"status": stage_status, "outputs": outputs}
    if location in ("all", "metrics") and not partial:
        record["metrics"] = {"barometer_fusion": copy.deepcopy(summary)}
    bundle.manifest["stages"][name] = record
    return record


def input_descriptor(status="ok"):
    samples = [{key: value for key, value in observation(i).items()
                if key not in ("frame_time_s", "time_error_s")} for i in range(4)] if status == "ok" else []
    return {
        "status": status, "reason": None if status == "ok" else f"Recorded input {status}",
        "samples": samples, "aligned_frames": 4 if samples else 0,
        "measurement": "relative_sensor_height_m", "absolute_datum": "not_interpreted",
        "clock": {"time_origin_s": 1000. if samples else None, "time_offset_s": -.05,
                  "max_time_error_s": .1, "alignment": "bounded_nearest_recorded_sample"},
        "uncertainty": {"sigma_column": "logged_sigma", "assumed_sigma_m": None,
                        "calibrated_accuracy": False},
    }


def record_input(bundle, *, status="ok", stage_status="done", partial=False):
    descriptor = input_descriptor(status)
    ledger = {"provided": status != "not_provided", "available": status == "ok",
              "parsed": status == "ok", "used": False, "sample_count": len(descriptor["samples"]),
              "aligned_frames": descriptor["aligned_frames"],
              "parsing_status": status, "alignment_status": status,
              "status": "aligned_not_fused" if status == "ok" else status,
              "reason": descriptor["reason"]}
    bundle.write_json("s0_ingest/barometer.json", descriptor)
    bundle.write_json("s0_ingest/frames.json", {"frames": [], "capabilities": {"baro": ledger}})
    outputs = {"partial": "s0_ingest"} if partial else {
        "barometer": "s0_ingest/barometer.json", "frames": "s0_ingest/frames.json"}
    bundle.manifest["stages"]["s0_ingest"] = {"status": stage_status, "outputs": outputs,
                                             "metrics": {"optional_inputs": {"baro": ledger}}}
    return descriptor, ledger


def add_geometry(bundle):
    directory = bundle.stage_dir("s7_dense")
    np.savez(directory / "dense.npz", points=np.array([[.5, .5, 1.], [1.5, .5, 2.]], np.float64),
             colors=np.full((2, 3), 100, np.uint8), confidence=np.array([.4, .9], np.float32),
             inferred=np.array([False, True]))
    bundle.manifest["stages"]["s7_dense"] = {"status": "done", "outputs": {"cloud": "s7_dense/dense.npz"}}


def read_report(bundle, cfg):
    outputs = build_report(bundle, cfg)
    assert outputs == {"json": "report/report.json", "html": "report/report.html"}
    report = bundle.read_json(outputs["json"])
    assert report["accuracy"]["status"] == "UNVALIDATED"
    assert report["accuracy"]["absolute_spatial_rmse_m"] is None
    assert report["gps_fit"]["independent_accuracy"] is False
    assert report["barometer_fusion"]["independent_accuracy"] is False
    assert report["barometer_fusion"]["fit_is_independent_accuracy"] is False
    assert report["barometer_fusion"]["model"] == MODEL
    assert report["barometer_fusion"]["absolute_datum"] == "not_interpreted"
    json.dumps(report, allow_nan=False)
    return report, bundle.path(outputs["html"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("enabled", [True, False, None])
def test_no_recorded_solve_is_unavailable_even_with_configuration_or_registry(report_fixture, enabled):
    bundle, cfg = report_fixture
    cfg.barometer.enabled = enabled
    bundle.manifest["models"] = [{"name": "detected_not_executed", "precision_m": .001}]
    report, markup = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["status"] == "unavailable" and sensor["current"] is False
    assert sensor["configuration"]["enabled"] is enabled
    assert sensor["stage"] is None and sensor["provenance"] == []
    assert all(sensor[key] is None for key in CURRENT_FIELDS)
    assert sensor["uncertainty_provenance"] is None
    assert "Optional barometer fusion" in markup


@pytest.mark.parametrize("stage", ["s2_poses", "s6_global"])
def test_old_bundle_does_not_infer_use_from_optional_input_ledger(report_fixture, stage):
    bundle, cfg = report_fixture
    bundle.write_json(f"{stage}/poses.json", {"optional_inputs": {"baro": {"used": True, "available": True}}})
    bundle.manifest["stages"][stage] = {"status": "done", "outputs": {"poses": f"{stage}/poses.json"}}
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["stage"] == stage and sensor["status"] == "unavailable"
    assert all(sensor[key] is None for key in CURRENT_FIELDS)
    assert sensor["recorded_status"] is None and report["issues"] == []


@pytest.mark.parametrize("location", ["all", "metrics", "poses", "diagnostics"])
@pytest.mark.parametrize("stage", ["s2_poses", "s6_global"])
def test_accepted_final_factors_preserve_numerical_fit_bias_uncertainty_and_attempts(report_fixture, location, stage):
    bundle, cfg = report_fixture
    summary = fusion(stage)
    record_fusion(bundle, summary, location=location, stage_status="degraded")
    report, markup = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["status"] == sensor["recorded_status"] == "used"
    assert sensor["current"] is True and sensor["stage_status"] == "degraded"
    assert sensor["used"] is True and sensor["stage"] == stage
    assert sensor["final_source"] == summary["final_source"]
    assert sensor["bias_m"] == -50.  # Relative nuisance offsets may legitimately be negative.
    assert sensor["fit_rmse_m"] == pytest.approx(math.sqrt(.075))
    assert [sensor[key] for key in CURRENT_FIELDS[3:]] == [4, 4, 4, 0]
    assert sensor["attempts"] == summary["attempts"] and sensor["inherited"] == summary.get("inherited")
    assert sensor["uncertainty_provenance"]["sigma_used"] == summary["barometer"]["sigma_used"]
    assert sensor["provenance"][0]["summary"] == summary
    assert sensor["provenance"][0]["validation_status"] == "valid"
    assert report["issues"] == []
    assert "Current nuisance bias: -50.0 m" in markup
    assert "not independent accuracy or measured calibration" in markup


@pytest.mark.parametrize("source", ["pose_graph", "retained_s2"])
def test_graph_fallback_and_retained_s2_report_actual_selected_diagnostics(report_fixture, source):
    bundle, cfg = report_fixture
    summary = fusion(source=source)
    record_fusion(bundle, summary)
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["used"] is True and sensor["final_source"] == source
    assert sensor["bias_m"] == summary["barometer"]["bias_m"]
    assert sensor["attempts"]["bundle_adjustment"]["barometer"]["factors_rejected"] == 4
    if source == "retained_s2":
        assert sensor["inherited"] == summary["inherited"]
        assert not sensor["attempts"]["pose_graph"]["used"]
    assert sensor["factors_rejected"] == 0  # Attempts are not final-factor counts.
    assert report["issues"] == []


def test_accepted_ba_without_sensor_does_not_promote_used_graph_or_s2(report_fixture):
    bundle, cfg = report_fixture
    summary = fusion(used=False)
    record_fusion(bundle, summary)
    report, markup = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["current"] and sensor["status"] == "not_used" and sensor["used"] is False
    assert sensor["final_source"] == "bundle_adjustment"
    assert sensor["attempts"]["pose_graph"]["used"] and sensor["inherited"]["used"]
    assert sensor["bias_m"] is None and sensor["fit_rmse_m"] is None
    assert [sensor[key] for key in CURRENT_FIELDS[3:]] == [0, 0, 0, 0]
    assert "current final use: False" in markup


@pytest.mark.parametrize("kind", ["solve", "descriptor"])
def test_disabled_requires_recorded_descriptor_or_solve_not_current_configuration(report_fixture, kind):
    bundle, cfg = report_fixture
    cfg.barometer.enabled = True  # Intent may differ from the recorded execution.
    if kind == "solve":
        record_fusion(bundle, fusion(used=False, disabled=True))
    else:
        record_input(bundle, status="disabled")
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["status"] == "disabled" and sensor["configuration"]["enabled"] is True
    assert sensor["used"] is (False if kind == "solve" else None)
    assert sensor["bias_m"] is None and sensor["fit_rmse_m"] is None


def test_configuration_disabled_can_still_report_accepted_retained_s2_factors(report_fixture):
    bundle, cfg = report_fixture
    cfg.barometer.enabled = False
    record_fusion(bundle, fusion(source="retained_s2"))
    report, _ = read_report(bundle, cfg)
    assert report["barometer_fusion"]["used"] is True
    assert report["barometer_fusion"]["configuration"]["enabled"] is False


@pytest.mark.parametrize("disabled", [True, False])
def test_legacy_solve_without_diagnostics_has_only_explicit_disabled_claim(report_fixture, disabled):
    bundle, cfg = report_fixture
    summary = fusion("s2_poses", used=False, disabled=disabled)
    summary["barometer"] = None
    summary["attempts"]["pose_graph"]["barometer"] = None
    record_fusion(bundle, summary)
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["status"] == ("disabled" if disabled else "unavailable")
    assert sensor["used"] is (False if disabled else None)
    assert sensor["factors_accepted"] is None and sensor["fit_rmse_m"] is None
    assert report["issues"] == []


def test_alignment_nonoverlap_retains_parsing_success_and_actual_clock(report_fixture):
    bundle, cfg = report_fixture
    descriptor, ledger = record_input(bundle)
    descriptor.update(status="unavailable", reason="no_decoded_frame_has_a_bounded_barometer_sample",
                      aligned_frames=0)
    ledger.update(status="unavailable", aligned_frames=0, alignment_status="unavailable",
                  reason=descriptor["reason"])
    bundle.write_json("s0_ingest/barometer.json", descriptor)
    report, _ = read_report(bundle, cfg)
    input_summary = report["barometer_fusion"]["input"]
    assert input_summary["parsed"] is True and input_summary["sample_count"] == 4
    assert input_summary["aligned_frames"] == 0 and input_summary["status"] == "unavailable"
    assert input_summary["clock_available"] is True and input_summary["clock"] == descriptor["clock"]
    assert report["barometer_fusion"]["used"] is None


def test_input_ledger_alone_cannot_create_disabled_descriptor_claim(report_fixture):
    bundle, cfg = report_fixture
    _, ledger = record_input(bundle, status="disabled")
    bundle.manifest["stages"]["s0_ingest"]["outputs"] = {"frames": "s0_ingest/frames.json"}
    report, _ = read_report(bundle, cfg)
    assert report["barometer_fusion"]["input"]["ledger"] == ledger
    assert report["barometer_fusion"]["input"]["descriptor_valid"] is False
    assert report["barometer_fusion"]["status"] == "unavailable"
    assert report["barometer_fusion"]["used"] is None


def test_factor_noise_comes_from_executed_diagnostics_not_current_configuration(report_fixture):
    bundle, cfg = report_fixture
    cfg.barometer.sigma_m = 999.
    summary = fusion()
    record_fusion(bundle, summary)
    report, _ = read_report(bundle, cfg)
    sigmas = report["barometer_fusion"]["uncertainty_provenance"]["sigma_used"]
    assert sigmas == summary["barometer"]["sigma_used"]
    assert all(item["sigma_m"] == .4 and item["sigma_source"] == "csv" for item in sigmas)


def test_retained_final_stage_input_ledger_is_inspectable_without_s0_invention(report_fixture):
    bundle, cfg = report_fixture
    summary = fusion()
    record_fusion(bundle, summary, location="poses", stage_status="failed")
    ledger = {"baro": {"provided": True, "available": True, "parsed": True,
                       "aligned_frames": 4, "used": True, "status": "used"}}
    bundle.write_json("s6_global/poses.json", {"barometer_fusion": summary, "optional_inputs": ledger})
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["input"]["stage_status"] is None and sensor["input"]["parsed"] is None
    assert sensor["provenance"][-1]["optional_inputs"] == ledger
    assert sensor["provenance"][-1]["stage_status"] == "failed"
    assert sensor["provenance"][-1]["current"] is False and sensor["used"] is None


@pytest.mark.parametrize("status", ["ok", "not_provided", "unavailable", "disabled"])
def test_s0_availability_parsing_alignment_clock_are_not_final_use(report_fixture, status):
    bundle, cfg = report_fixture
    descriptor, ledger = record_input(bundle, status=status)
    report, markup = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    input_summary = sensor["input"]
    assert input_summary["current"] and input_summary["stage_status"] == "done"
    assert input_summary["status"] == status and input_summary["used"] is False
    assert input_summary["ledger"] == ledger
    assert input_summary["sample_count"] == len(descriptor["samples"])
    assert input_summary["parsed"] is (status == "ok")
    assert input_summary["aligned_frames"] == descriptor["aligned_frames"]
    assert input_summary["clock"] == descriptor["clock"]
    assert input_summary["clock_available"] is (status == "ok")
    assert input_summary["uncertainty"] == descriptor["uncertainty"]
    assert sensor["status"] == ("disabled" if status == "disabled" else "unavailable")
    assert all(sensor[key] is None for key in CURRENT_FIELDS)
    assert "Parsing/alignment does not establish accepted factor use" in markup


@pytest.mark.parametrize("stage_status", ["stale", "failed", "blocked", "running", "partial", None])
@pytest.mark.parametrize("partial", [False, True])
def test_newest_s6_retained_metadata_never_falls_back_to_current_s2(report_fixture, stage_status, partial):
    bundle, cfg = report_fixture
    prior = fusion("s2_poses")
    latest = fusion()
    record_fusion(bundle, prior)
    record_fusion(bundle, latest, stage_status=stage_status, partial=partial)
    report, markup = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["stage"] == "s6_global" and sensor["stage_status"] == stage_status
    assert sensor["status"] == "unavailable" and sensor["recorded_status"] == "used"
    assert sensor["current"] is False and all(sensor[key] is None for key in CURRENT_FIELDS)
    assert sensor["uncertainty_provenance"] is None
    assert sensor["attempts"] == latest["attempts"] and sensor["inherited"] == latest["inherited"]
    assert "retained barometer metadata is not current" in sensor["availability_reason"]
    retained = sensor["provenance"][-1]
    assert retained["summary"] == latest and retained["stage_status"] == stage_status
    assert retained["current"] is False and retained["selected_final"] is True
    assert sensor["provenance"][0]["selected_final"] is False
    assert report["issues"] == [] and "retained barometer metadata is not current" in markup


@pytest.mark.parametrize("stage_status", ["failed", "stale", "running"])
def test_recorded_s6_without_summary_blocks_s2_current_use(report_fixture, stage_status):
    bundle, cfg = report_fixture
    record_fusion(bundle, fusion("s2_poses"))
    bundle.manifest["stages"]["s6_global"] = {"status": stage_status, "outputs": {}}
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["stage"] == "s6_global" and sensor["stage_status"] == stage_status
    assert sensor["status"] == "unavailable" and sensor["used"] is None
    assert sensor["provenance"][0]["summary"]["used"] is True


def test_failed_s0_partial_directory_preserves_clock_and_parsed_samples(report_fixture):
    bundle, cfg = report_fixture
    descriptor, _ = record_input(bundle, stage_status="failed", partial=True)
    bundle.manifest["stages"]["s0_ingest"].pop("metrics")
    report, markup = read_report(bundle, cfg)
    input_summary = report["barometer_fusion"]["input"]
    assert input_summary["current"] is False and input_summary["stage_status"] == "failed"
    assert input_summary["parsed"] is True and input_summary["sample_count"] == 4
    assert input_summary["clock"] == descriptor["clock"] and input_summary["clock_available"]
    assert input_summary["source_descriptor"] == "s0_ingest/barometer.json"
    assert report["barometer_fusion"]["used"] is None and report["issues"] == []
    assert "current input metadata: False" in markup


@pytest.mark.parametrize("stage_status", ["stale", "failed", None])
def test_retained_s2_partial_directory_has_provenance_but_no_current_use(report_fixture, stage_status):
    bundle, cfg = report_fixture
    summary = fusion("s2_poses")
    record_fusion(bundle, summary, stage_status=stage_status, partial=True)
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["stage"] == "s2_poses" and sensor["stage_status"] == stage_status
    assert sensor["recorded_status"] == "used" and sensor["status"] == "unavailable"
    assert sensor["provenance"][0]["summary"] == summary
    assert sensor["provenance"][0]["current"] is False
    assert all(sensor[key] is None for key in CURRENT_FIELDS)
    assert report["issues"] == []


def test_no_loose_discovery_of_unrecorded_artifacts(report_fixture):
    bundle, cfg = report_fixture
    bundle.write_json("s6_global/optimization.json", {"barometer_fusion": fusion()})
    bundle.write_json("s0_ingest/barometer.json", input_descriptor("disabled"))
    bundle.manifest["stages"]["s6_global"] = {"status": "done", "outputs": {}}
    report, _ = read_report(bundle, cfg)
    assert report["barometer_fusion"]["status"] == "unavailable"
    assert report["barometer_fusion"]["input"]["source_descriptor"] is None
    assert report["barometer_fusion"]["used"] is None and report["issues"] == []


def test_unsuccessful_attempt_with_real_rejected_factors_has_null_final_measurements(report_fixture):
    bundle, cfg = report_fixture
    summary = fusion("s2_poses", used=False)
    rejected = attempt(False, status="failed_retained_input")
    summary.update(method=rejected["method"], barometer=copy.deepcopy(rejected["barometer"]))
    summary["attempts"]["pose_graph"] = rejected
    record_fusion(bundle, summary)
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["status"] == "unavailable" and sensor["recorded_status"] == "not_used"
    assert all(sensor[key] is None for key in CURRENT_FIELDS)
    assert sensor["attempts"]["pose_graph"]["barometer"]["factors_added"] == 4
    assert sensor["attempts"]["pose_graph"]["barometer"]["factors_rejected"] == 4
    assert report["issues"] == []


@pytest.mark.parametrize("failure", [
    "nonobject", "status", "stage", "source", "attempt_status", "not_invoked", "diag_not_invoked",
    "zero_factors", "negative_factors", "boolean_factors", "missing_accepted", "rejected_accepted",
    "nan_bias", "infinite_fit", "negative_fit", "model", "absolute_datum", "independent_accuracy",
    "sigma_zero", "sigma_source", "uncertainty_mismatch", "duplicate_sample", "support_count",
    "final_mismatch", "inherited_missing", "inherited_invalid",
    "bias_prior", "excluded_index", "diagnostic_status",
])
def test_invalid_final_metadata_cannot_claim_use_or_break_geometry(report_fixture, failure):
    bundle, cfg = report_fixture
    summary = fusion(source="retained_s2" if failure.startswith("inherited") else "bundle_adjustment")
    diag = summary["barometer"]
    if failure == "nonobject":
        summary = []
    elif failure == "status":
        summary["status"] = []
    elif failure == "stage":
        summary["stage"] = "s2_poses"
    elif failure == "source":
        summary["final_source"] = "pose_graph"
    elif failure == "attempt_status":
        summary["attempts"]["bundle_adjustment"]["status"] = "failed_retained_input"
    elif failure == "not_invoked":
        summary["attempts"]["bundle_adjustment"]["optimizer_invoked"] = False
    elif failure == "diag_not_invoked":
        diag["optimizer_invoked"] = False
    elif failure in ("zero_factors", "negative_factors", "boolean_factors"):
        diag["factors_added"] = {"zero_factors": 0, "negative_factors": -1, "boolean_factors": True}[failure]
    elif failure == "missing_accepted":
        diag.pop("factors_accepted")
    elif failure == "rejected_accepted":
        diag["factors_rejected"] = 1
    elif failure == "nan_bias":
        diag["bias_m"] = float("nan")
    elif failure == "infinite_fit":
        diag["fit_rmse_m"] = float("inf")
    elif failure == "negative_fit":
        diag["fit_rmse_m"] = -.1
    elif failure == "model":
        diag["model"] = "absolute_altitude"
    elif failure == "absolute_datum":
        diag["absolute_datum"] = "ellipsoidal"
    elif failure == "independent_accuracy":
        diag["fit_is_independent_accuracy"] = True
    elif failure == "sigma_zero":
        diag["selected"][0]["sigma_m"] = 0.
    elif failure == "sigma_source":
        diag["selected"][0]["sigma_source"] = "registry_precision"
    elif failure == "uncertainty_mismatch":
        diag["sigma_used"][0]["sigma_m"] = .001
    elif failure == "duplicate_sample":
        diag["selected"][1]["source_index"] = 0
    elif failure == "support_count":
        diag["distinct_measurements"] = 5
    elif failure == "final_mismatch":
        summary["used"] = False
    elif failure == "inherited_missing":
        summary.pop("inherited")
    elif failure == "inherited_invalid":
        summary["inherited"]["attempts"]["pose_graph"]["optimizer_invoked"] = False
    elif failure == "bias_prior":
        diag["bias_prior"] = {"sigma_m": .01}
    elif failure == "excluded_index":
        diag["excluded"] = [{"camera_index": -1, "source_index": None, "reason": "invalid"}]
    elif failure == "diagnostic_status":
        diag["status"] = []
    if isinstance(summary, dict) and not failure.startswith("inherited"):
        summary["attempts"]["bundle_adjustment"]["barometer"] = copy.deepcopy(diag)
    bundle.manifest["stages"]["s6_global"] = {
        "status": "done", "outputs": {}, "metrics": {"barometer_fusion": summary}}
    add_geometry(bundle)
    report, _ = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["status"] == "unavailable" and sensor["current"] is False
    assert all(sensor[key] is None for key in CURRENT_FIELDS)
    assert report["issues"] and "Cannot validate barometer" in report["issues"][0]
    assert report["confidence"]["mean_observed_confidence"] == pytest.approx(.4)
    assert report["confidence"]["observed_points"] == 1


@pytest.mark.parametrize("failure", ["json", "nonobject", "missing", "traversal", "absolute", "symlink", "partial_escape"])
def test_corrupt_or_escaped_recorded_paths_are_optional_and_confined(report_fixture, failure):
    bundle, cfg = report_fixture
    summary = fusion()
    record = record_fusion(bundle, summary, location="diagnostics")
    path = bundle.path("s6_global/optimization.json")
    outside = bundle.root.parent / "outside.json"
    outside.write_text(json.dumps({"barometer_fusion": summary, "secret": "outside-secret"}), encoding="utf-8")
    if failure == "json":
        path.write_text("{invalid JSON", encoding="utf-8")
    elif failure == "nonobject":
        bundle.write_json("s6_global/optimization.json", [])
    elif failure == "missing":
        path.unlink()
    elif failure == "traversal":
        record["outputs"]["optimization"] = "../outside.json"
    elif failure == "absolute":
        record["outputs"]["optimization"] = str(outside)
    elif failure == "symlink":
        path.unlink()
        path.symlink_to(outside)
    else:
        record["outputs"] = {"partial": "../"}
    add_geometry(bundle)
    report, markup = read_report(bundle, cfg)
    assert report["barometer_fusion"]["status"] == "unavailable"
    assert report["barometer_fusion"]["used"] is None
    assert report["issues"] and "Cannot read barometer" in report["issues"][0]
    assert report["confidence"]["mean_observed_confidence"] == pytest.approx(.4)
    assert "outside-secret" not in markup


def test_valid_recorded_poses_can_preserve_provenance_after_corrupt_optimization_file(report_fixture):
    bundle, cfg = report_fixture
    summary = fusion()
    record_fusion(bundle, summary, location="artifacts")
    bundle.path("s6_global/optimization.json").write_text("{broken", encoding="utf-8")
    report, _ = read_report(bundle, cfg)
    assert report["barometer_fusion"]["used"] is True
    assert report["barometer_fusion"]["source_descriptor"] == "s6_global/poses.json"
    assert len(report["issues"]) == 1


def test_conflicting_recorded_summaries_never_choose_arbitrary_current_use(report_fixture):
    bundle, cfg = report_fixture
    record = record_fusion(bundle, fusion())
    record["metrics"]["barometer_fusion"] = fusion(used=False)
    report, _ = read_report(bundle, cfg)
    assert report["barometer_fusion"]["used"] is None
    assert report["barometer_fusion"]["current"] is False
    assert "summaries disagree" in report["issues"][0]


@pytest.mark.parametrize("failure", ["json", "negative_count", "nonfinite_clock", "sigma", "used_ledger",
                                     "assumed_uncertainty", "calibrated_accuracy"])
def test_malformed_s0_optional_input_does_not_erase_valid_final_diagnostics(report_fixture, failure):
    bundle, cfg = report_fixture
    descriptor, ledger = record_input(bundle)
    if failure == "json":
        bundle.path("s0_ingest/barometer.json").write_text("{corrupt", encoding="utf-8")
    else:
        if failure == "negative_count":
            descriptor["aligned_frames"] = -1
        elif failure == "nonfinite_clock":
            descriptor["clock"]["time_origin_s"] = float("nan")
        elif failure == "sigma":
            descriptor["samples"][0]["sigma_m"] = 0.
        elif failure == "assumed_uncertainty":
            descriptor["uncertainty"]["assumed_sigma_m"] = -1.
        elif failure == "calibrated_accuracy":
            descriptor["uncertainty"]["calibrated_accuracy"] = True
        else:
            ledger["used"] = True
        # JSON specimen deliberately exercises legacy nonfinite reader input.
        bundle.path("s0_ingest/barometer.json").write_text(json.dumps(descriptor), encoding="utf-8")
    record_fusion(bundle, fusion())
    report, _ = read_report(bundle, cfg)
    assert report["barometer_fusion"]["used"] is True
    assert report["barometer_fusion"]["fit_rmse_m"] == pytest.approx(math.sqrt(.075))
    assert report["issues"]


def test_html_escapes_input_paths_reasons_solver_and_inherited_provenance(report_fixture):
    bundle, cfg = report_fixture
    attack = '<img src=x onerror="alert(1)">'
    descriptor, _ = record_input(bundle)
    descriptor["reason"] = attack
    input_path = "s0_ingest/<svg onload=alert(1)>.json"
    bundle.write_json(input_path, descriptor)
    bundle.manifest["stages"]["s0_ingest"]["outputs"]["barometer"] = input_path
    summary = fusion()
    summary["reason"] = "</pre><script>alert('reason')</script>"
    summary["attempts"]["pose_graph"]["method"] = attack
    summary["inherited"]["reason"] = attack
    record = record_fusion(bundle, summary, location="poses", stage_status="failed")
    output_path = "s6_global/<svg onload=alert(1)>.json"
    bundle.write_json(output_path, {"barometer_fusion": summary})
    record["outputs"]["poses"] = output_path
    report, markup = read_report(bundle, cfg)
    section = markup.split("<h2>Optional barometer fusion</h2>")[1].split("<h2>Measured stage timings</h2>")[0]
    assert report["barometer_fusion"]["recorded_status"] == "used"
    for raw in (attack, summary["reason"], input_path, output_path):
        assert html.escape(raw, quote=True) in section
    assert "&lt;/pre&gt;&lt;script&gt;" in section
    assert "<script>" not in markup and "<img src=x" not in markup and "<svg onload" not in markup


def test_actual_s6_gtsam_scipy_unit_scene_to_report_is_not_mission_acceptance(report_fixture):
    pytest.importorskip("gtsam")
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    bundle, _ = report_fixture
    cfg = Config.model_validate({
        "dataset": {"name": "mathematical S6 report unit scene; NOT mission acceptance",
                    "video": bundle.root / "unused.unit",
                    "telemetry": {"format": "csv", "path": bundle.root / "unused-gps.unit",
                                  "altitude_datum": "ellipsoidal", "time_origin_s": 0.}},
        "barometer": {"enabled": True, "time_origin_s": 1000.},
    })
    points = np.array([[x, y, 8. + (x + y) / 4] for x in (-1., 0., 1., 2.)
                       for y in (-.7, .1, .8)], dtype=np.float64)
    cameras = []
    for index in range(4):
        true_center = np.array([float(index), .15 * (index % 2), .03 * index])
        center = true_center + [.01 * index, -.015 * index, .005 * index]
        pixels = (points - true_center)[:, :2] / (points - true_center)[:, 2, None] * 160. + [120., 100.]
        cameras.append({
            "frame_id": f"math{index}", "image_path": f"unused{index}.png",
            "time_s": index + .01, "width": 240, "height": 200,
            "K": [[160., 0., 120.], [0., 160., 100.], [0., 0., 1.]],
            "distortion": [], "projection_domain": "undistorted_pinhole",
            "R_cw": np.eye(3).tolist(), "center": center.tolist(),
            "visual_reference": {"R_cw": np.eye(3).tolist(), "center": center.tolist()},
            "gps": (true_center + [.12, .02, .03]).tolist(), "gps_inlier": True,
            "gps_sample": {"sigma_h_m": .2, "sigma_v_m": .3},
            "observations": [{"point_index": i, "uv": uv.tolist()} for i, uv in enumerate(pixels)],
            "barometer": observation(index), "barometer_reason": None, "anchors": [],
        })
    bundle.write_json("s2_poses/poses.json", {
        "frame": "projected_local", "origin": [500000., 3000000., 100.],
        "crs": {"epsg": 32643, "vertical": "ellipsoidal"}, "cameras": cameras,
        "pose_revision": "s2_metric_spine", "sparse_path": "s2_poses/sparse.npz",
    })
    np.savez_compressed(bundle.path("s2_poses/sparse.npz"), points=points + [.015, -.02, .04],
                        colors=np.full((len(points), 3), 128, dtype=np.uint8),
                        confidence=np.full(len(points), .6), inferred=np.zeros(len(points), bool),
                        reprojection_error_px=np.full(len(points), 1.),
                        track_length=np.full(len(points), len(cameras), dtype=np.int64))
    bundle.manifest["stages"]["s2_poses"] = {
        "status": "done", "outputs": {"poses": "s2_poses/poses.json", "sparse": "s2_poses/sparse.npz"}}
    result = s6_global.STAGE.run(bundle, cfg, None)
    bundle.manifest["stages"]["s6_global"] = {
        "status": "degraded" if result.degraded_reason else "done",
        "outputs": result.outputs, "metrics": result.metrics}
    poses = bundle.read_json(result.outputs["poses"])
    assert poses["global_optimization"]["status"] == "optimized"
    assert poses["bundle_adjustment"]["status"] == "optimized"
    assert poses["bundle_adjustment"]["poses_changed"]
    report, markup = read_report(bundle, cfg)
    sensor = report["barometer_fusion"]
    assert sensor["used"] and sensor["current"] and sensor["final_source"] == "bundle_adjustment"
    assert sensor["attempts"]["pose_graph"]["used"] and sensor["attempts"]["bundle_adjustment"]["used"]
    assert sensor["bias_m"] == poses["barometer_fusion"]["barometer"]["bias_m"]
    final_errors = [camera["center"][2] + sensor["bias_m"] - camera["barometer"]["height_m"]
                    for camera in poses["cameras"]]
    assert sensor["fit_rmse_m"] == pytest.approx(np.linalg.norm(final_errors) / 2, abs=1e-12)
    assert sensor["bias_m"] == pytest.approx(49.97, abs=.02)
    assert sensor["fit_rmse_m"] < .01
    assert sensor["factors_accepted"] == sensor["factors_added"] == 4
    assert sensor["uncertainty_provenance"]["sigma_used"] == poses["bundle_adjustment"]["barometer"]["sigma_used"]
    assert report["issues"] == []
    assert "NOT mission acceptance" in markup
