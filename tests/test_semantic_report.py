"""Constructed report/file-reference boundaries, not inference or scene acceptance.

Semantic asset specimens exercise confined references only; the report does not
validate raster contents. Geometry confidence uses an actual constructed NPZ.
"""

import copy
import html
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from drishti.bundle import Bundle
from drishti.report.builder import build_report

CLASSES = [{"id": 512, "name": "water", "color": [0, 40, 200], "source_ids": [3]}]
METHOD = "winning_source_class_softmax_probability_uncalibrated"


@pytest.fixture
def report_fixture(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    bundle = Bundle(root, {"bundle_version": "1", "status": "partial", "stages": {},
                           "inputs": [], "config_snapshot": {}})
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(name="constructed report boundary"),
        semantics=SimpleNamespace(model="configured-but-not-executed"),
        report=SimpleNamespace(check_points=None, accuracy_target_m=1., region_size_m=1.),
        dense=SimpleNamespace(max_points_per_chunk=1),
    )
    return bundle, cfg


def descriptor():
    return {
        "id": "semantic", "title": "Semantic classes", "status": "ok", "reason": None,
        "classes": copy.deepcopy(CLASSES), "nodata": 0,
        "raster_path": "s9_geo/semantic.tif", "confidence_path": "s9_geo/semantic_confidence.tif",
        "preview": {"path": "s9_geo/semantic_preview.png", "crs": "EPSG:4326",
                    "bounds": [75., 27., 75.001, 27.001], "width": 2, "height": 2},
        "confidence_method": METHOD, "accuracy_status": "UNVALIDATED", "supported_cells": 7,
        "provenance": {
            "source_descriptor": "s3_masking/semantics.json", "source_status": "ok",
            "model": {"name": "recorded-unit-segmenter", "revision": "constructed-report-input",
                      "license": "MIT", "backend": "transformers", "device": "cpu",
                      "files": {"weights.unit": sha256(b"unit specimen, not trained weights").hexdigest()},
                      "inference_frames": 2},
            "source_labels": {"3": "source water"},
            "projection": {"method": "final_camera_DSM_semantic_projection_with_geometry_zbuffer",
                           "supported_cells": 7, "inferred_excluded": True},
        },
    }


def record_semantics(bundle, desc, *, stage_status="done", standalone=True, partial=False):
    directory = bundle.stage_dir("s9_geo")
    for filename in ("semantic.tif", "semantic_confidence.tif", "semantic_preview.png"):
        (directory / filename).write_bytes(b"constructed file-reference specimen, not imagery")
    bundle.write_json("s9_geo/semantics.json", desc)
    bundle.write_json("s9_geo/geo.json", {
        "supported_cells": 9, "supported_area_m2": 9., "total_cells": 10,
        "bounding_grid_coverage_fraction": .9,
        "georeference": {"epsg": 32643, "vertical": "unknown", "origin": [500000., 3000000., 100.]},
        "semantics": desc,
    })
    outputs = {"metadata": "s9_geo/geo.json"}
    if standalone:
        outputs["semantic_metadata"] = "s9_geo/semantics.json"
    if partial:
        outputs = {"partial": "s9_geo"}
    bundle.manifest["stages"]["s9_geo"] = {"status": stage_status, "outputs": outputs,
                                            "confidence": .99}


def add_geometry(bundle):
    directory = bundle.stage_dir("s7_dense")
    np.savez(directory / "dense.npz", points=np.array([[.5, .5, 1.], [1.5, .5, 2.]], np.float64),
             colors=np.full((2, 3), 100, np.uint8), confidence=np.array([.4, .9], np.float32),
             inferred=np.array([False, True]))
    bundle.manifest["stages"]["s7_dense"] = {"status": "done",
                                            "outputs": {"cloud": "s7_dense/dense.npz"}}


def read_report(bundle, cfg):
    outputs = build_report(bundle, cfg)
    assert outputs == {"json": "report/report.json", "html": "report/report.html"}
    return bundle.read_json(outputs["json"]), bundle.path(outputs["html"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("state", ["missing", "disabled", "unavailable"])
def test_optional_absence_has_no_model_or_quality_claim(report_fixture, state):
    bundle, cfg = report_fixture
    if state != "missing":
        desc = descriptor()
        desc.update(status=state, reason="No semantic model configured" if state == "disabled"
                    else "Local checkpoint could not load", provenance=None, supported_cells=0,
                    raster_path=None, confidence_path=None, preview=None, confidence_method=None)
        record_semantics(bundle, desc)
    report, markup = read_report(bundle, cfg)
    semantics = report["semantics"]
    assert semantics["status"] == ("unavailable" if state == "missing" else state)
    assert semantics["current"] is (state != "missing")
    assert semantics["reason"]
    assert semantics["provenance"] is None and report["models"] == []
    assert semantics["supported_cells"] is None and semantics["raster_path"] is None
    assert semantics["confidence_path"] is None and semantics["preview"] is None
    assert semantics["accuracy_status"] == report["accuracy"]["status"] == "UNVALIDATED"
    assert "Optional semantic layer" in markup and semantics["reason"] in markup


@pytest.mark.parametrize("standalone", [True, False])
def test_current_semantics_preserve_exact_recorded_checkpoint_and_class_mapping(report_fixture, standalone):
    bundle, cfg = report_fixture
    desc = descriptor()
    record_semantics(bundle, desc, standalone=standalone)
    report, markup = read_report(bundle, cfg)
    semantics = report["semantics"]
    assert semantics["status"] == semantics["recorded_status"] == "ok"
    assert semantics["stage"] == "s9_geo" and semantics["stage_status"] == "done"
    assert semantics["current"] is True and semantics["supported_cells"] == 7
    assert semantics["source_descriptor"] == ("s9_geo/semantics.json" if standalone else "s9_geo/geo.json")
    assert semantics["source_field"] == (None if standalone else "semantics")
    assert semantics["classes"] == CLASSES and semantics["provenance"] == desc["provenance"]
    assert semantics["confidence_method"] == METHOD
    assert semantics["confidence_is_geometry_confidence"] is False
    assert semantics["confidence_is_accuracy"] is False
    assert semantics["accuracy_status"] == "UNVALIDATED"
    assert semantics["raster_path"] == desc["raster_path"] and semantics["preview"] == desc["preview"]
    assert report["coverage"]["supported_cells"] == 9  # Geometry support is separate.
    assert report["coverage"]["scene_completeness"] is None
    assert report["issues"] == []
    assert "Semantic probabilities are uncalibrated, distinct from geometry confidence and absolute accuracy" in markup
    assert "recorded-unit-segmenter" in markup and "Source channels" in markup


@pytest.mark.parametrize("partial_kind", ["source_frames", "preview_failure", "descriptor_partial"])
def test_partial_semantics_are_visible_with_actual_supported_products(report_fixture, partial_kind):
    bundle, cfg = report_fixture
    desc = descriptor()
    if partial_kind == "source_frames":
        desc["provenance"]["source_status"] = "partial"
    elif partial_kind == "preview_failure":
        desc.update(preview=None, reason="Geographic preview warp failed")
    else:
        desc.update(status="partial", reason="Only some cells were projected")
    record_semantics(bundle, desc, stage_status="degraded")
    report, _ = read_report(bundle, cfg)
    semantics = report["semantics"]
    assert semantics["status"] == "partial" and semantics["current"] is True
    assert semantics["recorded_status"] == desc["status"]
    assert semantics["stage_status"] == "degraded" and semantics["supported_cells"] == 7
    assert semantics["raster_path"] == desc["raster_path"]
    assert semantics["reason"] == desc["reason"] and semantics["provenance"] == desc["provenance"]


@pytest.mark.parametrize("stage_status,partial,standalone", [
    ("stale", False, True), ("stale", False, False), ("failed", False, True),
    ("blocked", False, True), ("running", False, True),
    ("failed", True, True), ("blocked", True, True), ("partial", False, True),
])
def test_retained_provenance_cannot_supply_current_coverage_confidence_or_georef(
    report_fixture, stage_status, partial, standalone,
):
    bundle, cfg = report_fixture
    desc = descriptor()
    desc["provenance"]["source_status"] = "partial"
    desc["reason"] = "Recorded frame failure"
    record_semantics(bundle, desc, stage_status=stage_status, partial=partial, standalone=standalone)
    # Retained provenance does not require all old product files to remain present.
    bundle.path(desc["raster_path"]).unlink()
    report, markup = read_report(bundle, cfg)
    semantics = report["semantics"]
    assert semantics["stage_status"] == stage_status and semantics["current"] is False
    assert semantics["status"] == "unavailable" and semantics["recorded_status"] == "ok"
    assert semantics["provenance"] == desc["provenance"] and semantics["classes"] == CLASSES
    assert semantics["reason"] == desc["reason"] and stage_status in semantics["availability_reason"]
    assert semantics["supported_cells"] is None and semantics["raster_path"] is None
    assert semantics["confidence_path"] is None and semantics["preview"] is None
    assert report["coverage"]["supported_cells"] is None
    assert report["coverage"]["bounding_grid_coverage_fraction"] is None
    assert report["confidence"]["mean_observed_confidence"] is None
    assert report["georeference"] is None and report["geospatial_products"] == {}
    assert report["stages"][0]["confidence"] is None
    assert report["issues"] == []
    assert "retained semantic metadata is not current" in markup and "recorded-unit-segmenter" in markup


def test_unavailable_semantics_keep_recorded_inference_after_optional_publication_failure(report_fixture):
    bundle, cfg = report_fixture
    desc = descriptor()
    desc.update(status="unavailable", reason="Semantic raster publication failed",
                raster_path=None, confidence_path=None, preview=None)
    record_semantics(bundle, desc, stage_status="degraded")
    report, _ = read_report(bundle, cfg)
    semantics = report["semantics"]
    assert semantics["status"] == "unavailable" and semantics["current"] is True
    assert semantics["provenance"] == desc["provenance"]
    assert semantics["provenance"]["model"]["inference_frames"] == 2
    assert semantics["supported_cells"] is None
    assert report["coverage"]["supported_cells"] == 9


def test_missing_current_raster_is_unavailable_without_erasing_recorded_provenance(report_fixture):
    bundle, cfg = report_fixture
    desc = descriptor()
    record_semantics(bundle, desc)
    bundle.path(desc["raster_path"]).unlink()
    report, _ = read_report(bundle, cfg)
    semantics = report["semantics"]
    assert semantics["current"] is True and semantics["recorded_status"] == "ok"
    assert semantics["status"] == "unavailable" and semantics["supported_cells"] is None
    assert semantics["provenance"] == desc["provenance"]
    assert semantics["raster_path"] is None and "raster_path" in semantics["availability_reason"]
    assert len(report["issues"]) == 1


@pytest.mark.parametrize("failure", [
    "nonobject", "json", "status", "classes", "class_mapping", "provenance", "method",
    "no_inference", "missing_reason",
    "supported_cells", "preview_bounds", "metadata_escape", "raster_escape",
    "confidence_escape", "preview_escape", "source_escape", "symlink_escape", "missing_metadata",
])
def test_invalid_optional_semantics_preserve_geometry_and_finish_both_reports(report_fixture, failure):
    bundle, cfg = report_fixture
    desc = descriptor()
    if failure == "nonobject":
        desc = []
    elif failure == "status":
        desc["status"] = []
    elif failure == "classes":
        desc["classes"] = {}
    elif failure == "class_mapping":
        desc["classes"].append(copy.deepcopy(CLASSES[0]))
    elif failure == "provenance":
        desc["provenance"] = "configured model, no inference"
    elif failure == "method":
        desc["confidence_method"] = "geometry_confidence"
    elif failure == "no_inference":
        desc["provenance"]["model"]["inference_frames"] = 0
    elif failure == "missing_reason":
        desc.update(status="disabled", reason=None)
    elif failure == "supported_cells":
        desc["supported_cells"] = True
    elif failure == "preview_bounds":
        desc["preview"]["bounds"] = [75., None, 75.001, 27.001]
    elif failure in ("raster_escape", "confidence_escape"):
        desc["raster_path" if failure == "raster_escape" else "confidence_path"] = "../outside.tif"
    elif failure == "preview_escape":
        desc["preview"]["path"] = "../outside.png"
    elif failure == "source_escape":
        desc["provenance"]["source_descriptor"] = "../outside.json"
    record_semantics(bundle, desc)
    add_geometry(bundle)
    path = bundle.path("s9_geo/semantics.json")
    if failure == "json":
        path.write_text("{invalid json", encoding="utf-8")
        # The embedded fallback must also be invalid for this isolation case.
        bundle.write_json("s9_geo/geo.json", {"supported_cells": 9, "semantics": []})
    elif failure == "metadata_escape":
        bundle.manifest["stages"]["s9_geo"]["outputs"]["semantic_metadata"] = "../outside.json"
        bundle.write_json("s9_geo/geo.json", {"supported_cells": 9, "semantics": []})
    elif failure == "symlink_escape":
        outside = bundle.root.parent / "outside.tif"
        outside.write_bytes(b"secret outside the bundle")
        raster = bundle.path("s9_geo/semantic.tif")
        raster.unlink()
        raster.symlink_to(outside)
    elif failure == "missing_metadata":
        path.unlink()
        bundle.write_json("s9_geo/geo.json", {"supported_cells": 9})
    report, markup = read_report(bundle, cfg)
    assert report["semantics"]["status"] == "unavailable"
    assert report["semantics"]["current"] is False
    assert report["semantics"]["supported_cells"] is None
    assert report["coverage"]["supported_cells"] == 9
    assert report["confidence"]["mean_observed_confidence"] == pytest.approx(.4)
    assert report["confidence"]["observed_points"] == 1
    assert report["accuracy"]["absolute_spatial_rmse_m"] is None
    assert report["issues"] and "Cannot read semantic" in report["issues"][0]
    assert "secret outside the bundle" not in markup


def test_bad_standalone_descriptor_can_use_valid_embedded_semantics(report_fixture):
    bundle, cfg = report_fixture
    desc = descriptor()
    record_semantics(bundle, desc)
    bundle.write_json("s9_geo/semantics.json", [])
    report, _ = read_report(bundle, cfg)
    assert report["semantics"]["status"] == "ok"
    assert report["semantics"]["source_descriptor"] == "s9_geo/geo.json"
    assert report["semantics"]["source_field"] == "semantics"
    assert report["semantics"]["provenance"] == desc["provenance"]
    assert len(report["issues"]) == 1


def test_optional_validation_exception_is_isolated_from_geometry(report_fixture, monkeypatch):
    bundle, cfg = report_fixture
    record_semantics(bundle, descriptor())
    add_geometry(bundle)

    def fail_validation(_descriptor):
        raise RuntimeError("constructed optional validation failure")

    monkeypatch.setattr("drishti.geo.semantics.validate_descriptor", fail_validation)
    report, _ = read_report(bundle, cfg)
    assert report["semantics"]["status"] == "unavailable"
    assert report["confidence"]["mean_observed_confidence"] == pytest.approx(.4)
    assert report["coverage"]["supported_cells"] == 9
    assert "constructed optional validation failure" in report["semantics"]["reason"]


def test_semantic_html_section_escapes_classes_reason_paths_and_model_provenance(report_fixture):
    bundle, cfg = report_fixture
    attack = '<img src=x onerror="alert(1)">'
    desc = descriptor()
    desc["classes"][0]["name"] = attack
    desc["reason"] = "<script>alert('reason')</script>"
    desc["provenance"]["model"]["name"] = attack
    desc["provenance"]["source_labels"]["3"] = "</pre><script>alert('labels')</script>"
    record_semantics(bundle, desc)
    source_path = "s9_geo/<svg onload=alert(1)>.json"
    bundle.write_json(source_path, desc)
    bundle.manifest["stages"]["s9_geo"]["outputs"]["semantic_metadata"] = source_path
    report, markup = read_report(bundle, cfg)
    section = markup.split("<h2>Optional semantic layer</h2>")[1].split("<h2>Measured stage timings</h2>")[0]
    assert report["semantics"]["status"] == "partial"
    assert report["semantics"]["classes"][0]["name"] == attack
    for raw in (attack, desc["reason"], source_path):
        assert html.escape(raw, quote=True) in section
    assert "&lt;/pre&gt;&lt;script&gt;" in section
    assert "<script>" not in markup and "<img src=x" not in markup and "<svg onload" not in markup
