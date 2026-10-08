"""Isolated API/security/math tests. Constructed fixtures are NOT reconstruction acceptance."""

import json
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server.app import create_app
from server.bundles import confined
from server.settings import Settings


@pytest.fixture
def mock_bundle(tmp_path):
    root = tmp_path / "runs"
    bundle = root / "unit-only"
    bundle.mkdir(parents=True)
    manifest = {"bundle_version": "1", "status": "partial", "crs": {"epsg": 32643},
                "stages": {"s6_global": {"outputs": {"poses": "poses.json"}}}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "poses.json").write_text(json.dumps({
        "frame": "projected_local", "origin": [500000, 1000000, 100],
        "crs": {"epsg": 32643, "vertical": "unknown"}, "cameras": [],
    }))
    settings = Settings(runs_root=root, dataset_root=tmp_path, viewer_dist=tmp_path / "viewer")
    with TestClient(create_app(settings)) as client:
        yield client, bundle, manifest, settings


def save_manifest(bundle, manifest):
    (bundle / "manifest.json").write_text(json.dumps(manifest))


def request(client, kind, points, **kwargs):
    return client.post("/api/runs/unit-only/measure", json={"kind": kind, "points": points, **kwargs})


def test_health_runs_scene_and_absent_report(mock_bundle):
    client, _, _, _ = mock_bundle
    assert client.get("/api/health").json()["mode"] == "offline_ground"
    assert client.get("/api/runs").json()["runs"][0]["id"] == "unit-only"
    scene = client.get("/api/runs/unit-only/scene").json()
    assert scene["origin"] == [500000, 1000000, 100]
    assert scene["crs"]["epsg"] == 32643
    assert client.get("/api/runs/unit-only/report").status_code == 404


@pytest.mark.parametrize("relative", ["../secret", "/etc/passwd", "a/../../secret", "a\\..\\secret"])
def test_path_confinement_rejects_traversal(mock_bundle, relative):
    _, bundle, _, _ = mock_bundle
    with pytest.raises(HTTPException) as exc:
        confined(bundle, relative)
    assert exc.value.status_code == 403


def test_artifact_and_directory_symlink_escape(mock_bundle, tmp_path):
    client, bundle, _, _ = mock_bundle
    secret = tmp_path / "secret.txt"
    secret.write_text("private-unit-test-value")
    (bundle / "escape").symlink_to(secret)
    (bundle / "escape-dir").symlink_to(tmp_path, target_is_directory=True)
    assert client.get("/api/runs/unit-only/files/escape").status_code == 403
    assert client.get("/api/runs/unit-only/files/escape-dir/secret.txt").status_code == 403
    assert client.get("/api/runs/unit-only/files/%2e%2e%2fsecret.txt").status_code == 403
    assert client.get("/api/runs/unit-only/files/%2Fetc%2Fpasswd").status_code == 403


def test_internal_symlink_is_served_and_missing_file_is_404(mock_bundle):
    client, bundle, _, _ = mock_bundle
    (bundle / "valid.txt").write_text("actual unit test bytes")
    (bundle / "alias.txt").symlink_to(bundle / "valid.txt")
    response = client.get("/api/runs/unit-only/files/alias.txt")
    assert response.text == "actual unit test bytes"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/runs/unit-only/files/missing.npz").status_code == 404


def test_run_manifest_report_and_pose_symlink_confinement(mock_bundle, tmp_path):
    client, bundle, manifest, settings = mock_bundle
    outside = tmp_path / "outside.json"
    outside.write_text('{"should_not_be_read":true}')
    (settings.runs_root / "outside-run").symlink_to(tmp_path, target_is_directory=True)
    assert client.get("/api/runs/outside-run/manifest").status_code == 403
    (bundle / "poses.json").unlink()
    (bundle / "poses.json").symlink_to(outside)
    assert client.get("/api/runs/unit-only/scene").status_code == 403
    manifest["stages"]["report"] = {"outputs": {"json": "outside-report.json"}}
    (bundle / "outside-report.json").symlink_to(outside)
    save_manifest(bundle, manifest)
    assert client.get("/api/runs/unit-only/report").status_code == 403
    (bundle / "manifest.json").unlink()
    (bundle / "manifest.json").symlink_to(outside)
    assert client.get("/api/runs/unit-only/manifest").status_code == 403


def test_corrupt_manifest_and_nonstandard_json_are_not_500(mock_bundle):
    client, bundle, _, _ = mock_bundle
    (bundle / "manifest.json").write_text("{bad json}")
    assert client.get("/api/runs/unit-only/manifest").status_code == 409
    assert client.get("/api/runs").json()["runs"][0]["status"] == "unreadable"
    (bundle / "manifest.json").write_text('{"status":NaN}')
    assert client.get("/api/runs/unit-only/manifest").status_code == 409
    (bundle / "manifest.json").write_text('{"status":"partial","wall_seconds":1e1000}')
    assert client.get("/api/runs/unit-only/manifest").status_code == 409
    (bundle / "manifest.json").write_text('{"stages":"invalid-stage-schema"}')
    assert client.get("/api/runs/unit-only/manifest").status_code == 409


@pytest.mark.parametrize(("kind", "points", "value", "method"), [
    ("distance", [[0, 0, 0], [3, 4, 12], [3, 4, 15]], 16, "local_XYZ_polyline_length"),
    ("height", [[0, 0, 2], [9, 3, 7], [5, 5, 4]], 5, "local_Z_max_minus_min"),
    ("area", [[0, 0, 1], [3, 0, 9], [3, 2, -4], [0, 2, 7]], 6, "horizontal_XY_polygon_area"),
    ("area", [[0, 0, 0], [3, 0, 0], [3, 3, 0], [2, 3, 0], [2, 1, 0], [0, 1, 0]], 5,
     "horizontal_XY_polygon_area"),
])
def test_metric_definitions_and_unvalidated_accuracy(mock_bundle, kind, points, value, method):
    client, _, _, _ = mock_bundle
    response = request(client, kind, points)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["value"] == pytest.approx(value)
    assert result["method"] == method
    assert result["uncertainty"]["status"] == "UNVALIDATED"
    assert result["uncertainty"]["absolute_uncertainty_m"] is None
    assert result["support"]["surface_support"] == "not_verified"


@pytest.mark.parametrize("points", [
    [[0, 0, 0], [2, 2, 0], [0, 2, 0], [2, 0, 0]],  # crossing
    [[0, 0, 0], [1, 0, 0], [2, 0, 0]],  # zero area
    [[0, 0, 0], [2, 0, 0], [2, 1, 0], [2, 0, 8], [0, 1, 0]],  # repeated XY
    [[0, 0, 0], [3, 0, 0], [1, 0, 0], [3, 3, 0], [0, 3, 0]],  # overlapping edge
    [[0, 0, 0], [3, 0, 0], [3, 3, 0], [1, 0, 0], [0, 3, 0]],  # touching edge
])
def test_invalid_polygons(mock_bundle, points):
    client, _, _, _ = mock_bundle
    assert request(client, "area", points).status_code == 422


def test_closed_polygon_is_valid(mock_bundle):
    client, _, _, _ = mock_bundle
    points = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [0, 0, 0]]
    assert request(client, "area", points).json()["value"] == 4


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", '"NaN"', "true"])
def test_nonfinite_and_boolean_points_return_422(mock_bundle, raw):
    client, _, _, _ = mock_bundle
    response = client.post("/api/runs/unit-only/measure", content=
                           '{"kind":"distance","points":[[0,0,0],[' + raw + ',1,1]]}',
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_volume_requires_real_surface_and_explicit_reference(mock_bundle):
    client, _, _, _ = mock_bundle
    points = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]
    assert request(client, "volume", points).status_code == 422
    response = request(client, "volume", points, reference_height=0)
    assert response.status_code == 409
    assert "No DSM" in response.json()["detail"]


def make_unit_rasters(bundle: Path, manifest: dict, *, provenance=True):
    """Explicit math inputs: these 2×2 files are not pipeline products or accuracy evidence."""
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(500000, 1000002, 1, 1)
    for name, values in (("dsm", [[105, 109], [-9999, 108]]),
                         ("mask", [[1, 1], [0, 2]]),
                         ("confidence", [[0.5, 0.9], [-9999, 0.7]])):
        with rasterio.open(bundle / f"{name}.tif", "w", driver="GTiff", width=2, height=2,
                           count=1, dtype="float64", crs="EPSG:32643", transform=transform,
                           nodata=-9999 if name != "mask" else 0) as dst:
            dst.write(np.array(values, dtype=np.float64), 1)
            if name == "dsm" and provenance:
                dst.update_tags(inferred_excluded="true", elevation_frame="projected_absolute")
    manifest["stages"]["s9_geo"] = {"outputs": {"dsm": "dsm.tif", "mask": "mask.tif",
                                                  "confidence": "confidence.tif"}}
    save_manifest(bundle, manifest)


def test_volume_excludes_inferred_nodata_reports_coverage_and_confidence(mock_bundle):
    client, bundle, manifest, _ = mock_bundle
    make_unit_rasters(bundle, manifest)
    points = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]
    response = request(client, "volume", points, reference_height=2)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["value"] == 10  # ((105-100)-2) + ((109-100)-2)
    support = result["support"]
    assert support["coverage_fraction"] == 0.5
    assert support["unsupported_area_m2"] == 2
    assert support["nodata_area_m2"] == 1
    assert support["inferred_excluded_area_m2"] == 1
    assert support["supported_cells"] == 2
    assert support["mean_source_confidence"] == pytest.approx(0.7)
    assert support["confidence_is_absolute_accuracy"] is False


def test_volume_partial_boundary_cells_and_above_surface_reference(mock_bundle):
    client, bundle, manifest, _ = mock_bundle
    make_unit_rasters(bundle, manifest)
    points = [[0.5, 0, 0], [2, 0, 0], [2, 2, 0], [0.5, 2, 0]]
    response = request(client, "volume", points, reference_height=2)
    assert response.status_code == 200, response.text
    assert response.json()["value"] == pytest.approx(8.5)
    assert response.json()["support"]["polygon_area_m2"] == 3
    assert request(client, "volume", points, reference_height=20).json()["value"] == 0


def test_volume_triangle_overlap_and_partly_outside_polygon(mock_bundle):
    client, bundle, manifest, _ = mock_bundle
    make_unit_rasters(bundle, manifest)
    triangle = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    response = request(client, "volume", triangle, reference_height=2)
    assert response.status_code == 200, response.text
    assert response.json()["value"] == pytest.approx(1.5)
    assert response.json()["support"]["coverage_fraction"] == pytest.approx(0.25)
    outside = [[-1, 0, 0], [2, 0, 0], [2, 2, 0], [-1, 2, 0]]
    result = request(client, "volume", outside, reference_height=2).json()
    assert result["value"] == 10
    assert result["support"]["outside_raster_area_m2"] == 2
    assert result["support"]["coverage_fraction"] == pytest.approx(1 / 3)


def test_volume_corrupt_and_misaligned_support_is_actionable(mock_bundle):
    import rasterio
    from rasterio.transform import from_origin

    client, bundle, manifest, _ = mock_bundle
    make_unit_rasters(bundle, manifest)
    points = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]
    with rasterio.open(bundle / "mask.tif", "r+") as mask:
        mask.transform = from_origin(500010, 1000002, 1, 1)
    response = request(client, "volume", points, reference_height=0)
    assert response.status_code == 409
    assert "grid-aligned" in response.json()["detail"]
    (bundle / "mask.tif").write_bytes(b"not-a-raster-unit-fixture")
    assert request(client, "volume", points, reference_height=0).status_code == 409


def test_volume_missing_provenance_no_support_and_cell_budget(mock_bundle):
    client, bundle, manifest, settings = mock_bundle
    make_unit_rasters(bundle, manifest, provenance=False)
    manifest["stages"]["s9_geo"]["outputs"].pop("mask")
    save_manifest(bundle, manifest)
    points = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]
    assert request(client, "volume", points, reference_height=0).status_code == 409
    make_unit_rasters(bundle, manifest)
    outside = [[10, 10, 0], [12, 10, 0], [12, 12, 0], [10, 12, 0]]
    assert request(client, "volume", outside, reference_height=0).status_code == 409
    with TestClient(create_app(settings.model_copy(update={"max_volume_cells": 1}))) as bounded:
        assert request(bounded, "volume", points, reference_height=0).status_code == 413


def test_actual_ply_bytes_decimate_preserve_flags_and_colors(mock_bundle):
    client, bundle, manifest, _ = mock_bundle
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 5\n"
              "property double x\nproperty double y\nproperty double z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property float confidence\nproperty uchar inferred\nend_header\n")
    rows = b"".join(struct.pack("<dddBBBfB", i, 2 * i, 3 * i, 12, 34, 56, i / 5, i % 2)
                    for i in range(5))
    (bundle / "unit.ply").write_bytes(header.encode("ascii") + rows)
    manifest["stages"]["s7_dense"] = {"outputs": {"ply": "unit.ply"}}
    save_manifest(bundle, manifest)
    result = client.get("/api/runs/unit-only/preview?limit=3").json()
    assert result["total_points"] == 5
    assert result["points"] == [[0, 0, 0], [2, 4, 6], [4, 8, 12]]
    assert result["colors"] == [[12, 34, 56]] * 3
    assert result["inferred"] == [False] * 3
    assert result["confidence"] == pytest.approx([0, 0.4, 0.8])


def test_npz_preview_actual_arrays_and_size_guard(mock_bundle):
    client, bundle, manifest, settings = mock_bundle
    points = np.arange(30, dtype=np.float64).reshape(10, 3)
    np.savez(bundle / "unit.npz", points=points, confidence=np.linspace(0, 1, 10),
             inferred=np.zeros(10, dtype=bool), colors=np.full((10, 3), 44, dtype=np.uint8))
    manifest["stages"]["s7_dense"] = {"outputs": {"cloud": "unit.npz"}}
    save_manifest(bundle, manifest)
    result = client.get("/api/runs/unit-only/preview?limit=2").json()
    assert result["points"] == points[[0, 5]].tolist()
    assert result["preview_points"] == 2
    with TestClient(create_app(settings.model_copy(update={"max_npz_bytes": 1}))) as bounded:
        assert bounded.get("/api/runs/unit-only/preview").status_code == 413


def test_background_run_returns_202_while_actual_callable_is_blocked(tmp_path):
    started, release = threading.Event(), threading.Event()
    descriptor = tmp_path / "unit-descriptor.yaml"
    descriptor.write_text("name: unit-service-only")

    class UnitService:
        def prepare(self, dataset, profile):
            assert dataset == descriptor and profile == "balanced"

            def execute():
                started.set()
                assert release.wait(5)
                return "partial"

            return SimpleNamespace(run_id="unit-job", execute=execute)

    with TestClient(create_app(Settings(dataset_root=tmp_path, runs_root=tmp_path / "runs"), UnitService())) as client:
        try:
            response = client.post("/api/runs", json={"dataset": descriptor.name, "profile": "balanced"})
            assert response.status_code == 202
            assert response.json()["id"] == "unit-job"
            assert started.wait(3)
            duplicate = client.post("/api/runs", json={"dataset": descriptor.name})
            assert duplicate.status_code == 409
            assert client.get("/api/runs").json()["runs"][0]["job"]["status"] == "running"
            assert client.post("/api/runs", json={"dataset": "../outside.yaml"}).status_code == 403
            assert client.post("/api/runs", json={"dataset": descriptor.name, "profile": "../bad"}).status_code == 422
        finally:
            release.set()


def test_real_descriptor_schema_rejection_and_doctor_do_not_claim_execution(tmp_path):
    descriptor = tmp_path / "invalid.yaml"
    descriptor.write_text("name: missing-video-and-telemetry\n")
    with TestClient(create_app(Settings(dataset_root=tmp_path, runs_root=tmp_path / "runs"))) as client:
        assert client.post("/api/runs", json={"dataset": descriptor.name}).status_code == 422
        result = client.get("/api/doctor").json()
        assert result["capability_detection_is_execution"] is False
        assert result["environment"]["python_executable"]
        assert not (tmp_path / "runs").exists()
