"""Constructed descriptor/file contract inputs, NOT mission or semantic accuracy evidence."""

import json
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.settings import Settings


def unit_tiff(*, probability=False):
    """Minimal real 2x2 TIFF, using only stdlib to keep API tests lightweight."""
    count = 11
    offset = 8 + 2 + count * 12 + 4
    pixels = struct.pack("<4f", 0.8, 0.6, -9999, 0.9) if probability else struct.pack("<4H", 1, 2, 0, 1)
    tags = [(256, 4, 2), (257, 4, 2), (258, 3, 32 if probability else 16),
            (259, 3, 1), (262, 3, 1), (273, 4, offset), (277, 3, 1),
            (278, 4, 2), (279, 4, len(pixels)), (284, 3, 1), (339, 3, 3 if probability else 1)]
    entries = b"".join(struct.pack("<HHII", tag, kind, 1, value) for tag, kind, value in tags)
    return b"II*\x00" + struct.pack("<IH", 8, count) + entries + b"\x00" * 4 + pixels


def unit_png():
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + b"\x14\x64\xc8\xff" * 2
                                          + b"\x00" + b"\x00\x00\x00\x00" * 2))
            + chunk(b"IEND", b""))


@pytest.fixture
def semantic_api(tmp_path):
    # The fixture isolates optional scene metadata; no perception/reconstruction is run.
    root = tmp_path / "runs"
    bundle = root / "semantic-unit"
    bundle.mkdir(parents=True)
    (bundle / "s9_geo").mkdir()
    manifest = {"bundle_version": "1", "status": "partial", "crs": {"epsg": 32643},
                "stages": {
                    "s6_global": {"outputs": {"poses": "poses.json"}},
                    "s7_dense": {"outputs": {"cloud": "dense.npz"}, "outputs_hash": "cloud-revision"},
                    "s8_mesh": {"outputs": {"mesh": "mesh.npz"}, "status": "stale"},
                    "s9_geo": {"outputs": {}, "outputs_hash": "geo-revision", "status": "failed"},
                }}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "poses.json").write_text(json.dumps({"origin": [500000, 1000000, 100]}))
    (bundle / "s9_geo" / "classes.tif").write_bytes(unit_tiff())
    (bundle / "s9_geo" / "probability.tif").write_bytes(unit_tiff(probability=True))
    (bundle / "s9_geo" / "preview.png").write_bytes(unit_png())
    descriptor = {
        "id": "semantic", "title": "Semantic classes", "status": "ok", "reason": None,
        "classes": [{"id": 1, "name": "Building", "color": [200, 80, 30], "source_ids": [2]},
                    {"id": 2, "name": "Water", "color": [20, 100, 200], "source_ids": [4, 5]}],
        "nodata": 0, "raster_path": "s9_geo/classes.tif",
        "confidence_path": "s9_geo/probability.tif", "supported_cells": 3,
        "preview": {"path": "s9_geo/preview.png", "crs": "EPSG:4326",
                    "bounds": [74.9, 8.9, 75.1, 9.1], "width": 2, "height": 2},
        "confidence_method": "mean_winning_source_softmax_probability_uncalibrated",
        "provenance": {"note": "constructed API unit fixture"}, "accuracy_status": "UNVALIDATED",
    }
    settings = Settings(runs_root=root, dataset_root=tmp_path, viewer_dist=tmp_path / "viewer")
    with TestClient(create_app(settings)) as client:
        yield client, bundle, manifest, descriptor


def publish(bundle, manifest, descriptor):
    manifest["stages"]["s9_geo"]["outputs"]["semantic_metadata"] = "s9_geo/semantics.json"
    (bundle / "s9_geo" / "semantics.json").write_text(json.dumps(descriptor))
    (bundle / "manifest.json").write_text(json.dumps(manifest))


def get_scene(client):
    response = client.get("/api/runs/semantic-unit/scene")
    assert response.status_code == 200, response.text
    return response.json()


def assert_geometry_retained(scene):
    assert scene["artifacts"]["cloud"] == "dense.npz"
    assert scene["artifacts"]["mesh"] == "mesh.npz"
    assert scene["stage_status"]["s8_mesh"] == "stale"
    assert scene["stage_status"]["s9_geo"] == "failed"
    assert scene["revision"]["s9_geo"] == "geo-revision"


def test_old_bundle_without_semantics_retains_scene(semantic_api):
    client, _, _, _ = semantic_api
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "No S9 semantic metadata" in scene["semantic_reason"]


def test_unregistered_metadata_is_not_treated_as_a_stage_product(semantic_api):
    client, bundle, _, descriptor = semantic_api
    (bundle / "s9_geo" / "semantics.json").write_text(json.dumps(descriptor))
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "No S9 semantic metadata" in scene["semantic_reason"]


def test_empty_bundle_without_semantics_remains_inspectable(semantic_api):
    client, bundle, _, _ = semantic_api
    (bundle / "manifest.json").write_text('{}')
    scene = get_scene(client)
    assert scene["origin"] is None
    assert scene["artifacts"]["mesh"] is None
    assert scene["semantic_layers"] == []
    assert scene["semantic_reason"]


@pytest.mark.parametrize("record_style", ["string", "path_object", "stage_list"])
def test_descriptor_discovery_and_actual_downloads(semantic_api, record_style):
    client, bundle, manifest, descriptor = semantic_api
    publish(bundle, manifest, descriptor)
    if record_style == "path_object":
        manifest["stages"]["s9_geo"]["outputs"]["semantic_metadata"] = {"path": "s9_geo/semantics.json"}
    elif record_style == "stage_list":
        manifest["stages"] = [{"name": name, **value} for name, value in manifest["stages"].items()]
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == [descriptor]
    assert scene["semantic_reason"] is None
    for path in [descriptor["raster_path"], descriptor["confidence_path"], descriptor["preview"]["path"]]:
        response = client.get(f"/api/runs/semantic-unit/files/{path}")
        assert response.status_code == 200
        assert response.content == (bundle / path).read_bytes()
        assert response.headers["x-content-type-options"] == "nosniff"


def test_geo_output_revision_refresh_is_preserved(semantic_api):
    client, bundle, manifest, descriptor = semantic_api
    publish(bundle, manifest, descriptor)
    assert get_scene(client)["revision"]["s9_geo"] == "geo-revision"
    manifest["stages"]["s9_geo"]["outputs_hash"] = "new-semantic-revision"
    descriptor["classes"][0]["color"] = [100, 200, 30]
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert scene["revision"]["s9_geo"] == "new-semantic-revision"
    assert scene["semantic_layers"][0]["classes"][0]["color"] == [100, 200, 30]


@pytest.mark.parametrize("bounds", [
    [75, 9, 75, 10], [76, 9, 75, 10], [75, 10, 76, 9], [75, 9, 76, 9],
    [-181, 9, 76, 10], [75, 9, 181, 10], [75, -91, 76, 10], [75, 9, 76, 91],
    [170, 9, -170, 10], [75, 9, 76], [True, 9, 76, 10], ["75", 9, 76, 10],
    [float("nan"), 9, 76, 10], [75, 9, float("inf"), 10], [75, 9, 76, -float("inf")],
])
def test_bad_geographic_bounds_only_disable_optional_overlay(semantic_api, bounds):
    client, bundle, manifest, descriptor = semantic_api
    descriptor["preview"]["bounds"] = bounds
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "Semantic layers unavailable" in scene["semantic_reason"]


@pytest.mark.parametrize(("key", "value"), [
    ("width", 0), ("height", -1), ("width", 4097), ("height", True), ("width", 2.5),
    ("height", "2"), ("crs", "EPSG:32643"), ("crs", None),
])
def test_bad_preview_dimensions_or_crs(semantic_api, key, value):
    client, bundle, manifest, descriptor = semantic_api
    descriptor["preview"][key] = value
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "preview" in scene["semantic_reason"]


@pytest.mark.parametrize(("key", "value"), [
    ("id", 0), ("id", 65535), ("id", True), ("id", 1.0), ("id", 2),
    ("name", ""), ("name", "   "), ("name", "water"), ("name", None),
    ("color", [255, 0]), ("color", [-1, 0, 0]), ("color", [0, 256, 0]),
    ("color", [0, 0.5, 0]), ("color", [True, 0, 0]), ("color", "red"),
    ("source_ids", [4]), ("source_ids", [2, 2]), ("source_ids", [True]), ("source_ids", []),
])
def test_invalid_class_map_never_reaches_legend(semantic_api, key, value):
    client, bundle, manifest, descriptor = semantic_api
    descriptor["classes"][0][key] = value
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "semantic" in scene["semantic_reason"]


@pytest.mark.parametrize(("key", "value"), [
    ("status", "partial"), ("status", "success"), ("status", None),
    ("id", "other"), ("title", None), ("classes", []), ("classes", [None]),
    ("nodata", -9999), ("nodata", False), ("supported_cells", 0), ("supported_cells", -1),
    ("supported_cells", True), ("supported_cells", 1.5), ("accuracy_status", "VALIDATED"),
    ("confidence_method", "geometry_confidence"), ("confidence_method", ""),
    ("confidence_method", "calibrated_softmax_probability"),
    ("confidence_method", "geometry_probability_uncalibrated"),
    ("confidence_method", None), ("confidence_method", {}), ("provenance", "unknown"),
    ("preview", "preview.png"), ("raster_path", ""), ("raster_path", None), ("reason", {}),
])
def test_bad_or_partial_descriptors_keep_geometry(semantic_api, key, value):
    client, bundle, manifest, descriptor = semantic_api
    descriptor[key] = value
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert scene["semantic_reason"]


@pytest.mark.parametrize("missing", [("preview",), ("confidence_path",), ("preview", "confidence_path")])
def test_partial_semantic_products_retain_real_downloads(semantic_api, missing):
    client, bundle, manifest, descriptor = semantic_api
    for key in missing:
        descriptor[key] = None
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == [descriptor]
    assert "unavailable" in scene["semantic_reason"]
    assert "downloadable" in scene["semantic_reason"]
    response = client.get(f"/api/runs/semantic-unit/files/{descriptor['raster_path']}")
    assert response.status_code == 200
    assert response.content == unit_tiff()


def test_partial_products_explain_availability_alongside_source_reason(semantic_api):
    client, bundle, manifest, descriptor = semantic_api
    descriptor.update(preview=None, confidence_path=None, reason="Some source frames were excluded.")
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert scene["semantic_layers"] == [descriptor]
    assert descriptor["reason"] in scene["semantic_reason"]
    assert "geographic preview" in scene["semantic_reason"]
    assert "semantic probability raster" in scene["semantic_reason"]
    assert "downloadable" in scene["semantic_reason"]


def test_class_raster_download_is_an_attachment_with_original_bytes(semantic_api):
    client, bundle, manifest, descriptor = semantic_api
    publish(bundle, manifest, descriptor)
    response = client.get(f"/api/runs/semantic-unit/files/{descriptor['raster_path']}?download=true&revision=geo-revision")
    assert response.status_code == 200
    assert response.content == unit_tiff()
    assert response.headers["content-type"] == "image/tiff"
    assert response.headers["content-disposition"] == 'attachment; filename="classes.tif"'
    assert response.headers["cache-control"] == "no-cache"
    image = client.get(f"/api/runs/semantic-unit/files/{descriptor['preview']['path']}?revision=geo-revision")
    assert image.headers["content-type"] == "image/png"
    assert "content-disposition" not in image.headers


def test_download_option_cannot_bypass_path_confinement(semantic_api, tmp_path):
    client, bundle, _, _ = semantic_api
    outside = tmp_path / "outside-download.tif"
    outside.write_bytes(unit_tiff())
    (bundle / "s9_geo" / "unsafe-download.tif").symlink_to(outside)
    response = client.get("/api/runs/semantic-unit/files/s9_geo/unsafe-download.tif?download=true")
    assert response.status_code == 403


@pytest.mark.parametrize("status", ["disabled", "unavailable"])
def test_explicit_unavailability_has_no_placeholder_paths(semantic_api, status):
    client, bundle, manifest, descriptor = semantic_api
    descriptor.update(status=status, reason="Local semantic checkpoint not available", classes=[],
                      raster_path=None, confidence_path=None, preview=None, supported_cells=0,
                      confidence_method=None, provenance=None)
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == [descriptor]
    assert scene["semantic_reason"] == descriptor["reason"]


def test_non_ok_retained_files_cannot_enable_imagery(semantic_api):
    client, bundle, manifest, descriptor = semantic_api
    descriptor.update(status="unavailable", supported_cells=0, reason="Preview projection failed")
    publish(bundle, manifest, descriptor)
    result = get_scene(client)["semantic_layers"][0]
    assert result["status"] == "unavailable"
    assert result["raster_path"] is None
    assert result["confidence_path"] is None
    assert result["preview"] is None


@pytest.mark.parametrize("file_key", ["metadata", "raster", "confidence", "preview"])
def test_missing_semantic_files_do_not_block_scene(semantic_api, file_key):
    client, bundle, manifest, descriptor = semantic_api
    publish(bundle, manifest, descriptor)
    paths = {"metadata": "s9_geo/semantics.json", "raster": descriptor["raster_path"],
             "confidence": descriptor["confidence_path"], "preview": descriptor["preview"]["path"]}
    (bundle / paths[file_key]).unlink()
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "HTTP 404" in scene["semantic_reason"]


@pytest.mark.parametrize("file_key", ["metadata", "raster", "confidence", "preview"])
@pytest.mark.parametrize("unsafe", ["../outside.tif", "/etc/passwd", "s9_geo\\escape.tif"])
def test_unsafe_descriptor_paths_are_explicitly_rejected(semantic_api, file_key, unsafe):
    client, bundle, manifest, descriptor = semantic_api
    if file_key == "raster":
        descriptor["raster_path"] = unsafe
    elif file_key == "confidence":
        descriptor["confidence_path"] = unsafe
    elif file_key == "preview":
        descriptor["preview"]["path"] = unsafe
    publish(bundle, manifest, descriptor)
    if file_key == "metadata":
        manifest["stages"]["s9_geo"]["outputs"]["semantic_metadata"] = unsafe
        (bundle / "manifest.json").write_text(json.dumps(manifest))
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "HTTP 403" in scene["semantic_reason"]


@pytest.mark.parametrize("file_key", ["metadata", "raster", "confidence", "preview"])
def test_semantic_symlink_escape_is_rejected_with_geometry_retained(semantic_api, tmp_path, file_key):
    client, bundle, manifest, descriptor = semantic_api
    publish(bundle, manifest, descriptor)
    paths = {"metadata": "s9_geo/semantics.json", "raster": descriptor["raster_path"],
             "confidence": descriptor["confidence_path"], "preview": descriptor["preview"]["path"]}
    target = bundle / paths[file_key]
    outside = tmp_path / target.name
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "symlink escapes" in scene["semantic_reason"]
    assert "HTTP 403" in scene["semantic_reason"]
    assert client.get(f"/api/runs/semantic-unit/files/{paths[file_key]}").status_code == 403


def test_internal_preview_symlink_remains_confined_and_usable(semantic_api):
    client, bundle, manifest, descriptor = semantic_api
    (bundle / "s9_geo" / "preview-alias.png").symlink_to(bundle / "s9_geo" / "preview.png")
    descriptor["preview"]["path"] = "s9_geo/preview-alias.png"
    publish(bundle, manifest, descriptor)
    assert get_scene(client)["semantic_layers"] == [descriptor]


def test_unavailable_status_does_not_silently_accept_unsafe_references(semantic_api):
    client, bundle, manifest, descriptor = semantic_api
    descriptor.update(status="unavailable", reason="Projection failed")
    descriptor["preview"]["path"] = "../outside.png"
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "HTTP 403" in scene["semantic_reason"]


def test_preview_directory_symlink_is_confined(semantic_api, tmp_path):
    client, bundle, manifest, descriptor = semantic_api
    outside = tmp_path / "outside-preview"
    outside.mkdir()
    (outside / "preview.png").write_bytes(unit_png())
    (bundle / "preview-dir").symlink_to(outside, target_is_directory=True)
    descriptor["preview"]["path"] = "preview-dir/preview.png"
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "HTTP 403" in scene["semantic_reason"]


def test_directory_reference_cannot_be_a_preview_file(semantic_api):
    client, bundle, manifest, descriptor = semantic_api
    (bundle / "directory.png").mkdir()
    descriptor["preview"]["path"] = "directory.png"
    publish(bundle, manifest, descriptor)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "not a file" in scene["semantic_reason"]


@pytest.mark.parametrize("contents", ['{broken}', '[]', '{"status":NaN}', '{"supported_cells":1e1000}'])
def test_unreadable_optional_metadata_is_reported(semantic_api, contents):
    client, bundle, manifest, descriptor = semantic_api
    publish(bundle, manifest, descriptor)
    (bundle / "s9_geo" / "semantics.json").write_text(contents)
    scene = get_scene(client)
    assert_geometry_retained(scene)
    assert scene["semantic_layers"] == []
    assert "HTTP 409" in scene["semantic_reason"]
