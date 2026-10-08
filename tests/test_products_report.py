"""Constructed mathematical/reader unit cases, not reconstruction acceptance runs."""

import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from drishti.geo.products import (
    Grid,
    georeference,
    orthoproject,
    project_camera,
    rasterize_cloud,
    read_cloud,
    validate_cloud,
    write_geotiff,
)
from drishti.io.exporters import (
    copy_geotiffs,
    copy_obj,
    export_gltf,
    export_las,
    export_optional,
    export_ply,
    recenter_mesh,
    run_tool,
)
from drishti.report.accuracy import validate_checkpoints
from drishti.report.builder import build_report, confidence_regions


def cloud(points, inferred=None, confidence=None):
    n = len(points)
    return {"points": np.asarray(points, dtype=np.float64),
            "colors": np.full((n, 3), [255, 0, 0], dtype=np.uint8),
            "confidence": np.asarray(confidence if confidence is not None else [0.6] * n),
            "inferred": np.asarray(inferred if inferred is not None else [False] * n)}


def geo_config(**kwargs):
    return SimpleNamespace(**{"resolution_m": 1., "max_cells": 1000,
                              "ground_method": "morphological", "ground_window_m": 3.,
                              "ground_threshold_m": .2, "nodata": -9999., **kwargs})


def reference(vertical="ellipsoidal"):
    pytest.importorskip("pyproj")
    return georeference({"frame": "projected_local", "origin": [500000., 3000000., 100.],
                         "crs": {"epsg": 32643, "vertical": vertical,
                                 "source_altitude_datum": vertical}})


def test_dsm_dtm_holes_inferred_and_morphological_roof_rejection():
    pytest.importorskip("scipy")
    points = [[x + .5, y + .5, 5. if (x, y) == (2, 2) else 0.]
              for x in range(5) for y in range(5) if (x, y) != (0, 0)]
    points += [[.5, .5, 50.], [6.5, .5, 10.]]
    data = cloud(points, inferred=[False] * 24 + [True, True])
    products = rasterize_cloud(data, geo_config())
    grid = products["grid"]
    row, col = grid.indices(data["points"])
    assert np.isnan(products["dsm"][row[-2], col[-2]])
    assert products["mask"][row[-2], col[-2]] == 2
    assert np.isnan(products["confidence"][row[-1], col[-1]])
    roof = points.index([2.5, 2.5, 5.])
    assert products["dsm"][row[roof], col[roof]] == 5.
    assert np.isnan(products["dtm"][row[roof], col[roof]])
    assert not products["ground"][roof]
    assert products["metadata"]["inferred_points_excluded"] == 2
    assert products["metadata"]["scene_completeness"] is None
    assert np.all(np.isnan(products["dsm"][products["mask"] != 1]))


def test_grid_guard_and_float32_input_rejected():
    pytest.importorskip("scipy")
    data = cloud([[0., 0., 0.], [100., 100., 0.]])
    with pytest.raises(ValueError, match="max_cells"):
        rasterize_cloud(data, geo_config(max_cells=1))
    data["points"] = data["points"].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        rasterize_cloud(data, geo_config())


def test_geotiff_precision_mask_and_unknown_datum_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    ref = reference("unknown")
    ref["origin"] = [500000.123456789, 3000000.987654321, 10000000.]
    grid = Grid(.125, .875, .25, 2, 2)
    data = np.array([[10000000.001, -9999.], [-9999., 10000000.002]])
    path = tmp_path / "dsm.tif"
    valid = data != -9999.
    write_geotiff(path, data, grid, ref, nodata=-9999., valid=valid)
    with rasterio.open(path) as raster:
        assert raster.crs.to_epsg() == 32643
        assert raster.tags()["vertical_datum"] == "unknown"
        assert raster.tags()["vertical_conversion"] == "null"
        assert raster.dtypes[0] == "float64"
        np.testing.assert_array_equal(raster.read(1, masked=True).mask, ~valid)
        np.testing.assert_array_equal(raster.read(1), data)
        assert raster.transform.c == ref["origin"][0] + grid.left
        assert raster.transform.f == ref["origin"][1] + grid.top


def camera():
    return {"frame_id": "frame", "image_path": "image.png", "width": 21, "height": 21,
            "center": [.5, .5, 10.], "R_cw": [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
            "K": [[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]], "distortion": []}


def test_orthomosaic_samples_observed_image_and_rejects_occlusion_and_mask(tmp_path):
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    image = np.zeros((21, 21, 3), dtype=np.uint8)
    image[10, 10] = [7, 125, 33]
    image_module.fromarray(image).save(tmp_path / "image.png")
    products = {"grid": Grid(0., 1., 1., 1, 1), "dsm": np.zeros((1, 1))}
    rgba, metadata = orthoproject(products, [camera()], tmp_path, chunk_size=1)
    np.testing.assert_array_equal(rgba[0, 0], [7, 125, 33, 255])
    assert metadata["textured_cells"] == 1
    occluder = {"vertices": np.array([[-2., -2., 5.], [3., -2., 5.], [.5, 3., 5.]]),
                "faces": np.array([[0, 1, 2]])}
    hidden, _ = orthoproject(products, [camera()], tmp_path, mesh=occluder, chunk_size=1)
    assert hidden[0, 0, 3] == 0
    image_module.fromarray(np.full((21, 21), 255, dtype=np.uint8)).save(tmp_path / "mask.png")
    masked, _ = orthoproject(products, [camera()], tmp_path, masks={"frame": "mask.png"}, chunk_size=1)
    assert masked[0, 0, 3] == 0
    bad = {**camera(), "image_path": "../outside.png"}
    invalid, meta = orthoproject(products, [bad], tmp_path, chunk_size=1)
    assert not invalid.any()
    assert len(meta["cameras_skipped"]) == 1


def precision_mesh():
    return {"vertices": np.array([[1000000.001, 1000000.001, 20.],
                                  [1000000.003, 1000000.001, 20.],
                                  [1000000.001, 1000000.003, 20.001]], dtype=np.float64),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "confidence": np.array([.1, .5, .9]), "inferred": np.array([False, False, True]),
            "uv": np.array([[0., 0.], [1., 0.], [0., 1.]])}


def test_recenter_precedes_float32_and_retains_axis_inverse():
    mesh, ref = precision_mesh(), reference()
    converted, sidecar = recenter_mesh(mesh, ref)
    assert np.ptp(converted[:, 0]) > .0019
    serialized = converted.astype(np.float32)
    restored = serialized.astype(np.float64) @ np.array(sidecar["source_to_gltf_rotation"])
    restored += np.array(sidecar["projected_anchor"])
    np.testing.assert_allclose(restored, mesh["vertices"] + ref["origin"], rtol=0., atol=1e-8)
    assert sidecar["float32_max_rounding_error_m"] < 1e-9


@pytest.mark.parametrize("extension", ["glb", "gltf"])
def test_textured_gltf_roundtrip_precision_and_custom_attributes(tmp_path, extension):
    trimesh = pytest.importorskip("trimesh")
    image_module = pytest.importorskip("PIL.Image")
    texture = tmp_path / "texture.png"
    image = np.full((2, 2, 4), [4, 9, 81, 255], dtype=np.uint8)
    image[0, 0, 3] = 0
    image_module.fromarray(image).save(texture)
    path = tmp_path / f"model.{extension}"
    mesh, ref = precision_mesh(), reference()
    sidecar = export_gltf(path, mesh, texture, ref)
    scene = trimesh.load(path, force="scene", process=False)
    geometry = next(iter(scene.geometry.values()))
    assert geometry.visual.kind == "texture"
    assert geometry.visual.material.alphaMode == "MASK"
    assert geometry.visual.material.alphaCutoff == pytest.approx(0.5)
    np.testing.assert_array_equal(np.asarray(geometry.visual.material.baseColorTexture)[..., 3],
                                  image[..., 3])
    assert "_CONFIDENCE" in geometry.vertex_attributes
    assert "_INFERRED" in geometry.vertex_attributes
    restored = geometry.vertices @ np.array(sidecar["source_to_gltf_rotation"])
    restored += sidecar["projected_anchor"]
    np.testing.assert_allclose(restored, mesh["vertices"] + ref["origin"], atol=1e-8, rtol=0.)
    assert path.with_suffix(path.suffix + ".georeference.json").is_file()


def test_obj_texture_material_copy_validated_by_standard_reader(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    image_module = pytest.importorskip("PIL.Image")
    obj, material, texture = (tmp_path / file for file in ("model.obj", "model.mtl", "texture.png"))
    obj.write_text("mtllib model.mtl\nusemtl observed\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
                   "vt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
    material.write_text("newmtl observed\nKd 1 1 1\nmap_Kd texture.png\n")
    image_module.fromarray(np.full((2, 2, 4), [10, 20, 30, 255], dtype=np.uint8)).save(texture)
    copied = copy_obj(obj, material, texture, tmp_path / "export")
    scene = trimesh.load(copied, force="scene", process=False)
    assert next(iter(scene.geometry.values())).visual.kind == "texture"
    assert (copied.parent / "texture.png").read_bytes() == texture.read_bytes()
    obj.write_text(obj.read_text().replace("mtllib model.mtl", "mtllib ../model.mtl"))
    with pytest.raises(ValueError, match="escaping"):
        copy_obj(obj, material, texture, tmp_path / "bad")


def test_geotiff_available_partial_product_is_preserved(tmp_path):
    pytest.importorskip("rasterio")
    ref = reference()
    path = tmp_path / "dsm.tif"
    write_geotiff(path, np.array([[101.]]), Grid(0., 1., 1., 1, 1), ref,
                  nodata=-9999., valid=np.ones((1, 1), dtype=bool))
    copied = copy_geotiffs({"dsm": path}, tmp_path / "export", ref)
    assert set(copied) == {"dsm"}
    assert copied["dsm"].read_bytes() == path.read_bytes()


def test_external_tool_returncode_and_stale_output_never_count_as_execution(tmp_path):
    output, log = tmp_path / "product.bin", tmp_path / "process.log"
    output.write_bytes(b"previous-product")
    with pytest.raises(RuntimeError, match="did not create"):
        run_tool([sys.executable, "-c", "pass"], log, output)
    assert not output.exists()
    with pytest.raises(RuntimeError, match="exited 7"):
        run_tool([sys.executable, "-c", "raise SystemExit(7)"], log, output)
    assert log.is_file()


def test_las_streaming_projected_crs_confidence_and_inferred_reader(tmp_path):
    laspy = pytest.importorskip("laspy")
    data = cloud([[.0004, .0026, .0014], [12.001, 3.002, 2.001]],
                 inferred=[False, True], confidence=[.2, .8])
    ref, path = reference(), tmp_path / "cloud.las"
    export_las(path, data, ref, scale_m=.001, chunk_size=1)
    loaded = laspy.read(path)
    assert loaded.header.parse_crs().to_epsg() == 32643
    np.testing.assert_allclose(np.column_stack((loaded.x, loaded.y, loaded.z)),
                               data["points"] + ref["origin"], rtol=0., atol=.0005)
    np.testing.assert_allclose(loaded.confidence, [.2, .8], atol=1e-7)
    np.testing.assert_array_equal(loaded.inferred, [0, 1])
    provenance = json.loads(next(vlr.record_data for vlr in loaded.vlrs if vlr.user_id == "DRISHTI"))
    assert provenance["vertical"] == "ellipsoidal"
    assert provenance["vertical_conversion"] is None


def test_point_ply_real_double_confidence_inferred(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    data = cloud([[1000000.001, 0., 1.], [1000000.002, 1., 2.]],
                 inferred=[False, True], confidence=[.2, .8])
    path = tmp_path / "cloud.ply"
    export_ply(path, data, reference(), chunk_size=1)
    loaded = plyfile.PlyData.read(path)["vertex"]
    np.testing.assert_array_equal(loaded["x"], data["points"][:, 0])
    np.testing.assert_array_equal(loaded["inferred"], [0, 1])
    np.testing.assert_allclose(loaded["confidence"], [.2, .8], atol=1e-7)


def checkpoint_csv(path, rows):
    fields = ["id", "independent", "vertical_datum", "x", "y", "z", "epsg",
              "longitude", "latitude", "altitude_m"]
    # Distinct schemas prevent blank longitude/latitude keys from selecting them.
    fields = [field for field in fields if any(field in row for row in rows)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_independent_checkpoints_projected_geographic_datum_nodata(tmp_path):
    pytest.importorskip("rasterio")
    pyproj = pytest.importorskip("pyproj")
    ref, grid = reference(), Grid(0., 2., 1., 2, 2)
    raster = tmp_path / "dsm.tif"
    values = np.array([[102., -9999.], [103., 104.]])
    write_geotiff(raster, values, grid, ref, nodata=-9999., valid=values != -9999.,
                  tags={"elevation_frame": "projected_absolute"})
    csv_path = tmp_path / "checks.csv"
    valid = {"id": "good", "independent": "true", "vertical_datum": "ellipsoidal",
             "x": 500000.5, "y": 3000001.5, "z": 101.5, "epsg": 32643}
    checkpoint_csv(csv_path, [valid, {**valid, "id": "hole", "x": 500001.5},
                              {**valid, "id": "datum", "vertical_datum": "orthometric"},
                              {**valid, "id": "fit", "independent": "false"},
                              {**valid, "id": "nan", "z": "nan"},
                              {**valid, "id": "outside", "x": 500020.}])
    measured = validate_checkpoints(csv_path, raster, target_m=1.)
    assert measured["status"] == "MEASURED"
    assert measured["valid_count"] == 1
    assert measured["excluded_count"] == 5
    assert measured["rmse_z_m"] == .5
    assert measured["full_3d_accuracy"] == "UNVALIDATED"
    assert measured["spatial_rmse_m"] is None
    longitude, latitude = pyproj.Transformer.from_crs(32643, 4326, always_xy=True).transform(
        valid["x"], valid["y"])
    checkpoint_csv(csv_path, [{"id": "geo", "independent": "true", "vertical_datum": "ellipsoidal",
                               "longitude": longitude, "latitude": latitude, "altitude_m": 101.5}])
    geographic = validate_checkpoints(csv_path, raster, target_m=1.)
    assert geographic["valid_count"] == 1
    assert geographic["rmse_z_m"] == .5


def test_unknown_datum_cannot_validate_accuracy(tmp_path):
    pytest.importorskip("rasterio")
    path = tmp_path / "dsm.tif"
    write_geotiff(path, np.array([[1.]]), Grid(0., 1., 1., 1, 1), reference("unknown"),
                  nodata=-9999., valid=np.ones((1, 1), dtype=bool),
                  tags={"elevation_frame": "projected_absolute"})
    checks = tmp_path / "checks.csv"
    checkpoint_csv(checks, [{"independent": "true", "vertical_datum": "unknown",
                             "x": 500000.5, "y": 3000000.5, "z": 1., "epsg": 32643}])
    measured = validate_checkpoints(checks, path, target_m=1.)
    assert measured["status"] == "UNVALIDATED"
    assert measured["rmse_z_m"] is None


def test_region_confidence_excludes_inferred_and_does_not_invent_accuracy():
    data = cloud([[0., 0., 1.], [1., 1., 1.], [25., 0., 2.]],
                 inferred=[False, True, True], confidence=[.2, .9, .8])
    regional = confidence_regions(data, 20.)
    assert regional["mean_observed_confidence"] == .2
    assert regional["regions"][0]["mean_observed_confidence"] == .2
    assert regional["regions"][1]["mean_observed_confidence"] is None
    assert all(region["absolute_accuracy_m"] is None for region in regional["regions"])


def partial_bundle(tmp_path, manifest=None):
    from drishti.bundle import Bundle

    return Bundle(tmp_path, {"bundle_version": "1", "status": "failed", "stages": {},
                             "inputs": [], "config_snapshot": {}, **(manifest or {})})


def test_failure_path_report_null_timings_confidence_crs_and_html_escaping(tmp_path):
    bundle = partial_bundle(tmp_path, {"bundle_version": "1", "run_id": "unit",
        "status": "failed", "error": "<script>alert(1)</script>",
        "stages": {"s2_poses": {"status": "failed", "wall_seconds": 1.25,
                    "environment": {"device": "cpu", "hostname": "<img src=x onerror=alert(1)>"},
                    "confidence": None, "metrics": {"gps_rmse_m": .3}},
                   "s7_dense": {"status": "skipped", "confidence": None}}})
    cfg = SimpleNamespace(dataset=SimpleNamespace(name="<script>evil</script>"),
                          report=SimpleNamespace(check_points=None, accuracy_target_m=1., region_size_m=20.))
    outputs = build_report(bundle, cfg)
    report = bundle.read_json(outputs["json"])
    assert report["accuracy"]["status"] == "UNVALIDATED"
    assert report["accuracy"]["absolute_spatial_rmse_m"] is None
    assert report["stages"][0]["confidence"] is None
    assert report["stages"][1]["wall_seconds"] is None
    assert report["timing"]["recorded_stage_wall_seconds_sum"] is None
    assert report["gps_fit"]["independent_accuracy"] is False
    assert report["georeference"] is None
    markup = (tmp_path / outputs["html"]).read_text()
    assert "<script>" not in markup
    assert "<img src=x" not in markup
    assert "&lt;script&gt;" in markup
    assert "&lt;img src=x" in markup


def test_partial_stages_record_missing_outputs_instead_of_crashing(tmp_path):
    from drishti.stages.s9_geo import STAGE as geo_stage
    from drishti.stages.s10_export import STAGE as export_stage

    bundle = partial_bundle(tmp_path)
    cfg = SimpleNamespace(geo=geo_config(), dense=SimpleNamespace(max_points_per_chunk=1),
                          export=SimpleNamespace(formats=["ply"], optional_formats=[],
                                                 las_scale_m=.001, blender="nonexistent-blender"))
    geo = geo_stage.run(bundle, cfg, None)
    assert geo.degraded_reason
    assert set(geo.outputs) == {"metadata", "semantic_metadata"}
    semantics = bundle.read_json(geo.outputs["semantic_metadata"])
    assert semantics["status"] == "unavailable" and semantics["reason"]
    assert semantics["raster_path"] is None and semantics["confidence_path"] is None
    assert semantics["preview"] is None and semantics["provenance"] is None
    assert semantics["classes"] == [] and semantics["supported_cells"] == 0
    assert semantics["accuracy_status"] == "UNVALIDATED"
    assert bundle.read_json(geo.outputs["metadata"])["semantics"] == semantics
    export = export_stage.run(bundle, cfg, None)
    records = bundle.read_json(export.outputs["exports"])
    assert export.degraded_reason
    assert records["formats"]["las"]["status"] == "unavailable"
    assert records["formats"]["ply"]["path"] is None
    assert "fbx" in records["missing_mandatory"]
    assert records["georeference"]["vertical"] == "unknown"


def test_report_rejects_malformed_metadata_and_stale_successful_export(tmp_path):
    bundle = partial_bundle(tmp_path)
    bundle.write_json("s9_geo/geo.json", [])
    bundle.write_json("s10_export/exports.json", {"formats": {
        "glb": {"status": "ok", "path": "s10_export/missing.glb", "reason": None},
        "las": "invalid_record"}, "georeference": "invalid_record"})
    bundle.manifest["stages"] = {
        "s9_geo": {"outputs": {"metadata": "s9_geo/geo.json"}},
        "s10_export": {"outputs": {"exports": "s10_export/exports.json"}}}
    cfg = SimpleNamespace(report=SimpleNamespace(check_points=None, accuracy_target_m=1., region_size_m=20.))
    outputs = build_report(bundle, cfg)
    report = bundle.read_json(outputs["json"])
    assert report["exports"]["formats"]["glb"]["status"] == "failed"
    assert report["exports"]["formats"]["glb"]["path"] is None
    assert report["exports"]["formats"]["las"]["status"] == "failed"
    assert report["georeference"] is None
    assert len(report["issues"]) == 3


def sidecar_fixture(tmp_path):
    """Final S7/S8 artifact-shape fixture, not a simulated footage reconstruction."""
    image_module = pytest.importorskip("PIL.Image")
    from drishti.geo.crs import crs_metadata
    from drishti.io.pointcloud import create_point_arrays, save_pointcloud
    from drishti.recon.texturing import write_material, write_obj

    bundle = partial_bundle(tmp_path)
    source = cloud([[.5, .5, 2.], [1.5, .5, 2.], [.5, 1.5, 2.], [1.5, 1.5, 9.]],
                   inferred=[False, False, False, True], confidence=[.2, .4, .6, .9])
    directory = bundle.stage_dir("s7_dense")
    arrays = create_point_arrays(directory / "arrays", len(source["points"]))
    for key, values in source.items():
        arrays[key][:] = values
    for value in arrays.values():
        value.flush()
    save_pointcloud(directory / "dense.npz", arrays)
    bundle.write_json("s7_dense/dense.json", {
        "points": len(source["points"]), "frame": "projected_local",
        "arrays": {key: (directory / "arrays" / f"{key}.npy").relative_to(bundle.root).as_posix()
                   for key in arrays}})
    crs = crs_metadata(32643, [{"latitude": 27., "longitude": 75.}],
                       SimpleNamespace(mode="epsg", vertical="source", geoid_path=None), "unknown")
    bundle.manifest["crs"] = crs
    cam = {**camera(), "projection_domain": "undistorted_pinhole"}
    bundle.write_json("s6_global/poses.json", {"frame": "projected_local", "origin": [500000., 3000000., 100.],
                                             "crs": crs, "cameras": [cam]})
    image_module.fromarray(np.full((21, 21, 3), [7, 125, 33], dtype=np.uint8)).save(tmp_path / "image.png")
    mesh_folder = bundle.stage_dir("s8_mesh")
    mesh = {"vertices": source["points"][:3], "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "confidence": arrays["confidence"][:3], "inferred": arrays["inferred"][:3],
            "colors": arrays["colors"][:3], "uv": np.array([[0., 0.], [1., 0.], [0., 1.]]),
            "source_views": np.array([0])}
    np.savez(mesh_folder / "mesh.npz", **mesh)
    write_obj(mesh_folder / "model.obj", mesh["vertices"], mesh["faces"], mesh["uv"], "model.mtl")
    write_material(mesh_folder / "model.mtl")
    image_module.fromarray(np.full((2, 2, 4), [12, 34, 56, 255], dtype=np.uint8)).save(mesh_folder / "texture.png")
    bundle.manifest["stages"] = {
        "s6_global": {"status": "done", "outputs": {"poses": "s6_global/poses.json"}},
        "s7_dense": {"status": "done", "outputs": {"cloud": "s7_dense/dense.npz",
                    "metadata": "s7_dense/dense.json", "arrays": "s7_dense/arrays"}},
        "s8_mesh": {"status": "done", "outputs": {key: f"s8_mesh/{filename}" for key, filename in
                    {"mesh": "mesh.npz", "obj": "model.obj", "material": "model.mtl", "texture": "texture.png"}.items()}}}
    cfg = SimpleNamespace(geo=geo_config(), dense=SimpleNamespace(max_points_per_chunk=1),
                          export=SimpleNamespace(formats=[], optional_formats=[], las_scale_m=.001,
                                                 blender="nonexistent-blender"),
                          report=SimpleNamespace(check_points=None, accuracy_target_m=1., region_size_m=1.))
    return bundle, cfg


def test_final_s7_mmap_loader_preserves_float32_scores_and_rejects_broken_sidecars(tmp_path, monkeypatch):
    pytest.importorskip("pyproj")
    bundle, _ = sidecar_fixture(tmp_path)

    def no_npz(_path):
        pytest.fail("S7 sidecar loading must not materialize the NPZ")

    monkeypatch.setattr("drishti.geo.products.read_npz", no_npz)
    mapped = read_cloud(bundle)
    checked = validate_cloud(mapped, chunk_size=1)
    assert isinstance(checked["points"], np.memmap)
    assert isinstance(checked["confidence"], np.memmap)
    assert checked["confidence"].dtype == np.float32
    assert not checked["points"].flags.writeable
    assert confidence_regions(checked, 1., chunk_size=1) == confidence_regions(checked, 1.)
    (tmp_path / "s7_dense/arrays/confidence.npy").unlink()
    with pytest.raises(OSError):
        read_cloud(bundle)
    (tmp_path / "s7_dense/dense.json").unlink()
    from drishti.errors import InputError

    with pytest.raises(InputError, match="Recorded artifact is missing"):
        read_cloud(bundle)


def test_final_artifact_adapters_output_paths_masks_and_report_semantics(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("laspy")
    pytest.importorskip("trimesh")
    pytest.importorskip("cv2")
    from drishti.stages.s9_geo import STAGE as geo_stage
    from drishti.stages.s10_export import STAGE as export_stage

    bundle, cfg = sidecar_fixture(tmp_path)
    geo = geo_stage.run(bundle, cfg, None)
    assert {"dsm", "dtm", "ortho", "confidence", "support", "mask", "metadata"}.issubset(geo.outputs)
    bundle.manifest["stages"]["s9_geo"] = {"status": "done", "outputs": geo.outputs, "metrics": geo.metrics}
    with rasterio.open(bundle.path(geo.outputs["dsm"])) as dsm:
        assert dsm.tags()["inferred_excluded"] == "true"
        assert dsm.tags()["elevation_frame"] == "projected_absolute"
        assert dsm.tags()["vertical_datum"] == "unknown"
        assert dsm.tags()["source_datum"] == "unknown"
        assert json.loads(dsm.tags()["vertical_conversion"]) == "none"
        assert dsm.read(1, masked=True).count() == 3
    with rasterio.open(bundle.path(geo.outputs["mask"])) as mask:
        assert sorted(mask.read(1).ravel().tolist()) == [1, 1, 1, 2]
    exports = export_stage.run(bundle, cfg, None)
    records = bundle.read_json(exports.outputs["exports"])
    for key in ("obj", "ply", "las", "geotiff", "glb", "gltf"):
        assert records["formats"][key]["status"] == "ok", records["formats"][key]
        assert bundle.path(records["formats"][key]["path"]).exists()
    for key, path in exports.outputs.items():
        assert not path.startswith("/")
        assert bundle.path(path).exists(), key
    assert {"obj_assets", "gltf_assets", "geotiff_assets"}.issubset(exports.outputs)
    assert records["georeference"]["origin"] == [500000., 3000000., 100.]
    assert len(records["georeference"]["glb_axes"]["gltf_to_local"]) == 16
    bundle.manifest["stages"]["s10_export"] = {"status": "degraded", "outputs": exports.outputs,
                                               "metrics": exports.metrics, "confidence": None}
    bundle.manifest["stages"]["s2_poses"] = {"status": "done", "confidence": None,
        "metrics": {"alignment": {"rmse_m": .8}, "optimization": {"gnss_fit_rmse_m": .5}}}
    report_outputs = build_report(bundle, cfg)
    report = bundle.read_json(report_outputs["json"])
    assert report["confidence"]["observed_points"] == 3
    assert report["confidence"]["inferred_points"] == 1
    assert report["accuracy"]["status"] == "UNVALIDATED"
    assert report["accuracy"]["absolute_spatial_rmse_m"] is None
    assert report["coverage"]["scene_completeness"] is None
    assert report["gps_fit"]["independent_accuracy"] is False
    assert next(item for item in report["gps_fit"]["stages"] if item["stage"] == "s2_poses")["metrics"]["optimization"]["gnss_fit_rmse_m"] == .5
    assert all(stage["confidence"] is None for stage in report["stages"])


def test_optional_converters_cannot_consume_las_from_a_failed_previous_export(tmp_path, monkeypatch):
    pytest.importorskip("trimesh")
    from drishti.stages.s10_export import STAGE

    bundle, cfg = sidecar_fixture(tmp_path)
    directory = bundle.stage_dir("s10_export")
    (directory / "cloud.las").write_bytes(b"previous-export")
    cfg.export.optional_formats = ["3dtiles", "potree"]

    def fail_las(*args, **kwargs):
        raise ValueError("Writer failed for current geometry")

    def no_converter(*args, **kwargs):
        pytest.fail("Optional converters must not read a stale LAS")

    monkeypatch.setattr("drishti.io.exporters.export_las", fail_las)
    monkeypatch.setattr("drishti.io.exporters.export_optional", no_converter)
    result = STAGE.run(bundle, cfg, None)
    records = bundle.read_json(result.outputs["exports"])
    assert records["formats"]["las"]["status"] == "failed"
    assert all(records["formats"][kind]["status"] == "unavailable" for kind in ("3dtiles", "potree"))


def test_projection_uses_final_image_domain_and_handles_no_supported_cells(tmp_path):
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    cam = {**camera(), "distortion": [.5], "camera_model": "SIMPLE_RADIAL",
           "projection_domain": "undistorted_pinhole"}
    points = np.array([[1.5, .5, 2.]])
    uv, z = project_camera(points, cam)
    np.testing.assert_allclose(uv, [[11.25, 10.]])
    np.testing.assert_allclose(z, [8.])
    image_module.fromarray(np.zeros((21, 21, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    rgba, metadata = orthoproject({"grid": Grid(0., 1., 1., 1, 1), "dsm": np.array([[np.nan]])},
                                  [cam], tmp_path, chunk_size=1)
    assert not rgba.any()
    assert metadata["textured_cells"] == 0
    assert metadata["cameras_skipped"] == []


def test_stale_artifacts_do_not_supply_current_confidence_or_checkpoint_accuracy(tmp_path):
    bundle = partial_bundle(tmp_path, {"stages": {
        "s7_dense": {"status": "stale", "confidence": .9, "outputs": {"cloud": "old.npz"}},
        "s9_geo": {"status": "stale", "outputs": {"dsm": "old.tif"}},
        "s10_export": {"status": "stale", "outputs": {"exports": "old.json"}}}})
    cfg = SimpleNamespace(report=SimpleNamespace(check_points=tmp_path / "checks.csv",
                                                 accuracy_target_m=1., region_size_m=20.))
    outputs = build_report(bundle, cfg)
    report = bundle.read_json(outputs["json"])
    assert report["confidence"]["mean_observed_confidence"] is None
    assert report["stages"][0]["confidence"] is None
    assert report["accuracy"]["elevation_validation"]["status"] == "UNVALIDATED"
    assert report["exports"]["formats"] == {}


def test_volume_api_consumes_real_s9_masks_and_absolute_elevation_tags(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("rasterio")
    pytest.importorskip("cv2")
    from drishti.stages.s9_geo import STAGE
    from server.bundles import BundleStore
    from server.measurements import volume

    root = tmp_path / "unit-contract"
    root.mkdir()
    bundle, cfg = sidecar_fixture(root)
    result = STAGE.run(bundle, cfg, None)
    bundle.manifest["stages"]["s9_geo"] = {"status": "done", "outputs": result.outputs}
    store = BundleStore(tmp_path, max_json_bytes=1024 * 1024)
    value, support = volume(store, root.name, bundle.manifest,
                            np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]]),
                            reference=0., max_cells=100)
    assert value == 6.
    assert support["supported_area_m2"] == 3.
    assert support["inferred_excluded_area_m2"] == 1.
    assert support["inferred_excluded_cells"] == 1
    assert support["mean_source_confidence"] == pytest.approx(.4)
    assert support["confidence_is_absolute_accuracy"] is False
    assert support["vertical_datum"] == "unknown"


def test_failed_geotiff_group_keeps_valid_copies_in_recorded_outputs(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("cv2")
    pytest.importorskip("trimesh")
    from drishti.stages.s9_geo import STAGE as geo_stage
    from drishti.stages.s10_export import STAGE as export_stage

    bundle, cfg = sidecar_fixture(tmp_path)
    geo = geo_stage.run(bundle, cfg, None)
    bundle.manifest["stages"]["s9_geo"] = {"status": "degraded", "outputs": geo.outputs}
    ref = georeference(bundle.read_json("s6_global/poses.json"))
    write_geotiff(bundle.path(geo.outputs["dtm"]), np.full((2, 2), -9999.),
                  Grid(0., 2., 1., 2, 2), ref, nodata=-9999., valid=np.zeros((2, 2), dtype=bool))
    result = export_stage.run(bundle, cfg, None)
    records = bundle.read_json(result.outputs["exports"])
    assert records["formats"]["geotiff"]["status"] == "failed"
    assert "geotiff_dsm" in result.outputs
    assert "geotiff_ortho" in result.outputs
    assert "dsm" in records["formats"]["geotiff"]["details"]["products"]
    assert "dtm" not in records["formats"]["geotiff"]["details"]["products"]
    assert "geotiff" in records["missing_mandatory"]


@pytest.mark.parametrize("datum", ["unknown", "orthometric"])
def test_ecef_tiles_cannot_implicitly_relabel_heights_as_ellipsoidal(tmp_path, datum):
    pytest.importorskip("laspy")
    from drishti.io.exporters import ExportUnavailable

    las = tmp_path / "cloud.las"
    export_las(las, cloud([[0., 0., 0.]]), reference(datum), scale_m=.001, chunk_size=1)
    with pytest.raises(ExportUnavailable, match="ellipsoidal heights"):
        export_optional("3dtiles", las, tmp_path / "tiles", reference(datum))


def test_checkpoint_reader_rejects_local_elevations_instead_of_reporting_absolute_accuracy(tmp_path):
    pytest.importorskip("rasterio")
    path = tmp_path / "dsm.tif"
    write_geotiff(path, np.array([[2.]]), Grid(0., 1., 1., 1, 1), reference(),
                  nodata=-9999., valid=np.ones((1, 1), dtype=bool),
                  tags={"elevation_frame": "projected_local"})
    checks = tmp_path / "checks.csv"
    checkpoint_csv(checks, [{"independent": "true", "vertical_datum": "ellipsoidal",
                             "x": 500000.5, "y": 3000000.5, "z": 102., "epsg": 32643}])
    with pytest.raises(ValueError, match="projected_absolute"):
        validate_checkpoints(checks, path, target_m=1.)


def test_partial_report_measures_elevation_without_inventing_a_missing_target(tmp_path):
    pytest.importorskip("rasterio")
    bundle = partial_bundle(tmp_path)
    dsm = bundle.stage_dir("s9_geo") / "dsm.tif"
    write_geotiff(dsm, np.array([[102.]]), Grid(0., 1., 1., 1, 1), reference(),
                  nodata=-9999., valid=np.ones((1, 1), dtype=bool),
                  tags={"elevation_frame": "projected_absolute"})
    checks = tmp_path / "checks.csv"
    checkpoint_csv(checks, [{"independent": "true", "vertical_datum": "ellipsoidal",
                             "x": 500000.5, "y": 3000000.5, "z": 101.5, "epsg": 32643}])
    bundle.manifest["stages"] = {"s9_geo": {"status": "done", "outputs": {"dsm": "s9_geo/dsm.tif"}}}
    # Failure-path callers may have only a dataset checkpoint declaration.
    cfg = SimpleNamespace(report=None, dataset=SimpleNamespace(report=SimpleNamespace(check_points=checks)))
    outputs = build_report(bundle, cfg)
    accuracy = bundle.read_json(outputs["json"])["accuracy"]
    measured = accuracy["elevation_validation"]
    assert accuracy["design_target_m"] is None
    assert accuracy["status"] == "UNVALIDATED"
    assert measured["status"] == "MEASURED"
    assert measured["rmse_z_m"] == .5
    assert measured["target_m"] is None
    assert measured["target_met_elevation_only"] is None
    assert measured["checkpoints"][0]["exceeds_target"] is None
