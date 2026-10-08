"""Constructed numerical/raster/stage contracts, not real reconstruction acceptance."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from drishti.bundle.core import Bundle
from drishti.geo.products import Grid, georeference, orthoproject
from drishti.geo.semantics import (
    geographic_preview,
    load_frame,
    project_semantics,
    validate_descriptor,
)
from drishti.stages.s9_geo import S9Geo
from drishti.stages.s10_export import S10Export

CLASSES = [{"id": 1, "name": "ground", "color": [13, 117, 29], "source_ids": [0]},
           {"id": 512, "name": "water", "color": [0, 0, 0], "source_ids": [3]}]
CONFIDENCE_METHOD = "winning_source_class_softmax_probability_uncalibrated"
METADATA_KEYS = {"id", "title", "status", "reason", "classes", "nodata", "raster_path",
                 "confidence_path", "preview", "confidence_method", "provenance",
                 "supported_cells", "accuracy_status"}


def camera(frame_id="frame", **kwargs):
    return {"frame_id": frame_id, "image_path": f"{frame_id}.png", "width": 21, "height": 21,
            "center": [.5, .5, 10.], "R_cw": [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
            "K": [[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]],
            "distortion": [], "projection_domain": "undistorted_pinhole", **kwargs}


def maps(tmp_path, cam=None, labels=None, probability=None):
    """Write only constructed map/image inputs for isolated projection tests."""
    image = pytest.importorskip("PIL.Image")
    cam = cam or camera()
    shape = cam["height"], cam["width"]
    labels = np.ones(shape, np.uint16) if labels is None else labels
    probability = np.full(shape, .9, np.float32) if probability is None else probability
    identifier = cam["frame_id"]
    image.fromarray(np.full((*shape, 3), [8, 31, 65], np.uint8)).save(tmp_path / cam["image_path"])
    np.save(tmp_path / f"{identifier}_labels.npy", labels)
    np.save(tmp_path / f"{identifier}_probability.npy", probability)
    return {"frame_id": identifier, "path": f"{identifier}_labels.npy",
            "confidence_path": f"{identifier}_probability.npy", "height": shape[0],
            "width": shape[1], "image_domain": "undistorted", "labeled_pixels": int((labels != 0).sum())}


def descriptor(frames, **kwargs):
    return {"status": "ok", "reason": None, "classes": copy.deepcopy(CLASSES), "nodata": 0,
            "model": {"revision": "constructed-contract-input"},
            "source_labels": {"0": "source ground", "3": "source water"},
            "confidence_method": CONFIDENCE_METHOD, "frames": frames, **kwargs}


def products(grid=None, dsm=None, support=None, mask=None):
    grid = grid or Grid(0., 1., 1., 1, 1)
    return {"grid": grid, "dsm": np.zeros(grid.shape) if dsm is None else np.asarray(dsm),
            "support": np.ones(grid.shape, np.uint32) if support is None else np.asarray(support),
            "mask": np.ones(grid.shape, np.uint8) if mask is None else np.asarray(mask)}


def project(tmp_path, desc, surface=None, cameras=None, **kwargs):
    pytest.importorskip("cv2")
    return project_semantics(surface or products(), cameras or [camera()], tmp_path, desc,
                             min_probability=.5, chunk_size=1, **kwargs)


@pytest.mark.parametrize("invalid,reason", [
    ({"status": "made_up"}, "status"),
    ({"nodata": 65535}, "nodata"),
    ({"classes": [{**CLASSES[0], "id": 0}]}, "IDs"),
    ({"classes": [{**CLASSES[0], "id": 65535}]}, "IDs"),
    ({"classes": [CLASSES[0], CLASSES[0]]}, "IDs"),
    ({"classes": [CLASSES[0], {**CLASSES[1], "name": " Ground "}]}, "names"),
    ({"classes": [{**CLASSES[0], "name": "Straße"},
                  {**CLASSES[1], "name": " STRASSE "}]}, "names"),
    ({"classes": [{**CLASSES[0], "color": [255, -1, 1]}]}, "color"),
    ({"classes": [CLASSES[0], {**CLASSES[1], "source_ids": [0]}]}, "source IDs"),
    ({"classes": [{**CLASSES[0], "source_ids": [1, 1]}]}, "source IDs"),
    ({"model": "executed"}, "provenance"),
    ({"reason": {}}, "reason"),
    ({"frames": [{"frame_id": "a"}, {"frame_id": "a"}]}, "frame IDs"),
    ({"frames": [{"frame_id": " "}]}, "frame IDs"),
    ({"confidence_method": "geometry_confidence"}, "winning source-class"),
])
def test_descriptor_rejects_ambiguous_class_mapping(invalid, reason):
    with pytest.raises(ValueError, match=reason):
        validate_descriptor({**descriptor([]), **invalid})


@pytest.mark.parametrize("failure,reason", [
    ("label_dtype", "uint16"), ("confidence_dtype", "float32"),
    ("label_shape", "camera-shaped"), ("confidence_shape", "camera-shaped"),
    ("unknown_class", "absent from descriptor"), ("nan", "finite"),
    ("inf", "finite"), ("negative", "\\[0, 1\\]"), ("too_high", "\\[0, 1\\]"),
    ("unknown_probability", "Unknown"), ("declared_shape", "shape"),
    ("declared_domain", "domain"), ("declared_count", "labeled_pixels"),
    ("escaping_path", "subpath"),
])
def test_frame_validation_rejects_invalid_maps_and_domains(tmp_path, failure, reason):
    labels = np.ones((21, 21), np.uint16)
    probability = np.full((21, 21), .9, np.float32)
    if failure == "label_dtype":
        labels = labels.astype(np.uint8)
    elif failure == "confidence_dtype":
        probability = probability.astype(np.float64)
    elif failure == "label_shape":
        labels = labels[:20]
    elif failure == "confidence_shape":
        probability = probability[:20]
    elif failure == "unknown_class":
        labels[0, 0] = 600
    elif failure in {"nan", "inf", "negative", "too_high"}:
        probability[0, 0] = {"nan": np.nan, "inf": np.inf, "negative": -.1, "too_high": 1.1}[failure]
    elif failure == "unknown_probability":
        labels[0, 0] = 0
    frame = maps(tmp_path, labels=labels, probability=probability)
    if failure == "declared_shape":
        frame["width"] = 20
    elif failure == "declared_domain":
        frame["image_domain"] = "distorted"
    elif failure == "declared_count":
        frame["labeled_pixels"] = 0
    elif failure == "escaping_path":
        frame["path"] = "../outside.npy"
    with pytest.raises(ValueError, match=reason):
        load_frame(tmp_path, frame, camera(), {1, 512})
    projected, chances, meta = project(tmp_path, descriptor([frame]))
    assert not projected.any() and not chances.any()
    assert len(meta["cameras_skipped"]) == 1


def test_categorical_nearest_threshold_and_unknown_remain_zero(tmp_path):
    labels = np.zeros((21, 21), np.uint16)
    chances = np.zeros((21, 21), np.float32)
    labels[10, 10:13] = [512, 1, 512]
    chances[10, 10:13] = [.8, .4, .95]
    cam = camera(center=[.25, .5, 10.])
    frame = maps(tmp_path, cam, labels, chances)
    surface = products(Grid(0., 1., 1., 4, 1))
    actual, probability, meta = project(tmp_path, descriptor([frame]), surface, [cam])
    np.testing.assert_array_equal(actual, [[512, 0, 512, 0]])
    np.testing.assert_allclose(probability, [[.8, 0., .95, 0.]])
    assert actual.dtype == np.uint16 and probability.dtype == np.float32
    assert meta["supported_cells"] == 2 and meta["sampling"].startswith("nearest")


@pytest.mark.parametrize("threshold,expected", [(.5, 1), (.500000001, 0)])
def test_probability_threshold_is_not_rounded_down_to_source_float32(tmp_path, threshold, expected):
    pytest.importorskip("cv2")
    frame = maps(tmp_path, probability=np.full((21, 21), .5, np.float32))
    actual, probability, _ = project_semantics(
        products(), [camera()], tmp_path, descriptor([frame]),
        min_probability=threshold, chunk_size=1,
    )
    assert actual[0, 0] == expected
    assert probability[0, 0] == (.5 if expected else 0.)


def test_supported_mask_and_inferred_geometry_cannot_establish_semantic_support(tmp_path):
    frame = maps(tmp_path)
    surface = products(Grid(0., 1., 1., 4, 1), dsm=[[0., 0., 0., np.nan]],
                       support=[[1, 1, 0, 1]], mask=[[1, 2, 0, 1]])
    actual, probability, meta = project(tmp_path, descriptor([frame]), surface)
    np.testing.assert_array_equal(actual, [[1, 0, 0, 0]])
    assert not probability[0, 1:].any()
    assert meta["supported_cells"] == 1


def test_semantics_and_ortho_share_occlusion_dynamic_masks_and_inferred_exclusion(tmp_path):
    image = pytest.importorskip("PIL.Image")
    frame = maps(tmp_path)
    occluder = {"vertices": np.array([[-2., -2., 5.], [3., -2., 5.], [.5, 3., 5.]]),
                "faces": np.array([[0, 1, 2]])}
    labels, probability, _ = project(tmp_path, descriptor([frame]), mesh=occluder)
    rgba, _ = orthoproject(products(), [camera()], tmp_path, mesh=occluder, chunk_size=1)
    assert not labels.any() and not probability.any() and not rgba[..., 3].any()
    inferred_mesh = {**occluder, "inferred": np.ones(3, bool)}
    labels, _, _ = project(tmp_path, descriptor([frame]), mesh=inferred_mesh)
    assert labels[0, 0] == 1
    image.fromarray(np.full((21, 21), 255, np.uint8)).save(tmp_path / "mask.png")
    labels, _, _ = project(tmp_path, descriptor([frame]), masks={"frame": "mask.png"})
    assert not labels.any()
    image.fromarray(np.zeros((20, 21), np.uint8)).save(tmp_path / "mask.png")
    labels, _, meta = project(tmp_path, descriptor([frame]), masks={"frame": "mask.png"})
    assert not labels.any() and "mask shape" in meta["cameras_skipped"][0]["reason"]


def test_dsm_samples_occlude_across_chunks_without_mesh(tmp_path):
    cam = camera(center=[0., .5, 10.], K=[[.1, 0., 10.], [0., .1, 10.], [0., 0., 1.]])
    frame = maps(tmp_path, cam)
    surface = products(Grid(-.5, 1., 1., 2, 1), dsm=[[0., 5.]])
    actual, probability, _ = project(tmp_path, descriptor([frame]), surface, [cam])
    np.testing.assert_array_equal(actual, [[0, 1]])
    assert probability[0, 0] == 0


def test_best_geometric_view_preserves_its_source_probability_and_partial_frames(tmp_path):
    near, far = camera("near"), camera("far", center=[.5, .5, 20.])
    near_frame = maps(tmp_path, near, np.full((21, 21), 512, np.uint16),
                      np.full((21, 21), .7, np.float32))
    far_frame = maps(tmp_path, far, probability=np.full((21, 21), .99, np.float32))
    missing_camera = {**near_frame, "frame_id": "not_registered"}
    bad = camera("bad")
    bad_frame = maps(tmp_path, bad)
    bad_frame["image_domain"] = "distorted"
    desc = descriptor([far_frame, near_frame, bad_frame, missing_camera], status="partial")
    actual, probability, meta = project(tmp_path, desc, cameras=[far, near, bad])
    assert actual[0, 0] == 512 and probability[0, 0] == pytest.approx(.7)
    assert {item["frame_id"] for item in meta["cameras_skipped"]} == {"bad", "not_registered"}


def test_broken_optional_frame_preserves_other_actual_frame_maps(tmp_path):
    valid, broken = camera("valid"), camera("broken")
    valid_frame, broken_frame = maps(tmp_path, valid), maps(tmp_path, broken)
    (tmp_path / broken_frame["path"]).unlink()
    desc = descriptor([broken_frame, valid_frame], status="partial")
    labels, confidence, meta = project(tmp_path, desc, cameras=[broken, valid])
    assert labels[0, 0] == 1 and confidence[0, 0] == pytest.approx(.9)
    assert meta["cameras_used"] == [{"frame_id": "valid", "selected_cell_updates": 1}]
    assert meta["cameras_skipped"][0]["frame_id"] == "broken"


@pytest.mark.parametrize("invalid", [{"model": None}, {"classes": []}, {"frames": []}])
def test_available_descriptor_requires_executed_mapped_frame_provenance(tmp_path, invalid):
    frame = maps(tmp_path)
    with pytest.raises(ValueError, match="actual frames and executed model provenance"):
        validate_descriptor({**descriptor([frame]), **invalid})


@pytest.mark.parametrize("near_value", [0, 1])
def test_invalid_near_view_cannot_replace_far_label_or_mix_geometry_confidence(tmp_path, near_value):
    near, far = camera("near"), camera("far", center=[.5, .5, 20.])
    near_frame = maps(tmp_path, near, np.full((21, 21), near_value, np.uint16),
                      np.full((21, 21), .4 if near_value else 0., np.float32))
    far_frame = maps(tmp_path, far, np.full((21, 21), 512, np.uint16),
                     np.full((21, 21), .73, np.float32))
    surface = {**products(), "confidence": np.array([[.001]])}
    for cameras in ([near, far], [far, near]):
        actual, probability, _ = project(tmp_path, descriptor([near_frame, far_frame]),
                                         surface, cameras)
        assert actual[0, 0] == 512 and probability[0, 0] == pytest.approx(.73)


def test_partial_dynamic_mask_descriptor_skips_only_unmasked_camera(tmp_path):
    image = pytest.importorskip("PIL.Image")
    near, far = camera("near"), camera("far", center=[.5, .5, 20.])
    near_frame = maps(tmp_path, near, np.full((21, 21), 512, np.uint16))
    far_frame = maps(tmp_path, far)
    image.fromarray(np.zeros((21, 21), np.uint8)).save(tmp_path / "far_mask.png")
    actual, probability, meta = project(tmp_path, descriptor([near_frame, far_frame]),
                                        cameras=[near, far], masks={"far": "far_mask.png"})
    assert actual[0, 0] == 1 and probability[0, 0] == pytest.approx(.9)
    assert meta["cameras_skipped"] == [{"frame_id": "near",
                                      "reason": "Recorded dynamic masks lack this final camera"}]
    assert meta["dynamic_masks"] == "required_per_projected_frame"


def test_native_projection_failure_preserves_explicit_actual_partial_cells(tmp_path, monkeypatch):
    from drishti.geo import semantics

    pytest.importorskip("cv2")
    frame = maps(tmp_path)
    real_samples = semantics.visible_dsm_samples

    def partial_samples(*args, **kwargs):
        yield next(real_samples(*args, **kwargs))
        raise RuntimeError("constructed native projection failure after one chunk")

    monkeypatch.setattr(semantics, "visible_dsm_samples", partial_samples)
    actual, probability, meta = project(tmp_path, descriptor([frame]),
                                        products(Grid(0., 1., 1., 2, 1)))
    np.testing.assert_array_equal(actual, [[1, 0]])
    np.testing.assert_allclose(probability, [[.9, 0.]])
    assert meta["supported_cells"] == 1
    assert meta["cameras_used"][0]["selected_cell_updates"] == 1
    assert "RuntimeError" in meta["cameras_used"][0]["partial_reason"]
    assert "after one chunk" in meta["cameras_skipped"][0]["reason"]


def test_distorted_projection_samples_actual_distorted_pixel_domain(tmp_path):
    cam = camera(distortion=[.5], camera_model="SIMPLE_RADIAL", projection_domain="distorted")
    labels = np.zeros((21, 21), np.uint16)
    probability = np.zeros((21, 21), np.float32)
    cam["K"][0][0] = 16.
    # x=5.5 gives pinhole u=18, distorted u=19: genuinely different label pixels.
    labels[10, 18:20], probability[10, 18:20] = [1, 512], [.9, .8]
    frame = maps(tmp_path, cam, labels, probability)
    frame["image_domain"] = "distorted"
    actual, _, _ = project(tmp_path, descriptor([frame]),
                           products(Grid(5., 1., 1., 1, 1)), [cam])
    assert actual[0, 0] == 512
    pinhole = {**cam, "projection_domain": "undistorted_pinhole"}
    actual, _, meta = project(tmp_path, descriptor([frame]),
                              products(Grid(5., 1., 1., 1, 1)), [pinhole])
    assert not actual.any() and "domain" in meta["cameras_skipped"][0]["reason"]
    frame["image_domain"] = "undistorted"
    actual, _, _ = project(tmp_path, descriptor([frame]),
                           products(Grid(5., 1., 1., 1, 1)), [pinhole])
    assert actual[0, 0] == 1


def test_geographic_preview_is_bounded_categorical_warp_with_transparent_nodata(tmp_path):
    pytest.importorskip("rasterio")
    pyproj = pytest.importorskip("pyproj")
    image = pytest.importorskip("PIL.Image")
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    x, y = pyproj.Transformer.from_crs(4326, 32631, always_xy=True).transform(8., 70.)
    ref = georeference({"frame": "projected_local", "origin": [x, y, 0.],
                        "crs": {"epsg": 32631, "vertical": "unknown"}})
    grid = Grid(0., 18000., 100., 300, 180)
    labels = np.full(grid.shape, 512, np.uint16)
    labels[:, :150] = 1
    labels[40:130, 100:200] = 0
    path = tmp_path / "preview.png"
    preview = geographic_preview(path, labels, grid, ref, CLASSES, long_side=96)
    rgba = np.asarray(image.open(path))
    assert preview["crs"] == "EPSG:4326" and preview["path"] == path.name
    assert rgba.shape == (preview["height"], preview["width"], 4)
    assert max(preview["width"], preview["height"]) <= 96
    assert {tuple(pixel) for pixel in rgba.reshape(-1, 4)} == {
        (0, 0, 0, 0), (0, 0, 0, 255), (13, 117, 29, 255)}
    expected_bounds = transform_bounds(ref["wkt"], 4326,
                                       *array_bounds(grid.height, grid.width,
                                                     grid.transform(ref["origin"])), densify_pts=21)
    np.testing.assert_allclose(preview["bounds"], expected_bounds, rtol=0., atol=1e-10)
    # Independently inverse-project destination pixel centers into original cells.
    west, south, east, north = preview["bounds"]
    rr, cc = np.indices(rgba.shape[:2])
    lon = west + (cc + .5) * (east - west) / preview["width"]
    lat = north - (rr + .5) * (north - south) / preview["height"]
    xx, yy = pyproj.Transformer.from_crs(4326, 32631, always_xy=True).transform(lon, lat)
    source_col = np.floor((xx - x) / grid.resolution).astype(int)
    source_row = np.floor((y + grid.top - yy) / grid.resolution).astype(int)
    inside = ((source_col >= 0) & (source_col < grid.width)
              & (source_row >= 0) & (source_row < grid.height))
    expected = np.zeros(rgba.shape[:2], np.uint16)
    expected[inside] = labels[source_row[inside], source_col[inside]]
    actual = np.where(rgba[..., 3] == 0, 0, np.where(rgba[..., 0] == 13, 1, 512))
    np.testing.assert_array_equal(actual, expected)
    assert not rgba[0, :, 3].all()  # genuine warped footprint, not a stretched source image


def test_geographic_preview_explicitly_disables_proj_networking_and_restores_host_state(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    pyproj = pytest.importorskip("pyproj")
    from rasterio.env import get_gdal_config

    ref = georeference({"frame": "projected_local", "origin": [500000., 3000000., 0.],
                        "crs": {"epsg": 32643, "vertical": "unknown"}})
    real_transformer = pyproj.Transformer.from_crs
    invoked = []

    def transformer(*args, **kwargs):
        assert not pyproj.network.is_network_enabled()
        assert get_gdal_config("PROJ_NETWORK") is False
        invoked.append(True)
        return real_transformer(*args, **kwargs)

    monkeypatch.setattr(pyproj.Transformer, "from_crs", transformer)
    previous = pyproj.network.is_network_enabled()
    try:
        pyproj.network.set_network_enabled(True)
        geographic_preview(tmp_path / "preview.png", np.ones((2, 2), np.uint16),
                           Grid(0., 2., 1., 2, 2), ref, CLASSES, long_side=16)
        assert invoked and pyproj.network.is_network_enabled()
    finally:
        pyproj.network.set_network_enabled(previous)


@pytest.mark.parametrize("geography", ["antimeridian", "unusable"])
def test_geographic_preview_rejects_unusable_extent_without_product(tmp_path, geography):
    pytest.importorskip("rasterio")
    pyproj = pytest.importorskip("pyproj")
    x, y = pyproj.Transformer.from_crs(4326, 32660, always_xy=True).transform(179.99, 0.)
    ref = georeference({"frame": "projected_local", "origin": [x, y, 0.],
                        "crs": {"epsg": 32660, "vertical": "unknown"}})
    if geography == "unusable":
        ref["origin"] = [1e30, 1e30, 0.]
    path = tmp_path / "preview.png"
    grid = Grid(-2000., 1000., 1000., 4, 2)
    with pytest.raises(ValueError, match="antimeridian|unusable"):
        geographic_preview(path, np.ones(grid.shape, np.uint16), grid, ref, CLASSES, long_side=16)
    assert not path.exists()


def stage_fixture(tmp_path, *, semantics=True):
    """Final-pose/observed-cloud artifact shapes, explicitly constructed unit input."""
    pytest.importorskip("scipy")
    pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    pytest.importorskip("cv2")
    bundle = Bundle(tmp_path, {"bundle_version": "1", "stages": {}, "inputs": [],
                               "config_snapshot": {}})
    cam = camera()
    frame = maps(tmp_path, cam)
    bundle.write_json("s6_global/poses.json", {
        "frame": "projected_local", "origin": [500000., 3000000., 100.],
        "crs": {"epsg": 32643, "vertical": "unknown"}, "cameras": [cam],
    })
    directory = bundle.stage_dir("s7_dense")
    points = np.array([[.5, .5, 2.], [1.5, .5, 2.], [.5, 1.5, 2.], [1.5, 1.5, 9.]])
    np.savez(directory / "dense.npz", points=points, colors=np.full((4, 3), 100, np.uint8),
             confidence=np.full(4, .6, np.float32), inferred=np.array([False, False, False, True]))
    directory = bundle.stage_dir("s8_mesh")
    np.savez(directory / "mesh.npz", vertices=points[:3], faces=np.array([[0, 1, 2]]),
             inferred=np.zeros(3, bool))
    mask_dir = bundle.stage_dir("s3_masking")
    image = pytest.importorskip("PIL.Image")
    image.fromarray(np.zeros((21, 21), np.uint8)).save(mask_dir / "mask.png")
    bundle.write_json("s3_masking/masks.json", {"frames": [{"frame_id": "frame",
                                                            "path": "s3_masking/mask.png"}]})
    bundle.manifest["stages"] = {
        "s6_global": {"outputs": {"poses": "s6_global/poses.json"}},
        "s7_dense": {"outputs": {"cloud": "s7_dense/dense.npz"}},
        "s8_mesh": {"outputs": {"mesh": "s8_mesh/mesh.npz"}},
        "s3_masking": {"outputs": {"masks": "s3_masking/masks.json"}},
    }
    if semantics:
        bundle.write_json("s3_masking/semantics.json", descriptor([frame]))
        bundle.manifest["stages"]["s3_masking"]["outputs"]["semantics"] = "s3_masking/semantics.json"
    cfg = SimpleNamespace(
        geo=SimpleNamespace(resolution_m=1., max_cells=1000, ground_method="morphological",
                            ground_window_m=1., ground_threshold_m=.2, nodata=-9999.),
        dense=SimpleNamespace(max_points_per_chunk=1),
        semantics=SimpleNamespace(model=None, min_probability=.5, preview_long_side=64, classes=[]),
        export=SimpleNamespace(formats=[], optional_formats=[], las_scale_m=.001,
                               blender="nonexistent-blender"),
    )
    return bundle, cfg


def test_s9_actual_products_tags_support_and_exact_status_schema(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    bundle, cfg = stage_fixture(tmp_path)
    assert "s3_masking" in S9Geo.requires and "semantics" in S9Geo.config_sections
    result = S9Geo().run(bundle, cfg, None)
    assert result.degraded_reason is None
    assert {"semantic", "semantic_confidence", "semantic_preview", "semantic_metadata"} <= result.outputs.keys()
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    assert set(meta) == METADATA_KEYS
    assert meta["status"] == "ok" and meta["supported_cells"] == 3
    assert meta["accuracy_status"] == "UNVALIDATED"
    assert meta["classes"] == CLASSES and meta["confidence_method"] == CONFIDENCE_METHOD
    assert meta["provenance"]["model"] == descriptor([])["model"]
    assert bundle.read_json(result.outputs["metadata"])["semantics"] == meta
    with rasterio.open(bundle.path(result.outputs["semantic"])) as raster:
        assert raster.crs.to_epsg() == 32643 and raster.nodata == 0
        assert raster.dtypes == ("uint16",)
        assert raster.read(1, masked=True).count() == 3
        assert json.loads(raster.tags()["semantic_classes"]) == CLASSES
        assert "uncalibrated" in raster.tags()["confidence_meaning"]
        assert json.loads(raster.tags()["model_provenance"]) == meta["provenance"]["model"]
        label_mask = raster.read(1) != 0
    with rasterio.open(bundle.path(result.outputs["semantic_confidence"])) as raster:
        assert raster.dtypes == ("float32",) and raster.nodata == -9999.
        np.testing.assert_array_equal(raster.dataset_mask() > 0, label_mask)
        np.testing.assert_allclose(raster.read(1)[label_mask], .9)
        np.testing.assert_array_equal(raster.read(1)[~label_mask], -9999.)
    for path in result.outputs.values():
        assert not path.startswith("/") and bundle.path(path).exists()


@pytest.mark.parametrize("state", ["absent", "disabled", "unavailable", "invalid", "zero_labels"])
def test_s9_optional_absence_errors_and_stale_files_preserve_mandatory_products(tmp_path, state):
    bundle, cfg = stage_fixture(tmp_path, semantics=state != "absent")
    if state != "absent":
        desc = bundle.read_json("s3_masking/semantics.json")
        if state in {"disabled", "unavailable"}:
            desc.update(status=state, reason="constructed optional status", frames=[], model=None)
        elif state == "invalid":
            desc["classes"][0]["id"] = 0
        else:
            labels = np.zeros((21, 21), np.uint16)
            frame = maps(tmp_path, labels=labels, probability=np.zeros((21, 21), np.float32))
            desc["frames"] = [frame]
        bundle.write_json("s3_masking/semantics.json", desc)
    directory = bundle.stage_dir("s9_geo")
    for filename in ("semantic.tif", "semantic_confidence.tif", "semantic_preview.png"):
        (directory / filename).write_bytes(b"previous invocation")
    result = S9Geo().run(bundle, cfg, None)
    assert {"dsm", "dtm", "ortho", "confidence"} <= result.outputs.keys()
    assert result.degraded_reason is None and result.metrics["status"] == "ok"
    assert not {"semantic", "semantic_confidence", "semantic_preview"}.intersection(result.outputs)
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    assert set(meta) == METADATA_KEYS
    assert meta["status"] == ("disabled" if state == "disabled" else "unavailable")
    assert meta["raster_path"] is None and meta["confidence_path"] is None and meta["preview"] is None
    assert meta["supported_cells"] == 0 and meta["reason"]
    assert not (directory / "semantic.tif").exists()
    assert not (directory / "semantic_confidence.tif").exists()
    assert not (directory / "semantic_preview.png").exists()


def test_s9_without_geometry_still_emits_semantic_status_descriptor(tmp_path):
    bundle = Bundle(tmp_path, {"bundle_version": "1", "stages": {}, "inputs": [],
                               "config_snapshot": {}})
    cfg = SimpleNamespace(semantics=SimpleNamespace(classes=[]))
    result = S9Geo().run(bundle, cfg, None)
    assert set(result.outputs) == {"metadata", "semantic_metadata"}
    assert result.degraded_reason
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    assert meta["status"] == "unavailable" and set(meta) == METADATA_KEYS


def test_s9_broken_declared_dynamic_masks_do_not_silently_establish_semantic_support(tmp_path):
    bundle, cfg = stage_fixture(tmp_path)
    bundle.write_json("s3_masking/masks.json", {"frames": [{"frame_id": "frame"}]})
    result = S9Geo().run(bundle, cfg, None)
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    assert {"dsm", "dtm", "ortho"} <= result.outputs.keys()
    assert result.degraded_reason is None
    assert meta["status"] == "unavailable" and "dynamic mask descriptor" in meta["reason"]
    assert "semantic" not in result.outputs


@pytest.mark.parametrize("mask_frames", [[], [{"frame_id": "other", "path": "missing.png"}],
                                       [{"frame_id": "frame", "path": "s3_masking/mask.png"}] * 2])
def test_s9_missing_or_duplicate_mask_records_cannot_label_observed_cells(tmp_path, mask_frames):
    bundle, cfg = stage_fixture(tmp_path)
    bundle.write_json("s3_masking/masks.json", {"frames": mask_frames})
    result = S9Geo().run(bundle, cfg, None)
    assert {"dsm", "dtm"} <= result.outputs.keys()
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    assert meta["status"] == "unavailable" and meta["reason"]
    assert "semantic" not in result.outputs


def test_s9_uses_final_pose_image_and_records_full_projection_provenance(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    bundle, cfg = stage_fixture(tmp_path)
    # An earlier camera revision sees the site; the final camera points elsewhere.
    poses = bundle.read_json("s6_global/poses.json")
    bundle.write_json("s2_poses/poses.json", poses)
    bundle.manifest["stages"]["s2_poses"] = {"outputs": {"poses": "s2_poses/poses.json"}}
    poses["cameras"][0]["center"] = [10000., .5, 10.]
    bundle.write_json("s6_global/poses.json", poses)
    result = S9Geo().run(bundle, cfg, None)
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    assert meta["status"] == "unavailable" and meta["supported_cells"] == 0
    assert "semantic" not in result.outputs
    poses["cameras"][0]["center"] = [.5, .5, 10.]
    bundle.write_json("s6_global/poses.json", poses)
    result = S9Geo().run(bundle, cfg, None)
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    with rasterio.open(bundle.path(result.outputs["semantic"])) as raster:
        assert json.loads(raster.tags()["semantic_provenance"]) == meta["provenance"]


def test_s9_real_antimeridian_preview_failure_preserves_semantic_and_mandatory_rasters(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    bundle, cfg = stage_fixture(tmp_path)
    poses = bundle.read_json("s6_global/poses.json")
    x, y = pyproj.Transformer.from_crs(4326, 32660, always_xy=True).transform(179.999999, 1.)
    poses.update(origin=[x, y, 0.], crs={"epsg": 32660, "vertical": "unknown"})
    bundle.write_json("s6_global/poses.json", poses)
    result = S9Geo().run(bundle, cfg, None)
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    assert {"dsm", "dtm", "ortho", "semantic", "semantic_confidence"} <= result.outputs.keys()
    assert result.degraded_reason is None and meta["status"] == "ok"
    assert meta["preview"] is None and "antimeridian" in meta["reason"]
    assert "semantic_preview" not in result.outputs


@pytest.mark.parametrize("failed", ["semantic", "semantic_confidence", "semantic_preview"])
def test_s9_optional_writers_fail_individually(tmp_path, monkeypatch, failed):
    from drishti.geo import semantics

    bundle, cfg = stage_fixture(tmp_path)
    real_write = semantics.write_geotiff

    def write(path, *args, **kwargs):
        if path.stem == failed:
            path.write_bytes(b"partial invalid output")
            path.with_suffix(path.suffix + ".msk").write_bytes(b"partial external mask")
            raise RuntimeError("constructed optional writer failure")
        return real_write(path, *args, **kwargs)

    def preview(path, *args, **kwargs):
        path.write_bytes(b"partial invalid preview")
        raise ValueError("constructed antimeridian failure")

    monkeypatch.setattr(semantics, "write_geotiff", write)
    if failed == "semantic_preview":
        monkeypatch.setattr(semantics, "geographic_preview", preview)
    result = S9Geo().run(bundle, cfg, None)
    meta = bundle.read_json(result.outputs["semantic_metadata"])
    assert {"dsm", "dtm", "ortho"} <= result.outputs.keys() and result.degraded_reason is None
    assert failed not in result.outputs and failed in meta["reason"]
    assert not (bundle.root / "s9_geo" / (failed + (".png" if failed == "semantic_preview" else ".tif"))).exists()
    assert not (bundle.root / "s9_geo" / f"{failed}.tif.msk").exists()
    if failed == "semantic":
        assert meta["status"] == "unavailable" and "semantic_confidence" in result.outputs
        assert "semantic_preview" not in result.outputs
    else:
        assert meta["status"] == "ok" and "semantic" in result.outputs


@pytest.mark.parametrize("failed", ["semantic", "semantic_confidence", "confidence"])
def test_s10_optional_product_failures_do_not_fail_mandatory_geotiffs(tmp_path, monkeypatch, failed):
    from drishti.io import exporters

    bundle, cfg = stage_fixture(tmp_path)
    geo = S9Geo().run(bundle, cfg, None)
    bundle.manifest["stages"]["s9_geo"] = {"outputs": geo.outputs}
    real_copy = exporters.copy_geotiffs

    def copy_products(sources, *args, **kwargs):
        if failed in sources:
            raise RuntimeError("constructed optional copy failure")
        return real_copy(sources, *args, **kwargs)

    monkeypatch.setattr(exporters, "copy_geotiffs", copy_products)
    result = S10Export().run(bundle, cfg, None)
    records = bundle.read_json(result.outputs["exports"])
    assert records["formats"]["geotiff"]["status"] == "ok"
    assert "geotiff" not in records["missing_mandatory"]
    assert records["optional_products"][f"geotiff_{failed}"]["status"] == "failed"
    assert records["optional_products"][f"geotiff_{failed}"]["path"] is None
    assert f"geotiff_{failed}" not in result.outputs
    assert {"geotiff_dsm", "geotiff_dtm", "geotiff_ortho"} <= result.outputs.keys()
    other_semantic = "semantic_confidence" if failed == "semantic" else "semantic"
    assert records["optional_products"][f"geotiff_{other_semantic}"]["status"] == "ok"
    assert failed not in records["formats"]["geotiff"]["details"]["products"]
    for path in result.outputs.values():
        assert not path.startswith("/") and bundle.path(path).exists()


def test_s10_copies_semantic_tags_and_never_registers_stale_absent_products(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    bundle, cfg = stage_fixture(tmp_path)
    geo = S9Geo().run(bundle, cfg, None)
    bundle.manifest["stages"]["s9_geo"] = {"outputs": geo.outputs}
    first = S10Export().run(bundle, cfg, None)
    with rasterio.open(bundle.path(first.outputs["geotiff_semantic"])) as copied:
        with rasterio.open(bundle.path(geo.outputs["semantic"])) as original:
            assert copied.tags() == original.tags()
            np.testing.assert_array_equal(copied.read(1), original.read(1))
            np.testing.assert_array_equal(copied.dataset_mask(), original.dataset_mask())
    for key in ("semantic", "semantic_confidence"):
        bundle.manifest["stages"]["s9_geo"]["outputs"].pop(key)
    second = S10Export().run(bundle, cfg, None)
    records = bundle.read_json(second.outputs["exports"])
    for key in ("semantic", "semantic_confidence"):
        assert f"geotiff_{key}" not in second.outputs
        assert records["optional_products"][f"geotiff_{key}"]["status"] == "unavailable"
        assert not (bundle.root / "s10_export/geotiff" / f"{key}.tif").exists()
    assert records["formats"]["geotiff"]["status"] == "ok"
