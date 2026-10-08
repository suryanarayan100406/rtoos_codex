"""Constructed raster/visibility boundary cases, not real mission acceptance."""

import numpy as np
import pytest
from test_semantic_products import camera, maps, products, stage_fixture

from drishti.geo.products import (
    Grid,
    georeference,
    mesh_zbuffer,
    orthoproject,
    write_geotiff,
)
from drishti.stages.s9_geo import S9Geo


@pytest.mark.parametrize("invalid", [None, []])
def test_s9_malformed_final_pose_descriptor_still_records_optional_status(tmp_path, invalid):
    from types import SimpleNamespace

    from drishti.bundle.core import Bundle

    bundle = Bundle(tmp_path, {"bundle_version": "1", "stages": {}, "inputs": [],
                               "config_snapshot": {}})
    bundle.write_json("s6_global/poses.json", invalid)
    bundle.manifest["stages"]["s6_global"] = {"outputs": {"poses": "s6_global/poses.json"}}
    result = S9Geo().run(bundle, SimpleNamespace(), None)
    assert set(result.outputs) == {"metadata", "semantic_metadata"}
    assert "Final poses descriptor must be an object" in result.degraded_reason
    assert result.metrics["semantics"]["status"] == "unavailable"


def test_raster_mask_cannot_be_silently_resampled_or_overflow_uint8(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    ref = georeference({"frame": "projected_local", "origin": [500000., 3000000., 0.],
                        "crs": {"epsg": 32643, "vertical": "unknown"}})
    grid = Grid(0., 2., 1., 2, 2)
    path = tmp_path / "raster.tif"
    with pytest.raises(ValueError, match="validity mask shape"):
        write_geotiff(path, np.ones(grid.shape), grid, ref, nodata=-9999., valid=[[True]])
    assert not path.exists()
    valid = np.array([[256, 0], [0, 256]], dtype=np.uint16)
    write_geotiff(path, np.ones(grid.shape), grid, ref, nodata=-9999., valid=valid)
    with rasterio.open(path) as raster:
        np.testing.assert_array_equal(raster.dataset_mask(), [[255, 0], [0, 255]])


@pytest.mark.parametrize("field", ["source_datum", "source_altitude_datum", "altitude_datum"])
def test_source_vertical_datum_alias_is_preserved_in_raster_provenance(tmp_path, field):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    ref = georeference({"frame": "projected_local", "origin": [500000., 3000000., 90.],
                        "crs": {"epsg": 32643, "vertical": "source", field: "orthometric"}})
    assert ref["vertical"] == ref["source_datum"] == "orthometric"
    assert ref["vertical_conversion"] is None
    path = tmp_path / "dsm.tif"
    write_geotiff(path, np.array([[92.]]), Grid(0., 1., 1., 1, 1), ref, nodata=-9999.)
    with rasterio.open(path) as raster:
        assert raster.tags()["source_datum"] == "orthometric"
        assert raster.tags()["vertical_datum"] == "orthometric"
        assert raster.tags()["vertical_conversion"] == "null"


def test_relative_datum_cannot_be_attached_to_absolute_geographic_products():
    pytest.importorskip("pyproj")
    with pytest.raises(ValueError, match="absolute altitude"):
        georeference({"frame": "projected_local", "origin": [500000., 3000000., 0.],
                      "crs": {"epsg": 32643, "vertical": "source", "altitude_datum": "relative"}})


@pytest.mark.parametrize("corruption", ["negative_face", "fractional_face", "inferred_shape"])
def test_visibility_mesh_rejects_ambiguous_indices_and_inferred_provenance(corruption):
    pytest.importorskip("cv2")
    mesh = {"vertices": np.array([[-2., -2., 5.], [3., -2., 5.], [.5, 3., 5.]]),
            "faces": np.array([[0, 1, 2]]), "inferred": np.zeros(3, bool)}
    if corruption == "negative_face":
        mesh["faces"][0, 0] = -1
    elif corruption == "fractional_face":
        mesh["faces"] = np.array([[.5, 1., 2.]])
    else:
        mesh["inferred"] = np.zeros(2, bool)
    with pytest.raises(ValueError, match="face indices|inferred flags"):
        mesh_zbuffer(mesh, camera(), (21, 21))


def test_rgb_projection_also_rejects_finite_unsupported_and_inferred_dsm_cells(tmp_path):
    pytest.importorskip("cv2")
    maps(tmp_path)
    surface = products(Grid(0., 1., 1., 3, 1), support=[[1, 0, 1]], mask=[[1, 1, 2]])
    rgba, metadata = orthoproject(surface, [camera()], tmp_path, chunk_size=1)
    np.testing.assert_array_equal(rgba[..., 3], [[255, 0, 0]])
    assert metadata["textured_cells"] == 1


@pytest.mark.parametrize("failed", ["dsm", "confidence", "support"])
def test_s9_failed_raster_writer_preserves_other_required_and_semantic_products(tmp_path, monkeypatch, failed):
    from drishti.geo import products as geo_products

    bundle, cfg = stage_fixture(tmp_path)
    real_write = geo_products.write_geotiff

    def write(path, *args, **kwargs):
        if path.stem == failed:
            path.write_bytes(b"partial invalid output")
            path.with_suffix(".tif.msk").write_bytes(b"partial mask")
            raise RuntimeError("constructed isolated raster failure")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(geo_products, "write_geotiff", write)
    result = S9Geo().run(bundle, cfg, None)
    assert {"dtm", "ortho", "semantic", "semantic_confidence"} <= result.outputs.keys()
    assert failed not in result.outputs
    assert result.metrics["products"][failed]["status"] == "failed"
    assert result.metrics["products"][failed]["path"] is None
    assert "constructed isolated raster failure" in result.metrics["products"][failed]["reason"]
    assert not (bundle.root / "s9_geo" / f"{failed}.tif").exists()
    assert not (bundle.root / "s9_geo" / f"{failed}.tif.msk").exists()


def test_s9_corrupt_mesh_visibility_falls_back_to_real_dsm_visibility(tmp_path):
    bundle, cfg = stage_fixture(tmp_path)
    np.savez(bundle.path("s8_mesh/mesh.npz"), vertices=np.zeros((3, 3)),
             faces=np.array([[-1, 1, 2]]), inferred=np.zeros(3, bool))
    result = S9Geo().run(bundle, cfg, None)
    assert {"dsm", "dtm", "ortho", "semantic"} <= result.outputs.keys()
    assert "Mesh visibility unavailable" in result.degraded_reason
    projection = result.metrics["semantics"]["provenance"]["projection"]
    assert projection["visibility_geometry"] == "DSM_samples"
