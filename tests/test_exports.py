"""Constructed S10 per-product failure/provenance checks, not mission acceptance."""

import json

import numpy as np
import pytest
from test_semantic_products import CLASSES, CONFIDENCE_METHOD, stage_fixture

from drishti.stages.s9_geo import S9Geo
from drishti.stages.s10_export import S10Export


def geo_bundle(tmp_path):
    bundle, cfg = stage_fixture(tmp_path)
    geo = S9Geo().run(bundle, cfg, None)
    bundle.manifest["stages"]["s9_geo"] = {"outputs": geo.outputs}
    return bundle, cfg, geo


def test_s10_optional_records_preserve_categorical_mapping_and_projection_provenance(tmp_path):
    bundle, cfg, geo = geo_bundle(tmp_path)
    result = S10Export().run(bundle, cfg, None)
    records = bundle.read_json(result.outputs["exports"])
    original = bundle.read_json(geo.outputs["semantic_metadata"])
    for key in ("semantic", "semantic_confidence"):
        record = records["optional_products"][f"geotiff_{key}"]
        assert record["status"] == "ok" and record["reason"] is None
        assert record["details"]["classes"] == CLASSES
        assert record["details"]["confidence_method"] == CONFIDENCE_METHOD
        assert record["details"]["provenance"] == original["provenance"]
        assert record["details"]["model"] == original["provenance"]["model"]


@pytest.mark.parametrize("corruption", ["mapping", "confidence_method", "nodata", "unmapped_label"])
def test_s10_invalid_semantic_storage_or_provenance_does_not_fail_required_geotiffs(tmp_path, corruption):
    rasterio = pytest.importorskip("rasterio")
    bundle, cfg, geo = geo_bundle(tmp_path)
    with rasterio.open(bundle.path(geo.outputs["semantic"]), "r+") as raster:
        if corruption == "mapping":
            raster.update_tags(semantic_classes=json.dumps([CLASSES[0], CLASSES[0]]))
        elif corruption == "confidence_method":
            raster.update_tags(confidence_method=json.dumps("geometry_confidence"))
        elif corruption == "nodata":
            raster.nodata = 65535
        else:
            values = raster.read(1)
            values[raster.dataset_mask() > 0] = 65535
            raster.write(values, 1)
    result = S10Export().run(bundle, cfg, None)
    records = bundle.read_json(result.outputs["exports"])
    assert records["formats"]["geotiff"]["status"] == "ok"
    record = records["optional_products"]["geotiff_semantic"]
    assert record["status"] == "failed" and record["path"] is None and record["reason"]
    assert "geotiff_semantic" not in result.outputs
    assert records["optional_products"]["geotiff_semantic_confidence"]["status"] == "ok"
    assert not bundle.path("s10_export/geotiff/semantic.tif").exists()


def test_s10_preserves_external_gdal_validity_mask_in_registered_assets(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    bundle, cfg, geo = geo_bundle(tmp_path)
    source = bundle.path(geo.outputs["dsm"])
    with rasterio.open(source) as original:
        profile, data, tags = original.profile, original.read(), original.tags()
    valid = np.zeros(data.shape[1:], dtype=np.uint8)
    valid[0, 0] = 255
    source.unlink()
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(source, "w", **profile) as original:
            original.write(data)
            original.update_tags(**tags)
            original.write_mask(valid)
    assert source.with_suffix(".tif.msk").exists()
    result = S10Export().run(bundle, cfg, None)
    with rasterio.open(bundle.path(result.outputs["geotiff_dsm"])) as copied:
        np.testing.assert_array_equal(copied.dataset_mask(), valid)
    assets = bundle.path(result.outputs["geotiff_assets"])
    assert (assets / "dsm.tif.msk").read_bytes() == source.with_suffix(".tif.msk").read_bytes()


def test_s10_required_product_failure_keeps_other_geo_and_semantic_copies(tmp_path):
    bundle, cfg, geo = geo_bundle(tmp_path)
    bundle.path(geo.outputs["dtm"]).write_bytes(b"constructed invalid raster")
    result = S10Export().run(bundle, cfg, None)
    records = bundle.read_json(result.outputs["exports"])
    assert records["formats"]["geotiff"]["status"] == "failed"
    assert "geotiff" in records["missing_mandatory"]
    assert {"geotiff_dsm", "geotiff_ortho", "geotiff_semantic"} <= result.outputs.keys()
    assert "dtm" not in records["formats"]["geotiff"]["details"]["products"]
    assert records["optional_products"]["geotiff_semantic"]["status"] == "ok"


def test_s10_missing_final_georeference_still_records_each_optional_geo_failure(tmp_path):
    bundle, cfg, _ = geo_bundle(tmp_path)
    poses = bundle.read_json("s6_global/poses.json")
    poses["crs"] = {"epsg": 4326, "vertical": "unknown"}
    bundle.write_json("s6_global/poses.json", poses)
    result = S10Export().run(bundle, cfg, None)
    records = bundle.read_json(result.outputs["exports"])
    assert records["formats"]["geotiff"]["status"] == "unavailable"
    for key in ("confidence", "support", "mask", "semantic", "semantic_confidence"):
        record = records["optional_products"][f"geotiff_{key}"]
        assert record["status"] == "unavailable" and record["path"] is None
        assert "projected CRS" in record["reason"]
