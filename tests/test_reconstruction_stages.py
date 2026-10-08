"""Constructed artifact-contract regression units, not real-data acceptance.

Only neural prediction/Poisson are isolated where indicated. File IO, camera
geometry, depth fitting, tiled fusion, and atlas baking use their actual code.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from drishti.bundle.core import Bundle
from drishti.config.models import Config
from drishti.errors import InputError
from drishti.recon.depth import pixel_rays, project_points
from drishti.stages import s4_depth, s7_dense, s8_mesh


def sparse_bundle(tmp_path):
    bundle = Bundle(tmp_path, {"bundle_version": "1", "stages": {}, "inputs": [], "config_snapshot": {}})
    for name in ("s2_poses", "s6_global"):
        folder = bundle.stage_dir(name)
        np.savez(folder/"sparse.npz", points=np.array([[0., 0., 1.]]))
        bundle.manifest["stages"][name] = {"outputs": {"sparse": f"{name}/sparse.npz"}}
    return bundle


def test_pose_declared_sparse_revision_has_priority(tmp_path):
    bundle = sparse_bundle(tmp_path)
    path, source, warning = s4_depth.pose_sparse_source(bundle, {"sparse_path": "s6_global/sparse.npz"})
    assert path == bundle.artifact("s6_global", "sparse")
    assert source == "final_pose_declared_sparse" and warning is None


def test_recorded_s6_sparse_when_pose_path_not_declared(tmp_path):
    bundle = sparse_bundle(tmp_path)
    path, source, warning = s4_depth.pose_sparse_source(bundle, {})
    assert path == bundle.artifact("s6_global", "sparse")
    assert source == "s6_recorded_sparse" and warning is None


def test_missing_final_sparse_does_not_silently_use_s2(tmp_path):
    bundle = sparse_bundle(tmp_path)
    with pytest.raises(InputError, match="missing"):
        s4_depth.pose_sparse_source(bundle, {"sparse_path": "s6_global/missing.npz"})
    bundle.artifact("s6_global", "sparse").unlink()
    with pytest.raises(InputError, match="missing"):
        s4_depth.pose_sparse_source(bundle, {})


def test_explicit_legacy_fallback_and_refined_pose_guard(tmp_path):
    bundle = sparse_bundle(tmp_path)
    bundle.manifest["stages"]["s6_global"]["outputs"].pop("sparse")
    path, source, warning = s4_depth.pose_sparse_source(bundle, {})
    assert path == bundle.artifact("s2_poses", "sparse")
    assert source == "legacy_s2_sparse_fallback" and warning
    with pytest.raises(InputError, match="matching sparse"):
        s4_depth.pose_sparse_source(bundle, {"pose_revision": "s6_global_visual_GNSS_refined"})


def test_declared_sparse_path_is_bundle_relative(tmp_path):
    bundle = sparse_bundle(tmp_path)
    with pytest.raises(InputError, match="confined"):
        s4_depth.pose_sparse_source(bundle, {"sparse_path": "../other/sparse.npz"})


def depth_stage_bundle(tmp_path, monkeypatch):
    bundle = sparse_bundle(tmp_path)
    camera = {"frame_id": "frame0", "image_path": "s6_global/frame0.png",
              "K": [[50, 0, 20], [0, 50, 15], [0, 0, 1]], "R_cw": np.eye(3).tolist(),
              "center": [1., -2., 3.], "width": 40, "height": 30,
              "distortion": [], "projection_domain": "undistorted_pinhole"}
    y, x = np.indices((30, 40))
    prediction = (1+x/20+y/40).astype(np.float32)
    uv = np.array([(x, y) for y in range(4, 25, 6) for x in range(4, 36, 6)])
    z = 1/(0.08*prediction[uv[:, 1], uv[:, 0]]+0.01)
    rays = pixel_rays(uv, camera)
    camera["observations"] = [{"point_index": i, "uv": point.tolist()} for i, point in enumerate(uv)]
    for stage, scale in (("s2_poses", 2.), ("s6_global", 1.)):
        points = rays*z[:, None]*scale+camera["center"]
        np.savez(bundle.root/stage/"sparse.npz", points=points.astype(np.float64),
                 confidence=np.full(len(points), 0.8), inferred=np.zeros(len(points), bool))
    image = np.stack((x*5, y*7, np.full_like(x, 100)), axis=-1).astype(np.uint8)
    Image.fromarray(image).save(bundle.root/camera["image_path"])
    poses = {"cameras": [camera], "frame": "projected_local", "origin": [1000., 2000., 40.],
             "crs": {"epsg": 32643}, "sparse_path": "s6_global/sparse.npz",
             "pose_revision": "s6_global_visual_GNSS_refined"}
    bundle.write_json("s6_global/poses.json", poses)
    bundle.manifest["stages"]["s6_global"]["outputs"]["poses"] = "s6_global/poses.json"
    folder = bundle.stage_dir("s3_masking")
    mask = np.zeros((30, 40), np.uint8)
    mask[:2] = 255
    Image.fromarray(mask).save(folder/"frame0.png")
    bundle.write_json("s3_masking/masks.json", {"frames": [{"frame_id": "frame0", "path": "s3_masking/frame0.png"}]})
    bundle.manifest["stages"]["s3_masking"] = {"outputs": {"masks": "s3_masking/masks.json"}}

    class UnitPrediction:
        device = "cpu"

        def __init__(self, *args):
            pass

        def predict(self, rgb):
            assert rgb.shape == image.shape
            return prediction.copy()

    monkeypatch.setattr(s4_depth, "resolve_model", lambda *args: {"name": "isolated_unit_prediction"})
    monkeypatch.setattr(s4_depth, "LocalDepth", UnitPrediction)
    return bundle, camera, prediction, Config()


def test_s4_fits_retriangulated_s6_points_and_s7_consumes_final_image_domain(tmp_path, monkeypatch):
    bundle, camera, prediction, cfg = depth_stage_bundle(tmp_path, monkeypatch)
    runtime = SimpleNamespace(device="cpu")
    result = s4_depth.STAGE.run(bundle, cfg, runtime)
    assert result.degraded_reason is None
    bundle.manifest["stages"]["s4_depth"] = {"outputs": result.outputs}
    index = bundle.read_json(result.outputs["depth"])
    entry = index["frames"][0]
    assert entry["metric"] and entry["image_domain"] == "undistorted"
    assert entry["sparse_path"] == index["sparse_path"] == "s6_global/sparse.npz"
    assert entry["anchor_method"] == "final_pose_declared_sparse_tracks_reprojected_with_pixel_z_buffer"
    np.testing.assert_allclose([entry["a"], entry["b"]], [0.08, 0.01], atol=1e-6)
    depth = np.load(bundle.root/entry["path"])
    np.testing.assert_allclose(depth[2:], 1/(0.08*prediction[2:]+0.01), rtol=1e-6)
    assert not np.any(depth[:2])
    # A non-metric map must be skipped before even attempting to open its path.
    index["frames"].append({"frame_id": "unaligned", "metric": False, "path": "must_not_be_opened.npy"})
    bundle.write_json(result.outputs["depth"], index)
    cfg = cfg.model_copy(update={"dense": cfg.dense.model_copy(update={"pixel_stride": 2, "max_points_per_chunk": 32})})
    fused = s7_dense.STAGE.run(bundle, cfg, runtime)
    metadata = bundle.read_json(fused.outputs["metadata"])
    assert metadata["points"] > 0 and metadata["fused_frames"] == 1
    assert metadata["skipped_frames"][0]["frame_id"] == "unaligned"
    arrays = {name: np.load(bundle.root/path, mmap_mode="r") for name, path in metadata["arrays"].items()}
    assert {"normals", "view_direction"}.issubset(arrays)
    assert arrays["points"].dtype == np.float64
    assert np.all(np.einsum("ij,ij->i", arrays["normals"], arrays["view_direction"]) > 0)
    projected, metric_z = project_points(arrays["points"], camera)
    expected = 1/(0.08*(1+projected[:, 0]/20+projected[:, 1]/40)+0.01)
    np.testing.assert_allclose(metric_z, expected, rtol=0.02)


def test_s4_does_not_publish_loaded_only_depth_model_as_executed(tmp_path, monkeypatch):
    bundle, _, _, cfg = depth_stage_bundle(tmp_path, monkeypatch)
    Image.fromarray(np.full((30, 40), 255, np.uint8)).save(bundle.root / "s3_masking/frame0.png")
    result = s4_depth.STAGE.run(bundle, cfg, SimpleNamespace(device="cpu"))
    assert result.models == []
    assert result.metrics["metric_frames"] == 0


def test_s7_does_not_count_fully_filtered_maps_as_fused_frames(tmp_path, monkeypatch):
    bundle, _, _, cfg = depth_stage_bundle(tmp_path, monkeypatch)
    result = s4_depth.STAGE.run(bundle, cfg, SimpleNamespace(device="cpu"))
    bundle.manifest["stages"]["s4_depth"] = {"outputs": result.outputs}
    entry = bundle.read_json(result.outputs["depth"])["frames"][0]
    np.save(bundle.root/entry["confidence_path"], np.zeros((30, 40), np.float32))
    fused = s7_dense.STAGE.run(bundle, cfg, SimpleNamespace(device="cpu"))
    metadata = bundle.read_json(fused.outputs["metadata"])
    assert metadata["points"] == metadata["fused_frames"] == 0
    assert metadata["processed_metric_frames"] == 1
    assert metadata["pose_revision"] == "s6_global_visual_GNSS_refined"
    assert "filters" in metadata["skipped_frames"][0]["reason"]
    assert fused.confidence is None and fused.degraded_reason


@pytest.mark.parametrize("failure", ["texture_write", "publication"])
def test_s8_texture_write_failure_preserves_mesh_artifact_and_metadata(tmp_path, monkeypatch, failure):
    bundle, camera, _, cfg = depth_stage_bundle(tmp_path, monkeypatch)
    folder = bundle.stage_dir("s7_dense")
    np.savez(folder/"dense.npz", points=np.zeros((0, 3), np.float64))
    np.save(folder/"points.npy", np.zeros((0, 3), np.float64))
    bundle.write_json("s7_dense/dense.json", {"arrays": {"points": "s7_dense/points.npy"}})
    bundle.manifest["stages"]["s7_dense"] = {"outputs": {"cloud": "s7_dense/dense.npz", "metadata": "s7_dense/dense.json"}}
    vertices = np.array([[-0.6, -0.6, 3], [-0.6, 0.6, 3], [0.6, -0.6, 3], [0.6, 0.6, 3]])+camera["center"]
    mesh = {"vertices": vertices, "faces": np.array([[0, 1, 2], [2, 1, 3]]),
            "colors": np.full((4, 3), 100, np.uint8), "confidence": np.full(4, 0.8), "inferred": np.zeros(4, bool)}
    monkeypatch.setattr(s8_mesh, "poisson_mesh", lambda *args: (mesh, {"method": "isolated_mesh_unit"}))

    if failure == "texture_write":
        def fail_save(*args, **kwargs):
            raise OSError("unit-injected texture write failure")

        monkeypatch.setattr(Image.Image, "save", fail_save)
    else:
        replace = Path.replace

        def fail_publish(path, destination):
            if path.name == "mesh.npz" and path.parent.name.startswith("texture-"):
                raise OSError("unit-injected publication failure")
            return replace(path, destination)

        monkeypatch.setattr(Path, "replace", fail_publish)
    result = s8_mesh.STAGE.run(bundle, cfg, SimpleNamespace(device="cpu"))
    assert "unit-injected" in result.degraded_reason
    assert "texture" not in result.outputs and "material" not in result.outputs
    metadata = bundle.read_json(result.outputs["metadata"])
    assert metadata["texture_status"] == "failed"
    with np.load(bundle.root/result.outputs["mesh"]) as saved:
        assert metadata["vertices"] == len(saved["vertices"]) == 4
        assert "uv" not in saved.files
    assert "mtllib" not in (bundle.root/result.outputs["obj"]).read_text()
    assert not (bundle.root/"s8_mesh"/"texture.png").exists()
    assert not (bundle.root/"s8_mesh"/"model.mtl").exists()
