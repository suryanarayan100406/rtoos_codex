"""Isolated coordinate, filtering, external reduction, and artifact unit tests."""

from types import SimpleNamespace

import numpy as np

from drishti.recon.fusion import OBSERVATION, TiledFusion, depth_observations
from drishti.recon.meshing import observed_support


def dense_config():
    return SimpleNamespace(pixel_stride=1, max_points_per_chunk=16, max_depth_m=100,
                           min_confidence=0.2, normal_max_relative_jump=0.1)


def test_fusion_transform_exclusion_and_normals_face_actual_camera():
    camera = {"K": [[20, 0, 4], [0, 20, 4], [0, 0, 1]], "R_cw": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
              "center": [100.125, -50.125, 8.125], "distortion": []}
    depth = np.full((9, 9), 4., np.float32)
    confidence = np.ones((9, 9), np.float32)
    excluded = np.zeros((9, 9), bool)
    excluded[3, 3] = True
    depth[5, 5] = np.nan
    confidence[4, 4] = 0.1
    rgb = np.full((9, 9, 3), 127, np.uint8)
    batches = list(depth_observations(depth, confidence, rgb, excluded, camera, dense_config()))
    assert batches and max(map(len, batches)) <= 16
    records = np.concatenate(batches)
    assert records["point"].dtype == np.float64
    np.testing.assert_allclose(records["point"][:, 2], 12.125)
    view = np.array(camera["center"])-records["point"]
    assert np.all(np.einsum("ij,ij->i", records["normal"], view) > 0)
    np.testing.assert_allclose(records["view"], view/np.linalg.norm(view, axis=1)[:, None])
    assert np.all(records["confidence"] >= 0.2)
    assert len(records) < 49


def test_large_tile_external_reduction_and_real_ply_attributes(tmp_path, monkeypatch):
    fusion = TiledFusion(tmp_path/"work", 1., 100., 16)
    records = np.zeros(16, OBSERVATION)
    records["point"][:8] = [-0.2, 0.1, 1.2]
    records["point"][8:] = [0.2, 0.1, 1.2]
    records["normal"] = [0, 0, -1]
    records["view"] = [0, 0, -1]
    records["color"] = [20, 40, 60]
    records["confidence"][:8] = 0.5
    records["confidence"][8:] = 1
    records["inferred"][1] = True
    original = np.fromfile
    reads = []

    def bounded_read(*args, **kwargs):
        reads.append(kwargs.get("count", -1))
        assert 0 < kwargs.get("count", -1) <= 4
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", bounded_read)
    for _ in range(40):
        fusion.add(records)
    output = tmp_path/"output"
    output.mkdir()
    arrays, metadata = fusion.finish(output)
    assert metadata["points"] == 2
    assert metadata["observations"] == 640
    assert len(reads) > 40
    np.testing.assert_array_equal(arrays["observations"], [320, 320])
    np.testing.assert_array_equal(arrays["inferred"], [True, False])
    np.testing.assert_allclose(arrays["points"], [[-0.2, 0.1, 1.2], [0.2, 0.1, 1.2]])
    np.testing.assert_allclose(arrays["confidence"], [0.5, 1])
    assert np.all(np.einsum("ij,ij->i", arrays["normals"], arrays["view_direction"]) > 0)
    with np.load(output/"dense.npz", allow_pickle=False) as npz:
        assert npz["points"].dtype == np.float64
        assert {"confidence", "inferred", "normals", "observations"}.issubset(npz.files)
    ply = (output/"dense.ply").read_bytes()
    header, binary = ply.split(b"end_header\n", 1)
    assert b"property float confidence" in header and b"property uchar inferred" in header
    assert b"property float nx" in header and b"property double x" in header
    dtype = np.dtype([("xyz", "<f8", (3,)), ("rgb", "u1", (3,)), ("normal", "<f4", (3,)),
                      ("confidence", "<f4"), ("inferred", "u1"), ("observations", "<u4")])
    stored = np.frombuffer(binary, dtype)
    np.testing.assert_array_equal(stored["inferred"], [1, 0])
    np.testing.assert_allclose(stored["confidence"], [0.5, 1])


def test_external_merge_combines_weighted_points_across_many_chunks(tmp_path):
    fusion = TiledFusion(tmp_path/"work", 1., 10., 8)
    for step in range(30):
        records = np.zeros(8, OBSERVATION)
        records["point"] = [0.1+step/100, 0.1, 0.1]
        records["color"] = [10+step, 50, 90]
        records["normal"] = [0, 0, -1]
        records["view"] = [0, 0, -1]
        records["confidence"] = (step+1)/30
        fusion.add(records)
    output = tmp_path/"output"
    output.mkdir()
    arrays, _ = fusion.finish(output)
    weight = np.arange(1, 31)/30
    expected_x = np.sum((0.1+np.arange(30)/100)*weight)/np.sum(weight)
    assert len(arrays["points"]) == 1
    np.testing.assert_allclose(arrays["points"][0, 0], expected_x, rtol=1e-7)
    np.testing.assert_allclose(arrays["confidence"][0], np.sum(weight**2)/np.sum(weight), rtol=1e-7)
    assert arrays["observations"][0] == 240


def test_inferred_mesh_flag_uses_full_observed_support_distances():
    arrays = {"points": np.array([[0, 0, 0], [5, 0, 0]], np.float64),
              "confidence": np.array([0.8, 0.5]), "inferred": np.array([False, True])}
    confidence, inferred, distance = observed_support(np.array([[0.1, 0, 0], [3, 0, 0], [5, 0, 0]]), arrays, 0.5, 1)
    np.testing.assert_allclose(distance, [0.1, 2, 0])
    np.testing.assert_array_equal(inferred, [False, True, True])
    assert confidence[0] > confidence[1]


def test_no_supported_observations_preserves_an_honest_empty_cloud(tmp_path):
    fusion = TiledFusion(tmp_path/"work", 1., 10., 8)
    output = tmp_path/"output"
    output.mkdir()
    arrays, metadata = fusion.finish(output)
    assert metadata["points"] == metadata["observations"] == 0
    assert arrays["points"].shape == (0, 3)
    with np.load(output/"dense.npz", allow_pickle=False) as cloud:
        assert not len(cloud["confidence"])
        assert not len(cloud["inferred"])
