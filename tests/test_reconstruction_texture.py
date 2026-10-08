"""Observed-image UV mapping units; these constructed cases are not acceptance data."""

from types import SimpleNamespace

import numpy as np

from drishti.recon.depth import bilinear, project_points
from drishti.recon.texturing import best_view_texture, triangle_pixels, write_material, write_obj


def texture_case():
    vertices = np.array([[-0.6, -0.6, 3], [-0.6, 0.6, 3], [0.6, -0.6, 3]], np.float64)
    faces = np.array([[0, 1, 2]])
    camera = {"K": [[50, 0, 32], [0, 50, 32], [0, 0, 1]], "R_cw": np.eye(3),
              "center": [0, 0, 0], "width": 65, "height": 65, "distortion": []}
    y, x = np.indices((65, 65))
    rgb = np.stack((3*x, 3*y, np.full_like(x, 200)), axis=-1).astype(np.uint8)
    cfg = SimpleNamespace(texture_size=32, texture_patch_px=16, texture_padding_px=2,
                          visibility_long_side=65, texture_occlusion_tolerance_m=0.05)
    return vertices, faces, camera, rgb, cfg


def test_atlas_texels_are_observed_reprojection_and_uv_faces_are_coherent(tmp_path):
    vertices, faces, camera, rgb, cfg = texture_case()
    result = best_view_texture(vertices, faces, [camera], lambda _: rgb,
                               lambda _: np.zeros((65, 65), bool), cfg)
    assert result.metadata["observed_texels"] > 0
    assert result.source_views[0] == 0
    np.testing.assert_array_equal(result.faces, [[0, 1, 2]])
    np.testing.assert_allclose(result.vertices[result.faces[0]], vertices[faces[0]])
    # UV convention maps pixel centres with a bottom-up V axis.
    corner_pixels = np.column_stack((result.uv[:, 0]*32-0.5, (1-result.uv[:, 1])*32-0.5))
    np.testing.assert_allclose(corner_pixels, [[2, 2], [13, 2], [2, 13]])
    xy, bary = triangle_pixels(corner_pixels, 32, 32)
    source_uv, _ = project_points(bary @ vertices, camera)
    expected = np.rint(bilinear(rgb, source_uv)).astype(np.uint8)
    present = result.texture[xy[:, 1], xy[:, 0], 3] > 0
    np.testing.assert_array_equal(result.texture[xy[present, 1], xy[present, 0], :3], expected[present])
    write_material(tmp_path/"model.mtl")
    write_obj(tmp_path/"model.obj", result.vertices, result.faces, result.uv, "model.mtl")
    assert "map_Kd texture.png" in (tmp_path/"model.mtl").read_text()
    assert "f 1/1 2/2 3/3" in (tmp_path/"model.obj").read_text()


def test_dynamic_texture_samples_remain_unobserved_and_inferred_faces_untextured():
    vertices, faces, camera, rgb, cfg = texture_case()
    excluded = np.zeros((65, 65), bool)
    excluded[:, :26] = True
    result = best_view_texture(vertices, faces, [camera], lambda _: rgb, lambda _: excluded, cfg)
    assert np.count_nonzero(result.texture[..., 3]) < 78
    inferred = best_view_texture(vertices, faces, [camera], lambda _: rgb, lambda _: excluded,
                                 cfg, np.array([True, False, False]))
    assert inferred.metadata["observed_texels"] == 0
    assert inferred.source_views[0] == -1


def test_texture_falls_back_to_an_alternate_observed_view():
    vertices, faces, camera, rgb, cfg = texture_case()
    cfg.texture_candidate_views = 2
    blocked = np.ones((65, 65), bool)
    alternate = dict(camera)
    alternate["center"] = [0.2, 0.0, 0.0]
    result = best_view_texture(vertices, faces, [camera, alternate],
                               lambda item: rgb,
                               lambda item: blocked if item is camera else np.zeros((65, 65), bool),
                               cfg)
    assert result.source_views[0] == 1
    assert result.metadata["fallback_view_faces"] == 1
    assert result.metadata["observed_texels"] > 0


def test_texture_uv_seams_duplicate_attributes_by_vertex_source():
    vertices, faces, camera, rgb, cfg = texture_case()
    result = best_view_texture(vertices, np.vstack((faces, faces)), [camera], lambda _: rgb,
                               lambda _: np.zeros((65, 65), bool), cfg)
    assert result.uv.shape == (6, 2)
    np.testing.assert_array_equal(result.vertex_source, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(result.faces, [[0, 1, 2], [3, 4, 5]])
    assert not np.array_equal(result.uv[:3], result.uv[3:])
