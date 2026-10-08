"""Real optional Open3D binding smoke on constructed geometry, NOT dataset acceptance."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from drishti.config.models import MeshConfig
from drishti.recon.meshing import poisson_mesh


def test_real_open3d_poisson_view_normals_support_and_vertex_budget():
    pytest.importorskip("open3d", reason="Real Open3D Poisson runtime is not installed")
    # A deterministic Fibonacci sphere isolates the binding and normal-sign path.
    # Directions are mathematical test vectors, not invented drone telemetry.
    count = 600
    index = np.arange(count, dtype=np.float64)
    height = 1-2*(index+0.5)/count
    azimuth = index*np.pi*(3-np.sqrt(5))
    radial = np.sqrt(1-height**2)
    directions = np.column_stack((radial*np.cos(azimuth), radial*np.sin(azimuth), height))
    center = np.array([12., -7., 4.])
    points = center+directions
    arrays = {"points": points, "normals": -directions, "view_direction": directions,
              "colors": np.full((count, 3), [40, 100, 180], np.uint8),
              "confidence": np.linspace(0.6, 0.9, count, dtype=np.float32),
              "inferred": np.zeros(count, bool)}
    cfg = MeshConfig(poisson_depth=5, normal_neighbors=20, density_quantile=0.05,
                     max_input_points=count, max_vertices=1500,
                     support_radius_m=0.08, texture_size=512, texture_patch_px=16)
    mesh, metadata = poisson_mesh(arrays, cfg, chunk_size=128)
    vertices, faces = mesh["vertices"], mesh["faces"]
    assert len(vertices) > 0 and len(faces) > 0
    assert vertices.dtype == np.float64 and np.isfinite(vertices).all()
    assert np.issubdtype(faces.dtype, np.integer)
    assert faces.min() >= 0 and faces.max() < len(vertices)
    assert len(vertices) <= cfg.max_vertices
    assert 3*len(faces) <= cfg.max_vertices  # Includes the downstream UV seam budget.
    assert len(faces) <= (cfg.texture_size//cfg.texture_patch_px)**2
    assert metadata["method"] == "open3d_poisson"
    assert metadata["poisson_input_points"] == metadata["source_points"] == count
    assert metadata["input_sampled"] is False
    assert metadata["normal_orientation"] == "open3d_knn_normals_signed_toward_observing_view_directions"
    # All initial normals deliberately oppose the viewing cues. An outward face
    # winding demonstrates that actual KNN estimation/Poisson plus sign correction
    # used the viewing directions rather than retaining those wrong input signs.
    centroids = vertices[faces].mean(axis=1)
    face_normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                            vertices[faces[:, 2]]-vertices[faces[:, 0]])
    nondegenerate = np.linalg.norm(face_normals, axis=1) > np.finfo(float).eps
    assert np.any(nondegenerate)
    signed = np.einsum("ij,ij->i", face_normals[nondegenerate], centroids[nondegenerate]-center)
    assert np.mean(signed > 0) > 0.95
    confidence, inferred = mesh["confidence"], mesh["inferred"]
    assert confidence.shape == inferred.shape == (len(vertices),)
    assert inferred.dtype == np.bool_
    assert np.isfinite(confidence).all() and np.all((confidence >= 0) & (confidence <= 1))
    assert np.all(mesh["colors"] == [40, 100, 180])
    distance, _ = cKDTree(points).query(vertices)
    np.testing.assert_array_equal(inferred, distance > cfg.support_radius_m)
    assert metadata["inferred_vertices"] == np.count_nonzero(inferred)
    assert metadata["support_method"] == "nearest_full_observed_cloud_distance"
    assert metadata["support_distance_max_m"] == pytest.approx(float(np.max(distance)))
    assert confidence.max() < arrays["confidence"].max()
