"""Actual Open3D Poisson reconstruction and observed-support provenance."""

import numpy as np

from drishti.errors import DependencyError, ReconstructionError


def observed_support(vertices, arrays, radius_m, chunk_size):
    """Nearest real observations, streamed over the full cloud, not a capped sample."""
    from scipy.spatial import cKDTree

    distances = np.full(len(vertices), np.inf)
    confidence = np.zeros(len(vertices), np.float32)
    inferred = np.ones(len(vertices), bool)
    for begin in range(0, len(arrays["points"]), chunk_size):
        end = begin+chunk_size
        points = np.asarray(arrays["points"][begin:end], np.float64)
        if not len(points):
            continue
        tree = cKDTree(points)
        source_conf = arrays["confidence"][begin:end]
        source_inferred = arrays["inferred"][begin:end]
        for start in range(0, len(vertices), chunk_size):
            stop = min(len(vertices), start+chunk_size)
            distance, index = tree.query(vertices[start:stop])
            better = distance < distances[start:stop]
            distances[start:stop][better] = distance[better]
            confidence[start:stop][better] = source_conf[index[better]]
            inferred[start:stop][better] = source_inferred[index[better]]
    inferred |= distances > radius_m
    confidence *= np.exp(-distances/radius_m)
    confidence[~np.isfinite(distances)] = 0
    return confidence, inferred, distances


def poisson_mesh(arrays, cfg, chunk_size):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise DependencyError("Poisson meshing requires Open3D; the real fused cloud is preserved") from exc
    from scipy.spatial import cKDTree

    count = len(arrays["points"])
    if count < cfg.normal_neighbors+1:
        raise ReconstructionError("Too few supported points for configured Poisson normal neighborhood")
    indices = np.linspace(0, count-1, min(count, cfg.max_input_points), dtype=np.int64)
    points = np.asarray(arrays["points"][indices], np.float64)
    normals = np.asarray(arrays["normals"][indices], np.float64)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    valid &= np.linalg.norm(normals, axis=1) > np.finfo(float).eps
    points, normals = points[valid], normals[valid]
    directions = np.asarray(arrays["view_direction"][indices][valid], np.float64) if "view_direction" in arrays else normals
    if len(points) < cfg.normal_neighbors+1:
        raise ReconstructionError("Too few points with actual camera-oriented normals")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    # Smooth geometric normals in a real local neighborhood, preserving the sign
    # of the observed camera-facing depth normals at each point. A global-up or
    # unrestricted graph orientation would lose those observing-view constraints.
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=cfg.normal_neighbors))
    smoothed = np.asarray(cloud.normals).copy()
    smoothed[np.einsum("ij,ij->i", smoothed, directions) < 0] *= -1
    normals = smoothed
    cloud.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=cfg.poisson_depth)
    densities = np.asarray(densities)
    if len(densities):
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, cfg.density_quantile))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    # UV seam duplication has three vertices per triangle; budget before texturing.
    atlas_capacity = (cfg.texture_size//cfg.texture_patch_px)**2
    face_limit = min(cfg.max_vertices//3, atlas_capacity)
    if face_limit < 1:
        raise ReconstructionError("Mesh/atlas configuration has no triangle capacity")
    if len(mesh.triangles) > face_limit:
        mesh = mesh.simplify_quadric_decimation(face_limit)
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > face_limit:
        raise ReconstructionError("Open3D could not satisfy the configured mesh/texture vertex budget")
    vertices = np.asarray(mesh.vertices, np.float64).copy()
    faces = np.asarray(mesh.triangles, np.int64).copy()
    if not len(vertices) or not len(faces):
        raise ReconstructionError("Poisson reconstruction has no supported surface after density filtering")
    # Face winding follows the nearest actual view-oriented observation normal.
    centroids = np.mean(vertices[faces], axis=1)
    _, nearest = cKDTree(points).query(centroids)
    face_normal = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    flip = np.einsum("ij,ij->i", face_normal, normals[nearest]) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    confidence, inferred, distances = observed_support(vertices, arrays, cfg.support_radius_m, chunk_size)
    _, nearest_vertex = cKDTree(points).query(vertices)
    colors = np.asarray(arrays["colors"][indices][valid])[nearest_vertex]
    return {"vertices": vertices, "faces": faces, "colors": colors,
            "confidence": confidence, "inferred": inferred}, {
                "method": "open3d_poisson", "poisson_input_points": len(points),
                "input_sampled": len(points) < count, "source_points": count,
                "normal_orientation": "open3d_knn_normals_signed_toward_observing_view_directions",
                "support_method": "nearest_full_observed_cloud_distance",
                "support_radius_m": cfg.support_radius_m,
                "inferred_vertices": int(np.count_nonzero(inferred)),
                "support_distance_max_m": float(np.max(distances)),
                "density_threshold": float(np.quantile(densities, cfg.density_quantile)) if len(densities) else None}
