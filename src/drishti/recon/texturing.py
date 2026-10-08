"""Observed best-view UV atlas, with occlusion/exclusion and honest uncovered pixels."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drishti.errors import InputError
from drishti.recon.depth import bilinear, project_points


def triangle_pixels(triangle, width, height):
    """Raster coordinates and screen barycentrics, including shared edges."""
    low = np.maximum(np.floor(np.min(triangle, axis=0)).astype(int), 0)
    high = np.minimum(np.ceil(np.max(triangle, axis=0)).astype(int), [width-1, height-1])
    if np.any(low > high):
        return np.empty((0, 2), int), np.empty((0, 3))
    x, y = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
    a, b, c = np.asarray(triangle, float)
    denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(denominator) <= np.finfo(float).eps:
        return np.empty((0, 2), int), np.empty((0, 3))
    u = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denominator
    v = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denominator
    weights = np.stack((u, v, 1-u-v), axis=-1)
    inside = np.all(weights >= -1e-9, axis=-1)
    return np.column_stack((x[inside], y[inside])), weights[inside]


def visibility_buffer(vertices, faces, camera, long_side):
    scale = min(1., long_side/max(camera["width"], camera["height"]))
    width, height = max(1, int(np.ceil(camera["width"]*scale))), max(1, int(np.ceil(camera["height"]*scale)))
    uv, z = project_points(vertices, camera)
    depth = np.full((height, width), np.inf, np.float64)
    for face in faces:
        if np.any(z[face] <= 0) or not np.isfinite(uv[face]).all():
            continue
        xy, bary = triangle_pixels(uv[face]*scale, width, height)
        if len(xy):
            metric_z = 1/(bary @ (1/z[face]))
            ix, iy = xy.T
            depth[iy, ix] = np.minimum(depth[iy, ix], metric_z)
    return depth, scale


def visible_samples(uv, z, depth, scale, tolerance):
    height, width = depth.shape
    xy = np.rint(uv*scale).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    result = np.zeros(len(z), bool)
    where = np.flatnonzero(inside)
    buffer_z = depth[xy[where, 1], xy[where, 0]]
    result[where] = np.isfinite(buffer_z) & (z[where] > 0) & (np.abs(z[where]-buffer_z) <= tolerance)
    return result


@dataclass
class TexturedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    texture: np.ndarray
    source_views: np.ndarray
    vertex_source: np.ndarray
    metadata: dict


def best_view_texture(vertices, faces, cameras, load_image, load_mask, cfg, inferred=None):
    """Assign each observed face a supported camera and bake its world-space patch.

    UV vertices are seam-duplicated coherently with faces/attributes. Unobserved
    texels remain alpha=0. Source images, not vertex colors, supply every texel.
    """
    vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    size, patch, padding = cfg.texture_size, cfg.texture_patch_px, cfg.texture_padding_px
    across = size//patch
    if padding < 0 or patch <= 2*padding+1 or len(faces) > across**2:
        raise InputError("Texture atlas capacity/padding is incompatible with the mesh")
    centroids = np.mean(vertices[faces], axis=1)
    normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], np.finfo(float).eps)
    candidate_count = int(getattr(cfg, "texture_candidate_views", 4))
    candidate_scores = np.full((len(faces), candidate_count), -np.inf, np.float64)
    candidate_sources = np.full((len(faces), candidate_count), -1, np.int32)
    supported = np.ones(len(faces), bool) if inferred is None else ~np.any(np.asarray(inferred)[faces], axis=1)
    # One visibility buffer and one image/mask at a time, never all camera images.
    for index, camera in enumerate(cameras):
        direction = np.asarray(camera["center"], np.float64)-centroids
        distance = np.linalg.norm(direction, axis=1)
        angle = np.einsum("ij,ij->i", normals, direction)/np.maximum(distance, np.finfo(float).eps)
        uv, z = project_points(centroids, camera)
        good = supported & (angle > 0) & np.isfinite(uv).all(axis=1) & (z > 0)
        good &= (uv[:, 0] >= 0) & (uv[:, 0] < camera["width"]-1)
        good &= (uv[:, 1] >= 0) & (uv[:, 1] < camera["height"]-1)
        where = np.flatnonzero(good)
        if not len(where):
            continue
        excluded = np.asarray(load_mask(camera), bool)
        if excluded.shape != (camera["height"], camera["width"]):
            raise InputError("Texture exclusion mask is in a different camera domain")
        ij = np.rint(uv[where]).astype(int)
        where = where[~excluded[ij[:, 1], ij[:, 0]]]
        depth, scale = visibility_buffer(vertices, faces, camera, cfg.visibility_long_side)
        where = where[visible_samples(uv[where], z[where], depth, scale, cfg.texture_occlusion_tolerance_m)]
        # Focal length and view angle favor the highest observed surface resolution.
        K = np.asarray(camera["K"], float)
        score = angle[where]*K[0, 0]*K[1, 1]/np.maximum(z[where]**2, np.finfo(float).eps)
        slots = np.argmin(candidate_scores[where], axis=1)
        better = score > candidate_scores[where, slots]
        selected = where[better]
        selected_slots = slots[better]
        candidate_scores[selected, selected_slots] = score[better]
        candidate_sources[selected, selected_slots] = index
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    candidate_scores = np.take_along_axis(candidate_scores, order, axis=1)
    candidate_sources = np.take_along_axis(candidate_sources, order, axis=1)
    texture = np.zeros((size, size, 4), np.uint8)
    local_triangle = np.array([[padding, padding], [patch-padding-1, padding], [padding, patch-padding-1]], float)
    local_xy, bary = triangle_pixels(local_triangle, patch, patch)
    uv = np.empty((len(faces)*3, 2), np.float64)
    for face_id in range(len(faces)):
        offset = np.array([(face_id % across)*patch, (face_id // across)*patch])
        coords = local_triangle+offset
        uv[3*face_id:3*face_id+3, 0] = (coords[:, 0]+0.5)/size
        uv[3*face_id:3*face_id+3, 1] = 1-(coords[:, 1]+0.5)/size
    textured_pixels, covered_faces, fallback_faces = 0, 0, 0
    sources = np.full(len(faces), -1, np.int32)
    # Try ranked cameras in passes so each camera image/mask/visibility buffer is
    # loaded at most once per candidate rank. A failed best view can therefore use
    # a valid alternate view instead of creating an avoidable transparent face.
    for rank in range(candidate_count):
        for index, camera in enumerate(cameras):
            chosen = np.flatnonzero((candidate_sources[:, rank] == index) & (sources < 0))
            if not len(chosen):
                continue
            rgb = np.asarray(load_image(camera))
            excluded = np.asarray(load_mask(camera), bool)
            if rgb.shape != (camera["height"], camera["width"], 3):
                raise InputError("Texture source dimensions differ from the camera")
            depth, scale = visibility_buffer(vertices, faces, camera, cfg.visibility_long_side)
            for face_id in chosen:
                world = bary @ vertices[faces[face_id]]
                source_uv, z = project_points(world, camera)
                good = np.isfinite(source_uv).all(axis=1) & (z > 0)
                good &= (source_uv[:, 0] >= 0) & (source_uv[:, 0] <= camera["width"]-1)
                good &= (source_uv[:, 1] >= 0) & (source_uv[:, 1] <= camera["height"]-1)
                where = np.flatnonzero(good)
                if not len(where):
                    continue
                # All four bilinear sample pixels must be unmasked, avoiding color bleed.
                floor = np.floor(source_uv[where]).astype(int)
                ceil = np.minimum(floor+1, [camera["width"]-1, camera["height"]-1])
                clear = ~(excluded[floor[:, 1], floor[:, 0]] | excluded[floor[:, 1], ceil[:, 0]]
                          | excluded[ceil[:, 1], floor[:, 0]] | excluded[ceil[:, 1], ceil[:, 0]])
                clear &= visible_samples(source_uv[where], z[where], depth, scale, cfg.texture_occlusion_tolerance_m)
                where = where[clear]
                if not len(where):
                    continue
                color = np.clip(np.rint(bilinear(rgb, source_uv[where])), 0, 255).astype(np.uint8)
                offset = np.array([(face_id % across)*patch, (face_id // across)*patch])
                xy = local_xy[where]+offset
                texture[xy[:, 1], xy[:, 0], :3] = color
                texture[xy[:, 1], xy[:, 0], 3] = 255
                textured_pixels += len(where)
                covered_faces += 1
                sources[face_id] = index
                if rank:
                    fallback_faces += 1
    vertex_source = faces.ravel()
    return TexturedMesh(vertices[vertex_source], np.arange(len(vertex_source)).reshape(-1, 3),
                        uv, texture, sources, vertex_source, {
                             "texture_method": "observed_ranked_multi_view_fallback_world_reprojected_per_face_uv_atlas",
                            "textured_faces": covered_faces, "faces": len(faces),
                            "textured_face_fraction": covered_faces/len(faces) if len(faces) else 0.,
                            "observed_texels": textured_pixels, "unobserved_texels": "alpha_zero",
                            "allocated_triangle_texels": len(local_xy)*len(faces),
                             "observed_triangle_texel_fraction": textured_pixels/(len(local_xy)*len(faces))
                             if len(local_xy)*len(faces) else 0.,
                             "fallback_view_faces": fallback_faces,
                             "candidate_views_per_face": candidate_count,
                            "visibility_method": "projected_mesh_camera_z_buffer",
                            "visibility_raster_domain": "distorted_projected_triangle_approximation",
                            "visibility_long_side": cfg.visibility_long_side,
                            "atlas_size": size, "patch_size": patch})


def write_obj(path, vertices, faces, uv=None, material=None):
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write("# DRISHTI float64 local-origin geometry, metres\n")
        if material:
            stream.write(f"mtllib {material}\nusemtl observed_atlas\n")
        for point in vertices:
            stream.write("v " + " ".join(f"{v:.17g}" for v in point) + "\n")
        if uv is not None:
            for coord in uv:
                stream.write(f"vt {coord[0]:.17g} {coord[1]:.17g}\n")
        for face in faces+1:
            tokens = [f"{i}/{i}" if uv is not None else str(i) for i in face]
            stream.write("f " + " ".join(tokens) + "\n")


def write_material(path, texture="texture.png"):
    Path(path).write_text("newmtl observed_atlas\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\n"
                          f"map_Kd {texture}\nmap_d {texture}\n", encoding="utf-8")
