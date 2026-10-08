"""Supported geospatial rasters in a projected, local-origin metric frame.

Dependencies are deliberately imported at execution time. No interpolation is used
to fill holes, and inferred points do not establish observed raster support.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Grid:
    left: float  # local coordinates, before adding the projected origin
    top: float
    resolution: float
    width: int
    height: int

    @property
    def shape(self):
        return self.height, self.width

    def indices(self, points):
        import numpy as np

        col = np.floor((points[:, 0] - self.left) / self.resolution).astype(np.int64)
        row = np.floor((self.top - points[:, 1]) / self.resolution).astype(np.int64)
        return row, col

    def transform(self, origin):
        from rasterio.transform import from_origin

        return from_origin(
            self.left + float(origin[0]), self.top + float(origin[1]),
            self.resolution, self.resolution,
        )


def validate_cloud(cloud: dict[str, Any], *, chunk_size: int | None = None) -> dict[str, Any]:
    """Reject ambiguous precision/provenance instead of inventing confidence."""
    import numpy as np

    points = np.asanyarray(cloud["points"])
    if points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Cloud points must be Nx3 float64 local metres")
    if not len(points):
        raise ValueError("Cloud is empty")
    n = len(points)
    colors = np.asanyarray(cloud["colors"])
    confidence = np.asanyarray(cloud["confidence"])
    inferred = np.asanyarray(cloud["inferred"])
    if colors.shape != (n, 3) or colors.dtype != np.uint8:
        raise ValueError("Cloud colors must be Nx3 uint8")
    if confidence.shape != (n,) or not np.issubdtype(confidence.dtype, np.floating):
        raise ValueError("Cloud confidence must contain one floating value per point")
    if inferred.shape != (n,) or inferred.dtype != np.bool_:
        raise ValueError("Cloud inferred must contain one boolean per point")
    step = chunk_size if chunk_size is not None else n
    if step <= 0:
        raise ValueError("Cloud validation chunk size must be positive")
    for start in range(0, n, step):
        if not np.isfinite(points[start:start + step]).all():
            raise ValueError("Cloud has nonfinite coordinates")
        values = confidence[start:start + step]
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Cloud confidence must be finite and lie in [0, 1]")
    return {**cloud, "points": points, "colors": colors,
            "confidence": confidence, "inferred": inferred}


def read_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_cloud(bundle) -> dict[str, Any]:
    """Prefer final S7's read-only NPY sidecars; never fall back from broken ones.

    The NPZ fallback supports older contract bundles without sidecar metadata.
    Declared sidecars are authoritative and confined to the bundle like artifacts.
    """
    import numpy as np

    outputs = bundle.manifest.get("stages", {}).get("s7_dense", {}).get("outputs", {})
    if "metadata" not in outputs:
        return read_npz(bundle.artifact("s7_dense", "cloud"))
    path = bundle.artifact("s7_dense", "metadata")
    metadata = bundle.read_json(path.relative_to(bundle.root).as_posix())
    if "arrays" not in metadata:
        return read_npz(bundle.artifact("s7_dense", "cloud"))
    arrays = metadata["arrays"]
    required = {"points", "colors", "confidence", "inferred"}
    if not isinstance(arrays, dict) or not required.issubset(arrays):
        raise ValueError("S7 mmap sidecars lack required point/provenance arrays")
    cloud = {key: np.load(bundle.path(relative), mmap_mode="r", allow_pickle=False)
             for key, relative in arrays.items()}
    if metadata.get("points") is not None and metadata["points"] != len(cloud["points"]):
        raise ValueError("S7 sidecar point count differs from its measured metadata")
    return cloud


def georeference(poses: dict, manifest_crs: dict | None = None) -> dict:
    import numpy as np
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    if not isinstance(poses, dict):
        raise ValueError("Final poses must be an object with a projected_local geometry frame")
    if poses.get("frame") != "projected_local":
        raise ValueError("Products require a projected_local geometry frame")
    origin = np.asarray(poses.get("origin"), dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("A finite projected XYZ origin is required")
    crs = poses.get("crs") or manifest_crs or {}
    if not isinstance(crs, dict):
        crs = {"epsg": crs}
    spec = crs.get("wkt") or crs.get("epsg")
    if spec is None:
        raise ValueError("Projected CRS is absent; geographic products cannot be tagged")
    try:
        parsed = CRS.from_user_input(spec)
    except CRSError as exc:
        raise ValueError(f"Projected CRS is invalid: {exc}") from exc
    horizontal = parsed.sub_crs_list[0] if parsed.is_compound else parsed
    if not horizontal.is_projected or any(
        not np.isclose(axis.unit_conversion_factor, 1.0)
        for axis in horizontal.axis_info[:2]
    ):
        raise ValueError("Products require a projected CRS with metre horizontal axes")
    datum = crs.get("vertical")
    source_datum = (crs.get("source_datum") or crs.get("source_altitude_datum")
                    or crs.get("altitude_datum") or "unknown")
    if datum in (None, "source"):
        datum = source_datum
    if (not isinstance(datum, str) or not datum.strip()
            or not isinstance(source_datum, str) or not source_datum.strip()):
        raise ValueError("Vertical and source datum provenance must be nonblank strings")
    if datum == "relative" or source_datum == "relative":
        raise ValueError("Geographic products require absolute altitude, not a relative datum")
    return {**crs, "epsg": horizontal.to_epsg(), "wkt": parsed.to_wkt(),
            "origin": origin.tolist(), "vertical": datum or "unknown",
            "source_datum": source_datum,
            "vertical_conversion": crs.get("vertical_conversion"),
            "frame": "projected_local", "units": "metres"}


def rasterize_cloud(cloud: dict, geo_cfg, *, chunk_size: int | None = None) -> dict:
    """Top observed surface and a conservative morphological ground baseline.

    Opening the per-cell minimum removes local elevated objects; original points
    within the configured height threshold are classified as ground. The DTM is
    their mean elevation in each cell, not an interpolated/filled terrain model.
    """
    import numpy as np
    from scipy.ndimage import maximum_filter, minimum_filter

    cloud = validate_cloud(cloud, chunk_size=chunk_size)
    points = cloud["points"]
    step = chunk_size if chunk_size is not None else len(points)
    resolution = float(geo_cfg.resolution_m)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("geo.resolution_m must be positive")
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for start in range(0, len(points), step):
        block = points[start:start + step]
        lower = np.minimum(lower, block.min(axis=0))
        upper = np.maximum(upper, block.max(axis=0))
    left = float(np.floor(lower[0] / resolution) * resolution)
    top = float((np.floor(upper[1] / resolution) + 1) * resolution)
    width = int(np.floor((upper[0] - left) / resolution)) + 1
    height = int(np.floor((top - lower[1]) / resolution)) + 1
    if width * height > geo_cfg.max_cells:
        raise ValueError("Raster exceeds geo.max_cells; increase resolution_m or tile the input")
    if geo_cfg.ground_method != "morphological":
        raise ValueError("Only the actual morphological ground classifier is implemented")
    grid = Grid(left, top, resolution, width, height)
    size = width * height
    counts = np.zeros(size, dtype=np.int64)
    confidence_sum = np.zeros(size, dtype=np.float64)
    mask = np.zeros(size, dtype=np.uint8)
    surface = np.full(size, -np.inf, dtype=np.float64)
    minimum = np.full(size, np.inf, dtype=np.float64)
    observed_points = 0
    for start in range(0, len(points), step):
        block = points[start:start + step]
        observed = ~cloud["inferred"][start:start + step]
        observed_points += int(observed.sum())
        row, col = grid.indices(block)
        flat = row * width + col
        obs_idx = flat[observed]
        np.add.at(counts, obs_idx, 1)
        np.add.at(confidence_sum, obs_idx, cloud["confidence"][start:start + step][observed])
        np.maximum.at(surface, obs_idx, block[observed, 2])
        np.minimum.at(minimum, obs_idx, block[observed, 2])
        mask[flat[~observed]] = 2
    counts = counts.reshape(grid.shape)
    surface = surface.reshape(grid.shape)
    minimum = minimum.reshape(grid.shape)
    supported = counts > 0
    window = max(1, int(np.ceil(geo_cfg.ground_window_m / resolution)))
    window += 1 - window % 2  # symmetric odd footprint, not a tuning threshold
    eroded = minimum_filter(minimum, size=window, mode="constant", cval=np.inf)
    opened = maximum_filter(
        np.where(np.isfinite(eroded), eroded, -np.inf), size=window,
        mode="constant", cval=-np.inf,
    )
    ground = np.zeros(len(points), dtype=bool)
    ground_counts, sums = np.zeros(size, dtype=np.int64), np.zeros(size, dtype=np.float64)
    for start in range(0, len(points), step):
        block = points[start:start + step]
        row, col = grid.indices(block)
        flat = row * width + col
        baseline = opened.ravel()[flat]
        keep = ~cloud["inferred"][start:start + step] & np.isfinite(baseline)
        keep &= block[:, 2] <= baseline + geo_cfg.ground_threshold_m
        ground[start:start + step] = keep
        np.add.at(ground_counts, flat[keep], 1)
        np.add.at(sums, flat[keep], block[keep, 2])
    dtm = np.full(size, np.nan, dtype=np.float64)
    np.divide(sums, ground_counts, out=dtm, where=ground_counts > 0)
    confidence_sum = confidence_sum.reshape(grid.shape)
    confidence = np.full(grid.shape, np.nan, dtype=np.float64)
    np.divide(confidence_sum, counts, out=confidence, where=supported)
    surface[~supported] = np.nan
    mask = mask.reshape(grid.shape)
    mask[supported] = 1
    return {"grid": grid, "dsm": surface, "dtm": dtm.reshape(grid.shape),
            "confidence": confidence, "support": counts, "mask": mask,
            "ground": ground, "ground_counts": ground_counts.reshape(grid.shape),
            "metadata": {
                "dsm_method": "maximum_observed_point_elevation_per_cell",
                "inferred_excluded": True,
                "elevation_frame": "projected_absolute",
                "dtm_method": "morphological_opening_ground_classification_cell_mean",
                "ground_window_cells": window, "ground_points": int(ground.sum()),
                "observed_points": observed_points,
                "inferred_points_excluded": len(points) - observed_points,
                "supported_cells": int(supported.sum()), "total_cells": size,
                "supported_area_m2": float(supported.sum() * resolution**2),
                "bounding_grid_coverage_fraction": float(supported.mean()),
                "scene_completeness": None,
                "completeness_reason": "Visible-scene target surface denominator is not supplied",
                "dtm_limitation": "Geometric baseline; ground under roofs/canopy is not recovered",
                "mask_values": {"0": "unsupported", "1": "observed", "2": "inferred_only"},
            }}


def write_geotiff(path: Path, data, grid: Grid, ref: dict, *, nodata,
                  valid=None, descriptions=(), tags=None) -> None:
    import json

    import numpy as np
    import rasterio

    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3 or not data.shape[0] or data.shape[1:] != grid.shape:
        raise ValueError("Raster shape does not match its grid")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != grid.shape:
            raise ValueError("Raster validity mask shape does not match its grid")
    if len(descriptions) > data.shape[0]:
        raise ValueError("Raster has more band descriptions than bands")
    profile = {"driver": "GTiff", "width": grid.width, "height": grid.height,
               "count": data.shape[0], "dtype": data.dtype, "crs": ref["wkt"],
               "transform": grid.transform(ref["origin"]), "nodata": nodata,
               "compress": "deflate", "BIGTIFF": "IF_SAFER"}
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
            if valid is not None:
                dst.write_mask(np.asarray(valid, dtype=np.uint8) * 255)
            for band, description in enumerate(descriptions, start=1):
                dst.set_band_description(band, description)
            dst.update_tags(
                vertical_datum=str(ref["vertical"]),
                source_datum=str(ref.get("source_datum") or ref.get("source_altitude_datum")
                                 or ref.get("altitude_datum") or "unknown"),
                vertical_conversion=json.dumps(ref.get("vertical_conversion")),
                local_origin=json.dumps(ref["origin"]),
                **(tags or {}),
            )


def project_camera(points, camera):
    """OpenCV world→camera projection, including declared lens distortion."""
    import numpy as np

    from drishti.recon.depth import project_points

    rotation = np.asarray(camera["R_cw"], dtype=np.float64)
    center = np.asarray(camera["center"], dtype=np.float64)
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    if rotation.shape != (3, 3) or intrinsic.shape != (3, 3) or center.shape != (3,):
        raise ValueError("Camera requires R_cw, center and K of the contracted shapes")
    if not all(np.isfinite(a).all() for a in (rotation, center, intrinsic)):
        raise ValueError("Camera transform has nonfinite values")
    return project_points(points, camera)


def validate_visibility_mesh(mesh: dict) -> dict:
    """Validate occlusion geometry without silently coercing corrupt face indices."""
    import numpy as np

    vertices, faces = np.asarray(mesh["vertices"]), np.asarray(mesh["faces"])
    if (vertices.dtype != np.float64 or vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()):
        raise ValueError("Visibility mesh vertices must be finite Nx3 float64 local metres")
    if (faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
        raise ValueError("Visibility mesh face indices must be valid integer triangles")
    inferred = mesh.get("inferred")
    if inferred is not None:
        inferred = np.asarray(inferred)
        if inferred.dtype != np.bool_ or inferred.shape != (len(vertices),):
            raise ValueError("Visibility mesh inferred flags must be one boolean per vertex")
    return {**mesh, "vertices": vertices, "faces": faces, "inferred": inferred}


def mesh_zbuffer(mesh: dict | None, camera: dict, shape):
    """Triangle visibility buffer with perspective-correct camera-Z interpolation."""
    import numpy as np

    height, width = shape
    buffer = np.full((height, width), np.inf, dtype=np.float64)
    if mesh is None:
        return buffer
    mesh = validate_visibility_mesh(mesh)
    vertices = mesh["vertices"]
    pixels, depth = project_camera(vertices, camera)
    faces = mesh["faces"]
    inferred = mesh.get("inferred")
    if inferred is not None:
        faces = faces[~np.asarray(inferred, dtype=bool)[faces].any(axis=1)]
    for face in faces:
        z = depth[face]
        uv = pixels[face]
        # Behind/near-plane crossing triangles are excluded, never inverted.
        if (z <= 0).any() or not np.isfinite(uv).all():
            continue
        lo = np.maximum(np.ceil(uv.min(axis=0)), 0).astype(np.int64)
        hi = np.minimum(np.floor(uv.max(axis=0)), [width - 1, height - 1]).astype(np.int64)
        if (lo > hi).any():
            continue
        a, b, c = uv
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) <= np.finfo(np.float64).eps:
            continue
        x = np.arange(lo[0], hi[0] + 1)
        for y in range(int(lo[1]), int(hi[1]) + 1):
            w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
            w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            inverse = w0 / z[0] + w1 / z[1] + w2 / z[2]
            visible_x = x[inside & (inverse > 0)]
            if visible_x.size:
                values = 1 / inverse[inside & (inverse > 0)]
                buffer[y, visible_x] = np.minimum(buffer[y, visible_x], values)
    return buffer


def confined_path(root: Path, relative: str, *, allow_absolute: bool = False) -> Path:
    """Resolve an artifact without allowing absolute or escaping bundle paths."""
    if (not isinstance(relative, str) or not relative
            or (Path(relative).is_absolute() and not allow_absolute)):
        raise ValueError("Artifact path must be bundle-relative")
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def dynamic_exclusion(root: Path, masks, frame_id: str, shape, *, allow_absolute: bool = False):
    import numpy as np
    from PIL import Image

    if masks is None:
        return None
    if frame_id not in masks:
        raise ValueError("Recorded dynamic masks lack this final camera")
    with Image.open(confined_path(root, masks[frame_id], allow_absolute=allow_absolute)) as image:
        excluded = np.asarray(image) != 0
    if excluded.shape != shape:
        raise ValueError("Dynamic mask shape differs from image")
    return excluded


def visible_dsm_samples(products: dict, camera: dict, shape, *, mesh=None,
                        excluded=None, chunk_size: int):
    """Yield cell indices, nearest pixels and geometric view scores in bounded chunks.

    The triangle z-buffer additionally represents facades/overhangs. Lens-distorted
    mesh edges are approximated as straight segments; this is recorded explicitly.
    DSM cells also occlude one another, including cells without semantic labels.
    This is shared by RGB and categorical projection; scores are not confidence.
    """
    import numpy as np

    grid, dsm = products["grid"], np.asarray(products["dsm"])
    if dsm.shape != grid.shape:
        raise ValueError("Projected DSM shape does not match its grid")
    valid = np.isfinite(dsm)
    for key in ("support", "mask"):
        if key in products:
            values = np.asarray(products[key])
            if values.shape != grid.shape:
                raise ValueError("Projected DSM support shape does not match its grid")
            valid &= values > 0 if key == "support" else values == 1
    supported = np.flatnonzero(valid.ravel())
    if chunk_size <= 0:
        raise ValueError("Projection chunk size must be positive")
    h, w = shape
    if excluded is not None and excluded.shape != shape:
        raise ValueError("Dynamic exclusion shape differs from image")
    zbuffer = mesh_zbuffer(mesh, camera, shape)

    def cells(indices):
        rows, cols = np.divmod(indices, grid.width)
        return np.column_stack((grid.left + (cols + 0.5) * grid.resolution,
                                grid.top - (rows + 0.5) * grid.resolution,
                                dsm.ravel()[indices]))

    def project(indices):
        xyz = cells(indices)
        uv, z = project_camera(xyz, camera)
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
        finite &= (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1)
        finite &= (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        take = np.flatnonzero(finite)
        pixel = np.rint(uv[take]).astype(np.int64)
        if excluded is not None:
            keep = ~excluded[pixel[:, 1], pixel[:, 0]]
            take, pixel = take[keep], pixel[keep]
        return xyz, z, take, pixel

    # DSM samples participate in visibility even without a viable mesh.
    for start in range(0, len(supported), chunk_size):
        indices = supported[start:start + chunk_size]
        _, z, take, pixel = project(indices)
        np.minimum.at(zbuffer, (pixel[:, 1], pixel[:, 0]), z[take])
    for start in range(0, len(supported), chunk_size):
        indices = supported[start:start + chunk_size]
        xyz, z, take, pixel = project(indices)
        tolerance = grid.resolution * np.sqrt(2) / 2
        visible = z[take] <= zbuffer[pixel[:, 1], pixel[:, 0]] + tolerance
        take, pixel = take[visible], pixel[visible]
        delta = np.asarray(camera["center"]) - xyz[take]
        distance = np.linalg.norm(delta, axis=1)
        score = np.divide(delta[:, 2], distance**3,
                          out=np.full_like(distance, -np.inf), where=distance > 0)
        above = delta[:, 2] > 0
        yield indices[take[above]], pixel[above], score[above]


def orthoproject(products: dict, cameras: list[dict], root: Path, *, mesh=None,
                 masks=None, chunk_size: int) -> tuple[Any, dict]:
    """Project supported DSM cells into final images, reject occlusion, select view."""
    import numpy as np
    from PIL import Image

    grid = products["grid"]
    rgba = np.zeros((*grid.shape, 4), dtype=np.uint8)
    best = np.full(grid.shape, -np.inf)
    used, failures = [], []
    if chunk_size <= 0:
        raise ValueError("Orthomosaic chunk size must be positive")
    for camera in cameras:
        frame_id = str(camera.get("frame_id", ""))
        selected = 0
        try:
            image_path = confined_path(root, camera["image_path"], allow_absolute=True)
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            h, w = rgb.shape[:2]
            if (w, h) != (camera["width"], camera["height"]):
                raise ValueError("Camera dimensions differ from observed image")
            excluded = dynamic_exclusion(root, masks, frame_id, (h, w), allow_absolute=True)
            for indices, pixel, score in visible_dsm_samples(
                products, camera, (h, w), mesh=mesh, excluded=excluded, chunk_size=chunk_size,
            ):
                chosen = score > best.ravel()[indices]
                output = indices[chosen]
                best.ravel()[output] = score[chosen]
                rgba.reshape(-1, 4)[output, :3] = rgb[pixel[chosen, 1], pixel[chosen, 0]]
                rgba.reshape(-1, 4)[output, 3] = 255
                selected += int(chosen.sum())
            used.append({"frame_id": frame_id, "selected_cell_updates": selected})
        except Exception as exc:
            if selected:
                used.append({"frame_id": frame_id, "selected_cell_updates": selected,
                             "partial_reason": f"{type(exc).__name__}: {exc}"})
            failures.append({"frame_id": frame_id, "reason": str(exc)})
    return rgba, {"method": "final_camera_DSM_image_projection_with_geometry_zbuffer",
                  "image_sampling": "nearest_observed_pixel", "cameras_used": used,
                  "cameras_skipped": failures, "textured_cells": int((rgba[..., 3] > 0).sum()),
                  "visibility_geometry": "mesh_triangles_and_DSM_samples" if mesh else "DSM_samples",
                  "lens_edge_approximation": "Straight projected triangle edges in distorted images",
                   "dynamic_masks": "required_per_projected_frame" if masks is not None else "unavailable"}
