"""Project actual S3 class maps onto observed geometry; never infer semantic labels.

Probabilities are uncalibrated source-model probabilities, separate from geometry
confidence and absolute accuracy. All optional failures remain inspectable.
"""

from pathlib import Path
from typing import Any

from drishti.geo.products import (
    Grid,
    confined_path,
    dynamic_exclusion,
    visible_dsm_samples,
    write_geotiff,
)

CONFIDENCE_METHOD = "winning_source_class_softmax_probability_uncalibrated"


def validate_descriptor(descriptor: Any) -> dict[str, Any]:
    """Validate categorical identities before trusting any frame from a descriptor."""
    if not isinstance(descriptor, dict):
        raise ValueError("Semantic descriptor must be an object")
    status = descriptor.get("status")
    if not isinstance(status, str) or status not in {"ok", "partial", "unavailable", "disabled"}:
        raise ValueError("Semantic descriptor has an invalid status")
    reason = descriptor.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("Semantic descriptor reason must be a string or null")
    if type(descriptor.get("nodata")) is not int or descriptor["nodata"] != 0:
        raise ValueError("Semantic descriptor nodata must be class 0")
    classes = descriptor.get("classes")
    if not isinstance(classes, list):
        raise ValueError("Semantic descriptor requires a classes list")
    ids: set[int] = set()
    names: set[str] = set()
    sources: set[int] = set()
    for item in classes:
        if not isinstance(item, dict):
            raise ValueError("Semantic class must be an object")
        identifier, name = item.get("id"), item.get("name")
        color, source_ids = item.get("color"), item.get("source_ids")
        if type(identifier) is not int or not 1 <= identifier <= 65534 or identifier in ids:
            raise ValueError("Semantic class IDs must be unique integers in [1, 65534]")
        if not isinstance(name, str) or not name.strip() or name.strip().casefold() in names:
            raise ValueError("Semantic class names must be nonblank and unique")
        if (not isinstance(color, list) or len(color) != 3
                or any(type(c) is not int or not 0 <= c <= 255 for c in color)):
            raise ValueError("Semantic class color must contain three uint8 values")
        if (not isinstance(source_ids, list) or not source_ids
                or any(type(s) is not int or s < 0 for s in source_ids)
                or len(set(source_ids)) != len(source_ids) or sources.intersection(source_ids)):
            raise ValueError("Semantic source IDs must be nonnegative and uniquely mapped")
        ids.add(identifier)
        names.add(name.strip().casefold())
        sources.update(source_ids)
    frames = descriptor.get("frames")
    if not isinstance(frames, list) or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError("Semantic descriptor requires a frames list")
    frame_ids = [frame.get("frame_id") for frame in frames]
    if (any(not isinstance(identifier, str) or not identifier.strip() for identifier in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)):
        raise ValueError("Semantic frame IDs must be nonblank and unique")
    if descriptor.get("model") is not None and not isinstance(descriptor["model"], dict):
        raise ValueError("Semantic model provenance must be an object or null")
    method = descriptor.get("confidence_method")
    if method is not None and (not isinstance(method, str) or not method.strip()):
        raise ValueError("Semantic confidence method must be nonblank or null")
    if status in {"ok", "partial"} and method != CONFIDENCE_METHOD:
        raise ValueError("Semantic confidence method must describe the winning source-class probability")
    if status in {"ok", "partial"} and (not classes or not frames or descriptor.get("model") is None):
        raise ValueError("Available semantics require classes, actual frames and executed model provenance")
    return descriptor


def load_frame(root: Path, frame: dict, camera: dict, class_ids: set[int]):
    """Read confined NPY maps and check the final camera's actual pixel domain."""
    import numpy as np
    from PIL import Image

    from drishti.recon.depth import image_domain

    shape = camera["height"], camera["width"]
    if any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("Final camera dimensions must be positive integers")
    declared_shape = frame.get("height"), frame.get("width")
    if (any(type(value) is not int for value in declared_shape) or declared_shape != shape):
        raise ValueError("Semantic frame shape differs from final camera")
    if frame.get("image_domain") != image_domain(camera):
        raise ValueError("Semantic frame image domain differs from final camera")
    with Image.open(confined_path(root, camera["image_path"])) as image:
        if image.size != (shape[1], shape[0]):
            raise ValueError("Final camera dimensions differ from observed image")
    labels = np.load(confined_path(root, frame["path"]), mmap_mode="r", allow_pickle=False)
    probability = np.load(confined_path(root, frame["confidence_path"]), mmap_mode="r",
                          allow_pickle=False)
    if not isinstance(labels, np.ndarray) or labels.dtype != np.uint16 or labels.shape != shape:
        raise ValueError("Semantic labels must be camera-shaped uint16 NPY")
    if (not isinstance(probability, np.ndarray) or probability.dtype != np.float32
            or probability.shape != shape):
        raise ValueError("Semantic probabilities must be camera-shaped float32 NPY")
    # Row checks bound validation temporaries even for large source images.
    labeled = 0
    for row, confidence in zip(labels, probability, strict=True):
        if not set(np.unique(row)).issubset({0, *class_ids}):
            raise ValueError("Semantic map contains class IDs absent from descriptor")
        if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("Semantic probabilities must be finite and in [0, 1]")
        if (confidence[row == 0] != 0).any():
            raise ValueError("Unknown semantic pixels must have class 0 and confidence 0")
        labeled += int(np.count_nonzero(row))
    if type(frame.get("labeled_pixels")) is not int or frame["labeled_pixels"] != labeled:
        raise ValueError("Semantic labeled_pixels differs from actual map")
    return labels, probability


def project_semantics(products: dict, cameras: list[dict], root: Path, descriptor: dict, *,
                      min_probability: float, mesh=None, masks=None, chunk_size: int):
    """Nearest-label projection using the orthomosaic's final-pose visibility test.

    The best geometric view among labeled, above-threshold visible pixels wins.
    Unmapped/low-probability pixels cannot overwrite a supported label from another
    view. Its original winning-source probability is retained without averaging.
    """
    import numpy as np

    validate_descriptor(descriptor)
    if descriptor["status"] not in {"ok", "partial"}:
        raise ValueError("No actual semantic maps are available for projection")
    if not np.isfinite(min_probability) or not 0 <= min_probability <= 1:
        raise ValueError("Semantic minimum probability must be in [0, 1]")
    if chunk_size <= 0:
        raise ValueError("Semantic projection chunk size must be positive")
    grid = products["grid"]
    dsm = np.asarray(products["dsm"])
    support, observed = np.asarray(products["support"]), np.asarray(products["mask"])
    if any(array.shape != grid.shape for array in (dsm, support, observed)):
        raise ValueError("Semantic DSM support arrays must match the raster grid")
    valid = np.isfinite(dsm) & (support > 0) & (observed == 1)
    visible_products = {**products, "dsm": np.where(valid, dsm, np.nan)}
    labels = np.zeros(grid.shape, dtype=np.uint16)
    probability = np.zeros(grid.shape, dtype=np.float32)
    best = np.full(grid.shape, -np.inf)
    class_ids = {item["id"] for item in descriptor["classes"]}
    frames = {item["frame_id"]: item for item in descriptor["frames"]}
    used, skipped = [], []
    camera_ids = {str(camera.get("frame_id", "")) for camera in cameras}
    for frame_id in sorted(frames.keys() - camera_ids):
        skipped.append({"frame_id": frame_id, "reason": "No matching final camera"})
    for camera in cameras:
        frame_id = str(camera.get("frame_id", ""))
        if frame_id not in frames:
            continue
        selected = 0
        try:
            source, confidence = load_frame(root, frames[frame_id], camera, class_ids)
            excluded = dynamic_exclusion(root, masks, frame_id, source.shape)
            for indices, pixel, score in visible_dsm_samples(
                visible_products, camera, source.shape, mesh=mesh, excluded=excluded,
                chunk_size=chunk_size,
            ):
                values = source[pixel[:, 1], pixel[:, 0]]
                chances = confidence[pixel[:, 1], pixel[:, 0]]
                # Compare the stored float32 value against the actual configured
                # threshold, without NumPy rounding that threshold back to float32.
                chosen = (values != 0) & (chances.astype(np.float64) >= min_probability)
                chosen &= score > best.ravel()[indices]
                output = indices[chosen]
                best.ravel()[output] = score[chosen]
                labels.ravel()[output] = values[chosen]
                probability.ravel()[output] = chances[chosen]
                selected += int(chosen.sum())
            used.append({"frame_id": frame_id, "selected_cell_updates": selected})
        except Exception as exc:
            # Keep actually projected cells from earlier chunks/cameras, including
            # native projector failures, and identify any partial camera explicitly.
            if selected:
                used.append({"frame_id": frame_id, "selected_cell_updates": selected,
                             "partial_reason": f"{type(exc).__name__}: {exc}"})
            skipped.append({"frame_id": frame_id, "reason": str(exc)})
    return labels, probability, {
        "method": "final_camera_DSM_semantic_projection_with_geometry_zbuffer",
        "sampling": "nearest_source_label_and_probability",
        "view_selection": "near_top_facing_best_geometric_view_among_valid_labels",
        "inferred_excluded": True, "unsupported_excluded": True,
        "min_probability": min_probability, "cameras_used": used, "cameras_skipped": skipped,
        "supported_cells": int(np.count_nonzero(labels)),
        "visibility_geometry": "mesh_triangles_and_DSM_samples" if mesh else "DSM_samples",
        "lens_edge_approximation": "Straight projected triangle edges in distorted images",
        "dynamic_masks": "required_per_projected_frame" if masks is not None else "unavailable",
    }


def geographic_preview(path: Path, labels, grid: Grid, ref: dict, classes: list[dict], *,
                       long_side: int) -> dict:
    """Make an offline geographic warp, even if the host enabled PROJ networking."""
    import rasterio
    from pyproj import network

    previous = network.is_network_enabled()
    try:
        network.set_network_enabled(False)
        with rasterio.Env(PROJ_NETWORK="OFF"):
            return _geographic_preview(path, labels, grid, ref, classes, long_side=long_side)
    finally:
        network.set_network_enabled(previous)


def _geographic_preview(path: Path, labels, grid: Grid, ref: dict, classes: list[dict], *,
                        long_side: int) -> dict:
    """Warp categorical pixels into a bounded geographic grid, then colorize RGBA.

    Nearest warping before the color lookup is equivalent to nearest RGBA warping
    and avoids treating a valid black class color as nodata. No projected image is
    stretched onto longitude/latitude bounds.
    """
    import numpy as np
    from PIL import Image
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds, from_bounds
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds

    if type(long_side) is not int or long_side <= 0:
        raise ValueError("Semantic preview long side must be a positive integer")
    labels = np.asarray(labels)
    if labels.shape != grid.shape or labels.dtype != np.uint16:
        raise ValueError("Preview requires grid-shaped uint16 labels")
    validate_descriptor({"status": "unavailable", "nodata": 0, "classes": classes, "frames": []})
    lookup = np.zeros((65536, 4), dtype=np.uint8)
    for item in classes:
        lookup[item["id"]] = [*item["color"], 255]
    if not set(np.unique(labels)).issubset({0, *(item["id"] for item in classes)}):
        raise ValueError("Preview contains class IDs absent from descriptor")
    source_transform = grid.transform(ref["origin"])
    left, bottom, right, top = array_bounds(grid.height, grid.width, source_transform)
    # Densification matches rasterio's standard transformed-bounds convention.
    edge_x, edge_y = np.linspace(left, right, 23), np.linspace(bottom, top, 23)
    x = np.concatenate((edge_x, np.full(23, right), edge_x[::-1], np.full(23, left)))
    y = np.concatenate((np.full(23, bottom), edge_y, np.full(23, top), edge_y[::-1]))
    transformer = Transformer.from_crs(ref["wkt"], "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    lon, lat = np.asarray(lon), np.asarray(lat)
    if (not np.isfinite(lon).all() or not np.isfinite(lat).all()
            or (np.abs(lon) > 180).any() or (np.abs(lat) >= 90).any()):
        raise ValueError("Semantic preview geography is unusable")
    if np.ptp(lon) >= 180 or (np.abs(np.diff(lon)) >= 180).any():
        raise ValueError("Semantic preview crosses the antimeridian")
    bounds = transform_bounds(ref["wkt"], "EPSG:4326", left, bottom, right, top,
                              densify_pts=21)
    west, south, east, north = bounds
    if (not np.isfinite(bounds).all() or not -180 <= west < east <= 180
            or not -90 < south < north < 90 or east - west >= 180):
        raise ValueError("Semantic preview geographic bounds are unusable or cross the antimeridian")
    suggested, _, _ = calculate_default_transform(
        ref["wkt"], "EPSG:4326", grid.width, grid.height, left, bottom, right, top,
    )
    if (not np.isfinite([suggested.a, suggested.e]).all()
            or suggested.a <= 0 or suggested.e >= 0):
        raise ValueError("Semantic preview geographic resolution is unusable")
    width = max(1, int(np.ceil((east - west) / suggested.a)))
    height = max(1, int(np.ceil((north - south) / -suggested.e)))
    scale = min(1., long_side / max(width, height))
    width, height = max(1, int(width * scale)), max(1, int(height * scale))
    target = np.zeros((height, width), dtype=np.uint16)
    reproject(source=labels, destination=target, src_transform=source_transform,
              src_crs=ref["wkt"], src_nodata=0,
              dst_transform=from_bounds(*bounds, width, height), dst_crs="EPSG:4326",
              dst_nodata=0, resampling=Resampling.nearest)
    if not target.any():
        raise ValueError("Bounded geographic preview contains no labeled pixels")
    Image.fromarray(lookup[target]).save(path)
    return {"path": path.name, "crs": "EPSG:4326", "bounds": list(bounds),
            "width": width, "height": height}


def metadata_descriptor(*, classes=None, reason: str | None = None) -> dict[str, Any]:
    """Exact shared S9 status schema, including honest null absent products."""
    return {"id": "semantic", "title": "Semantic classes", "status": "unavailable",
            "reason": reason, "classes": classes or [], "nodata": 0,
            "raster_path": None, "confidence_path": None, "preview": None,
            "confidence_method": None, "provenance": None, "supported_cells": 0,
            "accuracy_status": "UNVALIDATED"}


def semantic_products(bundle, cfg, products, cameras, ref, *, mesh=None, masks=None,
                      mask_error: str | None = None):
    """Publish optional actual raster products without leaking failures into F14."""
    import json

    import numpy as np

    directory = bundle.stage_dir("s9_geo")
    outputs: dict[str, str] = {}
    semantics_cfg = getattr(cfg, "semantics", None)
    classes = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item
               for item in getattr(semantics_cfg, "classes", [])]
    metadata = metadata_descriptor(classes=classes)
    reasons: list[str] = []

    def relative(path):
        return path.relative_to(bundle.root).as_posix()

    def remove_output(path):
        for candidate in (path, path.with_suffix(path.suffix + ".msk")):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                reasons.append(f"{candidate.name} cleanup: {type(exc).__name__}: {exc}")

    try:
        # These are known generated products; remove previous-invocation outputs.
        for filename in ("semantic.tif", "semantic_confidence.tif", "semantic_preview.png"):
            path = directory / filename
            remove_output(path)
        source_path = bundle.artifact("s3_masking", "semantics")
        descriptor = validate_descriptor(bundle.read_json(relative(source_path)))
        metadata.update(classes=descriptor["classes"],
                        confidence_method=descriptor.get("confidence_method"),
                        provenance={"source_descriptor": relative(source_path),
                                    "source_status": descriptor["status"],
                                    "model": descriptor.get("model"),
                                    "source_labels": descriptor.get("source_labels"),
                                    "projection": None})
        if descriptor.get("reason"):
            reasons.append(str(descriptor["reason"]))
        if descriptor["status"] in {"disabled", "unavailable"}:
            metadata["status"] = descriptor["status"]
            reasons.append("S3 semantics " + descriptor["status"])
        elif mask_error is not None:
            reasons.append(mask_error)
        elif products is None or ref is None:
            reasons.append("Supported observed DSM/final georeference unavailable")
        else:
            labels, probability, projection = project_semantics(
                products, cameras, bundle.root, descriptor,
                min_probability=getattr(semantics_cfg, "min_probability", 0.5),
                mesh=mesh, masks=masks, chunk_size=cfg.dense.max_points_per_chunk,
            )
            metadata["provenance"]["projection"] = projection
            metadata["supported_cells"] = projection["supported_cells"]
            reasons.extend(f"{item['frame_id']}: {item['reason']}"
                           for item in projection["cameras_skipped"])
            if not projection["supported_cells"]:
                reasons.append("No labeled supported observed DSM cells")
            else:
                valid = labels != 0
                tags = {"semantic_classes": json.dumps(descriptor["classes"]),
                        "confidence_method": json.dumps(descriptor.get("confidence_method")),
                        "confidence_meaning": "winning source-class softmax probability; uncalibrated; not geometry confidence or absolute accuracy",
                        "model_provenance": json.dumps(descriptor.get("model")),
                        "source_labels": json.dumps(descriptor.get("source_labels")),
                        "semantic_provenance": json.dumps(metadata["provenance"]),
                        "inferred_excluded": "true", "unsupported_excluded": "true",
                        "accuracy_status": "UNVALIDATED", "sampling": "nearest"}
                for key, data, nodata, description, field in (
                    ("semantic", labels, 0, "semantic_class_id", "raster_path"),
                    ("semantic_confidence", np.where(valid, probability, -9999.).astype(np.float32),
                     -9999., "winning_source_class_probability_uncalibrated", "confidence_path"),
                ):
                    path = directory / f"{key}.tif"
                    try:
                        write_geotiff(path, data, products["grid"], ref, nodata=nodata,
                                      valid=valid, descriptions=(description,), tags=tags)
                        outputs[key] = relative(path)
                        metadata[field] = outputs[key]
                    except Exception as exc:
                        remove_output(path)
                        reasons.append(f"{key}: {type(exc).__name__}: {exc}")
                if metadata["raster_path"] is not None:
                    metadata["status"] = "ok"
                    path = directory / "semantic_preview.png"
                    try:
                        preview = geographic_preview(
                            path, labels, products["grid"], ref, descriptor["classes"],
                            long_side=getattr(semantics_cfg, "preview_long_side", 1024),
                        )
                        preview["path"] = relative(path)
                        metadata["preview"] = preview
                        outputs["semantic_preview"] = relative(path)
                    except Exception as exc:
                        remove_output(path)
                        reasons.append(f"semantic_preview: {type(exc).__name__}: {exc}")
    except Exception as exc:
        # A semantic reader/model/map/warp failure must preserve mandatory rasters.
        reasons.append(f"Semantics unavailable: {type(exc).__name__}: {exc}")
    metadata["reason"] = "; ".join(reasons) or None
    return outputs, metadata
