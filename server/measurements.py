"""Metric local-coordinate measurements; no absolute accuracy inferred from geometry."""

import math
from typing import Annotated, Literal

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from server.bundles import BundleStore, artifact, confined

MetricFloat = Annotated[FiniteFloat, Field(strict=True)]


class MeasurementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["distance", "area", "height", "volume"]
    points: list[tuple[MetricFloat, MetricFloat, MetricFloat]] = Field(min_length=2, max_length=4096)
    reference_height: MetricFloat | None = None


def cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def polygon_area(points) -> float:
    # Translate before arithmetic to retain precision for large user coordinates.
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    xy = xy - xy[0]
    return abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                     - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))) / 2


def polygon(points) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    if np.array_equal(xy[0], xy[-1]):
        xy = xy[:-1]
    if len(xy) < 3:
        raise HTTPException(422, "A polygon needs at least three distinct XY vertices")
    extent = float(np.max(np.ptp(xy, axis=0)))
    if not math.isfinite(extent):
        raise HTTPException(422, "Polygon coordinate magnitudes overflow local geometry")
    tolerance = max(extent * 1e-12, 1e-10)
    if any(np.linalg.norm(xy[i] - xy[j]) <= tolerance
           for i in range(len(xy)) for j in range(i)):
        raise HTTPException(422, "Polygon repeats an XY vertex")

    def intersects(a, b, c, d):
        eps = tolerance * max(extent, 1)
        orientations = (cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b))
        o1, o2, o3, o4 = orientations
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        for o, p, q, r in ((o1, a, b, c), (o2, a, b, d), (o3, c, d, a), (o4, c, d, b)):
            if abs(o) <= eps and np.all(r >= np.minimum(p, q) - tolerance) and np.all(
                    r <= np.maximum(p, q) + tolerance):
                return True
        return False

    n = len(xy)
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        # Adjacent collinear edges doubling back overlap too.
        c = xy[(i + 2) % n]
        if abs(cross(a, b, c)) <= tolerance * max(extent, 1) and np.dot(a - b, c - b) > 0:
            raise HTTPException(422, "Polygon has overlapping edges")
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if intersects(a, b, xy[j], xy[(j + 1) % n]):
                raise HTTPException(422, "Polygon self-intersects or has touching non-adjacent edges")
    area = polygon_area(xy)
    if not math.isfinite(area):
        raise HTTPException(422, "Polygon area overflows local geometry")
    if area <= tolerance * max(extent, 1):
        raise HTTPException(422, "Polygon has zero horizontal area")
    return xy


def clipped_area(subject, cell) -> float:
    """Signed polygon clipping against a convex raster cell, including partial cells."""
    vertices = [np.asarray(p, dtype=np.float64) for p in subject]
    if cross(cell[0], cell[1], cell[2]) < 0:
        cell = cell[::-1]
    for a, b in zip(cell, np.roll(cell, -1, axis=0), strict=True):
        result = []
        if not vertices:
            return 0.0
        previous = vertices[-1]
        prev_cross = cross(a, b, previous)
        for current in vertices:
            current_cross = cross(a, b, current)
            if (current_cross >= 0) != (prev_cross >= 0):
                result.append(previous + (current - previous) *
                              (prev_cross / (prev_cross - current_cross)))
            if current_cross >= 0:
                result.append(current)
            previous, prev_cross = current, current_cross
        vertices = result
    return polygon_area(vertices) if len(vertices) >= 3 else 0.0


def accuracy(store: BundleStore, run_id: str, manifest: dict) -> dict:
    try:
        report = store.report(run_id, manifest)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        report = {}
    # Include the report's evidence as-is; a GPS fitting residual is never promoted.
    evidence = report.get("accuracy", report.get("validation", {}))
    status = evidence.get("status", "UNVALIDATED") if isinstance(evidence, dict) else "UNVALIDATED"
    return {"status": status, "absolute_uncertainty_m": None,
            "report_evidence": evidence,
            "note": "Geometry measurements are not independent absolute-accuracy validation."}


def volume(store: BundleStore, run_id: str, manifest: dict, xy: np.ndarray,
           reference: float, max_cells: int) -> tuple[float, dict]:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise HTTPException(503, "Volume integration requires local rasterio") from exc
    dsm = artifact(manifest, "s9_geo", "dsm")
    if not dsm:
        raise HTTPException(409, "No DSM is available for volume integration")
    scene = store.scene(run_id, manifest)
    origin = scene.get("origin")
    if (not isinstance(origin, list) or len(origin) != 3 or
        not all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in origin)):
        raise HTTPException(409, "Volume requires a recorded finite projected XYZ origin")
    metadata_path = artifact(manifest, "s9_geo", "metadata")
    metadata = store.json_artifact(run_id, metadata_path) if metadata_path else {}
    path = confined(store.bundle(run_id), dsm)
    try:
        dataset = rasterio.open(path)
    except rasterio.errors.RasterioError as exc:
        raise HTTPException(409, f"Cannot read DSM: {exc}") from exc
    with dataset as src:
        tags = src.tags()
        if not src.crs or not src.crs.is_projected or src.crs.linear_units_factor[1] != 1:
            raise HTTPException(409, "DSM must use a projected CRS with metre units")
        epsg = scene["crs"].get("epsg")
        if not epsg or src.crs != rasterio.crs.CRS.from_epsg(int(epsg)):
            raise HTTPException(409, "DSM CRS does not match the recorded local-coordinate CRS")
        exclusion = tags.get("inferred_excluded", metadata.get("inferred_excluded"))
        # With no exclusion provenance or explicit mask we cannot call the surface observed.
        inferred_path = artifact(manifest, "s9_geo", "inferred")
        mask_path = artifact(manifest, "s9_geo", "mask")
        support_path = artifact(manifest, "s9_geo", "support")
        if str(exclusion).lower() not in ("true", "1", "yes") and not inferred_path and not mask_path:
            raise HTTPException(409, "DSM lacks non-inferred support provenance; volume is unavailable")
        elevation_frame = tags.get("elevation_frame", metadata.get("elevation_frame"))
        if elevation_frame not in ("projected_absolute", "projected_local"):
            raise HTTPException(409, "DSM elevation coordinate frame is unsupported")
        absolute_xy = xy + np.asarray(origin[:2])
        inverse = ~src.transform
        corners = [inverse * (x, y) for x in (absolute_xy[:, 0].min(), absolute_xy[:, 0].max())
                   for y in (absolute_xy[:, 1].min(), absolute_xy[:, 1].max())]
        c0 = max(0, math.floor(min(c[0] for c in corners)))
        r0 = max(0, math.floor(min(c[1] for c in corners)))
        c1 = min(src.width, math.ceil(max(c[0] for c in corners)))
        r1 = min(src.height, math.ceil(max(c[1] for c in corners)))
        count = max(0, c1 - c0) * max(0, r1 - r0)
        if count > max_cells:
            raise HTTPException(413, "Volume polygon exceeds configured raster-cell limit")
        area = polygon_area(xy)
        supported_area = nodata_area = inferred_area = raster_area = total = 0.0
        supported_cells = unsupported_cells = inferred_cells = nodata_cells = 0
        confidence_sum = confidence_area = 0.0
        if count:
            window = Window(c0, r0, c1 - c0, r1 - r0)

            def read_aligned(relative: str, label: str):
                try:
                    with rasterio.open(confined(store.bundle(run_id), relative)) as raster:
                        if (raster.crs, raster.transform, raster.shape) != (src.crs, src.transform, src.shape):
                            raise HTTPException(409, f"DSM {label} raster is not grid-aligned")
                        return raster.read(1, window=window, masked=True)
                except rasterio.errors.RasterioError as exc:
                    raise HTTPException(409, f"Cannot read DSM {label}: {exc}") from exc

            heights = read_aligned(dsm, "elevation")
            inferred = np.zeros(heights.shape, dtype=bool)
            observed = np.ones(heights.shape, dtype=bool)
            confidence = None
            # S9's mask contract: 0 unsupported, 1 observed, 2 inferred-only.
            if mask_path:
                flags = read_aligned(mask_path, "support mask").filled(0)
                observed = flags == 1
                inferred = flags == 2
            if support_path:
                observed &= read_aligned(support_path, "support counts").filled(0) > 0
            confidence_path = artifact(manifest, "s9_geo", "confidence")
            if confidence_path:
                confidence = read_aligned(confidence_path, "confidence")
            if inferred_path:
                flags = read_aligned(inferred_path, "inference mask")
                inferred |= np.ma.getmaskarray(flags) | (flags.filled(1) != 0)
            for row in range(heights.shape[0]):
                for col in range(heights.shape[1]):
                    cell = np.array([src.transform * (col + c0 + dx, row + r0 + dy)
                                     for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]) - origin[:2]
                    overlap = clipped_area(xy, cell)
                    if overlap <= 0:
                        continue
                    raster_area += overlap
                    if inferred[row, col]:
                        inferred_area += overlap
                        inferred_cells += 1
                        unsupported_cells += 1
                    elif (not observed[row, col] or np.ma.is_masked(heights[row, col])
                          or not np.isfinite(heights[row, col])):
                        nodata_area += overlap
                        nodata_cells += 1
                        unsupported_cells += 1
                    else:
                        z = float(heights[row, col])
                        local_z = z - origin[2] if elevation_frame == "projected_absolute" else z
                        total += max(0, local_z - reference) * overlap
                        supported_area += overlap
                        supported_cells += 1
                        if confidence is not None and not np.ma.is_masked(confidence[row, col]):
                            conf_value = float(confidence[row, col])
                            if math.isfinite(conf_value) and 0 <= conf_value <= 1:
                                confidence_sum += conf_value * overlap
                                confidence_area += overlap
        if not supported_cells:
            raise HTTPException(409, "Polygon contains no supported non-inferred DSM cells")
        return total, {"polygon_area_m2": area, "supported_area_m2": supported_area,
                       "coverage_fraction": min(1.0, supported_area / area),
                       "nodata_area_m2": nodata_area,
                       "inferred_excluded_area_m2": inferred_area if mask_path or inferred_path else None,
                       "outside_raster_area_m2": max(0.0, area - raster_area),
                       "unsupported_area_m2": max(0.0, area - supported_area),
                       "supported_cells": supported_cells,
                       "unsupported_cells_in_raster": unsupported_cells,
                       "nodata_cells": nodata_cells,
                       "inferred_excluded_cells": inferred_cells if mask_path or inferred_path else None,
                       "mean_source_confidence": confidence_sum / confidence_area if confidence_area else None,
                       "confidence_is_absolute_accuracy": False,
                       "candidate_cells": count, "inferred_excluded": True,
                       "cell_elevation_model": "piecewise_constant_DSM",
                       "boundary_method": "polygon_cell_overlap_area",
                       "reference_height_local_m": reference,
                       "vertical_datum": scene["crs"].get("vertical"),
                       "surface": dsm}


def measure(store: BundleStore, run_id: str, request: MeasurementRequest,
            max_points: int, max_cells: int) -> dict:
    if len(request.points) > max_points:
        raise HTTPException(413, "Measurement exceeds configured point limit")
    manifest = store.manifest(run_id)
    points = np.asarray(request.points, dtype=np.float64)
    support = {"picked_points": len(points), "surface_support": "not_verified",
               "note": "Picked coordinates do not establish observed geometry or absolute accuracy."}
    warnings = ["Independent absolute accuracy is not established by this measurement."]
    if request.kind == "distance":
        value = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        method, units = "local_XYZ_polyline_length", "m"
    elif request.kind == "height":
        value = float(np.ptp(points[:, 2]))
        method, units = "local_Z_max_minus_min", "m"
    else:
        xy = polygon(points)
        if request.kind == "area":
            value, method, units = polygon_area(xy), "horizontal_XY_polygon_area", "m²"
        else:
            if request.reference_height is None:
                raise HTTPException(422, "Volume requires an explicit local reference_height in metres")
            value, support = volume(store, run_id, manifest, xy,
                                    request.reference_height, max_cells)
            method, units = "supported_non_inferred_DSM_above_local_reference", "m³"
            warnings.append("Volume covers supported cells only; gaps and inferred cells contribute nothing.")
    if not math.isfinite(value):
        raise HTTPException(422, "Measurement overflow; coordinate magnitudes are too large")
    return {"kind": request.kind, "value": value, "units": units, "method": method,
            "coordinate_frame": "projected_local_metres", "support": support,
            "uncertainty": accuracy(store, run_id, manifest), "warnings": warnings}
