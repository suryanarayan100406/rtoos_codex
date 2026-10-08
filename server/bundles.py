"""Read-only bundle access with confinement applied to every artifact and symlink."""

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import HTTPException


def confined(root: Path, relative: str, *, exists: bool = True) -> Path:
    if not relative or "\\" in relative or "\x00" in relative:
        raise HTTPException(403, "Invalid relative artifact path")
    name = PurePosixPath(relative)
    if name.is_absolute() or any(p in ("..", ".") for p in name.parts):
        raise HTTPException(403, "Artifact path escapes its root")
    base = root.resolve()
    try:
        result = (base / str(name)).resolve()
    except (RuntimeError, OSError) as exc:
        raise HTTPException(403, "Artifact symlink cannot be resolved safely") from exc
    if not result.is_relative_to(base):
        raise HTTPException(403, "Artifact symlink escapes its root")
    if exists and not result.exists():
        raise HTTPException(404, "Artifact not found")
    return result


def read_json(path: Path, max_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        raise HTTPException(404, "JSON artifact not found")
    if path.stat().st_size > max_bytes:
        raise HTTPException(413, "JSON artifact exceeds configured size limit")
    try:
        # Reject NaN/Infinity rather than letting non-standard JSON propagate.
        def bad_constant(value: str) -> None:
            raise ValueError(f"Non-finite JSON constant: {value}")

        def finite_float(value: str) -> float:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError("Non-finite JSON number")
            return parsed

        result = json.loads(path.read_text(encoding="utf-8"), parse_constant=bad_constant,
                            parse_float=finite_float)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result
    except (ValueError, OSError) as exc:
        raise HTTPException(409, f"Unreadable JSON artifact: {exc}") from exc


def stage(manifest: dict, name: str) -> dict:
    stages = manifest.get("stages") or {}
    if isinstance(stages, dict):
        return stages.get(name, {})
    return next((s for s in stages if s.get("name") == name), {})


def artifact(manifest: dict, name: str, key: str) -> str | None:
    value = stage(manifest, name).get("outputs", {}).get(key)
    if isinstance(value, dict):
        value = value.get("path")
    return value if isinstance(value, str) else None


def semantic_descriptor(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    """Validate optional S9 metadata before any of its files reach the viewer.

    This validates the display contract, not raster contents or semantic accuracy.
    Preview dimensions use the same hard ceiling as SemanticsConfig.
    """
    status = descriptor.get("status")
    if status not in ("ok", "unavailable", "disabled"):
        raise ValueError("semantic status must be ok, unavailable or disabled")
    if descriptor.get("id") != "semantic" or descriptor.get("title") != "Semantic classes":
        raise ValueError("semantic id/title do not match the S9 descriptor contract")
    reason = descriptor.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("semantic reason must be nonempty text or null")
    if status != "ok" and not reason:
        raise ValueError("unavailable/disabled semantics must explain their reason")
    if type(descriptor.get("nodata")) is not int or descriptor["nodata"] != 0:
        raise ValueError("semantic nodata must be class ID 0")
    supported = descriptor.get("supported_cells")
    if type(supported) is not int or supported < 0 or (status == "ok" and supported == 0):
        raise ValueError("supported_cells must be a nonnegative integer, positive for ok semantics")
    if descriptor.get("accuracy_status") != "UNVALIDATED":
        raise ValueError("semantic accuracy_status must be UNVALIDATED")
    provenance = descriptor.get("provenance")
    if not isinstance(provenance, dict) and not (provenance is None and status != "ok"):
        raise ValueError("semantic provenance must be an object")
    method = descriptor.get("confidence_method")
    if method is not None and (not isinstance(method, str) or not method.strip()):
        raise ValueError("semantic confidence_method must be nonempty text or null")
    if status == "ok" and (not method or "uncalibrated" not in method.casefold()
                           or "geometry" in method.casefold() or not any(
                               word in method.casefold()
                               for word in ("probability", "probabilities", "softmax")
                           )):
        raise ValueError("semantic confidence_method must describe uncalibrated semantic probability, independent of geometry confidence")

    classes = descriptor.get("classes")
    if not isinstance(classes, list) or (status == "ok" and not classes):
        raise ValueError("semantic classes must be a list, nonempty for ok semantics")
    ids: set[int] = set()
    names: set[str] = set()
    sources: set[int] = set()
    for entry in classes:
        if not isinstance(entry, dict):
            raise ValueError("semantic class entries must be objects")
        class_id, name, color = entry.get("id"), entry.get("name"), entry.get("color")
        if type(class_id) is not int or not 1 <= class_id <= 65534 or class_id in ids:
            raise ValueError("semantic class IDs must be unique integers in 1..65534")
        if not isinstance(name, str) or not name.strip() or name.strip().casefold() in names:
            raise ValueError("semantic class names must be nonempty and unique")
        if not isinstance(color, list) or len(color) != 3 or any(
            type(channel) is not int or not 0 <= channel <= 255 for channel in color
        ):
            raise ValueError("semantic class color must contain three integer RGB values in 0..255")
        ids.add(class_id)
        names.add(name.strip().casefold())
        if "source_ids" in entry:
            source_ids = entry["source_ids"]
            if not isinstance(source_ids, list) or not source_ids:
                raise ValueError("semantic source_ids must be a nonempty list when provided")
            for source_id in source_ids:
                if type(source_id) is not int or source_id < 0 or source_id in sources:
                    raise ValueError("semantic source IDs must be nonnegative, unique checkpoint channels")
                sources.add(source_id)

    def file_reference(value: Any, label: str, suffixes: tuple[str, ...], *,
                       required: bool = False) -> None:
        if value is None and not required:
            return
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"semantic {label} must name a real bundle-relative file")
        path = confined(root, value)
        if not path.is_file():
            raise ValueError(f"semantic {label} is not a file")
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"semantic {label} has an unsupported file type")

    file_reference(descriptor.get("raster_path"), "raster_path", (".tif", ".tiff"),
                   required=status == "ok")
    file_reference(descriptor.get("confidence_path"), "confidence_path", (".tif", ".tiff"))
    preview = descriptor.get("preview")
    if preview is not None:
        if not isinstance(preview, dict) or preview.get("crs") != "EPSG:4326":
            raise ValueError("semantic preview must declare EPSG:4326")
        file_reference(preview.get("path"), "preview.path", (".png",), required=True)
        bounds = preview.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in bounds
        ):
            raise ValueError("semantic preview bounds must be finite [west,south,east,north]")
        west, south, east, north = bounds
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("semantic preview bounds are outside valid longitude/latitude ranges or reversed")
        if any(type(preview.get(key)) is not int or not 1 <= preview[key] <= 4096
               for key in ("width", "height")):
            raise ValueError("semantic preview dimensions must be positive integers no larger than 4096")

    result = dict(descriptor)
    if status != "ok":
        # Even retained files in a non-ok descriptor cannot enable imagery/downloads.
        result.update(raster_path=None, confidence_path=None, preview=None)
    return result


class BundleStore:
    def __init__(self, root: Path, max_json_bytes: int):
        self.root = root.resolve()
        self.max_json_bytes = max_json_bytes

    def bundle(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise HTTPException(404, "Invalid run id")
        path = confined(self.root, run_id)
        if not path.is_dir():
            raise HTTPException(404, "Run not found")
        return path

    def manifest(self, run_id: str) -> dict:
        manifest = read_json(confined(self.bundle(run_id), "manifest.json"), self.max_json_bytes)
        records = manifest.get("stages") or {}
        if not isinstance(records, dict | list):
            raise HTTPException(409, "Manifest stages must be a dict or list")
        values = records.values() if isinstance(records, dict) else records
        if any(not isinstance(record, dict) or not isinstance(record.get("outputs", {}), dict)
               for record in values):
            raise HTTPException(409, "Manifest contains an invalid stage record")
        if manifest.get("crs") is not None and not isinstance(manifest["crs"], dict):
            raise HTTPException(409, "Manifest CRS must be an object")
        return manifest

    def json_artifact(self, run_id: str, path: str) -> dict:
        return read_json(confined(self.bundle(run_id), path), self.max_json_bytes)

    def report(self, run_id: str, manifest: dict | None = None) -> dict:
        data = manifest if manifest is not None else self.manifest(run_id)
        path = artifact(data, "report", "json")
        if not path:
            # Failure reports can precede the report-stage manifest entry.
            path = "report/report.json"
        return self.json_artifact(run_id, path)

    def semantic_layers(self, run_id: str, manifest: dict) -> tuple[list[dict], str | None]:
        path = artifact(manifest, "s9_geo", "semantic_metadata")
        if path is None:
            return [], "No S9 semantic metadata is recorded in this bundle."
        try:
            descriptor = self.json_artifact(run_id, path)
            validated = semantic_descriptor(self.bundle(run_id), descriptor)
        except HTTPException as exc:
            # Optional unsafe paths remain explicitly rejected, without blocking geometry.
            return [], f"Semantic layers unavailable: {exc.detail} (HTTP {exc.status_code})."
        except (ValueError, OSError, OverflowError) as exc:
            return [], f"Semantic layers unavailable: invalid S9 descriptor: {exc}"
        if validated["status"] != "ok":
            return [validated], validated["reason"]
        missing = []
        if validated.get("preview") is None:
            missing.append("geographic preview")
        if validated.get("confidence_path") is None:
            missing.append("semantic probability raster")
        reason = validated.get("reason")
        if missing:
            availability = (f"Semantic {', '.join(missing)} unavailable; "
                            "the class GeoTIFF remains downloadable.")
            reason = f"{reason} {availability}" if reason else availability
        return [validated], reason

    def scene(self, run_id: str, manifest: dict | None = None) -> dict:
        data = manifest if manifest is not None else self.manifest(run_id)
        crs = dict(data.get("crs") or {})
        origin = crs.get("origin")
        pose_path = artifact(data, "s6_global", "poses") or artifact(data, "s2_poses", "poses")
        if pose_path:
            poses = self.json_artifact(run_id, pose_path)
            origin = poses.get("origin", origin)
            crs.update(poses.get("crs") or {})
        exports = {}
        exports_path = artifact(data, "s10_export", "exports")
        if exports_path:
            exports = self.json_artifact(run_id, exports_path)
            georef = exports.get("georeference") or {}
            crs.update(georef.get("reference") or {})
            crs.update({k: v for k, v in georef.items() if v is not None})
            origin = georef.get("origin") or origin
        if crs.get("wkt") or crs.get("epsg"):
            try:
                from pyproj import CRS

                parsed = CRS.from_user_input(crs.get("wkt") or crs["epsg"])
                horizontal = parsed.sub_crs_list[0] if parsed.is_compound else parsed
                # Keep source WKT2/datum unchanged. proj4.js's WKT parser needs
                # a derived WKT1 definition for many custom projected CRSs.
                crs["viewer_wkt1"] = horizontal.to_wkt(version="WKT1_GDAL")
            except ImportError:
                pass  # Known EPSG transforms remain available without pyproj.
            except (ValueError, RuntimeError) as exc:
                crs["viewer_transform_error"] = str(exc)
        scene_stages = ("s6_global", "s7_dense", "s8_mesh", "s9_geo", "s10_export")
        semantic_layers, semantic_reason = self.semantic_layers(run_id, data)
        return {"coordinate_frame": "projected_local", "origin": origin, "crs": crs,
                "revision": {name: stage(data, name).get("outputs_hash") for name in scene_stages},
                "stage_status": {name: stage(data, name).get("status") for name in scene_stages},
                "semantic_layers": semantic_layers, "semantic_reason": semantic_reason,
                "exports": exports, "artifacts": {
                    "cloud": artifact(data, "s7_dense", "cloud"),
                    "ply": artifact(data, "s7_dense", "ply"),
                    "mesh": artifact(data, "s8_mesh", "mesh"),
                    "dsm": artifact(data, "s9_geo", "dsm"),
                    "confidence": artifact(data, "s9_geo", "confidence"),
                }}
