"""Bounded previews of real point arrays. PLY is streamed, not loaded wholesale."""

import math
import struct
import zipfile
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from server.bundles import BundleStore, artifact, confined

PLY_TYPES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h",
             "int16": "h", "ushort": "H", "uint16": "H", "int": "i", "int32": "i",
             "uint": "I", "uint32": "I", "float": "f", "float32": "f",
             "double": "d", "float64": "d"}


def ply_preview(path: Path, limit: int) -> tuple[dict, int]:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        properties = []
        count = 0
        current = None
        encoding = None
        seen_vertex = False
        header_bytes = 0
        while True:
            line = stream.readline(8192)
            header_bytes += len(line)
            if not line or header_bytes > 65536:
                raise ValueError("Missing or oversized PLY header")
            fields = line.decode("ascii").strip().split()
            if not fields:
                continue
            if fields[0] == "format":
                encoding = fields[1]
            elif fields[0] == "element":
                current = fields[1]
                if current == "vertex":
                    seen_vertex = True
                    count = int(fields[2])
                elif not seen_vertex and int(fields[2]) > 0:
                    raise ValueError("Preview requires vertex as the first populated PLY element")
            elif fields[0] == "property" and current == "vertex":
                if fields[1] == "list" or fields[1] not in PLY_TYPES:
                    raise ValueError("Unsupported PLY vertex property")
                properties.append((fields[2], fields[1]))
            elif fields[0] == "end_header":
                break
        names = [p[0] for p in properties]
        if count < 0 or not all(axis in names for axis in ("x", "y", "z")):
            raise ValueError("PLY lacks valid vertex XYZ fields")
        stride = max(1, math.ceil(count / limit))
        if len(set(names)) != len(names):
            raise ValueError("Duplicate PLY vertex properties")
        arrays: dict[str, list[float]] = {name: [] for name in names}
        if encoding in ("binary_little_endian", "binary_big_endian"):
            record = struct.Struct(("<" if encoding == "binary_little_endian" else ">") +
                                   "".join(PLY_TYPES[kind] for _, kind in properties))
            offset = stream.tell()
            if path.stat().st_size < offset + count * record.size:
                raise ValueError("Truncated binary PLY")
            for index in range(0, count, stride):
                stream.seek(offset + index * record.size)
                values = record.unpack(stream.read(record.size))
                if all(math.isfinite(float(values[names.index(a)])) for a in ("x", "y", "z")):
                    for name, value in zip(names, values, strict=True):
                        arrays[name].append(value)
        elif encoding == "ascii":
            for index in range(count):
                line = stream.readline()
                if not line:
                    raise ValueError("Truncated ASCII PLY")
                if index % stride:
                    continue
                values = tuple(float(v) for v in line.split())
                if len(values) != len(names):
                    raise ValueError("Malformed PLY vertex")
                if all(math.isfinite(values[names.index(a)]) for a in ("x", "y", "z")):
                    for name, value in zip(names, values, strict=True):
                        arrays[name].append(value)
        else:
            raise ValueError("Unsupported PLY encoding")
        points = np.array([arrays[a] for a in ("x", "y", "z")]).T.reshape(-1, 3)
        colors = None
        if all(a in arrays for a in ("red", "green", "blue")):
            colors = np.array([arrays[a] for a in ("red", "green", "blue")]).T
        return {"points": points, "colors": colors,
                "confidence": np.array(arrays["confidence"]) if "confidence" in arrays else None,
                "inferred": np.array(arrays["inferred"]) if "inferred" in arrays else None}, count


def npz_preview(path: Path, limit: int, max_bytes: int) -> tuple[dict, int]:
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > max_bytes:
            raise HTTPException(413, "NPZ exceeds preview memory limit; provide a streaming PLY or Potree export")
    with np.load(path, allow_pickle=False) as arrays:
        points = arrays["points"]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Point array must be Nx3")
        count = len(points)
        indices = np.arange(0, count, max(1, math.ceil(count / limit)))
        valid = np.isfinite(points[indices]).all(axis=1)
        indices = indices[valid]
        result = {"points": points[indices]}
        for key in ("colors", "confidence", "inferred"):
            if key in arrays:
                values = arrays[key]
                expected_shape = (count, 3) if key == "colors" else (count,)
                if values.shape != expected_shape:
                    raise ValueError(f"{key} does not align with points")
                result[key] = values[indices]
            else:
                result[key] = None
        return result, count


def preview(store: BundleStore, run_id: str, limit: int, max_bytes: int) -> dict:
    manifest = store.manifest(run_id)
    relative = artifact(manifest, "s7_dense", "ply") or artifact(manifest, "s7_dense", "cloud")
    if not relative:
        raise HTTPException(404, "No real dense cloud artifact is available")
    path = confined(store.bundle(run_id), relative)
    try:
        arrays, total = (ply_preview(path, limit) if path.suffix.lower() == ".ply"
                         else npz_preview(path, limit, max_bytes))
        points = arrays["points"]
        result = {"points": points.tolist(), "total_points": total,
                  "preview_points": len(points), "source": relative,
                  "coordinate_frame": "projected_local_metres",
                  "decimation": "deterministic_stride",
                  "bounds_scope": "preview_only",
                  "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()] if len(points) else None}
        for key in ("colors", "confidence", "inferred"):
            values = arrays.get(key)
            if values is None:
                result[key] = None
            elif key == "inferred":
                result[key] = [bool(v) if math.isfinite(float(v)) else None for v in values]
            elif key == "confidence":
                result[key] = [float(v) if math.isfinite(float(v)) else None for v in values]
            else:
                if values.shape != (len(points), 3) or not np.isfinite(values).all():
                    raise ValueError("Colors must be finite Nx3 values")
                result[key] = np.clip(values, 0, 255).astype(int).tolist()
        return result
    except HTTPException:
        raise
    except (ValueError, TypeError, IndexError, KeyError, OSError, zipfile.BadZipFile, struct.error) as exc:
        raise HTTPException(409, f"Cannot preview cloud: {exc}") from exc
