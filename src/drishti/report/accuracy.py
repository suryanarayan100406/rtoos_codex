"""Independent DSM elevation checks, deliberately not full 3D accuracy.

CSV rows declare `independent=true`, `vertical_datum`, optional `id`, and either
`longitude,latitude,altitude_m` (WGS84, optional epsg=4326), or
`x,y,z,epsg` / `easting,northing,elevation_m,epsg` (projected metres).
No vertical conversion or checkpoint-derived correction is performed here.
"""

import csv
import math
from pathlib import Path
from typing import Any


def validate_checkpoints(path: Path, dsm_path: Path, *, target_m: float | None) -> dict[str, Any]:
    import numpy as np
    import pyproj
    import rasterio
    from pyproj import CRS, Transformer

    if target_m is not None and (not math.isfinite(target_m) or target_m <= 0):
        raise ValueError("A supplied checkpoint accuracy target must be finite and positive")
    pyproj.network.set_network_enabled(False)
    result: dict[str, Any] = {"status": "UNVALIDATED", "method": "independent_DSM_cell_elevation_comparison",
              "scope": "elevation_only", "full_3d_accuracy": "UNVALIDATED",
              "horizontal_rmse_m": None, "spatial_rmse_m": None,
              "rmse_z_m": None, "bias_z_m": None, "mae_z_m": None,
              "max_absolute_z_error_m": None, "target_m": target_m,
              "target_met_elevation_only": None, "checkpoints": [], "excluded": [],
              "confidence_95_m": None,
              "confidence_95_reason": "No distribution/sample-size certification is inferred"}
    seen = set()
    with rasterio.open(dsm_path) as dsm, path.open(newline="", encoding="utf-8-sig") as handle:
        if not dsm.crs:
            raise ValueError("DSM lacks a CRS; checkpoint coordinates cannot be transformed")
        raster_crs = CRS.from_user_input(dsm.crs)
        if not raster_crs.is_projected or any(
            not np.isclose(axis.unit_conversion_factor, 1.) for axis in raster_crs.axis_info[:2]
        ):
            raise ValueError("DSM checkpoint validation requires a metre projected CRS")
        if dsm.tags().get("elevation_frame") != "projected_absolute":
            raise ValueError("DSM checkpoint elevations must explicitly use the projected_absolute frame")
        datum = dsm.tags().get("vertical_datum", "unknown")
        result["vertical_datum"] = datum
        result["dsm_crs"] = raster_crs.to_string()
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Checkpoint CSV has no header")
        for line, row in enumerate(reader, start=2):
            identity = row.get("id") or f"row_{line}"
            try:
                if identity in seen:
                    raise ValueError("Duplicate checkpoint id")
                seen.add(identity)
                if str(row.get("independent", "")).strip().lower() not in ("true", "1", "yes"):
                    raise ValueError("Checkpoint independence must be explicitly declared true")
                declared_datum = str(row.get("vertical_datum", "")).strip()
                if not declared_datum or declared_datum.lower() == "unknown" or datum == "unknown":
                    raise ValueError("A known matching vertical datum is required")
                if declared_datum != datum:
                    raise ValueError(f"Vertical datum mismatch: checkpoint {declared_datum}, DSM {datum}")
                if row.get("longitude") is not None and row.get("latitude") is not None:
                    x, y, z = float(row["longitude"]), float(row["latitude"]), float(row["altitude_m"])
                    if not -180 <= x <= 180 or not -90 <= y <= 90:
                        raise ValueError("Geographic checkpoint is outside longitude/latitude bounds")
                    epsg = int(row.get("epsg") or 4326)
                    if epsg != 4326:
                        raise ValueError("Longitude/latitude rows require WGS84 epsg=4326")
                    source = CRS.from_epsg(epsg)
                else:
                    x = float(row["x"] if "x" in row else row["easting"])
                    y = float(row["y"] if "y" in row else row["northing"])
                    z = float(row["z"] if "z" in row else row["elevation_m"])
                    source = CRS.from_epsg(int(row["epsg"]))
                    if not source.is_projected or any(
                        not np.isclose(axis.unit_conversion_factor, 1.) for axis in source.axis_info[:2]
                    ):
                        raise ValueError("Projected checkpoints require a metre projected EPSG")
                if not all(math.isfinite(value) for value in (x, y, z)):
                    raise ValueError("Checkpoint coordinates/elevation must be finite")
                transform = Transformer.from_crs(source, raster_crs, always_xy=True, allow_ballpark=False)
                # XY only: transforming Z could silently change the declared datum.
                x, y = transform.transform(x, y, errcheck=True)
                if not all(math.isfinite(value) for value in (x, y)):
                    raise ValueError("Checkpoint horizontal transform is nonfinite")
                raster_row, raster_col = dsm.index(x, y)
                if not (0 <= raster_row < dsm.height and 0 <= raster_col < dsm.width):
                    raise ValueError("Checkpoint is outside the DSM extent")
                sample = dsm.read(1, window=((raster_row, raster_row + 1),
                                             (raster_col, raster_col + 1)), masked=True)
                if np.ma.getmaskarray(sample)[0, 0] or not np.isfinite(sample[0, 0]):
                    raise ValueError("Checkpoint falls on unsupported/nodata DSM")
                predicted = float(sample[0, 0])
                error = predicted - z
                result["checkpoints"].append({"id": identity, "x": x, "y": y,
                    "reference_z_m": z, "dsm_z_m": predicted, "error_z_m": error,
                    "exceeds_target": abs(error) > target_m if target_m is not None else None,
                    "independence": "user_declared",
                    "raster_cell": [raster_row, raster_col]})
            except (KeyError, TypeError, ValueError, pyproj.exceptions.ProjError) as exc:
                result["excluded"].append({"id": identity, "line": line, "reason": str(exc)})
    errors = np.asarray([row["error_z_m"] for row in result["checkpoints"]])
    result["valid_count"] = len(errors)
    result["excluded_count"] = len(result["excluded"])
    if errors.size:
        result.update(status="MEASURED", rmse_z_m=float(np.sqrt(np.mean(errors**2))),
                      bias_z_m=float(errors.mean()), mae_z_m=float(np.abs(errors).mean()),
                      max_absolute_z_error_m=float(np.abs(errors).max()),
                      target_met_elevation_only=bool(np.abs(errors).max() <= target_m)
                      if target_m is not None else None)
    else:
        result["reason"] = "No independent, datum-compatible checkpoint overlaps supported DSM"
    return result
