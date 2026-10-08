"""GPS-derived projected CRS and float64 local coordinates with height provenance."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from drishti.errors import DependencyError, InputError


def utm_epsg(latitude: float, longitude: float) -> int:
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise InputError("UTM derivation requires finite coordinates")
    if not -80 <= latitude <= 84:
        raise InputError("Latitude is outside UTM [-80,84]; supply an explicit projected EPSG override")
    if not -180 <= longitude <= 180:
        raise InputError("Longitude is outside [-180,180]")
    # +180 belongs to zone 60, rather than the nonexistent zone 61.
    zone = min(60, math.floor((longitude + 180) / 6) + 1)
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        zone = 32  # Norway's official UTM exception.
    if 72 <= latitude <= 84 and 0 <= longitude < 42:
        zone = 31 if longitude < 9 else 33 if longitude < 21 else 35 if longitude < 33 else 37
    return (32600 if latitude >= 0 else 32700) + zone


def _pyproj() -> Any:
    try:
        import pyproj
    except ImportError as exc:
        raise DependencyError("Georeferencing requires pyproj") from exc
    pyproj.network.set_network_enabled(False)
    return pyproj


def derive_epsg(samples: list[dict[str, Any]], settings: Any) -> int:
    import numpy as np

    if not samples:
        raise InputError("Cannot derive CRS without GPS")
    if settings.mode == "epsg":
        if settings.epsg is None:
            raise InputError("crs.mode=epsg requires crs.epsg")
        epsg = int(settings.epsg)
    else:
        longitude = np.asarray([sample["longitude"] for sample in samples], dtype=np.float64)
        if np.ptp(longitude) > 180:
            raise InputError("Track crosses the antimeridian; supply an explicit projected CRS")
        epsg = utm_epsg(float(np.median([sample["latitude"] for sample in samples])),
                        float(np.median(longitude)))
        if any(not -80 <= sample["latitude"] <= 84 for sample in samples):
            raise InputError("Track leaves the UTM latitude domain; supply an explicit CRS")
    pyproj = _pyproj()
    try:
        crs = pyproj.CRS.from_epsg(epsg)
    except pyproj.exceptions.CRSError as exc:
        raise InputError(f"Invalid projected EPSG {epsg}: {exc}") from exc
    if not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-12 for axis in crs.axis_info[:2]):
        raise InputError("CRS must be projected with metre horizontal units")
    return epsg


def crs_metadata(epsg: int, samples: list[dict[str, Any]], settings: Any, datum: str) -> dict[str, Any]:
    import numpy as np

    target = datum if settings.vertical == "source" else settings.vertical
    if target != datum and (datum == "unknown" or settings.geoid_path is None):
        raise InputError("Vertical conversion requires a known source datum and an explicit local geoid grid")
    return {"epsg": epsg, "wkt": _pyproj().CRS.from_epsg(epsg).to_wkt(),
            "derived_from": "gps_median_lonlat" if settings.mode == "derive_from_gps" else "explicit_user_override",
            "median_latitude": float(np.median([sample["latitude"] for sample in samples])),
            "median_longitude": float(np.median([sample["longitude"] for sample in samples])),
            "vertical": target, "source_altitude_datum": datum, "vertical_validated": target != "unknown",
            "vertical_conversion": "local_PROJ_geoid_grid" if target != datum else "none",
            "geoid_grid": Path(settings.geoid_path).name if target != datum else None,
            "coordinate_units": "metres", "local_transform": "projected_xyz = local_xyz + origin"}


def project_gps(samples: list[dict[str, Any]], epsg: int, *, settings: Any = None,
                source_datum: str = "unknown", origin: Any = None) -> tuple[Any, Any]:
    import numpy as np

    pyproj = _pyproj()
    longitude = np.asarray([s["longitude"] for s in samples], dtype=np.float64)
    latitude = np.asarray([s["latitude"] for s in samples], dtype=np.float64)
    altitude = np.asarray([s["altitude_m"] for s in samples], dtype=np.float64)
    if settings is not None and settings.vertical not in ("source", source_datum):
        grid = settings.geoid_path
        if source_datum == "unknown" or grid is None or not Path(grid).is_file():
            raise InputError("Cannot convert altitude without known datum and a real local geoid grid")
        multiplier = -1 if source_datum == "ellipsoidal" else 1
        try:
            vertical = pyproj.Transformer.from_pipeline(
                f"+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
                f"+step +proj=vgridshift +grids={Path(grid).resolve()} +multiplier={multiplier} "
                "+step +proj=unitconvert +xy_in=rad +xy_out=deg")
            _, _, altitude = vertical.transform(longitude, latitude, altitude, errcheck=True)
        except pyproj.exceptions.ProjError as exc:
            raise InputError(f"Local geoid conversion failed: {exc}") from exc
    transform = pyproj.Transformer.from_crs(4326, epsg, always_xy=True, allow_ballpark=False)
    try:
        x, y = transform.transform(longitude, latitude, errcheck=True)
    except pyproj.exceptions.ProjError as exc:
        raise InputError(f"GPS projection failed: {exc}") from exc
    absolute = np.column_stack((x, y, altitude)).astype(np.float64)
    if not np.isfinite(absolute).all():
        raise InputError("GPS projection yielded non-finite coordinates")
    local_origin = np.median(absolute, axis=0) if origin is None else np.asarray(origin, dtype=np.float64)
    if local_origin.shape != (3,) or not np.isfinite(local_origin).all():
        raise InputError("Projected origin must be a finite XYZ vector")
    return absolute - local_origin, local_origin
