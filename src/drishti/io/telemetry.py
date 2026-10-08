"""Logged-clock telemetry adapters and bounded, non-extrapolating alignment."""

from __future__ import annotations

import bisect
import csv
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drishti.errors import DependencyError, InputError


def _options(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


def timestamp_seconds(value: Any) -> float:
    """Numeric seconds or an explicitly timezone-qualified ISO-8601 clock."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("ISO timestamp must include a UTC offset")
            result = stamp.timestamp()
        except ValueError as exc:
            raise InputError(f"Invalid telemetry timestamp {value!r}: {exc}") from exc
    if not math.isfinite(result):
        raise InputError(f"Non-finite telemetry timestamp: {value!r}")
    return result


def validate_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not samples:
        raise InputError("Telemetry contains no valid absolute GPS samples")
    for index, sample in enumerate(samples):
        for key in ("time_s", "latitude", "longitude", "altitude_m"):
            try:
                value = float(sample[key])
            except (KeyError, ValueError, TypeError) as exc:
                raise InputError(f"Telemetry record {index}: missing/invalid {key}") from exc
            if not math.isfinite(value):
                raise InputError(f"Telemetry record {index}: non-finite {key}")
            sample[key] = value
        if not -90 <= sample["latitude"] <= 90 or not -180 <= sample["longitude"] <= 180:
            raise InputError(f"Telemetry record {index}: latitude/longitude out of range")
    samples.sort(key=lambda sample: sample["time_s"])
    for left, right in zip(samples, samples[1:], strict=False):
        if left["time_s"] == right["time_s"]:
            raise InputError(f"Duplicate telemetry timestamp {left['time_s']}; resolve the clock ambiguity")
    return samples


def parse_csv(path: Path, options: Any = None) -> list[dict[str, Any]]:
    """CSV columns are explicit mappings; no row-count-derived clock or altitude."""
    opts = _options(options)
    columns = {"time": opts.get("time", "time_s"),
               "latitude": opts.get("lat", "latitude"),
               "longitude": opts.get("lon", "longitude"),
               "altitude": opts.get("alt", "altitude_m")}
    if columns["altitude"].lower() in {"rel_alt", "relative_alt", "relative_altitude", "altitude_relative", "height_agl", "hagl", "agl"}:
        raise InputError("Relative/AGL CSV altitude cannot be treated as absolute; supply absolute altitude")
    samples = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=opts.get("delimiter", ","))
        missing = set(columns.values()) - set(reader.fieldnames or ())
        if missing:
            raise InputError(f"CSV {path}: missing mandatory columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                sample = {
                    "time_s": timestamp_seconds(row[columns["time"]]),
                    "latitude": float(row[columns["latitude"]]),
                    "longitude": float(row[columns["longitude"]]),
                    "altitude_m": float(row[columns["altitude"]]),
                    "source_line": line,
                }
                # Only use uncertainty when supplied, never infer an RTK precision.
                for key in ("sigma_h_m", "sigma_v_m"):
                    if row.get(key, ""):
                        sigma = float(row[key])
                        if not math.isfinite(sigma) or sigma <= 0:
                            raise ValueError(f"{key} must be positive and finite")
                        sample[key] = sigma
                for key in ("yaw", "pitch", "roll"):
                    column = opts.get(key)
                    if column is not None and row.get(column, ""):
                        angle = float(row[column])
                        if not math.isfinite(angle):
                            raise ValueError(f"{key} must be finite")
                        sample[f"{key}_deg"] = angle
                samples.append(sample)
            except (ValueError, TypeError, InputError) as exc:
                raise InputError(f"CSV {path}, line {line}: {exc}") from exc
    return validate_samples(samples)


_NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->")


def parse_dji_srt(path: Path) -> list[dict[str, Any]]:
    samples = []
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8-sig").replace("\r", "")):
        if not block.strip():
            continue
        stamp = _SRT_TIME.search(block)
        if stamp is None:
            raise InputError("DJI SRT block lacks a subtitle timestamp")
        h, m, s, ms = map(int, stamp.groups())
        if m >= 60 or s >= 60:
            raise InputError("DJI SRT subtitle timestamp has invalid minutes/seconds")
        values = {}
        for key, names in {
            "latitude": r"(?:latitude|lat)",
            "longitude": r"(?:longitude|lon|lng)",
            "altitude_m": r"(?:abs_alt|absolute_altitude|absolute_alt|altitude)",
        }.items():
            match = re.search(r"(?<![\w])" + names + r"\s*[:=]\s*" + _NUMBER, block, re.I)
            if match:
                values[key] = float(match.group(1))
        # Older DJI files encode longitude/latitude in GPS(...), but HEIGHT and
        # rel_alt remain relative and must never be promoted to absolute heights.
        if "latitude" not in values or "longitude" not in values:
            gps = re.search(r"GPS\s*\(\s*" + _NUMBER + r"\s*,\s*" + _NUMBER, block, re.I)
            if gps:
                values.update(longitude=float(gps.group(1)), latitude=float(gps.group(2)))
        if set(values) != {"latitude", "longitude", "altitude_m"}:
            raise InputError("DJI SRT requires latitude, longitude and absolute altitude; rel_alt/HEIGHT is insufficient")
        samples.append({"time_s": h * 3600 + m * 60 + s + ms / 1000, **values})
    return validate_samples(samples)


def parse_mavlink(path: Path, altitude_datum: str) -> list[dict[str, Any]]:
    """tlog recorded timestamps; DataFlash GPS TimeUS (not record indices)."""
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise DependencyError("MAVLink/DataFlash telemetry requires pymavlink") from exc
    connection = mavutil.mavlink_connection(str(path), dialect="ardupilotmega")
    binary = Path(path).suffix.lower() == ".bin"
    samples = []
    raw_samples = []
    try:
        while (message := connection.recv_match(blocking=False)) is not None:
            kind = message.get_type()
            if binary:
                if kind != "GPS":
                    continue
                if int(getattr(message, "Status", 0)) < 3:
                    continue
                if altitude_datum == "ellipsoidal":
                    raise InputError("DataFlash GPS Alt is MSL, not ellipsoidal; declare orthometric or unknown")
                if not hasattr(message, "TimeUS"):
                    raise InputError("DataFlash GPS record needs logged TimeUS for frame synchronization")
                sample = {
                    "time_s": float(message.TimeUS) / 1e6,
                    "latitude": message.Lat, "longitude": message.Lng, "altitude_m": message.Alt,
                    "time_domain": "flight_boot_seconds", "message": kind,
                }
            else:
                if kind not in ("GLOBAL_POSITION_INT", "GPS_RAW_INT"):
                    continue
                if kind == "GPS_RAW_INT" and int(message.fix_type) < 3:
                    continue
                if not hasattr(message, "_timestamp"):
                    raise InputError("MAVLink tlog record lacks its recorded timestamp")
                if altitude_datum == "ellipsoidal":
                    if kind != "GPS_RAW_INT":
                        continue
                    if not hasattr(message, "alt_ellipsoid") or not message.alt_ellipsoid:
                        raise InputError("Ellipsoidal MAVLink altitude requires measured GPS_RAW_INT.alt_ellipsoid")
                    altitude = message.alt_ellipsoid
                else:
                    altitude = message.alt
                sample = {
                    "time_s": float(message._timestamp), "latitude": message.lat / 1e7,
                    "longitude": message.lon / 1e7, "altitude_m": altitude / 1000,
                    "time_domain": "recorded_tlog_seconds", "message": kind,
                }
                if altitude_datum != "ellipsoidal" and kind != "GLOBAL_POSITION_INT":
                    raw_samples.append(sample)
                    continue
            samples.append(sample)
    finally:
        connection.close()
    return validate_samples(samples or raw_samples)


def parse_exif(path: Path, altitude_datum: str = "unknown") -> list[dict[str, Any]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DependencyError("EXIF telemetry requires Pillow") from exc
    paths = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"}) if Path(path).is_dir() else [Path(path)]
    samples = []
    if altitude_datum == "ellipsoidal":
        raise InputError("Standard EXIF GPSAltitude is sea-level referenced, not ellipsoidal")
    for image_path in paths:
        try:
            with Image.open(image_path) as image:
                gps = image.getexif().get_ifd(34853)
            if not all(key in gps for key in (1, 2, 3, 4, 5, 6, 7, 29)):
                raise ValueError("needs GPS lat/lon, altitude/reference, UTC date and UTC time")
            def degrees(parts: Any) -> float:
                return float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600
            latitude = degrees(gps[2]) * (-1 if gps[1] == "S" else 1)
            longitude = degrees(gps[4]) * (-1 if gps[3] == "W" else 1)
            if gps[1] not in ("N", "S") or gps[3] not in ("E", "W"):
                raise ValueError("invalid GPS latitude/longitude reference")
            reference = gps[5][0] if isinstance(gps[5], bytes) else int(gps[5])
            if reference not in (0, 1):
                raise ValueError("invalid EXIF GPSAltitudeRef")
            day = datetime.strptime(str(gps[29]).strip("\x00"), "%Y:%m:%d").replace(tzinfo=UTC)
            clock = gps[7]
            if not (0 <= float(clock[0]) < 24 and 0 <= float(clock[1]) < 60 and 0 <= float(clock[2]) < 60):
                raise ValueError("invalid EXIF GPS UTC clock")
            seconds = float(clock[0]) * 3600 + float(clock[1]) * 60 + float(clock[2])
            samples.append({"time_s": day.timestamp() + seconds, "latitude": latitude,
                            "longitude": longitude, "altitude_m": float(gps[6]) * (-1 if reference else 1),
                            "source_image": image_path.name})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise InputError(f"EXIF {image_path}: {exc}") from exc
    return validate_samples(samples)


def load_telemetry(descriptor: Any) -> dict[str, Any]:
    datum = descriptor.altitude_datum
    if hasattr(descriptor, "model_fields_set") and "altitude_datum" not in descriptor.model_fields_set:
        raise InputError("Explicit telemetry.altitude_datum is required, including when it is unknown")
    if datum not in ("unknown", "ellipsoidal", "orthometric"):
        raise InputError("Declare absolute altitude_datum: ellipsoidal, orthometric, or unknown; relative altitude is unsupported")
    path = Path(descriptor.path)
    format_name = descriptor.format
    if format_name == "csv":
        samples = parse_csv(path, descriptor.csv)
    elif format_name == "dji_srt":
        samples = parse_dji_srt(path)
    elif format_name in ("mavlink", "tlog", "dataflash"):
        samples = parse_mavlink(path, datum)
    elif format_name == "exif":
        samples = parse_exif(path, datum)
    else:
        raise InputError(f"Unsupported telemetry format {format_name!r}")
    if descriptor.time_origin_s is None:
        if format_name != "dji_srt":
            raise InputError("Explicit telemetry.time_origin_s is required for non-subtitle logged clocks")
        origin = 0.0  # Subtitle timestamps are explicitly clip-relative.
    else:
        origin = float(descriptor.time_origin_s)
    if not math.isfinite(origin):
        raise InputError("telemetry.time_origin_s must be finite")
    for sample in samples:
        sample["source_time_s"] = sample["time_s"]
        sample["time_s"] -= origin
    return {"samples": samples, "altitude_datum": datum,
            "vertical_validated": datum != "unknown",
            "clock": {"time_origin_s": origin, "mapping": "clip_time = logged_time - time_origin_s"},
            "format": format_name}


class TelemetryInterpolator:
    def __init__(self, samples: list[dict[str, Any]], max_gap_s: float):
        self.samples = validate_samples([dict(sample) for sample in samples])
        self.times = [sample["time_s"] for sample in self.samples]
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise InputError("max_telemetry_gap_s must be positive")
        self.max_gap_s = max_gap_s

    def at(self, time_s: float) -> tuple[dict[str, Any] | None, str | None]:
        if not math.isfinite(time_s):
            raise InputError("Frame telemetry time must be finite")
        index = bisect.bisect_left(self.times, time_s)
        if index < len(self.times) and self.times[index] == time_s:
            sample = dict(self.samples[index])
            sample["interpolation"] = "exact"
            return sample, None
        if index == 0 or index == len(self.times):
            return None, "telemetry_outside_logged_range"
        left, right = self.samples[index - 1:index + 1]
        gap = right["time_s"] - left["time_s"]
        if gap > self.max_gap_s:
            return None, "telemetry_gap_exceeds_limit"
        alpha = (time_s - left["time_s"]) / gap
        longitude_delta = (right["longitude"] - left["longitude"] + 180) % 360 - 180
        result = {"time_s": time_s,
                  "latitude": left["latitude"] + alpha * (right["latitude"] - left["latitude"]),
                  "longitude": (left["longitude"] + alpha * longitude_delta + 180) % 360 - 180,
                  "altitude_m": left["altitude_m"] + alpha * (right["altitude_m"] - left["altitude_m"]),
                  "interpolation": "linear", "bracket_s": [left["time_s"], right["time_s"]]}
        for key in ("sigma_h_m", "sigma_v_m"):
            if key in left and key in right:
                result[key] = max(left[key], right[key])
        return result, None
