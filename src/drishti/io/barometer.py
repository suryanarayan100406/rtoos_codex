"""Recorded relative sensor heights and bounded nearest-sample clock alignment.

No pressure conversion, height interpolation, geographic datum, or fusion occurs
here. Source IDs refer to the original CSV order, not the timestamp-sorted order.
"""

from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path
from typing import Any, Literal, TypedDict

from drishti.config.models import BarometerConfig
from drishti.errors import InputError
from drishti.io.telemetry import timestamp_seconds


class BarometerSample(TypedDict):
    source_time_s: float
    time_s: float
    height_m: float
    sigma_m: float
    sigma_source: Literal["csv", "barometer.sigma_m"]
    source_index: int
    source_line: int


class BarometerObservation(BarometerSample):
    time_error_s: float
    frame_time_s: float


def parse_barometer_csv(path: Path, settings: BarometerConfig) -> list[BarometerSample]:
    """Strictly parse all records within max_samples; never guess a clock origin.

    source_index is zero-based excluding the header; source_line is the one-based
    physical start line of the record, including for quoted multiline fields.
    Heights may be negative: their arbitrary zero is not an absolute reference.
    A configured uncertainty column is required on every record, with no fallback.
    """
    origin = settings.time_origin_s
    if origin is None:
        raise InputError("Barometer CSV requires explicit barometer.time_origin_s")
    columns = settings.csv
    required = [columns.time, columns.height]
    if columns.sigma is not None:
        required.append(columns.sigma)
    samples: list[BarometerSample] = []
    source_times: set[float] = set()
    normalized_times: set[float] = set()
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=columns.delimiter, strict=True)
            header = next(reader, [])
            if len(set(header)) != len(header):
                raise InputError(f"Barometer CSV {path}: duplicate column names")
            missing = set(required) - set(header)
            if missing:
                raise InputError(f"Barometer CSV {path}: missing configured columns {sorted(missing)}")
            indices = {name: header.index(name) for name in required}
            while True:
                line = reader.line_num + 1
                row = next(reader, None)
                if row is None:
                    break
                if len(samples) >= settings.max_samples:
                    raise InputError(f"Barometer CSV {path}: exceeds barometer.max_samples={settings.max_samples}")
                if len(row) != len(header):
                    raise InputError(f"Barometer CSV {path}, line {line}: record width differs from header")
                try:
                    source_time = timestamp_seconds(row[indices[columns.time]])
                    time_s = source_time - origin
                    if not math.isfinite(time_s):
                        raise ValueError("normalized time_s must be finite")
                    if source_time in source_times or time_s in normalized_times:
                        raise ValueError("duplicate timestamp; resolve the clock ambiguity")
                    height = float(row[indices[columns.height]])
                    if not math.isfinite(height):
                        raise ValueError("height_m must be finite")
                    sigma = settings.sigma_m if columns.sigma is None else float(row[indices[columns.sigma]])
                    if not math.isfinite(sigma) or sigma <= 0:
                        raise ValueError("sigma_m must be positive and finite on every record")
                    samples.append({"source_time_s": source_time, "time_s": time_s,
                                    "height_m": height, "sigma_m": sigma,
                                    "sigma_source": "barometer.sigma_m" if columns.sigma is None else "csv",
                                    "source_index": len(samples), "source_line": line})
                    source_times.add(source_time)
                    normalized_times.add(time_s)
                except (ValueError, InputError) as exc:
                    raise InputError(f"Barometer CSV {path}, line {line}: {exc}") from exc
    except csv.Error as exc:
        raise InputError(f"Barometer CSV {path}: malformed CSV: {exc}") from exc
    if not samples:
        raise InputError(f"Barometer CSV {path}: contains no recorded height samples")
    return sorted(samples, key=lambda sample: sample["source_time_s"])


def load_barometer(path: Path | None, settings: BarometerConfig) -> dict[str, Any]:
    """Return an optional-input descriptor without changing mandatory GPS inputs.

    `ok` here records successful parsing. The caller fills aligned_frames and
    marks nonoverlap unavailable after inspecting decoded frame timestamps.
    No filesystem access is attempted when disabled or not provided.
    """
    document: dict[str, Any] = {
        "status": "disabled", "reason": "barometer_disabled", "samples": [], "aligned_frames": 0,
        "measurement": "relative_sensor_height_m", "absolute_datum": "not_interpreted",
        "positive_direction": "up",
        "clock": {
            "time_origin_s": settings.time_origin_s, "time_offset_s": settings.time_offset_s,
            "mapping": "clip_time = logged_time - time_origin_s",
            "query": "frame.time_s + barometer.time_offset_s",
            "alignment": "bounded_nearest_recorded_sample",
            "max_time_error_s": settings.max_time_error_s,
            "time_error_definition": "sample.time_s - query_time_s", "tie_break": "earlier_sample",
        },
        "uncertainty": {
            "sigma_column": settings.csv.sigma,
            "assumed_sigma_m": settings.sigma_m if settings.csv.sigma is None else None,
            "assumption": "configured_noise_not_calibrated_accuracy" if settings.csv.sigma is None else None,
            "calibrated_accuracy": False,
        },
    }
    if not settings.enabled:
        return document
    if path is None:
        document.update(status="not_provided", reason="barometer_not_provided")
        return document
    try:
        document["samples"] = parse_barometer_csv(path, settings)
    except (InputError, OSError, UnicodeError) as exc:
        document.update(status="unavailable", reason=str(exc))
        return document
    document.update(status="ok", reason=None)
    return document


class BarometerAligner:
    """Pure nearest-recorded-sample lookup, with no interpolation/extrapolation.

    Pass validated samples from parse_barometer_csv/load_barometer and the same
    config. at() receives the original frame time, applies the barometer offset,
    and returns a copy. Reusing one raw sample retains its same source_index.
    """

    def __init__(self, samples: list[BarometerSample], settings: BarometerConfig):
        self.samples = sorted((sample.copy() for sample in samples), key=lambda sample: sample["time_s"])
        self.times = [sample["time_s"] for sample in self.samples]
        self.time_offset_s = settings.time_offset_s
        self.max_time_error_s = settings.max_time_error_s

    def at(self, frame_time_s: float) -> tuple[BarometerObservation | None, str | None]:
        query = frame_time_s + self.time_offset_s
        if not math.isfinite(frame_time_s) or not math.isfinite(query):
            return None, "barometer_frame_time_nonfinite"
        if not self.times:
            return None, "barometer_no_samples"
        if query < self.times[0] or query > self.times[-1]:
            return None, "barometer_outside_logged_range"
        index = bisect.bisect_left(self.times, query)
        if index > 0 and query - self.times[index - 1] <= self.times[index] - query:
            index -= 1  # Deterministic ties select the earlier recorded sample.
        sample = self.samples[index]
        error = sample["time_s"] - query
        if abs(error) > self.max_time_error_s:
            return None, "barometer_time_error_exceeds_limit"
        observation: BarometerObservation = {**sample, "time_error_s": error, "frame_time_s": frame_time_s}
        return observation, None
