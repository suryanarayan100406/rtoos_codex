"""Isolated parser/clock policy units; constructed records are not flight acceptance."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from drishti.config.models import CSVColumns, TelemetryConfig
from drishti.errors import InputError
from drishti.io.telemetry import (
    TelemetryInterpolator,
    load_telemetry,
    parse_csv,
    parse_dji_srt,
    parse_exif,
    parse_mavlink,
    timestamp_seconds,
)


def sample(time, lon=1.0, altitude=20.0):
    return {"time_s": time, "latitude": 2.0, "longitude": lon, "altitude_m": altitude}


def test_csv_explicit_columns_iso_clock_and_real_altitude(tmp_path):
    path = tmp_path / "positions.csv"
    path.write_text("utc,lat,lon,height,yaw,sigma_h_m\n2026-01-01T00:00:00Z,12,77,123.4,45,0.1\n")
    records = parse_csv(path, CSVColumns(time="utc", lat="lat", lon="lon", alt="height", yaw="yaw"))
    assert records[0]["time_s"] == datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    assert records[0]["altitude_m"] == 123.4
    assert records[0]["yaw_deg"] == 45
    assert records[0]["sigma_h_m"] == 0.1


@pytest.mark.parametrize("contents", [
    "latitude,longitude,altitude_m\n12,77,2\n",
    "time_s,latitude,longitude,altitude_m\n0,12,77,\n",
    "time_s,latitude,longitude,altitude_m\n0,91,77,2\n",
    "time_s,latitude,longitude,altitude_m\n0,12,77,nan\n",
    "time_s,latitude,longitude,altitude_m\n0,12,77,2\n0,12,77,2\n",
])
def test_csv_mandatory_fields_fail_without_fabrication(tmp_path, contents):
    path = tmp_path / "invalid.csv"
    path.write_text(contents)
    with pytest.raises(InputError):
        parse_csv(path)


def test_timestamp_rejects_timezone_guess_and_nonfinite():
    with pytest.raises(InputError, match="UTC offset"):
        timestamp_seconds("2026-01-01T00:00:00")
    with pytest.raises(InputError, match="Non-finite"):
        timestamp_seconds("nan")


def test_known_relative_csv_column_cannot_be_relabelled_by_datum(tmp_path):
    path = tmp_path / "positions.csv"
    path.write_text("time_s,latitude,longitude,rel_alt\n0,12,77,30\n")
    with pytest.raises(InputError, match="Relative"):
        parse_csv(path, {"alt": "rel_alt"})


def test_dji_srt_uses_subtitle_clock_absolute_height(tmp_path):
    path = tmp_path / "clip.srt"
    path.write_text("1\n00:00:01,250 --> 00:00:01,500\n[latitude: 12.3] [longitude: 77.4] [rel_alt: 10 abs_alt: 120.5]\n")
    record = parse_dji_srt(path)[0]
    assert record["time_s"] == 1.25
    assert record["altitude_m"] == 120.5
    path.write_text("1\n00:00:01,250 --> 00:00:01,500\nGPS (77.4,12.3,0), HEIGHT: 10m\n")
    with pytest.raises(InputError, match="absolute altitude"):
        parse_dji_srt(path)


def test_descriptor_explicit_datum_clock_origin_and_relative_rejection(tmp_path):
    path = tmp_path / "clip.csv"
    path.write_text("timestamp,latitude,longitude,abs_alt\n1000,12,77,120\n1001,12,77,121\n")
    descriptor = TelemetryConfig(format="csv", path=path, altitude_datum="unknown", time_origin_s=1000.0)
    result = load_telemetry(descriptor)
    assert [row["time_s"] for row in result["samples"]] == [0, 1]
    assert result["samples"][0]["source_time_s"] == 1000
    assert result["vertical_validated"] is False
    with pytest.raises(ValidationError, match="altitude_datum"):
        load_telemetry(TelemetryConfig(format="csv", path=path, time_origin_s=1000.0))
    with pytest.raises(InputError, match="time_origin_s"):
        load_telemetry(TelemetryConfig(format="csv", path=path, altitude_datum="unknown"))
    with pytest.raises(InputError, match="relative"):
        load_telemetry(TelemetryConfig(format="csv", path=path, altitude_datum="relative", time_origin_s=1000.0))


def test_interpolation_has_no_endpoint_extrapolation_or_gap_bridging():
    track = TelemetryInterpolator([sample(0), sample(2, 3, 24), sample(10, 5, 30)], 2)
    value, reason = track.at(1)
    assert reason is None and value["longitude"] == 2 and value["altitude_m"] == 22
    assert track.at(0)[0]["interpolation"] == "exact"
    assert track.at(-0.1) == (None, "telemetry_outside_logged_range")
    assert track.at(10.1) == (None, "telemetry_outside_logged_range")
    assert track.at(3) == (None, "telemetry_gap_exceeds_limit")


def test_longitude_interpolation_crosses_dateline_without_zero_jump():
    value, _ = TelemetryInterpolator([sample(0, 179), sample(2, -179)], 2).at(1)
    assert abs(value["longitude"]) == 180


class LogConnection:
    def __init__(self, records):
        self.records = iter(records)
        self.closed = False

    def recv_match(self, blocking=False):
        return next(self.records, None)

    def close(self):
        self.closed = True


def message(kind, **fields):
    return SimpleNamespace(get_type=lambda: kind, **fields)


def test_mavlink_tlog_uses_recorded_clock_and_absolute_altitude(monkeypatch, tmp_path):
    mavutil = pytest.importorskip("pymavlink.mavutil")
    connection = LogConnection([message("GLOBAL_POSITION_INT", _timestamp=1000.25,
                                       lat=123000000, lon=774000000, alt=123456, relative_alt=10000)])
    monkeypatch.setattr(mavutil, "mavlink_connection", lambda *a, **k: connection)
    result = parse_mavlink(tmp_path / "flight.tlog", "orthometric")
    assert result[0]["time_s"] == 1000.25
    assert result[0]["altitude_m"] == 123.456
    assert result[0]["latitude"] == 12.3
    assert connection.closed


def test_mavlink_raw_only_log_and_ellipsoid_field(monkeypatch, tmp_path):
    mavutil = pytest.importorskip("pymavlink.mavutil")
    records = [message("GPS_RAW_INT", _timestamp=50.0, fix_type=3, lat=120000000,
                       lon=770000000, alt=120000, alt_ellipsoid=160000)]
    monkeypatch.setattr(mavutil, "mavlink_connection", lambda *a, **k: LogConnection(records))
    assert parse_mavlink(tmp_path / "flight.tlog", "unknown")[0]["altitude_m"] == 120
    assert parse_mavlink(tmp_path / "flight.tlog", "ellipsoidal")[0]["altitude_m"] == 160


def test_dataflash_requires_timeus_and_never_relabels_msl(monkeypatch, tmp_path):
    mavutil = pytest.importorskip("pymavlink.mavutil")
    records = [message("GPS", Status=3, TimeUS=1234567, Lat=12.0, Lng=77.0, Alt=100.0)]
    monkeypatch.setattr(mavutil, "mavlink_connection", lambda *a, **k: LogConnection(records))
    assert parse_mavlink(tmp_path / "flight.bin", "orthometric")[0]["time_s"] == 1.234567
    with pytest.raises(InputError, match="MSL"):
        parse_mavlink(tmp_path / "flight.bin", "ellipsoidal")
    records = [message("GPS", Status=3, Lat=12.0, Lng=77.0, Alt=100.0)]
    with pytest.raises(InputError, match="TimeUS"):
        parse_mavlink(tmp_path / "flight.bin", "orthometric")


def test_exif_utc_fractional_clock_and_below_sea_level(monkeypatch, tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    gps = {1: "S", 2: (12, 30, 0), 3: "E", 4: (77, 15, 0), 5: b"\x01", 6: 4,
           7: (1, 2, 3.25), 29: "2026:01:01"}
    class ExifImage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getexif(self):
            return SimpleNamespace(get_ifd=lambda key: gps)
    monkeypatch.setattr(image_module, "open", lambda path: ExifImage())
    result = parse_exif(tmp_path / "image.jpg")[0]
    assert result["latitude"] == -12.5 and result["longitude"] == 77.25
    assert result["altitude_m"] == -4
    assert result["time_s"] == datetime(2026, 1, 1, tzinfo=UTC).timestamp() + 3723.25
    del gps[7]
    with pytest.raises(InputError, match="UTC"):
        parse_exif(tmp_path / "image.jpg")
