"""Constructed parser/alignment/S0 contracts, not recorded-flight acceptance.

S0 tests explicitly replace only the video-decoding boundary with frame records.
The CSV parsers, geographic transform, typed config and atomic bundle I/O run.
No image, reconstruction, calibrated accuracy or mission result is claimed.
"""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from drishti.bundle import Bundle
from drishti.config.loader import load_config
from drishti.config.models import (
    BarometerConfig,
    BarometerCSVColumns,
    Config,
    DatasetDescriptor,
    IngestConfig,
    ModelsConfig,
    OptionalInputs,
    TelemetryConfig,
)
from drishti.errors import InputError
from drishti.io.barometer import BarometerAligner, load_barometer, parse_barometer_csv
from drishti.runtime import Runtime
from drishti.stages import s0_ingest

ROOT = Path(__file__).resolve().parents[1]


def csv_file(tmp_path, contents):
    path = tmp_path / "barometer.csv"
    path.write_text(contents, encoding="utf-8")
    return path


def settings(**overrides):
    return BarometerConfig.model_validate({"enabled": True, "time_origin_s": 100.0, **overrides})


@pytest.mark.parametrize("profile", ["fast", "balanced", "max"])
def test_yaml_profiles_and_typed_defaults_disable_barometer(profile):
    cfg = load_config(profile=profile, config_dir=ROOT / "configs", environ={})
    assert cfg.barometer == BarometerConfig()
    assert cfg.barometer.csv == BarometerCSVColumns()
    assert cfg.barometer.time_origin_s is None
    assert cfg.barometer.model_dump() == {
        "enabled": False, "time_origin_s": None, "time_offset_s": 0.0,
        "max_time_error_s": 0.1, "sigma_m": 2.0, "min_measurements": 3,
        "max_samples": 1000000,
        "csv": {"time": "timestamp", "height": "height_m", "sigma": None, "delimiter": ","},
    }


def test_legacy_saved_config_gains_disabled_default(tmp_path):
    snapshot = json.loads(Config().model_dump_json())
    del snapshot["barometer"]
    bundle = Bundle(tmp_path, {"bundle_version": "1", "stages": {}, "inputs": [],
                              "config_snapshot": snapshot})
    assert bundle.config().barometer == BarometerConfig()
    assert Config.model_validate(snapshot).model_dump()["barometer"]["enabled"] is False


def test_descriptor_and_overrides_keep_explicit_clock_and_resolve_optional_path(tmp_path):
    descriptor = tmp_path / "dataset.yaml"
    descriptor.write_text(yaml.safe_dump({
        "name": "isolated-clock-unit", "video": "video.unit",
        "telemetry": {"format": "csv", "path": "gps.csv", "altitude_datum": "ellipsoidal",
                      "time_origin_s": 10.0},
        "optional": {"baro": "sensor.csv"},
        "barometer": {"enabled": True, "time_origin_s": 100.0, "time_offset_s": 0.25},
    }), encoding="utf-8")
    cfg = load_config(descriptor, overrides=["barometer.time_offset_s=0.5"],
                      config_dir=ROOT / "configs", environ={"DRISHTI_BAROMETER__TIME_OFFSET_S": "0.3"})
    assert cfg.dataset is not None
    assert cfg.dataset.optional.baro == tmp_path / "sensor.csv"
    assert cfg.barometer.time_origin_s == 100.0
    assert cfg.barometer.time_offset_s == 0.5
    with pytest.raises(ValidationError, match="time_origin_s"):
        load_config(overrides=["barometer.enabled=true"], config_dir=ROOT / "configs", environ={})


@pytest.mark.parametrize("values", [
    {"enabled": True}, {"enabled": 1}, {"time_origin_s": float("nan")},
    {"time_origin_s": float("inf")}, {"time_offset_s": float("inf")},
    {"max_time_error_s": 0.0}, {"max_time_error_s": float("nan")},
    {"sigma_m": 0.0}, {"sigma_m": float("inf")}, {"min_measurements": 1},
    {"min_measurements": 3.0}, {"max_samples": 2}, {"max_samples": 10.0},
    {"pressure": True}, {"csv": {"unknown": "column"}},
    {"csv": {"time": " "}}, {"csv": {"height": ""}}, {"csv": {"sigma": "\t"}},
    {"csv": {"time": "height_m"}}, {"csv": {"sigma": "timestamp"}},
    {"csv": {"delimiter": "||"}}, {"csv": {"delimiter": ""}},
])
def test_config_rejects_ambiguous_clocks_columns_noise_and_budgets(values):
    with pytest.raises(ValidationError):
        BarometerConfig.model_validate(values)


def test_sorted_samples_retain_raw_source_ids_signed_height_and_noise_assumption(tmp_path):
    path = csv_file(tmp_path, "timestamp,height_m,pressure,sigma_m\n102,8,900,invalid\n100,-4,1000,invalid\n101,2,950,invalid\n")
    samples = parse_barometer_csv(path, settings(sigma_m=2.5))
    assert [sample["source_time_s"] for sample in samples] == [100.0, 101.0, 102.0]
    assert [sample["time_s"] for sample in samples] == [0.0, 1.0, 2.0]
    assert [sample["height_m"] for sample in samples] == [-4.0, 2.0, 8.0]
    assert [sample["source_index"] for sample in samples] == [1, 2, 0]
    assert [sample["source_line"] for sample in samples] == [3, 4, 2]
    assert all(sample["sigma_m"] == 2.5 and sample["sigma_source"] == "barometer.sigma_m" for sample in samples)
    assert all("altitude_m" not in sample and "pressure" not in sample for sample in samples)
    document = load_barometer(path, settings(sigma_m=2.5))
    assert document["measurement"] == "relative_sensor_height_m"
    assert document["absolute_datum"] == "not_interpreted"
    assert document["uncertainty"] == {"sigma_column": None, "assumed_sigma_m": 2.5,
                                       "assumption": "configured_noise_not_calibrated_accuracy",
                                       "calibrated_accuracy": False}


def test_explicit_iso_clock_custom_columns_bom_and_physical_source_lines(tmp_path):
    path = csv_file(tmp_path, '\ufeffclock;height;noise;note\n2026-01-01T00:00:01Z;-3;0.4;"first\nrecord"\n2026-01-01T00:00:00+00:00;4;0.2;second\n')
    cfg = settings(time_origin_s=datetime(2026, 1, 1, tzinfo=UTC).timestamp(),
                   csv={"time": "clock", "height": "height", "sigma": "noise", "delimiter": ";"})
    samples = parse_barometer_csv(path, cfg)
    assert [sample["time_s"] for sample in samples] == [0.0, 1.0]
    assert [sample["source_index"] for sample in samples] == [1, 0]
    assert [sample["source_line"] for sample in samples] == [4, 2]
    assert [sample["sigma_m"] for sample in samples] == [0.2, 0.4]
    assert all(sample["sigma_source"] == "csv" for sample in samples)
    assert load_barometer(path, cfg)["uncertainty"]["assumed_sigma_m"] is None


@pytest.mark.parametrize("contents,reason", [
    ("", "missing configured columns"),
    ("timestamp,pressure\n100,1000\n", "height_m"),
    ("timestamp,height_m\n", "no recorded height samples"),
    ("timestamp,height_m\n100,nan\n", "height_m must be finite"),
    ("timestamp,height_m\n100,inf\n", "height_m must be finite"),
    ("timestamp,height_m\n100,\n", "line 2"),
    ("timestamp,height_m\n,2\n", "timestamp"),
    ("timestamp,height_m\nnan,2\n", "Non-finite"),
    ("timestamp,height_m\n2026-01-01T00:00:00,2\n", "UTC offset"),
    ("timestamp,height_m\n100,2\n100.0,3\n", "duplicate timestamp"),
    ("timestamp,height_m\n2026-01-01T00:00:00Z,2\n1767225600,3\n", "duplicate timestamp"),
    ("timestamp,height_m,height_m\n100,2,3\n", "duplicate column names"),
    ("timestamp,height_m\n100\n", "record width"),
    ("timestamp,height_m\n100,2,extra\n", "record width"),
    ('timestamp,height_m\n100,"unterminated\n', "malformed CSV"),
    ("timestamp,height_m\n\n100,2\n", "record width"),
])
def test_strict_csv_never_silently_drops_invalid_records(tmp_path, contents, reason):
    path = csv_file(tmp_path, contents)
    with pytest.raises(InputError, match=reason):
        parse_barometer_csv(path, settings())
    document = load_barometer(path, settings())
    assert document["status"] == "unavailable" and document["samples"] == []
    assert document["reason"] and document["aligned_frames"] == 0


@pytest.mark.parametrize("sigma", ["", "0", "-0.5", "nan", "inf", "invalid"])
def test_configured_sigma_requires_every_record_without_assumed_fallback(tmp_path, sigma):
    path = csv_file(tmp_path, f"timestamp,height_m,noise\n100,2,0.5\n101,3,{sigma}\n")
    cfg = settings(csv={"sigma": "noise"}, sigma_m=99.0)
    with pytest.raises(InputError, match="line 3"):
        parse_barometer_csv(path, cfg)
    assert load_barometer(path, cfg)["samples"] == []


def test_configured_sigma_column_itself_is_mandatory(tmp_path):
    with pytest.raises(InputError, match="noise"):
        parse_barometer_csv(csv_file(tmp_path, "timestamp,height_m\n100,2\n"), settings(csv={"sigma": "noise"}))


def test_parser_never_guesses_origin_even_from_a_zero_first_row(tmp_path):
    with pytest.raises(InputError, match="time_origin_s"):
        parse_barometer_csv(csv_file(tmp_path, "timestamp,height_m\n0,2\n"), BarometerConfig())


def test_raw_sample_budget_is_enforced_before_validating_extra_records(tmp_path):
    cfg = settings(min_measurements=2, max_samples=2)
    path = csv_file(tmp_path, "timestamp,height_m\n100,2\n101,3\n")
    assert len(parse_barometer_csv(path, cfg)) == 2
    path.write_text("timestamp,height_m\n100,2\n101,3\ninvalid,invalid\n", encoding="utf-8")
    with pytest.raises(InputError, match="max_samples=2"):
        parse_barometer_csv(path, cfg)
    assert load_barometer(path, cfg)["status"] == "unavailable"


@pytest.mark.parametrize("origin,rows,reason", [
    (-1e308, "1e308,2\n", "normalized time_s must be finite"),
    (1e20, "0,2\n1,3\n", "duplicate timestamp"),
])
def test_normalization_cannot_publish_overflow_or_ambiguous_times(tmp_path, origin, rows, reason):
    with pytest.raises(InputError, match=reason):
        parse_barometer_csv(csv_file(tmp_path, "timestamp,height_m\n" + rows), settings(time_origin_s=origin))


@pytest.fixture
def parsed_track(tmp_path):
    return parse_barometer_csv(csv_file(tmp_path, "timestamp,height_m,noise\n100,-10,0.5\n101,99,0.7\n"),
                               settings(csv={"sigma": "noise"}))


@pytest.mark.parametrize("frame_time,source_index,height,error", [
    (0.0, 0, -10.0, 0.0), (1.0, 1, 99.0, 0.0),
    (0.05, 0, -10.0, -0.05), (0.95, 1, 99.0, 0.05),
])
def test_alignment_copies_actual_nearest_sample_with_signed_timing(parsed_track, frame_time, source_index, height, error):
    observation, reason = BarometerAligner(parsed_track, settings()).at(frame_time)
    assert reason is None and observation is not None
    assert observation["height_m"] == height
    assert observation["time_s"] == float(source_index)
    assert observation["source_index"] == source_index
    assert observation["source_line"] == source_index + 2
    assert observation["time_error_s"] == pytest.approx(error)
    assert observation["frame_time_s"] == frame_time
    assert observation["sigma_source"] == "csv"
    assert set(observation) == {"source_time_s", "time_s", "height_m", "sigma_m", "sigma_source",
                                "source_index", "source_line", "time_error_s", "frame_time_s"}


@pytest.mark.parametrize("frame_time", [-0.001, 1.001])
def test_alignment_refuses_outside_range_even_within_endpoint_tolerance(parsed_track, frame_time):
    assert BarometerAligner(parsed_track, settings()).at(frame_time) == (None, "barometer_outside_logged_range")


def test_alignment_tolerance_is_inclusive_and_never_bridges_gaps(parsed_track):
    aligner = BarometerAligner(parsed_track, settings(max_time_error_s=0.125))
    assert aligner.at(0.125)[0] is not None
    assert aligner.at(0.125001) == (None, "barometer_time_error_exceeds_limit")
    assert aligner.at(0.5) == (None, "barometer_time_error_exceeds_limit")


def test_alignment_ties_are_deterministic_and_offset_is_applied_once(parsed_track):
    aligner = BarometerAligner(parsed_track, settings(max_time_error_s=0.5, time_offset_s=0.25))
    observation, reason = aligner.at(0.25)  # query=0.5: earlier source record wins.
    assert reason is None and observation is not None
    assert observation["source_index"] == 0 and observation["time_error_s"] == -0.5
    assert observation["frame_time_s"] == 0.25
    later, _ = aligner.at(0.250001)
    assert later is not None and later["source_index"] == 1
    endpoint, _ = aligner.at(0.75)
    assert endpoint is not None and endpoint["time_s"] == 1.0 and endpoint["time_error_s"] == 0.0
    assert aligner.at(0.751) == (None, "barometer_outside_logged_range")


def test_alignment_preserves_source_reuse_and_never_mutates_inputs(parsed_track):
    before = copy.deepcopy(parsed_track)
    aligner = BarometerAligner(parsed_track, settings())
    first, _ = aligner.at(0.01)
    second, _ = aligner.at(0.02)
    assert first is not None and second is not None
    assert first["source_index"] == second["source_index"] == 0
    assert parsed_track == before
    first["height_m"] = 10000.0
    parsed_track[0]["height_m"] = 20000.0
    third, _ = aligner.at(0.0)
    assert third is not None and third["height_m"] == before[0]["height_m"]
    assert second["height_m"] == before[0]["height_m"]


@pytest.mark.parametrize("frame_time", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_alignment_query_is_optional_unavailability(parsed_track, frame_time):
    assert BarometerAligner(parsed_track, settings()).at(frame_time) == (None, "barometer_frame_time_nonfinite")


def test_empty_alignment_and_single_sample_do_not_invent_support(tmp_path):
    assert BarometerAligner([], settings()).at(0.0) == (None, "barometer_no_samples")
    samples = parse_barometer_csv(csv_file(tmp_path, "timestamp,height_m\n100,2\n"), settings())
    aligner = BarometerAligner(samples, settings())
    assert aligner.at(0.0)[0] is not None  # min_measurements gates fusion, not parsing.
    assert aligner.at(0.01) == (None, "barometer_outside_logged_range")


def test_disabled_and_unprovided_loader_skip_parsing(monkeypatch, tmp_path):
    def forbidden_parser(*args, **kwargs):
        pytest.fail("Disabled or unprovided barometer attempted parsing")
    monkeypatch.setattr("drishti.io.barometer.parse_barometer_csv", forbidden_parser)
    disabled = load_barometer(tmp_path / "absent.csv", BarometerConfig())
    assert disabled["status"] == "disabled" and disabled["samples"] == []
    assert load_barometer(None, settings())["status"] == "not_provided"


@pytest.mark.parametrize("source", ["missing", "directory", "non_utf8"])
def test_optional_filesystem_and_decoding_failures_return_descriptors(tmp_path, source):
    path = tmp_path / "sensor.csv"
    if source == "directory":
        path.mkdir()
    elif source == "non_utf8":
        path.write_bytes(b"\xff\xfe\x00")
    document = load_barometer(path, settings())
    assert document["status"] == "unavailable" and document["reason"]
    assert document["samples"] == [] and document["aligned_frames"] == 0


@pytest.fixture
def isolated_s0(tmp_path, monkeypatch):
    """Only decoder is isolated; no actual video or reconstruction is asserted."""
    video_path = tmp_path / "decoder_boundary.unit"
    video_path.write_bytes(b"isolated video boundary, not a playable recording")
    gps_path = tmp_path / "gps.csv"
    gps_path.write_text("timestamp,latitude,longitude,abs_alt\n100,12,77,120\n101,12.0001,77.0001,121\n102,12.0002,77.0002,122\n103,12.0003,77.0003,123\n", encoding="utf-8")
    runs = 0

    def run(baro_path=None, baro_settings=None, frame_times=(0.0, 0.5, 1.0), ingest_offset=0.0):
        nonlocal runs
        runs += 1
        cfg = Config(
            dataset=DatasetDescriptor(name="isolated-S0-contract", video=video_path,
                                      telemetry=TelemetryConfig(format="csv", path=gps_path,
                                                                time_origin_s=100.0, altitude_datum="ellipsoidal"),
                                      optional=OptionalInputs(baro=baro_path)),
            barometer=baro_settings if baro_settings is not None else settings(),
            ingest=IngestConfig(time_offset_s=ingest_offset),
            models=ModelsConfig(registry=ROOT / "configs/models.yaml", cache_dir=tmp_path / "models"),
        )
        bundle = Bundle.create(tmp_path / f"bundle-{runs}", cfg)

        def isolated_decoder(path, output_dir, bundle_root, *, target_fps, max_frames):
            assert path.is_file() and target_fps == cfg.ingest.target_fps
            assert max_frames == cfg.ingest.max_frames
            output_dir.mkdir(parents=True, exist_ok=True)
            frames = [{"id": f"f{index:08d}", "time_s": time_s, "width": 8, "height": 8,
                       "path": (output_dir / f"f{index:08d}.png").relative_to(bundle_root).as_posix()}
                      for index, time_s in enumerate(frame_times)]
            return frames, {"unit_test_video_boundary_isolated": True}

        monkeypatch.setattr(s0_ingest, "decode_video", isolated_decoder)
        result = s0_ingest.S0Ingest().run(bundle, bundle.config(), Runtime("cpu", {"unit_test": True}))
        return bundle, result

    return run


@pytest.mark.parametrize("case,status,parsed", [
    ("disabled_absent", "disabled", False), ("disabled_invalid", "disabled", False),
    ("absent", "not_provided", False), ("missing", "unavailable", False),
    ("invalid", "unavailable", False), ("nonoverlap", "unavailable", True),
    ("beyond_tolerance", "unavailable", True), ("bad_sigma", "unavailable", False),
    ("empty", "unavailable", False), ("non_utf8", "unavailable", False),
])
def test_optional_failures_preserve_valid_s0_gps_baseline(tmp_path, isolated_s0, case, status, parsed):
    cfg = BarometerConfig() if case.startswith("disabled") else settings()
    path = None
    frame_times = (0.0, 0.5, 1.0)
    if case not in ("disabled_absent", "absent"):
        path = tmp_path / "optional.csv"
        contents = {
            "disabled_invalid": "malformed input that must not be parsed\n",
            "invalid": "timestamp,height_m\n100,4\n101,nan\n",
            "nonoverlap": "timestamp,height_m\n110,4\n111,5\n",
            "beyond_tolerance": "timestamp,height_m\n100,4\n101,5\n",
            "bad_sigma": "timestamp,height_m,noise\n100,4,0.5\n101,5,\n",
            "empty": "timestamp,height_m\n",
        }
        if case in contents:
            path.write_text(contents[case], encoding="utf-8")
        elif case == "non_utf8":
            path.write_bytes(b"\xff\xfe\x00")
        if case == "bad_sigma":
            cfg = settings(csv={"sigma": "noise"})
        if case == "beyond_tolerance":
            frame_times = (0.25, 0.5, 0.75)
    baseline_bundle, baseline_result = isolated_s0(frame_times=frame_times, baro_settings=BarometerConfig())
    bundle, result = isolated_s0(path, cfg, frame_times=frame_times)
    document = bundle.read_json(result.outputs["frames"])
    frames = document["frames"]
    assert document["video"]["unit_test_video_boundary_isolated"] is True
    baseline_frames = baseline_bundle.read_json(baseline_result.outputs["frames"])["frames"]
    assert [frame["gps"] for frame in frames] == [frame["gps"] for frame in baseline_frames]
    assert all(frame["gps"] is not None and frame["flags"] == [] for frame in frames)
    assert all(frame["barometer"] is None and frame["barometer_reason"] for frame in frames)
    assert result.metrics["unsupported_telemetry_frames"] == 0 and result.degraded_reason is None
    assert bundle.read_json(result.outputs["telemetry"]) == baseline_bundle.read_json(baseline_result.outputs["telemetry"])
    sensor = bundle.read_json(result.outputs["barometer"])
    assert result.outputs["barometer"] == "s0_ingest/barometer.json"
    assert sensor["status"] == status and sensor["aligned_frames"] == 0
    assert sensor["measurement"] == "relative_sensor_height_m" and sensor["absolute_datum"] == "not_interpreted"
    assert bool(sensor["samples"]) == parsed
    ledger = document["capabilities"]["baro"]
    assert ledger["used"] is False and ledger["parsed"] == parsed
    assert ledger["aligned_frames"] == 0 and ledger["alignment_status"] == status
    assert ledger["provided"] == (path is not None)
    assert ledger["available"] == (path is not None and path.exists())
    assert result.metrics["optional_inputs"]["baro"] == ledger
    if parsed:
        assert ledger["parsing_status"] == "ok" and ledger["parsing_reason"] is None
        expected = "barometer_outside_logged_range" if case == "nonoverlap" else "barometer_time_error_exceeds_limit"
        assert all(frame["barometer_reason"] == expected for frame in frames)


def test_s0_aligns_explicit_independent_sensor_clock_and_preserves_ids_noise(tmp_path, isolated_s0):
    path = csv_file(tmp_path, "timestamp,height_m,noise\n201.25,-3,0.3\n200.25,4,0.5\n200.75,7,0.4\n")
    cfg = settings(time_origin_s=200.0, time_offset_s=0.25, csv={"sigma": "noise"})
    bundle, result = isolated_s0(path, cfg, ingest_offset=0.5)
    document = bundle.read_json(result.outputs["frames"])
    sensor = bundle.read_json(result.outputs["barometer"])
    assert sensor["status"] == "ok" and sensor["reason"] is None and sensor["aligned_frames"] == 3
    assert sensor["clock"]["time_origin_s"] == 200.0 and sensor["clock"]["time_offset_s"] == 0.25
    assert sensor["uncertainty"]["assumed_sigma_m"] is None
    observations = [frame["barometer"] for frame in document["frames"]]
    assert [observation["source_index"] for observation in observations] == [1, 2, 0]
    assert [observation["source_line"] for observation in observations] == [3, 4, 2]
    assert [observation["height_m"] for observation in observations] == [4.0, 7.0, -3.0]
    assert [observation["sigma_m"] for observation in observations] == [0.5, 0.4, 0.3]
    for index, (frame, sample) in enumerate(zip(document["frames"], sensor["samples"], strict=True)):
        assert frame["barometer"] == {**sample, "frame_time_s": frame["time_s"], "time_error_s": 0.0}
        assert frame["barometer_reason"] is None and frame["flags"] == []
        assert frame["gps"]["altitude_m"] == 120.5 + index * 0.5
        assert frame["barometer"]["sigma_source"] == "csv"
    ledger = document["capabilities"]["baro"]
    assert ledger["provided"] is True and ledger["available"] is True and ledger["parsed"] is True
    assert ledger["parsing_status"] == "ok" and ledger["alignment_status"] == "ok"
    assert ledger["status"] == "aligned_not_fused" and ledger["used"] is False
    assert ledger["sample_count"] == 3 and ledger["aligned_frames"] == 3
    assert "barometer" in s0_ingest.S0Ingest.config_sections


def test_s0_partial_alignment_reuses_only_real_samples_without_optional_drop_flags(tmp_path, isolated_s0):
    path = csv_file(tmp_path, "timestamp,height_m\n100,4\n101,9\n")
    frame_times = (-0.01, 0.0, 0.04, 0.5, 0.96, 1.0, 1.01)
    bundle, result = isolated_s0(path, settings(sigma_m=1.5), frame_times, ingest_offset=0.1)
    document = bundle.read_json(result.outputs["frames"])
    sensor = bundle.read_json(result.outputs["barometer"])
    assert sensor["status"] == "ok" and sensor["aligned_frames"] == 4
    assert len(sensor["samples"]) == 2  # S0 may align below the minimum fusion support.
    assert all(frame["gps"] is not None and frame["flags"] == [] for frame in document["frames"])
    assert [frame["barometer_reason"] for frame in document["frames"]] == [
        "barometer_outside_logged_range", None, None, "barometer_time_error_exceeds_limit",
        None, None, "barometer_outside_logged_range",
    ]
    observations = [frame["barometer"] for frame in document["frames"] if frame["barometer"] is not None]
    assert [observation["source_index"] for observation in observations] == [0, 0, 1, 1]
    assert [observation["height_m"] for observation in observations] == [4.0, 4.0, 9.0, 9.0]
    assert all(observation["sigma_source"] == "barometer.sigma_m" and observation["sigma_m"] == 1.5 for observation in observations)
    assert document["capabilities"]["baro"]["used"] is False
    assert result.degraded_reason is None


def test_s0_barometer_never_substitutes_for_unsupported_mandatory_gps(tmp_path, isolated_s0):
    path = csv_file(tmp_path, "timestamp,height_m\n100,4\n104,9\n")
    bundle, result = isolated_s0(path, settings(), frame_times=(0.0, 4.0))
    frames = bundle.read_json(result.outputs["frames"])["frames"]
    assert frames[1]["gps"] is None and frames[1]["flags"] == ["telemetry_outside_logged_range"]
    assert frames[1]["barometer"] is not None and frames[1]["barometer_reason"] is None
    assert result.metrics["unsupported_telemetry_frames"] == 1
    assert result.degraded_reason == "1 frames have unsupported telemetry and will be dropped"
    with pytest.raises(InputError, match="No decoded frame overlaps supported telemetry"):
        isolated_s0(path, settings(), frame_times=(4.0,))


def test_s0_retains_optional_descriptor_even_when_mandatory_parser_fails(tmp_path, monkeypatch, isolated_s0):
    path = csv_file(tmp_path, "timestamp,height_m\n100,4\n101,9\n")

    def invalid_mandatory_telemetry(descriptor):
        raise InputError("isolated mandatory GPS validation failure")

    monkeypatch.setattr(s0_ingest, "load_telemetry", invalid_mandatory_telemetry)
    with pytest.raises(InputError, match="mandatory GPS validation failure"):
        isolated_s0(path, settings())
    bundle = Bundle.load(tmp_path / "bundle-1")
    sensor = bundle.read_json("s0_ingest/barometer.json")
    assert sensor["status"] == "ok" and sensor["aligned_frames"] == 0
    assert [sample["height_m"] for sample in sensor["samples"]] == [4.0, 9.0]
    assert not bundle.path("s0_ingest/frames.json").exists()
