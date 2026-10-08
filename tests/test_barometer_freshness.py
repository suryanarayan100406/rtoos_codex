"""Filesystem freshness specimens; no recorded sensor/reconstruction execution claim."""

from dataclasses import dataclass

import pytest

from drishti.bundle import Bundle, content_hash, run_identity
from drishti.config import Config
from drishti.runtime import Runtime
from drishti.stages.base import StageResult
from drishti.stages.runner import STAGES, StageRunner

SENSOR_STAGES = tuple(spec for spec in STAGES if spec.name in {"s0_ingest", "s2_poses", "s6_global"})
ROSTER = tuple(spec for spec in STAGES if spec.name in {"s0_ingest", "s1_frameqa", "s2_poses", "s6_global"})


@pytest.fixture
def foundation(tmp_path, monkeypatch):
    from drishti.stages import runner as runner_module

    # Isolate sensor freshness from other agents' concurrent source edits.
    monkeypatch.setattr(runner_module, "source_digest", lambda: "isolated_barometer_freshness_source")
    video, gps, registry = tmp_path / "video.unit", tmp_path / "gps.unit", tmp_path / "registry.yaml"
    video.write_bytes(b"filesystem specimen, not recorded footage")
    gps.write_bytes(b"filesystem specimen, not mandatory GPS telemetry")
    registry.write_text("models: {}\n", encoding="utf-8")
    cfg = Config.model_validate({
        "dataset": {"name": "barometer-freshness-contract", "video": video,
                    "telemetry": {"format": "csv", "path": gps, "altitude_datum": "unknown"},
                    "optional": {"baro": tmp_path / "barometer.csv"}},
        "run": {"copy_inputs": False},
        "models": {"registry": registry, "cache_dir": tmp_path / "unused-model-cache"},
        "barometer": {"enabled": True, "time_origin_s": 1000.},
    })
    bundle = Bundle.create(tmp_path / "bundle", cfg)
    return StageRunner(bundle, cfg, Runtime("cpu", {"versions": {}, "filesystem_unit": True}), ROSTER)


def seal_roster(runner):
    for name in runner.order:
        spec = runner.stages[name]
        relative = f"{name}/filesystem-unit.json"
        runner.bundle.write_json(relative, {"filesystem_unit_only": True, "sensor_used": False})
        outputs = {"unit": relative}
        hashes = runner.bundle.output_hashes(outputs)
        record = {"name": name, "status": "done", "outputs": outputs,
                  "output_hashes": hashes, "outputs_hash": content_hash(hashes), "models": []}
        runner.bundle.manifest["stages"][name] = record
        record["inputs_hash"] = runner.fingerprint(spec)
    runner.bundle.save()


@pytest.mark.parametrize("spec", SENSOR_STAGES, ids=lambda spec: spec.name)
def test_sensor_config_registration_matches_implementation(spec):
    implementation = spec.load()
    assert "barometer" in spec.config_sections
    assert implementation.config_sections == spec.config_sections
    assert implementation.requires == spec.requires


@pytest.mark.parametrize("spec", SENSOR_STAGES, ids=lambda spec: spec.name)
@pytest.mark.parametrize("parameter,value", [
    ("enabled", False), ("time_origin_s", 2000.), ("time_offset_s", .1),
    ("max_time_error_s", .2), ("sigma_m", 3.), ("min_measurements", 4),
    ("max_samples", 2000000), ("csv.time", "sensor_time"),
    ("csv.height", "sensor_height"), ("csv.sigma", "sensor_sigma"), ("csv.delimiter", ";"),
])
def test_every_sensor_parameter_invalidates_stage_and_identity(foundation, spec, parameter, value):
    runner = foundation
    seal_roster(runner)
    before_params = runner.params(spec)
    before = run_identity(runner.cfg, "isolated-source")
    assert runner.is_fresh(spec.name)
    target = runner.cfg.barometer
    parts = parameter.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    assert runner.params(spec) != before_params
    assert not runner.is_fresh(spec.name, recursive=False)
    assert run_identity(runner.cfg, "isolated-source") != before


@pytest.mark.parametrize("transition", ["changed", "missing_to_present", "present_to_missing"])
def test_raw_sensor_content_and_availability_changes_invalidate_all_sensor_stages(foundation, transition):
    runner = foundation
    path = runner.cfg.dataset.optional.baro
    if transition != "missing_to_present":
        path.write_text("timestamp,height_m\n1000,5\n1001,6\n1002,7\n", encoding="utf-8")
    seal_roster(runner)
    before = run_identity(runner.cfg, "isolated-source")
    assert all(runner.is_fresh(spec.name) for spec in SENSOR_STAGES)
    if transition == "present_to_missing":
        path.unlink()
    else:
        path.write_text("timestamp,height_m\n1000,6\n1001,7\n1002,8\n", encoding="utf-8")
    assert all(not runner.is_fresh(spec.name, recursive=False) for spec in SENSOR_STAGES)
    assert run_identity(runner.cfg, "isolated-source") != before


@pytest.mark.parametrize("relative", ["s0_ingest/barometer.json", "s2_poses/poses.json"])
def test_tampered_aligned_observations_and_pose_provenance_invalidate_downstream(foundation, relative):
    runner = foundation
    seal_roster(runner)
    stage_name = relative.split("/")[0]
    record = runner.bundle.manifest["stages"][stage_name]
    runner.bundle.write_json(relative, {"status": "aligned_not_fused", "height_m": 5.})
    record["outputs"]["sensor_provenance"] = relative
    hashes = runner.bundle.output_hashes(record["outputs"])
    record.update(output_hashes=hashes, outputs_hash=content_hash(hashes))
    for name in runner.order:
        runner.bundle.manifest["stages"][name]["inputs_hash"] = runner.fingerprint(runner.stages[name])
    assert runner.is_fresh("s6_global")
    runner.bundle.write_json(relative, {"status": "aligned_not_fused", "height_m": 6.})
    assert not runner.is_fresh(stage_name)
    assert not runner.is_fresh("s6_global")


@dataclass
class FilesystemStage:
    name: str
    requires: tuple[str, ...]
    config_sections: tuple[str, ...]
    calls: int = 0

    def run(self, bundle, cfg, runtime):
        self.calls += 1
        path = f"{self.name}/filesystem-unit.json"
        bundle.write_json(path, {"filesystem_unit_only": True, "sensor_used": False})
        return StageResult({"unit": path})


@pytest.mark.parametrize("copy_inputs", [False, True])
def test_resume_retries_missing_optional_file_when_raw_sensor_arrives(foundation, copy_inputs):
    fixture = foundation
    fixture.cfg.run.copy_inputs = copy_inputs
    stages = [FilesystemStage(spec.name, spec.requires, spec.config_sections) for spec in ROSTER]
    runner = StageRunner(fixture.bundle, fixture.cfg, fixture.runtime, stages)
    runner.run()
    initial_id = runner.bundle.manifest["content_id"]
    item = next(item for item in runner.bundle.manifest["inputs"] if item["role"] == "dataset.optional.baro")
    assert item["kind"] == "missing" and item["sha256"] is None
    runner.run()
    assert [stage.calls for stage in stages] == [1] * len(stages)
    fixture.cfg.dataset.optional.baro.write_text("timestamp,height_m\n1000,5\n1001,6\n1002,7\n", encoding="utf-8")
    runner.run()
    assert [stage.calls for stage in stages] == [2] * len(stages)
    assert runner.bundle.manifest["content_id"] != initial_id
    runner.run()
    assert [stage.calls for stage in stages] == [2] * len(stages)
    assert runner.bundle.verify()["ok"]
    assert runner.bundle.manifest["models"] == []
    item = next(item for item in runner.bundle.manifest["inputs"] if item["role"] == "dataset.optional.baro")
    assert item["portable"] is copy_inputs


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("provided", [False, True])
def test_absent_optional_sensor_has_no_mandatory_bundle_or_runner_effect(foundation, enabled, provided):
    fixture = foundation
    fixture.cfg.barometer.enabled = enabled
    if not provided:
        fixture.cfg.dataset.optional.baro = None
    stages = [FilesystemStage(spec.name, spec.requires, spec.config_sections) for spec in ROSTER]
    runner = StageRunner(fixture.bundle, fixture.cfg, fixture.runtime, stages)
    runner.run()
    assert runner.bundle.manifest["status"] == "done"
    assert all(runner.is_fresh(spec.name) for spec in SENSOR_STAGES)
    assert runner.bundle.verify()["ok"]
    assert runner.bundle.manifest["models"] == []


def test_old_saved_configs_gain_disabled_sensor_defaults_and_no_new_input_role(foundation):
    legacy = foundation.cfg.model_dump(mode="python")
    legacy.pop("barometer")
    legacy["dataset"]["optional"]["baro"] = None
    cfg = Config.model_validate(legacy)
    assert not cfg.barometer.enabled and cfg.barometer.time_origin_s is None
    bundle = Bundle.create(foundation.bundle.root.parent / "legacy-bundle", cfg)
    assert "dataset.optional.baro" not in {item["role"] for item in bundle.manifest["inputs"]}
    assert bundle.verify()["ok"]
