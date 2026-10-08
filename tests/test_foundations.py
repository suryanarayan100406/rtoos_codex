"""Isolated foundation units. These bytes are not footage or reconstruction data."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from drishti.bundle import Bundle, hash_file, hash_path, run_identity, source_digest
from drishti.cli import app
from drishti.config import Config, load_config
from drishti.config.loader import merge
from drishti.errors import DependencyError, InputError
from drishti.models.registry import Registry
from drishti.runtime import Runtime
from drishti.stages.base import StageResult
from drishti.stages.runner import StageRunner

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def descriptor(tmp_path):
    (tmp_path / "video.unit").write_bytes(b"unit input, not a recorded video")
    (tmp_path / "telemetry.unit").write_bytes(b"unit input, not GPS measurements")
    path = tmp_path / "dataset.yaml"
    write_yaml(path, {"name": "isolated-unit", "video": "video.unit", "telemetry": {"format": "csv", "path": "telemetry.unit", "altitude_datum": "unknown"}})
    return path


@pytest.fixture
def cfg(descriptor):
    return load_config(descriptor, config_dir=CONFIGS, environ={})


@pytest.fixture
def runtime():
    return Runtime("cpu", {"versions": {}, "unit_test_runtime": True})


class UnitStage:
    """A filesystem unit of the runner, never a reconstruction algorithm."""
    config_sections = ("ingest",)

    def __init__(self, name="unit_a", requires=(), fail=False, confidence=None):
        self.name, self.requires, self.fail, self.confidence, self.calls = name, requires, fail, confidence, 0

    def run(self, bundle, cfg, runtime):
        self.calls += 1
        bundle.stage_dir(self.name)
        output = f"{self.name}/unit.json"
        bundle.write_json(output, {"unit": self.calls, "fps": cfg.ingest.target_fps})
        if self.fail:
            raise InputError("intentional isolated unit failure")
        return StageResult({"unit": output}, confidence=self.confidence)


def runner_for(tmp_path, cfg, runtime, stages):
    bundle = Bundle.create(tmp_path / "bundle", cfg)
    return StageRunner(bundle, bundle.config(), runtime, stages)


def test_precedence_and_descriptor_paths(descriptor):
    raw = yaml.safe_load(descriptor.read_text())
    raw["ingest"] = {"target_fps": 3.0}
    write_yaml(descriptor, raw)
    assert load_config(descriptor, "fast", config_dir=CONFIGS, environ={}).ingest.target_fps == 3.0
    assert load_config(descriptor, "fast", config_dir=CONFIGS, environ={"DRISHTI_INGEST__TARGET_FPS": "5.0"}).ingest.target_fps == 5.0
    cfg = load_config(descriptor, "fast", ["ingest.target_fps=7.0"], CONFIGS, {"DRISHTI_INGEST__TARGET_FPS": "5.0"})
    assert cfg.ingest.target_fps == 7.0
    assert cfg.dataset.video == descriptor.parent / "video.unit"
    assert cfg.models.registry == CONFIGS / "models.yaml"
    assert cfg.models.cache_dir == ROOT / "models"
    assert cfg.run.copy_inputs is True


def test_nested_descriptor_report_and_portable_model_paths(descriptor):
    raw = yaml.safe_load(descriptor.read_text())
    write_yaml(descriptor, {"dataset": {**raw, "report": {"check_points": "checks.unit"}}, "models": {"registry": "registry.yaml", "cache_dir": "cache"}})
    cfg = load_config(descriptor, config_dir=CONFIGS, environ={})
    assert cfg.report.check_points == descriptor.parent / "checks.unit"
    assert cfg.dataset.report.check_points == cfg.report.check_points
    assert cfg.models.registry == descriptor.parent / "registry.yaml"
    assert cfg.models.cache_dir == descriptor.parent / "cache"


@pytest.mark.parametrize("override", ["depth.typo=2", "ingest.max_frames=3.5", "run.copy_inputs=1", "spine.robust_delta=.nan", "models.offline=false", "dense.method=tsdf", "depth.max_depth_m=0.05", "crs.epsg=32643"])
def test_strict_config_rejection(descriptor, override):
    with pytest.raises(ValidationError):
        load_config(descriptor, overrides=[override], config_dir=CONFIGS, environ={})


@pytest.mark.parametrize("profile", ["fast", "balanced", "max"])
def test_profiles_choose_executable_baseline(profile):
    cfg = load_config(profile=profile, config_dir=CONFIGS, environ={})
    assert cfg.dense.method == "point_fusion"
    assert cfg.mesh.method == "poisson"
    assert cfg.mesh.texture == "best_view"
    assert not cfg.poses.use_glomap
    assert cfg.depth.model == "depth_anything_v2_small"


def test_recursive_inventory_hashes(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "unit").write_bytes(b"first")
    first = hash_path(root)
    (root / "unit").write_bytes(b"other")
    assert hash_path(root)["sha256"] != first["sha256"]
    before = hash_path(root)
    (root / "empty-directory").mkdir()
    assert hash_path(root)["sha256"] != before["sha256"]
    (root / "link").symlink_to(root / "unit")
    with pytest.raises(ValueError, match="Symlinks"):
        hash_path(root)


def test_actual_source_digest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "module.py"
    file.write_text("VALUE = 1\n")
    original = source_digest(source)
    file.write_text("VALUE = 2\n")
    assert source_digest(source) != original


def test_source_digest_includes_isolated_blender_exporter(tmp_path):
    root = tmp_path / "checkout"
    (root / "src/drishti").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "src/drishti/stage.py").write_text("VALUE = 1\n")
    exporter = root / "scripts/blender_export.py"
    exporter.write_text("VALUE = 1\n")
    original = source_digest(root)
    exporter.write_text("VALUE = 2\n")
    assert source_digest(root) != original


def checkpoint(tmp_path, *, name="approved-unit", license="MIT", revision=None, source=None):
    cache = tmp_path / "cache"
    cache.mkdir()
    folder = cache / "checkpoint"
    folder.mkdir()
    file = folder / "weights.unit"
    file.write_bytes(b"unit checksum specimen, not model weights")
    # This is a measured digest of explicit unit bytes, not a real checkpoint claim.
    digest = hash_file(file)
    spec = {"path": "checkpoint", "revision": revision or digest, "license": license, "files": {file.name: digest}}
    if source:
        spec["source"] = source
    registry = tmp_path / "registry.yaml"
    write_yaml(registry, {"models": {name: spec}})
    return registry, cache, file, spec


def test_registry_verifies_checkpoint_files(tmp_path):
    registry, cache, file, spec = checkpoint(tmp_path)
    resolved = Registry(registry, cache).resolve("approved-unit")
    assert resolved["path"] == str(file.parent)
    assert resolved["revision"] == spec["revision"]
    file.write_bytes(b"tamper")
    with pytest.raises(DependencyError, match="sha256 mismatch"):
        Registry(registry, cache).resolve("approved-unit")


@pytest.mark.parametrize("name,license,source", [
    ("approved-unit", "CC-BY-NC-4.0", None),
    ("depth_anything_v2_base", "Apache-2.0", None),
    ("depth_anything_v2_large", "MIT", None),
    ("alias", "MIT", "depth-anything/Depth-Anything-V2-Base"),
    ("dust3r", "Apache-2.0", None),
    ("pi3", "MIT", None),
    ("mapanything", "Apache-2.0", None),
    ("research_only_model", "BSD-3-Clause", None),
])
def test_checkpoint_license_gate_rejects_mislabeling(tmp_path, name, license, source):
    registry, cache, _, _ = checkpoint(tmp_path, name=name, license=license, source=source)
    with pytest.raises(DependencyError):
        Registry(registry, cache).resolve(name)


@pytest.mark.parametrize("license", ["Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"])
def test_permissive_allowlist(tmp_path, license):
    registry, cache, _, _ = checkpoint(tmp_path, license=license)
    assert Registry(registry, cache).resolve("approved-unit")["license"] == license


def test_registry_rejects_mutable_revision_and_unhashed_files(tmp_path):
    registry, cache, file, spec = checkpoint(tmp_path, revision="main")
    with pytest.raises(DependencyError, match="immutable"):
        Registry(registry, cache).resolve("approved-unit")
    spec["revision"] = hash_file(file)
    write_yaml(registry, {"models": {"approved-unit": spec}})
    (file.parent / "unhashed-config.unit").write_bytes(b"additional unit bytes")
    with pytest.raises(DependencyError, match="unhashed"):
        Registry(registry, cache).resolve("approved-unit")


def test_checkpoint_variant_gate_does_not_reject_small_for_unrelated_words(tmp_path):
    registry, cache, file, spec = checkpoint(tmp_path, name="depth_anything_v2_small", license="Apache-2.0")
    large_scene_cache = cache / "large-scene-cache"
    large_scene_cache.mkdir()
    shutil.move(str(file.parent), str(large_scene_cache / "small"))
    spec["path"] = "large-scene-cache/small"
    write_yaml(registry, {"models": {"depth_anything_v2_small": spec}})
    assert Registry(registry, cache).resolve("depth_anything_v2_small")["license"] == "Apache-2.0"


def test_bundle_relocation_collision_integrity(tmp_path, cfg):
    bundle = Bundle.create(tmp_path / "original", cfg)
    assert bundle.verify()["ok"]
    with pytest.raises(FileExistsError):
        Bundle.create(bundle.root, cfg)
    for item in bundle.manifest["inputs"]:
        assert not Path(item["path"]).is_absolute()
    destination = tmp_path / "relocated"
    shutil.copytree(bundle.root, destination)
    relocated = Bundle.load(destination)
    assert relocated.verify()["ok"]
    assert relocated.config().dataset.video.is_relative_to(destination)
    assert run_identity(relocated.config(), code_digest="unit-source-digest") == run_identity(cfg, code_digest="unit-source-digest")
    relocated.config().dataset.video.write_bytes(b"tampered copied source")
    assert not relocated.verify()["ok"]
    with pytest.raises(InputError, match="tampered"):
        relocated.refresh_inputs(relocated.config())


def test_manifest_tampering_detected(tmp_path, cfg):
    bundle = Bundle.create(tmp_path / "bundle", cfg)
    raw = json.loads((bundle.root / "manifest.json").read_text())
    raw["seeds"]["global"] = 999
    (bundle.root / "manifest.json").write_text(json.dumps(raw))
    assert not Bundle.load(bundle.root).verify()["ok"]


def test_bundle_path_confinement(tmp_path, cfg):
    bundle = Bundle.create(tmp_path / "bundle", cfg)
    for bad in ("../outside", str(tmp_path / "outside")):
        with pytest.raises(InputError):
            bundle.write_json(bad, {})


def test_resume_skips_fresh_and_rebuilds_tampered_outputs(tmp_path, cfg, runtime):
    a, b = UnitStage(), UnitStage("unit_b", ("unit_a",))
    runner = runner_for(tmp_path, cfg, runtime, [b, a])
    assert runner.order == ["unit_a", "unit_b"]
    runner.run()
    initial = runner.bundle.manifest["content_id"]
    runner.run()
    assert (a.calls, b.calls) == (1, 1)
    assert runner.bundle.manifest["content_id"] == initial
    runner.bundle.artifact("unit_a", "unit").write_bytes(b"tampered")
    assert not runner.bundle.verify()["ok"]
    runner.run()
    assert (a.calls, b.calls) == (2, 2)
    assert runner.bundle.verify()["ok"]
    assert runner.bundle.manifest["stages"]["unit_a"]["wall_seconds"] >= 0


def test_stage_confidence_status_is_explicit_when_summary_is_missing(tmp_path, cfg, runtime):
    unavailable = UnitStage("unavailable")
    measured = UnitStage("measured", confidence=.75)
    runner = runner_for(tmp_path, cfg, runtime, [unavailable, measured])
    runner.run()
    records = runner.bundle.manifest["stages"]
    assert records["unavailable"]["confidence"] is None
    assert records["unavailable"]["confidence_status"] == "unavailable"
    assert records["unavailable"]["confidence_reason"]
    assert records["measured"]["confidence"] == .75
    assert records["measured"]["confidence_status"] == "measured_proxy"
    assert records["measured"]["confidence_reason"] is None


def test_resume_invalidates_config_code_and_model_bytes(tmp_path, cfg, runtime, monkeypatch):
    from drishti.stages import runner as runner_module
    source = tmp_path / "source"
    source.mkdir()
    code = source / "unit.py"
    code.write_text("UNIT_REVISION = 1\n")
    monkeypatch.setattr(runner_module, "source_digest", lambda: source_digest(source))
    registry, cache, file, spec = checkpoint(tmp_path)
    modified = Config.model_validate(merge(cfg.model_dump(mode="python"), {"models": {"registry": registry, "cache_dir": cache}, "depth": {"model": "approved-unit"}}))
    stage = UnitStage()
    stage.config_sections = ("depth",)
    runner = runner_for(tmp_path, modified, runtime, [stage])
    runner.run()
    assert runner.is_fresh(stage.name)
    changed = runner.cfg.model_copy(deep=True)
    changed.depth.input_long_side = 532
    runner.cfg = changed
    assert not runner.is_fresh(stage.name)
    runner.run()
    code.write_text("UNIT_REVISION = 2\n")
    assert not runner.is_fresh(stage.name)
    runner.run()
    assert stage.calls == 3
    file.write_bytes(b"different measured unit model content")
    assert not runner.is_fresh(stage.name)


def test_failure_preserves_partial_and_blocks_dependents(tmp_path, cfg, runtime, monkeypatch):
    report_called = []
    real_import = importlib.import_module

    def builder(bundle, config):
        report_called.append(bundle.manifest["status"])
        bundle.write_json("report/report.json", {"unit_failure_report": True})
        return {"json": "report/report.json"}

    def lazy_import(name):
        return SimpleNamespace(build_report=builder) if name == "drishti.report.builder" else real_import(name)

    monkeypatch.setattr(importlib, "import_module", lazy_import)
    a, b = UnitStage(fail=True), UnitStage("unit_b", ("unit_a",))
    runner = runner_for(tmp_path, cfg, runtime, [a, b])
    runner.run()
    assert a.calls == 1 and b.calls == 0
    assert runner.bundle.manifest["status"] == "partial"
    assert runner.bundle.manifest["stages"][a.name]["status"] == "failed"
    assert runner.bundle.manifest["stages"][b.name]["status"] == "blocked"
    assert runner.bundle.artifact(a.name, "partial").is_dir()
    assert report_called == ["partial"]
    assert runner.bundle.verify()["ok"]


def test_only_requires_fresh_dependencies_and_dag_order(tmp_path, cfg, runtime):
    a, b = UnitStage(), UnitStage("unit_b", ("unit_a",))
    runner = runner_for(tmp_path, cfg, runtime, [a, b])
    runner.run(selected=["unit_b"], only=True)
    assert b.calls == 0
    assert runner.bundle.manifest["stages"][b.name]["status"] == "blocked"
    dag = StageRunner(runner.bundle, runner.cfg, runtime)
    assert dag.order.index("s6_global") < dag.order.index("s3_masking") < dag.order.index("s4_depth")
    assert "s0_ingest" in dag.plan(["s4"])
    with pytest.raises(InputError, match="Cycle"):
        StageRunner(runner.bundle, runner.cfg, runtime, [UnitStage("cycle", ("cycle",))])


def test_cli_help_stages_doctor_are_core_only(monkeypatch):
    monkeypatch.setattr(Runtime, "detect", lambda cfg: Runtime("cpu", {"unit_test_probe": True}))
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["run", "--help"]).exit_code == 0
    assert runner.invoke(app, ["resume", "--help"]).exit_code == 0
    result = runner.invoke(app, ["stages"])
    assert result.exit_code == 0, result.output
    assert "s6_global" in result.output
    result = runner.invoke(app, ["doctor", "--config-dir", str(CONFIGS)])
    assert result.exit_code == 0, result.output
    assert "no reconstruction algorithm" in result.output
    result = subprocess.run([sys.executable, "-c", "import sys; import drishti.cli; assert 'torch' not in sys.modules and 'open3d' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_runtime_reports_actual_host_without_placement_labels(monkeypatch):
    from drishti.runtime import probe as probe_module
    real_version = probe_module.importlib.metadata.version

    def version(name):
        if name in ("torch", "openvino"):
            raise probe_module.importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(probe_module.importlib.metadata, "version", version)
    cfg = Config()
    cfg.compute.environment = "requested-cloud-label"
    runtime = Runtime.detect(cfg)
    assert runtime.device == "cpu"
    assert runtime.environment["ram_total_bytes"] > 0
    assert runtime.environment["cpu_logical"] > 0
    assert runtime.environment["requested_environment_label"] == "requested-cloud-label"
    cfg.compute.device = "cuda"
    with pytest.raises(DependencyError, match="operational"):
        Runtime.detect(cfg)


def test_changed_external_source_rehashed_and_copied_on_resume(tmp_path, cfg, runtime):
    stage = UnitStage()
    runner = runner_for(tmp_path, cfg, runtime, [stage])
    runner.run()
    initial_content = runner.bundle.manifest["content_id"]
    cfg.dataset.video.write_bytes(b"new external isolated source input")
    runner.cfg = cfg
    runner.run()
    assert stage.calls == 2
    assert runner.cfg.dataset.video.is_relative_to(runner.bundle.root)
    assert runner.bundle.manifest["content_id"] != initial_content
    assert runner.bundle.verify()["ok"]


def test_unselected_downstream_marked_stale(tmp_path, cfg, runtime):
    a, b = UnitStage(), UnitStage("unit_b", ("unit_a",))
    runner = runner_for(tmp_path, cfg, runtime, [a, b])
    runner.run()
    runner.run(selected=[a.name], force=True)
    assert runner.bundle.manifest["stages"][b.name]["status"] == "stale"
    assert runner.bundle.artifact(b.name, "unit").is_file()
    runner.run()
    assert (a.calls, b.calls) == (2, 2)


def test_real_report_builder_final_status_and_partial_recovery(tmp_path, cfg, runtime):
    from drishti.stages.report_stage import STAGE as report_stage
    a = UnitStage("s10_export", fail=True)
    runner = runner_for(tmp_path, cfg, runtime, [a, report_stage])
    runner.run()
    assert runner.bundle.read_json("report/report.json")["status"] == "partial"
    assert runner.bundle.manifest["stages"]["report"]["status"] == "blocked"
    a.fail = False
    runner.run()
    assert "partial_report" not in runner.bundle.manifest
    assert runner.bundle.verify()["ok"]
    report = runner.bundle.read_json("report/report.json")
    assert report["status"] == "done"
    assert report["accuracy"]["status"] == "UNVALIDATED"
    assert report["stages"][-1]["status"] == "done"
    runner.run()
    assert a.calls == 2


def test_missing_mandatory_input_never_creates_bundle(tmp_path, cfg):
    cfg.dataset.video.unlink()
    with pytest.raises(InputError, match="Missing supplied input"):
        Bundle.create(tmp_path / "bundle", cfg)
    assert not (tmp_path / "bundle").exists()


def test_missing_optional_input_remains_optional_and_null(tmp_path, cfg, runtime):
    cfg.dataset.optional.imu = tmp_path / "absent-optional.unit"
    stage = UnitStage()
    runner = runner_for(tmp_path, cfg, runtime, [stage])
    assert runner.bundle.verify()["ok"]
    item = next(item for item in runner.bundle.manifest["inputs"] if item["role"] == "dataset.optional.imu")
    assert item["kind"] == "missing" and item["sha256"] is None
    runner.run()
    assert stage.calls == 1
    cfg.dataset.optional.imu.write_bytes(b"now supplied isolated optional input")
    runner.cfg = cfg
    runner.run()
    assert stage.calls == 2
    assert runner.bundle.verify()["ok"]


def test_cli_failure_inspection_verification_and_collision(descriptor, tmp_path):
    # Exercise actual command wiring with deliberately invalid unit bytes, not a clip.
    runner = CliRunner()
    output_root = tmp_path / "unit-runs"
    args = ["run", "--dataset", str(descriptor), "--output-root", str(output_root), "--run-id", "unit-run", "--upto", "s0", "--set", "ingest.target_fps=5.0", "--config-dir", str(CONFIGS)]
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    bundle = Bundle.load(output_root / "unit-run")
    assert bundle.manifest["status"] == "partial"
    assert bundle.config().ingest.target_fps == 5.0
    assert bundle.read_json("report/report.json")["accuracy"]["status"] == "UNVALIDATED"
    assert runner.invoke(app, ["inspect", str(bundle.root)]).exit_code == 0
    assert runner.invoke(app, ["verify", str(bundle.root)]).exit_code == 0
    collision = runner.invoke(app, [*args, "--force"])
    assert collision.exit_code == 1 and "File exists" in collision.output
    resumed = runner.invoke(app, ["resume", "--output-root", str(output_root), "--run-id", "unit-run", "--only", "s0", "--set", "ingest.target_fps=6.0"])
    assert resumed.exit_code == 1, resumed.output
    assert Bundle.load(bundle.root).config().ingest.target_fps == 6.0
