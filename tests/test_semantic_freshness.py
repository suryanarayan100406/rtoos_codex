"""Constructed filesystem/contract units, not semantic or mission acceptance.

Checkpoint specimens are explicitly non-model bytes with measured hashes. Stage
records isolate freshness and never claim inference or reconstruction executed.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from drishti.bundle import Bundle, content_hash, hash_file, run_identity
from drishti.config.models import Config, SemanticsConfig
from drishti.runtime import Runtime
from drishti.stages.base import StageResult
from drishti.stages.runner import STAGES, StageRunner, StageSpec

SEMANTIC_MODEL = "semantic-freshness-unit"
UNIT_CODE_DIGEST = "isolated-freshness-source"
CHECKPOINT_BYTES = b"isolated checksum specimen; not trained model weights"
S3 = next(spec for spec in STAGES if spec.name == "s3_masking")
S9 = next(spec for spec in STAGES if spec.name == "s9_geo")


def write_registry(path: Path, models: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")


def read_registry(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict) and isinstance(document["models"], dict)
    return document["models"]


def record_outputs(runner: StageRunner, spec: StageSpec, outputs: dict[str, str]) -> None:
    hashes = runner.bundle.output_hashes(outputs)
    record: dict[str, Any] = {
        "name": spec.name, "status": "degraded" if spec == S3 else "done",
        "params": runner.params(spec), "outputs": outputs, "output_hashes": hashes,
        "outputs_hash": content_hash(hashes), "models": [],
    }
    runner.bundle.manifest["stages"][spec.name] = record
    record["inputs_hash"] = runner.fingerprint(spec)


def record_roster(runner: StageRunner) -> None:
    """Seal filesystem-only records using the real registered dependency graph."""
    specs = {spec.name: spec for spec in STAGES}
    for name in runner.order:
        runner.bundle.stage_dir(name)
        relative = f"{name}/freshness-unit.json"
        descriptor: dict[str, Any] = {"filesystem_unit_only": True}
        if name == S3.name:
            descriptor.update(status="unavailable", model=None, frames=[],
                              reason="Isolated fixture; semantic inference did not execute")
        runner.bundle.write_json(relative, descriptor)
        record_outputs(runner, specs[name], {"unit": relative})
    runner.bundle.save()


@dataclass
class FreshnessFixture:
    cfg: Config
    bundle: Bundle
    runner: StageRunner

    @property
    def checkpoint(self) -> Path:
        return self.cfg.models.cache_dir / SEMANTIC_MODEL / "weights.unit"


@pytest.fixture
def foundation(tmp_path: Path) -> FreshnessFixture:
    video, telemetry = tmp_path / "video.unit", tmp_path / "telemetry.unit"
    video.write_bytes(b"isolated input hash specimen; not footage")
    telemetry.write_bytes(b"isolated input hash specimen; not flight measurements")
    registry, cache = tmp_path / "registry.yaml", tmp_path / "model-cache"
    models: dict[str, Any] = {}
    defaults = Config()
    for name in (defaults.masking.model, defaults.depth.model, SEMANTIC_MODEL):
        directory = cache / name
        directory.mkdir(parents=True)
        weights = directory / "weights.unit"
        weights.write_bytes(CHECKPOINT_BYTES)
        digest = hash_file(weights)
        models[name] = {
            "path": name, "revision": digest, "license": "MIT",
            "files": {weights.name: digest}, "runtime": "transformers",
            "source": "isolated freshness specimen",
            "output_domain": "semantic_logits" if name == SEMANTIC_MODEL else "unit",
        }
    write_registry(registry, models)
    cfg = Config.model_validate({
        "dataset": {
            "name": "semantic-foundation-unit", "video": video,
            "telemetry": {"format": "csv", "path": telemetry, "altitude_datum": "unknown"},
        },
        "run": {"copy_inputs": False},
        "models": {"registry": registry, "cache_dir": cache},
        "semantics": {
            "model": SEMANTIC_MODEL,
            "classes": [{"id": 11, "name": "land", "color": [10, 20, 30], "source_ids": [0]}],
        },
    })
    bundle = Bundle.create(tmp_path / "bundle", cfg)
    runner = StageRunner(bundle, cfg, Runtime("cpu", {"versions": {}, "unit_runtime": True}))
    record_roster(runner)
    return FreshnessFixture(cfg, bundle, runner)


@pytest.mark.parametrize("spec", STAGES, ids=lambda spec: spec.name)
def test_registered_dag_and_config_contract_match_implementation(spec: StageSpec) -> None:
    implementation = spec.load()
    assert implementation.name == spec.name
    assert implementation.requires == spec.requires
    assert implementation.config_sections == spec.config_sections


@pytest.mark.parametrize("previous", ["present", "missing", "hash_mismatch"])
def test_checkpoint_changes_invalidate_unexecuted_s3_and_downstream(
    foundation: FreshnessFixture, previous: str,
) -> None:
    unit = foundation
    if previous == "missing":
        shutil.rmtree(unit.checkpoint.parent)
    elif previous == "hash_mismatch":
        unit.checkpoint.write_bytes(b"incorrect isolated checkpoint contents")
    record_roster(unit.runner)
    before_identity = run_identity(unit.cfg, UNIT_CODE_DIGEST)
    s9_fingerprint = unit.runner.fingerprint(S9)
    assert unit.runner.is_fresh(S3.name) and unit.runner.is_fresh(S9.name)
    assert unit.bundle.manifest["stages"][S3.name]["models"] == []

    unit.checkpoint.parent.mkdir(exist_ok=True)
    unit.checkpoint.write_bytes(
        CHECKPOINT_BYTES if previous != "present" else CHECKPOINT_BYTES.replace(b"isolated", b"modified")
    )
    assert not unit.runner.is_fresh(S3.name)
    assert run_identity(unit.cfg, UNIT_CODE_DIGEST) != before_identity
    # S9 consumes S3 products, not the checkpoint. Recursive freshness catches
    # the upstream change without attributing semantic inference to S9.
    assert unit.runner.fingerprint(S9) == s9_fingerprint
    assert unit.runner.is_fresh(S9.name, recursive=False)
    assert not unit.runner.is_fresh(S9.name)
    assert unit.bundle.manifest["models"] == []
    assert unit.bundle.manifest["stages"][S3.name]["models"] == []


@pytest.mark.parametrize("metadata", ["runtime", "output_domain", "source"])
def test_semantic_registry_metadata_invalidates_freshness_and_identity(
    foundation: FreshnessFixture, metadata: str,
) -> None:
    unit = foundation
    before = run_identity(unit.cfg, UNIT_CODE_DIGEST)
    models = read_registry(unit.cfg.models.registry)
    models[SEMANTIC_MODEL][metadata] = "changed isolated metadata"
    write_registry(unit.cfg.models.registry, models)
    assert not unit.runner.is_fresh(S3.name)
    assert run_identity(unit.cfg, UNIT_CODE_DIGEST) != before
    assert unit.bundle.manifest["models"] == []


@pytest.mark.parametrize("stage", [S3, S9], ids=lambda spec: spec.name)
@pytest.mark.parametrize("parameter", ["model", "source_ids", "id", "name", "color", "min_probability", "preview_long_side"])
def test_semantic_config_changes_invalidate_stage_and_identity(
    foundation: FreshnessFixture, stage: StageSpec, parameter: str,
) -> None:
    unit = foundation
    before_params = unit.runner.params(stage)
    before_identity = run_identity(unit.cfg, UNIT_CODE_DIGEST)
    semantics = unit.cfg.semantics
    if parameter == "model":
        semantics.model = None
    elif parameter == "min_probability":
        semantics.min_probability = 0.75
    elif parameter == "preview_long_side":
        semantics.preview_long_side = 512
    else:
        value = {"source_ids": [1], "id": 12, "name": "ground", "color": [30, 20, 10]}[parameter]
        setattr(semantics.classes[0], parameter, value)
    assert unit.runner.params(stage) != before_params
    assert not unit.runner.is_fresh(stage.name, recursive=False)
    assert run_identity(unit.cfg, UNIT_CODE_DIGEST) != before_identity


@pytest.mark.parametrize("map_name", ["labels", "confidence"])
def test_tampered_published_semantic_maps_invalidate_recursive_freshness(
    foundation: FreshnessFixture, map_name: str,
) -> None:
    unit = foundation
    directory = unit.bundle.stage_dir(S3.name) / "semantics"
    directory.mkdir()
    labels, confidence = directory / "frame.labels.npy", directory / "frame.confidence.npy"
    np.save(labels, np.full((2, 3), 11, np.uint16), allow_pickle=False)
    np.save(confidence, np.full((2, 3), 0.75, np.float32), allow_pickle=False)
    outputs = unit.bundle.manifest["stages"][S3.name]["outputs"].copy()
    outputs["semantic_maps"] = directory.relative_to(unit.bundle.root).as_posix()
    record_outputs(unit.runner, S3, outputs)
    # Refresh dependency hashes without changing the descriptor or inferred status.
    specs = {spec.name: spec for spec in STAGES}
    for name in unit.runner.order:
        record_outputs(unit.runner, specs[name], unit.bundle.manifest["stages"][name]["outputs"])
    assert unit.runner.is_fresh(S3.name) and unit.runner.is_fresh(S9.name)
    descriptor_hash = hash_file(unit.bundle.path(outputs["unit"]))
    path, dtype = (labels, np.uint16) if map_name == "labels" else (confidence, np.float32)
    np.save(path, np.zeros((2, 3), dtype), allow_pickle=False)
    assert hash_file(unit.bundle.path(outputs["unit"])) == descriptor_hash
    assert not unit.runner.is_fresh(S3.name)
    assert not unit.runner.is_fresh(S9.name)
    assert not unit.bundle.verify()["ok"]


@pytest.mark.parametrize("product", ["semantic", "semantic_confidence", "semantic_preview"])
def test_tampered_published_s9_semantic_products_invalidate_freshness(
    foundation: FreshnessFixture, product: str,
) -> None:
    unit = foundation
    relative = f"{S9.name}/{product}.unit"
    unit.bundle.path(relative).write_bytes(b"published filesystem specimen; not a geospatial product")
    outputs = unit.bundle.manifest["stages"][S9.name]["outputs"].copy()
    outputs[product] = relative
    record_outputs(unit.runner, S9, outputs)
    assert unit.runner.is_fresh(S9.name)
    unit.bundle.path(relative).write_bytes(b"tampered published specimen")
    assert unit.runner.is_fresh(S3.name)
    assert not unit.runner.is_fresh(S9.name)
    assert not unit.bundle.verify()["ok"]


@pytest.mark.parametrize("previous", ["present", "missing"])
def test_disabled_semantics_ignore_unused_checkpoint_bytes_and_availability(
    foundation: FreshnessFixture, previous: str,
) -> None:
    unit = foundation
    unit.cfg.semantics.model = None
    if previous == "missing":
        shutil.rmtree(unit.checkpoint.parent)
    record_roster(unit.runner)
    before_identity = run_identity(unit.cfg, UNIT_CODE_DIGEST)
    unit.checkpoint.parent.mkdir(exist_ok=True)
    unit.checkpoint.write_bytes(b"unused isolated checkpoint now supplied or changed")
    assert unit.runner.is_fresh(S3.name) and unit.runner.is_fresh(S9.name)
    assert run_identity(unit.cfg, UNIT_CODE_DIGEST) == before_identity
    assert unit.bundle.manifest["models"] == []


@pytest.mark.parametrize("role", ["masking", "depth", "semantics"])
@pytest.mark.parametrize("layout", ["directory", "file"])
def test_identity_hashes_actual_configured_checkpoint_contents(
    foundation: FreshnessFixture, role: str, layout: str,
) -> None:
    unit = foundation
    name = getattr(unit.cfg, role).model
    weights = unit.cfg.models.cache_dir / name / "weights.unit"
    if layout == "file":
        models = read_registry(unit.cfg.models.registry)
        models[name]["path"] = f"{name}/{weights.name}"
        write_registry(unit.cfg.models.registry, models)
    before = run_identity(unit.cfg, UNIT_CODE_DIGEST)
    registry_hash = hash_file(unit.cfg.models.registry)
    weights.write_bytes(CHECKPOINT_BYTES.replace(b"isolated", b"modified"))
    assert hash_file(unit.cfg.models.registry) == registry_hash
    assert run_identity(unit.cfg, UNIT_CODE_DIGEST) != before


@pytest.mark.parametrize("available", [True, False], ids=["present", "missing"])
def test_semantic_identity_remains_stable_when_bundle_and_cache_relocate(
    foundation: FreshnessFixture, tmp_path: Path, available: bool,
) -> None:
    unit = foundation
    if not available:
        shutil.rmtree(unit.checkpoint.parent)
    cfg = unit.cfg.model_copy(deep=True)
    cfg.run.copy_inputs = True
    original = Bundle.create(tmp_path / "portable-original", cfg)
    destination, cache = tmp_path / "portable-relocated", tmp_path / "relocated-cache"
    shutil.copytree(original.root, destination)
    shutil.copytree(cfg.models.cache_dir, cache)
    relocated = Bundle.load(destination)
    moved_cfg = relocated.config()
    moved_cfg.models.cache_dir = cache
    moved_cfg.run.output_root = tmp_path / "unrelated-output-directory"
    moved_cfg.compute.environment = "unit label, not an execution claim"
    assert moved_cfg.dataset is not None
    assert moved_cfg.dataset.video.is_relative_to(destination)
    assert relocated.verify()["ok"]
    assert run_identity(moved_cfg, UNIT_CODE_DIGEST) == run_identity(cfg, UNIT_CODE_DIGEST)


@dataclass
class StatusOnlySemanticUnit:
    """Runner filesystem stand-in; never loads a model or performs inference."""

    name: str = S3.name
    requires: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = S3.config_sections
    calls: int = 0

    def run(self, bundle: Bundle, cfg: Config, runtime: Runtime) -> StageResult:
        self.calls += 1
        bundle.stage_dir(self.name)
        relative = f"{self.name}/status-unit.json"
        bundle.write_json(relative, {"status": "unavailable", "model": None, "frames": [],
                                     "reason": "Filesystem unit; inference did not execute"})
        return StageResult({"semantics": relative}, degraded_reason="Isolated filesystem unit")


def test_resume_retries_unavailable_semantics_when_checkpoint_is_supplied(
    foundation: FreshnessFixture,
) -> None:
    unit = foundation
    shutil.rmtree(unit.checkpoint.parent)
    unit.bundle.manifest["stages"] = {}
    unit.bundle.save()
    stage = StatusOnlySemanticUnit()
    runner = StageRunner(unit.bundle, unit.cfg, unit.runner.runtime, [stage])
    runner.run()
    first_identity = unit.bundle.manifest["content_id"]
    runner.run()
    assert stage.calls == 1 and unit.bundle.manifest["content_id"] == first_identity
    unit.checkpoint.parent.mkdir()
    unit.checkpoint.write_bytes(CHECKPOINT_BYTES)
    runner.run()
    assert stage.calls == 2
    assert unit.bundle.manifest["content_id"] != first_identity
    assert unit.bundle.manifest["stages"][stage.name]["models"] == []
    assert unit.bundle.manifest["models"] == []
    assert unit.bundle.verify()["ok"]


def test_runner_rejects_implementation_config_contract_drift(
    foundation: FreshnessFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = foundation
    unit.bundle.manifest["stages"] = {}
    unit.bundle.save()
    stage = StatusOnlySemanticUnit(config_sections=("masking",))
    spec = StageSpec(stage.name, (), S3.config_sections)
    monkeypatch.setattr(StageSpec, "load", lambda self: stage)
    runner = StageRunner(unit.bundle, unit.cfg, unit.runner.runtime, [spec])
    runner.run()
    record = unit.bundle.manifest["stages"][stage.name]
    assert stage.calls == 0 and record["status"] == "failed"
    assert record["error"]["type"] == "InputError"
    assert "registered DAG contract" in record["error"]["message"]


@pytest.mark.parametrize("names", [("Land", " LAND "), ("Straße", " STRASSE "), ("ς", " σ ")])
def test_semantic_config_rejects_strip_casefold_name_collisions(names: tuple[str, str]) -> None:
    classes = [
        {"id": index + 1, "name": name, "color": [10, 20, 30], "source_ids": [index]}
        for index, name in enumerate(names)
    ]
    with pytest.raises(ValidationError, match="nonblank names must be unique"):
        SemanticsConfig.model_validate({"model": SEMANTIC_MODEL, "classes": classes})


def test_semantic_config_preserves_distinct_display_names() -> None:
    cfg = SemanticsConfig.model_validate({"classes": [
        {"id": 1, "name": " Straße ", "color": [10, 20, 30], "source_ids": [0]},
        {"id": 2, "name": "Water", "color": [30, 20, 10], "source_ids": [1]},
    ]})
    assert [item.name for item in cfg.classes] == [" Straße ", "Water"]
