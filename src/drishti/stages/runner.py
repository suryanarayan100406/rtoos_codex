"""Lazy DAG execution with measured provenance and content-based resume."""

import importlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drishti.bundle import Bundle, content_hash, hash_path, run_identity, source_digest
from drishti.bundle.core import identity_config, input_paths, manifest_digest, source_hash, utc_now
from drishti.config.models import Config
from drishti.errors import DrishtiError, InputError
from drishti.models.registry import Registry
from drishti.runtime import Runtime, seed_runtime
from drishti.stages.base import Stage, StageResult


@dataclass(frozen=True)
class StageSpec:
    name: str
    requires: tuple[str, ...]
    config_sections: tuple[str, ...]

    def load(self) -> Stage:
        module = "report_stage" if self.name == "report" else self.name
        return importlib.import_module(f"drishti.stages.{module}").STAGE


STAGES = (
    StageSpec("s0_ingest", (), ("dataset", "ingest", "crs", "barometer")),
    StageSpec("s1_frameqa", ("s0_ingest",), ("frameqa", "run")),
    StageSpec("s2_poses", ("s0_ingest", "s1_frameqa"), ("dataset", "poses", "spine", "crs", "run", "barometer")),
    StageSpec("s6_global", ("s2_poses",), ("spine", "bundle_adjustment", "barometer")),
    StageSpec("s3_masking", ("s1_frameqa", "s6_global"), ("masking", "semantics", "models", "compute", "run")),
    StageSpec("s4_depth", ("s2_poses", "s6_global", "s3_masking"), ("depth", "models", "compute")),
    StageSpec("s7_dense", ("s4_depth", "s6_global", "s3_masking"), ("dense",)),
    StageSpec("s8_mesh", ("s7_dense", "s6_global", "s3_masking"), ("mesh", "dense")),
    StageSpec("s9_geo", ("s7_dense", "s8_mesh", "s6_global", "s3_masking"), ("geo", "dense", "crs", "semantics")),
    StageSpec("s10_export", ("s7_dense", "s8_mesh", "s9_geo", "s6_global"), ("export", "crs", "dense")),
    StageSpec("report", ("s10_export",), ("report",)),
)
SUCCESS = {"done", "degraded"}


def normalize_stage(name: str) -> str:
    lower = name.lower()
    if lower in {stage.name for stage in STAGES}:
        return lower
    for stage in STAGES:
        if lower == stage.name.split("_", 1)[0]:
            return stage.name
    raise InputError(f"Unknown stage {name!r}")


class StageRunner:
    def __init__(self, bundle: Bundle, cfg: Config, runtime: Runtime | None = None, stages: Iterable[Stage | StageSpec] | None = None):
        self.bundle = bundle
        self.cfg = cfg
        self.runtime = runtime
        roster = list(STAGES if stages is None else stages)
        self.stages = {stage.name: stage for stage in roster}
        if len(self.stages) != len(roster):
            raise InputError("Duplicate stage names in DAG")
        self.order = self._topological_order()
        self.code_digest = source_digest()
        self._input_states: dict[str, Any] | None = None
        self._model_states: dict[str, Any] | None = None

    def _topological_order(self) -> list[str]:
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise InputError(f"Cycle in stage DAG at {name}")
            if name in visited:
                return
            if name not in self.stages:
                raise InputError(f"Unregistered dependency {name}")
            visiting.add(name)
            for dependency in self.stages[name].requires:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in self.stages:
            visit(name)
        return order

    def _name(self, name: str) -> str:
        return name if name in self.stages else normalize_stage(name)

    def plan(self, selected: Sequence[str] | None = None, upto: str | None = None, only: bool = False) -> list[str]:
        if selected and upto:
            raise InputError("Choose stage selection or --upto, not both")
        if upto:
            target = self._name(upto)
            if target not in self.order:
                raise InputError(f"Unregistered stage {target}")
            return self.order[:self.order.index(target) + 1]
        if not selected:
            return self.order.copy()
        wanted = {self._name(name) for name in selected}
        if not wanted.issubset(self.stages):
            raise InputError("Selection contains an unregistered stage")
        if not only:
            def add_dependencies(name: str) -> None:
                for dependency in self.stages[name].requires:
                    if dependency not in wanted:
                        wanted.add(dependency)
                        add_dependencies(dependency)
            for name in list(wanted):
                add_dependencies(name)
        return [name for name in self.order if name in wanted]

    def params(self, stage: Stage | StageSpec) -> dict[str, Any]:
        canonical = identity_config(self.cfg)
        return {section: canonical[section] for section in stage.config_sections}

    def _model_state(self, name: str | None) -> dict[str, Any]:
        if name is None:
            return {"name": None, "available": False, "disabled": True}
        if self._model_states is not None and name in self._model_states:
            return self._model_states[name]
        registry = Registry(self.cfg.models.registry, self.cfg.models.cache_dir)
        raw = registry.models.get(name)
        if not isinstance(raw, dict):
            return {"name": name, "available": False}
        path = Path(str(raw.get("path", ""))).expanduser()
        root = (self.cfg.models.cache_dir / path).resolve() if not path.is_absolute() else path
        spec = {key: value for key, value in raw.items() if key != "path"}
        try:
            actual = hash_path(root)
        except (OSError, ValueError) as exc:
            actual = {"unavailable": str(exc)}
        state = {"name": name, "spec": spec, "actual_files": actual}
        if self._model_states is not None:
            self._model_states[name] = state
        return state

    def fingerprint(self, stage: Stage | StageSpec) -> str:
        records = self.bundle.manifest["stages"]
        models = []
        if "masking" in stage.config_sections and self.cfg.masking.model is not None:
            models.append(self._model_state(self.cfg.masking.model))
        if "depth" in stage.config_sections:
            models.append(self._model_state(self.cfg.depth.model))
        # S3 attempts optional inference even when no previous model ran. S9 only
        # projects published maps; its semantic config does not imply inference.
        if stage.name == "s3_masking" and self.cfg.semantics.model is not None:
            models.append(self._model_state(self.cfg.semantics.model))
        # Models actually used by a previous invocation must also be fresh.
        for spec in records.get(stage.name, {}).get("models", []):
            if spec.get("name") and spec["name"] not in {model["name"] for model in models}:
                models.append(self._model_state(spec["name"]))
        dependencies = {name: {"outputs_hash": records.get(name, {}).get("outputs_hash"), "inputs_hash": records.get(name, {}).get("inputs_hash"), "status": records.get(name, {}).get("status")} for name in stage.requires}
        payload: dict[str, Any] = {
            "name": stage.name, "requires": list(stage.requires), "params": self.params(stage),
            "inputs": self._input_states if self._input_states is not None else {role: source_hash(role, path) for role, path in input_paths(self.cfg).items()},
            "dependencies": dependencies, "code_digest": self.code_digest,
            "seed": self.cfg.run.seed, "models": models,
            "versions": self.runtime.environment.get("versions", {}) if self.runtime else {},
        }
        if stage.name == "report":
            # A changed failure/timing must regenerate the user-visible report.
            payload["stage_measurements"] = {name: record for name, record in records.items() if name != "report"}
        return content_hash(payload)

    def is_fresh(self, name: str, recursive: bool = True) -> bool:
        if self._input_states is None:
            self.code_digest = source_digest()
        memo: dict[str, bool] = {}

        def check(stage_name: str) -> bool:
            if stage_name in memo:
                return memo[stage_name]
            stage = self.stages[stage_name]
            record = self.bundle.manifest["stages"].get(stage.name, {})
            memo[stage_name] = False
            if record.get("status") not in SUCCESS:
                return False
            if recursive and any(not check(dep) for dep in stage.requires):
                return False
            try:
                if record.get("inputs_hash") != self.fingerprint(stage):
                    return False
                actual = self.bundle.output_hashes(record.get("outputs", {}))
                if actual != record.get("output_hashes") or content_hash(actual) != record.get("outputs_hash"):
                    return False
            except (OSError, ValueError, DrishtiError):
                return False
            memo[stage_name] = True
            return True

        return check(self._name(name))

    def _record_partial(self, name: str, record: dict[str, Any]) -> None:
        directory = self.bundle.path(name)
        if directory.is_dir() and any(directory.iterdir()):
            record["outputs"] = {"partial": name}
            try:
                record["output_hashes"] = self.bundle.output_hashes(record["outputs"])
                record["outputs_hash"] = content_hash(record["output_hashes"])
            except (OSError, ValueError, DrishtiError) as exc:
                record["partial_integrity_error"] = str(exc)

    def _partial_report(self, partial: bool = True) -> None:
        try:
            builder = importlib.import_module("drishti.report.builder")
        except ImportError:
            return
        started = time.perf_counter()
        try:
            result = builder.build_report(self.bundle, self.cfg)
            if isinstance(result, StageResult):
                outputs = result.outputs
            elif isinstance(result, dict) and all(isinstance(value, str) for value in result.values()):
                outputs = result
            else:
                outputs = {"json": "report/report.json", "html": "report/report.html"}
                outputs = {key: path for key, path in outputs.items() if self.bundle.path(path).is_file()}
            if outputs:
                hashes = self.bundle.output_hashes(outputs)
                if partial:
                    self.bundle.manifest["partial_report"] = {"outputs": outputs, "output_hashes": hashes, "outputs_hash": content_hash(hashes)}
                else:
                    self.bundle.manifest.pop("partial_report", None)
                self.bundle.manifest.pop("report_error", None)
                report_record = self.bundle.manifest["stages"].get("report")
                if report_record is not None:
                    report_record.update(outputs=outputs, output_hashes=hashes, outputs_hash=content_hash(hashes))
        except Exception as exc:
            self.bundle.manifest["report_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if not partial:
                self.bundle.manifest["status"] = "partial"
                self.bundle.manifest["stages"]["report"]["status"] = "failed"
        self.bundle.manifest["report_finalization_wall_seconds"] = time.perf_counter() - started
        self.bundle.save()

    def run(self, selected: Sequence[str] | None = None, upto: str | None = None, only: bool = False, force: bool = False) -> Bundle:
        plan = self.plan(selected, upto, only)
        if self.bundle.manifest.get("manifest_sha256") != manifest_digest(self.bundle.manifest):
            raise InputError("Manifest integrity mismatch; inspect/verify before resume")
        started_run = time.perf_counter()
        self.code_digest = source_digest()
        try:
            if self.runtime is None:
                self.runtime = Runtime.detect(self.cfg)
            self.cfg = self.bundle.refresh_inputs(self.cfg)
            self.bundle.manifest["content_id"] = run_identity(self.cfg, self.code_digest)
            self._input_states = {role: source_hash(role, path) for role, path in input_paths(self.cfg).items()}
            self._model_states = {}
        except Exception as exc:
            self.bundle.manifest.update(status="partial", error={"type": type(exc).__name__, "message": str(exc)})
            self.bundle.save()
            self._partial_report()
            raise
        self.bundle.snapshot(self.cfg)
        self.bundle.manifest["environment"] = self.runtime.environment
        self.bundle.manifest["status"] = "running"
        self.bundle.manifest["requested_stages"] = plan
        self.bundle.manifest["source_digest"] = self.code_digest
        self.bundle.manifest.pop("error", None)
        self.bundle.save()
        records = self.bundle.manifest["stages"]
        rerun: set[str] = set()
        failures = False
        verified: set[str] = set()
        record: dict[str, Any]
        for name in plan:
            spec = self.stages[name]
            blocked = [dep for dep in spec.requires if records.get(dep, {}).get("status") not in SUCCESS or (dep not in verified and not self.is_fresh(dep))]
            if blocked:
                reason = "Required stages are unavailable or stale"
                record = {"name": name, "status": "blocked", "blocked_by": blocked,
                          "confidence": None, "confidence_status": "unavailable",
                          "confidence_reason": reason, "degraded_reason": reason, "outputs": {}}
                self._record_partial(name, record)
                records[name] = record
                self.bundle.save()
                failures = True
                continue
            if not force and not any(dep in rerun for dep in spec.requires) and self.is_fresh(name, recursive=False):
                verified.add(name)
                continue
            started = time.perf_counter()
            record = {"name": name, "status": "running", "started_utc": utc_now(), "params": self.params(spec), "code_digest": self.code_digest, "environment": self.runtime.environment, "confidence": None, "confidence_status": None, "confidence_reason": None, "outputs": {}}
            records[name] = record
            self.bundle.save()
            try:
                record["inputs_hash"] = self.fingerprint(spec)
                implementation = spec.load() if isinstance(spec, StageSpec) else spec
                if (implementation.name != name
                        or tuple(implementation.requires) != tuple(spec.requires)
                        or tuple(implementation.config_sections) != tuple(spec.config_sections)):
                    raise InputError(f"Stage {name} does not match the registered DAG contract")
                seed_runtime(self.cfg.run.seed)
                result = implementation.run(self.bundle, self.cfg, self.runtime)
                if not isinstance(result, StageResult):
                    raise InputError(f"Stage {name} did not return StageResult")
                if not result.outputs:
                    raise InputError(f"Stage {name} returned no real outputs")
                if result.confidence is not None and not 0 <= result.confidence <= 1:
                    raise InputError(f"Stage {name} returned invalid confidence")
                content_hash({"metrics": result.metrics, "models": result.models})
                for model in result.models:
                    if not model.get("name"):
                        raise InputError("Executed model provenance must identify its registry name")
                    validated = Registry(self.cfg.models.registry, self.cfg.models.cache_dir).resolve(model["name"])
                    if any(model.get(key) != validated.get(key) for key in ("revision", "license", "files")):
                        raise InputError(f"Executed checkpoint provenance mismatch: {model['name']}")
                hashes = self.bundle.output_hashes(result.outputs)
                confidence_status = "measured_proxy" if result.confidence is not None else "unavailable"
                confidence_reason = None if result.confidence is not None else (
                    result.degraded_reason or "Stage did not publish a numeric confidence summary")
                record.update({"status": "degraded" if result.degraded_reason else "done", "outputs": result.outputs, "output_hashes": hashes, "outputs_hash": content_hash(hashes), "metrics": result.metrics, "confidence": result.confidence, "confidence_status": confidence_status, "confidence_reason": confidence_reason, "degraded_reason": result.degraded_reason, "models": result.models})
                # Include first-use model provenance in the stored freshness identity.
                record["inputs_hash"] = self.fingerprint(spec)
                for model in result.models:
                    if model not in self.bundle.manifest["models"]:
                        self.bundle.manifest["models"].append(model)
                verified.add(name)
            except Exception as exc:
                record.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}, "confidence_status": "unavailable", "confidence_reason": str(exc), "degraded_reason": str(exc)})
                self._record_partial(name, record)
                failures = True
            finally:
                record["ended_utc"] = utc_now()
                record["wall_seconds"] = time.perf_counter() - started
                self.bundle.save()
                rerun.add(name)
        self.bundle.manifest["status"] = "partial" if failures else ("degraded" if any(records[name].get("status") == "degraded" for name in plan) else "done")
        # Downstream artifacts from a prior solve are retained, but not presented as fresh.
        affected = set(rerun)
        for name in self.order:
            if name not in plan and any(dep in affected for dep in self.stages[name].requires):
                affected.add(name)
                if name in records:
                    records[name]["status"] = "stale"
                    records[name]["confidence_status"] = "unavailable"
                    records[name]["confidence_reason"] = "An upstream stage was rerun; this stage is stale"
                    records[name]["degraded_reason"] = "An upstream stage was rerun; resume this stage"
        invocation_wall = time.perf_counter() - started_run
        self.bundle.manifest["last_invocation_wall_seconds"] = invocation_wall
        self.bundle.manifest.setdefault("executions", []).append({"ended_utc": utc_now(), "wall_seconds": invocation_wall, "executed_stages": [name for name in plan if name in rerun], "requested_stages": plan, "environment": self.runtime.environment})
        if len(rerun) == len(self.order) and len(plan) == len(self.order):
            self.bundle.manifest["wall_seconds"] = invocation_wall
        elif rerun:
            # Multi-invocation/stage-scoped work is not one measured end-to-end run.
            self.bundle.manifest["wall_seconds"] = None
        self.bundle.save()
        if failures:
            self._partial_report()
        elif "report" in rerun:
            self._partial_report(partial=False)
        self._input_states = None
        self._model_states = None
        return self.bundle


Runner = StageRunner


def run_pipeline(bundle: Bundle, cfg: Config, **kwargs: Any) -> Bundle:
    return StageRunner(bundle, cfg).run(**kwargs)
