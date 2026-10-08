"""Atomic, relocatable project manifests and input/output integrity."""

import copy
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drishti import __version__
from drishti.bundle.hashing import content_hash, hash_file, hash_path, source_digest
from drishti.config.loader import resolve_paths, set_nested
from drishti.config.models import Config
from drishti.errors import InputError


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def manifest_digest(manifest: dict[str, Any]) -> str:
    return content_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def input_paths(cfg: Config) -> dict[str, Path]:
    if cfg.dataset is None:
        raise InputError("A dataset descriptor with video and telemetry is required")
    result = {"dataset.video": cfg.dataset.video, "dataset.telemetry.path": cfg.dataset.telemetry.path, "models.registry": cfg.models.registry}
    for name in ("imu", "baro", "intrinsics", "rtk"):
        value = getattr(cfg.dataset.optional, name)
        if value is not None:
            result[f"dataset.optional.{name}"] = value
    if cfg.report.check_points is not None:
        result["report.check_points"] = cfg.report.check_points
    if cfg.crs.geoid_path is not None:
        result["crs.geoid_path"] = cfg.crs.geoid_path
    return result


def source_hash(role: str, path: Path) -> dict[str, Any]:
    if not path.exists() and role not in {"dataset.video", "dataset.telemetry.path", "models.registry"}:
        return {"kind": "missing", "sha256": None, "bytes": None}
    return hash_path(path)


def identity_config(cfg: Config) -> dict[str, Any]:
    snapshot = cfg.model_dump(mode="json")
    snapshot["run"].pop("output_root", None)
    snapshot["run"].pop("copy_inputs", None)
    snapshot["models"].pop("cache_dir", None)
    snapshot["models"].pop("registry", None)
    snapshot["compute"].pop("environment", None)
    for role in input_paths(cfg):
        set_nested(snapshot, role, f"input:{role}")
    if snapshot.get("dataset"):
        snapshot["dataset"]["report"] = snapshot["report"]
    return snapshot


def run_identity(cfg: Config, code_digest: str | None = None) -> str:
    sources = {role: source_hash(role, path) for role, path in input_paths(cfg).items()}
    registry_hash = hash_file(cfg.models.registry) if cfg.models.registry.is_file() else None
    from drishti.models.registry import Registry
    registry = Registry(cfg.models.registry, cfg.models.cache_dir)
    models = {}
    for name in (cfg.masking.model, cfg.depth.model, cfg.semantics.model):
        if name is None:
            continue
        spec = registry.models.get(name)
        if isinstance(spec, dict) and isinstance(spec.get("path"), str):
            path = Path(spec["path"]).expanduser()
            root = cfg.models.cache_dir / path if not path.is_absolute() else path
            models[name] = hash_path(root) if root.exists() else None
    return content_hash({"config": identity_config(cfg), "inputs": sources, "code": code_digest or source_digest(), "registry": registry_hash, "models": models})


class Bundle:
    def __init__(self, root: str | Path, manifest: dict[str, Any] | None = None):
        self.root = Path(root).resolve()
        if manifest is None:
            with (self.root / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        if not isinstance(manifest, dict) or manifest.get("bundle_version") != "1":
            raise InputError("Unsupported or invalid bundle manifest")
        if not isinstance(manifest.get("stages"), dict) or not isinstance(manifest.get("inputs"), list) or not isinstance(manifest.get("config_snapshot"), dict):
            raise InputError("Invalid manifest: stages mapping, input list and config_snapshot required")
        self.manifest = manifest

    @classmethod
    def load(cls, root: str | Path) -> "Bundle":
        return cls(root)

    @classmethod
    def create(cls, root: str | Path, cfg: Config, run_id: str | None = None) -> "Bundle":
        paths = input_paths(cfg)
        for role, path in paths.items():
            if role not in {"dataset.video", "dataset.telemetry.path", "models.registry"}:
                continue
            if not path.exists():
                raise InputError(f"Missing supplied input {role}: {path}")
        if not cfg.models.registry.is_file():
            raise InputError(f"Missing model registry: {cfg.models.registry}")
        code = source_digest()
        identity = run_identity(cfg, code)
        destination = Path(root).resolve()
        identifier = run_id or destination.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier):
            raise InputError("run_id must be 1–128 safe filename characters")
        # Exclusive creation is deliberate; an existing run is only opened by resume.
        destination.mkdir(parents=True, exist_ok=False)
        snapshot = cfg.model_dump(mode="json")
        inputs = []
        for role, source in paths.items():
            source = source.resolve()
            before = source_hash(role, source)
            portable = cfg.run.copy_inputs and before["kind"] != "missing"
            if portable:
                target = destination / "inputs" / role.replace(".", "_") / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
                if hash_path(target) != before or hash_path(source) != before:
                    raise InputError(f"Input changed during portable copy: {source}")
                reference = target.relative_to(destination).as_posix()
            else:
                reference = str(source)
            inputs.append({"role": role, "path": reference, "portable": portable, **before})
            set_nested(snapshot, role, reference)
        if snapshot["dataset"] is not None:
            snapshot["dataset"]["report"] = snapshot["report"]
        snapshot["run"]["output_root"] = "."
        manifest = {
            "bundle_version": "1", "drishti_version": __version__, "run_id": identifier,
            "content_id": identity, "created_utc": utc_now(), "updated_utc": utc_now(),
            "status": "created", "config_snapshot": snapshot, "seeds": {"global": cfg.run.seed},
            "source_digest": code, "inputs": inputs, "stages": {}, "crs": {}, "models": [],
        }
        bundle = cls(destination, manifest)
        bundle.save()
        return bundle

    def path(self, relative: str | Path) -> Path:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise InputError(f"Expected bundle-relative confined path: {relative}")
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise InputError(f"Artifact path escapes bundle: {relative}")
        return result

    def stage_dir(self, name: str) -> Path:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise InputError(f"Invalid stage directory: {name}")
        path = self.path(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_json(self, relative: str | Path) -> Any:
        with self.path(relative).open(encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(self, relative: str | Path, data: Any) -> None:
        path = self.path(relative)
        if path == self.root / "manifest.json":
            raise InputError("Use Bundle.save() to write the manifest")
        atomic_json(path, data)

    def artifact(self, stage: str, key: str) -> Path:
        try:
            relative = self.manifest["stages"][stage]["outputs"][key]
        except KeyError as exc:
            raise InputError(f"No recorded artifact {stage}.{key}") from exc
        path = self.path(relative)
        if not path.exists():
            raise InputError(f"Recorded artifact is missing: {stage}.{key}: {relative}")
        return path

    def config(self) -> Config:
        return Config.model_validate(resolve_paths(self.manifest["config_snapshot"], self.root))

    def save(self) -> None:
        self.manifest["updated_utc"] = utc_now()
        self.manifest["manifest_sha256"] = manifest_digest(self.manifest)
        atomic_json(self.root / "manifest.json", self.manifest)

    def output_hashes(self, outputs: dict[str, str]) -> dict[str, Any]:
        result = {}
        for key, relative in outputs.items():
            path = self.path(relative)
            if path == self.root or path == self.root / "manifest.json" or path.is_relative_to(self.root / "inputs"):
                raise InputError(f"Stage output cannot be bundle root, manifest, or source input: {relative}")
            result[key] = {"path": relative, **hash_path(path)}
        return result

    def verify(self) -> dict[str, Any]:
        issues = []
        if self.manifest.get("manifest_sha256") != manifest_digest(self.manifest):
            issues.append("manifest: sha256 mismatch")
        for item in self.manifest.get("inputs", []):
            try:
                path = self.path(item["path"]) if item.get("portable") else Path(item["path"])
                if source_hash(item["role"], path)["sha256"] != item["sha256"]:
                    issues.append(f"input {item['role']}: sha256 mismatch")
            except (OSError, ValueError, InputError) as exc:
                issues.append(f"input {item['role']}: {exc}")
        for name, record in self.manifest.get("stages", {}).items():
            try:
                if "output_hashes" in record:
                    actual = self.output_hashes(record.get("outputs", {}))
                    if actual != record["output_hashes"] or content_hash(actual) != record.get("outputs_hash"):
                        issues.append(f"stage {name}: output integrity mismatch")
            except (OSError, ValueError, InputError) as exc:
                issues.append(f"stage {name}: {exc}")
        partial_report = self.manifest.get("partial_report")
        if partial_report:
            try:
                actual = self.output_hashes(partial_report["outputs"])
                if actual != partial_report["output_hashes"]:
                    issues.append("partial report: output integrity mismatch")
            except (OSError, ValueError, InputError) as exc:
                issues.append(f"partial report: {exc}")
        return {"ok": not issues, "issues": issues, "content_id": self.manifest.get("content_id"), "scope": "bundle integrity; this does not establish reconstruction accuracy or determinism"}

    def snapshot(self, cfg: Config) -> None:
        snapshot = cfg.model_dump(mode="json")
        for role, path in input_paths(cfg).items():
            if path.resolve().is_relative_to(self.root):
                set_nested(snapshot, role, path.resolve().relative_to(self.root).as_posix())
        snapshot["run"]["output_root"] = "."
        if snapshot["dataset"] is not None:
            snapshot["dataset"]["report"] = copy.deepcopy(snapshot["report"])
        self.manifest["config_snapshot"] = snapshot

    def refresh_inputs(self, cfg: Config) -> Config:
        """Rehash current sources, copying explicitly changed resume inputs portably."""
        previous = {item["role"]: item for item in self.manifest["inputs"]}
        snapshot = cfg.model_dump(mode="python")
        updated = []
        for role, source in input_paths(cfg).items():
            source = source.resolve()
            digest = source_hash(role, source)
            old = previous.get(role)
            if source.is_relative_to(self.root) and old and old.get("portable") and source == self.path(old["path"]):
                # In-bundle tampering is not silently accepted as a new source dataset.
                if digest["sha256"] != old["sha256"]:
                    raise InputError(f"Copied source input has been tampered with: {role}")
                reference = old["path"]
                portable = True
            elif cfg.run.copy_inputs and digest["kind"] != "missing":
                target = self.root / "inputs" / role.replace(".", "_") / digest["sha256"][:16] / source.name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_dir():
                        shutil.copytree(source, target)
                    else:
                        shutil.copy2(source, target)
                if hash_path(target) != digest or hash_path(source) != digest:
                    raise InputError(f"Input changed during portable copy: {source}")
                reference = target.relative_to(self.root).as_posix()
                portable = True
            else:
                reference = str(source)
                portable = False
            updated.append({"role": role, "path": reference, "portable": portable, **digest})
            set_nested(snapshot, role, self.root / reference if portable else source)
        snapshot["dataset"]["report"] = snapshot["report"]
        self.manifest["inputs"] = updated
        return Config.model_validate(snapshot)
