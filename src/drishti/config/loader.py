"""YAML layers: flags > DRISHTI_SECTION__FIELD > dataset > profile > defaults."""

import copy
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from drishti.config.models import Config


def default_config_dir() -> Path:
    checkout = Path(__file__).resolve().parents[3] / "configs"
    installed = Path(sys.prefix) / "share" / "drishti" / "configs"
    for candidate in (checkout, installed, Path.cwd() / "configs"):
        if (candidate / "default.yaml").is_file():
            return candidate
    raise FileNotFoundError("No configs/default.yaml found; pass --config-dir")


def merge(base: dict[str, Any], layer: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in layer.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected a YAML mapping with string keys")
    return value


def set_nested(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError(f"Invalid override key: {dotted}")
    node = target
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Conflicting override: {dotted}")
        node = child
    node[parts[-1]] = value


def overrides_from_strings(values: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=YAML_VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        set_nested(result, key, yaml.safe_load(raw))
    return result


_PATHS = (
    "dataset.video", "dataset.telemetry.path", "dataset.optional.imu",
    "dataset.optional.baro", "dataset.optional.intrinsics", "dataset.optional.rtk",
    "dataset.report.check_points", "report.check_points", "crs.geoid_path",
    "models.registry", "models.cache_dir", "run.output_root",
)


def resolve_paths(layer: dict[str, Any], base: Path) -> dict[str, Any]:
    result = copy.deepcopy(layer)
    for dotted in _PATHS:
        parts = dotted.split(".")
        node = result
        for key in parts[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                break
            node = child
        else:
            value = node.get(parts[-1])
            if isinstance(value, str | Path):
                path = Path(value).expanduser()
                node[parts[-1]] = (base / path).resolve() if not path.is_absolute() else path.resolve()
    return result


def report_layer(layer: dict[str, Any]) -> dict[str, Any]:
    """The descriptor's report fields and cfg.report share a precedence layer."""
    result = copy.deepcopy(layer)
    descriptor = result.get("dataset")
    if isinstance(descriptor, dict) and isinstance(descriptor.get("report"), dict):
        result["report"] = merge(descriptor["report"], result.get("report", {}))
    return result


def load_config(
    dataset: str | Path | None = None,
    profile: str | Path = "balanced",
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    config_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Config:
    directory = Path(config_dir).resolve() if config_dir else default_config_dir()
    # Defaults are relative to the config directory; run output is caller-relative.
    defaults = read_yaml(directory / "default.yaml")
    defaults.setdefault("run", {}).setdefault("output_root", str(Path.cwd() / "runs"))
    data = resolve_paths(defaults, directory)
    profile_path = Path(profile)
    if profile_path.suffix not in (".yaml", ".yml"):
        profile_path = directory / "profiles" / f"{profile}.yaml"
    profile_path = profile_path.resolve()
    data = merge(data, resolve_paths(read_yaml(profile_path), profile_path.parent))
    if dataset is not None:
        descriptor_path = Path(dataset).resolve()
        raw = read_yaml(descriptor_path)
        fields = {key: raw.pop(key) for key in ("name", "video", "telemetry", "optional") if key in raw}
        # A report section configures both descriptor provenance and pipeline validation.
        if "report" in raw:
            fields["report"] = raw["report"]
        if "dataset" in raw and fields:
            raise ValueError("Use either nested dataset or top-level descriptor fields")
        if fields:
            raw["dataset"] = fields
        data = merge(data, resolve_paths(report_layer(raw), descriptor_path.parent))
    environment: dict[str, Any] = {}
    for key, value in (os.environ if environ is None else environ).items():
        if key.startswith("DRISHTI_"):
            set_nested(environment, key[len("DRISHTI_"):].lower().replace("__", "."), yaml.safe_load(value))
    data = merge(data, resolve_paths(report_layer(environment), Path.cwd()))
    if overrides is None:
        flags = {}
    elif isinstance(overrides, Mapping):
        flags = dict(overrides)
    else:
        flags = overrides_from_strings(overrides)
    data = merge(data, resolve_paths(report_layer(flags), Path.cwd()))
    # Dataset-level check data participates in the same precedence as report tunables.
    if data.get("dataset") is not None:
        data["dataset"]["report"] = data.get("report", {})
    return Config.model_validate(data)
