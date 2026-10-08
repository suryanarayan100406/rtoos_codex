"""Portable offline CLI; heavy stage imports are deferred until execution."""

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import ValidationError

from drishti.bundle import Bundle, run_identity
from drishti.config import Config, load_config
from drishti.config.loader import (
    merge,
    overrides_from_strings,
    report_layer,
    resolve_paths,
    set_nested,
)
from drishti.errors import DrishtiError, InputError
from drishti.models.registry import Registry
from drishti.runtime import Runtime
from drishti.stages.runner import STAGES, StageRunner

app = typer.Typer(no_args_is_help=True, help="DRISHTI: offline, ground-only reconstruction.")


def _print(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, default=str, allow_nan=False))


def _error(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(1) from exc


def _flags(settings: list[str], output_root: Path | None) -> dict[str, Any]:
    flags = overrides_from_strings(settings)
    if output_root is not None:
        flags = merge(flags, {"run": {"output_root": str(output_root)}})
    return flags


DatasetOption = Annotated[Path, typer.Option("--dataset", help="Recorded-data YAML descriptor.")]
ProfileOption = Annotated[str, typer.Option("--profile", help="fast, balanced, max or YAML path.")]
OutputOption = Annotated[Path | None, typer.Option("--output-root")]
RunIDOption = Annotated[str | None, typer.Option("--run-id")]
StagesOption = Annotated[list[str] | None, typer.Option("--stage", help="Run stage(s) and their dependencies; repeatable.")]
OnlyOption = Annotated[list[str] | None, typer.Option("--only", help="Run only stage(s), requiring fresh dependencies; repeatable.")]
UptoOption = Annotated[str | None, typer.Option("--upto")]
SetOption = Annotated[list[str] | None, typer.Option("--set", help="Typed KEY=YAML_VALUE override; repeatable.")]
ConfigDirOption = Annotated[Path | None, typer.Option("--config-dir")]


def _execute(bundle: Bundle, cfg: Config, stage: list[str] | None, only: list[str] | None, upto: str | None, force: bool) -> None:
    if stage and only:
        raise InputError("Choose --stage or --only")
    result = StageRunner(bundle, cfg).run(selected=stage or only, upto=upto, only=bool(only), force=force)
    _print({"run_id": result.manifest["run_id"], "bundle": str(result.root), "status": result.manifest["status"]})
    if result.manifest["status"] == "partial":
        raise typer.Exit(1)


@app.command()
def run(
    dataset: DatasetOption,
    profile: ProfileOption = "balanced",
    output_root: OutputOption = None,
    run_id: RunIDOption = None,
    stage: StagesOption = None,
    only: OnlyOption = None,
    upto: UptoOption = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    settings: SetOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Create a new exclusive run; use resume for existing bundles."""
    try:
        cfg = load_config(dataset, profile, _flags(settings or [], output_root), config_dir)
        identifier = run_id or run_identity(cfg)[:16]
        bundle = Bundle.create(cfg.run.output_root / identifier, cfg, identifier)
        _execute(bundle, bundle.config(), stage, only, upto, force)
    except (DrishtiError, OSError, ValueError, ValidationError) as exc:
        _error(exc)


@app.command()
def resume(
    bundle_path: Annotated[Path | None, typer.Argument(help="Existing bundle directory.")] = None,
    dataset: Annotated[Path | None, typer.Option("--dataset")] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    output_root: OutputOption = None,
    run_id: RunIDOption = None,
    stage: StagesOption = None,
    only: OnlyOption = None,
    upto: UptoOption = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    settings: SetOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Resume fresh stages, re-running changed inputs/config/code/models/outputs."""
    try:
        if bundle_path is None:
            if run_id is None:
                raise InputError("Pass bundle directory or --run-id with --output-root")
            bundle_path = (output_root or Path("runs")) / run_id
        bundle = Bundle.load(bundle_path)
        if dataset is not None:
            cfg = load_config(dataset, profile or "balanced", _flags(settings or [], output_root), config_dir)
        else:
            base = bundle.config().model_dump(mode="python")
            # Load explicit profile overrides; defaults must not erase a saved run's tunables.
            if profile is not None:
                from drishti.config.loader import default_config_dir, read_yaml
                directory = config_dir or default_config_dir()
                path = Path(profile)
                if path.suffix not in (".yaml", ".yml"):
                    path = directory / "profiles" / f"{profile}.yaml"
                base = merge(base, resolve_paths(read_yaml(path), path.resolve().parent))
            environment: dict[str, Any] = {}
            for key, value in os.environ.items():
                if key.startswith("DRISHTI_"):
                    set_nested(environment, key[len("DRISHTI_"):].lower().replace("__", "."), yaml.safe_load(value))
            base = merge(base, resolve_paths(report_layer(environment), Path.cwd()))
            base = merge(base, resolve_paths(report_layer(_flags(settings or [], output_root)), Path.cwd()))
            if base.get("dataset") is not None:
                base["dataset"]["report"] = base["report"]
            cfg = Config.model_validate(base)
        _execute(bundle, cfg, stage, only, upto, force)
    except (DrishtiError, OSError, ValueError, ValidationError) as exc:
        _error(exc)


@app.command("inspect")
def inspect_bundle(bundle: Annotated[Path, typer.Argument()]) -> None:
    """Print the recorded manifest without loading optional compute packages."""
    try:
        _print(Bundle.load(bundle).manifest)
    except (DrishtiError, OSError, ValueError) as exc:
        _error(exc)


@app.command()
def verify(bundle: Annotated[Path, typer.Argument()]) -> None:
    """Rehash recursive artifacts and inputs; integrity is distinct from accuracy."""
    try:
        result = Bundle.load(bundle).verify()
        _print(result)
        if not result["ok"]:
            raise typer.Exit(1)
    except (DrishtiError, OSError, ValueError) as exc:
        _error(exc)


@app.command()
def doctor(config_dir: ConfigDirOption = None, profile: ProfileOption = "balanced") -> None:
    """Probe actual CPU/RAM/CUDA/OpenVINO and local checkpoint availability."""
    try:
        cfg = load_config(profile=profile, config_dir=config_dir)
        runtime = Runtime.detect(cfg)
        registry = Registry(cfg.models.registry, cfg.models.cache_dir)
        models = {}
        for name in registry.models:
            try:
                models[name] = {"valid": True, "spec": registry.resolve(name)}
            except DrishtiError as exc:
                models[name] = {"valid": False, "reason": str(exc)}
        _print({"device": runtime.device, "environment": runtime.environment, "models": models, "required_depth_checkpoint_registered": cfg.depth.model in models, "note": "Capabilities detected; no reconstruction algorithm or model inference was executed."})
    except (DrishtiError, OSError, ValueError) as exc:
        _error(exc)


@app.command()
def stages() -> None:
    """Show the offline DAG (S6 runs before S3/S4)."""
    _print([{"name": stage.name, "requires": stage.requires, "config_sections": stage.config_sections} for stage in STAGES])


if __name__ == "__main__":
    app()
