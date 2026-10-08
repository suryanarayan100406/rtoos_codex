"""Offline, stage-isolated Kaggle execution with a sampled memory watchdog.

GB is decimal (1e9 bytes). The watchdog is a monitored stop, not an OS hard
limit: sampled peaks are lower bounds and the CUDA allocator cap covers Torch,
not arbitrary native allocations. Importing this module needs no neural extras.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
from pydantic import Field, PrivateAttr, field_validator, model_validator

from drishti.bundle import Bundle
from drishti.bundle.core import atomic_json, input_paths, manifest_digest, source_hash, utc_now
from drishti.bundle.hashing import content_hash, hash_path
from drishti.config.loader import (
    load_config,
    merge,
    overrides_from_strings,
    read_yaml,
    report_layer,
    resolve_paths,
)
from drishti.config.models import Config, FilePath, Positive, StrictModel
from drishti.errors import DependencyError, InputError
from drishti.models.registry import Registry
from drishti.runtime import Runtime
from drishti.runtime.probe import PACKAGES
from drishti.stages.runner import STAGES, SUCCESS, StageRunner

GB = 1_000_000_000
MIB = 1_048_576
DEFAULT_SMI_TIMEOUT_S = 10.0
DEFAULT_REPORT_RECOVERY_TIMEOUT_S = 120.0
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PROJ_NETWORK": "OFF",
    "WANDB_MODE": "disabled",
}


class KaggleSettings(StrictModel):
    dataset: FilePath | None = None
    resume_bundle: FilePath | None = None
    output_root: FilePath = Path("/kaggle/working/drishti-runs")
    run_id: str = "kaggle-quality"
    profile: FilePath = Path("configs/profiles/kaggle_quality.yaml")
    registry: FilePath = Path("configs/models.yaml")
    model_cache: FilePath = Path("models")
    model_paths: dict[str, FilePath] = Field(default_factory=dict)
    gpu_index: int = Field(default=0, ge=0)
    ram_limit_gb: float = Field(default=30.0, gt=0, le=30)
    vram_limit_gb: float = Field(default=30.0, gt=0, le=30)
    ram_reserve_gb: Positive = 3.0
    vram_reserve_gb: Positive = 1.0
    gpu_memory_fraction: float = Field(default=0.90, gt=0, le=1)
    cpu_threads: int = Field(default=8, ge=1)
    monitor_interval_s: Positive = 0.5
    min_system_available_gb: Positive = 1.0
    termination_grace_s: Positive = 5.0
    smi_timeout_s: Positive = DEFAULT_SMI_TIMEOUT_S
    report_recovery_timeout_s: Positive = DEFAULT_REPORT_RECOVERY_TIMEOUT_S
    sample_frame_budget: int = Field(default=1800, ge=3)
    archive: bool = True
    archive_name: str = "drishti-kaggle-quality.zip"
    overrides: list[str] = Field(default_factory=list)

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)
    _explicit_fields: set[str] | None = PrivateAttr(default=None)

    @field_validator("run_id", "archive_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
            raise ValueError("Expected a safe basename of 1–128 filename characters")
        return value

    @model_validator(mode="after")
    def valid_budget_and_source(self) -> KaggleSettings:
        if self.dataset is None and self.resume_bundle is None:
            raise ValueError("Supply dataset or resume_bundle (or both)")
        if self.ram_reserve_gb >= self.ram_limit_gb:
            raise ValueError("ram_reserve_gb must be below ram_limit_gb")
        if self.vram_reserve_gb >= self.vram_limit_gb:
            raise ValueError("vram_reserve_gb must be below vram_limit_gb")
        return self


def load_settings(path: str | Path) -> KaggleSettings:
    """Strict YAML settings; every path, including defaults, is YAML-relative.

    On resume an omitted profile preserves saved tunables. An explicitly supplied
    profile is layered over the saved config, without resetting it to defaults.
    """
    source = Path(path).expanduser().resolve()
    settings = KaggleSettings.model_validate(read_yaml(source))
    settings._explicit_fields = settings.model_fields_set.copy()
    settings._base_dir = source.parent
    for key in ("dataset", "resume_bundle", "output_root", "profile", "registry", "model_cache"):
        value = getattr(settings, key)
        if value is not None:
            setattr(settings, key, (source.parent / value).resolve())
    rebound_paths: dict[str, Path] = {}
    for key, value in settings.model_paths.items():
        declared = source.parent / value
        if declared.is_symlink():
            raise InputError(f"Checkpoint symlinks are not a verified inventory: {declared}")
        rebound_paths[key] = declared.resolve()
    settings.model_paths = rebound_paths
    return settings


class TelemetryError(RuntimeError):
    """Enforcement cannot continue when its telemetry is unavailable."""


class _SignalInterrupt(KeyboardInterrupt):
    """Route parent SIGTERM through the same partial-safe path as Ctrl-C."""


@contextmanager
def _interrupt_on_sigterm() -> Iterator[None]:
    # Python permits signal handlers only in the main thread. Never replace a
    # notebook's SIGINT handler or inspect its unrelated sibling processes.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum: int, frame: Any) -> None:
        raise _SignalInterrupt("SIGTERM: interrupted by external signal")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def _uninterrupted_cleanup() -> Iterator[None]:
    """Let bounded TERM/KILL/reaping finish even after repeated stop signals."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    total_bytes: int
    free_bytes: int


def _smi(query: str, timeout_s: float = DEFAULT_SMI_TIMEOUT_S) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TelemetryError(f"nvidia-smi enforcement telemetry unavailable: {exc}") from exc
    return result.stdout


def _integer(value: str, label: str, *, positive: bool = False) -> int:
    if not re.fullmatch(r"[0-9]+", value.strip()):
        raise TelemetryError(f"Unavailable/invalid {label}: {value!r}")
    number = int(value)
    if positive and number == 0:
        raise TelemetryError(f"{label} must be positive")
    return number


def query_gpus(timeout_s: float = DEFAULT_SMI_TIMEOUT_S) -> list[GPU]:
    rows = csv.reader(io.StringIO(_smi("--query-gpu=index,uuid,name,memory.total,memory.free", timeout_s)))
    devices: list[GPU] = []
    for row in rows:
        if len(row) != 5:
            raise TelemetryError(f"Invalid GPU inventory row: {row!r}")
        index, uuid, name, total, free = (item.strip() for item in row)
        if not uuid.startswith("GPU-") or not name:
            raise TelemetryError("Expected a physical GPU UUID and measured GPU name")
        device = GPU(_integer(index, "GPU index"), uuid, name,
                     _integer(total, "GPU total MiB", positive=True) * MIB,
                     _integer(free, "GPU free MiB") * MIB)
        if device.free_bytes > device.total_bytes:
            raise TelemetryError("GPU free memory exceeds total memory")
        devices.append(device)
    if not devices or len({d.uuid for d in devices}) != len(devices) or len({d.index for d in devices}) != len(devices):
        raise TelemetryError("Missing or ambiguous physical GPU inventory")
    return devices


def query_process_vram(uuid: str, pids: set[int], timeout_s: float = DEFAULT_SMI_TIMEOUT_S) -> int:
    """Selected GPU's compute-app memory for the observed orchestration tree.

    An operational empty compute-app list measures zero. N/A, malformed rows or a
    failed query are unavailable, never zero. Other devices are never pooled.
    """
    rows = csv.reader(io.StringIO(_smi("--query-compute-apps=pid,gpu_uuid,used_gpu_memory", timeout_s)))
    # Some driver/toolkit combinations emit the same process more than once
    # (for example, once per compute context). `used_gpu_memory` is already a
    # per-process value, so duplicate rows must not be summed or treated as a
    # fatal telemetry outage. Keep the largest reported value for each PID.
    per_pid: dict[int, int] = {}
    for row in rows:
        if len(row) != 3:
            raise TelemetryError(f"Invalid GPU process telemetry row: {row!r}")
        raw_pid, device, memory = (item.strip() for item in row)
        if device != uuid:
            continue
        pid = _integer(raw_pid, "GPU process PID", positive=True)
        # Even another process's N/A indicates this GPU's telemetry is incomplete.
        used = _integer(memory, "GPU process used-memory MiB") * MIB
        per_pid[pid] = max(per_pid.get(pid, 0), used)
    return sum(memory for pid, memory in per_pid.items() if pid in pids)


@dataclass(frozen=True)
class Budgets:
    ram_bytes: int
    vram_bytes: int
    system_available_floor_bytes: int
    gpu_free_floor_bytes: int


def select_budget(settings: KaggleSettings, devices: Sequence[GPU], available_bytes: int) -> tuple[GPU, Budgets]:
    matches = [device for device in devices if device.index == settings.gpu_index]
    if len(matches) != 1:
        raise TelemetryError(f"No unique physical GPU at index {settings.gpu_index}")
    device = matches[0]
    ram = int(min(settings.ram_limit_gb * GB - settings.ram_reserve_gb * GB,
                  available_bytes - settings.ram_reserve_gb * GB))
    reserve = int(settings.vram_reserve_gb * GB)
    vram = int(min(settings.vram_limit_gb * GB - reserve,
                   device.total_bytes * settings.gpu_memory_fraction,
                   device.total_bytes - reserve, device.free_bytes - reserve))
    if ram <= 0 or vram <= 0 or available_bytes < settings.min_system_available_gb * GB:
        raise TelemetryError("Insufficient measured memory after configured headroom")
    return device, Budgets(ram, vram, int(settings.min_system_available_gb * GB), reserve)


def worker_environment(settings: KaggleSettings, gpu: GPU | None) -> dict[str, str]:
    env = {**os.environ, **OFFLINE_ENV}
    env["CUDA_VISIBLE_DEVICES"] = gpu.uuid if gpu is not None else ""
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(settings.cpu_threads)
    # A source-checkout entrypoint and its workers also work without an editable install.
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")])
    return env


def _tree_processes(worker_pid: int | None, known: dict[int, psutil.Process]) -> list[psutil.Process]:
    parent = psutil.Process(os.getpid())
    processes = [parent]
    if worker_pid is not None:
        try:
            worker = psutil.Process(worker_pid)
            processes.extend([worker, *worker.children(recursive=True)])
        except psutil.NoSuchProcess:
            pass
    # Keep previously observed descendants, including orphans/escaped children.
    # Group signalling cleans unobserved group members without scanning the host.
    for process in processes:
        known.setdefault(process.pid, process)
    alive: list[psutil.Process] = []
    for process in known.values():
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                alive.append(process)
        except psutil.NoSuchProcess:
            continue
    return alive


def sample_resources(worker_pid: int | None, gpu: GPU | None,
                     known: dict[int, psutil.Process], *,
                     smi_timeout_s: float = DEFAULT_SMI_TIMEOUT_S) -> dict[str, Any]:
    try:
        processes = _tree_processes(worker_pid, known)
        rss = 0
        pids: set[int] = set()
        for process in processes:
            try:
                rss += process.memory_info().rss
                pids.add(process.pid)
            except psutil.NoSuchProcess:
                continue
        available = psutil.virtual_memory().available
    except (psutil.Error, OSError) as exc:
        raise TelemetryError(f"RAM enforcement telemetry unavailable: {exc}") from exc
    used: int | None = None
    free: int | None = None
    if gpu is not None:
        current = next((item for item in query_gpus(smi_timeout_s) if item.uuid == gpu.uuid), None)
        if current is None or current.total_bytes != gpu.total_bytes:
            raise TelemetryError("Selected physical GPU disappeared or its inventory changed")
        free = current.free_bytes
        used = query_process_vram(gpu.uuid, pids, smi_timeout_s)
    return {"rss_bytes": rss, "vram_bytes": used, "system_available_bytes": available,
            "gpu_free_bytes": free, "process_count": len(pids)}


def _violation(sample: dict[str, Any], budget: Budgets) -> str | None:
    for field, maximum, label in (("rss_bytes", budget.ram_bytes, "aggregate process-tree RAM budget"),
                                  ("vram_bytes", budget.vram_bytes, "selected-GPU process-tree VRAM budget")):
        value = sample[field]
        if value is not None and value > maximum:
            return f"Exceeded {label}: {value} > {maximum} bytes"
    if sample["system_available_bytes"] < budget.system_available_floor_bytes:
        return "Host available memory fell below min_system_available_gb"
    if sample["gpu_free_bytes"] is not None and sample["gpu_free_bytes"] < budget.gpu_free_floor_bytes:
        return "Selected GPU free memory fell below vram_reserve_gb"
    return None


def terminate_group(process: subprocess.Popen[bytes], grace_s: float,
                    known: dict[int, psutil.Process]) -> list[int]:
    """Clean the worker group, including orphans and observed escaped children."""
    errors: list[str] = []
    try:
        _tree_processes(process.pid, known)
    except (psutil.Error, OSError) as exc:
        # Lost process telemetry must never bypass group TERM/KILL/reaping.
        errors.append(str(exc))
    targets = [item for pid, item in known.items() if pid not in {os.getpid(), process.pid}]
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            errors.append(str(exc))
    # Popen must reap its own child, otherwise psutil.wait_procs can consume its
    # wait status and turn a signal/nonzero exit into an incorrectly recorded zero.
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    _, alive = psutil.wait_procs(targets, timeout=grace_s)
    # Do this even if the group leader has exited: native descendants may survive.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            errors.append(str(exc))
    process.wait(timeout=grace_s)
    _, alive = psutil.wait_procs(alive, timeout=grace_s)
    if errors:
        raise TelemetryError("Worker group was signalled; descendant cleanup could not be fully verified: " + "; ".join(errors))
    return [item.pid for item in alive if item.is_running() and item.status() != psutil.STATUS_ZOMBIE]


def monitor_command(command: Sequence[str], *, env: dict[str, str], log: Path,
                    settings: KaggleSettings, budget: Budgets, gpu: GPU | None,
                    sampler: Callable[[int | None, GPU | None, dict[int, psutil.Process]], dict[str, Any]] | None = None,
                    timeout_s: float | None = None) -> dict[str, Any]:
    """Execute a real subprocess; bounded report recovery may set timeout_s."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, Any] = {"rss_peak_bytes": None, "vram_peak_bytes": None,
                              "system_available_min_bytes": None, "gpu_free_min_bytes": None,
                              "samples": 0, "stop_reason": None, "telemetry_error": None,
                              "last_sample": None, "cleanup_survivor_pids": []}
    known: dict[int, psutil.Process] = {}
    with _interrupt_on_sigterm(), log.open("ab", buffering=0) as stream:
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(list(command), env=env, stdout=stream, stderr=stream,
                                       start_new_session=True)
            result["pid"] = process.pid
            while True:
                try:
                    sample = (sample_resources(process.pid, gpu, known, smi_timeout_s=settings.smi_timeout_s)
                              if sampler is None else sampler(process.pid, gpu, known))
                except TelemetryError as exc:
                    result["telemetry_error"] = str(exc)
                    result["stop_reason"] = str(exc)
                    break
                result["samples"] += 1
                result["last_sample"] = sample
                for source, dest, reducer in (
                    ("rss_bytes", "rss_peak_bytes", max), ("vram_bytes", "vram_peak_bytes", max),
                    ("system_available_bytes", "system_available_min_bytes", min),
                    ("gpu_free_bytes", "gpu_free_min_bytes", min),
                ):
                    value = sample[source]
                    if value is not None:
                        previous = result[dest]
                        result[dest] = value if previous is None else reducer(previous, value)
                reason = _violation(sample, budget)
                if reason:
                    result["stop_reason"] = reason
                    break
                if process.poll() is not None:
                    break
                if timeout_s is not None and time.perf_counter() - started >= timeout_s:
                    result["stop_reason"] = "Bounded report recovery exceeded its wall-time allowance"
                    break
                time.sleep(settings.monitor_interval_s)
        except KeyboardInterrupt as exc:
            result["stop_reason"] = str(exc) or "KeyboardInterrupt: interrupted by user"
        finally:
            if process is not None:
                survivors: list[int] | None
                with _uninterrupted_cleanup():
                    try:
                        survivors = terminate_group(process, settings.termination_grace_s, known)
                    except (TelemetryError, psutil.Error, OSError, subprocess.SubprocessError) as exc:
                        survivors = None
                        result["cleanup_error"] = f"Worker cleanup could not be verified: {exc}"
                        result["stop_reason"] = result["stop_reason"] or result["cleanup_error"]
                result["cleanup_survivor_pids"] = survivors
                if survivors:
                    result["cleanup_error"] = f"Worker descendants survived cleanup: {survivors}"
                    result["stop_reason"] = result["stop_reason"] or result["cleanup_error"]
                result["returncode"] = process.returncode
            else:
                result["returncode"] = None
            result["wall_seconds"] = time.perf_counter() - started
    return result


def bind_registry(settings: KaggleSettings, cfg: Config) -> tuple[Path, dict[str, Any]]:
    """Rebind only paths; Registry verifies exact local inventories and hashes.

    The deterministic working registry lives outside stage outputs and is copied
    into bundle inputs by the core Bundle contract. No source registry is edited.
    Missing optional models retain their real fallback/unavailable behavior.
    """
    source = Registry(settings.registry, settings.model_cache)
    document = {"models": json.loads(json.dumps(source.models))}
    optional = {cfg.masking.model, cfg.semantics.model} - {None, cfg.depth.model}
    for name, location in settings.model_paths.items():
        if name not in source.models or not isinstance(source.models[name], dict):
            if name in optional:
                continue  # No invented spec; resolve() records optional unavailability below.
            raise InputError(f"model_paths names an absent/invalid registry entry: {name}")
        if location.is_symlink():
            raise InputError(f"Checkpoint symlinks are not a verified inventory: {location}")
        document["models"][name]["path"] = str(location.resolve())
    digest = content_hash(document)
    destination = settings.output_root.parent / ".drishti-kaggle" / "registries" / digest / "models.json"
    atomic_json(destination, document)  # JSON is valid YAML; avoid timestamps in this input.
    registry = Registry(destination, settings.model_cache)
    inventory: dict[str, Any] = {}
    selected = {cfg.depth.model, cfg.masking.model, *settings.model_paths}
    if cfg.semantics.model is not None:
        selected.add(cfg.semantics.model)
    for name in sorted(selected):
        try:
            resolved = registry.resolve(name)
            inventory[name] = {"valid": True, "spec": resolved, "executed": False}
        except DependencyError as exc:
            if name == cfg.depth.model or (name in settings.model_paths and name not in optional):
                raise
            inventory[name] = {"valid": False, "reason": str(exc), "executed": False}
    return destination, inventory


def declared_duration(path: Path) -> dict[str, Any]:
    """Probe real PyAV metadata only, never infer duration from a synthetic clip."""
    try:
        import av
    except ImportError as exc:
        raise DependencyError("Full-flight metadata probing requires the ingest extra (PyAV)") from exc
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise InputError(f"Video has no video stream: {path}")
        stream = container.streams.video[0]
        candidates = [
            ("video_stream.duration*time_base", float(stream.duration * stream.time_base)
             if stream.duration is not None and stream.time_base is not None else None),
            ("container.duration/av.time_base", float(container.duration / av.time_base)
             if container.duration is not None else None),
        ]
        for source, value in candidates:
            if value is not None and math.isfinite(value) and value > 0:
                return {"declared_duration_s": value, "duration_source": source,
                        "duration_kind": "declared_metadata", "duration_measured_by_decode": False}
    return {"declared_duration_s": None, "duration_source": None,
            "duration_kind": "unavailable_metadata", "duration_measured_by_decode": False}


def sampling_plan(cfg: Config, settings: KaggleSettings) -> tuple[Config, dict[str, Any]]:
    if cfg.dataset is None:
        raise InputError("A real dataset is required for full-flight sampling")
    metadata = declared_duration(cfg.dataset.video)
    duration = metadata["declared_duration_s"]
    fps = cfg.ingest.target_fps
    if duration is not None:
        fps = min(fps, (settings.sample_frame_budget - 1) / duration)
    plan = {**metadata, "profile_target_fps": cfg.ingest.target_fps, "target_fps": fps,
            "sample_frame_budget": settings.sample_frame_budget, "max_frames": 0,
            "estimated_frames": math.floor(duration * fps) + 1 if duration is not None else None,
            "estimate_status": "approximate_metadata_bound" if duration is not None else "unbounded_unknown_duration",
            "interpretation": "Full-clip timestamp sampling, never an early max_frames cutoff; metadata and input cadence do not guarantee a usable SfM solve or an exact frame count."}
    data = cfg.model_dump(mode="python")
    data["ingest"].update(target_fps=fps, max_frames=0)
    return Config.model_validate(data), plan


def _explicit(settings: KaggleSettings) -> set[str]:
    return settings.model_fields_set if settings._explicit_fields is None else settings._explicit_fields


def _configured(settings: KaggleSettings, bundle: Bundle | None) -> tuple[Config, dict[str, Any]]:
    overrides = resolve_paths(report_layer(overrides_from_strings(settings.overrides)), settings._base_dir)
    if settings.dataset is not None:
        cfg = load_config(settings.dataset, settings.profile, overrides, environ={})
    elif bundle is not None:
        data = bundle.config().model_dump(mode="python")
        if "profile" in _explicit(settings):
            data = merge(data, resolve_paths(read_yaml(settings.profile), settings.profile.parent))
        data = merge(data, overrides)
        if data.get("dataset") is not None:
            data["dataset"]["report"] = data["report"]
        cfg = Config.model_validate(data)
    else:
        raise InputError("dataset or an existing bundle is required")
    previous = bundle.manifest.get("cloud_execution", {}).get("sampling", {}) if bundle else {}
    requested = bundle is None or settings.dataset is not None or "profile" in _explicit(settings) or "sample_frame_budget" in _explicit(settings) or "ingest" in overrides
    if requested:
        if cfg.dataset is None:
            raise InputError("Mandatory dataset is absent")
        candidate_fps = cfg.ingest.target_fps
        if bundle is not None and settings.dataset is None and "profile" not in _explicit(settings) and "target_fps" not in overrides.get("ingest", {}):
            candidate_fps = previous.get("profile_target_fps", candidate_fps)
        request = {"video": hash_path(cfg.dataset.video),
                   "profile": hash_path(settings.profile) if (bundle is None or settings.dataset is not None or "profile" in _explicit(settings)) else previous.get("request", {}).get("profile"),
                   "sample_frame_budget": settings.sample_frame_budget,
                   "configured_target_fps": candidate_fps,
                   "ingest_overrides": overrides.get("ingest", {})}
        if previous.get("request") == request:
            # The same explicit plan must not compound the already reduced fps.
            cfg.ingest.target_fps = previous["target_fps"]
            cfg.ingest.max_frames = 0
            plan = {**previous, "resume_decision": "unchanged_explicit_plan_preserved"}
        else:
            # Budget-only changes use the originally configured fps, not its reduced value.
            cfg.ingest.target_fps = candidate_fps
            cfg, plan = sampling_plan(cfg, settings)
            plan.update(request=request, resume_decision="new_or_changed_explicit_plan")
    else:
        # Preserve the saved sampling cadence, but clear a legacy early cutoff.
        # This config change participates in normal S0/downstream freshness.
        plan = {**previous, "target_fps": cfg.ingest.target_fps, "max_frames": 0,
                "legacy_truncation_cleared": cfg.ingest.max_frames > 0,
                "resume_decision": "saved_fps_preserved"}
        cfg.ingest.max_frames = 0
    # Runner policy is explicit, even if a descriptor/override supplies stale paths.
    data = cfg.model_dump(mode="python")
    data["run"].update(copy_inputs=True, output_root=settings.output_root)
    data["models"].update(registry=settings.registry, cache_dir=settings.model_cache, offline=True)
    data["compute"]["device"] = "cuda"
    return Config.model_validate(data), plan


def _copy_bundle(source: Path, destination: Path) -> Bundle:
    import shutil

    original = Bundle.load(source)
    if original.manifest.get("manifest_sha256") != manifest_digest(original.manifest):
        raise InputError("Resume manifest integrity mismatch")
    if source.resolve() == destination.resolve():
        return original
    if destination.exists():
        raise InputError(f"Resume destination already exists; point resume_bundle at it: {destination}")
    if destination.resolve().is_relative_to(source.resolve()):
        raise InputError("Cannot copy a resume bundle into itself")
    # Reject links before copying, including a link in inputs or logs.
    hash_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return Bundle.load(destination)


def _mark_interrupted(bundle: Bundle, name: str, reason: str) -> None:
    record = bundle.manifest["stages"].setdefault(name, {"name": name, "outputs": {}})
    was_running = record.get("status") == "running"
    previous_status = record.get("status")
    if not was_running:
        for key in ("metrics", "models"):
            if key in record:
                record[f"retained_{key}"] = record.pop(key)
    record.update(status="failed", confidence=None, degraded_reason=reason,
                  ended_utc=utc_now(), error={"type": "CloudExecutionInterrupted", "message": reason},
                  interrupted_before_stage=not was_running, previous_status=previous_status,
                  outputs={})
    record.pop("output_hashes", None)
    record.pop("outputs_hash", None)
    # Core-stage wall time was never completed/measured in the terminated worker.
    record["wall_seconds"] = None
    directory = bundle.path(name)
    if directory.is_dir() and any(directory.iterdir()):
        record["outputs"] = {"partial": name}
        try:
            hashes = bundle.output_hashes(record["outputs"])
            record.update(output_hashes=hashes, outputs_hash=content_hash(hashes))
        except (OSError, ValueError, InputError) as exc:
            record["partial_integrity_error"] = str(exc)
            record.pop("output_hashes", None)
            record.pop("outputs_hash", None)
    affected = {name}
    for stage in STAGES:
        if stage.name != name and any(dep in affected for dep in stage.requires):
            affected.add(stage.name)
            existing = bundle.manifest["stages"].get(stage.name)
            if existing is not None:
                existing.update(status="stale", confidence=None,
                                degraded_reason="An upstream cloud worker was interrupted; retained artifacts are not current")
            else:
                bundle.manifest["stages"][stage.name] = {
                    "name": stage.name, "status": "blocked", "outputs": {}, "confidence": None,
                    "blocked_by": [dep for dep in stage.requires if dep in affected],
                    "degraded_reason": "An upstream cloud worker was interrupted before this stage could run",
                }
    bundle.manifest.update(status="partial", wall_seconds=None,
                           error={"type": "CloudExecutionInterrupted", "message": reason})
    bundle.save()


def archive_bundle(root: Path, destination: Path) -> Path:
    """Stream a real bundle to an atomic ZIP; reject links, specials and self-inclusion."""
    root = root.resolve()
    if destination.is_symlink() or destination.resolve().is_relative_to(root):
        raise InputError("Archive must be outside the bundle and cannot be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for directory, dirs, files in os.walk(root, followlinks=False):
                for name in sorted([*dirs, *files]):
                    item = Path(directory) / name
                    mode = item.lstat().st_mode
                    if stat.S_ISLNK(mode) or not item.resolve().is_relative_to(root):
                        raise InputError(f"Escaping/symlink bundle entry cannot be archived: {item}")
                    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                        raise InputError(f"Non-regular bundle entry cannot be archived: {item}")
                    relative = item.relative_to(root).as_posix()
                    if "\\" in relative:
                        raise InputError(f"Ambiguous cross-platform archive entry: {relative}")
                    archive.write(item, arcname=f"{root.name}/{relative}")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def _persist_cloud(bundle: Bundle, cloud: dict[str, Any]) -> None:
    bundle.manifest["cloud_execution"] = cloud
    bundle.manifest["wall_seconds"] = None
    bundle.write_json("cloud_run.json", cloud)
    bundle.save()


def _report_failure(bundle: Bundle, message: str) -> None:
    """Retain and rehash real report files if a recovery worker was interrupted."""
    bundle.manifest.update(status="partial", report_error={"type": "CloudReportRefreshFailed", "message": message})
    record = bundle.manifest["stages"].get("report")
    if record is not None:
        record.update(status="failed", confidence=None, degraded_reason=message)
    outputs = {key: path for key, path in {"json": "report/report.json", "html": "report/report.html"}.items()
               if bundle.path(path).is_file()}
    if outputs:
        try:
            hashes = bundle.output_hashes(outputs)
            bundle.manifest["partial_report"] = {"outputs": outputs, "output_hashes": hashes,
                                                 "outputs_hash": content_hash(hashes)}
            if record is not None:
                record.update(outputs=outputs, output_hashes=hashes, outputs_hash=content_hash(hashes))
        except (OSError, ValueError, InputError) as exc:
            bundle.manifest["report_integrity_error"] = str(exc)
            bundle.manifest.pop("partial_report", None)
            if record is not None:
                record.update(outputs=outputs)
                record.pop("output_hashes", None)
                record.pop("outputs_hash", None)


def _report_worker(bundle: Bundle) -> None:
    from drishti.report.builder import build_report

    outputs = build_report(bundle, bundle.config())
    hashes = bundle.output_hashes(outputs)
    record = bundle.manifest["stages"].get("report")
    if record is not None:
        record.update(outputs=outputs, output_hashes=hashes, outputs_hash=content_hash(hashes))
    if bundle.manifest["status"] == "partial":
        bundle.manifest["partial_report"] = {"outputs": outputs, "output_hashes": hashes,
                                             "outputs_hash": content_hash(hashes)}
    else:
        bundle.manifest.pop("partial_report", None)
    bundle.manifest.pop("report_error", None)
    bundle.save()


def _worker(bundle_path: Path, stage: str, result_path: Path, request_path: Path) -> int:
    os.environ.update(OFFLINE_ENV)
    result: dict[str, Any] = {"torch_max_allocated_bytes": None, "torch_max_reserved_bytes": None,
                              "torch_memory_fraction": None, "torch_total_bytes": None,
                              "stage_executed": None, "executed_stages": None, "stage_status": None,
                              "stage_execution_scope": "Core executed_stages records invocation/attempt, not proof that every algorithm or model ran",
                              "runtime": {"executed": False, "scope": "CUDA detection/allocator setup is not stage or model inference"}}
    torch: Any = None
    code = 1
    try:
        bundle = Bundle.load(bundle_path)
        if stage == "refresh_report":
            _report_worker(bundle)
            code = 0
        else:
            with request_path.open(encoding="utf-8") as stream:
                request = json.load(stream)
            import torch as torch_module

            torch = torch_module
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise DependencyError("Worker requires exactly one operational visible Torch CUDA GPU")
            total = int(torch.cuda.get_device_properties(0).total_memory)
            effective = min(int(request["vram_bytes"]), int(total * request["gpu_memory_fraction"]),
                            total - int(request["vram_reserve_bytes"]))
            if effective <= 0:
                raise DependencyError("Torch reports insufficient CUDA memory after headroom")
            fraction = effective / total
            # Before Runtime.detect(), model loaders, COLMAP, or the graph runtime.
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            torch.cuda.reset_peak_memory_stats(0)
            torch.set_num_threads(int(request["cpu_threads"]))
            result.update(torch_memory_fraction=fraction, torch_total_bytes=total,
                          torch_effective_vram_bytes=effective)
            cfg = bundle.config()
            outcome = StageRunner(bundle, cfg).run(selected=[stage], only=True)
            executed = outcome.manifest["executions"][-1]["executed_stages"]
            result.update(executed_stages=executed, stage_executed=stage in executed,
                          stage_status=outcome.manifest["stages"].get(stage, {}).get("status"))
            code = 0 if outcome.manifest["stages"].get(stage, {}).get("status") in SUCCESS else 1
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"Kaggle worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if torch is not None and result["torch_memory_fraction"] is not None:
            try:
                result["torch_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
                result["torch_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
            except Exception as exc:
                result["torch_peak_error"] = str(exc)
        result["exit_code"] = code
        atomic_json(result_path, result)
    return code


def _worker_command(bundle: Bundle, stage: str) -> tuple[list[str], Path]:
    result = bundle.path(f"cloud/workers/{stage}.json")
    # A killed worker must not inherit the previous invocation's allocator peaks.
    result.unlink(missing_ok=True)
    command = [sys.executable, "-m", "drishti.runtime.kaggle", "--worker", str(bundle.root),
               "--stage", stage, "--result", str(result),
               "--request", str(bundle.path("cloud/worker_request.json"))]
    return command, result


def _merge_peaks(cloud: dict[str, Any], measurements: dict[str, Any]) -> None:
    for key in ("rss_peak_bytes", "vram_peak_bytes"):
        value = measurements.get(key)
        if value is not None:
            previous = cloud[key]
            cloud[key] = value if previous is None else max(previous, value)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _progress(message: str) -> None:
    print(f"[DRISHTI Kaggle] {message}", file=sys.stderr, flush=True)


def _budget_summary(cloud: dict[str, Any]) -> None:
    def decimal_gb(value: int | None) -> str:
        return "unavailable" if value is None else f"{value / GB:.3f} decimal GB"

    limits = cloud.get("limits") or {}
    _progress(
        f"Final status={cloud['status']}; sampled RAM peak={decimal_gb(cloud['rss_peak_bytes'])} "
        f"(stop budget={decimal_gb(limits.get('ram_bytes'))}); "
        f"sampled selected-GPU VRAM peak={decimal_gb(cloud['vram_peak_bytes'])} "
        f"(stop budget={decimal_gb(limits.get('vram_bytes'))}). "
        "Sampled peaks are lower bounds; this is not a hard instantaneous memory limit."
    )


def _freeze_report_cloud(bundle: Bundle, cloud: dict[str, Any]) -> None:
    """Freeze the report's explicit pre-recovery scope; no self-referential hashes.

    The report builder reads manifest.cloud_execution. Its final refresh sees
    exactly this snapshot. Later recovery/ZIP measurements belong to cloud_run.json,
    not a recursive series of report refreshes that can never measure themselves.
    """
    cloud["report_cloud_snapshot"] = {
        "scope": "completed_stage_workers_before_final_report_recovery_and_archive",
        "captured_utc": utc_now(), "path": "cloud/report_snapshot.json",
        "authoritative_final_record": "cloud_run.json",
        "excludes": ["final CPU report recovery resources/result", "streaming archive result"],
        "interpretation": "Report cloud status, elapsed time and sampled peaks describe this snapshot; later results and peaks are in the final cloud_run.json. Model inventory/package/resource detection is not inference.",
    }
    bundle.write_json("cloud/report_snapshot.json", cloud)
    _persist_cloud(bundle, cloud)


def _invalidate_retained(bundle: Bundle, cfg: Config) -> None:
    """Do not report old geometry as current after a failed changed-input resume.

    Reuse core fingerprints, with actual package metadata and without importing
    Torch into the orchestrator. The core's per-invocation caches avoid repeatedly
    hashing a full flight video while checking the retained DAG.
    """
    versions = _package_versions()
    checker = StageRunner(bundle, cfg, runtime=Runtime("cuda", {"versions": versions}))
    checker._input_states = {role: source_hash(role, path) for role, path in input_paths(cfg).items()}
    checker._model_states = {}
    for spec in STAGES:
        record = bundle.manifest["stages"].get(spec.name)
        if record is not None and record.get("status") in SUCCESS:
            if any(bundle.manifest["stages"].get(dep, {}).get("status") not in SUCCESS for dep in spec.requires) or not checker.is_fresh(spec.name, recursive=False):
                record.update(status="stale", confidence=None,
                              degraded_reason="Retained products failed current content/config/code/package freshness after cloud interruption")


def run(settings: KaggleSettings) -> dict[str, Any]:
    """Run the actual offline DAG. Return paths/status; partial/aborted has exit_code=1.

    No retries with downgraded quality. Caller uses exit_code as its process exit
    status. Failures before a valid bundle can be created raise an actionable error.
    """
    with _interrupt_on_sigterm():
        return _run(settings)


def _run(settings: KaggleSettings) -> dict[str, Any]:
    os.environ.update(OFFLINE_ENV)
    started = time.perf_counter()
    source_bundle = Bundle.load(settings.resume_bundle) if settings.resume_bundle is not None else None
    if source_bundle is not None and source_bundle.manifest.get("manifest_sha256") != manifest_digest(source_bundle.manifest):
        raise InputError("Resume manifest integrity mismatch")
    cfg, plan = _configured(settings, source_bundle)
    registry, inventory = bind_registry(settings, cfg)
    cfg.models.registry = registry
    destination = settings.output_root / settings.run_id
    if settings.resume_bundle is None:
        bundle = Bundle.create(destination, cfg, settings.run_id)
        cfg = bundle.config()
    else:
        bundle = _copy_bundle(settings.resume_bundle, destination)
    cloud: dict[str, Any] = {
        "schema_version": "1", "status": "running", "started_utc": utc_now(),
        "ended_utc": None, "orchestration_wall_seconds": None, "stop_reason": None,
        "wall_time_scope": "input preparation through final report recovery, excluding streaming ZIP creation",
        "gpu": None, "limits": None, "settings": settings.model_dump(mode="json"),
        "runtime": {"executed": False, "package_versions": _package_versions(),
                    "python": sys.version, "python_executable": sys.executable,
                    "scope": "Detected resources/packages only; current stage invocation evidence is in workers, actual inference provenance in core stage records"},
        "sampling": plan, "model_inventory": inventory, "workers": [],
        "rss_peak_bytes": None, "vram_peak_bytes": None,
        "report_refresh": None, "report_cloud_snapshot": None,
        "report_recovery_wall_limit_s": settings.report_recovery_timeout_s,
        "archive_path": None, "archive_error": None,
        "archive_status": "pending" if settings.archive else "disabled",
        "archive_scope": "ZIP contains the pre-creation bundle snapshot; its archive result is pending there. Final cloud_run.json and stdout summary record the actual creation result, without recursive ZIP rewriting.",
        "output": "cloud_run.json", "memory_method": {
            "ram": "psutil aggregate orchestrator/current worker/native descendant RSS; shared pages may be counted multiple times",
            "vram": "nvidia-smi selected physical GPU UUID compute-app PID memory for observed process tree",
            "gpu_scope": "one selected physical GPU; device memories are never summed",
            "monitor_interval_s": settings.monitor_interval_s,
            "monitor_interval_kind": "polling sleep excluding query/collection time, not a maximum detection latency",
            "smi_timeout_s": settings.smi_timeout_s,
            "torch_allocator": "per-process fraction before stage runtime; does not cap every native CUDA allocation",
            "sampled_peaks_are_lower_bounds": True, "hard_instantaneous_limit": False,
            "scope": "stage workers and bounded report recovery; input preparation and streaming ZIP are not continuously sampled",
        },
    }
    _persist_cloud(bundle, cloud)
    gpu: GPU | None = None
    budget: Budgets | None = None
    active: str | None = None
    try:
        if settings.resume_bundle is not None:
            cfg = bundle.refresh_inputs(cfg)
            bundle.snapshot(cfg)
            bundle.save()
        available = psutil.virtual_memory().available
        gpu, budget = select_budget(settings, query_gpus(settings.smi_timeout_s), available)
        cloud["gpu"] = asdict(gpu)
        cloud["limits"] = {**asdict(budget), "ram_limit_bytes": int(settings.ram_limit_gb * GB),
                           "vram_limit_bytes": int(settings.vram_limit_gb * GB),
                           "initial_system_available_bytes": available,
                           "effective_torch_fraction_of_measured_total": budget.vram_bytes / gpu.total_bytes}
        bundle.write_json("cloud/worker_request.json", {
            "vram_bytes": budget.vram_bytes, "vram_reserve_bytes": int(settings.vram_reserve_gb * GB),
            "gpu_memory_fraction": settings.gpu_memory_fraction, "cpu_threads": settings.cpu_threads,
        })
        _persist_cloud(bundle, cloud)
        _progress(f"Selected physical GPU {gpu.index}: {gpu.name} ({gpu.uuid}); "
                  f"RAM stop budget={budget.ram_bytes / GB:.3f} decimal GB; "
                  f"VRAM stop budget={budget.vram_bytes / GB:.3f} decimal GB")
        _progress(f"Full-flight sampling target_fps={plan['target_fps']:.6g}; "
                  "ingest.max_frames=0; configured quality is never automatically downgraded")
        env = worker_environment(settings, gpu)
        # STAGES is the authoritative topologically ordered roster, including requires.
        for index, spec in enumerate(STAGES, start=1):
            active = spec.name
            command, result_path = _worker_command(bundle, active)
            log = bundle.path(f"logs/kaggle_{active}.log")
            _progress(f"[{index}/{len(STAGES)}] {active} started; log={log}")
            measurements = monitor_command(command, env=env, log=log,
                                           settings=settings, budget=budget, gpu=gpu)
            bundle = Bundle.load(bundle.root)
            worker_data = read_yaml(result_path) if result_path.is_file() else {
                "torch_max_allocated_bytes": None, "torch_max_reserved_bytes": None,
                "stage_executed": None, "executed_stages": None,
                "reason": "Worker terminated before allocator telemetry could be published",
            }
            cloud["workers"].append({"stage": active, "requires": list(spec.requires),
                                     **measurements, "worker": worker_data,
                                     "result_path": result_path.relative_to(bundle.root).as_posix() if result_path.is_file() else None})
            _merge_peaks(cloud, measurements)
            reason = measurements["stop_reason"]
            if reason is not None:
                _mark_interrupted(bundle, active, reason)
                cloud.update(status="aborted", stop_reason=reason)
            elif measurements["returncode"] != 0:
                record = bundle.manifest["stages"].get(active, {})
                if worker_data.get("error") or record.get("status") not in {"failed", "blocked"}:
                    reason = worker_data.get("error", {}).get("message", f"Worker exited with code {measurements['returncode']}")
                    _mark_interrupted(bundle, active, reason)
                    cloud.update(status="aborted", stop_reason=reason)
                else:
                    cloud["status"] = "partial"
            cloud["workers"][-1]["stage_status"] = bundle.manifest["stages"].get(active, {}).get("status")
            _progress(f"{active} status={cloud['workers'][-1]['stage_status']}; "
                      f"worker_exit={measurements['returncode']}; "
                      f"invoked_stage={worker_data.get('stage_executed', 'unavailable')}; "
                      f"log={log}" + (f"; stopped: {reason}" if reason else ""))
            _persist_cloud(bundle, cloud)
            if cloud["status"] == "aborted":
                break
            active = None
    except (Exception, KeyboardInterrupt) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        cloud.update(status="aborted", stop_reason=reason)
        bundle = Bundle.load(bundle.root)
        if active is not None:
            _mark_interrupted(bundle, active, reason)
        bundle.manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
    if cloud["status"] in {"partial", "aborted"}:
        try:
            _invalidate_retained(bundle, cfg)
        except (Exception, KeyboardInterrupt) as exc:
            cloud["retained_freshness_error"] = f"{type(exc).__name__}: {exc}"
            for record in bundle.manifest["stages"].values():
                if record.get("status") in SUCCESS:
                    record.update(status="stale", confidence=None,
                                  degraded_reason="Current freshness could not be established after cloud interruption")
    records = bundle.manifest["stages"]
    complete = all(records.get(spec.name, {}).get("status") in SUCCESS for spec in STAGES)
    if cloud["status"] == "running":
        cloud["status"] = "degraded" if complete and any(records[spec.name]["status"] == "degraded" for spec in STAGES) else "done" if complete else "partial"
    bundle.manifest["status"] = cloud["status"] if cloud["status"] in {"done", "degraded"} else "partial"
    cloud.update(ended_utc=utc_now(), orchestration_wall_seconds=time.perf_counter() - started)
    _freeze_report_cloud(bundle, cloud)
    # Best-effort recovery uses a CPU-only worker and a finite memory/time allowance.
    # No GPU measurements are invented for it if GPU enforcement failed.
    if budget is None:
        available = psutil.virtual_memory().available
        recovery_ram = int(min(settings.ram_limit_gb * GB - settings.ram_reserve_gb * GB,
                               available - settings.ram_reserve_gb * GB))
        budget = Budgets(max(1, recovery_ram), 0, int(settings.min_system_available_gb * GB), 0)
    try:
        command, _ = _worker_command(bundle, "refresh_report")
        _progress(f"Final CPU report recovery started; log={bundle.path('logs/kaggle_refresh_report.log')}")
        recovery = monitor_command(command, env=worker_environment(settings, None),
                                   log=bundle.path("logs/kaggle_refresh_report.log"), settings=settings,
                                   budget=budget, gpu=None,
                                   timeout_s=settings.report_recovery_timeout_s)
        bundle = Bundle.load(bundle.root)
        cloud["report_refresh"] = recovery
        _merge_peaks(cloud, recovery)
        if recovery["returncode"] != 0 or recovery["stop_reason"]:
            _report_failure(bundle, recovery["stop_reason"] or "Report refresh worker failed; see log")
            if cloud["status"] != "aborted":
                cloud["status"] = "partial"
        _progress(f"Final report recovery worker_exit={recovery['returncode']}; "
                  f"stop_reason={recovery['stop_reason']}")
    except (Exception, KeyboardInterrupt) as exc:
        cloud["report_refresh"] = {"error": f"{type(exc).__name__}: {exc}"}
        cloud["status"] = "partial" if cloud["status"] != "aborted" else "aborted"
        _report_failure(bundle, f"{type(exc).__name__}: {exc}")
    cloud.update(ended_utc=utc_now(), orchestration_wall_seconds=time.perf_counter() - started)
    archive_path = settings.output_root.parent / settings.archive_name
    if settings.archive:
        cloud["archive_path"] = str(archive_path)
    _persist_cloud(bundle, cloud)
    archive: Path | None = None
    if settings.archive:
        try:
            _progress(f"Streaming bundle archive to {archive_path}")
            archive = archive_bundle(bundle.root, archive_path)
            cloud.update(archive_status="created", archive_completed_utc=utc_now())
            _persist_cloud(bundle, cloud)
        except (Exception, KeyboardInterrupt) as exc:
            cloud.update(archive_path=None, archive_status="failed", archive_error=f"{type(exc).__name__}: {exc}")
            cloud["status"] = "partial" if cloud["status"] != "aborted" else "aborted"
            bundle.manifest["status"] = "partial"
            _persist_cloud(bundle, cloud)
    _budget_summary(cloud)
    return {"status": cloud["status"], "exit_code": 0 if cloud["status"] in {"done", "degraded"} else 1,
            "bundle": str(bundle.root), "cloud_run": str(bundle.path("cloud_run.json")),
            "manifest": str(bundle.path("manifest.json")),
            "report_json": str(bundle.path("report/report.json")) if bundle.path("report/report.json").is_file() else None,
            "report_html": str(bundle.path("report/report.html")) if bundle.path("report/report.html").is_file() else None,
            "archive": str(archive) if archive is not None else None,
            "stop_reason": cloud["stop_reason"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline quality-focused DRISHTI Kaggle GPU runner (sampled memory watchdog).")
    parser.add_argument("--settings", type=Path, help="Strict Kaggle YAML settings; paths are YAML-relative.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        if args.stage is None or args.result is None or args.request is None:
            parser.error("Internal worker requires stage, result and request")
        return _worker(args.worker, args.stage, args.result, args.request)
    if args.settings is None:
        parser.error("--settings is required")
    try:
        summary = run(load_settings(args.settings))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Kaggle runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
