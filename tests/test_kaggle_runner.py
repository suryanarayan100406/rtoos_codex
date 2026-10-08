"""Constructed GPU/contracts boundaries and real OS subprocess checks.

The tiny checkpoint/input files below test inventory and bundle mechanics. They
are not trained checkpoints, real footage, reconstruction or Kaggle acceptance.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path
from typing import Any

import psutil
import pytest
from pydantic import ValidationError

from drishti.bundle import Bundle
from drishti.bundle.core import manifest_digest
from drishti.bundle.hashing import content_hash, hash_file, hash_path
from drishti.config.loader import load_config, read_yaml
from drishti.config.models import Config
from drishti.errors import DependencyError, InputError
from drishti.models.registry import Registry
from drishti.runtime import Runtime
from drishti.runtime import kaggle as k
from drishti.runtime.probe import PACKAGES
from drishti.stages.base import StageResult
from drishti.stages.runner import STAGES, StageRunner


@pytest.fixture(autouse=True)
def restore_offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in k.OFFLINE_ENV:
        monkeypatch.setenv(key, os.environ.get(key, "test-before-run"))


def write_json(path: Path, document: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def local_settings(tmp_path: Path, **extra: Any) -> k.KaggleSettings:
    video = tmp_path / "recorded-input-boundary.bin"
    video.write_bytes(b"input inventory boundary; not footage")
    telemetry = tmp_path / "gps.csv"
    telemetry.write_text("timestamp,latitude,longitude,abs_alt\n", encoding="utf-8")
    dataset = write_json(tmp_path / "dataset.yaml", {
        "name": "contract-boundary", "video": video.name,
        "telemetry": {"path": telemetry.name, "format": "csv", "altitude_datum": "ellipsoidal"},
    })
    profile = write_json(tmp_path / "profile.yaml", {"ingest": {"target_fps": 4.0}})
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint hash boundary, not a trained model")
    registry = write_json(tmp_path / "registry.yaml", {"models": {
        "depth_anything_v2_small": {
            "path": "/old/workstation/checkpoint.bin", "revision": hash_file(checkpoint),
            "license": "Apache-2.0", "runtime": "torch", "files": {checkpoint.name: hash_file(checkpoint)},
        },
    }})
    return k.KaggleSettings.model_validate({
        "dataset": dataset, "profile": profile, "registry": registry,
        "model_cache": tmp_path / "cache", "model_paths": {"depth_anything_v2_small": checkpoint},
        "output_root": tmp_path / "runs", "termination_grace_s": 0.1,
        "monitor_interval_s": 0.02, **extra,
    })


def configured_bundle(settings: k.KaggleSettings, monkeypatch: pytest.MonkeyPatch) -> Bundle:
    monkeypatch.setattr(k, "declared_duration", lambda _: {
        "declared_duration_s": 600.0, "duration_kind": "declared_metadata",
        "duration_source": "constructed_unit_boundary", "duration_measured_by_decode": False,
    })
    cfg, plan = k._configured(settings, None)
    registry, _ = k.bind_registry(settings, cfg)
    cfg.models.registry = registry
    bundle = Bundle.create(settings.output_root / settings.run_id, cfg, settings.run_id)
    bundle.manifest["cloud_execution"] = {"sampling": plan}
    bundle.save()
    return bundle


@pytest.mark.parametrize("fields", [
    {}, {"dataset": "data.yaml", "unknown": 1},
    {"dataset": "data.yaml", "ram_limit_gb": 30.001},
    {"dataset": "data.yaml", "vram_limit_gb": 31.0},
    {"dataset": "data.yaml", "ram_limit_gb": 0.0},
    {"dataset": "data.yaml", "vram_reserve_gb": 30.0},
    {"dataset": "data.yaml", "ram_reserve_gb": 31.0},
    {"dataset": "data.yaml", "ram_reserve_gb": 0.0},
    {"dataset": "data.yaml", "gpu_memory_fraction": 0.0},
    {"dataset": "data.yaml", "gpu_memory_fraction": 1.01},
    {"dataset": "data.yaml", "monitor_interval_s": float("nan")},
    {"dataset": "data.yaml", "smi_timeout_s": 0.0},
    {"dataset": "data.yaml", "smi_timeout_s": float("inf")},
    {"dataset": "data.yaml", "report_recovery_timeout_s": -1.0},
    {"dataset": "data.yaml", "vram_limit_gb": float("inf")},
    {"dataset": "data.yaml", "cpu_threads": True},
    {"dataset": "data.yaml", "cpu_threads": "8"},
    {"dataset": "data.yaml", "archive": "true"},
    {"dataset": "data.yaml", "gpu_index": -1},
    {"dataset": "data.yaml", "sample_frame_budget": 2},
    {"dataset": "data.yaml", "run_id": "../escape"},
    {"dataset": "data.yaml", "archive_name": "/tmp/escape.zip"},
    {"dataset": "data.yaml", "archive_name": "a\\b.zip"},
])
def test_strict_settings_reject_invalid(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        k.KaggleSettings.model_validate(fields)


def test_settings_paths_defaults_and_explicit_tracking(tmp_path: Path) -> None:
    source = write_json(tmp_path / "settings.yaml", {
        "resume_bundle": "bundle", "model_paths": {"depth": "weights"},
    })
    settings = k.load_settings(source)
    assert settings.resume_bundle == tmp_path / "bundle"
    assert settings.profile == tmp_path / "configs/profiles/kaggle_quality.yaml"
    assert settings.registry == tmp_path / "configs/models.yaml"
    assert settings.model_cache == tmp_path / "models"
    assert settings.model_paths == {"depth": tmp_path / "weights"}
    assert "profile" not in k._explicit(settings)
    assert settings.output_root == Path("/kaggle/working/drishti-runs")
    assert settings.ram_limit_gb == settings.vram_limit_gb == 30.0
    assert settings.dataset is None
    assert settings.smi_timeout_s == 10.0
    assert settings.report_recovery_timeout_s == 120.0


def test_settings_accept_both_sources_and_decimal_budgets() -> None:
    settings = k.KaggleSettings(dataset=Path("data.yaml"), resume_bundle=Path("bundle"), gpu_memory_fraction=1.0)
    gpu = k.GPU(0, "GPU-one", "Constructed T4", 16 * k.GB, 15 * k.GB)
    _, budget = k.select_budget(settings, [gpu], 32 * k.GB)
    assert budget.ram_bytes == 27_000_000_000
    assert budget.vram_bytes == 14_000_000_000


def test_gpu_budget_never_pools_two_devices() -> None:
    settings = k.KaggleSettings(dataset=Path("data.yaml"), gpu_index=1)
    devices = [k.GPU(0, "GPU-a", "A", 16 * k.GB, 16 * k.GB),
               k.GPU(1, "GPU-b", "B", 16 * k.GB, 10 * k.GB)]
    device, budget = k.select_budget(settings, devices, 10 * k.GB)
    assert device.uuid == "GPU-b"
    assert budget.vram_bytes == 9 * k.GB
    assert budget.ram_bytes == 7 * k.GB
    with pytest.raises(k.TelemetryError, match="unique"):
        k.select_budget(settings, devices[:1], 10 * k.GB)
    with pytest.raises(k.TelemetryError, match="Insufficient"):
        k.select_budget(settings, devices, 2 * k.GB)


@pytest.mark.parametrize("text", ["", "0, GPU-a, A, N/A, 10\n", "0, GPU-a, A, 10, 11\n",
                                      "0, MIG-a, A, 10, 10\n", "0, GPU-a, A, 10\n",
                                      "0, GPU-a, A, 10, 10\n1, GPU-a, A, 10, 10\n"])
def test_bad_gpu_inventory_aborts(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(k, "_smi", lambda *args: text)
    with pytest.raises(k.TelemetryError):
        k.query_gpus()


def test_gpu_process_scope_and_missing_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(k, "_smi", lambda *args: "10, GPU-a, 20\n11, GPU-a, 30\n10, GPU-b, 90\n")
    assert k.query_process_vram("GPU-a", {10}) == 20 * k.MIB
    assert k.query_process_vram("GPU-a", {10, 11}) == 50 * k.MIB
    monkeypatch.setattr(k, "_smi", lambda *args: "")
    assert k.query_process_vram("GPU-a", {10}) == 0
    for text in ("10, GPU-a, N/A\n", "malformed\n"):
        monkeypatch.setattr(k, "_smi", lambda *args, text=text: text)
        with pytest.raises(k.TelemetryError):
            k.query_process_vram("GPU-a", {10})


def test_gpu_process_duplicate_rows_are_deduplicated_by_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(k, "_smi", lambda *args: "10, GPU-a, 20\n10, GPU-a, 24\n11, GPU-a, 30\n")
    assert k.query_process_vram("GPU-a", {10, 11}) == 54 * k.MIB


def test_smi_missing_executable_is_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("nvidia-smi absent")

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(k.TelemetryError, match="unavailable"):
        k.query_gpus()


def test_smi_valid_query_and_explicit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def boundary_run(command: list[str], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        return types.SimpleNamespace(stdout="17, GPU-a, 23\n")

    monkeypatch.setattr(subprocess, "run", boundary_run)
    assert k.query_process_vram("GPU-a", {17}, timeout_s=1.25) == 23 * k.MIB
    assert calls == [(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                       "--format=csv,noheader,nounits"],
                      {"check": True, "capture_output": True, "text": True, "timeout": 1.25})]


def test_rebind_registry_preserves_immutable_inventory_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path)
    before = settings.registry.read_bytes()
    cfg = load_config(settings.dataset, settings.profile, environ={})

    def no_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Registry rebinding must never open a network connection")

    monkeypatch.setattr(socket, "socket", no_network)
    generated, inventory = k.bind_registry(settings, cfg)
    assert generated == k.bind_registry(settings, cfg)[0]
    assert not generated.is_relative_to(settings.output_root / settings.run_id)
    assert settings.registry.read_bytes() == before
    original = read_yaml(settings.registry)["models"][cfg.depth.model]
    rebound = read_yaml(generated)["models"][cfg.depth.model]
    assert {key: value for key, value in rebound.items() if key != "path"} == {
        key: value for key, value in original.items() if key != "path"
    }
    resolved = Registry(generated, settings.model_cache).resolve(cfg.depth.model)
    assert resolved["files"] == original["files"]
    assert inventory[cfg.depth.model]["executed"] is False
    assert inventory["rtdetr"]["valid"] is False  # optional, fallback stays available


@pytest.mark.parametrize("model", ["rtdetr", "semantic_boundary"])
def test_supplied_optional_models_remain_optional(tmp_path: Path, model: str) -> None:
    settings = local_settings(tmp_path)
    cfg = load_config(settings.dataset, settings.profile, environ={})
    if model == "semantic_boundary":
        cfg.semantics.model = model
    checkpoint = settings.model_paths[cfg.depth.model]
    settings.model_paths[model] = checkpoint
    # Missing optional specs cannot be invented, even with an explicit path.
    _, inventory = k.bind_registry(settings, cfg)
    assert inventory[model]["valid"] is False
    assert inventory[model]["executed"] is False
    document = read_yaml(settings.registry)
    document["models"][model] = {
        **document["models"][cfg.depth.model], "files": {checkpoint.name: "0" * 64},
    }
    write_json(settings.registry, document)
    _, inventory = k.bind_registry(settings, cfg)
    assert inventory[model]["valid"] is False
    assert "sha256 mismatch" in inventory[model]["reason"]


@pytest.mark.parametrize("failure", ["hash", "inventory", "symlink", "missing_name", "revision", "license"])
def test_bad_rebound_models_fail_without_relabeling(tmp_path: Path, failure: str) -> None:
    settings = local_settings(tmp_path)
    cfg = load_config(settings.dataset, settings.profile, environ={})
    checkpoint = settings.model_paths[cfg.depth.model]
    if failure == "hash":
        checkpoint.write_bytes(b"changed")
    elif failure == "inventory":
        root = tmp_path / "weights"
        root.mkdir()
        (root / checkpoint.name).write_bytes(checkpoint.read_bytes())
        (root / "unhashed.bin").write_bytes(b"unhashed")
        settings.model_paths[cfg.depth.model] = root
    elif failure == "symlink":
        link = tmp_path / "link.bin"
        link.symlink_to(checkpoint)
        settings.model_paths[cfg.depth.model] = link
    elif failure == "missing_name":
        settings.model_paths["invented"] = checkpoint
    else:
        document = read_yaml(settings.registry)
        document["models"][cfg.depth.model][failure] = "main" if failure == "revision" else "CC-BY-NC-4.0"
        write_json(settings.registry, document)
    with pytest.raises((DependencyError, InputError)):
        k.bind_registry(settings, cfg)


def test_duration_reads_actual_pyav_boundary_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = types.SimpleNamespace(duration=1200, time_base=0.5)

    class Container:
        streams = types.SimpleNamespace(video=[stream])
        duration: int | None = 700_000_000

        def __enter__(self) -> Container:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    opened: list[str] = []

    def open_metadata(path: str) -> Container:
        opened.append(path)
        return Container()

    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=open_metadata, time_base=1_000_000))
    assert k.declared_duration(Path("supplied-video.mp4"))["declared_duration_s"] == 600.0
    assert opened == ["supplied-video.mp4"]
    stream.duration = None
    assert k.declared_duration(Path("supplied-video.mp4"))["duration_source"] == "container.duration/av.time_base"
    Container.duration = None
    assert k.declared_duration(Path("supplied-video.mp4"))["declared_duration_s"] is None


@pytest.mark.parametrize("duration", [600.0, 10.0, None])
def test_full_flight_sampling_no_early_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: float | None) -> None:
    settings = local_settings(tmp_path, sample_frame_budget=1800)
    cfg = load_config(settings.dataset, settings.profile, environ={})
    cfg.ingest.max_frames = 1800
    monkeypatch.setattr(k, "declared_duration", lambda _: {"declared_duration_s": duration})
    result, plan = k.sampling_plan(cfg, settings)
    assert result.ingest.max_frames == 0
    assert result.ingest.target_fps == (1799 / 600 if duration == 600 else 4.0)
    if duration is None:
        assert plan["estimated_frames"] is None
        assert plan["estimate_status"] == "unbounded_unknown_duration"
    else:
        assert plan["estimated_frames"] <= 1800


def test_resume_preserves_fps_until_explicit_plan_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path)
    bundle = configured_bundle(settings, monkeypatch)
    resumed = k.KaggleSettings(resume_bundle=bundle.root, registry=settings.registry,
                               model_cache=settings.model_cache, output_root=settings.output_root)
    cfg, plan = k._configured(resumed, bundle)
    assert cfg.ingest.target_fps == 1799 / 600
    assert plan["resume_decision"] == "saved_fps_preserved"
    assert cfg.models.registry == settings.registry
    assert cfg.models.cache_dir == settings.model_cache
    assert cfg.run.copy_inputs is True
    assert cfg.semantics.model is None and cfg.barometer.enabled is False
    resumed.profile = settings.profile
    resumed._explicit_fields = {"profile", "resume_bundle"}
    cfg, plan = k._configured(resumed, bundle)
    assert cfg.ingest.target_fps == 1799 / 600
    assert plan["resume_decision"] == "unchanged_explicit_plan_preserved"
    resumed.sample_frame_budget = 1000
    cfg, plan = k._configured(resumed, bundle)
    assert cfg.ingest.target_fps == 999 / 600
    assert plan["resume_decision"] == "new_or_changed_explicit_plan"


def test_resume_budget_increase_does_not_compound_reduced_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path, sample_frame_budget=600)
    bundle = configured_bundle(settings, monkeypatch)
    resumed = k.KaggleSettings(resume_bundle=bundle.root, sample_frame_budget=2400)
    cfg, plan = k._configured(resumed, bundle)
    assert cfg.ingest.target_fps == 2399 / 600
    assert plan["profile_target_fps"] == 4.0


def test_changed_descriptor_sampling_and_legacy_resume_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path)
    bundle = configured_bundle(settings, monkeypatch)
    assert settings.dataset is not None
    descriptor = read_yaml(settings.dataset)
    descriptor["ingest"] = {"target_fps": 0.5}
    write_json(settings.dataset, descriptor)
    cfg, plan = k._configured(settings, bundle)
    assert cfg.ingest.target_fps == 0.5
    assert plan["resume_decision"] == "new_or_changed_explicit_plan"
    saved = bundle.config()
    saved.ingest.max_frames = 20
    bundle.snapshot(saved)
    bundle.save()
    resumed = k.KaggleSettings(resume_bundle=bundle.root)
    cfg, plan = k._configured(resumed, bundle)
    assert cfg.ingest.target_fps == saved.ingest.target_fps
    assert cfg.ingest.max_frames == 0
    assert plan["legacy_truncation_cleared"] is True


def test_worker_env_is_single_uuid_offline_with_threads(tmp_path: Path) -> None:
    settings = local_settings(tmp_path)
    gpu = k.GPU(1, "GPU-second", "Constructed", 16 * k.GB, 16 * k.GB)
    env = k.worker_environment(settings, gpu)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-second"
    assert all(env[key] == value for key, value in k.OFFLINE_ENV.items())
    assert env["OMP_NUM_THREADS"] == "8"
    assert k.worker_environment(settings, None)["CUDA_VISIBLE_DEVICES"] == ""


def roomy_budget() -> k.Budgets:
    return k.Budgets(29 * k.GB, 0, 1, 0)


def assert_stopped(pid: int) -> None:
    try:
        process = psutil.Process(pid)
        assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_real_child_memory_watchdog_and_process_group_cleanup(tmp_path: Path) -> None:
    settings = local_settings(tmp_path)
    baseline = psutil.Process().memory_info().rss
    child = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('CHILD',os.getpid(),flush=True); a=bytearray(100_000_000); time.sleep(20)"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(20)"
    log = tmp_path / "child.log"
    result = k.monitor_command([sys.executable, "-c", parent], env=k.worker_environment(settings, None),
                               log=log, settings=settings, budget=k.Budgets(baseline + 35_000_000, 0, 1, 0), gpu=None,
                               timeout_s=5)
    assert "RAM budget" in result["stop_reason"]
    assert result["rss_peak_bytes"] > baseline + 35_000_000
    assert result["vram_peak_bytes"] is None
    assert result["returncode"] < 0
    assert result["cleanup_survivor_pids"] == []
    pid = int(log.read_text().split("CHILD ")[1].split()[0])
    assert_stopped(pid)
    assert_stopped(result["pid"])


def test_real_process_exit_status_is_not_consumed_by_psutil(tmp_path: Path) -> None:
    settings = local_settings(tmp_path)
    result = k.monitor_command([sys.executable, "-c", "raise SystemExit(7)"],
                               env=k.worker_environment(settings, None), log=tmp_path / "exit.log",
                               settings=settings, budget=roomy_budget(), gpu=None)
    assert result["returncode"] == 7
    assert result["stop_reason"] is None


def test_lost_process_telemetry_still_terminates_real_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path)

    def denied(*args: Any, **kwargs: Any) -> Any:
        raise psutil.AccessDenied(pid=os.getpid())

    monkeypatch.setattr(k, "_tree_processes", denied)
    result = k.monitor_command([sys.executable, "-c", "import time; time.sleep(20)"],
                               env=k.worker_environment(settings, None), log=tmp_path / "denied.log",
                               settings=settings, budget=roomy_budget(), gpu=None)
    assert result["returncode"] < 0
    assert result["samples"] == 0 and result["rss_peak_bytes"] is None
    assert result["cleanup_survivor_pids"] is None
    assert "cleanup could not be verified" in result["cleanup_error"]
    assert_stopped(result["pid"])


@pytest.mark.parametrize("stop", ["telemetry", "interrupt", "timeout"])
def test_monitor_stops_on_lost_telemetry_interrupt_and_bounded_timeout(tmp_path: Path, stop: str) -> None:
    settings = local_settings(tmp_path)
    samples = 0

    def sampler(pid: int | None, gpu: k.GPU | None, known: dict[int, psutil.Process]) -> dict[str, Any]:
        nonlocal samples
        samples += 1
        measured = k.sample_resources(pid, gpu, known)
        if samples >= 2:
            if stop == "telemetry":
                raise k.TelemetryError("selected GPU telemetry disappeared")
            if stop == "interrupt":
                raise KeyboardInterrupt
        return measured

    result = k.monitor_command([sys.executable, "-c", "import time; time.sleep(20)"],
                               env=k.worker_environment(settings, None), log=tmp_path / "stop.log",
                               settings=settings, budget=roomy_budget(), gpu=None, sampler=sampler,
                               timeout_s=0.06 if stop == "timeout" else 5)
    assert result["stop_reason"] is not None
    assert result["returncode"] < 0
    assert result["rss_peak_bytes"] is not None
    assert result["vram_peak_bytes"] is None
    assert_stopped(result["pid"])
    if stop == "telemetry":
        assert result["telemetry_error"] == "selected GPU telemetry disappeared"


def test_native_orphan_cleanup_after_group_leader_exits(tmp_path: Path) -> None:
    settings = local_settings(tmp_path)
    child = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ORPHAN',os.getpid(),flush=True); time.sleep(20)"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(.5)"
    log = tmp_path / "orphan.log"
    result = k.monitor_command([sys.executable, "-c", parent], env=k.worker_environment(settings, None),
                               log=log, settings=settings, budget=roomy_budget(), gpu=None)
    assert result["returncode"] == 0
    assert result["cleanup_survivor_pids"] == []
    assert_stopped(int(log.read_text().split("ORPHAN ")[1].split()[0]))


def test_cleanup_preserves_unrelated_orchestrator_sibling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path)
    def no_host_scan(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Runner must not scan unrelated host/Jupyter processes")

    monkeypatch.setattr(psutil, "process_iter", no_host_scan)
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
    try:
        result = k.monitor_command([sys.executable, "-c", "pass"],
                                   env=k.worker_environment(settings, None), log=tmp_path / "sibling.log",
                                   settings=settings, budget=roomy_budget(), gpu=None)
        assert result["returncode"] == 0
        assert sibling.poll() is None
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


def test_sigterm_parent_records_partial_and_cleans_real_native_worker(tmp_path: Path) -> None:
    """Real SIGTERM/session lifecycle; constructed GPU/input boundary, no inference."""
    settings = local_settings(tmp_path, termination_grace_s=0.15)
    source = write_json(tmp_path / "settings.json", settings.model_dump(mode="json"))
    worker = tmp_path / "native_boundary.py"
    worker.write_text('''
import os, subprocess, sys, time
from drishti.bundle import Bundle
bundle = Bundle.load(sys.argv[1])
bundle.write_json('s0_ingest/actual_partial.json', {'process_boundary_only': True})
bundle.manifest['stages']['s0_ingest'] = {'name': 's0_ingest', 'status': 'running', 'outputs': {}}
bundle.save()
child = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('NATIVE',os.getpid(),flush=True); a=bytearray(2_000_000); time.sleep(30)"
subprocess.Popen([sys.executable, '-c', child])
print('WORKER', os.getpid(), flush=True)
time.sleep(30)
''', encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text('''
import json, sys
from drishti.runtime import kaggle as k
k.declared_duration = lambda _: {'declared_duration_s': None}
gpu = k.GPU(0, 'GPU-boundary', 'Constructed GPU boundary', 16*k.GB, 16*k.GB)
k.query_gpus = lambda *args: [gpu]
k.query_process_vram = lambda *args: 0
original = k._worker_command
def command(bundle, stage):
    if stage == 'refresh_report':
        return original(bundle, stage)
    return [sys.executable, sys.argv[2], str(bundle.root)], bundle.path('cloud/workers/boundary.json')
k._worker_command = command
summary = k.run(k.load_settings(sys.argv[1]))
print(json.dumps(summary), flush=True)
raise SystemExit(summary['exit_code'])
''', encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(driver), str(source), str(worker)],
                               env=k.worker_environment(settings, None),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    log = settings.output_root / settings.run_id / "logs/kaggle_s0_ingest.log"
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if log.is_file() and "NATIVE " in log.read_text():
                break
            if process.poll() is not None:
                break
            time.sleep(0.02)
        assert log.is_file() and "NATIVE " in log.read_text()
        # Give the genuine process-tree sampler time to observe the native child.
        time.sleep(0.1)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 1, stderr
        summary = json.loads(stdout)
        assert summary["status"] == "aborted"
        assert "SIGTERM" in summary["stop_reason"]
        bundle = Bundle.load(summary["bundle"])
        assert bundle.verify()["ok"] is True
        assert bundle.manifest["stages"]["s0_ingest"]["status"] == "failed"
        assert all(bundle.manifest["stages"][spec.name]["status"] == "blocked" for spec in STAGES[1:])
        cloud = bundle.read_json("cloud_run.json")
        assert cloud["workers"][0]["cleanup_survivor_pids"] == []
        assert cloud["workers"][0]["returncode"] < 0
        assert cloud["report_refresh"]["returncode"] == 0
        assert cloud["archive_status"] == "created"
        assert bundle.read_json("report/report.json")["status"] == "partial"
        assert "[1/11] s0_ingest started; log=" in stderr
        assert "Final status=aborted; sampled RAM peak=" in stderr
        text = log.read_text()
        for marker in ("WORKER ", "NATIVE "):
            assert_stopped(int(text.split(marker)[1].split()[0]))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        # A failed assertion should not leave any of the test's native children.
        if log.is_file():
            text = log.read_text()
            for marker in ("WORKER ", "NATIVE "):
                if marker in text:
                    pid = int(text.split(marker)[1].split()[0])
                    try:
                        psutil.Process(pid).kill()
                    except psutil.NoSuchProcess:
                        pass


def test_sigterm_monitor_restores_callers_signal_handler(tmp_path: Path) -> None:
    settings = local_settings(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)

    def sampler(pid: int | None, gpu: k.GPU | None, known: dict[int, psutil.Process]) -> dict[str, Any]:
        sample = k.sample_resources(pid, gpu, known)
        os.kill(os.getpid(), signal.SIGTERM)
        return sample

    result = k.monitor_command([sys.executable, "-c", "import time; time.sleep(20)"],
                               env=k.worker_environment(settings, None), log=tmp_path / "signal.log",
                               settings=settings, budget=roomy_budget(), gpu=None, sampler=sampler)
    assert "SIGTERM" in result["stop_reason"]
    assert signal.getsignal(signal.SIGTERM) == previous
    assert_stopped(result["pid"])


def test_constructed_gpu_violation_and_device_disappearance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path)
    sample = {"rss_bytes": 10, "vram_bytes": 101, "system_available_bytes": 100, "gpu_free_bytes": 10}
    assert "VRAM budget" in str(k._violation(sample, k.Budgets(100, 100, 1, 1)))
    sample["vram_bytes"] = 0
    sample["gpu_free_bytes"] = 0
    assert "GPU free" in str(k._violation(sample, k.Budgets(100, 100, 1, 1)))
    gpu = k.GPU(0, "GPU-a", "Boundary", 100, 90)
    monkeypatch.setattr(k, "query_gpus", lambda *args: [k.GPU(1, "GPU-b", "Boundary", 100, 90)])
    with pytest.raises(k.TelemetryError, match="disappeared"):
        k.sample_resources(None, gpu, {})
    assert settings.gpu_index == 0


def test_configured_smi_timeout_reaches_inventory_and_process_sampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path, smi_timeout_s=0.75)
    gpu = k.GPU(0, "GPU-a", "Constructed", 16 * k.GB, 16 * k.GB)
    seen: list[float] = []

    def inventory(timeout_s: float) -> list[k.GPU]:
        seen.append(timeout_s)
        return [gpu]

    def memory(uuid: str, pids: set[int], timeout_s: float) -> int:
        seen.append(timeout_s)
        return 0

    monkeypatch.setattr(k, "query_gpus", inventory)
    monkeypatch.setattr(k, "query_process_vram", memory)
    result = k.monitor_command([sys.executable, "-c", "pass"],
                               env=k.worker_environment(settings, None), log=tmp_path / "timeout.log",
                               settings=settings, budget=roomy_budget(), gpu=gpu)
    assert result["returncode"] == 0
    assert seen and set(seen) == {0.75}
    assert len(seen) == 2 * result["samples"]


def fake_torch(monkeypatch: pytest.MonkeyPatch, calls: list[str], *, available: bool = True) -> Any:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: available, device_count=lambda: 1,
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=16 * k.GB),
        set_per_process_memory_fraction=lambda fraction, device: calls.append(f"cap:{fraction}:{device}"),
        reset_peak_memory_stats=lambda _: calls.append("reset"),
        max_memory_allocated=lambda _: 123, max_memory_reserved=lambda _: 456,
        manual_seed_all=lambda _: None,
    )
    module.manual_seed = lambda _: None  # type: ignore[attr-defined]
    module.set_num_threads = lambda _: None  # type: ignore[attr-defined]
    module.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=True, deterministic=False))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_worker_actual_core_resume_atomic_integrity_and_torch_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = configured_bundle(local_settings(tmp_path), monkeypatch)
    calls: list[str] = []
    fake_torch(monkeypatch, calls)

    class BoundaryStage:
        name = "s0_ingest"
        requires: tuple[str, ...] = ()
        config_sections = STAGES[0].config_sections

        def run(self, bundle: Bundle, cfg: Config, runtime: Runtime) -> StageResult:
            calls.append("stage")
            bundle.write_json("s0_ingest/unit_boundary.json", {"unit_boundary": True})
            return StageResult(outputs={"unit": "s0_ingest/unit_boundary.json"})

    class BoundaryRunner(StageRunner):
        def __init__(self, bundle: Bundle, cfg: Config) -> None:
            super().__init__(bundle, cfg, stages=[BoundaryStage()])

    def detect(cfg: Config | None = None) -> Runtime:
        calls.append("runtime")
        return Runtime("cuda", {"versions": {"unit_boundary": "1"}})

    monkeypatch.setattr(Runtime, "detect", detect)
    monkeypatch.setattr(k, "StageRunner", BoundaryRunner)
    request = write_json(bundle.path("cloud/worker_request.json"), {
        "vram_bytes": 8 * k.GB, "vram_reserve_bytes": k.GB,
        "gpu_memory_fraction": 0.9, "cpu_threads": 2,
    })
    result = bundle.path("cloud/workers/test.json")
    assert k._worker(bundle.root, "s0_ingest", result, request) == 0
    assert calls.index("cap:0.5:0") < calls.index("runtime") < calls.index("stage")
    measurements = read_yaml(result)
    assert measurements["torch_max_allocated_bytes"] == 123
    assert measurements["torch_max_reserved_bytes"] == 456
    assert measurements["stage_executed"] is True
    assert measurements["executed_stages"] == ["s0_ingest"]
    assert measurements["runtime"]["executed"] is False
    assert Bundle.load(bundle.root).verify()["ok"] is True
    identity = Bundle.load(bundle.root).manifest["content_id"]
    modified = Bundle.load(bundle.root)
    k._persist_cloud(modified, {"rss_peak_bytes": 100, "arbitrary_volatile_time": time.time()})
    assert k._worker(bundle.root, "s0_ingest", result, request) == 0
    assert calls.count("stage") == 1  # actual core content freshness, not a mocked success
    assert read_yaml(result)["stage_executed"] is False
    assert read_yaml(result)["executed_stages"] == []
    assert Bundle.load(bundle.root).manifest["content_id"] == identity
    bundle.path("s0_ingest/unit_boundary.json").write_text("{}")
    assert k._worker(bundle.root, "s0_ingest", result, request) == 0
    assert calls.count("stage") == 2
    manifest = bundle.path("manifest.json")
    tampered = json.loads(manifest.read_text())
    tampered["seeds"]["global"] = 9
    write_json(manifest, tampered)
    assert k._worker(bundle.root, "s0_ingest", result, request) == 1
    assert "integrity" in read_yaml(result)["error"]["message"]


def test_worker_requires_real_operational_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = configured_bundle(local_settings(tmp_path), monkeypatch)
    fake_torch(monkeypatch, [], available=False)
    request = write_json(bundle.path("request.json"), {})
    result = bundle.path("result.json")
    assert k._worker(bundle.root, "s0_ingest", result, request) == 1
    assert read_yaml(result)["torch_max_allocated_bytes"] is None
    assert read_yaml(result)["error"]["type"] == "DependencyError"
    assert bundle.manifest["stages"] == {}
    assert read_yaml(result)["stage_executed"] is None


def test_interruption_partial_paths_hashes_and_stale_downstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = configured_bundle(local_settings(tmp_path), monkeypatch)
    bundle.write_json("s0_ingest/real_partial_boundary.json", {"partial_written": True})
    bundle.manifest["stages"] = {"s0_ingest": {"name": "s0_ingest", "status": "running"},
                                  "s1_frameqa": {"status": "done", "outputs": {}}}
    bundle.save()
    k._mark_interrupted(bundle, "s0_ingest", "measured RSS limit")
    record = bundle.manifest["stages"]["s0_ingest"]
    assert record["status"] == "failed" and record["wall_seconds"] is None
    assert record["outputs"] == {"partial": "s0_ingest"}
    assert record["output_hashes"]["partial"]["sha256"] == hash_path(bundle.path("s0_ingest"))["sha256"]
    assert bundle.manifest["stages"]["s1_frameqa"]["status"] == "stale"
    assert bundle.manifest["stages"]["s2_poses"]["status"] == "blocked"
    assert bundle.manifest["stages"]["s2_poses"]["blocked_by"] == ["s0_ingest", "s1_frameqa"]
    assert bundle.manifest["stages"]["report"]["status"] == "blocked"
    k._report_worker(bundle)
    assert bundle.verify()["ok"] is True
    report = bundle.read_json("report/report.json")
    assert report["status"] == "partial"
    assert report["accuracy"]["absolute_spatial_rmse_m"] is None
    assert report["timing"]["end_to_end_wall_seconds"] is None
    assert "measured RSS limit" in bundle.path("report/report.html").read_text()


def test_interrupted_cached_stage_cannot_keep_missing_outputs_or_current_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = configured_bundle(local_settings(tmp_path), monkeypatch)
    bundle.write_json("s0_ingest/old.json", {"boundary_only": True})
    outputs = {"old": "s0_ingest/old.json"}
    hashes = bundle.output_hashes(outputs)
    bundle.manifest["stages"]["s0_ingest"] = {
        "name": "s0_ingest", "status": "done", "outputs": outputs,
        "output_hashes": hashes, "outputs_hash": content_hash(hashes),
        "metrics": {"old_sample_count": 12}, "models": [], "confidence": 0.9,
    }
    bundle.path("s0_ingest/old.json").unlink()
    k._mark_interrupted(bundle, "s0_ingest", "worker failed before validating the cache")
    record = bundle.manifest["stages"]["s0_ingest"]
    assert record["status"] == "failed" and record["previous_status"] == "done"
    assert record["interrupted_before_stage"] is True
    assert record["outputs"] == {} and "output_hashes" not in record
    assert "metrics" not in record and record["retained_metrics"] == {"old_sample_count": 12}
    assert bundle.verify()["ok"] is True


def test_real_report_recovery_and_partial_archive_after_gpu_preflight_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = local_settings(tmp_path, termination_grace_s=1.0)
    monkeypatch.setattr(k, "declared_duration", lambda _: {"declared_duration_s": None})

    def no_gpu(timeout_s: float) -> list[k.GPU]:
        raise k.TelemetryError("no measured GPU inventory")

    monkeypatch.setattr(k, "query_gpus", no_gpu)
    summary = k.run(settings)
    assert summary["status"] == "aborted" and summary["exit_code"] == 1
    assert summary["archive"] is not None
    bundle = Bundle.load(summary["bundle"])
    assert bundle.verify()["ok"] is True
    assert bundle.manifest["wall_seconds"] is None
    cloud = bundle.read_json("cloud_run.json")
    assert cloud == bundle.manifest["cloud_execution"]
    assert cloud["gpu"] is None and cloud["vram_peak_bytes"] is None
    assert cloud["report_refresh"]["returncode"] == 0
    assert cloud["runtime"]["executed"] is False
    assert cloud["report_recovery_wall_limit_s"] == 120.0
    assert cloud["archive_status"] == "created"
    snapshot = bundle.read_json("cloud/report_snapshot.json")
    assert snapshot["report_refresh"] is None
    assert snapshot["archive_path"] is None and snapshot["archive_status"] == "pending"
    assert snapshot["report_cloud_snapshot"] == cloud["report_cloud_snapshot"]
    assert snapshot["orchestration_wall_seconds"] < cloud["orchestration_wall_seconds"]
    assert cloud["orchestration_wall_seconds"] > 0
    assert bundle.read_json("report/report.json")["status"] == "partial"
    with zipfile.ZipFile(summary["archive"]) as archive:
        saved = json.loads(archive.read(f"{bundle.root.name}/manifest.json"))
        assert saved["manifest_sha256"] == manifest_digest(saved)
        assert saved["status"] == "partial"
        assert saved["cloud_execution"]["archive_status"] == "pending"
        assert f"{bundle.root.name}/report/report.html" in archive.namelist()


def test_report_recovery_failure_rehashes_real_partial_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = configured_bundle(local_settings(tmp_path), monkeypatch)
    bundle.manifest["stages"]["report"] = {"status": "done", "outputs": {}}
    k._report_worker(bundle)
    bundle.path("report/report.html").write_text("interrupted actual write")
    k._report_failure(bundle, "report worker was stopped")
    bundle.save()
    assert bundle.verify()["ok"] is True
    assert bundle.manifest["stages"]["report"]["status"] == "failed"
    assert bundle.manifest["partial_report"]["outputs"]["html"] == "report/report.html"


def test_report_receives_immutable_explicit_cloud_snapshot_without_hash_recursion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from drishti.report import builder

    bundle = configured_bundle(local_settings(tmp_path), monkeypatch)
    bundle.manifest["status"] = "partial"
    cloud: dict[str, Any] = {
        "status": "aborted", "workers": [], "report_refresh": None,
        "rss_peak_bytes": 123, "vram_peak_bytes": None, "archive_status": "pending",
        "runtime": {"executed": False}, "orchestration_wall_seconds": 1.0,
    }
    k._freeze_report_cloud(bundle, cloud)
    frozen_hash = hash_file(bundle.path("cloud/report_snapshot.json"))
    received: list[dict[str, Any]] = []
    original = builder.build_report

    def observe(bundle: Bundle, cfg: Config) -> dict[str, str]:
        received.append(json.loads(json.dumps(bundle.manifest["cloud_execution"])))
        return original(bundle, cfg)

    monkeypatch.setattr(builder, "build_report", observe)
    k._report_worker(Bundle.load(bundle.root))
    assert received == [bundle.read_json("cloud/report_snapshot.json")]
    assert received[0]["report_cloud_snapshot"]["authoritative_final_record"] == "cloud_run.json"
    assert len(received[0]["report_cloud_snapshot"]["excludes"]) == 2
    report_hashes = bundle.output_hashes({"json": "report/report.json", "html": "report/report.html"})
    # The report cannot include measurements of its own final recovery or ZIP.
    cloud.update(report_refresh={"rss_peak_bytes": 456}, rss_peak_bytes=456,
                 orchestration_wall_seconds=2.0, archive_status="failed")
    bundle = Bundle.load(bundle.root)
    k._persist_cloud(bundle, cloud)
    assert bundle.verify()["ok"] is True
    assert hash_file(bundle.path("cloud/report_snapshot.json")) == frozen_hash
    assert bundle.output_hashes({"json": "report/report.json", "html": "report/report.html"}) == report_hashes
    assert bundle.read_json("cloud_run.json")["rss_peak_bytes"] == 456


def test_report_timeout_and_archive_failure_have_explicit_final_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    settings = local_settings(tmp_path, report_recovery_timeout_s=17.5)
    monkeypatch.setattr(k, "declared_duration", lambda _: {"declared_duration_s": None})

    def no_gpu(*args: Any) -> list[k.GPU]:
        raise k.TelemetryError("GPU boundary unavailable")

    monkeypatch.setattr(k, "query_gpus", no_gpu)
    original = k.monitor_command
    timeouts: list[float | None] = []

    def observe(command: list[str], **kwargs: Any) -> dict[str, Any]:
        timeouts.append(kwargs.get("timeout_s"))
        return original(command, **kwargs)

    def unavailable_archive(*args: Any) -> Path:
        raise OSError("real archive IO failure boundary")

    monkeypatch.setattr(k, "monitor_command", observe)
    monkeypatch.setattr(k, "archive_bundle", unavailable_archive)
    summary = k.run(settings)
    assert timeouts == [17.5]
    assert summary["archive"] is None and summary["exit_code"] == 1
    bundle = Bundle.load(summary["bundle"])
    cloud = bundle.read_json("cloud_run.json")
    assert cloud["archive_status"] == "failed"
    assert "IO failure" in cloud["archive_error"]
    assert bundle.read_json("cloud/report_snapshot.json")["archive_status"] == "pending"
    assert bundle.verify()["ok"] is True
    output = capsys.readouterr()
    assert output.out == ""
    assert "VRAM peak=unavailable" in output.err


def test_copy_resume_bundle_rejects_tampered_manifest_and_existing_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = configured_bundle(local_settings(tmp_path), monkeypatch)
    copied = k._copy_bundle(original.root, tmp_path / "moved")
    assert copied.verify()["ok"] is True
    assert k._copy_bundle(copied.root, copied.root).root == copied.root
    with pytest.raises(InputError, match="already exists"):
        k._copy_bundle(original.root, copied.root)
    with pytest.raises(InputError, match="itself"):
        k._copy_bundle(original.root, original.root / "nested")
    document = json.loads(original.path("manifest.json").read_text())
    document["content_id"] = "tampered"
    write_json(original.path("manifest.json"), document)
    with pytest.raises(InputError, match="integrity"):
        k._copy_bundle(original.root, tmp_path / "tampered-copy")


def test_parent_orders_real_isolated_core_boundary_workers_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real subprocesses/core freshness; GPU and stage algorithms are unit boundaries."""
    settings = local_settings(tmp_path, archive=False, termination_grace_s=1.0)
    monkeypatch.setattr(k, "declared_duration", lambda _: {"declared_duration_s": 600.0})
    gpu = k.GPU(0, "GPU-constructed", "Constructed unit GPU", 16 * k.GB, 16 * k.GB)
    monkeypatch.setattr(k, "query_gpus", lambda *args: [gpu])
    monkeypatch.setattr(k, "query_process_vram", lambda *args: 0)
    # Replace algorithms at the stage boundary, not the real DAG/resume machinery.
    boundary_script = tmp_path / "boundary_worker.py"
    boundary_script.write_text('''
import importlib.metadata, os, sys
from pathlib import Path
from drishti.bundle import Bundle
from drishti.bundle.core import atomic_json
from drishti.runtime import Runtime
from drishti.runtime.kaggle import OFFLINE_ENV
from drishti.runtime.probe import PACKAGES
from drishti.stages.base import StageResult
from drishti.stages.runner import StageRunner, STAGES
assert all(os.environ[key] == value for key, value in OFFLINE_ENV.items())
assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-constructed'
versions = {}
for package in PACKAGES:
    try: versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: versions[package] = None
class Boundary:
    def __init__(self, spec):
        self.name, self.requires, self.config_sections = spec.name, spec.requires, spec.config_sections
    def run(self, bundle, cfg, runtime):
        output = self.name + '/unit_boundary.json'
        bundle.write_json(output, {'unit_boundary_only': True, 'pid': os.getpid()})
        return StageResult(outputs={'unit': output})
bundle = Bundle.load(sys.argv[1])
stage = sys.argv[2]
result = StageRunner(bundle, bundle.config(), Runtime('cuda', {'versions': versions, 'constructed_cuda_boundary': True}), stages=[Boundary(spec) for spec in STAGES]).run(selected=[stage], only=True)
atomic_json(Path(sys.argv[3]), {'unit_boundary_only': True, 'executed_stages': result.manifest['executions'][-1]['executed_stages'], 'torch_max_allocated_bytes': None, 'torch_max_reserved_bytes': None})
raise SystemExit(0 if result.manifest['stages'][stage]['status'] in {'done', 'degraded'} else 1)
''', encoding="utf-8")
    original_command = k._worker_command

    def boundary_command(bundle: Bundle, stage: str) -> tuple[list[str], Path]:
        if stage == "refresh_report":
            return original_command(bundle, stage)
        result = bundle.path(f"cloud/workers/{stage}.json")
        result.unlink(missing_ok=True)
        return [sys.executable, str(boundary_script), str(bundle.root), stage, str(result)], result

    monkeypatch.setattr(k, "_worker_command", boundary_command)
    summary = k.run(settings)
    assert summary["status"] == "done", summary
    bundle = Bundle.load(summary["bundle"])
    assert bundle.verify()["ok"] is True
    workers = bundle.read_json("cloud_run.json")["workers"]
    assert [entry["stage"] for entry in workers] == [spec.name for spec in STAGES]
    assert [entry["requires"] for entry in workers] == [list(spec.requires) for spec in STAGES]
    assert len({entry["pid"] for entry in workers}) == len(STAGES)
    assert all(entry["worker"]["unit_boundary_only"] is True for entry in workers)
    assert all(entry["worker"]["executed_stages"] == [entry["stage"]] for entry in workers)
    original_identity = bundle.manifest["content_id"]
    resumed = k.KaggleSettings(resume_bundle=bundle.root, output_root=settings.output_root,
                               registry=settings.registry, model_cache=settings.model_cache,
                               model_paths=settings.model_paths, archive=False,
                               termination_grace_s=1.0, monitor_interval_s=0.02)
    again = k.run(resumed)
    assert again["exit_code"] == 0, again
    bundle = Bundle.load(again["bundle"])
    assert bundle.verify()["ok"] is True
    assert bundle.manifest["content_id"] == original_identity
    assert all(not entry["worker"]["executed_stages"] for entry in bundle.read_json("cloud_run.json")["workers"])
    assert bundle.read_json("cloud_run.json")["sampling"]["resume_decision"] == "saved_fps_preserved"
    retained = bundle.path("s0_ingest/unit_boundary.json").read_bytes()

    def crash_before_cached_stage(bundle: Bundle, stage: str) -> tuple[list[str], Path]:
        if stage == "refresh_report":
            return original_command(bundle, stage)
        result = bundle.path(f"cloud/workers/{stage}.json")
        result.unlink(missing_ok=True)
        code = (
            "from pathlib import Path; from drishti.bundle.core import atomic_json; "
            f"atomic_json(Path({str(result)!r}), {{'error': {{'type': 'DependencyError', "
            "'message': 'constructed CUDA unavailable before cached stage'}}, 'stage_executed': None}); "
            "raise SystemExit(9)"
        )
        return [sys.executable, "-c", code], result

    monkeypatch.setattr(k, "_worker_command", crash_before_cached_stage)
    stopped = k.run(resumed)
    assert stopped["status"] == "aborted" and stopped["exit_code"] == 1
    bundle = Bundle.load(stopped["bundle"])
    assert bundle.verify()["ok"] is True
    assert bundle.manifest["stages"]["s0_ingest"]["status"] == "failed"
    assert bundle.manifest["stages"]["s0_ingest"]["interrupted_before_stage"] is True
    assert all(bundle.manifest["stages"][spec.name]["status"] == "stale" for spec in STAGES[1:])
    assert bundle.path("s0_ingest/unit_boundary.json").read_bytes() == retained
    cloud = bundle.read_json("cloud_run.json")
    assert len(cloud["workers"]) == 1
    assert cloud["workers"][0]["stage_status"] == "failed"
    assert cloud["workers"][0]["worker"]["stage_executed"] is None


def test_load_settings_rejects_model_symlink_before_resolution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"local fixture")
    (tmp_path / "link").symlink_to(checkpoint)
    settings = write_json(tmp_path / "settings.yaml", {
        "dataset": "data.yaml", "model_paths": {"depth": "link"},
    })
    with pytest.raises(InputError, match="symlink"):
        k.load_settings(settings)


def test_changed_resume_inputs_cannot_leave_retained_geometry_current(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = configured_bundle(local_settings(tmp_path), monkeypatch)
    cfg = bundle.config()
    # Build a legitimate core freshness record for a constructed stage boundary.
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    runner = StageRunner(bundle, cfg, Runtime("cuda", {"versions": versions}))
    bundle.write_json("s0_ingest/unit.json", {"unit_boundary": True})
    outputs = {"unit": "s0_ingest/unit.json"}
    hashes = bundle.output_hashes(outputs)
    bundle.manifest["stages"]["s0_ingest"] = {
        "name": "s0_ingest", "status": "done", "inputs_hash": runner.fingerprint(STAGES[0]),
        "outputs": outputs, "output_hashes": hashes, "outputs_hash": content_hash(hashes),
    }
    bundle.save()
    k._invalidate_retained(bundle, cfg)
    assert bundle.manifest["stages"]["s0_ingest"]["status"] == "done"
    cfg.ingest.target_fps = 0.25
    k._invalidate_retained(bundle, cfg)
    assert bundle.manifest["stages"]["s0_ingest"]["status"] == "stale"


def test_zip_streaming_confinement_and_preserves_existing_archive(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "data.bin").write_bytes(b"real archive bytes")
    archive = tmp_path / "bundle.zip"
    k.archive_bundle(root, archive)
    before = archive.read_bytes()
    with zipfile.ZipFile(archive) as saved:
        assert saved.read("bundle/data.bin") == b"real archive bytes"
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in saved.namelist())
    with pytest.raises(InputError, match="outside"):
        k.archive_bundle(root, root / "inside.zip")
    outside = tmp_path / "outside"
    outside.write_text("not part of bundle")
    (root / "escape").symlink_to(outside)
    with pytest.raises(InputError, match="symlink"):
        k.archive_bundle(root, archive)
    assert archive.read_bytes() == before
    assert list(tmp_path.glob(".bundle.zip.*.tmp")) == []


def test_zip_rejects_special_files(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    os.mkfifo(root / "fifo")
    with pytest.raises(InputError, match="Non-regular"):
        k.archive_bundle(root, tmp_path / "output.zip")


def test_zip_rejects_windows_traversal_filenames(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "..\\escape.bin").write_bytes(b"not a safe cross-platform zip name")
    with pytest.raises(InputError, match="cross-platform"):
        k.archive_bundle(root, tmp_path / "output.zip")


def test_help_imports_without_torch_and_cli_nonzero(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/kaggle_run.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "--settings" in result.stdout
    source = write_json(tmp_path / "invalid.yaml", {"dataset": "data.yaml", "vram_limit_gb": 40.0})
    failed = subprocess.run([sys.executable, str(script), "--settings", str(source)], capture_output=True, text=True, check=False)
    assert failed.returncode == 1 and "vram_limit_gb" in failed.stderr


def test_main_keeps_final_stdout_json_separate_from_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    settings = local_settings(tmp_path, archive=False)
    source = write_json(tmp_path / "settings.json", settings.model_dump(mode="json"))
    monkeypatch.setattr(k, "declared_duration", lambda _: {"declared_duration_s": None})

    def no_gpu(*args: Any) -> list[k.GPU]:
        raise k.TelemetryError("Constructed no-GPU CLI boundary")

    monkeypatch.setattr(k, "query_gpus", no_gpu)
    assert k.main(["--settings", str(source)]) == 1
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "aborted" and summary["exit_code"] == 1
    assert summary["report_json"] is not None
    assert "[DRISHTI Kaggle] Final status=aborted" in captured.err
    assert "[DRISHTI" not in captured.out
