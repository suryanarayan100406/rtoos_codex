"""Source/bootstrap/report contracts; no Kaggle GPU or mission acceptance claim."""

import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from drishti.bundle import Bundle
from drishti.config import Config
from drishti.runtime.kaggle import KaggleSettings, load_settings
from scripts.package_kaggle import package_source

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle_quality.ipynb"


def cells():
    document = json.loads(NOTEBOOK.read_text())
    return document, {cell["id"]: "".join(cell["source"]) for cell in document["cells"]}


def checkout(tmp_path):
    """Small explicit source-shape boundary, never model weights or footage."""
    root = tmp_path / "checkout"
    root.mkdir()
    for name in ("pyproject.toml", "README.md", "AGENTS.md"):
        (root / name).write_text("source archive unit boundary\n")
    for relative in ("src/drishti/cli.py", "src/drishti/models/registry.py",
                     "scripts/kaggle_run.py", "configs/profiles/kaggle_quality.yaml"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source archive unit boundary\n")
    return root


def bootstrap_scope(tmp_path, source):
    project = tmp_path / "working/project"
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text("isolated subprocess boundary")
    return {"Path": Path, "SOURCE_ZIP": source, "PROJECT_ROOT": project, "VENV_ROOT": venv,
            "INSTALL_BLENDER": False, "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu124"}


def isolated_setup(monkeypatch):
    import shutil

    calls = []
    monkeypatch.setattr(shutil, "which", lambda name: "/isolated-unit/" + name)

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="3.12\n")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_notebook_is_unexecuted_valid_python_and_typed_configuration():
    document, sources = cells()
    assert document["nbformat"] == 4
    assert document["metadata"]["kernelspec"]["name"] == "python3"
    assert len(sources) == len(document["cells"])
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
    scope = {}
    exec(sources["configuration"], scope)
    settings = KaggleSettings.model_validate({
        **scope["RESOURCE_SETTINGS"], "dataset": scope["DATASET_YAML"],
        "model_paths": scope["MODEL_PATHS"], "sample_frame_budget": scope["SAMPLE_FRAME_BUDGET"],
    })
    assert settings.ram_limit_gb <= 30 and settings.vram_limit_gb <= 30
    assert 0 < settings.gpu_memory_fraction < 1
    setup = sources["setup"]
    assert "TARGET_PYTHON = '3.12'" in setup
    assert "uv==0.6.14" in setup
    assert "Use a Kaggle Python 3.11/3.12" not in setup


def test_packaged_real_checkout_contains_runtime_models_profile_and_notebook(tmp_path):
    archive = package_source(ROOT, tmp_path / "source.zip")
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        assert "drishti/src/drishti/models/registry.py" in names
        assert "drishti/src/drishti/models/inference.py" in names
        assert "drishti/src/drishti/runtime/kaggle.py" in names
        assert "drishti/configs/profiles/kaggle_quality.yaml" in names
        assert "drishti/notebooks/kaggle_quality.ipynb" in names
        assert json.loads(zipped.read("drishti/notebooks/kaggle_quality.ipynb"))["nbformat"] == 4
        assert not any("node_modules" in name or "__pycache__" in name for name in names)
        assert all(not name.startswith(("drishti/models/", "drishti/data/", "drishti/runs/"))
                   for name in names)


def test_source_zip_excludes_weights_and_inputs_and_is_atomic_on_bad_link(tmp_path):
    root = checkout(tmp_path)
    for relative in ("models/model.safetensors", "data/flight.mp4", "runs/run/manifest.json",
                     "src/drishti/weights.bin", "src/drishti/.env"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"explicit source-filter boundary")
    destination = tmp_path / "source.zip"
    package_source(root, destination)
    before = destination.read_bytes()
    with zipfile.ZipFile(destination) as zipped:
        assert "drishti/src/drishti/models/registry.py" in zipped.namelist()
        assert not any(name.endswith((".safetensors", ".mp4", ".bin", ".env"))
                       for name in zipped.namelist())
    (root / "src/drishti/escape.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(ValueError, match="regular files"):
        package_source(root, destination)
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob(".drishti-source-*.tmp"))


def test_actual_notebook_bootstrap_extracts_source_and_reuses_matching_hash(tmp_path, monkeypatch):
    source = package_source(checkout(tmp_path), tmp_path / "source.zip")
    scope = bootstrap_scope(tmp_path, source)
    commands = isolated_setup(monkeypatch)
    _, sources = cells()
    exec(sources["setup"], scope)
    project = scope["PROJECT_ROOT"]
    assert (project / "src/drishti/models/registry.py").is_file()
    assert (project / ".kaggle-source-sha256").read_text().strip() == hashlib.sha256(source.read_bytes()).hexdigest()
    exec(sources["setup"], scope)  # A repeat setup must neither overwrite nor reject the same source.
    assert any("torch==2.6.0" in command for command in commands)
    assert any("[ingest,poses,neural,geo,server]" in str(command) for command in commands)
    assert not any("snapshot_download" in str(command) for command in commands)
    source.write_bytes(b"different unit source")
    with pytest.raises(RuntimeError, match="different/untracked"):
        exec(sources["setup"], scope)


def test_notebook_missing_input_reports_mounted_paths(tmp_path, monkeypatch):
    scope = bootstrap_scope(tmp_path, tmp_path / "missing-source.zip")
    _, sources = cells()
    with pytest.raises(FileNotFoundError, match="Attach the source Dataset"):
        exec(sources["setup"], scope)


def test_actual_notebook_bootstrap_extracts_mission_zip(tmp_path, monkeypatch):
    source = package_source(checkout(tmp_path), tmp_path / "source.zip")
    mission = tmp_path / "mission.zip"
    with zipfile.ZipFile(mission, "w") as archive:
        archive.writestr("mission.yaml", "name: unit\n")
        archive.writestr("flight.MP4", "unit video")
        archive.writestr("flight.SRT", "unit telemetry")
    models = tmp_path / "models.zip"
    with zipfile.ZipFile(models, "w") as archive:
        archive.writestr("models.yaml", "models: {}\n")
        archive.writestr("depth_anything_v2_small/config.json", "{}")
    scope = bootstrap_scope(tmp_path, source)
    scope.update(MISSION_ZIP=mission, MISSION_ROOT=tmp_path / "working/mission",
                 MODEL_ZIP=models, MODEL_ROOT=tmp_path / "working/models")
    isolated_setup(monkeypatch)
    _, sources = cells()
    exec(sources["setup"], scope)
    assert (scope["MISSION_ROOT"] / "mission.yaml").read_text() == "name: unit\n"
    assert scope["DATASET_YAML"] == scope["MISSION_ROOT"] / "mission.yaml"
    assert (scope["MODEL_ROOT"] / "models.yaml").is_file()
    assert (scope["MODEL_ROOT"] / "depth_anything_v2_small/config.json").is_file()


def test_actual_notebook_bootstrap_copies_extracted_source_dataset(tmp_path, monkeypatch):
    source_root = checkout(tmp_path)
    scope = bootstrap_scope(tmp_path, tmp_path / "unused-source.zip")
    scope.update(SOURCE_ZIP=None, SOURCE_ROOT=source_root)
    isolated_setup(monkeypatch)
    _, sources = cells()
    exec(sources["setup"], scope)
    assert (scope["PROJECT_ROOT"] / "pyproject.toml").is_file()
    assert (scope["PROJECT_ROOT"] / "src/drishti/models/registry.py").is_file()
    assert scope["SOURCE_ZIP"] is None


@pytest.mark.parametrize("bad", ["../outside.py", "drishti/../outside.py", "drishti\\escape.py",
                                 "drishti/source\\escape.py", "symlink", "duplicate"])
def test_notebook_bootstrap_rejects_escaping_or_ambiguous_source(tmp_path, monkeypatch, bad):
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        if bad == "symlink":
            member = zipfile.ZipInfo("drishti/escape.py")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            zipped.writestr(member, "../../outside.py")
        elif bad == "duplicate":
            zipped.writestr("drishti/duplicate.py", "unit boundary")
            with pytest.warns(UserWarning, match="Duplicate"):
                zipped.writestr("drishti/duplicate.py", "conflicting boundary")
        else:
            zipped.writestr(bad, "unit boundary")
    scope = bootstrap_scope(tmp_path, archive)
    isolated_setup(monkeypatch)
    _, sources = cells()
    with pytest.raises(ValueError, match="Invalid source archive"):
        exec(sources["setup"], scope)
    assert not scope["PROJECT_ROOT"].exists()
    assert not (tmp_path / "outside.py").exists()


@pytest.mark.parametrize("resume_kind", [None, "working", "attached"])
def test_notebook_settings_roundtrip_and_resume_does_not_reset_profile(tmp_path, monkeypatch, resume_kind):
    _, sources = cells()
    scope = {}
    exec(sources["configuration"], scope)
    working = tmp_path / "working"
    working.mkdir()
    registry = tmp_path / "models.yaml"
    registry.write_text("models: {}\n")
    descriptor = tmp_path / "mission.yaml"
    descriptor.write_text("unit descriptor boundary")
    scope.update(OUTPUT_ROOT=working / "runs", PROJECT_ROOT=tmp_path / "project",
                 MODEL_REGISTRY=registry, DATASET_YAML=descriptor,
                 PYTHON=Path(sys.executable), json=json, subprocess=subprocess)
    if resume_kind is not None:
        prior = (scope["OUTPUT_ROOT"] / scope["RUN_ID"] if resume_kind == "working"
                 else tmp_path / "input/prior-run")
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(json.dumps({"run_id": scope["RUN_ID"]}))
        if resume_kind == "attached":
            scope["RESUME_BUNDLE"] = prior
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    exec(sources["prepare-settings"], scope)
    settings = load_settings(scope["SETTINGS_FILE"])
    assert commands[-1][-1] == str(scope["SETTINGS_FILE"])
    if resume_kind is None:
        assert settings.dataset == descriptor
        assert "profile" in settings.model_fields_set
        assert settings.profile == scope["PROJECT_ROOT"] / "configs/profiles/kaggle_quality.yaml"
    else:
        assert settings.resume_bundle == prior
        assert settings.dataset is None
        declared = json.loads(scope["SETTINGS_FILE"].read_text())
        assert "profile" not in declared and "profile" not in settings._explicit_fields
        assert "sample_frame_budget" not in declared and "sample_frame_budget" not in settings._explicit_fields


def test_report_preserves_scoped_cloud_measurements_without_accuracy_claims(tmp_path):
    from drishti.report.builder import build_report

    # These numbers are explicit report-boundary inputs, not measured Kaggle results.
    cloud = {"status": "partial", "gpu": {"name": "<script>unit</script>"},
             "rss_peak_bytes": 1234, "vram_peak_bytes": None, "orchestration_wall_seconds": 2.0,
             "limits": {"ram_bytes": 27_000_000_000, "vram_bytes": 14_000_000_000},
             "report_cloud_snapshot": {
                 "scope": "completed_stage_workers_before_final_report_recovery_and_archive",
                 "excludes": ["final report recovery", "archive result"],
                 "authoritative_final_record": "cloud_run.json", "interpretation": "constructed boundary"}}
    bundle = Bundle(tmp_path, {"bundle_version": "1", "status": "partial", "stages": {},
                              "inputs": [], "config_snapshot": {}, "cloud_execution": cloud})
    output = build_report(bundle, Config())
    report = bundle.read_json(output["json"])
    assert report["cloud_execution"] == cloud
    assert report["accuracy"]["status"] == "UNVALIDATED"
    assert report["accuracy"]["absolute_spatial_rmse_m"] is None
    assert report["timing"]["end_to_end_wall_seconds"] is None
    markup = bundle.path(output["html"]).read_text()
    assert "Measured cloud execution snapshot" in markup
    assert "not evidence of a hard instantaneous" in markup
    assert "&lt;script&gt;unit&lt;/script&gt;" in markup
    assert "<script>" not in markup
    assert "cloud_run.json" in markup
