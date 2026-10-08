# DRISHTI — Complete Guide (Usage + How It Works)

> **Historical source document:** this guide describes an earlier checkout that
> was not supplied with this folder. Use [the current build guide](build/QUICKSTART.md)
> and [measured status](build/STATUS.md) for the code implemented here.

**DRISHTI** turns a *pre-recorded* drone video plus its GPS/flight telemetry into **one accurate,
georeferenced 3D model** and a set of standard deliverables — **entirely offline, on the ground**.
There is no on-drone/edge component and no live/streaming path: you hand it a recording, it produces a
model.

This is the practical, **code-accurate** guide to the software *as it actually runs today*. It has two
parts:

- **[Part 1 — Using DRISHTI](#part-1--using-drishti):** install, prepare inputs, run, resume, inspect,
  read the outputs, serve and view them.
- **[Part 2 — How it works](#part-2--how-it-works):** the bundle model, config system, CRS, compute
  placement, the honesty/confidence model, and every pipeline stage in detail.

> **Scope note.** The conceptual/architecture docs under `docs/` (e.g. `docs/02-HOW-IT-WORKS.md`)
> describe the *full envisioned system*, including forward-looking pieces. **This guide describes only
> what the shipped code does.** Where a stage detects an optional tool but does not yet run it, that is
> stated plainly in [Honest limitations](#18-honest-limitations--what-is-not-yet-wired). Nothing here is
> aspirational.

> **Honesty policy (binding).** Every number DRISHTI reports is *measured* from a real run. Absent data
> is `null`, never a placeholder. A stage that falls back to a lesser method still produces a real
> result and records a `degraded_reason`. Confidence is a measured 0..1 value per stage, capped by the
> method that actually ran. DRISHTI ships **no dataset** and fabricates nothing about the world.

---

## Table of contents

**Part 1 — Using DRISHTI**
1. [What you need (the input contract)](#1-what-you-need-the-input-contract)
2. [Install](#2-install)
3. [Prepare your data (the dataset descriptor)](#3-prepare-your-data-the-dataset-descriptor)
4. [Generate a realistic test dataset](#4-generate-a-realistic-test-dataset)
5. [Run the pipeline](#5-run-the-pipeline)
6. [Where each stage runs (local vs cloud-T4)](#6-where-each-stage-runs-local-vs-cloud-t4)
7. [Resume, inspect, verify, doctor, stages](#7-resume-inspect-verify-doctor-stages)
8. [The outputs — bundle and deliverables](#8-the-outputs--bundle-and-deliverables)
9. [Serve and view in the browser](#9-serve-and-view-in-the-browser)
10. [Troubleshooting](#10-troubleshooting)

**Part 2 — How it works**
11. [Architecture in one picture](#11-architecture-in-one-picture)
12. [The project bundle and manifest](#12-the-project-bundle-and-manifest)
13. [Configuration and precedence](#13-configuration-and-precedence)
14. [CRS derivation](#14-crs-derivation)
15. [Compute placement policy](#15-compute-placement-policy)
16. [The honesty model: confidence and graceful degradation](#16-the-honesty-model-confidence-and-graceful-degradation)
17. [The pipeline, stage by stage](#17-the-pipeline-stage-by-stage)
18. [Honest limitations — what is not yet wired](#18-honest-limitations--what-is-not-yet-wired)
19. [File map](#19-file-map)

---
---

# Part 1 — Using DRISHTI

## 1. What you need (the input contract)

DRISHTI ingests **a video file + a telemetry track**, described by a small YAML **dataset descriptor**.
That descriptor is the entire input contract. Concretely:

| Input | Required? | What it is |
|-------|-----------|------------|
| **Video** | **yes** | The recorded drone clip (e.g. `flight.MP4`). Decoded with PyAV or OpenCV. |
| **Telemetry** | **yes** | A GPS/flight track time-aligned to the video. One of: `dji_srt`, `csv`, `mavlink`, `exif`. |
| IMU / baro / intrinsics / RTK | optional | Improve or constrain later stages if present. |
| Ground check-points | optional | A CSV of surveyed points; enables *validated* elevation accuracy in the report. |

You do **not** need a drone, a GPU, or an internet connection to run the pipeline. You need the footage
and its metadata. Georeferencing is derived from the GPS track in the telemetry.

---

## 2. Install

DRISHTI's **core** is lightweight (Pydantic, Typer, Rich, NumPy). The heavy, environment-specific
pieces are **optional-dependency groups** you install per environment. Python **3.11+** is required.

```bash
# from the repo root
python -m venv .venv && . .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -e .                                    # core only (CLI works; heavy stages will report as blocked)
```

### Extras, and which stages they unlock

| Extra | Installs | Needed by |
|-------|----------|-----------|
| `video` | `av`, `opencv-python-headless` | S0 ingest, S1 frame-QA, S3/S4 mask & depth I/O |
| `poses` | `pycolmap` | S2 poses (SfM) |
| `geo` | `pyproj`, `rasterio`, `shapely` | S2 CRS projection, S9 GeoTIFF rasters, report validation |
| `recon` | `open3d`, `trimesh`, `pygltflib` | S7 dense, S8 mesh, S10 export |
| `depth` | `torch`, `openvino`, `onnx` | S3 masking, S4 metric depth |
| `telem` | `pymavlink`, `piexif` | MAVLink/CSV telemetry parsing |
| `spine` | `gtsam` | optional S2 tightening (*not yet wired* — see §18) |
| `server` | `fastapi`, `uvicorn` | the API server + web viewer |
| `dev` | `pytest`, `ruff`, `mypy`, `httpx` | development / CI |

**Local dev machine (no discrete GPU)** — a full CPU-runnable install:

```bash
pip install -e ".[video,poses,geo,recon,depth,telem]"
```

**Cloud T4 session (Colab/Kaggle)** — everything, so the neural stages run on CUDA:

```bash
pip install -e ".[video,geo,recon,poses,spine,telem,depth,server]"
```

> **On Colab/Kaggle, prefer the notebooks over this raw command.** Their current runtime is **Python
> 3.13**, for which **Open3D has no wheel** (it caps at cp312) — so `.[recon]` makes this atomic install
> abort, taking the `drishti` CLI down with it. `notebooks/drishti_colab_full.ipynb` (and the tier-split
> `drishti_cloud_t4.ipynb`) handle this: they install each group in isolation and, on a >3.12 kernel,
> auto-build a Python 3.12 env with `uv`. They also add `transformers` and `laspy`, which no extra lists
> but S3/S4 and S10 need (see §18). Use the raw command only on a machine with Python ≤3.12.

After installing, **always run the doctor** — it tells you the truth about *this* machine:

```bash
drishti doctor
```

It prints detected CPU/RAM/CUDA/OpenVINO, which optional modules import, and, per stage, whether it can
run here or will fail loudly. See [§7](#7-resume-inspect-verify-doctor-stages).

---

## 3. Prepare your data (the dataset descriptor)

A descriptor is a YAML file (conventionally under `configs/datasets/`). Minimal example:

```yaml
name: my_mission
video: data/my_mission/flight.MP4
telemetry:
  format: dji_srt              # dji_srt | csv | mavlink | exif
  path: data/my_mission/flight.SRT
optional:
  imu: null
  baro: null
  intrinsics: null
  rtk: null
crs:
  mode: derive_from_gps        # derive_from_gps (default) | set
  epsg: null                   # e.g. 32643 to force a CRS instead of deriving it
report:
  check_points: null           # path to a ground-truth CSV to validate elevation accuracy
```

### Telemetry formats

| `format` | Source | Notes |
|----------|--------|-------|
| `dji_srt` | DJI subtitle sidecar | Bracketed fields, e.g. `[latitude: ..] [longitude: ..] [abs_alt: ..]`. |
| `csv` | Any flight log exported to CSV | Column-mapped time/lat/lon/alt. |
| `mavlink` | ArduPilot/PX4 `.tlog` | Parsed via `pymavlink` (the `telem` extra). |
| `exif` | Per-image GPS EXIF | For image-sequence sources. |

### CRS options

- `derive_from_gps` (default): DRISHTI picks the correct UTM zone from the median GPS position — you do
  nothing. See [§14](#14-crs-derivation).
- `set` + `epsg`: force a specific projected CRS (e.g. `epsg: 32643`).

---

## 4. Generate a realistic test dataset

DRISHTI ships **no data** (honesty policy). To get a real input to run against, use
`scripts/make_sample_dataset.py`, which assembles the `video + telemetry + descriptor` contract in one
of two honest ways. Full details in [`scripts/README.md`](../scripts/README.md).

```bash
# REAL: fetch an openly-licensed aerial image set, read its real GPS EXIF,
#       and mux the real pixels + real coordinates into MP4 + DJI-style SRT.
python scripts/make_sample_dataset.py --dataset brighton_beach   # ~62 MB, 18 imgs — fast smoke test
python scripts/make_sample_dataset.py --dataset aukerman         # ~543 MB, real BUILDINGS (house + barn)
python scripts/make_sample_dataset.py --list                     # show the real registry

# SYNTHETIC: render a labelled procedural city with GROUND-TRUTH poses/GPS (no network).
python scripts/make_sample_dataset.py --synthetic
```

Each mode writes `data/<name>/` (git-ignored) plus a matching `configs/datasets/<name>.yaml` and a
`PROVENANCE.txt`. What each mode is — and is not:

| Mode | Pixels | GPS | Honesty note |
|------|--------|-----|--------------|
| `--dataset <name>` | **real** source imagery | **real** per-image EXIF | Only *playback timing* is synthesized. |
| `--synthetic` | rendered | ground-truth | Clearly labelled *synthetic*; a known-geometry target that makes accuracy *checkable*. |

Real downloads are cached under `data/.cache/` so re-running is instant. The large sets (`aukerman`,
`lewis`) are best fetched **on the T4**, where the full run happens — the cloud notebook has a cell for
exactly that (see [§6](#6-where-each-stage-runs-local-vs-cloud-t4)). On a no-GPU laptop, prefer
`brighton_beach` or `--synthetic`.

> Proven end to end: `brighton_beach` (real DJI pixels + real GPS EXIF) ingests and derives EPSG:32615;
> `synthetic_city` derives EPSG:32643. Both from real S0 runs.

---

## 5. Run the pipeline

The single entry point is `drishti`. To build a fresh bundle and run the pipeline:

```bash
drishti run --dataset configs/datasets/brighton_beach.yaml --profile balanced
```

**`--dataset/-d` is required** (there is no positional form). The full flag set for `run`:

| Flag | Meaning |
|------|---------|
| `--dataset, -d PATH` | **(required)** the dataset descriptor YAML. |
| `--profile, -p NAME` | `fast`, `balanced` (default), or `max`. |
| `--output-root, -o DIR` | where to create the bundle (default from config; typically `runs/`). |
| `--run-id ID` | explicit run id (default: timestamp + random suffix). |
| `--only STAGE` | run just this one stage (its dependencies must already be complete). |
| `--upto STAGE` | run from the start up to *and including* this stage. |
| `--force` | re-run stages even if their inputs/params are unchanged. |
| `--set, -s k=v` | override any config key, e.g. `-s dense.tsdf_voxel_m=0.03` (repeatable). |
| `--config-dir DIR` | directory holding `default.yaml` + `profiles/`. |

### Profiles

`--profile` selects a quality/speed tradeoff (files under `configs/profiles/`):

- **`fast`** — quick settings for a smoke test.
- **`balanced`** — the default; sensible quality at reasonable cost. Runs the whole pipeline end-to-end.
- **`max`** — the highest-quality settings the environment can hold (intended for the T4). **Caveat:**
  `max` selects `dense.method=gaussian` + `mesh.method=2dgs`, which this build does **not** implement —
  `s7_dense` / `s8_mesh` will **raise loudly** rather than silently fall back. Use it with the two
  methods overridden to the implemented path:
  `--profile max --set dense.method=tsdf --set mesh.method=poisson` (this keeps `max`'s other quality
  knobs). The cloud notebooks already do this.

### A quick smoke test

Run only the first stage to confirm your inputs parse and the CRS derives:

```bash
drishti run --dataset configs/datasets/brighton_beach.yaml --upto s0_ingest
drishti inspect runs/<run_id>
```

When `run` starts, it prints a header (run id, dataset, profile, bundle path, detected compute, and the
resolved stage order) and an **honest pre-flight warning** listing any stages that cannot run in this
environment — they are not skipped silently; they will fail loudly when reached, and the warning tells
you which extras to install or to use the T4 tier.

---

## 6. Where each stage runs (local vs cloud-T4)

The dev target is an **Intel Core Ultra 7 155H with no discrete GPU**. The neural stages are meant to
run on a **free cloud T4** (Colab/Kaggle, ~16 GB), and everything else on the CPU ground station. Each
stage declares its compute need; DRISHTI's placement policy maps that to a recorded environment label
(see [§15](#15-compute-placement-policy)). The practical tiering:

| Tier | Stages | Why |
|------|--------|-----|
| **Cloud T4** (recommended) | S0 → **S7** | S2 (COLMAP) is CPU-heavy and S3/S4/S7 need CUDA; run them where the GPU is. |
| **Ground station** (CPU) | S8 → report | Poisson meshing, rasterization, export, and report are CPU work. |

Because the **bundle is content-hashed and resumable**, stages finished on the T4 are recorded as done;
when you continue on the ground station, they are skipped and the run picks up at `s8_mesh`.

### The cloud-T4 workflow

`notebooks/drishti_cloud_t4.ipynb` runs the *same* package and the *same* CLI on a T4 — no stubs. The
flow:

1. Enable the GPU runtime; the notebook confirms the T4 with `nvidia-smi` and `drishti doctor`
   (expect `cuda: yes`).
2. Provide a mission — **Option D** generates a realistic dataset right there (the best place to fetch
   the large `aukerman` buildings set).
3. Run the heavy stages and stop after dense:

   ```bash
   drishti run --dataset "$MISSION" --profile max --set dense.method=tsdf --set mesh.method=poisson --upto s7_dense
   ```

   (The two `--set` overrides keep `max`'s quality knobs but use the implemented TSDF + Poisson path;
   plain `--profile max` raises at `s7_dense` — see [§5 Profiles](#profiles).)

4. `inspect` + `verify` the bundle, zip it, and download it.
5. On the ground station, unzip into `runs/` and **resume** — the GPU stages are already done:

   ```bash
   drishti resume runs/<run_id>          # continues at s8_mesh, reusing the recorded config
   ```

**Prefer a single environment?** `notebooks/drishti_colab_full.ipynb` runs **all 11 stages** on the T4
(with a full import → run → export flow and logs), then you only *view/serve* the finished bundle
locally — no heavy deps required on your machine. It uses the `balanced` profile (end-to-end on the
implemented path).

---

## 7. Resume, inspect, verify, doctor, stages

All commands fail loudly on missing inputs/dependencies — nothing is faked.

### `resume` — continue an existing bundle

```bash
drishti resume runs/<run_id>              # positional bundle path
drishti resume runs/<run_id> --upto s9_geo
drishti resume runs/<run_id> --force      # ignore freshness, re-run
```

`resume` **reuses the config recorded in the bundle's manifest** (so it is reproducible); it takes
`--only`, `--upto`, `--force` but not `--profile` (the profile is already baked into the snapshot).

### `inspect` — read the manifest

```bash
drishti inspect runs/<run_id>
```

Prints provenance (run id, dataset, creation time, versions), the derived **CRS**, the **inputs** table
(role / present / bytes / path), and the **per-stage table**: status, environment (`local`/`cloud-t4`),
wall-clock seconds, measured confidence, and any degradation note. This is the ground truth — not a
progress bar.

### `verify` — integrity check

```bash
drishti verify runs/<run_id>
```

Recomputes the hash of every completed stage's outputs and checks them against the manifest. Reports
`OK` / `MISMATCH` / `MISSING` per stage; exits non-zero if anything changed on disk since it was
recorded.

### `doctor` — what can run here

```bash
drishti doctor
```

Probes hardware (CPU, RAM, CUDA + VRAM, OpenVINO devices, and whether this is a cloud-GPU tier), which
optional **modules** import, per-stage **readiness**, and optional **external tools** (GLOMAP, COLMAP
CLI, Blender, PotreeConverter, GTSAM, PDAL, py3dtiles) that unlock non-degraded paths.

### `stages` — the canonical order

```bash
drishti stages
```

Lists the pipeline order and each stage's dependencies.

---

## 8. The outputs — bundle and deliverables

Every run produces a **project bundle**: one directory under `runs/<run_id>/` that is the single source
of truth for that run. Layout:

```
runs/<run_id>/
  manifest.json          # the run's spine — provenance, per-stage records, timings, confidence
  inputs/                # (bundle scaffolding)
  logs/
  s0_ingest/             # telemetry.json, frames.json, decoded frames
  s1_frameqa/            # frameqa.json, keyframes.json
  s2_poses/              # poses.json, spine.json, sparse_points.json, COLMAP workdir
  s3_masking/            # masks.json + per-keyframe PNG masks
  s4_depth/              # depth.json + per-frame .npy metric depth maps
  s6_global/             # poses_global.json, global.json
  s7_dense/              # dense.ply (+ dense.json)
  s8_mesh/               # mesh.ply (+ mesh.json)
  s9_geo/                # dsm.tif, dtm.tif, ortho.tif (+ geo.json)
  s10_export/            # deliverables (see below) + exports.json
  report/                # report.json, report.html
```

### Deliverables (from `s10_export`)

DRISHTI attempts each output format with a **real** tool; a format whose tool is missing is recorded as
*skipped, with a reason* — never faked. The **MUST** set is `obj, ply, las, geotiff, gltf, glb, fbx`;
if any MUST format is skipped, the export stage is marked *degraded*.

| Deliverable | Format | Tool |
|-------------|--------|------|
| Textured/colored mesh | `model.obj`, `model.glb`, `model.ply` | trimesh (per-vertex color) |
| Mesh (Autodesk) | `model.fbx` | headless Blender |
| Point cloud | `points.las` / `points.laz` | laspy (+ pyproj for CRS) |
| DSM / DTM / ortho | `dsm.tif`, `dtm.tif`, `ortho.tif` | rasterio (GeoTIFF) |
| Web tiles (optional) | `3dtiles/` | py3dtiles |
| Web point cloud (optional) | `potree/` | PotreeConverter |

### The accuracy report

`report/report.html` (and `report.json`) summarizes every stage's metrics, confidence, timing, and any
degradation. If you supplied ground **check-points**, it reports **measured** elevation error
(`rmse_z_m`, `mae_z_m`, `bias_z_m`) sampled from the DSM. If you did not, accuracy is reported as
**UNVALIDATED** — never a fabricated number.

---

## 9. Serve and view in the browser

The optional API server and web viewer are thin, honest windows onto the same core — every number they
show comes straight from `manifest.json`.

### API server (FastAPI)

```bash
pip install -e ".[server]"
DRISHTI_RUNS_DIR=runs python -m server.app        # or: uvicorn server.app:app --port 8000
```

Environment variables: `DRISHTI_RUNS_DIR` (where bundles live), `DRISHTI_CONFIG_DIR`,
`DRISHTI_CORS_ORIGINS`, `DRISHTI_HOST`, `DRISHTI_PORT`. Key endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | version + runs dir |
| `GET` | `/api/doctor` | same compute/stage readiness as the CLI |
| `GET` | `/api/runs` | list all bundles with rollup state |
| `POST` | `/api/runs` | start a run (background thread; returns `202` + run id) |
| `GET` | `/api/runs/{id}` | run summary (rollup + per-stage) |
| `GET` | `/api/runs/{id}/manifest` | the full manifest |
| `GET` | `/api/runs/{id}/report` | the HTML accuracy report |
| `GET` | `/api/runs/{id}/files/{path}` | a bundle artifact (path-traversal protected) |

The server runs real pipeline runs in a background thread and reports status by reading the on-disk
manifest, so status is truthful and survives a restart.

### Web viewer (React + Vite + CesiumJS)

```bash
cd viewer
npm install
npm run dev            # http://localhost:5173 ; dev-proxies /api -> :8000
```

The viewer lists runs (polling every 5 s so a running pipeline updates live), shows the per-stage table,
links the accuracy report and each deliverable (skipped formats show their reason), and renders the
georeferenced **mesh (3D Tiles)** on a **fully offline** Cesium globe (no Ion token, no CDN). For
production, `npm run build` produces `viewer/dist`, which the API server mounts at `/` (single origin,
no CORS).

---

## 10. Troubleshooting

- **A stage says it will "fail loudly" in the pre-flight warning.** It is missing its extras in *this*
  environment. Install them (`pip install -e ".[recon,geo,depth]"`) or run that stage on the T4. Run
  `drishti doctor` to see exactly what is missing.
- **`config error` / exit code 2.** The descriptor or an override failed to load/validate. The config
  model forbids unknown keys — check for typos in the descriptor or `--set`.
- **A run failed partway.** The bundle is still there and resumable. Fix the cause and
  `drishti resume runs/<run_id>`. The failed stage's real error string is recorded in the manifest and
  shown by `inspect`.
- **`make_sample_dataset.py` real mode reports no GPS.** Some sets include a preview image with no EXIF;
  the real images carry GPS. The script prints an EXIF diagnostic on the first image when it finds none.
  (Historical gotcha: DJI stores `GPSAltitudeRef` as a single byte — handled.)
- **Console shows odd characters on Windows.** The CLI reconfigures stdout to UTF-8 so box-drawing and
  status glyphs render even when redirected; if you still see mojibake, ensure your terminal codepage is
  UTF-8.

---
---

# Part 2 — How it works

## 11. Architecture in one picture

```
 dataset.yaml ──► [S0 ingest] ─► [S1 frame-QA] ─► [S2 poses] ─┬─► [S3 masking] ─► [S4 depth] ─┐
   video + telem                                              └─► [S6 global] ───────────────┤
                                                                                              ▼
   report.html ◄─ [report] ◄─ [S10 export] ◄─┬─ [S9 geo]  ◄──────────────────────  [S7 dense] ◄┘
                                             └─ [S8 mesh]  ◄─────────────────────── (S7 dense)
```

The canonical order (there is **no S5** — live fusion is folded into S7):

```
s0_ingest → s1_frameqa → s2_poses → s3_masking → s4_depth →
s6_global → s7_dense → s8_mesh → s9_geo → s10_export → report
```

Every stage reads and writes **only** through the project bundle. That is the whole trick behind the
local/cloud split: because stages exchange data via files in the bundle, any stage can run on the CPU
ground station or a cloud GPU, and the next stage continues from whatever was written.

---

## 12. The project bundle and manifest

A **bundle** (`src/drishti/bundle/`) is a versioned, content-hashed directory. `manifest.json` is its
spine and the single source of truth. It is written **atomically after every stage**, so a run is always
resumable and its status always reflects reality.

**Manifest contents** (`bundle/manifest.py`):

- Provenance: `bundle_version`, `drishti_version`, `run_id`, `created_utc`, `dataset_name`.
- `config_snapshot` + `dataset_snapshot` (so a run reproduces itself; `resume` reads these).
- `seeds`, `inputs` (each with a SHA-256 + byte size), and the derived `crs`.
- `stages[]` — one **StageRecord** each:

  | Field | Meaning |
  |-------|---------|
  | `status` | `pending` / `running` / `done` / `degraded` / `failed` / `skipped` |
  | `environment` | where it ran (`local` / `cloud-t4`) |
  | `wall_seconds` | measured wall-clock time |
  | `inputs_hash`, `params_hash` | drive freshness (skip when unchanged) |
  | `outputs` | logical name → bundle-relative path |
  | `outputs_hash` | combined hash of outputs (feeds downstream freshness) |
  | `confidence.summary` | measured 0..1, or `null` if not computed |
  | `metrics` | **measured values only** |
  | `degraded_reason` | set when a real fallback was used |
  | `error` | the real error string if the stage failed |

### Freshness, skipping, and resumability (`stages/runner.py`)

Before running a stage the runner:

1. **Checks dependencies:** every `requires` dependency must have a record with status `done` or
   `degraded`, else it stops and tells you to `drishti resume`.
2. **Computes hashes:** `params_hash` from the stage's config slice, and `inputs_hash` from each
   dependency's `outputs_hash` (or `missing:<dep>`) plus any extra input hashes (e.g. S0's raw files).
3. **Skips if fresh:** if a previous record exists, is `done`/`degraded`, and both hashes match — the
   stage is skipped (unless `--force`). This is why a stage re-runs exactly when something upstream it
   depends on actually changed.

On failure, the runner records `status=failed` + the real error, saves the manifest, and re-raises — no
faked success. A stage that used a fallback returns normally with `degraded_reason` set → `degraded`.

---

## 13. Configuration and precedence

Config is a validated Pydantic model (`config/models.py`) that **forbids unknown keys** (a typo is an
error, not a silent no-op). Sections: `run, compute, ingest, frameqa, poses, spine, crs, masking, depth,
dense, mesh, geo, export, report, models`. A few defaults worth knowing:

- `ingest.target_fps = 4.0` — frame sampling rate.
- `depth.model = "depth_anything_v2_base"` — the metric-depth model.
- `dense.method = "tsdf"`, `dense.tsdf_voxel_m = 0.05` — TSDF fusion voxel size.
- `mesh.method = "poisson"`, `mesh.poisson_depth = 11` — Poisson reconstruction depth.
- `crs.mode = "derive_from_gps"`.

**Precedence** (highest wins), from `config/loader.py`:

```
--set  >  env (DRISHTI__section__key)  >  dataset descriptor (crs/report)  >  profile  >  configs/default.yaml
```

- `--set dense.tsdf_voxel_m=0.03` overrides everything for that key.
- Environment overrides use a **double underscore**: `DRISHTI__dense__tsdf_voxel_m=0.03`.
- The descriptor contributes `crs` and `report` settings.
- `--profile` layers `configs/profiles/<name>.yaml` over `configs/default.yaml`.

> Do not confuse config env overrides (`DRISHTI__...`, double underscore) with the *server's* runtime
> env vars (`DRISHTI_RUNS_DIR`, single underscore) or the sample script's `DRISHTI_MISSION`.

---

## 14. CRS derivation

When `crs.mode = derive_from_gps` (the default), S0 picks the working coordinate reference system from
the GPS track itself, with pure-Python UTM math (no pyproj needed for the choice):

- Take the **median longitude/latitude** of the track.
- Compute the **UTM zone** from the median longitude.
- Choose the EPSG code: **`32600 + zone`** for the northern hemisphere, **`32700 + zone`** for the
  southern.

The result is recorded as `crs` in the manifest (`epsg`, `derived_from = gps_median_lonlat`, vertical =
`ellipsoidal`). All downstream geometry lives in this projected CRS, so distances are metric and the
outputs are georeferenced. Set `crs.mode = set` with an explicit `epsg` to override.

---

## 15. Compute placement policy

Each stage declares a `ComputeNeed` (`runtime/compute.py`): `cpu_only`, `needs_cuda`, `vram_gb`,
`openvino_ok`. Under the default `auto` policy, `place()` maps that to a recorded **environment label**:

- `cpu_only` → **`local`**.
- `needs_cuda` **or** `vram_gb > 0` → **`cloud-t4`**.
- `force_local` / `force_cloud` override the policy.

So S3, S4, and S7 are labeled `cloud-t4`; all others are `local`. Two clarifications:

- The label is **provenance + the recommended tier**, recorded per stage. It does not physically prevent
  a stage from running wherever you invoke the CLI — when you run on the T4, the CPU-labeled stages run
  there too.
- Under `auto`, `openvino_ok` does **not** keep a stage local; it only influences the backend chosen
  once a stage is placed. Local OpenVINO/CPU execution of the neural stages happens under `force_local`.

This is why the recommended split ([§6](#6-where-each-stage-runs-local-vs-cloud-t4)) runs S0–S7 on the
T4 (S2's COLMAP is CPU-heavy but benefits from the T4's faster CPU) and resumes S8–report on the ground.

---

## 16. The honesty model: confidence and graceful degradation

Two mechanisms keep DRISHTI's self-reporting honest.

**Confidence** is a measured 0..1 value per stage, stored in `manifest.stages[].confidence.summary`
(`null` when not computed). Each stage's value is a *real measurement* of that stage's output — the
exact formula per stage is in [§17](#17-the-pipeline-stage-by-stage). It is not a progress bar and not a
guess.

**Graceful degradation** (`runtime/reliability.py`) is a shared ladder so fallbacks are consistent and
honest. A stage lists candidate **tiers** (best first), each with an `available` flag it computed from
the runtime. `choose()` returns the highest-quality available tier, a human `degraded_reason` if a
better tier was skipped, and a **confidence ceiling** for the chosen tier. The stage then reports
`min(measured, ceiling)` — a fallback can never *claim* the confidence of the method it replaced. The
L0–L6 ceilings (conservative policy caps, not measurements):

| Level | Meaning | Cap |
|-------|---------|-----|
| L0 | full quality (preferred model + cloud GPU / RTK) | 1.00 |
| L1 | full method, local (OpenVINO/CPU) | 0.95 |
| L2 | reduced settings | 0.85 |
| L3 | alternative method (classical instead of neural) | 0.70 |
| L4 | prior-only geometry | 0.50 |
| L5 | near-passthrough (capability disabled, pipeline continues) | 0.35 |
| L6 | minimal viable | 0.20 |

If *no* tier is available, that is **absence, not degradation**: the stage raises `NoViableTier` and
fails loudly rather than reporting a low-confidence success.

---

## 17. The pipeline, stage by stage

For each stage: **purpose**, what it **requires**, its **compute** label, the **real libraries** it uses,
its **outputs**, its measured **confidence** formula, and its **degradation** behavior. All grounded in
`src/drishti/stages/`.

### S0 — `s0_ingest` — Ingest & time-align telemetry
- **Requires:** none (root stage). **Compute:** `local`.
- **Does:** decodes the clip, parses and time-aligns the GPS/flight telemetry, **derives the CRS** from
  the GPS track, samples frames at `ingest.target_fps`, and attaches interpolated GPS to each frame.
- **Libraries:** PyAV (`av`) preferred, else OpenCV for decode; telemetry parsers for
  dji_srt/csv/mavlink/exif; pure-Python UTM math for the CRS.
- **Outputs:** `s0_ingest/telemetry.json`, `s0_ingest/frames.json`, and a directory of decoded `.jpg`
  frames.
- **Confidence:** telemetry temporal coverage — `min(1.0, telemetry_duration / video_duration)`.
- **Degradation:** none. Fails loudly on missing video/telemetry or if no frames decode.

### S1 — `s1_frameqa` — Frame QA & keyframe selection
- **Requires:** `s0_ingest`. **Compute:** `local`.
- **Does:** scores each frame for sharpness (variance of Laplacian) and exposure clipping, drops
  unusable frames, and selects keyframes by real inter-frame parallax (ORB feature displacement) or a
  fixed stride.
- **Libraries:** OpenCV (ORB + BFMatcher), NumPy.
- **Outputs:** `s1_frameqa/frameqa.json` (all scored frames), `s1_frameqa/keyframes.json` (selected).
- **Confidence:** QA accept rate — `len(accepted) / len(scored)`.
- **Degradation:** `degraded_reason` set if fewer than 2 keyframes were selected. Fails loudly if every
  frame fails QA.

### S2 — `s2_poses` — Camera poses & metric spine
- **Requires:** `s1_frameqa`. **Compute:** `local` (COLMAP is CPU-bound).
- **Does:** COLMAP recovers intrinsics + camera poses in an arbitrary visual frame; a **Sim(3)**
  (Umeyama scale + rotation + translation) fit then anchors that frame to the GPS track in the projected
  CRS. The fit RMSE is the accuracy proxy.
- **Libraries:** pycolmap (feature extraction, matching, incremental mapping), NumPy, pyproj (projection),
  Umeyama fit.
- **Outputs:** `s2_poses/poses.json` (intrinsics + per-pose center/rotation/lat/lon/alt),
  `s2_poses/spine.json` (Sim3 params, origin, `fit_rmse_m`, correspondence count),
  `s2_poses/sparse_points.json` (sparse cloud in the ground frame), plus the COLMAP workdir.
- **Confidence:** shrinks as fit RMSE grows relative to GNSS sigma —
  `clamp(gnss_sigma_m / (fit_rmse_m + 1e-6), 0, 1)`.
- **Degradation:** `degraded_reason` notes when optional GTSAM factor-graph tightening is skipped
  (needs `gtsam` + IMU). **See [§18](#18-honest-limitations--what-is-not-yet-wired):** GTSAM is only
  *detected*, not yet invoked. Fails loudly with fewer than 3 keyframes, <3 registered images, or <3 GPS
  correspondences.

### S3 — `s3_masking` — Dynamic-object masking
- **Requires:** `s2_poses`. **Compute:** `cloud-t4` (`vram≈3 GB`, OpenVINO-capable locally).
- **Does:** detects dynamic classes (people/vehicles/animals) with RT-DETR and writes per-keyframe
  binary masks (255 = dynamic/exclude, 0 = keep), grown by a morphological dilation margin so moving
  objects don't leak into the reconstruction.
- **Libraries:** HuggingFace `transformers` RT-DETR on `torch`; OpenCV for mask raster + dilation; NumPy.
- **Outputs:** `s3_masking/masks.json` + a directory of per-keyframe `.png` masks.
- **Confidence:** uses the reliability ladder. Measured value `1.0 - min(1.0, mean_dynamic_fraction)`
  (less dynamic area ⇒ more usable static scene), then capped by the chosen tier.
- **Degradation:** the L0 tier (`sam2_refined`) is **not built in this version**, so the stage always
  runs the L2 `rtdetr_boxes` tier — confidence is therefore **capped at 0.85** and a `degraded_reason`
  is recorded. The chosen tier is also stored in `metrics.masking_tier`.

### S4 — `s4_depth` — Metric depth per keyframe
- **Requires:** `s2_poses`, `s3_masking`. **Compute:** `cloud-t4` (`vram≈4 GB`).
- **Does:** runs Depth Anything V2 for relative depth, then fits a **scale + shift** to the S2 sparse
  points that reproject into each frame to make it **metric**; dynamic pixels (from S3 masks) are zeroed.
  Frames with too few anchors are kept relative and flagged.
- **Libraries:** HuggingFace `transformers` Depth Anything V2 on `torch`; NumPy scale/shift fit; OpenCV
  to read masks; depth saved as `.npy`.
- **Outputs:** `s4_depth/depth.json` (per-frame anchor count, residual, degraded flag) + a directory of
  `.npy` metric depth maps.
- **Confidence:** fraction of frames successfully metric-aligned — `n_metric / n_frames` (a frame needs
  at least 8 sparse anchors).
- **Degradation:** `degraded_reason` set if fewer than 70% of frames became metric; per-frame, a frame
  with <8 anchors keeps relative depth and is flagged.

### S6 — `s6_global` — Global consistency
- **Requires:** `s2_poses`. **Compute:** `local`.
- **Does:** a global pass over the S2 spine — computes global consistency metrics (median neighbor
  spacing as a drift proxy, revisit/loop-closure candidates) and emits the poses S7 consumes.
- **Libraries:** NumPy; haversine for the revisit test; probes for a `glomap` binary on PATH.
- **Outputs:** `s6_global/poses_global.json`, `s6_global/global.json` (pose count, median step, loop
  candidates, `glomap_available`).
- **Confidence:** binary on GLOMAP presence — `0.9` if the `glomap` binary is found, else `0.75`.
- **Degradation:** `degraded_reason` notes GLOMAP is not installed. **See
  [§18](#18-honest-limitations--what-is-not-yet-wired):** the poses are currently a pass-through of the
  S2 spine in *both* cases — GLOMAP is detected but not yet invoked.

### S7 — `s7_dense` — Dense reconstruction (TSDF fusion)
- **Requires:** `s4_depth`, `s6_global`. **Compute:** `cloud-t4` (`vram≈6 GB`).
- **Does:** fuses per-keyframe metric depth + color (dynamic pixels already zeroed) into a TSDF volume
  and extracts a dense colored point cloud in the ground CRS.
- **Libraries:** Open3D (TSDF fusion, `sdf_trunc = 4·voxel`); a pure-Python PLY writer; NumPy.
- **Outputs:** `s7_dense/dense.ply` (colored cloud) + `s7_dense/dense.json` (method, voxel, frames
  fused, point count, bbox).
- **Confidence:** fraction of poses actually fused — `used_poses / total_poses`.
- **Degradation:** none. Non-TSDF methods (e.g. Gaussian splatting) fail loudly — the shipped dense path
  is TSDF; an empty cloud raises.

### S8 — `s8_mesh` — Surface mesh
- **Requires:** `s7_dense`. **Compute:** `local` (Open3D Poisson on CPU).
- **Does:** screened Poisson reconstruction over the dense cloud, with low-density/extrapolated vertices
  cropped and flagged; per-vertex color is the shipped texturing baseline.
- **Libraries:** Open3D (normal estimation/orientation, Poisson, density-quantile cropping, mesh write);
  NumPy.
- **Outputs:** `s8_mesh/mesh.ply` (vertex colors + normals) + `s8_mesh/mesh.json` (Poisson depth, vertex
  and face counts, `cropped_low_density_frac`, `texture = per_vertex_color`).
- **Confidence:** `1.0 - cropped_low_density_fraction` (fewer vertices cropped as low-density ⇒ higher).
- **Degradation:** `degraded_reason` set when `mesh.texture = mvs_texturing` is requested — the atlas
  path is **not built** (see [§18](#18-honest-limitations--what-is-not-yet-wired)); per-vertex color is
  used. Non-Poisson methods and zero-face results fail loudly.

### S9 — `s9_geo` — Georeferenced raster products
- **Requires:** `s7_dense`. **Compute:** `local`.
- **Does:** rasterizes the dense CRS cloud into a **DSM** (max elevation), **DTM** (ground filter), and
  RGB **orthomosaic**, written as georeferenced GeoTIFFs in the derived CRS.
- **Libraries:** rasterio (compressed GeoTIFF writer); NumPy-only grid math; probes for `pdal`.
- **Outputs:** `s9_geo/dsm.tif`, `s9_geo/dtm.tif`, optional `s9_geo/ortho.tif`, and `s9_geo/geo.json`
  (EPSG, bounds, resolutions, `dtm_method`, DSM coverage).
- **Confidence:** DSM cell coverage — `finite_cells / total_cells`.
- **Degradation:** `degraded_reason` notes PDAL SMRF is unavailable. **See
  [§18](#18-honest-limitations--what-is-not-yet-wired):** the DTM is *always* computed by morphological
  opening; PDAL is only detected and used to relabel `dtm_method`, not yet invoked.

### S10 — `s10_export` — Deliverables
- **Requires:** `s8_mesh`, `s9_geo`. **Compute:** `local`.
- **Does:** emits output formats from the canonical bundle products; each is attempted with a real tool,
  and a format whose tool is missing is recorded as *skipped with a reason*.
- **Libraries:** trimesh (OBJ/glTF/GLB), laspy + pyproj (LAS/LAZ), Blender via subprocess (FBX), file
  copy (PLY/GeoTIFF), external tools for 3D Tiles / Potree.
- **Outputs:** per-format files under `s10_export/` + `s10_export/exports.json` (per-format ok/path/
  reason; the MUST set and which succeeded).
- **Confidence:** fraction of **MUST** formats that succeeded — `must_ok / must_wanted`, where MUST =
  `{obj, ply, las, geotiff, gltf, glb, fbx}`.
- **Degradation:** `degraded_reason` lists any MUST formats skipped for missing tools. Optional formats
  (laz/3dtiles/potree) missing tools are recorded but do not degrade the stage.

### report — Final accuracy + confidence report
- **Requires:** `s10_export`. **Compute:** `local`.
- **Does:** reads the finished manifest and emits `report.json` + `report.html` summarizing every
  stage's metrics, confidence, timing, and degradation. If ground check-points are supplied, validates
  elevation error against the DSM; otherwise reports accuracy as **UNVALIDATED**.
- **Libraries:** the report renderer; rasterio + NumPy + stdlib `csv` + pyproj for optional check-point
  validation.
- **Outputs:** `report/report.json`, `report/report.html`.
- **Confidence:** a run-health flag — `1.0` if no stage failed, else `0.0` (not an accuracy measurement).
- **Degradation:** none of its own. Check-point accuracy (`rmse_z_m`, `mae_z_m`, `bias_z_m`) is reported
  only when a real check-points file, rasterio, and a DSM are all present.

---

## 18. Honest limitations — what is not yet wired

The shipped code genuinely computes: COLMAP SfM, the Sim(3) GPS georeferencing fit, Depth Anything V2
metric scale/shift, TSDF fusion, Poisson meshing, GeoTIFF rasterization, and multi-format export. The
following are **detected/declared but not yet implemented**, and are called out so the docs never
overstate the software:

- **S3 SAM2 pixel refinement (L0).** Not built; masking always runs the RT-DETR box tier (L2), so its
  confidence is capped at 0.85.
- **S8 MVS-Texturing atlas.** Not built; meshes ship with per-vertex color. Requesting
  `mesh.texture = mvs_texturing` records a `degraded_reason`.
- **S7 Gaussian splatting.** Not shipped; the dense path is TSDF, and other methods fail loudly.

Three stages currently **probe for an optional tool and adjust their label/confidence without actually
invoking it** — worth knowing when reading a manifest:

- **S6 GLOMAP.** Poses pass through from S2 unchanged whether or not the `glomap` binary is present, yet
  confidence is reported as 0.9 (vs 0.75) when it is on PATH.
- **S9 PDAL SMRF.** The DTM is always the morphological-opening result; when `pdal` imports,
  `dtm_method` is labeled `pdal_smrf` even though SMRF is not run.
- **S2 GTSAM.** No factor-graph tightening runs; having `gtsam` + IMU merely suppresses the
  `degraded_reason`.

> These three are the one place where a reported label/confidence can imply a refinement the code does
> not perform. Everything else in a manifest reflects work that genuinely ran. Wiring these tools (or
> making their labels reflect that they are pending) is tracked as follow-up work.

---

## 19. File map

| Path | What lives there |
|------|------------------|
| `src/drishti/cli.py` | the `drishti` CLI (run/resume/inspect/verify/doctor/stages) |
| `src/drishti/stages/` | the 11 pipeline stages + `runner.py` (order, freshness, execution) |
| `src/drishti/bundle/` | bundle + manifest (the on-disk contract) |
| `src/drishti/config/` | Pydantic config models + layered loader |
| `src/drishti/runtime/` | compute detection/placement, dependency probing, reliability ladder |
| `src/drishti/geo/` | CRS derivation, projection, Umeyama align, rasterization |
| `src/drishti/io/` | video decode, telemetry parsers, exporters, point-cloud/PLY I/O |
| `src/drishti/models/` | depth + detection model loaders |
| `src/drishti/recon/` | SfM, depth-fit, TSDF helpers |
| `src/drishti/report/` | report builder + HTML renderer |
| `configs/` | `default.yaml`, `profiles/`, `datasets/` |
| `scripts/make_sample_dataset.py` | generate a realistic input (real or synthetic) |
| `notebooks/drishti_colab_full.ipynb` | run the **entire** pipeline on a free T4 (import → run → export + logs) |
| `notebooks/drishti_cloud_t4.ipynb` | run the GPU-heavy stages on a free T4, then resume locally |
| `server/app.py` | FastAPI server over the same core |
| `viewer/` | React + Vite + CesiumJS web viewer |

---

*For agent/contributor context (status, decision log, honesty policy in full), see `AGENTS.md`. For the
conceptual architecture narrative, see `docs/02-HOW-IT-WORKS.md`.*
