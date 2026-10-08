# Kaggle GPU quality run

Import [`notebooks/kaggle_quality.ipynb`](../../notebooks/kaggle_quality.ipynb)
into Kaggle to run the existing offline pipeline with the
[`kaggle_quality` profile](../../configs/profiles/kaggle_quality.yaml).
The notebook installs dependencies, invokes stage-isolated workers, writes an
inspectable project bundle, and offers downloads. Its settings target separate
**30 decimal GB RAM and 30 decimal GB VRAM ceilings** with lower monitored stop
budgets. Quality, actual memory fit, accuracy and mission speed require a real run.

## 1. Package and attach the inputs

From the source checkout, create the source attachment using Python's standard
library; application dependencies are not needed for this command:

```bash
python scripts/package_kaggle.py
```

Upload `build/drishti-kaggle-source.zip` as a Kaggle Dataset. It includes the Python
application, configuration, scripts, notebook and runbooks. Mission data, model
weights, caches and the browser build are provisioned separately. Kaggle may mount
Dataset contents as extracted directories rather than preserving the uploaded ZIP;
the notebook supports both layouts. Rebuild this ZIP after changing application
code or configuration.

Attach two more Datasets:

1. **Your recorded mission:** video, original GPS/flight log and `mission.yaml`.
   Descriptor-relative paths must match the uploaded layout. GPS must include
   absolute altitude and an explicitly declared datum; numeric CSV/other logged
   clocks need an explicit origin. Use the
   [input reference](INPUTS-CONFIG.md) and
   [`mission.yaml.example`](../../configs/datasets/mission.yaml.example).
2. **Verified model files:** a registry plus complete local checkpoint directories.
   See [model provisioning](MODELS-LICENSES.md). Depth Anything V2 **Small** is the
   required permissive default; Base/Large are not substituted. Include processor,
   config and weight files in the registered inventory. RT-DETR is recommended;
   its optional failure retains the existing motion fallback. No notebook cell
   downloads model weights or creates substitute hashes/revisions.

Optional barometer/semantic settings belong in the original mission descriptor.
They remain disabled by default and are not prerequisites for video + GPS.

### Supplied `aukerman` input

The supplied folder is packaged as `build/aukerman-kaggle-input.zip`. Its
SHA-256 is
`ea4b313a2341a127f9ee37699b0869bd6f7ea8d7473cebdf62f6b07aa81a4c92`.
The archive contains:

```text
mission.yaml
flight.MP4       # 77 frames, 1600x1200, 2 fps, about 38.5 s
flight.SRT       # 77 DJI-style GPS/absolute-altitude samples
PROVENANCE.txt
```

Create a Kaggle Dataset named, for example, `aukerman-input`, and upload this ZIP
as its file. The notebook default expects
`/kaggle/input/aukerman-input/aukerman-kaggle-input.zip`; its setup cell safely
extracts the archive to `/kaggle/working/aukerman-input` and sets `DATASET_YAML`
to the extracted `mission.yaml`. If you upload the four files individually,
set `MISSION_ZIP = None`, set `MISSION_ROOT` to their `/kaggle/input/...`
directory, and leave `DATASET_YAML = MISSION_ROOT / 'mission.yaml'`.

The supplied provenance says the MP4 playback timing was assembled at 2 fps from
real source imagery and GPS EXIF, rather than being an original camera recording.
The descriptor therefore declares `altitude_datum: unknown`; the pipeline retains
that provenance and does not claim independently validated altitude or real-flight
acceptance.

### Supplied model package

The supported model package has been provisioned as
`build/drishti-models.zip`. It contains the Apache-2.0 Depth Anything V2 Small
Transformers checkpoint and `models.yaml` with measured hashes. Its SHA-256 is
`62fc698364d954896a447f54a498c7768f3d6e5244177818a294f09921137d00`.

Create a Kaggle Dataset named `drishti-models` and upload this ZIP as its file.
The notebook default expects:

```text
/kaggle/input/drishti-models/drishti-models.zip
```

The setup cell extracts it into `/kaggle/working/drishti-models`, then validates
the registry's immutable revision and every file hash before stages start. The
model package is not part of the source ZIP or Aukerman mission ZIP.

## 2. Import and configure the notebook

Use Kaggle's **Import Notebook** with `notebooks/kaggle_quality.ipynb`. Select a
**GPU** accelerator and enable Internet for dependency installation. The project
environment uses Python **3.11 or 3.12**; if Kaggle's host kernel is newer, setup
installs the pinned `uv` bootstrap tool and provisions Python 3.12 before creating
`/tmp/drishti-kaggle-venv`. Reconstruction itself uses attached local inputs and
offline model/PROJ flags. The setup installs Torch 2.6.0 / torchvision 0.21.0 from
the CUDA 12.4 wheel index, then the pinned application extras into that separate
environment. The host notebook kernel is not replaced.

Edit the first code cell to match the Input panel:

| Notebook variable | Meaning |
|---|---|
| `SOURCE_ZIP` | Attached `drishti-kaggle-source.zip` |
| `MISSION_ZIP` | Attached `aukerman-kaggle-input.zip`; set to `None` for extracted files |
| `MISSION_ROOT` | Writable extraction directory for the mission ZIP |
| `DATASET_YAML` | Actual mission descriptor for a new run |
| `MODEL_ZIP` | Attached `drishti-models.zip`; set to `None` for extracted files |
| `MODEL_ROOT` | Writable extraction directory for the model ZIP |
| `MODEL_REGISTRY`, `MODEL_CACHE` | Extracted registry and checkpoint root |
| `MODEL_PATHS` | Existing registry names mapped to actual Kaggle checkpoint paths |
| `RUN_ID`, `OUTPUT_ROOT` | Safe run basename and writable output directory |
| `RESOURCE_SETTINGS['gpu_index']` | Physical GPU index shown by `nvidia-smi` |
| `SAMPLE_FRAME_BUDGET` | Approximate full-flight sample budget; default 1800 |
| `OVERRIDES` | Typed pipeline overrides, e.g. `geo.resolution_m=0.15` |
| `RESUME_BUNDLE` | Previous complete bundle directory, or `None` for a new run |

`MODEL_PATHS` changes registry locations only; immutable revision, licensing and
file hashes are preserved and verified. Add a semantic model path only if the
descriptor enables that registered checkpoint. The source ZIP is extracted only
into a new directory or reused when its SHA-256 matches the recorded marker.
The mission ZIP is treated the same way. The attached Aukerman package does not
include neural weights; attach the separate `drishti-models` Dataset. If you upload
the model files individually, set `MODEL_ZIP = None`, set `MODEL_ROOT` to their
`/kaggle/input/...` directory, and leave `MODEL_REGISTRY = MODEL_ROOT / 'models.yaml'`
and `MODEL_CACHE = MODEL_ROOT`. An empty registry will correctly stop before depth
inference rather than downloading or inventing a checkpoint.

Run the cells in order. The setup attempts an external Blender installation when
`INSTALL_BLENDER=True`; Blender is needed for actual FBX export. Missing export
tools produce recorded unavailable/degraded results rather than placeholder files.
GPU/package detection is not evidence that inference or reconstruction ran.

## Quality and full-flight sampling

The quality profile uses the implemented SfM/GNSS graph, calibrated joint BA,
metric-aligned Small depth, point fusion, Poisson and observed-image texture path:

| Setting | Value |
|---|---|
| Profile frame cadence / maximum keyframes | 6 fps / 600 |
| Sequential matching overlap | 20 |
| BA work limits | 20,000 points / 200,000 observations / 200 evaluations |
| Depth inference long side | 1036 pixels; normative float32 inference/maps |
| Dense sampling / voxel / tiles | Stride 2 / 0.05 m / 20 m |
| Poisson depth / input point cap | 10 / 2,000,000 |
| Texture atlas / patch | 8192 pixels / 16 pixels |
| Raster sampling / maximum cells | 0.20 m / 12,000,000 |

The atlas and UV expansion impose an actual textured capacity of at most
262,144 faces / 786,432 vertices, below the configured one-million-vertex ceiling.
These are work/resolution settings, not independent accuracy measurements.
Support, metric-depth, visibility, confidence and nodata gates remain active. The
quality profile does not discard an entire frame when the optional RT-DETR detector
is absent or motion support is uncertain: it retains those pixels with an explicit
unknown-support/degraded record and rejects pathological motion masks above the
configured fraction. Supplying a verified RT-DETR checkpoint remains the stronger
dynamic-object path; retained unknown pixels are not a claim that every pixel is
static.
The selected neural adapters use CUDA; SfM, graph/BA, meshing, geospatial work and
external exporters retain their implemented CPU/native backends.

New runs set `ingest.max_frames=0`, avoiding a frame-count cutoff that would discard
the flight tail. When duration metadata is available, the runner reduces cadence
to `min(configured_fps, (sample_frame_budget - 1) / declared_duration_s)` across the
full clip. This is an approximate metadata-based count, not a decoded-duration
measurement or exact hard frame cap. Unknown duration is recorded as unbounded.
Full-resolution frame/depth files, copied inputs and the output ZIP may require
substantial disk space; check the session's working/output allowance.

## Resource controls and recovery

- GB means **1,000,000,000 bytes**. RAM and VRAM are separate; settings above 30 GB
  are rejected. Two T4s do not provide a pooled 32 GB device: one physical UUID is
  selected and made visible to each worker.
- Default reserves are **3 GB RAM / 1 GB VRAM**, with a **0.90 Torch allocator
  fraction**. Actual stop budgets are reduced further according to measured
  available RAM and the selected GPU's total/free memory. Workers recheck GPU
  headroom before allocation.
- Each stage has a fresh subprocess. RSS monitoring includes the orchestrator,
  active worker and observed descendants; shared pages may be counted repeatedly.
  VRAM monitoring uses the selected GPU's compute-process records for that tree.
  Unrelated Jupyter processes are not scanned or killed.
- Default polling sleep is **0.5 s**, excluding query time; GPU queries have a
  **10 s** timeout. Sampled peaks are lower bounds and may miss short spikes.
  Torch limits its allocator, not every native CUDA allocation. These mechanisms
  do not establish a hard instantaneous memory ceiling. Preparation, installation
  and streaming ZIP creation are not continuously sampled like stage workers.
- A resource/telemetry stop or interrupt terminates the worker tree, retains real
  partial files, marks interrupted/dependent results honestly and attempts a
  bounded CPU report refresh (default **120 s**), followed by a streaming ZIP.
  The runner does not automatically lower quality settings to hide a failure.

If you interrupt the run cell, allow cleanup/report/archive to finish before
stopping the session. A partial run returns nonzero; inspect its retained outputs.

## Outputs, downloads and resume

With the default run ID, inspect:

```text
/kaggle/working/drishti-runs/kaggle-quality/
  manifest.json
  inputs/                       # copied mission inputs and registry evidence
  s0_ingest/ ... s10_export/     # only actual produced artifacts
  logs/kaggle_*.log
  report/report.html
  report/report.json
  cloud/report_snapshot.json
  cloud_run.json
/kaggle/working/drishti-kaggle-quality.zip
```

The results cell shows measured budgets, sampled peaks, stop reason, archive
status and bundle integrity. Use its links or Kaggle's Output panel to download
the reconstruction ZIP, report and **final `cloud_run.json`**. Retain the complete
bundle, original georeference sidecars and the exact verified checkpoints.

The HTML/JSON report contains a labeled pre-recovery cloud snapshot; later recovery
measurements are in the final `cloud_run.json`. The ZIP is a pre-creation bundle
snapshot, so its internal archive result remains pending. The final loose
`cloud_run.json` records the actual archive result without recursively rewriting
the ZIP. Orchestration wall time includes preparation through final report recovery,
but excludes dependency setup and ZIP creation. Core stage times remain separate.

For a resumed session:

1. Attach/extract the previous ZIP so the complete run directory containing
   `manifest.json` is available.
2. Set `RESUME_BUNDLE` to that directory and attach the same verified model files.
3. Run the cells in order. The runner copies read-only attached bundles into the
   writable output root. An existing writable bundle resumes in place automatically.
4. Omitted resume profile/sample-budget settings preserve the saved quality and
   cadence. Changed inputs, code, models or overrides invalidate affected stages.

The same bundle can be transferred to a local ground station and checked with
`drishti verify <bundle>` before `drishti resume <bundle>`. Integrity is separate
from determinism and independent accuracy. No independent checks means accuracy
remains **UNVALIDATED**; see the [real-mission protocol](VALIDATION.md).

## Script entrypoint

The notebook invokes the same typed runner available from a GPU-equipped shell:

```bash
python scripts/kaggle_run.py --settings /kaggle/working/kaggle_run.yaml
# Installed console entrypoint:
drishti-kaggle --settings /kaggle/working/kaggle_run.yaml
```

Start from [`configs/kaggle.yaml.example`](../../configs/kaggle.yaml.example).
All paths resolve relative to that YAML. For resume, set `resume_bundle`, omit
`dataset`, and omit `profile`/`sample_frame_budget` to preserve the saved plan.
Current measured implementation checks and unvalidated GPU/mission boundaries are
recorded in [STATUS.md](STATUS.md).
