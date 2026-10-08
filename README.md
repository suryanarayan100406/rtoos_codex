# DRISHTI

**Recorded drone video + GPS + flight metadata → offline, georeferenced 3D products.**

This checkout implements the ground-only scope of the supplied
[PRD](docs/implementation/00-PRD.md). The build was planned first and developed
with five implementation agents. The [execution plan](docs/build/PLAN.md) and
[shared contracts](docs/build/CONTRACTS.md) document the decisions.

## Start here

- [Current quickstart](docs/build/QUICKSTART.md)
- [Input and configuration reference](docs/build/INPUTS-CONFIG.md)
- [Architecture and stage methods](docs/build/ARCHITECTURE.md)
- [Model provisioning and licensing](docs/build/MODELS-LICENSES.md)
- [API and measurement definitions](server/README.md)
- [Viewer and offline Potree setup](viewer/README.md)
- [Local/cloud deployment](docs/build/DEPLOYMENT.md)
- [Kaggle GPU notebook and quality-run instructions](docs/build/KAGGLE.md)
- [Testing and real-data acceptance](docs/build/VALIDATION.md)
- [Measured implementation status](docs/build/STATUS.md)

## Install and run

Use **Python 3.11 or 3.12** and Node.js 22+ for the viewer. Installation may use
the network; reconstruction loads local checkpoint files only.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ingest,poses,neural,geo,server]'
drishti doctor
```

On Windows PowerShell activate with `.venv\Scripts\Activate.ps1`.
Register license-cleared local checkpoints as described in the model guide.
Create a dataset descriptor from `configs/datasets/mission.yaml.example`:

```bash
drishti run --dataset /path/to/mission.yaml --profile balanced --run-id my-flight
drishti inspect runs/my-flight
drishti verify runs/my-flight
drishti resume runs/my-flight
```

Build the browser UI inside `viewer/`, then run the local server from the root:

```bash
npm ci --prefix viewer
npm run build --prefix viewer
python -m server.app
```

Open **http://127.0.0.1:8000**. Cesium uses local assets without an Ion token or
external basemap. Potree requires a locally vendored runtime and real octree;
the bounded Cesium point preview works independently.

### Kaggle GPU

Import [`notebooks/kaggle_quality.ipynb`](notebooks/kaggle_quality.ipynb) into
Kaggle. Attach the source ZIP made with `python scripts/package_kaggle.py`, your
recorded mission/video/GPS, and verified local checkpoints. Edit the notebook's
input paths and run the cells in order. The quality profile uses stage-isolated
workers and separate maximum 30 decimal GB RAM/VRAM settings with lower measured
stop budgets. See the [Kaggle runbook](docs/build/KAGGLE.md) for setup, outputs,
resume and sampled-memory scope.

The supplied `aukerman` sample is ready at
`build/aukerman-kaggle-input.zip`; upload it as the mission Dataset. It contains
the source-derived MP4/SRT package and descriptor, but no neural weights.
The ready-to-upload `build/drishti-models.zip` contains the registered Depth
Anything V2 Small checkpoint required by the quality notebook.

## What is implemented

Typed YAML configuration, deterministic PTS ingest, telemetry adapters, frame QA,
COLMAP SfM, robust GPS alignment, executed GTSAM pose graphs and joint calibrated
reprojection/GNSS bundle adjustment, local depth/detector
adapters, optional recorded-barometer height fusion, metric inverse-depth fitting,
tiled point fusion, Poisson meshing,
observed-image texture atlases, DSM/DTM/projected orthomosaic, optional local
semantic segmentation and geospatial class overlays, required export adapters,
accuracy reports, resumable content-hashed bundles, and a measurement UI.

Semantic overlays default to disabled. Enable them with a verified local checkpoint
and an explicit source-channel/class mapping; see the
[configuration reference](docs/build/INPUTS-CONFIG.md#optional-semantic-overlays).
Optional barometer fusion is also disabled by default; its recorded clock, CSV
columns and uncertainty settings are documented in
[barometer setup](docs/build/INPUTS-CONFIG.md#optional-barometer-height-fusion).

**Product acceptance remains unvalidated:** no real single-pass recording,
telemetry, approved neural checkpoints, or independent survey checks were supplied.
The ≤1 m and <15 min targets are targets until independently measured. Missing
inputs/tools produce explicit failures or degraded partial products and a report.
See the status document for actual test evidence and remaining gaps.

## Source documentation

The original `docs/` corpus is preserved. Its live-edge architecture describes
future scope; `GUIDE.md`, `RUN_GUIDE.md`, and the September audit contain historical
claims about an absent checkout. Use `docs/build/` for this implementation.
