# Running this checkout

## 1. Install

Use Python 3.11/3.12. From the checkout root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ingest,poses,neural,geo,server]'
drishti doctor
```

Windows: `python -m venv .venv`, then `.venv\Scripts\Activate.ps1`.
`pip install -e .` installs the CLI/config/bundle core; stages need their extras:

| Extra | Functions |
|---|---|
| `ingest` | PyAV decode, OpenCV QA/motion, Pillow, MAVLink |
| `poses` | COLMAP, GTSAM, projection and numerical geometry |
| `neural` | Torch/Transformers local inference |
| `openvino` | Local verified IR inference (optional alternate depth backend) |
| `geo` | Open3D mesh, projected imagery, GeoTIFF/LAS/mesh interchange |
| `server` | FastAPI and local Uvicorn |
| `dev` | pytest, Ruff, mypy, httpx |

Doctor reports actual detected packages/devices and registered models. Detection
does not certify stage execution. Blender must be separately installed for FBX;
optional LAZ needs a laspy compressor, tiles need py3dtiles, octrees need
PotreeConverter.

## 2. Prepare your inputs and checkpoints

Copy the descriptor template, set real paths and explicitly declare altitude
datum and timestamp alignment. See [INPUTS-CONFIG](INPUTS-CONFIG.md).
Provision local Depth Anything V2 Small and RT-DETR model directories; use the
[checkpoint registry guide](MODELS-LICENSES.md). There are no bundled weights or
runtime downloads.

Optional semantic overlays require a separately verified local semantic checkpoint
and explicit class mapping. See
[semantic configuration](INPUTS-CONFIG.md#optional-semantic-overlays); the default
`semantics.model=null` keeps this input optional.
Optional recorded barometer height can be enabled with `optional.baro` and the
[barometer configuration](INPUTS-CONFIG.md#optional-barometer-height-fusion).
Declare its actual clock origin and noise source; defaults keep fusion disabled.

## 3. Run and inspect

```bash
drishti stages
drishti run --dataset /path/to/mission.yaml --profile balanced --run-id survey
drishti inspect runs/survey
drishti verify runs/survey
```

For a local high-quality run with larger keyframe, depth, fusion, mesh and
texture budgets, use `configs/profiles/max_quality_local.yaml`. It removes the
Kaggle watchdog policy but still records unknown masking, rejects unsupported
relative depth, and cannot establish accuracy without independent checkpoints.

New runs use exclusive directories. Without `--run-id`, the ID derives from
input/config/code/model content. An existing ID must be resumed rather than
overwritten. `done` is stage completion, not proof of product accuracy. A
`degraded` run identifies actual reduced methods/missing products; `partial`
means required processing failed/blocked and the CLI exits nonzero.

Partial/stage-scoped execution:

```bash
drishti run --dataset /path/to/mission.yaml --run-id ingest-check --upto s1_frameqa
drishti resume runs/ingest-check --upto s6_global
drishti resume runs/ingest-check --set dense.voxel_m=0.2
drishti resume runs/ingest-check --only s10_export
```

`--stage` selects stages with dependencies; `--only` requires already fresh
dependencies. `--force` reruns selected stages. Resume reuses the manifest config
and verifies current code/models/inputs/outputs; changed upstream data invalidates
downstream products. `verify` checks integrity, not cross-run determinism or accuracy.

## 4. Read outputs

`report/report.html` and JSON describe real stages, errors, products, confidence
and independent elevation checks if supplied. With no independent 3D reference,
full absolute accuracy remains **UNVALIDATED**. GPS alignment residual is a fit
diagnostic. Missing scene denominator means visible-scene completeness is null.
The report's `barometer_fusion` section distinguishes provided/parsed/aligned
input from actual final factor use, with retained stage status, offset and fit.

`s10_export/` contains successful deliverables and `exports.json` with explicit
availability. OBJ/glTF/GLB carry observed texture; all geometry has origin/CRS
sidecars. LAS and GeoTIFF carry projected coordinates. Local-coordinate mesh/PLY
exports require their sidecars in GIS; they must not be interpreted as global XYZ.

## 5. View and measure

```bash
npm ci --prefix viewer
npm run build --prefix viewer
python -m server.app
```

Open http://127.0.0.1:8000. Select a bundle, inspect stage/product status, load
available geometry, and pick points for distance/area/height/DSM volume. Volume
requires a local reference plane and observed raster support. Cloud confidence
and inferred controls are explicitly scoped to the cloud preview.
When semantic products exist, the Cesium panel provides a class legend, selectable
2D geographic overlay and original class-GeoTIFF download. Semantic quality remains
UNVALIDATED; imagery is excluded from measurement picks.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing altitude datum | Set `telemetry.altitude_datum` explicitly; `unknown` stays unvalidated |
| Missing absolute altitude | Supply absolute measurements; relative height cannot substitute |
| Unsupported GPS frames | Check clock origin/offset and `ingest.max_telemetry_gap_s` |
| SfM cannot initialize | Inspect blur/overlap/parallax and QA records; no fabricated pose fallback |
| Weak/collinear GPS geometry | Read alignment conditioning warning; tilt uncertainty needs real constraints |
| Joint bundle adjustment retained its input | Inspect S6 `bundle_adjustment.reason`, selected-track support and solver termination; adjust budgets if appropriate |
| No registered checkpoint | Register local Small/detector files with actual hashes/revision |
| No metric depth | Inspect sparse support/alignment residuals; unaligned maps cannot be fused |
| Semantic overlay unavailable | Inspect S3/S9 `semantics.json`, checkpoint mapping and processor support; a valid class raster may still be downloadable |
| Barometer not used | Inspect S0 `barometer.json` and S2/S6 `barometer_fusion`: verify clock origin/offset, nearest-sample tolerance, distinct support and recorded noise |
| Memory exhaustion | Reduce chunk/input/keyframe/mesh limits or coarsen profile; rerun affected stages |
| FBX unavailable | Install Blender on PATH, then resume export |
| Empty Potree tab | Vendor assets and convert octree; read local Potree setup contract |
| Integrity mismatch | Inspect the changed file; preserve original bundle and rerun from trustworthy sources |

See [STATUS](STATUS.md) for validation scope before quoting results.
