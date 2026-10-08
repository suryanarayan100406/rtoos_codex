# Measured implementation status

**Date:** 2026-10-07  
**Checkout state:** the source-derived Aukerman mission and a verified Depth
Anything V2 Small checkpoint package are now supplied locally; no independent
survey/checkpoint measurements were supplied.

## Current continuation verification — supplied Aukerman input

The supplied `aukerman` folder is packaged as `build/aukerman-kaggle-input.zip`
for the Kaggle notebook. It contains a descriptor, the source-derived
`flight.MP4`, matching DJI-style `flight.SRT` and `PROVENANCE.txt`; it contains
no model weights. The MP4 has 77 frames at 1600x1200/2 fps and about 38.5 seconds;
the SRT has 77 absolute-altitude GPS samples from 0.0 through 38.0 seconds.

Using the actual files, local CPU validation completed S0 ingest and S1 frame QA
with 77 decoded frames, 77 telemetry samples, 77 selected QA frames and no QA
drop counts. S0 is marked degraded because the supplied altitude datum is
explicitly `unknown`; no datum conversion or accuracy claim was invented. The
source provenance says playback timing was assembled at 2 fps from real source
imagery/GPS EXIF rather than an original camera recording, so this is input and
pipeline validation, not official real-flight acceptance.

A local CPU continuation was attempted through S2 with `--upto s2_poses`. It
completed S0/S1, feature extraction and matching, and entered the COLMAP mapper,
but the audit command limit interrupted it before final S2 publication. No final
pose-graph or metric-spine acceptance result is claimed from that partial run.

The Apache-2.0 Depth Anything V2 Small Transformers checkpoint was provisioned
at immutable revision
`5426e4f0f36572d16453bbda7a8389317b1bef99`, registered with four measured file
hashes, and packaged as `build/drishti-models.zip` (SHA-256
`62fc698364d954896a447f54a498c7768f3d6e5244177818a294f09921137d00`). It has not been executed in
this local environment because Torch/CUDA is absent; trained-checkpoint quality,
GPU memory fit and reconstruction acceptance remain unvalidated.

## Current continuation verification — Kaggle GPU quality runner

Two disjoint agent workstreams implemented the typed stage-isolated GPU runner
and quality profile. Coordinator integration added the importable notebook,
atomic source ZIP helper, shipped entrypoint/data files, scoped cloud report
rendering, boundary tests and [Kaggle runbook](KAGGLE.md). The existing offline
pipeline, checkpoint-level gates, optional sensors/semantics and artifact contracts
remain in use.

The runner targets separate maximum **30 decimal GB RAM / 30 decimal GB VRAM**
configured ceilings, lower measured stop budgets and one physical GPU. Stage
workers release model/device state between invocations. Sampled process-tree
telemetry, allocator limits, interrupted-stage freshness, bounded report recovery
and streaming archive outcomes are recorded without claiming a hard instantaneous
cap. Full-flight sampling uses declared duration metadata and clears early frame
cutoffs; resume preserves the saved plan unless explicitly changed. Reports keep
a labeled pre-recovery snapshot; the loose final `cloud_run.json` is authoritative
for later recovery peaks/archive outcomes.

Current checks used `/tmp/omnirush/drishti-venv`, Python **3.12.3**, NumPy **1.26.4**,
SciPy **1.15.2**, GTSAM **4.2.1** and Rasterio **1.4.3**:

| Check | Measured result |
|---|---|
| `python -m pytest -q -rs --basetemp=/tmp/omnirush/drishti-audit-final-postchange` | **1214 passed, 3 skipped**, 307.13 s |
| Final Kaggle notebook/runner/profile boundary suites | **119 passed**, 90.33 s |
| `python -m ruff check src server tests scripts` | **Passed** |
| `python -m mypy --cache-dir /tmp/omnirush/drishti-audit-final-postchange-mypy src server` | **Passed**, 56 source files; configured regular mode |
| Source and installed `drishti-kaggle --help` | **Passed**; installed with `pip install -e . --no-deps` |
| `pip wheel . --no-deps` | **Passed**; final wheel size 208,304 bytes; SHA-256 `e5ba720165fe4bf22c512ae16b9985984e539718bc5f32f394b84f2de74ca143` |
| `python scripts/package_kaggle.py`, three ZIP integrity checks and extracted runner `--help` | **Passed**; source ZIP size is recorded at packaging time; integrity and extracted help passed |

The **119 Kaggle boundary checks** cover actual OS subprocess/process-group cleanup, atomic
ZIP/confined notebook extraction, full-flight/resume configuration, checkpoint
rebinding, interrupted freshness, optional model execution provenance, source-code
freshness and cloud report boundaries. GPU inventory/CUDA
allocator telemetry and notebook dependency-installation commands use isolated
doubles. These checks do not establish actual Kaggle installation, CUDA inference,
trained-checkpoint quality, 30 GB memory fit, accuracy or mission performance.

This audit also corrected concrete contract issues: configured optional intrinsics
now fall back to self-calibrated SfM with an explicit ledger/reason; graph-refined
but unsupported sparse points lose confidence before depth; the isolated Blender
exporter participates in code freshness; GLOMAP no longer claims supplied
intrinsics were refined merely because it ran; a loaded-but-never-inferred depth
model is not published as executed; and positive `max_frames` records every
unprocessed PTS tail frame instead of silently stopping without a drop record;
keyframe admission now cross-checks robust geometric residuals, inlier/coverage
support and calibrated parallax when optional intrinsics are valid; and stage
ledgers explicitly label confidence as `measured_proxy` or `unavailable` with a
reason.
The Kaggle quality profile now retains pixels when optional RT-DETR is unavailable
and motion support is uncertain, while labeling that support unknown and rejecting
pathological near-full-frame motion masks. This prevents optional detector absence
from collapsing metric depth coverage; it does not claim dynamic-object certainty.
The three skips remain absent OpenVINO, Open3D and optional Torch runtime execution;
the full suite emitted 503 existing third-party deprecation warnings. The latest
viewer typecheck, coordinate tests, semantic lifecycle tests and production build
also passed. Scoped strict solver evidence is from the barometer increment;
full-repository strict mypy was not rerun. No actual GPU or independent geometry
measurements were supplied. Real acceptance remains
**UNVALIDATED** under the [validation protocol](VALIDATION.md).

## Previous continuation verification — optional barometer fusion

Four disjoint workstreams implemented optional recorded-height input, actual
optimizer factors, stage/freshness integration and partial-safe reports. Barometer
fusion is disabled by default. S0 strictly parses positive-up metre heights with
explicit clocks and bounded nearest-sample association. S2/S6 jointly estimate a
free sensor-height offset in the actual GTSAM graph and SciPy BA; mandatory GNSS
remains the absolute anchor. Reused/conflicting raw records and inconsistent
clock fields are excluded rather than treated as independent sensor support.

Final-result usage is distinguished from input availability and solver attempts.
Failed/unsupported solves preserve real baseline/input geometry. Reports expose
accepted BA, graph or retained-S2 provenance, with current usage/bias/fit null
for stale or unsuccessful final-stage evidence. Sensor fit is not independent
accuracy, and the optimized offset is not measured calibration.

Current checks used `/tmp/omnirush/drishti-venv`, Python **3.12.3**, NumPy **1.26.4**,
SciPy **1.15.2**, GTSAM **4.2.1** and Rasterio **1.4.3**:

| Check | Measured result |
|---|---|
| `python -m pytest -q -rs --basetemp=/tmp/omnirush/drishti-barometer-integrated-pytest` | **1085 passed, 3 skipped**, 319.86 s |
| `python -m ruff check src server tests scripts` | **Passed** |
| `python -m mypy --cache-dir /tmp/omnirush/drishti-barometer-integrated-mypy src server` | **Passed**, 55 source files; configured regular mode |
| `python -m mypy --strict --follow-imports=silent` for `recon/barometer.py`, `recon/spine.py`, `recon/bundle_adjustment.py` | **Passed**, 3 owned solver modules |
| CLI `stages`, `doctor`, `run --help` | **Passed** |
| Configuration profiles and explicit-clock CLI overrides | **Passed**; all profiles keep barometer disabled by default |

The **389 new constructed checks** include parser/clock/noise provenance, actual
GTSAM/SciPy execution, tangent and sparse Jacobians, sensor/local-Z offset
invariance, noise weighting, duplicate/tampered support, unchanged-input fallback,
final anchor publication, freshness and a real-S6 mathematical result consumed
by the report. They are implementation evidence, not flight acceptance.
The same three absent-runtime skips and 503 third-party deprecation warnings
described below remain. Viewer code was unchanged, so viewer checks were not
rerun; the previous semantic-increment evidence below still applies.

Doctor measured WSL2 CPU execution, about **8.04 GB** visible RAM, no CUDA,
empty checkpoint registry and absent Blender/COLMAP CLI/GLOMAP/Potree tools.
No original barometer log, real mission dataset, checkpoint or independent
measurements were supplied. Recorded-flight sensor quality, absolute accuracy,
memory and mission performance remain unvalidated. Pressure conversion,
weather-drift compensation, measured sensor lever arms and IMU preintegration
are not implemented. Full-repository strict mypy was not rerun.

## Previous continuation verification — optional semantic overlays

Five disjoint parallel workstreams completed the in-progress F9/F15 semantic path:
S3 inference, S9/S10 geospatial products, server/viewer integration, foundation
freshness and reporting. Optional local semantic-logits checkpoints require the
existing exact-file license gate and an explicit source-channel/class mapping.
No semantic checkpoint was supplied or executed in this verification environment.

- S3 always records semantic availability; successful frame inference publishes
  camera-shaped uint16 labels and float32 winning-source probabilities. Dynamic,
  unknown, unmapped and low-probability pixels remain zero. Optional failures retain
  actual dynamic masks, and execution provenance requires validated frame inference.
- S9 uses final-camera/DSM/mesh visibility and nearest labels on supported observed
  geometry. It preserves nodata/CRS/datum provenance and emits class/probability
  GeoTIFFs plus a bounded, actually warped EPSG:4326 preview when available.
  Required raster products and optional semantic exports fail independently.
- The API validates/confines optional descriptors and file references. The viewer
  provides class legends, original class-raster downloads and a selectable geographic
  ellipsoid overlay, with revision/toggle/late-load cleanup and geometry-only picks.
- Resume identity covers configured semantic checkpoint contents even after a
  previously unavailable model. Reports retain old semantic provenance with stage
  status, without presenting stale support as a current measurement.

That increment's checks used `/tmp/omnirush/drishti-venv`, Python **3.12.3**, NumPy
**1.26.4**, SciPy **1.15.2**, GTSAM **4.2.1**, Rasterio **1.4.3**, and Node
**24.21.0** / npm **11.19.0**:

| Check | Measured result |
|---|---|
| `python -m pytest -q -rs` | **696 passed, 3 skipped**, 319.86 s |
| `python -m ruff check src server tests scripts` | **Passed** |
| `python -m mypy --cache-dir /tmp/omnirush/drishti-integrated-mypy src server` | **Passed**, 53 source files; configured regular mode |
| CLI `stages`, `doctor`, `run --help` | **Passed** |
| Viewer `npm run typecheck`, `test:coordinates`, `test:semantics`, `build` | **Passed** |

The three Python skips are actual OpenVINO IR execution, actual Open3D Poisson
execution and optional real-Torch constructed-logit math: those runtimes are
absent. Semantic loader/model boundaries use explicitly isolated doubles; map
projection/storage/readers, freshness and report checks use constructed contract
inputs. Viewer semantic checks exercise isolated React/Cesium lifecycle behavior,
not a real WebGL scene. This evidence does not establish trained-checkpoint quality
or mission acceptance. The suite emitted third-party Affine/Rasterio and
Starlette/AnyIO deprecation warnings; the viewer build emitted Cesium `eval` and
large-chunk warnings.

Doctor measured WSL2 CPU execution, about **8.04 GB** visible RAM, no CUDA, empty
model registry, and absent Blender/COLMAP CLI/GLOMAP/Potree tools; pycolmap was
installed. No real dataset, checkpoints or independent measurements were added.
Semantic class accuracy, scene placement, absolute accuracy, target-hardware memory
and mission performance remain unvalidated. Full-repository strict mypy was not
rerun; the earlier strict-mode limitation below remains recorded.

## Previous continuation verification — S6 joint refinement

S6 now executes joint fixed-calibration reprojection/GNSS bundle adjustment after
the existing global pose graph. The sparse SciPy solver jointly varies camera
rotations, centers and supported points; deterministic coverage-first whole-track
selection bounds the problem. Nonconvergence, inadequate support, invalid
geometry or failed positive-depth checks retain the real input result and record
the reason. Accepted geometry has matching final anchors, refreshed point
confidence, and no obsolete pose-graph covariance attached to changed poses.

That session's checks in Python **3.12.3**, NumPy **1.26.4**, SciPy **1.15.2** and
GTSAM **4.2.1**, using `/tmp/omnirush/drishti-ba-venv`:

- `python -m pytest -q -rs`: **268 passed, 2 skipped**. The skips are the actual
  OpenVINO IR and Open3D Poisson executions; those runtimes are absent in this
  environment. The 79 new BA/config/S6 checks include actual joint optimization,
  GNSS sigma consumption, late-camera coverage under budgets, deterministic
  selection, immutable-input fallback, final anchor publication and depth revision
  guards. These are constructed numerical/artifact contracts, not mission runs.
- `python -m ruff check src server tests scripts`: **passed**.
- `python -m mypy src server`: **passed, 51 source files**.
- `python -m mypy --strict src/drishti/recon/bundle_adjustment.py`: **passed**.
- CLI `stages` and `doctor`: **passed**; doctor measured CPU execution, no CUDA,
  no registered checkpoints, and absent Blender/Potree tools.
- Full-repository `mypy --strict src server` is **not green**: 294 pre-existing
  annotation/third-party-boundary errors in 25 files. The configured repository
  check is the regular mypy command above. This corrects the handoff's claim of
  full strict typing; only the new BA module was verified with `--strict` then.

Viewer checks were not rerun for that Python-only increment. Real single-pass
acceptance, independent absolute accuracy and mission performance remain
unvalidated; no data or checkpoints were added.

## Previous-session evidence (handoff)

Five delegated workstreams implemented the code and returned their owned checks.
The integrated Python suite subsequently passed **191 tests** in Python 3.12
with the pinned core/ingest/poses/geo/server environment. The checks include
actual GTSAM pose-graph optimization, Open3D Poisson execution, OpenVINO IR
adapter execution, GeoTIFF/LAS/PLY/mesh reader checks, API measurements, hash/
resume logic and strict model gating. Ruff passed the integrated source/server/
test/script set in the agent runs. Viewer agent checks passed `npm install`,
coordinate tests, TypeScript check, Vite build and an offline browser smoke.

The prior handoff described a root type review of broad mypy annotation errors in
third-party OpenCV/trimesh boundary stubs and mixed JSON dictionaries. Owned
agents fixed them. Its **no issues in 50 source files** result is consistent with
the configured regular check, but its full strict claim is not reproducible in
the S6 continuation. That viewer check passed `npm ci`, typecheck, coordinate tests
and production build. Current integrated evidence is recorded at the top of this
document.

## Requirement traceability

| Requirement | Current state |
|---|---|
| F1 deterministic video | Implemented, isolated/tests; real codec run pending |
| F2 SRT/CSV/MAVLink/EXIF | Implemented strict adapters/tests; real logs pending |
| F3 logged clock alignment | Implemented bounded PTS/clock policy; real offset validation pending |
| F4 QA/keyframes/drop log | QA/drop logging implemented; keyframe admission uses robust homography residual/inlier/coverage checks plus a rotation/planar-compensated geometric parallax proxy, with optional calibrated essential-matrix triangulation; real flight quality pending |
| F5 optional streams | Presence/hash, optional intrinsics self-calibration fallback, declared RTK paths and optional recorded-height barometer alignment/fusion implemented/tested; IMU preintegration and real sensor acceptance pending |
| F6 SfM/self-calibration | COLMAP execution implemented; real aerial run pending |
| F7 metric visual+GNSS graph | Actual GTSAM graph, Sim3 alignment, joint fixed-calibration reprojection/GNSS BA and optional barometer/free-offset factors implemented/tested; mission/independent accuracy pending |
| F8 GPS-derived CRS | Implemented/tests; real mission edge cases pending |
| F9 dynamic masking | RT-DETR + motion fallback and optional semantic-logits adapter implemented/tested; approved checkpoint runs pending |
| F10 metric depth | Correct inverse-domain alignment implemented; approved checkpoint/aerial quality pending |
| F11 feed-forward geometry | MAY, not shipped |
| F12 dense + textured mesh | Point fusion + Poisson + observed texture implemented; real scene pending |
| F13 confidence/inferred | Stored/propagated in cloud/mesh/raster/export paths; Poisson support-radius provenance is a proxy rather than direct ray-observation proof; calibration pending |
| F14 DSM/DTM/orthomosaic | Implemented with nodata and projection; GIS acceptance pending |
| F15 semantic overlays | Optional S3 labels → observed-DSM class/probability rasters → API/Cesium legend, overlay and downloads implemented/contract-tested; trained-checkpoint and scene acceptance pending |
| F16 required exports | Adapters implemented; Blender/optional external execution pending |
| F17 report | Implemented honest HTML/JSON/checkpoint path; no measured mission report supplied |
| F18 viewer/measurements | Implemented API/UI/Cesium and bounded cloud preview; real GLB/browser scene pending |
| F19 reproducible bundle | Implemented atomic content-hash/resume/registry provenance; post-change wheel/source package rebuilt and integrity-checked |
| N1 ≤1 m absolute | **UNVALIDATED** — no independent mission checks |
| N2 completeness | Support metrics implemented; visible-scene denominator unavailable |
| N3 performance | Measurement infrastructure implemented; no claim without mission run |
| N4 reliability | Partial/fallback paths implemented; invalid mandatory input correctly fails |
| N5 reproducibility | Stable identity + current integrity implemented; cross-run mission comparison pending |
| N6 offline | Core/run-time model network disabled; assets must be pre-provisioned |
| N7 licensing | Checkpoint-level gate implemented; exact user files still need evidence |
| N8 honesty | Missing/unsupported measurements remain null or explicit; audit on real run required |
| N9 compute fit | Tiling/bounds plus Kaggle single-GPU stage isolation and sampled memory-stop controls implemented/contract-tested; measured target hardware run pending |
| N10 single command/browser | CLI and local server/viewer implemented; mission execution pending |

## Explicit remaining gaps

- The supplied Aukerman files are source-derived real imagery/GPS with assembled
  playback timing, not an original recorded camera flight. Its approved Small
  checkpoint is provisioned in `build/drishti-models.zip` but has not executed
  locally. Independent check points, geoid grid, Blender, LAZ compressor,
  py3dtiles or Potree assets are not present in this checkout.
- No approved semantic checkpoint, independently labeled scene or real WebGL
  semantic-overlay acceptance run was supplied. Resize-only semantic processors
  are supported; crop/pad inverse pixel transforms are not implemented.
- Barometer fusion supports recorded metre heights with one constant unknown bias;
  pressure conversion, weather drift and measured sensor lever arms are not
  implemented. Real sensor/clock/noise acceptance remains pending.
- ByteTrack/SAM2/RAFT learned path, IMU
  preintegration, RTK factor consumption beyond corrected-track input, optional
  feed-forward geometry, 3DGS and PDAL SMRF are not claimed. Joint calibrated BA
  is implemented; further joint intrinsics refinement and mission validation
  remain pending.
- F4's image-displacement admission is deliberately reported as a heuristic until
  a real geometric-parallax/coverage selector and real-flight validation replace
  it. Mesh measured/inferred flags use observed-cloud support distance and do not
  certify that every Poisson vertex lies directly on an observed depth ray.
- The `<15 min` and `≤1 m` official targets remain design targets until measured
  on appropriate real hardware/data. This document intentionally does not check
  them off based on unit tests.
