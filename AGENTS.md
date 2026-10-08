# DRISHTI — implementation context

## 1. Source of truth
Read `docs/implementation/00-PRD.md`, `01-IMPLEMENTATION-PLAN.md`, and
`02-SYSTEM-DESIGN.md`. The official problem statement remains the evaluation
contract. `docs/build/PLAN.md` records this checkout's implementation decisions;
`docs/build/CONTRACTS.md` defines the shared code interfaces.

## 2. Scope
Recorded video + GPS + flight metadata → one offline, ground-only georeferenced
reconstruction. No live path or on-drone deployment. S5 is folded into S7.
Optional inputs must never become mandatory. Use the documented Python pipeline,
FastAPI server, and React/Cesium/Potree viewer architecture.

## 3. Honesty and correctness
No invented telemetry, model checksums, reconstruction artifacts, confidence, or
accuracy. Absent measurements are null. GPS fit residuals are not independent
accuracy. Invalid mandatory input fails clearly; recoverable failures preserve
real partial products and emit a report. Do not claim a method ran merely because
its package was detected. Keep inferred geometry distinct. Keep relative depth
out of metric fusion. Retain nodata and vertical datum provenance.

## 4. Licensing
Gate actual checkpoints, not model families. Depth Anything V2 Small is the
permissive default; Base/Large are not. Models are supplied in a local registry
with immutable revision and verified file hashes. Blender FBX export is an
isolated external process. No runtime model downloads.

## 5. Engineering
All tunables belong in typed config/YAML. Bundle-relative artifact paths;
atomic manifests; content-based freshness including code/models; measured actual
execution environment. Use local-origin float64 geometry and record its projected
CRS transform. GLB/glTF must be recentered before float32 serialization.

## 6. Delegation
The user explicitly requested agents. Parallel work is permitted using disjoint
file ownership. Read `docs/build/CONTRACTS.md` before implementing. Do not edit
another agent's files or commit. Use apply_patch for edits. Test your owned work;
tell the coordinator exactly what passed and what remains unvalidated.

## 7. Layout
`src/drishti/{config,bundle,runtime,models,stages,io,geo,recon,report}`;
`configs/`, `server/`, `viewer/`, `tests/`, `docker/`, `notebooks/`, `scripts/`.

## 8. Status
2026-10-07: two parallel workstreams plus coordinator integration added the Kaggle
GPU quality runner/profile, importable notebook, atomic source ZIP packaging and
scoped cloud reports. Typed settings target separate maximum 30 decimal GB
RAM/VRAM ceilings with lower measured stop budgets, one physical GPU, stage-isolated
workers, sampled process-tree monitoring, bounded recovery and full-flight/resume
sampling. Current Python suite: **1214 passed, 3 skipped** (OpenVINO/Open3D/Torch
absent); the full post-change run completed in 307.13 s. Ruff passed; configured
regular mypy passed for 56 source files. Wheel, installed console entrypoint,
source ZIP integrity and extracted runner help passed.
Optional-intrinsics fallback, graph confidence invalidation, Blender-exporter
freshness, loaded-versus-executed depth provenance, max-frame tail reporting,
geometric keyframe admission and explicit confidence status/reasons were audited
and contract-tested. GPU/installation boundaries use isolated doubles;
actual Kaggle/CUDA/checkpoint execution and memory fit remain unvalidated.
Prior barometer fusion, semantic overlays and S6 joint BA/final anchors remain
implemented/tested. Optional barometer/semantics remain disabled by default; GNSS
is the absolute anchor. Previous scoped strict mypy passed for barometer/spine/BA.
Viewer typecheck, coordinate/semantic lifecycle tests and production build passed
again after the post-change audit. No real
WebGL scene or trained-checkpoint acceptance was run. Full-repository strict mode
has previously recorded annotation errors and was not rerun this increment.
The supplied source-derived Aukerman package passed local S0/S1 validation with
77 decoded frames, 77 telemetry samples and no QA drops; its altitude datum is
unknown and its playback timing is assembled, so reconstruction acceptance,
absolute accuracy and mission performance remain unvalidated. The approved Depth
Anything V2 Small checkpoint is provisioned locally at its immutable revision and
packaged for Kaggle, but it has not been executed because Torch/CUDA is absent.
Historical GUIDE/RUN_GUIDE/audit claims are not current
verification. See `docs/build/STATUS.md` for measured evidence and
`docs/build/KAGGLE.md` for notebook/run/download/resume instructions.

## 9. Decisions
- Offline PRD governs this build; canonical live topology is future scope.
- Resolve licensing contradictions at checkpoint level; never silently relabel.
- Stable run identity excludes timestamps/timings; integrity and determinism are
  separate concepts.
- Geographic altitude datum is explicitly declared; relative altitude is rejected
  unless a user supplies a valid absolute reference. No implicit geoid correction.
- Only a real recorded single-pass dataset can establish product acceptance.
- S6 jointly refines fixed-calibration camera poses and supported sparse points
  against observed pixels and GNSS after the pose graph. Intrinsics stay at S2
  calibration; invalid/nonconverged adjustment retains graph products. Changed
  poses clear obsolete covariance and unoptimized point confidence before depth.
- Semantics are optional and disabled by default. Explicit source-channel mapping
  and verified local semantic logits produce uncalibrated winning-source probabilities,
  distinct from geometry confidence. S9 declares S3 mask/map dependency and labels
  only observed supported cells. Geographic preview imagery is displayed on the
  ellipsoid, never measurement geometry. Retained semantic provenance is labeled
  with stage status; stale support cannot become a current measurement.
- Optional barometer fusion consumes recorded positive-up metre heights with an
  explicit clock and jointly estimated free constant bias; it does not replace
  mandatory absolute GPS or assign a sensor datum. Reused/conflicting raw samples
  cannot become independent factors. Availability, attempts and accepted final
  usage are separate; estimated bias/fit are not calibration or accuracy. Weather
  drift, pressure conversion, measured sensor lever arms and IMU fusion remain
  unimplemented/unvalidated.
- Kaggle uses one selected physical GPU; multiple devices' VRAM is never pooled.
  GB is decimal; RAM/VRAM ceilings are separate. Sampled memory peaks are lower
  bounds, and polling/Torch allocator controls are not hard instantaneous caps.
  Reports retain a scoped pre-recovery cloud snapshot; final recovery/archive
  evidence belongs to `cloud_run.json`. Archive contents are a pre-creation
  snapshot, without recursive report/ZIP rewriting. Preserve full-flight sampling
  and saved resume settings rather than silently truncating or lowering quality.
