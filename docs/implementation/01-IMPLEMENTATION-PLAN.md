# DRISHTI — Implementation Plan (all phases)

**Purpose:** The complete, phase-by-phase plan to build DRISHTI — **recorded footage + telemetry → one
full-quality georeferenced 3D model, offline, ground-only**. All phases are planned here up front (as
requested); we then build them in order. Each phase lists its goal, the pipeline stages it realizes,
tasks, tech, **local vs cloud** compute placement, deliverables, exit criteria, and risks.

**Audience:** The build team and AI agents implementing DRISHTI. Pairs with the
[PRD](00-PRD.md) (what/why) and the [System Design](02-SYSTEM-DESIGN.md) (how the code is structured).
Requirements referenced as **F#/N#** are defined in the PRD.

**TL;DR**
- **Seven phases (0–6).** Phase 0 lays foundations; Phases 1–5 build a **thin end-to-end pipeline
  first**, then deepen each stage; Phase 6 hardens for production.
- **Guiding order:** *make it run end-to-end on a real clip early* (even coarsely), then improve
  accuracy/completeness/speed — so there is always a working system to measure and demo.
- **Compute split is explicit per phase:** light/orchestration/geo/export/viewer **local (Intel +
  OpenVINO)**; heavy neural stages **cloud (free T4)**; every stage exchanges data via the **project
  bundle**, so placement is configurable.
- **Every phase ends with measurable exit criteria** and updates [`../../AGENTS.md`](../../AGENTS.md)
  (status + decision log). No stubbed outputs — a phase is "done" only when it runs on real data.

---

## Global build principles

1. **Thin slice first.** Phase 1 wires S0→S10 end-to-end with the simplest real implementation of each
   stage (real poses, real depth, real mesh — just not yet the best). Later phases deepen quality. This
   guarantees a demoable, measurable system from early on.
2. **Real data only.** Each phase is validated on a **real** clip + telemetry. No synthetic stand-ins,
   no stubbed returns (per [`../../AGENTS.md` §5](../../AGENTS.md)).
3. **Bundle-first.** Every stage reads/writes the on-disk **project bundle** with a typed contract, so
   stages are independently runnable, resumable, and relocatable (local↔cloud).
4. **Measure, don't claim.** Each phase records **measured** timings/accuracy into the run manifest and
   report; targets stay labeled "design target" until measured.
5. **Config over code.** Every tunable goes to `configs/`; nothing hardcoded.
6. **License-clean & offline** at every step.

**Compute legend:** 🖥️ = local (Intel Ultra 7 155H, CPU + Arc iGPU/NPU via OpenVINO) · ☁️ = free cloud
GPU (Colab/Kaggle T4 ~16 GB).

---

## Phase map (at a glance)

| Phase | Name | Stages realized | Outcome |
|-------|------|-----------------|---------|
| **0** | Foundations & scaffolding | — | Repo, config, bundle, stage-runner, model registry, CI, Docker; no-op pipeline runs on a real input |
| **1** | Ingest, frame QA & metric spine | S0–S2 (+thin S3–S10) | **Thin end-to-end**: real poses + georeferenced sparse cloud + a coarse mesh + real exports |
| **2** | Depth, masking & prior geometry | S3–S4 | Metric depth + dynamic/semantic masks; optional feed-forward geometry |
| **3** | Dense reconstruction & meshing | S6–S8 | Dense fusion / 3DGS → high-quality textured mesh with confidence |
| **4** | Georeferencing, semantics & exports | S9–S10 | DSM/DTM + orthomosaic + semantic layers + full format set, all georeferenced |
| **5** | Accuracy report & web viewer | S10 + UI | Measured accuracy/confidence report + interactive web viewer with measurement |
| **6** | Robustness, performance & hardening | all | Graceful degradation, tiling/scale, speed tuning, reproducibility, packaging |

Dependencies are linear (each builds on the previous), except the **viewer (Phase 5)** and
**exports (Phase 4)** can be developed against Phase 1's thin outputs in parallel once the bundle schema
is stable.

---

## Phase 0 — Foundations & scaffolding

**Goal:** A running skeleton: the repo, config system, project-bundle model, stage runner, model
registry, and Docker/envs — such that a **no-op pipeline executes end-to-end over a real input** and
emits a valid (mostly empty) bundle + manifest. No reconstruction yet, but the rails exist.

**Scope / stages:** infrastructure only.

**Tasks**
- Repo scaffold per [`../../AGENTS.md` §7](../../AGENTS.md): `pyproject.toml`, `src/drishti/`, `configs/`,
  `tests/`, `docker/`, `notebooks/`, `viewer/`, `server/`.
- **Config system** (N5): Pydantic v2 models + YAML profiles (`default.yaml`, `profiles/{fast,balanced,max}.yaml`,
  `datasets/*.yaml`) + CLI/env overrides. No magic numbers anywhere else.
- **Project-bundle** (F19): on-disk schema (directory + JSON manifest), versioned, content-hashed,
  resumable; typed read/write API.
- **Stage runner:** a `Stage` protocol (typed inputs/outputs over the bundle), a DAG runner with
  resume/skip-if-fresh, per-stage timing + confidence capture, structured logging.
- **Model registry** (N5, N7): name → {version, sha256, license, runtime (OpenVINO/torch), source};
  offline cache; **license gate** that refuses to load a non-permissive model in the shipped path.
- **CLI** (`drishti`): `run`, `run --stage`, `resume`, `inspect`, `doctor` (env/compute check).
- **Compute abstraction:** a runtime selector (OpenVINO local vs torch/CUDA cloud) + a "where does this
  stage run" policy read from config.
- **Docker/envs:** a CPU/OpenVINO local image and a CUDA cloud image; a Colab/Kaggle bootstrap notebook.
- **CI:** pytest + lint + a smoke test that runs the no-op pipeline on a tiny real clip.

**Compute placement:** 🖥️ all (infrastructure).

**Deliverables:** runnable `drishti` CLI; a no-op pass producing a valid bundle+manifest on a real
clip; green CI; both Docker images build.

**Exit criteria**
- `drishti run --dataset <real-clip>` completes and writes a schema-valid bundle + manifest (input
  hashes, config snapshot, seeds) with **zero fabricated content**.
- `drishti doctor` reports detected compute (cores, iGPU/NPU via OpenVINO, CUDA if on cloud).
- Loading a non-permissive model is **refused** by the registry gate (unit-tested).
- `../../AGENTS.md` §8 status + §9 decision log updated.

**Risks:** over-engineering the framework before a real stage exists — keep Phase 0 minimal but real;
the bundle schema will evolve, so version it from day one.

---

## Phase 1 — Ingest, frame QA & metric spine (thin end-to-end)

**Goal:** Turn a **real** recording + telemetry into **real camera poses and a georeferenced sparse
reconstruction**, then push a *simple-but-real* result all the way to exports — the **thin end-to-end
slice**. After this phase, DRISHTI produces a georeferenced point cloud + a coarse mesh + valid OBJ/PLY/
LAS/GeoTIFF from real footage.

**Scope / stages:** S0 (ingest+align), S1 (frame QA/keyframes), S2 (poses + metric spine), plus a
**thin** S7/S8/S9/S10 (sparse→coarse mesh→georef→export) to close the loop.

**Tasks**
- **S0 ingest (F1–F3, F5):** deterministic video decode (PyAV/ffmpeg); telemetry adapters for **DJI
  SRT**, CSV, MAVLink, EXIF; time-align frames↔telemetry; record optional-stream presence. **Fail
  loudly** on missing mandatory fields.
- **S1 frame QA (F4):** blur/exposure gating; keyframe selection by parallax/coverage; log every drop
  with a reason.
- **S2 poses (F6):** SfM via **pycolmap + GLOMAP**, intrinsics self-calibrated.
- **S2 metric spine (F7–F8):** **GTSAM** factor graph fusing visual + GNSS (IMU/RTK when present);
  **derive CRS/UTM from GPS median lon/lat**; Umeyama/Sim(3) alignment to the geographic CRS; emit a
  metric georeferenced trajectory + sparse cloud with confidence.
- **Thin close-the-loop:** sparse→coarse surface (Open3D Poisson on sparse+normals or a low-res TSDF);
  georeference; export **OBJ/PLY/LAS/GeoTIFF** (the MUST set minus texture polish).

**Compute placement:** 🖥️ all of Phase 1 (COLMAP/GLOMAP + GTSAM + Open3D run on CPU; OpenVINO for any
feature/descriptor nets). No cloud needed yet — keeps the thin slice self-contained.

**Deliverables:** georeferenced camera trajectory; sparse+coarse point cloud (LAS/PLY); coarse mesh
(OBJ); a first GeoTIFF; per-stage timings in the manifest.

**Exit criteria**
- Poses + metric spine solve on a **real** clip; **CRS is derived from GPS** (never hardcoded);
  trajectory error vs. GPS is **measured and reported**.
- The MUST export set (OBJ/PLY/LAS/GeoTIFF) opens correctly in QGIS/CloudCompare/MeshLab, georeferenced.
- Missing mandatory telemetry fields cause a **loud, clear failure** (tested).
- `../../AGENTS.md` updated.

**Risks:** single-pass low parallax → weak SfM; mitigate with GNSS-constrained BA and keyframe
selection. Telemetry format variety → keep adapters small and test each on a real sample.

---

## Phase 2 — Depth, masking & prior geometry (S3–S4)

**Goal:** Add **metric depth** and **perception** so the reconstruction becomes dense-ready and clean:
per-keyframe metric depth scale-aligned to the spine, dynamic-object masks, and semantic labels;
optional feed-forward geometry as an enhancement.

**Scope / stages:** S3 (dynamic + semantic masking), S4 (metric depth + optional feed-forward geometry).

**Tasks**
- **S4 depth (F10):** **Depth Anything V2** (small/base, **OpenVINO** locally) and **Metric3D v2** (T4)
  for metric depth; **scale-align** depth to the metric spine (per-view scale/shift from sparse points);
  emit depth + confidence.
- **S3 masking (F9):** **RT-DETR + SAM 2 + ByteTrack** for dynamic objects; **RAFT** motion residual;
  semantic labels (ground/building/vegetation/water/vehicle). Masks feed dense fusion to exclude movers.
- **S4 feed-forward geometry (F11, optional):** **MapAnything/Pi3** pointmaps in **chunked windows** on
  T4 to strengthen low-parallax regions; pipeline MUST still pass without it.
- Wire depth/masks into the bundle; update the thin dense step to consume them.

**Compute placement:** ☁️ depth (full-res), masking (SAM2/RT-DETR), feed-forward geometry — heavy nets
on **T4**. 🖥️ a **Depth Anything V2 OpenVINO** path for local/offline runs and small scenes. Both write
depth/masks to the bundle identically.

**Deliverables:** per-keyframe metric depth + confidence; dynamic + semantic masks; (optional) pointmap
geometry; Colab/Kaggle notebooks for the cloud stages that sync via the bundle.

**Exit criteria**
- Metric depth is produced for real keyframes and **scale-aligned** to the spine (depth-vs-sparse
  residual measured & reported).
- Dynamic objects are masked out of the reconstruction inputs (visually verified on a real clip).
- The pipeline **succeeds with feed-forward geometry disabled** (proves it's optional) and, when
  enabled, fits in 16 GB via chunking.
- `../../AGENTS.md` updated.

**Risks:** aerial domain gap on depth (accuracy caveat, N1) — measure, don't assume; feed-forward
geometry OOM on 16 GB — keep chunked/optional.

---

## Phase 3 — Dense reconstruction & meshing (S6–S8)

**Goal:** Replace the thin coarse surface with a **high-quality dense reconstruction and textured
mesh**, carrying confidence and distinguishing measured vs inferred geometry.

**Scope / stages:** S6 (global optimization/BA refinement), S7 (dense: TSDF fusion / few-shot 3DGS),
S8 (meshing + texturing).

**Tasks**
- **S6:** global BA / pose-graph refinement over the metric spine (tighten before densification).
- **S7 dense (F12):** **Open3D** TSDF/point fusion from metric depth (masked); **gsplat** few-shot 3DGS
  with depth/normal/confidence regularization for view-consistent detail.
- **S8 mesh (F12–F13):** surface extraction — **2DGS/SuGaR** from the splat scene or **screened Poisson**
  (Open3D) — then **MVS-Texturing**/best-view texturing; compute **per-vertex confidence** and **flag
  inferred/completed** regions.
- Level-of-detail generation for downstream tiling.

**Compute placement:** ☁️ 3DGS training and any learned meshing on **T4** (tile large scenes to fit 16
GB). 🖥️ Open3D TSDF/Poisson meshing and texturing on CPU for smaller scenes / offline runs. Bundle
carries dense artifacts between them.

**Deliverables:** dense point cloud / 3DGS scene; watertight-where-measured textured mesh with
per-vertex confidence and an inferred-geometry flag layer; LOD tiles.

**Exit criteria**
- A **textured mesh** is produced from a real clip with visible fidelity improvement over Phase 1's
  coarse mesh (qualitative + coverage metric).
- **Confidence** and **inferred-vs-measured** flags are present in the mesh/point outputs.
- Large-scene path fits in **16 GB** via tiling (tested on the largest available real clip).
- `../../AGENTS.md` updated.

**Risks:** 3DGS memory/time on T4 — tile + cap iterations; texture seams — best-view + blending;
watertight vs honest holes — prefer honest holes flagged over fabricated fill (N8).

---

## Phase 4 — Georeferencing, semantics & exports (S9–S10)

**Goal:** Turn the reconstruction into **finished geospatial products** in **all required formats**,
fully georeferenced, plus semantic layers.

**Scope / stages:** S9 (georeference, DSM/DTM, orthomosaic, semantics), S10 (export).

**Tasks**
- **S9 (F14–F15):** georeference all products to the **GPS-derived CRS** (PROJ/pyproj, geoid handling);
  generate **DSM** and **DTM** (ground/non-ground separation) and an **orthomosaic** (GeoTIFF) via
  GDAL/rasterio/PDAL; render semantic layers as selectable overlays (GeoJSON/raster).
- **S10 (F16):** exporters for **OBJ · PLY · LAS(/LAZ) · GeoTIFF · glTF/GLB · FBX** and **3D Tiles**
  (py3dtiles) + Potree octree. **FBX via Blender headless** (`bpy`), **out-of-process** (GPL isolation).
- Validate every export against a standard reader; write CRS metadata into all geo products.

**Compute placement:** 🖥️ all (GDAL/PDAL/PROJ, py3dtiles, trimesh, Blender headless run on CPU). No
cloud needed.

**Deliverables:** DSM, DTM, orthomosaic (GeoTIFF); semantic layers; the full export set + 3D Tiles/Potree;
export validation report.

**Exit criteria**
- **All MUST formats** (F16) are produced from a real reconstruction and **open correctly** in QGIS,
  CloudCompare, and MeshLab, **correctly georeferenced**.
- DSM/DTM/orthomosaic have correct CRS and plausible geospatial extent (checked against the GPS track).
- FBX is produced without linking GPL into DRISHTI (out-of-process confirmed).
- `../../AGENTS.md` updated.

**Risks:** FBX toolchain fragility → isolate Blender; CRS/geoid mistakes → unit-test transforms against
known control; big rasters → tile with GDAL.

---

## Phase 5 — Accuracy report & web viewer (S10 + UI)

**Goal:** Make the results **inspectable and honest**: a measured accuracy/confidence report and an
interactive web viewer with measurement tools.

**Scope / stages:** S10 reporting + the viewer/serving UI.

**Tasks**
- **Accuracy report (F17, N8):** compute **measured** error vs. any held-out GPS/check points; per-region
  confidence; coverage/completeness; **measured wall-clock per stage per environment**; explicitly flag
  inferred geometry and any region exceeding the ≤ 1 m bar. If no check data exists, state accuracy is
  **unvalidated** (never invent a number). HTML + JSON.
- **Web viewer (F18):** **React (Vite) + CesiumJS + Potree**, loads the georeferenced model + 3D Tiles/
  octree; toggles semantic/confidence layers; **distance/area/volume/height** measurement tools.
- **FastAPI backend:** serve bundle artifacts + a measurement/report API; run locally.

**Compute placement:** 🖥️ all (React build, FastAPI, CesiumJS/Potree run locally in a browser).

**Deliverables:** an HTML+JSON accuracy report; a running web viewer + FastAPI server; a demo script.

**Exit criteria**
- The report shows **measured** accuracy (or an explicit "unvalidated"), per-region confidence, coverage,
  and **measured** timings labeled by environment — **no fabricated figures**.
- The viewer loads a **real** reconstruction georeferenced, toggles layers, and measures distance/area/
  volume/height correctly (checked against known dimensions where available).
- `../../AGENTS.md` updated.

**Risks:** large models in-browser → LOD/streaming via 3D Tiles/Potree; measurement accuracy in the
viewer must match the model's CRS/units (test).

---

## Phase 6 — Robustness, performance & production hardening

**Goal:** Make DRISHTI **dependable and efficient** on the target compute: graceful degradation, scale
to large scenes, speed tuning, full reproducibility, and packaging.

**Scope / stages:** cross-cutting (all).

**Tasks**
- **Reliability spine (N4):** implement the graceful-degradation ladder (L0–L6) — every stage has a
  fallback and emits a best-effort result + uncertainty rather than hard-failing.
- **Scale:** scene tiling/chunking + out-of-core handling so large clips run within 16 GB (cloud) and
  local RAM; multi-session Colab/Kaggle orchestration via the resumable bundle.
- **Performance (N3):** profile per stage; OpenVINO tuning on iGPU/NPU; batch/precision tuning on T4;
  cache/reuse; record **measured** improvements per environment.
- **Reproducibility (N5):** finalize pinned versions + checksums + seeds + run-manifest hashing; a
  `drishti verify` that re-checks a bundle's integrity.
- **Robustness tests:** degraded inputs (GPS gaps, motion blur, exposure swings, no IMU/RTK); confirm
  graceful degradation and honest reporting.
- **Packaging & docs:** finalize Docker images (local CPU/OpenVINO + cloud CUDA), Colab/Kaggle runners,
  quick-start, and an end-to-end demo on a real dataset.

**Compute placement:** 🖥️ + ☁️ (this phase spans both; the point is to make the split robust and fast).

**Deliverables:** degradation ladder implemented + tested; large-scene runs; a performance report with
**measured** per-environment timings; reproducibility tooling; finished packaging + demo.

**Exit criteria**
- Degraded/partial inputs produce a **best-effort model + honest uncertainty**, never a crash (tested
  across the L0–L6 scenarios).
- A large real scene completes within free-tier limits via tiling + resume.
- `drishti verify` confirms a rebuilt bundle matches manifest hashes (reproducibility).
- The performance report lists **measured** timings per environment; **no unearned "< 15 min" claim**.
- `../../AGENTS.md` status shows all phases ☑ with the decision log complete.

**Risks:** free-tier session limits vs large scenes → resume + tiling are essential; performance tuning
has diminishing returns → prioritize the stages the report shows are slowest.

---

## Cross-phase workstreams (run continuously)

| Workstream | Cadence | Notes |
|------------|---------|-------|
| **AGENTS.md upkeep** | every phase | Update §8 status + append §9 decisions — the contract for the next agent |
| **Testing** | every phase | pytest on real small clips; determinism (seeds+pins); no fabricated fixtures |
| **Honesty audit** | every phase | Grep for hardcoded params/fake numbers; confirm design-target vs measured labels |
| **License audit** | when adding a model/lib | Registry gate + ledger in [`../03-TECHNOLOGY-STACK.md`](../03-TECHNOLOGY-STACK.md) |
| **Docs sync** | when architecture shifts | Reconcile with [`../_internal/CANONICAL-ARCHITECTURE-SPEC.md`](../_internal/CANONICAL-ARCHITECTURE-SPEC.md); record deltas |

---

## Definition of done (project)

The PRD's [acceptance criteria](00-PRD.md#9-acceptance-criteria-definition-of-done-for-the-product) are
all met on a **real** dataset: one command → full georeferenced deliverable set + measured accuracy/
confidence report + web viewer, reproducible and offline, no hardcoded data, permissively licensed.

---

## Open questions / risks (plan-level)

- **Real dataset timing** — Phases 2–5 need a representative real clip to validate; without it we can
  build the machinery but not confirm accuracy/completeness. Acquire one early.
- **Free-tier variance** — Colab/Kaggle availability and limits vary; the resumable bundle + local
  fallbacks (OpenVINO depth, Open3D meshing) are the hedge.
- **Phase 3 is the risk peak** — dense/3DGS on 16 GB is where memory/time pressure concentrates; keep
  the Open3D-TSDF path as a reliable fallback to the 3DGS path.
- **Scope creep back toward "live"** — resist; this build is offline/ground-only by decision
  ([`../../AGENTS.md` §2](../../AGENTS.md)). Any change reconciles with the canonical spec first.
