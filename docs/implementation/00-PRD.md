# DRISHTI — Product Requirements Document (PRD)

**Purpose:** Define *what* DRISHTI must do and *how well*, for the build we are actually shipping:
**recorded drone footage + telemetry in → one full-quality georeferenced 3D model out, processed
entirely on the ground, offline.** This is the contract the implementation plan and system design build
against.

**Audience:** The build team and any AI agent implementing DRISHTI. For the evaluator-facing story see
[`../00-MASTER-OVERVIEW.md`](../00-MASTER-OVERVIEW.md); for architecture internals see
[`02-SYSTEM-DESIGN.md`](02-SYSTEM-DESIGN.md); for the authoritative pipeline see
[`../_internal/CANONICAL-ARCHITECTURE-SPEC.md`](../_internal/CANONICAL-ARCHITECTURE-SPEC.md).

**TL;DR**
- **Job to be done:** given a *pre-recorded* single-pass drone video plus its GPS track and flight
  metadata, produce a **georeferenced, metrically accurate, textured 3D model** and the full
  deliverable set (point cloud, DSM/DTM, orthomosaic, semantic layers, accuracy report, viewer).
- **Scope guardrails:** **offline and ground-only** — no on-drone compute, no Jetson/edge tier, no
  live/streaming path, **one** full-quality output (not live-preview-then-refine). See
  [`../../AGENTS.md` §2](../../AGENTS.md).
- **Accuracy goal:** **≤ 1 m** absolute (official). RTK/PPK inputs → few-cm design target; GPS-only
  baseline → sub-metre-to-≤ 1 m **design target**, with any region exceeding it **flagged, not hidden**.
- **Compute reality:** runs on an **Intel Core Ultra 7 155H (no dGPU) + free cloud T4**. Time targets
  are **reported per environment**; the official "< 15 min / 10-min video" is a GPU-class figure we do
  not claim on weak hardware.
- **Hard rules:** no hardcoded data / no stub outputs, honesty policy, confidence on every stage,
  permissive licenses only. (See [§7](#7-non-functional-requirements) and [`../../AGENTS.md` §5](../../AGENTS.md).)

---

## 1. Problem & product vision

Classical drone mapping requires multiple overlapping passes and hours of Structure-from-Motion +
Multi-View Stereo. The target scenario — disaster response, reconnaissance, rapid survey — gives **one
pass and a recording after the fact**. DRISHTI's job is to reconstruct **the whole scene, to metric
accuracy, without Ground Control Points**, from that single recording plus its GPS and flight metadata,
using **learned priors + sensor fusion** to compensate for the weak multi-view geometry a single pass
provides.

**Product vision (this build):** a reproducible, offline, ground-based pipeline and viewer that a user
points at a folder (video + telemetry) and gets back a complete, measurable, georeferenced 3D model
with an honest accuracy report — on commodity hardware with free cloud GPU, no proprietary services.

---

## 2. Scope

**In scope**
- Ingesting a **recorded** video (1080p/4K) plus **GPS track + flight metadata**, and the optional
  streams (IMU, barometric altitude, camera intrinsics, RTK/PPK) when the dataset provides them.
- **All processing on the ground, offline:** frame QA → poses + metric spine → depth/masking → dense
  reconstruction → mesh + texture → georeference → semantics → exports → accuracy report → web viewer.
- Producing the **full deliverable set** in the required formats (see [§6](#6-outputs--formats)).
- Running across a **local (Intel/OpenVINO) + free-cloud (T4)** split, offline-capable, reproducible.

**Out of scope (this build)**
- Any **on-drone / edge deployment** (Jetson, nvblox, in-flight compute) — descoped; see
  [`../../AGENTS.md` §2](../../AGENTS.md).
- Any **live / streaming / near-real-time** path or in-flight coverage HUD.
- A **two-path** (live-coarse + refine) design — collapsed to one offline full-quality pipeline.
- Flight planning, drone control, or capture hardware.
- Paid/managed cloud services or any mandatory network dependency at run time.

---

## 3. Users & use cases

| User | Goal | Primary artifacts |
|------|------|-------------------|
| Disaster-response / survey analyst | Measure a site (distances, areas, volumes, heights) from one pass | Textured mesh + DSM/DTM + orthomosaic in a viewer; measurement tools |
| GIS / mapping engineer | Bring the result into existing tools | LAS/LAZ, GeoTIFF, OBJ/PLY, 3D Tiles with correct CRS |
| NTRO / SIH evaluator | Verify accuracy, completeness, speed, and honesty | Accuracy report + confidence overlays + measured timings |
| DRISHTI developer / AI agent | Reproduce and extend a run | Project bundle + run manifest + config |

**Representative use case (end to end):** an analyst receives a 10-minute nadir/oblique clip of a
flood-affected area with a DJI SRT GPS track. They run `drishti run --dataset flood_site.yaml`. The
pipeline (local + one Colab session for heavy stages) returns a georeferenced textured mesh, a point
cloud, a DSM/DTM, an orthomosaic, building/vegetation/water semantic layers, and an accuracy report
that states measured RMSE against GPS check points, flags two low-confidence occluded regions, and
lists the actual wall-clock time per stage. They open the web viewer and measure a levee's height.

---

## 4. Assumptions & dependencies

- The **dataset is provided** (at the event) as files: video + GPS + flight metadata, optionally
  IMU/baro/intrinsics/RTK-PPK. Until then, development uses a **representative real clip** — never
  synthetic stand-ins (that would violate the no-hardcode/live rule).
- **Compute:** Intel Core Ultra 7 155H (16 cores, Arc iGPU, NPU), 16 GB+ RAM, **no dGPU**; free
  **Colab/Kaggle T4 (~16 GB)** for heavy neural stages. Must also run fully offline once models are
  cached locally.
- Camera intrinsics are **self-calibrated by default**; provided intrinsics/RTK only tighten the solve.
- GPS provides the metric/georeference anchor; **no GCPs** are assumed.

---

## 5. Functional requirements

Requirements are **MUST / SHOULD / MAY** (RFC-2119 sense). Each maps to a pipeline stage (S0–S10) from
the canonical spec, realized offline.

### 5.1 Input & ingest (S0–S1)
- **F1 (MUST)** Accept a recorded video file (common codecs; 1080p/4K) and decode it deterministically.
- **F2 (MUST)** Parse GPS + flight metadata from real telemetry formats via pluggable adapters:
  **DJI SRT**, generic **CSV**, **MAVLink** (`.tlog`/`.bin`), and EXIF; **fail loudly** on missing
  mandatory fields (no placeholder substitution).
- **F3 (MUST)** Time-align frames to telemetry timestamps (no hardware clock; use logged timestamps).
- **F4 (MUST)** Frame QA: reject blurred/over/under-exposed frames; select keyframes by
  parallax/coverage. Every dropped frame is logged with a reason.
- **F5 (SHOULD)** Ingest optional IMU/baro/intrinsics/RTK-PPK when present and record their presence in
  the bundle so downstream stages can use them.

### 5.2 Poses & metric spine (S2, S6)
- **F6 (MUST)** Estimate camera poses via SfM (self-calibrated intrinsics).
- **F7 (MUST)** Build a **metric spine**: a factor graph fusing visual structure + GNSS (and IMU/RTK
  when present) to recover **metric scale and georeference without GCPs**; align to a geographic CRS via
  Umeyama/Sim(3).
- **F8 (MUST)** Derive the **CRS/EPSG from the GPS track** (UTM zone from median lon/lat) — never
  hardcoded.

### 5.3 Perception, depth & geometry (S3–S4)
- **F9 (MUST)** Mask dynamic objects and (SHOULD) produce semantic labels
  (ground/building/vegetation/water/vehicle).
- **F10 (MUST)** Produce **metric depth** per keyframe, scale-aligned to the metric spine.
- **F11 (MAY)** Use feed-forward geometry (MapAnything/Pi3) as an enhancement when it fits in 16 GB;
  the pipeline MUST succeed without it.

### 5.4 Dense reconstruction, mesh & texture (S7–S8)
- **F12 (MUST)** Produce a dense representation (TSDF/point fusion and/or few-shot 3DGS) and a
  **textured mesh**.
- **F13 (MUST)** Carry **per-vertex/per-point confidence**; mark **inferred (completed) geometry**
  distinctly from measured geometry.

### 5.5 Georeference, semantics & products (S9)
- **F14 (MUST)** Georeference all products to the derived CRS; produce **DSM/DTM** and an
  **orthomosaic** (GeoTIFF).
- **F15 (SHOULD)** Produce semantic layers as selectable overlays.

### 5.6 Export, report & viewer (S10 + UI)
- **F16 (MUST)** Export the required set: **OBJ · PLY · LAS · GeoTIFF · glTF/GLB · FBX** (FBX via
  Blender headless, out-of-process). SHOULD also export LAZ and OGC **3D Tiles**.
- **F17 (MUST)** Produce an **accuracy report**: measured error vs. available check data, per-region
  confidence, coverage/completeness, and **measured wall-clock time per stage and per environment**.
- **F18 (MUST)** Provide a **web viewer** (React + CesiumJS + Potree) that loads the model
  georeferenced, toggles layers/confidence, and supports **distance/area/volume/height** measurement.
- **F19 (MUST)** Emit a **project bundle** with a run manifest (input hashes, model versions, config,
  seeds) so any run is **reproducible and resumable**.

---

## 6. Outputs & formats

| Product | Format(s) | Requirement |
|---------|-----------|-------------|
| Textured mesh | **OBJ**, **glTF/GLB**, **FBX** | MUST |
| Point cloud | **PLY**, **LAS** (SHOULD: LAZ) | MUST |
| Raster geo products | **GeoTIFF** (DSM, DTM, orthomosaic) | MUST |
| Streamable tiles | OGC **3D Tiles**, Potree octree | SHOULD |
| 3D Gaussian scene | `.ply`/`.splat` | MAY |
| Semantic layers | GeoJSON / raster masks | SHOULD |
| Accuracy & confidence report | HTML + JSON | MUST |
| Project bundle + manifest | directory + JSON | MUST |

All georeferenced products MUST carry correct CRS metadata derived from the GPS track.

---

## 7. Non-functional requirements

| # | Requirement | Target |
|---|-------------|--------|
| **N1 Accuracy** | Absolute spatial accuracy (official ≤ 1 m). RTK/PPK → few-cm *design target*; GPS-only → sub-metre-to-≤ 1 m *design target*. Regions exceeding the bar are **flagged, never averaged away**. | Design target until measured on a real dataset |
| **N2 Completeness** | Reconstruct the entire visible scene; report coverage %; inferred fills flagged. | Measured & reported |
| **N3 Performance** | **Measured wall-clock per stage, per environment.** Official "< 15 min / 10-min video" is a GPU-class target and is **not claimed** on Intel-iGPU/free-T4. | Honest, environment-labeled |
| **N4 Reliability** | Never hard-fail: graceful-degradation ladder (L0–L6); always emit a best-effort model + uncertainty report. | MUST |
| **N5 Reproducibility** | Deterministic given the same inputs+config: pinned versions + checksums + seeds + run manifest. | MUST |
| **N6 Offline-capable** | No mandatory network at run time once models are cached; no proprietary services. | MUST |
| **N7 Licensing** | Shipped path uses **permissive licenses only**; NC/military-excluded models are reference-only. | MUST |
| **N8 Honesty** | No fabricated numbers; label design-target vs measured; flag inferred geometry; report failures plainly. | MUST |
| **N9 Compute fit** | Every stage runs within **16 GB VRAM** (cloud) or on CPU/OpenVINO (local); larger scenes tile/chunk. | MUST |
| **N10 Usability** | A single command runs the pipeline over a dataset descriptor; the viewer runs in a browser. | SHOULD |

---

## 8. Success metrics (tied to the official evaluation weights)

| Evaluation criterion (weight) | How DRISHTI targets it | How we measure it |
|-------------------------------|------------------------|-------------------|
| **Accuracy (30%)** | Metric spine + GPS georeference + confidence | RMSE vs. held-out GPS/check points, per region, in the report |
| **Completeness (20%)** | Dense fusion + 3DGS + flagged completion | Coverage % of visible scene; hole statistics |
| **Speed (20%)** | Tiered local+cloud, resumable stages | Measured wall-clock per stage per environment |
| **Innovation (15%)** | Prior-assisted single-pass metric reconstruction without GCPs | Documented method + ablations |
| **Scalability (10%)** | Artifact-tiered stages; tiling for large scenes | Runs on free tier; scales to a workstation unchanged |
| **UI (5%)** | Web viewer + measurement + confidence overlays | Demo + usability of the viewer |

---

## 9. Acceptance criteria (definition of done for the product)

1. `drishti run --dataset <descriptor>` takes a **real** recorded clip + telemetry and produces a valid
   **project bundle** with **all MUST outputs** ([§6](#6-outputs--formats)) and a run manifest.
2. Outputs are **georeferenced** in a CRS **derived from the GPS track**, and open correctly in a
   standard GIS/3D tool (QGIS/CloudCompare/MeshLab) and in the DRISHTI web viewer.
3. The **accuracy report** shows **measured** error and per-region confidence, flags inferred geometry,
   and lists **measured** per-stage timings labeled by environment.
4. The run is **reproducible** (same inputs+config → same manifest hashes) and **offline** once models
   are cached.
5. **No hardcoded data, no stub outputs** anywhere in the delivered pipeline; missing required inputs
   **fail loudly**.
6. Every shipped component is **permissively licensed**.

---

## 10. Open questions / risks

- **Aerial domain gap** — depth/geometry priors are largely ground-trained; ≤ 1 m on aerial video is a
  **design target until measured**; may need calibration/fine-tuning.
- **Free-tier limits** — Colab/Kaggle T4 time/memory caps may force tiling and multi-session runs for
  large scenes; end-to-end time will exceed the GPU-workstation target — report honestly.
- **Check data for accuracy** — validating N1 needs held-out GPS/check points; if the provided dataset
  lacks them, the report states accuracy is **unvalidated** rather than inventing a number.
- **FBX** — emitted via **Blender headless (GPL, isolated process)**; confirm it never links into DRISHTI.
- **Dataset availability before the event** — need a representative real clip to develop against;
  otherwise Phases 2–5 cannot be validated on real data.
