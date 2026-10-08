# DRISHTI — Build Completion & Compromise Audit (A→Z)

> **Historical audit (2026-09-22):** cited code and measurements belong to an
> absent checkout. This audit informed the new implementation; it is not current
> verification. See [build/STATUS.md](build/STATUS.md) for measured results here.

**What this is.** A single, honest, evidence-backed answer to two questions:
1. **Where has the delivered system been compromised** relative to what the docs claim?
2. **What still needs to be built** to make the claims true — laid out as one dependency-ordered plan you can execute in a single pass.

**How it was produced.** Full re-read of the authoritative docs (`AGENTS.md`, `docs/00-MASTER-OVERVIEW.md`,
`docs/implementation/00-PRD.md`, `.../01-IMPLEMENTATION-PLAN.md`, `docs/GUIDE.md §16–18`) and a line-level read of
the whole `src/drishti/` tree, `server/`, `viewer/`, `configs/`, `docker/`, CI, and `tests/`. Every claim below is
cited to `file:line` and, where possible, **measured**, not asserted.

**Date:** 2026-09-22 · **HEAD:** `117b426` · **Scope contract:** ground-only, offline, footage-in, one full-quality
output (`AGENTS.md §2`, `PRD §2`) — this audit judges the code against *that* contract, not the aspirational
three-tier vision in `00-MASTER-OVERVIEW.md`.

> **A note on the two documents.** `00-MASTER-OVERVIEW.md` describes a three-tier live+refine system (Jetson/edge,
> RTK cm-accuracy, nvblox, TensorRT, two output paths). `AGENTS.md §2` and `PRD §2` explicitly descope all of that.
> Most "overclaims" are really *the overview describing a different product than the one being built.* This audit
> treats the PRD's F1–F19 / N1–N10 as the real contract. Wherever the overview and the PRD disagree, that gap is
> itself flagged in §10.

---

## 0. Verification basis (measured this session, not claimed)

| Check | Command | Result |
|---|---|---|
| Unit tests | `.venv\Scripts\python -m pytest -q` | **127 passed, 2 warnings, 46.3s** ✅ |
| `test_server` collection | (part of the run above) | **Runs clean** — no collection error (fastapi+httpx are installed in `.venv`). The "pre-existing test_server collection error" in `AGENTS.md §8` is **stale** — correct it. |
| Lint | `.venv\Scripts\python -m ruff check src tests server` | **3 errors, exit 1 — NOT clean** ❌ |
| CI parity | CI runs the exact same `ruff check src tests server` (`.github/workflows/ci.yml:42`) | **`main` is currently CI-red.** |

**The 3 ruff errors (introduced by the last commit `117b426`, never linted):**
- `src/drishti/stages/s9_geo.py:44` — `I001` import block un-sorted (auto-fixable).
- `tests/test_geo_raster.py:83` — `B905` `zip()` without `strict=`.
- `tests/test_geo_raster.py:115` — `B905` `zip()` without `strict=`.

So the `AGENTS.md §8` status line "**127 tests passing, ruff clean**" is **half true**: tests pass, ruff does not.
This is the smallest, most immediate compromise and the first thing to fix (§9, Phase A).

---

## 1. Verdict in one paragraph

**The skeleton is real and unusually honest; the muscles are missing or faked.** The pipeline *machinery* —
config, bundle/manifest, stage runner, resumability, content-hash freshness, the CLI, the API server, the license
gate, the graceful-degradation ladder, the accuracy-report builder, the web viewer's run-inspector — is genuinely
built, tested, and mostly does what it says. But **the load-bearing reconstruction claims are compromised in three
different ways**: (a) three stages **fabricate capability signals** (S2 "factor-graph spine", S6 "global bundle
adjustment", S9 "PDAL SMRF") — they toggle honesty flags / relabel / raise confidence for methods that never run;
(b) several stages carry **silent correctness bugs** that corrupt metric output without flagging it (the depth
domain mismatch is the worst); and (c) the headline **compute story ("local on Intel via OpenVINO") is
unimplemented** — no stage ever selects a backend, and no model ever loads through OpenVINO. Accuracy (N1) is, by
the report's own honest admission, **unvalidated**. The good news: a large fraction of the debt is *"make the label
honest"* (cheap) rather than *"build the missing engine"* (expensive), and one stage — S3 — already demonstrates
the correct honest pattern the others should copy.

---

## 2. What is genuinely built (give it its due)

These are real, run, and tested. Do **not** rewrite them; build on them.

| Area | Evidence | Status |
|---|---|---|
| **Config system** — typed Pydantic, `extra="forbid"` (typos fail loudly), full YAML tree + profiles + `--set` overrides + precedence | `config/models.py`, `configs/default.yaml`, `config/loader.py` | Real, tested (`test_config_loader.py`) |
| **Bundle + manifest** — provenance, input hashes, per-stage records, seeds, "measured-only" metrics | `bundle/manifest.py`, `bundle/bundle.py` | Real, tested (`test_bundle.py`) |
| **Stage runner** — dependency order, content-hash freshness skip, per-stage timing, persists manifest after every stage (resumable), fails loudly + records the real error | `stages/runner.py:79-166` | Real |
| **CLI** — `run / resume / inspect / verify / doctor / stages`; honest pre-flight that warns which stages will fail here | `cli.py` | Real, tested (`test_cli.py`) |
| **`verify` integrity check** — recomputes output hashes vs the manifest | `cli.py:330-375` | Real (checks *recorded* integrity, not cross-run determinism — see §3 Tier 4) |
| **API server** — thin, honest layer over the same core; background run thread; manifest is source of truth; path-traversal guarded; serves artifacts + report; serves built SPA | `server/app.py` | Real, tested (`test_server.py`, 8 tests) |
| **License gate** — permissive allowlist; reference-only / NC / military-excluded models refused unless explicitly overridden | `models/registry.py:105-136` | **Genuine strength** (N7) |
| **Graceful-degradation ladder** — L0–L6 caps, `choose()`/`tier()`/`Decision.blend()` | `runtime/reliability.py` | Real — but wired into **only one stage** (§3 Tier 3) |
| **Accuracy report** — summarizes strictly measured manifest data; labels accuracy "design proxy"; says **UNVALIDATED** when no check-points | `report/report.py` | **Genuine strength** (N8) |
| **CRS derivation** — UTM zone from median GPS lon/lat, never hardcoded | S2/`geo/crs.py` (F8) | Real |
| **S7 tiled TSDF fusion** — chunked to bound host RAM; the 5 cm `dense.ply` is preserved | `stages/s7_dense.py`, `dense.tile.*` | Real |
| **Web viewer (as a run inspector)** — React+Vite+TS, polls the API, renders the 3D-Tiles tileset fully offline (no Ion token), honest empty-states, download links | `viewer/src/*` | Real — but **not** the F18 viewer (§5) |
| **Compute detection** — probes torch.cuda, OpenVINO devices, RAM; never fabricates | `runtime/compute.py:51-95` | Real |
| **Offline capability** — guarded imports, local model cache, no mandatory network; viewer offline | throughout | Real (N6) |

---

## 3. The compromises, A→Z (the honest debt)

Ordered by severity. Each is tagged **[FAKE]** (fabricated signal / honesty violation), **[BUG]** (silently wrong
output), **[GAP]** (claimed capability not implemented), or **[DEBT]** (real but degraded/risky), with the fix.

### Tier 1 — Honesty-policy violations: fabricated capability signals

These violate `AGENTS.md §5 / PRD N8` directly, because they make the manifest *assert* a better method than ran.

- **[FAKE] S2 "metric spine / factor graph" is a plain Sim(3) GPS fit.**
  `s2_poses.py:128-132`: the code comments "optional GTSAM tightening", sets `degraded = None`, and only writes a
  degraded reason **when gtsam+IMU are absent**. When they're *present*, the flag is cleared — implying the
  factor-graph tightening happened — **but there is no gtsam code anywhere** (`grep gtsam src/` → only a doctor
  string, a deps mapping, and this flag toggle). `spine.json` always records `"align": "umeyama_sim3"`
  (`s2_poses.py:117`). **F7 ("factor graph fusing visual + GNSS + IMU/RTK") is not met.**
  *Fix (honest, cheap):* delete the gtsam branch; state plainly the spine is a robust Sim(3) GPS alignment.
  *Fix (real, expensive):* actually build the GTSAM factor graph and fuse IMU/RTK.

- **[FAKE] S6 "global bundle adjustment" never runs; presence of a binary alone raises confidence.**
  `s6_global.py:58` `glomap = shutil.which("glomap")` is the only glomap reference; poses are written back
  unchanged; `s6_global.py:61` clears `degraded` and `s6_global.py:72` sets `conf = 0.9 if glomap else 0.75`. So
  merely having the `glomap` binary on `PATH` bumps confidence and de-flags degradation for a **byte-identical
  passthrough**. **F6/F7 "global optimization" is not met.**
  *Fix (honest):* confidence must not depend on an uninvoked binary; keep S6 a labelled passthrough until GLOMAP is
  actually called. *Fix (real):* invoke `glomap mapper` and read its refined poses back.

- **[FAKE] S9 DTM is labelled `pdal_smrf` while always computing the morphological stand-in.**
  `s9_geo.py:134` `used_smrf = available("pdal")` (import check only); `s9_geo.py:160,170` write
  `"dtm_method": "pdal_smrf" if used_smrf else "morphological_opening"`. `geo/raster.py:155-156` openly admits its
  filter is "a simple, dependency-free stand-in for SMRF". There is **no `import pdal`, no PDAL pipeline** — so if
  `pdal` merely *imports*, the manifest **mislabels the method**. **F14 DTM provenance is falsified.**
  *Fix (honest):* label it `morphological_opening` unless a real PDAL pipeline ran. *Fix (real):* run a PDAL SMRF
  pipeline.

> **Contrast — S3 is the honest template.** `s3_masking.py:56-65` does the *opposite* of the three above: it
> declares the preferred SAM2 tier `available=False` with the note "SAM2 pixel refinement is not built in this
> pipeline version", falls to the RT-DETR box tier (L2), and caps confidence at that tier's ceiling. **Every faked
> stage above should be rewritten to look like S3.**

### Tier 2 — Silent correctness bugs: wrong numbers, not flagged

These produce a real output that is quietly wrong. They are the accuracy-critical fixes.

- **[BUG] Depth domain mismatch corrupts all metric depth (worst bug in the tree).**
  `models/depth.py:40` returns Depth-Anything-V2 **affine-invariant *inverse* depth**; `s4_depth.py:79-80` fits
  `metric ≈ a·rel + b` against true metric Z. Fitting a *metric depth* model against an *inverse-depth* predictor
  yields a wrong (often negative-slope) affine solve; near-field is then deleted by `metric[metric<=0]=0`.
  **F10 metric depth is systematically corrupted.** *Fix:* invert DA-V2 output to a depth proxy (`1/(disp+eps)`)
  before the affine fit, or switch to a genuine metric model and drop the fit.

- **[BUG] Degraded depth frames are fused as if trustworthy.**
  S4 flags degraded frames, but `s7_dense.py:61-75` never checks `d["degraded"]` before fusing them into the TSDF.
  Bad depth silently pollutes the dense cloud. *Fix:* skip or down-weight degraded frames in S7.

- **[BUG] S8 forces world-up normals on building-dense scenes, mis-orienting façades.**
  `s8_mesh.py:105-110` uses `orient_normals_to_align_with_direction([0,0,1])` above
  `mesh.consistent_normals_max_points` (5 M). Correct for nadir 2.5-D terrain, **wrong for vertical building
  walls** — a real accuracy loss on exactly the town/aerial datasets in `configs/datasets/`. *Fix:* only apply the
  up-prior to near-ground points; use a scalable consistent-orientation method for façades. (The point downsample
  to octree-leaf and the untouched 5 cm `dense.ply` are fine — not a quality loss.)

- **[BUG] DTM claims full coverage: empty cells filled with scene-max, no nodata mask.**
  S9 fills empty DTM cells with the scene maximum instead of emitting nodata, so unmeasured area is reported as
  covered terrain. Violates N2 (coverage honesty). *Fix:* emit a real nodata value + a measured/inferred coverage
  mask.

- **[BUG] S2 confidence saturates to 1.0 on overfit.**
  `s2_poses.py:135` `conf = min(1, gnss_sigma_m/(fit_rmse+1e-6))` → a few-correspondence fit with tiny RMSE reports
  confidence **1.0**. The Sim(3) alignment (`geo/align.py:33-65`) is plain least-squares (no RANSAC), so a handful
  of GPS points overfit and then self-report as perfect. *Fix:* RANSAC + penalize low correspondence count.

- **[BUG] `vertical: orthometric` is accepted but no geoid is ever applied.**
  `config/models.py:75` accepts `orthometric`, `manifest.py:29` records it, but heights stay ellipsoidal — a
  label-only ~tens-of-metres error if a user selects it. *Fix:* apply a geoid model, or reject `orthometric` as
  unimplemented.

### Tier 3 — Architecture-vs-claim gaps: the compute story

- **[GAP] "Local inference on Intel via OpenVINO" is unimplemented.** OpenVINO is detected
  (`compute.py:76-84`), advertised by `doctor`, declared per-stage (`openvino_ok=True` in S3/S4 `ComputeNeed`), and
  a backend string `"openvino"` *can* be computed (`compute.py:116-121`) — **but no stage ever calls
  `rt.backend()`** (`grep '\.backend('` → definition + export only, **zero call sites in `stages/`**), and
  `models/loaders.py:36-38 torch_device()` returns only `cuda`/`cpu`. **Local neural stages run on CPU torch, not
  the iGPU/NPU.** The whole compute narrative in `PRD §4 / N9` and the overview is aspirational.
  *Fix (honest):* stop advertising OpenVINO acceleration. *Fix (real):* add an OpenVINO load/infer path in
  `models/` and consume `rt.backend()` in the runner.

- **[GAP] The graceful-degradation ladder is wired into exactly one stage.** `choose`/`tier` appear only in
  `s3_masking.py`. `N4` says the ladder governs "every heavy stage"; S0/S1/S2/S4/S6/S7/S8/S9/S10 instead use ad-hoc
  confidence floats + free-text `degraded_reason` that is **not** capped by any ceiling. *Fix:* give each heavy
  stage a real tier list and route confidence through `Decision.blend()`.

- **[DEBT] "Placement (local vs cloud-t4)" is a manifest *label*, not a scheduler.** `runner.py:106`
  `placement = rt.place(stage.compute)` is recorded as `environment`, but `stage.run()` executes in-process
  regardless. The local↔cloud split is manual (notebooks). So `environment: "cloud-t4"` in a local run's manifest
  is advisory. *Fix:* document it as advisory, or implement real dispatch.

### Tier 4 — Reproducibility gaps (N5 / F19)

- **[GAP] No model version/checksum pinning, despite N5 "pinned versions + checksums".** Every `ModelSpec`
  defaults `version="unpinned"`, `sha256=None` (`registry.py:39-42`) and **none of the ~20 entries set either**.
  A HuggingFace checkpoint update silently changes results. *Fix:* pin `version` + `sha256` per shipped model and
  verify on load.
- **[GAP] The manifest records no model versions.** `F19` requires "model versions" in the manifest;
  `manifest.py` has fields for input/param/output hashes, seeds, and config — **but no model-version field**.
  *Fix:* add resolved model versions/hashes to each `StageRecord`.
- **[DEBT] `verify` proves *recorded* integrity, not re-run determinism.** `cli.py:330-375` recomputes hashes vs
  what this bundle stored — it does not prove a fresh run reproduces them. With unpinned weights, cross-run
  determinism (acceptance criterion 4) is not actually guaranteed.

### Tier 5 — Config-as-aspiration: dead / unimplemented knobs

`configs/default.yaml` is disciplined ("no magic numbers") but a large share of its surface drives capabilities
that don't exist. These knobs load and validate, then feed nothing (or feed a stand-in):

| Knob (`configs/default.yaml`) | Reality |
|---|---|
| `poses.sfm_backend`, `poses.use_glomap`, `poses.matcher`, `poses.min_track_len` | not consumed / GLOMAP never invoked |
| `spine.gnss_sigma_v_m`, `spine.use_baro`, `spine.align` | dead (no factor graph; baro never read) |
| `masking.segmenter: sam2`, `masking.tracker: bytetrack`, `masking.motion: raft`, `masking.semantic_classes` | **config-only** — only RT-DETR box masking runs; **no semantic labels produced** |
| `depth.metric_refiner: metric3d_v2`, `depth.scale_align`, `depth.input_long_side` | metric3d never used; knobs ignored |
| `depth.feedforward_geometry.*` | unwired (MAY / off — acceptable, but model not integrated) |
| `dense.method: gaussian` + `dense.gaussian.*` | `gaussian` raises RuntimeError (only `tsdf` works) |
| `mesh.method: 2dgs`/`sugar` | raise RuntimeError (only `poisson` works) |
| **`mesh.texture: mvs_texturing`** (the **default**) | **not implemented → every default run marks S8 degraded** (`config/models.py:151`, `default.yaml:97`) |
| `geo.ground_classifier: pdal_smrf` (default) | probe-only label; always morphological (Tier 1) |
| `export.also: [laz, 3dtiles, potree]` | conditional on external binaries (`PotreeConverter`, `py3dtiles`) |

*Fix:* for each row, either implement it or change the default to the method that actually runs (e.g.
`mesh.texture: best_view`) and mark the rest "planned".

### Tier 6 — Unwired optional inputs (F5)

- **[GAP] IMU / baro / intrinsics / RTK are registered and hashed but never consumed.** `s0_ingest.py:54`
  records their presence; nothing downstream reads them (intrinsics "used" = COLMAP self-calibration, not the
  user's). So `F5` ("so downstream stages can use them") is half-met: presence is recorded, use is not.
- **[DEBT] Time-alignment is "assume both start at zero" + a manual `time_offset_s`** (default 0.0); no
  cross-correlation (`F3`). Silent fallbacks: `dji_srt.py:66` rel-alt-or-0.0; `exif.py:47` uses record index as
  seconds. *Fix:* auto time-sync; make missing altitude/time loud, not silent.

---

## 4. Declared-but-unimplemented catalogue ("in name only")

Everything the docs/registry/config name that has **no working implementation**. This is the concrete "what else
needs to be made" list.

| Capability | Declared where | Reality |
|---|---|---|
| **SAM2** pixel-accurate masking | registry, `masking.segmenter`, S3 L0 tier | Not built (S3 honestly falls back to boxes) |
| **ByteTrack** temporal tracking | registry, `masking.tracker` | Not built |
| **RAFT** residual-motion check | registry, `masking.motion` | Not built |
| **Semantic labels** (ground/building/veg/water) | `F9 SHOULD`, `F15`, `masking.semantic_classes` | **Not produced at all** |
| **Metric3D v2** metric refiner | registry (primary), `depth.metric_refiner` | Never loaded/used |
| **Feed-forward geometry** (MapAnything / Pi3) | `F11 MAY`, `depth.feedforward_geometry` | Unwired (license-gated, off) |
| **Gaussian / 2DGS / SuGaR** dense+mesh | `dense.method`, `mesh.method` | `RuntimeError`; `max` profile crashes |
| **MVS-Texturing** (the default texture) | `mesh.texture` default | Not built → S8 degraded every default run |
| **GTSAM factor-graph spine** | `F7`, `spine.*` | Not built (Sim(3) only) |
| **GLOMAP global BA** | `F6/F7`, S6 | Never invoked (passthrough) |
| **PDAL SMRF** ground filter | `F14`, `geo.ground_classifier` | Never invoked (morphological stand-in) |
| **OpenVINO** local acceleration | `N9`, `compute.prefer_openvino`, per-stage `openvino_ok` | No inference path; backend never consumed |
| **Potree** in-browser point cloud | `F18` | Not integrated (download-only, viewer admits it) |
| **Viewer measurement** (distance/area/volume/height) | `F18` | **Absent** |
| **Viewer layer / confidence toggles** | `F18` | **Absent** |
| **3D Tiles / LAZ export** | `F16 SHOULD` | Conditional on absent external tools |
| **FBX export** | `F16 MUST` | Real, but needs a Blender ≥4.0 binary → absent ⇒ S10 degraded (fbx ∈ MUST) |
| **Geoid / orthometric heights** | `crs.vertical` | Label-only |

---

## 5. Viewer reality (F18 is the largest single UI gap)

`F18 (MUST)` = "React + CesiumJS + **Potree**, loads model georeferenced, **toggles layers/confidence**, supports
**distance/area/volume/height** measurement."

**What exists** (`viewer/src/`): a real React+Vite+TS app that lists runs, polls the API every 5 s, shows a
per-stage status/confidence table, opens the accuracy report, renders the **3D-Tiles** tileset offline
(`CesiumView.tsx`), and offers downloads. `package-lock.json` exists, so CI's `build-viewer` job actually builds
it.

**What's missing vs F18:**
- **Potree** — no dependency (`viewer/package.json`); point cloud is download-only (`CesiumView.tsx:6-9` says so).
- **Measurement tools** — none anywhere (no distance/area/volume/height handlers). This is the core of F18.
- **Layer / confidence overlays/toggles** — none; confidence is a per-stage table, not an on-model overlay.

So the viewer is an **honest run-inspector**, not the **measurement viewer** F18 specifies. Either build the
measurement + Potree + layers, or amend F18 to describe the inspector that exists.

---

## 6. Requirement scorecard (PRD F1–F19, N1–N10)

Legend: ✅ built · 🟡 partial/compromised · ❌ not built · (—) acceptable as-is.

| Req | Summary | State | Note |
|---|---|---|---|
| F1 | Decode video deterministically | ✅ | S0 real |
| F2 | Parse DJI SRT/CSV/MAVLink/EXIF, fail loudly | 🟡 | silent alt/time fallbacks (Tier 6) |
| F3 | Time-align frames↔telemetry | 🟡 | zero-start + manual offset only |
| F4 | Frame QA + keyframe by parallax | ✅ | QA constants hardcoded (minor) |
| F5 | Ingest optional IMU/baro/intrinsics/RTK + record | 🟡 | recorded, never consumed |
| F6 | SfM poses (self-calib) | ✅ | pycolmap |
| F7 | **Metric spine (factor graph, GNSS+IMU/RTK)** | 🟡 | **Sim(3) only; GTSAM faked** |
| F8 | Derive CRS/EPSG from GPS | ✅ | genuine |
| F9 | Mask dynamic objects + (SHOULD) semantics | 🟡 | RT-DETR boxes only; no semantics |
| F10 | **Metric depth per keyframe** | 🟡 | **inverse-depth bug corrupts scale** |
| F11 | Feed-forward geometry (MAY) | ❌ (—) | optional; unwired |
| F12 | Dense + **textured mesh** | 🟡 | dense ✅; texture default unbuilt ⇒ degraded |
| F13 | Per-vertex/point confidence; mark inferred | 🟡 | density-crop proxy; verify per-point emit |
| F14 | Georeference + DSM/DTM/ortho GeoTIFF | 🟡 | real rasters; DTM coverage overclaim + faked label |
| F15 | Semantic layers as overlays (SHOULD) | ❌ | not produced |
| F16 | Export OBJ·PLY·LAS·GeoTIFF·GLB·**FBX** | 🟡 | FBX needs Blender; 3DTiles/LAZ conditional; LAS whole-cloud RAM |
| F17 | Accuracy report (measured error, per-region conf, timings) | 🟡 | honest, but error UNVALIDATED; conf per-stage not per-region |
| F18 | **Web viewer + measurement + layers + Potree** | 🟡 | inspector only; no measurement/Potree/layers |
| F19 | Bundle + manifest reproducible+resumable | 🟡 | strong, but **no model versions/checksums** |
| N1 | Accuracy ≤1 m | ❌ (—) | **UNVALIDATED** by honest design |
| N2 | Completeness + coverage% + inferred flags | 🟡 | DTM coverage overclaim |
| N3 | Measured wall-clock per stage/env | ✅ | env = placement label |
| N4 | Reliability ladder L0–L6 | 🟡 | wired only in S3 |
| N5 | Reproducibility (pins+checksums+seeds+manifest) | 🟡 | seeds/config/inputs ✅; **versions/checksums ❌** |
| N6 | Offline-capable | ✅ | genuine |
| N7 | Permissive licenses only | ✅ | **genuine strength** |
| N8 | Honesty | 🟡 | report/manifest honest; **S2/S6/S9 + S3-confidence violate it** |
| N9 | Compute fit 16 GB / CPU / tiling | 🟡 | S7 tiling ✅; S10 LAS ~22 GB RAM ❌ |
| N10 | Single command + browser viewer | ✅ | genuine |

**Tally:** ✅ 8 · 🟡 15 · ❌ 4 (2 of the ❌ are acceptable-as-is). No MUST is *fully* met beyond the machinery
layer without a caveat — which is exactly the gap between "the spine runs" and "the product is done".

---

## 7. Test & CI reality

- **127 tests pass** — but they cover the **pure/core** layer: config, bundle, reliability, telemetry parsing, geo
  align/CRS/raster math, depthfit math, pointcloud, tiled-TSDF math, voxel mesh math, report builder, CLI wiring,
  server smoke. `tests/` has **no `__init__.py`** and there's no repo-root `conftest.py` (fine; they import the
  installed package).
- **What the 127 do *not* prove:** there is **no end-to-end pipeline test** — no `s0→report` run on even a tiny
  fixture. Every heavy stage (S3/S4/S7/S8/S9/S10 real execution) is **unexercised by CI** (`ci.yml` installs only
  `.[dev,server]`, never `video/geo/recon/poses/depth`). The heavy path is validated only manually via notebooks.
- **CI is currently red** on lint (§0). The `build-viewer` job will run (`package-lock.json` present) but the
  `lint-and-test` job fails at `ruff check`.

*Fix:* (a) clear the 3 ruff errors; (b) add one tiny synthetic end-to-end smoke test (a few frames + a short GPS
track → assert a bundle with all MUST outputs) gated behind the heavy extras, run on the cloud tier or a nightly.

---

## 8. Accuracy reality (read this before quoting any number)

The system **never validates N1 (≤1 m)** in a normal run. `report/report.py:36-46` reports the S2 spine-fit RMSE
**explicitly labelled "GPS-alignment fit residual (design proxy, not an independent check)"**, and if no
check-points are supplied it prints **"absolute accuracy is UNVALIDATED"**. None of the shipped datasets carry
held-out check-points. This is the *honest* behavior the PRD demands — but it means **every accuracy claim is a
design target, not a measurement.** Combined with the S2 inverse-depth bug (Tier 2) and confidence-saturation
(Tier 2), the current numbers should be treated as *unverified and probably optimistic* until a real check-point
dataset is run.

---

## 9. The build-it-in-one-go plan (dependency-ordered)

Each item says **[HONEST]** (make the claim/label truthful — cheap, hours) or **[REAL]** (build the missing engine
— days). Doing the **[HONEST]** column alone makes the system *defensible and non-fabricating*; the **[REAL]**
column makes the claims *true*. Do them in this order — later phases depend on earlier ones.

**Phase A — Make CI honest again (hours).**
- Fix the 3 ruff errors (`ruff check --fix` for the import block; add `strict=False` to the two `zip()`s). Re-run
  `pytest` + `ruff` green. Update `AGENTS.md §8`.

**Phase B — Stop the fabrications (½ day) [HONEST].**
- S2: remove the gtsam flag-clear; state "Sim(3) GPS fit" always (`s2_poses.py:128-132`).
- S6: make confidence independent of an uninvoked binary; keep it a labelled passthrough (`s6_global.py:61-72`).
- S9: label the DTM `morphological_opening` unless a real PDAL pipeline ran (`s9_geo.py:160,170`).
- Rewrite all three to mirror S3's honest tier pattern.

**Phase C — Fix the correctness bugs (2–4 days) [REAL] — the accuracy core.**
1. Depth domain: invert DA-V2 output before the affine fit, or adopt a true metric model (`depth.py:40`,
   `s4_depth.py:79-80`). *This is the single highest-value fix.*
2. Skip/down-weight degraded frames in S7 (`s7_dense.py:61-75`).
3. S2 robustness: RANSAC Sim(3) + correspondence-count penalty in confidence (`align.py`, `s2_poses.py:135`).
4. DTM nodata + coverage mask (S9).
5. S8 façade normals: up-prior only near ground; scalable orientation elsewhere (`s8_mesh.py:105-110`).

**Phase D — Reproducibility (1 day) [REAL].**
- Pin `version` + `sha256` on every shipped `ModelSpec`; verify on load; record resolved versions in each
  `StageRecord` (`registry.py`, `manifest.py`). Then acceptance-criterion 4 becomes real.

**Phase E — Implement the named capabilities (the big build; pick per priority).**
- **Compute story:** add an OpenVINO load/infer path in `models/` and consume `rt.backend()` in the runner — *or*
  drop the OpenVINO claim from PRD/overview (decide honestly first) **[REAL/HONEST]**.
- **Texture:** implement `best_view` texturing and make it the default (so S8 stops auto-degrading), then add
  MVS-Texturing **[REAL]**.
- **Masking/semantics:** add SAM2 refinement + a semantic head → satisfies F9 semantics + F15 layers **[REAL]**.
- **Spine/BA:** build the GTSAM factor graph (F7) and/or invoke GLOMAP (F6) **[REAL]**.
- **Ground:** wire a real PDAL SMRF pipeline (F14) **[REAL]**.

**Phase F — Viewer to spec (2–3 days) [REAL].**
- Add Cesium measurement (distance/area/volume/height), layer + confidence toggles, and Potree point-cloud
  rendering → satisfies F18. (Or formally amend F18 to the inspector scope.)

**Phase G — Robustness & scale [REAL/DEBT].**
- Stream LAS export instead of whole-cloud RAM (S10, N9).
- Extend the L0–L6 ladder to every heavy stage (N4).

**Phase H — Validate accuracy (needs data) [REAL].**
- Obtain/produce a check-point dataset; run the report's validated path; publish measured RMSE. Only after this can
  N1 be stated as anything but a design target.

**Cross-cutting doc fix:** reconcile `00-MASTER-OVERVIEW.md` with the ground-only build — either mark the
three-tier/live/RTK/nvblox material as "future vision, not this build" inline, or move it to `docs/_internal/`, so
the overview stops describing a different product than the PRD (§10 below).

---

## 10. The docs-vs-build gap (why "overclaims" exist)

`00-MASTER-OVERVIEW.md` sells the aspirational system; `AGENTS.md §2` + `PRD §2` build a strict subset. The
honest posture (`AGENTS.md §5`) is already the project's stated value — the debt in §3 is where the *code* fell
short of that posture, and the overview is where the *prose* runs ahead of it. Closing both is what "build it in
one go" means here: make the code stop signalling capability it lacks (Phase B), then either build the capability
(Phases C–H) or trim the claim to match. Nothing in this audit requires touching the 5 cm `dense.ply` or
degrading any working output.

---

*Generated 2026-09-22 against HEAD `117b426`. Every `file:line` above was read directly this session; the tests,
ruff, and grep results in §0/§3 were executed, not asserted.*
