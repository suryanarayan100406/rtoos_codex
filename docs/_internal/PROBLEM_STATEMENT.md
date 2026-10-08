# Problem Statement (Reference) + Official & Authored Output/Evaluation Tables

> This is the canonical restatement of the NTRO problem (PS-17 / **SIH26158**), now reconciled against
> the authoritative problem-statement PDF (`SIH26158.pdf`). **§1a holds the official Desired Output and
> Evaluation Criteria tables verbatim and is authoritative;** §2–§3 are our richer authored tables and
> must conform to §1a. Every document derives its scope from this file. Do not silently change scope; if
> scope must change, update this file first.

---

## 1. Problem (verbatim, condensed)

**Problem Statement ID:** SIH26158 (appears in the PDF as **Problem Statement – 17**)
**Title:** Single-Pass Drone Video to Accurate 3D Model Generation System
**Organization / Department:** National Technical Research Organisation (NTRO)
**Category:** Software · **Theme:** Drone / Robotics
**YouTube / Video link:** Nil · **Dataset:** "Will be provided real time" (at the event).

**Background.** Accurate 3D models of buildings, infrastructure, terrain and objects normally require
multiple drone passes, heavy image overlap, specialized flight planning and long post-processing. In
disaster response, surveillance, infrastructure inspection, military reconnaissance and rapid mapping
there is often **only one opportunity** to capture data. A system that produces an accurate, textured
3D model from a **single drone pass** would cut mission time, operator effort, data-acquisition
requirements and processing complexity, and enable **near-real-time situational awareness**.

**Task.** Build an AI-enabled system that generates a **georeferenced, metrically accurate 3D model**
from a **single-pass drone video** captured by a moving UAV, reconstructing: (i) 3D terrain &
structures, (ii) building facades & rooftops, (iii) roads & infrastructure, (iv) vegetation &
obstacles, (v) textured 3D meshes or point clouds — suitable for **visualization, measurement and
analysis**.

**Key challenges.** (i) Limited viewing angles from a single flight path; (ii) motion blur & video
compression artifacts; (iii) variable illumination & shadows; (iv) dynamic objects (vehicles, humans,
animals); (v) GPS inaccuracy & sensor noise; (vi) real-time / near-real-time processing; (vii)
reconstruction of occluded surfaces; (viii) metric accuracy **without extensive Ground Control Points
(GCPs)**.

**Inputs.**
- **Mandatory:** Drone video (1080p/4K); GPS coordinates; flight metadata.
- **Optional:** IMU data; barometric altitude; camera intrinsic parameters; RTK/PPK corrections.

**Applications.** Border & strategic-area mapping; disaster damage assessment; urban planning & smart
cities; infrastructure inspection; construction progress monitoring; archaeological documentation;
digital-twin generation; military reconnaissance & mission planning.

---

## 1a. Official "Desired Output" & "Evaluation Criteria" (verbatim from the PDF) — AUTHORITATIVE

> These two tables are the official targets from the problem-statement PDF (PS-17 / SIH26158). They are
> the bar every deliverable is measured against. The richer tables in §2–§3 elaborate on these but must
> **conform** to them; if they ever conflict, **these win**.

**Desired Output (official):**

| Parameter | Target |
|-----------|--------|
| Reconstruction Type | 3D Mesh / Point Cloud |
| Processing Time | **< 15 minutes for a 10-minute video** |
| Spatial Accuracy | **≤ 1 m** |
| Coverage | Entire visible scene |
| Output Formats | **OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx** |
| Visualization | Web-based or Desktop Viewer |

**Evaluation Criteria (official weights):**

| Criteria | Weightage |
|----------|-----------|
| Reconstruction Accuracy | **30%** |
| Model Completeness | **20%** |
| Processing Speed | **20%** |
| Innovation | **15%** |
| Scalability | **10%** |
| User Interface | **5%** |

**What these pin down (every document must reflect these):**
- **≤ 1 m spatial accuracy** is the absolute bar. We meet it cm-level with RTK/PPK and target
  sub-metre-to-~1 m with GPS(+IMU); regions whose positioning quality would exceed 1 m are **flagged**
  in the accuracy report, never silently reported. (Design targets, to be measured on the event dataset.)
- **< 15 min / 10-min video** concretizes "near-real-time": an edge coarse map builds during flight;
  the ground refine path delivers the final model within this budget. (Maps to **Processing Speed 20%**.)
- **Mandatory inputs are video (1080p/4K) + GPS + flight metadata only**; IMU, barometer, camera
  intrinsics and RTK/PPK are **optional**. The baseline pipeline therefore **self-calibrates intrinsics**
  and derives **scale + georeference from GPS baselines + visual structure**; IMU/baro/RTK are accuracy
  *enhancers*, not prerequisites. This is the single biggest technical constraint the design must honor.
- **Output formats must include `.fbx`** (CAD/DCC interchange) alongside OBJ / PLY / LAS / GeoTIFF /
  glTF-GLB. We may additionally emit LAZ, OGC 3D Tiles, CityJSON, COG — but the official six are required.
- **A web or desktop viewer/UI is a scored deliverable** (**User Interface 5%**; the Visualization row).
- Criteria → anchors mapping: Reconstruction Accuracy → **A2** metric spine; Model Completeness →
  coverage report + occlusion completion (**A1/A4**); Processing Speed → two-path edge/ground (**A3**) +
  TensorRT; Innovation → prior-assisted single-pass orchestration; Scalability → three tiers + swappable
  models + cloud tiling; User Interface → the web/desktop viewer.

---

## 2. Desired Output (our authored deliverable table)

> **Conforms to §1a.** This elaborates the official "3D Mesh / Point Cloud" row into the full
> deliverable set; the official targets (≤ 1 m, < 15 min, required formats incl. `.fbx`, viewer) govern.

Everything below is produced from **one flight** and exported in open, interoperable, GIS/CAD/
digital-twin-ready formats. Each artifact carries a per-region **confidence** value.

| # | Deliverable | Primary format(s) | Purpose / use |
|---|-------------|-------------------|----------------|
| 1 | **Georeferenced dense point cloud** | LAS/LAZ, PLY | Measurement, GIS ingest, ground truth for downstream products |
| 2 | **Textured 3D mesh (multi-LOD)** | OBJ+MTL, glTF/GLB, **FBX**, OGC **3D Tiles**, OSGB | Visualization, inspection, digital twin, CAD/DCC interchange |
| 3 | **3D Gaussian-Splat scene** | `.ply` / `.splat` / `.ksplat` | Photorealistic free-viewpoint situational awareness |
| 4 | **Digital Surface Model (DSM) + Digital Terrain Model (DTM)** | GeoTIFF (float32, COG) | Elevation, volumetrics, terrain & slope analysis |
| 5 | **True orthomosaic** | GeoTIFF (COG) | Top-down basemap, planning, change detection |
| 6 | **Semantic layers** | GeoJSON / Shapefile + labeled cloud & mesh | building/roof, road/infra, vegetation, terrain, obstacle; dynamic objects removed |
| 7 | **Measurements & analytics** | JSON + in-viewer tools | Distances, areas, volumes, heights, slopes, clearances |
| 8 | **Accuracy & confidence report** | PDF + JSON | RMSE (H/V), GSD, coverage %, per-region confidence, occlusion map |
| 9 | **Live coarse map + telemetry** | Streamed 3D + 2D (WebRTC/RTSP → web viewer) | Near-real-time situational awareness **during** the flight |
| 10 | **Reproducible project bundle** | Archive (video ref, poses, calibration, logs, metadata) | Audit, re-processing, chain-of-custody |

---

## 3. Evaluation Criteria (our authored assessment table)

> **Conforms to §1a.** The official weighted rubric (Accuracy 30 / Completeness 20 / Speed 20 /
> Innovation 15 / Scalability 10 / UI 5) is the scored one; the rows below are how we instrument each.
> Note the official **absolute bar is ≤ 1 m** and the official **speed bar is < 15 min / 10-min video**.

Targets assume a nadir/oblique single pass at typical AGL. Two accuracy regimes are reported because
RTK/PPK is *optional* input: **(A) RTK/PPK available** and **(B) GPS-only**.

| # | Criterion | Metric / how measured | Target (A: RTK/PPK · B: GPS-only) |
|---|-----------|------------------------|------------------------------------|
| 1 | **Absolute geometric accuracy** | Horizontal & vertical RMSE vs independent checkpoints/survey | **Official bar ≤ 1 m.** A: ≤ 2–5 cm + 1×GSD · B: target ≤ 1 m (0.5–2 m envelope by GPS quality; **flag > 1 m**) |
| 2 | **Relative accuracy** | Known-distance / scale-bar error; local RMSE | A: < 1% of distance · B: report measured value |
| 3 | **Ground Sampling Distance (GSD)** | cm/pixel from AGL & sensor geometry | ~1.5–3 cm/px at typical mapping altitude |
| 4 | **Completeness / coverage** | % target surface above confidence threshold; occlusion-flagged area | Maximize; **explicitly report** unseen / low-confidence regions |
| 5 | **Reconstruction fidelity** | Point density (pts/m²), mesh detail, texture sharpness, hole ratio | Quantitative + qualitative panel |
| 6 | **Quality vs reference** | Cloud-to-cloud (C2C), cloud-to-mesh (C2M), Chamfer vs COLMAP/Metashape or LiDAR | Minimize distance; report percentiles |
| 7 | **Processing latency** | (a) live-preview latency per keyframe; (b) time-to-final-model per minute of video / km² | **Official bar: < 15 min for a 10-min video (end-to-end, ground).** a: near-real-time (≈1–2 s/keyframe, edge) · b: within the < 15 min/10-min budget |
| 8 | **Robustness** | Degradation curve under injected blur, illumination change, GPS noise, dynamic-object density | Graceful, monotonic; **no hard failure** |
| 9 | **Dynamic-object rejection** | Precision/recall of masked movers; residual "ghost" density in map | High recall; near-zero ghosting |
| 10 | **Georeferencing correctness** | Absolute position error of known features; CRS/datum/geoid correctness | Within accuracy budget; correct EPSG + geoid model |
| 11 | **Real-world deployability** | Runs on Jetson Orin + ground GPU; integrates with DJI/PX4; power/thermal within budget | Demonstrated on real hardware |
| 12 | **Reliability / availability** | Recovery from tracking loss, link loss, sensor dropout; % missions yielding a usable model | High; documented fallback ladder |
| 13 | **Usability & interoperability** | Visualization + measurement UX; export into GIS/CAD/twin toolchains | Standards-compliant, tool-agnostic outputs |

---

## 4. Honesty guardrails (read before writing any claim)

- "Real-time" = **near-real-time live coarse preview on the edge** + **full refined model in minutes**
  on the ground tier. The official concretization is **< 15 minutes for a 10-minute video** end-to-end.
  We never claim a full 4K textured mesh in hard real-time on the drone.
- "Never fails" = **graceful degradation with no single point of failure**; the system always emits a
  best-effort model **plus an explicit uncertainty/completeness report**, rather than crashing or
  silently emitting garbage.
- "Metric without GCPs" = accurate **because** of a fused metric spine, with honest,
  sensor-configuration-dependent accuracy numbers — not a claim that GCP-grade survey accuracy comes
  for free. **Because IMU, intrinsics and RTK/PPK are all *optional* inputs, the honest baseline is
  GPS + visual scale with self-calibrated intrinsics**, targeting the official **≤ 1 m** bar; RTK/PPK
  and IMU tighten this toward cm-level when present.
