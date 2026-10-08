# Research dossier — competitive-landscape

> Auto-generated from the research workflow. Source material for document authoring.

## Executive summary

No existing product solves the stated problem end-to-end: single-pass drone video to a georeferenced, metrically accurate, textured 3D model in near-real-time with graceful degradation. The market splits into three camps that each solve only part of it. (1) Offline photogrammetry suites (Pix4Dmapper, Agisoft Metashape, DJI Terra, RealityCapture, Bentley ContextCapture, OpenDroneMap) deliver metric, georeferenced, textured 3D, but demand 70-80% frontal/side overlap, nadir-plus-oblique grid missions, GCPs or RTK, and minutes-to-hours of batch compute; they are neither single-pass nor real-time and tend to hard-fail or leave holes when overlap is insufficient. (2) Real-time SLAM/VIO (ORB-SLAM3, VINS-Fusion, OpenVINS, RTAB-Map) runs live on one moving camera and recovers metric scale from IMU, but outputs sparse or short-range maps built for localization, not textured survey deliverables. (3) NeRF / Gaussian-Splatting capture (Luma AI, Polycam, Nerfstudio, 3DGS) produces photorealistic 3D quickly, but is not natively metric or georeferenced, needs many well-posed views (usually COLMAP), and the consumer clouds are unusable for air-gapped defense data. The only incumbent explicitly built for rapid single-flight response, Pix4Dreact, is 2D-orthomosaic only. Our defensible wedge is exactly the unoccupied intersection: an on-prem, edge-assisted pipeline fusing VIO metric pose + RTK/PPK georeferencing + learned monocular depth to build a progressively refined textured model from one constrained flight path, that always emits a best-available product with confidence metadata instead of failing. We must be honest that absolute accuracy, facade completeness, and final mesh polish will trail a proper multi-pass mission with GCPs, so the system should ship an offline refinement tier and must not over-claim survey-grade certification.

### Visual-Inertial SLAM/Odometry front-end (VINS-Fusion / OpenVINS / ORB-SLAM3)  
*Maturity: production*

- **What:** Real-time camera-pose estimation from video plus IMU, with loop closure, pose-graph/bundle optimization, and optional GPS fusion. VINS-Fusion and OpenVINS are optimization/filter VIO estimators; ORB-SLAM3 adds visual-inertial multi-map SLAM.
- **Why it fits single-pass:** This is the one incumbent family that genuinely handles a single moving camera on one flight path in real time AND recovers METRIC scale from the IMU without GCPs, exactly the constraint set in the problem. It becomes our pose/scale backbone; GPS fusion gives drift-bounded global alignment.
- **Perf/accuracy:** Real-time on a strong CPU (ORB-SLAM3 needs ~i7; VINS ~10-20 Hz on embedded). EuRoC ATE typically a few cm to sub-dm. Metric scale from IMU. Output is SPARSE feature points, not a textured surface.
- **Failure modes:** Sparse map only (no facades/texture); scale drift and initialization failure under low-IMU-excitation constant-velocity flight; feature loss on textureless/blurred/compressed frames; monocular-only runs are scale-ambiguous without IMU/RTK.
- **Alternatives:** DROID-SLAM (deep, denser, more robust to blur but GPU-heavy); Kimera (adds metric-semantic mesh); OKVIS; classic COLMAP poses offline as an accuracy oracle.

### 3D Gaussian Splatting (+ SuGaR / 2DGS mesh extraction)  
*Maturity: near-production*

- **What:** Explicit radiance-field reconstruction that trains photorealistic, real-time-renderable 3D from posed images; SuGaR/2DGS extract a mesh, and texture is baked from source frames.
- **Why it fits single-pass:** Fastest path to a photorealistic, view-consistent textured model from limited/uneven viewing angles, and it tolerates the sparse/irregular view geometry of a single pass better than classic dense MVS. Renders in real time for the visualization/analysis requirement.
- **Perf/accuracy:** Training ~35-45 min (30k iter) or 5-10 min quick pass; real-time rendering (30-100+ FPS); >20 GB VRAM peaks on large scenes. Quality on par with Mip-NeRF360.
- **Failure modes:** NOT natively metric or georeferenced (must be anchored externally); needs decent poses (COLMAP/VIO); floaters/elongated splats and popping; degrades with illumination/shadow inconsistency and motion blur; large-scene memory blow-up; mesh from splats is still noisy.
- **Alternatives:** TSDF/volumetric fusion (metric, robust, less photoreal); NeRF/Instant-NGP; neural MVS (MVSNet family); classic Poisson-mesh from MVS point cloud for the survey-grade tier.

### RTK / PPK GNSS georeferencing  
*Maturity: production*

- **What:** Carrier-phase GNSS correction (real-time RTK via NTRIP base/CORS, or post-processed PPK) giving cm-level absolute positions for image/keyframe centers.
- **Why it fits single-pass:** This is THE realistic answer to 'metric accuracy WITHOUT extensive GCPs' called out in the problem, and it is an explicit optional input. Accurate image-center positions let us georeference and metrically scale the reconstruction with few or zero ground markers.
- **Perf/accuracy:** ~8 mm + 1 ppm horizontal, ~15 mm vertical relative to base; cm-level absolute; effective to ~20 km baseline (error grows ~1 mm/km).
- **Failure modes:** Needs a base station / NTRIP / CORS network in range; absolute accuracy capped by base-station coordinate quality; RTK fix can drop under multipath/jamming; many field drones only have single-frequency GPS (metres), so we must degrade gracefully to VIO scale.
- **Alternatives:** PPP (no base, slower convergence); a handful of GCPs/checkpoints for validation; visual scale from known-size targets; IMU-only metric scale as fallback.

### Learned monocular metric depth + neural MVS (Depth Anything V2 / Metric3D / MVSNet / DROID-SLAM)  
*Maturity: near-production*

- **What:** Deep networks that predict per-frame (metric) depth and/or fuse multi-view depth, injecting geometry where triangulation baselines are weak.
- **Why it fits single-pass:** Directly attacks the single-pass core problem: one flight path gives strong along-track but weak cross-track baselines, so classic MVS starves. Monocular priors + video temporal fusion fill density that overlap alone cannot provide.
- **Perf/accuracy:** Depth Anything V2 / Metric3D run at interactive rates on Jetson/GPU; relative depth strong, absolute metric depth improving but still needs scale/pose anchoring; MVSNet-class nets give dense depth per view.
- **Failure modes:** Monocular metric scale is domain-sensitive (altitude/GSD shifts); hallucinated geometry in occluded/unseen regions; struggles on water, glass, thin structures, dense vegetation; needs fusion + confidence to avoid over-trusting priors.
- **Alternatives:** Classic PatchMatch MVS (COLMAP/OpenMVS) where overlap is sufficient; stereo rig or LiDAR payload to sidestep monocular ambiguity entirely.

### Pix4Dreact (rapid-response incumbent, 2D only)  
*Maturity: production*

- **What:** Fast 2D orthomosaic/map generation on a laptop aimed at emergency response and public safety.
- **Why it fits single-pass:** It is the closest thing on the market to our 'rapid single overflight' target and proves the demand, which makes it the sharpest benchmark, but it is our opening precisely because it produces only 2D maps, no 3D terrain, facades, or textured mesh.
- **Perf/accuracy:** Near-real-time 2D orthomosaics in minutes on a field laptop; tolerates lower overlap than full 3D pipelines; no 3D output.
- **Failure modes:** No 3D reconstruction at all; no facades/rooftops geometry; 2D only, so unusable for volumetrics, mission planning in 3D, or digital twins; still a post-flight batch (not live-streaming).
- **Alternatives:** Pix4Dmapper for full offline 3D; DroneDeploy live map; our system for the 3D+near-real-time gap Pix4Dreact leaves open.

### DJI Terra (hardware-integrated commercial competitor)  
*Maturity: production*

- **What:** DJI's desktop reconstruction suite: 2D/3D from visible light, 3D mesh, LiDAR reconstruction, multispectral, and now 3D Gaussian Splatting; cluster processing for speed.
- **Why it fits single-pass:** Strongest turnkey competitor because most field UAVs are DJI, so Terra is the default for planned-mission mapping. But it targets planned grid/oblique missions and post-processing, not a single constrained pass with near-real-time output, and it is tied to the DJI ecosystem/cloud.
- **Perf/accuracy:** Efficient cluster reconstruction; free trial capped at <500 photos / 8 GB LiDAR / 1 month / 1 machine; paid Standard/Flagship via dealers. Post-processing, not live.
- **Failure modes:** Not single-pass or real-time; best with proper overlap/oblique missions; ecosystem lock-in and data-residency concerns for defense; limited graceful-degradation story.
- **Alternatives:** RealityCapture (free <$1M, faster batch); Metashape; our on-prem near-real-time pipeline for the field/degraded-mission niche.

### COLMAP (+ OpenMVS) offline SfM/MVS  
*Maturity: production*

- **What:** Gold-standard incremental Structure-from-Motion plus PatchMatch multi-view stereo producing sparse then dense point clouds; the de-facto pose backend for NeRF/GS.
- **Why it fits single-pass:** Not for the live path, but it is our accuracy oracle and offline refinement tier: when the operator can afford a batch pass, COLMAP+OpenMVS gives the highest-fidelity metric point cloud to validate and refine the real-time output against.
- **Perf/accuracy:** Point clouds comparable in density/accuracy to laser scanning; but OFFLINE and slow, incremental SfM scales poorly (minutes-to-hours; large sets need hours and heavy RAM/GPU).
- **Failure modes:** Not real-time; scale-ambiguous without georef/known baseline; incremental SfM can drift or fail to register weakly-overlapping single-pass frames; heavy compute.
- **Alternatives:** GLOMAP (faster global SfM); Metashape/RealityCapture; hloc for robust matching; our VIO for the online path.

### OpenDroneMap (open-source offline batch)  
*Maturity: production*

- **What:** AGPLv3 pipeline (OpenSfM + MVS) producing georeferenced orthophotos, DSM, point clouds and meshes from drone imagery.
- **Why it fits single-pass:** Best open baseline and a legitimate on-prem offline fallback tier for our system when time permits and overlap is adequate, with no licensing cost or data-residency issue, valuable for an air-gapped NTRO deployment.
- **Perf/accuracy:** Free/open; georeferencing via GPS/GDAL/PDAL; resource-heavy (recommends ~128 GB RAM for 2,500 images; split-merge for low-RAM). Offline batch.
- **Failure modes:** Not real-time; needs standard overlap (assumes grid missions); heavy RAM/CPU; quality/robustness below top commercial suites on hard scenes; single-pass low-overlap input produces gaps.
- **Alternatives:** MicMac, Meshroom (open); commercial suites for polish; our pipeline for the live/degraded case ODM cannot serve.

### Offline metric survey suites (Agisoft Metashape / RealityCapture / Bentley ContextCapture)  
*Maturity: production*

- **What:** Professional photogrammetry: dense point clouds, textured meshes, DSM/orthos, LOD reality meshes, georeferencing with GCP/RTK; ContextCapture scales to city/infrastructure datasets.
- **Why it fits single-pass:** These are where incumbents genuinely BEAT us and we must say so: for planned missions with good overlap and GCPs they deliver higher absolute accuracy and cleaner final deliverables than any single-pass real-time system. They are our benchmark ceiling and complementary refinement backends, not head-to-head competitors on our niche.
- **Perf/accuracy:** Survey-grade with GCP/RTK; RealityCapture now free for <$1M revenue and very fast (near-linear scaling); Metashape Pro ~$3,499 perpetual; ContextCapture enterprise-priced. All OFFLINE.
- **Failure modes:** Not single-pass, not real-time; require multi-pass/oblique coverage and often GCPs; can hard-fail or leave holes on sparse-overlap single-pass video; high cost (ContextCapture) or export/format friction.
- **Alternatives:** Each other; ODM (free); our system only for the rapid/degraded/GPS-contested niche they do not target.

### RTAB-Map (real-time dense mapping reference)  
*Maturity: production*

- **What:** Real-time appearance-based graph SLAM with RGB-D/stereo/LiDAR, loop closure, and dense point-cloud/mesh output; ROS/ROS2, BSD-licensed.
- **Why it fits single-pass:** Proves real-time DENSE mapping is feasible and gives reusable loop-closure/graph-optimization machinery, but its sweet spot is short-range RGB-D on ground robots, so it fits our architecture as reference more than as a drop-in for high-altitude monocular aerial video.
- **Perf/accuracy:** Real-time dense mapping with RGB-D/stereo; BSD license; good indoor/near-range metric maps and loop closure.
- **Failure modes:** Depth range limited (RGB-D ~metres) so weak for aerial altitudes; monocular aerial at scale is not its strength; georeferencing not built-in; large outdoor maps stress memory.
- **Alternatives:** Kimera (metric-semantic mesh); Voxblox/nvblox TSDF; our VIO+GS stack for aerial texture.

### Consumer/research NeRF-GS capture (Luma AI, Polycam, Nerfstudio)  
*Maturity: research*

- **What:** Nerfstudio is an open Apache-2.0 toolkit for NeRF/Gaussian-Splatting; Luma AI (now pivoting to generative video) and Polycam are consumer capture apps with cloud processing and rough on-device measurement.
- **Why it fits single-pass:** Mostly an anti-pattern that clarifies what we must NOT be: cloud-dependent, non-metric, non-georeferenced, consumer-grade. Nerfstudio is genuinely useful as our R&D harness for GS/NeRF experimentation, but none are deployable for air-gapped, survey-grade defense mapping.
- **Perf/accuracy:** Nerfstudio: real-time training viz for fast methods, poses from COLMAP, point-cloud export possible. Luma/Polycam: cloud, minutes, freemium; Polycam shows areas/dimensions but makes NO survey-grade/georeferenced claims.
- **Failure modes:** Not metric/georeferenced by default; cloud upload violates data sovereignty (fatal for NTRO); need many well-posed views; Luma has largely left 3D capture for generative media.
- **Alternatives:** Self-hosted gsplat/Nerfstudio in our pipeline; on-prem everything; classic photogrammetry for metric guarantees.

## Recommended stack

| Component | Choice | Rationale |
|---|---|---|
| Video + telemetry ingest | DJI Payload/Onboard SDK or MAVLink + RTSP/UDP stream via GStreamer; parse EXIF/XMP geotags and flight log | Covers the dominant DJI fleet and open PX4/ArduPilot drones; pulls the mandatory video + GPS + flight metadata and optional IMU/baro/intrinsics the problem specifies. |
| Real-time pose + metric scale | VINS-Fusion or OpenVINS (GPS-fused VIO); ORB-SLAM3 for loop-closure/multi-map; DROID-SLAM as robust GPU fallback | Only proven way to get live, metric, drift-bounded pose from a single moving camera without GCPs; GPS/RTK fusion anchors it globally. |
| Depth densification | Depth Anything V2 / Metric3D monocular depth + MVSNet-class fusion; PatchMatch MVS (OpenMVS) when overlap allows | Fills the weak cross-track baseline of a single pass; switches to true MVS when geometry supports it for higher fidelity. |
| Reconstruction + texturing | 3D Gaussian Splatting (gsplat/Nerfstudio) with SuGaR/2DGS mesh export; TSDF volumetric fusion (nvblox) as metric-robust fallback | GS gives fast photorealistic textured output tolerant of irregular views; TSDF gives a metric, robust surface when GS is unstable, satisfying graceful degradation. |
| Dynamic-object handling | YOLO/RT-DETR + SAM2 segmentation to mask vehicles/humans/animals before fusion | Turns a listed key challenge into a first-class feature; most photogrammetry suites treat movers as noise/artifacts. |
| Georeferencing (GCP-free) | RTK via NTRIP (RTKLIB) or PPK post-processing; 7-parameter Helmert alignment of geotagged keyframes; optional few checkpoints for validation | Delivers cm-level metric georeferencing without extensive GCPs, directly meeting the hardest accuracy constraint; degrades to GPS+IMU scale when no RTK. |
| Offline refinement tier | COLMAP/GLOMAP + OpenMVS, or OpenDroneMap, run on-prem when time permits | Provides a survey-grade accuracy ceiling and honest upgrade path beyond the near-real-time product; all open and air-gappable. |
| Edge + ground-station compute | NVIDIA Jetson Orin AGX (companion/onboard) + RTX-class ground-station GPU laptop over the datalink | Runs VIO + depth on the edge for near-real-time, offloads GS training/refinement to the ground station; realistic, field-deployable, no cloud. |
| Output, CRS + visualization | LAS/LAZ point cloud, glTF/OBJ + Cesium 3D Tiles mesh, GeoTIFF ortho/DSM, EPSG/UTM CRS; potree/CesiumJS viewer | Standard geospatial formats make outputs measurable/analyzable in existing GIS and mission-planning tools and support digital-twin visualization. |
| Time synchronization | Hardware/PTP or software timestamp alignment of frames, IMU, GPS/RTK | VIO metric scale and georeferencing accuracy hinge on tight frame-to-IMU-to-GNSS sync; the most common silent accuracy killer if skipped. |

## Real-world integration

Ingest the mandatory drone video as an RTSP/UDP H.264/H.265 stream or recorded MP4, with GPS and flight metadata from the DJI Payload/Onboard SDK or MAVLink telemetry; parse per-frame EXIF/XMP geotags and the flight log for gimbal attitude, altitude and (if present) camera intrinsics. Consume optional IMU at 100-400 Hz, barometric altitude, and RTK/PPK corrections over NTRIP (RTKLIB) from a base station or CORS/VRS network. The single most load-bearing integration detail is time synchronization: frames, IMU, and GNSS must share a common clock (hardware trigger/PTP where possible, otherwise cross-correlation software sync) or VIO scale and georeferencing silently degrade. Camera intrinsics should be self-calibrated online if not provided. Compute is tiered: a Jetson Orin AGX companion (or the drone's onboard compute) runs VIO + monocular depth + dynamic masking for near-real-time pose and a coarse model, streaming keyframes over the datalink to a ground-station RTX GPU that trains/refines the Gaussian-Splatting or TSDF model; an optional on-prem server runs the COLMAP/OpenMVS or ODM refinement pass. Outputs are standard geospatial formats: LAS/LAZ point clouds, glTF/OBJ + Cesium 3D Tiles textured meshes, GeoTIFF orthomosaics and DSMs, all tagged with an explicit CRS (EPSG/UTM). Everything is on-prem/air-gappable, which is mandatory for NTRO data sovereignty and rules out the consumer clouds (Luma, Polycam) and, for sensitive missions, the DJI cloud.

## Reliability & failure handling

Design as a degradation ladder that always emits a best-available product with confidence metadata, never a hard failure. Georeferencing/scale fallback chain: RTK fix -> PPK post-process -> GPS + IMU-derived metric scale -> monocular depth-prior scale -> relative (unscaled) model flagged as non-metric. Reconstruction fallback: full GS textured mesh -> TSDF metric surface -> sparse VIO point cloud + 2D orthomosaic -> localization/trajectory only. Because output is progressive/incremental (local bundle adjustment on a sliding window plus loop closure to bound drift), a usable coarse model exists within seconds and refines as more video arrives, so an aborted or shortened flight still yields something. Occluded and single-view regions are explicitly labeled as low-confidence/unobserved rather than hallucinated, giving analysts an honest coverage map instead of invented facades. Dynamic objects are segmented and masked so movers do not corrupt geometry, and are optionally re-inserted as tracked annotations. A watchdog monitors feature counts, VIO covariance, and RTK fix status; on motion blur, compression artifacts, or fix loss it automatically drops to the next-lower tier and records why in the output metadata. Every deliverable carries per-region uncertainty and a provenance/quality report so downstream measurement and mission-planning tools (and human operators) know what to trust, and an offline refinement pass can be queued to upgrade the near-real-time result when time and compute allow.

## Single-pass specifics

Traditional photogrammetry (Pix4D, Metashape, DJI Terra, RealityCapture, ContextCapture, ODM) is built on the opposite assumptions: 70-80%+ frontal/side overlap, planned nadir-plus-oblique grid missions for facades, GCPs (or RTK) for metric anchoring, and one large OFFLINE global bundle adjustment over hundreds/thousands of images. A single pass violates all of these, so the pipeline must change fundamentally. (1) Replace overlap-redundancy with video temporal density plus learned monocular depth priors: many closely-spaced frames give strong along-track baselines but weak cross-track ones, so triangulation must be augmented by neural depth rather than relying on wide multi-view geometry. (2) Replace GCPs with VIO-from-IMU metric scale and RTK/PPK georeferencing of keyframes (few or zero ground markers). (3) Replace the offline global solve with incremental/streaming SfM: sliding-window local bundle adjustment, keyframe selection, and loop closure to bound drift while producing output live. (4) Treat occlusion and facade backsides as first-class: a single flight sees mostly one side of structures, so the system must explicitly mark unobserved surfaces and never fabricate them, whereas multi-pass suites simply demand you fly more. (5) Actively reject dynamic objects (vehicles/humans/animals) via semantic masking instead of assuming a static scene. (6) Build in robustness to motion blur, rolling shutter, and video-compression artifacts (deblur/keyframe quality gating) that curated photo sets avoid. The net trade is deliberate: accept lower absolute accuracy and incomplete coverage in exchange for coverage-time, robustness, and near-real-time usability, then optionally recover accuracy via an offline refinement tier.

## Open risks

- Metric scale ambiguity/drift on a single monocular pass when IMU excitation is low (constant-velocity straight-line flight) and no RTK is available.
- Facade/backside/occlusion incompleteness is inherent to one flight path; we can label but not fill unobserved surfaces, which caps usefulness for full digital twins.
- Motion blur, rolling shutter, and H.264/H.265 compression artifacts degrade feature matching and GS/NeRF quality; deblur and keyframe gating add latency.
- Variable illumination and shadows break photometric consistency for Gaussian Splatting/NeRF and produce baked-in shadow artifacts in texture.
- Proving metric accuracy WITHOUT GCPs is hard to certify; without at least a few checkpoints, accuracy claims are difficult to validate for legal/survey use.
- Edge compute (Jetson-class) may not sustain dense real-time reconstruction; realistic target is near-real-time coarse output + ground-station refinement, which must be messaged honestly.
- GS/NeRF georeferencing and metric guarantees are still maturing research; mesh extraction from splats remains noisy.
- Water, glass, reflective and thin/vegetated structures fail for all methods including ours.
- GPS jamming/spoofing in contested/border environments can corrupt georeferencing; VIO helps locally but global anchoring suffers.
- Data sovereignty forbids consumer/DJI cloud paths, so we lose their compute scale and must run fully on-prem within power/thermal limits of a field ground station.

## Differentiation notes

Our defensible wedge is the unoccupied intersection, not any single capability: single flight path + limited overlap + near-real-time + metric/georeferenced (RTK/PPK, GCP-free) + textured 3D + graceful degradation + fully on-prem/air-gapped. Each incumbent owns one axis and abandons the others. Offline suites (Pix4Dmapper, Metashape, DJI Terra, RealityCapture, ContextCapture, ODM) own accuracy and polish but require planned multi-pass/oblique missions, GCPs, and batch compute, and they tend to hard-fail or leave holes on sparse single-pass video. SLAM/VIO (ORB-SLAM3, VINS-Fusion, OpenVINS, RTAB-Map) owns real-time metric localization but outputs sparse/short-range maps, not survey deliverables. NeRF/GS (Luma, Polycam, Nerfstudio, 3DGS) owns photorealism and speed but is non-metric, non-georeferenced by default, and cloud-bound at the consumer end. Pix4Dreact owns rapid response but is 2D only. Where we HONESTLY LOSE: for a planned mission with good overlap and GCPs, Pix4D/Metashape/DJI Terra/RealityCapture/ContextCapture will beat us on absolute accuracy, facade completeness, and final mesh cleanliness; RealityCapture is now free (<$1M revenue) and extremely fast for that offline batch niche; DJI Terra is turnkey for the DJI fleet; ODM is a free, proven open baseline. We should not claim survey-grade certification or full 360-degree building reconstruction from one pass. Where we genuinely WIN for NTRO use-cases (disaster response, border/recon, rapid mapping, mission planning): time-to-first-usable-3D under operational constraints where you get one overflight and cannot fly a grid; robustness in GPS-degraded/contested environments via VIO; graceful degradation that always yields a best-available product with an honest coverage/confidence map; dynamic-object masking as a feature; and complete on-prem/air-gapped operation for data sovereignty. The strategy is not to out-accurate photogrammetry but to serve the operational gap it structurally cannot: one pass, right now, on-site, never failing outright, with an offline refinement tier for when accuracy must be recovered.

## Citations / references

- ORB-SLAM3: Campos et al., IEEE T-RO 2021 (github.com/UZ-SLAMLab/ORB_SLAM3, GPL-3.0)
- VINS-Mono / VINS-Fusion: Qin, Li, Shen, IEEE T-RO 2018 (HKUST-Aerial-Robotics)
- OpenVINS: Geneva et al., ICRA 2020
- RTAB-Map: Labbe & Michaud, J. Field Robotics 2019 (introlab/rtabmap, BSD)
- DROID-SLAM: Teed & Deng, NeurIPS 2021
- COLMAP / Structure-from-Motion Revisited: Schonberger & Frahm, CVPR 2016; Pixelwise View Selection for MVS, ECCV 2016
- GLOMAP: Pan et al., ECCV 2024
- 3D Gaussian Splatting: Kerbl, Kopanas, Leimkuhler, Drettakis, SIGGRAPH 2023
- SuGaR mesh from Gaussians: Guedon & Lepetit, CVPR 2024; 2D Gaussian Splatting, SIGGRAPH 2024
- NeRF: Mildenhall et al., ECCV 2020; Instant-NGP: Muller et al., SIGGRAPH 2022
- Nerfstudio: Tancik et al., SIGGRAPH 2023 (Apache-2.0)
- MVSNet: Yao et al., ECCV 2018
- Depth Anything V2: Yang et al., 2024; Metric3D v2, 2024; ZoeDepth, 2023
- Pix4Dmapper and Pix4Dreact (pix4d.com) - Pix4Dreact is 2D rapid-response only
- Agisoft Metashape (Standard/Pro, proprietary, ~$3,499 Pro perpetual)
- DJI Terra (enterprise.dji.com/dji-terra, 2024 - adds 3D Gaussian Splatting, mesh, LiDAR)
- RealityCapture / RealityScan 2.0, Epic Games (free for <US$1M revenue since April 2024; airborne LiDAR support)
- Bentley ContextCapture / iTwin Capture Modeler (enterprise reality modeling)
- OpenDroneMap / ODM (opendronemap.org, AGPLv3; OpenSfM + PDAL/GDAL)
- Luma AI (lumalabs.ai, now generative video); Polycam (poly.cam, LiDAR+photo+splat, no survey-grade/georef claim)
- RTK positioning: ~8mm+1ppm horizontal, ~20km baseline (en.wikipedia.org/wiki/Real-time_kinematic_positioning); PPK post-processing
- Kimera metric-semantic SLAM: Rosinol et al., ICRA 2020; nvblox/Voxblox TSDF fusion
- SAM2 (Meta, 2024) and YOLO/RT-DETR for dynamic-object segmentation/masking
