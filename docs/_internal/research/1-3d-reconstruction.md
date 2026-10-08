# Research dossier — 3D Reconstruction backbone (pose/geometry, dense/textured, surface/mesh, volumetric fusion) for single-pass drone video

> Auto-generated from the research workflow. Source material for document authoring.

## Executive summary

For single-pass drone video, demote triangulation-first incremental SfM (COLMAP) from primary engine to fallback, and build the backbone on feed-forward geometric foundation models that regress camera poses AND dense (near-)metric pointmaps directly from frames. Core recommendation: VGGT (CVPR 2025 Best Paper; <1s inference; 1 to hundreds of views; outputs cameras+depth+pointmaps+tracks in one pass) as the geometry engine, CUT3R for online per-frame streaming and near-real-time preview, Pi3/MapAnything for reference-frame-robust metric output, MASt3R-SfM as a careful offline global solver, and VGGT-Long's chunk + overlap-alignment + loop-closure wrapper for kilometer-scale corridors. These models exploit learned monocular+multi-view priors, so they stay stable at the tiny baselines and limited viewing angles that make COLMAP/Metashape/Pix4D fail or hole out on a single pass. Reinforce them with a monocular metric-depth prior (Depth Anything V2 metric or Metric3D v2 / UniDepth) to fix scale and shape where parallax is near-degenerate. Recover metric scale and georeferencing WITHOUT GCPs by aligning the reconstructed camera-center trajectory to RTK/PPK GPS via a 7-DOF Umeyama (Sim3) fit — or SE3 when using a metric model + RTK — then refining with GPS-prior bundle adjustment. Emit a live volumetric TSDF (nvblox on Jetson / Open3D ScalableTSDF) for immediate situational awareness, then refine offline into a depth-regularized 3D Gaussian Splatting model (InstantSplat feed-forward init + FSGS/DNGaussian) and extract a textured, watertight mesh via 2DGS/GOF/SuGaR + Poisson/TSDF. Mask dynamic objects (SAM 2 + motion, or MonST3R) before fusion, complete occluded surfaces with learned priors/inpainting and explicitly flag them, and expose georeferenced LAS/LAZ, GeoTIFF DSM/orthophoto, and Cesium 3D Tiles/glTF outputs. The whole stack degrades gracefully — RTK→cm metric, GPS→meter-level, none→relative — and always returns at least the live TSDF so it never hard-fails.

### VGGT (Visual Geometry Grounded Transformer) + VGGT-Long  
*Maturity: near-production*

- **What:** Feed-forward transformer that in a single forward pass regresses camera intrinsics/extrinsics, dense depth maps, point maps, and 3D point tracks from 1 to hundreds of views; VGGT-Long wraps it with chunked processing + overlapping alignment + lightweight loop closure for kilometer-scale sequences.
- **Why it fits single-pass:** Single-pass = tiny/forward baselines and limited angles where triangulation is near-degenerate. VGGT infers geometry from learned priors rather than parallax, so it stays stable where COLMAP breaks; near-instant inference supports near-real-time; VGGT-Long solves the long-corridor/memory and drift problem that pure VGGT hits on drone strips.
- **Perf/accuracy:** CVPR 2025 Best Paper; reconstructs in <1s per view-set, no post-optimization needed; ~1B params, runs on one A100/RTX-class GPU; SOTA on camera-pose and point-map benchmarks. VGGT-Long demonstrated on 4500-frame KITTI sequences (disk-backed, ~50GiB scratch), supports VGGT/Pi3/MapAnything backends with SE(3) metric alignment.
- **Failure modes:** Base model OOMs beyond ~hundreds of views (VGGT-Long/chunking mitigates); anchors to a reference frame so long sequences drift without loop closure; can hallucinate plausible-but-wrong geometry in low-confidence/occluded regions; aerial nadir/oblique domain gap vs training data; some weights are research/non-commercial (need commercial weights for product).
- **Alternatives:** Fast3R (up to 1500 images/pass, ~7 img/s @1000 views on A100, no global alignment); Spann3R (spatial-memory online); DUSt3R (pairwise, OOM ~48+ views); Pow3R (prior-conditioned on intrinsics/depth).

### CUT3R (Continuous Updating Transformer for 3D Reconstruction)  
*Maturity: research*

- **What:** Stateful recurrent transformer that ingests an RGB stream one frame at a time and emits metric-scale per-pixel pointmaps + camera params online, accumulating into a growing dense reconstruction in a shared frame with no global optimization; can also query virtual viewpoints to infer unseen regions.
- **Why it fits single-pass:** Purpose-built for streaming single-pass video: true online/feed-forward operation matches the 'video stream from a moving UAV' input and the near-real-time constraint; persistent-state design gives a live, always-available reconstruction (graceful degradation) and its virtual-view query is a concrete lever for occluded-surface completion.
- **Perf/accuracy:** CVPR 2025; per-frame feed-forward, metric-scale pointmaps, RGB-only, no per-video optimization; 'revisiting' pass reprocesses frames for a refined final model.
- **Failure modes:** Online state can drift/accumulate error over long streams (no loop closure by default); virtual-view inference is a prior, not measured geometry — must be confidence-flagged; sensitive to dynamic content unless masked.
- **Alternatives:** Spann3R (memory-based online); StreamVGGT (streaming VGGT variant); VGGT windowed with sliding buffer; MonST3R for dynamic streams.

### MASt3R-SfM (metric) + MASt3R matching  
*Maturity: near-production*

- **What:** Fully-integrated feed-forward SfM: MASt3R's dense metric matching head + image-retrieval pairing + sparse global alignment, producing camera poses and a metric 3D reconstruction from unordered/low-overlap image collections.
- **Why it fits single-pass:** Best 'careful offline global solver' when the streaming pass needs tightening: robust on the low-overlap, limited-baseline sets that break classical incremental SfM, and its metric checkpoint gives real-scale output that anchors georeferencing without GCPs.
- **Perf/accuracy:** 3DV 2025; uses the '..._catmlpdpt_metric' checkpoint (metric regression + matching loss); retrieval + sparse_global_alignment pipeline; robust where COLMAP/GLOMAP fail on sparse overlap.
- **Failure modes:** Slower than pure feed-forward (global alignment step); still bounded scene scale; retrieval quality gates results; not a real-time engine.
- **Alternatives:** VGGT/Fast3R for speed; GLOMAP/COLMAP for a pure-geometry verification baseline; MapAnything for prior-fused metric SfM.

### Monocular metric depth prior — Depth Anything V2 (metric) / Metric3D v2 / UniDepth  
*Maturity: production*

- **What:** Large-scale monocular depth models; metric variants output real-scale per-pixel depth from a single image.
- **Why it fits single-pass:** The single-pass killer is scale/shape ambiguity in low-parallax (forward-motion) regions — a per-frame metric depth prior fills exactly that gap, provides an independent scale check for georeferencing, and is the regularizer that makes sparse-view Gaussian Splatting stop producing floaters.
- **Perf/accuracy:** Depth Anything V2: 25M–1.3B params; small (ViT-S ~25M) runs real-time (>30 FPS) on a GPU, ~10x faster than diffusion depth (Marigold); six metric models for indoor/outdoor. Metric3D v2/UniDepth predict metric depth + camera.
- **Failure modes:** Metric depth has a domain gap at drone altitudes/oblique views → biased scale unless fine-tuned or fused with RTK; relative variants need alignment; single-image depth is noisy on thin/reflective structures.
- **Alternatives:** Marigold/GeoWizard (diffusion, slower, higher detail); ZoeDepth; DepthPro; fuse multiple via median + confidence.

### InstantSplat (feed-forward-init 3D Gaussian Splatting)  
*Maturity: near-production*

- **What:** Pose-free sparse-view 3DGS: initializes Gaussians directly from a feed-forward model (MASt3R/DUSt3R) pointmap, then jointly optimizes scene + camera poses with co-visibility pruning and confidence-aware gradients — no COLMAP.
- **Why it fits single-pass:** Bridges the geometry backbone to a photorealistic, textured, renderable model using the same feed-forward init, and is explicitly designed for the extremely sparse views a single pass yields; produces measurable, visualization-ready output fast.
- **Perf/accuracy:** Reconstruction in seconds; claimed >30x faster than COLMAP+3DGS; demonstrated with as few as 3 views; supports 3DGS, 2DGS, and Mip-Splatting representations.
- **Failure modes:** Inherits scale/accuracy from the feed-forward init (not independently metric); very wide unseen areas still under-constrained; residual floaters without depth regularization.
- **Alternatives:** FSGS / DNGaussian (depth-regularized sparse GS); pixelSplat / MVSplat (feed-forward GS from 2–N views, cost-volume); DepthRegularized-GS; gsplat/Splatfacto with mono-depth loss.

### FSGS / DNGaussian (depth-regularized sparse-view Gaussian Splatting)  
*Maturity: research*

- **What:** Sparse-view 3DGS with monocular-depth regularization: FSGS uses Gaussian Unpooling to densify from sparse points guided by a pretrained depth estimator + augmented viewpoints; DNGaussian adds depth-normalized regularization.
- **Why it fits single-pass:** Directly targets the 3-to-few-view regime of a single pass; the depth prior + augmented-view supervision suppress the floaters and geometry collapse that plain 3DGS suffers from with limited angles, giving a cleaner mesh downstream.
- **Perf/accuracy:** Real-time rendering (100+ FPS at 3DGS quality); SOTA sparse-view PSNR on LLFF (3/6/9 views) and Mip-NeRF360 (16/24 views); FSGS reported >200 FPS rendering.
- **Failure modes:** Depth-prior bias propagates into geometry; still assumes reasonable coverage; per-scene optimization (minutes) not real-time to build.
- **Alternatives:** InstantSplat (faster init); Mip-Splatting (anti-aliasing); RegNeRF/FreeNeRF (NeRF sparse-view, slower).

### 2D Gaussian Splatting (2DGS) — surface/mesh extraction  
*Maturity: near-production*

- **What:** Collapses 3D Gaussians into oriented planar disks (surfels) with ray-splat intersection + depth-distortion/normal-consistency terms, giving view-consistent geometry from which a mesh is extracted by TSDF-fusing rendered depth.
- **Why it fits single-pass:** The problem demands textured meshes (not just point clouds) for measurement/analysis; 2DGS is the current sweet spot of accurate surfaces + fast training + real-time rendering, and pairs cleanly with the GS dense stage and metric georeferencing.
- **Perf/accuracy:** SIGGRAPH 2024; fast training (minutes) and real-time rendering; SOTA geometry among GS methods on DTU/Tanks&Temples; view-consistent normals enable clean Poisson/TSDF meshes.
- **Failure modes:** Thin structures/vegetation still hard; mesh quality follows GS density (sparse views → gaps); textureless facades under-constrained.
- **Alternatives:** GOF (Gaussian Opacity Fields, direct level-set mesh, SIGGRAPH Asia 2024); SuGaR (Poisson from GS level set, mesh in minutes); Neuralangelo (neural SDF, highest fidelity but hours); Open3D Poisson on fused point cloud.

### nvblox (real-time GPU TSDF + ESDF fusion)  
*Maturity: production*

- **What:** CUDA-accelerated TSDF/ESDF volumetric reconstruction library producing a live mesh + distance field, with C++/Python/ROS2 and NVIDIA Isaac integration.
- **Why it fits single-pass:** Provides the never-hard-fail, always-on live reconstruction: fuse per-frame metric depth/pointmaps into a growing mesh on a Jetson Orin onboard or edge box for immediate situational awareness (recon/disaster), independent of whether the heavy GS/mesh stage has converged. ESDF also gives obstacle clearance for the UAV.
- **Perf/accuracy:** Real-time TSDF+ESDF on GPU/Jetson; designed for robotic RGB-D; integrates with Isaac ROS; outputs mesh + 3D distance field.
- **Failure modes:** Needs metric depth input (from stereo/depth model/pointmap) — garbage-in on bad depth; voxel resolution trades detail vs memory; not photorealistic/textured to survey grade.
- **Alternatives:** Open3D ScalableTSDFVolume (CPU/GPU, easy); VDBFusion (OpenVDB, memory-efficient large scenes); Voxblox (CPU TSDF/ESDF for MAVs, ROS).

### Pi3 (permutation-equivariant visual geometry) + MapAnything  
*Maturity: research*

- **What:** Pi3 reconstructs point clouds + poses feed-forward WITHOUT a fixed reference view (permutation-equivariant), improving robustness/drift; MapAnything is a unified metric feed-forward transformer that optionally ingests intrinsics, poses, depth, or partial reconstructions and outputs depth maps + ray maps + poses + a metric scale factor.
- **Why it fits single-pass:** Pi3 removes the reference-frame dependence that hurts DUSt3R/VGGT on long drone strips (less drift). MapAnything is the ideal fusion point for the drone's OPTIONAL inputs (camera intrinsics, RTK poses, IMU, mono-depth) — it turns 'optional metadata' into a metric, globally consistent solution, directly attacking metric-accuracy-without-GCPs.
- **Perf/accuracy:** Pi3 (arXiv 2507.13347, 2025): reference-free, robust SOTA feed-forward geometry. MapAnything (2025): single model does uncalibrated SfM, MVS, mono-depth, localization, depth completion; directly regresses metric scene geometry + cameras.
- **Failure modes:** Both research-stage (evolving APIs/weights); metric claims still need aerial-domain validation; MapAnything's benefit depends on quality of the optional priors fed in.
- **Alternatives:** VGGT/CUT3R (mature-r); MASt3R-SfM (metric offline); Pow3R (prior-conditioned DUSt3R variant).

### Dynamic-object handling — MonST3R + SAM 2 masking  
*Maturity: research*

- **What:** MonST3R (Motion DUSt3R) estimates per-timestep pointmaps, per-frame poses/intrinsics, and video depth on dynamic scenes; SAM 2 + optical flow (RAFT) segments and masks moving vehicles/humans/animals before fusion.
- **Why it fits single-pass:** Directly addresses the 'dynamic objects corrupt geometry' challenge: masking transient objects before pointmap fusion / GS training keeps roads and structures clean, and MonST3R's motion-aware pose estimation is more robust than static-only models when the scene isn't perfectly static.
- **Perf/accuracy:** MonST3R (ICLR 2025 Spotlight): per-frame camera poses/intrinsics + video depth competitive with DepthCrafter/task-specific pose methods on dynamic scenes; feed-forward + flow-guided global optimization.
- **Failure modes:** Segmentation misses (partially occluded/small movers) leak into geometry; heavy dynamics (crowds/traffic) still degrade pose; extra compute per frame.
- **Alternatives:** Robust/confidence-weighted TSDF fusion to reject transients; explicit moving-object detector + track; DynaSLAM-style masking; semantic filtering (mask 'vehicle/person' classes).

### COLMAP / GLOMAP (classical SfM/MVS) — fallback & verification baseline  
*Maturity: production*

- **What:** Feature-based Structure-from-Motion + MVS; GLOMAP is a global-SfM variant an order of magnitude faster than COLMAP's incremental pipeline.
- **Why it fits single-pass:** Not the primary engine for single-pass, but essential as (a) a graceful-degradation fallback when learned models OOM/fail, (b) an independent geometric baseline to validate the feed-forward reconstruction's metric accuracy, and (c) a bundle-adjustment refiner (pyceres) for the final georeferenced solve.
- **Perf/accuracy:** COLMAP: robust, ubiquitous, but minutes-to-hours and needs ~60-80% overlap + texture; GLOMAP (ECCV 2024): ~10-100x faster global SfM on large sets, similar accuracy.
- **Failure modes:** Fails/produces broken or incomplete models on forward-motion low-overlap single-pass video (near-degenerate triangulation); poor on textureless/repetitive facades; slow dense MVS.
- **Alternatives:** OpenMVG+OpenMVS; hloc (deep features SuperPoint+SuperGlue/LightGlue front-end to rescue hard matches); Theia; feed-forward models (primary).

## Recommended stack

| Component | Choice | Rationale |
|---|---|---|
| Streaming geometry engine (near-real-time) | CUT3R for online per-frame pointmaps+poses; VGGT (windowed) as higher-accuracy chunk solver; VGGT-Long wrapper for long corridors (chunk + overlap-align + loop closure) | Feed-forward, works at tiny baselines/limited angles where COLMAP fails; CUT3R gives an always-available live model; VGGT-Long fixes memory/drift on drone strips and can swap in Pi3/MapAnything backends. |
| Offline global geometry / metric solver | MASt3R-SfM (metric checkpoint), optionally MapAnything for prior-fused metric SfM; refine with pyceres/GTSAM bundle adjustment | Careful global solution for accuracy pass; metric output anchors GCP-free georeferencing; BA folds in GPS/IMU priors. |
| Monocular depth prior | Depth Anything V2 (metric variant) primary; Metric3D v2 / UniDepth v2 as cross-check | Resolves scale/shape in low-parallax forward-motion regions; independent scale check; regularizer for sparse-view GS. Small model runs real-time. |
| Dynamic-object masking | SAM 2 + RAFT optical flow (mask vehicles/humans/animals); MonST3R when dynamics are heavy | Prevents transient objects from corrupting static terrain/structures before fusion; MonST3R adds motion-robust pose. |
| Georeferencing & metric scale (no GCPs) | Umeyama 7-DOF (Sim3) alignment of camera-center track to RTK/PPK GPS via evo/scipy; SE3 when metric model + RTK; GPS-prior BA; RTKLIB for PPK, NTRIP for live RTK | RTK camera centers give 1-3 cm horizontal / 2-5 cm vertical scale+datum without a GCP network; degrades to meter-level Sim3 on plain GPS, to baro/metric-depth scale if needed. |
| Live volumetric fusion (never-fail preview) | nvblox (Jetson Orin / Isaac ROS) onboard-edge; Open3D ScalableTSDF or VDBFusion on ground station | Always-on TSDF mesh + ESDF for situational awareness and obstacle clearance, independent of heavy stages; graceful degradation floor. |
| Photorealistic dense reconstruction | 3D Gaussian Splatting via gsplat/Nerfstudio (Splatfacto); InstantSplat for pose-free feed-forward init; FSGS/DNGaussian depth regularization for sparse views | Textured, measurable, real-time-renderable model from few views; depth reg kills floaters caused by limited angles. |
| Surface / textured mesh extraction | 2DGS primary; GOF or SuGaR alternatives; TSDF-fuse rendered depth then Poisson (Open3D); Neuralangelo only for offline max-fidelity | Watertight, view-consistent, textured mesh for measurement/analysis; minutes not hours; 2DGS best surface-vs-speed tradeoff. |
| Edge/onboard compute | NVIDIA Jetson Orin NX/AGX (onboard nvblox + depth); RTX 4090/A100-class ground station for VGGT/GS/mesh | Near-real-time preview onboard; heavy GS/mesh offloaded to ground station over the video/telemetry link. |
| Drone hardware & telemetry | DJI Matrice 350 RTK / Mavic 3 Enterprise RTK / Phantom 4 RTK (RTK + embedded SRT telemetry); or PX4/ArduPilot + RTK GPS + MAVSDK companion computer | RTK is the GCP-free metric enabler; DJI SRT gives per-frame lat/lon/alt/gimbal; open stack via MAVLink for custom platforms. |
| Geospatial I/O & visualization | GDAL (GeoTIFF DSM/ortho), PDAL/LAStools (LAS/LAZ with EPSG CRS), Cesium 3D Tiles / potree / OGC 3D Tiles for web viz, glTF/OBJ/OSGB mesh; OpenDroneMap for orthophoto/DSM rasterization | Standards-based, georeferenced, measurable outputs consumable by GIS/analysts; ODM covers the 2.5D ortho/DSM products classical pipelines expect. |
| Video ingest & frame conditioning | ffmpeg (I-frame extraction, SRT/KLV demux), variance-of-Laplacian blur rejection, parallax-based keyframe selection; MISB 0601 KLV parser for military FMV | Mitigates motion blur & H.264/H.265 compression artifacts; feeds clean keyframes to feed-forward models; supports full-motion-video metadata for recon. |

## Real-world integration

Drone platforms: DJI Matrice 350 RTK / Mavic 3 Enterprise RTK / Phantom 4 RTK expose per-frame telemetry as an SRT subtitle stream embedded alongside 4K H.264/H.265 video (lat/lon/ellipsoidal+relative alt, gimbal yaw/pitch/roll, focal/ISO); pull it via ffmpeg SRT demux and DJI PSDK/MSDK/Cloud API. Open platforms: PX4/ArduPilot + RTK GNSS + MAVLink/MAVSDK to a companion computer (Jetson Orin), logging pose at frame timestamps. Camera intrinsics from EXIF/XMP or a one-time checkerboard/Kalibr calibration; feed as the OPTIONAL intrinsics prior to VGGT/MapAnything/Pow3R. Time synchronization is critical: align GNSS/IMU to video frames via PTP or GPS timestamps and calibrate the GNSS-antenna→camera lever arm and gimbal offset — an uncorrected lever arm or 50-100 ms sync error directly biases metric accuracy. RTK corrections via NTRIP from a CORS/base station (live) or PPK post-processing with RTKLIB against RINEX logs (more robust for BVLOS). Georeference in WGS84 → project to UTM (EPSG:326xx/327xx) with EGM2008 geoid for orthometric heights. Onboard/edge: nvblox + Depth Anything V2 run on Jetson Orin via Isaac ROS 2 for a live TSDF mesh + ESDF; the heavy VGGT/Gaussian-Splatting/mesh stages run on an RTX/A100 ground station fed by the video+telemetry downlink or post-flight SD card. For military full-motion video, parse MISB 0601 KLV metadata for platform/sensor pose. Outputs plug into standard tooling: LAS/LAZ point clouds with embedded CRS (PDAL), GeoTIFF DSM/orthophoto (GDAL/ODM), Cesium/OGC 3D Tiles + glTF/OBJ/OSGB textured meshes for web and GIS (QGIS/ArcGIS/Cesium) visualization, measurement, and analysis.

## Reliability & failure handling

Design as a tiered ladder so the system always returns something useful. (1) Georeferencing tiers: RTK/PPK present → SE3 (metric model) or tight GPS-prior BA → cm-level metric georef; plain single-freq GPS → 7-DOF Umeyama Sim3 trajectory alignment → meter-level georef with scale cross-checked by metric mono-depth + barometric altitude; no GPS → relative reconstruction at arbitrary/estimated scale, explicitly flagged 'NOT georeferenced'. (2) Geometry tiers: streaming feed-forward (CUT3R/VGGT window) for the live preview → offline MASt3R-SfM/VGGT-Long global refine for accuracy → classical COLMAP/GLOMAP+hloc fallback if the learned model OOMs or diverges. (3) Always-on floor: an nvblox/Open3D TSDF is fused every frame from metric depth, so even if GS/mesh never converges the operator still gets a live mesh + obstacle field. (4) Confidence propagation: keep the models' per-point/per-pixel confidence plus multi-view consistency; render low-confidence and prior-completed (occluded/virtual-view/inpainted) regions in a distinct layer and exclude them from measurement by default — never present hallucinated geometry as measured. (5) Input conditioning: variance-of-Laplacian blur rejection, I-frame preference, and parallax-gated keyframe selection defend against motion blur and H.265 compression; drop rather than trust bad frames. (6) Dynamics: SAM 2 + flow masking before fusion; if masking is uncertain, robust/median + confidence-weighted TSDF integration rejects transient points. (7) Watchdog/backpressure: on a failed or OOM chunk, shrink the window, then skip and interpolate poses from IMU/GPS; cap memory with VGGT-Long disk-backed chunking; the mission pipeline never blocks or hard-crashes on a single bad segment.

## Single-pass specifics

Multi-pass photogrammetry (Pix4D, Agisoft Metashape, DJI Terra, Bentley ContextCapture, RealityCapture, OpenDroneMap) assumes a planned grid with 70-80% front/side overlap plus oblique cross-strips, then runs incremental feature-SfM + dense MVS in batch. A single pass violates every one of those assumptions, so the pipeline must change in specific ways: (1) Engine swap — replace triangulation-first incremental SfM (COLMAP) as the PRIMARY with feed-forward pointmap/pose regression (VGGT/CUT3R/MASt3R-SfM/Pi3/MapAnything). Forward flight puts the epipole inside the image and gives near-zero parallax there; depth uncertainty scales as ~1/baseline, so classical triangulation is degenerate exactly where single-pass geometry is thin — learned priors regress geometry without needing wide baselines. (2) Depth-prior injection — a metric monocular depth model (Depth Anything V2 metric / Metric3D v2) becomes a first-class input, not an afterthought, to constrain scale and shape in low-parallax regions; the dense stage must be depth-regularized GS (FSGS/DNGaussian/InstantSplat) or plain 3DGS collapses into floaters from the narrow view cone. (3) Metric scale/datum from RTK/PPK camera centers via Sim3/SE3 alignment + GPS-prior BA instead of a GCP network — the challenge explicitly bans extensive GCPs, and a single pass can't even see enough well-distributed GCPs. (4) Occlusion is fundamentally unrecoverable from one direction — building backs, undersides, and shadowed facades are never imaged, so they must be COMPLETED by learned priors (CUT3R virtual-view query, diffusion inpainting, Poisson hole-filling) and clearly flagged as inferred, not reconstructed. (5) Architecture — streaming/windowed processing with loop closure (CUT3R/VGGT-Long) replaces global batch bundle adjustment to hit near-real-time and handle long corridors, and dynamic objects must be masked before fusion because a single pass gives no second look to average them out. (6) Coverage-aware expectations — output a per-area confidence/coverage map; a single flight yields high-confidence nadir/near-track geometry and low-confidence oblique/occluded regions, and the product must communicate that rather than pretend uniform survey-grade accuracy.

## Open risks

- Aerial domain gap: VGGT/CUT3R/MASt3R/depth models are trained mostly on indoor, object-centric, and driving data — high-altitude nadir/oblique drone imagery is out-of-distribution, so metric accuracy at flight altitude is unproven and may require fine-tuning on aerial data (UseGeo, ISPRS Vaihingen/Potsdam, WildRGB, or synthetic/AirSim/Blender renders).
- Proving metric accuracy WITHOUT GCPs is the crux and is not free: it demands RTK/PPK + accurate lever-arm and time-sync calibration + at least a few independent survey checkpoints to VALIDATE (even if not to constrain) — otherwise you cannot claim cm-level to a customer.
- Scale drift over long single-pass corridors: feed-forward models anchored to a reference frame accumulate scale/pose drift; loop closure (VGGT-Long/VGGT-SLAM) helps only if the flight revisits areas, which a pure single pass may not.
- True real-time full-fidelity GS/mesh on edge hardware is not yet feasible — realistic target is near-real-time live TSDF preview + minutes-scale offline GS/mesh refinement; over-promising real-time textured meshes is a trap.
- Hallucination risk: feed-forward models and generative occlusion-completion produce plausible-but-wrong geometry in unseen/low-confidence regions — dangerous for measurement, infrastructure inspection, and military mission planning unless rigorously confidence-flagged and excluded from measurement.
- Video degradation: H.265 compression blocking, rolling-shutter skew on fast UAVs, and motion blur degrade correspondence and pointmap quality; rolling-shutter modeling and aggressive keyframe selection may be needed.
- Memory/scale limits: VGGT OOMs beyond hundreds of views and VGGT-Long needs tens of GiB of disk per few-thousand frames — resource planning and chunking add real engineering cost.
- Licensing/commercialization: some foundation-model weights (e.g., VGGT) carry research/non-commercial licenses; a product path needs commercial weights or permissively licensed alternatives (MASt3R/DUSt3R terms vary).

## Differentiation notes

Off-the-shelf drone-mapping software — OpenDroneMap/WebODM, Pix4Dmapper, Agisoft Metashape, DJI Terra, Bentley ContextCapture, Esri Drone2Map, RealityCapture — is uniformly classical multi-pass feature-SfM + dense MVS. Against this problem it has four disqualifying limits: (1) it requires a planned overlapping grid with oblique passes and degrades to broken/incomplete models on single-pass forward-motion low-overlap video (near-degenerate triangulation); (2) it is batch, taking tens of minutes to hours with no near-real-time or streaming preview — unusable for live recon/disaster response; (3) it leans on GCP networks for best metric accuracy, which the problem forbids; (4) it has no learned prior to fill low-parallax or occluded regions. Our differentiation: (a) feed-forward geometric foundation models (VGGT/CUT3R/MASt3R-SfM/Pi3/MapAnything) reconstruct from the tiny baselines and limited angles that make COLMAP-based tools fail; (b) a metric monocular-depth prior + depth-regularized Gaussian Splatting beat sparse-view floaters and give textured, measurable output where classical MVS holes out; (c) a genuinely streaming architecture (CUT3R + nvblox live TSDF) delivers near-real-time situational awareness on-drone/edge — no commercial photogrammetry package does this; (d) RTK/PPK-driven Sim3/SE3 georeferencing achieves metric scale + datum WITHOUT a GCP network; (e) an explicit graceful-degradation ladder (RTK→GPS→relative; feed-forward→classical fallback; always-on TSDF floor) means it never hard-fails on a mission; (f) learned occlusion completion with confidence layers, absent from all classical tools. Consumer NeRF/GS SaaS (Luma, Polycam, Scaniverse) is also non-georeferenced, non-metric, non-streaming, and not built for kilometer-scale corridor drone data — so it is not a substitute either.

## Citations / references

- Wang et al., VGGT: Visual Geometry Grounded Transformer, CVPR 2025 (Best Paper) — vgg-t.github.io
- Wang et al., CUT3R: Continuous Updating Transformer for 3D Reconstruction, CVPR 2025 — cut3r.github.io
- Yang et al., Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass, CVPR 2025 — fast3r-3d.github.io
- Wang et al., DUSt3R: Geometric 3D Vision Made Easy, CVPR 2024
- Leroy et al., MASt3R: Grounding Image Matching in 3D, ECCV 2024; Duisterhof et al., MASt3R-SfM, 3DV 2025 — github.com/naver/mast3r
- Yang et al., Pi3: Scalable Permutation-Equivariant Visual Geometry Learning, arXiv 2507.13347 (2025) — yyfz.github.io/pi3
- MapAnything: Universal Feed-Forward Metric 3D Reconstruction, 2025 — map-anything.github.io
- Deng et al., VGGT-Long: kilometer-scale monocular reconstruction (chunk + loop closure), 2025 — github.com/DengKaiCQ/VGGT-Long
- Maggio et al. (MIT-SPARK), VGGT-SLAM: Dense RGB SLAM on the SL(4) Manifold, 2026
- Zhang et al., MonST3R: A Simple Approach for Estimating Geometry of Dynamic Scenes, ICLR 2025 (Spotlight) — monst3r-project.github.io
- Wang et al., Spann3R: 3D Reconstruction with Spatial Memory, 2024; Jang et al., Pow3R (prior-conditioned), 2025
- Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
- Fan et al., InstantSplat: Sparse-view Gaussian Splatting in Seconds, 2024 — instantsplat.github.io
- Zhu et al., FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting, arXiv 2312.00451 (ECCV 2024) — zehaozhu.github.io/FSGS
- Li et al., DNGaussian: Depth-Normalized sparse-view GS, CVPR 2024
- Charatan et al., pixelSplat, CVPR 2024; Chen et al., MVSplat, ECCV 2024
- Huang et al., 2D Gaussian Splatting for Geometrically Accurate Radiance Fields, SIGGRAPH 2024, arXiv 2403.17888 — surfsplatting.github.io
- Guédon & Lepetit, SuGaR: Surface-Aligned Gaussian Splatting, CVPR 2024 — anttwo.github.io/sugar
- Yu et al., Gaussian Opacity Fields (GOF), SIGGRAPH Asia 2024; Li et al., Neuralangelo, CVPR 2023
- Yang et al., Depth Anything V2, arXiv 2406.09414; Metric3D v2; Piccinelli et al., UniDepth
- NVIDIA nvblox (GPU TSDF+ESDF, Isaac ROS) — github.com/nvidia-isaac/nvblox; Open3D ScalableTSDFVolume; Oleynikova et al., Voxblox; VDBFusion
- Schönberger & Frahm, COLMAP, CVPR 2016; Pan et al., GLOMAP: Global Structure-from-Motion Revisited, ECCV 2024; hloc (SuperPoint+LightGlue)
- OpenDroneMap/WebODM — opendronemap.org; GDAL, PDAL, Cesium 3D Tiles; Ravi et al., SAM 2, Meta 2024
- Nerfstudio gsplat/Splatfacto; RTKLIB (PPK/NTRIP)
