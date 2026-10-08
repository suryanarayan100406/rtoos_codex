# Research dossier — sota-currency

> Auto-generated from the research workflow. Source material for document authoring.

## Executive summary

The field has shifted decisively (2024-2026) from optimization-based photogrammetry to FEED-FORWARD geometry transformers that regress cameras + dense pointmaps in one pass, which is exactly the primitive a single-pass drone video needs. DUSt3R/MASt3R (2024) started it; VGGT (CVPR 2025 Best Paper) made it a foundation model reconstructing scenes in <1s; and as of late 2025/2026 the SOTA leader is Depth Anything 3 (Nov 2025), which beats VGGT by ~36-44% on camera pose and ~25% on geometry and even emits 3D Gaussians for rendering. For THIS problem the highest-value recent work is the "prior-injectable" family — MapAnything (3DV 2026) and Pow3R (CVPR 2025) — because they ingest the drone's GPS/RTK pose, IMU, intrinsics and even sparse depth as optional guidance and output METRIC geometry, which is the cleanest path to metric accuracy without GCPs. Near-real-time single-pass is covered by streaming/online variants (CUT3R, Point3R, StreamVGGT) that update a recurrent 3D state per frame, and by VGGT-Long/VGGT-SLAM which extend VGGT to kilometer-scale flights with chunking and loop closure. Metric scale is anchored by monocular metric-depth models (Metric3D v2, UniDepth V2, Apple Depth Pro at 2.25 MP in 0.3s, MoGe-2). Textured deliverables come from feed-forward/pose-free Gaussian Splatting (NoPoSplat, AnySplat, MVSplat, InstantSplat) converted to mesh via 2DGS or SuGaR. The two load-bearing risks are (1) LICENSING — VGGT's commercial checkpoint explicitly EXCLUDES military use and DUSt3R/MASt3R/UniDepth are non-commercial (CC BY-NC), a hard blocker for an NTRO/defense deployment unless CC-BY/Apache alternatives (Depth Anything, MapAnything, Pi3, Metric3D) are used; and (2) AERIAL GENERALIZATION — almost every model is trained on object-centric/indoor/driving data, so nadir/high-altitude/oblique imagery is out-of-distribution and will likely need domain fine-tuning.

### VGGT / VGGT-Long / VGGT-SLAM (+ VGGT-Omega 2026)  
*Maturity: near-production*

- **What:** Feed-forward transformer that from 1..hundreds of images directly regresses intrinsics+extrinsics, depth maps, dense pointmaps and 2D tracks in a single pass, no bundle adjustment. VGGT-Long (ICRA 2026) chunks + loop-closes it to kilometer-scale outdoor sequences with NO calibration/depth/retraining; VGGT-SLAM aligns submaps over the SL(4) homography manifold for uncalibrated video; VGGT-Omega (May 2026) is the successor and a May-2026 memory fix processes ~2-3x more frames per GPU.
- **Why it fits single-pass:** This is the core primitive for single-pass drone video: it turns an unposed/loosely-posed video into a consistent camera track + dense 3D from limited viewing angles without exhaustive matching or hours of global optimization. VGGT-Long directly targets the long, single-path outdoor trajectory a survey drone flies.
- **Perf/accuracy:** Scene reconstruction in <1s (feed-forward); ~1B params; Co3D AUC@30 ~90; VGGT-Long validated on KITTI/Waymo/Virtual-KITTI at km scale with loop closure.
- **Failure modes:** Base model OOMs beyond tens-to-low-hundreds of frames (mitigated by VGGT-Long chunking / VGGT-SLAM submaps / StreamVGGT / the 2026 memory fix); pointmaps not guaranteed metric; assumes largely static scenes; aerial nadir imagery is out-of-distribution.
- **Alternatives:** Depth Anything 3 (now beats it), Fast3R (1000+ imgs one pass), Pi3, MapAnything, CUT3R.

### Depth Anything 3 (DA3)  
*Maturity: near-production*

- **What:** Nov-2025 any-view geometry foundation model built on a plain DINOv2 backbone with a single depth-ray prediction target; predicts spatially consistent geometry from an arbitrary number of views with or without poses, does camera pose + monocular/any-view depth, and can drive novel-view synthesis via a 3DGS head. Trained exclusively on public academic datasets.
- **Why it fits single-pass:** Current SOTA and a single model that spans the whole pipeline (pose + geometry + rendering) — attractive as one backbone for a graceful-degradation drone system, and the public-data provenance is licensing-friendlier than VGGT for defense use.
- **Perf/accuracy:** New SOTA across pose+geometry+rendering benchmarks; reported to beat VGGT by ~35.7-44.3% on camera pose accuracy and ~25.1% on geometric accuracy; detail on par with Depth Anything V2; community TensorRT/ROS2 real-time integrations already appearing.
- **Failure modes:** Very new (Nov 2025) so field-hardening/robustness at scale unproven; metric scale not the headline claim (community metric-depth variants exist); license terms on weights still to confirm; no aerial-specific validation reported.
- **Alternatives:** VGGT/VGGT-Omega, Pi3, MapAnything.

### MapAnything  
*Maturity: near-production*

- **What:** Unified feed-forward transformer (3DV 2026) that takes one-to-many images PLUS any optional geometric inputs — camera intrinsics, poses, depth, or partial reconstructions — and outputs METRIC 3D geometry and cameras in one pass via a factored depth+ray+pose+scale representation. Released CC BY 4.0.
- **Why it fits single-pass:** Best structural match to the drone metadata problem: it consumes exactly the mandatory/optional inputs the challenge lists (GPS-derived pose, IMU, barometric-anchored scale, calibrated intrinsics, RTK/PPK) as guidance and returns metric, globally consistent geometry — the cleanest route to metric accuracy WITHOUT GCPs. Permissive license suits defense deployment.
- **Perf/accuracy:** Reported to outperform or match specialist feed-forward models across uncalibrated SfM, calibrated MVS, monocular depth, localization and depth completion while training more efficiently; metric-scale output.
- **Failure modes:** Very recent; aerial-domain performance unquantified; quality of metric scale still depends on quality of injected priors (noisy GPS degrades it).
- **Alternatives:** Pow3R (also prior-injectable), CUT3R (metric, online), Metric3D v2 for scale.

### Pow3R  
*Maturity: research*

- **What:** CVPR 2025 DUSt3R-style model that accepts any subset of auxiliary priors (intrinsics, relative pose, sparse or dense depth) at inference, trained by randomly dropping modalities so it works with full, partial, or no priors; also supports native-resolution inference and point-cloud completion.
- **Why it fits single-pass:** Lets you feed drone intrinsics + GPS/IMU relative pose + sparse depth (from RTK baselines or a metric-depth model) as guidance rather than solving from scratch, improving robustness under limited viewing angles and GPS noise, and degrading gracefully when a prior is missing.
- **Perf/accuracy:** Reported SOTA on 3D reconstruction, depth completion, multi-view depth, MVS, and multi-view pose estimation (numbers in paper).
- **Failure modes:** Code via NAVER Labs (check license — NAVER 3D releases are typically CC BY-NC, a defense blocker); pairwise DUSt3R lineage needs global alignment for many frames; aerial OOD.
- **Alternatives:** MapAnything (metric + more inputs, CC BY), Pi3.

### CUT3R / Point3R / StreamVGGT (online streaming)  
*Maturity: research*

- **What:** Online recurrent feed-forward reconstructors that maintain a persistent 3D state and update it per incoming frame. CUT3R (CVPR 2025) uses a stateful recurrent model outputting METRIC pointmaps and can probe virtual unobserved views to infer unseen regions; Point3R (2025) keeps an explicit spatial-pointer memory tied to 3D structure; StreamVGGT (2025/26) distills VGGT into a causal-attention transformer that caches historical keys/values as memory for efficient long streaming.
- **Why it fits single-pass:** Directly delivers the near-real-time / streaming requirement: process the video as it arrives with bounded per-frame cost instead of buffering the whole flight, and CUT3R's virtual-view probing partially addresses occluded/unseen surfaces from a single flight path.
- **Perf/accuracy:** CUT3R: metric-scale online pointmaps, competitive/SOTA across 3D/4D tasks; StreamVGGT: competitive with dense VGGT while enabling online inference with FlashAttention-style ops; specific FPS not published.
- **Failure modes:** Recurrent state drifts over long sequences without loop closure; accuracy typically below full bidirectional batch models; static-scene bias; memory growth for very long flights.
- **Alternatives:** VGGT-Long (batch+loop closure), Spann3R (spatial-memory incremental), MapAnything.

### MASt3R-SfM (+ MASt3R / DUSt3R)  
*Maturity: near-production*

- **What:** Feed-forward matching foundation (MASt3R adds a metric-accurate local feature head to DUSt3R) turned into a full, unconstrained SfM pipeline: uses the model as both matcher and image retriever to cut complexity from quadratic to linear, then does low-memory global alignment. Handles ordered or unordered, low-overlap image sets.
- **Why it fits single-pass:** Robust, well-understood FALLBACK/refinement path when pure feed-forward geometry is uncertain — it recovers accurate poses and metric matches even with little motion or overlap (the single-pass regime), and integrates with classic COLMAP-style tooling for a verifiable geometry backbone.
- **Perf/accuracy:** Steady across scales, strongest at small/medium scale; linear (vs quadratic) complexity via learned retrieval; metric matching head.
- **Failure modes:** CC BY-NC-SA 4.0 — NON-COMMERCIAL, a hard licensing blocker for NTRO/defense; slower than single-shot VGGT-class models; global alignment still an optimization step; aerial OOD.
- **Alternatives:** GLOMAP/COLMAP (classical, permissive), VGGT (faster feed-forward), MapAnything.

### Metric monocular depth: Metric3D v2 / UniDepth V2 / Apple Depth Pro / MoGe-2  
*Maturity: production*

- **What:** Zero-shot single-image METRIC depth (and normals/point maps). Metric3D v2 (TPAMI 2024) canonicalizes camera models to resolve metric ambiguity across 16M+ images; UniDepth V2 (TPAMI 2025) self-prompts a dense camera representation and outputs uncertainty; Apple Depth Pro (ICLR 2025) yields sharp boundaries and estimates focal length with no metadata; MoGe-2 (2025) outputs metric-scale point maps with fine detail.
- **Why it fits single-pass:** Provides the METRIC SCALE ANCHOR that pose+geometry transformers often lack, and a per-frame dense-depth prior that is robust when only one flight path / limited parallax is available. Depth Pro's focal-length estimation covers the case of unknown/missing intrinsics; uncertainty outputs feed graceful degradation.
- **Perf/accuracy:** Depth Pro: 2.25 MP depth in 0.3s on one GPU, SOTA sharpness + focal-length; Depth Anything V2 (feeder): 25M-1.3B params, >10x faster than diffusion-depth; Metric3D v2: zero-shot metric across thousands of camera models; UniDepth V2: strong zero-shot over 10 datasets.
- **Failure modes:** UniDepth V2 is CC BY-NC-SA (non-commercial); absolute scale error still several % (not survey-grade); large altitude ranges/sky/water/thin structures degrade; per-frame depth is not multi-view consistent without fusion.
- **Alternatives:** Depth Anything V2 metric heads (fast, permissive small models), fuse depth into VGGT/MapAnything instead of standalone.

### Pi3 (permutation-equivariant visual geometry)  
*Maturity: research*

- **What:** July-2025 (rev Mar 2026) feed-forward network that predicts affine-invariant poses and scale-invariant local point maps with a FULLY PERMUTATION-EQUIVARIANT architecture — no fixed reference view, so results are invariant to input frame ordering. CC BY 4.0.
- **Why it fits single-pass:** Removes the reference-frame fragility of DUSt3R/VGGT (which fail if the anchor view is poor) — valuable for drone video where any single keyframe may be motion-blurred or texture-poor; robustness to ordering suits unordered/looped flight segments.
- **Perf/accuracy:** Reported SOTA on camera pose, monocular/video depth, and dense point-map reconstruction; robustness to input permutation.
- **Failure modes:** Outputs affine/scale-invariant (needs a metric anchor); very new; aerial OOD; speed not published here.
- **Alternatives:** VGGT, Depth Anything 3, MapAnything.

### Pose-free / uncalibrated feed-forward Gaussian Splatting: NoPoSplat & AnySplat  
*Maturity: research*

- **What:** Feed-forward models that output 3D Gaussians directly from UNPOSED images. NoPoSplat (ICLR 2025) predicts Gaussians in one view's canonical space from sparse unposed images, real-time, trained only with photometric loss, using an intrinsics token for scale. AnySplat (2025) scales to ANY number of uncalibrated views, predicting Gaussians + intrinsics + extrinsics in one shot. Both CC BY 4.0.
- **Why it fits single-pass:** Produces the TEXTURED, photorealistic deliverable directly and tolerates the inaccurate/missing poses typical of drone GPS — no COLMAP pre-step — enabling near-real-time novel-view visualization for mission planning/inspection.
- **Perf/accuracy:** NoPoSplat: real-time reconstruction, SOTA NVS among pose-free methods, strong under low overlap; AnySplat: matches pose-aware baselines zero-shot across sparse and dense views, huge rendering-latency reduction vs optimization NeRF.
- **Failure modes:** Gaussians are for rendering, not directly metric/measurable (need mesh/point-cloud export + scale); trained on ground-level datasets (RealEstate10K/DL3DV) so aerial OOD; view-count/coverage limits completeness of occluded facades.
- **Alternatives:** MVSplat (cost-volume, 22fps), InstantSplat (opt+foundation init), pixelSplat, Splatt3R (MASt3R+GS).

### MVSplat / InstantSplat (few-view Gaussian Splatting)  
*Maturity: near-production*

- **What:** MVSplat (ECCV 2024): cost-volume feed-forward Gaussians from sparse posed views. InstantSplat (2024, rev 2025): initializes from a geometric foundation model (DUSt3R-style) then does Gaussian bundle-adjustment optimizing scene + poses, avoiding SfM.
- **Why it fits single-pass:** Practical, mature few-view GS for the sparse-coverage single-pass regime; InstantSplat's foundation-model init pairs naturally with the VGGT/MASt3R geometry backbone to go from video to renderable model in seconds/minutes.
- **Perf/accuracy:** MVSplat: 22 fps feed-forward, 10x fewer params and >2x faster than pixelSplat, SOTA on RealEstate10K/ACID; InstantSplat: >30x reconstruction speedup, SSIM 0.376->0.762 vs SfM+3DGS.
- **Failure modes:** MVSplat needs poses/intrinsics; both degrade outside training domain (aerial); limited extrapolation beyond captured views; static-scene assumption.
- **Alternatives:** NoPoSplat/AnySplat (pose-free), FSGS/DNGaussian (3-view NVS).

### Gaussian-to-mesh: 2D Gaussian Splatting (2DGS) & SuGaR  
*Maturity: near-production*

- **What:** Surface-extraction from Gaussian fields. 2DGS (SIGGRAPH 2024) collapses 3D Gaussians into view-consistent 2D oriented disks with depth-distortion + normal-consistency terms for accurate TSDF meshing. SuGaR (CVPR 2024) adds a surface-alignment regularizer then Poisson-reconstructs an editable mesh in minutes.
- **Why it fits single-pass:** Converts the GS radiance field into the TEXTURED MESH / measurable surface the challenge requires for measurement, analysis, and digital-twin/GIS export — bridging the photorealistic-render output to CAD/GIS-consumable geometry.
- **Perf/accuracy:** 2DGS: noise-free detailed geometry with real-time rendering; SuGaR: editable mesh in minutes vs hours for neural-SDF methods, better render quality.
- **Failure modes:** Mesh quality depends on Gaussian coverage — single-pass occluded backs produce holes; thin/vegetation structures noisy; scale inherited from upstream (must be metric before meshing).
- **Alternatives:** Gaussian Opacity Fields (GOF), classical TSDF fusion of metric depth, Poisson on fused point cloud.

## Recommended stack

| Component | Choice | Rationale |
|---|---|---|
| Primary feed-forward geometry backbone | Depth Anything 3 (SOTA, public-data license) OR VGGT-1B (mature, huge ecosystem) as the geometry core | One forward pass gives poses + dense pointmaps/depth from limited-angle video; DA3 is current accuracy leader and licensing-friendlier for defense, VGGT is the battle-tested option with the richest tooling (COLMAP export, SLAM/Long extensions). |
| Metric scale + drone-metadata fusion | MapAnything or Pow3R to ingest intrinsics/GPS-pose/IMU/RTK/sparse-depth as priors and emit metric geometry | This is how you hit metric accuracy WITHOUT GCPs — inject the mandatory GPS + optional RTK/PPK/IMU/intrinsics as guidance rather than solving pose from scratch; MapAnything is CC BY 4.0 and outputs metric directly. |
| Metric-depth anchor / per-frame prior | Metric3D v2 or Apple Depth Pro (permissive) as the scale anchor; Depth Anything V2 small (Apache) for a fast robust relative-depth prior | Fixes absolute scale and stabilizes reconstruction under low parallax; Depth Pro also estimates focal length when intrinsics are missing. Avoid UniDepth V2 for deployment (non-commercial). |
| Near-real-time streaming path | CUT3R or StreamVGGT for online per-frame updates; VGGT-Long / VGGT-SLAM for full-flight batch with chunking + loop closure | Meets the near-real-time constraint and bounds drift over a long single-pass trajectory; CUT3R's virtual-view probing partially fills unseen/occluded regions. |
| Robust SfM fallback (graceful degradation) | GLOMAP/COLMAP as permissive classical fallback; MASt3R-SfM only for R&D (non-commercial license) | When feed-forward confidence is low on a segment, fall back to verifiable classical geometry so the pipeline never hard-fails; keep a permissive option for the shippable build. |
| Textured output (render) | AnySplat / NoPoSplat (pose-free feed-forward 3DGS) or InstantSplat (foundation-init + GS-BA) | Produces the textured, photorealistic model for visualization/mission-planning without a COLMAP pre-step and tolerates imperfect drone pose. |
| Measurable mesh / point cloud | 2DGS or SuGaR for GS-to-mesh; export georeferenced LAS/LAZ point cloud + glTF/OBJ mesh + Cesium 3D Tiles | Turns radiance field into CAD/GIS-consumable, measurable surfaces for measurement, analysis and digital-twin delivery. |
| Dynamic-object handling | MonST3R-style dynamic pointmaps and/or a segmentation mask (YOLO/SAM) to exclude vehicles/humans/animals from static reconstruction | Directly addresses the dynamic-objects challenge; static geometry stays clean while moving objects are masked or tracked separately. |
| Edge/compute | Ground-station RTX/A-series GPU for near-real-time; NVIDIA Jetson Orin for degraded onboard preview (small models: DA V2-S, TensorRT) | 1B-param feed-forward models are impractical fully onboard; a stream-to-ground architecture gives near-real-time full quality with an onboard low-res preview fallback. |

## Real-world integration

INGEST: pull the live video as RTSP/RTP H.264/H.265 (DJI/most UAVs) or SRT; decode with hardware NVDEC. TELEMETRY: read GPS/IMU/barometric altitude/gimbal + camera intrinsics over MAVLink (ArduPilot/PX4) or the vendor SDK (DJI PSDK/Cloud API, Autel), and from video SEI/SRT metadata and image EXIF. SYNC: frames must be time-aligned to telemetry — use the drone's GPS/PTP timestamps or SEI-embedded per-frame time; interpolate pose to exact frame time (telemetry is typically 10-50 Hz vs 30-60 fps video). CALIBRATION: use factory intrinsics if present, else estimate per-frame focal length with Depth Pro or self-calibrate with VGGT/DA3; account for rolling shutter and lens distortion (fisheye vs pinhole). PRIORS INTO MODEL: feed intrinsics + GPS-derived relative pose + RTK/PPK-corrected positions + IMU orientation + sparse metric depth into MapAnything/Pow3R as guidance. GEOREFERENCING: the feed-forward reconstruction is in an arbitrary local frame — estimate a 7-parameter Helmert/similarity transform (scale, rotation, translation) from the flight's GPS/RTK camera positions to map the model into UTM/ECEF (WGS84); RTK/PPK gives cm-level camera positions that fix both scale and datum without GCPs. OUTPUT FORMATS: georeferenced point cloud as LAS/LAZ (with CRS/EPSG), textured mesh as OBJ/glTF/GLB, DSM/DEM as GeoTIFF, streamable web viz as Cesium 3D Tiles / potree, and COLMAP/NeRFStudio format for interop (VGGT already exports COLMAP). DEPLOY: containerized GPU service (Docker + CUDA/TensorRT), ROS2 nodes for live pose+depth streaming, gRPC/REST for chunked results; store per-point/per-face confidence for downstream QA. Interop with existing photogrammetry (Metashape/Pix4D/OpenDroneMap) by handing them the feed-forward poses as a warm start.

## Reliability & failure handling

Design as a confidence-gated degradation LADDER, never a single hard path. (1) Every stage emits confidence: VGGT/DA3 give per-pixel confidence, metric-depth models give uncertainty (UniDepth/Depth Pro) — mask or down-weight low-confidence points instead of dropping the frame. (2) FRAME QUALITY GATE: reject motion-blurred frames via Laplacian-variance / no-reference blur score and select sharp keyframes; drop frames with telemetry dropouts. (3) SCALE FALLBACK CHAIN: prefer RTK/PPK-derived metric scale -> else metric-depth model (Metric3D v2/Depth Pro) -> else GPS-baseline triangulation between camera centers -> else deliver relative-scale model flagged 'non-metric'. (4) GEOMETRY FALLBACK: if feed-forward confidence on a chunk is low or it OOMs, fall back to classical GLOMAP/COLMAP for that segment, or to monocular metric-depth + pose-from-telemetry TSDF fusion, so a segment always yields best-effort geometry. (5) DRIFT CONTROL: bound long-flight drift with VGGT-Long chunk overlap + lightweight loop closure or VGGT-SLAM submap alignment; cross-check against GPS trajectory and correct. (6) DYNAMIC OBJECTS: segment and exclude vehicles/humans/animals (SAM/YOLO) from the static model; optionally reconstruct them separately. (7) OCCLUSION: tag single-flight back-faces as holes with confidence rather than hallucinating unmarked geometry; optionally use CUT3R virtual-view inference or generative inpainting flagged as 'inferred'. (8) ALWAYS emit a partial georeferenced point cloud with a quality report (coverage %, estimated metric error, confidence heatmap) even on partial failure — the system degrades to a coarser but valid product, never a crash. (9) Watchdog + graceful timeout on the GPU service; onboard low-res preview continues if the ground link drops.

## Single-pass specifics

Traditional multi-pass photogrammetry (COLMAP/RealityCapture/Metashape/Pix4D) assumes many overlapping wide-baseline images, exhaustive pairwise matching, and hours of global bundle adjustment — it structurally CANNOT work from one flight path with limited angles, thin parallax, and near-real-time deadlines, and it hard-fails when matches are sparse. The single-pass regime forces five concrete changes: (1) Replace SfM+MVS+BA with FEED-FORWARD geometry regression (VGGT/DA3/MapAnything) that infers cameras+dense geometry in one pass and does not diverge on low overlap. (2) Inject METADATA AS PRIORS, not as things to solve — GPS/RTK pose, IMU, barometric altitude and intrinsics go INTO the network (Pow3R/MapAnything) to pin scale and pose, which is how you get metric accuracy without GCPs. (3) Use MONOCULAR METRIC DEPTH to supply geometry that multi-view triangulation can't recover from a single-sided trajectory, and to anchor absolute scale. (4) Process ONLINE/STREAMING (CUT3R/StreamVGGT/Point3R) or in overlapping chunks with loop closure (VGGT-Long) instead of a single global optimization, to meet near-real-time and long-trajectory needs. (5) ACCEPT AND MARK INCOMPLETENESS: a single pass sees mostly one side of buildings/terrain, so occluded facades and back-faces must be left as confidence-tagged holes or explicitly inferred (CUT3R virtual views, generative inpainting) — never silently filled. Learned matching (MASt3R) replaces hand-crafted features to survive low-overlap, motion blur, and compression artifacts, and pose-free GS (NoPoSplat/AnySplat) removes the COLMAP pre-step so texturing tolerates GPS-noisy poses. Net: trade some absolute metric precision and completeness for robustness, occlusion-awareness, and speed — with a classical fallback for verifiability.

## Open risks

- LICENSING IS A HARD BLOCKER FOR NTRO/DEFENSE: VGGT's commercial checkpoint license EXPLICITLY EXCLUDES military applications, and DUSt3R/MASt3R/MASt3R-SfM/Pow3R (NAVER) and UniDepth V2 are CC BY-NC / NC-SA (non-commercial). For a defense deployment you must build on permissive/public-data models: Depth Anything V2 (small=Apache), Depth Anything 3 (public-data, confirm weight license), MapAnything (CC BY 4.0), Pi3 (CC BY 4.0), Metric3D v2 (CC BY 4.0), NoPoSplat/AnySplat (CC BY 4.0), or fully classical GLOMAP/COLMAP + OpenDroneMap. This choice must be made up front.
- AERIAL/NADIR GENERALIZATION: nearly every feed-forward model is trained on object-centric (Co3D), indoor (ScanNet), or driving (KITTI/Waymo) and ground-level NVS (RealEstate10K/DL3DV) data. High-altitude nadir/oblique aerial imagery — repetitive texture, homogeneous terrain, thin structures (power lines, antennas, railings), water and dense vegetation, extreme depth range — is out-of-distribution and will likely require fine-tuning on aerial datasets (e.g. UseGeo, WHU, Mill-19/UrbanScene3D, ISPRS) before metric claims hold.
- METRIC ACCURACY WITHOUT GCPs IS BOUNDED: feed-forward/monocular metric scale still carries several-percent error; RTK/PPK camera positions can fix scale+datum to cm at the camera centers but do NOT guarantee survey-grade accuracy across the reconstructed surface — needs empirical validation against GCP/LiDAR ground truth and clear accuracy reporting per deliverable.
- COMPUTE/MEMORY vs REAL-TIME: 1B-param models (VGGT/DA3) exceed practical onboard budgets and OOM beyond ~tens-to-low-hundreds of frames; true onboard real-time is unrealistic — plan a stream-to-ground-station architecture with an onboard low-res preview, and rely on chunking (VGGT-Long), streaming (StreamVGGT/CUT3R) and the May-2026 VGGT memory fix. Even so, 'near-real-time' (seconds-to-minutes lag), not hard real-time, is the honest claim.
- DYNAMIC SCENES: most models assume static geometry; vehicles/humans/animals create ghosting unless explicitly masked (SAM/YOLO) or handled by dynamic variants (MonST3R/CUT3R). This is unsolved at production quality.
- BLEEDING-EDGE MATURITY: the strongest options (Depth Anything 3 Nov-2025, VGGT-Omega/StreamVGGT/Pi3 2026, MapAnything 3DV-2026) are months old with limited independent validation, evolving APIs, and little aerial track record — integration risk is real; pin versions and keep a mature fallback (VGGT + GLOMAP).
- OCCLUSION FROM SINGLE PASS: one flight path fundamentally cannot see building backs/undersides — completeness is capped; inferred/inpainted surfaces must be clearly flagged and are unsafe for measurement.

## Differentiation notes

Off-the-shelf drone-mapping products (DJI Terra, Pix4Dmapper/Pix4Dreact, Agisoft Metashape, RealityCapture, Bentley ContextCapture, OpenDroneMap) are MULTI-PASS photogrammetry: they need dense overlapping stills/flight grids, run exhaustive matching + global bundle adjustment for minutes-to-hours offline, and hard-fail or produce holes when overlap/parallax is low — they cannot ingest a single video pass, cannot run near-real-time, and their metric accuracy depends on GCPs or RTK. Their strength (survey-grade accuracy with GCPs) is exactly what this problem forbids the reliance on. This system differentiates by (1) FEED-FORWARD geometry that survives limited viewing angles and low overlap in one pass; (2) treating GPS/RTK/IMU/intrinsics as network PRIORS (MapAnything/Pow3R) to reach metric scale WITHOUT GCPs; (3) NEAR-REAL-TIME streaming/online reconstruction during or right after the flight (CUT3R/StreamVGGT/VGGT-Long) versus hours of offline processing; (4) OCCLUSION- and DYNAMIC-AWARE, confidence-tagged output with a graceful-degradation ladder that never hard-fails; and (5) an integrated video->geometry->metric-scale->textured-mesh/point-cloud->georeferenced-GIS pipeline built from the very latest (2025-2026) foundation models rather than classical SfM. Pix4Dreact does fast 2D orthos in real time but not accurate textured 3D; NeRF/GS SaaS (Luma, Polycam) render nicely but are not metric/georeferenced. The defensible edge is the specific fusion — feed-forward transformers + metric-depth anchoring + metadata priors + streaming + GS-to-mesh + graceful degradation — none of which any single off-the-shelf tool provides for the single-pass constraint, plus a deliberate permissive-license stack that makes defense deployment legal.

## Citations / references

- DUSt3R: Geometric 3D Vision Made Easy (CVPR 2024, arXiv:2312.14132)
- MASt3R: Grounding Image Matching in 3D (ECCV 2024, arXiv:2406.09756)
- MASt3R-SfM: Unconstrained Structure-from-Motion (arXiv:2409.19152, Sep 2024)
- VGGT: Visual Geometry Grounded Transformer (CVPR 2025 Best Paper, arXiv:2503.11651; VGGT-Omega + memory fix, May 2026; VGGT-1B-Commercial checkpoint, Jul 2025, non-military)
- Fast3R: 3D Reconstruction of 1000+ Images in One Forward Pass (CVPR 2025, arXiv:2501.13928)
- CUT3R: Continuous 3D Perception Model with Persistent State (CVPR 2025, arXiv:2501.12387)
- Spann3R: 3D Reconstruction with Spatial Memory (3DV 2025, arXiv:2408.16061)
- Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors (CVPR 2025, arXiv:2503.17316, NAVER Labs)
- Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory (arXiv:2507.02863, 2025)
- StreamVGGT: Streaming 4D Visual Geometry Transformer (arXiv:2507.11539, 2025, rev Mar 2026)
- VGGT-Long: Chunk it, Loop it, Align it — kilometer-scale (ICRA 2026, arXiv:2507.16443)
- VGGT-SLAM: Dense RGB SLAM via SL(4) submap alignment (arXiv:2505.12549, May 2025)
- Pi3: Scalable Permutation-Equivariant Visual Geometry Learning (arXiv:2507.13347, 2025, rev Mar 2026)
- MapAnything: Universal Feed-Forward Metric 3D Reconstruction (3DV 2026, arXiv:2509.13414, CC BY 4.0)
- Depth Anything 3: Recovering Visual Space from Any Views (arXiv:2511.10647, Nov 2025; beats VGGT ~36-44% pose, ~25% geometry)
- Depth Anything V2 (NeurIPS 2024, arXiv:2406.09414)
- Metric3D v2: Geometric Foundation Model for Zero-shot Metric Depth+Normal (TPAMI 2024, arXiv:2404.15506)
- UniDepth / UniDepthV2: Universal Monocular Metric Depth (CVPR 2024 / TPAMI 2025, arXiv:2502.20110, CC BY-NC-SA)
- Depth Pro: Sharp Monocular Metric Depth in Less Than a Second (Apple, ICLR 2025, arXiv:2410.02073)
- MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details (arXiv:2507.02546, 2025)
- pixelSplat (CVPR 2024, arXiv:2312.12337)
- MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images (ECCV 2024, arXiv:2403.14627; 22fps, 10x fewer params than pixelSplat)
- InstantSplat: Sparse-view Gaussian Splatting in Seconds (arXiv:2403.20309, 2024 rev 2025; >30x speedup)
- FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting (ECCV 2024, arXiv:2312.00451)
- DNGaussian: Depth-Normalized Sparse-view Gaussian Splatting (CVPR 2024)
- NoPoSplat: No Pose, No Problem — pose-free feed-forward Gaussians (ICLR 2025, arXiv:2410.24207, CC BY 4.0)
- AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views (arXiv:2505.23716, 2025, CC BY 4.0)
- Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs (2024)
- 2D Gaussian Splatting for Geometrically Accurate Radiance Fields (SIGGRAPH 2024, arXiv:2403.17888)
- SuGaR: Surface-Aligned Gaussian Splatting for Fast Mesh Extraction (CVPR 2024, arXiv:2311.12775)
- MonST3R: A Simple Approach for Estimating Geometry of Dynamic Scenes (2024)
