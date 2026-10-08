# Research dossier — AI / Deep Learning (learned priors for single-pass metric 3D reconstruction)

> Auto-generated from the research workflow. Source material for document authoring.

## Executive summary

Single-pass drone video breaks classical photogrammetry because a single flight path gives thin baselines and narrow angles, so wide-baseline triangulation is ill-conditioned and there is no redundant overlap for bundle adjustment or GCP-free scaling. The fix is to inject LEARNED PRIORS that supply the shape and absolute scale geometry cannot recover from limited views. The recommended architecture is a three-layer learned stack: (1) a monocular METRIC depth model (Metric3D v2, UniDepthV2, or MoGe-2/Depth Pro) run per-keyframe to anchor absolute scale and predict intrinsics/FOV without calibration; (2) a feed-forward video-geometry backbone (VGGT for batch keyframe chunks, or CUT3R / MASt3R-SLAM for true streaming) that jointly regresses poses + dense pointmaps directly, no COLMAP loop; (3) temporally consistent video depth (Video Depth Anything) plus learned MVS refinement (CasMVSNet/PatchmatchNet) wherever real parallax exists. Metric scale WITHOUT GCPs comes from fusing per-frame metric depth + IMU preintegration for local scale, then a 7-DoF Sim(3) Umeyama alignment of the visual trajectory to the GNSS track (RTK/PPK when available) for absolute georeferencing — the GPS track acts as continuous soft control points. Every learned output must emit a confidence/uncertainty map (UniDepthV2, Metric3D, VGGT, MoGe all provide one) so poorly constrained pixels are downweighted, enabling graceful degradation and honest accuracy reporting. On a Jetson Orin, the deployable real-time path is Depth Anything V2 ViT-S / Metric3D-S exported to TensorRT INT8/FP16 (~15-30 FPS at reduced res), with heavy transformer backbones (VGGT, MASt3R) run in near-real-time on keyframe chunks or offloaded. The single most under-appreciated risk is licensing: the strongest checkpoints (DA-V2 Large, UniDepth, Video Depth Anything Large, MoGe, Depth Pro) ship under non-commercial/research licenses — plan the production path around Apache-2.0 (DA-V2 Small) and BSD (Metric3D) weights.

### Monocular METRIC depth as scale anchor: Metric3D v2 / UniDepthV2  
*Maturity: production*

- **What:** Single-image zero-shot METRIC (absolute-scale) depth + surface normals + intrinsics/FOV prediction. Metric3D v2 maps any camera into a canonical camera space to achieve zero-shot metric output; UniDepthV2 predicts a metric point cloud AND camera intrinsics directly from RGB with a confidence map.
- **Why it fits single-pass:** This is the core reason single-pass metric reconstruction is possible without GCPs: it supplies absolute scale per-frame where triangulation from a thin baseline cannot. UniDepthV2 predicting its own intrinsics covers the 'camera intrinsics optional' input case. Normals feed man-made-structure planar regularization.
- **Perf/accuracy:** Metric3D v2 ViT-L: KITTI AbsRel 0.044 / d1 0.985, NYU d1 0.989; ViT-giant2 KITTI AbsRel 0.039. UniDepthV2 zero-shot d1: KITTI 98.9, NYU 98.8, huge gains on ETH3D 85.2 and IBims 94.5. Metric3D BSD-2 license (deployable); UniDepth CC-BY-NC (research only).
- **Failure modes:** Metric scale error typically 5-10% and DOMAIN-SHIFTED for aerial nadir/oblique views (training is mostly automotive/indoor) → expect scale bias on rooftops/terrain; wrong focal length badly distorts the point cloud; per-frame scale is noisy and drifts, so it must be globally re-anchored, never trusted frame-by-frame.
- **Alternatives:** Depth Pro (Apple): 2.25MP metric depth in 0.3s with SOTA focal-length estimation and razor-sharp boundaries (sample-code license). MoGe-2: metric point maps + normals + FOV, ~60ms/img A100 FP16 ViT-L. Depth Anything V2 metric variants (indoor Hypersim / outdoor VKITTI).

### Feed-forward video geometry backbone: VGGT  
*Maturity: near-production*

- **What:** A single feed-forward transformer (CVPR 2025 Best Paper) that from 1, a few, or hundreds of views directly regresses camera intrinsics+extrinsics, depth maps, dense point maps, and 3D point tracks — replacing the COLMAP SfM + MVS pipeline with one network pass.
- **Why it fits single-pass:** Directly solves the thin-baseline problem: it learns multi-view geometry priors instead of triangulating, so it stays well-conditioned when angles are narrow. Outputs poses AND geometry AND tracks in one shot, which is exactly the SfM+MVS product photogrammetry needs — but in near-real-time on keyframe chunks.
- **Perf/accuracy:** Reconstructs a scene in <1 second (feed-forward), hundreds of views; ~1.2B params. May-2026 memory fix allows ~2-3x more frames per GPU budget. Commercial checkpoint Co3D AUC@30 90.37. Needs an Ampere+ GPU (bf16); heavy VRAM — chunk keyframes on Jetson-class hardware.
- **Failure modes:** VRAM scales with view count (batch, not infinite streaming); very long single-pass flights must be tiled into overlapping keyframe windows then stitched; up-to-scale unless fused with metric depth/GNSS; 3D viz rendering adds tens of seconds (separate from inference).
- **Alternatives:** CUT3R: online recurrent 'persistent state', linear memory, 4-64 views at 512px — true streaming. MASt3R-SLAM: real-time dense monocular SLAM with the metric MASt3R prior, runs on RTX 4090, works with --no-calib. Fast3R / Spann3R / Fast3R-style multi-view DUSt3R successors for many-image batches.

### Temporally consistent video depth: Video Depth Anything  
*Maturity: near-production*

- **What:** Extends Depth Anything V2 with temporal attention to produce flicker-free, temporally consistent depth over arbitrarily long videos; metric variants and an experimental training-free streaming mode that caches temporal-attention hidden states for frame-by-frame inference.
- **Why it fits single-pass:** Per-frame monocular depth flickers, which poisons multi-frame fusion into a point cloud/TSDF. Temporal consistency is what makes fusing a moving-drone depth stream into one coherent metric model actually work. Streaming mode fits near-real-time on-drone use.
- **Perf/accuracy:** Single A100, 1x32x518x518 FP16: Small 7.5ms (28.4M params), Large 14ms (381.8M). Metric models trained on VKITTI+IRS. Streaming trades accuracy (ScanNet d1 0.926→0.836). Small = Apache-2.0 (deployable); Base/Large = CC-BY-NC.
- **Failure modes:** Metric variant trained on driving/synthetic data → aerial domain gap; streaming mode's accuracy drop; still relative-consistent, absolute scale must come from the metric variant + external anchoring.
- **Alternatives:** ChronoDepth, DepthCrafter (diffusion-based, higher quality but far slower, not real-time). Marigold (diffusion, single-image, high detail, slow — good for offline hero shots not streaming).

### Metric scale-alignment / georeferencing WITHOUT GCPs  
*Maturity: production*

- **What:** Fusion recipe, not a single model: (a) use per-keyframe metric depth to seed local scale; (b) IMU preintegration + barometric altitude to constrain inter-frame scale and gravity direction; (c) 7-DoF Sim(3) Umeyama alignment of the visual trajectory to the GNSS track (RTK/PPK when present) for absolute georeferencing; (d) a per-keyframe scale-correction term in a sliding-window factor graph to kill drift.
- **Why it fits single-pass:** Directly answers 'metric accuracy WITHOUT extensive GCPs.' The continuous GPS track supplies thousands of soft control points along the single pass; gravity from IMU fixes 2 of 3 rotational DoF; metric depth removes the global scale gauge freedom that monocular SfM cannot resolve.
- **Perf/accuracy:** RTK/PPK gives cm-level georef; single-point GPS ~2-5m absolute but sub-decimeter RELATIVE after Sim(3)+IMU fusion. Umeyama is closed-form least-squares (microseconds). Scale drift reducible to ~1-2% with metric-depth priors vs 5-15% for pure monocular VIO.
- **Failure modes:** GPS multipath/outages near structures corrupt the alignment (RANSAC the correspondences, gate on GNSS covariance); time sync error between video frames and GPS/IMU directly becomes position error at drone speed; barometer drifts with weather.
- **Alternatives:** GTSAM/Ceres sliding-window factor graph fusing VIO+GNSS+scale; OpenVINS / VINS-Fusion for the VIO front-end; loosely-coupled Sim(3) vs tightly-coupled GNSS factors.

### Uncertainty / confidence estimation on every learned output  
*Maturity: near-production*

- **What:** Per-pixel confidence maps and predictive uncertainty from the models themselves (UniDepthV2, Metric3D, VGGT, MoGe, MASt3R all emit confidence), plus test-time augmentation / multi-view depth agreement as an external consistency check.
- **Why it fits single-pass:** Under limited single-pass angles, large regions are weakly constrained; confidence maps are what let the system DOWNWEIGHT them in fusion, avoid emitting garbage geometry, report honest per-region accuracy, and degrade gracefully instead of hard-failing. Essential for measurement/analysis credibility.
- **Perf/accuracy:** Confidence is a native model output (no extra net for UniDepth/Metric3D/VGGT/MASt3R). Multi-view depth-agreement variance and left-right/flip TTA give calibrated-enough gating thresholds; cross-model depth disagreement flags occlusion/dynamic regions.
- **Failure modes:** Learned confidence is often poorly CALIBRATED (overconfident on textureless/repetitive facades and on out-of-domain aerial content); needs empirical threshold tuning on drone data; confidence ≠ correctness under systematic domain bias.
- **Alternatives:** Monte-Carlo dropout, deep ensembles (too slow on-device), photometric/geometric reprojection consistency as a model-agnostic confidence proxy for fusion weighting (TSDF weight = f(confidence, ray angle, blur)).

### Dynamic-object handling: MonST3R + semantic segmentation  
*Maturity: research*

- **What:** MonST3R estimates geometry+poses+intrinsics for DYNAMIC video and produces static/dynamic segmentation feed-forward; pair with fast semantic/instance segmentation (YOLO-seg, SAM2, or a lightweight panoptic net) to mask vehicles/humans/animals before fusion.
- **Why it fits single-pass:** Directly addresses the 'dynamic objects' challenge: moving cars/people otherwise smear into the terrain and corrupt both pose estimation and the mesh. Masking them keeps the static reconstruction clean and the trajectory accurate.
- **Perf/accuracy:** MonST3R outputs a time-varying point cloud + per-frame poses + static/dynamic labels (ICLR 2025). YOLOv8/11-seg runs real-time on Orin via TensorRT (>30 FPS at 640px) to provide the dynamic mask cheaply.
- **Failure modes:** MonST3R is heavy/research-grade for on-device real-time; segmentation misses partially occluded or unusual objects; stopped-but-movable objects (parked vs moving car) are ambiguous; fast small objects between keyframes.
- **Alternatives:** Simple flow/reprojection-residual outlier rejection in the factor graph; robust kernels (Huber/Cauchy) in fusion; SAM2 video propagation for coherent masks; class-based masking of 'movable' categories regardless of motion.

### Semantic + planar/normal priors for man-made structure regularization  
*Maturity: near-production*

- **What:** Use predicted surface normals (Metric3D v2, MoGe) and semantic labels (building/road/vegetation) to impose Manhattan-world / piecewise-planar constraints on facades, rooftops, and roads, and to drive category-specific meshing.
- **Why it fits single-pass:** Single-pass occlusion leaves gaps; planar/symmetry priors let you complete and regularize facades and rooftops that were only grazingly seen. Semantics separate the five required output classes (terrain, facades/roofs, roads, vegetation, obstacles) and let vegetation be modeled as point cloud while buildings become clean planar meshes.
- **Perf/accuracy:** Normals come free from Metric3D v2/MoGe. Planar RANSAC + normal snapping typically cuts facade/roof RMS noise substantially and yields watertight, measurable planes vs raw noisy points.
- **Failure modes:** Over-regularization erases real detail (ornamentation, curved roofs); Manhattan assumption breaks on organic/curved architecture and rubble (disaster scenes); semantic errors propagate into wrong geometry priors.
- **Alternatives:** PolyFit / KSR polygonal surface reconstruction for buildings; learned shape/scene completion priors for occluded volumes; footprint extrusion from segmented roof polygons + GIS building footprints when available.

### Learned MVS refinement where parallax exists: CasMVSNet / PatchmatchNet  
*Maturity: production*

- **What:** Cascade cost-volume learned multi-view stereo that refines depth to high metric accuracy on frame subsets that DO have usable baseline (oblique passes, altitude changes), given the poses from the feed-forward backbone.
- **Why it fits single-pass:** Where the single pass does provide real parallax, learned MVS gives sharper, more metrically accurate depth than pure monocular priors, fusing prior + geometry. Memory-efficient cascade formulation suits high-res 4K frames and constrained VRAM.
- **Perf/accuracy:** PatchmatchNet DTU overall 0.352mm (Acc 0.427 / Comp 0.277); designed to cut memory+runtime for high-res MVS. Requires known poses, per-view depth range, and pair.txt view selection (all provided by the VGGT/SLAM front-end).
- **Failure modes:** Degrades exactly where single-pass hurts most — tiny baselines make the cost volume flat/ambiguous; needs a decent depth-range prior (supply from monocular metric depth); textureless facades still fail.
- **Alternatives:** Vis-MVSNet (visibility-aware, better for occlusion), the 2025 diffmvs (repo states more efficient + better than PatchmatchNet), MVSNet/CasMVSNet baselines. Or skip explicit MVS entirely and rely on the feed-forward pointmaps + TSDF fusion.

### Feed-forward Gaussian Splatting for texture & novel views: MVSplat  
*Maturity: research*

- **What:** Predicts 3D Gaussians from sparse (2-3) posed views in a single feed-forward pass via a cost-volume encoder, giving photorealistic textured novel-view synthesis without per-scene NeRF/3DGS training.
- **Why it fits single-pass:** Delivers the 'textured' and 'visualization' deliverables in near-real-time. Feed-forward (no minutes-long per-scene optimization) fits the real-time constraint; works from sparse views, matching single-pass sparsity. Complements the metric mesh with a photoreal digital-twin view.
- **Perf/accuracy:** ECCV 2024 Oral; feed-forward from N=2-3 context views; built on UniMatch cost-volume + pixelSplat pipeline, more efficient than pixelSplat.
- **Failure modes:** Sparse-view generalization to wide aerial scenes is unproven; Gaussians are not directly measurable geometry (need mesh extraction for metrology); domain gap from indoor/object training sets to large outdoor terrain.
- **Alternatives:** Standard 3DGS/2DGS trained per-scene offline (higher quality, not real-time); NeRF/Instant-NGP for hero renders; classic textured Poisson/Delaunay meshing from the fused point cloud for the measurable deliverable (pair photoreal GS view + metric mesh).

### On-device inference optimization: TensorRT + INT8/FP16 + distillation  
*Maturity: production*

- **What:** Export models to ONNX → TensorRT engines with FP16/INT8 quantization and layer fusion; distill large depth backbones into ViT-S students; run the pipeline via DeepStream/GStreamer with NVDEC hardware video decode on Jetson Orin.
- **Why it fits single-pass:** This is what makes the whole learned stack actually run near-real-time on a drone/ground-station GPU. Depth Anything's design (heavy teacher → small student) is built for exactly this distillation path; TensorRT is the standard deployment route for NVIDIA embedded.
- **Perf/accuracy:** Depth Anything TensorRT on RTX 4090 FP16 (518px): ViT-S 3ms, ViT-B 6ms, ViT-L 12ms incl. pre/post. Jetson AGX Orin (275 TOPS INT8, 64GB) est. ViT-S depth ~15-30 FPS at reduced res; Orin NX (100 TOPS, 16GB) ~1/2-1/3 of that (est. scaled from desktop). NVDEC decodes 4K H.265 in hardware freeing the GPU.
- **Failure modes:** INT8 needs representative calibration data (aerial imagery) or accuracy drops; transformer attention/dynamic-shape ops can be unsupported or slow in TensorRT (plugin/opset issues); large models (VGGT 1.2B, UniDepth ViT-L) exceed Orin NX VRAM → must run ViT-S/B or offload to ground station.
- **Alternatives:** NVIDIA Triton for server-side, torch-tensorrt, ONNX Runtime with TensorRT EP; structured pruning; run heavy backbone on ground station over the video link and keep only lightweight depth/segmentation on the drone (edge-server split).

## Recommended stack

| Component | Choice | Rationale |
|---|---|---|
| Metric depth anchor (deployable license) | Metric3D v2 (ViT-S/L, BSD-2) — plus UniDepthV2 or Depth Pro for R&D/benchmarking only (CC-BY-NC / sample license) | Metric3D v2 gives zero-shot metric depth + normals with a permissive BSD-2 license, so it can ship in a product; UniDepthV2/Depth Pro are stronger on some metrics but non-commercial. Provides absolute scale + normals + intrinsics — the GCP-free scale foundation. |
| Feed-forward geometry backbone | VGGT for batch keyframe-chunk reconstruction; MASt3R-SLAM or CUT3R for streaming/online | Replaces COLMAP SfM+MVS with a single well-conditioned network pass that stays stable under thin single-pass baselines; VGGT for quality on chunks, MASt3R-SLAM/CUT3R for real-time incremental operation with metric priors. |
| Temporally consistent depth stream | Video Depth Anything (Small metric, Apache-2.0) with streaming mode | Flicker-free depth is required for clean multi-frame fusion; the Small variant is Apache-2.0 and fast (7.5ms FP16 A100), and streaming mode fits on-drone near-real-time. |
| Sensor fusion / georeferencing | VINS-Fusion or OpenVINS front-end + GTSAM/Ceres sliding-window factor graph + closed-form Sim(3) Umeyama to GNSS/RTK | Fuses metric-depth scale + IMU + barometer + GPS track into a drift-corrected, georeferenced trajectory WITHOUT GCPs; RANSAC-gated GNSS factors handle GPS noise/outages. |
| Dynamic-object masking | YOLO11-seg (real-time, Orin TensorRT) for movable-class masks; SAM2 for coherent video propagation; MonST3R for R&D dynamic geometry | Cheap real-time masking of vehicles/humans/animals prevents them corrupting pose and mesh; SAM2 gives temporally coherent masks; MonST3R is the research-grade fallback for heavy dynamic scenes. |
| Fusion & meshing | Confidence-weighted TSDF (Open3D / VDBFusion / nvblox on Jetson) → Poisson/Delaunay mesh + planar snapping; optional MVSplat/3DGS photoreal view | TSDF weighted by learned confidence + ray geometry gives graceful degradation and a measurable metric mesh; nvblox is GPU-accelerated for Jetson; planar priors clean facades/roofs; 3DGS adds the photoreal digital-twin visualization. |
| Edge inference runtime | NVIDIA Jetson AGX Orin 64GB + TensorRT (FP16/INT8) + DeepStream/GStreamer NVDEC; Triton on a ground station for heavy backbones | AGX Orin (275 TOPS) runs ViT-S depth + segmentation in real-time; hardware NVDEC decodes the 4K/1080p drone stream; split heavy VGGT/MASt3R to a ground-station GPU over the link when VRAM/latency demands. |
| Camera intrinsics handling | Use EXIF/metadata when present; otherwise UniDepthV2 / MoGe / Depth Pro self-predicted intrinsics + focal length, refined in the factor graph | Directly covers the 'intrinsics optional' input: the learned models estimate FOV/focal length so the pipeline never hard-requires calibration, and estimates are refined online. |

## Real-world integration

VIDEO INGEST: decode the drone H.264/H.265 stream in hardware via NVDEC through GStreamer/DeepStream (4K HEVC decodes in real-time, freeing the GPU for inference); pull frames as NV12/RGB CUDA buffers with zero-copy to TensorRT. TIME SYNC is the make-or-break integration detail: video frames must be timestamped against GPS/IMU on a common clock — use the drone's embedded telemetry. For tactical/NTRO-relevant platforms this is MISB KLV metadata muxed into the MPEG-TS (STANAG 4609) carrying per-frame sensor lat/lon/alt, platform attitude, and gimbal angles; for civilian/PX4/ArduPilot drones it is MAVLink CAMERA_TRIGGER/GLOBAL_POSITION_INT and the DJI SDK/SRT telemetry sidecar. Interpolate GPS/IMU to each frame timestamp; unsynchronized data becomes position error at flight speed (10 m/s = 10cm per 10ms of skew). MODEL SERVING: export every model to ONNX (opset 17+), build per-model TensorRT engines with FP16 default and INT8 for the ViT-S depth/segmentation nets (calibrate INT8 on real aerial frames, not COCO); wrap in a DeepStream pipeline or Triton (edge-server split — lightweight nets on the Jetson, VGGT/MASt3R on a ground-station RTX GPU over the datalink). GEOREFERENCING OUTPUT: emit georeferenced products in standard formats — LAS/LAZ or COPC for point clouds (with CRS EPSG code, typically UTM zone from GPS), OBJ/glTF/3D Tiles for textured meshes, GeoTIFF DSM/orthomosaic, and PLY for Gaussians — so they open directly in QGIS, CloudCompare, Cesium, Potree, and existing GIS. CALIBRATION: accept camera intrinsics from EXIF/metadata, else self-calibrate via the learned FOV heads, refining in the factor graph; apply lens distortion undistortion pre-inference. HARDWARE TARGET: Jetson AGX Orin 64GB (Orin NX 16GB as the constrained tier) for on-drone/on-vehicle; a laptop/ground-station RTX 4090-class GPU for near-real-time full-quality passes.

## Reliability & failure handling

Design as a DEGRADATION LADDER so the system never hard-fails, only lowers fidelity: (1) full path = feed-forward backbone (VGGT/MASt3R-SLAM) + metric depth + MVS refinement + GNSS fusion; (2) if the backbone OOMs or the GPU is saturated, drop to monocular metric depth (Metric3D-S) + VIO-only pose, tiling/downscaling frames to fit VRAM rather than crashing; (3) if visual tracking is lost (motion blur, low texture, tunnel/occlusion), dead-reckon pose from IMU preintegration + barometer and resume when features return; (4) if GNSS drops out, continue in local metric frame and re-anchor georeferencing when signal returns. CONFIDENCE-GATED FUSION: every learned depth/point output carries a confidence map; TSDF integration weight = f(model confidence, ray-incidence angle, blur score) so weakly constrained single-pass regions are downweighted, not blindly meshed — and the final product ships with a per-region uncertainty/accuracy layer for honest measurement. INPUT QC: run a cheap blur/exposure detector (variance-of-Laplacian, over/under-exposure histograms) to skip or downweight motion-blurred and blown-out frames before they enter fusion. CROSS-CHECK: disagreement between monocular metric depth and multi-view geometry flags occlusion, dynamic objects, or domain-shift failure — those pixels are masked. WATCHDOGS: per-stage timeouts and frame-drop policies keep the pipeline real-time under load (process latest keyframe, drop backlog) instead of accumulating unbounded lag. Everything runs bounded-memory (sliding window / linear-memory streaming models like CUT3R) so a long single-pass flight cannot exhaust RAM.

## Single-pass specifics

What MUST change versus traditional multi-pass photogrammetry (Pix4D/RealityCapture/COLMAP+MVS): (1) NO wide-baseline triangulation — a single flight path gives thin baselines and near-parallel rays, so classical dense MVS cost volumes go flat/ambiguous; you MUST inject learned monocular + multi-view priors (VGGT/Metric3D/UniDepth) that supply geometry from a single or few views instead of triangulating everything. (2) NO redundant overlap / loop closure / large bundle adjustment — multi-pass relies on many overlapping strips and loop closures to constrain scale and drift; single-pass has none, so replace global BA with a feed-forward backbone + sliding-window VIO/GNSS factor graph and IMU preintegration. (3) METRIC SCALE cannot come from redundant views or GCPs — it MUST come from monocular METRIC depth (absolute scale per-frame) + IMU + a Sim(3) fit to the continuous GPS track (the track is the GCP substitute). (4) OCCLUSION is permanent — you only see each surface from one direction, so backsides/undersides of buildings are never observed; you MUST add shape/scene-completion priors, planar/Manhattan and symmetry regularization, and footprint-extrusion to plausibly complete occluded surfaces (and clearly flag them low-confidence). (5) UNCERTAINTY becomes first-class — with limited angles a large fraction of pixels are weakly constrained, so per-pixel confidence must gate fusion and be reported, unlike multi-pass where redundancy hides weak observations. (6) REAL-TIME feed-forward, not offline optimization — replace the hours-long COLMAP→MVS→NeRF-training loop with single-pass networks producing results as the drone flies. (7) DYNAMIC objects can't be voted out by many views — a moving car seen once smears in, so explicit segmentation/masking is mandatory, not optional. Net: photogrammetry's redundancy-based geometry is replaced by prior-based geometry + tight sensor fusion + explicit uncertainty and completion.

## Open risks

- LICENSING is the top deployment risk: the strongest checkpoints (Depth Anything V2 Base/Large, UniDepth/UniDepthV2, Video Depth Anything Base/Large, MoGe, Depth Pro sample code) are CC-BY-NC or research/sample licenses — NOT usable in a shipped/commercial or possibly govt product. Production must be built on Apache-2.0 (DA-V2 Small, Video-DA Small) and BSD-2 (Metric3D) weights, or licenses negotiated; verify VGGT's checkpoint license before relying on it.
- DOMAIN GAP: nearly all these models are trained on ground-level, automotive, or indoor data; aerial nadir/oblique drone views (rooftops, terrain, high-altitude scale) are out-of-distribution, so published KITTI/NYU accuracy numbers will NOT transfer directly — expect scale bias and require fine-tuning/validation on real drone footage.
- METRIC SCALE DRIFT over a long single pass: per-frame metric depth is 5-10% off and noisy; without careful IMU+GNSS anchoring and per-keyframe scale correction, the model accumulates scale drift that violates the 'metrically accurate' requirement.
- VRAM / COMPUTE on embedded: VGGT (~1.2B), UniDepth/MASt3R ViT-L exceed Jetson Orin NX (16GB) budgets and are near-real-time at best even on AGX Orin — the heavy backbone likely must run on a ground station, adding datalink-latency and connectivity dependence.
- CONFIDENCE MISCALIBRATION: learned confidence maps are often overconfident on textureless facades and out-of-domain content, so uncertainty-based graceful degradation needs empirical recalibration on drone data or it will silently emit wrong-but-confident geometry.
- DYNAMIC-OBJECT LEAKAGE: segmentation misses partially occluded / unusual / stopped-movable objects; residual dynamics corrupt pose and mesh, and MonST3R-class dynamic handling is research-grade and heavy for real-time.
- TensorRT PORTABILITY: transformer attention and dynamic shapes in these models can hit unsupported/slow ops or need custom plugins during ONNX→TensorRT INT8 conversion, and INT8 needs representative aerial calibration data — a real engineering-time risk for the hackathon-to-product path.
- EVALUATION/GROUND TRUTH: proving metric accuracy without GCPs requires some independent check (a few survey points, RTK, or a reference LiDAR/photogrammetry scan) — otherwise accuracy claims are unverifiable.

## Differentiation notes

Off-the-shelf photogrammetry (Pix4D, DJI Terra, RealityCapture, Agisoft Metashape, OpenDroneMap, bare COLMAP+OpenMVS) is fundamentally MULTI-PASS and OFFLINE: it needs 70-80% front/side image overlap from a grid/orbit flight, GCPs (or RTK) for metric accuracy, and minutes-to-hours of SfM+dense-MVS+meshing — and it HARD-FAILS on a single linear pass because the sparse-then-dense triangulation is under-constrained (thin baseline → failed/holey reconstruction, unresolved scale). This solution differs by replacing redundancy-based geometry with LEARNED-PRIOR geometry: monocular metric depth supplies absolute scale from single views, feed-forward transformers (VGGT/MASt3R/CUT3R) regress poses+dense points in one pass instead of iterative BA, and tight IMU+GNSS fusion (Sim(3) to the GPS track) georeferences WITHOUT GCPs — so it produces a usable metric model from exactly the input photogrammetry rejects. It runs NEAR-REAL-TIME/on-device (TensorRT on Jetson) rather than offline on a workstation, DEGRADES GRACEFULLY via a confidence-gated fallback ladder instead of failing, explicitly masks DYNAMIC objects (which photogrammetry smears), and emits PER-PIXEL UNCERTAINTY so measurements come with honest error bounds. Versus naive 'run COLMAP on the video frames': COLMAP will typically fail to register or diverge on low-parallax, motion-blurred single-pass video, whereas the learned backbone is trained to stay stable there. The trade-off to be honest about: for a well-planned multi-pass survey with GCPs, classical photogrammetry still wins on raw metric precision — this system's value is producing a good-enough georeferenced metric model from the constrained single-pass, near-real-time, no-GCP scenario the problem statement mandates.

## Citations / references

- VGGT: Visual Geometry Grounded Transformer, CVPR 2025 (Best Paper), facebookresearch/vggt
- Depth Anything V2, 2024 (ViT-S 24.8M Apache-2.0 / B 97.5M / L 335M CC-BY-NC; metric variants) — DepthAnything/Depth-Anything-V2
- Video Depth Anything, CVPR 2025 Highlight (Small 28.4M Apache-2.0, metric + streaming) — DepthAnything/Video-Depth-Anything
- Metric3D v2: A Versatile Monocular Geometric Foundation Model, TPAMI 2024 (BSD-2; ViT-L KITTI AbsRel 0.044) — YvanYin/Metric3D
- UniDepth (CVPR 2024) & UniDepthV2 (arXiv:2502.20110, 2025), CC-BY-NC — lpiccinelli-eth/UniDepth
- Depth Pro: Sharp Monocular Metric Depth in Less Than a Second, Apple 2024 (2.25MP in 0.3s) — apple/ml-depth-pro
- MoGe / MoGe-2 (metric scale) / MoGe-3, Microsoft 2024-2025 — microsoft/MoGe
- DUSt3R (CVPR 2024) & MASt3R (ECCV 2024) & MASt3R-SfM (3DV 2025), metric checkpoint — naver/mast3r, naver/dust3r
- MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors, CVPR 2025 (arXiv:2412.12392) — rmurai0610/MASt3R-SLAM
- CUT3R: Continuous 3D Perception Model with Persistent State, CVPR 2025 Oral (arXiv:2501.12387) — CUT3R/CUT3R
- MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion, ICLR 2025 — Junyi42/monst3r
- MVSNet (ECCV 2018), CasMVSNet (CVPR 2020), PatchmatchNet (CVPR 2021 Oral, DTU overall 0.352mm), Vis-MVSNet — FangjinhuaWang/PatchmatchNet
- MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images, ECCV 2024 Oral — donydchen/mvsplat
- Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth, CVPR 2024
- ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth, 2023
- SAM 2 (Segment Anything in Images and Videos), Meta 2024; YOLO11-seg, Ultralytics 2024
- VINS-Fusion / OpenVINS (VIO), GTSAM & Ceres (factor-graph fusion); Umeyama Sim(3) alignment (1991)
- NVIDIA TensorRT, DeepStream, nvblox, Triton; Jetson AGX Orin (275 TOPS INT8, 64GB) / Orin NX benchmarks; depth-anything-tensorrt (RTX 4090 FP16 ViT-S 3ms) — spacewalk01/depth-anything-tensorrt
- STANAG 4609 / MISB KLV motion-imagery metadata; MAVLink; Open3D / VDBFusion / CloudCompare / Potree / Cesium 3D Tiles
