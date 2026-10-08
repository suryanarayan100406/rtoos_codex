# Research dossier — Computer Vision & Video Intelligence (front-end: compressed video to clean, geometry-ready observations)

> Auto-generated from the research workflow. Source material for document authoring.

## Executive summary

For single-pass drone reconstruction, the CV front-end's job is to convert one compressed, motion-blurred, dynamically-cluttered flight video into a sparse set of sharp, well-exposed, wide-parallax keyframes with reliable inter-frame correspondences and clean dynamic-object masks — because everything downstream (SfM/BA, MVS, meshing) inherits these errors and single-pass gives you no second flight to average them out. The pipeline is: hardware-decode (NVDEC/DeepStream) -> per-frame quality gating (Laplacian-variance blur + exposure) -> parallax-aware keyframe selection -> dynamic masking (YOLO-seg + SAM2 mask propagation + geometric moving-object test) -> feature matching (SuperPoint+LightGlue sparse for real-time; RoMa/LoFTR dense as a wide-baseline fallback). The single biggest lever versus classic multi-pass photogrammetry is adopting learned feed-forward 3D priors — MASt3R-SLAM (real-time 15 FPS dense SLAM, works uncalibrated) as the online front-end and VGGT (CVPR 2025 Best Paper, feed-forward pose+depth+pointmaps in <1s) for near-real-time refinement — because limited baseline makes pure multi-view triangulation ill-conditioned, and these networks inject monocular/learned geometric priors exactly where classical geometry degenerates. Metric scale without GCPs comes from fusing RTK/PPK GPS baselines and IMU with a metric monocular depth prior (Depth Anything V2-metric / UniDepth / Metric3Dv2) rather than from geometry alone. On deblur, the correct default is reject-not-restore: gate out heavily blurred frames and only lightly restore borderline ones, because learned deblur hallucinates texture that poisons photogrammetric matching. A critical realism flag for this NTRO/defense context: several strongest models (VGGT commercial checkpoint, MASt3R, Depth Anything V2-Base/Large) carry non-commercial or explicitly no-military licenses, and Ultralytics YOLO is AGPL-3.0 — so the deployable stack must be assembled from permissively-licensed components (SuperPoint/DISK/LightGlue Apache-2.0, RoMa MIT, DROID-SLAM BSD) or licensed/retrained equivalents. Every stage must degrade gracefully: gating never empties the keyframe set, masking falls back to detection-only, and the geometry front-end falls back from learned-dense to classical sparse SfM so the system slows and coarsens rather than hard-failing.

### Learned feed-forward 3D front-end: MASt3R-SLAM (online) + VGGT (batch refine)  
*Maturity: research*

- **What:** MASt3R-SLAM is a CVPR 2025 real-time dense SLAM that ingests MP4/RGB directly and outputs globally-consistent dense pointmaps + poses at 15 FPS, with a --no-calib mode for unknown intrinsics. VGGT (CVPR 2025 Best Paper) is a single feed-forward transformer that predicts camera extrinsics+intrinsics, depth maps, dense pointmaps, and 3D point tracks from 1-to-hundreds of frames in <1s, exportable to COLMAP for downstream Gaussian Splatting.
- **Why it fits single-pass:** Single-pass gives a narrow baseline where classical triangulation is ill-conditioned; these models carry learned monocular+multiview geometric priors that fill degenerate regions, produce metric-ish depth from limited parallax, and complete partially-occluded surfaces instead of leaving holes. Feed-forward = near-real-time and no fragile iterative convergence.
- **Perf/accuracy:** MASt3R-SLAM: 15 FPS on RTX 4090, dense. VGGT: reconstruction <1s, VGGT-1B, handles hundreds of views (May-2026 memory fix = 2-3x more frames/GPU-budget), AUC@30 90.37 (commercial ckpt) on Co3D.
- **Failure modes:** Both trained largely on global-shutter, near-nadir/oblique distributions unlike some drone geometries; VGGT frame count bounded by GPU VRAM; can hallucinate plausible-but-wrong geometry in truly unseen regions (dangerous for measurement); MASt3R-SLAM drift without loop closure on a straight single pass. LICENSE: VGGT commercial checkpoint excludes military use; MASt3R weights are CC-BY-NC-SA (non-commercial).
- **Alternatives:** DUSt3R / MASt3R-SfM (3DV 2025, unconstrained SfM), DROID-SLAM (NeurIPS 2021, BSD-3, robust deep-flow SLAM, needs calibration + 11GB VRAM) as a permissively-licensed fallback, Fast3R / Spann3R / MonST3R (dynamic-scene variant), classical COLMAP/GLOMAP as the ultimate fallback.

### Dynamic-object masking: YOLO-seg detection + SAM2 mask propagation + geometric motion test  
*Maturity: production*

- **What:** Two-track masking of vehicles/humans/animals: (1) semantic — YOLO-seg (YOLO26/YOLO11-seg) or Mask2Former panoptic per keyframe, with SAM2 streaming-memory video propagation to keep masks temporally consistent and fill frames the detector misses; (2) geometric — flag pixels whose optical-flow vector violates the epipolar constraint (residual to the estimated epipolar line) or whose reprojection is inconsistent with the static-scene model, catching movers of unknown class.
- **Why it fits single-pass:** In single-pass you cannot average out a moving car across multiple flights, so any un-masked dynamic pixel becomes a permanent ghost/streak in the mesh. Semantic catches known movable classes even while parked-vs-moving is ambiguous; geometric catches unknown/animal movers and is class-agnostic. SAM2's video memory gives coherent masks across the sequence cheaply.
- **Perf/accuracy:** SAM2.1 hiera_tiny 91 FPS / small 85 / base+ 64 / large 39.5 FPS on A100; supports adding objects mid-track (Dec-2024 per-object inference). YOLO26-seg: n 2.7M params @2.1ms, m 23.6M @6.7ms, x 62.8M @16.4ms on T4 (mask mAP 33.9->47.0).
- **Failure modes:** Segmentation misses camouflaged/small/animal targets; masking away real static structure loses coverage; epipolar test is degenerate for objects moving along the epipolar/flight direction (aperture-style ambiguity) and near the epipole in forward flight. Ultralytics YOLO is AGPL-3.0 (copyleft) — licensing problem for a closed defense product.
- **Alternatives:** RT-DETR / YOLOX / RTMDet (Apache-2.0, avoids AGPL), Mask2Former (panoptic), Grounded-SAM2 (open-vocabulary), plus MonST3R or motion-segmentation nets for scene-flow-based movers; dilate masks + shadow segmentation to remove moving shadows too.

### Sparse feature matching: SuperPoint / DISK / ALIKED + LightGlue  
*Maturity: production*

- **What:** Learned sparse keypoints (SuperPoint, DISK, or ALIKED) matched by LightGlue, an adaptive-depth/width GNN matcher returning correspondence indices; drop-in replacement for SIFT+NN or SuperGlue in the SfM/SLAM front-end.
- **Why it fits single-pass:** The real-time, low-VRAM backbone for pose/track estimation and for feeding classical BA. Learned features survive the motion blur, compression blocking, and moderate illumination change of drone video far better than SIFT, and LightGlue's adaptive pruning gives the FPS budget headroom needed for near-real-time.
- **Perf/accuracy:** LightGlue 150 FPS @1024 kpts / 50 FPS @4096 kpts on RTX 3080; 20 FPS on CPU @512 kpts; 4-10x faster than SuperGlue at comparable accuracy; now in HuggingFace Transformers. Apache-2.0 core (SuperPoint/ALIKED carry own licenses; DISK/ALIKED permissive).
- **Failure modes:** Sparse matching thins out on low-texture (water, sand, fresh snow, uniform rooftops) and very wide baselines / large viewpoint change — exactly the hard cases in sparse single-pass coverage; SuperPoint's original license is research-oriented.
- **Alternatives:** XFeat/XFeat* (lightweight, CPU/edge-friendly, Apache), DISK+LightGlue, ALIKED+LightGlue; escalate to dense/detector-free matching (next row) when inlier counts fall below threshold.

### Dense / detector-free matching for wide baseline & appearance change: RoMa / LoFTR / MASt3R-matching  
*Maturity: near-production*

- **What:** Detector-free dense matchers that estimate pixel-dense warps + per-pixel confidence for image pairs: RoMa (robust dense, DINOv2-backed), LoFTR/ASpanFormer (transformer coarse-to-fine), and MASt3R's 3D-grounded reciprocal-NN matching.
- **Why it fits single-pass:** Single-pass leaves sparse overlap and forces wide-baseline / oblique pairings where sparse detectors fail; dense matchers recover correspondences on low-texture and across illumination/shadow change, densifying weak links so the reconstruction stays connected. Use selectively (only on low-inlier pairs) to control cost.
- **Perf/accuracy:** RoMa (CVPR 2024) SOTA robust dense matching, MIT-licensed (except DINOv2); Tiny RoMa v1 AUC@5 56.4 on Mega1500 for the light variant; MASt3R matching grounds correspondences in metric 3D (ViT-L/ViT-B + DPT head).
- **Failure modes:** Heavy: full RoMa/LoFTR are GPU- and latency-expensive, not per-frame real-time — must be triggered sparingly; dense warps can over-smooth across depth discontinuities; MASt3R weights non-commercial.
- **Alternatives:** DKM (RoMa's predecessor), ASpanFormer, Efficient LoFTR / XFeat* semi-dense for a cheaper middle ground; gate usage by inlier-count threshold from the sparse stage.

### Frame quality gating & parallax-aware keyframe selection  
*Maturity: production*

- **What:** Per-frame scoring then selection: reject blur via Laplacian-of-image variance (Pech-Pacheco) or a learned no-reference sharpness score; reject over/under-exposure via histogram clipping fraction; then select keyframes that maximize baseline/parallax subject to a minimum feature-overlap (median optical-flow magnitude or matched-inlier count between candidate and last keyframe) and a minimum GPS/IMU-implied translation.
- **Why it fits single-pass:** This is THE single-pass-specific stage. Multi-pass keyframing removes redundancy; single-pass keyframing must instead deliberately harvest the little parallax the one flight offers — too little baseline = degenerate triangulation, too much = lost overlap. Gating first guarantees only sharp, well-exposed frames enter geometry.
- **Perf/accuracy:** Laplacian-variance and histogram gating are microseconds/frame on CPU/GPU; overlap/parallax scoring reuses the optical-flow or matcher already in the pipeline (near-free). Typical target: 60-80% inlier overlap between adjacent keyframes.
- **Failure modes:** A single global blur threshold fails under scene-content and altitude changes (low-texture scenes look 'blurry'); can starve the keyframe set on a uniformly blurry segment — must fall back to 'best-available in window' rather than emitting nothing; parallax scoring confounds camera translation with scene motion if dynamics aren't masked first.
- **Alternatives:** Learned NR-IQA/sharpness nets, frequency-domain (FFT high-freq energy) blur scores, information-gain / covariance-aware keyframe selection from SLAM; content-adaptive per-window thresholds instead of global.

### Hardware video ingest & decode: NVDEC + GStreamer / NVIDIA DeepStream  
*Maturity: production*

- **What:** GPU-accelerated H.264/H.265 decode via NVDEC exposed through GStreamer (nvv4l2decoder) or the NVIDIA DeepStream SDK, with explicit GOP/keyframe (I/P/B) awareness, zero-copy to CUDA/TensorRT, and timestamp/PTS alignment to GPS/IMU telemetry.
- **Why it fits single-pass:** This is the mandatory real-hardware ingest layer. Drone links deliver compressed H.264/H.265 (often RTSP/RTP); NVDEC decode on Jetson Orin / discrete NVIDIA keeps the whole front-end on-GPU for near-real-time, and DeepStream gives a production, batteries-included video-analytics pipeline that already runs YOLO/segmentation inline.
- **Perf/accuracy:** NVDEC decodes multiple 4K H.265 streams real-time on Jetson Orin / RTX; DeepStream runs detection+tracking inline at video rate; zero-copy avoids CPU<->GPU transfers that otherwise dominate latency.
- **Failure modes:** Compression blocking/mosquito artifacts and I-frame-only lossy previews corrupt features; packet loss over RF link causes decode corruption/frame drops — must detect and skip corrupt frames; B-frame reordering and PTS jitter break naive video-to-telemetry sync; low-bitrate links quantize away the texture photogrammetry needs.
- **Alternatives:** FFmpeg/PyAV with hardware accel, Jetson Multimedia API, Intel VAAPI/Quick Sync on x86; extract only I-frames as a cheap low-quality fallback when bitrate/telemetry is poor.

### Optical flow for motion segmentation & overlap scoring: SEA-RAFT / RAFT / NeuFlow v2  
*Maturity: near-production*

- **What:** Dense optical flow used two ways: (a) as the geometric signal for moving-object detection (flow inconsistent with the epipolar/ego-motion field = mover), and (b) as a cheap inter-frame overlap/parallax measure for keyframe selection.
- **Why it fits single-pass:** Provides the class-agnostic motion cue that complements semantic masking, and a dense-correspondence prior that helps matching under blur. SEA-RAFT and NeuFlow v2 bring RAFT-class accuracy into the real-time regime needed on-drone/near-real-time.
- **Perf/accuracy:** RAFT (ECCV 2020) is the accuracy baseline but heavy; SEA-RAFT (ECCV 2024) is markedly faster + more accurate; NeuFlow v2 (2024) targets real-time/edge flow at large speedups.
- **Failure modes:** Large-displacement flow under strong motion blur is unreliable; flow alone cannot separate camera motion from object motion without the epipolar/ego-motion model; expensive at 4K (tile or downscale).
- **Alternatives:** GMFlow/Unimatch, classical DIS/Farneback for a CPU fallback, or reuse LightGlue/dense-matcher correspondences directly instead of a separate flow net.

### Motion deblur policy: IMU-aided restoration with reject-first gating  
*Maturity: research*

- **What:** A decision policy, not just a model: heavily-blurred frames are REJECTED at the gate; borderline frames are lightly restored using an IMU/gyro-derived blur-kernel (PSF from angular velocity during exposure) or a conservative learned video deblur net; heavily-restored frames are never promoted to keyframes for measurement.
- **Why it fits single-pass:** Single-pass has fewer frames so the temptation is to salvage blurry ones — but aggressive learned deblur hallucinates edges/texture that create false matches and biased geometry. IMU-aided kernels are physically grounded (the drone's gyro tells you the motion), making restoration trustworthy for mild blur while gating protects metric accuracy.
- **Perf/accuracy:** Learned video deblur (e.g. RVRT / Shift-Net / recent event- or IMU-guided methods) reaches ~32-34 dB PSNR on GoPro/DVD benchmarks, but benchmark PSNR does not equal geometric fidelity — the load-bearing metric here is downstream match inlier count, not PSNR.
- **Failure modes:** Restoration hallucination corrupting matches (the central risk); rolling-shutter blur is spatially variant and not a single PSF; deblur latency can break real-time budget — hence reject-first as default.
- **Alternatives:** Prefer avoidance: higher shutter/frame-rate capture, global-shutter sensor, drop-and-interpolate poses over blurred frames; event-camera-guided deblur if the platform has one; simply skip and rely on neighboring sharp keyframes.

### Metric monocular depth prior for scale & graceful degradation: Depth Anything V2-metric / UniDepth / Metric3D v2  
*Maturity: near-production*

- **What:** A monocular metric-depth network run on keyframes to (a) supply a metric-scale prior that, fused with RTK/PPK GPS baselines and IMU, fixes absolute scale without GCPs, and (b) provide a full-frame depth fallback when multi-view geometry is too weak to triangulate.
- **Why it fits single-pass:** Single-pass without GCPs makes scale observability marginal; a metric depth prior turns weak geometry into usable metric structure and densifies low-parallax regions. It is also the graceful-degradation floor: even one usable frame yields a coarse depth surface rather than nothing.
- **Perf/accuracy:** Depth Anything V2 (NeurIPS 2024): Small 24.8M (Apache-2.0) / Base 97.5M / Large 335.3M (CC-BY-NC-4.0), metric_depth fine-tuned track; input 518px+; UniDepth/Metric3Dv2 predict metric depth + intrinsics zero-shot. Prompt Depth Anything reaches 4K metric depth when prompted by low-res LiDAR.
- **Failure modes:** Monocular metric depth has real scale error and domain gap at high/oblique drone altitudes (training data is mostly ground-level); must be fused/anchored to GPS/IMU/RTK, never trusted alone for measurement; Base/Large checkpoints non-commercial.
- **Alternatives:** UniDepth (CVPR 2024), Metric3D v2, ZoeDepth; anchor scale on RTK/PPK baseline geometry or known-size objects; use the Apache-2.0 Small model or retrain a metric head for a commercial/defense build.

### Illumination, shadow & appearance normalization  
*Maturity: near-production*

- **What:** Photometric conditioning before matching/texturing: per-frame exposure/white-balance compensation, shadow detection+masking (or shadow-invariant transforms), and multi-band / gain-normalized texture blending so mesh texture isn't seamed by changing light.
- **Why it fits single-pass:** Over a single continuous flight, sun angle and auto-exposure shift; learned matchers tolerate this, but texture quality and dynamic-shadow ghosting still degrade results. Masking moving shadows also prevents them being reconstructed as fake geometry — a single-pass-specific hazard since you can't average shadows out.
- **Perf/accuracy:** Exposure/gain normalization and multi-band blending are standard, real-time; learned shadow segmentation adds one lightweight net per keyframe.
- **Failure modes:** Aggressive normalization can erase real radiometric information; shadow removal can hallucinate ground texture under the shadow; specular/water regions remain hard.
- **Alternatives:** Retinex / homomorphic filtering, learned relighting/shadow-removal nets, or defer to texture-baking with view-dependent blending and outlier rejection in the meshing stage.

## Recommended stack

| Component | Choice | Rationale |
|---|---|---|
| Video decode / ingest | NVDEC via GStreamer (nvv4l2decoder) or NVIDIA DeepStream SDK, zero-copy to CUDA; PyAV/FFmpeg CPU fallback | Production H.264/H.265 hardware decode that keeps the pipeline on-GPU for near-real-time on Jetson Orin or discrete NVIDIA; DeepStream also runs the detector/tracker inline. Handles RTSP/RTP drone links and I/P/B GOP structure. |
| Frame quality gate | Laplacian-variance blur score + histogram-clip exposure gate, content-adaptive per-window thresholds | Near-free, deterministic, CPU/GPU; guarantees only sharp, well-exposed frames enter geometry and never emits an empty set (best-available fallback). |
| Keyframe selection | Parallax/overlap-maximizing selector driven by optical-flow magnitude or matcher inlier count + GPS/IMU translation | Single-pass must harvest scarce baseline deliberately; reuses signals already computed, so it is effectively free and keeps overlap in the 60-80% band. |
| Dynamic-object masking | RT-DETR / RTMDet or YOLO-seg (license-permitting) for semantics + SAM2.1 (hiera-small/base+) for video mask propagation + epipolar-residual motion test | Two independent cues (semantic + geometric) catch both known classes and unknown movers; SAM2 gives temporally consistent masks at 60-85 FPS. Choose Apache-licensed detectors to avoid Ultralytics AGPL for a defense product. |
| Sparse features + matcher | SuperPoint (or DISK/ALIKED) + LightGlue, TensorRT/ONNX-exported | Real-time (150 FPS @1024 kpts on RTX 3080), low-VRAM, blur/compression-robust backbone for pose/tracks and classical BA; Apache-2.0 core. |
| Wide-baseline fallback matcher | RoMa (MIT) or Efficient LoFTR, triggered only when sparse inliers < threshold | Detector-free dense matching rescues low-texture / wide-baseline / illumination-change pairs that break sparse matching, without paying its cost every frame. |
| Geometry front-end (online) | MASt3R-SLAM (real-time dense, --no-calib) as primary; DROID-SLAM (BSD-3) as permissive fallback; classical COLMAP/GLOMAP as ultimate fallback | Learned dense priors make limited-baseline single-pass reconstructable at 15 FPS and tolerate unknown intrinsics; layered fallbacks guarantee graceful degradation to a slower classical path. |
| Geometry refinement (batch, near-real-time) | VGGT (use VGGT-1B-Commercial checkpoint, or retrain, given military-license limits) -> export to COLMAP -> gsplat | Feed-forward pose+depth+pointmaps in <1s over the selected keyframes tightens the model and hands a clean COLMAP model to meshing/Gaussian-splatting. |
| Optical flow | SEA-RAFT (accuracy) or NeuFlow v2 (edge real-time) | RAFT-class flow at real-time speeds for the motion-segmentation cue and overlap scoring. |
| Metric depth prior | Depth Anything V2 metric (Apache Small, or retrained head) / UniDepth / Metric3D v2, fused with RTK/PPK + IMU | Recovers absolute scale without GCPs and provides a depth floor when triangulation is too weak — the graceful-degradation backstop. |
| Deblur | IMU/gyro-PSF light restoration with reject-first policy; heavy learned deblur disabled for keyframes | Physically-grounded mild restoration only; avoids hallucinated texture corrupting metric geometry. |
| Edge compute target | NVIDIA Jetson Orin (AGX/NX) on-drone for gating/masking/SLAM; ground-station RTX for VGGT refine | Splits the near-real-time front-end (on-platform) from heavier batch refinement (ground), matching bandwidth and power budgets. |

## Real-world integration

INPUTS/INTERFACES: Ingest the drone RF/RTSP video as H.264/H.265 over RTP; decode with NVDEC. Pull GPS/IMU/baro/gimbal telemetry via MAVLink (PX4/ArduPilot) or the vendor SDK (DJI PSDK/OSDK, Skydio); parse camera intrinsics from EXIF/XMP or a one-time calibration. TIME SYNC is the make-or-break detail: align video PTS to telemetry timestamps (ideally PPS/GPS-time or the autopilot's monotonic clock), correcting for encoder latency and B-frame reordering — a 50-100 ms video/telemetry offset at drone speed is meters of georeferencing error. Convert GPS (WGS84) to a local metric frame (UTM/ENU) for reconstruction, then georeference the output (GeoTIFF/LAS/LAZ/OBJ+world-file/3D Tiles/Cesium ion; COLMAP model as the interchange to meshing/splatting). CALIBRATION: run camera intrinsic + camera-IMU extrinsic (hand-eye) + gimbal calibration offline; estimate rolling-shutter readout time per sensor. SCALE/GEOREF WITHOUT GCP: fuse RTK/PPK GPS baselines + IMU pre-integration + metric monocular depth to set absolute scale and datum; expose optional GCP/checkpoint ingestion to validate. DEPLOYMENT: containerize (Docker/L4T) on Jetson Orin for the on-drone near-real-time front-end (decode->gate->mask->SLAM), stream keyframes+poses+masks to a ground RTX station for VGGT/BA refinement and meshing; models exported to TensorRT/ONNX with FP16/INT8. Standard glue: ROS 2 for telemetry/timing plumbing, GStreamer/DeepStream for the video graph. LICENSING is an integration constraint, not an afterthought (see risks).

## Reliability & failure handling

Design every stage with a fallback so the system slows/coarsens instead of hard-failing. DECODE: detect corrupt/dropped frames (RF loss) and skip; on severe loss, fall back to I-frame-only extraction. GATING: never emit an empty keyframe set — if a whole window is blurry/over-exposed, pass the best-available frame flagged low-confidence. MASKING: if segmentation net stalls or misses, fall back to detection-only + geometric motion test; dilate masks conservatively (better to lose a little coverage than keep a ghost). MATCHING: tiered — sparse SuperPoint+LightGlue first; if inliers < threshold, escalate to dense RoMa/LoFTR on that pair; if still degenerate, mark the link weak and lean on IMU/GPS + metric-depth prior to bridge. GEOMETRY: MASt3R-SLAM primary -> DROID-SLAM -> classical COLMAP/GLOMAP, each a strict superset-of-robustness fallback; if online SLAM diverges (tracking loss), reinitialize from the last good keyframe using GPS/IMU dead-reckoning. SCALE: if RTK absent, fall back to IMU+metric-depth scale with widened uncertainty. Attach PER-VERTEX/PER-REGION CONFIDENCE (from matcher certainty, view count, depth-prior agreement) and NEVER fabricate geometry for measurement — occluded/unseen surfaces are flagged 'unmeasured' or 'inferred', not silently filled. Health monitor watches inlier counts, tracking covariance, pose-vs-GPS residual, and frame-drop rate; on threshold breach it degrades mode (drop to sparse, reduce resolution, switch to batch-only) and surfaces a status rather than crashing. Golden rule: hallucination-prone steps (learned deblur, monocular depth, learned completion) are allowed for visualization but down-weighted or excluded from metric outputs.

## Single-pass specifics

What must change versus multi-pass photogrammetry: (1) KEYFRAMING INVERTS — multi-pass removes redundant overlapping frames; single-pass must actively MAXIMIZE parallax from one trajectory while holding minimum overlap, because the only baseline you get is along the flight line. (2) TRIANGULATION IS OFTEN DEGENERATE — narrow baseline and, in straight forward flight, the epipole sits inside the frame so points near it are unobservable; the system must detect low-parallax/near-epipole configurations and switch from triangulation to learned/monocular depth priors there. (3) LEARNED PRIORS BECOME MANDATORY, NOT OPTIONAL — VGGT/MASt3R/DUSt3R-family and metric monocular depth substitute learned geometry where multi-view geometry fails; classic pure-SfM will hole out. (4) OCCLUSION IS STRUCTURAL — one viewing direction means facades on the far side, undersides, and street canyons are simply never seen; you either complete them with priors (flag as inferred) or leave them empty — you cannot fly again. (5) DYNAMIC OBJECTS CAN'T BE AVERAGED OUT — with multiple passes a moving car appears in few frames and is outvoted; single-pass has no such redundancy, so masking must be aggressive and high-recall (semantic + geometric + shadow). (6) NO LOOP CLOSURE / DRIFT UNBOUNDED — a non-revisiting path gives no loop constraints, so drift is bounded only by GPS/IMU fusion and the learned front-end, not by SfM self-consistency. (7) SCALE OBSERVABILITY IS MARGINAL — fewer geometric constraints for metric scale without GCPs, forcing reliance on RTK/PPK baseline + IMU + metric depth. (8) ROLLING SHUTTER + MOTION BLUR ARE FIRST-ORDER — continuous forward motion means every frame has RS distortion and possible blur, and you can't cherry-pick a still frame from another pass; gating + RS-aware handling matter more. (9) REDUNDANCY-FOR-DENOISING IS GONE — noise/artifacts that multi-pass averages out persist, so front-end frame quality is disproportionately load-bearing.

## Open risks

- LICENSING vs the defense/NTRO context: VGGT's commercial checkpoint explicitly excludes MILITARY use and the original is non-commercial; MASt3R weights are CC-BY-NC-SA (non-commercial); Depth Anything V2 Base/Large are CC-BY-NC; Ultralytics YOLO11/YOLO26 are AGPL-3.0 (copyleft). A deployable military/commercial product must be built from permissive components (SuperPoint/DISK/LightGlue Apache, RoMa MIT, DROID-SLAM BSD, RT-DETR/RTMDet/YOLOX Apache) or retrained/licensed equivalents — this is the single biggest realism risk.
- Learned-prior HALLUCINATION corrupting metric accuracy: VGGT/MASt3R/monocular-depth/deblur can produce plausible-but-wrong geometry in unseen/low-parallax regions; must be excluded or heavily down-weighted for measurement and flagged for visualization only.
- Domain gap: most learned 3D and metric-depth models are trained on ground-level/object-centric data, not high-altitude oblique/nadir drone imagery — accuracy and scale can degrade; may need drone-domain fine-tuning.
- Rolling shutter is under-modeled: most learned SLAM/feed-forward nets assume global shutter; continuous drone motion induces RS distortion that biases geometry unless RS-aware BA or a global-shutter sensor is used.
- Real-time budget: MASt3R-SLAM's 15 FPS is on an RTX 4090; hitting near-real-time on Jetson Orin for 4K with masking + matching concurrently is tight and likely needs resolution/rate reduction, TensorRT INT8, and on-drone/ground split.
- Video/telemetry time-sync and GPS noise: sub-100ms desync or uncorrected GPS bias translates directly into georeferencing error; RTK/PPK not always available on the RF link.
- Compression floor: low-bitrate RF video quantizes away texture; below some bitrate the front-end simply cannot recover reliable features regardless of algorithm.
- Motion-segmentation blind spots: objects moving along the epipolar/flight direction and near the epipole evade the geometric test, and camouflaged/animal movers evade semantics.
- VRAM ceiling on VGGT/dense matchers bounds how many keyframes can be jointly processed; long flights need chunking + stitching, which reintroduces drift.
- Occluded surfaces are physically unrecoverable in single-pass — any 'reconstruction' there is inference; over-claiming measurability on inferred geometry is a correctness and trust risk.

## Differentiation notes

Off-the-shelf photogrammetry (DJI Terra, Pix4D, RealityCapture/RealityScan, OpenDroneMap, Metashape) assumes multi-pass, high-overlap (>70-80% front/side) captures and runs offline batch SfM+MVS that HARD-FAILS or holes out on a single sparse pass — they were never built for limited-baseline, near-real-time, or aggressive dynamic-object handling. Our differentiation: (1) a single-pass-native front-end whose keyframing MAXIMIZES rather than reduces parallax and that explicitly detects and routes around triangulation-degenerate/near-epipole geometry; (2) learned feed-forward 3D priors (MASt3R-SLAM online + VGGT refine) that reconstruct where classical MVS gives up, giving near-real-time output instead of hours of offline batch; (3) first-class dynamic masking (semantic + geometric + shadow) so movers never ghost — something consumer photogrammetry largely ignores; (4) GCP-free metric scale via tight RTK/PPK+IMU+metric-depth fusion rather than requiring surveyed control points; (5) a graceful-degradation architecture with tiered fallbacks (dense->sparse->classical, learned->geometric) so it never hard-fails on bad video — the opposite of black-box desktop tools; and (6) confidence-tagged outputs that separate measured from inferred geometry, which matters for defense/measurement use where fabricated surfaces are unacceptable. Crucially, it is assembled to run on-drone (Jetson Orin) + ground-station and integrates via MAVLink/RTSP/COLMAP/3D-Tiles into real UAV stacks, with a licensing-clean component set suitable for the NTRO/defense context — none of which the commercial photogrammetry suites offer.

## Citations / references

- LightGlue: Local Feature Matching at Light Speed, Lindenberger et al., ICCV 2023 (cvg/LightGlue; now in HF Transformers)
- SuperPoint: Self-Supervised Interest Point Detection, DeTone et al., CVPRW 2018
- SuperGlue, Sarlin et al., CVPR 2020
- DISK, Tyszkiewicz et al., NeurIPS 2020; ALIKED, Zhao et al., 2023
- SAM 2 / SAM 2.1: Segment Anything in Images and Videos, Ravi et al., Meta FAIR 2024 (facebookresearch/sam2, Apache-2.0)
- VGGT: Visual Geometry Grounded Transformer, Wang et al., CVPR 2025 Best Paper (facebookresearch/vggt; VGGT-1B-Commercial, non-military)
- DUSt3R, Wang et al., CVPR 2024 (arXiv 2312.14132)
- MASt3R: Grounding Image Matching in 3D, Leroy et al., 2024 (arXiv 2406.09756, CC-BY-NC-SA)
- MASt3R-SfM: Fully-Integrated Unconstrained SfM, 3DV 2025
- MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors, Murai/Dexheimer/Davison, CVPR 2025 (arXiv 2412.12392; 15 FPS)
- DROID-SLAM, Teed & Deng, NeurIPS 2021 (BSD-3, mono/stereo/RGB-D)
- RoMa: Robust Dense Feature Matching, Edstedt et al., CVPR 2024 (MIT except DINOv2); DKM, CVPR 2023
- LoFTR: Detector-Free Local Feature Matching, Sun et al., CVPR 2021; Efficient LoFTR 2024
- RAFT, Teed & Deng, ECCV 2020; SEA-RAFT, ECCV 2024; NeuFlow v2, 2024
- Depth Anything V2, Yang et al., NeurIPS 2024 (Small Apache-2.0; Base/Large CC-BY-NC); metric_depth track; Prompt Depth Anything 2024
- UniDepth, Piccinelli et al., CVPR 2024; Metric3D v2, 2024
- Ultralytics YOLO11 / YOLO26 segmentation (AGPL-3.0) — YOLO26-seg mask mAP 33.9-47.0, 2.1-16.4ms T4 TensorRT
- RT-DETR (Apache), RTMDet, YOLOX (Apache) as permissive detector alternatives; Mask2Former, Cheng et al., CVPR 2022; Grounded-SAM2
- MonST3R / Spann3R / Fast3R (dynamic & scalable pointmap variants, 2024-2025)
- Laplacian-variance focus measure, Pech-Pacheco et al., ICPR 2000
- NVIDIA DeepStream SDK, NVDEC/NVENC, Jetson Orin; GStreamer nvv4l2decoder; gsplat / 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)
