# Research dossier — Systems & Edge / Compute Optimization (real-time edge+cloud execution for single-pass drone-video 3D reconstruction)

> Auto-generated from the research workflow. Source material for document authoring.

## Executive summary

Treat single-pass reconstruction as a two-tier system: an on-drone/ground LIVE tier that yields a coverage-quality preview in ~1-3 s glass-to-glass, and a DETERMINISTIC OFFLINE tier that reprocesses the fully-recorded, hardware-timestamped stream into the georeferenced, metrically-accurate textured model. The single most important systems decision is to ALWAYS record near-raw H.265 plus full-rate GNSS/IMU/baro to onboard NVMe (store-and-forward), because there is no second flight — the accurate model must never depend on live link quality or real-time compute pressure. On the edge, use a Jetson Orin NX 16GB (10-40 W, 117 sparse-INT8 TOPS) for endurance or AGX Orin 64GB / Jetson Thor T4000 for heavy loads, running Isaac ROS (cuVSLAM VIO + nvblox incremental TSDF) with DeepStream/GStreamer hardware-accelerated ingest, detection/segmentation offloaded to the DLA and depth run on the GPU via TensorRT INT8/FP8. Because a single narrow-baseline pass makes classical multi-view triangulation ill-conditioned (forward-motion parallax degeneracy near the epipole), shift compute AWAY from wide-baseline dense MVS toward learned metric-depth/pointmap priors (Metric3D/UniDepth/Depth Anything V2, and VGGT/MASt3R-SLAM feed-forward geometry) fused tightly with RTK/PPK-GNSS + IMU to lock metric scale and georeference WITHOUT GCPs. Downlink only keyframes + poses + downsampled depth + confidence over SRT (sub-second, ARQ+FEC) to fit a lossy 2-20 Mbps RF link; the ground/cloud tier runs COLMAP's global mapper (GLOMAP, 1-2 orders of magnitude faster than incremental) with GNSS-constrained bundle adjustment, then dense/3DGS + texturing, emitting progressive LOD (Cesium 3D Tiles, COPC point clouds). Reliability is engineered as an explicit graceful-degradation ladder gated by confidence and health monitors (RTK to PPK to VIO-only to IMU/GNSS dead-reckon) so capability degrades rather than hard-failing, with ROS 2 lifecycle nodes, watchdogs, and idempotent/resumable checkpointing throughout.

### Two-tier architecture: on-edge live preview + deterministic offline refine on the recorded stream  
*Maturity: production*

- **What:** Split the workload into (1) a real-time tier producing a low-latency coverage/quality preview and obstacle map during flight, and (2) an offline tier that deterministically reprocesses the full recorded video+telemetry into the final metric georeferenced model. Version-pin every stage, fix RNG seeds, and index frames by monotonic ID + hardware timestamp so the refined model is reproducible.
- **Why it fits single-pass:** Single-pass gives exactly one flight and no re-fly, so you cannot let the accurate model depend on whatever the real-time tier managed under compute/thermal/link pressure. Recording near-raw onboard and refining later is what turns a limited, once-seen dataset into an accurate deliverable; the live tier's real job is to tell the pilot in-flight what coverage was missed.
- **Perf/accuracy:** Live preview target ~1-3 s glass-to-glass; refined model minutes after landing (or streamed to cloud mid-flight for a head start). Recording continuity target 100% regardless of link/compute state.
- **Failure modes:** If timestamps/clock sync drift between camera, IMU and GNSS, deterministic refine silently loses metric accuracy; refine tier can diverge from preview if pipeline versions are not pinned.
- **Alternatives:** Single monolithic real-time-only pipeline (loses accuracy, unrecoverable on any live failure); pure offline batch photogrammetry (no live guidance, no graceful degradation).

### On-edge VIO + incremental TSDF fusion (NVIDIA Isaac ROS cuVSLAM + nvblox)  
*Maturity: near-production*

- **What:** cuVSLAM GPU stereo/visual-inertial odometry provides real-time camera pose; nvblox performs GPU-accelerated incremental TSDF+ESDF fusion of depth into a live mesh/occupancy volume with per-voxel weighting. Both are Isaac ROS 4.6 packages (isaac_ros_visual_slam, isaac_ros_nvblox) with Nav2 integration.
- **Why it fits single-pass:** Gives an immediately usable live mesh + obstacle field on-drone for the preview and for sense-and-avoid, and the ESDF supports safe flight. TSDF's per-voxel confidence weighting lets low-confidence, once-seen single-pass surfaces be down-weighted rather than trusted blindly.
- **Perf/accuracy:** cuVSLAM runs at camera rate (~30-60 Hz) on Orin; nvblox integrates depth frames well above 30 Hz at 5-10 cm voxels on Orin-class. Memory-bandwidth-bound on Orin (102-205 GB/s) vs 273 GB/s on Thor.
- **Failure modes:** Monocular-only input gives scale-ambiguous VIO (needs GNSS/IMU/stereo to fix scale); dynamic objects corrupt TSDF unless masked; fast yaw/low-texture causes tracking loss.
- **Alternatives:** ORB-SLAM3 / OpenVINS / VINS-Fusion (CPU, no GPU accel), Voxblox/OpenVDB TSDF (CPU-bound), Open3D scalable TSDF.

### Learned metric-depth & feed-forward pointmap priors (Metric3D/UniDepth/Depth Anything V2; VGGT; MASt3R-SLAM)  
*Maturity: research*

- **What:** Run a monocular metric-depth network per keyframe for dense geometry, and/or feed-forward multi-view geometry transformers (VGGT predicts intrinsics+extrinsics+depth+pointmaps in one pass; MASt3R-SLAM does real-time dense SLAM with reconstruction priors) to recover structure where triangulation is weak.
- **Why it fits single-pass:** THE key single-pass change: a narrow-baseline forward-moving pass makes classical multi-view triangulation ill-conditioned (near-zero parallax at the epipole). Learned priors supply geometry and metric-scaled depth exactly where MVS fails, and produce confidence maps for occluded/once-seen surfaces.
- **Perf/accuracy:** VGGT reconstructs a scene in <1 s and handles hundreds of views in seconds on a datacenter GPU (bf16, Ampere+); MASt3R-SLAM and MonoGS ~10 FPS on RTX 4090. On Orin, run a light metric-depth net in TensorRT INT8 at reduced resolution (~10-30 Hz); heavy transformers belong on ground/cloud.
- **Failure modes:** Learned depth can hallucinate plausible-but-wrong geometry (must gate on confidence and flag); metric scale from a single net drifts without GNSS/IMU anchoring; VGGT memory scales with view count (recent fix ~2-3x more frames).
- **Alternatives:** Classical dense MVS (COLMAP patch-match / OpenMVS) for the refine tier where baseline permits; DUSt3R/MASt3R offline; stereo depth (Isaac ROS ESS/FoundationStereo) if a stereo rig exists.

### Tightly-coupled GNSS(RTK/PPK)-Visual-Inertial fusion for GCP-free metric georeferencing  
*Maturity: production*

- **What:** Fuse RTK/PPK GNSS, IMU pre-integration, barometric altitude and known camera intrinsics with VIO in a sliding-window/pose-graph estimator, adding absolute GNSS position factors and (post-flight) PPK-corrected factors to bound drift and lock global metric scale and datum.
- **Why it fits single-pass:** Delivers metric accuracy and georeferencing WITHOUT extensive GCPs, and — crucially for single-pass — replaces loop-closure (which a non-revisiting flight cannot provide) with absolute GNSS factors as the drift-bounding mechanism. Lever-arm and time-offset calibration convert cm-level RTK into cm-level model georef.
- **Perf/accuracy:** RTK/PPK: cm-level absolute position; single-point GNSS fallback: meters. VIO drift typically <1% of trajectory when GNSS-aided. Requires accurate GNSS-camera lever arm and camera-IMU time offset (<~1 ms).
- **Failure modes:** Poor RTK fix / multipath near structures degrades scale; wrong lever-arm or timing offset injects systematic metric error; GNSS outage under bridges/canyons forces VIO-only relative scale.
- **Alternatives:** VINS-Fusion (GPS-fused), PX4 EKF2 / robot_localization EKF as fallback state estimator; GTSAM/Ceres custom factor graph for the refine tier; minimal GCP check-points only for QA.

### TensorRT INT8/FP8 quantization + DLA offload + mixed precision (compute/VRAM/thermal engineering)  
*Maturity: production*

- **What:** Compile depth/detection/segmentation nets with TensorRT 10.x (PTQ/QAT via Model Optimizer, INT8/FP8/FP4/AWQ), run detection+segmentation on the Orin DLA to free the GPU for depth+fusion, use fp16/bf16 for transformers, batch keyframes, and pin memory with NITROS zero-copy to avoid host-device copies on the shared-memory SoC.
- **Why it fits single-pass:** Fits the whole live perception stack into a UAV power/thermal budget and Orin's shared 8-16 GB memory. On Orin the DLA adds ~40 (NX) to ~92 (AGX, 2x DLA) sparse-INT8 TOPS of offload, letting depth+TSDF keep the main GPU. Bandwidth-bound fusion benefits most from fp16 + tiling.
- **Perf/accuracy:** INT8 typically ~2-4x throughput vs FP16 with minor accuracy loss; AGX Orin 248 / Orin NX 117 / Orin Nano 67 sparse-INT8 TOPS; Thor 2070 FP4 TFLOPS @ 40-130 W (7.5x AGX Orin). Manage with nvpmodel (power mode) + jetson_clocks (clock lock).
- **Failure modes:** INT8 without good calibration data degrades depth accuracy; DLA supports a limited op set (unsupported layers fall back to GPU); thermal throttling under sustained load silently cuts clocks — must be monitored and fed into the degradation ladder.
- **Alternatives:** ONNX Runtime / TVM; FP16-only if calibration is unavailable; CUDA-graph capture to cut per-frame launch overhead; cloud Triton Inference Server for heavy nets.

### DeepStream/GStreamer hardware-accelerated ingest + dynamic-object masking  
*Maturity: production*

- **What:** Build the on-drone ingest as a GStreamer/DeepStream 8.0 pipeline: NVDEC hardware decode (nvv4l2decoder) -> nvvideoconvert -> nvinfer (TensorRT detector/segmenter, DLA) -> appsink to the fusion node, with a semantic/instance mask that removes vehicles, humans and animals before they enter TSDF/SfM.
- **Why it fits single-pass:** Single-pass scenes contain dynamic objects that, seen once, would be baked in as permanent geometry; masking at ingest is the clean fix. Hardware decode/encode + DLA inference keeps CPU free and sustains real-time on Orin. DeepStream 8.0 supports Blackwell and multi-stream.
- **Perf/accuracy:** DeepStream on Jetson Thor benchmarks: RT-DETR ~195 and PeopleNet+tracker ~363 concurrent 1080p30 streams — enormous headroom for one 4K stream plus reconstruction. NVDEC decodes 4K H.265 in real time on Orin.
- **Failure modes:** Aggressive masking removes legitimate static objects (parked-vs-moving ambiguity); missed detections leave ghosts; decoding the lossy downlinked stream (not the onboard near-raw) imports compression artifacts.
- **Alternatives:** Plain GStreamer + custom appsink; segmentation models (SegFormer, YOLO-seg) on DLA; optical-flow motion segmentation for un-modeled movers.

### SRT store-and-forward transport + onboard NVMe recording (link-loss resilience)  
*Maturity: production*

- **What:** Record near-raw H.265 + full-rate telemetry to an onboard NVMe circular buffer with monotonic/PTP timestamps; downlink keyframes+poses+depth+confidence over SRT (ARQ + optional FEC, tunable latency buffer) for the live preview. On link loss the buffer persists; on reconnect/landing the gap is backfilled for the deterministic refine.
- **Why it fits single-pass:** Directly answers 'never hard-fail' and 'GPS/link noise': the accurate model is built from the guaranteed onboard recording, never from the best-effort live link. SRT rides through the lossy 2-20 Mbps RF downlink that plain RTP/RTSP/RTMP cannot.
- **Perf/accuracy:** SRT: sub-second latency with configurable 120-500 ms recovery buffer, recovers from severe packet loss; keyframe+pose+depth stream fits a few Mbps vs 20-50 Mbps for raw 4K H.265. NVMe sustains 4K record + read-back.
- **Failure modes:** NVMe fills on very long flights (size the circular buffer / evict-after-forward); SRT latency buffer set too low drops packets, too high adds delay; unencrypted links risk interception (use SRT AES).
- **Alternatives:** MPEG-TS over UDP/RTP for lowest latency preview; QUIC/WebRTC for cloud legs; rosbag2 (mcap) as the on-disk record format; MQTT for lightweight event/telemetry backchannel.

### Graceful degradation ladder with ROS 2 lifecycle nodes, watchdogs, and confidence gating  
*Maturity: production*

- **What:** Define explicit capability tiers and transition rules driven by health/confidence: (1) full RTK-VIO+depth+downlink; (2) RTK lost -> PPK-deferred, flag reduced georef; (3) compute/thermal starved -> raise keyframe interval, sparser geometry; (4) downlink lost -> store-and-forward, preview freezes, recording continues; (5) GNSS lost -> VIO-only relative, flag not-georeferenced; (6) VIO lost -> IMU/GNSS/baro dead-reckon via EKF, mark gap, attempt relocalization. Every fusion input is confidence-gated (reject low-confidence depth/pose).
- **Why it fits single-pass:** This IS the 'never hard-fail' requirement made concrete, and it is tuned for single-pass where any dropped capability is unrecoverable in-flight — the system keeps the guaranteed recording and only lowers live fidelity/annotates confidence, never crashes.
- **Perf/accuracy:** Deterministic bounded transition latency (lifecycle transitions in tens of ms); recording uptime 100%; each output tile carries a confidence/provenance flag for downstream measurement trust.
- **Failure modes:** Flapping between tiers if thresholds lack hysteresis; a fallback estimator that is itself unmonitored; over-conservative gating discards usable data.
- **Alternatives:** ROS 2 managed/lifecycle nodes + diagnostics; systemd/hardware watchdog; behavior-tree supervisor (BehaviorTree.CPP); Nav2 recovery behaviors as a model.

### Global SfM refine (COLMAP global mapper / GLOMAP) + GNSS-constrained bundle adjustment  
*Maturity: near-production*

- **What:** Offline tier: feature extraction/matching (or VGGT/MASt3R-initialized correspondences) -> COLMAP's global mapper (GLOMAP, now merged into COLMAP) for fast global SfM -> bundle adjustment with GNSS/RTK prior factors and fixed intrinsics -> dense MVS or 3DGS -> mesh + texture -> geo-export.
- **Why it fits single-pass:** Global SfM is 1-2 orders of magnitude faster than incremental COLMAP with on-par/superior quality — essential to hit near-real-time turnaround on the recorded stream. GNSS-constrained BA delivers the metric georeference without GCPs; learned-correspondence init rescues weak single-pass baselines.
- **Perf/accuracy:** GLOMAP/global mapper: 1-2 orders of magnitude faster than incremental COLMAP, comparable/better accuracy; enables minutes-scale refine on a workstation/cloud GPU rather than hours.
- **Failure modes:** Degenerate near-collinear camera motion still challenges SfM (mitigate with learned priors + IMU); textureless/repetitive facades cause mismatches; without GNSS priors the global frame is arbitrary.
- **Alternatives:** Incremental COLMAP (robust, slow); Theia/OpenSfM; RTG-SLAM/MonoGS for GS-based; OpenDroneMap as an integrated open pipeline; commercial Metashape/Pix4D/RealityCapture as reference.

### ROS 2 QoS tuning + NITROS zero-copy + Zenoh WAN bridging + backpressure control  
*Maturity: production*

- **What:** Use ROS 2 with tuned DDS QoS (BEST_EFFORT/KEEP_LAST for preview video, RELIABLE for keyframe metadata/poses), NITROS negotiated zero-copy for intra-machine GPU-to-GPU node handoff, and rmw_zenoh for the drone<->ground<->cloud WAN legs. Enforce bounded queues: drop-oldest on the preview path, spill-to-disk (never drop) on the record path.
- **Why it fits single-pass:** Wireless/WAN links make DDS multicast discovery brittle; Zenoh scales and bridges cleanly over lossy/routed networks with shared-memory optimization. Zero-copy is mandatory to avoid CPU-GPU memcpy stalls on Orin's shared memory. Explicit backpressure prevents the classic real-time failure where a slow consumer stalls capture.
- **Perf/accuracy:** NITROS avoids multi-GB/s of redundant host-device copies; Zenoh adds shared-memory transport and client/router modes for efficient remote RViz/telemetry. QoS deadlines/liveliness give bounded staleness.
- **Failure modes:** Mismatched QoS profiles silently drop messages; unbounded queues cause latency blowup/OOM; Zenoh under high CPU load can silently drop some service calls (known issue).
- **Alternatives:** CycloneDDS/FastDDS defaults with careful QoS; eCAL; raw shared memory; MAVROS/MAVSDK bridge for the MAVLink side.

### Progressive LOD streaming for visualization, measurement & analysis  
*Maturity: production*

- **What:** Emit the model as streamable, level-of-detail tiles: Cesium 3D Tiles (glTF/b3dm) for meshes and COPC/EPT/Potree octrees for point clouds, with a coarse preview available first and detail streamed on demand; georeferenced in a real CRS via PDAL/GDAL, plus DSM/DTM and orthomosaic products.
- **Why it fits single-pass:** Makes the single-pass output immediately usable for measurement/analysis in a browser or GIS without downloading a huge model, and lets the coarse live preview and the refined tiles share one representation. Georeferenced tiles support cm-level measurement when RTK/PPK is available.
- **Perf/accuracy:** 3D Tiles/COPC stream interactively over the network at LOD; standard OGC/Cesium tooling; export to LAS/LAZ/COPC, glTF, GeoTIFF (DSM/ortho).
- **Failure modes:** LOD generation is compute-heavy for the final pass; incorrect CRS/EPSG or geoid model produces vertical offsets; measurement on low-confidence once-seen surfaces misleads unless flagged.
- **Alternatives:** Potree (point clouds), CesiumJS/Cesium ion, Nexus/3DHOP meshes, Deck.gl; direct mesh export (OBJ/PLY/glTF) for CAD/BIM.

### Observability, telemetry & idempotent/resumable processing  
*Maturity: production*

- **What:** Instrument every stage: tegrastats/jtop (GPU/DLA/thermal/power), ROS 2 diagnostics + per-stage latency histograms, dropped-frame and queue-depth counters, GNSS fix quality, VRAM/thermal, exported to Prometheus/Grafana or OpenTelemetry. Checkpoint the pose graph + TSDF + processed-frame cursor periodically so any crash resumes idempotently from the last checkpoint using the recorded stream.
- **Why it fits single-pass:** You cannot manage graceful degradation without measuring health, and single-pass demands that a mid-refine crash resume rather than restart hours of work. Idempotent, resumable, deterministic reprocessing is what makes the offline tier trustworthy and auditable.
- **Perf/accuracy:** Sub-second health telemetry; checkpoint/resume bounds re-work to one interval; deterministic re-runs reproduce the same model (fixed seeds, pinned versions, content-addressed inputs).
- **Failure modes:** Excessive telemetry overhead on the edge; non-atomic checkpoints corrupt on power loss (write-then-rename); hidden nondeterminism (unpinned CUDA/library versions, multi-GPU race) breaks reproducibility.
- **Alternatives:** MLflow/DVC for run/version tracking; ros2 bag replay for regression; Sentry/Loki for logs; systemd journald + healthchecks.

## Recommended stack

| Component | Choice | Rationale |
|---|---|---|
| Onboard compute (endurance) | NVIDIA Jetson Orin NX 16GB (10-40 W, 1024 CUDA / 32 Tensor cores, 1x DLA, 102.4 GB/s, 117 sparse-INT8 TOPS) | Best perf-per-watt for a UAV power budget; runs cuVSLAM+nvblox+INT8 depth with DLA offload within thermal limits. |
| Onboard compute (heavy / large UAV) | Jetson AGX Orin 64GB (15-60 W, 248 TOPS, 204.8 GB/s, 2x DLA) or Jetson Thor T4000 (Blackwell, 64GB @273 GB/s, 40-70 W, 1200 FP4 TFLOPS) | Headroom to run feed-forward geometry transformers and richer live fusion on-drone; Thor is ~7.5x AGX Orin for near-future platforms. |
| Edge OS / SDK | JetPack 6.x (Ubuntu 22.04, CUDA 12.x) + TensorRT 10.9 + TensorRT Model Optimizer (INT8/FP8/FP4/AWQ) | Standard, supported Jetson toolchain with quantization and DLA support for real-time inference. |
| Perception middleware | NVIDIA Isaac ROS 4.6 (isaac_ros_visual_slam/cuVSLAM, isaac_ros_nvblox, ESS/FoundationStereo, NITROS zero-copy) | Production GPU-accelerated VIO + incremental TSDF/ESDF + zero-copy transport, purpose-built for Jetson. |
| Video ingest / analytics | NVIDIA DeepStream 8.0 + GStreamer (NVDEC/NVENC, DLA inference) | Hardware-accelerated multi-stream decode + TensorRT inference for keyframe/dynamic-object masking with huge stream headroom. |
| Robotics framework + transport | ROS 2 Humble/Jazzy with rmw_zenoh for WAN legs; tuned DDS QoS for LAN | Zenoh bridges lossy drone<->ground<->cloud links far better than DDS multicast; QoS gives explicit reliability/backpressure control. |
| Live reconstruction | nvblox incremental TSDF mesh + cuVSLAM pose (on edge); optional MASt3R-SLAM/MonoGS on ground GPU | Immediate usable mesh/obstacle field and pilot coverage feedback; dense priors on the ground tier where a stronger GPU exists. |
| Metric geometry priors | Metric3D v2 / UniDepth / Depth Anything V2 (metric) in TensorRT INT8; VGGT / DUSt3R-MASt3R for refine | Recovers geometry and metric-scaled depth where single-pass triangulation is ill-conditioned; emits confidence for occluded surfaces. |
| State estimation / fallback | Tightly-coupled GNSS(RTK/PPK)-VIO factor graph (GTSAM/Ceres or VINS-Fusion); PX4 EKF2 / robot_localization EKF as fallback | GCP-free metric georeferencing; EKF fallback keeps a pose flowing when VIO drops (graceful degradation). |
| Offline refine (cloud/ground) | COLMAP with global mapper (GLOMAP) + GNSS-constrained bundle adjustment -> OpenMVS / 3D Gaussian Splatting + texturing | 1-2 orders of magnitude faster global SfM; produces the accurate final mesh/point cloud from the recorded stream. |
| Downlink transport + recording | SRT (ARQ+FEC, AES) for live keyframe/pose/depth; onboard NVMe circular buffer recording H.265 + telemetry as rosbag2/mcap | Rides lossy 2-20 Mbps RF; guarantees the full dataset survives link loss for deterministic refine. |
| Flight-stack integration | PX4 or ArduPilot + MAVLink via MAVROS/MAVSDK (or DJI Payload SDK); GPS PPS + PTP time sync | Real telemetry (GLOBAL_POSITION_INT, ATTITUDE, HIGHRES_IMU) and hardware timestamps needed for metric fusion. |
| Cloud inference / scale-out | NVIDIA Triton Inference Server + containerized COLMAP/3DGS on RTX/L4/A-series; object store for stream + artifacts | Offloads heavy transformers/MVS/3DGS; scales the deterministic refine tier elastically. |
| Geo output + visualization | PDAL/GDAL, LAS/LAZ/COPC, glTF + Cesium 3D Tiles, Potree; DSM/DTM/orthomosaic (GeoTIFF) | Progressive LOD streaming for measurement/analysis in browser/GIS with correct CRS/geoid. |
| Observability | tegrastats/jtop + ROS 2 diagnostics + Prometheus/Grafana + OpenTelemetry; MLflow/DVC + content-addressed inputs for reproducibility | Health signals drive the degradation ladder; versioned deterministic re-runs make the model auditable. |

## Real-world integration

CAMERA/SENSOR INTERFACES: onboard camera via MIPI CSI-2, GMSL2 (Fakra), or USB3/UVC into the Jetson; a synchronized global-shutter stereo/tracking pair (or RealSense/Zed) enables cuVSLAM VIO and scale. Prefer processing the onboard near-raw/high-bitrate feed (before the lossy RF encoder) for reconstruction, and downlink a separate compressed preview — this directly reduces the motion-blur/compression-artifact challenge. TELEMETRY: MAVLink over serial/UDP (PX4/ArduPilot) via MAVROS/MAVSDK gives GLOBAL_POSITION_INT (GPS), GPS_RAW_INT (fix type/HDOP), ATTITUDE/ATTITUDE_QUATERNION, HIGHRES_IMU, and altitude; DJI drones use the Payload/Onboard SDK. RTK/PPK via an onboard u-blox ZED-F9P-class receiver + a base station or NTRIP caster; PPK done post-flight against logged RINEX. VIDEO TRANSPORT: H.264/H.265 hardware-encoded on the SoC, carried by RTP/RTSP or (preferred for lossy links) SRT/MPEG-TS; the ground station terminates SRT and re-publishes into ROS 2. TIME SYNC (make-or-break for metric accuracy): discipline the Jetson clock to GPS PPS and run PTP (IEEE-1588) across compute nodes; timestamp every frame at capture, and calibrate the camera-IMU time offset (<~1 ms) and the GNSS-antenna-to-camera lever arm (cm). CALIBRATION: intrinsics + distortion (OpenCV/Kalibr checkerboard), camera-IMU extrinsics (Kalibr), gimbal mounting angle (favor an oblique/side-look angle to create cross-track parallax and see facades), and the geoid/EPSG datum for vertical accuracy. FORMATS/HANDOFF: rosbag2 (mcap) for the recorded stream; keyframe messages carry image + pose + intrinsics + depth + confidence + timestamp; outputs as LAS/LAZ/COPC (points), glTF/OBJ + 3D Tiles (mesh), GeoTIFF (DSM/DTM/ortho). DEPLOYMENT: containerized (Isaac ROS Docker / L4T base images) on the drone and cloud for version-pinned, reproducible pipelines; nvpmodel power mode and jetson_clocks set per platform to trade endurance vs throughput.

## Reliability & failure handling

Engineer 'never hard-fail' as an explicit, monitored graceful-degradation ladder rather than try/except patching. TIERS (highest to lowest capability, with hysteresis on transitions): (1) full RTK/PPK-VIO + learned depth + live fusion + downlink; (2) RTK lost -> continue on float/single-point GNSS + VIO, flag reduced georef, defer to PPK; (3) compute/thermal starved (tegrastats shows throttle) -> raise keyframe interval, drop depth-net resolution/rate, fuse sparser geometry; (4) downlink lost -> store-and-forward to NVMe continues at full rate, live preview freezes but recording NEVER stops; (5) GNSS fully lost -> VIO-only relative reconstruction, scale from IMU/baro/known intrinsics, output flagged 'metric-relative, not georeferenced'; (6) VIO/tracking lost -> dead-reckon on IMU+GNSS+baro via a fallback EKF (PX4 EKF2 / robot_localization), mark the gap, attempt relocalization; (7) perception node crash -> ROS 2 lifecycle + hardware/software watchdog restarts it and it resumes idempotently from the last checkpoint using the recorded stream. CONFIDENCE GATING: every fusion input is thresholded — reject low-confidence depth pixels (network uncertainty), high-reprojection-error poses, and poor-HDOP GNSS; nvblox per-voxel weighting down-weights once-seen single-pass surfaces; each output tile carries a confidence/provenance flag so measurement tools can distinguish trusted from inferred/occluded geometry. HEALTH/WATCHDOGS: heartbeats between drone and ground, per-node health topics, bounded queues with drop-oldest on preview and spill-to-disk (never drop) on the record path, and deadline/liveliness QoS for bounded staleness. IDEMPOTENT/RESUMABLE: monotonic frame IDs + atomic (write-then-rename) checkpoints of pose graph, TSDF, and processed-frame cursor; deterministic reprocessing (pinned versions, fixed seeds, content-addressed inputs) so the refined model is reproducible and a crash costs at most one checkpoint interval.

## Single-pass specifics

What must change versus traditional multi-pass photogrammetry, from the systems/compute angle: (1) COMPUTE REALLOCATION — spend the budget on learned monocular metric-depth and feed-forward pointmap networks (Metric3D/UniDepth/DAv2, VGGT/MASt3R) plus tight sensor fusion, NOT on wide-baseline dense MVS, because a narrow forward-moving single pass has near-zero parallax at the epipole and ill-conditioned triangulation; MVS only helps in the refine tier where oblique views provide baseline. (2) DRIFT CONTROL WITHOUT LOOP CLOSURE — a non-revisiting flight offers no loop closures, so the back-end is a chain/sliding-window pose graph whose drift is bounded by absolute GNSS/RTK factors and IMU pre-integration, not by revisit constraints. (3) METRIC SCALE WITHOUT GCPs — scale and georeference come from tightly-coupled RTK/PPK-GNSS + IMU + baro + known intrinsics with accurate lever-arm and time-offset calibration; monocular-only scale is ambiguous and must be anchored. (4) RECORD-EVERYTHING IS MANDATORY — there is no re-fly, so near-raw video + full-rate telemetry MUST be recorded onboard (store-and-forward) and the accurate model produced by DETERMINISTIC offline reprocessing, decoupled from live link/compute quality. (5) IN-FLIGHT COVERAGE FEEDBACK — because gaps cannot be re-flown, the live tier's primary value is a real-time coverage/quality HUD telling the pilot what facade/rooftop was missed while there is still time to capture it. (6) FLIGHT+SENSOR INTEGRATION — favor an oblique/side-looking gimbal angle to manufacture cross-track parallax and see building facades that a straight-down nadir single pass never resolves. (7) OCCLUSION AS A FIRST-CLASS OUTPUT — once-seen and never-seen surfaces are pervasive; confidence maps and per-voxel weights must propagate to the final tiles so downstream measurement trusts only well-observed geometry, and any generative/learned completion is explicitly flagged low-confidence. (8) PREFER NEAR-RAW OVER DOWNLINKED FRAMES — reconstruct from onboard high-bitrate frames, not the lossy RF-compressed preview, to sidestep compression artifacts.

## Open risks

- Metric accuracy without GCPs hinges entirely on RTK/PPK fix quality plus lever-arm and camera-IMU time-offset calibration; a few-ms timing error or cm lever-arm error injects systematic metric bias that no software recovers.
- Learned depth/pointmap priors (the crutch for weak single-pass baselines) can hallucinate plausible-but-wrong geometry; without rigorous confidence gating and flagging, measurements on inferred surfaces mislead.
- Thermal throttling on a power-constrained UAV silently cuts Jetson clocks under sustained load, degrading live capability — must be actively monitored (tegrastats) and wired into the degradation ladder, not discovered post-hoc.
- RF downlink bandwidth (often 2-20 Mbps, lossy) cannot carry real-time 4K, so live preview fidelity is inherently limited; the near-real-time promise depends on onboard recording + deferred refine, which stakeholders may misread as 'not real-time'.
- Full 3D Gaussian Splatting needs ~8-24 GB VRAM and is too heavy for Orin-class edge; it must live on ground/cloud, adding a network dependency for the highest-fidelity output.
- Feed-forward transformers (VGGT) scale memory with view count; long flights need chunking/keyframe budgeting or they OOM even on datacenter GPUs.
- Un-masked dynamic objects (vehicles/people/animals) seen once get baked as permanent geometry into TSDF/SfM; detector misses directly corrupt the model.
- GNSS multipath and outages near tall structures/canyons — exactly the border/urban/infrastructure targets — degrade the very georeferencing the system depends on, forcing VIO-only relative segments.
- Hidden nondeterminism (unpinned CUDA/library versions, multi-GPU races, GPU float reductions) can break the 'deterministic reprocessing' guarantee that the single-pass value proposition rests on.
- Newest, most capable methods (VGGT, MASt3R-SLAM, MonoGS) are 2024-2025 research with RTX-4090-class benchmarks and limited edge/aerial validation — integration and on-Orin performance are real hackathon-to-product risks.

## Differentiation notes

Off-the-shelf photogrammetry (DJI Terra, Pix4D, Agisoft Metashape, RealityCapture, OpenDroneMap) is fundamentally MULTI-PASS and OFFLINE-BATCH: it assumes many overlapping wide-baseline images (grid/orbit missions), frequently needs GCPs for accuracy, offers no in-flight guidance, and hard-stops on missing data — none of which survives a single narrow-baseline pass with limited angles. This system is single-pass-NATIVE on four axes: (1) it reallocates compute from wide-baseline MVS to learned metric-depth/pointmap priors + tightly-coupled RTK/PPK-GNSS-VIO, recovering geometry and metric scale exactly where triangulation is ill-conditioned and GCPs are absent; (2) it is a two-tier edge+cloud design giving a ~1-3 s live coverage/obstacle preview (Isaac ROS cuVSLAM + nvblox on Jetson) so gaps can be caught while still flyable, plus a deterministic offline refine that no consumer tool provides; (3) it is engineered to NEVER hard-fail via an explicit confidence-gated degradation ladder, store-and-forward recording, watchdogs, and idempotent resumable reprocessing — where desktop tools simply error out; and (4) it integrates directly with real drone hardware/flight stacks (MAVLink/PX4/ArduPilot/DJI PSDK, PPS/PTP time sync, SRT transport) as a live streaming pipeline rather than a folder-of-JPEGs importer. Net: comparable final-model tooling (COLMAP global mapper, 3DGS, OpenMVS) is reused in the refine tier, but the surrounding real-time, single-pass, reliability, and integration engineering is the differentiator.

## Citations / references

- NVIDIA Jetson Orin module datasheet — AGX Orin 64GB (2048 CUDA/64 Tensor cores, 204.8 GB/s, 15-60W, 248 sparse-INT8 TOPS, 2x NVDLA v2), Orin NX 16GB (117 TOPS, 102.4 GB/s, 10-40W), Orin Nano 8GB (67 TOPS, 7-25W) (nvidia.com, 2025)
- NVIDIA Jetson AGX Thor / T5000 / T4000 — Blackwell GPU, 2070 FP4 TFLOPS (sparse), 128GB LPDDR5X @273 GB/s, 40-130W, 7.5x AGX Orin (nvidia.com, 2025-2026)
- NVIDIA Isaac ROS release 4.6 — isaac_ros_nvblox, isaac_ros_visual_slam (cuVSLAM), ESS/FoundationStereo, NITROS zero-copy (nvidia-isaac-ros.github.io, updated Aug 2026)
- nvblox — GPU-accelerated real-time TSDF+ESDF reconstruction library, ROS 2/Nav2 integration (github.com/nvidia-isaac/nvblox, 2024-2026)
- NVIDIA DeepStream SDK 8.0 — GStreamer-based, Blackwell support, DLA offload; Jetson Thor benchmarks (RT-DETR ~195, PeopleNet+tracker ~363 concurrent 1080p30 streams) (developer.nvidia.com, 2025-2026)
- NVIDIA TensorRT 10.9 + TensorRT Model Optimizer — FP8/FP4/INT8/INT4/AWQ, PTQ+QAT, Jetson/edge deployment (developer.nvidia.com, 2025)
- VGGT: Visual Geometry Grounded Transformer — feed-forward cameras+depth+pointmaps+tracks, <1s per scene, hundreds of views in seconds, CVPR 2025 Best Paper (github.com/facebookresearch/vggt)
- MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors — monocular, RTX 4090, metric checkpoint, CVPR 2025 (github.com/rmurai0610/MASt3R-SLAM)
- MonoGS / Gaussian Splatting SLAM — first monocular 3DGS SLAM, up to ~10 FPS on RTX 4090, CVPR 2024 (github.com/muskie82/MonoGS)
- 3D Gaussian Splatting — ~24GB VRAM to paper quality (~8GB minimum), Taming-3DGS/sparse_adam 1.6-2.7x training speedup, SIGGRAPH 2023 (github.com/graphdeco-inria/gaussian-splatting)
- GLOMAP global SfM — 1-2 orders of magnitude faster than incremental COLMAP, on-par/superior quality, now merged into COLMAP as the 'global' mapper (github.com/colmap/glomap, arXiv:2407.20219, 2024)
- rmw_zenoh — ROS 2 RMW over Zenoh, shared-memory transport, WAN client/router bridging (github.com/ros2/rmw_zenoh)
- SRT (Secure Reliable Transport) — sub-second latency, ARQ+FEC, tunable latency buffer, HEVC ingest, IETF draft (github.com/Haivision/srt)
- MAVLink camera & telemetry protocol; PX4/ArduPilot flight stacks; MAVROS/MAVSDK; DJI Payload SDK (mavlink.io / px4.io, 2025)
- Metric monocular depth: Metric3D v2, UniDepth, Depth Anything V2 (metric) (2024-2025)
- State estimation fallbacks: VINS-Fusion (GPS-aided VIO), PX4 EKF2, robot_localization EKF; GTSAM/Ceres factor-graph BA
- Geo/LOD tooling: PDAL, GDAL, LAS/LAZ/COPC, Cesium 3D Tiles, Potree; OpenDroneMap / OpenMVS for refine (2024-2025)
