# Research dossier — Drone & Sensor / Hardware Integration (physical capture system + software plumbing) for single-pass drone-video-to-3D

> Auto-generated from the research workflow. Source material for document authoring.

## Executive summary

Standardize on two proven, obtainable capture platforms. (1) DJI enterprise: Matrice 350 RTK carrying a Zenmuse P1 (45 MP full-frame, mechanical shutter) or L2 (LiDAR+RGB), or the lighter Mavic 3E — all ship factory RTK (1 cm + 1 ppm horizontal, 1.5 cm + 1 ppm vertical), calibrated intrinsics embedded in EXIF/XMP (DewarpData), and documented SDKs. (2) A custom PX4/ArduPilot UAV with a GLOBAL-shutter machine-vision camera for teams needing full sensor/timestamp control. Put an NVIDIA Jetson Orin NX (100-157 TOPS, 10-40 W — DJI Manifold 3, or a third-party PSDK carrier for the M350) on the aircraft as companion compute, OR run the identical stack on a ground Jetson AGX Orin fed by a 1080p RTMP/SRT downlink for near-real-time preview. The single most important hardware function is PER-FRAME GEOREFERENCING: hardware-timestamp every video frame to GNSS time (DjiTimeSync/PPS on DJI; PPS+PTP with gpsd/chrony on custom) and interpolate the RTK/PPK trajectory plus gimbal/IMU attitude onto each frame, then apply the camera-to-GNSS-antenna lever arm — this is what delivers metric georeferencing with few or zero GCPs. Prefer PPK (RTKLIB/Emlid Studio) over live-only RTK as the accuracy anchor because it survives datalink loss and can be reprocessed; and always record full-resolution video onboard while streaming only a compressed proxy, so a dropped link never loses the mission. Because a single straight strip is weak, narrow-baseline geometry, the capture SOP must change versus multi-pass photogrammetry: fly oblique (30-45 deg gimbal), slow and steady for >=90% frame overlap, add a slight lateral weave if airspace allows, and FIX camera intrinsics from a pre-flight calibration rather than trusting SfM self-calibration. The whole thing is buildable in a hackathon (DJI Pilot 2 RTMP + MediaMTX + GStreamer/NVDEC + MAVSDK) and hardens into a product (PSDK payload, NTRIP RTK, Kalibr-calibrated custom rigs).

### Per-frame georeferencing (GNSS-time frame stamping + trajectory interpolation + lever-arm)  
*Maturity: production*

- **What:** Timestamp every decoded video frame to a common GNSS/GPS time base, then spline-interpolate the RTK/PPK position and the gimbal/IMU attitude to each frame's exact epoch, and apply the fixed camera-sensor-to-antenna lever arm to get a metric camera pose per frame. These per-frame priors are injected into SfM/SLAM as pose priors (COLMAP rig/pose priors, or a factor-graph).
- **Why it fits single-pass:** This is THE mechanism that makes single-pass video metrically accurate WITHOUT dense GCPs: the trajectory substitutes for the missing multi-view constraints and control points. It is the crux hardware-software contract for the whole system.
- **Perf/accuracy:** Sync-error budget dominates: position error = sync_error x ground_speed. At 8 m/s, 10 ms sync = 8 cm smear; keep sync < 2-3 ms for cm-class work. RTK-fix priors give 1-3 cm absolute after lever-arm correction.
- **Failure modes:** Un-modeled sync offset or lever arm biases the entire model; attitude interpolation errors on aggressive maneuvers; RTK dropping to float mid-strip injects jumps.
- **Alternatives:** GCP/checkpoint-based bundle adjustment (needs ground marks); pure visual SLAM scale-from-IMU (metric but not georeferenced); DJI Terra offline PPK batch.

### Oblique single-pass capture SOP (gimbal pitch, GSD, speed, overlap, micro-weave)  
*Maturity: production*

- **What:** Fly the single pass with the gimbal at 30-45 deg oblique (not pure nadir) so facades AND rooftops AND ground are seen; hold slow, constant ground speed (5-8 m/s at 80-120 m AGL) for >=90% along-track frame overlap with usable keyframe parallax; if airspace permits, add a gentle lateral weave or one cross line; use fast/mechanical shutter to kill motion blur.
- **Why it fits single-pass:** Directly attacks the 'limited viewing angles from one flight path' challenge at the capture stage, where it is cheapest to fix. Oblique + slight lateral motion is the difference between a reconstructable strip and an ambiguous one.
- **Perf/accuracy:** GSD ~= H x pixel_pitch / focal. Mavic 3E (4/3, 3.28 um px, 12.29 mm) ~2.7 cm/px at 100 m; M350+P1 (35 mm) ~1.25 cm/px at 100 m. Motion blur = speed x exposure / GSD; 8 m/s x 1/1000 s = 8 mm << one pixel.
- **Failure modes:** Pure nadir single strip = no facade geometry + doming/bowl error; too fast = motion blur + baseline gaps; too slow = battery limit; hard shadows in low sun.
- **Alternatives:** Multi-strip grid (violates single-pass); orbit/POI for a single structure; terrain-follow to hold constant GSD.

### PPK geotagging as the metric anchor (RTKLIB / Emlid Studio / DJI Terra)  
*Maturity: production*

- **What:** Log raw GNSS observations (RINEX) on the drone and a base simultaneously, and post-process the trajectory after the flight against base/CORS data to get cm-level camera positions; feed those as the per-frame pose prior.
- **Why it fits single-pass:** Single-pass missions at borders/disaster sites often have poor or no live correction link. PPK needs no real-time link, can be reprocessed with better ephemerides, and degrades gracefully — it is the robustness-friendly accuracy source.
- **Perf/accuracy:** Typical no-GCP mapping: ~1-3 cm horizontal RMSE, ~2-6 cm vertical RMSE with mechanical shutter + correct lever arm; vertical can carry a systematic 5-10 cm bias unless one checkpoint is used.
- **Failure modes:** Systematic vertical/geoid/lever-arm bias with zero checkpoints; cycle slips; short baseline to base needed (<10-30 km); DJI raw-obs export varies by model.
- **Alternatives:** Live RTK (needs link); single-GNSS geotags (sub-meter, needs GCPs); LiDAR direct georef (Zenmuse L2).

### Fixed / strongly-prior camera intrinsics from EXIF-XMP + pre-flight calibration  
*Maturity: production*

- **What:** Pull DJI's calibrated focal, principal point and DewarpData distortion from image EXIF/XMP; for video (no per-frame EXIF) inject a single fixed intrinsic model obtained by a pre-flight OpenCV/Kalibr checkerboard calibration of that exact camera/zoom, and FIX (not free) intrinsics during bundle adjustment.
- **Why it fits single-pass:** A single straight strip cannot reliably self-calibrate — free focal length trades against depth/scale and causes doming. Fixing intrinsics is the specific change that stabilizes single-pass reconstruction.
- **Perf/accuracy:** Pre-flight OpenCV calibration reprojection error <0.3 px on a good target; removes the focal-depth ambiguity that can distort single-strip Z by several percent.
- **Failure modes:** Zoom/focus changes invalidate the fixed model; temperature-induced focal drift; consumer rolling-shutter cameras need a rolling-shutter model too.
- **Alternatives:** Full SfM self-calibration (risky single-pass); per-image XMP intrinsics for stills; rolling-shutter-aware BA (COLMAP/OpenSfM).

### DJI PSDK payload + onboard Jetson via E-Port (DjiTimeSync)  
*Maturity: near-production*

- **What:** Mount a Jetson Orin NX as a PSDK payload on the E-Port/SkyPort; use the DjiTimeSync API to discipline the payload clock to aircraft GPS time (PPS), subscribe to position/attitude/RTK telemetry, and draw 24 V power/thermal budget through the port.
- **Why it fits single-pass:** Gives on-aircraft compute + hardware time sync + telemetry on a supported, certified DJI platform — the realistic productization path for onboard near-real-time processing.
- **Perf/accuracy:** Manifold 3 = Orin NX, up to 100 TOPS INT8, 16 GB LPDDR5, 256 GB SSD, 33 W, 120 g; PSDK 3.16, DjiTimeSync PPS. Orin NX NVDEC decodes multiple 4K H.265 streams in real time.
- **Failure modes:** Manifold 3 officially targets M400/M4-series, NOT M350 (needs a third-party PSDK carrier or older Manifold 2 on M350); PSDK does not expose high-rate raw IMU or camera-IMU extrinsics; payload weight cuts endurance; in-flight thermal/power caps (15-25 W).
- **Alternatives:** Ground-station Jetson AGX Orin (275 TOPS) processing a downlinked stream; Manifold 2 (Jetson TX2, weaker); record-then-process off-board.

### Live video ingest: DJI Pilot 2 / MSDK / Cloud-API RTMP-RTSP -> MediaMTX -> GStreamer NVDEC  
*Maturity: production*

- **What:** Push the 1080p live feed from DJI Pilot 2 'Live Streaming' (RTMP/RTSP/GB28181/Agora) or the MSDK LiveStreamManager to a MediaMTX/SRS media server, and pull it on the compute node with GStreamer using hardware NVDEC (nvv4l2decoder) for zero-CPU H.264/H.265 decode.
- **Why it fits single-pass:** A no-custom-firmware, hackathon-friendly path to get real drone video into the pipeline in near-real-time for progressive/coarse reconstruction and operator preview.
- **Perf/accuracy:** O3 downlink 1080p/30, ~200 ms air-to-controller latency; end-to-end glass-to-glass typically 0.3-1 s via RTMP/RTSP; SRT adds ARQ loss recovery for lossy 4G/5G links at sub-second latency.
- **Failure modes:** Downlink capped at 1080p and lossy/compressed -> NOT metric-grade (must reconcile with full-res offline); RTMP has no loss recovery over WAN (use SRT); network stalls.
- **Alternatives:** SRT (srt-live-transmit) for lossy links; WebRTC/Agora for lowest latency; direct RTSP from custom drone GStreamer.

### DJI per-frame .SRT telemetry extraction  
*Maturity: production*

- **What:** Parse the .SRT subtitle sidecar DJI writes next to recorded video, which carries per-frame timestamp, GPS lat/lon, absolute/relative altitude and often gimbal/ISO/shutter — giving a ready-made per-frame pose stream with no SDK code.
- **Why it fits single-pass:** A concrete, zero-integration hook to georeference DJI video frames immediately in a hackathon; pairs with PPK to upgrade accuracy later.
- **Perf/accuracy:** Frame-rate-cadence GPS (~30 Hz interpolated); positions are standard-GNSS-tagged in many models (sub-meter), RTK precision only when RTK-fixed and model-dependent.
- **Failure modes:** SRT GPS often not full-RTK precision; field set varies by firmware/model; downlinked-recording SRT is coarser than onboard-recording SRT; no attitude on some cameras.
- **Alternatives:** MSDK KeyManager telemetry log; PSDK telemetry subscription; MAVLink log (custom).

### Dual-stream capture: full-res onboard recording + compressed proxy stream  
*Maturity: production*

- **What:** Record 4K/5.1K video (or interval mechanical-shutter 20-45 MP stills) to onboard SSD/SD for the metric deliverable, while transmitting only a 1080p proxy for live preview/coarse model. Reconcile the two by timecode after landing.
- **Why it fits single-pass:** Reconciles the real-time mandate with metric accuracy AND link-loss robustness: the accurate data never depends on the radio link.
- **Perf/accuracy:** Onboard 4K H.265 ~100-150 Mbps vs 1080p downlink ~10-15 Mbps; stills at 0.7 s interval (Mavic 3E mechanical shutter) give photogrammetry-grade, per-image-EXIF frames.
- **Failure modes:** Storage fills; timecode drift between proxy and master; extra offload/reconcile step; heat from onboard 4K encode.
- **Alternatives:** Stream-only (loses accuracy on link drop); record-only (loses real-time preview).

### RTK via NTRIP / RTCM3 (own base or CORS)  
*Maturity: production*

- **What:** Feed RTCM3 corrections to the drone rover over 4G/NTRIP from a national CORS caster or your own base (Emlid Reach RS2/RS3), giving cm-level camera positions in real time during the pass.
- **Why it fits single-pass:** Delivers immediate cm georef for near-real-time output when a correction link exists, and populates the per-frame pose prior live.
- **Perf/accuracy:** DJI RTK-fix 1 cm + 1 ppm H / 1.5 cm + 1 ppm V; Emlid RS3 base ~7 mm+1 ppm H PPK; fix acquisition seconds under open sky.
- **Failure modes:** Link/cell dead zones at borders/disaster sites; float/single fallback; base-rover baseline limits; ionospheric/multipath.
- **Alternatives:** PPK (no link needed); local RTK base broadcasting over 900 MHz radio; PPP (slow convergence).

### Multi-sensor time synchronization: PPS + PTP + gpsd/chrony/linuxptp (custom rigs)  
*Maturity: production*

- **What:** Route GNSS PPS into both the flight controller and the Jetson (pps-gpio + gpsd + chrony) and/or run linuxptp (ptp4l/phc2sys) for a hardware-disciplined clock; hardware-trigger the camera and log CAMERA_FEEDBACK (GPS time + pose) per exposure via MAVLink.
- **Why it fits single-pass:** On custom UAVs this is what makes per-frame georeferencing possible at the few-millisecond level required for cm accuracy at flight speed.
- **Perf/accuracy:** PPS discipline to <1 us; PTP <10 us on wired; camera hardware trigger + strobe timestamp aligns frame exposure to GPS time within one frame.
- **Failure modes:** USB/CSI cameras without hardware trigger have unknown exposure latency; software timestamps jitter 10-50 ms; USB video adds buffering delay.
- **Alternatives:** Post-hoc IMU-to-optical-flow cross-correlation to recover offset; genlock/timecode generators; DjiTimeSync on DJI.

### Camera-IMU spatial/temporal calibration with Kalibr (custom VIO)  
*Maturity: production*

- **What:** Use Kalibr (AprilGrid target) to solve camera intrinsics, camera-IMU extrinsics AND the temporal offset, with allan_variance_ros/imu_utils for the IMU noise model; export YAML consumed by the VIO/SfM front end.
- **Why it fits single-pass:** Tightly-coupled VIO on a custom global-shutter rig recovers metric scale and robust pose through the weak single-pass geometry and GPS gaps; requires accurate cam-IMU calibration to work.
- **Perf/accuracy:** Kalibr temporal offset to ~1 ms, extrinsic rotation <0.5 deg with a good AprilGrid sequence; enables VINS-Fusion/OpenVINS-class VIO drift ~0.1-1% of trajectory.
- **Failure modes:** Poor excitation (must rotate/translate all axes) gives bad calibration; DJI does not expose raw IMU so Kalibr is custom-rig-only; calibration drifts with temperature/vibration.
- **Alternatives:** Factory extrinsics (DJI); online self-calibration in VINS-Fusion; loosely-coupled GPS+visual.

### Global-shutter machine-vision camera + hardware trigger (custom alternative to consumer rolling shutter)  
*Maturity: near-production*

- **What:** On custom UAVs, use a global-shutter sensor (e.g., Sony IMX296/IMX264 via Arducam/e-con/Leopard on Jetson CSI, or a GigE Vision cam) with external hardware trigger and exposure-strobe feedback, on a Gremsy gimbal.
- **Why it fits single-pass:** Single fast passes maximize rolling-shutter jello and motion blur; global shutter + fast exposure eliminates the per-row distortion that corrupts video SfM and makes each frame a clean measurement.
- **Perf/accuracy:** Global shutter removes rolling-shutter skew entirely; IMX296 1.6 MP global @60 fps; hardware trigger ties exposure to GPS time; strobe timestamp <1 ms.
- **Failure modes:** Lower resolution than DJI 20-45 MP sensors; more integration effort; lens/mount/gimbal tuning; CSI cable length limits.
- **Alternatives:** Mechanical-shutter DJI stills (Mavic 3E/P1); rolling-shutter model in BA; industrial RGB (FLIR Blackfly).

## Recommended stack

| Component | Choice | Rationale |
|---|---|---|
| Primary aircraft (accuracy) | DJI Matrice 350 RTK + Zenmuse P1 (45 MP full-frame, mechanical shutter) or Zenmuse L2 (LiDAR+RGB) | Factory RTK 1 cm+1 ppm, mechanical shutter (no rolling-shutter distortion), calibrated intrinsics in EXIF/XMP, 55 min endurance, PSDK/MSDK support; L2 gives a direct-georeferenced point cloud fallback for vegetation/occlusion. |
| Primary aircraft (lightweight/agile) | DJI Mavic 3E (or 3T for thermal) | 4/3 20 MP with true mechanical shutter and built-in RTK in a sub-1 kg package; ~2.7 cm/px GSD at 100 m; ideal cheap hackathon-to-field platform; 3T adds 640x512 radiometric thermal for disaster/inspection. |
| Onboard companion compute | NVIDIA Jetson Orin NX 16 GB — DJI Manifold 3 (M400/M4-series) or third-party PSDK carrier / Manifold 2 for M350 | 100-157 TOPS at 10-40 W, hardware NVDEC for real-time 4K H.265 decode, 16 GB for SLAM/SfM; Manifold 3 is the certified DJI option (PSDK 3.6/OSDK 5.0, 33 W, 120 g). |
| Ground-station compute (fallback / heavy lift) | NVIDIA Jetson AGX Orin 64 GB or an RTX laptop | 275 TOPS / 204 GB/s to run near-real-time reconstruction on the downlinked proxy and full offline batch on the recorded master when onboard thermal/power is constrained. |
| Custom UAV option | PX4 on Pixhawk 6X (or ArduPilot) + global-shutter cam (Sony IMX296/264 via Arducam/e-con) + Gremsy Pixy gimbal | Full control of hardware trigger, raw IMU, PPS, and camera model for tightly-coupled VIO and sub-ms sync that DJI does not expose. |
| GNSS correction | NTRIP (national CORS or own Emlid Reach RS3 base, RTCM3 over 4G) for RTK; RTKLIB / Emlid Studio for PPK | RTK for live cm georef when a link exists; PPK as the link-independent, reprocessable metric anchor for border/disaster sites — the graceful path. |
| Video transport | DJI Pilot 2 / MSDK LiveStreamManager -> RTMP/SRT -> MediaMTX; GStreamer + NVDEC ingest; SRT for lossy 4G/5G | No-firmware live path from DJI; SRT ARQ recovers packet loss over cellular at sub-second latency; MediaMTX is a single-binary RTSP/RTMP/SRT/WebRTC server. |
| Telemetry / SDK | DJI MSDK v5 (KeyManager telemetry, LiveStream) / PSDK 3.16 onboard; MAVSDK + pymavlink for PX4/ArduPilot | MSDK/PSDK expose RTK position, gimbal attitude and time sync on DJI; MAVSDK gives GLOBAL_POSITION_INT, ATTITUDE_QUATERNION, CAMERA_FEEDBACK, TIMESYNC on custom. |
| Time synchronization | DjiTimeSync (PPS) on DJI; linuxptp (ptp4l/phc2sys) + gpsd + pps-gpio + chrony on custom | Sub-ms frame-to-GNSS-time alignment; sync error x ground speed is the dominant per-frame position error, so this must be <2-3 ms. |
| Calibration toolchain | Kalibr (AprilGrid) + allan_variance_ros/imu_utils + OpenCV checkerboard | Camera intrinsics, cam-IMU extrinsic+temporal, and IMU noise model for custom VIO; OpenCV pre-flight intrinsics to FIX the camera model for single-pass BA. |
| Per-frame telemetry parser | DJI .SRT sidecar parser + geotag/lever-arm module | Immediate per-frame GPS/altitude/gimbal for DJI video with zero SDK code; upgradeable to PPK trajectory. |

## Real-world integration

DJI path (fastest to real hardware): PSDK payload connects over the E-Port/SkyPort providing 24 V power, USB/UART/Ethernet and a data link; call DjiTimeSync to discipline the Jetson clock to aircraft GPS time (PPS) and subscribe to RTK position + gimbal attitude. Get video off the aircraft two ways in parallel: (a) LIVE — DJI Pilot 2 'Live Streaming' or MSDK v5 LiveStreamManager pushes 1080p RTMP/RTSP/GB28181/Agora to a MediaMTX/SRS server that the compute node pulls with GStreamer using NVDEC hardware decode (~0.3-1 s glass-to-glass, ~200 ms O3 air link); (b) MASTER — 4K/5.1K H.265 (or 0.7 s interval mechanical-shutter 20-45 MP stills) recorded to onboard storage, offloaded after landing. Exploit the DJI .SRT subtitle sidecar for per-frame timestamp/GPS/altitude/gimbal with no SDK code. Custom PX4/ArduPilot path: MAVLink over SiK 915 MHz or RFD900x to the GCS and to the companion via serial/UDP (MAVSDK/pymavlink); camera on Jetson CSI (global shutter, hardware-triggered) or USB3/GigE Vision; downlink via GStreamer RTSP/SRT over 4G/5G/WiFi-HaLow; PPS from the GNSS routed to both the flight controller and Jetson (pps-gpio + gpsd + chrony, or linuxptp for PTP); camera hardware trigger logs CAMERA_FEEDBACK (GPS time + pose) per exposure; gimbal via Gremsy over MAVLink gimbal protocol v2. Positioning: NTRIP client feeds RTCM3 to the rover; log raw obs (RINEX) for PPK; enter the measured camera-to-antenna LEVER ARM (typ. 10-20 cm) in mission config and correct it — an un-modeled lever arm biases the whole model. Convert RTK ellipsoidal height to orthometric with a geoid model (EGM2008 or local). Data formats to standardize on: video H.264/H.265 in .mp4/.ts; per-frame telemetry as DJI .SRT or a resampled CSV keyed to GPS time; MAVLink .tlog/.ulog; GNSS RINEX for PPK; intrinsics/extrinsics as Kalibr/OpenCV YAML; geotags/intrinsics in image EXIF/XMP for the stills path. Deployment: onboard Jetson runs coarse near-real-time reconstruction + preview; ground Jetson AGX/laptop runs the full-res + PPK batch for the accurate deliverable; DJI Cloud API / Dock 2 covers autonomous, repeat-monitoring digital-twin use cases.

## Reliability & failure handling

Design explicit accuracy TIERS and always emit something rather than hard-fail: (A) RTK-fix + fixed intrinsics + tight sync -> live metric cm; (B) PPK reprocessed offline -> metric cm even if the live link died; (C) GNSS single/float only -> sub-meter georef, flagged; (D) no GNSS -> VIO/SfM relative model with IMU/lever-arm-derived scale, marked 'un-georeferenced, relative scale'. Link loss: because full-res video and raw GNSS are logged ONBOARD independent of the radio, a dropped downlink loses only the live preview, not the mission; the drone executes RTH. RTK->float transitions are detected from the fix-quality flag and those frames are down-weighted or handed to PPK. Time-sync loss: fall back to post-hoc cross-correlation of IMU angular rate against frame optical flow to recover the frame-to-pose offset. Motion blur / compression artifacts: score frames by variance-of-Laplacian and reject blurry/keyframe-redundant ones before they enter BA; prefer I-frames. Dynamic objects: not a hardware fix, but capture SOP (higher altitude, oblique) plus downstream masking; hardware helps by giving accurate pose so moving points are easily flagged as reprojection outliers. Onboard thermal/power: cap Jetson to a 15-25 W power mode with a thermal watchdog; on overheat, degrade to record-only and defer processing to the ground node. Storage: pre-flight free-space check, bitrate cap, and circular-buffer option so recording never stalls the encoder. GNSS-denied/RF-denied (borders): switch to PPK-only + onboard/local RTK base over 900 MHz radio instead of cellular NTRIP. Every stage writes a health/QA record (fix quality, sync residual, blur rate, coverage/overlap) so the output carries an honest per-region uncertainty rather than silently failing.

## Single-pass specifics

What must change versus traditional multi-pass photogrammetry, at the hardware/capture layer: (1) FIX camera intrinsics from a pre-flight calibration (or EXIF/XMP DewarpData) instead of SfM self-calibration — a single straight strip cannot separate focal length from depth/scale, producing systematic 'doming'/bowl error; multi-pass grids average this out, a single pass cannot. (2) Per-frame RTK/PPK+IMU pose is NOT optional — it replaces both the missing cross-strip multi-view constraints and the missing GCP network; get it wrong (lever arm, sync) and there is no redundant geometry to absorb the error. (3) Shoot OBLIQUE (30-45 deg), not nadir — a nadir single strip sees no facades and has near-zero convergence angle; oblique buys the viewing-angle diversity that a second strip normally provides. (4) Fly slow and steady for >=90% along-track overlap with enough keyframe baseline for parallax; with only one strip there is no lateral overlap to fall back on, so along-track geometry must carry everything. (5) Add even a slight lateral weave or a single cross line if airspace allows — a tiny amount of cross-track motion dramatically improves self-consistency and kills the bowl error, and is far cheaper than a full second pass. (6) Rolling shutter and motion blur hurt more because a fast single pass has no redundant observations to dilute a bad frame — mandate mechanical/global shutter and fast exposure (>=1/1000 s). (7) Time-sync tolerance is tighter: with one strip, a frame's pose error is not cross-checked by an overlapping strip, so keep frame-to-GNSS sync <2-3 ms. (8) Lever-arm and geoid/vertical-datum correction matter more: without a multi-strip block or GCPs you cannot regress out a systematic vertical bias, so budget for one checkpoint or accept ~5-10 cm Z. (9) Occlusion is inherent to one viewpoint set — lean on an oblique gimbal, and consider LiDAR (Zenmuse L2) or NeRF/Gaussian-splatting-style view synthesis (downstream domains) to fill single-viewpoint gaps that multi-pass overlap would otherwise cover.

## Open risks

- DJI Manifold 3 officially targets M400/M4-series, NOT the M350 RTK — onboard compute on M350 depends on a third-party PSDK carrier board or the older/weaker Manifold 2 (Jetson TX2); verify the exact carrier + PSDK aircraft support before committing, and note M400 is very new (2025) with supply/availability uncertainty.
- DJI live downlink is capped at 1080p H.264 and is lossy/compressed — it is NOT metric-grade, so the pipeline MUST reconcile a live proxy against full-res onboard recordings, adding real complexity and a post-flight offload step.
- DJI SDKs do not expose high-rate raw IMU or camera-IMU extrinsics, so tightly-coupled VIO is impractical on DJI; you must rely on RTK + gimbal pose, which limits GNSS-denied robustness (custom global-shutter rigs are the answer but add integration effort).
- 'Metric cm accuracy without GCPs' is optimistic on the VERTICAL axis: PPK/RTK still carry a systematic geoid/lever-arm/antenna bias that typically needs one checkpoint; realistically budget ~5-10 cm Z with zero ground control.
- Time-sync precision is a hard dependency: sync_error x ground_speed sets per-frame position error (10 ms at 8 m/s = 8 cm); USB/CSI cameras without hardware trigger have unknown, jittery exposure latency that is hard to pin down.
- Connectivity assumptions break at borders/disaster zones: 4G/5G for NTRIP corrections and for streaming may be absent, forcing PPK-only + local RTK base and eliminating true real-time cm output there.
- Regulatory/BVLOS, RF spectrum, and payload-weight-vs-endurance tradeoffs constrain the flight profile and the onboard compute you can carry; onboard 4K encode + Jetson heat in a sealed enclosure is a real thermal risk.
- Rolling shutter on consumer/Mavic 3T cameras plus fast video creates skew that corrupts SfM unless explicitly modeled — prefer mechanical-shutter stills (Mavic 3E/P1) or a global-shutter custom camera for metric video.

## Differentiation notes

Off-the-shelf mapping suites (DJI Terra, Pix4D, Agisoft Metashape, RealityCapture, OpenDroneMap) ingest geotagged STILLS in a post-flight batch, generally assume multi-strip overlap (60-80% front/side) and often want GCPs, and produce results in minutes-to-hours of cloud/desktop compute — none is built for a single oblique VIDEO pass or for near-real-time edge output. This system differentiates by: (1) treating VIDEO as the primary metric sensor with per-frame GNSS-time pose (DjiTimeSync/PPS + trajectory interpolation + lever-arm), so a single pass is sufficient where COTS tools need a grid; (2) running near-real-time ON THE EDGE (Jetson Orin NX/AGX Orin, NVDEC + incremental SfM/SLAM) with a live preview and progressive refinement instead of an offline batch; (3) explicit graceful-degradation TIERS (RTK-fix -> PPK -> single-GNSS -> relative VIO) so it never hard-fails on link or GNSS loss — a robustness posture commercial suites lack; and (4) directly exploiting the DJI per-frame .SRT telemetry + PPK to georeference video WITHOUT GCPs, which no COTS product does natively. Practical stance for the hackathon: use DJI Terra / ODM as an offline 'oracle' and accuracy fallback for the high-fidelity deliverable, while the custom real-time, single-pass, degrade-gracefully edge pipeline is the actual differentiator that meets the NTRO mandate the COTS tools cannot satisfy alone.

## Citations / references

- DJI Matrice 350 RTK official specifications, enterprise.dji.com, 2024 (RTK 1 cm+1 ppm H / 1.5 cm+1 ppm V, 55 min, O3 Enterprise Transmission, PSDK payloads)
- DJI Mavic 3E/3T Enterprise specifications, enterprise.dji.com, 2024 (4/3 20 MP wide, mechanical shutter 8-1/2000 s, RTK 1 cm+1 ppm, O3 ~200 ms, 3T 640x512 thermal)
- DJI Developer Payload SDK (PSDK) v3.16.0 README and docs, developer.dji.com, 2025 (X-Port/SkyPort/extension port, DjiTimeSync, Manifold 3 default target, M4T/4E/M400)
- DJI Manifold 3 product specs (via reseller listings), 2024-2025 (NVIDIA Jetson Orin NX, up to 100 TOPS INT8, 16 GB LPDDR5, 256 GB SSD, USB 3.2/HDMI/Ethernet/UART/CAN, 33 W, 120 g, E-Port V2, OSDK 5.0/PSDK 3.6)
- NVIDIA Jetson Orin module datasheet, nvidia.com, 2024 (Orin NX 16 GB 157 TOPS 10-40 W; AGX Orin 64 GB 275 TOPS 15-60 W; Ampere GPU, Cortex-A78AE)
- Furgale, Rehder, Siegwart, 'Unified Temporal and Spatial Calibration for Multi-Sensor Systems' (Kalibr), IROS 2013; ethz-asl/kalibr wiki (CAM-IMU spatial/temporal + IMU intrinsics, AprilGrid, rolling-shutter model)
- T. Takasu, RTKLIB open-source GNSS (RTK/PPK, RTCM3, RINEX), 2013+; Emlid Studio and Reach RS2/RS3 PPK documentation
- Haivision / SRT Alliance, Secure Reliable Transport (SRT) — sub-second latency, ARQ packet-loss recovery over unpredictable networks
- MAVLink / MAVSDK and PX4/ArduPilot camera-trigger (CAMERA_FEEDBACK), gimbal protocol v2, TIMESYNC; linuxptp (ptp4l/phc2sys), gpsd + PPS + chrony
- Schonberger & Frahm, 'Structure-from-Motion Revisited' (COLMAP), CVPR 2016 — camera models, pose priors, fixed-intrinsics bundle adjustment
- MDPI Drones / Remote Sensing UAV RTK/PPK no-GCP accuracy studies, 2020-2023 (~1-3 cm horizontal, ~2-6 cm vertical RMSE with mechanical shutter; benefit of a single checkpoint for vertical)
- MediaMTX (RTSP/RTMP/SRT/WebRTC server) and GStreamer NVDEC/DeepStream hardware video pipeline documentation
