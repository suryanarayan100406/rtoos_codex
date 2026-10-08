# DRISHTI — Role: Drone & Sensor / Hardware Integration

**Purpose:** Define the physical capture system for DRISHTI and the S0 (Capture & Sync) stage — the
airframe, sensor suite, time synchronization, calibration, comms/downlink, and the physical
integration of the edge computer — and the time-synchronized sensor contract this role hands to the
rest of the pipeline.

**Audience:** National Technical Research Organisation (NTRO) technical evaluators (mapping, systems,
and signals engineers), hackathon judges, and the DRISHTI build team wiring drone hardware to the edge
software.

**TL;DR**
- This role **owns the physical capture system and stage S0** (Capture & Sync): airframe, sensors,
  clocks, calibration, downlink, and the physical mounting/power/thermal of the on-Uncrewed-Aerial-
  Vehicle (UAV) computer. It hands **time-synchronized sensor streams** to the Computer Vision role
  (S1) and the metric spine (S2); it does **not** own the reconstruction math (cite the pipeline in
  [`CANONICAL-ARCHITECTURE-SPEC`](../_internal/CANONICAL-ARCHITECTURE-SPEC.md), not this document).
- We run a **dual-track platform**: a Commercial-Off-The-Shelf (COTS) DJI **Matrice 350 RTK** +
  Zenmuse P1/L2 for fastest time-to-field and factory Real-Time Kinematic (RTK), and an open
  **PX4/ArduPilot** custom airframe with a global-shutter camera, survey Global Navigation Satellite
  System (GNSS), tactical Inertial Measurement Unit (IMU), and Jetson companion for full sensor access
  and sovereignty.
- The crux hardware function is **per-frame georeferencing**: hardware-timestamp every frame to GPS
  time, interpolate the RTK/Post-Processed-Kinematic (PPK) trajectory and gimbal/IMU attitude onto that
  epoch, and apply the antenna→camera lever arm. Per-frame position error ≈ `sync_error × ground_speed`,
  so we target **frame-to-GNSS sync < 2–3 ms**.
- **The official input contract is a hardware constraint, and it is generous to us in exactly one
  direction only.** Only **drone video (1080p/4K), GPS coordinates and flight metadata** are *mandatory*;
  **IMU, barometric altitude, camera intrinsics and RTK/PPK are all optional**. So the sensor suite below
  is the *ideal* instrument, not a prerequisite: the pipeline must run on a video + a GPS log from an
  unknown airframe, with intrinsics self-calibrated. This role therefore owns two jobs — specify the good
  instrument, **and** define what S0 emits when most of it is absent (§4, §9).
- Honesty (per [`PROBLEM_STATEMENT`](../_internal/PROBLEM_STATEMENT.md) §4): **"metric without Ground
  Control Points (GCPs)"** = fused GNSS + visual scale (with RTK/PPK and IMU folded in when fitted),
  sensor-configuration-dependent and measured against the official **≤ 1 m** bar, with the **vertical
  axis as the weak axis** (a ~5–30 cm bias typically needs one checkpoint, and any region past 1 m is
  flagged rather than averaged away).
- **Store-and-forward** is non-negotiable: near-raw H.265 + full-rate telemetry are recorded to onboard
  Non-Volatile Memory express (NVMe) so a dropped link loses only the live preview, never the mission —
  the accurate model is built from the guaranteed recording (reliability spine A4).
- The single-pass capture Standard Operating Procedure (SOP) is deliberately **not** multi-pass
  photogrammetry: oblique gimbal, slow/steady flight, ≥90% along-track overlap, intrinsics held stable
  through the pass (locked zoom/focus) so self-calibration has one consistent camera to solve for.
- **Turn diversity is now an accuracy requirement, not a nicety.** With no IMU on the mandatory-only
  configuration, metric scale rests on GNSS-baseline geometry, so the SOP asks for gentle heading and
  altitude variation across the pass. A dead-straight, constant-height line is the worst case for scale
  observability and the SOP says so (§8).
- **Capture parameters are the pipeline's cost function, so this role co-owns the < 15 minute ceiling.**
  The official bar is a finished model in **< 15 minutes for a 10-minute video**, and the ground tier's
  cost scales with *keyframe count × resolution* — both decided in the air by speed, altitude, overlap and
  capture resolution. A 90%-overlap 4K pass at 5 m/s hands downstream several times the work a leaner
  profile would, so the SOP states a target keyframe density explicitly (§8) instead of leaving it a
  by-product. The other speed lever this role owns is the **mid-flight head-start**: keyframe packages
  stream while the aircraft is still flying (§6), so the ground tier can begin refining before landing.
- **On the official rubric this role's evidence is indirect but load-bearing.** Frame-to-GNSS sync and
  lever-arm/boresight discipline feed **reconstruction accuracy (30%)**; the coverage SOP feeds **model
  completeness (20%)**; capture resolution and the head-start uplink feed **processing speed (20%)**. What
  this role publishes for the record is the *capture configuration* — `sensor_caps`, sync residual, fix
  quality, overlap achieved — because none of those three scores is interpretable without it (§9).

> The authoritative pipeline (three tiers, stages S0–S10, model choices, the metric spine A2 and
> reliability spine A4) is defined once in
> [`CANONICAL-ARCHITECTURE-SPEC`](../_internal/CANONICAL-ARCHITECTURE-SPEC.md). This role document
> conforms to it and details only the physical layer and S0.

---

## 1. Mandate & scope

This role delivers the physical instrument. Everything downstream — Visual-Inertial Odometry (VIO),
depth priors, fusion, meshing — assumes it is fed clean, time-stamped, calibrated sensor data on one
clock. Our mandate is to make that assumption true, and to be honest in the health record wherever it
is not.

We own the capture system end-to-end and stage **S0 (Capture & Sync)** as defined in the spec: hardware
time-stamping, aligning every frame to **GNSS and flight metadata** on a common clock — adding IMU and
barometer to that alignment *when those optional sensors exist* — **defaulting to self-calibrated
intrinsics** and using an operator-supplied calibration only as an initial guess, and emitting
per-sample quality flags. The output is the **time-synchronized sensor stream** that S1 (Ingest & Frame
QA) and S2 (Odometry & Localization) consume — in message terms, the `Keyframe` contract carrying
capture timestamp, pose prior, intrinsics, a **`sensor_caps` capability flag** naming which optional
sensors were actually present, and confidence (see [Integration](../04-INTEGRATION.md) §2).

That capability flag is this role's most important single output. Every downstream stage branches on it,
so S0 must never fake a sensor it did not have: a missing IMU is reported as absent, not silently
substituted with a zero-rate stream.

Boundaries with peers are drawn to avoid overlap:

| This role **owns** | Peer owns (boundary) |
|--------------------|----------------------|
| Airframe & platform selection, payload integration | Reconstruction stages S1–S10 — [`CANONICAL-ARCHITECTURE-SPEC`](../_internal/CANONICAL-ARCHITECTURE-SPEC.md) |
| Sensor suite, S0 capture & time synchronization | VIO / factor-graph math, and *what to do with GNSS numerically* — [Geospatial & Accuracy](5-geospatial-accuracy-research.md) |
| Calibration: intrinsics, camera↔IMU extrinsics, lever arm, boresight | Coordinate Reference System (CRS) / geoid transforms, uncertainty propagation — [Geospatial & Accuracy](5-geospatial-accuracy-research.md) |
| Comms/downlink transport + store-and-forward recording | Frame QA, keyframing, masking (S1/S3) — [Computer Vision & Video Intelligence](2-computer-vision-video-intelligence.md) |
| **Physical** mounting, power, thermal, vibration, camera wiring of the Jetson | **Runtime software** on the edge box (ROS 2, Isaac ROS, TensorRT, degradation-ladder logic) — [Systems & Edge/Compute](6-systems-edge-compute-optimization.md) |

The Geospatial boundary is worth stating twice: **we produce the raw material** — GNSS/RTK/PPK samples
tagged to GPS time, the measured lever arm and boresight, the raw GNSS observations (Receiver
Independent Exchange, RINEX) — and Geospatial decides *how to fuse them* (factor graph, robust kernels,
ellipsoidal→orthometric height). Get the timestamp or lever arm wrong here and no amount of downstream
math recovers it.

---

## 2. Platform strategy

We standardize on **two proven, obtainable platforms** rather than betting on one. Track A gets us onto
real hardware fastest and is defensible as a certified COTS system; Track B gives full sensor and
timestamp control and an indigenous, auditable path for defense deployment. Both present the *same edge
software contract*, so the pipeline is one system on two topologies, not two systems.

| Axis | Track A — DJI Matrice 350 RTK + Zenmuse P1/L2 | Track B — Custom PX4/ArduPilot airframe |
|------|-----------------------------------------------|------------------------------------------|
| Control / SDK | Payload SDK (PSDK) 3.x / Mobile SDK v5; closed flight stack | MAVLink / MAVSDK / pymavlink; open flight stack |
| Sensor access | RTK pose + gimbal attitude; **no raw high-rate IMU, no camera–IMU extrinsics** | Full raw IMU, Pulse-Per-Second (PPS), hardware trigger, open camera model |
| Camera / shutter | Zenmuse P1 (45 MP full-frame, **mechanical shutter**); L2 (LiDAR+RGB) | Global-shutter machine-vision sensor (Sony IMX296/264-class) |
| RTK / positioning | Factory RTK, **1 cm + 1 ppm** horizontal (vendor-reported) | Survey rover (u-blox ZED-F9P-class) + NTRIP / PPK |
| Time sync | DjiTimeSync (PPS) over E-Port | PPS + Precision Time Protocol (PTP) via `linuxptp` + gpsd/chrony |
| Tightly-coupled VIO | Impractical (no raw IMU) → relies on RTK + gimbal pose | Full VIO → anchors metric scale in GNSS-denied segments |
| Defensibility / time-to-field | **Fastest**; certified, warrantied platform | More integration effort; complete control |
| Sovereignty | Foreign COTS; firmware/supply-chain dependence | **Indigenous-buildable, auditable, air-gap friendly** |

**Why dual-track.** The DJI path is the realistic way to demonstrate on real hardware quickly: factory
RTK, a mechanical-shutter camera that sidesteps rolling-shutter distortion, calibrated intrinsics in
image Exchangeable Image File Format (EXIF) / Extensible Metadata Platform (XMP), and documented SDKs.
But DJI SDKs **do not expose high-rate raw IMU or camera–IMU extrinsics**, so tightly-coupled VIO — the
mechanism that carries metric scale through GNSS dropouts (spec §4) — is impractical there; DJI leans on
RTK + gimbal pose, which limits GNSS-denied robustness. The custom PX4/ArduPilot rig closes that gap
and removes the foreign-COTS sovereignty concern that matters for an NTRO deployment. The P1's
mechanical shutter and the custom rig's global shutter are two routes to the *same* requirement — a
clean, distortion-free frame from a moving UAV — which §3 explains.

Platform specifics, PSDK-carrier caveats, and the shared edge contract are detailed in
[Integration](../04-INTEGRATION.md) §6; hardware choices trace to spec §8.

---

## 3. Sensor suite deep-dive

The camera is the primary metric sensor; every other sensor exists to *georeference and scale* what the
camera sees. This section explains why each spec matters for a single fast pass, where honest limits
lie, and how each maps to a failure mode the health record must expose.

**Camera — global vs rolling shutter.** On a moving UAV a **rolling shutter** exposes image rows
sequentially, so a frame captured while translating/rotating is skewed row-by-row ("jello"); unmodeled,
this biases sub-pixel feature locations and therefore metric geometry — and a single pass has no
redundant overlapping strip to dilute a bad frame. We therefore mandate either a **mechanical shutter**
(DJI Zenmuse P1/Mavic 3E) or a **global shutter** (custom machine-vision sensor), plus a fast exposure
(≥ 1/1000 s). Motion blur scales as `speed × exposure / Ground-Sampling-Distance (GSD)`; at 8 m/s and
1/1000 s that is ~8 mm, well under one pixel. **GSD** (cm/pixel) follows `GSD ≈ AGL × pixel_pitch /
focal_length`: the M350 + P1 gives ~1.25 cm/px and the Mavic 3E ~2.7 cm/px at 100 m Above Ground Level
(AGL) (vendor-derived geometry). Higher resolution is not free — it raises encode/thermal load onboard.

**IMU — the scale and gravity anchor.** The accelerometer observes gravity (fixing roll/pitch and the
gravity-aligned frame) and, through pre-integration between keyframes, supplies the **metric scale** a
monocular camera cannot recover — this is what removes scale ambiguity *without* GCPs (spec §4). What
matters is **grade**: bias stability and noise density. A tactical/industrial-grade IMU on the custom
rig, run at 100–400 Hz and Kalibr-characterized, gives a trustworthy scale anchor; a cheap MEMS IMU
under low-excitation (near-constant-velocity) flight leaves scale weakly observable. DJI does not expose
raw IMU, which is precisely why Track B exists.

**GNSS — absolute georeference.** Multi-constellation reception (GPS/GLONASS/Galileo/BeiDou, and
**NavIC/IRNSS** — the Indian Regional Navigation Satellite System, relevant for sovereignty and
in-region availability) bounds drift and anchors an absolute datum. **RTK** via a Networked Transport of
RTCM (RTCM3) stream over NTRIP (Networked Transport of RTCM via Internet Protocol) — from a national
Continuously Operating Reference Station (CORS) or a local base — gives cm-level positions in real time;
**PPK** post-processes logged raw observations against a base after landing. We prefer PPK as the
*accuracy anchor* because it survives datalink loss and can be reprocessed with better ephemerides.

**Barometer, magnetometer, gimbal encoders.** The barometer damps vertical drift as an altitude factor;
the magnetometer initializes heading (and is down-weighted near steel or under electronic warfare); the
gimbal encoders supply the dynamic camera-to-body attitude per frame. All feed S0's per-frame pose
assembly.

| Sensor | Role in pipeline | Key spec that matters | Failure mode |
|--------|------------------|-----------------------|--------------|
| Camera (RGB) | Primary metric sensor → frames to S1, depth in S4 | Shutter type (global/mechanical); GSD (cm/px); resolution | Rolling-shutter skew; motion blur; H.265 compression artifacts |
| IMU | Metric **scale** + gravity anchor for the S2 factor graph | Grade / bias stability; noise density; 100–400 Hz rate | Bias drift; weak scale under low excitation; MEMS noise |
| GNSS (multi-constellation + NavIC) | Absolute georeference + drift bound (S2) | Fix type (single/float/RTK-fix); constellations incl. NavIC | Multipath/canyon; jamming/spoofing; float fallback |
| RTK / PPK correction | cm-level absolute position prior | Base baseline < 10–30 km; RTCM3 link **or** logged RINEX | RTK link loss; PPK cycle slips; base too distant |
| Barometer | Vertical-drift damping; altitude factor | Relative-altitude stability | Pressure drift with weather / temperature |
| Magnetometer | Heading initialization & yaw disambiguation | Heading reference quality | Distortion near ferrous mass / EW; must be down-weighted |
| Gimbal encoders | Camera↔body attitude per frame | Encoder resolution; dynamic transform | Backlash / latency; uncalibrated boresight |

---

## 4. S0 — capture & synchronization

This is the hard part, and the make-or-break one. S0's job (spec S0 definition) is to place **every
frame, IMU sample, GNSS fix, and barometer reading on one GPS-disciplined clock**, attach a metric pose
prior to each frame, log intrinsics, and stamp per-sample quality flags — then hand the result to S1/S2.

**Why sync error directly limits accuracy.** A frame's pose is only as good as the timestamp that ties
it to the trajectory. Per-frame position error ≈ `sync_error × ground_speed`: at 8 m/s, a 10 ms desync
smears a frame ~8 cm. For cm-class work we budget **< 2–3 ms**. On a single pass this tolerance is
*tighter* than in multi-pass photogrammetry, because there is no overlapping strip to cross-check a
mis-timed frame.

**Hardware time-stamping.** On the custom rig, GNSS **PPS** disciplines both the flight controller and
the Jetson (`pps-gpio` + `gpsd` + `chrony`), and **PTP (IEEE-1588)** via `linuxptp` aligns compute
nodes; a hardware camera trigger logs exposure at GPS time (PPS to < 1 µs, PTP to < 10 µs wired —
vendor/protocol figures). On DJI, **DjiTimeSync** disciplines the payload Jetson clock over the E-Port
PPS; for zero-SDK ingest, the per-frame `.SRT` sidecar carries timestamp/GPS/gimbal at frame cadence.
S0 then interpolates the RTK/PPK trajectory and gimbal/IMU attitude onto each frame's exact epoch and
applies the antenna→camera **lever arm**.

```mermaid
sequenceDiagram
  participant GNSS as GNSS + PPS
  participant CLK as Host clock (chrony/PTP)
  participant IMU as IMU 100-400 Hz
  participant CAM as Camera
  participant S0 as S0 Capture & Sync
  participant DS as Downstream S1 / S2
  GNSS->>CLK: PPS + GPS time (discipline < 1 us)
  loop per frame epoch
    IMU->>S0: samples @ GPS-time stamps
    GNSS->>S0: position / RTK fix @ 5-10 Hz
    CAM->>S0: frame + exposure epoch (trigger or PTS)
    Note over S0: stamp frame on GPS clock;<br/>correct decoder + B-frame latency
    S0->>S0: interpolate trajectory + gimbal/IMU attitude to epoch
    S0->>S0: apply antenna to camera lever arm
    S0->>S0: attach per-sample validity + temporal-uncertainty flags
    S0->>DS: time-synced keyframe bundle (frame + pose prior + confidence)
  end
```

**Intrinsics logging.** Camera intrinsics are an **optional** input, so S0 handles three cases and says
which one it is. (1) Stills/EXIF path: pull DJI's calibrated focal, principal point, and DewarpData
distortion from EXIF/XMP. (2) Our own aircraft: inject a single intrinsic model from the pre-flight
calibration (§5) as a **strong prior** — `intrinsics_fixed = true` where the optics are genuinely locked,
which is the specific change that stabilizes weak single-pass bundle adjustment. (3) **Default, and the
only case the official contract guarantees:** no calibration and no usable EXIF, so S0 emits
`intrinsics_fixed = false` with a focal-length estimate from the depth backbone as an initial guess, and
S6 **self-calibrates**. Case 3 is a supported path, not a flag-and-hope; what S0 must never do is present
a guessed intrinsic model as a measured one.

**Per-sample quality flags.** Each sample carries validity: GNSS fix type (single/float/RTK-fix) and
Horizontal Dilution of Precision (HDOP), IMU saturation/clipping, gimbal-encoder validity, and a
**temporal-uncertainty flag**. These flags are the confidence signal S0 contributes to the reliability
spine (A4).

**Fallback (spec S0 fallback).** If PPS is absent, S0 drops to **software timestamp interpolation** and
**raises the temporal-uncertainty flag** on affected samples (software stamps jitter 10–50 ms — enough
to matter at speed). If sync is lost entirely, a post-hoc cross-correlation of IMU angular rate against
frame optical flow recovers the offset offline. Nothing is silently trusted; the offset is made
observable so Geospatial can down-weight it.

---

## 5. Calibration

Calibration is where single-pass reconstruction is quietly won or lost. A single straight strip is the
hard case for self-calibration — free focal length trades against depth/scale and produces systematic
"doming." So wherever we own the aircraft we supply strong, pre-measured priors; and because calibration
is an **optional** input under the official contract, we also make the un-calibrated path work rather
than merely tolerating it (ADR-16 in [Design Decisions](../05-DESIGN-DECISIONS.md)).

- **Camera intrinsics.** Pre-flight OpenCV checkerboard calibration of the exact camera/zoom/focus,
  targeting reprojection error < 0.3 px (design target on a good board), or DJI EXIF/XMP DewarpData for
  the stills path. Where that calibration exists and the optics are locked, intrinsics enter bundle
  adjustment **tightly constrained** rather than free-solved. Where it does not — the guaranteed
  mandatory-only case — S6 solves them, initialized from the depth backbone's focal estimate and
  conditioned by the turn/altitude diversity the SOP asks for (§8); the accuracy report then states that
  the pass was self-calibrated.
- **Camera↔IMU extrinsics + temporal offset.** Solved once with **Kalibr** (AprilGrid), with
  `allan_variance_ros` / `imu_utils` for the IMU noise model; Kalibr recovers the temporal offset to
  ~1 ms and extrinsic rotation to < 0.5° given a well-excited sequence. This is a **custom-rig-only**
  step — DJI does not expose raw IMU. Output is a YAML consumed by S2 (per the coordinate-frame handoff
  in spec §6: camera in OpenCV Right-Down-Forward, body/IMU in ROS REP-103 Forward-Left-Up).
- **Gimbal boresight & GNSS lever arm.** Measure the antenna→camera lever arm (typically 10–20 cm) to
  **1–2 cm** and the gimbal mounting boresight; an unmodeled lever arm biases the *entire* model. These
  are recorded in mission config and handed to Geospatial.
- **Pre-calibrate vs self-calibrate — which is the default depends on who owns the aircraft.** Camera
  intrinsics are an **optional** input under the official contract, so **self-calibration (S6) is the
  system default** and must be first-class: unknown intrinsics is the configuration we are guaranteed to
  face, not an exception. Where *we* fly the aircraft, pre-calibration is the specific stabilizer for
  single-pass and we take it — locked lens/zoom, verified reprojection error, intrinsics passed as a
  strong prior rather than a hard constraint so S6 can still absorb thermal/focus drift. Where we are
  handed someone else's video, S6 solves intrinsics outright and the accuracy report states that the
  pass was self-calibrated. Both paths are supported; neither is labelled a failure.

**Pre-mission SOP (calibration).** (1) Verify lens/zoom/focus locked; (2) run OpenCV intrinsics on the
day's rig, confirm reprojection < 0.3 px; (3) confirm Kalibr cam–IMU YAML is current (re-run after any
remount, hard landing, or large temperature change — vibration and thermal cycling drift extrinsics);
(4) measure/confirm lever arm and boresight; (5) confirm geoid model and target CRS with Geospatial;
(6) log all of the above into the reproducible project bundle before takeoff.

---

## 6. Comms & downlink

The link carries a **preview**; the recording carries the **truth**. This is how DRISHTI honors
"real-time" honestly — near-real-time edge preview plus minutes-scale ground refinement — without ever
letting the accurate model depend on radio quality.

**Dual-stream capture.** We record near-raw H.265 (or interval mechanical-shutter stills) + full-rate
telemetry to onboard NVMe, while downlinking only a compressed proxy for the live coarse map and
operator situational awareness. Video transport is **RTSP** (Real-Time Streaming Protocol) or, preferred
on lossy 4G/5G/RF links, **SRT** (Secure Reliable Transport) — Automatic-Repeat-reQuest + Forward Error
Correction with a 120–500 ms recovery buffer — carried through a MediaMTX-class server and decoded with
hardware NVDEC. Telemetry rides **MAVLink** (or PSDK subscription / `.SRT` sidecar on DJI). RTK
corrections come **up** the link as RTCM3 over an NTRIP client.

**Store-and-forward.** S0 writes to an NVMe circular buffer with monotonic IDs and hardware timestamps,
independent of the radio. On link loss the buffer persists and the live preview freezes, but **recording
never stops**; on reconnect or after landing the gap is backfilled so the Ground Tier sees a complete
stream (reliability ladder, `Always` row). Where no correction link exists (borders, disaster zones), we
log RINEX and rely on PPK — the link-independent anchor.

**The link is also a speed mechanism, not only a resilience one.** Because keyframe packages (pose priors
+ downsampled depth + masks + confidence) flow up *during* the flight, the Ground Tier can start S6/S7 on
the earliest keyframes before the aircraft lands — which is the difference between the **< 15 minute**
clock starting at landing and starting shortly after take-off. When the link is poor the clock simply
starts later and the ground tier's deadline scheduler absorbs it
([Systems & Edge/Compute](6-systems-edge-compute-optimization.md) §2); the mission itself is never at
risk, because the master recording is onboard either way.

| Data | Onboard NVMe (master) | Downlink (SRT) — indicative budget |
|------|----------------------|--------------------------------------|
| Reconstruction video | near-raw H.265, ~100–150 Mbps @ 4K | — (rebuilt from master) |
| Preview video | — | 1080p H.265 proxy, ~2–10 Mbps |
| Telemetry | full-rate IMU/GNSS/baro/RTK + RINEX raw | keyframe poses only |
| Keyframe package | full-res refs + depth + masks | downsampled depth + masks + confidence, a few Mbps |
| Live map | — | incremental map blocks, ~0.5–2 Mbps |

The keyframe package fits comfortably inside a lossy 2–20 Mbps RF link — versus 20–50 Mbps for raw 4K —
which is precisely why we stream keyframes, not full video. Vendor link figures (DJI O3: 1080p/30,
~200 ms air-to-controller; glass-to-glass ~0.3–1 s) are reported as such; the byte budgets above are
**design targets**. The transport/QoS split is detailed in [Systems & Edge/Compute](6-systems-edge-compute-optimization.md)
and [Integration](../04-INTEGRATION.md) §5.

---

## 7. Edge compute integration

This role owns the **physical** integration of the edge computer — mounting, power, thermal, wiring,
vibration — while [Systems & Edge/Compute](6-systems-edge-compute-optimization.md) owns the **software**
that runs on it. The edge computer is the on-UAV NVIDIA Jetson Orin of the Edge Tier (spec §2).

**Board & mounting.** The endurance choice is a **Jetson Orin NX 16 GB** (vendor: 100–157 TOPS,
10–40 W, 102.4 GB/s, 1× Deep Learning Accelerator (DLA)); the heavy-lift choice is a **Jetson AGX Orin
64 GB** (vendor: 248–275 TOPS, 15–60 W, 204.8 GB/s, 2× DLA). On DJI, the certified route is a Manifold
3-class carrier (Orin NX, vendor: 33 W, 120 g) on the E-Port — with the caveat that Manifold 3 targets
the M400/M4-series, **not** the M350, so onboard compute on the M350 needs a third-party PSDK carrier
(open risk, §11). The board is hard-mounted to the payload frame with its heatsink in clean airflow.

**Camera interface.** Fixed global-shutter machine-vision sensors connect over **Camera Serial Interface
(CSI)** or **Gigabit Multimedia Serial Link (GMSL)** with hardware trigger; quick-integration cameras
use **USB3 / USB Video Class (UVC)** at the cost of unknown exposure latency (flagged in the sync
budget, §4). Decode is zero-copy through NVDEC.

**Power & thermal (indicative / design targets).** In flight the Jetson is capped to a **~15–25 W**
power mode (`nvpmodel` + `jetson_clocks`) drawing off the E-Port 24 V rail or airframe supply. A sealed
UAV enclosure plus onboard 4K encode is a real thermal risk; a **thermal watchdog** reads `tegrastats`
and, on sustained throttle, transitions the edge to **ladder L6** (point-splat preview, defer meshing to
the ground) so heat degrades fidelity rather than crashing a node. Detection/segmentation offload to the
DLA frees the GPU for depth + fusion.

**Weight, payload & vibration.** Every gram of compute trades against endurance, so the board, carrier,
and cabling are weight-budgeted against the airframe's payload limit. The camera and IMU are mounted on
**vibration isolation** (the IMU rigidly co-located with the camera to preserve the Kalibr extrinsic);
motor vibration corrupts IMU integration and induces rolling-shutter/mechanical artifacts if it reaches
the sensor. Cable strain relief protects the CSI/GMSL runs, whose length is physically limited.

---

## 8. Capture SOP for single-pass

The flight profile is a first-class part of the design: it is the cheapest place to fix the
"limited-viewing-angles" challenge (spec §9, challenge i). What must change versus a multi-pass
lawnmower grid:

- **Oblique gimbal, 30–45°** (not pure nadir). A nadir single strip sees no facades and has near-zero
  convergence angle; oblique buys the viewing-angle diversity a second strip would normally provide, and
  lets one pass see rooftops **and** facades **and** ground.
- **Slow, steady, ~5–8 m/s at ~80–120 m AGL** for **≥ 90% along-track overlap** with usable keyframe
  parallax. With only one strip there is no lateral overlap to fall back on, so along-track geometry must
  carry everything.
- **Slight lateral weave or a single cross line if airspace allows.** A little cross-track motion
  dramatically improves self-consistency and kills the "bowl"/doming error — far cheaper than a full
  second pass.
- **Fast/mechanical (or global) shutter, ≥ 1/1000 s** to keep motion blur under a pixel (§3).
- **Intrinsics held stable through the pass** — locked lens/zoom/focus so there is one consistent camera
  to solve for. Where pre-flight calibration exists (§5) it is passed as a strong prior; where it does
  not, S6 self-calibrates, which is the default for third-party footage.
- **Gentle heading and altitude variation across the pass.** With no IMU (the mandatory-only
  configuration) metric scale rests on GNSS-baseline geometry, so a dead-straight constant-height line is
  the worst case for scale observability. A lazy S-curve or a slow altitude ramp costs nothing and
  conditions the scale solution.

**The SOP is a deadline decision as well as an accuracy one.** Every parameter above prices the ground
tier's work: at 5–8 m/s with ≥ 90% along-track overlap a 10-minute pass yields on the order of a few
hundred usable keyframes, S4→S8 cost is roughly linear in that count, and depth and texture cost scale
with resolution. The SOP therefore carries a **target keyframe density** — a working figure of ~2–4
keyframes per second of flight at 4K, to be tightened once the ground tier's throughput is measured — so
that a longer or faster pass is traded off *in the flight plan* rather than discovered as a missed
**< 15 minute** deadline after landing. Flying 4K rather than 1080p is likewise a deliberate purchase of
ground-sample distance at a known cost in processing time, and it goes in the mission log because the
accuracy report has to state which was flown.

**Honest single-pass constraints.** Occlusion is *inherent* to one viewpoint set — backsides and
grazing-angle surfaces cannot be triangulated and will be flagged low-confidence or filled by learned
priors downstream (S7), never presented as measured. The **vertical axis remains weak**: with zero GCPs
a systematic ~5–30 cm vertical bias (geoid/lever-arm/boresight) is common, so we budget for one
checkpoint or a precise local geoid to remove it (accuracy budget, spec §12). The SOP maximizes
*reconstructability in one pass*; it does not pretend a single pass equals a survey grid.

---

## 9. Reliability & confidence

This role's contribution to the reliability spine (A4) is to **never lose data and never hand downstream
an unlabeled sample**. Sensor dropouts are mapped explicitly onto the canonical graceful-degradation
ladder (spec §5), and each is *detected*, *flagged*, and *acted on* rather than allowed to fail silently.

| Trigger (this role detects) | Detection signal | Response | Ladder |
|-----------------------------|------------------|----------|--------|
| No RTK/PPK correction, or none fitted (it is *optional*) | `gnss_fix_type` leaves RTK-fix, or the `sensor_caps` RTK bit is unset | Continue on GNSS + visual scale (plus IMU where fitted); flag reduced georef; defer to PPK if RINEX was logged | **L1** |
| No IMU fitted (mandatory-only capture) | `sensor_caps` IMU bit unset | Emit keyframes with no attitude prior and no gravity vector; S2 runs visual odometry scaled by GNSS baselines — the **default path, not a fallback**; report wider vertical uncertainty | **L1** |
| No camera intrinsics supplied (they are *optional*) | no calibration file, no usable EXIF | `intrinsics_fixed = false`; S6 self-calibrates; flag the pass as self-calibrated in the accuracy report | **L1** |
| GNSS dropout (canyon / denied / jammed) | fix = 0, HDOP spikes | VIO+IMU dead-reckon; re-anchor on GNSS re-acquire | **L2** |
| Brief visual loss (blur/occlusion) | frame QA + IMU cross-check | IMU inertial propagation; short gap flagged high-uncertainty | **L3** |
| PPS absent / clock sync lost | `clock_sync_residual_ms` over budget | Software timestamp interpolation; raise temporal-uncertainty flag; post-hoc IMU/flow correlation | S0 fallback |
| Downlink lost | heartbeat timeout | Store-and-forward continues; preview freezes; **recording never stops** | Always |
| Edge thermal/power saturated | `soc_temp_c`, throttle | Point-splat preview; defer meshing to ground | **L6** |

**What this role guarantees downstream.** (1) Every sample is stamped on a single GPS-disciplined clock
and carries a validity + temporal-uncertainty flag. (2) The onboard recording is **complete** regardless
of link or compute pressure (store-and-forward), so the accurate model is always recoverable. (3) The
calibration set — intrinsics (measured where available, self-calibrated otherwise), Kalibr cam–IMU
extrinsics, lever arm, boresight — is delivered and versioned into the project bundle, each entry marked
measured or solved so nothing downstream mistakes an estimate for a measurement.
 (4) RTK-fix→float and GNSS-denied transitions are surfaced as fix-
quality flags so Geospatial can down-weight or hand those frames to PPK. These are the honest inputs the
metric spine (A2) needs; the numeric fusion and CRS/geoid handling belong to
[Geospatial & Accuracy](5-geospatial-accuracy-research.md).

---

## 10. Hackathon Minimum-Viable-Product (MVP) responsibilities

The event provides a dataset (drone video + GPS + flight metadata), not a drone — which is precisely the
**mandatory-only** configuration the official brief guarantees. So this role does not fly hardware at
the hackathon; it **demonstrates the capture contract** that real hardware would satisfy, proves the
pipeline runs with the optional sensors absent, and tells a credible hardware-readiness story.


**What we demonstrate:**

1. **Sync / calibration / SOP design** — this document plus the calibration YAML schema and the
   pre-mission SOP checklist, presented as the design that makes single-pass metric accuracy possible.
2. **Emulation of the capture contract on the provided data.** Parse the DJI `.SRT` sidecar (or MAVLink
   log) for per-frame timestamp/GPS/altitude/gimbal; extract frame presentation timestamps with ffmpeg
   and prefer I-frames; align frame Presentation-Time-Stamps to the telemetry clock; spline-interpolate
   the trajectory and attitude onto each frame; apply a nominal lever arm; and emit **`Keyframe`-shaped
   records** (timestamp + pose prior + fix type + intrinsics + confidence) — exactly the S0 output S1/S2
   expect (see [Integration](../04-INTEGRATION.md) §2). The edge/ground split is emulated on one
   workstation with a loopback link (spec §11).
3. **Graceful-degradation demo.** Inject GPS noise and motion blur into the provided stream and show the
   flags flipping and the ladder transitioning (RTK→GNSS-only→VIO-relative), proving the reliability
   spine is real rather than asserted.
4. **Hardware-readiness story.** Show the dual-track platform table, the PSDK/MAVLink integration path,
   the power/thermal budget, and the store-and-forward recorder — i.e. *what real integration looks
   like* — so evaluators see a clear line from the emulated contract to a flown system.

5. **Account for S0's share of the 15 minutes.** Time the emulated capture contract end-to-end — sidecar
   parse, PTS extraction, trajectory and attitude interpolation, `Keyframe` emission — and report it
   against the ~1-minute ingest allowance in the ground-tier budget
   ([Systems & Edge/Compute](6-systems-edge-compute-optimization.md) §2). This is the cheapest stage in the
   chain and must stay that way: a slow sidecar parser is an easy way to spend two minutes of a
   fifteen-minute budget before any geometry has run.

This makes the "real hardware" and "single-workstation" demos the **same S0 contract on a different
topology**, not two different stories.

---

## 11. Open questions / risks

- **Onboard compute on the DJI Matrice 350.** Manifold 3 targets the M400/M4-series, not the M350;
  onboard Jetson on the M350 needs a third-party PSDK carrier (or the older, weaker Manifold 2). Verify
  the exact carrier + PSDK aircraft support before committing; the M400 is new (2025) with supply
  uncertainty. (Dossier `../_internal/research/4-drone-sensor-hardware.md`.)
- **Time-sync on COTS cameras.** USB/CSI cameras without a hardware trigger have jittery, unknown
  exposure latency; hitting the < 2–3 ms budget may require a global-shutter rig with strobe feedback,
  which DJI SDKs do not expose. This bounds GNSS-denied VIO robustness on DJI platforms.
- **No raw IMU on DJI.** Tightly-coupled VIO is impractical on DJI, so GNSS-denied resilience on Track A
  is limited to RTK + gimbal pose; the custom Track B rig is the answer but adds integration effort.
- **Vertical accuracy without a checkpoint.** A systematic ~5–30 cm vertical bias (geoid/lever-arm/
  boresight) is common with zero GCPs; "metric without GCPs" is honest only with sensor-configuration-
  dependent numbers, and sub-decimetre vertical effectively needs one checkpoint or a precise local
  geoid.
- **Resolution and overlap versus the deadline is an unmeasured trade.** 4K at ≥ 90% overlap is the right
  capture for the **≤ 1 m** accuracy bar, and it is simultaneously the most expensive input the ground tier
  can be handed. Whether the full-fidelity chain clears **< 15 minutes** on a 10-minute 4K pass — and if
  not, whether the honest answer is a leaner keyframe density, 1080p capture, or a ground-tier quality dial
  — is a measurement, not a preference. First test: the same scene at 4K and at 1080p with **both**
  accuracy and wall-clock recorded, so the trade is stated in numbers.
- **Connectivity at borders/disaster zones.**
 4G/5G for NTRIP and streaming may be absent, forcing
  PPK-only + a local RTK base over 900 MHz radio and eliminating true real-time cm output there.
- **Thermal on a sealed UAV.** Onboard 4K encode + Jetson heat in a sealed enclosure is a genuine risk;
  the thermal watchdog → L6 path must be validated on the real airframe, not assumed.
- **Rolling shutter on consumer cameras.** Consumer/Mavic 3T video needs an explicit rolling-shutter
  model; we prefer mechanical-shutter stills (Mavic 3E/P1) or a global-shutter custom camera for metric
  video.
- **Spec conformance.** This document follows the canonical spec's tiers, stages, frames (§6), hardware
  reference (§8), reliability ladder (§5), and accuracy budget (§12). No conflicts were found; message-
  contract details are authored in [Integration](../04-INTEGRATION.md) and are proposals pending
  implementation, not measured results.

## 12. Further reading

- [`CANONICAL-ARCHITECTURE-SPEC`](../_internal/CANONICAL-ARCHITECTURE-SPEC.md) — authoritative tiers,
  stages S0–S10, coordinate frames (§6), hardware reference (§8), reliability ladder (§5), accuracy
  budget (§12).
- [`PROBLEM_STATEMENT`](../_internal/PROBLEM_STATEMENT.md) — problem, Desired Output & Evaluation tables,
  honesty guardrails.
- [Integration](../04-INTEGRATION.md) — message contracts (`Keyframe`/`MapUpdate`/`HealthStatus`), time
  synchronization, edge↔ground protocol, control vs data plane.
- [Systems & Edge/Compute](6-systems-edge-compute-optimization.md) — runtime software on the edge box,
  transport/QoS, degradation-ladder logic, observability.
- [Geospatial & Accuracy](5-geospatial-accuracy-research.md) — factor-graph fusion, lever-arm/boresight
  numerics, CRS/geoid handling, ASPRS accuracy assessment.
- [Computer Vision & Video Intelligence](2-computer-vision-video-intelligence.md) — the S1 video
  front-end, keyframing, and dynamic masking that consume this role's S0 output.
- Dossier: [`../_internal/research/4-drone-sensor-hardware.md`](../_internal/research/4-drone-sensor-hardware.md)
  — capture platforms, per-frame georeferencing, time sync, PSDK/MAVLink integration detail.
