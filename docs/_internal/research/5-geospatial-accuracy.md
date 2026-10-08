# Research dossier — Geospatial & Accuracy: metric correctness, georeferencing, and validation without extensive GCPs for single-pass drone video reconstruction

> Auto-generated from the research workflow. Source material for document authoring.

## Executive summary

For single-pass drone video, metric accuracy cannot be recovered from photogrammetric self-calibration alone: one flight line gives weak convergence geometry, few/no loop closures, and near-degenerate bundle adjustment, so scale and georeferencing MUST be injected from onboard sensors rather than solved from imagery. The recommended core is a tightly-coupled Visual-Inertial-GNSS estimator on a factor graph (GTSAM 4.2 / iSAM2 with Forster IMU preintegration, GNSS/RTK constraints, and visual reprojection factors); the IMU supplies metric scale, GNSS bounds drift and provides an absolute datum. With RTK/PPK, direct georeferencing reaches roughly 1-3 cm horizontal and 2-7 cm vertical RMSE with no GCPs, but a single ground checkpoint is strongly advised because an uncorrected vertical bias of 10-30 cm is common. All elevations must be converted from GNSS ellipsoidal height to orthometric using a geoid model (H = h - N) — critical in India where geoid undulation reaches about -100 m in the south — and every product tagged with an explicit EPSG/UTM CRS and vertical datum. Because single-pass accuracy is highly heterogeneous (well-triangulated nadir ground vs. barely-seen facades and occlusions), the system must propagate and expose per-region uncertainty (BA covariance, ray convergence angle, observation count, GSD) and validate against ASPRS Positional Accuracy Standards Edition 2 (2023) checkpoint RMSE (horizontal 95% = RMSEr x 1.7308, vertical 95% = RMSEz x 1.96). Graceful degradation comes from a tiered pipeline — RTK -> PPK -> GNSS+IMU -> VIO-only local frame — that always emits a labeled, georeferenced result and never hard-fails. Outputs use open, interoperable georeferenced formats (LAS 1.4/LAZ/COPC, Cloud-Optimized GeoTIFF DSM/DTM/orthomosaic, glTF/GLB, OGC 3D Tiles 1.1, CityJSON 2.0) consumable in QGIS, CesiumJS, and digital-twin platforms.

### Tightly-coupled VIO+GNSS factor-graph fusion (GTSAM 4.2 / iSAM2)  
*Maturity: production*

- **What:** Maximum-a-posteriori estimation over a factor graph fusing on-manifold IMU preintegration factors, GNSS position (or RTK) constraints, barometric-altitude factors, and visual reprojection/landmark factors; iSAM2 gives incremental near-real-time re-optimization as new keyframes arrive. GTSAM ships RTK GNSS double-difference factors and IMU preintegration.
- **Why it fits single-pass:** Single-pass video has weak multi-view geometry and almost no loop closures, so scale and absolute position cannot come from imagery; the IMU delivers metric scale and GNSS bounds drift and anchors an absolute datum. The graph naturally carries covariances, giving the per-region uncertainty the single-pass problem demands.
- **Perf/accuracy:** iSAM2 incremental updates run in milliseconds; relative trajectory error typically 0.1-1% of path length, with absolute accuracy pinned to the GNSS constraint (1-3 cm with RTK, 1-5 m with standard GNSS). Covariance recovery is native.
- **Failure modes:** GNSS multipath/spoofing injects large errors unless robust/switchable constraints (DCS, Huber) are used; poor IMU initialization or low motion excitation corrupts scale; requires accurate camera-IMU-GNSS time sync (sub-ms) and lever-arm/boresight calibration.
- **Alternatives:** Custom Ceres or g2o graph; VINS-Fusion (loose GPS); OpenVINS+GNSS; maplab 2.0; Kimera.

### Direct georeferencing with RTK/PPK GNSS + lever-arm  
*Maturity: production*

- **What:** Camera exterior orientation obtained directly from onboard RTK/PPK GNSS position plus INS attitude and the measured GNSS-antenna-to-camera lever arm and boresight, so each frame's projection center is known in an absolute CRS to centimeters without a GCP network.
- **Why it fits single-pass:** Removes dependence on dense GCPs and strong image overlap — exactly the constraints a single pass cannot satisfy — and supplies absolute scale and position when bundle adjustment is otherwise weakly conditioned.
- **Perf/accuracy:** Reported ~1-3 cm horizontal RMSE and ~2-7 cm vertical RMSE with no GCPs (DJI Phantom 4 RTK / WingtraOne / Matrice RTK class); an unmodeled systematic vertical offset of ~10-30 cm is common and is removed by one checkpoint. Needs time sync <1 ms and lever arm known to <1-2 cm.
- **Failure modes:** GNSS outage / urban canyon / canopy; unmodeled boresight and lever arm; wrong vertical datum/geoid; rolling-shutter timing error scales with UAV speed; RTK float (not fix) degrades to dm-m.
- **Alternatives:** PPK post-processing with RTKLIB against a CORS/base RINEX (no live link needed); standard GNSS+IMU direct georef (dm-m); assisted with 1 checkpoint.

### IMU preintegration for metric scale recovery  
*Maturity: production*

- **What:** Forster et al. on-manifold IMU preintegration summarizes accelerometer/gyro between keyframes into a single relative-motion factor that observes gravity direction and metric scale, resolving monocular scale ambiguity and providing high-rate motion between images.
- **Why it fits single-pass:** A single monocular pass is scale-ambiguous and lacks the wide stereo baselines that would otherwise fix scale; the IMU is the practical metric anchor whenever RTK is unavailable, and it bridges GNSS dropouts.
- **Perf/accuracy:** Scale error typically <1-3% after a proper initialization requiring ~10-30 s of accelerated motion; standard in VINS-Fusion, OpenVINS, ORB-SLAM3, GTSAM.
- **Failure modes:** Constant-velocity / low-excitation flight leaves scale weakly observable; accelerometer/gyro bias drift; poor initialization; cheap MEMS IMU noise. Requires IMU-camera extrinsic + time-offset calibration (Kalibr).
- **Alternatives:** GNSS-baseline scale between frames; stereo/RGB-D camera; a physical scale bar or known object dimension in scene.

### GVINS - tightly-coupled raw-GNSS + visual-inertial fusion  
*Maturity: research*

- **What:** Nonlinear optimization that fuses raw GNSS pseudorange and Doppler (multi-constellation GPS/GLONASS/Galileo/BeiDou) with VIO, estimating global 6-DoF in ECEF with online local-ENU alignment (HKUST, 2021).
- **Why it fits single-pass:** Border/strategic single-pass flights often traverse GNSS-degraded terrain; raw-measurement fusion still contributes global constraints with fewer than 4 satellites, where an RTK fix is impossible, giving robust drift-free global pose.
- **Perf/accuracy:** Meter-level global positioning without RTK, smooth and drift-free, bridges GNSS gaps and recovers global pose after outages; VIO front-end adapted from VINS-Mono.
- **Failure modes:** Requires access to raw GNSS observables (RINEX/UBX) many drones do not expose; ROS1 / older Ceres/Eigen dependencies; GPL-3; not RTK-level absolute accuracy on its own.
- **Alternatives:** GICI-LIB (2024, newer GNSS-VI-LiDAR); VINS-Fusion loose GPS; custom GTSAM with pseudorange factors.

### OpenVINS (MSCKF) onboard real-time VIO  
*Maturity: near-production*

- **What:** Filter-based sliding-window visual-inertial estimator (Multi-State Constraint Kalman Filter) from UDel RPNG, with online camera/IMU intrinsic-extrinsic and time-offset calibration, mono/stereo, and zero-velocity updates.
- **Why it fits single-pass:** Lightweight enough for onboard Jetson-class compute to produce live keyframe poses during the single flight, enabling near-real-time incremental reconstruction and low-latency situational awareness.
- **Perf/accuracy:** >30-100 Hz tracking on embedded hardware; EuRoC absolute trajectory error typically ~0.1-0.6% of trajectory length; winner of the IROS 2019 FPV Drone Racing VIO competition.
- **Failure modes:** Filter inconsistency under aggressive motion; no absolute geo-anchor unless a GNSS update is added; loop closure is only loosely coupled (ov_secondary) and does not feed back; GPL-3.
- **Alternatives:** VINS-Fusion, ORB-SLAM3, ROVIO, OKVIS2, Fast-LIO2 (if LiDAR added).

### VINS-Fusion with loose GNSS global fusion  
*Maturity: near-production*

- **What:** Optimization-based mono/stereo(+IMU) VIO with a separate global_fusion pose-graph module that fuses GPS in the world frame via GeographicLib to align the local VIO trajectory to an absolute CRS.
- **Why it fits single-pass:** A mature, well-tested baseline that is fast to stand up for a hackathon and cleanly demonstrates aligning drift-free local VIO to GNSS-absolute coordinates for single-pass flights.
- **Perf/accuracy:** Real-time on embedded platforms; global fusion aligns output to GNSS accuracy while retaining sub-percent relative VIO; scale from IMU.
- **Failure modes:** GPS is loosely coupled (not inside the VIO cost), so poor GPS is not optimally down-weighted; needs reasonable GNSS; scale wholly dependent on IMU init.
- **Alternatives:** OpenVINS+GNSS update; tight GTSAM graph; maplab 2.0; SVO Pro.

### COLMAP 4.2 / GLOMAP SfM+MVS with pose priors and geo-registration  
*Maturity: production*

- **What:** Incremental (COLMAP) or global (GLOMAP, 2024) structure-from-motion to refine poses and self-calibrate intrinsics, followed by PatchMatch MVS dense reconstruction; GNSS/pose priors constrain BA and a model aligner geo-registers output into an absolute CRS. PyCOLMAP for scripting.
- **Why it fits single-pass:** Offline refinement tightens the fused poses and recovers intrinsics; feeding GNSS pose priors keeps the otherwise ill-conditioned single-pass bundle adjustment stable and produces a metric, georeferenced point cloud/mesh for measurement.
- **Perf/accuracy:** Sub-pixel reprojection error (~0.5-1 px) typical; GLOMAP performs global SfM roughly an order of magnitude faster than incremental on large scenes; MVS yields dense clouds/meshes. COLMAP 4.2 current (2026).
- **Failure modes:** Weak baseline causes drift/degeneracy without priors; default pipeline does not model rolling shutter; struggles on textureless or repetitive facades; MVS is GPU/time heavy (offline).
- **Alternatives:** OpenMVG+OpenMVS; OpenDroneMap (full georeferenced pipeline); Agisoft Metashape / Pix4D / RealityCapture (commercial); MASt3R-SfM.

### Geoid/datum handling and CRS transforms (PROJ 9 / GDAL 3 / GeographicLib)  
*Maturity: production*

- **What:** Transform among WGS84 geographic (EPSG:4326), ECEF (EPSG:4978), UTM zones, and local ENU; convert GNSS ellipsoidal height h to orthometric H via a geoid model (H = h - N); attach EPSG codes and WKT2 CRS strings to every product.
- **Why it fits single-pass:** GNSS returns ellipsoidal height while all deliverables and measurements need orthometric elevation; mishandling silently injects tens-of-meters vertical error (India N reaches about -100 m in the south, +/- across the country), dwarfing cm-level survey targets.
- **Perf/accuracy:** Numeric transform accuracy is sub-mm; geoid-model accuracy is the limiter — EGM2008 gives few-cm to dm regionally, a national geoid grid (e.g., Indian geoid) is better. UTM keeps scale distortion <1:1000 within a 6-degree zone.
- **Failure modes:** Wrong datum/realization or epoch (ITRF vs WGS84 vs local, plate motion for cm work); missing geoid grid; UTM zone-edge crossings; assuming EPSG:4326 axis order.
- **Alternatives:** National geoid grids; Cesium World Terrain; verticaldatum/PROJ pipelines; GeographicLib GeoidEval.

### ASPRS Edition 2 (2023) checkpoint accuracy assessment  
*Maturity: production*

- **What:** Compute horizontal radial RMSEr and vertical RMSEz against independent, higher-accuracy checkpoints and report accuracy at the 95% confidence level (horizontal = RMSEr x 1.7308, vertical = RMSEz x 1.96). Edition 2 dropped fixed map-scale/pixel classes in favor of user-specified RMSE thresholds.
- **Why it fits single-pass:** To credibly claim GCP-free metric accuracy for measurement and analysis, the system must quantify and certify correctness with a recognized standard rather than assert it.
- **Perf/accuracy:** Recommends >=20-30 well-distributed checkpoints; checkpoints should be ~3x more accurate than the product; report H and V separately with CRS, geoid, and epoch documented.
- **Failure modes:** Too few or poorly distributed checkpoints; checkpoints not independent of the solution; ignoring systematic bias (RMSE assumes it is removed); mixing datums.
- **Alternatives:** Scale-bar / known-distance checks; cloud-to-cloud M3C2 vs reference LiDAR/TLS; reprojection-error and reconstruction-uncertainty proxies when no checkpoints exist.

### Per-region uncertainty propagation and confidence maps  
*Maturity: near-production*

- **What:** Derive per-point / per-cell uncertainty from bundle-adjustment covariance, ray convergence (intersection) angle, number of observing rays, and local GSD, then publish a confidence raster or per-vertex attribute alongside the model.
- **Why it fits single-pass:** Single-pass accuracy is intrinsically non-uniform (dense, well-converged nadir ground vs. sparse, grazing-angle facades and occluded backsides); measurement users must be told which parts are trustworthy so the system degrades gracefully instead of presenting uniform false confidence.
- **Perf/accuracy:** Flag regions with convergence angle <5-10 degrees or <3 observing rays as low-confidence; propagate covariance into downstream measurement tools; typical GSD at 100 m AGL is ~2-3 cm/px.
- **Failure modes:** Covariance underestimates error in the presence of unmodeled systematic bias; full covariance is expensive (needs approximation/marginalization); confidence must be calibrated against checkpoints.
- **Alternatives:** Monte-Carlo bundle adjustment; ensemble reconstruction; learned per-pixel depth uncertainty (VGGT/DUSt3R confidence heads).

### Rolling-shutter and motion modeling in pose/BA  
*Maturity: near-production*

- **What:** Model per-row exposure timing (rolling shutter) and camera motion during exposure, gate keyframes by image sharpness, deblur where feasible, and use robust cost functions to tolerate H.264/H.265 compression artifacts.
- **Why it fits single-pass:** The input is compressed video from a moving UAV, not crisp survey stills; unmodeled rolling shutter and motion blur bias sub-pixel feature locations and therefore metric accuracy — the single biggest reason video accuracy trails still-photo photogrammetry.
- **Perf/accuracy:** Unmodeled rolling shutter can bias geometry by many pixels / decimeters at typical UAV speeds; explicit modeling recovers most of it. Sharpness gating and I-frame preference cut compression noise.
- **Failure modes:** Extra per-frame unknowns further weaken conditioning; needs sensor readout time; severe blur is unrecoverable and those frames must be dropped; variable GOP complicates PTS.
- **Alternatives:** Use a global-shutter camera (best fix); higher frame rate / lower speed; learned deblurring (NAFNet, Restormer); extract only keyframes at low motion.

### Feed-forward 3D priors: VGGT (2025) / MASt3R (2024) for weak-baseline regions  
*Maturity: research*

- **What:** Transformer models that predict camera poses, depth, and (metric, for MASt3R) point maps from one-to-many views in seconds, usable to initialize BA or to fill regions where classical triangulation is degenerate; COLMAP and gsplat export supported.
- **Why it fits single-pass:** Strong learned monocular/few-view priors partially compensate for the limited angles and single-viewpoint occlusions that break classical multi-view stereo on a single pass, and give near-instant initialization for near-real-time operation.
- **Perf/accuracy:** VGGT (CVPR 2025 Best Paper) reconstructs a scene in <1 s feed-forward and matches or beats dedicated monocular-depth methods; MASt3R adds metric point maps and scalable global alignment; MUSt3R extends to many views without global alignment.
- **Failure modes:** Absolute metric scale not guaranteed (must be tied to GNSS/IMU); geometry hallucination on out-of-distribution aerial scenes; high VRAM; VGGT's commercial checkpoint EXCLUDES military use and the base model is non-commercial - a licensing blocker for NTRO/defense deployment.
- **Alternatives:** MASt3R-SfM; DUSt3R; monocular metric depth (Metric3D v2, UniDepth, Depth Anything v2) fused into BA.

## Recommended stack

| Component | Choice | Rationale |
|---|---|---|
| High-accuracy pose/fusion (offline/near-real-time) | GTSAM 4.2 factor graph (IMU preintegration + RTK/GNSS + visual + baro factors, iSAM2) | Best metric accuracy and native uncertainty for GCP-free georeferencing; production-grade BSD library with RTK double-difference and IMU factors already implemented. |
| Onboard real-time VIO | OpenVINS (MSCKF) or VINS-Fusion, with a GNSS update | Run in real time on Jetson-class compute to produce live keyframe poses; OpenVINS is proven on agile UAVs (IROS 2019 FPV winner), VINS-Fusion has a ready GPS global-fusion module. |
| GNSS-degraded / raw-GNSS fusion | GVINS or GICI-LIB | Tightly fuse raw pseudorange/Doppler for robust global pose in border/canopy areas where an RTK fix is impossible; essential anti-jamming/anti-outage fallback for strategic use. |
| SfM refinement + dense MVS | COLMAP 4.2 + GLOMAP, then OpenMVS | Refine poses, self-calibrate intrinsics with GNSS pose priors, and densify into a metric mesh; GLOMAP gives ~10x faster global SfM for large single-pass scenes. |
| Turnkey georeferenced mapping baseline | OpenDroneMap (ODM/NodeODM) | Open-source, produces georeferenced orthophoto GeoTIFF, DSM, LAZ point cloud, and textured OBJ, and reads GPS from DJI SRT - a fast working end-to-end fallback (note AGPL-3.0). |
| Fast init / weak-baseline & occlusion priors | MASt3R / MUSt3R (prefer over VGGT for deployment) | Feed-forward metric point maps in seconds to seed BA and fill single-viewpoint gaps; MASt3R avoids VGGT's military-use license exclusion that blocks NTRO deployment. |
| PPK GNSS post-processing | RTKLIB (demo5 fork) | Open-source carrier-phase post-processing of drone raw GNSS against a base/CORS RINEX to reach cm positions when no live RTK link is available. |
| Geodesy / CRS / geoid | PROJ 9 + GDAL 3 + GeographicLib, geoid = EGM2008 or Indian national geoid grid | Correct WGS84/ECEF/UTM/ENU transforms and ellipsoidal-to-orthometric height conversion; prevents the tens-of-meters vertical datum errors that are catastrophic in India. |
| Point-cloud processing & delivery | PDAL + Entwine, output LAS 1.4 / LAZ / COPC | Reproject, classify, tile, and cloud-optimize point clouds with embedded CRS for streaming and GIS ingestion. |
| Dynamic-object masking | Mask2Former / YOLO / Segment Anything (SAM) | Segment and remove vehicles, people, and animals before pose estimation and reconstruction so moving objects do not corrupt static geometry or introduce ghost points. |
| Onboard compute + live mapping | NVIDIA Jetson Orin (AGX/NX) + nvblox (Isaac ROS) | GPU-accelerated real-time TSDF/ESDF reconstruction for near-real-time situational output during flight, with offline high-fidelity densification later. |
| Raster deliverables | Cloud-Optimized GeoTIFF (COG) via GDAL for DSM/DTM/orthomosaic | Standard georeferenced raster with embedded CRS, streamable and directly consumable by QGIS/ArcGIS and web clients. |
| 3D model / web / digital-twin delivery | glTF/GLB + OGC 3D Tiles 1.1 (CesiumJS / Cesium ion); CityJSON 2.0 / CityGML 3.0 for buildings | 3D Tiles 1.1 streams city-scale meshes and point clouds with glTF content and metadata; CityJSON 2.0 (OGC standard) encodes LOD building models for digital twins and planning. |
| Telemetry & video ingest | pymavlink (PX4/ArduPilot logs), DJI SRT parser, ffmpeg (frames + PTS) | Extract GPS/IMU/attitude and per-frame timestamps and synchronize them to video frames - the glue that enables direct georeferencing from real drone hardware. |
| GIS / QA / visualization | QGIS + CloudCompare (M3C2) + MeshLab/Blender | Open-source measurement, checkpoint RMSE computation, cloud-to-cloud validation against reference LiDAR, and mesh inspection. |

## Real-world integration

INPUT INGEST: Video arrives as H.264/H.265 in MP4/MOV; extract frames and their presentation timestamps (PTS) with ffmpeg and prefer I-frames for feature work. DJI aircraft embed per-frame GPS (lat/lon/rel+abs altitude), timestamp, gimbal and sometimes ISO/shutter in SRT subtitle sidecar files - parse these directly. PX4/ArduPilot platforms log MAVLink (.ulog/.tlog/.bin): read GLOBAL_POSITION_INT, GPS_RAW_INT/GPS2_RAW, HIGHRES_IMU, ATTITUDE, and camera-trigger messages via pymavlink. TIME SYNC is the make-or-break step: align video PTS to telemetry clock via camera-trigger events, hardware PPS, or software cross-correlation of angular rates; VINS-Fusion and OpenVINS estimate the residual camera-IMU time offset online. GEOMETRIC CALIBRATION: obtain camera intrinsics from EXIF/XMP (focal length, sensor size) or calibrate/self-calibrate (Kalibr for camera-IMU extrinsics and time offset, COLMAP for intrinsics); measure the GNSS-antenna-to-camera lever arm and gimbal boresight to cm; record rolling-shutter readout time. GNSS CORRECTIONS: for live RTK, stream RTCM3 via an NTRIP client (mountpoint from a national CORS/VRS network) to the drone/base; for PPK, log raw GNSS (u-blox UBX or RINEX) onboard plus a base-station/CORS RINEX and post-process in RTKLIB to cm. OUTPUT CRS: project into the correct UTM zone (India spans EPSG:32642-32646, WGS84 UTM 42N-46N; use ITRF-realized frames for cm work) and convert ellipsoidal to orthometric height with EGM2008 or a national geoid via GeographicLib/PROJ; embed the full WKT2 CRS + EPSG code and vertical datum in every GeoTIFF (via GDAL), LAS 1.4 (georeference VLR), and 3D Tiles/CityJSON boundingVolume/metadata. DELIVERY & PLATFORMS: point clouds as LAZ/COPC (PDAL, Entwine), rasters as COG DSM/DTM/orthomosaic, meshes as OBJ+MTL and glTF/GLB, city-scale as OGC 3D Tiles 1.1 for CesiumJS/Cesium ion and building models as CityJSON 2.0/CityGML 3.0; all open in QGIS, and 3D Tiles feed digital-twin stacks (Bentley iTwin, NVIDIA Omniverse). EDGE: run VIO + nvblox on Jetson Orin for near-real-time onboard pose and coarse mesh, offloading dense MVS/mesh/texture to a ground GPU station over the datalink or after landing.

## Reliability & failure handling

TIERED GEOREFERENCING (never hard-fail): the pipeline always emits a georeferenced, labeled result at the best tier the sensors allow - Tier 1 RTK/PPK (few-cm absolute) -> Tier 2 standard GNSS + IMU + baro (dm-to-m absolute, metric scale from IMU) -> Tier 3 VIO-only in a local ENU frame (metric scale, no absolute geo) -> Tier 4 up-to-scale reconstruction (image-only). Each product carries an explicit accuracy label, CRS, geoid/epoch, and processing tier so downstream measurement tools show trust rather than silently presenting wrong numbers. GNSS DROPOUT/JAMMING: IMU preintegration and VIO bridge the gap in the factor graph; on GNSS return the trajectory re-anchors via new global factors; raw-GNSS fusion (GVINS-style) keeps contributing with <4 satellites - critical for strategic/border areas where spoofing and jamming are expected, so the system must never fully depend on GNSS. MOTION BLUR / COMPRESSION: sharpness-based keyframe gating, I-frame preference, optional deblurring, and robust M-estimators (Huber/Cauchy) in BA reject artifact-driven outliers; unusable frames are dropped, not forced. DYNAMIC OBJECTS: semantic masks (vehicles/people/animals) are removed before pose estimation and densification so moving objects never enter static geometry; residual movers appear only as low-confidence and are filtered by consensus across views. CONVERGENCE FAILURE: a region whose BA fails to converge or has grazing convergence angles is marked low-confidence and excluded from measurement rather than aborting the whole job; dense reconstruction gracefully degrades to sparse where MVS fails. DATUM SAFETY: validate CRS/geoid at startup, refuse to silently mix datums, warn and fall back to clearly-labeled ellipsoidal height if a geoid grid is missing. PROVENANCE: every tile/point stores its sensor configuration, tier, and uncertainty so operators and analysts can audit and safely act on the model.

## Single-pass specifics

What must change versus traditional multi-pass photogrammetry: (1) DO NOT rely on photogrammetric self-calibration for scale or georeferencing - a single flight line gives narrow convergence angles and near-degenerate bundle adjustment, so metric scale must be injected from IMU preintegration and absolute position from GNSS/RTK (direct georeferencing), with camera intrinsics supplied as a strong prior or fixed rather than freely solved. (2) FEED GNSS pose priors as constraints inside BA (soft or hard) to keep the weakly-conditioned single-pass network stable; classical free-network BA that works with 70-80% cross-strip overlap will drift or collapse here. (3) EXPECT AND MODEL heterogeneous accuracy: nadir ground is well triangulated but building facades and any surface seen from only one grazing viewpoint have large depth uncertainty - so per-region confidence (convergence angle, ray count, covariance, GSD) is mandatory output, not optional. (4) TREAT INPUT AS VIDEO, NOT PHOTOS: model rolling shutter, deblur/keyframe-select against motion blur, and handle H.264/H.265 compression noise, all of which cap sub-pixel accuracy and make cm-level claims harder than with still-photo surveys. (5) DO NOT COUNT ON LOOP CLOSURES: a single pass rarely revisits, so drift cannot be corrected by loops and must instead be bounded by continuous global GNSS constraints. (6) ACCEPT OCCLUSION AS UNDERDETERMINED: backsides and occluded surfaces seen from one viewpoint cannot be triangulated - fill with learned priors (MASt3R/monocular depth) or symmetry, but flag those regions as inferred, never as measured. (7) GEOREFERENCE INCREMENTALLY/ONLINE: direct georeferencing does not need the full BA network to finish, enabling near-real-time output that multi-pass workflows defer to a batch solve. (8) VERTICAL is the weak axis: with no GCPs a systematic vertical bias appears - budget for at least one checkpoint or an accurate geoid + calibrated boresight to remove it.

## Open risks

- Vertical accuracy without any GCP: a systematic vertical bias of ~10-30 cm from residual boresight/lever-arm and camera-calibration error is the single biggest metric risk; one ground checkpoint or a precise geoid+calibration is effectively required to hit sub-decimeter vertical.
- Single-pass facades and occluded/backside surfaces are geometrically underdetermined - expect holes and low fidelity there; learned priors reduce but cannot guarantee metric correctness, so these regions must be labeled inferred.
- Rolling shutter plus H.264/H.265 compression cap sub-pixel feature accuracy, so cm-level accuracy claims transferred from still-photo studies are optimistic for video and must be re-validated on the actual codec/sensor.
- Geoid / vertical-datum accuracy in Indian border and strategic regions may be limited; EGM2008 vs a local geoid can differ by decimeters, and using WGS84 vs an ITRF realization/epoch matters at the cm level.
- True real-time full dense metric reconstruction on edge hardware (Jetson) is not yet feasible; the realistic split is near-real-time onboard sparse/coarse (VIO + nvblox) with offline dense MVS/mesh/texture - manage expectations accordingly.
- Licensing/export constraints for defense: VGGT's commercial checkpoint explicitly excludes military use and its base model is non-commercial; VINS-Fusion/OpenVINS/GVINS/COLMAP are GPL and ODM is AGPL - copyleft and use-restrictions must be vetted for NTRO deployment (favor BSD GTSAM, permissive components, or obtain licenses).
- GNSS spoofing/jamming in contested areas can silently corrupt georeferencing; robust/switchable GNSS factors and a VIO-only fallback are mandatory, and RTK float (not fix) must be detected and down-weighted.
- CRS/geoid/UTM-zone misconfiguration is a common silent source of tens-of-meters error; needs automated validation, zone-edge handling, and provenance rather than trust in defaults.

## Differentiation notes

Off-the-shelf photogrammetry (Pix4D, DJI Terra, Agisoft Metashape, RealityCapture, OpenDroneMap) is built for MULTI-PASS still-image capture: it assumes 70-80% front/side overlap and multiple crossing flight lines, treats frames as independent photos with no IMU or tightly-coupled sensor fusion, typically still needs GCPs to certify cm accuracy, and reports a single global accuracy number rather than per-region confidence. Given single-pass video with weak geometry, these tools either fail to converge, silently return wrong scale, or produce a plausible-looking but unvalidated model - and none integrate the raw drone telemetry (SRT/MAVLink/NTRIP/PPK) or degrade gracefully. This solution differs on six concrete axes: (1) single-pass, VIDEO-native front end - rolling-shutter and motion-blur handling, keyframe selection, and tightly-coupled Visual-Inertial-GNSS fusion (GTSAM factor graph) rather than photo-only SfM; (2) GCP-FREE metric accuracy via RTK/PPK direct georeferencing + IMU scale + self-calibration, with rigorous uncertainty instead of assumed accuracy; (3) ASPRS-Edition-2-compliant, SELF-REPORTED PER-REGION CONFIDENCE (convergence angle, ray count, BA covariance, GSD) so each measurement carries a trust level - something no consumer tool exposes; (4) TIERED GRACEFUL DEGRADATION (RTK->PPK->GNSS+IMU->VIO-only) that always emits a labeled, georeferenced result and never hard-fails, with explicit GNSS-jamming/dropout fallback for strategic use; (5) rigorous DATUM/GEOID correctness (ellipsoidal->orthometric, EPSG/UTM, WKT2 embedded) that prevents the tens-of-meters vertical errors endemic to naive pipelines, tuned for the Indian geoid; (6) OPEN, INTEROPERABLE georeferenced outputs (LAS/LAZ/COPC, COG GeoTIFF, glTF/3D Tiles 1.1, CityJSON 2.0) that plug straight into QGIS, CesiumJS, and digital-twin platforms, plus permissive/vettable licensing suitable for defense deployment.

## Citations / references

- GVINS: Tightly Coupled GNSS-Visual-Inertial Fusion (Cao, Lu, Shen, HKUST, 2021; arXiv:2103.07899)
- VINS-Fusion / VINS-Mono (Qin, Li, Shen, HKUST, 2018-2019)
- OpenVINS (Geneva, Eckenhoff, Lee, Yang, Huang, UDel RPNG, ICRA 2020; MSCKF, IROS 2019 FPV winner)
- GTSAM 4.2 / iSAM2 incremental smoothing (Dellaert, Kaess, Georgia Tech; iSAM2 IJRR 2012); RTK double-difference and IMU factors
- On-Manifold IMU Preintegration (Forster, Carlone, Dellaert, Scaramuzza, T-RO 2017)
- ORB-SLAM3 visual-inertial (Campos et al., T-RO 2021)
- COLMAP 4.2 SfM+MVS (Schoenberger & Frahm, CVPR 2016; PatchMatch MVS ECCV 2016)
- GLOMAP global SfM (Pan, Barath, Pollefeys, Schoenberger, ECCV 2024)
- DUSt3R: Geometric 3D Vision Made Easy (Wang et al., CVPR 2024)
- MASt3R: Grounding Image Matching in 3D (Leroy, Cabon, Revaud, 2024) and MUSt3R (2025)
- VGGT: Visual Geometry Grounded Transformer (Wang et al., CVPR 2025 Best Paper; commercial checkpoint excludes military use)
- 3D Gaussian Splatting (Kerbl, Kopanas, Leimkuehler, Drettakis, SIGGRAPH 2023) and SuGaR mesh extraction (Guedon & Lepetit, CVPR 2024)
- ASPRS Positional Accuracy Standards for Digital Geospatial Data, Edition 2 (2023): horizontal 95% = RMSEr x 1.7308, vertical 95% = RMSEz x 1.96
- EGM2008 / EGM96 Earth Gravitational Models; geoid undulation ranges ~ +85 m to -106 m (southern India)
- OGC 3D Tiles 1.1 (Cesium/OGC Community Standard; glTF tile content) and CesiumJS
- CityJSON 2.0 (OGC standard) / CityGML 3.0 data model
- LAS 1.4 R15 (ASPRS), LAZ (LASzip), COPC (Cloud Optimized Point Cloud, Hobu)
- PROJ 9 / GDAL 3 / GeographicLib for CRS transforms and geoid evaluation; PDAL & Entwine for point clouds
- RTKLIB (demo5) for PPK/RTK GNSS post-processing
- OpenDroneMap / NodeODM (AGPL-3.0) georeferenced ortho/DSM/point-cloud/mesh pipeline
- NVIDIA nvblox / Isaac ROS real-time GPU TSDF/ESDF on Jetson Orin
- Studies on RTK UAV direct georeferencing without GCPs (e.g., Forlani et al. 2018; Taddia et al. 2019; Stroner et al. 2020): ~1-3 cm horizontal, ~2-7 cm vertical, vertical bias removed by a single checkpoint
