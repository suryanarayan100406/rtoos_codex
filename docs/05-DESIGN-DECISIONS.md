# DRISHTI — Design Decisions

**Purpose:** The *why* behind DRISHTI. Every consequential architectural choice recorded as an Architecture Decision Record (ADR): the context that forced it, the options considered, what we chose, the single-pass rationale, the trade-offs we accepted, and the condition that would make us revisit. This is the document that shows our reasoning is deliberate, not accidental.

**Audience:** National Technical Research Organisation (NTRO) technical evaluators and the DRISHTI build team. Assumes familiarity with [How It Works](02-HOW-IT-WORKS.md) and the [Technology Stack](03-TECHNOLOGY-STACK.md); the pipeline is defined once in [`_internal/CANONICAL-ARCHITECTURE-SPEC.md`](_internal/CANONICAL-ARCHITECTURE-SPEC.md).

**TL;DR**

- Seventeen decisions define DRISHTI. The through-line: **the single-pass, no-second-look constraint changes the right answer** at nearly every step versus a conventional multi-pass photogrammetry pipeline.
- Every decision is scored against the **official weighted rubric** (PS-17 / SIH26158) — reconstruction accuracy **30%**, model completeness **20%**, processing speed **20%**, innovation **15%**, scalability **10%**, user interface **5%** — and against the hard targets: **≤ 1 m** spatial accuracy, **< 15 min for a 10-minute video**, full visible-scene coverage, **OBJ · PLY · LAS · GeoTIFF · .glb/.gltf · .fbx**, web or desktop viewer.
- The input bet (ADR-16): **design for the mandatory inputs alone** — video + GPS + flight metadata, intrinsics self-calibrated — and treat IMU, barometer, RTK/PPK and operator-supplied intrinsics as *optional upgrades*, because that is exactly what the official contract makes them.
- The biggest bet (ADR-01): **prior-assisted feed-forward geometry** over classical Structure-from-Motion (SfM) as the primary path — because short single-pass baselines starve triangulation — with classical SfM kept as the verifiable fallback, not discarded.
- The metric bet (ADR-03/04): **sensor-injected scale from a factor graph whose required factors are visual + GNSS** (IMU, baro and RTK/PPK folded in when those optional sensors exist), not learned-only scale and not Ground Control Points (GCPs) — the only way to be metric *and* GCP-free in one pass.
- The reliability bet (ADR-09): **confidence is a first-class output at every stage**, and every stage has a fallback — so the system degrades gracefully instead of hard-failing.
- The honesty bet, present in every ADR's trade-off section: we name what we *lose* (absolute accuracy on planned GCP missions, facade completeness, mesh polish) versus offline incumbents, and why the single-pass niche is still worth owning.
- Trade-offs and external numbers follow the project honesty policy: performance figures attributed to a method are **as reported by its authors**; our targets are labelled *(design target — to be measured)*.

---

## 1. How decisions are recorded

Each decision is an **ADR** with a fixed shape, so an evaluator can audit our reasoning uniformly:

| Field | What it captures |
|-------|------------------|
| **Context** | The forcing constraint — usually a consequence of single-pass capture. |
| **Options** | The real alternatives we weighed (not strawmen). |
| **Decision** | What we chose. |
| **Rationale (single-pass angle)** | Why this choice specifically serves the single-pass, no-second-look regime. |
| **Trade-offs & risks** | What we give up, honestly, and what could bite us. |
| **Revisit if** | The concrete signal that would reopen the decision. |

ADRs are immutable once accepted; a superseding decision gets a new number and references the old one. This mirrors how the [Technology Stack](03-TECHNOLOGY-STACK.md) treats every model as replaceable — decisions, like models, are versioned rather than silently overwritten.

---

## 2. Decision index

| ADR | Decision | Primary driver |
|-----|----------|----------------|
| 01 | Prior-assisted feed-forward geometry as primary; classical SfM/MVS as fallback | Short single-pass baselines starve triangulation |
| 02 | Feed-forward backbone with a mandatory classical fallback | Bleeding-edge models are OOD on aerial data |
| 03 | Sensor-injected metric scale, not learned-only, not GCPs | Metric *and* GCP-free in one pass |
| 04 | Tightly-coupled factor graph over loose EKF fusion | Robust scale/georef from noisy single-pass sensors |
| 05 | Two output paths (Live S0–S5 · Refine S6–S10), not one monolith | "Real-time" and "accurate" are different jobs |
| 06 | Few-shot 3D Gaussian Splatting over NeRF / classical MVS for dense fill | Sparse single-pass views, fast textured output |
| 07 | Surface-aware meshing (2DGS/SuGaR) with Poisson fallback | Measurable surface from a Gaussian field |
| 08 | Belt-and-suspenders dynamic masking | An unmasked mover is a permanent ghost |
| 09 | Confidence as a first-class, propagated output | No second look to catch errors |
| 10 | ROS 2 + Isaac ROS + DeepStream over a custom runtime | Determinism, reuse, tri-tier data plane |
| 11 | TensorRT INT8/FP16 on the edge over FP32 | UAV power/thermal budget |
| 12 | Store-and-forward + deterministic ground reprocess | Link loss must lose nothing; auditability |
| 13 | nvblox (edge) + Open3D (ground) for fusion | Right fusion engine per tier |
| 14 | Dual-track hardware: DJI COTS + PX4 open/sovereign | De-risk procurement & sovereignty |
| 15 | Explicit CRS/geoid handling (UTM + EGM2008) | Orthometric height, not ellipsoidal |
| 16 | Design to the mandatory input contract; optional sensors are upgrades | Only video + GPS + metadata are guaranteed |
| 17 | Ship the official format set + both viewers; GPL writers out-of-process | `.fbx` and the viewer are graded deliverables |

---

## 3. The decision records

### ADR-01 — Prior-assisted feed-forward geometry as the primary reconstruction path

- **Context.** A single pass produces short baselines and narrow view-angle diversity. Classical SfM/Multi-View Stereo (MVS) needs wide, overlapping, multi-angle views to triangulate well; starve it and it produces holes, drift, and broken geometry — exactly where single-pass capture is weakest.
- **Options.** (a) Classical SfM/MVS (COLMAP + dense MVS) as primary; (b) feed-forward geometry transformers (predict pose + dense 3D in one forward pass) as primary; (c) NeRF/3DGS optimized directly from images.
- **Decision.** Feed-forward geometry (option b) is the **primary** geometry engine; classical SfM (GLOMAP/COLMAP) is retained as a **verifiable fallback and accuracy oracle**, not thrown away.
- **Rationale (single-pass angle).** Feed-forward models regress geometry from limited-angle input in ~seconds, tolerating the short baselines that break triangulation. They give us a *dense first estimate everywhere*, which we then constrain with sensor priors and refine — rather than failing to converge.
- **Trade-offs & risks.** Feed-forward models are newer, less battle-tested, and largely trained off-distribution from aerial imagery (see ADR-02). Classical SfM, when it *does* have the views, is more accurate and fully interpretable — so we keep it as the oracle and the fallback, and use it to validate feed-forward output on the event dataset.
- **Revisit if.** The provided dataset turns out to have rich multi-pass overlap (then classical SfM becomes primary), or feed-forward metric accuracy on aerial data fails to reach the [accuracy budget](_internal/CANONICAL-ARCHITECTURE-SPEC.md) after fine-tuning.

### ADR-02 — Feed-forward backbone with a *mandatory* classical fallback

- **Context.** The 2024–2026 geometry SOTA (VGGT, Depth Anything 3, MapAnything, Pi3) is months old and validated mostly on object-centric, indoor, or driving data — not high-altitude aerial. A backbone that silently degrades OOD is dangerous in a no-second-look setting.
- **Options.** (a) Commit to one backbone; (b) backbone + classical fallback triggered by confidence; (c) permanent ensemble of two backbones.
- **Decision.** One primary backbone (permissive default: **Depth Anything 3 / MapAnything**, per [Technology Stack §5](03-TECHNOLOGY-STACK.md)) **plus a confidence-gated classical fallback** (GLOMAP/COLMAP). Not a permanent ensemble — that doubles cost for marginal gain most of the time.
- **Rationale (single-pass angle).** When the backbone's confidence collapses (OOD terrain, degenerate motion), we fall to classical geometry rather than emitting a confident-but-wrong model. This is reliability ladder rung **L5** (neural failure → classical path) from the spec.
- **Trade-offs & risks.** Maintaining two geometry engines is more integration work and more to calibrate. The fallback is slower; a mission that leans on it heavily loses the near-real-time preview quality. We accept that: a slow correct answer beats a fast wrong one.
- **Revisit if.** Independent aerial benchmarks show the backbone is robust enough to drop the fallback, or a single model subsumes both roles with confidence.

### ADR-03 — Sensor-injected metric scale, not learned-only scale, not GCPs

- **Context.** The deliverable is a **metric, georeferenced** model **without** placing GCPs (which would defeat rapid single-pass survey). Monocular vision is scale-ambiguous; learned metric depth is impressive but not survey-grade on its own; GCPs are accurate but operationally expensive and slow.
- **Options.** (a) GCPs (classical photogrammetry); (b) learned metric depth alone for scale; (c) fuse onboard GNSS (+ RTK/PPK and IMU where fitted) to inject metric scale and georeference, using learned depth only as a *relative* prior.
- **Decision.** Option (c): **the sensor suite sets absolute scale and georeference**; learned depth contributes relative structure and gets scale-aligned to the sensor-anchored trajectory (7-DoF Umeyama/Sim3 alignment + GNSS-prior bundle adjustment). This is the **metric spine (anchor A2)**. Because GPS is the *only* mandatory positioning input, the load-bearing case is **GNSS baselines + visual structure with self-calibrated intrinsics** (ADR-16); RTK/PPK and IMU tighten it when present.
- **Rationale (single-pass angle).** The GNSS track is the one metric anchor available in a single autonomous pass with zero ground setup — thousands of soft control points along the flight line, centimetre-class when RTK/PPK happens to be fitted. Learned depth alone would leave us with plausible-but-unverifiable scale.
- **Trade-offs & risks.** Accuracy becomes **sensor-configuration-dependent**, and it is measured against the official **≤ 1 m** bar: centimetre-class with RTK/PPK (Regime A), **target ≤ 1 m** on the GPS-only mandatory baseline (Regime B, 0.5–2 m envelope by GPS quality). Vertical accuracy is the weak axis. We state this explicitly rather than claiming a single accuracy number — and **flag any region that would exceed 1 m** instead of averaging it away — per the honesty policy and the [Evaluation Criteria](_internal/PROBLEM_STATEMENT.md) tables.
- **Revisit if.** A future learned model demonstrates survey-grade metric accuracy from vision alone (then sensors become a cross-check, not the anchor), or a mission profile permits sparse GCPs for a hybrid boost.

### ADR-04 — Tightly-coupled factor graph over loose EKF fusion

- **Context.** We must fuse visual odometry and GNSS — the mandatory pair — plus IMU, RTK corrections and barometer *when those optional inputs exist*, into one trajectory. A loosely-coupled Extended Kalman Filter (EKF) fuses *post-hoc estimates*; a tightly-coupled factor graph fuses *raw measurements* jointly. The graph shape must therefore be **variable**: which factor types are instantiated depends on the capture capability flag, not on a fixed sensor suite.
- **Options.** (a) Loosely-coupled EKF (simpler, common in flight controllers); (b) tightly-coupled factor-graph smoothing (GTSAM/iSAM2) fusing raw factors with global bundle adjustment on the ground.
- **Decision.** **Tightly-coupled factor graph** (GTSAM iSAM2), incrementally on the edge and re-optimized globally on the ground.
- **Rationale (single-pass angle).** With only one pass and no loop closures from revisiting, every measurement counts. Jointly optimizing raw GNSS/IMU/visual factors extracts maximum consistency and yields per-estimate covariance — which feeds the confidence spine (ADR-09). A loose EKF discards cross-correlations we cannot afford to lose.
- **Trade-offs & risks.** More complex, heavier to implement and tune than an EKF; real-time incremental smoothing on the edge must be carefully bounded. Mitigation: incremental iSAM2 on-edge for the Live path, full batch re-optimization on the Ground path.
- **Revisit if.** Edge compute cannot sustain incremental smoothing within budget (fall back to a tighter sliding-window filter on-edge while keeping the factor graph on the ground).

### ADR-05 — Two output paths, not one monolithic pipeline

- **Context.** "Do everything in real time" and "produce a survey-accurate textured model" are in direct tension: the full-fidelity model cannot be computed in hard real time on a drone.
- **Options.** (a) One pipeline tuned for speed (sacrifice fidelity); (b) one tuned for accuracy (sacrifice liveness); (c) two coordinated paths sharing state.
- **Decision.** **Two paths** (anchor A3): a **Live path (S0–S5)** on the edge for a near-real-time coarse preview + coverage/quality feedback in flight, and a **Refine path (S6–S10)** on the ground for the accurate, textured, georeferenced deliverables — sharing the same recorded stream and metric spine.
- **Rationale (single-pass angle).** The Live path's real job is to *guarantee the single pass was good enough* — coverage gaps, blur, exposure, GNSS dropouts surfaced *while still flyable* (even if the plan permits only opportunistic re-observation). The Refine path then spends minutes, not milliseconds, to hit accuracy. This is the honest reading of "real-time": near-real-time preview + minutes-scale refinement, never a full 4K textured mesh in hard real time on the drone.
- **Trade-offs & risks.** Two paths mean two codepaths to keep consistent; the preview is deliberately coarser than the final model, which must be communicated so no one mistakes the live view for the deliverable.
- **Revisit if.** Edge compute grows enough to move Refine-path stages forward, or a mission needs a single deferred batch product only (then the Live path becomes optional QA).

### ADR-06 — Few-shot 3D Gaussian Splatting for dense reconstruction

- **Context.** After sparse/metric geometry we need dense, textured surfaces. Single-pass gives sparse, low-overlap views — hard for classical dense MVS and slow for vanilla NeRF.
- **Options.** (a) Classical dense MVS (PatchMatch); (b) NeRF; (c) 3D Gaussian Splatting (3DGS) with few-shot regularization and feed-forward initialization.
- **Decision.** **Few-shot 3DGS** (permissive **gsplat**, InstantSplat-style init from feed-forward geometry, with depth/normal/confidence regularization), with classical dense MVS available where views suffice.
- **Rationale (single-pass angle).** 3DGS trains and renders far faster than NeRF and, initialized from feed-forward geometry plus depth priors, fills plausible texture from sparse views. Regularization fights the under-constraint that sparse single-pass coverage causes.
- **Trade-offs & risks.** Gaussian fields are not natively a measurable surface (addressed in ADR-07) and can hallucinate in unobserved regions — so splatted fill in unseen areas is flagged as **inferred, not measured** (honesty policy). NeRF may still win on certain view-dependent effects; we accept the speed/robustness trade.
- **Revisit if.** A feed-forward splat model (NoPoSplat/AnySplat class) matures enough to replace the optimization step, or dense MVS proves sufficient on the event data.

### ADR-07 — Surface-aware meshing (2DGS/SuGaR) with Poisson fallback

- **Context.** Deliverables include a **measurable textured mesh** and point cloud. A raw Gaussian field is not directly a watertight, metric surface.
- **Options.** (a) Screened Poisson on the fused point cloud only; (b) surface-aligned Gaussians (2DGS/SuGaR) to extract a view-consistent mesh; (c) marching cubes on the TSDF only.
- **Decision.** **Surface-aware extraction (2DGS/SuGaR)** as primary, **screened Poisson on the metric point cloud as the robust fallback** when the Gaussian field is under-constrained.
- **Rationale (single-pass angle).** Surface-aligned Gaussians tie the fast 3DGS representation back to a measurable surface, preserving texture fidelity; Poisson guarantees a watertight mesh even when splatting is unreliable — reliability over polish.
- **Trade-offs & risks.** Surface extraction from Gaussians is research-stage and must be reimplemented on a permissive base (see [Technology Stack §5](03-TECHNOLOGY-STACK.md)); Poisson can over-smooth and close real gaps, so filled regions are marked inferred.
- **Revisit if.** A robust, permissive Gaussian-to-mesh method matures, or evaluation shows the TSDF mesh alone meets the accuracy budget.

### ADR-08 — Belt-and-suspenders dynamic masking

- **Context.** Moving objects (vehicles, people, animals, swaying vegetation) violate the static-scene assumption. In multi-pass capture a mover seen once is outvoted; in **single-pass, an unmasked mover becomes a permanent ghost or smear** with no second look to correct it.
- **Options.** (a) Semantic segmentation only; (b) geometric/motion consistency only; (c) both, fused (segment likely-dynamic classes *and* detect motion via flow-vs-epipolar residual, tracked across frames).
- **Decision.** **Both cues fused** — SAM 2 + detector + ByteTrack for appearance/semantics, RAFT flow vs epipolar geometry for motion — because either alone has failure modes.
- **Rationale (single-pass angle).** High recall matters more than precision here: a missed mover is unrecoverable, while over-masking merely costs some coverage that the confidence map will flag. Two independent cues raise recall.
- **Trade-offs & risks.** Over-masking removes real static structure and adds compute on the edge; a static-but-movable object (parked car) is ambiguous. Mitigation: track persistence and confidence weighting rather than hard binary removal.
- **Revisit if.** A single model reliably handles both semantic and motion segmentation within the edge budget.

### ADR-09 — Confidence as a first-class, propagated output

- **Context.** The single-pass, no-second-look constraint means we cannot re-fly to catch an error; the system must instead *tell us where it is unsure*.
- **Options.** (a) Post-hoc quality heuristics on the final model; (b) confidence emitted and propagated by every stage end-to-end.
- **Decision.** **Every stage emits a confidence/uncertainty signal that propagates** into the final per-point/per-region completeness-and-uncertainty report (anchor A4, the reliability spine). No learned output enters the pipeline without a confidence wrapper (multi-view agreement, geometric residual, or model uncertainty).
- **Rationale (single-pass angle).** A georeferenced model is only actionable if the user knows which parts are trustworthy. Confidence also drives the graceful-degradation ladder (which fallback fires, when) and lets us mark inferred/occlusion-completed surfaces honestly.
- **Trade-offs & risks.** Extra computation and engineering to produce, calibrate, and carry confidence everywhere; calibration itself must be validated so the numbers mean something. We accept this as core to the value proposition, not optional.
- **Revisit if.** Never fully — this is foundational. Calibration *methods* will evolve.

### ADR-10 — ROS 2 + Isaac ROS + DeepStream over a custom runtime

- **Context.** We need a deterministic data plane spanning edge and ground, with GPU-accelerated perception and video handling.
- **Options.** (a) Bespoke async framework; (b) ROS 2 + NVIDIA Isaac ROS + DeepStream/GStreamer.
- **Decision.** **ROS 2 (Humble/Jazzy) + Isaac ROS + DeepStream.**
- **Rationale (single-pass angle).** ROS 2's Quality-of-Service (QoS) pub/sub gives deterministic, replayable dataflow (essential for the deterministic reprocess in ADR-12); Isaac ROS provides GPU-accelerated nodes (nvblox, image pipeline) already tuned for Jetson; DeepStream gives zero-copy NVDEC→inference. Reuse over reinvention, matching the adopt-don't-invent philosophy.
- **Trade-offs & risks.** ROS 2 has a learning curve and runtime overhead; some Isaac components carry NVIDIA licensing (see stack). We accept the overhead for determinism and ecosystem.
- **Revisit if.** Real-time constraints demand a leaner runtime on the edge for specific nodes (we can drop to raw CUDA nodes selectively).

### ADR-11 — TensorRT INT8/FP16 on the edge over FP32

- **Context.** The edge (Jetson Orin) has a hard power/thermal/latency budget on the UAV; full-precision models won't fit the Live-path latency target.
- **Options.** (a) FP32/ONNX portable; (b) TensorRT with INT8/FP16 quantization and calibration.
- **Decision.** **TensorRT INT8/FP16** for edge inference; keep FP32 on the ground where accuracy is paramount.
- **Rationale (single-pass angle).** Quantization is what makes the near-real-time Live preview physically possible within the UAV budget; the ground path re-runs at full precision so the *deliverable* accuracy is never quantization-limited.
- **Trade-offs & risks.** Quantization can cost accuracy; INT8 needs a representative calibration set (aerial), and accuracy-vs-latency must be measured on the event hardware. Mitigation: the authoritative product comes from the FP32 ground path, so edge quantization only affects the preview.
- **Revisit if.** Measured INT8 accuracy loss is unacceptable even for preview (fall to FP16), or edge hardware gains headroom.

### ADR-12 — Store-and-forward + deterministic ground reprocess

- **Context.** The radio link can drop; a single-pass mission cannot be repeated cheaply, so **no captured data may be lost to a link outage**, and accuracy claims must be auditable.
- **Options.** (a) Rely on the live stream only; (b) record near-raw H.265 + telemetry to onboard NVMe and forward opportunistically, then deterministically reprocess on the ground.
- **Decision.** **Store-and-forward** to onboard storage with **deterministic ground reprocess** (monotonic frame IDs, pinned versions, fixed seeds) from the recorded near-raw stream + telemetry.
- **Rationale (single-pass angle).** The recording is the ground truth of the one pass; the Live path is best-effort over the radio, but the authoritative model is rebuilt from the complete on-board record. Determinism makes the result reproducible and the accuracy report checkable. Also underpins sovereignty/air-gap (no cloud dependency).
- **Trade-offs & risks.** Onboard storage, I/O bandwidth, and post-flight transfer time; the authoritative product is deferred to after landing. Accepted: correctness and auditability outrank immediacy for the deliverable.
- **Revisit if.** Links become reliably high-bandwidth enough to stream near-raw in real time (unlikely in contested environments).

### ADR-13 — nvblox on the edge, Open3D on the ground for fusion

- **Context.** Both tiers fuse depth into a volume, but under very different compute budgets and goals (live coarse map vs scalable accurate fusion).
- **Options.** (a) One fusion engine everywhere; (b) tier-appropriate engines: nvblox on edge, Open3D on ground.
- **Decision.** **nvblox (edge)** for GPU-accelerated live TSDF within the Jetson budget; **Open3D (ground)** for scalable, accurate TSDF/point fusion.
- **Rationale (single-pass angle).** The edge needs a fast, dynamic-masked coarse map *now* for the Live preview; the ground needs scale and accuracy. Matching the engine to the tier beats compromising on one. Both are permissively licensed (Apache/MIT).
- **Trade-offs & risks.** Two fusion codepaths; representations must reconcile between tiers. Mitigation: shared metric frame and voxel conventions from the spec's coordinate-frames section.
- **Revisit if.** One engine covers both budgets, or a superior permissive fusion library appears.

### ADR-14 — Dual-track hardware: DJI COTS + PX4 open/sovereign

- **Context.** We need a platform for development and demos now, and a sovereign, customizable platform for deployment — with different procurement, licensing, and data-control profiles.
- **Options.** (a) DJI-only (fast, integrated RTK, but closed and foreign); (b) open PX4/ArduPilot-only (sovereign but slower to stand up); (c) both tracks.
- **Decision.** **Both** — DJI Matrice 350 RTK + Zenmuse as the fast COTS track for development/benchmarking; a **PX4/ArduPilot + Jetson + open GNSS (NavIC-capable)** track for the sovereign deployment.
- **Rationale (single-pass angle).** DJI lets us validate the single-pass pipeline immediately on integrated RTK hardware; the open track ensures the *deployed* system is customizable, air-gappable, and free of foreign platform dependence — the sovereignty requirement. The software is hardware-agnostic behind a sensor-abstraction layer, so both feed the same pipeline.
- **Trade-offs & risks.** Supporting two hardware profiles doubles integration/calibration surface; DJI's closed ecosystem limits low-level access. Mitigation: sensor-abstraction interface so the pipeline is platform-neutral.
- **Revisit if.** The sovereign platform matures enough to be the sole track, or mission constraints forbid DJI entirely (then open-track only).

### ADR-15 — Explicit CRS and geoid handling (UTM + EGM2008)

- **Context.** "Metric and georeferenced" requires an explicit Coordinate Reference System (CRS) and a vertical datum. GNSS gives ellipsoidal height; users and maps expect **orthometric** height (above mean sea level).
- **Options.** (a) Report raw WGS84 lat/long + ellipsoidal height; (b) project to local UTM/EPSG and apply a geoid model (EGM2008 + local geoid) for orthometric height.
- **Decision.** **Project to the appropriate UTM/EPSG zone (PROJ/GDAL) and apply EGM2008 (+ local geoid where available) for orthometric height**; carry CRS/datum metadata in every exported product.
- **Rationale (single-pass angle).** A single autonomous pass still has to drop into the user's mapping frame correctly the first time; getting CRS/geoid wrong silently corrupts every downstream measurement. Explicit handling makes the georeferenced deliverables interoperable and auditable.
- **Trade-offs & risks.** Geoid grids must be bundled for offline/air-gapped use; wrong zone selection is a footgun. Mitigation: derive zone from GNSS, validate against telemetry, and record datum in metadata.
- **Revisit if.** A deployment standardizes on a different national CRS/geoid (swap the grids; the mechanism is unchanged).

### ADR-16 — Design to the mandatory input contract; optional sensors are upgrades

- **Context.** The official problem statement makes only **drone video (1080p/4K), GPS coordinates and flight metadata** mandatory. IMU, barometric altitude, camera intrinsics and RTK/PPK are listed as **optional**. A design that quietly assumes an IMU or a calibration file is a design that fails on the dataset we are actually handed.
- **Options.** (a) Assume the full sensor suite and degrade if something is missing; (b) design the mandatory-only configuration as the *reference* path and treat every optional sensor as an additive upgrade; (c) build two separate pipelines.
- **Decision.** Option (b). The **reference configuration is video + GPS + flight metadata, with self-calibrated intrinsics and scale from GNSS baselines + visual structure**. Optional sensors enter as *additional factors* in the same graph (ADR-04) and *additional constraints* on the same depth stack — never as prerequisites. A **capability flag** travels with every keyframe (see [Integration](04-INTEGRATION.md) §2) and every stage branches on it explicitly.
- **Rationale (single-pass angle).** This inverts the usual failure mode. Calling the no-IMU, no-RTK, unknown-intrinsics case a "fallback" makes it the least-tested path — and it is the *only* path the contract guarantees. Making it the reference means our headline numbers are numbers an evaluator can actually reproduce.
- **Trade-offs & risks.** We give up the gravity prior and inertial scale observability in the reference case, so scale conditioning rests on GNSS-baseline geometry (path length, turn diversity) and vertical uncertainty is wider. Self-calibration can trade focal length against scale on a straight, constant-height pass. Mitigations: parallax-aware keyframe selection (S1), metric-depth cross-checks on scale (S4), focal-length priors from the depth backbone, and honest per-region flagging wherever the ≤ 1 m bar is at risk.
- **Revisit if.** The provided dataset turns out to carry full IMU + RTK + calibration (then Regime A becomes the headline and this stays the guaranteed floor), or a capture SOP can mandate a minimum turn diversity for scale conditioning.

### ADR-17 — Ship the official format set and both viewers; GPL writers run out-of-process

- **Context.** The Desired Output names specific formats — **OBJ, PLY, LAS, GeoTIFF, .glb/.gltf, .fbx** — and a **web-based or desktop viewer**; *User Interface* is 5% of the score and *Model Completeness* 20%. `.fbx` is the awkward one: an Autodesk format with no mature permissive native writer, while our licensing posture (ADR-14, [Technology Stack §5](03-TECHNOLOGY-STACK.md)) forbids shipping restricted code into a military-scope build.
- **Options.** (a) Skip `.fbx` and offer glTF as "equivalent"; (b) link the Autodesk FBX SDK; (c) write `.fbx` by driving headless Blender as an **isolated CLI process**; (d) hand-roll an FBX writer.
- **Decision.** Option (c) for `.fbx`, with the rest of the set produced natively (GDAL for GeoTIFF, PDAL for LAS/LAZ, trimesh/Assimp for OBJ and glTF/GLB). Ship **both** viewers: a web viewer (CesiumJS/Potree over streamed OGC 3D Tiles) *and* a desktop path (QGIS/CloudCompare), each with measurement tools and a per-region confidence overlay. QGIS and Blender are separate GPL applications we invoke, never libraries we link.
- **Rationale (single-pass angle).** A single pass produces one authoritative model, so its portability *is* its value — the deliverable must open in the evaluator's tool of choice with no DRISHTI-specific reader. Two viewers also mean the confidence/coverage layer (ADR-09) is visible in whichever environment the operator already uses, which is where completeness is actually judged.
- **Trade-offs & risks.** Out-of-process `.fbx` adds a heavyweight dependency and a slower export step; a hand-rolled writer would be lighter but is a correctness risk on a format we do not control. Two viewers double the UI surface for a 5% criterion. Mitigations: `.fbx` is generated in the export tail (S10) where minutes are available inside the < 15 min budget, and both viewers share one tiled data source and one measurement API ([Integration](04-INTEGRATION.md) §8) rather than being two products.
- **Revisit if.** A permissive, well-tested FBX writer matures (drop the Blender hop), or the evaluator standardizes on one viewer.

---

## 4. Cross-cutting trade-off map

First, against the **official weighted rubric** — which decisions carry which percentage:

| Official criterion | Weight | Load-bearing decisions |
|--------------------|--------|------------------------|
| **Reconstruction accuracy** | **30%** | ADR-03, 04, 15, 16 — metric spine, variable factor graph, CRS/geoid, self-calibrated mandatory-input baseline |
| **Model completeness** | **20%** | ADR-01, 02, 06, 07, 09 — prior-assisted geometry, few-shot 3DGS, meshing, coverage & confidence report |
| **Processing speed** | **20%** | ADR-05, 11, 13 — two paths, TensorRT INT8/FP16, right fusion engine per tier; budgeted to < 15 min / 10-min video |
| **Innovation** | **15%** | ADR-01, 03, 09, 16 — prior-assisted single-pass orchestration, confidence as a first-class output, mandatory-input-first design |
| **Scalability** | **10%** | ADR-05, 10, 12, 14 — three tiers, ROS 2 data plane, store-and-forward, dual-track hardware |
| **User interface** | **5%** | ADR-17, 09 — web + desktop viewer, measurement API, confidence overlay |

Then, how each decision answers a specific single-pass challenge (challenge→mechanism matrix in the [spec §9](_internal/CANONICAL-ARCHITECTURE-SPEC.md)):

| Challenge (single-pass) | Decision(s) | Official criterion served |
|-------------------------|-------------|---------------------------|
| Short baselines starve triangulation | ADR-01, 06 | Model completeness (20%) |
| Metric scale without GCPs | ADR-03, 04, 15 | Reconstruction accuracy (30%) — both regimes vs the ≤ 1 m bar |
| Only video + GPS + metadata are guaranteed | ADR-16, 03 | Reconstruction accuracy (30%); innovation (15%) |
| Bleeding-edge models OOD on aerial | ADR-02 | Reconstruction accuracy (30%); robustness |
| "Real-time" vs "accurate" tension | ADR-05, 11, 13 | Processing speed (20%) — < 15 min / 10-min video |
| Moving objects → permanent ghosts | ADR-08 | Model completeness (20%); cleanliness |
| No second look to catch errors | ADR-09, 12 | Reconstruction accuracy (30%); auditability |
| Deliverables must open in the evaluator's tools | ADR-17 | User interface (5%); model completeness (20%) |
| Link loss / contested comms | ADR-12, 14 | Scalability (10%); reliability; data sovereignty |
| Sovereign, offline deployment | ADR-14, 12, 15 | Deployability in the NTRO context |
| Cross-tier determinism & reuse | ADR-10, 13 | Scalability (10%); reproducibility |

**What we knowingly lose** (the honest column): against offline photogrammetry incumbents (Pix4D, Metashape, DJI Terra, RealityCapture, ContextCapture, OpenDroneMap) on a *planned multi-pass mission with GCPs*, we expect to lose on **absolute accuracy, facade completeness, and final mesh polish**. Against real-time SLAM/VIO systems we produce a far richer product than their sparse maps. Against NeRF/GS cloud services we are metric, georeferenced, and offline where they are not. DRISHTI's defensible niche — confirmed by the [competitive landscape](_internal/research/8-competitive-landscape.md) — is **single-pass → georeferenced metric textured 3D, near-real-time, with graceful degradation**, which no incumbent currently occupies. We compete there, not on incumbents' home turf.

---

## Open questions / risks

- **Cost of the permissive backbone.** ADR-02's shipped backbone (Depth Anything 3 / MapAnything / Pi3) is now also what [`CANONICAL-ARCHITECTURE-SPEC.md`](_internal/CANONICAL-ARCHITECTURE-SPEC.md) §7 registers, so the naming is reconciled. What remains open is the *accuracy cost* of ruling out the restricted VGGT/MASt3R family — it must be measured on aerial data, not assumed negligible. Tracked jointly with [Technology Stack §5](03-TECHNOLOGY-STACK.md).
- **Scale conditioning with no IMU (ADR-16).** On the mandatory-only baseline there is no gravity prior and no inertial scale observability. Whether GNSS-baseline geometry alone holds the ≤ 1 m bar on a straight-line, constant-height pass is the single biggest unmeasured risk in the design, and the first thing to test on the provided dataset.

- **Aerial fine-tuning is a plan, not a result.** ADR-01/02/06 assume domain fine-tuning brings feed-forward metric accuracy into budget; this is unproven until measured on aerial data (owned with [AI/Deep Learning](roles/3-ai-deep-learning-research.md)).
- **Confidence calibration** (ADR-09) must itself be validated — an uncalibrated confidence is worse than none.
- **Edge smoothing bound** (ADR-04) and **INT8 accuracy** (ADR-11) both need measurement on the event hardware.
- **Two-of-everything cost.** ADR-02/05/13/14 each maintain a primary + fallback or per-tier variant; total integration surface is a real schedule risk for the hackathon MVP (mitigated by the [MVP subset](03-TECHNOLOGY-STACK.md)).

## Further reading

- [How It Works](02-HOW-IT-WORKS.md) — the pipeline these decisions shape.
- [Technology Stack](03-TECHNOLOGY-STACK.md) — the concrete components chosen, with licensing.
- [Integration](04-INTEGRATION.md) — how the decisions compose across tiers.
- [Theory](01-THEORY.md) — the first principles behind the single-pass reasoning.
- Internal: [`CANONICAL-ARCHITECTURE-SPEC.md`](_internal/CANONICAL-ARCHITECTURE-SPEC.md) §9 (challenge→mechanism), research dossier [`8-competitive-landscape.md`](_internal/research/8-competitive-landscape.md).
