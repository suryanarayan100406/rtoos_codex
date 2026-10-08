# DRISHTI — execution plan for this checkout

## Goal and authority
Implement the offline, ground-only product specified by the existing PRD F1–F19
and N1–N10. The supplied docs are retained as source material. Historical guides
and audit describe an absent earlier checkout, so their test/results claims are
not evidence for this build.

## Sequence and agent work packages
1. **Review:** read the entire source corpus, identify conflicts, and fix shared
   artifact/API contracts before coding. Completed before implementation.
2. **Foundation agent:** Python packaging, strict layered configuration, bundle
   manifests and hashing, stage runner, CLI, runtime detection and license/cache
   gate. Own config/bundle/runtime/models-registry/runner/CLI/packaging.
3. **Input and metric-spine agent:** deterministic PyAV decoding; DJI SRT/CSV/
   MAVLink/EXIF adapters; frame QA; COLMAP self-calibration; robust similarity
   alignment and a real GTSAM visual+GNSS pose graph. Own S0/S1/S2/S6 and helpers.
4. **Perception and reconstruction agent:** offline RT-DETR and motion masking;
   Small depth inference with inverse-depth-correct metric fitting; masked,
   confidence-filtered bounded point fusion; Open3D meshing and observed-image
   UV texturing. Own S3/S4/S7/S8 and helpers. Optional neural enhancers remain
   clearly identified until actually implemented.
5. **Geospatial/output agent:** DSM/ground-classified DTM, image-projected
   orthomosaic, CRS metadata, required exporters, isolated Blender FBX, report
   generation and checkpoint validation. Own S9/S10/report and helpers.
6. **Application agent:** local FastAPI bundle/measurement API; React, offline
   Cesium georeferenced mesh and Potree cloud viewer; real measurement and layer
   controls. Own server/viewer and their tests.
7. **Coordinator:** integration, runbooks/API/data/license/validation docs,
   Docker/cloud runner/CI, dependency installs and tests, cross-agent review.

Each agent reads the source requirements plus `CONTRACTS.md`; all operate on
disjoint paths. The coordinator resolves integration failures before completion.

## Required correctness gates
- Missing GPS/time/absolute altitude → actionable failure, never fabricated data.
- Frame PTS and telemetry clocks/offset/gaps explicitly accounted for.
- Similarity uses camera centers, checks conditioning, flags weak flight geometry.
- GTSAM must actually optimize; visual+GNSS baseline does not require IMU.
- Relative inverse depth fits inverse metric Z, then inverts. Unanchored maps are
  excluded from metric fusion. Confidence measures residual/support, not success.
- Local projected origin and camera transform direction are explicit. Float64
  working coordinates; display transforms persisted in sidecars.
- Inferred mesh regions remain flagged; empty raster cells remain nodata.
- GPS-fit error is separate from independent check-point error. No fabricated
  ≤1 m, completeness, timing, or 16 GB compliance claim.
- Cache registry verifies exact model files and immutable revision offline.
- Resume checks current inputs, parameters, code, model files and output integrity.
- All profiles select executable baseline methods.

## Continuation — S6 joint refinement
The 2026-10-05 increment added joint fixed-calibration reprojection/GNSS
bundle adjustment after the existing global pose graph. SciPy's sparse solver
optimizes supported camera poses and sparse points together, with deterministic
track budgets and unchanged-input failure fallback. S6 publishes matching
poses/points/anchors; unsupported points cannot retain stale depth-anchor confidence
after a pose change. Intrinsics stay at S2 calibration. Constructed numerical and
stage-contract checks are evidence for the implementation; real-data acceptance
remains a separate required gate.

## Continuation — optional semantic overlays
The 2026-10-06 increment completes the in-progress F9/F15 semantic path in five
disjoint parallel workstreams: S3 inference, S9/S10 raster products, server/viewer,
foundation freshness, and reporting. Semantics default to disabled. When enabled,
the checkpoint registry gates a supplied local semantic-logits model and explicit
source-channel mapping. Dynamic exclusion, camera-shaped labels/probabilities,
final-camera observed-DSM projection, bounded geographic previews, optional exports,
class legends/downloads and honest retained-provenance reports are connected.

S9 explicitly depends on S3 for its masks and semantic descriptor. S3 freshness
includes configured checkpoint contents even after unavailable inference; S9
includes semantic projection/preview parameters and upstream revisions. Model
detection does not create execution provenance. Semantic failures preserve required
geometry and raster products. Constructed contract/lifecycle checks verify this
implementation; approved checkpoint and real-scene acceptance remain pending.

## Continuation — optional barometer fusion
The subsequent 2026-10-06 increment implements optional barometer fusion in four
disjoint workstreams: typed ingest/alignment, real GTSAM/SciPy factors, stage/
freshness integration, and partial-safe reports. A free constant sensor offset
preserves mandatory GNSS as the absolute anchor. Exact logged clocks, distinct
source support, uncertainty provenance and accepted final usage are explicit.
See `BAROMETER-CONTRACT.md`; real-flight acceptance remains pending.

## Validation and exit criteria
Run Python unit/contract/API tests, lint, CLI doctor/help, config/schema checks,
viewer typecheck/production build, and bundle integrity/resume tests. Tests for
isolated math/parser units may use explicitly constructed inputs; they cannot
be represented as real reconstruction acceptance. Real end-to-end acceptance
requires supplied footage, telemetry, approved checkpoints, and toolchain.

Publish a requirement-by-requirement status with implemented/tested/unvalidated/
missing distinctions. A missing dataset, GPU runtime, geoid, or Blender is recorded
as a validation/dependency gap rather than falsely checking off a phase.

## Source conflicts resolved
| Conflict | Resolution |
|---|---|
| Live core docs vs offline PRD | Offline scope, no S5 or edge deployment |
| V2 Base called Apache in design | Use Small only; reject NC checkpoints |
| Pi3 and default MapAnything called permissive | No default adoption without exact checkpoint evidence |
| GTSAM/GLOMAP/PDAL detection treated as execution | Record only algorithms actually invoked |
| GUIDE `mode: set` vs RUN_GUIDE `mode: epsg` | `mode: epsg` explicit override |
| Epoch timestamps vs clip time | User-declared clock origin/offset; never guess |
| Orthometric label without conversion | Require datum declaration and geoid for conversion |
| Identical whole manifests despite timestamps | Stable content identity separate from measured execution metadata |
| No real dataset in this checkout | Machinery/tests can be verified; product accuracy remains unvalidated |

## Documentation deliverables
Root README and AGENTS; current quickstart; configuration and input contracts;
architecture/artifact contracts; API and measurement definitions; model/license
ledger; local/cloud/offline deployment; testing and real-data acceptance protocol;
troubleshooting; measured status and F1–F19/N1–N10 traceability.
