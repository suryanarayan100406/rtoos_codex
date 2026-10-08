# Shared implementation contracts — v1

All paths below are relative to the project bundle. Units are metres, seconds,
degrees for latitude/longitude, and OpenCV camera axes. Never invent missing data.

## Python interfaces
The optional recorded-barometer increment is specified in
[`BAROMETER-CONTRACT.md`](BAROMETER-CONTRACT.md). Its default is disabled; relative
sensor height uses an optimized nuisance offset while GPS remains the absolute
georeference anchor. Optional alignment/fusion cannot become mandatory input.

`from drishti.stages.base import StageResult, Stage`:

```python
@dataclass
class StageResult:
    outputs: dict[str, str]  # logical name -> bundle-relative file/directory
    metrics: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    degraded_reason: str | None = None
    models: list[dict[str, Any]] = field(default_factory=list)

class Stage:
    name: str
    requires: tuple[str, ...]
    config_sections: tuple[str, ...]
    def run(self, bundle: Bundle, cfg: Config, runtime: Runtime) -> StageResult: ...
```

Each module exposes `STAGE = S0Ingest()` (with its appropriate class). Heavy
imports happen within execution, so core CLI imports without optional extras.
`Bundle.root` is a Path; `Bundle.manifest` is a JSON-compatible dict.
`Bundle.stage_dir(name)` creates/returns a Path. `read_json(relative)` and
`write_json(relative, data)` read/write atomically; `artifact(stage, key)` returns
an absolute Path from the recorded stage outputs. `save()` atomically saves the
manifest. Agent code can use direct NumPy/image writes inside its stage directory.

`Config.dataset` is a DatasetDescriptor: `name`, `video: Path`,
`telemetry.{format,path,csv,time_origin_s,altitude_datum}`,
`optional.{imu,baro,intrinsics,rtk}: Path|None`, `report.check_points: Path|None`.
Paths resolve relative to the descriptor directory. `cfg` sections and fields:

| Section | Fields |
|---|---|
| run | seed=1234, output_root=Path('runs'), copy_inputs=True |
| ingest | target_fps=4.0, max_frames=0, time_offset_s=0.0, max_telemetry_gap_s=2.0 |
| barometer | enabled=False, time_origin_s=None (required when enabled), time_offset_s=0.0, max_time_error_s=0.1, sigma_m=2.0, min_measurements=3, max_samples=1000000, csv={time:'timestamp',height:'height_m',sigma:None,delimiter:','} |
| frameqa | min_sharpness=30.0, max_dark_fraction=0.6, max_bright_fraction=0.6, dark_threshold=10, bright_threshold=245, min_parallax_px=8.0, max_interval_s=2.0, max_keyframes=300, orb_features=2000 |
| poses | matcher='sequential', overlap=10, camera_model='SIMPLE_RADIAL', min_registered=3, min_track_length=3, use_glomap=False |
| spine | gnss_sigma_h_m=3.0, gnss_sigma_v_m=5.0, visual_sigma_m=0.2, rotation_sigma_rad=0.05, robust_delta=1.345, alignment_threshold_m=10.0, ransac_iterations=300, min_baseline_m=1.0, min_rank_ratio=0.001, optimize=True, max_iterations=100 |
| bundle_adjustment | enabled=True, reprojection_sigma_px=1.0, min_track_length=3, max_points=10000, max_observations=100000, max_nfev=100, ftol=1e-6, xtol=1e-6, gtol=1e-6, min_depth_m=0.01 |
| crs | mode='derive_from_gps'|'epsg', epsg=None, vertical='source', geoid_path=None |
| masking | model='rtdetr' or null, threshold=0.5, dilation_px=5, dynamic_classes=['person','car','truck','bus','motorcycle','bicycle','bird','dog','cat','horse'], motion_fallback=True, motion_threshold_px=3.0, min_motion_matches=16, motion_unknown_policy='exclude|retain', max_motion_exclusion_fraction=1.0 |
| semantics | model=None (disabled), min_probability=0.5, preview_long_side=1024, classes=[{id:1..65534,name,color:[R,G,B],source_ids:[checkpoint output channel IDs]}]; model requires explicit classes; IDs/names/source mappings unique |
| depth | model='depth_anything_v2_small', backend='torch'|'openvino', input_long_side=518, min_anchors=8, robust_delta=1.345, min_depth_m=0.1, max_depth_m=300.0, max_relative_residual=0.3 |
| dense | method='point_fusion', voxel_m=0.1, pixel_stride=4, min_confidence=0.2, max_points_per_chunk=1000000, tile_m=40.0, max_depth_m=300.0 |
| mesh | method='poisson', poisson_depth=9, density_quantile=0.05, max_vertices=500000, support_radius_m=0.5, texture='best_view', texture_size=2048, texture_candidate_views=4, normal_neighbors=30 |
| geo | resolution_m=0.5, max_cells=20000000, ground_method='morphological', ground_window_m=10.0, ground_threshold_m=0.5, nodata=-9999.0 |
| export | formats=['obj','ply','las','geotiff','glb','gltf','fbx'], optional_formats=['laz','3dtiles','potree'], blender='blender', las_scale_m=0.001 |
| report | check_points=None, accuracy_target_m=1.0, region_size_m=20.0 |
| models | registry=Path('configs/models.yaml'), cache_dir=Path('models'), offline=True |
| compute | device='auto', openvino_device='CPU', environment=None |

`runtime.device` is actual torch-compatible `cpu` or `cuda`; `runtime.environment`
describes actual detected host/GPU, never a placement recommendation. Model loader
uses `Registry(cfg.models.registry, cfg.models.cache_dir).resolve(name)` returning
validated spec dict with absolute `path`, `revision`, `license`, `files` hashes,
and optional `runtime`, `source`, `output_domain`. No network downloads in run.

## Artifacts

### S0 (`requires=()`)
Outputs `frames -> s0_ingest/frames.json`, `telemetry -> .../telemetry.json`,
`images -> .../frames/`. Frames object: `{'frames': [...], 'video': {...},
'capabilities': {...}}`; each frame has `id: str`, `path`, `time_s`, `width`,
`height`, `gps: {'latitude','longitude','altitude_m','time_s'}|None`.
Telemetry `{'samples': [{'time_s','latitude','longitude','altitude_m', ...}],
'altitude_datum': ..., 'clock': ...}`. `manifest.crs` includes `epsg`,
`derived_from`, `vertical`, and source datum. All-absolute datum unknown is
explicitly unvalidated, not ellipsoidal by assumption.
Output `barometer -> s0_ingest/barometer.json` is always a real optional status
descriptor (`disabled|not_provided|unavailable|ok`) with recorded samples/clock,
alignment count and uncertainty source. Frames also carry `barometer` (nearest
recorded observation or null) and `barometer_reason`. Optional failures never
replace GPS or become mandatory frame-drop flags. S0 records parsing/alignment
with `used=false`; see `BAROMETER-CONTRACT.md` for exact sample fields.

### S1 (`requires=('s0_ingest',)`)
Outputs `keyframes -> s1_frameqa/keyframes.json` containing `{'frames': [...]}`
with unchanged frame fields and measured QA; `qa -> .../qa.json` logs all drops.

### S2 (`requires=('s0_ingest','s1_frameqa')`)
Outputs `poses -> s2_poses/poses.json`, `sparse -> .../sparse.npz`,
`spine -> .../spine.json`, optional `reconstruction -> .../colmap/`.
Poses object: `frame='projected_local'`, `origin=[projected x,y,z]`, `crs`,
`cameras`: `frame_id`, `image_path`, `time_s`, `width`,`height`, `K` (3x3),
`distortion` list, `R_cw` (3x3 **world/local→camera**), `center` (local XYZ),
`gps` (local XYZ or null), `anchors` list of `[u,v,metric_z,reprojection_error]`.
Points `sparse.npz`: `points` Nx3 float64 local metres, `colors` Nx3 uint8,
`confidence` N floats, `inferred` N bool. Origin must be subtracted before any
float32 cast. Metric graph is actually invoked, or degradation says it was not.
Registered cameras retain aligned barometer observations by frame identity.
`poses.barometer_fusion` and `spine.barometer_fusion` identify actual accepted
graph factor use, solver attempts and raw availability separately. Sensor height
has no interpreted absolute datum; the graph estimates one free constant bias.

### S6 (`requires=('s2_poses',)`)
Outputs `poses -> s6_global/poses.json`, `optimization -> .../optimization.json`,
`sparse -> .../sparse.npz`; poses declare `sparse_path` referencing this final-pose
refined structure. S4 must consume that revision, not the original S2 cloud.
Same schema, real global visual+GNSS refinement followed by joint fixed-calibration
reprojection/GNSS bundle adjustment. `poses.bundle_adjustment` and the optimization
artifact record actual invocation, support selection, solver termination, fit
diagnostics, and any retained-input fallback. Successful joint adjustment declares
`pose_revision='s6_joint_reprojection_GNSS_bundle_adjusted'`; cameras and sparse
points are published together and anchors are recomputed. Point indices/provenance
are retained. Unsupported points lose confidence if poses change; obsolete pose
graph covariance is cleared rather than attached to the joint-adjusted poses.
Depth S4 consumes S6 poses, so
its anchors/calibration match the final pose revision. Runner orders S6 before
S3/S4 for this dependency, documenting the justified correctness delta.
`poses.barometer_fusion` and `optimization.barometer_fusion` identify the selected
final source (`bundle_adjustment|pose_graph|retained_s2`). Accepted final BA governs
usage even when its precursor graph used different factors. Actual attempts and
inherited S2 provenance remain inspectable; `optional_inputs.barometer` and the
legacy `baro` entry retain availability/alignment separately from final `used`.

### S3 (`requires=('s1_frameqa','s6_global')`)
Outputs `masks -> s3_masking/masks.json`, `images -> .../masks/`.
`{'frames':[{'frame_id','path','method','dynamic_fraction', ...}]}`.
PNG 255 = dynamic/exclude, 0 = keep; image shape exactly camera shape.
Optional semantic artifacts are labeled with class mapping and confidence.

Optional output `semantics -> s3_masking/semantics.json`, always a real status
descriptor, with `status='ok|partial|unavailable|disabled'`, `reason`, `classes`
(resolved config class mapping), `nodata=0`, `model` (executed provenance or null),
`source_labels` (actual checkpoint id2label or null), `confidence_method`, and
`frames`. Each successfully inferred frame has `frame_id`, `path` (uint16 NPY
class IDs), `confidence_path` (float32 NPY), `height`, `width`, `image_domain`
('distorted' or 'undistorted' matching the final camera), `labeled_pixels`.
Both maps are exactly camera/image shape. Raw argmax source labels are mapped
explicitly; unmapped/low-probability/dynamically excluded pixels have class 0 and
confidence 0. Confidence is the winning source-class softmax probability,
uncalibrated and separate from geometry confidence. Local transformers semantic
inference requires registry `output_domain='semantic_logits'`; no download or
remote code. Failed optional inference preserves dynamic masks and records a
reason; no invented maps or model-execution claims.
The descriptor may also contain `failures:[{frame_id,reason}]`; failed entries
have no artifact paths. `model.inference_frames` counts actual validated inference,
separately from published map pairs. Output `semantic_maps` references the directory
of published map pairs so resume hashes their contents. Resize-only processors
are supported; crop/pad preprocessing fails optionally without an inverse pixel
transform. S3 configuration/freshness covers `semantics` and the configured
checkpoint inventory even after previously unavailable inference.

### S4 (`requires=('s2_poses','s6_global','s3_masking')`)
Outputs `depth -> s4_depth/depth.json`, `maps -> .../maps/`.
`{'frames':[{'frame_id','path','confidence_path','metric':bool,'anchor_count',
'rmse_m':number|null,...}], 'domain':'camera_z_metres'}`. NPY arrays float32
camera-Z metres, invalid/excluded=0; confidence float32 [0,1]. Never fuse a
`metric=false` map. Relative inverse output fit is `1/Z=a*p+b`, then invert.

### S7 (`requires=('s4_depth','s6_global','s3_masking')`)
Outputs `cloud -> s7_dense/dense.npz`, `ply -> .../dense.ply`,
`metadata -> .../dense.json`. NPZ same points/colors/confidence/inferred contract
plus optional `normals` Nx3 oriented toward observing camera and `labels`.
Point fusion is allowed by F12; use tiled streaming to bound working memory.

### S8 (`requires=('s7_dense','s6_global','s3_masking')`)
Outputs `mesh -> s8_mesh/mesh.npz`, `obj -> .../model.obj`,
`material -> .../model.mtl`, `texture -> .../texture.png`, `metadata -> .../mesh.json`.
NPZ `vertices` Nx3 float64 local coordinates, `faces` Mx3 int, `colors` Nx3,
`confidence` N, `inferred` N, `uv` Nx2 optional; retain sparse provenance.
Observed-image best-view texture, not vertex color labeled as texture.

### S9 (`requires=('s7_dense','s8_mesh','s6_global','s3_masking')`)
Outputs `dsm`, `dtm`, `ortho`, `confidence`, `metadata` under s9_geo.
GeoTIFF rasters carry actual CRS/datum, nodata/alpha, measured coverage masks.
DTM method name must match actual ground classification. Orthomosaic uses
geometry/image reprojection, not colored-cloud gridding relabeled orthophoto.

Optional outputs `semantic` (uint16 class GeoTIFF, nodata=0),
`semantic_confidence` (float32 probability, nodata=-9999), `semantic_preview`
(bounded RGBA PNG actually nearest-neighbor warped into EPSG:4326), and
`semantic_metadata -> s9_geo/semantics.json`. The metadata descriptor has
`id='semantic'`, `title='Semantic classes'`, `status='ok|unavailable|disabled'`,
`reason`, `classes`, `nodata=0`, `raster_path`, `confidence_path`, `preview`
({path,crs:'EPSG:4326',bounds:[west,south,east,north],width,height} or null),
`confidence_method`, `provenance`, `supported_cells`, `accuracy_status='UNVALIDATED'`.
Class projection uses final camera/DSM visibility and dynamic exclusion, samples
categorical labels nearest-neighbor, and never fills unsupported/inferred cells.
Semantic failures are independent of mandatory raster/export availability.
Semantic metadata also appears under `geo.json.semantics` for report provenance.

### S10 (`requires=('s7_dense','s8_mesh','s9_geo','s6_global')`)
Outputs `exports -> s10_export/exports.json` plus each real deliverable.
`{'formats': {'glb':{'status':'ok|unavailable|failed','path':..., 'reason':...},...},
'georeference': {'epsg', 'origin', 'vertical', 'glb_axes':...}}`.
Missing MUST formats degrade status. Optional tools never produce placeholder files.
Available semantic rasters are copied as optional GeoTIFF products, preserving
class mapping/provenance. Per-product optional failures are recorded separately
from the mandatory DSM/DTM/orthomosaic export status.
`exports.json.optional_products` records each optional GeoTIFF product's
`status`, `path`, `reason` and available mapping/provenance details independently.

### report (`requires=('s10_export',)`)
Outputs `json -> report/report.json`, `html -> report/report.html`.
Reports read manifest stages, target/measurement distinction, per-region point
support/confidence, measured timings, actual outputs and independent checkpoint
validation. Missing check data means accuracy `UNVALIDATED`. Runner also calls
the report builder on failure so partial runs remain inspectable.
Each stage record also retains `confidence_status` (`measured_proxy` or
`unavailable`) and `confidence_reason`. A missing numeric confidence summary is
therefore explicit; it is never rendered as a fabricated zero or as evidence of
absolute accuracy.
`report.json.semantics` summarizes recorded S9 semantic status/provenance with
`stage_status`, `current`, `recorded_status`, `source_descriptor`, `classes`,
`confidence_method`, `supported_cells` and available paths. Summary status may be
`partial` when real products survive with missing optional assets. Retained
non-current provenance remains inspectable but has null current support and
download paths. Semantic probabilities never become geometry confidence or
absolute accuracy; semantic accuracy remains `UNVALIDATED`.
`report.json.barometer_fusion` separates configured enablement, recorded S0 input
availability/clock, accepted final solver use and retained attempts/provenance.
It adds `stage_status`/`current`; current use, bias, fit and factor counts remain
null for unavailable or non-current final-stage evidence. A recorded stale/failed
S6 never falls back to presenting S2 sensor usage as current. Bias is an optimized
nuisance variable and fit is not independent accuracy or measured calibration.

## Optional Kaggle orchestration

`drishti.runtime.kaggle` exposes `KaggleSettings`, `load_settings(path)`,
`run(settings)` and `main(argv)`. The source-checkout entrypoint is
`scripts/kaggle_run.py --settings <yaml>`; the installed command is
`drishti-kaggle --settings <yaml>`. Strict settings and defaults are defined in
that module and `configs/kaggle.yaml.example`; YAML paths are settings-relative.
Each worker invokes the existing `StageRunner` with one selected stage, retaining
core artifact schemas, freshness, optional-input policy and real execution records.

`manifest.cloud_execution` and `cloud_run.json` record separate orchestration
status, declared-metadata sampling, one selected physical GPU, measured stop
budgets, sampled process-tree peaks, actual worker results and recovery/archive
outcomes. Package/model inventory remains detection (`executed=false`), distinct
from core inference provenance. Missing measurements remain null; sampled peaks
do not establish a hard instantaneous cap. Core end-to-end wall time is null when
the stage-isolated run did not measure it as one core execution.

`cloud/report_snapshot.json` is the frozen pre-recovery cloud record rendered in
the report. `report.json.cloud_execution.report_cloud_snapshot` declares scope,
exclusions and the authoritative final `cloud_run.json`. Final recovery peaks and
archive outcomes are not retroactively inserted into the report. ZIPs contain a
pre-creation bundle snapshot, with their archive result pending there. The loose
final `cloud_run.json` records the actual creation outcome without recursive
report/ZIP rewriting. See [KAGGLE.md](KAGGLE.md) for operational details.

## Server/UI contract
GET `/api/health`, `/api/doctor`, `/api/runs`, `/api/runs/{id}/manifest`,
`/api/runs/{id}/report` (JSON), `/api/runs/{id}/files/{path}`.
POST `/api/runs` accepts `{dataset,profile}` and returns 202 with run id.
POST `/api/runs/{id}/measure` accepts `{kind:'distance|area|height|volume',
points:[[local_x,local_y,local_z],...], reference_height:number|null}`.
Distance=polyline XYZ; area=horizontal polygon; height=max Z-min Z; volume=DSM
integration inside a horizontal polygon over an explicit local reference plane,
only supported non-inferred cells. Return method, units, support, warning/accuracy
validation alongside a real numeric result; absent surface means error, not an
estimated placeholder. Validate polygons/finite points and keep paths confined.

GET `/api/runs/{id}/scene` adds `semantic_layers` (a list containing the validated
S9 semantic descriptor when present) and `semantic_reason` (availability reason
when absent). Confine all descriptor file paths, including preview paths; invalid
optional descriptors must not block mesh/cloud inspection. Scene S9 output hashes
govern refresh. The viewer selects a 2D geographic semantic imagery overlay,
shows its class legend and unvalidated semantic quality, and exposes its original
GeoTIFF download. Imagery is displayed on the ellipsoid, not claimed to be a
reconstructed surface or used as measurement geometry.
