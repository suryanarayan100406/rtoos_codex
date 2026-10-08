# Input and configuration reference

## Required descriptor

`name`, recorded `video`, and `telemetry.{format,path,altitude_datum}` are required.
Flight metadata is represented by actual video codec/dimensions/PTS and recorded
telemetry clock/position/optional attitude. Unknown camera settings or sensors
remain absent, not synthesized. Descriptor paths resolve relative to that file.

```yaml
name: mission
video: flight.mp4
telemetry:
  format: csv
  path: track.csv
  altitude_datum: ellipsoidal
  time_origin_s: 0.0
  csv:
    time: timestamp
    lat: latitude
    lon: longitude
    alt: abs_alt
```

CSV columns are configurable; optional yaw/pitch/roll mappings and recorded
`sigma_h_m`/`sigma_v_m` may be supplied. Mandatory coordinates and time must be
finite, geographically valid, and unambiguous. Duplicate timestamps fail.

## Telemetry and clocks

| Format | Required measured metadata |
|---|---|
| `dji_srt` | Subtitle times, latitude, longitude, **absolute** altitude |
| `csv` | Explicit column map and timestamp clock origin |
| `mavlink` `.tlog` | Recorded message timestamps and valid absolute GPS |
| `mavlink` `.bin` | DataFlash GPS with logged `TimeUS` and absolute MSL height |
| `exif` | Actual GPS coordinates/altitude reference and GPS UTC date/time |

Video must remain a recorded file; EXIF may supply a time-stamped sidecar image
track, not a synthesized video acceptance dataset. S0 keeps decoded frame PTS.
Telemetry is aligned as `source_time - time_origin_s + ingest.time_offset_s`.
SRT already uses clip-relative time. Other clocks require explicit origin; ISO
times require timezone. No index-as-time, automatic epoch guessing, endpoint
extrapolation, or long-gap constant GPS substitution is performed.

Inspect `s0_ingest/frames.json`, `telemetry.json`, and decode logs before geometry.
`gps=null` marks unsupported temporal association.

## Heights and coordinate provenance

- `unknown`: absolute altitude supplied, datum unspecified, vertical accuracy
  unvalidated. This is an explicit declaration, not a default.
- `ellipsoidal`: measured ellipsoidal height.
- `orthometric`: measured MSL/geoid-related height; retain the source definition.
- `relative`: rejected for georeferenced reconstruction. Supply a valid absolute
  track after applying a surveyed reference; this build does not infer one.

MAVLink MSL and ellipsoid fields are distinct. EXIF standard height cannot be
relabeled ellipsoidal. `crs.vertical: source` preserves supplied height. Changing
datum requires a real local PROJ geoid grid and known source datum. Grid identity
is hashed as an input; conversion failure is explicit. No implicit geoid download.

Horizontal CRS derives from median GPS using UTM including official Norway and
Svalbard exceptions. High latitudes/antimeridian tracks require an explicit
metre-projected CRS (`mode: epsg`, `epsg: ...`). Float64 local geometry obeys
`projected_xyz = local_xyz + origin`.

## Optional streams

`optional.intrinsics` is a JSON camera calibration used by SfM; its exact schema
is validated in `recon/sfm.py` and covered by binding tests. `optional.rtk` is a
corrected CSV track in the same declared datum/clock. Actual association/use is
recorded. IMU files are retained/hashed with availability recorded; preintegration
is not implemented. Optional barometer height CSVs can now be aligned and fused
when explicitly enabled, as described below. Their input availability and actual
accepted factor use are reported separately.
Optional data never becomes a prerequisite for the video+GPS baseline.

### Optional barometer height fusion

Supply `optional.baro` as a CSV of recorded **positive-up sensor height in metres**.
Pressure-only logs need a separately documented conversion before use. The sensor's
zero does not need an absolute geographic reference: each solver jointly estimates
one unknown constant offset while GNSS remains the absolute height anchor. This
does not permit relative altitude to replace mandatory GPS/telemetry altitude.

The following descriptor addition assumes the barometer timestamps are already
clip-relative; replace the origin and mappings with the real logged-clock values.

```yaml
optional:
  baro: recorded_height.csv
barometer:
  enabled: true
  time_origin_s: 0.0
  time_offset_s: 0.0
  max_time_error_s: 0.1
  sigma_m: 2.0
  min_measurements: 3
  max_samples: 1000000
  csv:
    time: timestamp
    height: height_m
    sigma: null
    delimiter: ','
```

Enabled fusion requires an explicit clock origin. Timestamps may be numeric seconds
or timezone-qualified ISO-8601, and must be unique; heights must be finite. A
configured sigma column must supply positive finite values on every row. With
`sigma: null`, `sigma_m` is a recorded noise assumption, **not measured accuracy**.
Column mappings must be nonblank and distinct. Unknown configuration keys fail.

S0 matches `frame.time_s + barometer.time_offset_s` against
`logged_time - barometer.time_origin_s`, choosing the nearest recorded sample
within `max_time_error_s`. It never interpolates heights, extrapolates outside the
logged range or guesses the clock from a row index. Missing, malformed or
nonoverlapping barometer input records a reason while preserving mandatory GPS.
Optional alignment failures do not become QA drop flags.

S2/S6 validate clock consistency and distinct raw support before fusion. Reused
samples contribute at most one factor; conflicting raw provenance is excluded.
At least `min_measurements` distinct samples are required. The residual is
`(local_camera_z + estimated_bias - recorded_height) / sigma`. The free bias is a
solver variable with no fabricated calibration prior. The constant-offset model
does not compensate weather drift or estimate a camera/sensor lever arm.

Inspect `s0_ingest/barometer.json` for parsing/alignment, and S2/S6
`barometer_fusion` for accepted final usage, solver attempts, retained provenance,
offset/fit and noise sources. An accepted BA result determines final usage;
otherwise the accepted graph or retained S2 source is identified. Reports null
current usage/fit for stale or failed stages. Sensor-fit RMSE is not independent
accuracy. Resume freshness covers the input file, clock/noise settings and stage
revisions. See [BAROMETER-CONTRACT](BAROMETER-CONTRACT.md) for the full interfaces.

## Precedence and profiles

Highest wins: `--set`/explicit CLI flags → environment → descriptor → profile →
`configs/default.yaml`. Unknown fields and invalid numeric/enum values fail.
Environment example: `DRISHTI_DENSE__VOXEL_M=0.2`.

```bash
drishti run --dataset mission.yaml --profile fast \
  --set ingest.max_frames=120 --set dense.voxel_m=0.2
```

`fast`, `balanced`, `max`, and `kaggle_quality` all use implemented point-fusion/Poisson/best-view
methods. Quality changes sampling, resolution and bounded processing settings;
no profile silently selects an unimplemented Gaussian method.

`ingest.max_frames=0` keeps the complete decoded clip. A positive value limits
published frame products for an intentional smoke test, but the decoder continues
through the recorded PTS stream and records every unprocessed tail frame with
`reason: max_frames_limit`; it does not silently claim that the tail was processed.

`kaggle_quality` increases matching/BA budgets, uses 1036-pixel Small depth,
0.05 m fusion voxels, an 8192-pixel atlas and 0.20 m raster sampling. Use the
[Kaggle notebook/runner](KAGGLE.md) for its separate RAM/VRAM controls and
metadata-based full-flight cadence. Loading the profile through the core CLI
alone does not enable the Kaggle resource watchdog. Runner operational tunables
are defined by `KaggleSettings` in `src/drishti/runtime/kaggle.py` and
`configs/kaggle.yaml.example`; paths resolve relative to that settings YAML.

### Joint visual/GNSS refinement

`bundle_adjustment` configures S6's calibrated reprojection/GNSS solve. It defaults
to enabled and uses `poses` calibration without modifying K or the undistorted
images. It requires connected, non-collinear multi-view feature support and at
least three valid GNSS factors. Insufficient support or nonconvergence preserves
the real pose-graph result and records a degraded reason.

```yaml
bundle_adjustment:
  enabled: true
  reprojection_sigma_px: 1.0
  min_track_length: 3
  max_points: 10000
  max_observations: 100000
  max_nfev: 100
  ftol: 0.000001
  xtol: 0.000001
  gtol: 0.000001
  min_depth_m: 0.01
```

Limits apply to whole selected tracks; selection prioritizes camera coverage and
connectivity. Solver convergence and positive camera-Z acceptance are required
before updating any geometry. `spine.robust_delta`, `rotation_sigma_rad`, and
GNSS sigma assumptions also apply. Actual per-sample GNSS uncertainties are used
when supplied. `spine.optimize=false` disables both graph and joint refinement.
`--set bundle_adjustment.enabled=false` keeps the graph-only path. Resume includes
this section in S6 freshness and invalidates downstream depth after a changed solve.

The complete typed reference is `src/drishti/config/models.py`; every shipped
algorithm knob is visible in `configs/default.yaml`. `CONTRACTS.md` summarizes
fields, while the typed model/YAML are authoritative for current defaults.
Server operational tunables are in `server/settings.py`; browser build-time
poll/pick limits are documented in `viewer/README.md`.

## Optional semantic overlays

Semantics default to `model: null`, with no semantic checkpoint required by the
video+GPS baseline. Provision a license-cleared local Transformers semantic
segmentation directory using the [registry guide](MODELS-LICENSES.md#semantic-checkpoints).
Its registry entry must declare `output_domain: semantic_logits`. Use resize-only
image preprocessing; crop/pad processors are explicitly unavailable until their
inverse pixel transform is supported.

Add a `semantics` section alongside the dataset fields in your descriptor. This
mapping is an **illustration**, not a supplied checkpoint's label inventory:
replace the source channel IDs and class names with those from your exact model's
`config.json`/`id2label` and record the intended grouping.

```yaml
semantics:
  model: aerial_semantics
  min_probability: 0.5
  preview_long_side: 1024
  classes:
    - {id: 1, name: Ground, color: [170, 140, 100], source_ids: [0]}
    - {id: 2, name: Building, color: [220, 100, 60], source_ids: [1]}
    - {id: 3, name: Vegetation, color: [60, 160, 80], source_ids: [2]}
    - {id: 4, name: Water, color: [50, 110, 210], source_ids: [3]}
```

Output IDs must be unique integers in 1–65534; 0 is reserved for nodata. Names
must be nonblank and unique after trimming and Unicode case folding. RGB values
are integers in 0–255. Source IDs are nonnegative checkpoint channel indices;
each may map to only one output class. Unmapped, low-probability and dynamically
excluded pixels have class/probability 0. Combining source channels into an output
class does not sum their probabilities: confidence retains the raw winning
channel's softmax probability and is uncalibrated.

Inspect `s3_masking/semantics.json` for `disabled|unavailable|partial|ok`, actual
checkpoint provenance and frame failures. S9 emits `s9_geo/semantics.json` even
when semantic rasters are unavailable. Successful products are class uint16
GeoTIFF (nodata 0), float32 probability GeoTIFF (nodata -9999), and a bounded
RGBA geographic preview. Labels are projected only onto observed, supported DSM
cells with final-camera visibility and dynamic exclusion. Preview failure can
leave the original class raster downloadable.

Resume detects checkpoint-content, class-mapping, threshold and preview changes,
including a newly supplied checkpoint after previous unavailability. Reports
separate semantic status/probability from geometric confidence and accuracy. The
Cesium overlay is 2D ellipsoid imagery and never measurement geometry.

## Independent checkpoint CSV

See `report/accuracy.py`: use `latitude,longitude,elevation_m,vertical_datum`, or
projected `x,y,z,epsg,vertical_datum` with `coordinate_frame=projected_absolute`.
Points must be independent, not fitted GPS/GCPs. Datum must match the DSM. Empty
or inferred/nodata/out-of-footprint observations are excluded and counted.
The validation is DSM elevation-only, not a full 3D acceptance certificate.
