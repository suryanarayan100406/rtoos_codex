# Optional barometer integration contract

This increment consumes recorded, positive-up barometric **height in metres** from
an optional CSV. It does not convert pressure, supply a geographic altitude datum,
replace mandatory absolute GPS, or implement IMU preintegration. Barometer fusion
is disabled by default; malformed/missing/nonoverlapping optional input preserves
the visual/GNSS baseline with a recorded reason.

## Typed configuration

Root `barometer` section:

- `enabled: bool = false`
- `time_origin_s: float | null = null`: explicit recorded clock origin required
  when enabled; numeric seconds or timezone-qualified ISO source timestamps use
  the existing timestamp parser. No first-row/epoch guessing.
- `time_offset_s: float = 0.0`: matching query is
  `frame.time_s + time_offset_s` against `logged_time - time_origin_s`.
- `max_time_error_s: positive float = 0.1`: bounded nearest recorded sample;
  no interpolation, extrapolation, or matching outside the logged range.
- `sigma_m: positive float = 2.0`: explicitly recorded assumed measurement noise
  when no uncertainty column is configured; not calibrated accuracy.
- `min_measurements: int >= 2 = 3`: minimum distinct sensor samples for fusion.
- `max_samples: int >= min_measurements = 1000000`: parser memory budget.
- `csv: {time: 'timestamp', height: 'height_m', sigma: null, delimiter: ','}`:
  explicit nonblank column mappings, distinct configured columns, one-character
  delimiter. A configured sigma column must contain finite positive values on
  every record. Height/time must be finite, timestamps unique. Unknown keys fail.

Input remains `dataset.optional.baro: Path | null`, already retained and hashed by
the bundle. Old saved configs gain the disabled default.

## S0 → S1 → S2 camera observations

S0 always publishes output `barometer -> s0_ingest/barometer.json`:
`status='disabled|not_provided|unavailable|ok'`, `reason`, `clock`, `samples`,
`aligned_frames`, `measurement='relative_sensor_height_m'`,
`absolute_datum='not_interpreted'`. Samples are real parsed records, sorted by
logged timestamp, with raw `source_time_s`, normalized `time_s`, `height_m`,
`sigma_m`, `sigma_source='csv|barometer.sigma_m'`, `source_index` and `source_line`.

Frames retain `barometer: observation | null` and `barometer_reason`; optional
alignment failures never become mandatory GPS/QA drop flags. Observations are
copies of actual nearest samples, adding signed `time_error_s` and
`frame_time_s`. S0 capability ledger records parsing/alignment separately from
fusion (`used=false`, `status='aligned_not_fused'` for successful alignment).
S1 retains these frame fields. S2 explicitly transfers by frame ID to registered
cameras; undistorting pixels does not change recorded sensor timestamps.

## Optimizer interfaces and model

Backward-compatible signatures:

```python
optimize_pose_graph(cameras, spine_settings, *, global_refinement=False,
                    barometer_settings=None) -> (cameras, metadata)
optimize_bundle(cameras, sparse, ba_settings, spine_settings, *,
                barometer_settings=None) -> (cameras, sparse, metadata)
```

A shared `recon/barometer.py` helper validates optional observations and selects
distinct raw `source_index` values. For repeated associations choose the camera
with smallest absolute timing error, tie by camera index. Do not turn reuse of one
record into independent factors. Metadata records selection/exclusion reasons,
actual uncertainty sources and the nearest-sample timing approximation.
Both clock equations are revalidated against current origin/offset settings with
binary64 arithmetic-roundoff tolerance only; the configured association bound
stays strict. Conflicting height/uncertainty/timestamp/line provenance for one raw
source index, or one raw timestamp reused across different source indices, is
excluded rather than counted as independent support.

When at least `min_measurements` distinct observations remain, optimize one free
constant sensor-offset variable `bias_m` jointly with camera poses (and points for
BA). Each residual is
`(camera.center[2] + bias_m - observed.height_m) / observed.sigma_m`.
GNSS supplies absolute height/georeference; the free bias avoids promoting a
relative sensor reading to an absolute altitude. No fabricated bias prior is
added. Initialize bias from the median observed height minus current local Z;
that is a solver initial value, not a measured calibration. This constant-offset
model does not estimate weather drift or a measured camera/sensor lever arm.

GTSAM uses actual Pose3/Double custom factors and the configured robust noise
model; Jacobians must match its local tangent coordinates. Joint SciPy BA extends
the state and sparse Jacobian with one bias variable and per-camera Z residuals.
Retain fixed-calibration support/positive-depth/convergence gates. Bad optional
observations or inadequate support skip barometer factors while preserving the
baseline solve; solver failure retains real input geometry as before.

Both optimizer metadata objects expose `baro_used` (true only for an accepted
result with actual barometer factors), and nested `barometer` with `status`,
`reason`, `factors_added`, `optimizer_invoked`, selected/excluded source support,
uncertainty provenance, `bias_m`, `fit_rmse_m`, `fit_is_independent_accuracy=false`,
`model='relative_height_with_constant_unknown_bias'`, and
`absolute_datum='not_interpreted'`. Bias/fit remain null without an accepted
barometer solution. Existing no-barometer method/revision names stay compatible;
enabled execution is identifiable from actual factor metadata.
`source_index` refers to zero-based CSV record order before sorting;
`source_line` is the one-based physical start line. BA uses central finite
differences for the active barometer model and preserves the prior differentiation
path when barometer factors are absent.

## Stages, freshness and reporting

S0/S2/S6 config sections add `barometer`. S2 and S6 pass enabled settings to both
actual solvers; no merely detected sensor can set `used=true`. Final-pose optional
input provenance reflects the accepted solve: accepted BA's factors, otherwise
the accepted graph's factors, otherwise retained input's prior provenance. Final
poses/sparse anchors remain coherent and BA clears obsolete covariance as before.

S6 publishes an explicit sensor summary identifying final-result usage separately
from attempts. Reports retain input availability, alignment, solver attempts and
final usage with stage status; stale/failed support cannot become current sensor
use or independent accuracy. Input hashes and barometer config participate in
resume freshness. Constructed parser/solver/contract tests are implementation
evidence, not recorded-flight or accuracy acceptance.

`report.json.barometer_fusion` includes configured enablement, recorded S0 input
descriptor/ledger, final stage/status/currentness, selected final source, current
use/bias/fit/factor counts, uncertainty provenance and exact annotated recorded
attempts/inherited provenance. Current measurements stay null for stale, failed,
blocked or unavailable evidence. A recorded S6 always takes precedence over S2
as final-stage evidence, including when S6 is unsuccessful.
