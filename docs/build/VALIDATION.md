# Testing and real-data acceptance protocol

## Automated checks in this checkout

The test suite is divided into isolated contracts and must not be described as
an end-to-end reconstruction acceptance test. Current categories are:

- typed config, precedence, registry license/hash gate, bundle atomicity,
  freshness, input hash and CLI failure behavior;
- telemetry parsing, timestamp/gap policy, UTM/vertical metadata, Sim(3) rank/
  outlier behavior, actual GTSAM pose graph objective execution, and joint
  reprojection/GNSS BA with support budgets, cheirality and unchanged-input fallback;
- inverse-depth fitting, OpenVINO adapter, detector/motion masks, confidence
  filtering, tiled fusion and actual Open3D Poisson geometry;
- GeoTIFF nodata/support/orthomosaic products, LAS/PLY/OBJ/glTF reader checks,
  GLB recentering/alpha, isolated export status and honest report rendering;
- FastAPI path confinement, background boundary, measurement geometry, DSM
  support/inferred exclusion and viewer coordinate/pick transform scripts;
- frontend TypeScript/build/offline/no-external-request smoke checks.
- optional semantic source-channel mapping, offline loader boundaries, S3 masks/
  labels/provenance, final-camera categorical projection, EPSG:4326 preview warp,
  optional export isolation, checkpoint freshness, reports and imagery lifecycle.
- optional recorded barometer CSV validation, bounded clock association, distinct
  source support, actual GTSAM/SciPy nuisance-offset optimization, tangent/sparse
  Jacobians, offset invariance, noise weighting, fallback and final-use reports.
- Kaggle typed settings, exact checkpoint rebinding, full-flight cadence/resume,
  single-GPU selection, budget/telemetry-stop boundaries, real subprocess cleanup,
  confined atomic ZIPs, actual notebook bootstrap/settings and cloud report scope.
- optional intrinsics self-calibration fallback, final-pose sparse-confidence
  invalidation, isolated-exporter freshness, loaded-versus-executed depth model
  provenance, and explicit max-frame tail drop records.

Run the available checks:

```bash
python -m pytest -q
python -m ruff check src server tests scripts
python -m mypy src server
npm ci --prefix viewer
npm run typecheck --prefix viewer
npm run test:coordinates --prefix viewer
npm run test:semantics --prefix viewer
npm run build --prefix viewer
```

Some external checks are conditional on real Blender/Potree/py3dtiles binaries
and are recorded unavailable when absent. Trained-model acceptance must use the
registered exact local checkpoint. Isolated mocked-loader and constructed-logit
tests verify boundaries and tensor math, and are labeled separately from actual
trained-checkpoint inference.

Targeted joint-refinement checks:

```bash
python -m pytest -q tests/test_bundle_adjustment.py tests/test_global_bundle_adjustment.py
python -m mypy --strict src/drishti/recon/bundle_adjustment.py
```

These constructed numerical cases exercise the actual SciPy solver and S6 graph/
joint-adjustment/anchor publication. They establish implementation behavior, not
aerial reconstruction quality, independent accuracy, or target-hardware speed.

Targeted optional-semantic checks:

```bash
python -m pytest -q tests/test_semantic_inference.py tests/test_semantic_stage.py \
  tests/test_semantic_products.py tests/test_semantic_api.py \
  tests/test_semantic_freshness.py tests/test_semantic_report.py
npm run test:semantics --prefix viewer
```

These constructed IO/geometry/loader/React-Cesium lifecycle cases verify optional
failure isolation, actual artifact storage/readers, offline loader flags and
revision handling. They do not validate checkpoint quality or real WebGL scene
placement. The optional Torch tensor case is skipped when Torch is absent.

### Optional recorded-barometer checks

Targeted recorded-barometer implementation checks:

```bash
python -m pytest -q tests/test_barometer_ingest.py tests/test_barometer_optimization.py \
  tests/test_barometer_stages.py tests/test_barometer_freshness.py tests/test_barometer_report.py
python -m mypy --strict --follow-imports=silent src/drishti/recon/barometer.py \
  src/drishti/recon/spine.py src/drishti/recon/bundle_adjustment.py
```

These constructed tests execute actual GTSAM custom factors and SciPy joint BA,
including a real-S6-to-report mathematical contract. They verify solver behavior
and honest provenance; they do not establish aerial barometer quality, absolute
accuracy or mission performance. A real optional-sensor run requires the original
height log, actual clock mapping and independently characterized noise/drift.

### Kaggle runner and notebook checks

```bash
python -m pytest -q tests/test_kaggle_runner.py tests/test_kaggle_profile.py \
  tests/test_kaggle_notebook.py
python scripts/kaggle_run.py --help
python scripts/package_kaggle.py
```

These checks include actual OS subprocess/process-group cleanup, ZIP storage and
the notebook's extraction/settings code. GPU inventory/CUDA allocator boundaries
and dependency-installation commands use isolated doubles; no actual Kaggle host,
CUDA inference, trained checkpoint or installation benchmark is implied. Full
acceptance additionally needs an actual GPU session, original mission inputs,
checkpoint evidence, output integrity, measured resource scope/peaks and independent
geometry checks. Review the final `cloud_run.json` separately from the report's
pre-recovery and archive's pre-creation snapshots; see [KAGGLE.md](KAGGLE.md).

## Acceptance with a real mission

The evaluator/developer must supply a real single-pass recording plus its actual
GPS/flight metadata. A valid acceptance package contains:

1. raw video and original telemetry, hashes, declared clock/altitude datum,
   optional streams and descriptor;
2. at least one independent check set not used for alignment, with surveyed
   horizontal/vertical datum and uncertainty metadata;
3. approved model files and exact registry evidence;
4. measured host/GPU memory, versions and external tools;
5. a fresh run from an empty bundle and a second resume/rebuild verification.

Review every dropped-frame reason, telemetry gap and S2/S6 conditioning warning.
Open products in a standard GIS/point-cloud/mesh reader. Confirm final CRS,
origin, height datum, no-data masks, texture references, and inferred flags.

Report separately:

- horizontal and vertical independent checkpoint errors, bias, coverage and
  excluded samples;
- GPS/visual alignment residuals as fit diagnostics only;
- observed support and per-region confidence, not as accuracy;
- visible-scene denominator and completeness, only when actually known;
- wall time per stage with actual environment, cached stages and peak memory;
- required export success and the reason for every unavailable format.
- optional semantic class mapping, dynamic exclusion, observed-only support and
  geographic placement; assess semantic class accuracy against independent labels
  separately from geometric accuracy and uncalibrated source probabilities.

Accuracy compliance is **not established** by a small GPS fit residual, a
photogrammetry reference comparison, input GPS points used in the solve, or a
successful export. Full product acceptance is impossible without the real data.
