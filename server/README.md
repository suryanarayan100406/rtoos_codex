# Local survey API

From the checkout root, install the Python package's `server` extra and run:

```bash
python -m pip install -e '.[server]'
python -m server.app
```

The default bind is `127.0.0.1:8000`. `uvicorn server.app:app` is also supported.
There are no mandatory external services. Build `viewer/` first to serve the UI at
the same origin; Vite development proxies `/api` to this local port.

## Typed settings

`server/settings.py` defines application limits and paths. Override individual
fields with `DRISHTI_SERVER_<FIELD>` environment variables:

| Field | Default | Meaning |
|---|---|---|
| `RUNS_ROOT` | `runs` | Bundle directory |
| `DATASET_ROOT` | `.` | Root allowed for submitted descriptor paths |
| `PROFILES_ROOT` | `configs/profiles` | Local fast/balanced/max profiles |
| `VIEWER_DIST` | `viewer/dist` | Production static build |
| `HOST`, `PORT` | `127.0.0.1`, `8000` | Local executable bind |
| `WORKERS` | `1` | Background pipeline concurrency |
| `MAX_POINTS` | `100000` | Bounded browser cloud preview |
| `MAX_NPZ_BYTES` | `256000000` | Uncompressed NPZ memory guard |
| `MAX_VOLUME_CELLS` | `1000000` | Raster cells per integration |
| `MAX_MEASURE_POINTS` | `512` | Picked vertices per measurement |
| `MAX_JSON_BYTES` | `32000000` | Manifest/sidecar/report read limit |

Pipeline `DRISHTI_SECTION__FIELD` overrides remain supported. The adapter excludes
`DRISHTI_SERVER_` settings from the shared config loader. Submitted profiles cannot
override the server's bundle output root.

## API

- `GET /api/health`: local server status.
- `GET /api/doctor`: measured core environment probe and application settings.
  Capability/package detection is explicitly separate from algorithm execution.
- `GET /api/runs`: on-disk bundles and in-process queued/running job state.
- `GET /api/runs/{id}/manifest`: root manifest dict, including real stage statuses.
- `GET /api/runs/{id}/report`: real report JSON; 404 until available. Partial reports
  use the recorded report path or the shared failure-path `report/report.json`.
- `GET /api/runs/{id}/scene`: recorded CRS, S6 (or S2) projected origin, export
  georeference/axis metadata, stage status/output-hash revisions and available
  artifact paths. No invented scene. Revisions let the viewer reload changed
  content at stable artifact paths after a resume.
  `semantic_layers` contains validated S9 optional class-layer descriptors;
  `semantic_reason` explains disabled/unavailable/partial product availability.
  Invalid semantic metadata leaves mesh/cloud inspection available. S9 output
  hashes govern imagery refresh independently of the geometry revision.
- `GET /api/runs/{id}/files/{path}`: bundle-relative file response. Absolute paths,
  traversal, and symlinks outside a bundle are rejected, including run/manifest,
  sidecar and raster paths. Internal symlinks remain usable.
  Add `?download=true` for original-file attachment responses, including class
  GeoTIFF downloads; MIME types and cache revalidation are retained.
- `GET /api/runs/{id}/preview?limit=N`: actual decimated local points, RGB,
  confidence and inferred flags. ASCII and little/big-endian binary PLY vertices
  are streamed; NPZ arrays are unpacked only under the memory limit. Missing
  attributes are null. Bounds describe the preview, not the complete cloud.
- `POST /api/runs` with `{"dataset":"path.yaml","profile":"balanced"}`:
  validates the real typed descriptor, supplied files and absolute-altitude
  declaration; creates a real core `Bundle`; returns **202** and the real run id;
  invokes `StageRunner.run()` in a background executor. Stage/report failures are
  retained by the shared runner. Job states are in-process, manifests persistent.
- `POST /api/runs/{id}/measure`: metric definitions below.

Descriptors resolve within `DATASET_ROOT`, including symlinks. The dataset's input
files resolve according to the shared descriptor contract. The OpenAPI schema is
available locally at `/docs` and `/openapi.json`.

## Measurement definitions

```json
{
  "kind": "volume",
  "points": [[0,0,0],[10,0,0],[10,10,0],[0,10,0]],
  "reference_height": 0
}
```

All input points are **local projected XYZ metres**, before adding the recorded
S6 origin. Numeric inputs must be finite (booleans and numeric strings rejected).

- **distance**: sum of XYZ segment lengths along the picked polyline.
- **area**: horizontal XY polygon area, not sloped surface area.
- **height**: maximum minus minimum local Z among picked points.
- **volume**: sum of `max(DSM_local_z - reference_height, 0) × polygon/cell overlap
  area`. The plane is an explicit **local Z** reference. GeoTIFF projected-absolute
  elevations have the recorded origin Z subtracted first. Cell elevation is a
  piecewise-constant DSM approximation; boundary cells use polygon overlap area,
  not center inclusion. Projected metre CRS and aligned support/provenance grids
  are required. S9 mask `0=unsupported,1=observed,2=inferred-only` and optional
  support counts/confidence are respected. DSM nodata and inferred cells never
  contribute. If the mask is absent, explicit `inferred_excluded=true` provenance
  is required; pre-excluded inferred area remains null rather than a made-up zero.

Polygons may be concave or explicitly closed. Repeated XY vertices, zero area,
self-intersection, non-adjacent touching, and overlapping edges are rejected.
Missing DSM, non-inferred provenance, origin, grid alignment, or observed support
returns an actionable **409**. Missing volume reference returns **422**.

Results carry units, method, support/coverage, unsupported/nodata/inferred area,
confidence provenance and report validation evidence. Absolute measurement
uncertainty is null: no independent error propagation model is available. Point
confidence, GPS-fit residuals and picked coordinates never become absolute accuracy.
Non-volume picks report `surface_support=not_verified`; volume reports only the
observed support it integrated, not full polygon or full scene completeness.

## Verification scope

`tests/test_server_api.py` uses explicitly constructed unit/math fixtures and a
mocked job boundary. It checks confinement, API contracts, finite/polygon
validation, raster integration, provenance and preview parsing. These are not
reconstruction acceptance, real-data accuracy or visual-placement validation.

`tests/test_semantic_api.py` checks optional descriptor/file validation, Unicode
class identity, S9 revision refresh, partial-product downloads and isolation from
mesh/cloud availability. These constructed contracts do not establish semantic
accuracy or actual model execution.
