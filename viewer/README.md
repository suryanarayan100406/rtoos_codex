# DRISHTI offline survey console

```bash
npm install
npm run typecheck
npm run test:coordinates
npm run test:semantics
npm run build
```

Serve `dist/` through `python -m server.app` from the checkout root. For development,
run the API on `127.0.0.1:8000`, then `npm run dev`. The Vite proxy keeps `/api`
same-origin. `VITE_POLL_MS` (default 5000) and `VITE_MAX_PICK_POINTS` (default 512)
configure UI polling and picked-vertex limits at build time.

The viewer uses **real CesiumJS**, with local Cesium assets/workers copied during
build. The globe has no Ion imagery, geocoder, terrain service or external basemap.
No CDN, external fonts, hardcoded reconstruction, runs or fabricated metrics.
The only network requests during normal operation are same-origin assets/API.
Pinned compatible Cesium engine/widgets/zip transitive versions are in package.json
and package-lock.json. The prebuilt Cesium ESM entry avoids thousands of source
module filesystem reads on Windows-mounted checkouts.

## Scene and coordinate contract

- The picker reads `/api/runs`, then the selected manifest, report and scene.
- Successful GLB/glTF exports load through `Cesium.Model.fromGltfAsync`.
- Export origin/EPSG and `glb_axes.gltf_to_local` from `exports.json` are honored,
  including the export's additional mesh recenter translation. The matrix is a
  **row-major 4×4** inverse of serialized glTF axes/recentering.
- `proj4` transforms the recorded projected origin to geographic coordinates.
  A local projected→ECEF Jacobian places the recentered mesh, respecting grid north
  and metre scale; picks invert the same affine transform to recover source local
  coordinates. This is local-affine globe rendering, not a full nonlinear warp of
  an arbitrarily large mesh. Dense points use the exact per-point projection.
- UTM EPSG 32601–32660/32701–32760 and built-in EPSG:3857 work offline. Other CRS
  definitions use recorded proj4 or a WKT1 browser definition derived by the local
  API from recorded WKT/EPSG using pyproj. Source WKT2/datum remains retained.
  Missing transforms display
  an actionable availability state rather than an invented position.
- Globe heights use the recorded source Z; the UI explicitly labels vertical datum
  and unvalidated globe elevation alignment if not ellipsoidal. No implicit geoid
  correction. Metric measurements use original local Z, preserving source datum.
- Optional successful 3D Tiles are loaded through `Cesium3DTileset.fromUrl`.
- Dense PLY/NPZ output is converted by the local API into bounded actual point
  arrays. Preview counts/decimation are shown; NPZ is never sent to the browser as
  if it were directly renderable. Point color/confidence/inferred values are real
  attributes, absent values remain unavailable. Purple identifies inferred points.

Layer controls operate on mesh, bounded cloud and optional tiles. Confidence
coloring, inferred exclusion and minimum-confidence filtering operate on the
cloud; the UI labels their scope. Mesh/tiles are complete exported geometry.
Run status, stage timings/confidence/degradation, report evidence and actual
deliverables are read from the bundle. No check data means accuracy UNVALIDATED.
Recorded stage output hashes refresh changed content at stable artifact paths;
stale retained outputs are identified, and measurements reset on scene revision.

## Optional semantic overlay

When S9 supplies a validated semantic descriptor, the Cesium panel offers a class
legend, original class-GeoTIFF download, checkpoint/probability provenance and a
selectable bounded EPSG:4326 preview. The preview has already been geographically
warped with nearest-neighbor sampling; it is displayed as imagery on the offline
ellipsoid. It is not surface semantics or measurement geometry. Semantic quality
is UNVALIDATED and source probabilities are uncalibrated, independently of cloud
geometry confidence.

Missing previews keep valid raster downloads/legends available. Invalid optional
metadata does not block geometry. S9 revisions refresh imagery without rebuilding
unchanged geometry; toggles, run changes, late image loads and unmounts dispose of
the previous imagery layer. The overlay is available in the Cesium tab.

## Measurements

Click reconstructed model/cloud/tiles, never empty globe terrain, to pick vertices.
Cloud picks preserve exact source local coordinates; model picks invert placement;
tile picks use geographic→projected transformation. Choose distance, horizontal
polygon area, picked-point Z range (height), or DSM volume. Volume requires an
explicit local Z reference plane. The API validates the polygon and computes the
result with support/coverage/inferred exclusion and uncertainty labels. Undo/Clear
keep the input explicit. See `server/README.md` for mathematical definitions.

## Optional Potree

The Potree tab invokes a **real local Potree adapter** only when a vendored asset
manifest and a successful octree export are present. See
`public/vendor/potree/README.md`. Missing assets are shown explicitly, and the
functional Cesium point preview remains available. No fake Potree UI or CDN.

Typecheck/build verify frontend compilation. Rendering, real georeferenced product
placement and measurement acceptance still require a supplied real bundle and
browser/WebGL environment; those checks cannot be claimed from compilation.

`npm run test:coordinates` checks isolated EPSG/axis/recentering/pick-inverse math,
not scene acceptance. Set `DRISHTI_PYTHON` to a Python with pyproj to additionally
check the export's real WKT2 definition format. `scripts/smoke.cjs` can use an
existing local Playwright (`PLAYWRIGHT_MODULE`) and Python (`DRISHTI_PYTHON`) to
launch `python -m server.app`, verify the built empty-state console/run dialog/
doctor, responsive mobile width, and absence of external requests/page errors.
It downloads nothing and uses no fabricated reconstruction. Real-data visual
placement and actual local Potree assets remain separate validation requirements.

`npm run test:semantics` exercises isolated React/Cesium boundaries for legends,
downloads, unavailable products, image loading, revision/toggle cleanup races and
geometry-only measurement picks. It does not launch a real WebGL browser or verify
mission/checkpoint semantic quality.
