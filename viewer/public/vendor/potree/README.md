# Optional, real local Potree adapter

No Potree implementation or external CDN is substituted. Install an audited local
Potree distribution **with its dependencies and workers** in this directory. Keep
its license notices. Potree's own BSD-2-Clause license does not cover every bundled
dependency: review the exact distribution. No assets are downloaded at runtime.

Add `assets.json` describing that distribution's script order and styles. Example
for the usual Potree 1.8 directory layout (adjust to the actual files you supply):

```json
{
  "pointBudget": 1000000,
  "styles": [
    "/vendor/potree/build/potree/potree.css",
    "/vendor/potree/libs/jquery-ui/jquery-ui.min.css",
    "/vendor/potree/libs/spectrum/spectrum.css"
  ],
  "scripts": [
    "/vendor/potree/libs/jquery/jquery-3.1.1.min.js",
    "/vendor/potree/libs/spectrum/spectrum.js",
    "/vendor/potree/libs/jquery-ui/jquery-ui.min.js",
    "/vendor/potree/libs/other/BinaryHeap.js",
    "/vendor/potree/libs/tween/tween.min.js",
    "/vendor/potree/libs/d3/d3.js",
    "/vendor/potree/libs/proj4/proj4.js",
    "/vendor/potree/libs/i18next/i18next.js",
    "/vendor/potree/libs/plasio/js/laslaz.js",
    "/vendor/potree/build/potree/potree.js"
  ]
}
```

The adapter calls the real `Potree.Viewer`, `Potree.loadPointCloud`,
`scene.addPointCloud`, and `fitToScreen`. The export contract is
`exports.json.formats.potree = {"status":"ok","path":"<bundle-relative cloud.js or metadata.json>"}`.
The exporter must retain all referenced octree files next to that metadata; the
API serves them with path and symlink confinement. The Potree view navigates the
cloud; measurements and provenance filters are in the Cesium view. Missing assets
or exports are visibly unavailable. The default Cesium preview is bounded and
explicitly labeled as decimated, with no fabricated points or confidence.
