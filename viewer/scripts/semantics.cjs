/* Constructed UI/lifecycle units, NOT browser rendering or semantic accuracy evidence.
   Real React server rendering and Cesium coordinate/filter types; only WebGL/image I/O
   and the effect scheduler are isolated. No browser or network dependencies. */
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve, dirname } = require("node:path");
const Module = require("node:module");
const ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const cesium = require("cesium");

const pending = [], viewers = [], imageryInstances = [];
class Imagery {
  constructor(provider, options) {
    this.provider = provider; this.options = options; this.destroyed = false;
    imageryInstances.push(this);
  }
  isDestroyed() { return this.destroyed; }
  destroy() { assert(!this.destroyed, "Imagery is destroyed exactly once"); this.destroyed = true; }
}
class FakeViewer {
  constructor(host, options) {
    this.options = options; this.destroyed = false; this.depthPicks = 0;
    this.imageryLayers = {
      items: [],
      add: layer => { assert(!this.destroyed); this.imageryLayers.items.push(layer); },
      remove: (layer, destroy) => {
        assert(!this.destroyed);
        const index = this.imageryLayers.items.indexOf(layer);
        if (index === -1) return false;
        this.imageryLayers.items.splice(index, 1);
        if (destroy) layer.destroy();
        return true;
      },
    };
    this.scene = {
      globe: {}, canvas: {}, primitives: { add: value => value }, requestRender: () => {},
      pick: () => this.picked, pickPositionSupported: true,
      pickPosition: () => { this.depthPicks++; return this.position; },
    };
    this.camera = { lookAt: () => {}, lookAtTransform: () => {}, flyToBoundingSphere: () => {},
      setView: options => { this.destination = options.destination; } };
    this.entities = { removeAll: () => {}, add: () => {} };
    viewers.push(this);
  }
  isDestroyed() { return this.destroyed; }
  destroy() {
    assert(!this.destroyed);
    this.imageryLayers.items.forEach(layer => layer.destroy());
    this.imageryLayers.items = [];
    this.destroyed = true;
  }
}
class Handler {
  setInputAction(callback) { viewers.at(-1).click = callback; }
  destroy() {}
}
const fakeCesium = { ...cesium, Viewer: FakeViewer, ImageryLayer: Imagery,
  ScreenSpaceEventHandler: Handler,
  SingleTileImageryProvider: { fromUrl: (url, options) => new Promise((resolve, reject) => {
    pending.push({ url, options, resolve, reject });
  }) },
  Model: { fromGltfAsync: async () => ({ show: true, readyEvent: { addEventListener: () => {} }, destroy: () => {} }) },
};

// Commit all previous cleanups before creating the next effects, matching React.
let slots = [], cursor = 0, scheduled = [];
const sameDeps = (a, b) => a && b && a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
const hooks = {
  useRef: value => { const i = cursor++; return slots[i] ||= { current: value }; },
  useState: initial => {
    const i = cursor++;
    slots[i] ||= { value: initial };
    return [slots[i].value, value => { slots[i].value = typeof value === "function" ? value(slots[i].value) : value; }];
  },
  useMemo: (create, deps) => {
    const i = cursor++;
    if (!sameDeps(slots[i]?.deps, deps)) slots[i] = { value: create(), deps };
    return slots[i].value;
  },
  useEffect: (create, deps) => {
    const i = cursor++;
    if (!sameDeps(slots[i]?.deps, deps)) scheduled.push({ i, create, deps });
  },
};
const loaded = new Map();
function loadSource(path) {
  if (loaded.has(path)) return loaded.get(path).exports;
  const compiled = new Module(path, module);
  compiled.filename = path;
  compiled.paths = Module._nodeModulePaths(dirname(path));
  const originalRequire = compiled.require.bind(compiled);
  compiled.require = name => {
    if (name === "cesium") return fakeCesium;
    if (name === "react" && path.endsWith("Survey.tsx")) return hooks;
    if (name.startsWith("./")) {
      const extension = name === "./Survey" || name === "./SemanticPanel" ? ".tsx" : ".ts";
      return loadSource(resolve(dirname(path), name + extension));
    }
    return originalRequire(name);
  };
  loaded.set(path, compiled);
  compiled._compile(ts.transpileModule(readFileSync(path, "utf8"), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
    jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true,
  } }).outputText, path);
  return compiled.exports;
}
const src = name => resolve(__dirname, "../src", name);
const { default: Survey } = loadSource(src("Survey.tsx"));
const { default: SemanticPanel } = loadSource(src("SemanticPanel.tsx"));
const { geometryKey } = loadSource(src("sceneLayers.ts"));
const { fileURL, downloadURL } = loadSource(src("api.ts"));
const { coordinates } = loadSource(src("coordinates.ts"));
function renderSurvey(props) {
  cursor = 0; scheduled = [];
  Survey(props);
  scheduled.forEach(({ i }) => slots[i]?.cleanup?.());
  scheduled.forEach(({ i, create, deps }) => { slots[i] = { deps, cleanup: create() }; });
}
function unmount() { slots.forEach(slot => slot?.cleanup?.()); slots = []; }
const settle = () => new Promise(resolve => setImmediate(resolve));
const provider = { tileWidth: 2, tileHeight: 2 };
const descriptor = {
  id: "semantic", title: "Semantic classes", status: "ok", reason: null,
  classes: [{ id: 1, name: "Building", color: [200, 80, 30], source_ids: [2] }],
  nodata: 0, supported_cells: 3, accuracy_status: "UNVALIDATED",
  confidence_method: "winning_source_softmax_probability_uncalibrated", provenance: {},
  raster_path: "s9_geo/classes.tif", confidence_path: "s9_geo/probability.tif",
  preview: { path: "s9_geo/preview.png", crs: "EPSG:4326", bounds: [74.9, 8.9, 75.1, 9.1], width: 2, height: 2 },
};
const scene = { coordinate_frame: "projected_local", origin: [500000, 1000000, 100],
  crs: { epsg: 32643, glb_axes: "local_xyz_z_up" },
  exports: { formats: { glb: { status: "ok", path: "s10_export/model.glb" } } }, artifacts: {},
  revision: { s9_geo: "geo-1", s10_export: "mesh-1" }, semantic_layers: [descriptor] };
const layers = { mesh: true, points: true, tiles: false, confidence: false,
  inferred: true, minConfidence: 0, semantic: true };
const picks = [];
const props = { runId: "unit-run", scene, layers, picks: [], kind: "distance",
  onPick: point => picks.push(point), onPreview: () => {} };
function panel(layer, extra = {}) {
  return renderToStaticMarkup(React.createElement(SemanticPanel, { runId: props.runId,
    scene: { ...scene, semantic_layers: layer ? [layer] : [] }, enabled: true,
    view: "cesium", onToggle: () => {}, ...extra }));
}

(async () => {
  assert.equal(fileURL("run id", "folder/preview image.png", "rev & 1"),
    "/api/runs/run%20id/files/folder/preview%20image.png?revision=rev%20%26%201");
  assert.equal(downloadURL("unit", "classes.tif", "rev"), "/api/runs/unit/files/classes.tif?revision=rev&download=true");
  assert.equal(downloadURL("unit", "classes.tif"), "/api/runs/unit/files/classes.tif?download=true");
  for (const path of ["../outside.tif", "/etc/passwd", "folder\\escape.png", "bad\x00.png"]) {
    assert.throws(() => downloadURL("unit", path), /Invalid bundle-relative/);
  }
  const markup = panel(descriptor);
  assert.match(markup, /Semantic quality UNVALIDATED/);
  assert.match(markup, /rgb\(200,80,30\)/);
  assert.match(markup, /Building/);
  assert.match(markup, /No data \/ unsupported/);
  assert.match(markup, /uncalibrated and independent of geometry confidence/);
  assert.match(markup, /excluded from measurement picks/);
  assert.match(markup, /revision=geo-1&amp;download=true/);
  assert(!markup.includes("disabled="));
  const partial = panel({ ...descriptor, preview: null, confidence_path: null });
  assert.match(partial, /disabled=""/);
  assert.match(partial, /Geographic preview unavailable/);
  assert.match(partial, /Download original class GeoTIFF/);
  const unavailable = panel({ ...descriptor, status: "unavailable", reason: "Checkpoint unavailable" });
  assert.match(unavailable, /Checkpoint unavailable/);
  assert(!unavailable.includes("download="));
  assert.match(panel(undefined), /no semantic descriptor/);
  assert.match(panel(descriptor, { view: "potree" }), /disabled=""/);
  assert(!panel(descriptor, { view: "potree" }).includes("checked="));

  renderSurvey(props);
  await settle();
  const v = viewers.at(-1);
  assert.equal(v.options.baseLayer, false);
  assert(v.options.terrainProvider instanceof cesium.EllipsoidTerrainProvider);
  for (const setting of ["geocoder", "baseLayerPicker", "skyBox", "skyAtmosphere"]) assert.equal(v.options[setting], false);
  assert.equal(pending.length, 1);
  assert.match(pending[0].url, /^\/api\/runs\/unit-run\/files\/.*\?revision=geo-1$/);
  const rect = pending[0].options.rectangle;
  assert(Math.abs(cesium.Math.toDegrees(rect.west) - 74.9) < 1e-10);
  assert(Math.abs(cesium.Math.toDegrees(rect.north) - 9.1) < 1e-10);
  pending.shift().resolve(provider);
  await settle();
  assert.equal(v.imageryLayers.items.length, 1);
  const image = v.imageryLayers.items[0];
  assert.equal(image.options.minificationFilter, cesium.TextureMinificationFilter.NEAREST);
  assert.equal(image.options.magnificationFilter, cesium.TextureMagnificationFilter.NEAREST);
  for (const picked of [undefined, {}, { primitive: image }, { id: { annotation: true } }]) {
    v.picked = picked; v.click({ position: {} });
  }
  assert.equal(picks.length, 0, "Ellipsoid/imagery/annotation clicks cannot measure");
  assert.equal(v.depthPicks, 0);
  v.picked = { primitive: slots.find(slot => slot?.current?.readyEvent).current };
  v.position = coordinates(scene).toFixed([0, 0, 0]);
  v.click({ position: {} });
  assert.equal(picks.length, 1, "Real model primitive remains pickable with imagery enabled");
  picks[0].forEach(value => assert(Math.abs(value) < 1e-6));

  renderSurvey({ ...props, layers: { ...layers, semantic: false } });
  assert(image.isDestroyed());
  assert.equal(v.imageryLayers.items.length, 0);
  assert.equal(viewers.length, 1, "Overlay toggles retain loaded geometry");
  renderSurvey(props);
  const late = pending.shift();
  renderSurvey({ ...props, layers: { ...layers, semantic: false } });
  late.resolve(provider); await settle();
  assert.equal(v.imageryLayers.items.length, 0, "Cancelled image load cannot re-enable imagery");

  renderSurvey(props);
  pending.shift().resolve(provider); await settle();
  const firstRevision = v.imageryLayers.items[0];
  const nextScene = { ...scene, revision: { ...scene.revision, s9_geo: "geo-2" },
    semantic_layers: [{ ...descriptor, classes: [{ ...descriptor.classes[0], color: [10, 20, 30] }] }] };
  assert.equal(geometryKey(nextScene), geometryKey(scene));
  renderSurvey({ ...props, scene: nextScene });
  assert(firstRevision.isDestroyed());
  assert.equal(viewers.length, 1, "S9 revisions preserve mesh/cloud viewer");
  const next = pending.shift(); assert.match(next.url, /revision=geo-2$/);
  next.resolve(provider); await settle();
  assert.equal(v.imageryLayers.items.length, 1);
  const partialScene = { ...nextScene, semantic_layers: [{ ...descriptor, preview: null }] };
  renderSurvey({ ...props, scene: partialScene });
  assert.equal(v.imageryLayers.items.length, 0);
  assert.equal(pending.length, 0, "A download-only product never requests imagery");
  for (const semantic_layers of [[], [{ ...descriptor, status: "unavailable", reason: "Projection failed" }],
    [{ ...descriptor, status: "disabled", reason: "Disabled in config" }]]) {
    renderSurvey({ ...props, scene: { ...nextScene, semantic_layers } });
    assert.equal(viewers.length, 1, "Unavailable/invalid optional descriptors retain geometry");
    assert.equal(pending.length, 0);
  }
  renderSurvey({ ...props, scene: nextScene });
  pending.shift().resolve(provider); await settle();
  renderSurvey({ ...props, runId: "other-run", scene: nextScene });
  assert(v.isDestroyed());
  const other = viewers.at(-1);
  assert.notEqual(other, v);
  const otherPending = pending.shift(); assert.match(otherPending.url, /\/other-run\/files\//);
  unmount();
  otherPending.resolve(provider); await settle();
  assert(other.isDestroyed());
  assert.equal(other.imageryLayers.items.length, 0, "Run/view/unmount cleanup rejects late completions");

  // Geographic raster inspection works even without a valid local geometry transform.
  renderSurvey({ ...props, scene: { ...scene, origin: null, crs: {}, exports: {} } });
  const rasterOnly = viewers.at(-1);
  assert(rasterOnly.destination instanceof cesium.Rectangle);
  pending.shift().resolve(provider); await settle();
  assert.equal(rasterOnly.imageryLayers.items.length, 1);
  rasterOnly.picked = { primitive: rasterOnly.imageryLayers.items[0] };
  rasterOnly.click({ position: {} });
  assert.equal(picks.length, 1);
  unmount();

  renderSurvey(props);
  pending.shift().reject(new Error("PNG unavailable")); await settle();
  assert.equal(viewers.at(-1).imageryLayers.items.length, 0);
  assert(slots.some(slot => typeof slot?.value === "string" && slot.value.includes("GeoTIFF remains downloadable")));
  unmount();
  renderSurvey(props);
  pending.shift().resolve({ tileWidth: 3, tileHeight: 2 }); await settle();
  assert.equal(viewers.at(-1).imageryLayers.items.length, 0);
  assert(slots.some(slot => typeof slot?.value === "string" && slot.value.includes("dimensions differ")));
  unmount();
  assert(imageryInstances.every(image => image.isDestroyed()));
  console.log("PASS: React legend/availability/downloads; offline Cesium options; EPSG:4326 nearest imagery; geometry-only picks; toggle/run/revision/unmount races; raster-only viewing; failed/mismatched previews");
})().catch(error => { console.error(error); process.exitCode = 1; });
