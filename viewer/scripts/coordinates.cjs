/* Isolated projection/axis/pick math, NOT real reconstruction placement acceptance. */
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve, dirname } = require("node:path");
const { spawnSync } = require("node:child_process");
const Module = require("node:module");
const ts = require("typescript");
const { Cartesian3, Cartographic, Math: CesiumMath, Matrix4 } = require("cesium");
const source = resolve(__dirname, "../src/coordinates.ts");
const compiled = new Module(source, module);
compiled.filename = source;
compiled.paths = Module._nodeModulePaths(dirname(source));
compiled._compile(ts.transpileModule(readFileSync(source, "utf8"), { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
} }).outputText, source);
const { coordinates, axesToLocal } = compiled.exports;
const scene = { origin: [500000, 1000000, 100], crs: { epsg: 32643 }, exports: {}, artifacts: {} };
const transforms = coordinates(scene);
const origin = transforms.toFixed([0, 0, 0]);
const geo = Cartographic.fromCartesian(origin);
assert(Math.abs(CesiumMath.toDegrees(geo.longitude) - 75) < 1e-9);
assert(Math.abs(geo.height - 100) < 1e-6);
const local = [12.5, -7.25, 2];
const fixed = Matrix4.multiplyByPoint(transforms.localToFixed, Cartesian3.fromArray(local), new Cartesian3());
const inverse = transforms.fromMeshPick(fixed);
inverse.forEach((value, i) => assert(Math.abs(value - local[i]) < 1e-6));
const exact = transforms.fromGeographicPick(transforms.toFixed(local));
exact.forEach((value, i) => assert(Math.abs(value - local[i]) < 1e-6));
const sidecar = { gltf_to_local: [1, 0, 0, 10, 0, 0, -1, 20, 0, 1, 0, 30, 0, 0, 0, 1] };
const serialized = new Cartesian3(1, 2, 3);
const recovered = Matrix4.multiplyByPoint(axesToLocal(sidecar), serialized, new Cartesian3());
assert.deepEqual([recovered.x, recovered.y, recovered.z], [11, 17, 32]);
assert.throws(() => axesToLocal(undefined), /missing or unsupported/);
assert.throws(() => coordinates({ ...scene, origin: null }), /recorded EPSG/);
if (process.env.DRISHTI_PYTHON) {
  const processResult = spawnSync(process.env.DRISHTI_PYTHON, ["-c", "from pyproj import CRS; print(CRS.from_epsg(32643).to_wkt())"], { encoding: "utf8" });
  assert.equal(processResult.status, 0, processResult.stderr);
  const withWKT = coordinates({ ...scene, crs: { epsg: 32643, wkt: processResult.stdout.trim() } });
  assert(Cartesian3.distance(withWKT.toFixed(local), transforms.toFixed(local)) < 1e-6);
}
console.log("PASS: offline UTM projection, source-local mesh/tile pick inversion, row-major glTF recenter/axis sidecar, missing-transform rejection" +
  (process.env.DRISHTI_PYTHON ? ", recorded pyproj WKT2 compatibility" : ""));
