import proj4 from "proj4";
import { Cartesian3, Matrix3, Matrix4 } from "cesium";
import type { Scene, XYZ } from "./types";

export function coordinates(scene: Scene) {
  const epsg = scene.crs.epsg;
  const origin = scene.origin;
  if (!origin || origin.length !== 3 || !origin.every(Number.isFinite) || !epsg) {
    throw new Error("No recorded EPSG and finite local origin are available.");
  }
  const code = `EPSG:${epsg}`;
  if (scene.crs.proj4) {
    proj4.defs(code, scene.crs.proj4);
  } else if (epsg >= 32601 && epsg <= 32660) {
    proj4.defs(code, `+proj=utm +zone=${epsg - 32600} +datum=WGS84 +units=m +no_defs`);
  } else if (epsg >= 32701 && epsg <= 32760) {
    proj4.defs(code, `+proj=utm +zone=${epsg - 32700} +south +datum=WGS84 +units=m +no_defs`);
  } else if (epsg === 3857) {
    // proj4 ships this definition. Other projected CRSs need their recorded definition.
  } else if (scene.crs.viewer_wkt1 || scene.crs.wkt) {
    proj4.defs(code, scene.crs.viewer_wkt1 || scene.crs.wkt!);
  } else if (!proj4.defs(code)) {
    throw new Error(`EPSG:${epsg} requires a recorded proj4/WKT definition for offline transformation.`);
  }
  const toFixed = (p: XYZ) => {
    const [lon, lat] = proj4(code, "EPSG:4326", [origin[0] + p[0], origin[1] + p[1]]);
    return Cartesian3.fromDegrees(lon, lat, origin[2] + p[2]);
  };
  const toLocal = (p: Cartesian3): XYZ => {
    // The inverse affine is used only for glTF pick coordinates below; cloud picks retain originals.
    return [p.x, p.y, p.z];
  };
  const fixedOrigin = toFixed([0, 0, 0]);
  const basis = (p: XYZ) => {
    const v = Cartesian3.subtract(toFixed(p), fixedOrigin, new Cartesian3());
    return [v.x, v.y, v.z];
  };
  const rotation = Matrix3.fromColumnMajorArray([
    ...basis([1, 0, 0]), ...basis([0, 1, 0]), ...basis([0, 0, 1]),
  ]);
  const localToFixed = Matrix4.fromRotationTranslation(rotation, fixedOrigin);
  const fixedToLocal = Matrix4.inverse(localToFixed, new Matrix4());
  const fromMeshPick = (fixed: Cartesian3): XYZ => toLocal(
    Matrix4.multiplyByPoint(fixedToLocal, fixed, new Cartesian3()));
  const fromGeographicPick = (fixed: Cartesian3): XYZ => {
    // Use exact projection for georeferenced tiles, not the mesh's local tangent approximation.
    const cartographic = Cartesian3ToGeographic(fixed);
    const [x, y] = proj4("EPSG:4326", code, [cartographic[0], cartographic[1]]);
    return [x - origin[0], y - origin[1], cartographic[2] - origin[2]];
  };
  return { toFixed, localToFixed, fromMeshPick, fromGeographicPick };
}

import { Cartographic, Math as CesiumMath } from "cesium";
function Cartesian3ToGeographic(p: Cartesian3): XYZ {
  const c = Cartographic.fromCartesian(p);
  return [CesiumMath.toDegrees(c.longitude), CesiumMath.toDegrees(c.latitude), c.height];
}

export function axesToLocal(axes: unknown): Matrix4 {
  // Cesium is configured with Z-up / X-forward so it adds no implicit axis rotation.
  if (axes === "local_xyz_z_up" || axes === "projected_local_xyz" || axes === "Z_UP" ||
      axes === "z_up" || axes === "x_east_y_north_z_up") return Matrix4.clone(Matrix4.IDENTITY);
  if (axes === "x_east_y_up_z_south" || axes === "X=east,Y=up,Z=south") {
    return Matrix4.fromRotationTranslation(Matrix3.fromColumnMajorArray([1, 0, 0, 0, 0, 1, 0, -1, 0]));
  }
  if (axes && typeof axes === "object") {
    const a = axes as Record<string, unknown>;
    const matrix = a.gltf_to_local ?? a.matrix;
    if (Array.isArray(matrix) && matrix.length === 16 && matrix.every(Number.isFinite)) {
      return Matrix4.fromRowMajorArray(matrix);
    }
  }
  throw new Error("The export's glb_axes sidecar is missing or unsupported; mesh placement is unavailable.");
}
