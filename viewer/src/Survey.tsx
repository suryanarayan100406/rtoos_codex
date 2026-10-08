import { useEffect, useMemo, useRef, useState } from "react";
import {
  Axis, BoundingSphere, Cartesian2, Cartesian3, Cesium3DTileset, Color,
  EllipsoidTerrainProvider, HeadingPitchRange, Matrix4, Model, PointPrimitiveCollection,
  Rectangle, ScreenSpaceEventHandler, ScreenSpaceEventType, Viewer,
} from "cesium";
import { api, fileURL } from "./api";
import { axesToLocal, coordinates } from "./coordinates";
import { geometryKey, mountSemanticImagery, semanticLayer } from "./sceneLayers";
import type { Kind, Layers, Preview, Scene, SemanticLayer, XYZ } from "./types";

type Props = { runId: string; scene: Scene; layers: Layers; picks: XYZ[]; kind: Kind;
  onPick: (point: XYZ) => void; onPreview: (preview: Preview | null) => void };

export default function Survey({ runId, scene, layers, picks, kind, onPick, onPreview }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<Viewer>();
  const model = useRef<Model>();
  const tiles = useRef<Cesium3DTileset>();
  const cloud = useRef<PointPrimitiveCollection>();
  const data = useRef<Preview>();
  const transform = useRef<ReturnType<typeof coordinates>>();
  const callbacks = useRef({ onPick, onPreview });
  callbacks.current = { onPick, onPreview };
  const activeLayers = useRef(layers);
  activeLayers.current = layers;
  const [messages, setMessages] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [semanticNotice, setSemanticNotice] = useState("");
  const geometryRevision = geometryKey(scene);
  const geometryScene = useMemo(() => JSON.parse(geometryRevision) as Scene, [geometryRevision]);
  const semanticRevision = JSON.stringify({ layer: semanticLayer(scene), revision: scene.revision?.s9_geo });
  const semantic = useMemo(() => JSON.parse(semanticRevision) as {
    layer?: SemanticLayer; revision?: string | null;
  }, [semanticRevision]);
  const stageNotices = Object.entries(scene.stage_status || {})
    .filter(([, status]) => status === "stale" || status === "failed")
    .map(([name, status]) => `${name}: ${status}. Retained artifacts are not a fresh reconstruction.`);

  useEffect(() => {
    let alive = true;
    const abort = new AbortController();
    setMessages([]);
    setLoading(true);
    callbacks.current.onPreview(null);
    const message = (text: string) => { if (alive) setMessages(m => [...m, text]); };
    const coords = (() => {
      try { return coordinates(geometryScene); }
      catch (error) { message(`Geometry placement unavailable: ${String(error)}`); return undefined; }
    })();
    transform.current = coords;
    let v: Viewer;
    try { v = new Viewer(host.current!, {
      baseLayer: false, terrainProvider: new EllipsoidTerrainProvider(),
      geocoder: false, baseLayerPicker: false, animation: false, timeline: false,
      navigationHelpButton: false, homeButton: false, sceneModePicker: false,
      selectionIndicator: false, infoBox: false, fullscreenButton: false,
      skyBox: false, skyAtmosphere: false, requestRenderMode: true,
    }); } catch (error) {
      message(`Cesium / WebGL unavailable: ${String(error)}`);
      setLoading(false); return;
    }
    viewer.current = v;
    v.scene.backgroundColor = Color.fromCssColorString("#0b111b");
    v.scene.globe.baseColor = Color.fromCssColorString("#172332");
    v.scene.globe.showGroundAtmosphere = false;
    if (coords) {
      v.camera.lookAt(coords.toFixed([0, 0, 0]), new HeadingPitchRange(0, -0.7, 150));
      v.camera.lookAtTransform(Matrix4.IDENTITY);
    }
    const handler = new ScreenSpaceEventHandler(v.scene.canvas);
    handler.setInputAction((event: { position: Cartesian2 }) => {
      if (!coords) return; // Geographic imagery alone supplies no local measurement geometry.
      const picked = v.scene.pick(event.position);
      if (!picked) return;
      if (picked.id?.drishtiLocal) {
        callbacks.current.onPick(picked.id.drishtiLocal as XYZ);
        return;
      }
      const meshPick = !!model.current && picked.primitive === model.current;
      const tilePick = !!tiles.current && (picked.primitive === tiles.current || picked.content?.tileset === tiles.current);
      if (!meshPick && !tilePick) return; // Globe/imagery and annotation picks are not geometry.
      if (!v.scene.pickPositionSupported) { message("This browser cannot depth-pick the mesh."); return; }
      const fixed = v.scene.pickPosition(event.position);
      if (!fixed) return;
      const point = meshPick ? coords.fromMeshPick(fixed) : coords.fromGeographicPick(fixed);
      if (point.every(Number.isFinite)) callbacks.current.onPick(point);
    }, ScreenSpaceEventType.LEFT_CLICK);

    async function load() {
      if (!coords) { setLoading(false); return; }
      const formats = geometryScene.exports.formats || {};
      const meshFormat = formats.glb?.status === "ok" ? formats.glb : formats.gltf;
      const operations: Promise<void>[] = [];
      if (meshFormat?.status === "ok" && meshFormat.path) {
        operations.push((async () => {
          try {
            const axis = axesToLocal(geometryScene.crs.glb_axes);
            const loaded = await Model.fromGltfAsync({
              url: fileURL(runId, meshFormat.path!, geometryScene.revision?.s10_export),
              modelMatrix: Matrix4.multiply(coords.localToFixed, axis, new Matrix4()),
              upAxis: Axis.Z, forwardAxis: Axis.X,
              allowPicking: true, minimumPixelSize: 0,
            });
            if (!alive) { loaded.destroy(); return; }
            loaded.show = activeLayers.current.mesh;
            model.current = v.scene.primitives.add(loaded);
            loaded.readyEvent.addEventListener(() => {
              if (!alive) return;
              v.camera.flyToBoundingSphere(loaded.boundingSphere, { duration: 0.7 });
              v.scene.requestRender();
            });
          } catch (error) { message(`Mesh: ${String(error)}`); }
        })());
      } else message("No successful GLB/glTF export is available.");
      if (formats["3dtiles"]?.status === "ok" && formats["3dtiles"].path) {
        operations.push((async () => {
          try {
            const loaded = await Cesium3DTileset.fromUrl(fileURL(runId, formats["3dtiles"].path!, geometryScene.revision?.s10_export));
            if (!alive) { loaded.destroy(); return; }
            loaded.show = activeLayers.current.tiles;
            tiles.current = v.scene.primitives.add(loaded);
          } catch (error) { message(`3D Tiles: ${String(error)}`); }
        })());
      }
      if (geometryScene.artifacts.ply || geometryScene.artifacts.cloud) {
        operations.push((async () => {
          try {
            const preview = await api<Preview>(`/runs/${encodeURIComponent(runId)}/preview`, { signal: abort.signal });
            if (!alive) return;
            data.current = preview;
            const collection = new PointPrimitiveCollection();
            collection.show = activeLayers.current.points;
            cloud.current = v.scene.primitives.add(collection);
            for (let index = 0; index < preview.points.length; index++) {
              const point = preview.points[index];
              const rgb = preview.colors?.[index];
              collection.add({ position: coords.toFixed(point), pixelSize: 3,
                color: rgb ? Color.fromBytes(rgb[0], rgb[1], rgb[2]) : Color.LIGHTGRAY,
                id: { drishtiLocal: point, index }, show: true });
            }
            callbacks.current.onPreview(preview);
            if (preview.points.length && !model.current) {
              const bounds = BoundingSphere.fromPoints(preview.points.map(coords.toFixed));
              v.camera.flyToBoundingSphere(bounds, { duration: 0.7 });
            }
            if (!preview.inferred) message("Cloud inference flags are absent; observed-only filtering hides unknown points.");
            if (!preview.confidence) message("Cloud confidence is absent; no confidence values are assumed.");
            v.scene.requestRender();
          } catch (error) { if (alive) message(`Point preview: ${String(error)}`); }
        })());
      } else message("No dense cloud is available for point preview.");
      await Promise.all(operations);
      if (alive) { setLoading(false); v.scene.requestRender(); }
    }
    void load();
    return () => {
      alive = false; abort.abort(); handler.destroy(); v.destroy();
      viewer.current = undefined; model.current = undefined; cloud.current = undefined;
      tiles.current = undefined; data.current = undefined; transform.current = undefined;
    };
  }, [runId, geometryScene]);

  useEffect(() => {
    // Geographic imagery has its own lifecycle; toggling it never rebuilds geometry.
    const v = viewer.current;
    setSemanticNotice("");
    if (!layers.semantic || !v || !semantic.layer) return;
    const cleanup = mountSemanticImagery(v, runId, semantic.layer, semantic.revision, setSemanticNotice);
    if (!transform.current && semantic.layer.status === "ok" && semantic.layer.preview) {
      v.camera.setView({ destination: Rectangle.fromDegrees(...semantic.layer.preview.bounds) });
    }
    return cleanup;
  }, [runId, geometryScene, semantic, layers.semantic]);

  useEffect(() => {
    if (model.current) model.current.show = layers.mesh;
    if (tiles.current) tiles.current.show = layers.tiles;
    const collection = cloud.current, preview = data.current;
    if (collection && preview) {
      collection.show = layers.points;
      for (let i = 0; i < collection.length; i++) {
        const point = collection.get(i);
        const inferred = preview.inferred?.[i];
        const confidence = preview.confidence?.[i];
        point.show = (layers.inferred || inferred === false) &&
          (layers.minConfidence === 0 || (confidence != null && confidence >= layers.minConfidence));
        if (layers.confidence) {
          point.color = confidence == null ? Color.GRAY : Color.fromHsl(
            Math.max(0, Math.min(1, confidence)) * 0.36, 0.8, 0.55);
        } else if (inferred === true) {
          point.color = Color.fromCssColorString("#d796ff");
        } else {
          const rgb = preview.colors?.[i];
          point.color = rgb ? Color.fromBytes(rgb[0], rgb[1], rgb[2]) : Color.LIGHTGRAY;
        }
      }
    }
    viewer.current?.scene.requestRender();
  }, [layers, loading]);

  useEffect(() => {
    const v = viewer.current, coords = transform.current;
    if (!v || !coords) return;
    v.entities.removeAll();
    const positions = picks.map(coords.toFixed);
    for (let i = 0; i < positions.length; i++) v.entities.add({
      position: positions[i], point: { pixelSize: 9, color: Color.fromCssColorString("#62ebbd"),
        outlineColor: Color.BLACK, outlineWidth: 2 },
      label: { text: String(i + 1), font: "12px monospace", pixelOffset: new Cartesian2(0, -20),
        fillColor: Color.WHITE, showBackground: true },
    });
    if (positions.length > 1) v.entities.add({ polyline: {
      positions: (kind === "area" || kind === "volume") && positions.length >= 3
        ? [...positions, positions[0]] : positions,
      width: 2, material: Color.fromCssColorString("#62ebbd"),
    } });
    v.scene.requestRender();
  }, [picks, kind, loading]);

  return <div className="survey"><div className="cesium-host" ref={host} />
    {loading && <div className="scene-note">Loading local reconstruction…</div>}
    {semanticNotice && <div className="semantic-scene-note" role="status">{semanticNotice}</div>}
    {messages.length + stageNotices.length > 0 && <details className="scene-messages"><summary>Scene availability · {messages.length + stageNotices.length} notices</summary>
      {[...stageNotices, ...messages].map((text, i) => <p key={i}>{text}</p>)}</details>}
    <div className="scene-foot"><span>LOCAL / EPSG:{scene.crs.epsg ?? "unknown"}</span>
      <span>Click reconstructed geometry to add a vertex</span></div>
  </div>;
}
