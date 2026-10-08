import { downloadURL } from "./api";
import { hasSemanticPreview, semanticLayer } from "./sceneLayers";
import type { Scene } from "./types";

type Props = { runId: string; scene: Scene | null; enabled: boolean;
  view: "cesium" | "potree"; onToggle: () => void };

export default function SemanticPanel({ runId, scene, enabled, view, onToggle }: Props) {
  const layer = scene ? semanticLayer(scene) : undefined;
  const rasterAvailable = layer?.status === "ok" && !!layer.raster_path;
  const previewAvailable = hasSemanticPreview(layer);
  return <section><div className="section-heading"><h2>Semantic classes</h2><span>2D GEOGRAPHIC</span></div>
    <label className="layer"><span><b>Semantic overlay</b><small>EPSG:4326 imagery on the ellipsoid</small></span>
      <input type="checkbox" checked={enabled && previewAvailable && view === "cesium"}
        disabled={!previewAvailable || view !== "cesium"} onChange={onToggle} /></label>
    {rasterAvailable && layer ? <>
      <span className="validation-badge">Semantic quality UNVALIDATED</span>
      <p className="micro">2D geographic imagery is not surface semantics or a reconstructed mesh texture.
        It is excluded from measurement picks.</p>
      {view !== "cesium" && <p className="micro">Open the georeferenced scene to display this overlay.</p>}
      {(scene?.semantic_reason || layer.reason) &&
        <p className="unavailable">{scene?.semantic_reason || layer.reason}</p>}
      {!layer.preview && <p className="micro">Geographic preview unavailable. Download the original class raster for GIS inspection.</p>}
      <ul className="semantic-legend" aria-label="Semantic class legend">{layer.classes.map(item => <li key={item.id}>
        <i style={{ backgroundColor: `rgb(${item.color.join(",")})` }} aria-hidden="true" />
        <span>{item.name}</span><code>{item.id}</code></li>)}
        <li><i className="semantic-nodata" aria-hidden="true" /><span>No data / unsupported</span><code>{layer.nodata}</code></li>
      </ul>
      <p className="micro">{layer.supported_cells.toLocaleString()} supported raster cells.
        Native semantic probabilities are uncalibrated and independent of geometry confidence.</p>
      <details className="semantic-evidence"><summary>Semantic provenance & probability method</summary>
        <pre>{JSON.stringify({ confidence_method: layer.confidence_method,
          accuracy_status: layer.accuracy_status, provenance: layer.provenance }, null, 2)}</pre></details>
      <a className="text-link" href={downloadURL(runId, layer.raster_path!, scene?.revision?.s9_geo)} download>
        Download original class GeoTIFF ↗</a>
    </> : <p className="unavailable">{scene?.semantic_reason || layer?.reason ||
      (scene ? "This bundle has no semantic descriptor." : "No bundle loaded.")}</p>}
  </section>;
}
