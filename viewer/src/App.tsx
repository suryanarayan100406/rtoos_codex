import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, fileURL } from "./api";
import Survey from "./Survey";
import PotreePanel from "./PotreePanel";
import SemanticPanel from "./SemanticPanel";
import type { Kind, Layers, Manifest, Measurement, Preview, Run, Scene, Stage, XYZ } from "./types";

const POLL_MS = Number(import.meta.env.VITE_POLL_MS || 5000);
const MAX_PICKS = Number(import.meta.env.VITE_MAX_PICK_POINTS || 512);
const number = (value: unknown, digits = 2) => typeof value === "number" && Number.isFinite(value)
  ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
const stagesOf = (manifest: Manifest | null): [string, Stage][] => {
  const stages = manifest?.stages || {};
  return Array.isArray(stages) ? stages.map(s => [s.name || "unnamed", s]) : Object.entries(stages);
};
const toolNames: Record<Kind, string> = { distance: "Distance", area: "Area", height: "Height", volume: "Volume" };
const definitions: Record<Kind, string> = {
  distance: "Polyline length in local XYZ metres. Pick two or more vertices.",
  area: "Horizontal XY polygon area. Pick at least three vertices in boundary order.",
  height: "Maximum minus minimum local Z among the picked points.",
  volume: "Supported, non-inferred DSM volume above an explicit local Z reference plane. Pick a polygon.",
};

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState("");
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [reportError, setReportError] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);
  const [layers, setLayers] = useState<Layers>({ mesh: true, points: true, tiles: false,
    confidence: false, inferred: true, minConfidence: 0, semantic: false });
  const [kind, setKind] = useState<Kind>("distance");
  const [picks, setPicks] = useState<XYZ[]>([]);
  const [reference, setReference] = useState("");
  const [measurement, setMeasurement] = useState<Measurement | null>(null);
  const [measuring, setMeasuring] = useState(false);
  const [view, setView] = useState<"cesium" | "potree">("cesium");
  const [newRun, setNewRun] = useState(false);
  const [dataset, setDataset] = useState("");
  const [profile, setProfile] = useState("balanced");
  const [starting, setStarting] = useState(false);
  const [doctor, setDoctor] = useState<Record<string, unknown> | null>(null);
  const [checking, setChecking] = useState(false);
  const activeMeasurementKey = useRef("");
  activeMeasurementKey.current = `${runId}:${kind}:${reference}:${JSON.stringify(picks)}:${JSON.stringify(scene)}`;

  const refresh = useCallback(async () => {
    try {
      const result = await api<{ runs: Run[] }>("/runs");
      setRuns(result.runs);
      setConnected(true);
      setRunId(current => current || result.runs[0]?.id || "");
    } catch (e) { setConnected(false); setError(String(e)); }
  }, []);
  useEffect(() => {
    void refresh();
    const interval = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    setManifest(null); setScene(null); setReport(null); setPreview(null); setReportError("");
    setPicks([]); setMeasurement(null); setError("");
    setLayers(current => ({ ...current, semantic: false }));
    if (!runId) return;
    const abort = new AbortController();
    const base = `/runs/${encodeURIComponent(runId)}`;
    const options = { signal: abort.signal };
    let generation = 0;
    const read = async () => {
      const token = ++generation;
      try {
        const [m, s] = await Promise.all([api<Manifest>(`${base}/manifest`, options), api<Scene>(`${base}/scene`, options)]);
        if (abort.signal.aborted || token !== generation) return;
        setManifest(m);
        setScene(previous => JSON.stringify(previous) === JSON.stringify(s) ? previous : s);
        try {
          const r = await api<Record<string, unknown>>(`${base}/report`, options);
          if (!abort.signal.aborted && token === generation) { setReport(r); setReportError(""); }
        } catch (e) {
          // Report generation may not have run yet. Keep the availability state explicit.
          if (!abort.signal.aborted && token === generation) {
            setReport(null);
            setReportError(e instanceof ApiError && e.status === 404 ? "" : String(e));
          }
        }
      } catch (e) { if (!abort.signal.aborted && token === generation) setError(String(e)); }
    };
    void read();
    const interval = setInterval(() => void read(), POLL_MS);
    return () => { abort.abort(); clearInterval(interval); };
  }, [runId]);

  useEffect(() => { setPicks([]); setMeasurement(null); }, [scene]);

  const stages = useMemo(() => stagesOf(manifest), [manifest]);
  const seconds = stages.reduce((sum, [, s]) => sum + (s.wall_seconds ?? 0), 0);
  const hasTiming = stages.some(([, s]) => typeof s.wall_seconds === "number");
  const done = stages.filter(([, s]) => ["done", "completed", "degraded", "ok"].includes(s.status || "")).length;
  const selected = runs.find(r => r.id === runId);
  const runError = manifest?.error || selected?.job?.error || selected?.error;
  const accuracy = report?.accuracy ?? report?.validation;
  const changeTool = (next: Kind) => { setKind(next); setPicks([]); setMeasurement(null); };
  const onPick = useCallback((point: XYZ) => {
    setMeasurement(null);
    setPicks(current => current.length >= MAX_PICKS ? current : [...current, point]);
  }, []);
  const toggle = (key: keyof Omit<Layers, "minConfidence">) =>
    setLayers(current => ({ ...current, [key]: !current[key] }));

  async function measure() {
    setMeasuring(true); setError(""); setMeasurement(null);
    const key = activeMeasurementKey.current;
    try {
      const localReference = reference.trim() === "" ? null : Number(reference);
      if (kind === "volume" && (localReference === null || !Number.isFinite(localReference))) {
        throw new Error("Enter an explicit finite local reference height in metres.");
      }
      const result = await api<Measurement>(`/runs/${encodeURIComponent(runId)}/measure`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, points: picks, reference_height: kind === "volume" ? localReference : null }),
      });
      if (key === activeMeasurementKey.current) setMeasurement(result);
    } catch (e) { if (key === activeMeasurementKey.current) setError(String(e)); }
    finally { setMeasuring(false); }
  }
  async function start(event: React.FormEvent) {
    event.preventDefault(); setStarting(true); setError("");
    try {
      const result = await api<{ id: string }>("/runs", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset, profile }) });
      await refresh(); setRunId(result.id); setNewRun(false);
    } catch (e) { setError(String(e)); } finally { setStarting(false); }
  }
  async function inspectDoctor() {
    setChecking(true); setError("");
    try { setDoctor(await api<Record<string, unknown>>("/doctor")); }
    catch (e) { setError(String(e)); } finally { setChecking(false); }
  }

  return <div className="app-shell">
    <header className="topbar"><a href="/" className="brand"><span className="brand-mark">◈</span>
      <span>DRISHTI<small>GROUND RECONSTRUCTION</small></span></a>
      <div className="run-selector"><label htmlFor="run">PROJECT BUNDLE</label>
        <select id="run" value={runId} onChange={e => setRunId(e.target.value)}>
          {runs.length === 0 && <option value="">No runs available</option>}
          {runs.map(run => <option key={run.id} value={run.id}>{run.id} · {run.job?.status || run.status || "unknown"}</option>)}
        </select></div>
      <div className="header-actions"><span className={`connection ${connected ? "online" : ""}`}>
        <i />{connected ? "LOCAL API" : "API UNAVAILABLE"}</span>
        <button className="quiet" onClick={() => void refresh()} title="Refresh bundle list">↻</button>
        <button className="primary" onClick={() => setNewRun(true)}>+ New run</button></div>
    </header>

    {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")}>Dismiss</button></div>}

    <div className="workspace"><aside className="left-panel">
      <div className="panel-title"><span className="eyebrow">SURVEY WORKSPACE</span><span className="pill">OFFLINE</span></div>
      <h1>{runId || "Your next survey"}</h1><p className="subtle">Recorded footage. Metric products. Traceable evidence.</p>
      <div className="status-line"><span className={`status-dot ${manifest?.status || ""}`} />
        {selected?.job?.status || manifest?.status || "Awaiting a project bundle"}</div>
      {runError && <p className="failure">{typeof runError === "string" ? runError : JSON.stringify(runError)}</p>}
      <div className="stat-grid"><div><strong>{manifest ? `${done}/${stages.length}` : "—"}</strong><span>Stages finished</span></div>
        <div><strong>{hasTiming ? `${number(seconds, 1)} s` : "—"}</strong><span>Recorded stage time</span></div>
        <div><strong>{number(preview?.total_points, 0)}</strong><span>Source cloud points</span></div>
        <div><strong>{scene?.crs.epsg ? `EPSG:${scene.crs.epsg}` : "—"}</strong><span>Recorded CRS</span></div></div>

      <section><div className="section-heading"><h2>Scene layers</h2><span>LOCAL ARTIFACTS</span></div>
        <label className="layer"><span><b>Textured mesh</b><small>GLB / glTF export</small></span>
          <input type="checkbox" checked={layers.mesh} onChange={() => toggle("mesh")} /></label>
        <label className="layer"><span><b>Point preview</b><small>{preview ? `${number(preview.preview_points, 0)} decimated points` : "Bounded real point arrays"}</small></span>
          <input type="checkbox" checked={layers.points} onChange={() => toggle("points")} /></label>
        <label className="layer"><span><b>3D Tiles</b><small>{scene?.exports.formats?.["3dtiles"]?.status || "No successful export"}</small></span>
          <input type="checkbox" checked={layers.tiles} onChange={() => toggle("tiles")} /></label>
        <div className="cloud-options"><label><input type="checkbox" checked={layers.confidence} onChange={() => toggle("confidence")} />
          Color points by confidence</label><label><input type="checkbox" checked={layers.inferred} onChange={() => toggle("inferred")} />
          Include inferred / unknown points</label><label className="threshold">Minimum point confidence <b>{layers.minConfidence.toFixed(2)}</b>
          <input type="range" min="0" max="1" step="0.05" value={layers.minConfidence}
            onChange={e => setLayers(current => ({ ...current, minConfidence: Number(e.target.value) }))} /></label>
          <div className="confidence-scale"><span>LOW</span><i /><span>HIGH</span></div>
          <p className="micro">Cloud filters only. Inferred points are purple. Mesh and tiles display their complete exported geometry.</p></div>
      </section>
      <SemanticPanel runId={runId} scene={scene} enabled={layers.semantic} view={view}
        onToggle={() => toggle("semantic")} />
      <section><div className="section-heading"><h2>Pipeline ledger</h2><span>{stages.length ? "ACTUAL EXECUTION" : "NO STAGES"}</span></div>
        <ol className="stage-list">{stages.map(([name, s]) => {
          const confidence = typeof s.confidence === "number" ? s.confidence : s.confidence?.summary;
          return <li key={name}><span className={`stage-marker ${s.status}`} /><div><b>{name.replace(/^s(\d+)_/, "S$1 · ").replaceAll("_", " ")}</b>
            <small>{s.status || "unknown"}{s.degraded_reason ? ` · ${s.degraded_reason}` : ""}</small></div>
            <span className="stage-metric">{number(s.wall_seconds, 1)} s<small>conf. {number(confidence)}</small></span></li>;
        })}</ol></section>
      <button className="wide quiet" onClick={() => void inspectDoctor()} disabled={checking}>
        {checking ? "Inspecting local runtime…" : "Inspect runtime / doctor"}</button>
    </aside>

    <main className="scene-panel"><div className="scene-toolbar"><div className="view-tabs">
      <button className={view === "cesium" ? "active" : ""} onClick={() => setView("cesium")}>Georeferenced scene</button>
      <button className={view === "potree" ? "active" : ""} onClick={() => setView("potree")}>Potree octree</button></div>
      <span className="micro">NO ION · NO EXTERNAL BASEMAP</span></div>
      {runId && scene ? view === "cesium" ? <Survey runId={runId} scene={scene} layers={layers} picks={picks}
        kind={kind} onPick={onPick} onPreview={setPreview} /> : <PotreePanel runId={runId} scene={scene} />
        : <div className="empty-scene"><div className="empty-symbol">◈</div><span className="eyebrow">A CLEAR VIEW, GROUNDED IN EVIDENCE</span>
          <h2>{runId ? "Reading the project bundle…" : "No reconstruction loaded"}</h2>
          <p>Select an existing bundle or start a run from a recorded-video dataset descriptor.
            Reconstructed geometry appears here when the pipeline produces real artifacts.</p>
          {!runId && <button className="primary" onClick={() => setNewRun(true)}>Start a local run</button>}</div>}
      <div className="datum-bar"><span>VERTICAL DATUM</span><b>{scene ? String(scene.crs.vertical || scene.crs.source_altitude_datum || "unvalidated") : "—"}</b>
        <small>Globe elevation alignment is unvalidated unless the recorded heights are ellipsoidal. No implicit geoid correction.</small></div>
    </main>

    <aside className="right-panel"><section><div className="section-heading"><h2>Measure the site</h2><span>METRIC LOCAL</span></div>
      <div className="tool-grid">{(Object.keys(toolNames) as Kind[]).map(tool => <button key={tool}
        className={kind === tool ? "active" : ""} onClick={() => changeTool(tool)}>
        <span>{({ distance: "↔", area: "▱", height: "↕", volume: "◇" })[tool]}</span>{toolNames[tool]}</button>)}</div>
      <p className="tool-definition">{definitions[kind]}</p>
      {kind === "volume" && <label className="field">Reference height · local Z (m)
        <input type="number" step="any" value={reference} onChange={e => { setReference(e.target.value); setMeasurement(null); }}
          placeholder="Explicit local height" /></label>}
      <div className="pick-heading"><b>{picks.length} picked vertices</b><div>
        <button className="quiet" disabled={!picks.length} onClick={() => { setPicks(p => p.slice(0, -1)); setMeasurement(null); }}>Undo</button>
        <button className="quiet" disabled={!picks.length} onClick={() => { setPicks([]); setMeasurement(null); }}>Clear</button></div></div>
      <div className="pick-list">{picks.map((point, i) => <div key={i}><span>{i + 1}</span>
        <code>{point.map(v => number(v, 3)).join(" / ")}</code></div>)}
        {!picks.length && <p className="micro">Pick model or cloud geometry in the scene. Blank globe clicks are ignored.</p>}</div>
      <button className="primary wide" disabled={!runId || measuring || picks.length < (kind === "area" || kind === "volume" ? 3 : 2)}
        onClick={() => void measure()}>{measuring ? "Computing…" : `Compute ${toolNames[kind].toLowerCase()}`}</button>
      {measurement && <div className="measurement-result"><span className="eyebrow">{measurement.kind.toUpperCase()}</span>
        <strong>{number(measurement.value, 3)} <small>{measurement.units}</small></strong>
        <p className="mono">{measurement.method.replaceAll("_", " ")}</p>
        {typeof measurement.support.coverage_fraction === "number" && <p>Supported polygon coverage: <b>{number(measurement.support.coverage_fraction * 100, 1)}%</b></p>}
        {typeof measurement.support.unsupported_area_m2 === "number" && <p>Unsupported area: <b>{number(measurement.support.unsupported_area_m2)} m²</b></p>}
        <span className="validation-badge">{measurement.uncertainty.status}</span>
        {measurement.warnings.map((warning, i) => <p className="micro" key={i}>{warning}</p>)}
        <details><summary>Support & uncertainty evidence</summary><pre>{JSON.stringify({ support: measurement.support, uncertainty: measurement.uncertainty }, null, 2)}</pre></details>
      </div>}
    </section>
    <section><div className="section-heading"><h2>Accuracy & evidence</h2><span>REPORT</span></div>
      <p className="micro">The ≤ 1 m requirement is a design target until independently measured. GPS alignment residuals do not validate absolute accuracy.</p>
      {report ? <><details className="report-details" open><summary>Recorded accuracy evidence</summary>
        <pre>{JSON.stringify(accuracy ?? { status: "UNVALIDATED", note: "No accuracy evidence field in report" }, null, 2)}</pre></details>
        <details className="report-details"><summary>Full report JSON</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
        <a className="text-link" href={`/api/runs/${encodeURIComponent(runId)}/report`} target="_blank" rel="noreferrer">Open report JSON ↗</a></>
        : <p className="unavailable">Accuracy UNVALIDATED · {reportError || "Report not available."}</p>}
    </section>
    <section><div className="section-heading"><h2>Deliverables</h2><span>BUNDLE FILES</span></div>
      <div className="deliverables">{Object.entries(scene?.exports.formats || {}).map(([name, format]) => <div key={name}>
        <b>{name.toUpperCase()}</b>{format.details?.products && (format.status === "ok" || name === "geotiff")
          ? <span className="product-links">{format.status !== "ok" && <span title={format.reason}>{format.status}</span>}
            {Object.entries(format.details.products).filter(([, path]) => typeof path === "string" && !!path).map(([product, path]) =>
              <a key={product} href={fileURL(runId, path)} download>{product.toUpperCase()} ↗</a>)}</span>
          : format.status === "ok" && format.path
          ? <a href={fileURL(runId, format.path)} download>File ↗</a>
          : <span title={format.reason}>{format.status || "unknown"}</span>}</div>)}
        {!scene?.exports.formats && <p className="micro">Exports appear after execution.</p>}</div>
    </section></aside></div>
    <footer className="bottom-bar"><span>DRISHTI / OFFLINE SURVEY CONSOLE</span>
      <span>Coordinates: projected local XYZ · metres · source confidence ≠ absolute accuracy</span></footer>

    {newRun && <div className="modal-backdrop"><form className="modal" onSubmit={event => void start(event)}>
      <div className="section-heading"><h2>Start a recorded-data run</h2><button type="button" className="quiet" onClick={() => setNewRun(false)}>Close</button></div>
      <p>Use an existing local YAML dataset descriptor. The API validates the inputs, then starts the real offline runner.</p>
      <label className="field">Dataset descriptor<input required value={dataset} onChange={e => setDataset(e.target.value)} placeholder="Path relative to configured dataset root" /></label>
      <label className="field">Processing profile<select value={profile} onChange={e => setProfile(e.target.value)}>
        <option value="fast">Fast</option><option value="balanced">Balanced</option><option value="max">Max</option></select></label>
      <button className="primary wide" disabled={starting}>{starting ? "Validating & queuing…" : "Start reconstruction"}</button>
    </form></div>}
    {doctor && <div className="modal-backdrop"><div className="modal doctor-modal"><div className="section-heading"><h2>Detected local environment</h2>
      <button className="quiet" onClick={() => setDoctor(null)}>Close</button></div><pre>{JSON.stringify(doctor, null, 2)}</pre></div></div>}
  </div>;
}
