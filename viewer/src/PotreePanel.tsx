import { useEffect, useState } from "react";
import { fileURL } from "./api";
import type { Scene } from "./types";

export default function PotreePanel({ runId, scene }: { runId: string; scene: Scene }) {
  const [available, setAvailable] = useState<boolean | null>(null);
  useEffect(() => {
    const abort = new AbortController();
    fetch("/vendor/potree/assets.json", { signal: abort.signal }).then(async response => {
      if (!response.ok) throw new Error("missing");
      const manifest = await response.json();
      if (!Array.isArray(manifest.scripts) || !manifest.scripts.length) throw new Error("invalid");
      setAvailable(true);
    }).catch(() => { if (!abort.signal.aborted) setAvailable(false); });
    return () => abort.abort();
  }, []);
  const cloud = scene.exports.formats?.potree;
  if (available == null) return <div className="empty-scene"><h2>Checking local Potree assets…</h2></div>;
  if (!available) return <div className="empty-scene"><span className="eyebrow">OPTIONAL LOCAL ADAPTER</span>
    <h2>Potree assets are not installed</h2><p>The Cesium point preview uses real decimated cloud data.
      Full octree navigation requires a local Potree distribution and a successful Potree export.</p>
    <p className="mono">See viewer/public/vendor/potree/README.md</p></div>;
  if (cloud?.status !== "ok" || !cloud.path) return <div className="empty-scene"><h2>No Potree octree export</h2>
    <p>{cloud?.reason || "This bundle has no successful Potree export. Use the Cesium point preview."}</p></div>;
  return <iframe className="potree-frame" title="Local Potree point cloud"
    src={`/potree-adapter.html?cloud=${encodeURIComponent(fileURL(runId, cloud.path))}`} />;
}
