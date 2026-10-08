const status = document.getElementById("status");
try {
  const cloud = new URLSearchParams(location.search).get("cloud");
  const cloudURL = new URL(cloud || "", location.origin);
  if (cloudURL.origin !== location.origin || !cloudURL.pathname.startsWith("/api/runs/") ||
      !cloudURL.pathname.includes("/files/")) throw new Error("Invalid local cloud URL");
  const response = await fetch("/vendor/potree/assets.json");
  if (!response.ok) throw new Error("Local Potree asset manifest is missing");
  const assets = await response.json();
  if (!Number.isSafeInteger(assets.pointBudget) || assets.pointBudget < 1) {
    throw new Error("Local Potree assets.json requires a positive integer pointBudget");
  }
  const localAsset = value => {
    if (typeof value !== "string" || !value.startsWith("/vendor/potree/") ||
        value.includes("..") || value.includes("\\") || value.includes("?")) {
      throw new Error("All Potree dependencies must be vendored local paths");
    }
    return value;
  };
  for (const href of assets.styles || []) {
    const link = document.createElement("link"); link.rel = "stylesheet"; link.href = localAsset(href);
    document.head.append(link);
  }
  for (const src of assets.scripts || []) await new Promise((resolve, reject) => {
    const script = document.createElement("script"); script.src = localAsset(src);
    script.onload = resolve; script.onerror = () => reject(new Error(`Missing local asset: ${src}`));
    document.head.append(script);
  });
  if (!window.Potree?.Viewer || !window.Potree?.loadPointCloud) throw new Error("A real Potree build was not loaded");
  const viewer = new window.Potree.Viewer(document.getElementById("render"));
  viewer.setEDLEnabled(true);
  viewer.setBackground("black");
  viewer.setPointBudget(assets.pointBudget);
  window.Potree.loadPointCloud(cloudURL.href, "Reconstruction", event => {
    const points = event.pointcloud;
    if (!points) { status.textContent = "Potree did not return a point cloud"; return; }
    viewer.scene.addPointCloud(points);
    points.material.size = 1;
    points.material.pointSizeType = window.Potree.PointSizeType.ADAPTIVE;
    viewer.fitToScreen();
    status.textContent = "Local Potree octree · measurements use the Cesium projected-coordinate tools";
  });
} catch (error) { status.textContent = `Potree unavailable: ${error.message}`; }
