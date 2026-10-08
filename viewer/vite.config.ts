import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [react(), viteStaticCopy({ targets: [
    { src: "node_modules/cesium/Build/Cesium/Assets", dest: "cesium" },
    { src: "node_modules/cesium/Build/Cesium/Workers", dest: "cesium" },
    { src: "node_modules/cesium/Build/Cesium/ThirdParty", dest: "cesium" },
    { src: "node_modules/cesium/Build/Cesium/Widgets", dest: "cesium" },
  ] })],
  define: { CESIUM_BASE_URL: JSON.stringify("/cesium/") },
  // Use Cesium's real, prebuilt ESM distribution. This avoids traversing thousands
  // of source modules on Windows-mounted checkouts without replacing the engine.
  resolve: { alias: [{ find: /^cesium$/, replacement: resolve("node_modules/cesium/Build/Cesium/index.js") }] },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { chunkSizeWarningLimit: 1800, rollupOptions: { output: {
    manualChunks: { cesium: ["cesium"], react: ["react", "react-dom"] },
  } } },
});
