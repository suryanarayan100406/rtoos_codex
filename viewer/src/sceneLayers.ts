import {
  ImageryLayer, Rectangle, SingleTileImageryProvider,
  TextureMagnificationFilter, TextureMinificationFilter,
} from "cesium";
import type { Viewer } from "cesium";
import { fileURL } from "./api";
import type { Scene, SemanticLayer } from "./types";

export function semanticLayer(scene: Scene): SemanticLayer | undefined {
  return scene.semantic_layers?.find(layer => layer.id === "semantic");
}

export function hasSemanticPreview(layer: SemanticLayer | undefined): boolean {
  return layer?.status === "ok" && !!layer.raster_path && !!layer.preview;
}

export function geometryKey(scene: Scene): string {
  // S9 imagery revisions must not destroy/reload unrelated mesh and cloud products.
  const stages = ["s6_global", "s7_dense", "s8_mesh", "s10_export"];
  return JSON.stringify({ coordinate_frame: scene.coordinate_frame, origin: scene.origin,
    crs: scene.crs, exports: scene.exports,
    artifacts: { cloud: scene.artifacts.cloud, ply: scene.artifacts.ply, mesh: scene.artifacts.mesh },
    revision: Object.fromEntries(stages.map(name => [name, scene.revision?.[name]])),
  });
}

export function mountSemanticImagery(viewer: Viewer, runId: string, layer: SemanticLayer,
  revision: string | null | undefined, notice: (text: string) => void): () => void {
  let active = true;
  let imagery: ImageryLayer | undefined;
  const dispose = () => {
    active = false;
    if (imagery && !viewer.isDestroyed()) {
      viewer.imageryLayers.remove(imagery, true);
      if (!imagery.isDestroyed()) imagery.destroy();
      viewer.scene.requestRender();
    }
    imagery = undefined;
  };
  if (!hasSemanticPreview(layer) || viewer.isDestroyed()) return dispose;
  const preview = layer.preview!;
  notice("Loading geographic semantic overlay…");
  void (async () => {
    try {
      const provider = await SingleTileImageryProvider.fromUrl(
        fileURL(runId, preview.path, revision),
        { rectangle: Rectangle.fromDegrees(...preview.bounds) },
      );
      if (!active || viewer.isDestroyed()) return;
      if (provider.tileWidth !== preview.width || provider.tileHeight !== preview.height) {
        throw new Error("Decoded preview dimensions differ from the S9 descriptor.");
      }
      // A nearest-warped RGBA image on the offline ellipsoid, never a scene primitive.
      imagery = new ImageryLayer(provider, {
        minificationFilter: TextureMinificationFilter.NEAREST,
        magnificationFilter: TextureMagnificationFilter.NEAREST,
      });
      viewer.imageryLayers.add(imagery);
      notice("");
      viewer.scene.requestRender();
    } catch (error) {
      if (active && !viewer.isDestroyed()) {
        dispose();
        notice(`Semantic overlay unavailable: ${String(error)} The class GeoTIFF remains downloadable.`);
      }
    }
  })();
  return dispose;
}
