export type XYZ = [number, number, number];
export type Kind = "distance" | "area" | "height" | "volume";
export type Stage = {
  name?: string; status?: string; confidence?: number | { summary?: number } | null;
  wall_seconds?: number; metrics?: Record<string, unknown>; degraded_reason?: string;
  outputs?: Record<string, string | { path: string }>;
};
export type Manifest = {
  status?: string; error?: string | Record<string, unknown>; created_utc?: string;
  stages?: Record<string, Stage> | Stage[];
  crs?: Record<string, unknown>; environment?: Record<string, unknown>;
};
export type Run = { id: string; status?: string; created_utc?: string;
  job?: { status: string; error?: string }; error?: string | Record<string, unknown> };
export type ExportFormat = { status: string; path?: string; reason?: string;
  details?: { products?: Record<string, string> } };
export type SemanticLayer = {
  id: "semantic"; title: "Semantic classes"; status: "ok" | "unavailable" | "disabled";
  reason: string | null; classes: { id: number; name: string; color: XYZ; source_ids?: number[] }[];
  nodata: 0; supported_cells: number; accuracy_status: "UNVALIDATED";
  confidence_method: string | null; provenance: Record<string, unknown> | null;
  raster_path: string | null; confidence_path: string | null;
  preview: { path: string; crs: "EPSG:4326"; bounds: [number, number, number, number];
    width: number; height: number } | null;
};
export type Scene = {
  coordinate_frame: string; origin: XYZ | null;
  revision?: Record<string, string | null>; stage_status?: Record<string, string | null>;
  crs: { epsg?: number; vertical?: unknown; source_altitude_datum?: string; glb_axes?: unknown;
    proj4?: string; wkt?: string; viewer_wkt1?: string; [key: string]: unknown };
  exports: { formats?: Record<string, ExportFormat>; georeference?: Record<string, unknown> };
  artifacts: { cloud?: string; ply?: string; mesh?: string; dsm?: string; confidence?: string };
  semantic_layers?: SemanticLayer[]; semantic_reason?: string | null;
};
export type Preview = {
  points: XYZ[]; colors: XYZ[] | null; confidence: (number | null)[] | null;
  inferred: (boolean | null)[] | null; total_points: number; preview_points: number;
  bounds: [XYZ, XYZ] | null; source: string;
};
export type Measurement = {
  kind: Kind; value: number; units: string; method: string;
  support: Record<string, unknown>;
  uncertainty: { status: string; absolute_uncertainty_m: number | null; note: string;
    report_evidence?: unknown };
  warnings: string[];
};
export type Layers = { mesh: boolean; points: boolean; tiles: boolean; confidence: boolean;
  inferred: boolean; minConfidence: number; semantic: boolean };
