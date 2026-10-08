"""Strict, offline pipeline configuration; all tunables live here or in YAML."""

from pathlib import Path
from sys import float_info
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def _path(value: object) -> object:
    return Path(value).expanduser() if isinstance(value, str) else value


FilePath = Annotated[Path, BeforeValidator(_path)]
Positive = Annotated[float, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]
PositiveFraction = Annotated[float, Field(gt=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True, allow_inf_nan=False)


class CSVColumns(StrictModel):
    time: str = "timestamp"
    lat: str = "latitude"
    lon: str = "longitude"
    alt: str = "abs_alt"
    yaw: str | None = None
    pitch: str | None = None
    roll: str | None = None
    delimiter: str = Field(default=",", min_length=1, max_length=1)


class TelemetryConfig(StrictModel):
    format: Literal["dji_srt", "csv", "mavlink", "exif"]
    path: FilePath
    csv: CSVColumns = Field(default_factory=CSVColumns)
    time_origin_s: float | None = None
    altitude_datum: Literal["unknown", "ellipsoidal", "orthometric", "relative"]


class OptionalInputs(StrictModel):
    imu: FilePath | None = None
    baro: FilePath | None = None
    intrinsics: FilePath | None = None
    rtk: FilePath | None = None


class ReportConfig(StrictModel):
    check_points: FilePath | None = None
    accuracy_target_m: Positive = 1.0
    region_size_m: Positive = 20.0


class DatasetDescriptor(StrictModel):
    name: str = Field(min_length=1)
    video: FilePath
    telemetry: TelemetryConfig
    optional: OptionalInputs = Field(default_factory=OptionalInputs)
    report: ReportConfig = Field(default_factory=ReportConfig)


class RunConfig(StrictModel):
    seed: int = Field(default=1234, ge=0)
    output_root: FilePath = Path("runs")
    copy_inputs: bool = True


class IngestConfig(StrictModel):
    target_fps: Positive = 4.0
    max_frames: int = Field(default=0, ge=0)
    time_offset_s: float = 0.0
    max_telemetry_gap_s: Positive = 2.0


class BarometerCSVColumns(StrictModel):
    time: str = "timestamp"
    height: str = "height_m"
    sigma: str | None = None
    delimiter: str = Field(default=",", min_length=1, max_length=1)

    @model_validator(mode="after")
    def column_mapping(self) -> "BarometerCSVColumns":
        columns = [self.time, self.height]
        if self.sigma is not None:
            columns.append(self.sigma)
        if any(not column.strip() for column in columns):
            raise ValueError("Barometer CSV column mappings must be nonblank")
        if len(set(columns)) != len(columns):
            raise ValueError("Barometer CSV column mappings must be distinct")
        return self


class BarometerConfig(StrictModel):
    enabled: bool = False
    time_origin_s: float | None = None
    time_offset_s: float = 0.0
    max_time_error_s: Positive = 0.1
    sigma_m: Positive = 2.0
    min_measurements: int = Field(default=3, ge=2)
    max_samples: int = Field(default=1000000, ge=2)
    csv: BarometerCSVColumns = Field(default_factory=BarometerCSVColumns)

    @model_validator(mode="after")
    def clock_and_budget(self) -> "BarometerConfig":
        if self.enabled and self.time_origin_s is None:
            raise ValueError("Enabled barometer requires explicit barometer.time_origin_s")
        if self.max_samples < self.min_measurements:
            raise ValueError("barometer.max_samples must be at least min_measurements")
        return self


class FrameQAConfig(StrictModel):
    min_sharpness: float = Field(default=30.0, ge=0)
    max_dark_fraction: Fraction = 0.6
    max_bright_fraction: Fraction = 0.6
    dark_threshold: int = Field(default=10, ge=0, le=255)
    bright_threshold: int = Field(default=245, ge=0, le=255)
    # This legacy name is retained for config/API compatibility. It is now the
    # minimum robust homography residual used as a rotation/planar-compensated
    # geometric parallax proxy, never raw image displacement.
    min_parallax_px: float = Field(default=8.0, ge=0)
    min_geometric_matches: int = Field(default=12, ge=4)
    homography_ransac_threshold_px: Positive = 3.0
    homography_ransac_iterations: int = Field(default=1000, ge=1)
    homography_ransac_confidence: float = Field(default=0.995, gt=0, lt=1)
    homography_min_inlier_fraction: Fraction = 0.5
    geometric_residual_quantile: PositiveFraction = 0.75
    min_match_coverage_fraction: Fraction = 0.125
    essential_ransac_threshold_px: Positive = 1.0
    essential_ransac_iterations: int = Field(default=1000, ge=1)
    essential_ransac_confidence: float = Field(default=0.999, gt=0, lt=1)
    essential_min_inlier_fraction: Fraction = 0.5
    essential_min_positive_depth_fraction: Fraction = 0.5
    essential_homography_inlier_margin: Fraction = 0.05
    min_angular_parallax_deg: float = Field(default=0.1, ge=0)
    max_interval_s: Positive = 2.0
    max_keyframes: int = Field(default=300, ge=3)
    orb_features: int = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def thresholds(self) -> "FrameQAConfig":
        if self.dark_threshold >= self.bright_threshold:
            raise ValueError("dark_threshold must be less than bright_threshold")
        return self


class PosesConfig(StrictModel):
    matcher: Literal["sequential", "exhaustive"] = "sequential"
    overlap: int = Field(default=10, ge=1)
    camera_model: Literal["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"] = "SIMPLE_RADIAL"
    min_registered: int = Field(default=3, ge=3)
    min_track_length: int = Field(default=3, ge=2)
    use_glomap: bool = False


class SpineConfig(StrictModel):
    gnss_sigma_h_m: Positive = 3.0
    gnss_sigma_v_m: Positive = 5.0
    visual_sigma_m: Positive = 0.2
    rotation_sigma_rad: Positive = 0.05
    robust_delta: Positive = 1.345
    alignment_threshold_m: Positive = 10.0
    ransac_iterations: int = Field(default=300, ge=1)
    min_baseline_m: Positive = 1.0
    min_rank_ratio: Fraction = 0.001
    optimize: bool = True
    max_iterations: int = Field(default=100, ge=1)


class CRSConfig(StrictModel):
    mode: Literal["derive_from_gps", "epsg"] = "derive_from_gps"
    epsg: int | None = Field(default=None, gt=0)
    vertical: Literal["source", "ellipsoidal", "orthometric"] = "source"
    geoid_path: FilePath | None = None

    @model_validator(mode="after")
    def explicit_epsg(self) -> "CRSConfig":
        if (self.mode == "epsg") != (self.epsg is not None):
            raise ValueError("set both crs.mode=epsg and crs.epsg, or derive without epsg")
        return self


class BundleAdjustmentConfig(StrictModel):
    enabled: bool = True
    reprojection_sigma_px: Positive = 1.0
    min_track_length: int = Field(default=3, ge=2)
    max_points: int = Field(default=10000, ge=3)
    max_observations: int = Field(default=100000, ge=9)
    max_nfev: int = Field(default=100, ge=1)
    ftol: float = Field(default=1e-6, gt=float_info.epsilon, lt=1)
    xtol: float = Field(default=1e-6, gt=float_info.epsilon, lt=1)
    gtol: float = Field(default=1e-6, gt=float_info.epsilon, lt=1)
    min_depth_m: Positive = 0.01


class MaskingConfig(StrictModel):
    model: str | None = "rtdetr"
    threshold: Fraction = 0.5
    dilation_px: int = Field(default=5, ge=0)
    dynamic_classes: list[str] = Field(default_factory=lambda: ["person", "car", "truck", "bus", "motorcycle", "bicycle", "bird", "dog", "cat", "horse"])
    motion_fallback: bool = True
    motion_threshold_px: Positive = 3.0
    min_motion_matches: int = Field(default=16, ge=4)
    motion_with_detector: bool = True
    motion_features: int = Field(default=2000, ge=4)
    motion_quality: float = Field(default=0.01, gt=0, le=1)
    motion_spacing_px: Positive = 7.0
    motion_ransac_iterations: int = Field(default=2000, ge=1)
    motion_ransac_confidence: float = Field(default=0.99, gt=0, lt=1)
    motion_min_inlier_fraction: Fraction = 0.5
    flow_window_px: int = Field(default=21, ge=3)
    flow_levels: int = Field(default=3, ge=1)
    flow_fb_threshold_px: Positive = 1.5
    flow_pyr_scale: float = Field(default=0.5, gt=0, lt=1)
    flow_iterations: int = Field(default=3, ge=1)
    flow_poly_n: Literal[5, 7] = 5
    flow_poly_sigma: Positive = 1.2
    motion_unknown_policy: Literal["exclude", "retain"] = "exclude"
    max_motion_exclusion_fraction: Fraction = 1.0

    @model_validator(mode="after")
    def flow_window(self) -> "MaskingConfig":
        if self.flow_window_px % 2 != 1:
            raise ValueError("flow_window_px must be odd")
        return self


class SemanticClass(StrictModel):
    id: int = Field(ge=1, le=65534)
    name: str = Field(min_length=1)
    color: list[Annotated[int, Field(ge=0, le=255)]] = Field(min_length=3, max_length=3)
    source_ids: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)


class SemanticsConfig(StrictModel):
    model: str | None = Field(default=None, min_length=1)
    min_probability: Fraction = 0.5
    preview_long_side: int = Field(default=1024, ge=16, le=4096)
    classes: list[SemanticClass] = Field(default_factory=list)

    @model_validator(mode="after")
    def class_mapping(self) -> "SemanticsConfig":
        if self.model is not None and not self.classes:
            raise ValueError("semantics.model requires an explicit classes/source_ids mapping")
        ids = [item.id for item in self.classes]
        names = [item.name.strip().casefold() for item in self.classes]
        sources = [source for item in self.classes for source in item.source_ids]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names) or "" in names:
            raise ValueError("Semantic class IDs and nonblank names must be unique")
        if len(set(sources)) != len(sources):
            raise ValueError("A source class ID may map to only one semantic class")
        return self


class DepthConfig(StrictModel):
    model: str = "depth_anything_v2_small"
    backend: Literal["torch", "openvino"] = "torch"
    input_long_side: int = Field(default=518, ge=14)
    min_anchors: int = Field(default=8, ge=2)
    robust_delta: Positive = 1.345
    min_depth_m: Positive = 0.1
    max_depth_m: Positive = 300.0
    max_relative_residual: Positive = 0.3
    max_anchors: int = Field(default=10000, ge=2)
    support_radius_px: Positive = 100.0
    robust_iterations: int = Field(default=20, ge=1)
    min_inlier_fraction: Fraction = 0.5
    max_anchor_reprojection_error_px: Positive = 3.0

    @model_validator(mode="after")
    def depth_range(self) -> "DepthConfig":
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("min_depth_m must be less than max_depth_m")
        if self.max_anchors < self.min_anchors:
            raise ValueError("max_anchors must be at least min_anchors")
        return self


class DenseConfig(StrictModel):
    method: Literal["point_fusion"] = "point_fusion"
    voxel_m: Positive = 0.1
    pixel_stride: int = Field(default=4, ge=1)
    min_confidence: Fraction = 0.2
    max_points_per_chunk: int = Field(default=1000000, ge=4)
    tile_m: Positive = 40.0
    max_depth_m: Positive = 300.0
    normal_max_relative_jump: Positive = 0.1


class MeshConfig(StrictModel):
    method: Literal["poisson"] = "poisson"
    poisson_depth: int = Field(default=9, ge=2, le=15)
    density_quantile: Fraction = 0.05
    max_vertices: int = Field(default=500000, ge=3)
    support_radius_m: Positive = 0.5
    texture: Literal["best_view"] = "best_view"
    texture_size: int = Field(default=2048, ge=16)
    normal_neighbors: int = Field(default=30, ge=3)
    max_input_points: int = Field(default=1000000, ge=4)
    texture_patch_px: int = Field(default=16, ge=4)
    texture_padding_px: int = Field(default=1, ge=0)
    visibility_long_side: int = Field(default=1024, ge=16)
    texture_occlusion_tolerance_m: Positive = 0.1
    texture_candidate_views: int = Field(default=4, ge=1, le=16)

    @model_validator(mode="after")
    def texture_atlas(self) -> "MeshConfig":
        if self.texture_patch_px <= 2 * self.texture_padding_px + 1:
            raise ValueError("texture_patch_px must exceed 2 * texture_padding_px + 1")
        if self.texture_patch_px > self.texture_size:
            raise ValueError("texture_patch_px must fit within texture_size")
        if self.max_input_points <= self.normal_neighbors:
            raise ValueError("max_input_points must exceed normal_neighbors")
        return self


class GeoConfig(StrictModel):
    resolution_m: Positive = 0.5
    max_cells: int = Field(default=20000000, ge=1)
    ground_method: Literal["morphological"] = "morphological"
    ground_window_m: Positive = 10.0
    ground_threshold_m: Positive = 0.5
    nodata: float = -9999.0


ExportFormat = Literal["obj", "ply", "las", "geotiff", "glb", "gltf", "fbx", "laz", "3dtiles", "potree"]


def required_formats() -> list[ExportFormat]:
    return ["obj", "ply", "las", "geotiff", "glb", "gltf", "fbx"]


def optional_formats() -> list[ExportFormat]:
    return ["laz", "3dtiles", "potree"]


class ExportConfig(StrictModel):
    formats: list[ExportFormat] = Field(default_factory=required_formats)
    optional_formats: list[ExportFormat] = Field(default_factory=optional_formats)
    blender: str = "blender"
    las_scale_m: Positive = 0.001


class ModelsConfig(StrictModel):
    registry: FilePath = Path("configs/models.yaml")
    cache_dir: FilePath = Path("models")
    offline: Literal[True] = True


class ComputeConfig(StrictModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    openvino_device: str = "CPU"
    environment: str | None = None


class Config(StrictModel):
    dataset: DatasetDescriptor | None = None
    run: RunConfig = Field(default_factory=RunConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    barometer: BarometerConfig = Field(default_factory=BarometerConfig)
    frameqa: FrameQAConfig = Field(default_factory=FrameQAConfig)
    poses: PosesConfig = Field(default_factory=PosesConfig)
    spine: SpineConfig = Field(default_factory=SpineConfig)
    bundle_adjustment: BundleAdjustmentConfig = Field(default_factory=BundleAdjustmentConfig)
    crs: CRSConfig = Field(default_factory=CRSConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    semantics: SemanticsConfig = Field(default_factory=SemanticsConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True, allow_inf_nan=False)
