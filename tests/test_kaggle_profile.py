"""Kaggle quality reader contracts, not GPU-memory or real-flight acceptance.

Constructed PTS, depth and mesh boundaries exercise actual configuration/geometry
code. The isolated Open3D boundary is not evidence that Poisson ran on a GPU/scene.
"""

import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from drishti.config import Config, load_config
from drishti.config.loader import read_yaml
from drishti.errors import ReconstructionError
from drishti.io.video import decode_video
from drishti.recon.depth import align_inverse_depth
from drishti.recon.meshing import poisson_mesh
from drishti.recon.texturing import best_view_texture
from drishti.runtime import kaggle

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PROFILE = CONFIGS / "profiles" / "kaggle_quality.yaml"


@pytest.fixture
def quality():
    return load_config(profile="kaggle_quality", config_dir=CONFIGS, environ={})


@pytest.fixture
def maximum():
    return load_config(profile="max", config_dir=CONFIGS, environ={})


def test_profile_validates_as_typed_overrides_and_through_named_and_path_loading(quality):
    # Standalone validation rejects unsupported keys even when defaults mask typos.
    Config.model_validate(read_yaml(PROFILE))
    by_path = load_config(profile=PROFILE, config_dir=CONFIGS, environ={})
    assert by_path == quality
    snapshot = quality.model_dump(mode="json")
    assert Config.model_validate(snapshot) == quality
    assert snapshot["ingest"]["max_frames"] == 0
    assert quality.dataset is None  # A profile does not invent a dataset or sensors.


def test_only_executable_local_permissive_methods_and_actual_device_request(quality, maximum):
    assert quality.dense.method == "point_fusion"
    assert quality.mesh.method == "poisson"
    assert quality.mesh.texture == "best_view"
    assert quality.poses.matcher == "sequential" and not quality.poses.use_glomap
    assert quality.depth.model == "depth_anything_v2_small"
    assert quality.depth.backend == "torch"
    assert quality.models.offline is True
    assert quality.models.registry == maximum.models.registry
    assert quality.models.cache_dir == maximum.models.cache_dir
    assert quality.compute.device == "cuda"
    assert quality.compute.environment is None  # Requested placement is not measured provenance.
    assert quality.semantics.model is None and not quality.semantics.classes
    assert not quality.barometer.enabled and quality.barometer.time_origin_s is None


def test_quality_work_and_resolution_increase_with_smaller_streaming_and_grid_budgets(
    quality, maximum,
):
    assert quality.ingest.target_fps >= maximum.ingest.target_fps
    qa = quality.frameqa
    assert qa.orb_features > maximum.frameqa.orb_features
    assert qa.max_keyframes >= maximum.frameqa.max_keyframes
    assert 0 < qa.min_parallax_px <= maximum.frameqa.min_parallax_px
    assert 1 / quality.ingest.target_fps <= qa.max_interval_s < maximum.frameqa.max_interval_s
    assert maximum.poses.overlap < quality.poses.overlap < qa.max_keyframes
    assert quality.spine.optimize and quality.bundle_adjustment.enabled
    assert quality.spine.ransac_iterations > maximum.spine.ransac_iterations
    assert quality.spine.max_iterations > maximum.spine.max_iterations
    ba = quality.bundle_adjustment
    assert ba.max_points > maximum.bundle_adjustment.max_points
    assert ba.max_observations > maximum.bundle_adjustment.max_observations
    assert ba.min_track_length >= quality.poses.min_track_length >= 3
    assert ba.max_observations >= ba.max_points * ba.min_track_length
    assert ba.max_nfev > maximum.bundle_adjustment.max_nfev
    # Patch-compatible resolution, bounded to at most 4x baseline input pixel area.
    assert quality.depth.input_long_side % 14 == 0
    assert maximum.depth.input_long_side < quality.depth.input_long_side <= 2 * maximum.depth.input_long_side
    assert maximum.depth.max_anchors < quality.depth.max_anchors <= ba.max_points
    dense, mesh, geo = quality.dense, quality.mesh, quality.geo
    assert dense.pixel_stride <= maximum.dense.pixel_stride
    assert dense.voxel_m <= maximum.dense.voxel_m
    assert dense.max_points_per_chunk < maximum.dense.max_points_per_chunk
    assert dense.tile_m < maximum.dense.tile_m
    assert dense.tile_m / dense.voxel_m == pytest.approx(round(dense.tile_m / dense.voxel_m))
    assert mesh.poisson_depth >= maximum.mesh.poisson_depth
    assert dense.max_points_per_chunk < maximum.mesh.max_input_points < mesh.max_input_points
    assert mesh.max_vertices >= maximum.mesh.max_vertices
    assert mesh.texture_size > maximum.mesh.texture_size
    assert maximum.mesh.visibility_long_side < mesh.visibility_long_side < mesh.texture_size
    assert dense.voxel_m <= geo.resolution_m < maximum.geo.resolution_m
    assert geo.max_cells < maximum.geo.max_cells


@pytest.mark.parametrize("section,work_knobs", [
    ("ingest", {"target_fps", "max_frames"}),
    ("frameqa", {"orb_features", "max_keyframes", "min_parallax_px", "max_interval_s"}),
    ("poses", {"overlap"}),
    ("spine", {"ransac_iterations", "max_iterations"}),
    ("bundle_adjustment", {"max_points", "max_observations", "max_nfev"}),
    ("depth", {"input_long_side", "max_anchors"}),
    ("dense", {"pixel_stride", "voxel_m", "max_points_per_chunk", "tile_m"}),
    ("mesh", {"poisson_depth", "max_input_points", "max_vertices", "texture_size", "visibility_long_side"}),
    ("geo", {"resolution_m", "max_cells"}),
    ("crs", set()),
    ("masking", set()),
    ("export", set()),
    ("report", set()),
])
def test_quality_preserves_support_confidence_nodata_and_provenance_filters(
    quality, maximum, section, work_knobs,
):
    # Compare every remaining typed field, including future provenance thresholds;
    # cost/resolution increases must not silently relax evidence requirements.
    actual = getattr(quality, section).model_dump()
    baseline = getattr(maximum, section).model_dump()
    assert {k: v for k, v in actual.items() if k not in work_knobs} == {
        k: v for k, v in baseline.items() if k not in work_knobs
    }


@pytest.fixture
def descriptor(tmp_path):
    # Descriptor-only unit: these deliberately nonexistent paths are not footage.
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump({
        "name": "profile-reader-unit", "video": "absent-video.mp4",
        "telemetry": {"format": "csv", "path": "absent-gps.csv", "altitude_datum": "unknown"},
    }), encoding="utf-8")
    return path


def test_dataset_can_explicitly_enable_optional_inputs_without_profile_requirements(descriptor):
    before = load_config(descriptor, PROFILE, config_dir=CONFIGS, environ={})
    assert before.dataset.optional.baro is None
    assert not before.barometer.enabled and before.semantics.model is None
    data = read_yaml(descriptor)
    data.update({
        "optional": {"baro": "recorded-heights.csv"},
        "barometer": {"enabled": True, "time_origin_s": 1700000000.0},
        "semantics": {"model": "local-semantic-unit", "classes": [
            {"id": 1, "name": "ground", "color": [80, 100, 60], "source_ids": [0]},
        ]},
    })
    descriptor.write_text(yaml.safe_dump(data), encoding="utf-8")
    configured = load_config(descriptor, PROFILE, config_dir=CONFIGS, environ={})
    assert configured.barometer.enabled
    assert configured.barometer.time_origin_s == data["barometer"]["time_origin_s"]
    assert configured.dataset.optional.baro == descriptor.parent / data["optional"]["baro"]
    assert configured.semantics.model == data["semantics"]["model"]
    assert configured.dataset.telemetry == before.dataset.telemetry  # GPS/datum remain mandatory.
    assert configured.depth == before.depth


@pytest.mark.parametrize("duration_s", [60, 600])
def test_runner_sample_budget_reduces_cadence_and_decoder_keeps_flight_tail(
    descriptor, tmp_path, monkeypatch, duration_s,
):
    cfg = load_config(descriptor, PROFILE, config_dir=CONFIGS, environ={})
    settings = kaggle.KaggleSettings(dataset=descriptor)
    monkeypatch.setattr(kaggle, "declared_duration", lambda _: {
        "declared_duration_s": float(duration_s), "duration_kind": "declared_metadata",
        "duration_source": "constructed_PTS_boundary", "duration_measured_by_decode": False,
    })
    planned, plan = kaggle.sampling_plan(cfg, settings)
    assert 0 < planned.ingest.target_fps <= cfg.ingest.target_fps
    assert planned.ingest.max_frames == cfg.ingest.max_frames == 0
    assert plan["estimated_frames"] <= settings.sample_frame_budget
    assert plan["duration_measured_by_decode"] is False
    if duration_s * cfg.ingest.target_fps > settings.sample_frame_budget:
        assert planned.ingest.target_fps < cfg.ingest.target_fps
    else:
        assert planned.ingest.target_fps == cfg.ingest.target_fps

    class PTSContainer:
        """Isolate decoder sampling; neither encoded footage nor actual PNG output."""
        stream = SimpleNamespace(
            codec_context=SimpleNamespace(name="unit-PTS"), start_time=0,
            time_base=Fraction(1, 12),
        )
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            for pts in range(duration_s * 12 + 1):
                yield SimpleNamespace(
                    pts=pts, time_base=stream.time_base, width=1, height=1,
                    to_image=lambda: SimpleNamespace(save=lambda *args, **kwargs: None),
                )

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _: PTSContainer()))
    frames, metadata = decode_video(
        cfg.dataset.video, tmp_path / "frames", tmp_path,
        target_fps=planned.ingest.target_fps, max_frames=planned.ingest.max_frames,
    )
    assert frames[0]["time_s"] == 0
    assert frames[-1]["time_s"] >= duration_s - 1 / planned.ingest.target_fps
    assert metadata["decoded_frames"] == duration_s * 12 + 1
    assert not metadata["truncated_by_max_frames"]
    assert len(frames) <= settings.sample_frame_budget


def test_unknown_duration_keeps_full_clip_with_honest_unbounded_sample_estimate(
    descriptor, monkeypatch,
):
    cfg = load_config(descriptor, PROFILE, config_dir=CONFIGS, environ={})
    monkeypatch.setattr(kaggle, "declared_duration", lambda _: {
        "declared_duration_s": None, "duration_kind": "unavailable_metadata",
        "duration_source": None, "duration_measured_by_decode": False,
    })
    planned, plan = kaggle.sampling_plan(cfg, kaggle.KaggleSettings(dataset=descriptor))
    assert planned.ingest == cfg.ingest and planned.ingest.max_frames == 0
    assert plan["estimated_frames"] is None
    assert plan["estimate_status"] == "unbounded_unknown_duration"


def test_metric_depth_remains_float32_masked_and_requires_real_anchor_support(quality):
    # Exact inverse-depth mathematics, not a neural prediction or mission accuracy.
    y, x = np.indices((24, 32))
    prediction = (1 + x / 32 + y / 24).astype(np.float32)
    excluded = np.zeros(prediction.shape, bool)
    excluded[:2] = True
    uv = np.array([(x, y) for y in (4, 10, 16) for x in (4, 10, 16, 22)])
    z = 1 / (0.05 * prediction[uv[:, 1], uv[:, 0]] + 0.01)
    anchors = np.column_stack((uv, z))
    settings = quality.depth.model_dump(exclude={"model", "backend", "input_long_side", "max_anchors", "max_anchor_reprojection_error_px"})
    fit = align_inverse_depth(prediction, anchors, excluded, **settings)
    assert fit.depth.dtype == fit.confidence.dtype == np.float32
    assert not fit.depth[excluded].any() and not fit.confidence[excluded].any()
    np.testing.assert_allclose(fit.depth[~excluded], 1 / (0.05 * prediction[~excluded] + 0.01), rtol=1e-6)
    with pytest.raises(ReconstructionError, match="Too few valid"):
        align_inverse_depth(prediction, anchors[:quality.depth.min_anchors - 1], excluded, **settings)


@pytest.mark.parametrize("limiter", ["atlas", "vertices"])
def test_actual_poisson_face_cap_and_uv_expansion_respect_both_profile_budgets(
    quality, maximum, monkeypatch, limiter,
):
    cfg = quality.mesh
    atlas_faces = (cfg.texture_size // cfg.texture_patch_px) ** 2
    if limiter == "vertices":
        cfg = type(cfg).model_validate(cfg.model_dump() | {"max_vertices": 3 * (atlas_faces // 2)})
    expected_faces = min(cfg.max_vertices // 3, atlas_faces)
    assert expected_faces > 0
    assert 3 * expected_faces <= cfg.max_vertices
    assert expected_faces <= atlas_faces
    if limiter == "atlas":
        baseline_capacity = min(maximum.mesh.max_vertices // 3,
                                (maximum.mesh.texture_size // maximum.mesh.texture_patch_px) ** 2)
        assert expected_faces > baseline_capacity
        assert 3 * expected_faces < cfg.max_vertices  # Vertex ceiling is not attainable.
    calls = []

    class FaceCount:
        def __len__(self):
            return expected_faces + 1

    class IsolatedMesh:
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        triangles = FaceCount()

        def remove_vertices_by_mask(self, mask):
            pass

        def remove_degenerate_triangles(self):
            pass

        def remove_duplicated_triangles(self):
            pass

        def remove_unreferenced_vertices(self):
            pass

        def simplify_quadric_decimation(self, face_limit):
            calls.append(face_limit)
            self.triangles = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
            return self

    class IsolatedCloud:
        def estimate_normals(self, search):
            pass

    def isolated_poisson(cloud, *, depth):
        assert depth == cfg.poisson_depth
        assert len(cloud.points) <= cfg.max_input_points
        return IsolatedMesh(), np.ones(4)

    monkeypatch.setitem(sys.modules, "open3d", SimpleNamespace(
        geometry=SimpleNamespace(
            PointCloud=IsolatedCloud, KDTreeSearchParamKNN=lambda **kwargs: None,
            TriangleMesh=SimpleNamespace(create_from_point_cloud_poisson=isolated_poisson),
        ),
        utility=SimpleNamespace(Vector3dVector=np.asarray),
    ))
    side = math.ceil(math.sqrt(cfg.normal_neighbors + 1))
    x, y = np.meshgrid(np.arange(side), np.arange(side))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float64)
    arrays = {
        "points": points, "normals": np.tile([0., 0., 1.], (len(points), 1)),
        "colors": np.zeros((len(points), 3), np.uint8),
        "confidence": np.full(len(points), 0.8, np.float32), "inferred": np.zeros(len(points), bool),
    }
    mesh, metadata = poisson_mesh(arrays, cfg, quality.dense.max_points_per_chunk)
    # This captures the actual reconstruction code's requested decimation limit,
    # without allocating a million-vertex mesh or claiming real Open3D execution.
    assert calls == [expected_faces]
    assert metadata["support_method"] == "nearest_full_observed_cloud_distance"
    # Bake only a tiny no-camera atlas to verify the real 3-vertices/face contract.
    tiny_atlas = type(cfg).model_validate(cfg.model_dump() | {"texture_size": cfg.texture_patch_px * 2})
    textured = best_view_texture(mesh["vertices"], mesh["faces"], [], None, None,
                                 tiny_atlas, mesh["inferred"])
    assert len(textured.vertices) == 3 * len(mesh["faces"]) <= cfg.max_vertices
    np.testing.assert_array_equal(textured.vertices, mesh["vertices"][mesh["faces"].ravel()])
    assert textured.metadata["observed_texels"] == 0
    assert not textured.texture[..., 3].any()  # Unobserved geometry gains no invented texture.
