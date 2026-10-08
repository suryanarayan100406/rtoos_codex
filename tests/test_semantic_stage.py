"""Constructed S3 boundaries, not real checkpoint/recorded-scene acceptance.

Registry hashing, image/NPY IO, dynamic box masking, descriptor publication, and
freshness are real. Neural and selected motion boundaries are explicitly isolated.
The registry specimens contain unit bytes, never a claim of trained model weights.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from PIL import Image

from drishti.bundle import Bundle, content_hash, hash_file
from drishti.config.models import Config
from drishti.errors import DependencyError, InputError, ReconstructionError
from drishti.models.semantics import map_prediction
from drishti.runtime import Runtime
from drishti.stages import s3_masking
from drishti.stages.runner import StageRunner, StageSpec

CLASSES = [
    {"id": 11, "name": "land", "color": [10, 20, 30], "source_ids": [0, 1]},
    {"id": 40000, "name": "canopy", "color": [30, 80, 60], "source_ids": [2]},
]
SOURCE_LABELS = {0: "Bare earth", 1: "Built area", 2: "Tree", 3: "Unmapped sky"}
DESCRIPTOR_KEYS = {"status", "reason", "classes", "nodata", "model", "source_labels",
                   "confidence_method", "frames", "failures"}
RUNTIME = Runtime("cpu", {"versions": {}, "isolated_unit_runtime": True})


def stage_fixture(tmp_path, *, count=3, enabled=True):
    bundle = Bundle(tmp_path, {"bundle_version": "1", "stages": {}, "inputs": [],
                               "config_snapshot": {}, "models": []})
    cameras = []
    for index in range(count):
        image = np.zeros((2, 3, 3), np.uint8)
        image[..., 0] = 17 + index
        image[..., 1] = np.arange(3)
        image[..., 2] = np.arange(2)[:, None]
        fid = f"frame{index}"
        image_path = f"s6_global/{fid}.png"
        bundle.stage_dir("s6_global")
        Image.fromarray(image).save(bundle.path(image_path))
        camera = {"frame_id": fid, "image_path": image_path, "height": 2, "width": 3,
                  "image_domain": "distorted" if index % 2 else "undistorted"}
        cameras.append(camera)
    bundle.write_json("s6_global/poses.json", {"cameras": cameras, "pose_revision": "unit_final_poses"})
    bundle.write_json("s1_frameqa/keyframes.json", {"frames": []})
    bundle.manifest["stages"] = {
        "s6_global": {"outputs": {"poses": "s6_global/poses.json"}},
        "s1_frameqa": {"outputs": {"keyframes": "s1_frameqa/keyframes.json"}},
    }
    registry = {}
    for name, domain in (("detector-unit", "object_detection"), ("semantic-unit", "semantic_logits")):
        folder = tmp_path / "cache" / name
        folder.mkdir(parents=True)
        specimen = folder / "config.unit"
        specimen.write_bytes(f"Isolated {name} registry specimen, not model weights".encode())
        digest = hash_file(specimen)
        registry[name] = {"path": name, "revision": digest, "license": "MIT",
                          "files": {specimen.name: digest}, "source": "isolated-unit-boundary",
                          "output_domain": domain}
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump({"models": registry}), encoding="utf-8")
    video, telemetry = tmp_path / "video.unit", tmp_path / "telemetry.unit"
    video.write_bytes(b"Isolated input bytes, not recorded video")
    telemetry.write_bytes(b"Isolated input bytes, not GPS telemetry")
    cfg = Config.model_validate({
        "dataset": {"name": "semantic-stage-unit", "video": video,
                    "telemetry": {"format": "csv", "path": telemetry, "altitude_datum": "unknown"}},
        "run": {"copy_inputs": False},
        "masking": {"model": "detector-unit", "dilation_px": 0, "motion_fallback": False,
                    "motion_with_detector": False},
        "semantics": {"model": "semantic-unit" if enabled else None, "classes": CLASSES,
                      "min_probability": .5},
        "models": {"registry": registry_path, "cache_dir": tmp_path / "cache"},
    })
    return bundle, cfg, cameras


def install_boundaries(monkeypatch, *, errors=None, load_error=None):
    errors = {} if errors is None else errors
    detector = Mock()
    detector.detect.return_value = [{"class": "person", "box": [0, 0, 1, 1], "score": .8}]
    detector_factory = Mock(return_value=detector)
    monkeypatch.setattr(s3_masking, "LocalDetector", detector_factory)
    calls = []

    class UnitSegmenter:
        """Raw winners/probabilities deliberately constructed for a stage unit."""
        source_labels = SOURCE_LABELS

        def __init__(self, spec, classes, device):
            self.device, self.classes = device, classes

        def predict(self, rgb, threshold):
            index = int(rgb[0, 0, 0]) - 17
            calls.append((index, rgb.copy(), threshold))
            if index in errors:
                raise errors[index]
            raw = np.array([[0, 1, 2], [2, 3, 0]], np.int64)
            winning = np.array([[.9, .7, .5], [.49, .8, .95]], np.float32)
            return map_prediction(raw, winning, self.classes, threshold)

    segmenter_factory = Mock(side_effect=load_error or UnitSegmenter)
    monkeypatch.setattr(s3_masking, "LocalSegmenter", segmenter_factory)
    return SimpleNamespace(detector=detector, detector_factory=detector_factory,
                           segmenter_factory=segmenter_factory, calls=calls)


def read_outputs(bundle, result):
    descriptor = bundle.read_json(result.outputs["semantics"])
    assert set(descriptor) == DESCRIPTOR_KEYS
    assert descriptor["nodata"] == 0 and descriptor["classes"] == CLASSES
    assert descriptor["confidence_method"] == s3_masking.SEMANTIC_CONFIDENCE_METHOD
    masking = bundle.read_json(result.outputs["masks"])
    assert masking["semantic_status"] == descriptor["status"]
    assert masking["semantic_labels"] == result.outputs["semantics"]
    for relative in result.outputs.values():
        assert not Path(relative).is_absolute() and bundle.path(relative).exists()
    return descriptor, masking


def assert_dynamic_masks(bundle, masking, count=3):
    assert len(masking["frames"]) == count
    for frame in masking["frames"]:
        assert frame["method"] == "rtdetr_conservative_dilated_boxes"
        assert frame["dynamic_fraction"] == pytest.approx(1 / 6)
        assert not frame["unknown"]
        np.testing.assert_array_equal(np.asarray(Image.open(bundle.path(frame["path"]))),
                                      [[255, 0, 0], [0, 0, 0]])


def seed_stale_products(bundle):
    directory = bundle.stage_dir("s3_masking")
    maps = directory / "semantics"
    maps.mkdir(exist_ok=True)
    for name in ("frame0.labels.npy", "frame0.confidence.npy", "removed.labels.npy"):
        np.save(maps / name, np.full((2, 3), 7))
    masks = directory / "masks"
    masks.mkdir(exist_ok=True)
    Image.fromarray(np.full((2, 3), 255, np.uint8)).save(masks / "removed.png")
    bundle.write_json("s3_masking/semantics.json", {"status": "ok", "model": {"name": "stale"},
                                                 "frames": [{"frame_id": "removed"}]})
    bundle.write_json("s3_masking/masks.json", {"frames": [{"frame_id": "removed"}]})
    return maps


def test_disabled_always_publishes_descriptor_without_semantic_loading_and_removes_stale_maps(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path, enabled=False)
    boundaries = install_boundaries(monkeypatch)
    maps = seed_stale_products(bundle)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "disabled" and desc["reason"]
    assert desc["model"] is None and desc["source_labels"] is None
    assert desc["frames"] == desc["failures"] == []
    boundaries.segmenter_factory.assert_not_called()
    assert not boundaries.calls and not list(maps.glob("*.npy"))
    assert "semantic_maps" not in result.outputs
    assert not bundle.path("s3_masking/masks/removed.png").exists()
    assert result.metrics["semantic_inference_frames"] == 0
    assert_dynamic_masks(bundle, masking)
    assert [model["name"] for model in result.models] == ["detector-unit"]


def test_success_resolves_verified_checkpoint_and_publishes_aligned_excluded_maps(tmp_path, monkeypatch):
    bundle, cfg, cameras = stage_fixture(tmp_path)
    boundaries = install_boundaries(monkeypatch)
    maps = seed_stale_products(bundle)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "ok" and desc["reason"] is None and not desc["failures"]
    assert len(desc["frames"]) == len(boundaries.calls) == 3
    assert_dynamic_masks(bundle, masking)
    expected_labels = np.array([[0, 11, 40000], [0, 0, 11]], np.uint16)
    expected_probability = np.array([[0, .7, .5], [0, 0, .95]], np.float32)
    for entry, camera, (_, observed, threshold) in zip(desc["frames"], cameras, boundaries.calls, strict=True):
        assert entry["frame_id"] == camera["frame_id"]
        assert entry["image_domain"] == camera["image_domain"]
        assert (entry["height"], entry["width"]) == (camera["height"], camera["width"])
        assert entry["labeled_pixels"] == 3
        np.testing.assert_array_equal(observed, np.asarray(Image.open(bundle.path(camera["image_path"]))))
        assert threshold == cfg.semantics.min_probability
        labels = np.load(bundle.path(entry["path"]), allow_pickle=False)
        probability = np.load(bundle.path(entry["confidence_path"]), allow_pickle=False)
        assert labels.dtype == np.uint16 and probability.dtype == np.float32
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(probability, expected_probability)
    assert result.confidence == pytest.approx(.8)  # Dynamic support, not semantic probabilities.
    assert result.degraded_reason is None and result.metrics["semantic_frames"] == 3
    spec, classes, device = boundaries.segmenter_factory.call_args.args
    assert spec["name"] == "semantic-unit" and spec["output_domain"] == "semantic_logits"
    assert Path(spec["path"]).is_absolute() and device == "cpu"
    assert [item.model_dump(mode="json") for item in classes] == CLASSES
    assert spec["files"] == {"config.unit": hash_file(Path(spec["path"]) / "config.unit")}
    assert desc["source_labels"] == {str(key): value for key, value in SOURCE_LABELS.items()}
    assert desc["model"] == result.models[1]
    assert desc["model"]["revision"] == spec["revision"]
    assert desc["model"]["license"] == spec["license"]
    assert desc["model"]["files"] == spec["files"]
    assert desc["model"]["inference_frames"] == 3
    assert desc["model"]["backend"] == "transformers" and desc["model"]["device"] == "cpu"
    assert result.outputs["semantic_maps"] == maps.relative_to(bundle.root).as_posix()
    assert not (maps / "removed.labels.npy").exists()


def test_s3_executes_actual_local_segmenter_through_isolated_transformers_runtime(tmp_path, monkeypatch):
    from test_semantic_inference import _expected_winning_probabilities, _install_runtime

    from drishti.models.semantics import LocalSegmenter

    bundle, cfg, _ = stage_fixture(tmp_path, count=1)
    install_boundaries(monkeypatch)  # Only the detector remains isolated at S3's boundary.
    runtime = _install_runtime(monkeypatch)
    monkeypatch.setattr(s3_masking, "LocalSegmenter", LocalSegmenter)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "ok" and desc["model"]["inference_frames"] == 1
    runtime.model.assert_called_once_with(pixel_values=runtime.pixel_values)
    for factory in (runtime.processor_factory, runtime.model_factory):
        factory.from_pretrained.assert_called_once_with(
            str(cfg.models.cache_dir / "semantic-unit"), local_files_only=True, trust_remote_code=False)
    expected_labels = np.full((2, 3), 11, np.uint16)
    expected_probability = np.tile(_expected_winning_probabilities(), (2, 1))
    expected_labels[0, 0], expected_probability[0, 0] = 0, 0
    entry = desc["frames"][0]
    np.testing.assert_array_equal(np.load(bundle.path(entry["path"])), expected_labels)
    np.testing.assert_allclose(np.load(bundle.path(entry["confidence_path"])), expected_probability, rtol=1e-6)
    assert desc["source_labels"] == {"0": "Bare Earth", "1": "Built Area", "2": "Tree"}
    assert_dynamic_masks(bundle, masking, count=1)


@pytest.mark.parametrize("declaration,expected", [
    ({}, "distorted"), ({"projection_domain": "undistorted_pinhole"}, "undistorted"),
    ({"projection_domain": "distorted"}, "distorted"),
])
def test_semantic_maps_use_shared_final_camera_legacy_domain_rules(tmp_path, monkeypatch, declaration, expected):
    bundle, cfg, cameras = stage_fixture(tmp_path, count=1)
    install_boundaries(monkeypatch)
    cameras[0].pop("image_domain")
    cameras[0].update(declaration)
    bundle.write_json("s6_global/poses.json", {"cameras": cameras})
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, _ = read_outputs(bundle, result)
    assert desc["status"] == "ok" and desc["frames"][0]["image_domain"] == expected


def test_unsupported_semantic_camera_domain_fails_before_frame_inference(tmp_path, monkeypatch):
    bundle, cfg, cameras = stage_fixture(tmp_path, count=1)
    boundaries = install_boundaries(monkeypatch)
    cameras[0]["image_domain"] = "unsupported"
    bundle.write_json("s6_global/poses.json", {"cameras": cameras})
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "unavailable" and "Unsupported camera image domain" in desc["reason"]
    assert desc["model"] is None and not boundaries.calls
    assert_dynamic_masks(bundle, masking, count=1)


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_partial_inference_preserves_successful_pairs_and_records_only_actual_execution(tmp_path, monkeypatch, failed_index):
    bundle, cfg, _ = stage_fixture(tmp_path)
    boundaries = install_boundaries(monkeypatch, errors={failed_index: RuntimeError("isolated frame failure")})
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "partial" and "isolated frame failure" in desc["reason"]
    assert [frame["frame_id"] for frame in desc["frames"]] == [f"frame{i}" for i in range(3) if i != failed_index]
    assert desc["failures"] == [{"frame_id": f"frame{failed_index}", "reason": "RuntimeError: isolated frame failure"}]
    assert desc["model"]["inference_frames"] == result.metrics["semantic_inference_frames"] == 2
    assert result.metrics["semantic_frames"] == 2 and result.metrics["semantic_failed_frames"] == 1
    assert len(boundaries.calls) == 3
    assert not bundle.path(f"s3_masking/semantics/frame{failed_index}.labels.npy").exists()
    assert not bundle.path(f"s3_masking/semantics/frame{failed_index}.confidence.npy").exists()
    assert_dynamic_masks(bundle, masking)
    assert result.degraded_reason is None


def test_loaded_but_all_failed_model_has_no_execution_provenance_or_placeholder_maps(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path)
    maps = seed_stale_products(bundle)
    boundaries = install_boundaries(monkeypatch, errors={i: InputError("nonfinite logits") for i in range(3)})
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "unavailable" and len(desc["failures"]) == 3
    assert desc["frames"] == [] and desc["model"] is None and desc["source_labels"] is None
    assert boundaries.segmenter_factory.call_count == 1 and len(boundaries.calls) == 3
    assert [model["name"] for model in result.models] == ["detector-unit"]
    assert result.metrics["semantic_inference_frames"] == 0
    assert "semantic_maps" not in result.outputs and not list(maps.glob("*.npy"))
    assert_dynamic_masks(bundle, masking)


@pytest.mark.parametrize("error", [DependencyError("torch absent"), RuntimeError("checkpoint load failed"),
                                   AttributeError("processor cannot load")])
def test_optional_loader_failure_preserves_dynamic_products_without_execution_claim(tmp_path, monkeypatch, error):
    bundle, cfg, _ = stage_fixture(tmp_path)
    boundaries = install_boundaries(monkeypatch, load_error=error)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "unavailable" and str(error) in desc["reason"]
    assert desc["model"] is None and desc["source_labels"] is None
    assert desc["frames"] == desc["failures"] == []  # No frame inference was attempted.
    assert boundaries.segmenter_factory.call_count == 1 and not boundaries.calls
    assert_dynamic_masks(bundle, masking)
    assert result.degraded_reason is None


@pytest.mark.parametrize("failure,reason", [("missing", "No local checkpoint"), ("hash", "sha256 mismatch"),
                                          ("license", "not permitted"), ("inventory", "unhashed")])
def test_registry_refusal_precedes_optional_loading_and_cannot_publish_stale_products(tmp_path, monkeypatch, failure, reason):
    bundle, cfg, _ = stage_fixture(tmp_path)
    boundaries = install_boundaries(monkeypatch)
    maps = seed_stale_products(bundle)
    document = yaml.safe_load(cfg.models.registry.read_text())
    if failure == "missing":
        document["models"].pop("semantic-unit")
    elif failure == "license":
        document["models"]["semantic-unit"]["license"] = "CC-BY-NC-4.0"
    elif failure == "hash":
        (cfg.models.cache_dir / "semantic-unit/config.unit").write_bytes(b"Changed unit checkpoint specimen")
    else:
        (cfg.models.cache_dir / "semantic-unit/unhashed.unit").write_bytes(b"Unhashed unit specimen")
    cfg.models.registry.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "unavailable" and reason in desc["reason"]
    assert desc["frames"] == [] and desc["model"] is None
    boundaries.segmenter_factory.assert_not_called()
    assert not list(maps.glob("*.npy")) and "semantic_maps" not in result.outputs
    assert_dynamic_masks(bundle, masking)


@pytest.mark.parametrize("failure", ["write", "publication"])
def test_optional_map_pair_failure_removes_partial_files_and_retains_inference_provenance(tmp_path, monkeypatch, failure):
    bundle, cfg, _ = stage_fixture(tmp_path)
    install_boundaries(monkeypatch)
    if failure == "write":
        save = np.save

        def fail_write(path, *args, **kwargs):
            if Path(path).name == "frame1.confidence.npy":
                Path(path).write_bytes(b"Unit partial write")
                raise OSError("isolated semantic write failure")
            return save(path, *args, **kwargs)

        monkeypatch.setattr(np, "save", fail_write)
    else:
        replace = Path.replace

        def fail_publication(path, destination):
            if path.name == "frame1.confidence.npy":
                raise OSError("isolated semantic publication failure")
            return replace(path, destination)

        monkeypatch.setattr(Path, "replace", fail_publication)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "partial" and "isolated semantic" in desc["reason"]
    assert desc["model"]["inference_frames"] == 3  # All valid inference ran, two pairs published.
    assert result.metrics["semantic_frames"] == 2 and result.metrics["semantic_inference_frames"] == 3
    assert [frame["frame_id"] for frame in desc["frames"]] == ["frame0", "frame2"]
    assert desc["failures"][0]["frame_id"] == "frame1"
    directory = bundle.path(result.outputs["semantic_maps"])
    assert sorted(path.name for path in directory.iterdir()) == [
        "frame0.confidence.npy", "frame0.labels.npy", "frame2.confidence.npy", "frame2.labels.npy"]
    assert_dynamic_masks(bundle, masking)


def test_interrupted_staging_outputs_are_removed_before_new_publication(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path)
    install_boundaries(monkeypatch)
    maps = seed_stale_products(bundle)
    interrupted = maps / ".semantic-unit-interrupted"
    interrupted.mkdir()
    np.save(interrupted / "frame0.labels.npy", np.full((2, 3), 99, np.uint16))
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, _ = read_outputs(bundle, result)
    assert desc["status"] == "ok" and not interrupted.exists()
    assert len(list(maps.iterdir())) == 6


def test_optional_output_cleanup_failure_preserves_masks_and_never_registers_old_maps(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path)
    boundaries = install_boundaries(monkeypatch)
    maps = seed_stale_products(bundle)
    unlink = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path.parent == maps:
            raise OSError("isolated semantic cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "unavailable" and "cleanup failure" in desc["reason"]
    assert desc["model"] is None and desc["frames"] == [] and "semantic_maps" not in result.outputs
    boundaries.segmenter_factory.assert_not_called()
    assert_dynamic_masks(bundle, masking)


@pytest.mark.parametrize("failure", ["shape", "label_dtype", "confidence_dtype", "nan", "unknown_class"])
def test_invalid_inference_boundary_never_resizes_or_relabels_bad_maps(tmp_path, monkeypatch, failure):
    bundle, cfg, _ = stage_fixture(tmp_path, count=1)
    boundaries = install_boundaries(monkeypatch)
    labels, probability = np.full((2, 3), 11, np.uint16), np.full((2, 3), .9, np.float32)
    if failure == "shape":
        labels, probability = labels[:1], probability[:1]
    elif failure == "label_dtype":
        labels = labels.astype(np.uint8)
    elif failure == "confidence_dtype":
        probability = probability.astype(np.float64)
    elif failure == "nan":
        probability[0, 1] = np.nan
    else:
        labels[0, 1] = 65534
    model = SimpleNamespace(device="cpu", source_labels=SOURCE_LABELS,
                            predict=Mock(return_value=(labels, probability)))
    boundaries.segmenter_factory.side_effect = None
    boundaries.segmenter_factory.return_value = model
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "unavailable" and len(desc["failures"]) == 1
    assert desc["model"] is None and desc["frames"] == [] and "semantic_maps" not in result.outputs
    assert_dynamic_masks(bundle, masking, count=1)


def test_unknown_dynamic_support_excludes_all_semantics_and_does_not_claim_detector_execution(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path, count=1)
    cfg.masking.motion_fallback = True
    boundaries = install_boundaries(monkeypatch)
    boundaries.detector.detect.side_effect = RuntimeError("isolated detector inference failure")
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "ok" and desc["frames"][0]["labeled_pixels"] == 0
    entry = desc["frames"][0]
    assert not np.load(bundle.path(entry["path"]), allow_pickle=False).any()
    assert not np.load(bundle.path(entry["confidence_path"]), allow_pickle=False).any()
    assert masking["frames"][0]["unknown"] and masking["frames"][0]["dynamic_fraction"] is None
    assert np.all(np.asarray(Image.open(bundle.path(masking["frames"][0]["path"]))) == 255)
    assert result.confidence is None and result.degraded_reason
    assert [model["name"] for model in result.models] == ["semantic-unit"]
    assert desc["model"]["inference_frames"] == 1


def test_high_quality_unknown_policy_retains_pixels_with_explicit_unknown_support(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path, count=1, enabled=False)
    cfg.masking.model = ""
    cfg.masking.motion_fallback = True
    cfg.masking.motion_unknown_policy = "retain"
    install_boundaries(monkeypatch)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    _, masking = read_outputs(bundle, result)
    entry = masking["frames"][0]
    assert entry["unknown"] is True
    assert entry["exclusion_fraction"] == 0
    assert np.all(np.asarray(Image.open(bundle.path(entry["path"]))) == 0)
    assert "pixels retained" in (result.degraded_reason or "")


def test_semantic_failure_preserves_supported_motion_exclusions(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path, count=2)
    cfg.masking.model = ""
    cfg.masking.motion_fallback = True
    install_boundaries(monkeypatch, errors={1: RuntimeError("isolated semantic frame failure")})
    excluded = np.array([[False, False, True], [True, False, False]])

    def unit_motion(*args):
        return excluded.copy(), {"status": "ok", "method": "isolated_motion_boundary",
                                 "matches": 20, "inlier_matches": 18, "reliable_fraction": .8}

    monkeypatch.setattr(s3_masking, "motion_mask", unit_motion)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "partial" and len(desc["frames"]) == 1
    for frame in masking["frames"]:
        assert frame["method"] == "isolated_motion_boundary" and not frame["unknown"]
        np.testing.assert_array_equal(np.asarray(Image.open(bundle.path(frame["path"]))) > 0, excluded)
    labels = np.load(bundle.path(desc["frames"][0]["path"]), allow_pickle=False)
    probability = np.load(bundle.path(desc["frames"][0]["confidence_path"]), allow_pickle=False)
    assert not labels[excluded].any() and not probability[excluded].any()
    assert [model["name"] for model in result.models] == ["semantic-unit"]
    assert result.confidence == pytest.approx(.72)


def test_reinvocation_failures_and_disable_replace_previous_successes(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path)
    boundaries = install_boundaries(monkeypatch)
    first = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    assert len(bundle.read_json(first.outputs["semantics"])["frames"]) == 3
    boundaries.segmenter_factory.side_effect = DependencyError("checkpoint now unavailable")
    second = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, second)
    assert desc["status"] == "unavailable" and desc["model"] is None and not desc["frames"]
    assert not list(bundle.path("s3_masking/semantics").glob("*.npy"))
    assert "semantic_maps" not in second.outputs
    assert_dynamic_masks(bundle, masking)
    cfg.semantics.model = None
    third = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    assert bundle.read_json(third.outputs["semantics"])["status"] == "disabled"


@pytest.mark.parametrize("failure", ["image_shape", "no_mask_method"])
def test_required_masking_failure_resets_stale_descriptors_and_still_fails_clearly(tmp_path, monkeypatch, failure):
    bundle, cfg, cameras = stage_fixture(tmp_path)
    boundaries = install_boundaries(monkeypatch)
    maps = seed_stale_products(bundle)
    error: type[RuntimeError]
    if failure == "image_shape":
        cameras[0]["width"] = 4
        bundle.write_json("s6_global/poses.json", {"cameras": cameras})
        error = InputError
    else:
        boundaries.detector_factory.side_effect = DependencyError("isolated missing detector")
        error = ReconstructionError
    with pytest.raises(error):
        s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc = bundle.read_json("s3_masking/semantics.json")
    assert desc["status"] == "unavailable" and not desc["frames"] and desc["model"] is None
    assert bundle.read_json("s3_masking/masks.json")["frames"] == []
    assert not list(maps.glob("*.npy"))
    boundaries.segmenter_factory.assert_not_called()


def test_no_cameras_does_not_turn_loaded_models_into_execution(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path, count=0)
    boundaries = install_boundaries(monkeypatch)
    result = s3_masking.STAGE.run(bundle, cfg, RUNTIME)
    desc, masking = read_outputs(bundle, result)
    assert desc["status"] == "unavailable" and "No registered cameras" in desc["reason"]
    assert desc["model"] is None and result.models == [] and masking["frames"] == []
    boundaries.segmenter_factory.assert_not_called()
    boundaries.detector.detect.assert_not_called()


def record_result(bundle, stage, outputs, runner, models=None):
    hashes = bundle.output_hashes(outputs)
    bundle.manifest["stages"][stage.name] = {
        "status": "done", "outputs": outputs, "output_hashes": hashes,
        "outputs_hash": content_hash(hashes), "models": [] if models is None else models,
    }
    bundle.manifest["stages"][stage.name]["inputs_hash"] = runner.fingerprint(stage)


def fresh_stage_fixture(tmp_path, monkeypatch):
    bundle, cfg, _ = stage_fixture(tmp_path)
    install_boundaries(monkeypatch)
    stage = s3_masking.S3Masking()
    prerequisites = [StageSpec(name, (), ()) for name in stage.requires]
    runner = StageRunner(bundle, cfg, RUNTIME, [*prerequisites, stage])
    for dependency in prerequisites:
        record_result(bundle, dependency, bundle.manifest["stages"][dependency.name]["outputs"], runner)
    result = stage.run(bundle, cfg, RUNTIME)
    record_result(bundle, stage, result.outputs, runner, result.models)
    assert runner.is_fresh(stage.name)
    return bundle, cfg, stage, runner, result


@pytest.mark.parametrize("parameter", ["model", "min_probability", "classes"])
def test_s3_declared_semantic_parameters_invalidate_freshness(tmp_path, monkeypatch, parameter):
    _, cfg, stage, runner, _ = fresh_stage_fixture(tmp_path, monkeypatch)
    assert "semantics" in stage.config_sections
    if parameter == "model":
        cfg.semantics.model = None
    elif parameter == "min_probability":
        cfg.semantics.min_probability = .75
    else:
        cfg.semantics.classes[0].source_ids = [0]
    assert not runner.is_fresh(stage.name)


def test_freshness_hashes_actual_semantic_map_contents_not_only_descriptor(tmp_path, monkeypatch):
    bundle, _, stage, runner, result = fresh_stage_fixture(tmp_path, monkeypatch)
    descriptor = bundle.read_json(result.outputs["semantics"])
    np.save(bundle.path(descriptor["frames"][0]["path"]), np.zeros((2, 3), np.uint16))
    assert not runner.is_fresh(stage.name)


def test_executed_semantic_checkpoint_bytes_invalidate_freshness(tmp_path, monkeypatch):
    _, cfg, stage, runner, _ = fresh_stage_fixture(tmp_path, monkeypatch)
    (cfg.models.cache_dir / "semantic-unit/config.unit").write_bytes(b"Changed measured unit specimen")
    assert not runner.is_fresh(stage.name)
