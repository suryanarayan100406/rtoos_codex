"""Constructed numerical and isolated loader-boundary units, not checkpoint acceptance.

No trained checkpoint or recorded image is supplied here. Mock loaders isolate
offline flags, actual configuration validation, and inference IO; the optional
torch case exercises real tensor operations on explicitly constructed logits.
"""

import builtins
import importlib
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from drishti.config.models import SemanticClass
from drishti.errors import DependencyError, InputError
from drishti.models import semantics
from drishti.models.semantics import LocalSegmenter, map_prediction


def _class(class_id=11, name="land", source_ids=None):
    return SemanticClass(
        id=class_id, name=name, color=[10, 20, 30],
        source_ids=[0, 1] if source_ids is None else source_ids,
    )


def _classes():
    return [_class(), _class(40000, "canopy", [2])]


def test_constructed_mapping_retains_raw_winner_confidence_and_unknown_nodata():
    raw = np.array([[0, 1, 2], [3, -1, 0]], dtype=np.int64)
    probability = np.array([[0.7, 0.8, 0.5], [0.99, 0.9, 0.49]], dtype=np.float64)
    before_raw, before_probability = raw.copy(), probability.copy()
    labels, confidence = map_prediction(raw, probability, _classes(), 0.5)
    np.testing.assert_array_equal(labels, [[11, 11, 40000], [0, 0, 0]])
    np.testing.assert_array_equal(
        confidence, np.array([[0.7, 0.8, 0.5], [0, 0, 0]], dtype=np.float32)
    )
    assert labels.dtype == np.uint16
    assert confidence.dtype == np.float32
    assert labels.shape == confidence.shape == raw.shape
    np.testing.assert_array_equal(raw, before_raw)
    np.testing.assert_array_equal(probability, before_probability)
    assert not np.shares_memory(labels, raw)
    assert not np.shares_memory(confidence, probability)


def test_constructed_uint64_unknown_ids_do_not_wrap_into_configured_source_channels():
    raw = np.array([[0, 2**64 - 1, 2**32]], dtype=np.uint64)
    labels, confidence = map_prediction(raw, np.ones((1, 3)), [_class(source_ids=[0])], 0)
    np.testing.assert_array_equal(labels, [[11, 0, 0]])
    np.testing.assert_array_equal(confidence, [[1, 0, 0]])


def test_constructed_oversized_configured_source_id_cannot_match_via_float_promotion():
    raw = np.array([[2**63 - 1]], dtype=np.int64)
    labels, confidence = map_prediction(
        raw, np.ones((1, 1)), [_class(source_ids=[2**63])], 0
    )
    np.testing.assert_array_equal(labels, [[0]])
    np.testing.assert_array_equal(confidence, [[0]])


def test_constructed_probability_threshold_is_inclusive_without_rounding_up():
    raw = np.zeros((1, 3), dtype=np.int32)
    probabilities = np.array([[0.5, np.nextafter(0.5, 0), 1.0]], dtype=np.float64)
    labels, confidence = map_prediction(raw, probabilities, _classes(), 0.5)
    np.testing.assert_array_equal(labels, [[11, 0, 11]])
    np.testing.assert_array_equal(confidence, [[0.5, 0, 1]])
    labels, _ = map_prediction(raw, probabilities, _classes(), 1.0)
    np.testing.assert_array_equal(labels, [[0, 0, 11]])


def test_float32_probabilities_do_not_round_a_higher_configured_threshold_down():
    labels, confidence = map_prediction(
        np.zeros((1, 2), np.int64), np.array([[.5, 1.]], np.float32),
        _classes(), np.nextafter(.5, 1.),
    )
    np.testing.assert_array_equal(labels, [[0, 11]])
    np.testing.assert_array_equal(confidence, [[0, 1]])


@pytest.mark.parametrize(
    "threshold", [-0.01, 1.01, np.nan, np.inf, -np.inf, True, "0.5", 2**2000]
)
def test_constructed_mapping_rejects_invalid_thresholds(threshold):
    with pytest.raises(InputError, match="min_probability"):
        map_prediction(np.zeros((1, 1), np.int64), np.ones((1, 1)), _classes(), threshold)


@pytest.mark.parametrize(
    ("raw", "probabilities", "reason"),
    [
        (np.zeros((2, 2), np.int64), np.ones((2, 1)), "shape"),
        (np.zeros((2,), np.int64), np.ones(2), "shape"),
        (np.zeros((1, 2, 2), np.int64), np.ones((1, 2, 2)), "shape"),
        (np.zeros((0, 2), np.int64), np.ones((0, 2)), "shape"),
        (np.zeros((1, 1), float), np.ones((1, 1)), "integer"),
        (np.zeros((1, 1), bool), np.ones((1, 1)), "integer"),
        (np.array([["0"]]), np.ones((1, 1)), "integer"),
        (np.zeros((1, 1), np.int64), np.ones((1, 1), np.int64), "floating-point"),
        (np.zeros((1, 1), np.int64), np.ones((1, 1), complex), "floating-point"),
        (np.zeros((1, 1), np.int64), np.array([[np.nan]]), "finite"),
        (np.zeros((1, 1), np.int64), np.array([[np.inf]]), "finite"),
        (np.zeros((1, 1), np.int64), np.array([[-0.01]]), "\\[0, 1\\]"),
        (np.zeros((1, 1), np.int64), np.array([[1.01]]), "\\[0, 1\\]"),
        ([[0], [0, 1]], [[1.0], [1.0, 1.0]], "rectangular"),
    ],
)
def test_constructed_mapping_rejects_malformed_predictions(raw, probabilities, reason):
    with pytest.raises(InputError, match=reason):
        map_prediction(raw, probabilities, _classes(), 0.5)


@pytest.mark.parametrize(
    ("classes", "reason"),
    [
        ([], "explicit"),
        ([{"id": 11, "source_ids": [0]}], "SemanticClass"),
        ([_class(), _class(11, "other", [2])], "unique"),
        ([_class(), _class(12, " LAND ", [2])], "unique"),
        ([_class(name="Straße"), _class(12, " STRASSE ", [2])], "unique"),
        ([_class(name="ς"), _class(12, "σ", [2])], "unique"),
        ([_class(name=" ")], "nonblank"),
        ([_class(source_ids=[0, 0])], "unique"),
        ([_class(), _class(12, "other", [1])], "unique"),
        ([_class().model_copy(update={"id": 0})], "class IDs"),
        ([_class().model_copy(update={"id": 65535})], "class IDs"),
        ([_class().model_copy(update={"id": True})], "class IDs"),
        ([_class().model_copy(update={"name": None})], "nonblank"),
        ([_class().model_copy(update={"source_ids": [-1]})], "nonnegative"),
        ([_class().model_copy(update={"source_ids": [True]})], "nonnegative"),
        ([_class().model_copy(update={"source_ids": []})], "requires source_ids"),
    ],
)
def test_constructed_mapping_rejects_ambiguous_or_invalid_configuration(classes, reason):
    with pytest.raises(InputError, match=reason):
        map_prediction(np.zeros((1, 1), np.int64), np.ones((1, 1)), classes, 0.5)


class _Tensor:
    """Small NumPy-backed torch boundary double, never a neural runtime claim."""

    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def is_floating_point(self):
        return self.data.dtype.kind == "f"

    def float(self):
        return _Tensor(self.data.astype(np.float32))

    def max(self, dim):
        return _Tensor(self.data.max(axis=dim)), _Tensor(self.data.argmax(axis=dim))

    def __getitem__(self, key):
        return _Tensor(self.data[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.data


def _constructed_logits():
    return np.array([[[[4, 0]], [[0, 2]], [[-2, -2]]]], dtype=np.float32)


def _constructed_resized_logits():
    # Closed-form bilinear values for width 2 -> 3, height 1 -> 2;
    # align_corners=False yields endpoint, midpoint, endpoint in this unit case.
    return np.array([
        [[[4, 2, 0], [4, 2, 0]], [[0, 1, 2], [0, 1, 2]], [[-2, -2, -2], [-2, -2, -2]]]
    ], dtype=np.float32)


def _expected_winning_probabilities():
    # Independent analytical values: probability of each winning raw channel.
    return np.array([
        np.exp(4) / (np.exp(4) + np.exp(0) + np.exp(-2)),
        np.exp(2) / (np.exp(2) + np.exp(1) + np.exp(-2)),
        np.exp(2) / (np.exp(0) + np.exp(2) + np.exp(-2)),
    ], dtype=np.float32)


def _install_runtime(monkeypatch, *, num_labels=3, id2label=None, torch_runtime=None):
    """Isolate optional loader APIs; directory contents are not a checkpoint."""
    if torch_runtime is None:
        torch_runtime = ModuleType("torch")
        torch_runtime.Tensor = _Tensor
        torch_runtime.isfinite = lambda tensor: np.isfinite(tensor.data)
        torch_runtime.inference_mode = Mock(side_effect=nullcontext)
        resized = _Tensor(_constructed_resized_logits())
        interpolate = Mock(return_value=resized)
        torch_runtime.nn = SimpleNamespace(functional=SimpleNamespace(interpolate=interpolate))

        def softmax(tensor, dim):
            shifted = tensor.data - tensor.data.max(axis=dim, keepdims=True)
            exponent = np.exp(shifted)
            return _Tensor(exponent / exponent.sum(axis=dim, keepdims=True))

        torch_runtime.softmax = Mock(side_effect=softmax)
        logits = _Tensor(_constructed_logits())
    else:
        resized = None
        interpolate = None
        logits = torch_runtime.tensor(_constructed_logits())
    monkeypatch.setitem(sys.modules, "torch", torch_runtime)

    processor = Mock()
    inputs = Mock()
    pixel_values = object()
    inputs.to.return_value = {"pixel_values": pixel_values}
    processor.return_value = inputs
    model = Mock()
    model.config = SimpleNamespace(
        num_labels=num_labels,
        id2label={"2": "Tree", "0": "Bare Earth", "1": "Built Area"}
        if id2label is None else id2label,
    )
    model.to.return_value = model
    model.eval.return_value = model
    model.return_value = SimpleNamespace(logits=logits)
    processor_factory, model_factory = Mock(), Mock()
    processor_factory.from_pretrained.return_value = processor
    model_factory.from_pretrained.return_value = model
    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = processor_factory
    transformers.AutoModelForSemanticSegmentation = model_factory
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return SimpleNamespace(
        torch=torch_runtime, processor=processor, inputs=inputs, model=model,
        processor_factory=processor_factory, model_factory=model_factory,
        pixel_values=pixel_values, resized=resized, interpolate=interpolate,
    )


def _spec(path):
    return {"path": str(path), "output_domain": "semantic_logits"}


def test_semantic_module_import_keeps_optional_runtimes_lazy(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError(f"Optional runtime imported at module import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(semantics)


@pytest.mark.parametrize("domain", [None, "depth", "relative_inverse", "probabilities"])
def test_output_domain_is_rejected_before_optional_runtime_load(monkeypatch, tmp_path, domain):
    runtime = _install_runtime(monkeypatch)
    with pytest.raises(InputError, match="semantic_logits"):
        LocalSegmenter({"path": str(tmp_path), "output_domain": domain}, _classes())
    runtime.processor_factory.from_pretrained.assert_not_called()
    runtime.model_factory.from_pretrained.assert_not_called()


@pytest.mark.parametrize("path", [None, "", "organization/model", "/nonexistent-semantic-unit"])
def test_loader_requires_an_existing_local_directory(monkeypatch, path):
    runtime = _install_runtime(monkeypatch)
    with pytest.raises(InputError, match="local directory"):
        LocalSegmenter(_spec(path) if path is not None else {"output_domain": "semantic_logits"}, _classes())
    runtime.processor_factory.from_pretrained.assert_not_called()
    runtime.model_factory.from_pretrained.assert_not_called()


@pytest.mark.parametrize("missing_name", ["torch", "transformers"])
def test_loader_reports_missing_optional_runtime_without_download(monkeypatch, tmp_path, missing_name):
    _install_runtime(monkeypatch)
    real_import = builtins.__import__

    def missing_runtime(name, *args, **kwargs):
        if name == missing_name:
            raise ModuleNotFoundError(f"{missing_name} missing in isolated dependency unit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_runtime)
    with pytest.raises(DependencyError, match="torch and transformers"):
        LocalSegmenter(_spec(tmp_path), _classes())


def test_local_loader_flags_actual_source_labels_and_no_execution_claim(monkeypatch, tmp_path):
    runtime = _install_runtime(monkeypatch)
    classes = _classes()
    segmenter = LocalSegmenter(_spec(tmp_path), classes)
    for factory in (runtime.processor_factory, runtime.model_factory):
        factory.from_pretrained.assert_called_once_with(
            str(tmp_path), local_files_only=True, trust_remote_code=False
        )
    runtime.model.to.assert_called_once_with("cpu")
    runtime.model.eval.assert_called_once_with()
    runtime.model.assert_not_called()
    assert segmenter.num_labels == 3
    assert segmenter.source_labels == {0: "Bare Earth", 1: "Built Area", 2: "Tree"}
    # Provenance retains checkpoint names, distinct from output class names.
    classes[0].source_ids.append(100)
    assert segmenter.classes[0].source_ids == [0, 1]


@pytest.mark.parametrize("flag", ["do_center_crop", "do_pad", "do_crop_margin"])
def test_loader_rejects_pixel_domain_changing_preprocessing_before_model_load(
    monkeypatch, tmp_path, flag
):
    runtime = _install_runtime(monkeypatch)
    setattr(runtime.processor, flag, True)
    with pytest.raises(InputError, match="must not crop or pad"):
        LocalSegmenter(_spec(tmp_path), _classes())
    runtime.model_factory.from_pretrained.assert_not_called()
    runtime.model.assert_not_called()


def test_resize_only_processor_retains_original_camera_grid(monkeypatch, tmp_path):
    runtime = _install_runtime(monkeypatch)
    runtime.processor.do_center_crop = False
    runtime.processor.do_pad = False
    runtime.processor.do_resize = True
    segmenter = LocalSegmenter(_spec(tmp_path), _classes())
    labels, probability = segmenter.predict(np.zeros((2, 3, 3), np.uint8), .5)
    assert labels.shape == probability.shape == (2, 3)
    np.testing.assert_allclose(probability, np.tile(_expected_winning_probabilities(), (2, 1)), rtol=1e-6)


@pytest.mark.parametrize(
    ("num_labels", "id2label", "reason"),
    [
        (0, {}, "num_labels"),
        (3.0, {}, "num_labels"),
        (True, {}, "num_labels"),
        (3, [], "actual id2label"),
        (3, {0: "a", 1: "b"}, "every num_labels channel"),
        (3, {0: "a", 1: "b", 3: "c"}, "out-of-range"),
        (3, {-1: "a", 1: "b", 2: "c"}, "out-of-range"),
        (3, {0: "a", "0": "duplicate", 1: "b", 2: "c"}, "duplicate"),
        (3, {"channel": "a", 1: "b", 2: "c"}, "integer channel IDs"),
        (3, {False: "a", 1: "b", 2: "c"}, "integer channel IDs"),
        (3, {0: " ", 1: "b", 2: "c"}, "nonblank"),
        (3, {0: 123, 1: "b", 2: "c"}, "nonblank"),
    ],
)
def test_loader_validates_actual_checkpoint_channel_metadata(
    monkeypatch, tmp_path, num_labels, id2label, reason
):
    runtime = _install_runtime(monkeypatch, num_labels=num_labels, id2label=id2label)
    with pytest.raises(InputError, match=reason):
        LocalSegmenter(_spec(tmp_path), _classes())
    runtime.model.assert_not_called()
    runtime.model.to.assert_not_called()


def test_loader_rejects_configured_source_ids_outside_checkpoint_channels(monkeypatch, tmp_path):
    runtime = _install_runtime(monkeypatch)
    with pytest.raises(InputError, match="source_id 3.*checkpoint channels"):
        LocalSegmenter(_spec(tmp_path), [_class(source_ids=[3])])
    runtime.model.assert_not_called()


@pytest.mark.parametrize("missing_name", ["num_labels", "id2label"])
def test_loader_does_not_synthesize_missing_checkpoint_channel_metadata(
    monkeypatch, tmp_path, missing_name
):
    runtime = _install_runtime(monkeypatch)
    delattr(runtime.model.config, missing_name)
    with pytest.raises(InputError, match=missing_name):
        LocalSegmenter(_spec(tmp_path), _classes())
    runtime.model.assert_not_called()


def test_constructed_logits_are_resized_before_softmax_and_never_grouped(monkeypatch, tmp_path):
    runtime = _install_runtime(monkeypatch)
    segmenter = LocalSegmenter(_spec(tmp_path), _classes())
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    labels, confidence = segmenter.predict(rgb, 0.5)
    np.testing.assert_array_equal(labels, np.full((2, 3), 11, np.uint16))
    expected = np.tile(_expected_winning_probabilities(), (2, 1))
    np.testing.assert_allclose(confidence, expected, rtol=1e-6)
    assert labels.dtype == np.uint16
    assert confidence.dtype == np.float32
    runtime.processor.assert_called_once_with(images=rgb, return_tensors="pt")
    runtime.inputs.to.assert_called_once_with("cpu")
    runtime.model.assert_called_once_with(pixel_values=runtime.pixel_values)
    runtime.torch.inference_mode.assert_called_once_with()
    args, kwargs = runtime.interpolate.call_args
    np.testing.assert_array_equal(args[0].data, _constructed_logits())
    assert kwargs == {"size": (2, 3), "mode": "bilinear", "align_corners": False}
    runtime.torch.softmax.assert_called_once_with(runtime.resized, dim=1)
    # Grouped land score would add channel 1 and exceed 0.75 in the middle.
    thresholded_labels, thresholded_confidence = segmenter.predict(rgb, 0.75)
    np.testing.assert_array_equal(thresholded_labels, [[11, 0, 11], [11, 0, 11]])
    assert np.all(thresholded_confidence[:, 1] == 0)


def test_constructed_unmapped_raw_winner_is_not_replaced_by_a_mapped_runner_up(monkeypatch, tmp_path):
    _install_runtime(monkeypatch)
    segmenter = LocalSegmenter(_spec(tmp_path), [_class(source_ids=[0])])
    labels, confidence = segmenter.predict(np.zeros((2, 3, 3), np.uint8), 0)
    np.testing.assert_array_equal(labels, [[11, 11, 0], [11, 11, 0]])
    assert np.all(confidence[:, 2] == 0)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((2, 3), np.uint8), np.zeros((2, 3, 4), np.uint8),
        np.zeros((0, 3, 3), np.uint8), np.zeros((2, 0, 3), np.uint8),
        np.zeros((2, 3, 3), np.float32), np.full((2, 3, 3), np.nan),
        np.zeros((2, 3, 3), np.uint16), [[[0, 0, 0]]],
    ],
)
def test_predict_rejects_invalid_rgb_before_executing_model(monkeypatch, tmp_path, rgb):
    runtime = _install_runtime(monkeypatch)
    segmenter = LocalSegmenter(_spec(tmp_path), _classes())
    with pytest.raises(InputError, match="uint8 HxWx3 RGB"):
        segmenter.predict(rgb, 0.5)
    runtime.processor.assert_not_called()
    runtime.model.assert_not_called()


def test_predict_rejects_threshold_before_executing_model(monkeypatch, tmp_path):
    runtime = _install_runtime(monkeypatch)
    segmenter = LocalSegmenter(_spec(tmp_path), _classes())
    with pytest.raises(InputError, match="min_probability"):
        segmenter.predict(np.zeros((2, 3, 3), np.uint8), np.nan)
    runtime.processor.assert_not_called()
    runtime.model.assert_not_called()


@pytest.mark.parametrize(
    ("logits", "reason"),
    [
        (None, "logits must"),
        (np.zeros((1, 3, 1, 2)), "logits must"),
        (_Tensor(np.zeros((3, 1, 2), np.float32)), "logits must"),
        (_Tensor(np.zeros((2, 3, 1, 2), np.float32)), "logits must"),
        (_Tensor(np.zeros((1, 4, 1, 2), np.float32)), "logits must"),
        (_Tensor(np.zeros((1, 3, 0, 2), np.float32)), "logits must"),
        (_Tensor(np.zeros((1, 3, 1, 2), np.int64)), "logits must"),
        (_Tensor(np.full((1, 3, 1, 2), np.nan)), "nonfinite logits"),
        (_Tensor(np.full((1, 3, 1, 2), np.inf)), "nonfinite logits"),
    ],
)
def test_predict_rejects_invalid_actual_model_logits(monkeypatch, tmp_path, logits, reason):
    runtime = _install_runtime(monkeypatch)
    runtime.model.return_value = SimpleNamespace(logits=logits)
    segmenter = LocalSegmenter(_spec(tmp_path), _classes())
    with pytest.raises(InputError, match=reason):
        segmenter.predict(np.zeros((2, 3, 3), np.uint8), 0.5)
    runtime.interpolate.assert_not_called()
    runtime.torch.softmax.assert_not_called()


def test_predict_rejects_nonfinite_resized_logits(monkeypatch, tmp_path):
    runtime = _install_runtime(monkeypatch)
    runtime.interpolate.return_value = _Tensor(np.full((1, 3, 2, 3), np.nan))
    segmenter = LocalSegmenter(_spec(tmp_path), _classes())
    with pytest.raises(InputError, match="nonfinite resized logits"):
        segmenter.predict(np.zeros((2, 3, 3), np.uint8), 0.5)
    runtime.torch.softmax.assert_not_called()


def test_predict_rejects_wrong_final_image_grid(monkeypatch, tmp_path):
    runtime = _install_runtime(monkeypatch)
    runtime.interpolate.return_value = _Tensor(np.zeros((1, 3, 1, 1), np.float32))
    segmenter = LocalSegmenter(_spec(tmp_path), _classes())
    with pytest.raises(InputError, match="original RGB image shape"):
        segmenter.predict(np.zeros((2, 3, 3), np.uint8), 0.5)


def test_optional_real_torch_constructed_logits_with_isolated_transformers_boundary(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch", reason="Optional torch runtime not installed")
    _install_runtime(monkeypatch, torch_runtime=torch)
    segmenter = LocalSegmenter(_spec(tmp_path), _classes())
    labels, confidence = segmenter.predict(np.zeros((2, 3, 3), np.uint8), 0.5)
    np.testing.assert_array_equal(labels, np.full((2, 3), 11, np.uint16))
    np.testing.assert_allclose(
        confidence, np.tile(_expected_winning_probabilities(), (2, 1)), rtol=1e-6
    )
    assert confidence.dtype == np.float32
