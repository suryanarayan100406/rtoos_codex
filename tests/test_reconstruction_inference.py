"""Local inference IO contracts; optional IR case is a runtime unit, not a trained model."""

import hashlib

import numpy as np
import pytest
import yaml

from drishti.errors import InputError
from drishti.models.inference import LocalDepth, verified_file
from drishti.models.registry import Registry


def test_output_domain_must_be_explicit_before_loading_any_runtime():
    with pytest.raises(InputError, match="relative_inverse"):
        LocalDepth({"path": "/nonexistent", "output_domain": None}, "openvino")
    with pytest.raises(InputError, match="relative_inverse"):
        LocalDepth({"path": "/nonexistent", "output_domain": "depth"}, "torch")


def test_ir_paths_must_be_in_verified_registry_inventory(tmp_path):
    (tmp_path/"model.xml").write_text("unit-test IO file, not a model")
    spec = {"path": str(tmp_path),
            "files": {"model.xml": hashlib.sha256((tmp_path/"model.xml").read_bytes()).hexdigest()}}
    assert verified_file(spec, "model.xml") == tmp_path/"model.xml"
    with pytest.raises(InputError, match="verified"):
        verified_file(spec, "unregistered.bin")
    with pytest.raises(InputError, match="escapes"):
        verified_file(spec, "../outside.bin")


def test_incomplete_aspect_ir_preprocessing_fails_clearly_before_runtime_load():
    spec = {"path": "/nonexistent", "output_domain": "relative_inverse",
            "preprocessing": {"layout": "NCHW", "color": "RGB", "scale": 1/255,
                              "mean": [0.5]*3, "std": [0.5]*3, "resize": "aspect_multiple"}}
    with pytest.raises(InputError, match="positive integer multiple"):
        LocalDepth(spec, "openvino")


def test_final_registry_fields_interoperate_with_ir_file_resolution(tmp_path):
    # These are IO/schema unit files, not a trained checkpoint or acceptance input.
    root = tmp_path/"ir_unit"
    root.mkdir()
    for name in ("model.xml", "model.bin"):
        (root/name).write_bytes(f"schema unit file: {name}".encode())
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
    spec = {"path": "ir_unit", "revision": hashes["model.xml"], "license": "MIT",
            "files": hashes, "runtime": "openvino", "model_file": "model.xml",
            "weights_file": "model.bin", "output_name": "relative_inverse",
            "output_domain": "relative_inverse", "preprocessing": {
                "layout": "NCHW", "color": "RGB", "scale": 1/255,
                "mean": [0.5]*3, "std": [0.5]*3, "resize": "aspect_multiple", "multiple": 14}}
    registry_path = tmp_path/"registry.yaml"
    registry_path.write_text(yaml.safe_dump({"models": {"ir_io_unit": spec}}))
    resolved = Registry(registry_path, tmp_path).resolve("ir_io_unit")
    assert resolved["preprocessing"]["multiple"] == 14
    assert "height" not in resolved["preprocessing"]
    assert verified_file(resolved, resolved["model_file"]) == root/"model.xml"
    assert verified_file(resolved, resolved["weights_file"]) == root/"model.bin"


def test_real_openvino_ir_inference_and_explicit_preprocessing(tmp_path):
    ov = pytest.importorskip("openvino", reason="OpenVINO runtime not installed")
    from openvino import opset13 as ops

    image = ops.parameter([1, 3, 4, 4], np.float32, name="image")
    prediction = ops.reduce_mean(image, ops.constant(np.array([1], np.int64)), keep_dims=True)
    model = ov.Model([prediction], [image], "preprocessing_io_unit")
    xml, binary = tmp_path/"model.xml", tmp_path/"model.bin"
    ov.serialize(model, str(xml), str(binary))
    spec = {"path": str(tmp_path), "output_domain": "relative_inverse",
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (xml, binary)},
            "preprocessing": {"layout": "NCHW", "color": "RGB", "scale": 1/255,
                              "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "resize": "stretch"}}
    rgb = np.full((8, 6, 3), [64, 128, 192], np.uint8)
    output = LocalDepth(spec, "openvino", openvino_device="CPU").predict(rgb)
    assert output.shape == (8, 6)
    expected = np.mean((np.array([64, 128, 192])/255-0.5)/0.5)
    np.testing.assert_allclose(output, expected, atol=2e-7)
