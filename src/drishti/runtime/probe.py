"""Measured host capabilities. Detection is never recorded as algorithm execution."""

import importlib.metadata
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import psutil

from drishti.config.models import Config
from drishti.errors import DependencyError

PACKAGES = ("drishti", "pydantic", "numpy", "av", "opencv-python-headless", "pycolmap", "gtsam", "pyproj", "scipy", "rasterio", "open3d", "trimesh", "laspy", "transformers", "torch", "safetensors", "openvino", "pymavlink", "Pillow")


def probe_environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    memory = psutil.virtual_memory()
    environment: dict[str, Any] = {
        "host": platform.node(), "os": platform.platform(), "machine": platform.machine(),
        "cpu": platform.processor() or None, "cpu_logical": os.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False), "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available, "python": sys.version,
        "python_executable": sys.executable, "versions": versions,
        "cuda_available": False, "cuda_devices": [], "openvino_devices": [],
        "tools": {name: shutil.which(name) for name in ("blender", "colmap", "glomap", "PotreeConverter")},
        "probe_errors": {},
    }
    if versions["torch"]:
        try:
            import torch
            environment["cuda_available"] = bool(torch.cuda.is_available())
            environment["torch_cuda_version"] = torch.version.cuda
            if environment["cuda_available"]:
                environment["cuda_devices"] = [
                    {"index": index, "name": torch.cuda.get_device_properties(index).name, "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory}
                    for index in range(torch.cuda.device_count())
                ]
        except Exception as exc:
            environment["probe_errors"]["torch"] = str(exc)
    if versions["openvino"]:
        try:
            import openvino
            core = openvino.Core()
            environment["openvino_devices"] = [{"id": device, "name": core.get_property(device, "FULL_DEVICE_NAME")} for device in core.available_devices]
        except Exception as exc:
            environment["probe_errors"]["openvino"] = str(exc)
    return environment


@dataclass
class Runtime:
    device: str
    environment: dict[str, Any]

    @classmethod
    def detect(cls, cfg: Config | None = None) -> "Runtime":
        environment = probe_environment()
        requested = cfg.compute.device if cfg else "auto"
        if requested == "cuda" and not environment["cuda_available"]:
            raise DependencyError("compute.device=cuda requested but an operational torch CUDA device was not detected")
        device = "cuda" if requested != "cpu" and environment["cuda_available"] else "cpu"
        environment["selected_torch_device"] = device
        environment["requested_environment_label"] = cfg.compute.environment if cfg else None
        return cls(device, environment)


def seed_runtime(seed: int) -> None:
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    # Do not import the neural stack for CPU-only stages.
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
