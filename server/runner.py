"""Adapter to the executable shared pipeline, not an external job service."""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException

from server.settings import Settings


@dataclass
class PreparedRun:
    run_id: str
    bundle: Any
    config: Any

    def execute(self) -> str:
        from drishti.stages.runner import StageRunner

        result = StageRunner(self.bundle, self.config).run()
        return result.manifest["status"]


class LocalRunService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def doctor(self) -> dict:
        try:
            from drishti.runtime.probe import probe_environment
        except ImportError as exc:
            raise HTTPException(503, f"Install the DRISHTI Python package: {exc}") from exc
        return {"environment": probe_environment(), "offline": True,
                "capability_detection_is_execution": False,
                "application": self.settings.model_dump(mode="json")}

    def prepare(self, dataset: Path, profile: str) -> PreparedRun:
        try:
            from drishti.bundle import Bundle
            from drishti.bundle.core import input_paths
            from drishti.config.loader import load_config
            from drishti.errors import DrishtiError
        except ImportError as exc:
            raise HTTPException(503, f"Install the DRISHTI Python package: {exc}") from exc
        try:
            config = load_config(
                dataset=dataset, profile=self.settings.profiles_root.resolve() / f"{profile}.yaml",
                overrides={"run": {"output_root": str(self.settings.runs_root.resolve())}},
                # Server settings share the DRISHTI prefix but are not pipeline sections.
                environ={k: v for k, v in os.environ.items() if not k.startswith("DRISHTI_SERVER_")},
            )
            descriptor = config.dataset
            if descriptor is None:
                raise ValueError("A dataset descriptor with video and telemetry is required")
            for role, path in input_paths(config).items():
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Missing or empty supplied input {role}: {path}")
            if descriptor.telemetry.altitude_datum == "relative":
                raise ValueError("Relative altitude is not an absolute metric reference; provide absolute telemetry")
            identifier = uuid.uuid4().hex
            root = self.settings.runs_root.resolve() / identifier
            bundle = Bundle.create(root, config, run_id=identifier)
            return PreparedRun(identifier, bundle, bundle.config())
        except (ValueError, OSError, DrishtiError, yaml.YAMLError) as exc:
            raise HTTPException(422, str(exc)) from exc
