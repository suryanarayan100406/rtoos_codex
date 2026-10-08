"""Typed application settings. Override with DRISHTI_SERVER_<FIELD> environment variables."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs_root: Path = Path("runs")
    dataset_root: Path = Path(".")
    profiles_root: Path = Path("configs/profiles")
    viewer_dist: Path = Path("viewer/dist")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=8)
    max_points: int = Field(default=100_000, ge=1, le=1_000_000)
    max_npz_bytes: int = Field(default=256_000_000, ge=1024)
    max_volume_cells: int = Field(default=1_000_000, ge=1)
    max_measure_points: int = Field(default=512, ge=3, le=4096)
    max_json_bytes: int = Field(default=32_000_000, ge=1024)

    @classmethod
    def from_env(cls) -> "Settings":
        prefix = "DRISHTI_SERVER_"
        return cls.model_validate({k[len(prefix):].lower(): v for k, v in os.environ.items()
                                   if k.startswith(prefix)})
