"""Uniform stage contracts shared by the runner and implementations."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class StageResult:
    outputs: dict[str, str]
    metrics: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    degraded_reason: str | None = None
    models: list[dict[str, Any]] = field(default_factory=list)


class Stage(Protocol):
    name: str
    requires: tuple[str, ...]
    config_sections: tuple[str, ...]

    def run(self, bundle: Any, cfg: Any, runtime: Any) -> StageResult: ...
