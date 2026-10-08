"""Actionable pipeline errors."""


class DrishtiError(RuntimeError):
    """Base for recoverable, reportable failures."""


class InputError(DrishtiError):
    """Required input is absent, invalid or temporally unsupported."""


class DependencyError(DrishtiError):
    """An optional runtime or tool is required for the requested algorithm."""


class ReconstructionError(DrishtiError):
    """No viable reconstruction can be produced from available observations."""
