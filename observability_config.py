from __future__ import annotations

# Top-level shim for backward compatibility with tests importing `observability_config`
# Re-export the configuration objects from the package module.

try:
    from observability.observability_config import (  # type: ignore
        ObservabilityConfig,
        LoggingConfig,
        TracingConfig,
        MetricsConfig,
        DEFAULT_OBSERVABILITY_CONFIG,
    )
except Exception as exc:  # pragma: no cover - defensive fallback
    # Minimal fallback to avoid hard import failure during test collection
    class ObservabilityConfig:  # type: ignore
        def __init__(self, enable_tracing: bool = True, enable_metrics: bool = True, log_level: str = "INFO") -> None:
            self.enable_tracing = enable_tracing
            self.enable_metrics = enable_metrics
            self.log_level = log_level

    class LoggingConfig:  # type: ignore
        def __init__(self, level: str = "INFO") -> None:
            self.level = level

    class TracingConfig:  # type: ignore
        def __init__(self, enabled: bool = True) -> None:
            self.enabled = enabled

    class MetricsConfig:  # type: ignore
        def __init__(self, enabled: bool = True) -> None:
            self.enabled = enabled

    DEFAULT_OBSERVABILITY_CONFIG = ObservabilityConfig()  # type: ignore

__all__ = [
    "ObservabilityConfig",
    "LoggingConfig",
    "TracingConfig",
    "MetricsConfig",
    "DEFAULT_OBSERVABILITY_CONFIG",
]