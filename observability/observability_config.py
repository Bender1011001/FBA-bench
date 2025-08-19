from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional import
    yaml = None  # type: ignore


@dataclass
class ObservabilityConfig:
    """
    Configuration settings for observability features in FBA-Bench.
    """
    enable_trace_analysis: bool = True
    llm_friendly_tools: bool = True
    auto_error_correction: bool = True
    real_time_alerts: bool = False
    insight_generation_interval: int = 100  # ticks
    error_pattern_window: int = 50  # events
    performance_alert_threshold: float = 0.8  # 0.0 - 1.0
    trace_retention_days: int = 30  # days

    def validate(self) -> None:
        """Validate settings; raise ValueError with informative messages."""
        if self.insight_generation_interval <= 0:
            raise ValueError("insight_generation_interval must be greater than 0")
        if self.error_pattern_window <= 0:
            raise ValueError("error_pattern_window must be greater than 0")
        if not (0.0 <= self.performance_alert_threshold <= 1.0):
            raise ValueError("performance_alert_threshold must be between 0.0 and 1.0")
        if self.trace_retention_days < 0:
            raise ValueError("trace_retention_days cannot be negative")

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """
        Load configuration from environment variables with safe defaults.

        Environment variables:
          - OBS_ENABLE_TRACE_ANALYSIS
          - OBS_LLM_FRIENDLY_TOOLS
          - OBS_AUTO_ERROR_CORRECTION
          - OBS_REAL_TIME_ALERTS
          - OBS_INSIGHT_INTERVAL
          - OBS_ERROR_WINDOW
          - OBS_PERF_THRESHOLD
          - OBS_TRACE_RETENTION_DAYS
        """
        def _b(name: str, default: bool) -> bool:
            return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")

        def _i(name: str, default: int) -> int:
            val = os.getenv(name)
            try:
                return int(val) if val is not None else default
            except Exception:
                return default

        def _f(name: str, default: float) -> float:
            val = os.getenv(name)
            try:
                return float(val) if val is not None else default
            except Exception:
                return default

        cfg = cls(
            enable_trace_analysis=_b("OBS_ENABLE_TRACE_ANALYSIS", True),
            llm_friendly_tools=_b("OBS_LLM_FRIENDLY_TOOLS", True),
            auto_error_correction=_b("OBS_AUTO_ERROR_CORRECTION", True),
            real_time_alerts=_b("OBS_REAL_TIME_ALERTS", False),
            insight_generation_interval=_i("OBS_INSIGHT_INTERVAL", 100),
            error_pattern_window=_i("OBS_ERROR_WINDOW", 50),
            performance_alert_threshold=_f("OBS_PERF_THRESHOLD", 0.8),
            trace_retention_days=_i("OBS_TRACE_RETENTION_DAYS", 30),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "ObservabilityConfig":
        """
        Load configuration from a YAML file.

        The YAML file may contain keys matching the dataclass fields.

        Raises:
            ValueError: if the file is missing, YAML is invalid, or validation fails.
        """
        if not path or not isinstance(path, str):
            raise ValueError("path must be a non-empty string")
        if not os.path.exists(path):
            raise ValueError(f"YAML config file not found: {path}")
        if yaml is None:
            raise ValueError("PyYAML is required to load YAML configurations")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to read YAML file: {e}")

        if not isinstance(data, dict):
            raise ValueError("YAML content must be a mapping/object")

        # Map keys defensively
        def _get_bool(k: str, default: bool) -> bool:
            v = data.get(k, default)
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ("1", "true", "yes", "on")
            return bool(v)

        def _get_int(k: str, default: int) -> int:
            v = data.get(k, default)
            try:
                return int(v)
            except Exception:
                return default

        def _get_float(k: str, default: float) -> float:
            v = data.get(k, default)
            try:
                return float(v)
            except Exception:
                return default

        cfg = cls(
            enable_trace_analysis=_get_bool("enable_trace_analysis", True),
            llm_friendly_tools=_get_bool("llm_friendly_tools", True),
            auto_error_correction=_get_bool("auto_error_correction", True),
            real_time_alerts=_get_bool("real_time_alerts", False),
            insight_generation_interval=_get_int("insight_generation_interval", 100),
            error_pattern_window=_get_int("error_pattern_window", 50),
            performance_alert_threshold=_get_float("performance_alert_threshold", 0.8),
            trace_retention_days=_get_int("trace_retention_days", 30),
        )
        cfg.validate()
        return cfg