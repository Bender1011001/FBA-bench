import os
import tempfile
from textwrap import dedent

import pytest

from observability.observability_config import ObservabilityConfig


def test_from_env_defaults_and_types(monkeypatch):
    # Clear env to use defaults
    for k in [
        "OBS_ENABLE_TRACE_ANALYSIS",
        "OBS_LLM_FRIENDLY_TOOLS",
        "OBS_AUTO_ERROR_CORRECTION",
        "OBS_REAL_TIME_ALERTS",
        "OBS_INSIGHT_INTERVAL",
        "OBS_ERROR_WINDOW",
        "OBS_PERF_THRESHOLD",
        "OBS_TRACE_RETENTION_DAYS",
    ]:
        monkeypatch.delenv(k, raising=False)

    cfg = ObservabilityConfig.from_env()
    assert isinstance(cfg.enable_trace_analysis, bool)
    assert isinstance(cfg.llm_friendly_tools, bool)
    assert isinstance(cfg.auto_error_correction, bool)
    assert isinstance(cfg.real_time_alerts, bool)
    assert isinstance(cfg.insight_generation_interval, int)
    assert isinstance(cfg.error_pattern_window, int)
    assert isinstance(cfg.performance_alert_threshold, float)
    assert isinstance(cfg.trace_retention_days, int)
    # Validate passes
    cfg.validate()


def test_from_env_with_values(monkeypatch):
    monkeypatch.setenv("OBS_ENABLE_TRACE_ANALYSIS", "false")
    monkeypatch.setenv("OBS_LLM_FRIENDLY_TOOLS", "0")
    monkeypatch.setenv("OBS_AUTO_ERROR_CORRECTION", "no")
    monkeypatch.setenv("OBS_REAL_TIME_ALERTS", "1")
    monkeypatch.setenv("OBS_INSIGHT_INTERVAL", "200")
    monkeypatch.setenv("OBS_ERROR_WINDOW", "75")
    monkeypatch.setenv("OBS_PERF_THRESHOLD", "0.65")
    monkeypatch.setenv("OBS_TRACE_RETENTION_DAYS", "10")

    cfg = ObservabilityConfig.from_env()
    assert cfg.enable_trace_analysis is False
    assert cfg.llm_friendly_tools is False
    assert cfg.auto_error_correction is False
    assert cfg.real_time_alerts is True
    assert cfg.insight_generation_interval == 200
    assert cfg.error_pattern_window == 75
    assert pytest.approx(cfg.performance_alert_threshold, rel=1e-6) == 0.65
    assert cfg.trace_retention_days == 10
    cfg.validate()


def test_from_yaml_success_and_validation():
    content = dedent(
        """
        enable_trace_analysis: true
        llm_friendly_tools: false
        auto_error_correction: true
        real_time_alerts: true
        insight_generation_interval: 150
        error_pattern_window: 60
        performance_alert_threshold: 0.7
        trace_retention_days: 45
        """
    )
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        cfg = ObservabilityConfig.from_yaml(tmp.name)

    assert cfg.enable_trace_analysis is True
    assert cfg.llm_friendly_tools is False
    assert cfg.auto_error_correction is True
    assert cfg.real_time_alerts is True
    assert cfg.insight_generation_interval == 150
    assert cfg.error_pattern_window == 60
    assert pytest.approx(cfg.performance_alert_threshold, rel=1e-6) == 0.7
    assert cfg.trace_retention_days == 45
    cfg.validate()


def test_from_yaml_invalid_file_raises(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ValueError):
        ObservabilityConfig.from_yaml(str(missing))


def test_from_yaml_invalid_values_raise():
    bad_content = dedent(
        """
        insight_generation_interval: 0
        error_pattern_window: -1
        performance_alert_threshold: 2.0
        trace_retention_days: -5
        """
    )
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as tmp:
        tmp.write(bad_content)
        tmp.flush()
        with pytest.raises(ValueError):
            ObservabilityConfig.from_yaml(tmp.name)