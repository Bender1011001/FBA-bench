"""
Base classes for multi-dimensional evaluation metrics.

This module provides the foundation for all benchmarking metrics, including
abstract base classes and concrete implementations for cognitive, business,
technical, and ethical metrics.
"""

from __future__ import annotations

import abc
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

# Re-export MetricResult so tests that import from benchmarking.metrics.base work
from ..core.results import MetricResult


@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    description: str
    unit: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    weight: float = 1.0
    enabled: bool = True


class MetricCategory(str, Enum):
    cognitive = "cognitive"
    business = "business"
    technical = "technical"
    ethical = "ethical"


class BaseMetric(abc.ABC):
    """
    Abstract base class for all metrics.

    This class defines the interface that all metrics must implement,
    including methods for calculation, validation, and aggregation.
    """

    def __init__(self, config: MetricConfig):
        """
        Initialize the metric.

        Args:
            config: Metric configuration
        """
        self.config = config
        self._values: List[float] = []
        self._timestamps: List[datetime] = []
        self._metadata: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Get the metric name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get the metric description."""
        return self.config.description

    @property
    def unit(self) -> str:
        """Get the metric unit."""
        return self.config.unit

    @property
    def values(self) -> List[float]:
        """Get all recorded values."""
        return self._values.copy()

    @property
    def timestamps(self) -> List[datetime]:
        """Get all recorded timestamps."""
        return self._timestamps.copy()

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Get all recorded metadata."""
        return self._metadata.copy()

    def record_value(self, value: float, timestamp: datetime | None = None, metadata: Dict[str, Any] | None = None) -> None:
        """
        Record a metric value.

        Args:
            value: The metric value
            timestamp: When the value was recorded (defaults to now)
            metadata: Additional metadata about the value
        """
        if timestamp is None:
            timestamp = datetime.now()

        if metadata is None:
            metadata = {}

        # Validate value range
        if self.config.min_value is not None and value < self.config.min_value:
            raise ValueError(f"Value {value} is below minimum {self.config.min_value}")

        if self.config.max_value is not None and value > self.config.max_value:
            raise ValueError(f"Value {value} is above maximum {self.config.max_value}")

        self._values.append(value)
        self._timestamps.append(timestamp)
        self._metadata.append(metadata)

    def clear(self) -> None:
        """Clear all recorded values."""
        self._values.clear()
        self._timestamps.clear()
        self._metadata.clear()

    @abc.abstractmethod
    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate the metric value from raw data.

        Args:
            data: Raw data for calculation

        Returns:
            Calculated metric value
        """
        raise NotImplementedError

    def get_latest_value(self) -> Optional[float]:
        """Get the most recently recorded value."""
        return self._values[-1] if self._values else None

    def get_average(self) -> Optional[float]:
        """Get the average of all recorded values."""
        return statistics.mean(self._values) if self._values else None

    def get_median(self) -> Optional[float]:
        """Get the median of all recorded values."""
        return statistics.median(self._values) if self._values else None

    def get_std_dev(self) -> Optional[float]:
        """Get the standard deviation of all recorded values."""
        return statistics.stdev(self._values) if len(self._values) > 1 else None

    def get_min(self) -> Optional[float]:
        """Get the minimum recorded value."""
        return min(self._values) if self._values else None

    def get_max(self) -> Optional[float]:
        """Get the maximum recorded value."""
        return max(self._values) if self._values else None

    def get_count(self) -> int:
        """Get the number of recorded values."""
        return len(self._values)

    def to_metric_results(self) -> List[MetricResult]:
        """
        Convert recorded values to MetricResult objects.

        Returns:
            List of metric results
        """
        results: List[MetricResult] = []
        for value, timestamp, metadata in zip(self._values, self._timestamps, self._metadata):
            results.append(MetricResult(
                name=self.name,
                value=value,
                unit=self.unit,
                timestamp=timestamp,
                metadata=metadata
            ))
        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the metric.

        Returns:
            Summary statistics
        """
        if not self._values:
            return {"error": "No values recorded"}

        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "count": self.get_count(),
            "latest": self.get_latest_value(),
            "average": self.get_average(),
            "median": self.get_median(),
            "std_dev": self.get_std_dev(),
            "min": self.get_min(),
            "max": self.get_max(),
            "target": self.config.target_value
        }


class CognitiveMetrics(BaseMetric):
    """
    Metrics for evaluating cognitive capabilities of agents.

    This includes reasoning, planning, memory, and other cognitive functions.
    """

    def __init__(self, config: MetricConfig | None = None):
        if config is None:
            config = MetricConfig(
                name="cognitive_performance",
                description="Overall cognitive performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=80.0
            )
        super().__init__(config)

    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate cognitive performance score.

        Args:
            data: dict containing submetric values like reasoning_score, planning_score, memory_score, learning_score

        Returns:
            float overall cognitive score (0-100)
        """
        r = float(data.get("reasoning_score", data.get("reasoning", 0.0)) or 0.0)
        p = float(data.get("planning_score", data.get("planning", 0.0)) or 0.0)
        m = float(data.get("memory_score", data.get("memory", 0.0)) or 0.0)
        l = float(data.get("learning_score", data.get("learning", 0.0)) or 0.0)
        score = max(0.0, min(100.0, (r + p + m + l) / 4.0))
        self.record_value(score, metadata={"r": r, "p": p, "m": m, "l": l})
        return score


class BusinessMetrics(BaseMetric):
    """Business-domain performance metrics (e.g., profit, revenue growth)."""

    def __init__(self, config: MetricConfig | None = None):
        if config is None:
            config = MetricConfig(
                name="business_performance",
                description="Overall business performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=75.0,
            )
        super().__init__(config)

    def calculate(self, data: Dict[str, Any]) -> float:
        revenue = float(data.get("revenue", 0.0) or 0.0)
        profit = float(data.get("profit", 0.0) or 0.0)
        margin_pct = float(data.get("margin_pct", (profit / revenue * 100.0) if revenue else 0.0) or 0.0)
        score = max(0.0, min(100.0, margin_pct))
        self.record_value(score, metadata={"revenue": revenue, "profit": profit})
        return score


class TechnicalMetrics(BaseMetric):
    """Technical performance metrics (latency, success rates, etc.)."""

    def __init__(self, config: MetricConfig | None = None):
        if config is None:
            config = MetricConfig(
                name="technical_performance",
                description="Overall technical performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=80.0,
            )
        super().__init__(config)

    def calculate(self, data: Dict[str, Any]) -> float:
        latency_ms = float(data.get("latency_ms", 0.0) or 0.0)
        # Score is 100 at 0ms, linearly decays to 0 at 3000ms
        score = max(0.0, min(100.0, 100.0 * (1.0 - (latency_ms / 3000.0))))
        self.record_value(score, metadata={"latency_ms": latency_ms})
        return score


class EthicalMetrics(BaseMetric):
    """Ethical/compliance metrics (policy violations, safety, etc.)."""

    def __init__(self, config: MetricConfig | None = None):
        if config is None:
            config = MetricConfig(
                name="ethical_compliance",
                description="Compliance and safety performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=90.0,
            )
        super().__init__(config)

    def calculate(self, data: Dict[str, Any]) -> float:
        violations = int(data.get("policy_violations", 0) or 0)
        score = max(0.0, min(100.0, 100.0 - (violations * 10.0)))
        self.record_value(score, metadata={"policy_violations": violations})
        return score