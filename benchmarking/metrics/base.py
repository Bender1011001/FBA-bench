"""
Base classes for multi-dimensional evaluation metrics.

This module provides the foundation for all benchmarking metrics, including
abstract base classes and concrete implementations for cognitive, business,
technical, and ethical metrics.
"""

import abc
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

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
    
    def record_value(self, value: float, timestamp: datetime = None, metadata: Dict[str, Any] = None) -> None:
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
        pass
    
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
        results = []
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
    
    def __init__(self, config: MetricConfig = None):
        """
        Initialize cognitive metrics.
        
        Args:
            config: Metric configuration (uses defaults if None)
        """
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
        
        # Sub-metrics
        self.reasoning_score = MetricConfig(
            name="reasoning_score",
            description="Logical reasoning capability",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
        
        self.planning_score = MetricConfig(
            name="planning_score", 
            description="Planning and strategy capability",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
        
        self.memory_score = MetricConfig(
            name="memory_score",
            description="Memory and recall capability", 
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
        
        self.learning_score = MetricConfig(
            name="learning_score",
            description="Learning and adaptation capability",
            unit="score", 
            min_value=0.0,
            max_value=100.0
        )
    
    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate cognitive performance score.
        
        Args:
            data: Data containing cognitive metrics
            
        Returns:
            Overall cognitive score
        """
        # Extract sub-metric values
        reasoning = data.get('reasoning_score', 0.0)
        planning = data.get('planning_score', 0.0)
        memory = data.get('memory_score', 0.0)
        learning = data.get('learning_score', 0.0)
        
        # Calculate weighted average
        weights = {
            'reasoning': 0.3,
            'planning': 0.3,
            'memory': 0.2,
            'learning': 0.2
        }
        
        overall_score = (
            reasoning * weights['reasoning'] +
            planning * weights['planning'] +
            memory * weights['memory'] +
            learning * weights['learning']
        )
        
        return overall_score
    
    def calculate_reasoning(self, data: Dict[str, Any]) -> float:
        """
        Calculate reasoning score.
        
        Args:
            data: Data containing reasoning metrics
            
        Returns:
            Reasoning score
        """
        logic_score = data.get('logic_score', 0.0)
        inference_score = data.get('inference_score', 0.0)
        problem_solving_score = data.get('problem_solving_score', 0.0)
        
        return (logic_score + inference_score + problem_solving_score) / 3.0
    
    def calculate_planning(self, data: Dict[str, Any]) -> float:
        """
        Calculate planning score.
        
        Args:
            data: Data containing planning metrics
            
        Returns:
            Planning score
        """
        goal_setting_score = data.get('goal_setting_score', 0.0)
        strategy_score = data.get('strategy_score', 0.0)
        adaptability_score = data.get('adaptability_score', 0.0)
        
        return (goal_setting_score + strategy_score + adaptability_score) / 3.0
    
    def calculate_memory(self, data: Dict[str, Any]) -> float:
        """
        Calculate memory score.
        
        Args:
            data: Data containing memory metrics
            
        Returns:
            Memory score
        """
        recall_score = data.get('recall_score', 0.0)
        retention_score = data.get('retention_score', 0.0)
        organization_score = data.get('organization_score', 0.0)
        
        return (recall_score + retention_score + organization_score) / 3.0
    
    def calculate_learning(self, data: Dict[str, Any]) -> float:
        """
        Calculate learning score.
        
        Args:
            data: Data containing learning metrics
            
        Returns:
            Learning score
        """
        adaptation_score = data.get('adaptation_score', 0.0)
        improvement_score = data.get('improvement_score', 0.0)
        generalization_score = data.get('generalization_score', 0.0)
        
        return (adaptation_score + improvement_score + generalization_score) / 3.0


class BusinessMetrics(BaseMetric):
    """
    Metrics for evaluating business performance of agents.
    
    This includes ROI, efficiency, strategic alignment, and other business metrics.
    """
    
    def __init__(self, config: MetricConfig = None):
        """
        Initialize business metrics.
        
        Args:
            config: Metric configuration (uses defaults if None)
        """
        if config is None:
            config = MetricConfig(
                name="business_performance",
                description="Overall business performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=75.0
            )
        
        super().__init__(config)
        
        # Sub-metrics
        self.roi_metric = MetricConfig(
            name="roi_score",
            description="Return on investment",
            unit="percentage",
            min_value=0.0,
            max_value=100.0
        )
        
        self.efficiency_metric = MetricConfig(
            name="efficiency_score",
            description="Operational efficiency",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
        
        self.strategic_alignment_metric = MetricConfig(
            name="strategic_alignment_score",
            description="Strategic alignment with goals",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
    
    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate business performance score.
        
        Args:
            data: Data containing business metrics
            
        Returns:
            Overall business score
        """
        # Extract sub-metric values
        roi = data.get('roi_score', 0.0)
        efficiency = data.get('efficiency_score', 0.0)
        strategic_alignment = data.get('strategic_alignment_score', 0.0)
        
        # Calculate weighted average
        weights = {
            'roi': 0.4,
            'efficiency': 0.3,
            'strategic_alignment': 0.3
        }
        
        overall_score = (
            roi * weights['roi'] +
            efficiency * weights['efficiency'] +
            strategic_alignment * weights['strategic_alignment']
        )
        
        return overall_score
    
    def calculate_roi(self, data: Dict[str, Any]) -> float:
        """
        Calculate ROI score.
        
        Args:
            data: Data containing ROI metrics
            
        Returns:
            ROI score
        """
        investment = data.get('investment', 0.0)
        returns = data.get('returns', 0.0)
        
        if investment <= 0:
            return 0.0
        
        roi = ((returns - investment) / investment) * 100
        return min(100.0, max(0.0, roi))
    
    def calculate_efficiency(self, data: Dict[str, Any]) -> float:
        """
        Calculate efficiency score.
        
        Args:
            data: Data containing efficiency metrics
            
        Returns:
            Efficiency score
        """
        resources_used = data.get('resources_used', 0.0)
        output_produced = data.get('output_produced', 0.0)
        
        if resources_used <= 0:
            return 0.0
        
        efficiency = (output_produced / resources_used) * 100
        return min(100.0, max(0.0, efficiency))
    
    def calculate_strategic_alignment(self, data: Dict[str, Any]) -> float:
        """
        Calculate strategic alignment score.
        
        Args:
            data: Data containing strategic alignment metrics
            
        Returns:
            Strategic alignment score
        """
        goal_achievement = data.get('goal_achievement', 0.0)
        priority_alignment = data.get('priority_alignment', 0.0)
        resource_allocation = data.get('resource_allocation', 0.0)
        
        return (goal_achievement + priority_alignment + resource_allocation) / 3.0


class TechnicalMetrics(BaseMetric):
    """
    Metrics for evaluating technical performance of agents.
    
    This includes performance, reliability, resource usage, and other technical metrics.
    """
    
    def __init__(self, config: MetricConfig = None):
        """
        Initialize technical metrics.
        
        Args:
            config: Metric configuration (uses defaults if None)
        """
        if config is None:
            config = MetricConfig(
                name="technical_performance",
                description="Overall technical performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=85.0
            )
        
        super().__init__(config)
        
        # Sub-metrics
        self.performance_metric = MetricConfig(
            name="performance_score",
            description="Execution performance",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
        
        self.reliability_metric = MetricConfig(
            name="reliability_score",
            description="System reliability",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
        
        self.resource_usage_metric = MetricConfig(
            name="resource_usage_score",
            description="Resource usage efficiency",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
    
    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate technical performance score.
        
        Args:
            data: Data containing technical metrics
            
        Returns:
            Overall technical score
        """
        # Extract sub-metric values
        performance = data.get('performance_score', 0.0)
        reliability = data.get('reliability_score', 0.0)
        resource_usage = data.get('resource_usage_score', 0.0)
        
        # Calculate weighted average
        weights = {
            'performance': 0.4,
            'reliability': 0.4,
            'resource_usage': 0.2
        }
        
        overall_score = (
            performance * weights['performance'] +
            reliability * weights['reliability'] +
            resource_usage * weights['resource_usage']
        )
        
        return overall_score
    
    def calculate_performance(self, data: Dict[str, Any]) -> float:
        """
        Calculate performance score.
        
        Args:
            data: Data containing performance metrics
            
        Returns:
            Performance score
        """
        execution_time = data.get('execution_time', 0.0)
        throughput = data.get('throughput', 0.0)
        latency = data.get('latency', 0.0)
        
        # Normalize metrics (lower is better for time and latency)
        time_score = max(0.0, 100.0 - execution_time)
        throughput_score = min(100.0, throughput)
        latency_score = max(0.0, 100.0 - latency)
        
        return (time_score + throughput_score + latency_score) / 3.0
    
    def calculate_reliability(self, data: Dict[str, Any]) -> float:
        """
        Calculate reliability score.
        
        Args:
            data: Data containing reliability metrics
            
        Returns:
            Reliability score
        """
        uptime = data.get('uptime', 0.0)
        error_rate = data.get('error_rate', 100.0)
        failure_count = data.get('failure_count', 0)
        
        uptime_score = min(100.0, uptime)
        error_score = max(0.0, 100.0 - error_rate)
        failure_score = max(0.0, 100.0 - failure_count * 10)
        
        return (uptime_score + error_score + failure_score) / 3.0
    
    def calculate_resource_usage(self, data: Dict[str, Any]) -> float:
        """
        Calculate resource usage score.
        
        Args:
            data: Data containing resource usage metrics
            
        Returns:
            Resource usage score
        """
        memory_usage = data.get('memory_usage', 100.0)
        cpu_usage = data.get('cpu_usage', 100.0)
        disk_usage = data.get('disk_usage', 100.0)
        
        # Lower usage is better
        memory_score = max(0.0, 100.0 - memory_usage)
        cpu_score = max(0.0, 100.0 - cpu_usage)
        disk_score = max(0.0, 100.0 - disk_usage)
        
        return (memory_score + cpu_score + disk_score) / 3.0


class EthicalMetrics(BaseMetric):
    """
    Metrics for evaluating ethical aspects of agents.
    
    This includes bias detection, safety, transparency, and other ethical metrics.
    """
    
    def __init__(self, config: MetricConfig = None):
        """
        Initialize ethical metrics.
        
        Args:
            config: Metric configuration (uses defaults if None)
        """
        if config is None:
            config = MetricConfig(
                name="ethical_performance",
                description="Overall ethical performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=90.0
            )
        
        super().__init__(config)
        
        # Sub-metrics
        self.bias_metric = MetricConfig(
            name="bias_score",
            description="Bias detection and mitigation",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
        
        self.safety_metric = MetricConfig(
            name="safety_score",
            description="Safety and harm prevention",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
        
        self.transparency_metric = MetricConfig(
            name="transparency_score",
            description="Transparency and explainability",
            unit="score",
            min_value=0.0,
            max_value=100.0
        )
    
    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate ethical performance score.
        
        Args:
            data: Data containing ethical metrics
            
        Returns:
            Overall ethical score
        """
        # Extract sub-metric values
        bias = data.get('bias_score', 0.0)
        safety = data.get('safety_score', 0.0)
        transparency = data.get('transparency_score', 0.0)
        
        # Calculate weighted average
        weights = {
            'bias': 0.4,
            'safety': 0.4,
            'transparency': 0.2
        }
        
        overall_score = (
            bias * weights['bias'] +
            safety * weights['safety'] +
            transparency * weights['transparency']
        )
        
        return overall_score
    
    def calculate_bias(self, data: Dict[str, Any]) -> float:
        """
        Calculate bias score.
        
        Args:
            data: Data containing bias metrics
            
        Returns:
            Bias score
        """
        demographic_parity = data.get('demographic_parity', 0.0)
        equal_opportunity = data.get('equal_opportunity', 0.0)
        fairness_awareness = data.get('fairness_awareness', 0.0)
        
        return (demographic_parity + equal_opportunity + fairness_awareness) / 3.0
    
    def calculate_safety(self, data: Dict[str, Any]) -> float:
        """
        Calculate safety score.
        
        Args:
            data: Data containing safety metrics
            
        Returns:
            Safety score
        """
        harm_prevention = data.get('harm_prevention', 0.0)
        risk_assessment = data.get('risk_assessment', 0.0)
        compliance_score = data.get('compliance_score', 0.0)
        
        return (harm_prevention + risk_assessment + compliance_score) / 3.0
    
    def calculate_transparency(self, data: Dict[str, Any]) -> float:
        """
        Calculate transparency score.
        
        Args:
            data: Data containing transparency metrics
            
        Returns:
            Transparency score
        """
        explainability = data.get('explainability', 0.0)
        interpretability = data.get('interpretability', 0.0)
        documentation_score = data.get('documentation_score', 0.0)
        
        return (explainability + interpretability + documentation_score) / 3.0