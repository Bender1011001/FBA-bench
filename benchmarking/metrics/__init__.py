"""
Multi-dimensional evaluation metrics for FBA-Bench.

This package provides comprehensive metrics for evaluating agent performance across
multiple dimensions including cognitive capabilities, business metrics, technical performance,
and ethical considerations. The metrics follow HELM and BIG-bench standards and include
statistical validation methods.
"""

from .base import BaseMetric, CognitiveMetrics, BusinessMetrics, TechnicalMetrics, EthicalMetrics
from .statistical import StatisticalValidator
from .registry import MetricRegistry, metrics_registry
from .advanced_cognitive import AdvancedCognitiveMetrics
from .business_intelligence import BusinessIntelligenceMetrics
from .technical_performance import TechnicalPerformanceMetrics
from .ethical_safety import EthicalSafetyMetrics
from .cross_domain import CrossDomainMetrics
from .statistical_analysis import StatisticalAnalysisFramework
from .comparative_analysis import ComparativeAnalysisEngine

# Extensible metrics framework
from .extensible_metrics import (
    # Types and enums
    MetricValueType,
    AggregationMethod,
    MetricMetadata,
    MetricResult,
    MetricValidationResult,
    
    # Base metric classes
    BaseMetric as ExtensibleBaseMetric,
    NumericMetric,
    BooleanMetric,
    StringMetric,
    ListMetric,
    CustomMetric,
    
    # Registry and management
    MetricRegistry as ExtensibleMetricRegistry,
    MetricSuite,
    
    # Validation
    ValidationRule,
    ThresholdValidationRule,
    CustomValidationRule,
    ValidationEngine,
    
    # Global instances
    metric_registry as extensible_metric_registry,
    validation_engine
)

__all__ = [
    # Legacy metrics
    "BaseMetric",
    "CognitiveMetrics",
    "BusinessMetrics",
    "TechnicalMetrics",
    "EthicalMetrics",
    "StatisticalValidator",
    "MetricRegistry",
    "metrics_registry",
    "AdvancedCognitiveMetrics",
    "BusinessIntelligenceMetrics",
    "TechnicalPerformanceMetrics",
    "EthicalSafetyMetrics",
    "CrossDomainMetrics",
    "StatisticalAnalysisFramework",
    "ComparativeAnalysisEngine",
    
    # Extensible metrics framework
    "MetricValueType",
    "AggregationMethod",
    "MetricMetadata",
    "MetricResult",
    "MetricValidationResult",
    "ExtensibleBaseMetric",
    "NumericMetric",
    "BooleanMetric",
    "StringMetric",
    "ListMetric",
    "CustomMetric",
    "ExtensibleMetricRegistry",
    "MetricSuite",
    "ValidationRule",
    "ThresholdValidationRule",
    "CustomValidationRule",
    "ValidationEngine",
    "extensible_metric_registry",
    "validation_engine"
]