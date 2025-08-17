"""
Evaluation module for FBA-Bench.

This module provides comprehensive evaluation capabilities for assessing agent performance
across multiple dimensions, including comparative analysis, trend analysis, and statistical
validation.
"""

from .enhanced_evaluation_framework import (
    EnhancedEvaluationFramework,
    EvaluationDimension,
    EvaluationGranularity,
    DimensionWeight,
    EvaluationConfig,
    DimensionScore,
    MultiDimensionalEvaluation,
    EvaluationProfile
)

__all__ = [
    "EnhancedEvaluationFramework",
    "EvaluationDimension",
    "EvaluationGranularity",
    "DimensionWeight",
    "EvaluationConfig",
    "DimensionScore",
    "MultiDimensionalEvaluation",
    "EvaluationProfile"
]