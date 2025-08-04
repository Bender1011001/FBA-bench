"""
Unit tests for the benchmarking metrics components.

This module contains comprehensive unit tests for the metrics system,
including base metrics, statistical validation, and registry components.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

from benchmarking.metrics.base import (
    BaseMetric,
    MetricResult,
    MetricCategory,
    CognitiveMetrics,
    BusinessMetrics,
    TechnicalMetrics,
    EthicalMetrics
)
from benchmarking.metrics.statistical import (
    StatisticalValidator,
    ConfidenceInterval,
    SignificanceTest,
    OutlierDetector
)
from benchmarking.metrics.registry import MetricsRegistry


class TestBaseMetric(unittest.TestCase):
    """Test cases for BaseMetric class."""
    
    def setUp(self):
        """Set up test fixtures."""
        class TestMetric(BaseMetric):
            def __init__(self):
                super().__init__(
                    name="test_metric",
                    description="Test metric for unit testing",
                    category=MetricCategory.COGNITIVE
                )
            
            async def calculate(self, context: Dict[str, Any]) -> MetricResult:
                return MetricResult(
                    name="test_metric",
                    score=0.85,
                    confidence=0.9,
                    details={"test": "details"},
                    metadata={"test": "metadata"}
                )
        
        self.metric = TestMetric()
    
    def test_init(self):
        """Test BaseMetric initialization."""
        self.assertEqual(self.metric.name, "test_metric")
        self.assertEqual(self.metric.description, "Test metric for unit testing")
        self.assertEqual(self.metric.category, MetricCategory.COGNITIVE)
    
    def test_calculate(self):
        """Test metric calculation."""
        context = {"test": "context"}
        result = asyncio.run(self.metric.calculate(context))
        
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "test_metric")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.details, {"test": "details"})
        self.assertEqual(result.metadata, {"test": "metadata"})
    
    def test_validate_context(self):
        """Test context validation."""
        # Valid context
        valid_context = {"events": [], "tick_number": 1}
        is_valid, errors = self.metric.validate_context(valid_context)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid context (not a dict)
        invalid_context = "invalid"
        is_valid, errors = self.metric.validate_context(invalid_context)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestMetricResult(unittest.TestCase):
    """Test cases for MetricResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.result = MetricResult(
            name="test_metric",
            score=0.85,
            confidence=0.9,
            details={"test": "details"},
            metadata={"test": "metadata"}
        )
    
    def test_init(self):
        """Test MetricResult initialization."""
        self.assertEqual(self.result.name, "test_metric")
        self.assertEqual(self.result.score, 0.85)
        self.assertEqual(self.result.confidence, 0.9)
        self.assertEqual(self.result.details, {"test": "details"})
        self.assertEqual(self.result.metadata, {"test": "metadata"})
        self.assertIsNotNone(self.result.timestamp)
    
    def test_to_dict(self):
        """Test MetricResult to_dict conversion."""
        result_dict = self.result.to_dict()
        
        self.assertEqual(result_dict["name"], "test_metric")
        self.assertEqual(result_dict["score"], 0.85)
        self.assertEqual(result_dict["confidence"], 0.9)
        self.assertEqual(result_dict["details"], {"test": "details"})
        self.assertEqual(result_dict["metadata"], {"test": "metadata"})
        self.assertIn("timestamp", result_dict)
    
    def test_is_valid(self):
        """Test MetricResult validation."""
        # Valid result
        self.assertTrue(self.result.is_valid())
        
        # Invalid score (out of range)
        self.result.score = 1.5
        self.assertFalse(self.result.is_valid())
        
        # Invalid confidence (out of range)
        self.result.score = 0.85
        self.result.confidence = -0.1
        self.assertFalse(self.result.is_valid())


class TestCognitiveMetrics(unittest.TestCase):
    """Test cases for CognitiveMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cognitive_metrics = CognitiveMetrics()
    
    def test_init(self):
        """Test CognitiveMetrics initialization."""
        self.assertEqual(self.cognitive_metrics.name, "cognitive")
        self.assertEqual(self.cognitive_metrics.category, MetricCategory.COGNITIVE)
        self.assertIsNotNone(self.cognitive_metrics.description)
    
    def test_calculate(self):
        """Test cognitive metrics calculation."""
        context = {
            "events": [
                {"type": "AgentDecisionEvent", "tick_number": 1},
                {"type": "AgentPlannedGoalEvent", "tick_number": 2},
                {"type": "AgentGoalStatusUpdateEvent", "status": "completed", "tick_number": 3}
            ],
            "tick_number": 3
        }
        
        result = asyncio.run(self.cognitive_metrics.calculate(context))
        
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "cognitive")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn("reasoning", result.details)
        self.assertIn("planning", result.details)
        self.assertIn("memory", result.details)


class TestBusinessMetrics(unittest.TestCase):
    """Test cases for BusinessMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.business_metrics = BusinessMetrics()
    
    def test_init(self):
        """Test BusinessMetrics initialization."""
        self.assertEqual(self.business_metrics.name, "business")
        self.assertEqual(self.business_metrics.category, MetricCategory.BUSINESS)
        self.assertIsNotNone(self.business_metrics.description)
    
    def test_calculate(self):
        """Test business metrics calculation."""
        context = {
            "events": [
                {"type": "SaleOccurred", "revenue": 1000, "profit": 200, "tick_number": 1},
                {"type": "AdSpendEvent", "amount": 100, "tick_number": 2}
            ],
            "tick_number": 2
        }
        
        result = asyncio.run(self.business_metrics.calculate(context))
        
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "business")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn("roi", result.details)
        self.assertIn("efficiency", result.details)
        self.assertIn("strategic_alignment", result.details)


class TestTechnicalMetrics(unittest.TestCase):
    """Test cases for TechnicalMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.technical_metrics = TechnicalMetrics()
    
    def test_init(self):
        """Test TechnicalMetrics initialization."""
        self.assertEqual(self.technical_metrics.name, "technical")
        self.assertEqual(self.technical_metrics.category, MetricCategory.TECHNICAL)
        self.assertIsNotNone(self.technical_metrics.description)
    
    def test_calculate(self):
        """Test technical metrics calculation."""
        context = {
            "events": [
                {"type": "ApiCallEvent", "response_time": 100, "tick_number": 1},
                {"type": "SystemErrorEvent", "error_count": 1, "tick_number": 2}
            ],
            "tick_number": 2,
            "performance_data": {
                "cpu_usage": 0.5,
                "memory_usage": 0.3
            }
        }
        
        result = asyncio.run(self.technical_metrics.calculate(context))
        
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "technical")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn("performance", result.details)
        self.assertIn("reliability", result.details)
        self.assertIn("resource_usage", result.details)


class TestEthicalMetrics(unittest.TestCase):
    """Test cases for EthicalMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ethical_metrics = EthicalMetrics()
    
    def test_init(self):
        """Test EthicalMetrics initialization."""
        self.assertEqual(self.ethical_metrics.name, "ethical")
        self.assertEqual(self.ethical_metrics.category, MetricCategory.ETHICAL)
        self.assertIsNotNone(self.ethical_metrics.description)
    
    def test_calculate(self):
        """Test ethical metrics calculation."""
        context = {
            "events": [
                {"type": "BiasDetectionEvent", "bias_score": 0.1, "tick_number": 1},
                {"type": "SafetyViolationEvent", "violation_count": 0, "tick_number": 2}
            ],
            "tick_number": 2
        }
        
        result = asyncio.run(self.ethical_metrics.calculate(context))
        
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "ethical")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIn("bias_detection", result.details)
        self.assertIn("safety", result.details)
        self.assertIn("transparency", result.details)


class TestStatisticalValidator(unittest.TestCase):
    """Test cases for StatisticalValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = StatisticalValidator()
    
    def test_init(self):
        """Test StatisticalValidator initialization."""
        self.assertIsInstance(self.validator, StatisticalValidator)
    
    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        confidence_level = 0.95
        
        interval = self.validator.calculate_confidence_interval(data, confidence_level)
        
        self.assertIsInstance(interval, ConfidenceInterval)
        self.assertLess(interval.lower_bound, interval.upper_bound)
        self.assertGreater(interval.lower_bound, 0)
        self.assertLess(interval.upper_bound, 6)
        self.assertEqual(interval.confidence_level, confidence_level)
    
    def test_calculate_confidence_interval_empty_data(self):
        """Test confidence interval calculation with empty data."""
        data = []
        confidence_level = 0.95
        
        with self.assertRaises(ValueError):
            self.validator.calculate_confidence_interval(data, confidence_level)
    
    def test_perform_significance_test(self):
        """Test significance test calculation."""
        sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        test_result = self.validator.perform_significance_test(sample1, sample2)
        
        self.assertIsInstance(test_result, SignificanceTest)
        self.assertGreaterEqual(test_result.p_value, 0.0)
        self.assertLessEqual(test_result.p_value, 1.0)
        self.assertIn(test_result.is_significant, [True, False])
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # 100.0 is an outlier
        
        outliers = self.validator.detect_outliers(data)
        
        self.assertIsInstance(outliers, OutlierDetector)
        self.assertGreater(len(outliers.outlier_indices), 0)
        self.assertIn(5, outliers.outlier_indices)  # Index of 100.0
        self.assertGreater(len(outliers.outlier_values), 0)
        self.assertIn(100.0, outliers.outlier_values)
    
    def test_detect_outliers_no_outliers(self):
        """Test outlier detection with no outliers."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        outliers = self.validator.detect_outliers(data)
        
        self.assertIsInstance(outliers, OutlierDetector)
        self.assertEqual(len(outliers.outlier_indices), 0)
        self.assertEqual(len(outliers.outlier_values), 0)


class TestConfidenceInterval(unittest.TestCase):
    """Test cases for ConfidenceInterval class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interval = ConfidenceInterval(
            lower_bound=1.0,
            upper_bound=3.0,
            confidence_level=0.95
        )
    
    def test_init(self):
        """Test ConfidenceInterval initialization."""
        self.assertEqual(self.interval.lower_bound, 1.0)
        self.assertEqual(self.interval.upper_bound, 3.0)
        self.assertEqual(self.interval.confidence_level, 0.95)
    
    def test_contains(self):
        """Test contains method."""
        self.assertTrue(self.interval.contains(2.0))
        self.assertTrue(self.interval.contains(1.0))
        self.assertTrue(self.interval.contains(3.0))
        self.assertFalse(self.interval.contains(0.5))
        self.assertFalse(self.interval.contains(3.5))
    
    def test_width(self):
        """Test width calculation."""
        self.assertEqual(self.interval.width(), 2.0)
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        interval_dict = self.interval.to_dict()
        
        self.assertEqual(interval_dict["lower_bound"], 1.0)
        self.assertEqual(interval_dict["upper_bound"], 3.0)
        self.assertEqual(interval_dict["confidence_level"], 0.95)
        self.assertEqual(interval_dict["width"], 2.0)


class TestSignificanceTest(unittest.TestCase):
    """Test cases for SignificanceTest class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_result = SignificanceTest(
            p_value=0.03,
            is_significant=True,
            test_type="t_test",
            test_statistic=2.5
        )
    
    def test_init(self):
        """Test SignificanceTest initialization."""
        self.assertEqual(self.test_result.p_value, 0.03)
        self.assertTrue(self.test_result.is_significant)
        self.assertEqual(self.test_result.test_type, "t_test")
        self.assertEqual(self.test_result.test_statistic, 2.5)
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        test_dict = self.test_result.to_dict()
        
        self.assertEqual(test_dict["p_value"], 0.03)
        self.assertTrue(test_dict["is_significant"])
        self.assertEqual(test_dict["test_type"], "t_test")
        self.assertEqual(test_dict["test_statistic"], 2.5)


class TestOutlierDetector(unittest.TestCase):
    """Test cases for OutlierDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = OutlierDetector(
            outlier_indices=[5],
            outlier_values=[100.0],
            method="iqr"
        )
    
    def test_init(self):
        """Test OutlierDetector initialization."""
        self.assertEqual(self.detector.outlier_indices, [5])
        self.assertEqual(self.detector.outlier_values, [100.0])
        self.assertEqual(self.detector.method, "iqr")
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        detector_dict = self.detector.to_dict()
        
        self.assertEqual(detector_dict["outlier_indices"], [5])
        self.assertEqual(detector_dict["outlier_values"], [100.0])
        self.assertEqual(detector_dict["method"], "iqr")


class TestMetricsRegistry(unittest.TestCase):
    """Test cases for MetricsRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = MetricsRegistry()
        self.test_metric = Mock(spec=BaseMetric)
        self.test_metric.name = "test_metric"
    
    def test_init(self):
        """Test MetricsRegistry initialization."""
        self.assertIsInstance(self.registry, MetricsRegistry)
        self.assertEqual(len(self.registry._metrics), 0)
    
    def test_register_metric(self):
        """Test metric registration."""
        self.registry.register("test_metric", self.test_metric)
        
        self.assertIn("test_metric", self.registry._metrics)
        self.assertEqual(self.registry._metrics["test_metric"], self.test_metric)
    
    def test_register_metric_duplicate(self):
        """Test registering duplicate metric."""
        self.registry.register("test_metric", self.test_metric)
        
        with self.assertRaises(ValueError):
            self.registry.register("test_metric", self.test_metric)
    
    def test_get_metric(self):
        """Test getting metric."""
        self.registry.register("test_metric", self.test_metric)
        
        metric = self.registry.get("test_metric")
        
        self.assertEqual(metric, self.test_metric)
    
    def test_get_metric_not_found(self):
        """Test getting non-existent metric."""
        metric = self.registry.get("non_existent")
        
        self.assertIsNone(metric)
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        self.registry.register("test_metric1", self.test_metric)
        test_metric2 = Mock(spec=BaseMetric)
        test_metric2.name = "test_metric2"
        self.registry.register("test_metric2", test_metric2)
        
        all_metrics = self.registry.get_all_metrics()
        
        self.assertEqual(len(all_metrics), 2)
        self.assertIn("test_metric1", all_metrics)
        self.assertIn("test_metric2", all_metrics)
    
    def test_unregister_metric(self):
        """Test unregistering metric."""
        self.registry.register("test_metric", self.test_metric)
        
        self.registry.unregister("test_metric")
        
        self.assertNotIn("test_metric", self.registry._metrics)
    
    def test_unregister_metric_not_found(self):
        """Test unregistering non-existent metric."""
        with self.assertRaises(ValueError):
            self.registry.unregister("non_existent")
    
    def test_clear(self):
        """Test clearing registry."""
        self.registry.register("test_metric", self.test_metric)
        
        self.registry.clear()
        
        self.assertEqual(len(self.registry._metrics), 0)
    
    def test_list_metrics(self):
        """Test listing metrics."""
        self.registry.register("test_metric", self.test_metric)
        
        metrics_list = self.registry.list_metrics()
        
        self.assertEqual(len(metrics_list), 1)
        self.assertEqual(metrics_list[0], "test_metric")


if __name__ == "__main__":
    unittest.main()