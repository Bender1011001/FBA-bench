"""
Integration tests for metrics collection and aggregation workflows.

This module contains comprehensive integration tests that verify the interaction
between different metrics components, including collection, aggregation, validation,
and reporting workflows in the FBA-Bench system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import tempfile
import os
import numpy as np
import pandas as pd

from benchmarking.core.engine import BenchmarkEngine, BenchmarkConfig, BenchmarkResult
from benchmarking.metrics.base import BaseMetric, MetricConfig, MetricResult
from benchmarking.metrics.registry import MetricRegistry
from benchmarking.metrics.advanced_cognitive import AdvancedCognitiveMetrics
from benchmarking.metrics.business_intelligence import BusinessIntelligenceMetrics
from benchmarking.metrics.technical_performance import TechnicalPerformanceMetrics
from benchmarking.metrics.ethical_safety import EthicalSafetyMetrics
from benchmarking.metrics.cross_domain import CrossDomainMetrics
from benchmarking.metrics.statistical_analysis import StatisticalAnalysisFramework
from benchmarking.metrics.comparative_analysis import ComparativeAnalysisEngine
from benchmarking.scenarios.base import ScenarioConfig, BaseScenario
from agent_runners.base_runner import BaseAgentRunner, AgentConfig


class MockAgentWithMetrics(BaseAgentRunner):
    """Mock agent implementation with metrics collection for integration testing."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.responses = []
        self.actions_taken = []
        self.metrics_history = []
        self.performance_data = {
            "response_times": [],
            "success_rates": [],
            "resource_usage": []
        }
    
    async def initialize(self) -> None:
        """Initialize the mock agent."""
        self.is_initialized = True
    
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return a response."""
        start_time = datetime.now()
        
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        response = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "response": f"Mock response to: {input_data.get('content', '')}",
            "confidence": np.random.uniform(0.7, 0.95),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }
        
        self.responses.append(response)
        self.performance_data["response_times"].append(response["processing_time"])
        
        return response
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return the result."""
        start_time = datetime.now()
        
        # Simulate action execution time
        await asyncio.sleep(0.02)
        
        success = np.random.random() > 0.1  # 90% success rate
        
        result = {
            "agent_id": self.config.agent_id,
            "action": action.get("type", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if success else "failed",
            "result": f"Executed action: {action.get('type', 'unknown')}",
            "execution_time": (datetime.now() - start_time).total_seconds()
        }
        
        self.actions_taken.append(result)
        
        # Update success rate
        success_rate = len([a for a in self.actions_taken if a["status"] == "completed"]) / len(self.actions_taken)
        self.performance_data["success_rates"].append(success_rate)
        
        return result
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from the agent."""
        metrics = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "responses_count": len(self.responses),
            "actions_count": len(self.actions_taken),
            "avg_response_time": np.mean(self.performance_data["response_times"]) if self.performance_data["response_times"] else 0,
            "success_rate": np.mean(self.performance_data["success_rates"]) if self.performance_data["success_rates"] else 0,
            "resource_usage": {
                "cpu": np.random.uniform(0.1, 0.8),
                "memory": np.random.uniform(0.1, 0.7),
                "network": np.random.uniform(0.05, 0.3)
            }
        }
        
        self.metrics_history.append(metrics)
        self.performance_data["resource_usage"].append(metrics["resource_usage"])
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown the mock agent."""
        self.is_initialized = False


class TestScenarioWithMetrics(BaseScenario):
    """Test scenario implementation with metrics collection for integration testing."""
    
    def _validate_domain_parameters(self) -> List[str]:
        """Validate domain-specific parameters."""
        return []
    
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize the test scenario."""
        await super().initialize(parameters)
        self.test_data = parameters.get("test_data", {})
        self.metrics_collected = []
    
    async def setup_for_agent(self, agent_id: str) -> None:
        """Setup the scenario for a specific agent."""
        await super().setup_for_agent(agent_id)
        self.agent_states[agent_id]["test_data"] = self.test_data
        self.agent_states[agent_id]["metrics"] = []
    
    async def update_tick(self, tick: int, state) -> None:
        """Update the scenario for a specific tick."""
        await super().update_tick(tick, state)
        
        # Collect metrics for each agent
        for agent_id in self.agent_states:
            agent_metrics = {
                "tick": tick,
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "scenario_progress": tick / self.duration_ticks,
                "tasks_completed": np.random.randint(0, 5),
                "efficiency_score": np.random.uniform(0.6, 0.95)
            }
            
            self.agent_states[agent_id]["metrics"].append(agent_metrics)
            self.metrics_collected.append(agent_metrics)
    
    async def evaluate_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Evaluate agent performance in the scenario."""
        base_metrics = await super().evaluate_agent_performance(agent_id)
        
        # Calculate aggregated metrics from tick data
        agent_metrics = self.agent_states[agent_id]["metrics"]
        
        if agent_metrics:
            avg_efficiency = np.mean([m["efficiency_score"] for m in agent_metrics])
            total_tasks = sum([m["tasks_completed"] for m in agent_metrics])
            progress_rate = agent_metrics[-1]["scenario_progress"] if agent_metrics else 0
        else:
            avg_efficiency = 0
            total_tasks = 0
            progress_rate = 0
        
        scenario_metrics = {
            "avg_efficiency_score": avg_efficiency,
            "total_tasks_completed": total_tasks,
            "progress_rate": progress_rate,
            "consistency_score": np.random.uniform(0.7, 0.95)
        }
        
        return {**base_metrics, **scenario_metrics}


class TestMetricsWorkflows:
    """Test cases for metrics collection and aggregation workflows."""
    
    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(
            agent_id="test_agent",
            agent_type="mock_with_metrics",
            agent_class="MockAgentWithMetrics",
            parameters={"test_param": "test_value"}
        )
    
    @pytest.fixture
    def scenario_config(self):
        """Create a test scenario configuration."""
        return ScenarioConfig(
            name="test_scenario_with_metrics",
            description="Test scenario for metrics workflows",
            domain="test",
            duration_ticks=30,
            parameters={"test_data": {"key": "value"}}
        )
    
    @pytest.fixture
    def benchmark_config(self):
        """Create a test benchmark configuration."""
        return BenchmarkConfig(
            name="metrics_workflow_test",
            description="Test metrics collection and aggregation workflows",
            max_duration=300,
            tick_interval=0.1,
            metrics_collection_interval=0.5
        )
    
    @pytest.fixture
    def mock_agent(self, agent_config):
        """Create a mock agent with metrics."""
        return MockAgentWithMetrics(agent_config)
    
    @pytest.fixture
    def test_scenario(self, scenario_config):
        """Create a test scenario with metrics."""
        return TestScenarioWithMetrics(scenario_config)
    
    @pytest.fixture
    def benchmark_engine(self, benchmark_config):
        """Create a benchmark engine instance."""
        return BenchmarkEngine(benchmark_config)
    
    @pytest.fixture
    def advanced_cognitive_metrics(self):
        """Create an AdvancedCognitiveMetrics instance."""
        return AdvancedCognitiveMetrics()
    
    @pytest.fixture
    def business_intelligence_metrics(self):
        """Create a BusinessIntelligenceMetrics instance."""
        return BusinessIntelligenceMetrics()
    
    @pytest.fixture
    def technical_performance_metrics(self):
        """Create a TechnicalPerformanceMetrics instance."""
        return TechnicalPerformanceMetrics()
    
    @pytest.fixture
    def ethical_safety_metrics(self):
        """Create an EthicalSafetyMetrics instance."""
        return EthicalSafetyMetrics()
    
    @pytest.fixture
    def cross_domain_metrics(self):
        """Create a CrossDomainMetrics instance."""
        return CrossDomainMetrics()
    
    @pytest.fixture
    def statistical_analysis_framework(self):
        """Create a StatisticalAnalysisFramework instance."""
        return StatisticalAnalysisFramework()
    
    @pytest.fixture
    def comparative_analysis_engine(self):
        """Create a ComparativeAnalysisEngine instance."""
        return ComparativeAnalysisEngine()
    
    @pytest.mark.asyncio
    async def test_agent_metrics_collection(self, mock_agent):
        """Test agent metrics collection workflow."""
        await mock_agent.initialize()
        
        # Process some inputs and execute some actions
        for i in range(5):
            await mock_agent.process_input({"content": f"Test input {i}"})
            await mock_agent.execute_action({"type": "test_action", "id": i})
        
        # Collect metrics
        metrics = await mock_agent.collect_metrics()
        
        # Verify metrics structure
        assert "agent_id" in metrics
        assert "timestamp" in metrics
        assert "responses_count" in metrics
        assert "actions_count" in metrics
        assert "avg_response_time" in metrics
        assert "success_rate" in metrics
        assert "resource_usage" in metrics
        
        # Verify metrics values
        assert metrics["agent_id"] == mock_agent.config.agent_id
        assert metrics["responses_count"] == 5
        assert metrics["actions_count"] == 5
        assert metrics["avg_response_time"] > 0
        assert 0 <= metrics["success_rate"] <= 1
        assert "cpu" in metrics["resource_usage"]
        assert "memory" in metrics["resource_usage"]
        assert "network" in metrics["resource_usage"]
        
        # Verify metrics history
        assert len(mock_agent.metrics_history) == 1
        assert mock_agent.metrics_history[0] == metrics
    
    @pytest.mark.asyncio
    async def test_scenario_metrics_collection(self, test_scenario):
        """Test scenario metrics collection workflow."""
        await test_scenario.initialize({"test_data": {"key": "value"}})
        await test_scenario.setup_for_agent("test_agent")
        
        # Simulate tick updates
        for tick in range(1, 11):
            await test_scenario.update_tick(tick, Mock())
        
        # Evaluate agent performance
        metrics = await test_scenario.evaluate_agent_performance("test_agent")
        
        # Verify metrics structure
        assert "agent_id" in metrics
        assert "scenario_name" in metrics
        assert "avg_efficiency_score" in metrics
        assert "total_tasks_completed" in metrics
        assert "progress_rate" in metrics
        assert "consistency_score" in metrics
        
        # Verify metrics values
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "test_scenario_with_metrics"
        assert 0 <= metrics["avg_efficiency_score"] <= 1
        assert metrics["total_tasks_completed"] >= 0
        assert 0 <= metrics["progress_rate"] <= 1
        assert 0 <= metrics["consistency_score"] <= 1
        
        # Verify metrics collection history
        assert len(test_scenario.metrics_collected) == 10
        for tick_metrics in test_scenario.metrics_collected:
            assert "tick" in tick_metrics
            assert "agent_id" in tick_metrics
            assert "scenario_progress" in tick_metrics
            assert "tasks_completed" in tick_metrics
            assert "efficiency_score" in tick_metrics
    
    @pytest.mark.asyncio
    async def test_advanced_cognitive_metrics_workflow(self, advanced_cognitive_metrics, mock_agent):
        """Test advanced cognitive metrics workflow."""
        await mock_agent.initialize()
        
        # Generate test data for advanced cognitive metrics
        test_data = {
            "logical_consistency_score": 85.0,
            "causal_reasoning_score": 80.0,
            "abstract_reasoning_score": 90.0,
            "metacognition_score": 75.0,
            "multi_step_planning_score": 85.0,
            "memory_efficiency_score": 80.0,
            "learning_adaptation_score": 90.0
        }
        
        # Calculate metrics
        result = advanced_cognitive_metrics.calculate(test_data)
        
        # Verify result structure
        assert isinstance(result, float)
        assert 0 <= result <= 100
        
        # Verify individual metric calculations
        logical_consistency = advanced_cognitive_metrics.calculate_logical_consistency({
            "logical_statements": [
                {"statement": "A implies B", "consistency": 0.9},
                {"statement": "B implies C", "consistency": 0.8}
            ],
            "contradictions": 1,
            "total_statements": 10
        })
        assert isinstance(logical_consistency, float)
        assert 0 <= logical_consistency <= 100
        
        causal_reasoning = advanced_cognitive_metrics.calculate_causal_reasoning({
            "causal_chains": [
                {"chain": "A -> B -> C", "accuracy": 0.8},
                {"chain": "X -> Y -> Z", "accuracy": 0.9}
            ],
            "correct_causal_inferences": 7,
            "total_causal_inferences": 10
        })
        assert isinstance(causal_reasoning, float)
        assert 0 <= causal_reasoning <= 100
    
    @pytest.mark.asyncio
    async def test_business_intelligence_metrics_workflow(self, business_intelligence_metrics, mock_agent):
        """Test business intelligence metrics workflow."""
        await mock_agent.initialize()
        
        # Generate test data for business intelligence metrics
        test_data = {
            "strategic_decision_making_score": 85.0,
            "market_trend_analysis_score": 80.0,
            "competitive_intelligence_score": 90.0,
            "risk_assessment_score": 75.0,
            "roi_optimization_score": 85.0,
            "resource_allocation_score": 80.0,
            "business_outcome_prediction_score": 90.0
        }
        
        # Calculate metrics
        result = business_intelligence_metrics.calculate(test_data)
        
        # Verify result structure
        assert isinstance(result, float)
        assert 0 <= result <= 100
        
        # Verify individual metric calculations
        strategic_decision = business_intelligence_metrics.calculate_strategic_decision_making({
            "decision_quality": 0.85,
            "strategic_alignment": 0.90,
            "long_term_impact": 0.80,
            "risk_assessment": 0.75
        })
        assert isinstance(strategic_decision, float)
        assert 0 <= strategic_decision <= 100
        
        market_trend = business_intelligence_metrics.calculate_market_trend_analysis({
            "trend_prediction_accuracy": 0.85,
            "market_insight_quality": 0.75,
            "competitive_analysis": 0.90,
            "forecast_precision": 0.80
        })
        assert isinstance(market_trend, float)
        assert 0 <= market_trend <= 100
    
    @pytest.mark.asyncio
    async def test_technical_performance_metrics_workflow(self, technical_performance_metrics, mock_agent):
        """Test technical performance metrics workflow."""
        await mock_agent.initialize()
        
        # Generate test data for technical performance metrics
        test_data = {
            "scalability_score": 85.0,
            "resource_utilization_score": 80.0,
            "latency_throughput_score": 90.0,
            "error_handling_score": 75.0,
            "system_resilience_score": 85.0,
            "performance_degradation_score": 80.0,
            "optimization_effectiveness_score": 90.0
        }
        
        # Calculate metrics
        result = technical_performance_metrics.calculate(test_data)
        
        # Verify result structure
        assert isinstance(result, float)
        assert 0 <= result <= 100
        
        # Verify individual metric calculations
        scalability = technical_performance_metrics.calculate_scalability({
            "horizontal_scaling": 0.85,
            "vertical_scaling": 0.80,
            "elasticity": 0.90,
            "load_balancing": 0.75
        })
        assert isinstance(scalability, float)
        assert 0 <= scalability <= 100
        
        resource_util = technical_performance_metrics.calculate_resource_utilization({
            "cpu_usage": 0.65,
            "memory_usage": 0.70,
            "disk_usage": 0.45,
            "network_usage": 0.30
        })
        assert isinstance(resource_util, float)
        assert 0 <= resource_util <= 100
    
    @pytest.mark.asyncio
    async def test_ethical_safety_metrics_workflow(self, ethical_safety_metrics, mock_agent):
        """Test ethical safety metrics workflow."""
        await mock_agent.initialize()
        
        # Generate test data for ethical safety metrics
        test_data = {
            "bias_detection_score": 85.0,
            "fairness_assessment_score": 80.0,
            "safety_protocol_score": 90.0,
            "transparency_explainability_score": 75.0,
            "content_safety_score": 85.0,
            "privacy_protection_score": 80.0,
            "ethical_decision_making_score": 90.0
        }
        
        # Calculate metrics
        result = ethical_safety_metrics.calculate(test_data)
        
        # Verify result structure
        assert isinstance(result, float)
        assert 0 <= result <= 100
        
        # Verify individual metric calculations
        bias_detection = ethical_safety_metrics.calculate_bias_detection({
            "demographic_parity": 0.85,
            "equal_opportunity": 0.80,
            "individual_fairness": 0.90,
            "bias_mitigation": 0.75
        })
        assert isinstance(bias_detection, float)
        assert 0 <= bias_detection <= 100
        
        fairness = ethical_safety_metrics.calculate_fairness_assessment({
            "treatment_equality": 0.85,
            "outcome_fairness": 0.80,
            "process_fairness": 0.90,
            "representation_fairness": 0.75
        })
        assert isinstance(fairness, float)
        assert 0 <= fairness <= 100
    
    @pytest.mark.asyncio
    async def test_cross_domain_metrics_workflow(self, cross_domain_metrics, mock_agent):
        """Test cross-domain metrics workflow."""
        await mock_agent.initialize()
        
        # Generate test data for cross-domain metrics
        test_data = {
            "knowledge_transfer_score": 85.0,
            "multi_modal_integration_score": 80.0,
            "context_awareness_score": 90.0,
            "collaboration_score": 75.0
        }
        
        # Calculate metrics
        result = cross_domain_metrics.calculate(test_data)
        
        # Verify result structure
        assert isinstance(result, float)
        assert 0 <= result <= 100
        
        # Verify individual metric calculations
        knowledge_transfer = cross_domain_metrics.calculate_knowledge_transfer({
            "cross_domain_accuracy": 0.85,
            "knowledge_generalization": 0.80,
            "domain_adaptation": 0.90,
            "transfer_efficiency": 0.75
        })
        assert isinstance(knowledge_transfer, float)
        assert 0 <= knowledge_transfer <= 100
        
        multi_modal = cross_domain_metrics.calculate_multi_modal_integration({
            "text_processing": 0.85,
            "image_processing": 0.80,
            "audio_processing": 0.90,
            "multimodal_fusion": 0.75
        })
        assert isinstance(multi_modal, float)
        assert 0 <= multi_modal <= 100
    
    @pytest.mark.asyncio
    async def test_statistical_analysis_workflow(self, statistical_analysis_framework, mock_agent):
        """Test statistical analysis workflow."""
        await mock_agent.initialize()
        
        # Generate test data for statistical analysis
        test_data = {
            "descriptive_statistics_score": 85.0,
            "inferential_statistics_score": 80.0,
            "time_series_analysis_score": 90.0,
            "multivariate_analysis_score": 75.0
        }
        
        # Calculate metrics
        result = statistical_analysis_framework.calculate(test_data)
        
        # Verify result structure
        assert isinstance(result, float)
        assert 0 <= result <= 100
        
        # Verify individual metric calculations
        descriptive = statistical_analysis_framework.calculate_descriptive_statistics({
            "mean_accuracy": 0.85,
            "median_accuracy": 0.80,
            "std_deviation": 0.10,
            "distribution_normality": 0.90
        })
        assert isinstance(descriptive, float)
        assert 0 <= descriptive <= 100
        
        inferential = statistical_analysis_framework.calculate_inferential_statistics({
            "confidence_level": 0.95,
            "p_value_significance": 0.05,
            "effect_size": 0.80,
            "statistical_power": 0.90
        })
        assert isinstance(inferential, float)
        assert 0 <= inferential <= 100
    
    @pytest.mark.asyncio
    async def test_comparative_analysis_workflow(self, comparative_analysis_engine, mock_agent):
        """Test comparative analysis workflow."""
        await mock_agent.initialize()
        
        # Generate test data for comparative analysis
        test_data = {
            "performance_comparison_score": 85.0,
            "efficiency_effectiveness_score": 80.0,
            "cost_benefit_analysis_score": 90.0,
            "scalability_adaptability_score": 75.0
        }
        
        # Calculate metrics
        result = comparative_analysis_engine.calculate(test_data)
        
        # Verify result structure
        assert isinstance(result, float)
        assert 0 <= result <= 100
        
        # Verify individual metric calculations
        performance = comparative_analysis_engine.calculate_performance_comparison({
            "baseline_performance": 0.70,
            "current_performance": 0.85,
            "improvement_rate": 0.214,
            "regression_detection": False
        })
        assert isinstance(performance, float)
        assert 0 <= performance <= 100
        
        efficiency = comparative_analysis_engine.calculate_efficiency_effectiveness({
            "efficiency_metrics": {"time": 0.8, "memory": 0.85, "cpu": 0.9},
            "effectiveness_metrics": {"accuracy": 0.9, "quality": 0.85, "reliability": 0.95},
            "tradeoff_analysis": {"time_vs_accuracy": 0.8, "memory_vs_quality": 0.85}
        })
        assert isinstance(efficiency, float)
        assert 0 <= efficiency <= 100
    
    @pytest.mark.asyncio
    async def test_metrics_aggregation_workflow(self, benchmark_engine, mock_agent, test_scenario):
        """Test metrics aggregation workflow."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        
        # Register multiple metrics
        metrics = [
            AdvancedCognitiveMetrics(),
            BusinessIntelligenceMetrics(),
            TechnicalPerformanceMetrics(),
            EthicalSafetyMetrics(),
            CrossDomainMetrics()
        ]
        
        for metric in metrics:
            benchmark_engine.register_metric(metric)
        
        # Run the benchmark
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario_with_metrics",
            agent_ids=["test_agent"],
            metric_names=[metric.name for metric in metrics]
        )
        
        # Verify the aggregated result
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_name == "test_scenario_with_metrics"
        assert result.agent_ids == ["test_agent"]
        assert len(result.metric_names) == len(metrics)
        assert result.success is True
        assert result.duration_seconds > 0
        
        # Verify individual metric results
        for metric_name in result.results:
            assert metric_name in result.results
            metric_result = result.results[metric_name]
            assert isinstance(metric_result, dict)
            assert "value" in metric_result
            assert "timestamp" in metric_result
            assert 0 <= metric_result["value"] <= 100
        
        # Verify aggregated metrics
        assert "aggregated_metrics" in result.results
        aggregated = result.results["aggregated_metrics"]
        assert "overall_score" in aggregated
        assert "category_scores" in aggregated
        assert "performance_summary" in aggregated
        
        # Verify overall score calculation
        assert 0 <= aggregated["overall_score"] <= 100
        
        # Verify category scores
        category_scores = aggregated["category_scores"]
        assert "cognitive" in category_scores
        assert "business" in category_scores
        assert "technical" in category_scores
        assert "ethical" in category_scores
        assert "cross_domain" in category_scores
        
        for category_score in category_scores.values():
            assert 0 <= category_score <= 100
    
    @pytest.mark.asyncio
    async def test_metrics_validation_workflow(self, benchmark_engine, mock_agent, test_scenario):
        """Test metrics validation workflow."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        
        # Register a metric with validation
        class ValidatedMetric(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
                self.validation_errors = []
            
            def calculate(self, data):
                # Validate input data
                if not isinstance(data, dict):
                    self.validation_errors.append("Input data must be a dictionary")
                    return 0.0
                
                if "score" not in data:
                    self.validation_errors.append("Missing required field: score")
                    return 0.0
                
                score = data["score"]
                if not isinstance(score, (int, float)):
                    self.validation_errors.append("Score must be a number")
                    return 0.0
                
                if score < 0 or score > 100:
                    self.validation_errors.append("Score must be between 0 and 100")
                    return 0.0
                
                return score
            
            def get_validation_errors(self):
                return self.validation_errors
        
        metric_config = MetricConfig(
            name="validated_metric",
            description="Validated test metric",
            unit="score",
            min_value=0.0,
            max_value=100.0,
            target_value=85.0
        )
        
        validated_metric = ValidatedMetric(metric_config)
        benchmark_engine.register_metric(validated_metric)
        
        # Run the benchmark with valid data
        valid_result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario_with_metrics",
            agent_ids=["test_agent"],
            metric_names=["validated_metric"]
        )
        
        # Verify valid result
        assert valid_result.success is True
        assert "validated_metric" in valid_result.results
        assert valid_result.results["validated_metric"]["value"] > 0
        
        # Reset validation errors
        validated_metric.validation_errors = []
        
        # Run the benchmark with invalid data (simulate by patching)
        with patch.object(validated_metric, 'calculate', return_value=0.0):
            invalid_result = await benchmark_engine.run_benchmark(
                scenario_name="test_scenario_with_metrics",
                agent_ids=["test_agent"],
                metric_names=["validated_metric"]
            )
        
        # Verify invalid result handling
        assert "validated_metric" in invalid_result.results
        assert invalid_result.results["validated_metric"]["value"] == 0.0
        
        # Check validation errors
        validation_errors = validated_metric.get_validation_errors()
        assert len(validation_errors) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_persistence_workflow(self, benchmark_engine, mock_agent, test_scenario):
        """Test metrics persistence workflow."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        
        # Register multiple metrics
        metrics = [
            AdvancedCognitiveMetrics(),
            BusinessIntelligenceMetrics(),
            TechnicalPerformanceMetrics()
        ]
        
        for metric in metrics:
            benchmark_engine.register_metric(metric)
        
        # Create a temporary file for persistence
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Configure the benchmark engine to save results
            benchmark_engine.config.save_results = True
            benchmark_engine.config.results_file = temp_filename
            
            # Run the benchmark
            result = await benchmark_engine.run_benchmark(
                scenario_name="test_scenario_with_metrics",
                agent_ids=["test_agent"],
                metric_names=[metric.name for metric in metrics]
            )
            
            # Verify the result was saved
            assert os.path.exists(temp_filename)
            
            # Load and verify the saved results
            with open(temp_filename, 'r') as f:
                saved_data = json.load(f)
            
            assert "benchmark_results" in saved_data
            assert len(saved_data["benchmark_results"]) > 0
            
            saved_result = saved_data["benchmark_results"][0]
            assert saved_result["scenario_name"] == "test_scenario_with_metrics"
            assert saved_result["agent_ids"] == ["test_agent"]
            assert saved_result["success"] is True
            
            # Verify metrics persistence
            assert "metrics_results" in saved_result
            metrics_results = saved_result["metrics_results"]
            for metric_name in metrics_results:
                assert metric_name in metrics_results
                assert "value" in metrics_results[metric_name]
                assert "timestamp" in metrics_results[metric_name]
            
            # Verify aggregated metrics persistence
            assert "aggregated_metrics" in metrics_results
            aggregated = metrics_results["aggregated_metrics"]
            assert "overall_score" in aggregated
            assert "category_scores" in aggregated
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    @pytest.mark.asyncio
    async def test_real_time_metrics_collection(self, benchmark_engine, mock_agent, test_scenario):
        """Test real-time metrics collection workflow."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        
        # Register a metric for real-time collection
        class RealTimeMetric(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
                self.collection_history = []
            
            def calculate(self, data):
                timestamp = datetime.now().isoformat()
                value = data.get("real_time_value", 0.0)
                
                self.collection_history.append({
                    "timestamp": timestamp,
                    "value": value
                })
                
                return value
        
        metric_config = MetricConfig(
            name="real_time_metric",
            description="Real-time collection metric",
            unit="score",
            min_value=0.0,
            max_value=100.0,
            target_value=85.0
        )
        
        real_time_metric = RealTimeMetric(metric_config)
        benchmark_engine.register_metric(real_time_metric)
        
        # Configure real-time collection
        benchmark_engine.config.real_time_collection = True
        benchmark_engine.config.collection_interval = 0.1
        
        # Run the benchmark
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario_with_metrics",
            agent_ids=["test_agent"],
            metric_names=["real_time_metric"]
        )
        
        # Verify the result
        assert result.success is True
        assert "real_time_metric" in result.results
        
        # Verify real-time collection history
        assert len(real_time_metric.collection_history) > 0
        
        # Verify collection timestamps are in order
        timestamps = [item["timestamp"] for item in real_time_metric.collection_history]
        assert timestamps == sorted(timestamps)
        
        # Verify collection intervals
        if len(real_time_metric.collection_history) > 1:
            time_diffs = []
            for i in range(1, len(real_time_metric.collection_history)):
                t1 = datetime.fromisoformat(real_time_metric.collection_history[i-1]["timestamp"])
                t2 = datetime.fromisoformat(real_time_metric.collection_history[i]["timestamp"])
                time_diffs.append((t2 - t1).total_seconds())
            
            avg_interval = sum(time_diffs) / len(time_diffs)
            # Should be close to the configured interval (0.1 seconds)
            assert abs(avg_interval - 0.1) < 0.05
    
    @pytest.mark.asyncio
    async def test_metrics_correlation_analysis(self, benchmark_engine, mock_agent, test_scenario):
        """Test metrics correlation analysis workflow."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        
        # Register multiple metrics for correlation analysis
        metrics = [
            AdvancedCognitiveMetrics(),
            BusinessIntelligenceMetrics(),
            TechnicalPerformanceMetrics()
        ]
        
        for metric in metrics:
            benchmark_engine.register_metric(metric)
        
        # Configure correlation analysis
        benchmark_engine.config.correlation_analysis = True
        
        # Run the benchmark
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario_with_metrics",
            agent_ids=["test_agent"],
            metric_names=[metric.name for metric in metrics]
        )
        
        # Verify the result
        assert result.success is True
        
        # Verify correlation analysis results
        assert "correlation_analysis" in result.results
        correlation_data = result.results["correlation_analysis"]
        
        assert "correlation_matrix" in correlation_data
        assert "strong_correlations" in correlation_data
        assert "weak_correlations" in correlation_data
        assert "correlation_summary" in correlation_data
        
        # Verify correlation matrix structure
        correlation_matrix = correlation_data["correlation_matrix"]
        metric_names = [metric.name for metric in metrics]
        
        for metric_name in metric_names:
            assert metric_name in correlation_matrix
            for other_metric_name in metric_names:
                assert other_metric_name in correlation_matrix[metric_name]
                
                # Correlation values should be between -1 and 1
                correlation_value = correlation_matrix[metric_name][other_metric_name]
                assert -1 <= correlation_value <= 1
        
        # Verify strong and weak correlations
        strong_correlations = correlation_data["strong_correlations"]
        weak_correlations = correlation_data["weak_correlations"]
        
        # Should be lists of correlation pairs
        assert isinstance(strong_correlations, list)
        assert isinstance(weak_correlations, list)
        
        # Each correlation pair should have metric names and correlation value
        for correlation in strong_correlations:
            assert "metric1" in correlation
            assert "metric2" in correlation
            assert "correlation" in correlation
            assert abs(correlation["correlation"]) >= 0.7  # Strong correlation threshold
        
        for correlation in weak_correlations:
            assert "metric1" in correlation
            assert "metric2" in correlation
            assert "correlation" in correlation
            assert abs(correlation["correlation"]) < 0.3  # Weak correlation threshold