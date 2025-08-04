"""
Integration tests for agent integration with the benchmarking engine.

This module contains comprehensive integration tests that verify the interaction
between different agent implementations and the FBA-Bench benchmarking engine,
including scenario execution, metrics collection, and result validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import tempfile
import os

from benchmarking.core.engine import BenchmarkEngine, BenchmarkConfig, BenchmarkResult
from benchmarking.scenarios.base import ScenarioConfig, BaseScenario
from benchmarking.scenarios.registry import ScenarioRegistry
from benchmarking.metrics.base import BaseMetric, MetricConfig, MetricResult
from benchmarking.metrics.registry import MetricRegistry
from agent_runners.base_runner import BaseAgentRunner, AgentConfig
from agent_runners.simulation_runner import SimulationRunner, SimulationState


class MockAgent(BaseAgentRunner):
    """Mock agent implementation for integration testing."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.responses = []
        self.actions_taken = []
        self.metrics_collected = {}
    
    async def initialize(self) -> None:
        """Initialize the mock agent."""
        self.is_initialized = True
    
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return a response."""
        response = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "response": f"Mock response to: {input_data.get('content', '')}",
            "confidence": 0.8,
            "processing_time": 0.1
        }
        self.responses.append(response)
        return response
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return the result."""
        result = {
            "agent_id": self.config.agent_id,
            "action": action.get("type", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "result": f"Executed action: {action.get('type', 'unknown')}"
        }
        self.actions_taken.append(result)
        return result
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from the agent."""
        metrics = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "responses_count": len(self.responses),
            "actions_count": len(self.actions_taken),
            "avg_response_time": 0.1,
            "success_rate": 0.95
        }
        self.metrics_collected = metrics
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown the mock agent."""
        self.is_initialized = False


class TestScenario(BaseScenario):
    """Test scenario implementation for integration testing."""
    
    def _validate_domain_parameters(self) -> List[str]:
        """Validate domain-specific parameters."""
        return []
    
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize the test scenario."""
        await super().initialize(parameters)
        self.test_data = parameters.get("test_data", {})
    
    async def setup_for_agent(self, agent_id: str) -> None:
        """Setup the scenario for a specific agent."""
        await super().setup_for_agent(agent_id)
        self.agent_states[agent_id]["test_data"] = self.test_data
    
    async def update_tick(self, tick: int, state: SimulationState) -> None:
        """Update the scenario for a specific tick."""
        await super().update_tick(tick, state)
        # Simulate some scenario state changes
        if tick % 10 == 0:
            self.global_state["milestone"] = tick
    
    async def evaluate_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Evaluate agent performance in the scenario."""
        base_metrics = await super().evaluate_agent_performance(agent_id)
        
        # Add scenario-specific metrics
        scenario_metrics = {
            "test_scenario_score": 0.85,
            "task_completion_rate": 0.9,
            "efficiency_metric": 0.8
        }
        
        return {**base_metrics, **scenario_metrics}


class TestAgentIntegration:
    """Test cases for agent integration with the benchmarking engine."""
    
    @pytest.fixture
    def benchmark_config(self):
        """Create a test benchmark configuration."""
        return BenchmarkConfig(
            name="agent_integration_test",
            description="Test agent integration with benchmarking engine",
            max_duration=300,
            tick_interval=0.1,
            metrics_collection_interval=1.0
        )
    
    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(
            agent_id="test_agent",
            agent_type="mock",
            agent_class="MockAgent",
            parameters={"test_param": "test_value"}
        )
    
    @pytest.fixture
    def scenario_config(self):
        """Create a test scenario configuration."""
        return ScenarioConfig(
            name="test_scenario",
            description="Test scenario for agent integration",
            domain="test",
            duration_ticks=50,
            parameters={"test_data": {"key": "value"}}
        )
    
    @pytest.fixture
    def metric_config(self):
        """Create a test metric configuration."""
        return MetricConfig(
            name="test_metric",
            description="Test metric for agent integration",
            unit="score",
            min_value=0.0,
            max_value=100.0,
            target_value=85.0
        )
    
    @pytest.fixture
    def mock_agent(self, agent_config):
        """Create a mock agent instance."""
        return MockAgent(agent_config)
    
    @pytest.fixture
    def test_scenario(self, scenario_config):
        """Create a test scenario instance."""
        return TestScenario(scenario_config)
    
    @pytest.fixture
    def benchmark_engine(self, benchmark_config):
        """Create a benchmark engine instance."""
        return BenchmarkEngine(benchmark_config)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.is_initialized is False
        
        await mock_agent.initialize()
        
        assert mock_agent.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_agent_input_processing(self, mock_agent):
        """Test agent input processing."""
        await mock_agent.initialize()
        
        input_data = {"content": "Test input"}
        response = await mock_agent.process_input(input_data)
        
        assert response["agent_id"] == mock_agent.config.agent_id
        assert response["response"] == "Mock response to: Test input"
        assert len(mock_agent.responses) == 1
    
    @pytest.mark.asyncio
    async def test_agent_action_execution(self, mock_agent):
        """Test agent action execution."""
        await mock_agent.initialize()
        
        action = {"type": "test_action", "parameters": {"key": "value"}}
        result = await mock_agent.execute_action(action)
        
        assert result["agent_id"] == mock_agent.config.agent_id
        assert result["action"] == "test_action"
        assert result["status"] == "completed"
        assert len(mock_agent.actions_taken) == 1
    
    @pytest.mark.asyncio
    async def test_agent_metrics_collection(self, mock_agent):
        """Test agent metrics collection."""
        await mock_agent.initialize()
        
        # Process some inputs and execute some actions first
        await mock_agent.process_input({"content": "Test input 1"})
        await mock_agent.execute_action({"type": "test_action"})
        await mock_agent.process_input({"content": "Test input 2"})
        
        metrics = await mock_agent.collect_metrics()
        
        assert metrics["agent_id"] == mock_agent.config.agent_id
        assert metrics["responses_count"] == 2
        assert metrics["actions_count"] == 1
        assert "avg_response_time" in metrics
        assert "success_rate" in metrics
    
    @pytest.mark.asyncio
    async def test_scenario_initialization(self, test_scenario):
        """Test scenario initialization."""
        parameters = {"test_data": {"key": "value"}}
        
        await test_scenario.initialize(parameters)
        
        assert test_scenario.is_initialized is True
        assert test_scenario.test_data == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_scenario_agent_setup(self, test_scenario):
        """Test scenario setup for an agent."""
        await test_scenario.initialize({"test_data": {"key": "value"}})
        
        agent_id = "test_agent"
        await test_scenario.setup_for_agent(agent_id)
        
        assert agent_id in test_scenario.agent_states
        assert test_scenario.agent_states[agent_id]["test_data"] == {"key": "value"}
        assert test_scenario.is_setup is True
    
    @pytest.mark.asyncio
    async def test_scenario_tick_update(self, test_scenario):
        """Test scenario tick update."""
        await test_scenario.initialize({"test_data": {"key": "value"}})
        await test_scenario.setup_for_agent("test_agent")
        
        state = Mock(spec=SimulationState)
        
        # Update for tick 5
        await test_scenario.update_tick(5, state)
        
        assert test_scenario.current_tick == 5
        assert len(test_scenario.results) == 1
        assert test_scenario.results[0]["tick"] == 5
        
        # Update for tick 10 (should trigger milestone)
        await test_scenario.update_tick(10, state)
        
        assert test_scenario.current_tick == 10
        assert test_scenario.global_state["milestone"] == 10
    
    @pytest.mark.asyncio
    async def test_scenario_performance_evaluation(self, test_scenario):
        """Test scenario performance evaluation."""
        await test_scenario.initialize({"test_data": {"key": "value"}})
        await test_scenario.setup_for_agent("test_agent")
        
        metrics = await test_scenario.evaluate_agent_performance("test_agent")
        
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "test_scenario"
        assert "test_scenario_score" in metrics
        assert "task_completion_rate" in metrics
        assert "efficiency_metric" in metrics
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_initialization(self, benchmark_engine):
        """Test benchmark engine initialization."""
        assert benchmark_engine.config.name == "agent_integration_test"
        assert benchmark_engine.is_running is False
        assert benchmark_engine.current_tick == 0
        assert len(benchmark_engine.results) == 0
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_agent_registration(self, benchmark_engine, mock_agent):
        """Test agent registration with benchmark engine."""
        await mock_agent.initialize()
        
        benchmark_engine.register_agent(mock_agent)
        
        assert mock_agent.config.agent_id in benchmark_engine.agents
        assert benchmark_engine.agents[mock_agent.config.agent_id] == mock_agent
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_scenario_registration(self, benchmark_engine, test_scenario):
        """Test scenario registration with benchmark engine."""
        benchmark_engine.register_scenario(test_scenario)
        
        assert test_scenario.config.name in benchmark_engine.scenarios
        assert benchmark_engine.scenarios[test_scenario.config.name] == test_scenario
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_metric_registration(self, benchmark_engine, metric_config):
        """Test metric registration with benchmark engine."""
        # Create a simple metric for testing
        class TestMetric(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
            
            def calculate(self, data):
                return 85.0
        
        metric = TestMetric(metric_config)
        benchmark_engine.register_metric(metric)
        
        assert metric.config.name in benchmark_engine.metrics
        assert benchmark_engine.metrics[metric.config.name] == metric
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_execution(self, benchmark_engine, mock_agent, test_scenario, metric_config):
        """Test benchmark engine execution with agent and scenario."""
        # Create and register a simple metric
        class TestMetric(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
            
            def calculate(self, data):
                return 85.0
        
        metric = TestMetric(metric_config)
        
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        benchmark_engine.register_metric(metric)
        
        # Run the benchmark
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario",
            agent_ids=["test_agent"],
            metric_names=["test_metric"]
        )
        
        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_name == "test_scenario"
        assert result.agent_ids == ["test_agent"]
        assert result.metric_names == ["test_metric"]
        assert result.success is True
        assert result.duration_seconds > 0
        assert len(result.results) > 0
        assert "test_metric" in result.results
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_multiple_agents(self, benchmark_engine, scenario_config, metric_config):
        """Test benchmark engine with multiple agents."""
        # Create multiple mock agents
        agent1_config = AgentConfig(agent_id="agent1", agent_type="mock", agent_class="MockAgent")
        agent2_config = AgentConfig(agent_id="agent2", agent_type="mock", agent_class="MockAgent")
        
        agent1 = MockAgent(agent1_config)
        agent2 = MockAgent(agent2_config)
        
        # Create a test scenario
        scenario = TestScenario(scenario_config)
        
        # Create and register a simple metric
        class TestMetric(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
            
            def calculate(self, data):
                return 85.0
        
        metric = TestMetric(metric_config)
        
        # Initialize and register components
        await agent1.initialize()
        await agent2.initialize()
        benchmark_engine.register_agent(agent1)
        benchmark_engine.register_agent(agent2)
        benchmark_engine.register_scenario(scenario)
        benchmark_engine.register_metric(metric)
        
        # Run the benchmark with multiple agents
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario",
            agent_ids=["agent1", "agent2"],
            metric_names=["test_metric"]
        )
        
        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_name == "test_scenario"
        assert set(result.agent_ids) == {"agent1", "agent2"}
        assert result.metric_names == ["test_metric"]
        assert result.success is True
        assert result.duration_seconds > 0
        assert len(result.results) > 0
        assert "test_metric" in result.results
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_multiple_scenarios(self, benchmark_engine, agent_config, metric_config):
        """Test benchmark engine with multiple scenarios."""
        # Create a mock agent
        agent = MockAgent(agent_config)
        
        # Create multiple test scenarios
        scenario1_config = ScenarioConfig(name="scenario1", description="Test scenario 1", domain="test", duration_ticks=30)
        scenario2_config = ScenarioConfig(name="scenario2", description="Test scenario 2", domain="test", duration_ticks=30)
        
        scenario1 = TestScenario(scenario1_config)
        scenario2 = TestScenario(scenario2_config)
        
        # Create and register a simple metric
        class TestMetric(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
            
            def calculate(self, data):
                return 85.0
        
        metric = TestMetric(metric_config)
        
        # Initialize and register components
        await agent.initialize()
        benchmark_engine.register_agent(agent)
        benchmark_engine.register_scenario(scenario1)
        benchmark_engine.register_scenario(scenario2)
        benchmark_engine.register_metric(metric)
        
        # Run benchmarks for both scenarios
        result1 = await benchmark_engine.run_benchmark(
            scenario_name="scenario1",
            agent_ids=["test_agent"],
            metric_names=["test_metric"]
        )
        
        result2 = await benchmark_engine.run_benchmark(
            scenario_name="scenario2",
            agent_ids=["test_agent"],
            metric_names=["test_metric"]
        )
        
        # Verify the results
        assert isinstance(result1, BenchmarkResult)
        assert result1.scenario_name == "scenario1"
        assert result1.success is True
        
        assert isinstance(result2, BenchmarkResult)
        assert result2.scenario_name == "scenario2"
        assert result2.success is True
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_multiple_metrics(self, benchmark_engine, agent_config, scenario_config):
        """Test benchmark engine with multiple metrics."""
        # Create a mock agent and test scenario
        agent = MockAgent(agent_config)
        scenario = TestScenario(scenario_config)
        
        # Create multiple test metrics
        class TestMetric1(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
            
            def calculate(self, data):
                return 85.0
        
        class TestMetric2(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
            
            def calculate(self, data):
                return 90.0
        
        metric1_config = MetricConfig(name="metric1", description="Test metric 1", unit="score", min_value=0.0, max_value=100.0)
        metric2_config = MetricConfig(name="metric2", description="Test metric 2", unit="score", min_value=0.0, max_value=100.0)
        
        metric1 = TestMetric1(metric1_config)
        metric2 = TestMetric2(metric2_config)
        
        # Initialize and register components
        await agent.initialize()
        benchmark_engine.register_agent(agent)
        benchmark_engine.register_scenario(scenario)
        benchmark_engine.register_metric(metric1)
        benchmark_engine.register_metric(metric2)
        
        # Run the benchmark with multiple metrics
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario",
            agent_ids=["test_agent"],
            metric_names=["metric1", "metric2"]
        )
        
        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_name == "test_scenario"
        assert result.agent_ids == ["test_agent"]
        assert set(result.metric_names) == {"metric1", "metric2"}
        assert result.success is True
        assert result.duration_seconds > 0
        assert len(result.results) > 0
        assert "metric1" in result.results
        assert "metric2" in result.results
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_error_handling(self, benchmark_engine, agent_config, scenario_config):
        """Test benchmark engine error handling."""
        # Create a mock agent and test scenario
        agent = MockAgent(agent_config)
        scenario = TestScenario(scenario_config)
        
        # Initialize and register components
        await agent.initialize()
        benchmark_engine.register_agent(agent)
        benchmark_engine.register_scenario(scenario)
        
        # Run the benchmark with non-existent metric
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario",
            agent_ids=["test_agent"],
            metric_names=["nonexistent_metric"]
        )
        
        # Verify the result handles the error gracefully
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_name == "test_scenario"
        assert result.agent_ids == ["test_agent"]
        assert result.success is False  # Should fail due to non-existent metric
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_persistence(self, benchmark_engine, agent_config, scenario_config, metric_config):
        """Test benchmark engine result persistence."""
        # Create a mock agent and test scenario
        agent = MockAgent(agent_config)
        scenario = TestScenario(scenario_config)
        
        # Create and register a simple metric
        class TestMetric(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
            
            def calculate(self, data):
                return 85.0
        
        metric = TestMetric(metric_config)
        
        # Initialize and register components
        await agent.initialize()
        benchmark_engine.register_agent(agent)
        benchmark_engine.register_scenario(scenario)
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
                scenario_name="test_scenario",
                agent_ids=["test_agent"],
                metric_names=["test_metric"]
            )
            
            # Verify the result was saved
            assert os.path.exists(temp_filename)
            
            # Load and verify the saved results
            with open(temp_filename, 'r') as f:
                saved_data = json.load(f)
            
            assert "benchmark_results" in saved_data
            assert len(saved_data["benchmark_results"]) > 0
            
            saved_result = saved_data["benchmark_results"][0]
            assert saved_result["scenario_name"] == "test_scenario"
            assert saved_result["agent_ids"] == ["test_agent"]
            assert saved_result["metric_names"] == ["test_metric"]
            assert saved_result["success"] is True
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_concurrent_execution(self, benchmark_engine, agent_config, scenario_config, metric_config):
        """Test benchmark engine concurrent execution."""
        # Create multiple mock agents
        agent1_config = AgentConfig(agent_id="agent1", agent_type="mock", agent_class="MockAgent")
        agent2_config = AgentConfig(agent_id="agent2", agent_type="mock", agent_class="MockAgent")
        
        agent1 = MockAgent(agent1_config)
        agent2 = MockAgent(agent2_config)
        
        # Create multiple test scenarios
        scenario1_config = ScenarioConfig(name="scenario1", description="Test scenario 1", domain="test", duration_ticks=20)
        scenario2_config = ScenarioConfig(name="scenario2", description="Test scenario 2", domain="test", duration_ticks=20)
        
        scenario1 = TestScenario(scenario1_config)
        scenario2 = TestScenario(scenario2_config)
        
        # Create and register a simple metric
        class TestMetric(BaseMetric):
            def __init__(self, config):
                super().__init__(config)
            
            def calculate(self, data):
                return 85.0
        
        metric = TestMetric(metric_config)
        
        # Initialize and register components
        await agent1.initialize()
        await agent2.initialize()
        benchmark_engine.register_agent(agent1)
        benchmark_engine.register_agent(agent2)
        benchmark_engine.register_scenario(scenario1)
        benchmark_engine.register_scenario(scenario2)
        benchmark_engine.register_metric(metric)
        
        # Run benchmarks concurrently
        tasks = [
            benchmark_engine.run_benchmark(
                scenario_name="scenario1",
                agent_ids=["agent1"],
                metric_names=["test_metric"]
            ),
            benchmark_engine.run_benchmark(
                scenario_name="scenario2",
                agent_ids=["agent2"],
                metric_names=["test_metric"]
            )
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify the results
        assert len(results) == 2
        
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.success is True
            assert result.duration_seconds > 0
            assert len(result.results) > 0
            assert "test_metric" in result.results
        
        # Verify different scenarios and agents were used
        assert results[0].scenario_name != results[1].scenario_name
        assert results[0].agent_ids != results[1].agent_ids