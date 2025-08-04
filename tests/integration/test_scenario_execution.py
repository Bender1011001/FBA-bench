"""
Integration tests for scenario execution and validation pipelines.

This module contains comprehensive integration tests that verify the interaction
between different scenario components, including execution, validation, error handling,
and result processing in the FBA-Bench system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import tempfile
import os
import numpy as np
import uuid

from benchmarking.core.engine import BenchmarkEngine, BenchmarkConfig, BenchmarkResult
from benchmarking.scenarios.base import ScenarioConfig, ScenarioResult, BaseScenario
from benchmarking.scenarios.registry import ScenarioRegistry
from benchmarking.scenarios.templates import (
    ECommerceScenario,
    HealthcareScenario,
    FinancialScenario,
    LegalScenario,
    ScientificScenario
)
from benchmarking.validators.base import BaseValidator, ValidationResult
from benchmarking.validators.registry import ValidatorRegistry
from benchmarking.validators.statistical import StatisticalValidator
from benchmarking.validators.reproducibility import ReproducibilityValidator
from benchmarking.validators.compliance import ComplianceValidator
from agent_runners.base_runner import BaseAgentRunner, AgentConfig
from agent_runners.simulation_runner import SimulationRunner, SimulationState


class MockAgentForScenario(BaseAgentRunner):
    """Mock agent implementation for scenario execution testing."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.responses = []
        self.actions_taken = []
        self.scenario_interactions = []
        self.performance_metrics = {
            "task_completion_rate": 0.0,
            "efficiency_score": 0.0,
            "accuracy_score": 0.0,
            "adaptability_score": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the mock agent."""
        self.is_initialized = True
    
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return a response."""
        response = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "response": f"Mock response to: {input_data.get('content', '')}",
            "confidence": np.random.uniform(0.7, 0.95),
            "processing_time": np.random.uniform(0.01, 0.1)
        }
        
        self.responses.append(response)
        self.scenario_interactions.append({
            "type": "input_processing",
            "timestamp": response["timestamp"],
            "input": input_data,
            "output": response
        })
        
        return response
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return the result."""
        success = np.random.random() > 0.1  # 90% success rate
        
        result = {
            "agent_id": self.config.agent_id,
            "action": action.get("type", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if success else "failed",
            "result": f"Executed action: {action.get('type', 'unknown')}",
            "execution_time": np.random.uniform(0.01, 0.2)
        }
        
        self.actions_taken.append(result)
        self.scenario_interactions.append({
            "type": "action_execution",
            "timestamp": result["timestamp"],
            "action": action,
            "result": result
        })
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return result
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from the agent."""
        metrics = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "responses_count": len(self.responses),
            "actions_count": len(self.actions_taken),
            "interactions_count": len(self.scenario_interactions),
            "performance_metrics": self.performance_metrics.copy()
        }
        
        return metrics
    
    def _update_performance_metrics(self):
        """Update performance metrics based on actions taken."""
        if not self.actions_taken:
            return
        
        # Calculate task completion rate
        completed_actions = len([a for a in self.actions_taken if a["status"] == "completed"])
        self.performance_metrics["task_completion_rate"] = completed_actions / len(self.actions_taken)
        
        # Update other metrics with random values for testing
        self.performance_metrics["efficiency_score"] = np.random.uniform(0.6, 0.95)
        self.performance_metrics["accuracy_score"] = np.random.uniform(0.7, 0.9)
        self.performance_metrics["adaptability_score"] = np.random.uniform(0.65, 0.85)
    
    async def shutdown(self) -> None:
        """Shutdown the mock agent."""
        self.is_initialized = False


class TestScenarioWithValidation(BaseScenario):
    """Test scenario implementation with validation for integration testing."""
    
    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        self.validation_results = []
        self.validation_errors = []
        self.execution_history = []
        self.checkpoints = {}
    
    def _validate_domain_parameters(self) -> List[str]:
        """Validate domain-specific parameters."""
        errors = []
        
        if "difficulty" in self.parameters:
            difficulty = self.parameters["difficulty"]
            if difficulty not in ["easy", "medium", "hard"]:
                errors.append("Difficulty must be one of: easy, medium, hard")
        
        if "complexity" in self.parameters:
            complexity = self.parameters["complexity"]
            if not isinstance(complexity, (int, float)) or complexity < 1 or complexity > 10:
                errors.append("Complexity must be a number between 1 and 10")
        
        return errors
    
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize the test scenario."""
        await super().initialize(parameters)
        self.test_data = parameters.get("test_data", {})
        self.validation_config = parameters.get("validation_config", {})
        
        # Initialize checkpoints
        self.checkpoints = {
            "initialization": {
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "parameters": parameters
            }
        }
        
        self.execution_history.append({
            "phase": "initialization",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        })
    
    async def setup_for_agent(self, agent_id: str) -> None:
        """Setup the scenario for a specific agent."""
        await super().setup_for_agent(agent_id)
        self.agent_states[agent_id]["test_data"] = self.test_data
        self.agent_states[agent_id]["validation_results"] = []
        self.agent_states[agent_id]["execution_history"] = []
        
        # Record setup checkpoint
        self.checkpoints[f"setup_{agent_id}"] = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "agent_id": agent_id
        }
        
        self.execution_history.append({
            "phase": "setup",
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "agent_id": agent_id
        })
    
    async def update_tick(self, tick: int, state: SimulationState) -> None:
        """Update the scenario for a specific tick."""
        await super().update_tick(tick, state)
        
        # Simulate scenario-specific state changes
        for agent_id in self.agent_states:
            agent_state = self.agent_states[agent_id]
            
            # Update agent progress
            progress = tick / self.duration_ticks
            agent_state["progress"] = progress
            
            # Simulate task completion
            if tick % 5 == 0:
                tasks_completed = agent_state.get("tasks_completed", 0)
                agent_state["tasks_completed"] = tasks_completed + np.random.randint(1, 3)
            
            # Record tick execution
            agent_state["execution_history"].append({
                "tick": tick,
                "timestamp": datetime.now().isoformat(),
                "progress": progress,
                "tasks_completed": agent_state.get("tasks_completed", 0)
            })
            
            # Validate agent state at checkpoints
            if tick % 10 == 0:
                validation_result = self._validate_agent_state(agent_id, agent_state)
                agent_state["validation_results"].append(validation_result)
        
        # Record tick checkpoint
        self.checkpoints[f"tick_{tick}"] = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "tick": tick,
            "agent_states": {aid: state.copy() for aid, state in self.agent_states.items()}
        }
        
        self.execution_history.append({
            "phase": "tick_update",
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "tick": tick
        })
    
    async def evaluate_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Evaluate agent performance in the scenario."""
        base_metrics = await super().evaluate_agent_performance(agent_id)
        
        agent_state = self.agent_states[agent_id]
        
        # Calculate scenario-specific metrics
        progress = agent_state.get("progress", 0.0)
        tasks_completed = agent_state.get("tasks_completed", 0)
        
        # Calculate efficiency metrics
        efficiency_score = progress / (self.current_tick + 1) if self.current_tick > 0 else 0
        
        # Calculate task completion rate
        expected_tasks = self.duration_ticks / 5 * 2  # Expected tasks based on simulation
        task_completion_rate = tasks_completed / expected_tasks if expected_tasks > 0 else 0
        
        # Aggregate validation results
        validation_results = agent_state.get("validation_results", [])
        validation_success_rate = len([r for r in validation_results if r["valid"]]) / len(validation_results) if validation_results else 1.0
        
        scenario_metrics = {
            "progress": progress,
            "tasks_completed": tasks_completed,
            "efficiency_score": efficiency_score,
            "task_completion_rate": task_completion_rate,
            "validation_success_rate": validation_success_rate,
            "overall_score": (progress + efficiency_score + task_completion_rate + validation_success_rate) / 4
        }
        
        return {**base_metrics, **scenario_metrics}
    
    def _validate_agent_state(self, agent_id: str, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent state at checkpoints."""
        validation_result = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate progress
        progress = agent_state.get("progress", 0.0)
        if progress < 0 or progress > 1:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Invalid progress value: {progress}")
        
        # Validate tasks completed
        tasks_completed = agent_state.get("tasks_completed", 0)
        if tasks_completed < 0:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Negative tasks completed: {tasks_completed}")
        
        # Check for warnings
        if progress < 0.3 and self.current_tick > self.duration_ticks * 0.5:
            validation_result["warnings"].append("Low progress for current tick")
        
        if tasks_completed < self.current_tick / 5:
            validation_result["warnings"].append("Low task completion rate")
        
        self.validation_results.append(validation_result)
        
        return validation_result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation results."""
        total_validations = len(self.validation_results)
        successful_validations = len([r for r in self.validation_results if r["valid"]])
        
        total_errors = sum(len(r["errors"]) for r in self.validation_results)
        total_warnings = sum(len(r["warnings"]) for r in self.validation_results)
        
        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": total_validations - successful_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "validation_errors": self.validation_errors
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution history."""
        return {
            "total_phases": len(self.execution_history),
            "checkpoints": len(self.checkpoints),
            "execution_history": self.execution_history,
            "checkpoints_data": self.checkpoints
        }


class TestScenarioExecution:
    """Test cases for scenario execution and validation pipelines."""
    
    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(
            agent_id="test_agent",
            agent_type="mock_for_scenario",
            agent_class="MockAgentForScenario",
            parameters={"test_param": "test_value"}
        )
    
    @pytest.fixture
    def scenario_config(self):
        """Create a test scenario configuration."""
        return ScenarioConfig(
            name="test_scenario_with_validation",
            description="Test scenario for execution and validation",
            domain="test",
            duration_ticks=30,
            parameters={
                "difficulty": "medium",
                "complexity": 5,
                "test_data": {"key": "value"},
                "validation_config": {
                    "enable_validation": True,
                    "validation_interval": 10
                }
            }
        )
    
    @pytest.fixture
    def benchmark_config(self):
        """Create a test benchmark configuration."""
        return BenchmarkConfig(
            name="scenario_execution_test",
            description="Test scenario execution and validation pipelines",
            max_duration=300,
            tick_interval=0.1,
            metrics_collection_interval=1.0
        )
    
    @pytest.fixture
    def mock_agent(self, agent_config):
        """Create a mock agent for scenario testing."""
        return MockAgentForScenario(agent_config)
    
    @pytest.fixture
    def test_scenario(self, scenario_config):
        """Create a test scenario with validation."""
        return TestScenarioWithValidation(scenario_config)
    
    @pytest.fixture
    def benchmark_engine(self, benchmark_config):
        """Create a benchmark engine instance."""
        return BenchmarkEngine(benchmark_config)
    
    @pytest.fixture
    def statistical_validator(self):
        """Create a statistical validator instance."""
        return StatisticalValidator()
    
    @pytest.fixture
    def reproducibility_validator(self):
        """Create a reproducibility validator instance."""
        return ReproducibilityValidator()
    
    @pytest.fixture
    def compliance_validator(self):
        """Create a compliance validator instance."""
        return ComplianceValidator()
    
    @pytest.mark.asyncio
    async def test_scenario_initialization(self, test_scenario):
        """Test scenario initialization with validation."""
        parameters = {
            "difficulty": "medium",
            "complexity": 5,
            "test_data": {"key": "value"},
            "validation_config": {"enable_validation": True}
        }
        
        await test_scenario.initialize(parameters)
        
        # Verify initialization
        assert test_scenario.is_initialized is True
        assert test_scenario.test_data == {"key": "value"}
        assert test_scenario.validation_config == {"enable_validation": True}
        
        # Verify checkpoints
        assert "initialization" in test_scenario.checkpoints
        assert test_scenario.checkpoints["initialization"]["status"] == "completed"
        
        # Verify execution history
        assert len(test_scenario.execution_history) == 1
        assert test_scenario.execution_history[0]["phase"] == "initialization"
        assert test_scenario.execution_history[0]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_scenario_agent_setup(self, test_scenario):
        """Test scenario setup for an agent."""
        await test_scenario.initialize({
            "difficulty": "medium",
            "complexity": 5,
            "test_data": {"key": "value"}
        })
        
        agent_id = "test_agent"
        await test_scenario.setup_for_agent(agent_id)
        
        # Verify agent setup
        assert agent_id in test_scenario.agent_states
        assert test_scenario.agent_states[agent_id]["test_data"] == {"key": "value"}
        assert "validation_results" in test_scenario.agent_states[agent_id]
        assert "execution_history" in test_scenario.agent_states[agent_id]
        assert test_scenario.is_setup is True
        
        # Verify checkpoints
        assert f"setup_{agent_id}" in test_scenario.checkpoints
        assert test_scenario.checkpoints[f"setup_{agent_id}"]["status"] == "completed"
        assert test_scenario.checkpoints[f"setup_{agent_id}"]["agent_id"] == agent_id
        
        # Verify execution history
        assert len(test_scenario.execution_history) == 2
        assert test_scenario.execution_history[1]["phase"] == "setup"
        assert test_scenario.execution_history[1]["status"] == "completed"
        assert test_scenario.execution_history[1]["agent_id"] == agent_id
    
    @pytest.mark.asyncio
    async def test_scenario_tick_execution(self, test_scenario):
        """Test scenario tick execution with validation."""
        await test_scenario.initialize({
            "difficulty": "medium",
            "complexity": 5,
            "test_data": {"key": "value"}
        })
        await test_scenario.setup_for_agent("test_agent")
        
        state = Mock(spec=SimulationState)
        
        # Execute ticks
        for tick in range(1, 11):
            await test_scenario.update_tick(tick, state)
            
            # Verify tick state
            assert test_scenario.current_tick == tick
            assert len(test_scenario.results) == tick
            
            # Verify agent state
            agent_state = test_scenario.agent_states["test_agent"]
            assert agent_state["progress"] == tick / test_scenario.duration_ticks
            assert "execution_history" in agent_state
            assert len(agent_state["execution_history"]) == tick
            
            # Verify checkpoints
            assert f"tick_{tick}" in test_scenario.checkpoints
            assert test_scenario.checkpoints[f"tick_{tick}"]["status"] == "completed"
            assert test_scenario.checkpoints[f"tick_{tick}"]["tick"] == tick
            
            # Verify execution history
            assert len(test_scenario.execution_history) == 1 + tick  # initialization + setup + ticks
            assert test_scenario.execution_history[1 + tick]["phase"] == "tick_update"
            assert test_scenario.execution_history[1 + tick]["status"] == "completed"
            assert test_scenario.execution_history[1 + tick]["tick"] == tick
            
            # Verify validation at checkpoints (every 10 ticks)
            if tick % 10 == 0:
                assert len(agent_state["validation_results"]) > 0
                validation_result = agent_state["validation_results"][-1]
                assert validation_result["agent_id"] == "test_agent"
                assert "valid" in validation_result
                assert "errors" in validation_result
                assert "warnings" in validation_result
    
    @pytest.mark.asyncio
    async def test_scenario_performance_evaluation(self, test_scenario):
        """Test scenario performance evaluation."""
        await test_scenario.initialize({
            "difficulty": "medium",
            "complexity": 5,
            "test_data": {"key": "value"}
        })
        await test_scenario.setup_for_agent("test_agent")
        
        # Execute some ticks
        state = Mock(spec=SimulationState)
        for tick in range(1, 21):
            await test_scenario.update_tick(tick, state)
        
        # Evaluate agent performance
        metrics = await test_scenario.evaluate_agent_performance("test_agent")
        
        # Verify metrics structure
        assert "agent_id" in metrics
        assert "scenario_name" in metrics
        assert "progress" in metrics
        assert "tasks_completed" in metrics
        assert "efficiency_score" in metrics
        assert "task_completion_rate" in metrics
        assert "validation_success_rate" in metrics
        assert "overall_score" in metrics
        
        # Verify metrics values
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "test_scenario_with_validation"
        assert 0 <= metrics["progress"] <= 1
        assert metrics["tasks_completed"] >= 0
        assert metrics["efficiency_score"] >= 0
        assert 0 <= metrics["task_completion_rate"] <= 1
        assert 0 <= metrics["validation_success_rate"] <= 1
        assert 0 <= metrics["overall_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_scenario_validation_summary(self, test_scenario):
        """Test scenario validation summary."""
        await test_scenario.initialize({
            "difficulty": "medium",
            "complexity": 5,
            "test_data": {"key": "value"}
        })
        await test_scenario.setup_for_agent("test_agent")
        
        # Execute ticks to trigger validation
        state = Mock(spec=SimulationState)
        for tick in range(1, 21):
            await test_scenario.update_tick(tick, state)
        
        # Get validation summary
        summary = test_scenario.get_validation_summary()
        
        # Verify summary structure
        assert "total_validations" in summary
        assert "successful_validations" in summary
        assert "failed_validations" in summary
        assert "success_rate" in summary
        assert "total_errors" in summary
        assert "total_warnings" in summary
        assert "validation_errors" in summary
        
        # Verify summary values
        assert summary["total_validations"] > 0
        assert summary["successful_validations"] >= 0
        assert summary["failed_validations"] >= 0
        assert 0 <= summary["success_rate"] <= 1
        assert summary["total_errors"] >= 0
        assert summary["total_warnings"] >= 0
        assert isinstance(summary["validation_errors"], list)
    
    @pytest.mark.asyncio
    async def test_scenario_execution_summary(self, test_scenario):
        """Test scenario execution summary."""
        await test_scenario.initialize({
            "difficulty": "medium",
            "complexity": 5,
            "test_data": {"key": "value"}
        })
        await test_scenario.setup_for_agent("test_agent")
        
        # Execute some ticks
        state = Mock(spec=SimulationState)
        for tick in range(1, 11):
            await test_scenario.update_tick(tick, state)
        
        # Get execution summary
        summary = test_scenario.get_execution_summary()
        
        # Verify summary structure
        assert "total_phases" in summary
        assert "checkpoints" in summary
        assert "execution_history" in summary
        assert "checkpoints_data" in summary
        
        # Verify summary values
        assert summary["total_phases"] > 0
        assert summary["checkpoints"] > 0
        assert isinstance(summary["execution_history"], list)
        assert isinstance(summary["checkpoints_data"], dict)
        
        # Verify execution history
        for phase in summary["execution_history"]:
            assert "phase" in phase
            assert "timestamp" in phase
            assert "status" in phase
        
        # Verify checkpoints data
        for checkpoint_name, checkpoint_data in summary["checkpoints_data"].items():
            assert "timestamp" in checkpoint_data
            assert "status" in checkpoint_data
    
    @pytest.mark.asyncio
    async def test_scenario_domain_parameter_validation(self):
        """Test scenario domain parameter validation."""
        # Test with valid parameters
        valid_config = ScenarioConfig(
            name="valid_scenario",
            description="Valid scenario",
            domain="test",
            duration_ticks=30,
            parameters={
                "difficulty": "medium",
                "complexity": 5
            }
        )
        
        valid_scenario = TestScenarioWithValidation(valid_config)
        errors = valid_scenario.validate()
        
        assert len(errors) == 0
        
        # Test with invalid difficulty
        invalid_difficulty_config = ScenarioConfig(
            name="invalid_difficulty_scenario",
            description="Invalid difficulty scenario",
            domain="test",
            duration_ticks=30,
            parameters={
                "difficulty": "invalid",
                "complexity": 5
            }
        )
        
        invalid_difficulty_scenario = TestScenarioWithValidation(invalid_difficulty_config)
        errors = invalid_difficulty_scenario.validate()
        
        assert len(errors) > 0
        assert any("Difficulty must be one of" in error for error in errors)
        
        # Test with invalid complexity
        invalid_complexity_config = ScenarioConfig(
            name="invalid_complexity_scenario",
            description="Invalid complexity scenario",
            domain="test",
            duration_ticks=30,
            parameters={
                "difficulty": "medium",
                "complexity": 15
            }
        )
        
        invalid_complexity_scenario = TestScenarioWithValidation(invalid_complexity_config)
        errors = invalid_complexity_scenario.validate()
        
        assert len(errors) > 0
        assert any("Complexity must be a number between 1 and 10" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_scenario_template_execution(self, scenario_config):
        """Test scenario template execution."""
        # Test ECommerce scenario
        ecommerce_config = ScenarioConfig(
            name="ecommerce_test",
            description="ECommerce test scenario",
            domain="ecommerce",
            duration_ticks=20,
            parameters={
                "customer_count": 10,
                "product_count": 50
            }
        )
        
        ecommerce_scenario = ECommerceScenario(ecommerce_config)
        await ecommerce_scenario.initialize(ecommerce_config.parameters)
        await ecommerce_scenario.setup_for_agent("test_agent")
        
        # Execute some ticks
        state = Mock(spec=SimulationState)
        for tick in range(1, 6):
            await ecommerce_scenario.update_tick(tick, state)
        
        # Evaluate agent performance
        metrics = await ecommerce_scenario.evaluate_agent_performance("test_agent")
        
        # Verify metrics
        assert "agent_id" in metrics
        assert "scenario_name" in metrics
        assert metrics["scenario_name"] == "ecommerce_test"
        
        # Test Healthcare scenario
        healthcare_config = ScenarioConfig(
            name="healthcare_test",
            description="Healthcare test scenario",
            domain="healthcare",
            duration_ticks=20,
            parameters={
                "patient_count": 10,
                "condition_count": 5
            }
        )
        
        healthcare_scenario = HealthcareScenario(healthcare_config)
        await healthcare_scenario.initialize(healthcare_config.parameters)
        await healthcare_scenario.setup_for_agent("test_agent")
        
        # Execute some ticks
        for tick in range(1, 6):
            await healthcare_scenario.update_tick(tick, state)
        
        # Evaluate agent performance
        metrics = await healthcare_scenario.evaluate_agent_performance("test_agent")
        
        # Verify metrics
        assert "agent_id" in metrics
        assert "scenario_name" in metrics
        assert metrics["scenario_name"] == "healthcare_test"
        
        # Test Financial scenario
        financial_config = ScenarioConfig(
            name="financial_test",
            description="Financial test scenario",
            domain="financial",
            duration_ticks=20,
            parameters={
                "transaction_count": 100,
                "account_count": 20
            }
        )
        
        financial_scenario = FinancialScenario(financial_config)
        await financial_scenario.initialize(financial_config.parameters)
        await financial_scenario.setup_for_agent("test_agent")
        
        # Execute some ticks
        for tick in range(1, 6):
            await financial_scenario.update_tick(tick, state)
        
        # Evaluate agent performance
        metrics = await financial_scenario.evaluate_agent_performance("test_agent")
        
        # Verify metrics
        assert "agent_id" in metrics
        assert "scenario_name" in metrics
        assert metrics["scenario_name"] == "financial_test"
        
        # Test Legal scenario
        legal_config = ScenarioConfig(
            name="legal_test",
            description="Legal test scenario",
            domain="legal",
            duration_ticks=20,
            parameters={
                "document_count": 20,
                "case_count": 5
            }
        )
        
        legal_scenario = LegalScenario(legal_config)
        await legal_scenario.initialize(legal_config.parameters)
        await legal_scenario.setup_for_agent("test_agent")
        
        # Execute some ticks
        for tick in range(1, 6):
            await legal_scenario.update_tick(tick, state)
        
        # Evaluate agent performance
        metrics = await legal_scenario.evaluate_agent_performance("test_agent")
        
        # Verify metrics
        assert "agent_id" in metrics
        assert "scenario_name" in metrics
        assert metrics["scenario_name"] == "legal_test"
        
        # Test Scientific scenario
        scientific_config = ScenarioConfig(
            name="scientific_test",
            description="Scientific test scenario",
            domain="scientific",
            duration_ticks=20,
            parameters={
                "dataset_count": 10,
                "research_field": "biology"
            }
        )
        
        scientific_scenario = ScientificScenario(scientific_config)
        await scientific_scenario.initialize(scientific_config.parameters)
        await scientific_scenario.setup_for_agent("test_agent")
        
        # Execute some ticks
        for tick in range(1, 6):
            await scientific_scenario.update_tick(tick, state)
        
        # Evaluate agent performance
        metrics = await scientific_scenario.evaluate_agent_performance("test_agent")
        
        # Verify metrics
        assert "agent_id" in metrics
        assert "scenario_name" in metrics
        assert metrics["scenario_name"] == "scientific_test"
    
    @pytest.mark.asyncio
    async def test_scenario_with_benchmark_engine(self, benchmark_engine, mock_agent, test_scenario):
        """Test scenario execution with benchmark engine."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        
        # Run the benchmark
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario_with_validation",
            agent_ids=["test_agent"]
        )
        
        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_name == "test_scenario_with_validation"
        assert result.agent_ids == ["test_agent"]
        assert result.success is True
        assert result.duration_seconds > 0
        
        # Verify scenario execution
        assert test_scenario.current_tick > 0
        assert len(test_scenario.results) > 0
        assert len(test_scenario.execution_history) > 0
        assert len(test_scenario.checkpoints) > 0
        
        # Verify validation results
        validation_summary = test_scenario.get_validation_summary()
        assert validation_summary["total_validations"] > 0
        
        # Verify execution summary
        execution_summary = test_scenario.get_execution_summary()
        assert execution_summary["total_phases"] > 0
        assert execution_summary["checkpoints"] > 0
    
    @pytest.mark.asyncio
    async def test_scenario_error_handling(self, test_scenario):
        """Test scenario error handling."""
        await test_scenario.initialize({
            "difficulty": "medium",
            "complexity": 5,
            "test_data": {"key": "value"}
        })
        await test_scenario.setup_for_agent("test_agent")
        
        # Test with invalid tick
        with pytest.raises(Exception):
            await test_scenario.update_tick(-1, Mock())
        
        # Test with invalid state
        with pytest.raises(Exception):
            await test_scenario.update_tick(1, None)
        
        # Test evaluation for non-existent agent
        with pytest.raises(Exception):
            await test_scenario.evaluate_agent_performance("nonexistent_agent")
    
    @pytest.mark.asyncio
    async def test_scenario_concurrent_execution(self, benchmark_engine, scenario_config):
        """Test scenario concurrent execution."""
        # Create multiple mock agents
        agent1_config = AgentConfig(agent_id="agent1", agent_type="mock_for_scenario", agent_class="MockAgentForScenario")
        agent2_config = AgentConfig(agent_id="agent2", agent_type="mock_for_scenario", agent_class="MockAgentForScenario")
        
        agent1 = MockAgentForScenario(agent1_config)
        agent2 = MockAgentForScenario(agent2_config)
        
        # Create multiple test scenarios
        scenario1_config = ScenarioConfig(
            name="scenario1",
            description="Test scenario 1",
            domain="test",
            duration_ticks=20,
            parameters={"difficulty": "medium", "complexity": 5}
        )
        
        scenario2_config = ScenarioConfig(
            name="scenario2",
            description="Test scenario 2",
            domain="test",
            duration_ticks=20,
            parameters={"difficulty": "hard", "complexity": 8}
        )
        
        scenario1 = TestScenarioWithValidation(scenario1_config)
        scenario2 = TestScenarioWithValidation(scenario2_config)
        
        # Initialize and register components
        await agent1.initialize()
        await agent2.initialize()
        benchmark_engine.register_agent(agent1)
        benchmark_engine.register_agent(agent2)
        benchmark_engine.register_scenario(scenario1)
        benchmark_engine.register_scenario(scenario2)
        
        # Run benchmarks concurrently
        tasks = [
            benchmark_engine.run_benchmark(
                scenario_name="scenario1",
                agent_ids=["agent1"]
            ),
            benchmark_engine.run_benchmark(
                scenario_name="scenario2",
                agent_ids=["agent2"]
            )
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify the results
        assert len(results) == 2
        
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.success is True
            assert result.duration_seconds > 0
        
        # Verify different scenarios and agents were used
        assert results[0].scenario_name != results[1].scenario_name
        assert results[0].agent_ids != results[1].agent_ids
        
        # Verify scenario execution
        assert scenario1.current_tick > 0
        assert scenario2.current_tick > 0
        assert len(scenario1.results) > 0
        assert len(scenario2.results) > 0
    
    @pytest.mark.asyncio
    async def test_scenario_validation_integration(self, benchmark_engine, mock_agent, test_scenario, 
                                                  statistical_validator, reproducibility_validator, compliance_validator):
        """Test scenario validation integration with benchmark engine."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        
        # Register validators
        benchmark_engine.register_validator(statistical_validator)
        benchmark_engine.register_validator(reproducibility_validator)
        benchmark_engine.register_validator(compliance_validator)
        
        # Configure validation
        benchmark_engine.config.enable_validation = True
        benchmark_engine.config.validation_interval = 5
        
        # Run the benchmark
        result = await benchmark_engine.run_benchmark(
            scenario_name="test_scenario_with_validation",
            agent_ids=["test_agent"]
        )
        
        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.success is True
        
        # Verify validation results
        assert "validation_results" in result.results
        validation_results = result.results["validation_results"]
        
        assert "statistical_validation" in validation_results
        assert "reproducibility_validation" in validation_results
        assert "compliance_validation" in validation_results
        
        # Verify statistical validation
        statistical = validation_results["statistical_validation"]
        assert "valid" in statistical
        assert "confidence" in statistical
        assert "significance" in statistical
        
        # Verify reproducibility validation
        reproducibility = validation_results["reproducibility_validation"]
        assert "valid" in reproducibility
        assert "reproducibility_score" in reproducibility
        assert "consistency" in reproducibility
        
        # Verify compliance validation
        compliance = validation_results["compliance_validation"]
        assert "valid" in compliance
        assert "compliance_score" in compliance
        assert "violations" in compliance
        
        # Verify scenario validation
        scenario_validation = test_scenario.get_validation_summary()
        assert scenario_validation["total_validations"] > 0
        assert 0 <= scenario_validation["success_rate"] <= 1
    
    @pytest.mark.asyncio
    async def test_scenario_result_persistence(self, benchmark_engine, mock_agent, test_scenario):
        """Test scenario result persistence."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        
        # Create a temporary file for persistence
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Configure the benchmark engine to save results
            benchmark_engine.config.save_results = True
            benchmark_engine.config.results_file = temp_filename
            
            # Run the benchmark
            result = await benchmark_engine.run_benchmark(
                scenario_name="test_scenario_with_validation",
                agent_ids=["test_agent"]
            )
            
            # Verify the result was saved
            assert os.path.exists(temp_filename)
            
            # Load and verify the saved results
            with open(temp_filename, 'r') as f:
                saved_data = json.load(f)
            
            assert "benchmark_results" in saved_data
            assert len(saved_data["benchmark_results"]) > 0
            
            saved_result = saved_data["benchmark_results"][0]
            assert saved_result["scenario_name"] == "test_scenario_with_validation"
            assert saved_result["agent_ids"] == ["test_agent"]
            assert saved_result["success"] is True
            
            # Verify scenario-specific data persistence
            assert "scenario_data" in saved_result
            scenario_data = saved_result["scenario_data"]
            assert "execution_summary" in scenario_data
            assert "validation_summary" in scenario_data
            
            # Verify execution summary persistence
            execution_summary = scenario_data["execution_summary"]
            assert "total_phases" in execution_summary
            assert "checkpoints" in execution_summary
            assert "execution_history" in execution_summary
            
            # Verify validation summary persistence
            validation_summary = scenario_data["validation_summary"]
            assert "total_validations" in validation_summary
            assert "successful_validations" in validation_summary
            assert "success_rate" in validation_summary
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)