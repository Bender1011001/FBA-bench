"""
Unit tests for the scenario framework.

This module provides comprehensive tests for the scenario framework components,
including base classes, registry, and template implementations.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from benchmarking.scenarios.base import (
    ScenarioConfig,
    ScenarioResult,
    BaseScenario,
    ScenarioTemplate
)
from benchmarking.scenarios.registry import (
    ScenarioRegistration,
    ScenarioRegistry,
    registry
)
from benchmarking.scenarios.templates import (
    ECommerceScenario,
    HealthcareScenario,
    FinancialScenario,
    LegalScenario,
    ScientificScenario
)
from agent_runners.base_runner import SimulationState


class TestScenarioConfig:
    """Test cases for ScenarioConfig class."""
    
    def test_scenario_config_creation(self):
        """Test creating a scenario configuration."""
        config = ScenarioConfig(
            name="test_scenario",
            description="Test scenario description",
            domain="test_domain",
            duration_ticks=100,
            parameters={"param1": "value1"},
            difficulty="medium"
        )
        
        assert config.name == "test_scenario"
        assert config.description == "Test scenario description"
        assert config.domain == "test_domain"
        assert config.duration_ticks == 100
        assert config.parameters == {"param1": "value1"}
        assert config.enabled is True
        assert config.difficulty == "medium"
        assert config.metadata == {}
    
    def test_scenario_config_defaults(self):
        """Test scenario configuration with default values."""
        config = ScenarioConfig(
            name="test_scenario",
            description="Test scenario description",
            domain="test_domain",
            duration_ticks=100
        )
        
        assert config.parameters == {}
        assert config.enabled is True
        assert config.difficulty == "medium"
        assert config.metadata == {}


class TestScenarioResult:
    """Test cases for ScenarioResult class."""
    
    def test_scenario_result_creation(self):
        """Test creating a scenario result."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=60)
        
        result = ScenarioResult(
            scenario_name="test_scenario",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=60.0,
            success=True,
            errors=["error1"],
            metadata={"key": "value"},
            tick_results=[{"tick": 1, "data": "test"}]
        )
        
        assert result.scenario_name == "test_scenario"
        assert result.start_time == start_time
        assert result.end_time == end_time
        assert result.duration_seconds == 60.0
        assert result.success is True
        assert result.errors == ["error1"]
        assert result.metadata == {"key": "value"}
        assert result.tick_results == [{"tick": 1, "data": "test"}]
    
    def test_scenario_result_defaults(self):
        """Test scenario result with default values."""
        result = ScenarioResult(
            scenario_name="test_scenario",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0
        )
        
        assert result.success is True
        assert result.errors == []
        assert result.metadata == {}
        assert result.tick_results == []


class MockScenario(BaseScenario):
    """Mock scenario implementation for testing."""
    
    def _validate_domain_parameters(self) -> List[str]:
        return []
    
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        self.is_initialized = True
    
    async def setup_for_agent(self, agent_id: str) -> None:
        self.is_setup = True
    
    async def update_tick(self, tick: int, state: SimulationState) -> None:
        self.current_tick = tick
    
    async def get_scenario_state(self) -> Dict[str, Any]:
        return {"tick": self.current_tick}
    
    async def evaluate_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        return {"agent_id": agent_id, "score": 0.5}


class TestBaseScenario:
    """Test cases for BaseScenario class."""
    
    @pytest.fixture
    def scenario_config(self):
        """Create a test scenario configuration."""
        return ScenarioConfig(
            name="test_scenario",
            description="Test scenario description",
            domain="test_domain",
            duration_ticks=100
        )
    
    @pytest.fixture
    def mock_scenario(self, scenario_config):
        """Create a mock scenario instance."""
        return MockScenario(scenario_config)
    
    def test_scenario_initialization(self, scenario_config):
        """Test scenario initialization."""
        scenario = MockScenario(scenario_config)
        
        assert scenario.config == scenario_config
        assert scenario.name == "test_scenario"
        assert scenario.description == "Test scenario description"
        assert scenario.domain == "test_domain"
        assert scenario.duration_ticks == 100
        assert scenario.current_tick == 0
        assert scenario.is_initialized is False
        assert scenario.is_setup is False
        assert scenario.results == []
        assert scenario.validation_errors == []
    
    def test_scenario_is_valid_property(self, mock_scenario):
        """Test is_valid property."""
        assert mock_scenario.is_valid is True
        
        mock_scenario.validation_errors = ["error1"]
        assert mock_scenario.is_valid is False
    
    def test_scenario_validate_valid_config(self, mock_scenario):
        """Test validation with valid configuration."""
        errors = mock_scenario.validate()
        assert errors == []
    
    @pytest.mark.parametrize("missing_field,expected_error", [
        ("name", "Scenario name cannot be empty"),
        ("description", "Scenario description cannot be empty"),
        ("domain", "Scenario domain cannot be empty")
    ])
    def test_scenario_validate_missing_fields(self, missing_field, expected_error):
        """Test validation with missing required fields."""
        config = ScenarioConfig(
            name="test",
            description="test",
            domain="test",
            duration_ticks=100
        )
        setattr(config, missing_field, "")
        
        scenario = MockScenario(config)
        errors = scenario.validate()
        
        assert expected_error in errors
    
    def test_scenario_validate_invalid_duration(self):
        """Test validation with invalid duration."""
        config = ScenarioConfig(
            name="test",
            description="test",
            domain="test",
            duration_ticks=0
        )
        
        scenario = MockScenario(config)
        errors = scenario.validate()
        
        assert "Duration ticks must be positive" in errors
    
    @pytest.mark.parametrize("difficulty,valid", [
        ("easy", True),
        ("medium", True),
        ("hard", True),
        ("invalid", False)
    ])
    def test_scenario_validate_difficulty(self, difficulty, valid):
        """Test validation of difficulty field."""
        config = ScenarioConfig(
            name="test",
            description="test",
            domain="test",
            duration_ticks=100,
            difficulty=difficulty
        )
        
        scenario = MockScenario(config)
        errors = scenario.validate()
        
        if valid:
            assert "Difficulty must be one of" not in errors
        else:
            assert "Difficulty must be one of" in errors
    
    @pytest.mark.asyncio
    async def test_scenario_start_valid(self, mock_scenario):
        """Test starting a valid scenario."""
        await mock_scenario.start()
        
        assert mock_scenario.current_tick == 0
        assert mock_scenario.results == []
        assert mock_scenario.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_scenario_start_invalid(self, mock_scenario):
        """Test starting an invalid scenario."""
        mock_scenario.validation_errors = ["Invalid configuration"]
        
        with pytest.raises(ValueError, match="Scenario configuration is invalid"):
            await mock_scenario.start()
    
    @pytest.mark.asyncio
    async def test_scenario_stop(self, mock_scenario):
        """Test stopping a scenario."""
        mock_scenario.is_setup = True
        
        await mock_scenario.stop()
        
        assert mock_scenario.is_setup is False
    
    @pytest.mark.asyncio
    async def test_scenario_reset(self, mock_scenario):
        """Test resetting a scenario."""
        mock_scenario.current_tick = 50
        mock_scenario.results = [{"tick": 1}]
        mock_scenario.is_setup = True
        
        await mock_scenario.reset()
        
        assert mock_scenario.current_tick == 0
        assert mock_scenario.results == []
        assert mock_scenario.is_setup is False
        assert mock_scenario.is_initialized is True
    
    def test_scenario_get_summary(self, mock_scenario):
        """Test getting scenario summary."""
        summary = mock_scenario.get_summary()
        
        expected_keys = [
            "name", "description", "domain", "duration_ticks",
            "current_tick", "is_valid", "is_initialized",
            "is_setup", "parameters", "metadata", "validation_errors"
        ]
        
        for key in expected_keys:
            assert key in summary
    
    def test_scenario_record_tick_result(self, mock_scenario):
        """Test recording tick results."""
        mock_scenario.record_tick_result(1, {"data": "test"})
        
        assert len(mock_scenario.results) == 1
        assert mock_scenario.results[0]["tick"] == 1
        assert mock_scenario.results[0]["data"] == "test"
        assert "timestamp" in mock_scenario.results[0]
    
    def test_scenario_get_tick_results_all(self, mock_scenario):
        """Test getting all tick results."""
        mock_scenario.results = [
            {"tick": 1, "data": "test1"},
            {"tick": 2, "data": "test2"}
        ]
        
        results = mock_scenario.get_tick_results()
        
        assert len(results) == 2
        assert results[0]["tick"] == 1
        assert results[1]["tick"] == 2
    
    def test_scenario_get_tick_results_specific(self, mock_scenario):
        """Test getting tick results for a specific tick."""
        mock_scenario.results = [
            {"tick": 1, "data": "test1"},
            {"tick": 2, "data": "test2"},
            {"tick": 2, "data": "test3"}
        ]
        
        results = mock_scenario.get_tick_results(2)
        
        assert len(results) == 2
        assert results[0]["tick"] == 2
        assert results[1]["tick"] == 2
    
    def test_scenario_get_latest_tick_result(self, mock_scenario):
        """Test getting the latest tick result."""
        mock_scenario.results = [
            {"tick": 1, "data": "test1"},
            {"tick": 2, "data": "test2"}
        ]
        
        result = mock_scenario.get_latest_tick_result()
        
        assert result["tick"] == 2
        assert result["data"] == "test2"
    
    def test_scenario_get_latest_tick_result_empty(self, mock_scenario):
        """Test getting the latest tick result when no results exist."""
        result = mock_scenario.get_latest_tick_result()
        
        assert result is None
    
    def test_scenario_calculate_scenario_metrics_no_results(self, mock_scenario):
        """Test calculating scenario metrics with no results."""
        metrics = mock_scenario.calculate_scenario_metrics()
        
        assert "error" in metrics
        assert metrics["error"] == "No results available"
    
    def test_scenario_calculate_scenario_metrics_with_results(self, mock_scenario):
        """Test calculating scenario metrics with results."""
        mock_scenario.results = [
            {"tick": 1, "completed": True, "score": 0.8},
            {"tick": 2, "completed": False, "score": 0.6},
            {"tick": 3, "completed": True, "score": 0.9}
        ]
        
        metrics = mock_scenario.calculate_scenario_metrics()
        
        assert metrics["total_ticks"] == 3
        assert metrics["completed_ticks"] == 2
        assert metrics["success_rate"] == 2/3
        assert metrics["average_score"] == (0.8 + 0.6 + 0.9) / 3
        assert metrics["min_score"] == 0.6
        assert metrics["max_score"] == 0.9


class TestScenarioTemplate:
    """Test cases for ScenarioTemplate class."""
    
    @pytest.fixture
    def scenario_config(self):
        """Create a test scenario configuration."""
        return ScenarioConfig(
            name="test_template",
            description="Test template description",
            domain="test_domain",
            duration_ticks=100
        )
    
    @pytest.fixture
    def template_scenario(self, scenario_config):
        """Create a template scenario instance."""
        return ScenarioTemplate(scenario_config)
    
    def test_template_initialization(self, scenario_config):
        """Test template scenario initialization."""
        template = ScenarioTemplate(scenario_config)
        
        assert template.agent_states == {}
        assert template.global_state == {}
    
    @pytest.mark.asyncio
    async def test_template_initialize(self, template_scenario):
        """Test template scenario initialization."""
        parameters = {"param1": "value1"}
        
        await template_scenario.initialize(parameters)
        
        assert template_scenario.parameters == parameters
        assert "start_time" in template_scenario.global_state
        assert template_scenario.global_state["total_agents"] == 0
        assert template_scenario.global_state["active_agents"] == 0
        assert template_scenario.global_state["completed_agents"] == 0
    
    @pytest.mark.asyncio
    async def test_template_setup_for_agent(self, template_scenario):
        """Test setting up template for an agent."""
        agent_id = "test_agent"
        
        await template_scenario.setup_for_agent(agent_id)
        
        assert agent_id in template_scenario.agent_states
        assert template_scenario.agent_states[agent_id]["start_time"] is not None
        assert template_scenario.agent_states[agent_id]["current_tick"] == 0
        assert template_scenario.agent_states[agent_id]["score"] == 0.0
        assert template_scenario.agent_states[agent_id]["completed"] is False
        assert template_scenario.agent_states[agent_id]["errors"] == []
        
        assert template_scenario.global_state["total_agents"] == 1
        assert template_scenario.global_state["active_agents"] == 1
        assert template_scenario.is_setup is True
    
    @pytest.mark.asyncio
    async def test_template_update_tick(self, template_scenario):
        """Test updating template tick."""
        # Setup agent first
        await template_scenario.setup_for_agent("test_agent")
        
        state = Mock(spec=SimulationState)
        tick = 5
        
        await template_scenario.update_tick(tick, state)
        
        assert template_scenario.current_tick == tick
        assert template_scenario.agent_states["test_agent"]["current_tick"] == tick
        assert len(template_scenario.results) == 1
        assert template_scenario.results[0]["tick"] == tick
    
    @pytest.mark.asyncio
    async def test_template_get_scenario_state(self, template_scenario):
        """Test getting template scenario state."""
        # Setup agent first
        await template_scenario.setup_for_agent("test_agent")
        
        state = await template_scenario.get_scenario_state()
        
        assert state["name"] == "test_template"
        assert state["current_tick"] == 0
        assert "global_state" in state
        assert "agent_states" in state
        assert "parameters" in state
        assert "metadata" in state
    
    @pytest.mark.asyncio
    async def test_template_evaluate_agent_performance_existing_agent(self, template_scenario):
        """Test evaluating agent performance for existing agent."""
        # Setup agent first
        await template_scenario.setup_for_agent("test_agent")
        
        metrics = await template_scenario.evaluate_agent_performance("test_agent")
        
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "test_template"
        assert metrics["score"] == 0.0
        assert metrics["completed"] is False
        assert metrics["completion_rate"] == 0.0
        assert metrics["tick_progress"] == 0.0
        assert "start_time" in metrics
        assert "evaluation_time" in metrics
    
    @pytest.mark.asyncio
    async def test_template_evaluate_agent_performance_nonexistent_agent(self, template_scenario):
        """Test evaluating agent performance for non-existent agent."""
        metrics = await template_scenario.evaluate_agent_performance("nonexistent_agent")
        
        assert "error" in metrics
        assert metrics["error"] == "Agent nonexistent_agent not found in scenario"
    
    def test_template_update_agent_score(self, template_scenario):
        """Test updating agent score."""
        # Setup agent first
        template_scenario.agent_states["test_agent"] = {"score": 0.5}
        
        template_scenario.update_agent_score("test_agent", 0.3)
        
        assert template_scenario.agent_states["test_agent"]["score"] == 0.8
    
    def test_template_mark_agent_completed(self, template_scenario):
        """Test marking agent as completed."""
        # Setup agent first
        template_scenario.agent_states["test_agent"] = {"completed": False}
        template_scenario.global_state["active_agents"] = 1
        template_scenario.global_state["completed_agents"] = 0
        
        template_scenario.mark_agent_completed("test_agent")
        
        assert template_scenario.agent_states["test_agent"]["completed"] is True
        assert template_scenario.global_state["active_agents"] == 0
        assert template_scenario.global_state["completed_agents"] == 1
    
    def test_template_add_agent_error(self, template_scenario):
        """Test adding error to agent."""
        # Setup agent first
        template_scenario.agent_states["test_agent"] = {"errors": []}
        
        template_scenario.add_agent_error("test_agent", "Test error")
        
        assert template_scenario.agent_states["test_agent"]["errors"] == ["Test error"]


class TestScenarioRegistration:
    """Test cases for ScenarioRegistration class."""
    
    def test_scenario_registration_creation(self):
        """Test creating a scenario registration."""
        config = ScenarioConfig(
            name="test",
            description="test",
            domain="test",
            duration_ticks=100
        )
        
        registration = ScenarioRegistration(
            name="test_scenario",
            description="Test scenario",
            domain="test_domain",
            scenario_class=MockScenario,
            default_config=config,
            tags=["tag1", "tag2"],
            enabled=True
        )
        
        assert registration.name == "test_scenario"
        assert registration.description == "Test scenario"
        assert registration.domain == "test_domain"
        assert registration.scenario_class == MockScenario
        assert registration.default_config == config
        assert registration.tags == ["tag1", "tag2"]
        assert registration.enabled is True
    
    def test_scenario_registration_default_tags(self):
        """Test scenario registration with default tags."""
        config = ScenarioConfig(
            name="test",
            description="test",
            domain="test",
            duration_ticks=100
        )
        
        registration = ScenarioRegistration(
            name="test_scenario",
            description="Test scenario",
            domain="test_domain",
            scenario_class=MockScenario,
            default_config=config
        )
        
        assert registration.tags == []


class TestScenarioRegistry:
    """Test cases for ScenarioRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance."""
        return ScenarioRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert len(registry._scenarios) > 0  # Should have built-in scenarios
        assert len(registry._domains) > 0
        assert len(registry._tags) > 0
    
    def test_register_scenario(self, registry):
        """Test registering a new scenario."""
        config = ScenarioConfig(
            name="new_scenario",
            description="New scenario",
            domain="new_domain",
            duration_ticks=50
        )
        
        registry.register_scenario(
            name="new_scenario",
            description="New scenario",
            domain="new_domain",
            scenario_class=MockScenario,
            default_config=config,
            tags=["new_tag"]
        )
        
        assert "new_scenario" in registry._scenarios
        assert "new_domain" in registry._domains
        assert "new_tag" in registry._tags
        assert "new_scenario" in registry._domains["new_domain"]
        assert "new_scenario" in registry._tags["new_tag"]
    
    def test_register_scenario_overwrite(self, registry):
        """Test overwriting an existing scenario."""
        config = ScenarioConfig(
            name="ecommerce",  # Existing scenario
            description="Updated scenario",
            domain="updated_domain",
            duration_ticks=50
        )
        
        with patch('benchmarking.scenarios.registry.logger') as mock_logger:
            registry.register_scenario(
                name="ecommerce",
                description="Updated scenario",
                domain="updated_domain",
                scenario_class=MockScenario,
                default_config=config
            )
            
            mock_logger.warning.assert_called_once()
    
    def test_get_scenario_existing(self, registry):
        """Test getting an existing scenario."""
        registration = registry.get_scenario("ecommerce")
        
        assert registration is not None
        assert registration.name == "ecommerce"
        assert registration.domain == "ecommerce"
    
    def test_get_scenario_nonexistent(self, registry):
        """Test getting a non-existent scenario."""
        registration = registry.get_scenario("nonexistent")
        
        assert registration is None
    
    def test_list_scenarios_all(self, registry):
        """Test listing all scenarios."""
        scenarios = registry.list_scenarios()
        
        assert len(scenarios) > 0
        assert "ecommerce" in scenarios
        assert "healthcare" in scenarios
    
    def test_list_scenarios_by_domain(self, registry):
        """Test listing scenarios by domain."""
        scenarios = registry.list_scenarios(domain="ecommerce")
        
        assert len(scenarios) > 0
        assert "ecommerce" in scenarios
        assert "healthcare" not in scenarios
    
    def test_list_scenarios_nonexistent_domain(self, registry):
        """Test listing scenarios for non-existent domain."""
        scenarios = registry.list_scenarios(domain="nonexistent")
        
        assert scenarios == []
    
    def test_list_scenarios_disabled_only(self, registry):
        """Test listing only enabled scenarios."""
        # Disable a scenario
        registry.disable_scenario("ecommerce")
        
        scenarios = registry.list_scenarios(enabled_only=True)
        
        assert "ecommerce" not in scenarios
        
        scenarios = registry.list_scenarios(enabled_only=False)
        
        assert "ecommerce" in scenarios
    
    def test_get_scenarios_by_domain(self, registry):
        """Test getting scenarios grouped by domain."""
        domains = registry.get_scenarios_by_domain()
        
        assert "ecommerce" in domains
        assert "healthcare" in domains
        assert isinstance(domains["ecommerce"], list)
        assert len(domains["ecommerce"]) > 0
    
    def test_get_scenarios_by_tag(self, registry):
        """Test getting scenarios by tag."""
        scenarios = registry.get_scenarios_by_tag("ecommerce")
        
        assert len(scenarios) > 0
        assert "ecommerce" in scenarios
    
    def test_get_scenarios_by_nonexistent_tag(self, registry):
        """Test getting scenarios by non-existent tag."""
        scenarios = registry.get_scenarios_by_tag("nonexistent_tag")
        
        assert scenarios == []
    
    def test_create_scenario_existing(self, registry):
        """Test creating an existing scenario."""
        scenario = registry.create_scenario("ecommerce")
        
        assert scenario is not None
        assert isinstance(scenario, ECommerceScenario)
    
    def test_create_scenario_nonexistent(self, registry):
        """Test creating a non-existent scenario."""
        with patch('benchmarking.scenarios.registry.logger') as mock_logger:
            scenario = registry.create_scenario("nonexistent")
            
            assert scenario is None
            mock_logger.error.assert_called_once()
    
    def test_create_scenario_disabled(self, registry):
        """Test creating a disabled scenario."""
        # Disable a scenario
        registry.disable_scenario("ecommerce")
        
        with patch('benchmarking.scenarios.registry.logger') as mock_logger:
            scenario = registry.create_scenario("ecommerce")
            
            assert scenario is None
            mock_logger.warning.assert_called_once()
    
    def test_create_scenario_with_config(self, registry):
        """Test creating a scenario with custom config."""
        config = ScenarioConfig(
            name="custom_ecommerce",
            description="Custom e-commerce scenario",
            domain="ecommerce",
            duration_ticks=200,
            parameters={"product_count": 50}
        )
        
        scenario = registry.create_scenario("ecommerce", config)
        
        assert scenario is not None
        assert scenario.config == config
    
    def test_enable_scenario_existing(self, registry):
        """Test enabling an existing scenario."""
        # Disable first
        registry.disable_scenario("ecommerce")
        
        result = registry.enable_scenario("ecommerce")
        
        assert result is True
        assert registry._scenarios["ecommerce"].enabled is True
    
    def test_enable_scenario_nonexistent(self, registry):
        """Test enabling a non-existent scenario."""
        result = registry.enable_scenario("nonexistent")
        
        assert result is False
    
    def test_disable_scenario_existing(self, registry):
        """Test disabling an existing scenario."""
        result = registry.disable_scenario("ecommerce")
        
        assert result is True
        assert registry._scenarios["ecommerce"].enabled is False
    
    def test_disable_scenario_nonexistent(self, registry):
        """Test disabling a non-existent scenario."""
        result = registry.disable_scenario("nonexistent")
        
        assert result is False
    
    def test_get_scenario_info_existing(self, registry):
        """Test getting info for an existing scenario."""
        info = registry.get_scenario_info("ecommerce")
        
        assert info["name"] == "ecommerce"
        assert info["domain"] == "ecommerce"
        assert "class" in info
        assert "module" in info
        assert "default_config" in info
        assert "tags" in info
        assert "enabled" in info
    
    def test_get_scenario_info_nonexistent(self, registry):
        """Test getting info for a non-existent scenario."""
        info = registry.get_scenario_info("nonexistent")
        
        assert "error" in info
        assert info["error"] == "Scenario 'nonexistent' not found"
    
    def test_create_scenario_suite(self, registry):
        """Test creating a scenario suite."""
        scenario_names = ["ecommerce", "healthcare"]
        configs = {
            "ecommerce": ScenarioConfig(
                name="custom_ecommerce",
                description="Custom e-commerce",
                domain="ecommerce",
                duration_ticks=150
            )
        }
        
        suite = registry.create_scenario_suite(scenario_names, configs)
        
        assert len(suite) == 2
        assert "ecommerce" in suite
        assert "healthcare" in suite
        assert suite["ecommerce"].config.name == "custom_ecommerce"
    
    def test_validate_scenario_config_valid(self, registry):
        """Test validating a valid scenario config."""
        config = ScenarioConfig(
            name="test",
            description="Test scenario",
            domain="ecommerce",
            duration_ticks=100,
            difficulty="medium"
        )
        
        errors = registry.validate_scenario_config("ecommerce", config)
        
        assert errors == []
    
    def test_validate_scenario_config_unknown_scenario(self, registry):
        """Test validating config for unknown scenario."""
        config = ScenarioConfig(
            name="test",
            description="Test scenario",
            domain="test",
            duration_ticks=100
        )
        
        errors = registry.validate_scenario_config("unknown", config)
        
        assert "Unknown scenario: unknown" in errors
    
    @pytest.mark.parametrize("missing_field,expected_error", [
        ("name", "Scenario name cannot be empty"),
        ("description", "Scenario description cannot be empty"),
        ("domain", "Scenario domain cannot be empty")
    ])
    def test_validate_scenario_config_missing_fields(self, registry, missing_field, expected_error):
        """Test validating config with missing fields."""
        config = ScenarioConfig(
            name="test",
            description="test",
            domain="test",
            duration_ticks=100
        )
        setattr(config, missing_field, "")
        
        errors = registry.validate_scenario_config("ecommerce", config)
        
        assert expected_error in errors
    
    def test_validate_scenario_config_invalid_duration(self, registry):
        """Test validating config with invalid duration."""
        config = ScenarioConfig(
            name="test",
            description="test",
            domain="test",
            duration_ticks=0
        )
        
        errors = registry.validate_scenario_config("ecommerce", config)
        
        assert "Duration ticks must be positive" in errors
    
    def test_validate_scenario_config_invalid_difficulty(self, registry):
        """Test validating config with invalid difficulty."""
        config = ScenarioConfig(
            name="test",
            description="test",
            domain="test",
            duration_ticks=100,
            difficulty="invalid"
        )
        
        errors = registry.validate_scenario_config("ecommerce", config)
        
        assert "Difficulty must be one of" in errors
    
    def test_get_registry_summary(self, registry):
        """Test getting registry summary."""
        summary = registry.get_registry_summary()
        
        assert "total_scenarios" in summary
        assert "enabled_scenarios" in summary
        assert "disabled_scenarios" in summary
        assert "domains" in summary
        assert "tags" in summary
        
        assert summary["total_scenarios"] > 0
        assert summary["enabled_scenarios"] > 0
        assert summary["disabled_scenarios"] >= 0
        assert len(summary["domains"]) > 0
        assert len(summary["tags"]) > 0


class TestECommerceScenario:
    """Test cases for ECommerceScenario class."""
    
    @pytest.fixture
    def scenario_config(self):
        """Create an e-commerce scenario configuration."""
        return ScenarioConfig(
            name="ecommerce_test",
            description="E-commerce test scenario",
            domain="ecommerce",
            duration_ticks=100,
            parameters={
                "product_count": 10,
                "customer_count": 20,
                "initial_budget": 50000
            }
        )
    
    @pytest.fixture
    def ecommerce_scenario(self, scenario_config):
        """Create an e-commerce scenario instance."""
        return ECommerceScenario(scenario_config)
    
    def test_ecommerce_initialization(self, scenario_config):
        """Test e-commerce scenario initialization."""
        scenario = ECommerceScenario(scenario_config)
        
        assert scenario.products == []
        assert scenario.customers == []
        assert scenario.orders == []
        assert scenario.competitors == []
        assert scenario.market_demand == 1.0
        assert scenario.seasonal_factor == 1.0
        assert scenario.competition_level == 0.5
    
    def test_validate_domain_parameters_valid(self, ecommerce_scenario):
        """Test validation with valid parameters."""
        errors = ecommerce_scenario._validate_domain_parameters()
        
        assert errors == []
    
    @pytest.mark.parametrize("param,value,expected_error", [
        ("product_count", 0, "product_count must be a positive integer"),
        ("product_count", "invalid", "product_count must be a positive integer"),
        ("customer_count", -1, "customer_count must be a positive integer"),
        ("customer_count", "invalid", "customer_count must be a positive integer"),
        ("initial_budget", 0, "initial_budget must be a positive number"),
        ("initial_budget", "invalid", "initial_budget must be a positive number")
    ])
    def test_validate_domain_parameters_invalid(self, ecommerce_scenario, param, value, expected_error):
        """Test validation with invalid parameters."""
        ecommerce_scenario.parameters[param] = value
        
        errors = ecommerce_scenario._validate_domain_parameters()
        
        assert expected_error in errors
    
    @pytest.mark.asyncio
    async def test_ecommerce_initialize(self, ecommerce_scenario):
        """Test e-commerce scenario initialization."""
        await ecommerce_scenario.initialize(ecommerce_scenario.parameters)
        
        assert len(ecommerce_scenario.products) == 10
        assert len(ecommerce_scenario.customers) == 20
        assert len(ecommerce_scenario.competitors) == 3
        assert 0.5 <= ecommerce_scenario.market_demand <= 1.5
        assert 0.8 <= ecommerce_scenario.seasonal_factor <= 1.2
        assert 0.3 <= ecommerce_scenario.competition_level <= 0.7
        
        assert ecommerce_scenario.global_state["products"] == 10
        assert ecommerce_scenario.global_state["customers"] == 20
        assert ecommerce_scenario.global_state["competitors"] == 3
        assert ecommerce_scenario.global_state["initial_budget"] == 50000
    
    def test_generate_products(self, ecommerce_scenario):
        """Test product generation."""
        products = ecommerce_scenario._generate_products(5)
        
        assert len(products) == 5
        
        for product in products:
            assert "id" in product
            assert "name" in product
            assert "category" in product
            assert "base_price" in product
            assert "current_price" in product
            assert "inventory" in product
            assert "cost" in product
            assert "popularity" in product
    
    def test_generate_customers(self, ecommerce_scenario):
        """Test customer generation."""
        customers = ecommerce_scenario._generate_customers(5)
        
        assert len(customers) == 5
        
        for customer in customers:
            assert "id" in customer
            assert "budget" in customer
            assert "preferences" in customer
            assert "purchase_history" in customer
            
            assert "price_sensitivity" in customer["preferences"]
            assert "quality_preference" in customer["preferences"]
            assert "brand_loyalty" in customer["preferences"]
    
    def test_generate_competitors(self, ecommerce_scenario):
        """Test competitor generation."""
        competitors = ecommerce_scenario._generate_competitors(3)
        
        assert len(competitors) == 3
        
        for competitor in competitors:
            assert "id" in competitor
            assert "name" in competitor
            assert "market_share" in competitor
            assert "pricing_strategy" in competitor
            assert "reputation" in competitor
    
    @pytest.mark.asyncio
    async def test_ecommerce_update_tick(self, ecommerce_scenario):
        """Test e-commerce scenario tick update."""
        # Initialize first
        await ecommerce_scenario.initialize(ecommerce_scenario.parameters)
        
        # Setup agent
        await ecommerce_scenario.setup_for_agent("test_agent")
        
        state = Mock(spec=SimulationState)
        
        # Test regular tick
        await ecommerce_scenario.update_tick(5, state)
        assert ecommerce_scenario.current_tick == 5
        
        # Test market condition update (every 10 ticks)
        initial_demand = ecommerce_scenario.market_demand
        await ecommerce_scenario.update_tick(10, state)
        # Market conditions should change
        assert ecommerce_scenario.current_tick == 10
        
        # Test competitor actions (every 5 ticks)
        initial_price = ecommerce_scenario.products[0]["current_price"]
        await ecommerce_scenario.update_tick(15, state)
        # Prices might change due to competitor actions
        assert ecommerce_scenario.current_tick == 15
    
    def test_update_market_conditions(self, ecommerce_scenario):
        """Test market condition updates."""
        initial_demand = ecommerce_scenario.market_demand
        initial_seasonal = ecommerce_scenario.seasonal_factor
        
        ecommerce_scenario._update_market_conditions()
        
        # Values should change but stay within bounds
        assert 0.5 <= ecommerce_scenario.market_demand <= 1.5
        assert 0.8 <= ecommerce_scenario.seasonal_factor <= 1.2
    
    def test_simulate_customer_behavior(self, ecommerce_scenario):
        """Test customer behavior simulation."""
        # Initialize with some products
        ecommerce_scenario.products = [
            {"id": "p1", "current_price": 50, "inventory": 100},
            {"id": "p2", "current_price": 100, "inventory": 50}
        ]
        ecommerce_scenario.customers = [
            {"id": "c1", "budget": 500, "purchase_history": []},
            {"id": "c2", "budget": 1000, "purchase_history": []}
        ]
        ecommerce_scenario.market_demand = 1.0
        ecommerce_scenario.seasonal_factor = 1.0
        
        initial_orders = len(ecommerce_scenario.orders)
        
        ecommerce_scenario._simulate_customer_behavior(1)
        
        # Orders might increase
        assert len(ecommerce_scenario.orders) >= initial_orders
    
    def test_simulate_competitor_actions(self, ecommerce_scenario):
        """Test competitor actions simulation."""
        # Initialize with products and competitors
        ecommerce_scenario.products = [
            {"id": "p1", "current_price": 50},
            {"id": "p2", "current_price": 100}
        ]
        ecommerce_scenario.competitors = [
            {"id": "comp1", "pricing_strategy": "aggressive"},
            {"id": "comp2", "pricing_strategy": "premium"}
        ]
        
        initial_prices = [p["current_price"] for p in ecommerce_scenario.products]
        
        ecommerce_scenario._simulate_competitor_actions(1)
        
        # Prices might change
        final_prices = [p["current_price"] for p in ecommerce_scenario.products]
        assert len(final_prices) == len(initial_prices)
    
    @pytest.mark.asyncio
    async def test_ecommerce_evaluate_agent_performance(self, ecommerce_scenario):
        """Test e-commerce agent performance evaluation."""
        # Setup agent
        await ecommerce_scenario.setup_for_agent("test_agent")
        
        # Add some orders
        ecommerce_scenario.orders = [
            {"agent_id": "test_agent", "price": 100, "quantity": 2},
            {"agent_id": "test_agent", "price": 50, "quantity": 1}
        ]
        
        metrics = await ecommerce_scenario.evaluate_agent_performance("test_agent")
        
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "ecommerce_test"
        assert "total_revenue" in metrics
        assert "total_orders" in metrics
        assert "average_order_value" in metrics
        assert "profit" in metrics
        assert "market_share" in metrics
        assert "inventory_turnover" in metrics
        assert "customer_satisfaction" in metrics
    
    def test_calculate_inventory_turnover(self, ecommerce_scenario):
        """Test inventory turnover calculation."""
        # Setup products and orders
        ecommerce_scenario.products = [
            {"id": "p1", "inventory": 50},
            {"id": "p2", "inventory": 30}
        ]
        ecommerce_scenario.orders = [
            {"product_id": "p1", "quantity": 10},
            {"product_id": "p1", "quantity": 5},
            {"product_id": "p2", "quantity": 15}
        ]
        
        turnover = ecommerce_scenario._calculate_inventory_turnover()
        
        expected_turnover = (10 + 5 + 15) / (50 + 30)
        assert turnover == expected_turnover
    
    def test_calculate_inventory_turnover_no_inventory(self, ecommerce_scenario):
        """Test inventory turnover calculation with no inventory."""
        ecommerce_scenario.products = [
            {"id": "p1", "inventory": 0},
            {"id": "p2", "inventory": 0}
        ]
        
        turnover = ecommerce_scenario._calculate_inventory_turnover()
        
        assert turnover == 0.0


class TestHealthcareScenario:
    """Test cases for HealthcareScenario class."""
    
    @pytest.fixture
    def scenario_config(self):
        """Create a healthcare scenario configuration."""
        return ScenarioConfig(
            name="healthcare_test",
            description="Healthcare test scenario",
            domain="healthcare",
            duration_ticks=150,
            parameters={
                "patient_count": 10,
                "medical_staff_count": 5
            }
        )
    
    @pytest.fixture
    def healthcare_scenario(self, scenario_config):
        """Create a healthcare scenario instance."""
        return HealthcareScenario(scenario_config)
    
    def test_healthcare_initialization(self, scenario_config):
        """Test healthcare scenario initialization."""
        scenario = HealthcareScenario(scenario_config)
        
        assert scenario.patients == []
        assert scenario.medical_conditions == []
        assert scenario.treatments == []
        assert scenario.medical_staff == []
        assert scenario.diagnostic_accuracy == 0.0
        assert scenario.treatment_effectiveness == 0.0
        assert scenario.patient_satisfaction == 0.0
    
    def test_validate_domain_parameters_valid(self, healthcare_scenario):
        """Test validation with valid parameters."""
        errors = healthcare_scenario._validate_domain_parameters()
        
        assert errors == []
    
    @pytest.mark.parametrize("param,value,expected_error", [
        ("patient_count", 0, "patient_count must be a positive integer"),
        ("patient_count", "invalid", "patient_count must be a positive integer"),
        ("medical_staff_count", -1, "medical_staff_count must be a positive integer"),
        ("medical_staff_count", "invalid", "medical_staff_count must be a positive integer")
    ])
    def test_validate_domain_parameters_invalid(self, healthcare_scenario, param, value, expected_error):
        """Test validation with invalid parameters."""
        healthcare_scenario.parameters[param] = value
        
        errors = healthcare_scenario._validate_domain_parameters()
        
        assert expected_error in errors
    
    @pytest.mark.asyncio
    async def test_healthcare_initialize(self, healthcare_scenario):
        """Test healthcare scenario initialization."""
        await healthcare_scenario.initialize(healthcare_scenario.parameters)
        
        assert len(healthcare_scenario.medical_conditions) > 0
        assert len(healthcare_scenario.treatments) > 0
        assert len(healthcare_scenario.patients) == 10
        assert len(healthcare_scenario.medical_staff) == 5
        
        assert healthcare_scenario.global_state["patients"] == 10
        assert healthcare_scenario.global_state["medical_conditions"] == len(healthcare_scenario.medical_conditions)
        assert healthcare_scenario.global_state["treatments"] == len(healthcare_scenario.treatments)
        assert healthcare_scenario.global_state["medical_staff"] == 5
    
    def test_generate_medical_conditions(self, healthcare_scenario):
        """Test medical condition generation."""
        conditions = healthcare_scenario._generate_medical_conditions()
        
        assert len(conditions) > 0
        
        for condition in conditions:
            assert "id" in condition
            assert "name" in condition
            assert "symptoms" in condition
            assert "severity" in condition
            assert "prevalence" in condition
            assert isinstance(condition["symptoms"], list)
            assert 0.0 <= condition["severity"] <= 1.0
            assert 0.0 <= condition["prevalence"] <= 1.0
    
    def test_generate_treatments(self, healthcare_scenario):
        """Test treatment generation."""
        treatments = healthcare_scenario._generate_treatments()
        
        assert len(treatments) > 0
        
        for treatment in treatments:
            assert "id" in treatment
            assert "name" in treatment
            assert "effectiveness" in treatment
            assert "cost" in treatment
            assert "duration" in treatment
            assert 0.0 <= treatment["effectiveness"] <= 1.0
            assert treatment["cost"] > 0
            assert treatment["duration"] > 0
    
    def test_generate_patients(self, healthcare_scenario):
        """Test patient generation."""
        # Setup medical conditions first
        healthcare_scenario.medical_conditions = [
            {"id": "flu", "symptoms": ["fever", "cough"]},
            {"id": "diabetes", "symptoms": ["thirst", "fatigue"]}
        ]
        
        patients = healthcare_scenario._generate_patients(5)
        
        assert len(patients) == 5
        
        for patient in patients:
            assert "id" in patient
            assert "age" in patient
            assert "gender" in patient
            assert "condition" in patient
            assert "symptoms" in patient
            assert "severity" in patient
            assert "medical_history" in patient
            assert "treatment_plan" in patient
            assert "status" in patient
            
            assert 18 <= patient["age"] <= 80
            assert patient["gender"] in ["male", "female"]
            assert patient["status"] == "waiting"
            assert isinstance(patient["symptoms"], list)
            assert 0.0 <= patient["severity"] <= 1.0
    
    def test_generate_medical_staff(self, healthcare_scenario):
        """Test medical staff generation."""
        staff = healthcare_scenario._generate_medical_staff(5)
        
        assert len(staff) == 5
        
        for member in staff:
            assert "id" in member
            assert "name" in member
            assert "role" in member
            assert "experience" in member
            assert "specialization" in member
            assert "workload" in member
            assert "patients_assigned" in member
            
            assert member["experience"] > 0
            assert member["workload"] == 0
            assert isinstance(member["patients_assigned"], list)
    
    @pytest.mark.asyncio
    async def test_healthcare_update_tick(self, healthcare_scenario):
        """Test healthcare scenario tick update."""
        # Initialize first
        await healthcare_scenario.initialize(healthcare_scenario.parameters)
        
        # Setup agent
        await healthcare_scenario.setup_for_agent("test_agent")
        
        state = Mock(spec=SimulationState)
        
        # Test regular tick
        await healthcare_scenario.update_tick(5, state)
        assert healthcare_scenario.current_tick == 5
        
        # Test patient arrival (every 5 ticks)
        initial_patients = len(healthcare_scenario.patients)
        await healthcare_scenario.update_tick(10, state)
        # New patients might arrive
        assert healthcare_scenario.current_tick == 10
    
    def test_simulate_patient_arrival(self, healthcare_scenario):
        """Test patient arrival simulation."""
        # Setup medical conditions
        healthcare_scenario.medical_conditions = [
            {"id": "flu", "symptoms": ["fever", "cough"]}
        ]
        
        initial_patients = len(healthcare_scenario.patients)
        
        # Force patient arrival
        healthcare_scenario.current_tick = 5
        with patch('random.random', return_value=0.2):  # 20% < 30%
            healthcare_scenario._simulate_patient_arrival()
        
        assert len(healthcare_scenario.patients) == initial_patients + 1
    
    def test_simulate_treatment_progress(self, healthcare_scenario):
        """Test treatment progress simulation."""
        # Setup patients with treatment plans
        healthcare_scenario.patients = [
            {
                "id": "p1",
                "status": "treating",
                "treatment_plan": {
                    "progress": 0.5,
                    "effectiveness": 0.8
                }
            },
            {
                "id": "p2",
                "status": "treating",
                "treatment_plan": {
                    "progress": 0.9,
                    "effectiveness": 0.7
                }
            }
        ]
        
        healthcare_scenario._simulate_treatment_progress(10)
        
        # Progress should increase
        assert healthcare_scenario.patients[0]["treatment_plan"]["progress"] > 0.5
        assert healthcare_scenario.patients[1]["treatment_plan"]["progress"] >= 0.9
    
    def test_update_staff_workload(self, healthcare_scenario):
        """Test staff workload update."""
        # Setup medical staff and patients
        healthcare_scenario.medical_staff = [
            {"id": "s1", "workload": 0, "patients_assigned": []},
            {"id": "s2", "workload": 0, "patients_assigned": []}
        ]
        healthcare_scenario.patients = [
            {"id": "p1", "status": "diagnosed"},
            {"id": "p2", "status": "treating"},
            {"id": "p3", "status": "waiting"}
        ]
        
        healthcare_scenario._update_staff_workload()
        
        # Workload should be updated
        total_workload = sum(staff["workload"] for staff in healthcare_scenario.medical_staff)
        assert total_workload == 2  # Only diagnosed and treating patients
    
    @pytest.mark.asyncio
    async def test_healthcare_evaluate_agent_performance(self, healthcare_scenario):
        """Test healthcare agent performance evaluation."""
        # Setup agent
        await healthcare_scenario.setup_for_agent("test_agent")
        
        # Setup patients
        healthcare_scenario.patients = [
            {"id": "p1", "status": "recovered", "diagnosed_by": "test_agent", "diagnosis_correct": True},
            {"id": "p2", "status": "treating", "diagnosed_by": "test_agent", "diagnosis_correct": False},
            {"id": "p3", "status": "recovered", "treated_by": "test_agent", "treatment_success": True},
            {"id": "p4", "status": "waiting"}
        ]
        
        metrics = await healthcare_scenario.evaluate_agent_performance("test_agent")
        
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "healthcare_test"
        assert metrics["total_patients"] == 4
        assert metrics["diagnosed_patients"] == 2
        assert metrics["treated_patients"] == 1
        assert metrics["recovered_patients"] == 1
        assert "diagnostic_accuracy" in metrics
        assert "treatment_effectiveness" in metrics
        assert "patient_satisfaction" in metrics
        assert "average_wait_time" in metrics
    
    def test_calculate_average_wait_time(self, healthcare_scenario):
        """Test average wait time calculation."""
        # Setup patients
        healthcare_scenario.patients = [
            {"id": "p1", "status": "waiting", "arrival_time": 5},
            {"id": "p2", "status": "waiting", "arrival_time": 8},
            {"id": "p3", "status": "diagnosed", "arrival_time": 2}
        ]
        healthcare_scenario.current_tick = 10
        
        wait_time = healthcare_scenario._calculate_average_wait_time()
        
        # Only waiting patients should be considered
        expected_wait_time = ((10 - 5) + (10 - 8)) / 2
        assert wait_time == expected_wait_time
    
    def test_calculate_average_wait_time_no_waiting_patients(self, healthcare_scenario):
        """Test average wait time calculation with no waiting patients."""
        healthcare_scenario.patients = [
            {"id": "p1", "status": "diagnosed", "arrival_time": 2},
            {"id": "p2", "status": "treating", "arrival_time": 5}
        ]
        
        wait_time = healthcare_scenario._calculate_average_wait_time()
        
        assert wait_time == 0.0


class TestFinancialScenario:
    """Test cases for FinancialScenario class."""
    
    @pytest.fixture
    def scenario_config(self):
        """Create a financial scenario configuration."""
        return ScenarioConfig(
            name="financial_test",
            description="Financial test scenario",
            domain="financial",
            duration_ticks=200,
            parameters={
                "initial_capital": 100000,
                "instrument_count": 10
            }
        )
    
    @pytest.fixture
    def financial_scenario(self, scenario_config):
        """Create a financial scenario instance."""
        return FinancialScenario(scenario_config)
    
    def test_financial_initialization(self, scenario_config):
        """Test financial scenario initialization."""
        scenario = FinancialScenario(scenario_config)
        
        assert scenario.market_data == []
        assert scenario.portfolios == {}
        assert scenario.instruments == []
        assert scenario.market_conditions == {
            "volatility": 0.2,
            "trend": "stable",
            "liquidity": 0.7
        }
    
    def test_validate_domain_parameters_valid(self, financial_scenario):
        """Test validation with valid parameters."""
        errors = financial_scenario._validate_domain_parameters()
        
        assert errors == []
    
    @pytest.mark.parametrize("param,value,expected_error", [
        ("initial_capital", 0, "initial_capital must be a positive number"),
        ("initial_capital", "invalid", "initial_capital must be a positive number"),
        ("instrument_count", 0, "instrument_count must be a positive integer"),
        ("instrument_count", "invalid", "instrument_count must be a positive integer")
    ])
    def test_validate_domain_parameters_invalid(self, financial_scenario, param, value, expected_error):
        """Test validation with invalid parameters."""
        financial_scenario.parameters[param] = value
        
        errors = financial_scenario._validate_domain_parameters()
        
        assert expected_error in errors
    
    @pytest.mark.asyncio
    async def test_financial_initialize(self, financial_scenario):
        """Test financial scenario initialization."""
        await financial_scenario.initialize(financial_scenario.parameters)
        
        assert len(financial_scenario.instruments) == 10
        assert len(financial_scenario.market_data) == 10
        
        assert financial_scenario.global_state["initial_capital"] == 100000
        assert financial_scenario.global_state["instruments"] == 10
        assert financial_scenario.global_state["market_volatility"] == 0.2
        assert financial_scenario.global_state["market_trend"] == "stable"
        assert financial_scenario.global_state["market_liquidity"] == 0.7
    
    def test_generate_instruments(self, financial_scenario):
        """Test financial instrument generation."""
        instruments = financial_scenario._generate_instruments(5)
        
        assert len(instruments) == 5
        
        for instrument in instruments:
            assert "id" in instrument
            assert "name" in instrument
            assert "type" in instrument
            assert "current_price" in instrument
            assert "volatility" in instrument
            assert "trend" in instrument
            assert "liquidity" in instrument
            assert "market_cap" in instrument
            
            assert instrument["current_price"] > 0
            assert 0.0 <= instrument["volatility"] <= 1.0
            assert instrument["trend"] in ["bullish", "bearish", "stable"]
            assert 0.0 <= instrument["liquidity"] <= 1.0
            assert instrument["market_cap"] > 0
    
    def test_initialize_market_data(self, financial_scenario):
        """Test market data initialization."""
        # Setup instruments first
        financial_scenario.instruments = [
            {"id": "i1", "current_price": 100, "trend": "bullish"},
            {"id": "i2", "current_price": 200, "trend": "bearish"}
        ]
        
        market_data = financial_scenario._initialize_market_data()
        
        assert len(market_data) == 2
        
        for data in market_data:
            assert "instrument_id" in data
            assert "prices" in data
            assert "volume" in data
            assert len(data["prices"]) == 30  # 30 days of history
            assert len(data["volume"]) == 30
    
    @pytest.mark.asyncio
    async def test_financial_update_tick(self, financial_scenario):
        """Test financial scenario tick update."""
        # Initialize first
        await financial_scenario.initialize(financial_scenario.parameters)
        
        # Setup agent
        await financial_scenario.setup_for_agent("test_agent")
        
        state = Mock(spec=SimulationState)
        
        # Test regular tick
        initial_prices = [i["current_price"] for i in financial_scenario.instruments]
        await financial_scenario.update_tick(5, state)
        final_prices = [i["current_price"] for i in financial_scenario.instruments]
        
        assert financial_scenario.current_tick == 5
        # Prices should change
        assert len(initial_prices) == len(final_prices)
    
    def test_update_market_conditions(self, financial_scenario):
        """Test market condition updates."""
        initial_volatility = financial_scenario.market_conditions["volatility"]
        initial_trend = financial_scenario.market_conditions["trend"]
        initial_liquidity = financial_scenario.market_conditions["liquidity"]
        
        financial_scenario._update_market_conditions(5)
        
        # Values should change but stay within bounds
        assert 0.1 <= financial_scenario.market_conditions["volatility"] <= 0.5
        assert financial_scenario.market_conditions["trend"] in ["bullish", "bearish", "stable"]
        assert 0.3 <= financial_scenario.market_conditions["liquidity"] <= 1.0
    
    def test_update_instrument_prices(self, financial_scenario):
        """Test instrument price updates."""
        # Setup instruments
        financial_scenario.instruments = [
            {"id": "i1", "current_price": 100, "volatility": 0.2, "trend": "bullish"},
            {"id": "i2", "current_price": 200, "volatility": 0.3, "trend": "bearish"}
        ]
        financial_scenario.market_data = [
            {"instrument_id": "i1", "prices": [100], "volume": [1000]},
            {"instrument_id": "i2", "prices": [200], "volume": [2000]}
        ]
        
        initial_prices = [i["current_price"] for i in financial_scenario.instruments]
        
        financial_scenario._update_instrument_prices(5)
        
        # Prices should change
        final_prices = [i["current_price"] for i in financial_scenario.instruments]
        assert len(initial_prices) == len(final_prices)
        
        # Market data should be updated
        assert len(financial_scenario.market_data[0]["prices"]) == 2
        assert len(financial_scenario.market_data[1]["prices"]) == 2
    
    def test_generate_market_event(self, financial_scenario):
        """Test market event generation."""
        # Setup instruments
        financial_scenario.instruments = [
            {"id": "i1", "current_price": 100, "trend": "bullish"},
            {"id": "i2", "current_price": 200, "trend": "stable"}
        ]
        
        initial_prices = [i["current_price"] for i in financial_scenario.instruments]
        
        financial_scenario._generate_market_event(5)
        
        # Prices might change due to market event
        final_prices = [i["current_price"] for i in financial_scenario.instruments]
        assert len(initial_prices) == len(final_prices)
    
    @pytest.mark.asyncio
    async def test_financial_evaluate_agent_performance(self, financial_scenario):
        """Test financial agent performance evaluation."""
        # Setup agent
        await financial_scenario.setup_for_agent("test_agent")
        
        # Setup portfolio
        financial_scenario.portfolios["test_agent"] = {
            "current_value": 120000,
            "value_history": [100000, 105000, 110000, 115000, 120000],
            "holdings": {"i1": 10, "i2": 5},
            "number_of_trades": 20,
            "win_rate": 0.6
        }
        
        metrics = await financial_scenario.evaluate_agent_performance("test_agent")
        
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "financial_test"
        assert metrics["initial_capital"] == 100000
        assert metrics["current_value"] == 120000
        assert metrics["total_return"] == 0.2  # 20% return
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "diversification" in metrics
        assert metrics["number_of_trades"] == 20
        assert metrics["win_rate"] == 0.6


class TestLegalScenario:
    """Test cases for LegalScenario class."""
    
    @pytest.fixture
    def scenario_config(self):
        """Create a legal scenario configuration."""
        return ScenarioConfig(
            name="legal_test",
            description="Legal test scenario",
            domain="legal",
            duration_ticks=120,
            parameters={
                "document_count": 20,
                "case_complexity": "medium"
            }
        )
    
    @pytest.fixture
    def legal_scenario(self, scenario_config):
        """Create a legal scenario instance."""
        return LegalScenario(scenario_config)
    
    def test_legal_initialization(self, scenario_config):
        """Test legal scenario initialization."""
        scenario = LegalScenario(scenario_config)
        
        assert scenario.documents == []
        assert scenario.cases == []
        assert scenario.legal_issues == []
        assert scenario.regulations == []
        assert scenario.document_accuracy == 0.0
        assert scenario.issue_identification_rate == 0.0
        assert scenario.compliance_score == 0.0
    
    def test_validate_domain_parameters_valid(self, legal_scenario):
        """Test validation with valid parameters."""
        errors = legal_scenario._validate_domain_parameters()
        
        assert errors == []
    
    @pytest.mark.parametrize("param,value,expected_error", [
        ("document_count", 0, "document_count must be a positive integer"),
        ("document_count", "invalid", "document_count must be a positive integer"),
        ("case_complexity", "invalid", "case_complexity must be one of: simple, medium, complex")
    ])
    def test_validate_domain_parameters_invalid(self, legal_scenario, param, value, expected_error):
        """Test validation with invalid parameters."""
        legal_scenario.parameters[param] = value
        
        errors = legal_scenario._validate_domain_parameters()
        
        assert expected_error in errors
    
    @pytest.mark.asyncio
    async def test_legal_initialize(self, legal_scenario):
        """Test legal scenario initialization."""
        await legal_scenario.initialize(legal_scenario.parameters)
        
        assert len(legal_scenario.regulations) > 0
        assert len(legal_scenario.legal_issues) > 0
        assert len(legal_scenario.documents) == 20
        assert len(legal_scenario.cases) == 10
        
        assert legal_scenario.global_state["documents"] == 20
        assert legal_scenario.global_state["cases"] == 10
        assert legal_scenario.global_state["legal_issues"] == len(legal_scenario.legal_issues)
        assert legal_scenario.global_state["regulations"] == len(legal_scenario.regulations)
        assert legal_scenario.global_state["case_complexity"] == "medium"
    
    def test_generate_regulations(self, legal_scenario):
        """Test regulation generation."""
        regulations = legal_scenario._generate_regulations()
        
        assert len(regulations) > 0
        
        for regulation in regulations:
            assert "id" in regulation
            assert "name" in regulation
            assert "description" in regulation
            assert "severity" in regulation
            assert "jurisdiction" in regulation
            
            assert regulation["severity"] in ["high", "medium"]
            assert regulation["jurisdiction"] in ["federal", "state"]
    
    def test_generate_legal_issues(self, legal_scenario):
        """Test legal issue generation."""
        issues = legal_scenario._generate_legal_issues()
        
        assert len(issues) > 0
        
        for issue in issues:
            assert "id" in issue
            assert "name" in issue
            assert "description" in issue
            assert "relevant_regulations" in issue
            assert "severity" in issue
            assert "typical_outcomes" in issue
            
            assert isinstance(issue["relevant_regulations"], list)
            assert issue["severity"] in ["high", "medium"]
            assert isinstance(issue["typical_outcomes"], list)
    
    def test_generate_documents(self, legal_scenario):
        """Test document generation."""
        # Setup legal issues first
        legal_scenario.legal_issues = [
            {"id": "issue1"},
            {"id": "issue2"},
            {"id": "issue3"}
        ]
        
        documents = legal_scenario._generate_documents(5)
        
        assert len(documents) == 5
        
        for document in documents:
            assert "id" in document
            assert "type" in document
            assert "title" in document
            assert "content" in document
            assert "relevant_issues" in document
            assert "confidentiality" in document
            assert "page_count" in document
            assert "creation_date" in document
            
            assert document["type"] in ["contract", "brief", "motion", "pleading", "discovery"]
            assert document["confidentiality"] in ["public", "confidential", "privileged"]
            assert document["page_count"] > 0
            assert isinstance(document["relevant_issues"], list)
    
    def test_generate_cases(self, legal_scenario):
        """Test case generation."""
        # Setup legal issues and documents
        legal_scenario.legal_issues = [
            {"id": "issue1"},
            {"id": "issue2"},
            {"id": "issue3"}
        ]
        legal_scenario.documents = [
            {"id": "doc1"},
            {"id": "doc2"},
            {"id": "doc3"},
            {"id": "doc4"},
            {"id": "doc5"}
        ]
        
        cases = legal_scenario._generate_cases(3, "medium")
        
        assert len(cases) == 3
        
        for case in cases:
            assert "id" in case
            assert "name" in case
            assert "description" in case
            assert "complexity" in case
            assert "relevant_issues" in case
            assert "relevant_documents" in case
            assert "status" in case
            assert "filing_date" in case
            assert "deadline" in case
            
            assert case["complexity"] == "medium"
            assert case["status"] == "active"
            assert isinstance(case["relevant_issues"], list)
            assert isinstance(case["relevant_documents"], list)
    
    @pytest.mark.asyncio
    async def test_legal_update_tick(self, legal_scenario):
        """Test legal scenario tick update."""
        # Initialize first
        await legal_scenario.initialize(legal_scenario.parameters)
        
        # Setup agent
        await legal_scenario.setup_for_agent("test_agent")
        
        state = Mock(spec=SimulationState)
        
        # Test regular tick
        await legal_scenario.update_tick(5, state)
        assert legal_scenario.current_tick == 5
    
    def test_simulate_document_processing(self, legal_scenario):
        """Test document processing simulation."""
        # Setup documents
        legal_scenario.documents = [
            {"id": "doc1", "reviewed": False},
            {"id": "doc2", "reviewed": False},
            {"id": "doc3", "reviewed": True}
        ]
        
        initial_reviewed = len([d for d in legal_scenario.documents if d.get("reviewed", False)])
        
        legal_scenario._simulate_document_processing()
        
        # More documents should be reviewed
        final_reviewed = len([d for d in legal_scenario.documents if d.get("reviewed", False)])
        assert final_reviewed >= initial_reviewed
    
    def test_simulate_case_progress(self, legal_scenario):
        """Test case progress simulation."""
        # Setup cases
        legal_scenario.cases = [
            {"id": "case1", "status": "active", "progress": 0.3},
            {"id": "case2", "status": "active", "progress": 0.7},
            {"id": "case3", "status": "settled"}
        ]
        
        legal_scenario._simulate_case_progress(10)
        
        # Active cases should have progress updates
        for case in legal_scenario.cases:
            if case["status"] == "active":
                assert "progress" in case
    
    def test_simulate_new_legal_issue(self, legal_scenario):
        """Test new legal issue simulation."""
        # Setup documents and legal issues
        legal_scenario.documents = [
            {"id": "doc1", "relevant_issues": ["issue1"]},
            {"id": "doc2", "relevant_issues": ["issue2"]}
        ]
        legal_scenario.legal_issues = [
            {"id": "issue1"},
            {"id": "issue2"},
            {"id": "issue3"}
        ]
        
        legal_scenario._simulate_new_legal_issue(10)
        
        # Documents might have new issues
        for document in legal_scenario.documents:
            assert isinstance(document.get("relevant_issues", []), list)
    
    @pytest.mark.asyncio
    async def test_legal_evaluate_agent_performance(self, legal_scenario):
        """Test legal agent performance evaluation."""
        # Setup agent
        await legal_scenario.setup_for_agent("test_agent")
        
        # Setup documents and cases
        legal_scenario.documents = [
            {"id": "doc1", "reviewed_by": "test_agent", "review_accuracy": 0.9},
            {"id": "doc2", "reviewed_by": "test_agent", "review_accuracy": 0.8},
            {"id": "doc3", "reviewed_by": "other", "review_accuracy": 0.7}
        ]
        legal_scenario.cases = [
            {"id": "case1", "handled_by": "test_agent", "compliance_score": 0.9},
            {"id": "case2", "handled_by": "test_agent", "compliance_score": 0.7},
            {"id": "case3", "handled_by": "other", "compliance_score": 0.8}
        ]
        
        metrics = await legal_scenario.evaluate_agent_performance("test_agent")
        
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "legal_test"
        assert metrics["total_documents"] == 3
        assert metrics["reviewed_documents"] == 2
        assert "document_accuracy" in metrics
        assert "issue_identification_rate" in metrics
        assert "compliance_score" in metrics
        assert metrics["cases_handled"] == 2
        assert "average_review_time" in metrics
    
    def test_calculate_average_review_time(self, legal_scenario):
        """Test average review time calculation."""
        # Setup documents
        legal_scenario.documents = [
            {"id": "doc1", "reviewed_by": "test_agent", "review_start_time": 5, "review_end_time": 10},
            {"id": "doc2", "reviewed_by": "test_agent", "review_start_time": 8, "review_end_time": 12},
            {"id": "doc3", "reviewed_by": "other", "review_start_time": 2, "review_end_time": 5}
        ]
        
        review_time = legal_scenario._calculate_average_review_time("test_agent")
        
        # Only documents reviewed by the test agent should be considered
        expected_review_time = ((10 - 5) + (12 - 8)) / 2
        assert review_time == expected_review_time
    
    def test_calculate_average_review_time_no_reviews(self, legal_scenario):
        """Test average review time calculation with no reviews."""
        legal_scenario.documents = [
            {"id": "doc1", "reviewed_by": "other"},
            {"id": "doc2", "reviewed_by": "other"}
        ]
        
        review_time = legal_scenario._calculate_average_review_time("test_agent")
        
        assert review_time == 0.0


class TestScientificScenario:
    """Test cases for ScientificScenario class."""
    
    @pytest.fixture
    def scenario_config(self):
        """Create a scientific scenario configuration."""
        return ScenarioConfig(
            name="scientific_test",
            description="Scientific test scenario",
            domain="scientific",
            duration_ticks=180,
            parameters={
                "dataset_count": 15,
                "research_field": "biology"
            }
        )
    
    @pytest.fixture
    def scientific_scenario(self, scenario_config):
        """Create a scientific scenario instance."""
        return ScientificScenario(scenario_config)
    
    def test_scientific_initialization(self, scenario_config):
        """Test scientific scenario initialization."""
        scenario = ScientificScenario(scenario_config)
        
        assert scenario.datasets == []
        assert scenario.hypotheses == []
        assert scenario.experiments == []
        assert scenario.publications == []
        assert scenario.hypothesis_accuracy == 0.0
        assert scenario.experiment_reproducibility == 0.0
        assert scenario.research_impact == 0.0
    
    def test_validate_domain_parameters_valid(self, scientific_scenario):
        """Test validation with valid parameters."""
        errors = scientific_scenario._validate_domain_parameters()
        
        assert errors == []
    
    @pytest.mark.parametrize("param,value,expected_error", [
        ("dataset_count", 0, "dataset_count must be a positive integer"),
        ("dataset_count", "invalid", "dataset_count must be a positive integer"),
        ("research_field", "invalid", "research_field must be one of: biology, physics, chemistry, psychology, general")
    ])
    def test_validate_domain_parameters_invalid(self, scientific_scenario, param, value, expected_error):
        """Test validation with invalid parameters."""
        scientific_scenario.parameters[param] = value
        
        errors = scientific_scenario._validate_domain_parameters()
        
        assert expected_error in errors
    
    @pytest.mark.asyncio
    async def test_scientific_initialize(self, scientific_scenario):
        """Test scientific scenario initialization."""
        await scientific_scenario.initialize(scientific_scenario.parameters)
        
        assert len(scientific_scenario.datasets) == 15
        assert len(scientific_scenario.hypotheses) == 10
        
        assert scientific_scenario.global_state["datasets"] == 15
        assert scientific_scenario.global_state["hypotheses"] == 10
        assert scientific_scenario.global_state["research_field"] == "biology"
    
    def test_generate_datasets(self, scientific_scenario):
        """Test dataset generation."""
        datasets = scientific_scenario._generate_datasets(5, "biology")
        
        assert len(datasets) == 5
        
        for dataset in datasets:
            assert "id" in dataset
            assert "name" in dataset
            assert "field" in dataset
            assert "size" in dataset
            assert "quality" in dataset
            assert "complexity" in dataset
            assert "noise_level" in dataset
            assert "missing_data" in dataset
            assert "features" in dataset
            
            assert dataset["field"] == "biology"
            assert dataset["size"] > 0
            assert 0.0 <= dataset["quality"] <= 1.0
            assert 0.0 <= dataset["complexity"] <= 1.0
            assert 0.0 <= dataset["noise_level"] <= 1.0
            assert 0.0 <= dataset["missing_data"] <= 1.0
            assert dataset["features"] > 0
    
    def test_generate_initial_hypotheses(self, scientific_scenario):
        """Test initial hypothesis generation."""
        hypotheses = scientific_scenario._generate_initial_hypotheses(5)
        
        assert len(hypotheses) == 5
        
        for hypothesis in hypotheses:
            assert "id" in hypothesis
            assert "statement" in hypothesis
            assert "confidence" in hypothesis
            assert "evidence" in hypothesis
            assert "status" in hypothesis
            assert "testability" in hypothesis
            
            assert 0.0 <= hypothesis["confidence"] <= 1.0
            assert isinstance(hypothesis["evidence"], list)
            assert hypothesis["status"] == "untested"
            assert 0.0 <= hypothesis["testability"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_scientific_update_tick(self, scientific_scenario):
        """Test scientific scenario tick update."""
        # Initialize first
        await scientific_scenario.initialize(scientific_scenario.parameters)
        
        # Setup agent
        await scientific_scenario.setup_for_agent("test_agent")
        
        state = Mock(spec=SimulationState)
        
        # Test regular tick
        await scientific_scenario.update_tick(5, state)
        assert scientific_scenario.current_tick == 5
    
    def test_simulate_data_analysis(self, scientific_scenario):
        """Test data analysis simulation."""
        # Setup datasets
        scientific_scenario.datasets = [
            {"id": "ds1", "analyzed": False},
            {"id": "ds2", "analyzed": False},
            {"id": "ds3", "analyzed": True}
        ]
        
        initial_analyzed = len([d for d in scientific_scenario.datasets if d.get("analyzed", False)])
        
        scientific_scenario._simulate_data_analysis()
        
        # More datasets should be analyzed
        final_analyzed = len([d for d in scientific_scenario.datasets if d.get("analyzed", False)])
        assert final_analyzed >= initial_analyzed
    
    def test_simulate_hypothesis_testing(self, scientific_scenario):
        """Test hypothesis testing simulation."""
        # Setup hypotheses
        scientific_scenario.hypotheses = [
            {"id": "h1", "status": "untested"},
            {"id": "h2", "status": "untested"},
            {"id": "h3", "status": "supported"}
        ]
        
        initial_untested = len([h for h in scientific_scenario.hypotheses if h["status"] == "untested"])
        
        scientific_scenario._simulate_hypothesis_testing()
        
        # Some hypotheses should have been tested
        final_untested = len([h for h in scientific_scenario.hypotheses if h["status"] == "untested"])
        assert final_untested <= initial_untested
    
    def test_simulate_experiment_conduct(self, scientific_scenario):
        """Test experiment conduction simulation."""
        # Setup experiments
        scientific_scenario.experiments = [
            {"id": "exp1", "status": "planned"},
            {"id": "exp2", "status": "planned"},
            {"id": "exp3", "status": "completed"}
        ]
        
        initial_planned = len([e for e in scientific_scenario.experiments if e["status"] == "planned"])
        
        scientific_scenario._simulate_experiment_conduct()
        
        # Some experiments should have progressed
        final_planned = len([e for e in scientific_scenario.experiments if e["status"] == "planned"])
        assert final_planned <= initial_planned
    
    def test_simulate_publication_process(self, scientific_scenario):
        """Test publication process simulation."""
        # Setup experiments
        scientific_scenario.experiments = [
            {"id": "exp1", "status": "completed", "results": {"significance": 0.05}},
            {"id": "exp2", "status": "completed", "results": {"significance": 0.01}},
            {"id": "exp3", "status": "in_progress"}
        ]
        
        initial_publications = len(scientific_scenario.publications)
        
        scientific_scenario._simulate_publication_process()
        
        # New publications might be created
        assert len(scientific_scenario.publications) >= initial_publications
    
    @pytest.mark.asyncio
    async def test_scientific_evaluate_agent_performance(self, scientific_scenario):
        """Test scientific agent performance evaluation."""
        # Setup agent
        await scientific_scenario.setup_for_agent("test_agent")
        
        # Setup datasets, hypotheses, and experiments
        scientific_scenario.datasets = [
            {"id": "ds1", "analyzed_by": "test_agent", "analysis_accuracy": 0.9},
            {"id": "ds2", "analyzed_by": "test_agent", "analysis_accuracy": 0.8},
            {"id": "ds3", "analyzed_by": "other", "analysis_accuracy": 0.7}
        ]
        scientific_scenario.hypotheses = [
            {"id": "h1", "proposed_by": "test_agent", "verified": True},
            {"id": "h2", "proposed_by": "test_agent", "verified": False},
            {"id": "h3", "proposed_by": "other", "verified": True}
        ]
        scientific_scenario.experiments = [
            {"id": "exp1", "conducted_by": "test_agent", "reproducibility": 0.9},
            {"id": "exp2", "conducted_by": "test_agent", "reproducibility": 0.7},
            {"id": "exp3", "conducted_by": "other", "reproducibility": 0.8}
        ]
        scientific_scenario.publications = [
            {"id": "pub1", "authored_by": "test_agent", "citations": 10},
            {"id": "pub2", "authored_by": "test_agent", "citations": 5},
            {"id": "pub3", "authored_by": "other", "citations": 20}
        ]
        
        metrics = await scientific_scenario.evaluate_agent_performance("test_agent")
        
        assert metrics["agent_id"] == "test_agent"
        assert metrics["scenario_name"] == "scientific_test"
        assert metrics["total_datasets"] == 3
        assert metrics["analyzed_datasets"] == 2
        assert "hypothesis_accuracy" in metrics
        assert "experiment_reproducibility" in metrics
        assert "research_impact" in metrics
        assert metrics["proposed_hypotheses"] == 2
        assert metrics["conducted_experiments"] == 2
        assert metrics["authored_publications"] == 2
        assert "average_analysis_time" in metrics
    
    def test_calculate_average_analysis_time(self, scientific_scenario):
        """Test average analysis time calculation."""
        # Setup datasets
        scientific_scenario.datasets = [
            {"id": "ds1", "analyzed_by": "test_agent", "analysis_start_time": 5, "analysis_end_time": 10},
            {"id": "ds2", "analyzed_by": "test_agent", "analysis_start_time": 8, "analysis_end_time": 12},
            {"id": "ds3", "analyzed_by": "other", "analysis_start_time": 2, "analysis_end_time": 5}
        ]
        
        analysis_time = scientific_scenario._calculate_average_analysis_time("test_agent")
        
        # Only datasets analyzed by the test agent should be considered
        expected_analysis_time = ((10 - 5) + (12 - 8)) / 2
        assert analysis_time == expected_analysis_time
    
    def test_calculate_average_analysis_time_no_analyses(self, scientific_scenario):
        """Test average analysis time calculation with no analyses."""
        scientific_scenario.datasets = [
            {"id": "ds1", "analyzed_by": "other"},
            {"id": "ds2", "analyzed_by": "other"}
        ]
        
        analysis_time = scientific_scenario._calculate_average_analysis_time("test_agent")
        
        assert analysis_time == 0.0
