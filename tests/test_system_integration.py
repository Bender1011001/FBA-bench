"""
System Integration Tests - Comprehensive testing of the FBA-Bench system.

This module provides comprehensive integration tests for the FBA-Bench system,
testing the interaction between all components and ensuring they work together
correctly.
"""

import asyncio
import json
import logging
import os
import pytest
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# Import the components to test
from api_server import app, lifespan
from benchmarking.core.engine import BenchmarkEngine
from benchmarking.agents.unified_agent import BaseUnifiedAgent, AgentFactory
from agent_runners.agent_manager import AgentManager
from services.world_store import WorldStore, ProductState, CommandArbitrationResult
from constraints.token_counter import TokenCounter, TokenCountResult
from benchmarking.scenarios.refined_scenarios import (
    ScenarioFactory, ScenarioConfig, ScenarioDifficulty, ScenarioType,
    PricingScenario, InventoryScenario, CompetitiveScenario
)
from benchmarking.config.pydantic_config import PydanticConfig, LLMConfig, AgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSystemIntegration:
    """
    Test class for system integration tests.
    
    This class tests the integration between all components of the FBA-Bench system.
    """
    
    @pytest.fixture
    def test_config(self) -> PydanticConfig:
        """
        Create a test configuration.
        
        Returns:
            Test configuration
        """
        return PydanticConfig(
            llm_config=LLMConfig(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000,
                api_key="test-api-key"
            ),
            agent_configs=[
                AgentConfig(
                    name="test-agent",
                    type="diy",
                    config={
                        "llm_config": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.7,
                            "api_key": "test-api-key"
                        },
                        "system_prompt": "You are a helpful assistant."
                    }
                )
            ],
            benchmark_config={
                "max_ticks": 100,
                "time_limit": 60.0,
                "metrics": ["revenue", "profit", "costs"]
            }
        )
    
    @pytest.fixture
    def world_store(self) -> WorldStore:
        """
        Create a test WorldStore.
        
        Returns:
            Test WorldStore
        """
        return WorldStore()
    
    @pytest.fixture
    def agent_manager(self, world_store: WorldStore, test_config: PydanticConfig) -> AgentManager:
        """
        Create a test AgentManager.
        
        Args:
            world_store: Test WorldStore
            test_config: Test configuration
            
        Returns:
            Test AgentManager
        """
        return AgentManager(
            world_store=world_store,
            openrouter_api_key=test_config.llm_config.api_key,
            use_unified_agents=True
        )
    
    @pytest.fixture
    def benchmark_engine(self, test_config: PydanticConfig, agent_manager: AgentManager) -> BenchmarkEngine:
        """
        Create a test BenchmarkEngine.
        
        Args:
            test_config: Test configuration
            agent_manager: Test AgentManager
            
        Returns:
            Test BenchmarkEngine
        """
        return BenchmarkEngine(config=test_config)
    
    @pytest.fixture
    def token_counter(self) -> TokenCounter:
        """
        Create a test TokenCounter.
        
        Returns:
            Test TokenCounter
        """
        return TokenCounter()
    
    @pytest.fixture
    def pricing_scenario(self) -> PricingScenario:
        """
        Create a test PricingScenario.
        
        Returns:
            Test PricingScenario
        """
        config = ScenarioConfig(
            name="test-pricing",
            description="Test pricing scenario",
            difficulty=ScenarioDifficulty.EASY,
            scenario_type=ScenarioType.PRICING,
            max_ticks=50,
            time_limit=30.0,
            initial_state={
                "base_price": 10.0,
                "cost_per_unit": 5.0,
                "max_price": 20.0,
                "min_price": 5.0,
                "price_elasticity": -1.5,
                "competitor_price": 10.0,
                "competitor_price_volatility": 0.1
            },
            success_criteria={
                "min_profit": 100.0
            },
            failure_criteria={
                "max_loss": -50.0
            }
        )
        return PricingScenario(config)
    
    @pytest.fixture
    def inventory_scenario(self) -> InventoryScenario:
        """
        Create a test InventoryScenario.
        
        Returns:
            Test InventoryScenario
        """
        config = ScenarioConfig(
            name="test-inventory",
            description="Test inventory scenario",
            difficulty=ScenarioDifficulty.MEDIUM,
            scenario_type=ScenarioType.INVENTORY,
            max_ticks=50,
            time_limit=30.0,
            initial_state={
                "initial_inventory": 100,
                "holding_cost_per_unit": 0.1,
                "stockout_cost_per_unit": 2.0,
                "order_cost_per_order": 10.0,
                "lead_time": 3,
                "max_order_quantity": 200,
                "demand_mean": 20,
                "demand_std": 5
            },
            success_criteria={
                "max_cost": 500.0
            },
            failure_criteria={
                "min_fulfillment_rate": 0.5
            }
        )
        return InventoryScenario(config)
    
    @pytest.fixture
    def competitive_scenario(self) -> CompetitiveScenario:
        """
        Create a test CompetitiveScenario.
        
        Returns:
            Test CompetitiveScenario
        """
        config = ScenarioConfig(
            name="test-competitive",
            description="Test competitive scenario",
            difficulty=ScenarioDifficulty.HARD,
            scenario_type=ScenarioType.COMPETITIVE,
            max_ticks=50,
            time_limit=30.0,
            initial_state={
                "num_competitors": 3,
                "market_size": 1000,
                "competitor_strategies": ["aggressive", "moderate", "conservative"],
                "price_sensitivity": 1.0,
                "quality_sensitivity": 0.8,
                "marketing_sensitivity": 0.5,
                "agent_price": 10.0,
                "agent_quality": 0.7,
                "agent_marketing": 0.5
            },
            success_criteria={
                "min_market_share": 0.3
            },
            failure_criteria={
                "max_market_share": 0.1
            }
        )
        return CompetitiveScenario(config)
    
    def test_world_store_initialization(self, world_store: WorldStore):
        """
        Test WorldStore initialization.
        
        Args:
            world_store: Test WorldStore
        """
        assert world_store is not None
        assert world_store.event_bus is not None
        assert world_store.storage_backend is not None
        assert len(world_store._product_state) == 0
    
    def test_world_store_product_state(self, world_store: WorldStore):
        """
        Test WorldStore product state management.
        
        Args:
            world_store: Test WorldStore
        """
        # Create a product state
        product_id = "test-product"
        product_state = ProductState(
            product_id=product_id,
            price=10.0,
            inventory=100,
            quality=0.8
        )
        
        # Set the product state
        world_store.set_product_state(product_id, product_state)
        
        # Get the product state
        retrieved_state = world_store.get_product_state(product_id)
        
        # Verify the state
        assert retrieved_state is not None
        assert retrieved_state.product_id == product_id
        assert retrieved_state.price == 10.0
        assert retrieved_state.inventory == 100
        assert retrieved_state.quality == 0.8
    
    def test_world_store_command_arbitration(self, world_store: WorldStore):
        """
        Test WorldStore command arbitration.
        
        Args:
            world_store: Test WorldStore
        """
        # Create a product state
        product_id = "test-product"
        product_state = ProductState(
            product_id=product_id,
            price=10.0,
            inventory=100,
            quality=0.8
        )
        
        # Set the product state
        world_store.set_product_state(product_id, product_state)
        
        # Create conflicting commands
        command1 = {
            "type": "update_price",
            "product_id": product_id,
            "price": 15.0,
            "timestamp": time.time()
        }
        
        command2 = {
            "type": "update_price",
            "product_id": product_id,
            "price": 12.0,
            "timestamp": time.time() + 1  # Later timestamp
        }
        
        # Arbitrate the commands
        result = world_store.arbitrate_commands([command1, command2])
        
        # Verify the result
        assert result is not None
        assert result.winning_command == command2
        assert result.reason == "timestamp"
    
    def test_agent_manager_initialization(self, agent_manager: AgentManager):
        """
        Test AgentManager initialization.
        
        Args:
            agent_manager: Test AgentManager
        """
        assert agent_manager is not None
        assert agent_manager.event_bus is not None
        assert agent_manager.world_store is not None
        assert agent_manager.use_unified_agents is True
        assert len(agent_manager.agent_registry._agents) == 0
    
    def test_agent_manager_create_agent(self, agent_manager: AgentManager, test_config: PydanticConfig):
        """
        Test AgentManager agent creation.
        
        Args:
            agent_manager: Test AgentManager
            test_config: Test configuration
        """
        # Create an agent
        agent_config = test_config.agent_configs[0]
        agent_id = agent_manager.create_agent(agent_config)
        
        # Verify the agent was created
        assert agent_id is not None
        assert agent_manager.agent_registry.get_agent(agent_id) is not None
        
        # Clean up
        agent_manager.remove_agent(agent_id)
    
    def test_agent_manager_decision_cycle(self, agent_manager: AgentManager, test_config: PydanticConfig):
        """
        Test AgentManager decision cycle.
        
        Args:
            agent_manager: Test AgentManager
            test_config: Test configuration
        """
        # Create an agent
        agent_config = test_config.agent_configs[0]
        agent_id = agent_manager.create_agent(agent_config)
        
        # Create a context
        context = {
            "tick": 1,
            "product_state": {
                "test-product": {
                    "price": 10.0,
                    "inventory": 100,
                    "quality": 0.8
                }
            },
            "market_state": {
                "demand": 50,
                "competitor_price": 12.0
            }
        }
        
        # Run a decision cycle
        with patch('agent_runners.agent_manager.AgentManager._execute_agent_decision') as mock_execute:
            mock_execute.return_value = {"action": "set_price", "price": 11.0}
            
            decisions = agent_manager.decision_cycle(context)
            
            # Verify the decisions
            assert decisions is not None
            assert len(decisions) == 1
            assert decisions[0]["agent_id"] == agent_id
            assert decisions[0]["action"] == "set_price"
            assert decisions[0]["price"] == 11.0
        
        # Clean up
        agent_manager.remove_agent(agent_id)
    
    def test_benchmark_engine_initialization(self, benchmark_engine: BenchmarkEngine):
        """
        Test BenchmarkEngine initialization.
        
        Args:
            benchmark_engine: Test BenchmarkEngine
        """
        assert benchmark_engine is not None
        assert benchmark_engine.config is not None
        assert benchmark_engine.event_bus is not None
        assert benchmark_engine.world_store is not None
        assert benchmark_engine.agent_manager is not None
        assert benchmark_engine.is_running is False
    
    def test_benchmark_engine_run_benchmark(self, benchmark_engine: BenchmarkEngine, pricing_scenario: PricingScenario):
        """
        Test BenchmarkEngine benchmark execution.
        
        Args:
            benchmark_engine: Test BenchmarkEngine
            pricing_scenario: Test PricingScenario
        """
        # Mock the agent manager to avoid actual LLM calls
        with patch.object(benchmark_engine.agent_manager, 'decision_cycle') as mock_decision_cycle:
            mock_decision_cycle.return_value = [
                {"agent_id": "test-agent", "action": "set_price", "price": 11.0}
            ]
            
            # Run the benchmark
            results = benchmark_engine.run_benchmark(
                scenario=pricing_scenario,
                agent_configs=benchmark_engine.config.agent_configs
            )
            
            # Verify the results
            assert results is not None
            assert "scenario" in results
            assert "agents" in results
            assert "metrics" in results
            assert "execution_time" in results
            assert results["scenario"]["name"] == "test-pricing"
            assert len(results["agents"]) == 1
            assert results["metrics"]["total_ticks"] > 0
    
    def test_token_counter_initialization(self, token_counter: TokenCounter):
        """
        Test TokenCounter initialization.
        
        Args:
            token_counter: Test TokenCounter
        """
        assert token_counter is not None
        assert token_counter.default_model == "gpt-3.5-turbo"
        assert token_counter.encoding_cache == {}
    
    def test_token_counter_count_tokens(self, token_counter: TokenCounter):
        """
        Test TokenCounter token counting.
        
        Args:
            token_counter: Test TokenCounter
        """
        # Test with empty text
        result = token_counter.count_tokens("")
        assert result.count == 0
        assert result.method == "empty"
        
        # Test with simple text
        text = "Hello, world!"
        result = token_counter.count_tokens(text)
        assert result.count > 0
        assert result.text_sample == text
        assert result.estimated == (not token_counter.is_available())
    
    def test_token_counter_count_messages(self, token_counter: TokenCounter):
        """
        Test TokenCounter message counting.
        
        Args:
            token_counter: Test TokenCounter
        """
        # Test with empty messages
        result = token_counter.count_messages([])
        assert result.count == 0
        assert result.method == "empty"
        
        # Test with simple messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        result = token_counter.count_messages(messages)
        assert result.count > 0
        assert result.estimated == (not token_counter.is_available())
    
    def test_pricing_scenario_initialization(self, pricing_scenario: PricingScenario):
        """
        Test PricingScenario initialization.
        
        Args:
            pricing_scenario: Test PricingScenario
        """
        assert pricing_scenario is not None
        assert pricing_scenario.config.name == "test-pricing"
        assert pricing_scenario.config.difficulty == ScenarioDifficulty.EASY
        assert pricing_scenario.config.scenario_type == ScenarioType.PRICING
        assert pricing_scenario.is_complete is False
        assert pricing_scenario.is_success is False
    
    def test_pricing_scenario_execution(self, pricing_scenario: PricingScenario):
        """
        Test PricingScenario execution.
        
        Args:
            pricing_scenario: Test PricingScenario
        """
        # Start the scenario
        pricing_scenario.start()
        
        # Execute a few steps
        for i in range(5):
            action = {"price": 10.0 + i}  # Simple pricing action
            result = pricing_scenario.step(action)
            
            # Verify the result
            assert result is not None
            assert result["tick"] == i + 1
            assert result["price"] == 10.0 + i
            assert "competitor_price" in result
            assert "demand" in result
            assert "units_sold" in result
            assert "revenue" in result
            assert "profit" in result
            
            # Stop if complete
            if result["is_complete"]:
                break
        
        # Verify the scenario state
        assert pricing_scenario.context.tick > 0
        assert pricing_scenario.metrics.revenue >= 0
        assert len(pricing_scenario.context.history) > 0
    
    def test_inventory_scenario_initialization(self, inventory_scenario: InventoryScenario):
        """
        Test InventoryScenario initialization.
        
        Args:
            inventory_scenario: Test InventoryScenario
        """
        assert inventory_scenario is not None
        assert inventory_scenario.config.name == "test-inventory"
        assert inventory_scenario.config.difficulty == ScenarioDifficulty.MEDIUM
        assert inventory_scenario.config.scenario_type == ScenarioType.INVENTORY
        assert inventory_scenario.is_complete is False
        assert inventory_scenario.is_success is False
    
    def test_inventory_scenario_execution(self, inventory_scenario: InventoryScenario):
        """
        Test InventoryScenario execution.
        
        Args:
            inventory_scenario: Test InventoryScenario
        """
        # Start the scenario
        inventory_scenario.start()
        
        # Execute a few steps
        for i in range(5):
            action = {"order_quantity": 20}  # Simple ordering action
            result = inventory_scenario.step(action)
            
            # Verify the result
            assert result is not None
            assert result["tick"] == i + 1
            assert "inventory" in result
            assert "demand" in result
            assert "units_sold" in result
            assert "stockout_quantity" in result
            assert "total_cost" in result
            
            # Stop if complete
            if result["is_complete"]:
                break
        
        # Verify the scenario state
        assert inventory_scenario.context.tick > 0
        assert inventory_scenario.metrics.costs >= 0
        assert len(inventory_scenario.context.history) > 0
    
    def test_competitive_scenario_initialization(self, competitive_scenario: CompetitiveScenario):
        """
        Test CompetitiveScenario initialization.
        
        Args:
            competitive_scenario: Test CompetitiveScenario
        """
        assert competitive_scenario is not None
        assert competitive_scenario.config.name == "test-competitive"
        assert competitive_scenario.config.difficulty == ScenarioDifficulty.HARD
        assert competitive_scenario.config.scenario_type == ScenarioType.COMPETITIVE
        assert competitive_scenario.is_complete is False
        assert competitive_scenario.is_success is False
    
    def test_competitive_scenario_execution(self, competitive_scenario: CompetitiveScenario):
        """
        Test CompetitiveScenario execution.
        
        Args:
            competitive_scenario: Test CompetitiveScenario
        """
        # Start the scenario
        competitive_scenario.start()
        
        # Execute a few steps
        for i in range(5):
            action = {
                "price": 10.0 + i,
                "quality": 0.7,
                "marketing": 0.5
            }  # Simple competitive action
            result = competitive_scenario.step(action)
            
            # Verify the result
            assert result is not None
            assert result["tick"] == i + 1
            assert "agent_price" in result
            assert "agent_quality" in result
            assert "agent_marketing" in result
            assert "agent_market_share" in result
            assert "agent_sales" in result
            assert "agent_revenue" in result
            assert "agent_profit" in result
            assert "competitors" in result
            
            # Stop if complete
            if result["is_complete"]:
                break
        
        # Verify the scenario state
        assert competitive_scenario.context.tick > 0
        assert competitive_scenario.metrics.revenue >= 0
        assert len(competitive_scenario.context.history) > 0
    
    def test_scenario_factory(self):
        """
        Test ScenarioFactory.
        """
        # Test pricing scenario creation
        pricing_config = ScenarioConfig(
            name="factory-pricing",
            description="Factory pricing scenario",
            difficulty=ScenarioDifficulty.EASY,
            scenario_type=ScenarioType.PRICING
        )
        pricing_scenario = ScenarioFactory.create_scenario(pricing_config)
        assert isinstance(pricing_scenario, PricingScenario)
        
        # Test inventory scenario creation
        inventory_config = ScenarioConfig(
            name="factory-inventory",
            description="Factory inventory scenario",
            difficulty=ScenarioDifficulty.MEDIUM,
            scenario_type=ScenarioType.INVENTORY
        )
        inventory_scenario = ScenarioFactory.create_scenario(inventory_config)
        assert isinstance(inventory_scenario, InventoryScenario)
        
        # Test competitive scenario creation
        competitive_config = ScenarioConfig(
            name="factory-competitive",
            description="Factory competitive scenario",
            difficulty=ScenarioDifficulty.HARD,
            scenario_type=ScenarioType.COMPETITIVE
        )
        competitive_scenario = ScenarioFactory.create_scenario(competitive_config)
        assert isinstance(competitive_scenario, CompetitiveScenario)
        
        # Test convenience methods
        pricing_scenario = ScenarioFactory.create_pricing_scenario("convenience-pricing")
        assert isinstance(pricing_scenario, PricingScenario)
        
        inventory_scenario = ScenarioFactory.create_inventory_scenario("convenience-inventory")
        assert isinstance(inventory_scenario, InventoryScenario)
        
        competitive_scenario = ScenarioFactory.create_competitive_scenario("convenience-competitive")
        assert isinstance(competitive_scenario, CompetitiveScenario)
    
    def test_full_system_integration(self, benchmark_engine: BenchmarkEngine, pricing_scenario: PricingScenario):
        """
        Test full system integration.
        
        Args:
            benchmark_engine: Test BenchmarkEngine
            pricing_scenario: Test PricingScenario
        """
        # Mock the agent manager to avoid actual LLM calls
        with patch.object(benchmark_engine.agent_manager, 'decision_cycle') as mock_decision_cycle:
            mock_decision_cycle.return_value = [
                {"agent_id": "test-agent", "action": "set_price", "price": 11.0}
            ]
            
            # Run the benchmark
            results = benchmark_engine.run_benchmark(
                scenario=pricing_scenario,
                agent_configs=benchmark_engine.config.agent_configs
            )
            
            # Verify the results
            assert results is not None
            assert "scenario" in results
            assert "agents" in results
            assert "metrics" in results
            assert "execution_time" in results
            
            # Verify scenario results
            assert results["scenario"]["name"] == "test-pricing"
            assert results["scenario"]["is_complete"] is True
            
            # Verify agent results
            assert len(results["agents"]) == 1
            assert results["agents"][0]["agent_id"] == "test-agent"
            assert "metrics" in results["agents"][0]
            
            # Verify metrics
            assert results["metrics"]["total_ticks"] > 0
            assert results["metrics"]["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_api_server_lifespan(self, test_config: PydanticConfig):
        """
        Test API server lifespan management.
        
        Args:
            test_config: Test configuration
        """
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config.dict(), f)
            config_path = f.name
        
        try:
            # Mock the environment variable
            with patch.dict(os.environ, {"FBA_CONFIG_PATH": config_path}):
                # Create a lifespan context
                async with lifespan(app):
                    # Verify the app state
                    assert hasattr(app.state, 'config')
                    assert hasattr(app.state, 'world_store')
                    assert hasattr(app.state, 'agent_manager')
                    assert hasattr(app.state, 'benchmark_engine')
                    assert hasattr(app.state, 'token_counter')
                    
                    # Verify the components
                    assert app.state.config is not None
                    assert app.state.world_store is not None
                    assert app.state.agent_manager is not None
                    assert app.state.benchmark_engine is not None
                    assert app.state.token_counter is not None
        finally:
            # Clean up
            os.unlink(config_path)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])