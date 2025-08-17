"""
Supply Chain Disruption Scenario for FBA-Bench.

This scenario tests an agent's ability to respond to and mitigate the impact of
supply chain disruptions, such as unexpected delays or reduced availability.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import random

from benchmarking.scenarios.base import BaseScenario, ScenarioConfig
from benchmarking.core.results import AgentRunResult
from benchmarking.agents.base import BaseAgent
from services.world_store import WorldStore
from pydantic import Field, PositiveInt, confloat

logger = logging.getLogger(__name__)

class SupplyChainDisruptionConfig(ScenarioConfig):
    """Configuration for the Supply Chain Disruption Scenario."""
    product_asin: str = Field(
        "B0CSUPPLYCHAIN",
        description="ASIN of the product affected by supply chain disruption."
    )
    initial_inventory: PositiveInt = Field(
        1000,
        description="Initial inventory level for the product."
    )
    disruption_magnitude: confloat(ge=0.0, le=1.0) = Field(
        0.5,
        description="Magnitude of the disruption (0.0 to 1.0, e.g., 0.5 means 50% supply reduction)."
    )
    disruption_duration_ticks: PositiveInt = Field(
        3,
        description="Duration of the disruption in ticks."
    )
    disruption_start_tick: PositiveInt = Field(
        2,
        description="Tick at which the supply chain disruption begins."
    )
    demand_per_tick: PositiveInt = Field(
        100,
        description="Simulated base demand for the product per tick."
    )

class SupplyChainDisruptionScenario(BaseScenario):
    """
    Simulates a supply chain disruption and evaluates an agent's response.
    """

    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        self.scenario_config: SupplyChainDisruptionConfig = SupplyChainDisruptionConfig(**config.model_dump())

        self.product_asin: str = self.scenario_config.product_asin
        self.initial_inventory: int = self.scenario_config.initial_inventory
        self.disruption_magnitude: float = self.scenario_config.disruption_magnitude
        self.disruption_duration_ticks: int = self.scenario_config.disruption_duration_ticks
        self.disruption_start_tick: int = self.scenario_config.disruption_start_tick
        self.demand_per_tick: int = self.scenario_config.demand_per_tick

        self.current_tick: int = 0
        self.current_inventory: int = self.initial_inventory
        self.total_revenue_lost: float = 0.0
        self.original_supply_per_tick: int = self.demand_per_tick * 1.2 # Assume some buffer in normal supply
        self.current_supply_per_tick: int = self.original_supply_per_tick

    async def setup(self, *args, **kwargs) -> None:
        """
        Set up the supply chain disruption scenario.
        Initialize inventory and supply levels.
        """
        logger.info(f"Setting up Supply Chain Disruption Scenario: {self.scenario_id}")
        self.current_tick = 0
        self.current_inventory = self.initial_inventory
        self.total_revenue_lost = 0.0
        self.current_supply_per_tick = self.original_supply_per_tick
        logger.info(f"Initial inventory for {self.product_asin}: {self.initial_inventory}")

    async def run(self, agent: BaseAgent, run_number: int, *args, **kwargs) -> AgentRunResult:
        """
        Execute a single tick of the supply chain disruption scenario.
        Apply disruption, simulate demand, and measure agent's mitigation.
        """
        start_time = datetime.now()
        logger.info(f"Running Supply Chain Disruption for agent {agent.agent_id}, run {run_number}, tick {self.current_tick}")

        # Apply disruption if within the disruption window
        if self.disruption_start_tick <= self.current_tick < \
           self.disruption_start_tick + self.disruption_duration_ticks:
            reduced_supply = int(self.original_supply_per_tick * (1 - self.disruption_magnitude))
            self.current_supply_per_tick = max(0, reduced_supply) # Ensure supply doesn't go negative
            logger.warning(f"Supply chain disruption active! Supply reduced to {self.current_supply_per_tick} units/tick.")
        else:
            self.current_supply_per_tick = self.original_supply_per_tick

        # Simulate demand
        actual_demand = self.demand_per_tick # Simplified for now, can add variability
        
        # Agent decides on actions (e.g., re-order, change pricing, communicate with customers)
        agent_decision_output: Dict[str, Any] = {}
        success = True
        errors: List[str] = []

        try:
            agent_input = {
                "product_asin": self.product_asin,
                "current_inventory": self.current_inventory,
                "current_supply_rate": self.current_supply_per_tick,
                "estimated_demand": actual_demand,
                "tick": self.current_tick,
                "disruption_active": (self.disruption_start_tick <= self.current_tick < \
                                      self.disruption_start_tick + self.disruption_duration_ticks)
            }
            agent_decision_output = await agent.decide(agent_input)

            # Agent's mitigation actions could be processed here by WorldStore
            # For this scenario, we primarily track inventory and revenue loss
            # Example: Agent might request a new supply order, which would update current_inventory
            # This would typically involve an interaction with WorldStore through EventBus
            # For now, we'll assume the agent's "decision" itself is the metric.
            
        except Exception as e:
            logger.error(f"Error in Supply Chain Disruption scenario run for agent {agent.agent_id}: {e}")
            success = False
            errors.append(str(e))
            # If agent fails, assume no effective mitigation

        # Update inventory based on supply and demand
        self.current_inventory += self.current_supply_per_tick # New supply arrives
        
        # Fulfill demand as much as possible
        units_sold = min(actual_demand, self.current_inventory)
        self.current_inventory -= units_sold

        # Calculate lost revenue due to unmet demand
        unmet_demand = actual_demand - units_sold
        if unmet_demand > 0:
            # Assuming a standard price of 100 per unit for lost revenue calculation
            lost_revenue_this_tick = unmet_demand * 100.0 # Placeholder price
            self.total_revenue_lost += lost_revenue_this_tick
            logger.warning(f"Unmet demand: {unmet_demand}. Lost revenue this tick: ${lost_revenue_this_tick:.2f}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        metrics = {
            "current_inventory": self.current_inventory,
            "units_sold_this_tick": units_sold,
            "unmet_demand_this_tick": unmet_demand,
            "total_revenue_lost_so_far": self.total_revenue_lost,
            "supply_rate_this_tick": self.current_supply_per_tick,
            "is_disruption_active": (self.disruption_start_tick <= self.current_tick < \
                                     self.disruption_start_tick + self.disruption_duration_ticks)
        }

        self.current_tick += 1

        return AgentRunResult(
            agent_id=agent.agent_id,
            scenario_name=self.config.name,
            run_number=run_number,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            errors=errors,
            metrics=metrics
        )

    async def teardown(self, *args, **kwargs) -> None:
        """
        Clean up resources after the supply chain disruption scenario.
        """
        logger.info(f"Tearing down Supply Chain Disruption Scenario: {self.scenario_id}")
        self.current_tick = 0
        self.current_inventory = self.initial_inventory
        self.total_revenue_lost = 0.0
        logger.info("Supply Chain Disruption Scenario resources cleaned up.")

    async def get_progress(self) -> Dict[str, Any]:
        """
        Provides current progress of the scenario.
        """
        return {
            "current_tick": self.current_tick,
            "current_inventory": self.current_inventory,
            "total_revenue_lost": self.total_revenue_lost,
            "disruption_active": (self.disruption_start_tick <= self.current_tick < \
                                  self.disruption_start_tick + self.disruption_duration_ticks)
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """
        Returns the configuration schema for the Supply Chain Disruption Scenario.
        """
        return SupplyChainDisruptionConfig.model_json_schema()