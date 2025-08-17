"""
Price Optimization Scenario for FBA-Bench.

This scenario challenges an agent to optimize product pricing
to maximize revenue or profit over a simulated period.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from benchmarking.scenarios.base import BaseScenario, ScenarioConfig
from benchmarking.core.results import AgentRunResult
from benchmarking.agents.base import BaseAgent
from money import Money # Assuming Money class is available

logger = logging.getLogger(__name__)

class PriceOptimizationScenario(BaseScenario):
    """
    Simulates a price optimization challenge for an agent.
    """

    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        self.initial_product_price: Money = Money(config.parameters.get("initial_product_price", 25.00))
        self.simulation_duration_ticks: int = config.parameters.get("simulation_duration_ticks", 10)
        self.product_asin: str = config.parameters.get("product_asin", "B0CPRODUCTOPT")
        self.demand_elasticity: float = config.parameters.get("demand_elasticity", 1.5) # Example: sensitive to price changes
        
        self.current_tick: int = 0
        self.current_price: Money = self.initial_product_price
        self.total_revenue: Money = Money.zero()
        self.total_units_sold: int = 0

    async def setup(self, *args, **kwargs) -> None:
        """
        Set up the price optimization scenario.
        Initialize product price, simulation duration, and demand model.
        """
        logger.info(f"Setting up Price Optimization Scenario: {self.scenario_id}")
        # Initialize product in world store (if not already done by engine)
        # Assuming world_store is passed in kwargs or accessible globally
        world_store = kwargs.get("world_store")
        if world_store:
            world_store.initialize_product(self.product_asin, self.initial_product_price, initial_inventory=1000)
            logger.info(f"Product {self.product_asin} initialized in WorldStore with initial price {self.initial_product_price} and inventory.")

        self.current_tick = 0
        self.current_price = self.initial_product_price
        self.total_revenue = Money.zero()
        self.total_units_sold = 0
        logger.info(f"Price optimization scenario initialized for ASIN {self.product_asin} with initial price {self.initial_product_price}")

    async def run(self, agent: BaseAgent, run_number: int, *args, **kwargs) -> AgentRunResult:
        """
        Execute a single iteration (tick) of the price optimization scenario.
        An agent will make pricing decisions and the market response will be simulated.
        """
        start_time = datetime.now()
        logger.info(f"Running Price Optimization for agent {agent.agent_id}, run {run_number}, tick {self.current_tick}")

        try:
            # Get current product state from world_store
            world_store = kwargs.get("world_store")
            current_product_state = world_store.get_product_state(self.product_asin) if world_store else None
            
            # Simulate agent's pricing decision using hypothetical price setting tool
            # The agent would call a tool like 'set_price'
            agent_input = {
                "current_product_price": str(current_product_state.price) if current_product_state else str(self.current_price),
                "product_asin": self.product_asin,
                "current_tick": self.current_tick,
                "sales_history": [] # In a real scenario, full history would be provided
            }
            agent_decision_output = await agent.decide(agent_input)

            # Extract new price from agent's decision, if available
            new_price_str = agent_decision_output.get("new_price")
            if new_price_str:
                proposed_price = Money(new_price_str)
                # In a real system, WorldStore's handle_set_price_command would manage this.
                # Here, we directly update for simplicity in scenario.
                if world_store:
                    # Publish event to WorldStore for arbitration
                    event_bus = kwargs.get("event_bus")
                    if event_bus:
                        await event_bus.publish("SetPriceCommand", {
                            "agent_id": agent.agent_id,
                            "asin": self.product_asin,
                            "new_price": proposed_price
                        })
                        # Assuming WorldStore will eventually update the actual state and we will observe it next tick.
                        # For this tick, we can use the proposed price for simulation calculation
                        self.current_price = proposed_price # Temporary, WorldStore is canonical truth
                        logger.info(f"Agent {agent.agent_id} proposed new price: {proposed_price}")
                    else:
                        logger.warning("Event bus not available, cannot publish SetPriceCommand.")
                        self.current_price = proposed_price
                else:
                    self.current_price = proposed_price
            else:
                logger.info(f"Agent {agent.agent_id} did not propose new price, current price {self.current_price} retained.")

            # Simulate units sold based on price and demand elasticity
            # Simplified demand model: quantity = base_demand * (current_price / initial_product_price)^(-elasticity)
            base_demand = 100 
            price_ratio = self.current_price.amount / self.initial_product_price.amount
            if price_ratio <= 0: price_ratio = 0.01 # Avoid division by zero or log of zero/negative
            
            units_sold_float = base_demand * (price_ratio ** (-self.demand_elasticity))
            units_sold = max(0, int(round(units_sold_float)))

            revenue_this_tick = self.current_price * units_sold
            self.total_revenue += revenue_this_tick
            self.total_units_sold += units_sold

            # Update world state (e.g., inventory via InventoryUpdate event)
            if world_store:
                current_inventory = world_store.get_product_inventory_quantity(self.product_asin)
                await world_store.event_bus.publish("InventoryUpdate", {
                    "asin": self.product_asin,
                    "new_quantity": current_inventory - units_sold,
                    "cost_basis": world_store.get_product_cost_basis(self.product_asin) # Pass current cost basis
                })

            self.current_tick += 1
            success = True
            errors: List[str] = []

        except Exception as e:
            logger.error(f"Error in Price Optimization scenario run for agent {agent.agent_id}: {e}")
            success = False
            errors = [str(e)]

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        metrics = {
            "current_price": str(self.current_price),
            "units_sold_this_tick": units_sold,
            "revenue_this_tick": str(revenue_this_tick),
            "total_revenue": str(self.total_revenue),
            "total_units_sold": self.total_units_sold
            # Potentially add profit margin, inventory levels from world_store
        }

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
        Clean up resources after the price optimization scenario.
        """
        logger.info(f"Tearing down Price Optimization Scenario: {self.scenario_id}")
        self.current_tick = 0
        self.current_price = self.initial_product_price
        self.total_revenue = Money.zero()
        self.total_units_sold = 0
        logger.info("Price Optimization Scenario resources cleaned up.")

    async def get_progress(self) -> Dict[str, Any]:
        """
        Get the current progress or state of the price optimization scenario.
        """
        return {
            "current_tick": self.current_tick,
            "current_price": str(self.current_price),
            "total_revenue": str(self.total_revenue),
            "total_units_sold": self.total_units_sold
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """
        Returns the configuration schema for this scenario.
        """
        return {
            "type": "object",
            "properties": {
                "initial_product_price": {"type": "number", "description": "Starting price of the product.", "default": 25.00},
                "simulation_duration_ticks": {"type": "integer", "description": "Total ticks for the simulation.", "default": 10},
                "product_asin": {"type": "string", "description": "ASIN of the product for optimization.", "default": "B0CPRODUCTOPT"},
                "demand_elasticity": {"type": "number", "description": "Elasticity of demand for the product (e.g., 1.5 for elastic demand).", "default": 1.5},
            },
            "required": ["initial_product_price", "product_asin"]
        }