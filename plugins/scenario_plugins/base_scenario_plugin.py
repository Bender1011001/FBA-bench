import logging
from typing import Dict, Any, List, Protocol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScenarioPlugin(Protocol):
    """
    Base protocol for scenario plugins (type checking only).
    """
    __is_fba_plugin__: bool
    plugin_id: str
    version: str
    name: str
    description: str
    scenario_type: str

    def initialize(self, config: Dict[str, Any]): ...
    async def generate_initial_state(self) -> Dict[str, Any]: ...
    async def inject_events(self, current_time: int) -> List[Dict[str, Any]]: ...
    async def validate_scenario_constraints(self, scenario_data: Dict[str, Any]) -> bool: ...
    def get_documentation_template(self) -> str: ...
    def get_plugin_info(self) -> Dict[str, Any]: ...


class BaseScenarioPlugin:
    """
    Backwards-compat concrete base class expected by tests:
    from plugins.scenario_plugins.base_scenario_plugin import BaseScenarioPlugin
    """
    __is_fba_plugin__ = True  # Marker for PluginManager

    plugin_id: str = "base_scenario_plugin"
    version: str = "0.1.0"
    name: str = "Base Scenario Plugin"
    description: str = "A base class for creating FBA-Bench custom scenarios."
    scenario_type: str = "general"  # e.g., "seasonal_demand", "competitor_actions"

    def initialize(self, config: Dict[str, Any]) -> None:
        logging.info(f"Initializing ScenarioPlugin: {self.name} with config: {config}")

    async def generate_initial_state(self) -> Dict[str, Any]:
        logging.info(f"Generating initial state for scenario: {self.name}")
        return {
            "initial_cash": 10000.0,
            "initial_inventory": {"product_a": 500, "product_b": 200},
            "market_volatility": 0.1,
            "demand_pattern": "flat",
            "competitor_count": 2,
        }

    async def inject_events(self, current_time: int) -> List[Dict[str, Any]]:
        logging.info(f"Injecting events for scenario {self.name} at time {current_time}")
        events: List[Dict[str, Any]] = []
        if current_time == 10:
            events.append({
                "type": "market_price_shock",
                "product_id": "product_a",
                "price_change_factor": 0.8,
                "description": "Sudden market price drop due to new competitor.",
            })
        if current_time == 25:
            events.append({
                "type": "supply_chain_disruption",
                "affected_product": "product_b",
                "duration": 5,
                "impact": "inventory_reduction",
                "value": 0.2,
                "description": "Major port strike impacting product B supply.",
            })
        return events

    async def validate_scenario_constraints(self, scenario_data: Dict[str, Any]) -> bool:
        logging.info(f"Validating scenario constraints for {self.name}")
        if scenario_data.get("initial_cash", 0) < 0:
            logging.error(f"Scenario validation failed: initial_cash cannot be negative for {self.name}.")
            return False
        return True

    def get_documentation_template(self) -> str:
        return f"""
# Scenario Plugin: {self.name} Documentation

## Overview
{self.description}

## Configuration
This plugin accepts the following configuration parameters in its `initialize` method:
- `param_name_1`: (Type) Description of parameter 1.
- `param_name_2`: (Type) Description of parameter 2.

## Initial State Generation (`generate_initial_state`)
This method sets up the starting conditions for the simulation.
Example output:
```json
{{
    "initial_cash": 10000.0,
    "initial_inventory": {{"product_a": 500, "product_b": 200}},
    "market_volatility": 0.1,
    "demand_pattern": "flat",
    "competitor_count": 2
}}
```

## Event Injection (`inject_events`)
This method injects dynamic events throughout the simulation.
It's called at each time step.
Events should be returned as a list of dictionaries.
Example event structure:
```json
{{
    "type": "event_type_name",
    "param1": "value1",
    "param2": "value2",
    "description": "A human-readable description of the event."
}}
```
Common event types to consider:
- `market_price_shock`: Sudden change in product price.
- `demand_surge`/`demand_drop`: Sudden increase/decrease in product demand.
- `supply_chain_disruption`: Affects inventory availability.
- `competitor_action`: New competitor entry, price war, etc.

## Validation (`validate_scenario_constraints`)
This method ensures the scenario definition is logically sound and safe.
Implement checks for:
- Valid numerical ranges for economic parameters.
- Consistency of inventory and sales data.
- Prevention of illogical or game-breaking scenarios.

## Plugin Information (`get_plugin_info`)
Returns basic metadata about the plugin.
"""

    def get_plugin_info(self) -> Dict[str, Any]:
        """Returns metadata about the plugin."""
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "scenario_type": self.scenario_type,
        }
