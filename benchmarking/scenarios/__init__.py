"""
Scenarios module for the FBA-Bench benchmarking framework.

This module provides the core components for defining and managing scenarios
that are used to benchmark agents. It includes base classes, configurations,
and a registry for scenario implementations.
"""

from .registry import ScenarioRegistry, scenario_registry
from .base import BaseScenario, ScenarioConfig
from .marketing_campaign import MarketingCampaignScenario
from .price_optimization import PriceOptimizationScenario
from .demand_forecasting import DemandForecastingScenario
from .supply_chain_disruption import SupplyChainDisruptionScenario

# Ensure that necessary components from other modules are importable for scenarios
from ..agents.base import BaseAgent
from ..core.results import AgentRunResult

# Refined scenarios framework
from .refined_scenarios import (
    # Types and enums
    ScenarioType,
    ScenarioDifficulty,
    ScenarioMetrics,
    
    # Context and config
    ScenarioContext,
    ScenarioConfig,
)

__all__ = [
    # Legacy scenarios
    "ScenarioRegistry",
    "scenario_registry",
    "BaseScenario",
    "ScenarioConfig",
    "MarketingCampaignScenario",
    "MarketingCampaignScenarioConfig",
    "PriceOptimizationScenario",
    "PriceOptimizationScenarioConfig",
    "DemandForecastingScenario",
    "DemandForecastingScenarioConfig",
    "SupplyChainDisruptionScenario",
    "SupplyChainDisruptionConfig",
    "BaseAgent", # Exposed as it's a common dependency for scenario development
    "AgentRunResult", # Exposed as scenario's run method returns this.
    
    # Refined scenarios framework
    "ScenarioComponentType",
    "ComponentStatus",
    "ScenarioVersion",
    "ComponentContext",
    "ComponentResult",
    "ScenarioManifest",
    "BaseScenarioComponent",
    "PreconditionComponent",
    "ActionComponent",
    "ObservationComponent",
    "ValidationComponent",
    "CleanupComponent",
    "ComposableScenario",
    "RefinedScenarioRegistry",
    "refined_scenario_registry"
]

# You can add logic here to automatically register built-in scenarios if desired
def _register_builtin_scenarios():
    """Register built-in scenarios on module import."""
    scenario_registry.register("marketing_campaign", MarketingCampaignScenario)
    scenario_registry.register("price_optimization", PriceOptimizationScenario)
    scenario_registry.register("demand_forecasting", DemandForecastingScenario)
    scenario_registry.register("supply_chain_disruption", SupplyChainDisruptionScenario)

_register_builtin_scenarios()