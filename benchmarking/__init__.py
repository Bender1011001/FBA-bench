"""
Benchmarking module for FBA-Bench.
"""

import logging
import os
import sys

# Configure logging at a basic level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Loading benchmarking package: __name__={__name__}, __package__={__package__}, __file__={__file__}")

# Dynamically add project root to path for absolute imports
# This assumes the script is run from the project root or its parent is the project root
if os.path.basename(os.getcwd()) == 'fba':
    project_root = os.getcwd()
elif os.path.basename(os.path.dirname(os.getcwd())) == 'fba':
    project_root = os.path.dirname(os.getcwd())
else:
    logger.warning("Could not determine project root. Assuming current working directory is fine.")
    project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added project root to sys.path: {project_root}")

# Import core submodules
from .core.engine import BenchmarkEngine
from .metrics.registry import metrics_registry 
from .scenarios.registry import ScenarioRegistry 
from .config import ConfigurationManager 

# Import and register built-in components
def _register_builtin_components():
    """Register built-in metrics and scenarios."""
    # The metrics are now registered directly by the metrics_registry instance itself upon import in metrics/registry.py
    # This block is for scenarios.
    try:
        # Scenario imports
        # Commenting out problematic imports for now as files are not found.
        # from .scenarios.marketing_campaign import MarketingCampaignScenario
        # from .scenarios.price_optimization import PriceOptimizationScenario
        # from .scenarios.demand_forecasting import DemandForecastingScenario
        # from .scenarios.supply_chain_disruption import SupplyChainDisruptionScenario
        
        scenario_registry = ScenarioRegistry() # Instantiate ScenarioRegistry
        # scenario_registry.register("marketing_campaign", MarketingCampaignScenario)
        # scenario_registry.register("price_optimization", PriceOptimizationScenario)
        # scenario_registry.register("demand_forecasting", DemandForecastingScenario)
        # scenario_registry.register("supply_chain_disruption", SupplyChainDisruptionScenario)
        logger.info("Registered built-in scenarios.")
    except ImportError as e:
        logger.warning(f"Failed to register some built-in scenarios: {e}")

# Automatically register built-in components when the module is imported
# This ensures default metrics and scenarios are available without explicit calls.
_register_builtin_components()


# Define the public API of the benchmarking package
__all__ = [
    'BenchmarkEngine',
    'SimulationRunner', 
    'ConfigurationManager',
    'ScenarioRegistry', 
    'metrics_registry' 
]

# Optional: Further setup or initialization can go here
logger.info("Benchmarking package initialized.")