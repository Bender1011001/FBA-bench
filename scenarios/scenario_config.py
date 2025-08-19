from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os
import yaml
import logging
from scenarios.scenario_framework import ScenarioConfig
from scenarios.dynamic_generator import DynamicScenarioGenerator

@dataclass
class ScenarioConfiguration:
    """
    Dataclass for holding a comprehensive scenario configuration.
    This mirrors the structure expected in the YAML files.
    """
    scenario_name: str
    difficulty_tier: int
    expected_duration: int # simulation days
    success_criteria: Dict[str, float]
    market_conditions: Dict[str, Any]
    external_events: List[Dict[str, Any]]
    agent_constraints: Dict[str, Any]
    multi_agent_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    # Additional fields can be added here as the simulation evolves
    product_catalog: Optional[List[Dict[str,Any]]] = field(default_factory=list)
    business_parameters: Optional[Dict[str,Any]] = field(default_factory=dict)

class ScenarioConfigManager:
    """
    Manages scenario templates, validation rules, difficulty metrics,
    and provides integration hooks for simulation systems.
    """
    def __init__(self, scenario_base_path: str = 'scenarios/'):
        self.scenario_base_path = scenario_base_path
        self.business_types_path = os.path.join(scenario_base_path, 'business_types')
        self.multi_agent_path = os.path.join(scenario_base_path, 'multi_agent')
        self.dynamic_generator = DynamicScenarioGenerator()
        self._load_all_scenario_metadata()
        self.default_tier_configs = {
            0: os.path.join(scenario_base_path, "tier_0_baseline.yaml"),
            1: os.path.join(scenario_base_path, "tier_1_moderate.yaml"),
            2: os.path.join(scenario_base_path, "tier_2_advanced.yaml"),
            3: os.path.join(scenario_base_path, "tier_3_expert.yaml"),
        }
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def _load_all_scenario_metadata(self):
        """
        Loads metadata for all available scenarios by scanning directories.
        """
        self.available_scenarios: Dict[str, str] = {} # {scenario_name: filepath}
        
        # Load standard tier scenarios
        for tier_file in os.listdir(self.scenario_base_path):
            if tier_file.startswith('tier_') and tier_file.endswith('.yaml'):
                name = os.path.splitext(tier_file)[0]
                self.available_scenarios[name] = os.path.join(self.scenario_base_path, tier_file)

        # Load business_types scenarios
        if os.path.exists(self.business_types_path):
            for file_name in os.listdir(self.business_types_path):
                if file_name.endswith('.yaml'):
                    name = os.path.splitext(file_name)[0]
                    self.available_scenarios[name] = os.path.join(self.business_types_path, file_name)

        # Load multi_agent scenarios
        if os.path.exists(self.multi_agent_path):
            for file_name in os.listdir(self.multi_agent_path):
                if file_name.endswith('.yaml'):
                    name = os.path.splitext(file_name)[0]
                    self.available_scenarios[name] = os.path.join(self.multi_agent_path, file_name)
        
        logging.info(f"Loaded {len(self.available_scenarios)} available scenarios.")
        logging.debug(f"Available Scenarios: {self.available_scenarios.keys()}")

    def get_scenario_path(self, scenario_name: str) -> Optional[str]:
        """Returns the file path for a given scenario name."""
        return self.available_scenarios.get(scenario_name)

    def load_specific_scenario(self, scenario_name: str) -> ScenarioConfig:
        """Loads a specific scenario by name."""
        file_path = self.get_scenario_path(scenario_name)
        if not file_path:
            raise ValueError(f"Scenario '{scenario_name}' not found.")
        return ScenarioConfig.from_yaml(file_path)

    def get_scenarios_by_tier(self, tier: int) -> List[ScenarioConfig]:
        """
        Retrieves all scenarios (including business_types) associated with a given tier.
        This operation requires loading each file to check its tier.
        """
        tier_scenarios = []
        # First, check default tier configs
        if tier in self.default_tier_configs:
            try:
                tier_scenarios.append(ScenarioConfig.from_yaml(self.default_tier_configs[tier]))
            except Exception as e:
                logging.warning(f"Could not load default tier {tier} scenario: {e}")
        
        # Then, scan all available scenarios to find those matching the tier
        for name, filepath in self.available_scenarios.items():
            try:
                config = ScenarioConfig.from_yaml(filepath)
                if config.config_data.get('difficulty_tier') == tier and config not in tier_scenarios:
                    tier_scenarios.append(config)
            except Exception as e:
                logging.warning(f"Could not load metadata for '{name}': {e}")
        
        if not tier_scenarios:
            logging.info(f"No scenarios found for tier {tier}.")
        return tier_scenarios

    def validate_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Ensures scenario configurations are valid based on a schema or rules."""
        try:
            scenario_config = ScenarioConfig(config_data)
            return scenario_config.validate_scenario_consistency()
        except ValueError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

    def quantify_difficulty(self, scenario_config: ScenarioConfig) -> Dict[str, Any]:
        """
        Quantifies scenario complexity objectively based on various parameters.
        This provides a programmatic way to get difficulty metrics beyond just the tier.
        """
        metrics = {
            "num_external_events": len(scenario_config.config_data.get('external_events', [])),
            "economic_volatility": scenario_config.config_data.get('market_conditions', {}).get('economic_cycles', 'stable'),
            "competition_level": scenario_config.config_data.get('market_conditions', {}).get('competition_levels', 'low'),
            "supply_chain_complexity": scenario_config.config_data.get('business_parameters', {}).get('supply_chain_complexity', 'simple'),
            "initial_capital_normalized": scenario_config.config_data.get('agent_constraints', {}).get('initial_capital', 100000), # Higher capital = easier
            "has_multi_agent": 'multi_agent_config' in scenario_config.config_data and scenario_config.config_data['multi_agent_config'].get('num_agents', 0) > 1,
            "information_asymmetry_present": scenario_config.config_data.get('agent_constraints', {}).get('information_asymmetry', False) != False
        }
        # A more sophisticated approach would assign numerical scores or weights to these
        # For example, mapping 'extreme' competition_level to a score of 1.0, 'low' to 0.2
        return metrics

    def integration_hook_example(self, simulation_system_api: Any):
        """
        Example-only integration hook.

        This method demonstrates how scenario configs could hook into other simulation
        systems. It is DISABLED by default for safety and has no side effects.

        Enable explicitly by setting environment variable:
          SCENARIO_INTEGRATION_HOOK_ENABLED=true

        Behavior:
        - When disabled (default): log at DEBUG and return immediately (no-ops).
        - When enabled: perform a safe no-op and log that the demo hook executed.
        """
        enabled = os.environ.get("SCENARIO_INTEGRATION_HOOK_ENABLED", "").lower() in {"1", "true", "yes"}
        if not enabled:
            logging.debug("ScenarioConfigManager.integration_hook_example disabled by default; set SCENARIO_INTEGRATION_HOOK_ENABLED to enable demo mode.")
            return

        # Demo mode: do not mutate external state; just log once to show the path is reachable.
        logging.info("ScenarioConfigManager.integration_hook_example executed in demonstration mode (no-op). No external state was modified.")
        # Intentionally avoid calling any simulation_system_api mutators here.
        return

    def generate_dynamic_scenario(self, base_template_name: str, randomization_config: Dict[str, Any], target_tier: Optional[int] = None) -> ScenarioConfig:
        """
        Generates a dynamic scenario using the DynamicScenarioGenerator, optionally scaled to a tier.
        """
        template_filepath = self.get_scenario_path(base_template_name)
        if not template_filepath:
            raise ValueError(f"Base template '{base_template_name}' not found for dynamic generation.")
        
        template_obj = ScenarioConfig.from_yaml(template_filepath)
        
        generated_scenario = self.dynamic_generator.generate_scenario(base_template_name, randomization_config)

        if target_tier is not None:
            logging.info(f"Scaling dynamically generated scenario to Tier {target_tier}...")
            generated_scenario = self.dynamic_generator.scale_difficulty(generated_scenario, target_tier)
        
        logging.info(f"Dynamically generated scenario '{generated_scenario.config_data['scenario_name']}' for Tier {generated_scenario.config_data['difficulty_tier']}.")
        return generated_scenario
