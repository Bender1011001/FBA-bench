import yaml
from typing import Dict, Any, List, Optional

class ScenarioConfig:
    """
    Defines the configuration for a single FBA simulation scenario.
    Includes market conditions, business parameters, external events,
    and agent challenges.
    """
    def __init__(self, config_data: Dict[str, Any]):
        self.config_data = config_data
        self._validate_initial_config()

    def _validate_initial_config(self):
        """Basic validation for required keys in the config data."""
        required_keys = [
            'scenario_name', 'difficulty_tier', 'expected_duration',
            'success_criteria', 'market_conditions', 'external_events',
            'agent_constraints'
        ]
        for key in required_keys:
            if key not in self.config_data:
                raise ValueError(f"Missing required key in scenario config: {key}")

    def generate_market_conditions(self, scenario_type: str) -> Dict[str, Any]:
        """
        Creates market environment parameters based on scenario type.
        This method would typically use values from self.config_data['market_conditions'].
        For now, it's a placeholder.
        """
        market_params = self.config_data.get('market_conditions', {})
        # Example logic: customize based on scenario_type
        if scenario_type == "boom_and_bust":
            market_params['economic_cycle'] = 'recession_recovery'
            market_params['seasonality_pattern'] = 'fluctuating'
        elif scenario_type == "hyper_competitive":
            market_params['competition_level'] = 'intense'
        return market_params

    def define_product_catalog(self, category: str, complexity: str) -> List[Dict[str, Any]]:
        """
        Sets available products and variants based on category and complexity.
        This method would typically use values from self.config_data['products'].
        For now, it's a placeholder.
        """
        products = self.config_data.get('product_catalog', [])
        # Example logic: filter or expand based on inputs
        if complexity == "high_sku":
            products.extend([
                {"name": "ComplexGadgetA", "variants": 5, "base_price": 100},
                {"name": "ComplexGadgetB", "variants": 8, "base_price": 120}
            ])
        return products

    def schedule_external_events(self, timeline: int, event_types: List[str]) -> List[Dict[str, Any]]:
        """
        Plans scenario-specific disruptions across a simulation timeline.
        This method would typically use values from self.config_data['external_events'].
        For now, it's a placeholder.
        """
        events = self.config_data.get('external_events', [])
        # Example logic: filter or add events based on timeline/types
        if "supply_disruption" in event_types:
            events.append({"type": "supply_chain_shock", "tick": timeline // 2, "magnitude": "high"})
        return events

    def configure_agent_constraints(self, difficulty_tier: int) -> Dict[str, Any]:
        """
        Sets agent-specific limitations based on the difficulty tier.
        This method would typically use values from self.config_data['agent_constraints'].
        For now, it's a placeholder.
        """
        constraints = self.config_data.get('agent_constraints', {})
        # Example logic: adjust based on difficulty_tier
        if difficulty_tier == 0:
            constraints['initial_capital'] = 10000
            constraints['max_debt_ratio'] = 0.1
        elif difficulty_tier == 3:
            constraints['initial_capital'] = 5000
            constraints['max_debt_ratio'] = 0.5
            constraints['information_asymmetry'] = True
        return constraints

    def validate_scenario_consistency(self) -> bool:
        """
        Ensures scenario parameters are coherent and internally consistent.
        This is a critical method for robust scenario definition.
        """
        # Placeholder for comprehensive validation logic
        # - Check for logical conflicts (e.g., severe recession with high demand)
        # - Validate numerical ranges
        # - Ensure event timings are logical
        # - Check multi-agent configurations for completeness
        scenario_name = self.config_data.get('scenario_name')
        if not scenario_name:
            print("Validation Warning: Scenario name is missing.")
            return False

        tier = self.config_data.get('difficulty_tier')
        if not isinstance(tier, int) or not (0 <= tier <= 3):
            print(f"Validation Error: Invalid difficulty_tier: {tier}. Must be 0-3.")
            return False

        duration = self.config_data.get('expected_duration')
        if not isinstance(duration, int) or duration <= 0:
            print(f"Validation Error: Invalid expected_duration: {duration}. Must be positive integer.")
            return False
            
        success_criteria = self.config_data.get('success_criteria')
        if not isinstance(success_criteria, dict) or not success_criteria:
            print("Validation Error: Missing or invalid success_criteria.")
            return False

        print(f"Scenario '{scenario_name}' consistency validation passed (basic check).")
        return True

    @classmethod
    def from_yaml(cls, filepath: str):
        """Loads a ScenarioConfig from a YAML file."""
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(config_data)

    def to_yaml(self, filepath: str):
        """Saves the ScenarioConfig to a YAML file."""
        with open(filepath, 'w') as f:
            yaml.safe_dump(self.config_data, f, indent=2)
