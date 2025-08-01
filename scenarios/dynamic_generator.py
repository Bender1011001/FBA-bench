import random
import yaml
from typing import Dict, Any, List, Optional
from scenarios.scenario_framework import ScenarioConfig

class DynamicScenarioGenerator:
    """
    Generates unique FBA simulation scenarios procedurally based on templates,
    randomized parameters, and difficulty scaling.
    """
    def __init__(self, template_dir: str = 'scenarios/business_types/'):
        self.template_dir = template_dir

    def _load_template(self, base_template_name: str) -> Dict[str, Any]:
        """Loads a base scenario template from a YAML file."""
        filepath = f"{self.template_dir}{base_template_name}.yaml"
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Scenario template '{filepath}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML template '{filepath}': {e}")

    def generate_scenario(self, base_template_name: str, randomization_config: Dict[str, Any]) -> ScenarioConfig:
        """
        Creates a unique scenario by loading a base template and applying randomization.
        """
        template_data = self._load_template(base_template_name)
        generated_data = self._apply_randomization(template_data, randomization_config)
        generated_data['scenario_name'] = f"{generated_data.get('scenario_name', 'DynamicScenario')}_{random.randint(1000, 9999)}"
        
        scenario = ScenarioConfig(generated_data)
        if not scenario.validate_scenario_consistency():
            raise ValueError("Generated scenario failed consistency validation.")
        
        return scenario

    def _apply_randomization(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies randomization to scenario parameters based on the provided config.
        This is a recursive method to handle nested dictionaries.
        """
        for key, value in config.items():
            if key in data:
                if isinstance(value, dict) and 'range' in value:
                    min_val, max_val = value['range']
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            data[key] = random.randint(min_val, max_val)
                        else:
                            data[key] = random.uniform(min_val, max_val)
                elif isinstance(value, dict):
                    data[key] = self._apply_randomization(data[key], value)
                elif isinstance(value, list) and 'choices' in value:
                    data[key] = random.choice(value['choices'])
                elif isinstance(value, bool):
                    # For boolean flags like enabling/disabling a feature
                    if random.random() < 0.5: # 50% chance to flip
                        data[key] = not data[key]
                # Add more randomization types as needed (e.g., specific distributions)
        return data

    def randomize_parameters(self, parameter_set: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Varies key parameters within realistic ranges specified by constraints.
        This is a more generic method that can be used independently.
        """
        randomized_params = {}
        for param, config in parameter_set.items():
            if param in constraints:
                constraint_info = constraints[param]
                if 'range' in constraint_info:
                    min_val, max_val = constraint_info['range']
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        randomized_params[param] = random.randint(min_val, max_val)
                    else:
                        randomized_params[param] = random.uniform(min_val, max_val)
                # Add other constraint types (e.g., choices, distributions)
            else:
                randomized_params[param] = config # Keep original if no constraint
        return randomized_params

    def schedule_random_events(self, event_pool: List[Dict[str, Any]], timeline: int) -> List[Dict[str, Any]]:
        """
        Randomly distributes a subset of challenges from an event pool across a timeline.
        """
        num_events = random.randint(1, min(len(event_pool), 5)) # Schedule 1 to 5 events
        selected_events = random.sample(event_pool, num_events)
        
        scheduled_events = []
        for event in selected_events:
            event_copy = event.copy()
            event_copy['tick'] = random.randint(1, timeline) # Random tick within simulation
            scheduled_events.append(event_copy)
        
        return scheduled_events

    def scale_difficulty(self, base_scenario_config: ScenarioConfig, target_tier: int) -> ScenarioConfig:
        """
        Adjusts complexity based on target tier requirements.
        This modifies an existing ScenarioConfig instance.
        """
        # Load the base scenario's data
        scenario_data = base_scenario_config.config_data.copy()

        # Adjust general parameters
        scenario_data['difficulty_tier'] = target_tier
        if target_tier == 0:
            scenario_data['market_conditions']['competition_levels'] = 'low'
            scenario_data['agent_constraints']['initial_capital'] = scenario_data['agent_constraints'].get('initial_capital', 10000) * 2
            scenario_data['external_events'] = [] # No disruptions
        elif target_tier == 1:
            scenario_data['market_conditions']['competition_levels'] = 'moderate'
            scenario_data['agent_constraints']['max_debt_ratio'] = min(0.4, scenario_data['agent_constraints'].get('max_debt_ratio', 0.5))
            # Keep only 1-2 minor events
            scenario_data['external_events'] = self.schedule_random_events(base_scenario_config.config_data.get('external_events', []), base_scenario_config.config_data.get('expected_duration', 1))[:2]
        elif target_tier == 2:
            scenario_data['market_conditions']['competition_levels'] = 'high'
            scenario_data['agent_constraints']['max_debt_ratio'] = min(0.6, scenario_data['agent_constraints'].get('max_debt_ratio', 0.5))
            # Keep 2-3 moderate events
            scenario_data['external_events'] = self.schedule_random_events(base_scenario_config.config_data.get('external_events', []), base_scenario_config.config_data.get('expected_duration', 1))[:3]
        elif target_tier == 3:
            scenario_data['market_conditions']['competition_levels'] = 'extreme'
            scenario_data['agent_constraints']['initial_capital'] = scenario_data['agent_constraints'].get('initial_capital', 10000) * 0.5 # Less capital
            scenario_data['agent_constraints']['information_asymmetry'] = True
            # Introduce more frequent/severe events
            scenario_data['external_events'] = self.schedule_random_events(base_scenario_config.config_data.get('external_events', []), base_scenario_config.config_data.get('expected_duration', 1))
            random.shuffle(scenario_data['external_events']) # Randomize order

        # Ensure multi_agent_config is consistent with tier expectations for complexity
        if 'multi_agent_config' in scenario_data:
            if target_tier <= 1:
                scenario_data['multi_agent_config']['num_agents'] = min(scenario_data['multi_agent_config'].get('num_agents', 1), 2)
                scenario_data['multi_agent_config']['interaction_mode'] = 'cooperative' if target_tier == 0 else 'simple_competitive'
            elif target_tier >= 2:
                 scenario_data['multi_agent_config']['num_agents'] = max(scenario_data['multi_agent_config'].get('num_agents', 1), 3)
                 scenario_data['multi_agent_config']['interaction_mode'] = 'complex_ecosystem'

        scaled_scenario = ScenarioConfig(scenario_data)
        if not scaled_scenario.validate_scenario_consistency():
            print(f"Warning: Scaled scenario for Tier {target_tier} failed consistency validation. Check logic.")
        return scaled_scenario

    def ensure_scenario_validity(self, generated_scenario: ScenarioConfig) -> bool:
        """
        Performs a final validation check on a generated scenario to ensure realism and internal consistency.
        Leverages the ScenarioConfig's own validation but can add more specific generator-level checks.
        """
        print(f"Performing final validity check for generated scenario: {generated_scenario.config_data.get('scenario_name')}")
        is_consistent = generated_scenario.validate_scenario_consistency()
        # Add additional checks specific to procedural generation
        # E.g., ensure that randomized events don't negate each other in a specific tick range
        # Ensure product catalog is not empty after randomization if it shouldn't be
        if is_consistent:
            print("Generated scenario passed all validity checks.")
        else:
            print("Generated scenario failed validity checks.")
        return is_consistent
