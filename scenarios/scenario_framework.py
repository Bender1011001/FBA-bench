import yaml
import random
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
        Uses values from self.config_data['market_conditions'] and applies scenario-specific logic.
        """
        market_params = self.config_data.get('market_conditions', {}).copy()
        
        # Apply scenario-specific market condition logic
        if scenario_type == "boom_and_bust":
            market_params.update({
                'economic_cycle': 'recession_recovery',
                'seasonality_pattern': 'fluctuating',
                'market_volatility': 'high',
                'consumer_confidence': 'low_to_moderate',
                'interest_rates': 'variable',
                'inflation_rate': 'unstable'
            })
        elif scenario_type == "hyper_competitive":
            market_params.update({
                'competition_level': 'intense',
                'market_saturation': 'high',
                'price_sensitivity': 'high',
                'barrier_to_entry': 'low',
                'innovation_rate': 'high',
                'customer_loyalty': 'low'
            })
        elif scenario_type == "supply_chain_crisis":
            market_params.update({
                'supply_disruption': 'severe',
                'logistics_challenges': 'high',
                'inventory_shortages': 'critical',
                'supplier_reliability': 'low',
                'shipping_costs': 'elevated',
                'lead_times': 'extended'
            })
        elif scenario_type == "international_expansion":
            market_params.update({
                'market_diversity': 'high',
                'cultural_complexity': 'moderate',
                'regulatory_variations': 'significant',
                'currency_fluctuation': 'moderate',
                'trade_barriers': 'low_to_moderate',
                'local_competition': 'variable'
            })
        elif scenario_type == "multi_agent_ecosystem":
            market_params.update({
                'network_effects': 'strong',
                'platform_dynamics': 'active',
                'ecosystem_maturity': 'growing',
                'stakeholder_interdependence': 'high',
                'value_chain_complexity': 'high',
                'collaboration_opportunities': 'abundant'
            })
        else:
            # Default market conditions
            market_params.update({
                'economic_cycle': 'stable',
                'seasonality_pattern': 'predictable',
                'market_volatility': 'low',
                'consumer_confidence': 'stable',
                'competition_level': 'moderate'
            })
        
        # Validate and normalize market parameters
        market_params = self._validate_market_params(market_params)
        
        return market_params

    def define_product_catalog(self, category: str, complexity: str) -> List[Dict[str, Any]]:
        """
        Sets available products and variants based on category and complexity.
        Uses values from self.config_data['product_catalog'] and applies business logic.

        Robustness improvements:
        - Accepts category as either a string or a list[str]; when a list is provided,
          will attempt to use the product's own 'category' field when present, otherwise
          fall back to the first category in the list.
        - Handles complexity values safely, falling back to 'medium_sku' when unknown.
        """
        base_products = self.config_data.get('product_catalog', []).copy()

        # Normalize categories to a list[str]
        if isinstance(category, list):
            categories: List[str] = [c for c in category if isinstance(c, str) and c]
        elif isinstance(category, str) and category:
            categories = [category]
        else:
            categories = []

        # Define product templates for different categories
        product_templates = {
            "electronics": {
                "base_attributes": ["warranty", "technical_support", "software_updates"],
                "variants": ["storage_size", "color", "connectivity"],
                "pricing_factors": ["brand_premium", "feature_complexity", "market_demand"]
            },
            "apparel": {
                "base_attributes": ["material_quality", "design_complexity", "brand_reputation"],
                "variants": ["size", "color", "style"],
                "pricing_factors": ["seasonal_demand", "brand_value", "production_cost"]
            },
            "food_beverage": {
                "base_attributes": ["organic_certification", "shelf_life", "nutritional_value"],
                "variants": ["flavor", "packaging_size", "dietary_category"],
                "pricing_factors": ["ingredient_quality", "brand_recognition", "distribution_costs"]
            },
            "home_goods": {
                "base_attributes": ["durability", "aesthetic_design", "functionality"],
                "variants": ["size", "color", "material"],
                "pricing_factors": ["material_quality", "design_complexity", "brand_premium"]
            },
            "automotive": {
                "base_attributes": ["safety_rating", "fuel_efficiency", "performance_metrics"],
                "variants": ["model", "trim_level", "color"],
                "pricing_factors": ["engine_size", "luxury_features", "technology_package"]
            }
        }

        # Apply complexity-based modifications
        complexity_modifiers = {
            "low_sku": {
                "variant_multiplier": 1.2,
                "price_adjustment": 0.9,
                "inventory_depth": "high"
            },
            "medium_sku": {
                "variant_multiplier": 1.5,
                "price_adjustment": 1.0,
                "inventory_depth": "medium"
            },
            "high_sku": {
                "variant_multiplier": 2.0,
                "price_adjustment": 1.1,
                "inventory_depth": "low"
            }
        }

        # Normalize/validate complexity
        if complexity not in complexity_modifiers:
            # Best-effort default
            complexity = "medium_sku"
        modifier = complexity_modifiers[complexity]

        # Helper to pick a category for a given product
        def _select_category_for_product(prod: Dict[str, Any]) -> Optional[str]:
            # Product-level category may override scenario-level category
            prod_cat = prod.get("category")
            if isinstance(prod_cat, list):
                prod_cat = prod_cat[0] if prod_cat else None
            if isinstance(prod_cat, str) and prod_cat:
                return prod_cat
            # Fallback to first category from scenario if provided
            if categories:
                return categories[0]
            return None

        # Process products based on category and complexity
        processed_products: List[Dict[str, Any]] = []

        for product in base_products:
            processed_product = product.copy()
            selected_category = _select_category_for_product(processed_product)

            if selected_category and selected_category in product_templates:
                template = product_templates[selected_category]

                # Add base attributes if not present
                for attr in template["base_attributes"]:
                    if attr not in processed_product:
                        processed_product[attr] = self._generate_attribute_value(attr, selected_category)

                # Generate variants based on complexity
                base_variants = processed_product.get("variants", 1)
                try:
                    base_variants_int = int(base_variants) if not isinstance(base_variants, list) else max(1, len(base_variants))
                except (ValueError, TypeError):
                    base_variants_int = 1

                variant_count = max(1, int(round(base_variants_int * modifier["variant_multiplier"])))

                # Generate variant details
                variants = []
                for i in range(variant_count):
                    variant = {
                        "variant_id": f"{processed_product.get('name', 'product')}_{i+1}",
                        "attributes": self._generate_variant_attributes(template["variants"]),
                        "price_adjustment": modifier["price_adjustment"] * (0.9 + (i * 0.1))
                    }
                    variants.append(variant)

                processed_product["variants"] = variants
                processed_product["inventory_strategy"] = modifier["inventory_depth"]

            processed_products.append(processed_product)

        # If no base products, generate default products for the provided categories (or a sensible default)
        if not processed_products:
            defaults_built = False
            for cat in (categories or []):
                if cat in product_templates:
                    processed_products.extend(self._generate_default_products(cat, complexity))
                    defaults_built = True
            if not defaults_built:
                # Fallback to a stable default category
                processed_products = self._generate_default_products("electronics", complexity)

        # Validate and normalize product catalog
        processed_products = self._validate_product_catalog(processed_products)

        return processed_products

    def schedule_external_events(self, timeline: int, event_types: List[str]) -> List[Dict[str, Any]]:
        """
        Plans scenario-specific disruptions across a simulation timeline.
        Uses values from self.config_data['external_events'] and applies scheduling logic.
        """
        base_events = self.config_data.get('external_events', []).copy()
        scheduled_events = []
        
        # Define event templates for different event types
        event_templates = {
            "supply_disruption": {
                "types": ["supplier_bankruptcy", "logistics_failure", "quality_issue", "natural_disaster"],
                "impact_levels": ["low", "medium", "high", "critical"],
                "duration_range": (1, 5),  # in simulation ticks
                "probability_factors": ["supplier_concentration", "geographic_risk", "inventory_levels"]
            },
            "market_shift": {
                "types": ["demand_surge", "demand_decline", "price_war", "new_competitor"],
                "impact_levels": ["low", "medium", "high"],
                "duration_range": (3, 10),
                "probability_factors": ["market_saturation", "economic_conditions", "competitive_landscape"]
            },
            "financial_crisis": {
                "types": ["credit_crunch", "currency_fluctuation", "interest_rate_spike", "liquidity_crisis"],
                "impact_levels": ["medium", "high", "critical"],
                "duration_range": (5, 15),
                "probability_factors": ["debt_levels", "cash_reserves", "market_volatility"]
            },
            "regulatory_change": {
                "types": ["compliance_requirement", "trade_tariff", "environmental_regulation", "labor_law"],
                "impact_levels": ["low", "medium", "high"],
                "duration_range": (10, 30),
                "probability_factors": ["industry_regulation", "political_climate", "compliance_history"]
            },
            "technology_disruption": {
                "types": ["innovation_breakthrough", "system_failure", "cybersecurity_breach", "automation_threat"],
                "impact_levels": ["low", "medium", "high", "critical"],
                "duration_range": (2, 8),
                "probability_factors": ["technology_adoption", "it_investment", "digital_transformation"]
            }
        }
        
        # Process base events
        for event in base_events:
            scheduled_event = event.copy()
            
            # Ensure event has a scheduled tick
            if "tick" not in scheduled_event:
                scheduled_event["tick"] = self._calculate_event_timing(event, timeline)
            
            # Add event metadata
            scheduled_event["event_id"] = f"event_{len(scheduled_events) + 1}"
            scheduled_event["status"] = "scheduled"
            
            scheduled_events.append(scheduled_event)
        
        # Generate additional events based on requested event types
        for event_type in event_types:
            if event_type in event_templates:
                template = event_templates[event_type]
                
                # Determine number of events to generate for this type
                event_count = self._calculate_event_count(event_type, timeline)
                
                for i in range(event_count):
                    # Generate event based on template
                    new_event = self._generate_event_from_template(template, event_type, timeline)
                    
                    # Ensure no timing conflicts
                    if self._validate_event_timing(new_event, scheduled_events):
                        scheduled_events.append(new_event)
        
        # Sort events by tick
        scheduled_events.sort(key=lambda x: x.get("tick", 0))
        
        # Validate event schedule
        scheduled_events = self._validate_event_schedule(scheduled_events, timeline)
        
        return scheduled_events

    def configure_agent_constraints(self, difficulty_tier: int) -> Dict[str, Any]:
        """
        Sets agent-specific limitations based on the difficulty tier.
        Uses values from self.config_data['agent_constraints'] and applies tier-specific logic.
        """
        base_constraints = self.config_data.get('agent_constraints', {}).copy()
        
        # Define tier-specific constraint templates
        tier_templates = {
            0: {  # Tier 0 - Beginner
                "initial_capital": 15000,
                "max_debt_ratio": 0.1,
                "information_asymmetry": False,
                "market_transparency": "high",
                "decision_time_limit": None,
                "resource_constraints": "minimal",
                "risk_tolerance": "low",
                "learning_curve": "gradual",
                "feedback_frequency": "high",
                "error_tolerance": "high"
            },
            1: {  # Tier 1 - Intermediate
                "initial_capital": 10000,
                "max_debt_ratio": 0.25,
                "information_asymmetry": False,
                "market_transparency": "medium",
                "decision_time_limit": 30,  # seconds
                "resource_constraints": "moderate",
                "risk_tolerance": "medium",
                "learning_curve": "moderate",
                "feedback_frequency": "medium",
                "error_tolerance": "medium"
            },
            2: {  # Tier 2 - Advanced
                "initial_capital": 7500,
                "max_debt_ratio": 0.4,
                "information_asymmetry": True,
                "market_transparency": "low",
                "decision_time_limit": 15,  # seconds
                "resource_constraints": "significant",
                "risk_tolerance": "high",
                "learning_curve": "steep",
                "feedback_frequency": "low",
                "error_tolerance": "low"
            },
            3: {  # Tier 3 - Expert
                "initial_capital": 5000,
                "max_debt_ratio": 0.5,
                "information_asymmetry": True,
                "market_transparency": "very_low",
                "decision_time_limit": 5,  # seconds
                "resource_constraints": "severe",
                "risk_tolerance": "very_high",
                "learning_curve": "very_steep",
                "feedback_frequency": "very_low",
                "error_tolerance": "very_low"
            }
        }
        
        # Apply tier-specific constraints
        if difficulty_tier in tier_templates:
            tier_constraints = tier_templates[difficulty_tier]
            
            # Update base constraints with tier-specific values
            for key, value in tier_constraints.items():
                base_constraints[key] = value
            
            # Apply additional tier-specific logic
            base_constraints = self._apply_tier_specific_logic(base_constraints, difficulty_tier)
        
        # Validate constraints
        base_constraints = self._validate_agent_constraints(base_constraints, difficulty_tier)
        
        return base_constraints

    def validate_scenario_consistency(self) -> bool:
        """
        Ensures scenario parameters are coherent and internally consistent.
        Performs comprehensive validation of all scenario parameters.
        """
        scenario_name = self.config_data.get('scenario_name')
        if not scenario_name:
            print("Validation Warning: Scenario name is missing.")
            return False

        # Validate difficulty tier
        tier = self.config_data.get('difficulty_tier')
        if not isinstance(tier, int) or not (0 <= tier <= 3):
            print(f"Validation Error: Invalid difficulty_tier: {tier}. Must be 0-3.")
            return False

        # Validate duration
        duration = self.config_data.get('expected_duration')
        if not isinstance(duration, int) or duration <= 0:
            print(f"Validation Error: Invalid expected_duration: {duration}. Must be positive integer.")
            return False
            
        # Validate success criteria
        success_criteria = self.config_data.get('success_criteria')
        if not isinstance(success_criteria, dict) or not success_criteria:
            print("Validation Error: Missing or invalid success_criteria.")
            return False

        # Validate market conditions
        market_conditions = self.config_data.get('market_conditions', {})
        if not self._validate_market_conditions_consistency(market_conditions, tier):
            print(f"Validation Error: Market conditions are inconsistent with difficulty tier {tier}.")
            return False

        # Validate external events
        external_events = self.config_data.get('external_events', [])
        if not self._validate_external_events_consistency(external_events, duration):
            print("Validation Error: External events are inconsistent with scenario duration or timeline.")
            return False

        # Validate agent constraints
        agent_constraints = self.config_data.get('agent_constraints', {})
        if not self._validate_agent_constraints_consistency(agent_constraints, tier):
            print(f"Validation Error: Agent constraints are inconsistent with difficulty tier {tier}.")
            return False

        # Check for logical conflicts between scenario parameters
        if not self._check_logical_conflicts():
            print("Validation Error: Logical conflicts detected in scenario parameters.")
            return False

        # Validate numerical ranges
        if not self._validate_numerical_ranges():
            print("Validation Error: Invalid numerical ranges in scenario parameters.")
            return False

        # Validate multi-agent configurations if present
        if 'multi_agent_config' in self.config_data:
            if not self._validate_multi_agent_configuration(self.config_data['multi_agent_config']):
                print("Validation Error: Multi-agent configuration is incomplete or inconsistent.")
                return False

        print(f"Scenario '{scenario_name}' consistency validation passed (comprehensive check).")
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

    # Helper methods for scenario configuration

    def _validate_market_params(self, market_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and normalizes market parameters."""
        # Ensure required market parameters are present
        required_params = ['economic_cycle', 'market_volatility', 'consumer_confidence']
        for param in required_params:
            if param not in market_params:
                market_params[param] = 'stable'  # Default value
        
        # Normalize parameter values
        volatility_levels = ['low', 'medium', 'high', 'very_high']
        if market_params.get('market_volatility') not in volatility_levels:
            market_params['market_volatility'] = 'medium'
        
        confidence_levels = ['low', 'low_to_moderate', 'moderate', 'high']
        if market_params.get('consumer_confidence') not in confidence_levels:
            market_params['consumer_confidence'] = 'moderate'
        
        return market_params

    def _generate_attribute_value(self, attribute: str, category: str) -> Any:
        """Generates a realistic value for a product attribute based on category."""
        attribute_values = {
            "electronics": {
                "warranty": ["1 year", "2 years", "3 years", "extended"],
                "technical_support": ["24/7", "business hours", "email only", "premium"],
                "software_updates": ["2 years", "3 years", "5 years", "lifetime"]
            },
            "apparel": {
                "material_quality": ["basic", "standard", "premium", "luxury"],
                "design_complexity": ["simple", "moderate", "complex", "haute couture"],
                "brand_reputation": ["emerging", "established", "premium", "luxury"]
            },
            "food_beverage": {
                "organic_certification": ["none", "partial", "certified", "premium organic"],
                "shelf_life": ["short", "medium", "long", "extended"],
                "nutritional_value": ["basic", "standard", "enhanced", "premium"]
            },
            "home_goods": {
                "durability": ["basic", "standard", "heavy-duty", "commercial"],
                "aesthetic_design": ["functional", "modern", "designer", "luxury"],
                "functionality": ["basic", "standard", "multi-function", "smart"]
            },
            "automotive": {
                "safety_rating": ["3-star", "4-star", "5-star", "top safety pick+"],
                "fuel_efficiency": ["standard", "efficient", "hybrid", "electric"],
                "performance_metrics": ["economy", "standard", "sport", "high-performance"]
            }
        }
        
        if category in attribute_values and attribute in attribute_values[category]:
            return random.choice(attribute_values[category][attribute])
        
        return "standard"  # Default value

    def _generate_variant_attributes(self, variant_types: List[str]) -> Dict[str, Any]:
        """Generates attributes for product variants."""
        variant_attributes = {}
        
        for variant_type in variant_types:
            if variant_type == "storage_size":
                variant_attributes[variant_type] = random.choice(["64GB", "128GB", "256GB", "512GB", "1TB"])
            elif variant_type == "color":
                variant_attributes[variant_type] = random.choice(["black", "white", "silver", "blue", "red", "green"])
            elif variant_type == "size":
                variant_attributes[variant_type] = random.choice(["XS", "S", "M", "L", "XL", "XXL"])
            elif variant_type == "style":
                variant_attributes[variant_type] = random.choice(["casual", "formal", "sport", "classic"])
            elif variant_type == "connectivity":
                variant_attributes[variant_type] = random.choice(["WiFi", "WiFi+Cellular", "5G", "Bluetooth only"])
            elif variant_type == "model":
                variant_attributes[variant_type] = random.choice(["base", "sport", "luxury", "performance"])
            elif variant_type == "trim_level":
                variant_attributes[variant_type] = random.choice(["base", "mid", "high", "premium"])
            elif variant_type == "material":
                variant_attributes[variant_type] = random.choice(["plastic", "metal", "wood", "glass", "fabric"])
            elif variant_type == "flavor":
                variant_attributes[variant_type] = random.choice(["vanilla", "chocolate", "strawberry", "mint", "caramel"])
            elif variant_type == "packaging_size":
                variant_attributes[variant_type] = random.choice(["small", "medium", "large", "family", "bulk"])
            elif variant_type == "dietary_category":
                variant_attributes[variant_type] = random.choice(["regular", "low-fat", "sugar-free", "gluten-free", "vegan"])
            else:
                variant_attributes[variant_type] = "standard"
        
        return variant_attributes

    def _generate_default_products(self, category: str, complexity: str) -> List[Dict[str, Any]]:
        """Generates default products for a category when no base products are provided."""
        default_products = []
        
        # Define default product templates
        default_templates = {
            "electronics": [
                {"name": "Smartphone", "base_price": 699, "variants": 3},
                {"name": "Laptop", "base_price": 1299, "variants": 2},
                {"name": "Tablet", "base_price": 399, "variants": 2}
            ],
            "apparel": [
                {"name": "T-Shirt", "base_price": 29, "variants": 5},
                {"name": "Jeans", "base_price": 79, "variants": 4},
                {"name": "Jacket", "base_price": 149, "variants": 3}
            ],
            "food_beverage": [
                {"name": "Energy Bar", "base_price": 3, "variants": 4},
                {"name": "Protein Shake", "base_price": 5, "variants": 3},
                {"name": "Organic Tea", "base_price": 8, "variants": 3}
            ],
            "home_goods": [
                {"name": "Coffee Mug", "base_price": 15, "variants": 4},
                {"name": "Throw Pillow", "base_price": 25, "variants": 3},
                {"name": "Table Lamp", "base_price": 45, "variants": 2}
            ],
            "automotive": [
                {"name": "Sedan", "base_price": 25000, "variants": 3},
                {"name": "SUV", "base_price": 35000, "variants": 2},
                {"name": "Truck", "base_price": 40000, "variants": 2}
            ]
        }
        
        if category in default_templates:
            for template in default_templates[category]:
                product = template.copy()
                
                # Apply complexity-based pricing adjustments
                if complexity == "low_sku":
                    product["base_price"] *= 0.9
                elif complexity == "high_sku":
                    product["base_price"] *= 1.1
                
                default_products.append(product)
        
        return default_products

    def _validate_product_catalog(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validates and normalizes the product catalog."""
        validated_products = []
        
        for product in products:
            validated_product = product.copy()
            
            # Ensure required fields are present
            if "name" not in validated_product:
                validated_product["name"] = "Unnamed Product"
            
            if "base_price" not in validated_product:
                validated_product["base_price"] = 100  # Default price
            
            # Validate price is positive
            if validated_product["base_price"] <= 0:
                validated_product["base_price"] = 100
            
            # Ensure variants are properly structured
            if "variants" in validated_product and isinstance(validated_product["variants"], list):
                for i, variant in enumerate(validated_product["variants"]):
                    if isinstance(variant, dict) and "variant_id" not in variant:
                        variant["variant_id"] = f"{validated_product['name']}_variant_{i+1}"
            
            validated_products.append(validated_product)
        
        return validated_products

    def _calculate_event_timing(self, event: Dict[str, Any], timeline: int) -> int:
        """Calculates the optimal timing for an event within the timeline."""
        # If event already has a timing preference, use it
        if "timing_preference" in event:
            preference = event["timing_preference"]
            if preference == "early":
                return random.randint(1, timeline // 3)
            elif preference == "middle":
                return random.randint(timeline // 3, 2 * timeline // 3)
            elif preference == "late":
                return random.randint(2 * timeline // 3, timeline)
        
        # Otherwise, distribute evenly across timeline
        return random.randint(1, timeline)

    def _calculate_event_count(self, event_type: str, timeline: int) -> int:
        """Calculates the number of events of a given type to generate based on timeline."""
        # Base event counts per type
        base_counts = {
            "supply_disruption": 2,
            "market_shift": 3,
            "financial_crisis": 1,
            "regulatory_change": 1,
            "technology_disruption": 2
        }
        
        # Scale with timeline duration
        base_count = base_counts.get(event_type, 1)
        timeline_factor = max(1, timeline // 50)  # Adjust based on timeline length
        
        return min(base_count * timeline_factor, 10)  # Cap at 10 events per type

    def _generate_event_from_template(self, template: Dict[str, Any], event_type: str, timeline: int) -> Dict[str, Any]:
        """Generates a new event based on a template."""
        # Select random event type from template
        event_subtype = random.choice(template["types"])
        impact_level = random.choice(template["impact_levels"])
        duration = random.randint(template["duration_range"][0], template["duration_range"][1])
        
        # Calculate timing
        timing = random.randint(1, timeline)
        
        # Create event
        event = {
            "event_id": f"{event_type}_{len(template['types'])}",
            "type": event_subtype,
            "category": event_type,
            "impact_level": impact_level,
            "duration": duration,
            "tick": timing,
            "status": "scheduled",
            "probability_factors": template["probability_factors"]
        }
        
        return event

    def _validate_event_timing(self, new_event: Dict[str, Any], existing_events: List[Dict[str, Any]]) -> bool:
        """Validates that an event's timing doesn't conflict with existing events."""
        new_tick = new_event.get("tick", 0)
        new_duration = new_event.get("duration", 1)
        
        for existing_event in existing_events:
            existing_tick = existing_event.get("tick", 0)
            existing_duration = existing_event.get("duration", 1)
            
            # Check for overlap
            if (new_tick >= existing_tick and new_tick <= existing_tick + existing_duration) or \
               (existing_tick >= new_tick and existing_tick <= new_tick + new_duration):
                return False
        
        return True

    def _validate_event_schedule(self, events: List[Dict[str, Any]], timeline: int) -> List[Dict[str, Any]]:
        """Validates and normalizes the event schedule."""
        validated_events = []
        
        for event in events:
            validated_event = event.copy()
            
            # Ensure required fields are present
            if "event_id" not in validated_event:
                validated_event["event_id"] = f"event_{len(validated_events) + 1}"
            
            if "type" not in validated_event:
                validated_event["type"] = "generic_event"
            
            if "tick" not in validated_event:
                validated_event["tick"] = 1
            
            # Validate timing is within timeline
            if validated_event["tick"] < 1:
                validated_event["tick"] = 1
            elif validated_event["tick"] > timeline:
                validated_event["tick"] = timeline
            
            # Ensure duration is positive
            if "duration" not in validated_event or validated_event["duration"] <= 0:
                validated_event["duration"] = 1
            
            validated_events.append(validated_event)
        
        return validated_events

    def _apply_tier_specific_logic(self, constraints: Dict[str, Any], difficulty_tier: int) -> Dict[str, Any]:
        """Applies tier-specific logic to agent constraints."""
        # Adjust constraints based on tier-specific rules
        if difficulty_tier == 0:  # Beginner
            # More forgiving constraints
            constraints["error_penalty_multiplier"] = 0.5
            constraints["learning_acceleration"] = 1.5
            constraints["hint_frequency"] = "high"
        elif difficulty_tier == 1:  # Intermediate
            constraints["error_penalty_multiplier"] = 1.0
            constraints["learning_acceleration"] = 1.0
            constraints["hint_frequency"] = "medium"
        elif difficulty_tier == 2:  # Advanced
            constraints["error_penalty_multiplier"] = 1.5
            constraints["learning_acceleration"] = 0.8
            constraints["hint_frequency"] = "low"
        elif difficulty_tier == 3:  # Expert
            constraints["error_penalty_multiplier"] = 2.0
            constraints["learning_acceleration"] = 0.5
            constraints["hint_frequency"] = "minimal"
        
        return constraints

    def _validate_agent_constraints(self, constraints: Dict[str, Any], difficulty_tier: int) -> Dict[str, Any]:
        """Validates and normalizes agent constraints."""
        validated_constraints = constraints.copy()
        
        # Ensure required constraints are present
        required_constraints = ["initial_capital", "max_debt_ratio"]
        for constraint in required_constraints:
            if constraint not in validated_constraints:
                # Set default based on tier
                if constraint == "initial_capital":
                    validated_constraints[constraint] = 10000 - (difficulty_tier * 2500)
                elif constraint == "max_debt_ratio":
                    validated_constraints[constraint] = 0.1 + (difficulty_tier * 0.15)
        
        # Validate numerical ranges
        if validated_constraints["initial_capital"] <= 0:
            validated_constraints["initial_capital"] = 5000
        
        if not (0 <= validated_constraints["max_debt_ratio"] <= 1):
            validated_constraints["max_debt_ratio"] = 0.5
        
        return validated_constraints

    def _validate_market_conditions_consistency(self, market_conditions: Dict[str, Any], difficulty_tier: int) -> bool:
        """Validates that market conditions are consistent with the difficulty tier."""
        # Check for logical consistency between market conditions and difficulty
        high_volatility_tiers = [2, 3]
        low_transparency_tiers = [2, 3]
        
        # High difficulty should correlate with challenging market conditions
        if difficulty_tier in high_volatility_tiers:
            if market_conditions.get('market_volatility') in ['low', 'very_low']:
                return False
        
        if difficulty_tier in low_transparency_tiers:
            if market_conditions.get('market_transparency') in ['high', 'very_high']:
                return False
        
        return True

    def _validate_external_events_consistency(self, external_events: List[Dict[str, Any]], duration: int) -> bool:
        """Validates that external events are consistent with the scenario duration."""
        for event in external_events:
            # Check that event timing is within scenario duration
            if "tick" in event and event["tick"] > duration:
                return False
            
            # Check that event duration doesn't exceed scenario duration
            if "duration" in event and event["duration"] > duration:
                return False
        
        return True

    def _validate_agent_constraints_consistency(self, agent_constraints: Dict[str, Any], difficulty_tier: int) -> bool:
        """Validates that agent constraints are consistent with the difficulty tier."""
        # Check that constraints align with difficulty tier
        if difficulty_tier == 0:  # Beginner
            if agent_constraints.get('initial_capital', 0) < 10000:
                return False
            if agent_constraints.get('max_debt_ratio', 0) > 0.2:
                return False
        elif difficulty_tier == 3:  # Expert
            if agent_constraints.get('initial_capital', 0) > 7500:
                return False
            if agent_constraints.get('max_debt_ratio', 0) < 0.4:
                return False
        
        return True

    def _check_logical_conflicts(self) -> bool:
        """Checks for logical conflicts between scenario parameters."""
        market_conditions = self.config_data.get('market_conditions', {})
        external_events = self.config_data.get('external_events', [])
        
        # Check for conflicting market conditions
        if market_conditions.get('economic_cycle') == 'recession':
            if market_conditions.get('consumer_confidence') in ['high', 'very_high']:
                return False
        
        # Check for conflicting events
        recession_events = [e for e in external_events if e.get('type') in ['economic_crisis', 'recession']]
        boom_events = [e for e in external_events if e.get('type') in ['economic_boom', 'growth_surge']]
        
        if recession_events and boom_events:
            return False
        
        return True

    def _validate_numerical_ranges(self) -> bool:
        """Validates that all numerical parameters are within acceptable ranges."""
        # Validate market condition numerical values
        market_conditions = self.config_data.get('market_conditions', {})
        
        for key, value in market_conditions.items():
            if isinstance(value, (int, float)):
                if value < 0:
                    return False
                if key in ['interest_rates', 'inflation_rate'] and value > 1.0:
                    return False
        
        # Validate agent constraint numerical values
        agent_constraints = self.config_data.get('agent_constraints', {})
        
        for key, value in agent_constraints.items():
            if isinstance(value, (int, float)):
                if value < 0:
                    return False
                if key == 'max_debt_ratio' and value > 1.0:
                    return False
        
        return True

    def _validate_multi_agent_configuration(self, multi_agent_config: Dict[str, Any]) -> bool:
        """Validates multi-agent configuration for completeness and consistency."""
        # Check required fields
        required_fields = ['agent_count', 'agent_roles', 'interaction_rules']
        for field in required_fields:
            if field not in multi_agent_config:
                return False
        
        # Validate agent count
        agent_count = multi_agent_config['agent_count']
        if not isinstance(agent_count, int) or agent_count <= 0:
            return False
        
        # Validate agent roles
        agent_roles = multi_agent_config['agent_roles']
        if not isinstance(agent_roles, list) or len(agent_roles) != agent_count:
            return False
        
        # Validate interaction rules
        interaction_rules = multi_agent_config['interaction_rules']
        if not isinstance(interaction_rules, dict) or not interaction_rules:
            return False
        
        return True
