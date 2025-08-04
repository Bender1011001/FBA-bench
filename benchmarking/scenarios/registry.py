"""
Scenario registry for managing available scenarios.

This module provides a centralized registry for all available scenarios,
allowing for dynamic registration, discovery, and instantiation of scenarios.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass

from .base import BaseScenario, ScenarioConfig
from .templates import (
    ECommerceScenario, 
    HealthcareScenario, 
    FinancialScenario, 
    LegalScenario, 
    ScientificScenario
)

logger = logging.getLogger(__name__)


@dataclass
class ScenarioRegistration:
    """Information about a registered scenario."""
    name: str
    description: str
    domain: str
    scenario_class: Type[BaseScenario]
    default_config: ScenarioConfig
    tags: List[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ScenarioRegistry:
    """
    Registry for managing available scenarios.
    
    This class provides a centralized way to register, discover, and instantiate
    scenarios. It supports dynamic registration and categorization of scenarios.
    """
    
    def __init__(self):
        """Initialize the scenario registry."""
        self._scenarios: Dict[str, ScenarioRegistration] = {}
        self._domains: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}
        
        # Register built-in scenarios
        self._register_builtin_scenarios()
    
    def _register_builtin_scenarios(self) -> None:
        """Register all built-in scenarios."""
        # E-commerce scenario
        self.register_scenario(
            name="ecommerce",
            description="E-commerce scenario for online retail benchmarking",
            domain="ecommerce",
            scenario_class=ECommerceScenario,
            default_config=ScenarioConfig(
                name="ecommerce",
                description="E-commerce scenario for online retail benchmarking",
                domain="ecommerce",
                duration_ticks=100,
                parameters={
                    "product_count": 10,
                    "customer_count": 100,
                    "initial_budget": 10000
                },
                difficulty="medium"
            ),
            tags=["ecommerce", "retail", "pricing", "inventory", "marketing"]
        )
        
        # Healthcare scenario
        self.register_scenario(
            name="healthcare",
            description="Healthcare scenario for medical diagnostics benchmarking",
            domain="healthcare",
            scenario_class=HealthcareScenario,
            default_config=ScenarioConfig(
                name="healthcare",
                description="Healthcare scenario for medical diagnostics benchmarking",
                domain="healthcare",
                duration_ticks=150,
                parameters={
                    "patient_count": 50,
                    "medical_staff_count": 10
                },
                difficulty="medium"
            ),
            tags=["healthcare", "medical", "diagnostics", "treatment", "patients"]
        )
        
        # Financial scenario
        self.register_scenario(
            name="financial",
            description="Financial scenario for investment analysis benchmarking",
            domain="financial",
            scenario_class=FinancialScenario,
            default_config=ScenarioConfig(
                name="financial",
                description="Financial scenario for investment analysis benchmarking",
                domain="financial",
                duration_ticks=200,
                parameters={
                    "initial_capital": 100000,
                    "instrument_count": 20
                },
                difficulty="hard"
            ),
            tags=["financial", "investment", "trading", "portfolio", "market"]
        )
        
        # Legal scenario
        self.register_scenario(
            name="legal",
            description="Legal scenario for document review benchmarking",
            domain="legal",
            scenario_class=LegalScenario,
            default_config=ScenarioConfig(
                name="legal",
                description="Legal scenario for document review benchmarking",
                domain="legal",
                duration_ticks=120,
                parameters={
                    "document_count": 100,
                    "case_complexity": "medium"
                },
                difficulty="medium"
            ),
            tags=["legal", "document", "review", "compliance", "regulation"]
        )
        
        # Scientific scenario
        self.register_scenario(
            name="scientific",
            description="Scientific scenario for research benchmarking",
            domain="scientific",
            scenario_class=ScientificScenario,
            default_config=ScenarioConfig(
                name="scientific",
                description="Scientific scenario for research benchmarking",
                domain="scientific",
                duration_ticks=180,
                parameters={
                    "dataset_count": 20,
                    "research_field": "general"
                },
                difficulty="hard"
            ),
            tags=["scientific", "research", "data", "hypothesis", "experiment"]
        )
    
    def register_scenario(
        self,
        name: str,
        description: str,
        domain: str,
        scenario_class: Type[BaseScenario],
        default_config: ScenarioConfig,
        tags: List[str] = None,
        enabled: bool = True
    ) -> None:
        """
        Register a new scenario.
        
        Args:
            name: Unique name for the scenario
            description: Description of the scenario
            domain: Domain of the scenario
            scenario_class: Class implementing the scenario
            default_config: Default configuration for the scenario
            tags: List of tags for categorization
            enabled: Whether the scenario is enabled by default
        """
        if name in self._scenarios:
            logger.warning(f"Scenario '{name}' already registered, overwriting")
        
        registration = ScenarioRegistration(
            name=name,
            description=description,
            domain=domain,
            scenario_class=scenario_class,
            default_config=default_config,
            tags=tags or [],
            enabled=enabled
        )
        
        self._scenarios[name] = registration
        
        # Update domain index
        if domain not in self._domains:
            self._domains[domain] = []
        if name not in self._domains[domain]:
            self._domains[domain].append(name)
        
        # Update tag index
        for tag in registration.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            if name not in self._tags[tag]:
                self._tags[tag].append(name)
        
        logger.info(f"Registered scenario: {name} ({domain})")
    
    def get_scenario(self, name: str) -> Optional[ScenarioRegistration]:
        """
        Get a scenario registration by name.
        
        Args:
            name: Name of the scenario
            
        Returns:
            Scenario registration or None if not found
        """
        return self._scenarios.get(name)
    
    def list_scenarios(self, domain: Optional[str] = None, enabled_only: bool = True) -> List[str]:
        """
        List available scenarios.
        
        Args:
            domain: Filter by domain (optional)
            enabled_only: Only return enabled scenarios
            
        Returns:
            List of scenario names
        """
        scenarios = self._scenarios
        
        # Filter by domain
        if domain is not None:
            if domain in self._domains:
                scenario_names = self._domains[domain]
                scenarios = {name: self._scenarios[name] for name in scenario_names}
            else:
                return []
        
        # Filter by enabled status
        if enabled_only:
            scenarios = {name: reg for name, reg in scenarios.items() if reg.enabled}
        
        return list(scenarios.keys())
    
    def get_scenarios_by_domain(self) -> Dict[str, List[str]]:
        """
        Get scenarios grouped by domain.
        
        Returns:
            Dictionary mapping domains to scenario names
        """
        return {domain: names.copy() for domain, names in self._domains.items()}
    
    def get_scenarios_by_tag(self, tag: str) -> List[str]:
        """
        Get scenarios by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of scenario names with the specified tag
        """
        return self._tags.get(tag, []).copy()
    
    def create_scenario(
        self, 
        name: str, 
        config: Optional[ScenarioConfig] = None
    ) -> Optional[BaseScenario]:
        """
        Create a scenario instance.
        
        Args:
            name: Name of the scenario
            config: Scenario configuration (optional)
            
        Returns:
            Scenario instance or None if not found
        """
        registration = self.get_scenario(name)
        if registration is None:
            logger.error(f"Scenario '{name}' not found")
            return None
        
        if not registration.enabled:
            logger.warning(f"Scenario '{name}' is disabled")
            return None
        
        # Use provided config or default
        scenario_config = config or registration.default_config
        
        try:
            scenario = registration.scenario_class(scenario_config)
            logger.info(f"Created scenario instance: {name}")
            return scenario
        except Exception as e:
            logger.error(f"Failed to create scenario '{name}': {e}")
            return None
    
    def enable_scenario(self, name: str) -> bool:
        """
        Enable a scenario.
        
        Args:
            name: Name of the scenario
            
        Returns:
            True if successful, False if scenario not found
        """
        if name not in self._scenarios:
            return False
        
        self._scenarios[name].enabled = True
        logger.info(f"Enabled scenario: {name}")
        return True
    
    def disable_scenario(self, name: str) -> bool:
        """
        Disable a scenario.
        
        Args:
            name: Name of the scenario
            
        Returns:
            True if successful, False if scenario not found
        """
        if name not in self._scenarios:
            return False
        
        self._scenarios[name].enabled = False
        logger.info(f"Disabled scenario: {name}")
        return True
    
    def get_scenario_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a scenario.
        
        Args:
            name: Name of the scenario
            
        Returns:
            Dictionary with scenario information
        """
        registration = self.get_scenario(name)
        if registration is None:
            return {"error": f"Scenario '{name}' not found"}
        
        return {
            "name": registration.name,
            "description": registration.description,
            "domain": registration.domain,
            "class": registration.scenario_class.__name__,
            "module": registration.scenario_class.__module__,
            "default_config": {
                "name": registration.default_config.name,
                "description": registration.default_config.description,
                "domain": registration.default_config.domain,
                "duration_ticks": registration.default_config.duration_ticks,
                "parameters": registration.default_config.parameters,
                "difficulty": registration.default_config.difficulty,
                "enabled": registration.default_config.enabled
            },
            "tags": registration.tags,
            "enabled": registration.enabled
        }
    
    def create_scenario_suite(
        self, 
        scenario_names: List[str], 
        configs: Dict[str, ScenarioConfig] = None
    ) -> Dict[str, BaseScenario]:
        """
        Create a suite of scenarios.
        
        Args:
            scenario_names: List of scenario names to include
            configs: Custom configurations for scenarios (optional)
            
        Returns:
            Dictionary of scenario instances
        """
        suite = {}
        configs = configs or {}
        
        for name in scenario_names:
            config = configs.get(name)
            scenario = self.create_scenario(name, config)
            if scenario is not None:
                suite[name] = scenario
        
        return suite
    
    def validate_scenario_config(self, name: str, config: ScenarioConfig) -> List[str]:
        """
        Validate a scenario configuration.
        
        Args:
            name: Name of the scenario
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        registration = self.get_scenario(name)
        if registration is None:
            errors.append(f"Unknown scenario: {name}")
            return errors
        
        # Validate required fields
        if not config.name:
            errors.append("Scenario name cannot be empty")
        
        if not config.description:
            errors.append("Scenario description cannot be empty")
        
        if not config.domain:
            errors.append("Scenario domain cannot be empty")
        
        if config.duration_ticks <= 0:
            errors.append("Duration ticks must be positive")
        
        # Validate difficulty
        if config.difficulty not in ["easy", "medium", "hard"]:
            errors.append("Difficulty must be one of: easy, medium, hard")
        
        # Create a temporary scenario instance to validate parameters
        try:
            temp_scenario = registration.scenario_class(config)
            validation_errors = temp_scenario.validate()
            errors.extend(validation_errors)
        except Exception as e:
            errors.append(f"Failed to validate scenario: {e}")
        
        return errors
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry.
        
        Returns:
            Dictionary with registry summary
        """
        enabled_count = sum(1 for r in self._scenarios.values() if r.enabled)
        disabled_count = len(self._scenarios) - enabled_count
        
        return {
            "total_scenarios": len(self._scenarios),
            "enabled_scenarios": enabled_count,
            "disabled_scenarios": disabled_count,
            "domains": {
                domain: len(scenarios)
                for domain, scenarios in self._domains.items()
            },
            "tags": {
                tag: len(scenarios)
                for tag, scenarios in self._tags.items()
            }
        }


# Global registry instance
scenario_registry = ScenarioRegistry()