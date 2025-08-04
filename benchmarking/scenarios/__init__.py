"""
Extensible scenario framework for FBA-Bench.

This package provides a framework for creating and managing benchmarking scenarios
across different domains including e-commerce, healthcare, finance, legal, and
scientific research. The framework is designed to be extensible and supports
scenario validation and metadata management.
"""

from .base import BaseScenario, ScenarioConfig, ScenarioResult
from .templates import (
    ECommerceScenario, 
    HealthcareScenario, 
    FinancialScenario, 
    LegalScenario, 
    ScientificScenario
)
from .registry import ScenarioRegistry, registry as scenario_registry

__all__ = [
    "BaseScenario",
    "ScenarioConfig",
    "ScenarioResult",
    "ECommerceScenario",
    "HealthcareScenario",
    "FinancialScenario",
    "LegalScenario",
    "ScientificScenario",
    "ScenarioRegistry",
    "scenario_registry"
]