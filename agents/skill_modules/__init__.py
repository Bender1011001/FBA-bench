"""
Skill Module Framework for FBA-Bench Multi-Domain Agent Capabilities.

This module provides a specialized skill-based architecture for agents to operate
across multiple business domains including supply management, marketing, customer service,
and financial analysis with event-driven coordination and LLM-driven decision making.

Exports:
    - BaseSkill: Abstract base class for all skill modules
    - SupplyManagerSkill: Inventory and supplier management capabilities
    - MarketingManagerSkill: Advertising and pricing strategy capabilities  
    - CustomerServiceSkill: Customer interaction and satisfaction capabilities
    - FinancialAnalystSkill: Budget management and financial planning capabilities
    - SkillCoordinator: Event-driven coordination system for skill modules
    - MultiDomainController: CEO-level coordination and resource allocation
    - SkillConfig: Configuration dataclass for skill management
"""

from .base_skill import BaseSkill
from .supply_manager import SupplyManagerSkill
from .marketing_manager import MarketingManagerSkill
from .customer_service import CustomerServiceSkill
from .financial_analyst import FinancialAnalystSkill

__all__ = [
    'BaseSkill',
    'SupplyManagerSkill', 
    'MarketingManagerSkill',
    'CustomerServiceSkill',
    'FinancialAnalystSkill'
]