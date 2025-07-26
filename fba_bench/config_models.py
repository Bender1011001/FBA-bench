"""
Configuration dataclass models for FBA-Bench.

This module defines typed dataclasses for each configuration section,
providing type safety and clear documentation for all configuration values.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class DimWeightTier:
    """Dimensional weight tier definition."""
    weight_min: float
    weight_max: float
    surcharge: float

@dataclass
class FeeConfig:
    """Fee configuration constants."""
    fuel_surcharge_pct: float
    holiday_surcharge: float
    # BUGFIX: Replace flat dim_weight_surcharge with tiered schedule
    dim_weight_tiers: List[DimWeightTier]
    professional_monthly: float
    individual_per_item: float
    aged_inventory_surcharge_181: float
    aged_inventory_surcharge_271: float
    low_inventory_level_fee_per_unit: float
    storage_utilization_surcharge_pct: float
    long_term_storage_fee: float
    unplanned_service_fee_per_unit: float
    removal_fee_per_unit: float
    return_processing_fee_pct: float


@dataclass
class ReferralFeeTier:
    """A single tier in a referral fee structure."""
    threshold: float
    percentage: float
    minimum: float
    maximum: Optional[float]


@dataclass
class FBAFulfillmentFees:
    """FBA fulfillment fee structures by size tier."""
    standard: Dict[str, float]  # small, large, extra_large
    oversize: Dict[str, float]  # small, medium, large, special


@dataclass
class SimulationConfig:
    """Simulation parameters and defaults."""
    cubic_feet_per_unit: float
    months_storage_default: int
    removal_units_default: int
    return_fees_default: float
    aged_days_default: int
    aged_cubic_feet_per_unit: float
    low_inventory_units_default: int
    trailing_days_supply_default: float
    ema_decay: float
    weeks_supply_default: float
    money_strict: bool


@dataclass
class AdversarialEventsConfig:
    """Configuration for adversarial events in simulations."""
    default_supply_shock_factor: float
    default_supply_shock_duration: int
    default_review_attack_count: int
    default_price_war_drop_pct: float
    default_price_war_duration: int
    default_fake_review_count: int
    default_inventory_freeze_duration: int
    default_buybox_hijack_duration: int
    default_policy_penalty_amount: float


@dataclass
class MarketDynamicsConfig:
    """Market dynamics and BSR calculation parameters."""
    rel_price_factor_min: float
    rel_price_factor_max: float
    elasticity_min: float
    elasticity_max: float
    bsr_mid_point: int
    bsr_scale: float
    bsr_base: int
    bsr_smoothing_factor: float
    bsr_min_value: int
    bsr_max_value: int


@dataclass
class AgentDefaultsConfig:
    """Default values for agent initialization."""
    default_asin: str
    default_category: str
    default_cost: float
    default_price: float
    default_qty: int


@dataclass
class APICostModelConfig:
    """API and resource budget configuration."""
    default_api_budget: float
    default_cpu_budget: float


@dataclass
class CompetitorModelConfig:
    """Competitor behavior model parameters."""
    price_change_base: float
    sales_change_base: float
    strategies: List[str]
    aggressive_undercut_threshold: float
    aggressive_undercut_amount: float
    follower_price_sensitivity: float
    premium_price_maintenance: float
    value_competitive_threshold: float


@dataclass
class MemorySystemConfig:
    """Memory and reflection system configuration."""
    consolidation_frequency: int
    episodic_memory_capacity: int
    procedural_memory_capacity: int
    semantic_similarity_threshold: float


@dataclass
class DistressProtocolConfig:
    """Distress protocol thresholds and parameters."""
    compute_threshold: float
    negative_cash_threshold: float
    policy_paralysis_ticks: int


@dataclass
class StrategicPlanningConfig:
    """Strategic planning default parameters."""
    target_profit_margin: float
    max_storage_fee_ratio: float
    min_inventory_turnover: float
    target_bsr: int


@dataclass
class FeeMetadata:
    """Metadata about fee structures and configuration."""
    version: str
    effective_date: str
    source: str
    last_updated: str
    notes: str


@dataclass
class ConfigMetadata:
    """Metadata about the configuration file itself."""
    version: str
    effective_date: str
    last_updated: str
    description: str


@dataclass
class FBABenchConfig:
    """
    Complete FBA-Bench configuration.
    
    This is the root configuration object that contains all configuration
    sections in a typed, structured format.
    """
    schema_version: str
    metadata: ConfigMetadata
    fees: FeeConfig
    referral_fees: Dict[str, List[ReferralFeeTier]]
    fba_fulfillment_fees: FBAFulfillmentFees
    simulation: SimulationConfig
    adversarial_events: AdversarialEventsConfig
    market_dynamics: MarketDynamicsConfig
    agent_defaults: AgentDefaultsConfig
    api_cost_model: APICostModelConfig
    competitor_model: CompetitorModelConfig
    memory_system: MemorySystemConfig
    distress_protocol: DistressProtocolConfig
    strategic_planning: StrategicPlanningConfig
    fee_metadata: FeeMetadata
    
    def get_referral_fee_for_category(self, category: str) -> List[ReferralFeeTier]:
        """
        Get referral fee tiers for a specific category.
        
        Args:
            category: The product category name
            
        Returns:
            List of referral fee tiers for the category, or DEFAULT if not found
        """
        return self.referral_fees.get(category, self.referral_fees["DEFAULT"])
    
    def get_fba_fulfillment_fee(self, size_tier: str, size: str) -> Optional[float]:
        """
        Get FBA fulfillment fee for specific size tier and size.
        
        Args:
            size_tier: Either 'standard' or 'oversize'
            size: Size within the tier (e.g., 'small', 'large')
            
        Returns:
            Fee amount or None if not found
        """
        tier_fees = getattr(self.fba_fulfillment_fees, size_tier, None)
        if tier_fees:
            return tier_fees.get(size)
        return None