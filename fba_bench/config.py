"""
FBA-Bench Configuration Module

This module provides backward-compatible access to configuration values
that are now loaded from the unified YAML configuration system.

DEPRECATION WARNING: Direct imports from this module are deprecated.
Please use the unified configuration system:

    from fba_bench.config_loader import load_config
    config = load_config()
    fee = config.fees.professional_monthly

For backward compatibility, all existing constants are still available
but now loaded from the unified configuration.
"""

import warnings
from .config_loader import load_config

# Load the unified configuration
_config = load_config()

# Issue deprecation warning for direct imports
def _deprecation_warning():
    warnings.warn(
        "Direct imports from fba_bench.config are deprecated. "
        "Please use the unified configuration system: "
        "from fba_bench.config_loader import load_config",
        DeprecationWarning,
        stacklevel=3
    )

# Fee Engine Configuration (backward compatibility)
FUEL_SURCHARGE_PCT = _config.fees.fuel_surcharge_pct
HOLIDAY_SURCHARGE = _config.fees.holiday_surcharge
# BUGFIX: Handle new tiered dim_weight structure - provide backward compatibility
# Use the middle tier (1-5 lbs) as the default for backward compatibility
DIM_WEIGHT_SURCHARGE = 1.25  # Default fallback for backward compatibility
if hasattr(_config.fees, 'dim_weight_tiers') and _config.fees.dim_weight_tiers:
    # Find the 1-5 lb tier for backward compatibility
    for tier in _config.fees.dim_weight_tiers:
        if tier.weight_min <= 2.5 < tier.weight_max:  # 2.5 lbs is middle of 1-5 range
            DIM_WEIGHT_SURCHARGE = tier.surcharge
            break
PROFESSIONAL_MONTHLY = _config.fees.professional_monthly
INDIVIDUAL_PER_ITEM = _config.fees.individual_per_item
AGED_INVENTORY_SURCHARGE_181 = _config.fees.aged_inventory_surcharge_181
AGED_INVENTORY_SURCHARGE_271 = _config.fees.aged_inventory_surcharge_271
LOW_INVENTORY_LEVEL_FEE_PER_UNIT = _config.fees.low_inventory_level_fee_per_unit
STORAGE_UTILIZATION_SURCHARGE_PCT = _config.fees.storage_utilization_surcharge_pct
UNPLANNED_SERVICE_FEE_PER_UNIT = _config.fees.unplanned_service_fee_per_unit
LONG_TERM_STORAGE_FEE = _config.fees.long_term_storage_fee
REMOVAL_FEE_PER_UNIT = _config.fees.removal_fee_per_unit
RETURN_PROCESSING_FEE_PCT = _config.fees.return_processing_fee_pct

# Simulation Configuration (backward compatibility)
CUBIC_FEET_PER_UNIT = _config.simulation.cubic_feet_per_unit
MONTHS_STORAGE_DEFAULT = _config.simulation.months_storage_default
REMOVAL_UNITS_DEFAULT = _config.simulation.removal_units_default
RETURN_FEES_DEFAULT = _config.simulation.return_fees_default
AGED_DAYS_DEFAULT = _config.simulation.aged_days_default
AGED_CUBIC_FEET_PER_UNIT = _config.simulation.aged_cubic_feet_per_unit
LOW_INVENTORY_UNITS_DEFAULT = _config.simulation.low_inventory_units_default
TRAILING_DAYS_SUPPLY_DEFAULT = _config.simulation.trailing_days_supply_default
EMA_DECAY = _config.simulation.ema_decay
WEEKS_SUPPLY_DEFAULT = _config.simulation.weeks_supply_default

# Money Migration Feature Flag (backward compatibility)
MONEY_STRICT = _config.simulation.money_strict

# Adversarial Events Configuration (backward compatibility)
DEFAULT_SUPPLY_SHOCK_FACTOR = _config.adversarial_events.default_supply_shock_factor
DEFAULT_SUPPLY_SHOCK_DURATION = _config.adversarial_events.default_supply_shock_duration
DEFAULT_REVIEW_ATTACK_COUNT = _config.adversarial_events.default_review_attack_count
DEFAULT_PRICE_WAR_DROP_PCT = _config.adversarial_events.default_price_war_drop_pct
DEFAULT_PRICE_WAR_DURATION = _config.adversarial_events.default_price_war_duration
DEFAULT_FAKE_REVIEW_COUNT = _config.adversarial_events.default_fake_review_count
DEFAULT_INVENTORY_FREEZE_DURATION = _config.adversarial_events.default_inventory_freeze_duration
DEFAULT_BUYBOX_HIJACK_DURATION = _config.adversarial_events.default_buybox_hijack_duration
DEFAULT_POLICY_PENALTY_AMOUNT = _config.adversarial_events.default_policy_penalty_amount

# Market Dynamics Configuration (backward compatibility)
REL_PRICE_FACTOR_MIN = _config.market_dynamics.rel_price_factor_min
REL_PRICE_FACTOR_MAX = _config.market_dynamics.rel_price_factor_max
ELASTICITY_MIN = _config.market_dynamics.elasticity_min
ELASTICITY_MAX = _config.market_dynamics.elasticity_max
BSR_MID_POINT = _config.market_dynamics.bsr_mid_point
BSR_SCALE = _config.market_dynamics.bsr_scale

# Agent Defaults (backward compatibility)
DEFAULT_ASIN = _config.agent_defaults.default_asin
DEFAULT_CATEGORY = _config.agent_defaults.default_category
DEFAULT_COST = _config.agent_defaults.default_cost
DEFAULT_PRICE = _config.agent_defaults.default_price
DEFAULT_QTY = _config.agent_defaults.default_qty

# BSR Calculation Constants (backward compatibility)
BSR_BASE = _config.market_dynamics.bsr_base
BSR_SMOOTHING_FACTOR = _config.market_dynamics.bsr_smoothing_factor
BSR_MIN_VALUE = _config.market_dynamics.bsr_min_value
BSR_MAX_VALUE = _config.market_dynamics.bsr_max_value

# API Cost Model Constants (backward compatibility)
DEFAULT_API_BUDGET = _config.api_cost_model.default_api_budget
DEFAULT_CPU_BUDGET = _config.api_cost_model.default_cpu_budget

# Competitor Model Constants (backward compatibility)
COMPETITOR_PRICE_CHANGE_BASE = _config.competitor_model.price_change_base
COMPETITOR_SALES_CHANGE_BASE = _config.competitor_model.sales_change_base
COMPETITOR_STRATEGIES = _config.competitor_model.strategies
AGGRESSIVE_UNDERCUT_THRESHOLD = _config.competitor_model.aggressive_undercut_threshold
AGGRESSIVE_UNDERCUT_AMOUNT = _config.competitor_model.aggressive_undercut_amount
FOLLOWER_PRICE_SENSITIVITY = _config.competitor_model.follower_price_sensitivity
PREMIUM_PRICE_MAINTENANCE = _config.competitor_model.premium_price_maintenance
VALUE_COMPETITIVE_THRESHOLD = _config.competitor_model.value_competitive_threshold

# Reflection and Memory Constants (backward compatibility)
MEMORY_CONSOLIDATION_FREQUENCY = _config.memory_system.consolidation_frequency
EPISODIC_MEMORY_CAPACITY = _config.memory_system.episodic_memory_capacity
PROCEDURAL_MEMORY_CAPACITY = _config.memory_system.procedural_memory_capacity
SEMANTIC_SIMILARITY_THRESHOLD = _config.memory_system.semantic_similarity_threshold

# Distress Protocol Constants (backward compatibility)
DISTRESS_COMPUTE_THRESHOLD = _config.distress_protocol.compute_threshold
DISTRESS_NEGATIVE_CASH_THRESHOLD = _config.distress_protocol.negative_cash_threshold
DISTRESS_POLICY_PARALYSIS_TICKS = _config.distress_protocol.policy_paralysis_ticks

# Strategic Plan Constants (backward compatibility)
DEFAULT_TARGET_PROFIT_MARGIN = _config.strategic_planning.target_profit_margin
DEFAULT_MAX_STORAGE_FEE_RATIO = _config.strategic_planning.max_storage_fee_ratio
DEFAULT_MIN_INVENTORY_TURNOVER = _config.strategic_planning.min_inventory_turnover
DEFAULT_TARGET_BSR = _config.strategic_planning.target_bsr

# Provide access to the unified configuration object
def get_unified_config():
    """
    Get the unified configuration object.
    
    Returns:
        FBABenchConfig: The complete configuration object
    """
    return _config

# For modules that want to check if they're using the new system
UNIFIED_CONFIG_AVAILABLE = True