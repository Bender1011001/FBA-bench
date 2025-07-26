"""
Configuration loader for FBA-Bench.

This module provides functionality to load and validate the unified YAML configuration,
with caching to avoid repeated file I/O and comprehensive error handling.
"""

import os
import yaml
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from .config_models import (
    FBABenchConfig, ConfigMetadata, FeeConfig, ReferralFeeTier, DimWeightTier,
    FBAFulfillmentFees, SimulationConfig, AdversarialEventsConfig,
    MarketDynamicsConfig, AgentDefaultsConfig, APICostModelConfig,
    CompetitorModelConfig, MemorySystemConfig, DistressProtocolConfig,
    StrategicPlanningConfig, FeeMetadata
)

logger = logging.getLogger(__name__)

# Global cache for loaded configuration
_cached_config: Optional[FBABenchConfig] = None
_config_file_path: Optional[str] = None


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


def _get_default_config_path() -> str:
    """Get the default path to the configuration file."""
    # Look for config file relative to this module's directory
    current_dir = Path(__file__).parent.parent
    config_path = current_dir / "config" / "fba_bench_config.yaml"
    return str(config_path)


def _validate_config_structure(config_data: Dict[str, Any]) -> None:
    """
    Validate that the configuration has required sections and basic structure.
    
    Args:
        config_data: Raw configuration dictionary from YAML
        
    Raises:
        ConfigurationError: If required sections are missing or invalid
    """
    required_sections = [
        'schema_version', 'metadata', 'fees', 'referral_fees', 
        'fba_fulfillment_fees', 'simulation', 'adversarial_events',
        'market_dynamics', 'agent_defaults', 'api_cost_model',
        'competitor_model', 'memory_system', 'distress_protocol',
        'strategic_planning', 'fee_metadata'
    ]
    
    missing_sections = [section for section in required_sections 
                       if section not in config_data]
    
    if missing_sections:
        raise ConfigurationError(
            f"Missing required configuration sections: {missing_sections}"
        )
    
    # Validate schema version
    schema_version = config_data.get('schema_version')
    if not schema_version or not isinstance(schema_version, str):
        raise ConfigurationError("Invalid or missing schema_version")
    
    # Basic validation of numeric values
    fees = config_data.get('fees', {})
    for fee_name, fee_value in fees.items():
        # Special handling for dim_weight_tiers (list structure)
        if fee_name == 'dim_weight_tiers':
            if not isinstance(fee_value, list):
                raise ConfigurationError(
                    f"Fee '{fee_name}' must be a list of tier objects, got: {type(fee_value)}"
                )
            for i, tier in enumerate(fee_value):
                if not isinstance(tier, dict):
                    raise ConfigurationError(
                        f"Tier {i} in '{fee_name}' must be a dict, got: {type(tier)}"
                    )
                required_keys = ['weight_min', 'weight_max', 'surcharge']
                for key in required_keys:
                    if key not in tier:
                        raise ConfigurationError(
                            f"Tier {i} in '{fee_name}' missing required key: {key}"
                        )
                    if not isinstance(tier[key], (int, float)) or tier[key] < 0:
                        raise ConfigurationError(
                            f"Tier {i} in '{fee_name}' key '{key}' must be non-negative number, got: {tier[key]}"
                        )
        else:
            # Regular numeric fee validation
            if not isinstance(fee_value, (int, float)) or fee_value < 0:
                raise ConfigurationError(
                    f"Fee '{fee_name}' must be a non-negative number, got: {fee_value}"
                )
    
    # Validate referral fees structure
    referral_fees = config_data.get('referral_fees', {})
    if 'DEFAULT' not in referral_fees:
        raise ConfigurationError("referral_fees must contain a 'DEFAULT' category")
    
    for category, tiers in referral_fees.items():
        if not isinstance(tiers, list) or not tiers:
            raise ConfigurationError(
                f"Referral fees for category '{category}' must be a non-empty list"
            )
        
        for tier in tiers:
            required_tier_fields = ['threshold', 'percentage', 'minimum']
            missing_fields = [field for field in required_tier_fields 
                            if field not in tier]
            if missing_fields:
                raise ConfigurationError(
                    f"Referral fee tier in category '{category}' missing fields: {missing_fields}"
                )


def _parse_referral_fees(referral_fees_data: Dict[str, Any]) -> Dict[str, list[ReferralFeeTier]]:
    """Parse referral fees data into typed objects."""
    result = {}
    
    for category, tiers_data in referral_fees_data.items():
        tiers = []
        for tier_data in tiers_data:
            tier = ReferralFeeTier(
                threshold=float(tier_data['threshold']),
                percentage=float(tier_data['percentage']),
                minimum=float(tier_data['minimum']),
                maximum=float(tier_data['maximum']) if tier_data['maximum'] is not None else None
            )
            tiers.append(tier)
        result[category] = tiers
    
    return result


def _parse_dim_weight_tiers(dim_weight_tiers_data: list[Dict[str, Any]]) -> list[DimWeightTier]:
    """Parse dimensional weight tiers data into typed objects."""
    tiers = []
    for tier_data in dim_weight_tiers_data:
        tier = DimWeightTier(
            weight_min=float(tier_data['weight_min']),
            weight_max=float(tier_data['weight_max']),
            surcharge=float(tier_data['surcharge'])
        )
        tiers.append(tier)
    return tiers


def _load_config_from_file(config_path: str) -> FBABenchConfig:
    """
    Load configuration from YAML file and parse into typed objects.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Parsed and validated configuration object
        
    Raises:
        ConfigurationError: If file cannot be loaded or is invalid
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}. "
            f"Please ensure the file exists and is readable."
        )
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error reading configuration file: {e}")
    
    if not config_data:
        raise ConfigurationError("Configuration file is empty or invalid")
    
    # Validate structure
    _validate_config_structure(config_data)
    
    try:
        # Parse into typed objects
        # Special handling for fees with dim_weight_tiers
        fees_data = config_data['fees'].copy()
        dim_weight_tiers = _parse_dim_weight_tiers(fees_data.pop('dim_weight_tiers'))
        
        config = FBABenchConfig(
            schema_version=config_data['schema_version'],
            metadata=ConfigMetadata(**config_data['metadata']),
            fees=FeeConfig(dim_weight_tiers=dim_weight_tiers, **fees_data),
            referral_fees=_parse_referral_fees(config_data['referral_fees']),
            fba_fulfillment_fees=FBAFulfillmentFees(**config_data['fba_fulfillment_fees']),
            simulation=SimulationConfig(**config_data['simulation']),
            adversarial_events=AdversarialEventsConfig(**config_data['adversarial_events']),
            market_dynamics=MarketDynamicsConfig(**config_data['market_dynamics']),
            agent_defaults=AgentDefaultsConfig(**config_data['agent_defaults']),
            api_cost_model=APICostModelConfig(**config_data['api_cost_model']),
            competitor_model=CompetitorModelConfig(**config_data['competitor_model']),
            memory_system=MemorySystemConfig(**config_data['memory_system']),
            distress_protocol=DistressProtocolConfig(**config_data['distress_protocol']),
            strategic_planning=StrategicPlanningConfig(**config_data['strategic_planning']),
            fee_metadata=FeeMetadata(**config_data['fee_metadata'])
        )
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
        
    except TypeError as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error parsing configuration: {e}")


def load_config(config_path: Optional[str] = None, force_reload: bool = False) -> FBABenchConfig:
    """
    Load the FBA-Bench configuration.
    
    This function loads and caches the configuration to avoid repeated file I/O.
    The configuration is validated during loading to ensure all required fields
    are present and have valid values.
    
    Args:
        config_path: Path to configuration file. If None, uses default location.
        force_reload: If True, reload even if already cached
        
    Returns:
        Validated configuration object
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid
        
    Example:
        >>> config = load_config()
        >>> fee = config.fees.professional_monthly
        >>> referral_rate = config.referral_fees["Electronics"][0].percentage
    """
    global _cached_config, _config_file_path
    
    if config_path is None:
        config_path = _get_default_config_path()
    
    # Return cached config if available and not forcing reload
    if not force_reload and _cached_config is not None and _config_file_path == config_path:
        return _cached_config
    
    # Load and cache the configuration
    _cached_config = _load_config_from_file(config_path)
    _config_file_path = config_path
    
    return _cached_config


def clear_config_cache() -> None:
    """Clear the cached configuration, forcing a reload on next access."""
    global _cached_config, _config_file_path
    _cached_config = None
    _config_file_path = None


def get_config_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded configuration.
    
    Returns:
        Dictionary with configuration metadata and status
    """
    if _cached_config is None:
        return {
            "loaded": False,
            "config_path": None,
            "schema_version": None,
            "metadata": None
        }
    
    return {
        "loaded": True,
        "config_path": _config_file_path,
        "schema_version": _cached_config.schema_version,
        "metadata": {
            "version": _cached_config.metadata.version,
            "effective_date": _cached_config.metadata.effective_date,
            "last_updated": _cached_config.metadata.last_updated,
            "description": _cached_config.metadata.description
        }
    }


# Convenience function for backward compatibility
def get_config() -> FBABenchConfig:
    """
    Get the current configuration (loads if not already loaded).
    
    This is a convenience function that ensures configuration is loaded
    and returns the cached instance.
    
    Returns:
        Current configuration object
    """
    return load_config()