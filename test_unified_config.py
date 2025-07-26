#!/usr/bin/env python3
"""
Test script for the unified configuration system.

This script tests that:
1. The unified configuration loads correctly
2. Backward compatibility is maintained
3. The fee engine works with the new configuration
4. All configuration values are accessible
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_loader():
    """Test the configuration loader."""
    print("Testing configuration loader...")
    
    try:
        from fba_bench.config_loader import load_config, get_config_info
        
        # Load configuration
        config = load_config()
        print(f"✓ Configuration loaded successfully")
        print(f"  Schema version: {config.schema_version}")
        print(f"  Config version: {config.metadata.version}")
        
        # Test config info
        info = get_config_info()
        print(f"✓ Configuration info retrieved")
        print(f"  Loaded: {info['loaded']}")
        print(f"  Schema version: {info['schema_version']}")
        
        # Test some specific values
        assert config.fees.professional_monthly == 39.99
        assert config.fees.fuel_surcharge_pct == 0.02
        assert config.simulation.money_strict == False
        print(f"✓ Configuration values are correct")
        
        # Test referral fees
        default_referral = config.referral_fees["DEFAULT"]
        assert len(default_referral) == 1
        assert default_referral[0].percentage == 0.15
        assert default_referral[0].minimum == 0.30
        print(f"✓ Referral fees structure is correct")
        
        # Test FBA fulfillment fees
        assert config.fba_fulfillment_fees.standard["small"] == 3.22
        assert config.fba_fulfillment_fees.oversize["large"] == 10.50
        print(f"✓ FBA fulfillment fees are correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration loader test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility with the old config.py imports."""
    print("\nTesting backward compatibility...")
    
    try:
        # Test importing from the old config module
        from fba_bench.config import (
            PROFESSIONAL_MONTHLY, FUEL_SURCHARGE_PCT, MONEY_STRICT,
            DEFAULT_ASIN, BSR_BASE, get_unified_config
        )
        
        # Verify values are correct
        assert PROFESSIONAL_MONTHLY == 39.99
        assert FUEL_SURCHARGE_PCT == 0.02
        assert MONEY_STRICT == False
        assert DEFAULT_ASIN == "B000TEST"
        assert BSR_BASE == 100000
        print(f"✓ Old config constants are accessible and correct")
        
        # Test unified config access
        unified_config = get_unified_config()
        assert unified_config.fees.professional_monthly == 39.99
        print(f"✓ Unified config access works")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False


def test_fee_engine():
    """Test the fee engine with the unified configuration."""
    print("\nTesting fee engine with unified configuration...")
    
    try:
        from fba_bench.fee_engine import FeeEngine
        
        # Initialize fee engine (should use unified config)
        fee_engine = FeeEngine()
        print(f"✓ Fee engine initialized successfully")
        
        # Test basic fee calculations
        referral_fee = fee_engine.referral_fee("DEFAULT", 10.0)
        print(f"✓ Referral fee calculation: ${referral_fee.to_float()}")
        
        fba_fee = fee_engine.fba_fulfillment_fee("standard", "small")
        print(f"✓ FBA fulfillment fee: ${fba_fee.to_float()}")
        
        holiday_fee = fee_engine.holiday_surcharge(True)
        print(f"✓ Holiday surcharge: ${holiday_fee.to_float()}")
        
        # Test total fees calculation
        total_fees = fee_engine.total_fees(
            category="DEFAULT",
            price=20.0,
            size_tier="standard",
            size="small"
        )
        print(f"✓ Total fees calculation completed")
        print(f"  Total: ${total_fees['total']}")
        
        # Verify fee metadata
        assert fee_engine.fee_metadata["version"] == "2025.1"
        assert "unified" in fee_engine.fee_metadata["notes"].lower()
        print(f"✓ Fee metadata is correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Fee engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_sections():
    """Test that all configuration sections are accessible."""
    print("\nTesting configuration sections...")
    
    try:
        from fba_bench.config_loader import load_config
        
        config = load_config()
        
        # Test all major sections
        sections = [
            'fees', 'referral_fees', 'fba_fulfillment_fees', 'simulation',
            'adversarial_events', 'market_dynamics', 'agent_defaults',
            'api_cost_model', 'competitor_model', 'memory_system',
            'distress_protocol', 'strategic_planning', 'fee_metadata'
        ]
        
        for section in sections:
            assert hasattr(config, section), f"Missing section: {section}"
            section_obj = getattr(config, section)
            assert section_obj is not None, f"Section {section} is None"
        
        print(f"✓ All {len(sections)} configuration sections are accessible")
        
        # Test some specific values from different sections
        assert config.market_dynamics.bsr_base == 100000
        assert config.adversarial_events.default_supply_shock_factor == 0.5
        assert config.agent_defaults.default_category == "DEFAULT"
        print(f"✓ Values from different sections are correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration sections test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("UNIFIED CONFIGURATION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        test_config_loader,
        test_backward_compatibility,
        test_fee_engine,
        test_configuration_sections,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The unified configuration system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())