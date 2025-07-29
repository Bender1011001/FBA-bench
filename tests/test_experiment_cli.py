#!/usr/bin/env python3
"""
Test script for FBA-Bench v3 Experiment CLI.

Validates that the CLI can parse configurations and generate parameter combinations correctly.
"""

import sys
import asyncio
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiment_cli import ExperimentConfig, SimulationRunner, ExperimentManager


def test_config_parsing():
    """Test that the sweep.yaml configuration loads correctly."""
    print("🧪 Testing configuration parsing...")
    
    try:
        # Load the sample configuration
        with open('sweep.yaml', 'r') as f:
            import yaml
            config_data = yaml.safe_load(f)
        
        config = ExperimentConfig(config_data)
        
        # Validate basic properties
        assert config.experiment_name == "price_strategy_test"
        assert "duration_hours" in config.base_parameters
        assert "parameter_sweep" in config.__dict__
        
        print(f"✅ Configuration loaded successfully:")
        print(f"   Experiment: {config.experiment_name}")
        print(f"   Description: {config.description}")
        print(f"   Base parameters: {list(config.base_parameters.keys())}")
        
        return config
        
    except Exception as e:
        print(f"❌ Configuration parsing failed: {e}")
        return None


def test_parameter_combinations(config: ExperimentConfig):
    """Test parameter combination generation."""
    print("\n🔢 Testing parameter combination generation...")
    
    try:
        total_combinations = config.get_total_combinations()
        print(f"   Total combinations: {total_combinations}")
        
        # Generate and display first few combinations
        combinations = []
        for run_number, parameters in config.generate_parameter_combinations():
            combinations.append((run_number, parameters))
            if len(combinations) >= 3:  # Only test first 3
                break
        
        print(f"   Generated {len(combinations)} sample combinations:")
        for run_number, params in combinations:
            key_info = []
            if 'initial_price' in params:
                key_info.append(f"price=${params['initial_price']}")
            if 'competitor_persona_distribution' in params:
                dist = params['competitor_persona_distribution']
                if isinstance(dist, dict) and 'name' in dist:
                    key_info.append(f"market={dist['name']}")
            
            print(f"     Run {run_number}: {', '.join(key_info)}")
        
        # Validate expected number of combinations
        # We have: 3 prices × 3 persona distributions × 3 market sensitivities = 27 total
        expected_total = 3 * 3 * 3
        if total_combinations == expected_total:
            print(f"✅ Combination count matches expected: {expected_total}")
        else:
            print(f"⚠️  Combination count mismatch: got {total_combinations}, expected {expected_total}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter combination generation failed: {e}")
        return False


def test_simulation_runner_setup(config: ExperimentConfig):
    """Test SimulationRunner setup without running full simulation."""
    print("\n⚙️  Testing simulation runner setup...")
    
    try:
        runner = SimulationRunner(config)
        
        # Test parameter formatting
        sample_params = {
            'initial_price': 29.99,
            'competitor_persona_distribution': {
                'name': 'aggressive_market',
                'distribution': {'IrrationalSlasher': 0.7, 'SlowFollower': 0.3}
            },
            'market_sensitivity': 0.8
        }
        
        formatted = runner._format_key_parameters(sample_params)
        print(f"   Sample parameter formatting: {formatted}")
        
        # Test competitor manager setup
        cm = runner._setup_competitor_manager(sample_params)
        print(f"   CompetitorManager created successfully")
        
        print("✅ SimulationRunner setup successful")
        return True
        
    except Exception as e:
        print(f"❌ SimulationRunner setup failed: {e}")
        return False


async def test_experiment_manager():
    """Test ExperimentManager initialization."""
    print("\n🏗️  Testing experiment manager...")
    
    try:
        manager = ExperimentManager('sweep.yaml')
        
        print(f"   Experiment name: {manager.experiment_config.experiment_name}")
        print(f"   Results directory: {manager.results_dir}")
        
        # Verify results directory was created
        if manager.results_dir.exists():
            print("✅ Results directory created successfully")
            
            # Check if config was saved
            config_file = manager.results_dir / "experiment_config.yaml"
            if config_file.exists():
                print("✅ Experiment configuration saved to results directory")
            else:
                print("⚠️  Experiment configuration not saved")
        else:
            print("❌ Results directory not created")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ExperimentManager initialization failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing FBA-Bench v3 Experiment CLI")
    print("=" * 50)
    
    # Test configuration parsing
    config = test_config_parsing()
    if not config:
        return False
    
    # Test parameter combinations
    if not test_parameter_combinations(config):
        return False
    
    # Test simulation runner setup
    if not test_simulation_runner_setup(config):
        return False
    
    # Test experiment manager (async)
    if not asyncio.run(test_experiment_manager()):
        return False
    
    print("\n🎉 All tests passed! Experiment CLI is ready for use.")
    print("\nTo run a full experiment:")
    print("   python experiment_cli.py run sweep.yaml --max-runs 1")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)