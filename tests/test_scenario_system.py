import pytest
import os
import shutil
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Adjust path to import modules correctly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scenarios.scenario_framework import ScenarioConfig
from scenarios.scenario_engine import ScenarioEngine
from scenarios.curriculum_validator import CurriculumValidator
from scenarios.dynamic_generator import DynamicScenarioGenerator
from scenarios.scenario_config import ScenarioConfigManager
from experiment_cli import main as cli_main # Import the main function from experiment_cli.py

# Define temporary directories for tests
TEST_SCENARIO_DIR = "test_scenarios_temp"
TEST_BUSINESS_TYPES_DIR = os.path.join(TEST_SCENARIO_DIR, "business_types")
TEST_MULTI_AGENT_DIR = os.path.join(TEST_SCENARIO_DIR, "multi_agent")
TEST_RESULTS_DIR = "test_scenario_results"


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Sets up a temporary test environment with scenario files."""
    # Create test directories
    os.makedirs(TEST_BUSINESS_TYPES_DIR, exist_ok=True)
    os.makedirs(TEST_MULTI_AGENT_DIR, exist_ok=True)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    # Create dummy scenario files for testing
    # Tier 0 Baseline
    with open(os.path.join(TEST_SCENARIO_DIR, "tier_0_baseline.yaml"), "w") as f:
        f.write("""
scenario_name: Test Tier 0 Baseline
difficulty_tier: 0
expected_duration: 5
success_criteria: {profit_target: 1000.0}
market_conditions: {economic_cycles: stable}
external_events: []
agent_constraints: {initial_capital: 10000}
""")

    # Tier 1 Moderate
    with open(os.path.join(TEST_SCENARIO_DIR, "tier_1_moderate.yaml"), "w") as f:
        f.write("""
scenario_name: Test Tier 1 Moderate
difficulty_tier: 1
expected_duration: 10
success_criteria: {profit_target: 5000.0}
market_conditions: {economic_cycles: seasonal}
external_events: [{name: "Minor Price Drop", tick: 5, type: "market_event"}]
agent_constraints: {initial_capital: 5000}
""")

    # International Expansion (Tier 3 Business Type)
    with open(os.path.join(TEST_BUSINESS_TYPES_DIR, "international_expansion.yaml"), "w") as f:
        f.write("""
scenario_name: International Expansion Dummy
difficulty_tier: 3
expected_duration: 15
success_criteria: {profit_target: 10000.0, market_share_europe: 0.1}
market_conditions: {economic_cycles: global_variations}
external_events: [{name: "Tariff Change", tick: 7, type: "regulatory_change"}]
agent_constraints: {initial_capital: 2000}
""")

    # Cooperative Joint Venture (Multi-Agent)
    with open(os.path.join(TEST_MULTI_AGENT_DIR, "cooperative_joint_venture.yaml"), "w") as f:
        f.write("""
scenario_name: Cooperative Joint Venture Dummy
difficulty_tier: 2
expected_duration: 10
success_criteria: {joint_profit_target: 10000.0}
market_conditions: {}
external_events: [{name: "Joint Marketing", tick: 3, type: "opportunity"}]
agent_constraints: {}
multi_agent_config: {num_agents: 2, interaction_mode: cooperative}
""")
    
    # Dummy randomization config for dynamic generator
    with open(os.path.join(TEST_SCENARIO_DIR, "dynamic_rand_config.yaml"), "w") as f:
        f.write("""
market_conditions:
  economic_cycles: {choices: [stable, recession_recovery]}
agent_constraints:
  initial_capital: {range: [1000, 5000]}
""")

    # Yield control to tests
    yield

    # Teardown: Clean up test directories
    if os.path.exists(TEST_SCENARIO_DIR):
        shutil.rmtree(TEST_SCENARIO_DIR)
    if os.path.exists(TEST_RESULTS_DIR):
        shutil.rmtree(TEST_RESULTS_DIR)

class MockAgent:
    """A simple mock agent for testing purposes."""
    def __init__(self, name="MockAgent"):
        self._name = name
    
    def name(self):
        return self._name

    async def step(self):
        """Simulate an agent's step."""
        pass # No actual logic needed for tests here

@pytest.mark.asyncio
async def test_scenario_loading_and_validation():
    """Test scenario loading and internal consistency validation."""
    engine = ScenarioEngine()

    # Test loading a valid scenario
    scenario = engine.load_scenario(os.path.join(TEST_SCENARIO_DIR, "tier_0_baseline.yaml"))
    assert scenario.config_data['scenario_name'] == "Test Tier 0 Baseline"
    assert scenario.config_data['difficulty_tier'] == 0
    assert scenario.validate_scenario_consistency() is True

    # Test invalid scenario (e.g., missing key) - requires a malformed test file
    # For now, rely on internal validation within DummyScenarioConfig
    invalid_config_data = {
        'scenario_name': 'Invalid Test',
        'difficulty_tier': 0,
        # missing expected_duration
        'success_criteria': {},
        'market_conditions': {},
        'external_events': [],
        'agent_constraints': {}
    }
    with pytest.raises(ValueError, match="Missing required key in scenario config: expected_duration"):
        ScenarioConfig(invalid_config_data)

    # Test loading a non-existent file
    with pytest.raises(FileNotFoundError):
        engine.load_scenario("non_existent_scenario.yaml")

@pytest.mark.asyncio
async def test_scenario_engine_run_simulation():
    """Test the scenario engine's ability to run a simulation."""
    engine = ScenarioEngine()
    mock_agent = MockAgent()
    
    # Patch the BotFactory.create_bot if it's used in cli_main logic
    with patch('baseline_bots.bot_factory.BotFactory.create_bot', return_value=mock_agent):
        results = await engine.run_simulation(os.path.join(TEST_SCENARIO_DIR, "tier_0_baseline.yaml"), {"MockAgent": mock_agent})
    
    assert results is not None
    assert results['success_status'] in ["success", "fail"] # Depends on dummy metrics
    assert results['scenario_name'] == "Test Tier 0 Baseline"
    assert results['tier'] == 0
    assert 'profit_target' in results['metrics']
    assert results['simulation_duration'] == 5 # From tier_0_baseline.yaml

@pytest.mark.asyncio
async def test_curriculum_progression_validation():
    """Tests curriculum validation system for proper difficulty progression."""
    validator = CurriculumValidator()
    
    # Simulate a few runs for different tiers (dummy results)
    validator.benchmark_agent_performance("AgentA", 0, "Test Scenario T0", {'profit': 1200, 'success_status': 'success', 'simulation_duration': 5})
    validator.benchmark_agent_performance("AgentA", 0, "Test Scenario T0", {'profit': 1100, 'success_status': 'success', 'simulation_duration': 5})
    validator.benchmark_agent_performance("AgentA", 1, "Test Scenario T1", {'profit': 4000, 'success_status': 'fail', 'simulation_duration': 10})
    validator.benchmark_agent_performance("AgentA", 1, "Test Scenario T1", {'profit': 5500, 'success_status': 'success', 'simulation_duration': 10}) # 50% success for T1
    validator.benchmark_agent_performance("AgentA", 2, "Test Scenario T2", {'profit': 500, 'success_status': 'fail', 'simulation_duration': 15})
    validator.benchmark_agent_performance("AgentA", 2, "Test Scenario T2", {'profit': 1000, 'success_status': 'fail', 'simulation_duration': 15}) # 0% success for T2
    validator.benchmark_agent_performance("AgentA", 3, "Test Scenario T3", {'profit': 100, 'success_status': 'fail', 'simulation_duration': 20})

    report = validator.validate_tier_progression()
    assert report["tier_progression_check"] is True # Assuming our dummy data still allows this after fixing
    assert "observations" in report
    assert len(report['tier_summaries']) == 4 # T0, T1, T2, T3

    recommendations = validator.recommend_tier_adjustments(report)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0 # Should have some recommendations based on dummy data

@pytest.mark.asyncio
async def test_dynamic_scenario_generation():
    """Test dynamic scenario generation and scaling."""
    generator = DynamicScenarioGenerator(template_dir=TEST_SCENARIO_DIR + os.sep)
    
    rand_config_path = os.path.join(TEST_SCENARIO_DIR, "dynamic_rand_config.yaml")
    with open(rand_config_path, 'r') as f:
        dynamic_rand_config = yaml.safe_load(f)

    # Generate a scenario from tier_0_baseline template
    generated_scenario = generator.generate_scenario("tier_0_baseline", dynamic_rand_config)
    assert generated_scenario.config_data['scenario_name'].startswith("Test Tier 0 Baseline")
    assert generated_scenario.config_data['difficulty_tier'] == 0 # Base tier from template
    assert isinstance(generated_scenario.config_data['agent_constraints']['initial_capital'], int) # Randomized

    # Test scaling difficulty
    scaled_scenario_t3 = generator.scale_difficulty(generated_scenario, target_tier=3)
    assert scaled_scenario_t3.config_data['difficulty_tier'] == 3
    assert scaled_scenario_t3.config_data['market_conditions']['competition_levels'] == 'extreme'
    assert scaled_scenario_t3.config_data['agent_constraints']['information_asymmetry'] is True
    assert len(scaled_scenario_t3.config_data['external_events']) > 0 # Should have events after scaling

@pytest.mark.asyncio
@patch('sys.argv', ['experiment_cli.py', 'run', '--scenario', 'Test Tier 0 Baseline', '--agents', 'MockAgent'])
@patch('scenarios.scenario_engine.ScenarioEngine.run_simulation', new_callable=AsyncMock)
@patch('baseline_bots.bot_factory.BotFactory.create_bot', return_value=MockAgent())
async def test_cli_run_scenario(mock_create_bot, mock_run_simulation):
    """Test CLI integration for running a single scenario."""
    # Mock the return value of run_simulation
    mock_run_simulation.return_value = {
        'scenario_name': 'Test Tier 0 Baseline',
        'tier': 0,
        'success_status': 'success',
        'profit_target': 1200.0,
        'simulation_duration': 5
    }

    await cli_main()
    mock_run_simulation.assert_called_once()
    args, kwargs = mock_run_simulation.call_args
    assert args[0].endswith("tier_0_baseline.yaml")
    assert 'MockAgent' in kwargs['agent_models']

    # Verify output file was created (needs to bypass sys.exit(0))
    # This requires more advanced mocking or checking the file system manually.
    # For now, confirm the mock was called.
    output_files = list(Path(TEST_RESULTS_DIR).glob("scenario_run_*.json"))
    assert len(output_files) == 1
    with open(output_files[0], 'r') as f:
        results = json.load(f)
        assert len(results) == 1
        assert results[0]['scenario_name'] == 'Test Tier 0 Baseline'
        assert results[0]['agent_name'] == 'MockAgent'

@pytest.mark.asyncio
@patch('sys.argv', ['experiment_cli.py', 'run', '--tier', '0', '--agents', 'MockAgent', '--validate-curriculum'])
@patch('scenarios.scenario_engine.ScenarioEngine.run_simulation', new_callable=AsyncMock)
@patch('baseline_bots.bot_factory.BotFactory.create_bot', return_value=MockAgent())
async def test_cli_run_tier_and_validate_curriculum(mock_create_bot, mock_run_simulation):
    """Test CLI integration for running scenarios in a tier and curriculum validation."""
    mock_run_simulation.side_effect = [
        {'scenario_name': 'Test Tier 0 Baseline', 'tier': 0, 'success_status': 'success', 'profit_target': 1500, 'simulation_duration': 5},
        # Add more mock results if tier 0 had other scenarios
    ]

    await cli_main()
    assert mock_run_simulation.call_count >= 1 # At least one T0 scenario was run

    # Verify curriculum report was generated
    output_files = list(Path(TEST_RESULTS_DIR).glob("scenario_run_*.json"))
    assert len(output_files) >= 1
    
    # Find the report associated with this run
    report_files = [f for f in output_files[0].parent.glob("curriculum_validation_report.json")]
    assert len(report_files) == 1
    with open(report_files[0], 'r') as f:
        report = json.load(f)
        assert "overall_performance_summary" in report
        assert "tier_progression_validation" in report
        assert report['tier_progression_validation']['tier_summaries']['0']['avg_success_rate'] > 0

@pytest.mark.asyncio
@patch('sys.argv', ['experiment_cli.py', 'run', '--generate-scenario', 'tier_0_baseline', '--dynamic-randomization-config', os.path.join(TEST_SCENARIO_DIR, "dynamic_rand_config.yaml"), '--dynamic-scenario-output', os.path.join(TEST_RESULTS_DIR, "generated_scenario_cli.yaml")])
async def test_cli_generate_scenario(mock_cli_parse_args):
    """Test CLI integration for dynamic scenario generation."""
    # The actual cli_main will sys.exit(0) after generating.
    # We need to mock sys.exit to prevent test runner from stopping.
    with patch('sys.exit') as mock_sys_exit:
        await cli_main()
        mock_sys_exit.assert_called_once_with(0) # Ensure it exits as expected

    output_file_path = Path(TEST_RESULTS_DIR) / "generated_scenario_cli.yaml"
    assert output_file_path.exists()
    
    generated_config = ScenarioConfig.from_yaml(str(output_file_path))
    assert generated_config.config_data['scenario_name'].startswith("Test Tier 0 Baseline")
    assert isinstance(generated_config.config_data['agent_constraints']['initial_capital'], int)
    assert generated_config.validate_scenario_consistency() is True

@pytest.mark.asyncio
@patch('sys.argv', ['experiment_cli.py', 'run', '--benchmark-scenarios', '--agents', 'MockAgent'])
@patch('scenarios.scenario_engine.ScenarioEngine.run_simulation', new_callable=AsyncMock)
@patch('baseline_bots.bot_factory.BotFactory.create_bot', return_value=MockAgent())
async def test_cli_benchmark_scenarios(mock_create_bot, mock_run_simulation):
    """Test CLI integration for benchmarking all scenarios."""
    mock_run_simulation.side_effect = [
        {'scenario_name': 'Test Tier 0 Baseline', 'tier': 0, 'success_status': 'success', 'profit_target': 1500, 'simulation_duration': 5},
        {'scenario_name': 'Test Tier 1 Moderate', 'tier': 1, 'success_status': 'fail', 'profit_target': 3000, 'simulation_duration': 10},
        {'scenario_name': 'International Expansion Dummy', 'tier': 3, 'success_status': 'fail', 'profit_target': 5000, 'simulation_duration': 15},
        {'scenario_name': 'Cooperative Joint Venture Dummy', 'tier': 2, 'success_status': 'success', 'joint_profit_target': 12000, 'simulation_duration': 10},
    ]

    await cli_main()
    
    # There are 4 dummy scenarios, so simulation should be called 4 times.
    assert mock_run_simulation.call_count == 4 

    output_files = list(Path(TEST_RESULTS_DIR).glob("benchmark_run_*.json"))
    assert len(output_files) == 1
    with open(output_files[0], 'r') as f:
        results = json.load(f)
        assert len(results) == 4
        assert any(r['scenario_name'] == 'Test Tier 0 Baseline' for r in results)
        assert any(r['scenario_name'] == 'International Expansion Dummy' for r in results)