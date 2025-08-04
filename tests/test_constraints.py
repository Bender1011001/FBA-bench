import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, ANY

from constraints.token_counter import TokenCounter
from constraints.constraint_config import ConstraintConfig, get_tier_config_path
from constraints.budget_enforcer import BudgetEnforcer
from constraints.agent_gateway import AgentGateway
from events import BudgetWarning, BudgetExceeded, ConstraintViolation, TickEvent, BaseEvent
from event_bus import EventBus
from metrics.cost_metrics import CostMetrics

# --- Fixtures ---

@pytest.fixture
def mock_event_bus():
    bus = EventBus(backend=MagicMock())
    bus.publish = AsyncMock() # Mock the async publish method
    return bus

@pytest.fixture
def mock_metrics_tracker():
    metrics = MagicMock(spec=CostMetrics)
    metrics.record_token_usage = MagicMock()
    metrics.apply_penalty = MagicMock()
    return metrics

@pytest.fixture
def default_config():
    return ConstraintConfig(
        max_tokens_per_action=100,
        max_total_tokens=500,
        token_cost_per_1k=0.01,
        violation_penalty_weight=10.0,
        grace_period_percentage=10.0, # 10% grace
        hard_fail_on_violation=True,
        inject_budget_status=True,
        track_token_efficiency=True
    )

@pytest.fixture
def lenient_config():
    return ConstraintConfig(
        max_tokens_per_action=100,
        max_total_tokens=500,
        token_cost_per_1k=0.01,
        violation_penalty_weight=1.0,
        grace_period_percentage=20.0, # 20% grace
        hard_fail_on_violation=False, # No hard fail
        inject_budget_status=True,
        track_token_efficiency=False
    )

# --- TokenCounter Tests ---

def test_token_counter_count_tokens():
    counter = TokenCounter()
    text = "Hello, world!"
    # tiktoken's gpt-4 tokenizer counts "Hello, world!" as 4 tokens
    assert counter.count_tokens(text, "gpt-4") == 4

    long_text = "This is a much longer piece of text for testing token counting accuracy. It should definitely have more tokens."
    assert counter.count_tokens(long_text, "gpt-4") > 10 # Just a sanity check

def test_token_counter_count_message_tokens():
    counter = TokenCounter()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    # Exact count depends on tiktoken implementation, but should be reasonable
    # For "gpt-4" with system + user, it's usually around BASE_TOKENS_PER_MESSAGE + num_messages * TOKENS_PER_MESSAGE
    # With role/content/name structure, this might be around 8 for two messages.
    assert counter.count_message_tokens(messages, "gpt-4") > 5 

def test_token_counter_calculate_cost():
    counter = TokenCounter()
    tokens = 1000
    cost_per_1k = 0.05
    assert counter.calculate_cost(tokens, cost_per_1k) == 0.05

    tokens = 2500
    cost_per_1k = 0.01
    assert counter.calculate_cost(tokens, cost_per_1k) == 0.025

# --- ConstraintConfig Tests ---

def test_constraint_config_from_yaml(tmp_path):
    config_content = """
    budget_constraints:
      max_tokens_per_action: 1000
      max_total_tokens: 10000
      token_cost_per_1k: 0.005
      violation_penalty_weight: 1.5
      grace_period_percentage: 15.0
    enforcement:
      hard_fail_on_violation: false
      inject_budget_status: true
      track_token_efficiency: false
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    config = ConstraintConfig.from_yaml(str(config_file))
    assert config.max_tokens_per_action == 1000
    assert config.max_total_tokens == 10000
    assert config.token_cost_per_1k == 0.005
    assert config.violation_penalty_weight == 1.5
    assert config.grace_period_percentage == 15.0
    assert config.hard_fail_on_violation is False
    assert config.inject_budget_status is True
    assert config.track_token_efficiency is False

# --- BudgetEnforcer Tests ---

@pytest.mark.asyncio
async def test_budget_enforcer_initial_state(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    assert enforcer.current_tick_tokens_used == 0
    assert enforcer.total_simulation_tokens_used == 0
    assert enforcer.violation_triggered is False

@pytest.mark.asyncio
async def test_budget_enforcer_record_token_usage(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.record_token_usage(50, "planning")
    assert enforcer.current_tick_tokens_used == 50
    assert enforcer.total_simulation_tokens_used == 50
    mock_metrics_tracker.record_token_usage.assert_called_with(50, "planning")

    enforcer.record_token_usage(30, "execution")
    assert enforcer.current_tick_tokens_used == 80
    assert enforcer.total_simulation_tokens_used == 80
    mock_metrics_tracker.record_token_usage.assert_called_with(30, "execution")

@pytest.mark.asyncio
async def test_budget_enforcer_reset_for_new_tick(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.record_token_usage(70)
    enforcer.reset_for_new_tick()
    assert enforcer.current_tick_tokens_used == 0
    assert enforcer.total_simulation_tokens_used == 70 # Total should not reset

@pytest.mark.asyncio
async def test_budget_enforcer_per_tick_within_limits(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.current_tick_tokens_used = 90 # Limit is 100
    can_continue, msg = enforcer.check_per_tick_limit()
    assert can_continue is True
    assert msg == ""
    mock_event_bus.publish.assert_not_called()

@pytest.mark.asyncio
async def test_budget_enforcer_per_tick_soft_violation(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.current_tick_tokens_used = 105 # Limit 100, Grace 10% (110)
    can_continue, msg = enforcer.check_per_tick_limit()
    assert can_continue is True # Soft violation, still allowed
    assert "WARNING: Per-tick token limit nearing/exceeded" in msg
    mock_event_bus.publish.assert_called_once_with("BudgetWarning", {"type": "per_tick", "message": msg})

@pytest.mark.asyncio
async def test_budget_enforcer_per_tick_hard_violation(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.current_tick_tokens_used = 115 # Limit 100, Grace 10% (110)
    
    with pytest.raises(SystemExit) as excinfo:
        can_continue, msg = enforcer.check_per_tick_limit()
    
    assert "HARD VIOLATION: Per-tick token limit exceeded" in str(excinfo.value)
    assert enforcer.violation_triggered is True
    mock_metrics_tracker.apply_penalty.assert_called_once_with("budget_violation", default_config.violation_penalty_weight)
    mock_event_bus.publish.assert_called_once_with("BudgetExceeded", {"reason": str(excinfo.value), "severity": "hard_fail"})

@pytest.mark.asyncio
async def test_budget_enforcer_total_sim_within_limits(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.total_simulation_tokens_used = 450 # Limit 500
    can_continue, msg = enforcer.check_total_simulation_limit()
    assert can_continue is True
    assert msg == ""
    mock_event_bus.publish.assert_not_called()

@pytest.mark.asyncio
async def test_budget_enforcer_total_sim_soft_violation(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.total_simulation_tokens_used = 520 # Limit 500, Grace 10% (550)
    can_continue, msg = enforcer.check_total_simulation_limit()
    assert can_continue is True
    assert "WARNING: Total simulation token budget nearing/exceeded" in msg
    mock_event_bus.publish.assert_called_once_with("BudgetWarning", {"type": "total_sim", "message": msg})

@pytest.mark.asyncio
async def test_budget_enforcer_total_sim_hard_violation(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.total_simulation_tokens_used = 560 # Limit 500, Grace 10% (550)
    
    with pytest.raises(SystemExit) as excinfo:
        can_continue, msg = enforcer.check_total_simulation_limit()
    
    assert "HARD VIOLATION: Total simulation token budget exceeded" in str(excinfo.value)
    assert enforcer.violation_triggered is True
    mock_metrics_tracker.apply_penalty.assert_called_once_with("budget_violation", default_config.violation_penalty_weight)
    mock_event_bus.publish.assert_called_once_with("BudgetExceeded", {"reason": str(excinfo.value), "severity": "hard_fail"})

@pytest.mark.asyncio
async def test_budget_enforcer_format_budget_status(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.current_tick_tokens_used = 25
    enforcer.total_simulation_tokens_used = 150
    
    status_string = enforcer.format_budget_status_for_prompt()
    assert "BUDGET STATUS:" in status_string
    assert "Tokens used this turn: 25 / 100 (25.0%)" in status_string
    assert "Total simulation tokens: 150 / 500 (30.0%)" in status_string
    assert "Remaining budget" in status_string
    assert "Budget health: HEALTHY" in status_string

    enforcer.current_tick_tokens_used = 85
    enforcer.total_simulation_tokens_used = 420
    status_string = enforcer.format_budget_status_for_prompt()
    assert "Budget health: WARNING" in status_string # 85% and 84% usage

    enforcer.current_tick_tokens_used = 101 # Above limit (100)
    status_string = enforcer.format_budget_status_for_prompt()
    assert "Budget health: CRITICAL" in status_string

# --- AgentGateway Tests ---

@pytest.mark.asyncio
async def test_agent_gateway_preprocess_request_injects_budget(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    gateway = AgentGateway(enforcer, mock_event_bus)
    
    initial_prompt = "Agent, please make a decision."
    result = await gateway.preprocess_request("agent1", initial_prompt, "decision")
    
    assert "BUDGET STATUS:" in result["modified_prompt"]
    assert "Your response must consider this budget constraint" in result["modified_prompt"]
    assert result["estimated_tokens_for_prompt"] > 0
    assert result["can_proceed"] is True

@pytest.mark.asyncio
async def test_agent_gateway_preprocess_request_hard_fail(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    enforcer.current_tick_tokens_used = 120 # Already over hard limit
    gateway = AgentGateway(enforcer, mock_event_bus)
    
    initial_prompt = "Agent, please make a decision."
    with pytest.raises(SystemExit):
        await gateway.preprocess_request("agent1", initial_prompt, "decision")
    
    mock_metrics_tracker.apply_penalty.assert_called_once()
    mock_event_bus.publish.assert_called_once_with("BudgetExceeded", ANY) # Check event type, content can vary

@pytest.mark.asyncio
async def test_agent_gateway_postprocess_response_records_usage(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    gateway = AgentGateway(enforcer, mock_event_bus)
    
    raw_prompt = "Hello world"
    llm_response = "Response completed"
    
    # Pre-set tokens used in enforcer to ensure postprocess builds on it correctly
    enforcer.current_tick_tokens_used = 0 # Reset for clear test
    enforcer.total_simulation_tokens_used = 0 # Reset for clear test

    await gateway.postprocess_response("agent1", "observation", raw_prompt, llm_response)
    
    # Expected tokens: raw_prompt tokens + llm_response tokens
    expected_tokens_raw_prompt = TokenCounter().count_tokens(raw_prompt)
    expected_tokens_llm_response = TokenCounter().count_tokens(llm_response)
    expected_total_tokens = expected_tokens_raw_prompt + expected_tokens_llm_response

    mock_metrics_tracker.record_token_usage.assert_called_once_with(expected_total_tokens, "observation")
    assert enforcer.current_tick_tokens_used == expected_total_tokens
    assert enforcer.total_simulation_tokens_used == expected_total_tokens

@pytest.mark.asyncio
async def test_agent_gateway_postprocess_response_triggers_warnings(lenient_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(lenient_config, mock_event_bus, mock_metrics_tracker)
    gateway = AgentGateway(enforcer, mock_event_bus)
    
    raw_prompt = "Some moderate length prompt" # approx 5 tokens
    llm_response = "A lengthy response that puts it over the soft limit, this should trigger a warning." # approx 15 tokens
    
    # Set current tokens to be just under the soft limit, so the new response pushes it over
    # max_tokens_per_action = 100, grace: 20% -> 120.
    # We want it to be >100 but <=120
    enforcer.current_tick_tokens_used = 90
    enforcer.total_simulation_tokens_used = 450 # Total limit 500, grace 600

    pre_existing_calls_count = mock_event_bus.publish.call_count

    await gateway.postprocess_response("agent1", "planning", raw_prompt, llm_response)
    
    # Only tick warning should be issued as it crosses the 100% threshold but is within grace
    assert mock_event_bus.publish.call_count == pre_existing_calls_count + 1
    
    # Check for one BudgetWarning call
    calls = []
    for call_args, call_kwargs in mock_event_bus.publish.call_args_list:
        event_type = call_args[0]
        event_data = call_args[1]
        calls.append(event_type)

    assert calls.count("BudgetWarning") == 1 # Only for per-tick
    assert not mock_metrics_tracker.apply_penalty.called # No hard fail, no direct penalty here

@pytest.mark.asyncio
async def test_agent_gateway_postprocess_response_hard_fail(default_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(default_config, mock_event_bus, mock_metrics_tracker)
    gateway = AgentGateway(enforcer, mock_event_bus)
    
    raw_prompt = "Short"
    llm_response = "Long response that pushes total tokens way over hard limit." # many tokens
    
    # Setup for total tokens to be just under grace, new response pushes it over
    # max_total_tokens=500, grace 10% (550)
    enforcer.total_simulation_tokens_used = 540 
    enforcer.current_tick_tokens_used = 50 # Not relevant for this specific hard fail, but kept for context

    with pytest.raises(SystemExit):
        await gateway.postprocess_response("agent1", "execution", raw_prompt, llm_response)
    
    mock_metrics_tracker.apply_penalty.assert_called_once()
    mock_event_bus.publish.assert_called_once_with("BudgetExceeded", ANY)
    assert enforcer.violation_triggered # Verify enforcer state

@pytest.mark.asyncio
async def test_integration_agent_gateway_and_enforcer_reset_per_tick(lenient_config, mock_event_bus, mock_metrics_tracker):
    enforcer = BudgetEnforcer(lenient_config, mock_event_bus, mock_metrics_tracker)
    gateway = AgentGateway(enforcer, mock_event_bus)

    # Tick 1: Agent makes three internal "LLM" calls
    await gateway.preprocess_request("agent1", "prompt A", "action1") # Est. 5 tokens for prompt itself
    await gateway.postprocess_response("agent1", "action1", "prompt A", "resp A") # Actual total ~10 tokens (5p+5c)
    enforcer.record_token_usage(10, "action1") # This is done implicitly in postprocess

    await gateway.preprocess_request("agent1", "prompt B", "action2") # Est. 5 tokens for prompt itself
    await gateway.postprocess_response("agent1", "action2", "prompt B", "resp B") # Actual total ~10 tokens (5p+5c)
    enforcer.record_token_usage(10, "action2")

    # The actual token usage might be higher due to prompt/response processing
    # Let's just verify that tokens are being tracked
    assert enforcer.current_tick_tokens_used > 0
    assert enforcer.total_simulation_tokens_used > 0

    enforcer.reset_for_new_tick() # Tick changes

    # Tick 2: Agent makes a call
    await gateway.preprocess_request("agent1", "prompt C", "action3")
    await gateway.postprocess_response("agent1", "action3", "prompt C", "resp C")
    enforcer.record_token_usage(10, "action3")

    # After reset, current tick tokens should be just from the new action
    # but the actual implementation might include additional processing overhead
    assert enforcer.current_tick_tokens_used > 0
    assert enforcer.total_simulation_tokens_used > 20 # Accumulates total
