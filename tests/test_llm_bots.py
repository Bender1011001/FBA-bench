import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
import uuid

from money import Money
from llm_interface.contract import BaseLLMClient
from llm_interface.prompt_adapter import PromptAdapter
from llm_interface.response_parser import LLMResponseParser
from constraints.agent_gateway import AgentGateway
from events import SetPriceCommand # Ensure this is actual path
from models.product import Product # Ensure this is actual path

# Import the LLM bots and their SimulationState
from baseline_bots.gpt_3_5_bot import GPT35Bot, SimulationState as LLMSimulationState
from baseline_bots.gpt_4o_mini_bot import GPT4oMiniBot
from baseline_bots.grok_4_bot import Grok4Bot
from baseline_bots.claude_sonnet_bot import ClaudeSonnetBot

# --- Mocks for Dependencies ---

# Mock WorldStore for PromptAdapter
class MockWorldStore:
    def get_all_product_states(self):
        # Return a dummy product state for prompt generation
        # NOTE: ProductState is internal to WorldStore, need to mock its structure if used directly
        # For simplicity, returning a dict that `_get_product_portfolio_summary` can process
        # It expects product.price.amount and product.inventory_quantity, product.cost_basis.amount
        class MockProductState:
            def __init__(self, asin, price, inventory_quantity, cost_basis):
                self.asin = asin
                self.price = Money.from_dollars(price)
                self.inventory_quantity = inventory_quantity
                self.cost_basis = Money.from_dollars(cost_basis)

        return {
            "B0EXAMPLE01": MockProductState("B0EXAMPLE01", 20.0, 100, 10.0),
            "B0EXAMPLE02": MockProductState("B0EXAMPLE02", 50.0, 50, 25.0)
        }

# Mock BudgetEnforcer for PromptAdapter and AgentGateway
class MockBudgetEnforcer:
    def __init__(self, max_tokens_per_action=1000, max_total_tokens=10000, token_cost_per_1k=0.01):
        self.max_tokens_per_action = max_tokens_per_action
        self.max_total_tokens = max_total_tokens
        self.token_cost_per_1k = token_cost_per_1k
        self.current_tick_tokens_used = 0
        self.total_simulation_tokens_used = 0
        self.config = MagicMock()
        self.config.inject_budget_status = True
        self.config.hard_fail_on_violation = True # For testing hard stops

    def format_budget_status_for_prompt(self):
        return "BUDGET STATUS: Mocked Status"

    def record_token_usage(self, tokens, action_type="general"):
        self.current_tick_tokens_used += tokens
        self.total_simulation_tokens_used += tokens

    def check_per_tick_limit(self) -> (bool, str):
        if self.current_tick_tokens_used > self.max_tokens_per_action:
            if self.config.hard_fail_on_violation:
                raise SystemExit("Mock hard fail tick")
            return False, "Mock per-tick soft warning"
        return True, ""

    def check_total_simulation_limit(self) -> (bool, str):
        if self.total_simulation_tokens_used > self.max_total_tokens:
            if self.config.hard_fail_on_violation:
                raise SystemExit("Mock hard fail total")
            return False, "Mock total soft warning"
        return True, ""

# Mock TrustMetrics for LLMResponseParser
class MockTrustMetrics:
    def apply_penalty(self, agent_id: str, penalty_amount: float, reason: str):
        pass # Do nothing for tests

@pytest.fixture
def mock_llm_client():
    client = AsyncMock(spec=BaseLLMClient)
    # Mock 'generate_response' to return a valid JSON string as content
    client.generate_response.return_value = {
        "choices": [
            {"message": {"content": '{"actions": [{"type": "set_price", "parameters": {"asin": "B0EXAMPLE01", "price": 24.99}}], "reasoning": "Test price adjustment", "confidence": 0.9}'}}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50}
    }
    client.get_token_count.return_value = 10 # Dummy token count
    return client

@pytest.fixture
def mock_prompt_adapter():
    world_store = MockWorldStore()
    budget_enforcer = MockBudgetEnforcer()
    return PromptAdapter(world_store, budget_enforcer)

@pytest.fixture
def mock_response_parser():
    trust_metrics = MockTrustMetrics()
    return LLMResponseParser(trust_metrics)

@pytest.fixture
def mock_agent_gateway():
    # AgentGateway needs an event bus, but we can mock it here or pass a dummy
    mock_event_bus = MagicMock()
    mock_event_bus.publish = AsyncMock() # Ensure publish is async mock
    budget_enforcer = MockBudgetEnforcer()
    return AgentGateway(budget_enforcer, mock_event_bus)

@pytest.fixture
def sample_llm_simulation_state():
    # A realistic-ish product for LLM consumption
    product1 = Product(
        asin="B0EXAMPLE01",
        category="electronics",
        cost=Money.from_dollars(10.0),
        price=Money.from_dollars(20.0),
        base_demand=100.0,
        inventory_units=50,
        # competitor_prices not directly in Product; simulate its presence if needed by prompt adapter
        # Assuming prompt adapter extracts this from broader simulation state or world store
    )
    product2 = Product(
        asin="B0EXAMPLE02",
        category="home",
        cost=Money.from_dollars(5.0),
        price=Money.from_dollars(12.0),
        base_demand=80.0,
        inventory_units=30,
    )
    return LLMSimulationState(
        products=[product1, product2],
        current_tick=5,
        simulation_time=datetime.utcnow(),
        recent_events=[] # For simplicity
    )

class TestLLMBots:
    """
    Tests for LLM-based bots, focusing on their interaction with LLM client,
    prompt adapter, response parser, and agent gateway.
    """

    @pytest.mark.asyncio
    async def test_gpt35_bot_decide_success(
        self, 
        mock_llm_client, 
        mock_prompt_adapter, 
        mock_response_parser, 
        mock_agent_gateway, 
        sample_llm_simulation_state
    ):
        bot = GPT35Bot(
            agent_id="test-gpt35-bot",
            llm_client=mock_llm_client,
            prompt_adapter=mock_prompt_adapter,
            response_parser=mock_response_parser,
            agent_gateway=mock_agent_gateway,
            model_params={"temperature": 0.1, "max_tokens_per_action": 1000}
        )

        actions = await bot.decide(sample_llm_simulation_state)

        # Assert LLM client was called with the modified prompt
        mock_llm_client.generate_response.assert_awaited_once()
        call_args, call_kwargs = mock_llm_client.generate_response.call_args
        assert "BUDGET STATUS: Mocked Status" in call_kwargs['prompt'] # From agent_gateway injection
        assert call_kwargs['temperature'] == 0.1
        assert call_kwargs['max_tokens'] == 1000

        # Assert agent gateway was used for preprocessing and postprocessing
        mock_agent_gateway.preprocess_request.assert_awaited_once()
        mock_agent_gateway.postprocess_response.assert_awaited_once()

        # Assert actions were parsed correctly
        assert len(actions) == 1
        assert isinstance(actions[0], SetPriceCommand)
        assert actions[0].asin == "B0EXAMPLE01"
        assert actions[0].new_price.to_float() == 24.99
        assert actions[0].agent_id == "test-gpt35-bot"
        assert actions[0].reason == "Test price adjustment"

    @pytest.mark.asyncio
    async def test_gpt4o_mini_bot_decide_no_actions_on_parsing_error(
        self, 
        mock_llm_client, 
        mock_prompt_adapter, 
        mock_response_parser, 
        mock_agent_gateway, 
        sample_llm_simulation_state
    ):
        # Configure mock LLM client to return invalid JSON
        mock_llm_client.generate_response.return_value = {
            "choices": [
                {"message": {"content": '{"actions_bad": [], "reasoning": "invalid json"}'}} # Valid JSON but schema violation
            ]
        }
        
        bot = GPT4oMiniBot(
            agent_id="test-gpt4o-bot",
            llm_client=mock_llm_client,
            prompt_adapter=mock_prompt_adapter,
            response_parser=mock_response_parser,
            agent_gateway=mock_agent_gateway,
            model_params={"temperature": 0.2, "max_tokens_per_action": 500}
        )

        actions = await bot.decide(sample_llm_simulation_state)

        # Should return no actions due to parsing/validation error
        assert len(actions) == 0
        mock_response_parser.parse_and_validate.assert_called_once()
        
        # Ensure trust penalty was applied
        mock_response_parser.trust_metrics.apply_penalty.assert_called_once()

    @pytest.mark.asyncio
    async def test_grok4_bot_decide_budget_exceeded_before_llm_call(
        self, 
        mock_llm_client, 
        mock_prompt_adapter, 
        mock_response_parser, 
        sample_llm_simulation_state
    ):
        # Configure agent gateway to hard fail on preprocess
        mock_gateway = MagicMock(spec=AgentGateway)
        mock_gateway.preprocess_request = AsyncMock(side_effect=SystemExit("Mock hard fail during preprocess"))
        mock_gateway.postprocess_response = AsyncMock() # This should not be called

        bot = Grok4Bot(
            agent_id="test-grok4-bot",
            llm_client=mock_llm_client,
            prompt_adapter=mock_prompt_adapter,
            response_parser=mock_response_parser,
            agent_gateway=mock_gateway,
            model_params={"temperature": 0.5, "max_tokens_per_action": 2000}
        )

        actions = await bot.decide(sample_llm_simulation_state)

        # Should return no actions because of SystemExit, and LLM not called
        assert len(actions) == 0
        mock_gateway.preprocess_request.assert_awaited_once() # Should be called
        mock_llm_client.generate_response.assert_not_awaited() 
        mock_gateway.postprocess_response.assert_not_awaited() # Should not be called
        
    @pytest.mark.asyncio
    async def test_claude_sonnet_bot_decide_empty_response(
        self, 
        mock_llm_client, 
        mock_prompt_adapter, 
        mock_response_parser, 
        mock_agent_gateway, 
        sample_llm_simulation_state
    ):
        # Configure mock LLM client to return empty actions
        mock_llm_client.generate_response.return_value = {
            "choices": [{"message": {"content": '{"actions": [], "reasoning": "no actions", "confidence": 0.5}'}}]
        }

        bot = ClaudeSonnetBot(
            agent_id="test-claude-bot",
            llm_client=mock_llm_client,
            prompt_adapter=mock_prompt_adapter,
            response_parser=mock_response_parser,
            agent_gateway=mock_agent_gateway,
            model_params={"temperature": 0.3, "max_tokens_per_action": 3000, "top_p": 0.9}
        )

        actions = await bot.decide(sample_llm_simulation_state)

        assert len(actions) == 0 # No actions should be returned
        mock_llm_client.generate_response.assert_awaited_once()
        mock_response_parser.parse_and_validate.assert_called_once()
        # No penalty should be applied for valid but empty response
        mock_response_parser.trust_metrics.apply_penalty.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_llm_bots_use_correct_model_params(self):
        # This test ensures that different bots pass their specific model_params correctly.
        
        # Setup common mocks
        mock_llm_client_generic = AsyncMock(spec=BaseLLMClient)
        mock_llm_client_generic.generate_response.return_value = {
            "choices": [
                {"message": {"content": '{"actions": [], "reasoning": "", "confidence": 0.5}'}}
            ]
        }
        mock_llm_client_generic.get_token_count.return_value = 10

        mock_prompt_adapter_generic = MockPromptAdapter(MockWorldStore(), MockBudgetEnforcer())
        mock_response_parser_generic = MockLLMResponseParser(MockTrustMetrics())
        mock_agent_gateway_generic = MockAgentGateway()
        sample_state = sample_llm_simulation_state()

        # Test GPT-3.5 Bot
        gpt35_params = {"temperature": 0.1, "max_tokens_per_action": 8000, "top_p": 1.0}
        gpt35_bot = GPT35Bot("gpt35", mock_llm_client_generic, mock_prompt_adapter_generic, 
                             mock_response_parser_generic, mock_agent_gateway_generic, gpt35_params)
        await gpt35_bot.decide(sample_state)
        call_args, call_kwargs = mock_llm_client_generic.generate_response.call_args
        assert call_kwargs['temperature'] == 0.1
        assert call_kwargs['max_tokens'] == 8000
        assert call_kwargs['top_p'] == 1.0

        # Reset mock for next bot
        mock_llm_client_generic.generate_response.reset_mock()

        # Test GPT-4o mini-budget Bot
        gpt4o_params = {"temperature": 0.2, "max_tokens_per_action": 1000, "top_p": 1.0}
        gpt4o_bot = GPT4oMiniBot("gpt4o", mock_llm_client_generic, mock_prompt_adapter_generic,
                                 mock_response_parser_generic, mock_agent_gateway_generic, gpt4o_params)
        await gpt4o_bot.decide(sample_state)
        call_args, call_kwargs = mock_llm_client_generic.generate_response.call_args
        assert call_kwargs['temperature'] == 0.2
        assert call_kwargs['max_tokens'] == 1000
        assert call_kwargs['top_p'] == 1.0

        # Reset mock for next bot
        mock_llm_client_generic.generate_response.reset_mock()

        # Test Grok-4 Bot
        grok4_params = {"temperature": 0.5, "max_tokens_per_action": 16000, "top_p": 1.0}
        grok4_bot = Grok4Bot("grok4", mock_llm_client_generic, mock_prompt_adapter_generic,
                             mock_response_parser_generic, mock_agent_gateway_generic, grok4_params)
        await grok4_bot.decide(sample_state)
        call_args, call_kwargs = mock_llm_client_generic.generate_response.call_args
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 16000
        assert call_kwargs['top_p'] == 1.0

        # Reset mock for next bot
        mock_llm_client_generic.generate_response.reset_mock()

        # Test Claude 3.5 Sonnet Bot
        claude_params = {"temperature": 0.3, "max_tokens_per_action": 32000, "top_p": 0.9}
        claude_bot = ClaudeSonnetBot("claude", mock_llm_client_generic, mock_prompt_adapter_generic,
                                     mock_response_parser_generic, mock_agent_gateway_generic, claude_params)
        await claude_bot.decide(sample_state)
        call_args, call_kwargs = mock_llm_client_generic.generate_response.call_args
        assert call_kwargs['temperature'] == 0.3
        assert call_kwargs['max_tokens'] == 32000
        assert call_kwargs['top_p'] == 0.9
