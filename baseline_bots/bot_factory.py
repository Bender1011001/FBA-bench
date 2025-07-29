import os
import yaml
import inspect
from typing import Dict, Any, Type, Optional

from llm_interface.contract import BaseLLMClient
from llm_interface.openrouter_client import OpenRouterClient
from llm_interface.prompt_adapter import PromptAdapter
from llm_interface.response_parser import LLMResponseParser

from constraints.budget_enforcer import BudgetEnforcer
from constraints.agent_gateway import AgentGateway

from services.world_store import WorldStore
from metrics.trust_metrics import TrustMetrics # Assuming this will be used by ResponseParser

# Import bot implementations
from baseline_bots.greedy_script_bot import GreedyScriptBot, SimulationState as GreedySimulationState
from baseline_bots.gpt_3_5_bot import GPT35Bot, SimulationState as LLMSimulationState
from baseline_bots.gpt_4o_mini_bot import GPT4oMiniBot
from baseline_bots.grok_4_bot import Grok4Bot
from baseline_bots.claude_sonnet_bot import ClaudeSonnetBot

class BotFactory:
    def __init__(self, 
                 config_dir: str = "baseline_bots/configs",
                 world_store: Optional[WorldStore] = None,
                 budget_enforcer: Optional[BudgetEnforcer] = None,
                 trust_metrics: Optional[TrustMetrics] = None, # Used by LLMResponseParser
                 agent_gateway: Optional[AgentGateway] = None,
                 openrouter_api_key: Optional[str] = None):
        
        self.config_dir = os.path.join(os.getcwd(), config_dir)
        self.world_store = world_store
        self.budget_enforcer = budget_enforcer
        self.trust_metrics = trust_metrics
        self.agent_gateway = agent_gateway
        self.openrouter_api_key = openrouter_api_key

        self.bot_configs: Dict[str, Dict[str, Any]] = self._load_bot_configs()
        self.bot_classes: Dict[str, Type] = {
            "GreedyScript": GreedyScriptBot,
            "GPT-3.5": GPT35Bot,
            "GPT-4o mini-budget": GPT4oMiniBot,
            "Grok-4": Grok4Bot,
            "Claude 3.5 Sonnet": ClaudeSonnetBot,
        }

    def _load_bot_configs(self) -> Dict[str, Dict[str, Any]]:
        configs = {}
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".yaml"):
                filepath = os.path.join(self.config_dir, filename)
                with open(filepath, 'r') as f:
                    config = yaml.safe_load(f)
                    bot_name = config.get("bot_name")
                    if bot_name:
                        configs[bot_name] = config
        return configs

    def create_bot(self, bot_name: str, tier: str):
        config = self.bot_configs.get(bot_name)
        if not config:
            raise ValueError(f"Bot configuration not found for: {bot_name}")

        bot_class = self.bot_classes.get(bot_name)
        if not bot_class:
            raise ValueError(f"Bot class not found for: {bot_name}")

        tier_config = config.get("tier_configs", {}).get(tier)
        if not tier_config:
            raise ValueError(f"Tier configuration '{tier}' not found for bot: {bot_name}")

        # Common parameters for LLM bots
        llm_client = None
        prompt_adapter = None
        response_parser = None
        agent_gateway = None

        if issubclass(bot_class, (GPT35Bot, GPT4oMiniBot, Grok4Bot, ClaudeSonnetBot)):
            # LLM-based bot, ensure dependencies are available
            if not self.world_store or not self.budget_enforcer or not self.trust_metrics:
                raise ValueError("WorldStore, BudgetEnforcer, and TrustMetrics instances are required for LLM-based bots.")
            
            # Re-initialize AgentGateway if not already provided or if it needs specific WorldStore/BudgetEnforcer
            # For now, let's assume it's passed or can be instantiated simply if dependencies are there.
            if not self.agent_gateway:
                # If agent_gateway is not explicitly passed, create a new one.
                # Note: This might lead to multiple gateway instances per simulation depending on how it's used.
                # Ideally, a single AgentGateway instance is managed by the simulation orchestrator.
                # For this factory, we'll ensure it has the core dependencies required.
                from event_bus import get_event_bus # Assuming event_bus is accessible
                agent_gateway = AgentGateway(self.budget_enforcer, get_event_bus())
            else:
                agent_gateway = self.agent_gateway

            # LLM Client
            llm_client = OpenRouterClient(
                model_name=config["model"], 
                api_key=self.openrouter_api_key
            )
            
            # Prompt Adapter
            prompt_adapter = PromptAdapter(self.world_store, self.budget_enforcer)
            
            # Response Parser
            response_parser = LLMResponseParser(self.trust_metrics) # Requires TrustMetrics

            # Model specific parameters for LLM bots
            model_params = {
                "max_tokens_per_action": tier_config.get("max_tokens_per_action"),
                "temperature": tier_config.get("temperature"),
                "top_p": tier_config.get("top_p", 1.0) # Default for top_p if not in config
            }

            # Instantiate LLM bots with their specific dependencies
            if bot_class == GPT35Bot:
                return GPT35Bot(bot_name, llm_client, prompt_adapter, response_parser, agent_gateway, model_params)
            elif bot_class == GPT4oMiniBot:
                return GPT4oMiniBot(bot_name, llm_client, prompt_adapter, response_parser, agent_gateway, model_params)
            elif bot_class == Grok4Bot:
                return Grok4Bot(bot_name, llm_client, prompt_adapter, response_parser, agent_gateway, model_params)
            elif bot_class == ClaudeSonnetBot:
                return ClaudeSonnetBot(bot_name, llm_client, prompt_adapter, response_parser, agent_gateway, model_params)

        elif bot_class == GreedyScriptBot:
            # GreedyScriptBot does not need LLM-related dependencies
            reorder_threshold = tier_config.get("reorder_threshold", 10) # Example default
            reorder_quantity = tier_config.get("reorder_quantity", 50) # Example default
            return GreedyScriptBot(reorder_threshold=reorder_threshold, reorder_quantity=reorder_quantity)
        
        else:
            raise NotImplementedError(f"Bot type {bot_name} not supported by factory.")

# Example usage (for testing/demonstration)
# if __name__ == "__main__":
#     # These would typically be provided by the main simulation orchestrator
#     from services.world_store import get_world_store, set_world_store
#     from constraints.budget_enforcer import BudgetEnforcer
#     from metrics.trust_metrics import TrustMetrics
#     from event_bus import EventBus
#     import asyncio

#     # Initialize core simulation components (mock or real)
#     mock_event_bus = EventBus() # Requires an event loop
#     set_world_store(WorldStore(event_bus=mock_event_bus))
#     world_store_instance = get_world_store()
#     budget_enforcer_instance = BudgetEnforcer.from_tier_config("T0", event_bus=mock_event_bus)
#     trust_metrics_instance = TrustMetrics(trust_score_service=None) # Mock TrustScoreService for now
#     agent_gateway_instance = AgentGateway(budget_enforcer_instance, mock_event_bus)

#     try:
#         # Ensure configs directory exists for testing factory
#         os.makedirs("baseline_bots/configs", exist_ok=True)
#         # Create dummy config files for testing factory directly
#         with open("baseline_bots/configs/dummy_greedy_config.yaml", "w") as f:
#             f.write("bot_name: GreedyScript\n")
#             f.write("expected_score: 10\n")
#             f.write("tier_configs:\n  T0:\n    reorder_threshold: 10\n")

#         with open("baseline_bots/configs/dummy_gpt_config.yaml", "w") as f:
#             f.write("bot_name: GPT-3.5\n")
#             f.write("model: openai/gpt-3.5-turbo\n")
#             f.write("expected_score: 35\n")
#             f.write("tier_configs:\n  T0:\n    max_tokens_per_action: 8000\n    temperature: 0.1\n")
            
#         factory = BotFactory(
#             world_store=world_store_instance,
#             budget_enforcer=budget_enforcer_instance,
#             trust_metrics=trust_metrics_instance,
#             agent_gateway=agent_gateway_instance,
#             openrouter_api_key="sk-test-key" #dummy
#         )

#         greedy_bot = factory.create_bot("GreedyScript", "T0")
#         print(f"Created: {greedy_bot.agent_id}")

#         # Mock a simulation state for decide method
#         from money import Money
#         from datetime import datetime
#         # Creating a dummy Product for simulation state
#         dummy_product = Product(
#             asin="B00000000A",
#             category="electronics",
#             cost=Money.from_dollars(50.00),
#             price=Money.from_dollars(100.00),
#             base_demand=10.0,
#             competitor_prices=[("C1", Money.from_dollars(95.00)), ("C2", Money.from_dollars(98.00))], # Example competitor prices
#             inventory_units=15,
#             trust_score=0.9
#         )
#         # Update product class to accept competitor prices in __init__ for this mock
#         # Original models/product.py does not have competitor_prices. Adding it temporarily if needed.
#         # For now, just bypass for GreedyScript example.
        
#         # For GreedyScript, we need a SimulationState that contains products with competitor_prices
#         # As product.py doesn't have competitor_prices, assuming mock data structure or a more complete SimulationState class
#         # will be used by the orchestrator. For this example, let's inject a simplified structure.
#         class MockGreedyProduct:
#             def __init__(self, asin, price, cost, inventory_units, competitor_prices):
#                 self.asin = asin
#                 self.price = price
#                 self.cost = cost
#                 self.inventory_units = inventory_units
#                 self.competitor_prices = competitor_prices # Tuple of (asin, Money)
            
#         mock_greedy_products = [
#             MockGreedyProduct(
#                 asin="B07XAMPLE", 
#                 price=Money.from_dollars(25.00), 
#                 cost=Money.from_dollars(15.00), 
#                 inventory_units=20, 
#                 competitor_prices=[("COMP1", Money.from_dollars(24.00)), ("COMP2", Money.from_dollars(26.00))]
#             )
#         ]
#         mock_greedy_state = GreedySimulationState(
#             products=mock_greedy_products,
#             current_tick=1,
#             simulation_time=datetime.now()
#         )
#         greedy_actions = greedy_bot.decide(mock_greedy_state)
#         print(f"GreedyScriptBot actions: {greedy_actions}")


#         if os.getenv("RUN_LLM_TEST", "false").lower() == "true": # Only run if env var is set
#             # Dummy setup for LLM bot
#             llm_product = Product(
#                 asin="B00000000B",
#                 category="books",
#                 cost=Money.from_dollars(5.00),
#                 price=Money.from_dollars(10.00),
#                 base_demand=5.0,
#                 inventory_units=50,
#                 trust_score=0.8
#             )
#             llm_products = [llm_product]
#             llm_state = LLMSimulationState(
#                 products=llm_products,
#                 current_tick=1,
#                 simulation_time=datetime.now(),
#                 recent_events=[] # Empty for example
#             )

#             # Create an LLM bot
#             gpt_bot = factory.create_bot("GPT-3.5", "T0")
#             print(f"Created: {gpt_bot.agent_id}")

#             print("Attempting LLM bot decision (this will call OpenRouter API)...")
#             # Run async decide method
#             async def run_decide():
#                 llm_actions = await gpt_bot.decide(llm_state)
#                 print(f"GPT-3.5 Bot actions: {llm_actions}")

#             asyncio.run(run_decide())

#     except ValueError as e:
#         print(f"Error: {e}")
#     except FileNotFoundError as e:
#         print(f"Configuration Error: {e}. Please ensure config files are in 'baseline_bots/configs'.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}", exc_info=True)
#     finally:
#         # Clean up dummy config files
#         if os.path.exists("baseline_bots/configs/dummy_greedy_config.yaml"):
#             os.remove("baseline_bots/configs/dummy_greedy_config.yaml")
#         if os.path.exists("baseline_bots/configs/dummy_gpt_config.yaml"):
#             os.remove("baseline_bots/configs/dummy_gpt_config.yaml")