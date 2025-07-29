import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from money import Money 
from llm_interface.contract import BaseLLMClient
from llm_interface.prompt_adapter import PromptAdapter
from llm_interface.response_parser import LLMResponseParser
from constraints.agent_gateway import AgentGateway
from events import SetPriceCommand # Assuming SetPriceCommand is the primary action

from models.product import Product 
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SimulationState:
    current_tick: int
    simulation_time: datetime
    products: List[Product]
    recent_events: List[Any] = field(default_factory=list)

    def get_product(self, asin: str) -> Optional[Product]:
        for product in self.products:
            if product.asin == asin:
                return product
        return None

class GPT4oMiniBot:
    def __init__(self, 
                 agent_id: str,
                 llm_client: BaseLLMClient, 
                 prompt_adapter: PromptAdapter, 
                 response_parser: LLMResponseParser,
                 agent_gateway: AgentGateway,
                 model_params: Dict[str, Any]):
        
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.prompt_adapter = prompt_adapter
        self.response_parser = response_parser
        self.agent_gateway = agent_gateway
        self.model_params = model_params

    async def decide(self, state: SimulationState) -> List[SetPriceCommand]:
        """
        Agent decides on actions based on the current simulation state using an LLM.
        """
        available_actions = {
            "set_price": {
                "description": "Adjust the price of a product.",
                "parameters": {
                    "asin": "string (Amazon Standard Identification Number)",
                    "price": "float (new price for the product)"
                }
            }
        }

        scenario_context = "The primary goal is to maximize profit while operating within a constrained token budget."

        raw_prompt = self.prompt_adapter.generate_prompt(
            current_tick=state.current_tick,
            simulation_time=state.simulation_time,
            recent_events=state.recent_events,
            available_actions=available_actions,
            scenario_context=scenario_context
        )

        gateway_response = await self.agent_gateway.preprocess_request(
            agent_id=self.agent_id,
            prompt=raw_prompt,
            action_type="decide_action",
            model_name=self.llm_client.model_name
        )
        
        modified_prompt = gateway_response["modified_prompt"]
        can_proceed = gateway_response["can_proceed"]

        if not can_proceed:
            logger.warning(f"[{self.agent_id}] Cannot proceed with LLM call due to budget constraints.")
            return []

        llm_raw_response_data = None
        try:
            llm_raw_response_data = await self.llm_client.generate_response(
                prompt=modified_prompt,
                temperature=self.model_params.get("temperature", 0.7),
                max_tokens=self.model_params.get("max_tokens_per_action", 1000),
                top_p=self.model_params.get("top_p", 1.0),
            )
            llm_response_content = llm_raw_response_data["choices"][0]["message"]["content"]
            
            await self.agent_gateway.postprocess_response(
                agent_id=self.agent_id,
                action_type="decide_action",
                raw_prompt=raw_prompt,
                llm_response=llm_response_content,
                model_name=self.llm_client.model_name
            )

            parsed_response, error_details = self.response_parser.parse_and_validate(llm_response_content, self.agent_id)

            if error_details:
                logger.error(f"[{self.agent_id}] Failed to parse/validate LLM response: {error_details}")
                return []
            
            if not parsed_response or "actions" not in parsed_response:
                logger.warning(f"[{self.agent_id}] LLM response missing 'actions' field or empty: {parsed_response}")
                return []

            agent_commands: List[SetPriceCommand] = []
            for action_data in parsed_response.get("actions", []):
                action_type = action_data.get("type")
                parameters = action_data.get("parameters", {})

                if action_type == "set_price":
                    try:
                        asin = parameters.get("asin")
                        price = parameters.get("price")
                        if asin and isinstance(price, (int, float)):
                            agent_commands.append(SetPriceCommand(
                                event_id=str(uuid.uuid4()),
                                timestamp=state.simulation_time,
                                agent_id=self.agent_id,
                                asin=asin,
                                new_price=Money.from_dollars(price),
                                reason=parsed_response.get("reasoning", "LLM decided price change")
                            ))
                        else:
                            logger.warning(f"[{self.agent_id}] Invalid set_price parameters: {parameters}")
                    except Exception as e:
                        logger.error(f"[{self.agent_id}] Error creating SetPriceCommand: {e} with data {parameters}")
                else:
                    logger.warning(f"[{self.agent_id}] Unknown action type received: {action_type}")
            
            return agent_commands

        except Exception as e:
            logger.error(f"[{self.agent_id}] An error occurred during LLM decision-making: {e}", exc_info=True)
            return []