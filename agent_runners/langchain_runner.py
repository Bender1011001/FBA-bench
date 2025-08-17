"""
LangChain Agent Runner for FBA-Bench.

This module implements the AgentRunner interface for LangChain agents,
enabling them to participate in the benchmarking system.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base_runner import AgentRunner, AgentRunnerStatus, AgentRunnerError, AgentRunnerInitializationError, AgentRunnerDecisionError
from benchmarking.config.pydantic_config import FrameworkType, LLMConfig, MemoryConfig
from pydantic import ValidationError
from fba_bench.core.llm_outputs import FbaDecision

logger = logging.getLogger(__name__)


class LangChainRunner(AgentRunner):
    """
    Agent runner for LangChain agents.
    
    This class integrates LangChain agents into the FBA-Bench system,
    allowing them to make pricing decisions in the simulation.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the LangChain agent runner."""
        super().__init__(agent_id, config)
        self.agent = None
        self.llm = None
        self.memory = None
        self.llm_config = None
        self.memory_config = None
        self.tools = []
        
    def _do_initialize(self) -> None:
        """Initialize the LangChain agent and its components."""
        try:
            # Extract configurations
            self.llm_config = self._extract_llm_config()
            self.memory_config = self._extract_memory_config()
            
            # Create the LLM
            self._create_llm()
            
            # Create the memory
            self._create_memory()
            
            # Create tools
            self._create_tools()
            
            # Create the agent
            self._create_agent()
            
            logger.info(f"LangChain agent runner {self.agent_id} initialized successfully")
            
        except ImportError as e:
            raise AgentRunnerInitializationError(
                f"LangChain not available: {e}. Install with: pip install langchain",
                agent_id=self.agent_id,
                framework="LangChain"
            ) from e
        except Exception as e:
            raise AgentRunnerInitializationError(
                f"Failed to initialize LangChain agent {self.agent_id}: {e}",
                agent_id=self.agent_id,
                framework="LangChain"
            ) from e
    
    def _extract_llm_config(self) -> LLMConfig:
        """Extract LLM configuration from the agent config."""
        llm_config_dict = self.config.get('llm_config', {})
        
        # Create LLMConfig with defaults
        return LLMConfig(
            name=f"{self.agent_id}_llm",
            model=llm_config_dict.get('model', 'gpt-4'),
            api_key=llm_config_dict.get('api_key'),
            base_url=llm_config_dict.get('base_url'),
            max_tokens=llm_config_dict.get('max_tokens', 2048),
            temperature=llm_config_dict.get('temperature', 0.7),
            top_p=llm_config_dict.get('top_p', 1.0),
            timeout=llm_config_dict.get('timeout', 30),
            max_retries=llm_config_dict.get('max_retries', 3)
        )
    
    def _extract_memory_config(self) -> MemoryConfig:
        """Extract Memory configuration from the agent config."""
        memory_config_dict = self.config.get('memory_config', {})
        
        # Create MemoryConfig with defaults
        return MemoryConfig(
            name=f"{self.agent_id}_memory",
            type=memory_config_dict.get('type', 'buffer'),
            window_size=memory_config_dict.get('window_size', 10),
            max_tokens=memory_config_dict.get('max_tokens')
        )
    
    def _create_llm(self) -> None:
        """Create the LangChain LLM with JSON structured output enabled."""
        from langchain_openai import ChatOpenAI

        # Create the LLM
        self.llm = ChatOpenAI(
            model=self.llm_config.model,
            api_key=self.llm_config.api_key.get_secret_value() if self.llm_config.api_key else None,
            base_url=self.llm_config.base_url,
            max_tokens=self.llm_config.max_tokens,
            temperature=self.llm_config.temperature,
            # Enforce JSON-object responses to align with Pydantic schema parsing
            model_kwargs={
                "top_p": self.llm_config.top_p,
                "request_timeout": self.llm_config.timeout,
                "response_format": {"type": "json_object"},
            },
        )

        logger.debug(f"Created LangChain LLM with model: {self.llm_config.model}")
    
    def _create_memory(self) -> None:
        """Create the LangChain memory."""
        from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
        
        if self.memory_config.type == "buffer":
            self.memory = ConversationBufferMemory(
                k=self.memory_config.window_size,
                return_messages=True
            )
        elif self.memory_config.type == "summary":
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                max_token_limit=self.memory_config.max_tokens or 1000
            )
        else:
            # Default to buffer memory
            self.memory = ConversationBufferMemory(
                k=self.memory_config.window_size,
                return_messages=True
            )
        
        logger.debug(f"Created LangChain memory with type: {self.memory_config.type}")
    
    def _create_tools(self) -> None:
        """Create tools for the LangChain agent."""
        from langchain.tools import Tool
        
        # Create a pricing tool
        pricing_tool = Tool(
            name="set_product_price",
            func=lambda x: "Price set successfully",
            description="Set the price for a product. Input should be a JSON string with 'asin' and 'price' fields."
        )
        
        # Create a market analysis tool
        market_analysis_tool = Tool(
            name="analyze_market",
            func=lambda x: "Market analysis complete",
            description="Analyze market conditions for a product. Input should be a JSON string with product details."
        )
        
        self.tools = [pricing_tool, market_analysis_tool]
        
        logger.debug(f"Created {len(self.tools)} tools for LangChain agent")
    
    def _create_agent(self) -> None:
        """Create the LangChain agent."""
        from langchain.agents import AgentType, initialize_agent
        
        # Get agent-specific parameters
        agent_params = self.config.get('parameters', {})
        
        # Create the agent
        schema_text = json.dumps(FbaDecision.model_json_schema(), indent=2)
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            memory=self.memory,
            agent_kwargs={
                "prefix": agent_params.get(
                    'prefix',
                    "You are an FBA pricing expert. Your goal is to optimize product pricing for maximum profit."
                ),
                "suffix": agent_params.get(
                    'suffix',
                    "Respond ONLY with a valid JSON object that conforms exactly to this schema:\n"
                    f"{schema_text}\n"
                    "Do not include any prose or explanation outside the JSON. "
                    "Use floating-point dollars for prices (e.g., 19.99)."
                ),
            },
        )

        logger.debug("Created LangChain agent")
    
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a pricing decision using the LangChain agent.
        
        Args:
            context: Context information including market state and products
            
        Returns:
            Dictionary containing the decision and metadata
        """
        try:
            # Update context
            self.update_context(context)
            
            # Format the input for the agent
            input_text = self._format_agent_input(context)
            
            # Execute the agent
            result = self.agent.run(input_text)
            
            # Parse and validate structured output using Pydantic
            decision = self._parse_langchain_result(result)
            
            # Update metrics
            self.update_metrics({
                'decision_timestamp': datetime.now().isoformat(),
                'decision_type': 'pricing',
                'raw_result': str(result),
                'parsed_decision': decision
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in LangChain decision making: {e}")
            raise AgentRunnerDecisionError(
                f"Decision making failed for LangChain agent {self.agent_id}: {e}",
                agent_id=self.agent_id,
                framework="LangChain"
            ) from e
    
    def _format_agent_input(self, context: Dict[str, Any]) -> str:
        """Format the input for the LangChain agent."""
        # Extract relevant information from context
        products = context.get('products', [])
        market_conditions = context.get('market_conditions', {})
        recent_events = context.get('recent_events', [])
        tick = context.get('tick', 0)
        
        # Format the input
        input_text = f"You are an FBA pricing expert. Analyze the following data and make optimal pricing decisions for tick {tick}:\n\n"
        
        # Add products information
        if products:
            input_text += "PRODUCTS:\n"
            for product in products:
                input_text += f"- ASIN: {product.get('asin', 'unknown')}, "
                input_text += f"Current Price: ${product.get('current_price', 0):.2f}, "
                input_text += f"Cost: ${product.get('cost', 0):.2f}, "
                input_text += f"Sales Rank: {product.get('sales_rank', 'unknown')}, "
                input_text += f"Inventory: {product.get('inventory', 0)}\n"
        
        # Add market conditions
        if market_conditions:
            input_text += "\nMARKET CONDITIONS:\n"
            for key, value in market_conditions.items():
                input_text += f"- {key}: {value}\n"
        
        # Add recent events
        if recent_events:
            input_text += "\nRECENT EVENTS:\n"
            for event in recent_events[-5:]:  # Only include last 5 events
                input_text += f"- {event}\n"
        
        input_text += "\nUse the available tools to analyze the market and set optimal prices. "
        input_text += "Provide your final answer in JSON format with ASINs as keys and objects containing 'price' (numeric) and 'reasoning' (string) as values."
        
        return input_text
    
    def _parse_langchain_result(self, result: Any) -> Dict[str, Any]:
        """
        Validate the LLM output against the FbaDecision schema and adapt to legacy dict format.
        Raises AgentRunnerDecisionError if validation fails.
        """
        # Ensure we have raw JSON text
        raw_text = result if isinstance(result, str) else str(result)

        # Attempt to extract a JSON object if the response accidentally contains prose
        json_text = raw_text
        if not raw_text.strip().startswith("{"):
            try:
                start = raw_text.index("{")
                end = raw_text.rindex("}") + 1
                json_text = raw_text[start:end]
            except Exception:
                json_text = raw_text  # fallback; validation will fail with clear error

        try:
            validated: FbaDecision = FbaDecision.model_validate_json(json_text)
        except ValidationError as e:
            logger.error(f"LLM output validation failed: {e}")
            raise AgentRunnerDecisionError(
                f"Decision making failed for LangChain agent {self.agent_id}: invalid structured output",
                agent_id=self.agent_id,
                framework="LangChain",
            ) from e

        # Adapt to legacy decision dict shape expected downstream
        adapted: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "framework": "LangChain",
            "timestamp": datetime.now().isoformat(),
            "pricing_decisions": {},
            "reasoning": "Validated structured output",
        }
        for pd in validated.pricing_decisions:
            adapted["pricing_decisions"][pd.asin] = {
                "price": float(pd.new_price),
                "confidence": 0.9,
                "reasoning": pd.reasoning,
            }
        if validated.meta:
            adapted["meta"] = validated.meta
        return adapted
    
    def _do_cleanup(self) -> None:
        """Clean up LangChain resources."""
        # LangChain doesn't require explicit cleanup
        self.agent = None
        self.llm = None
        self.memory = None
        self.tools = []
        logger.info(f"LangChain agent runner {self.agent_id} cleaned up")