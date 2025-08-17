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
        """Create the LangChain LLM."""
        from langchain_openai import ChatOpenAI
        
        # Create the LLM
        self.llm = ChatOpenAI(
            model=self.llm_config.model,
            api_key=self.llm_config.api_key.get_secret_value() if self.llm_config.api_key else None,
            base_url=self.llm_config.base_url,
            max_tokens=self.llm_config.max_tokens,
            temperature=self.llm_config.temperature,
            model_kwargs={
                "top_p": self.llm_config.top_p,
                "request_timeout": self.llm_config.timeout
            }
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
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            memory=self.memory,
            agent_kwargs={
                "prefix": agent_params.get('prefix', "You are an FBA pricing expert. Your goal is to optimize product pricing for maximum profit."),
                "suffix": agent_params.get('suffix', "Begin! When you have a final answer, provide it in JSON format with ASINs as keys and objects containing 'price' (numeric) and 'reasoning' (string) as values.")
            }
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
            
            # Parse the result
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
        """Parse the result from the LangChain agent."""
        decision = {
            'agent_id': self.agent_id,
            'framework': 'LangChain',
            'timestamp': datetime.now().isoformat(),
            'pricing_decisions': {},
            'reasoning': ''
        }
        
        try:
            # Convert result to string if it's not already
            result_str = str(result)
            
            # Try to parse as JSON
            try:
                pricing_data = json.loads(result_str)
                
                # Extract pricing decisions
                if isinstance(pricing_data, dict):
                    for asin, price_data in pricing_data.items():
                        if isinstance(price_data, dict):
                            decision['pricing_decisions'][asin] = {
                                'price': float(price_data.get('price', 0)),
                                'confidence': float(price_data.get('confidence', 0.8)),
                                'reasoning': price_data.get('reasoning', 'LangChain pricing decision')
                            }
                        elif isinstance(price_data, (int, float)):
                            decision['pricing_decisions'][asin] = {
                                'price': float(price_data),
                                'confidence': 0.8,
                                'reasoning': 'LangChain pricing decision'
                            }
                
                decision['reasoning'] = "Successfully parsed LangChain pricing decisions"
                
            except json.JSONDecodeError:
                # If not JSON, try to extract pricing information using regex
                price_pattern = r'(?:price|pricing|cost)\s*[:=]?\s*\$?(\d+(?:\.\d+)?)'
                matches = re.findall(price_pattern, result_str, re.IGNORECASE)
                
                if matches:
                    # Create generic pricing decisions
                    for i, price in enumerate(matches):
                        decision['pricing_decisions'][f'product_{i+1}'] = {
                            'price': float(price),
                            'confidence': 0.6,
                            'reasoning': 'Extracted from LangChain text response'
                        }
                    
                    decision['reasoning'] = "Extracted pricing information from LangChain text response"
                else:
                    # Fallback to a default decision
                    decision['pricing_decisions']['default'] = {
                        'price': 10.0,
                        'confidence': 0.3,
                        'reasoning': 'Fallback decision due to unparseable LangChain response'
                    }
                    decision['reasoning'] = "Could not parse LangChain response, using fallback"
            
        except Exception as e:
            logger.error(f"Error parsing LangChain result: {e}")
            decision['pricing_decisions']['default'] = {
                'price': 10.0,
                'confidence': 0.1,
                'reasoning': f'Error parsing result: {str(e)}'
            }
            decision['reasoning'] = f"Error parsing LangChain result: {str(e)}"
        
        return decision
    
    def _do_cleanup(self) -> None:
        """Clean up LangChain resources."""
        # LangChain doesn't require explicit cleanup
        self.agent = None
        self.llm = None
        self.memory = None
        self.tools = []
        logger.info(f"LangChain agent runner {self.agent_id} cleaned up")