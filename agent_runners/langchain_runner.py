"""
LangChain Agent Runner - Advanced reasoning and tool chain integration.

This runner integrates with LangChain's agent framework, enabling
sophisticated reasoning chains, memory systems, and tool ecosystems.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from .base_runner import AgentRunner, SimulationState, ToolCall, AgentRunnerError

logger = logging.getLogger(__name__)

# LangChain imports with graceful fallback
try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.agents.agent_types import AgentType
    from langchain.agents.format_scratchpad import format_to_openai_function_messages
    from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
    from langchain.tools import BaseTool, StructuredTool, tool
    from langchain.schema import AgentAction, AgentFinish
    from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema.messages import SystemMessage
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain")
    # Create mock classes for development
    class AgentExecutor: pass
    class BaseTool: pass
    class StructuredTool: pass
    class ConversationBufferWindowMemory: pass
    class ChatOpenAI: pass
    class ChatPromptTemplate: pass
    class MessagesPlaceholder: pass
    class SystemMessage: pass
    class BaseModel: pass
    def Field(*args, **kwargs): pass
    def tool(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class FBAPricingToolInput(BaseModel):
    """Input schema for FBA pricing tool."""
    asin: str = Field(description="Product ASIN to price")
    current_price: float = Field(description="Current product price")
    competitor_prices: List[float] = Field(description="List of competitor prices")
    reasoning: str = Field(description="Reasoning for the pricing decision", default="")


class FBAInventoryToolInput(BaseModel):
    """Input schema for FBA inventory tool."""
    asin: str = Field(description="Product ASIN to manage")
    current_inventory: int = Field(description="Current inventory level")
    sales_velocity: float = Field(description="Daily sales velocity")


class FBAMarketToolInput(BaseModel):
    """Input schema for FBA market analysis tool."""
    market_data: str = Field(description="JSON string of market data to analyze")


@tool("fba_pricing_analysis", args_schema=FBAPricingToolInput, return_direct=False)
def fba_pricing_analysis(asin: str, current_price: float, competitor_prices: List[float], reasoning: str = "") -> str:
    """
    Analyze pricing strategy for an FBA product based on market conditions.
    
    Args:
        asin: Product ASIN to analyze
        current_price: Current price of the product
        competitor_prices: List of competitor prices for comparison
        reasoning: Additional reasoning for the analysis
    
    Returns:
        JSON string with pricing recommendation
    """
    if not competitor_prices:
        recommended_price = current_price
        strategy = "maintain"
    else:
        min_competitor = min(competitor_prices)
        avg_competitor = sum(competitor_prices) / len(competitor_prices)
        
        # Simple pricing strategy
        if current_price > avg_competitor * 1.1:
            recommended_price = avg_competitor * 0.95  # Undercut by 5%
            strategy = "competitive_reduction"
        elif current_price < min_competitor * 0.9:
            recommended_price = min_competitor * 0.99  # Match lowest
            strategy = "competitive_match"
        else:
            recommended_price = current_price
            strategy = "maintain"
    
    return json.dumps({
        "tool": "set_price",
        "asin": asin,
        "current_price": current_price,
        "recommended_price": round(recommended_price, 2),
        "strategy": strategy,
        "confidence": 0.85,
        "reasoning": reasoning or f"LangChain pricing analysis: {strategy}"
    })


@tool("fba_inventory_analysis", args_schema=FBAInventoryToolInput, return_direct=False)
def fba_inventory_analysis(asin: str, current_inventory: int, sales_velocity: float) -> str:
    """
    Analyze inventory levels and recommend actions.
    
    Args:
        asin: Product ASIN to analyze
        current_inventory: Current inventory units
        sales_velocity: Daily sales velocity
    
    Returns:
        JSON string with inventory recommendation
    """
    days_of_supply = current_inventory / max(sales_velocity, 0.1)  # Avoid division by zero
    
    if days_of_supply < 7:
        action = "urgent_restock"
        confidence = 0.95
    elif days_of_supply < 14:
        action = "plan_restock"
        confidence = 0.8
    elif days_of_supply > 60:
        action = "reduce_inventory"
        confidence = 0.7
    else:
        action = "maintain"
        confidence = 0.9
    
    return json.dumps({
        "tool": "manage_inventory",
        "asin": asin,
        "current_inventory": current_inventory,
        "days_of_supply": round(days_of_supply, 1),
        "recommended_action": action,
        "confidence": confidence,
        "reasoning": f"LangChain inventory analysis: {days_of_supply:.1f} days of supply"
    })


@tool("fba_market_analysis", args_schema=FBAMarketToolInput, return_direct=False)
def fba_market_analysis(market_data: str) -> str:
    """
    Analyze market conditions and trends.
    
    Args:
        market_data: JSON string containing market data
    
    Returns:
        JSON string with market analysis
    """
    try:
        data = json.loads(market_data)
        product_count = data.get('product_count', 0)
        total_events = data.get('total_events', 0)
        
        if total_events > 10:
            market_condition = "highly_active"
            recommendation = "monitor_closely"
        elif total_events > 5:
            market_condition = "moderately_active"
            recommendation = "standard_monitoring"
        else:
            market_condition = "quiet"
            recommendation = "opportunity_seeking"
        
        return json.dumps({
            "market_condition": market_condition,
            "recommendation": recommendation,
            "confidence": 0.8,
            "reasoning": f"LangChain market analysis: {total_events} recent events detected"
        })
    
    except (json.JSONDecodeError, KeyError) as e:
        return json.dumps({
            "market_condition": "unknown",
            "recommendation": "maintain_status_quo",
            "confidence": 0.5,
            "reasoning": f"LangChain market analysis failed: {str(e)}"
        })


class LangChainRunner(AgentRunner):
    """
    LangChain runner implementation for advanced reasoning and tool chains.
    
    Features:
    - Function calling agents with structured reasoning
    - Memory systems for maintaining context across decisions
    - Tool chains for complex multi-step operations
    - Customizable prompts and reasoning patterns
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.agent_executor: Optional[AgentExecutor] = None
        self.memory: Optional[Any] = None
        self.llm: Optional[Any] = None
        self.tools: List[BaseTool] = []
        self.prompt_template: Optional[ChatPromptTemplate] = None
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the LangChain agent and tools."""
        if not LANGCHAIN_AVAILABLE:
            raise AgentRunnerError(
                "LangChain not available. Install with: pip install langchain",
                agent_id=self.agent_id,
                framework="LangChain"
            )
        
        try:
            # Initialize LLM
            await self._initialize_llm(config)
            
            # Initialize memory
            await self._initialize_memory(config)
            
            # Initialize tools
            self._initialize_tools(config)
            
            # Create prompt template
            self._create_prompt_template(config)
            
            # Create agent
            await self._create_agent(config)
            
            self._initialized = True
            logger.info(f"LangChain runner initialized for agent {self.agent_id}")
            
        except Exception as e:
            raise AgentRunnerError(
                f"Failed to initialize LangChain agent: {str(e)}",
                agent_id=self.agent_id,
                framework="LangChain"
            ) from e
    
    async def _initialize_llm(self, config: Dict[str, Any]) -> None:
        """Initialize the language model."""
        llm_config = config.get('llm_config', {})
        model_name = llm_config.get('model', 'gpt-4')
        
        if 'gpt' in model_name.lower():
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=llm_config.get('temperature', 0.1),
                openai_api_key=llm_config.get('api_key')
            )
        else:
            raise AgentRunnerError(
                f"Unsupported model: {model_name}",
                agent_id=self.agent_id,
                framework="LangChain"
            )
    
    async def _initialize_memory(self, config: Dict[str, Any]) -> None:
        """Initialize memory system."""
        memory_config = config.get('memory_config', {})
        memory_type = memory_config.get('type', 'buffer')
        
        if memory_type == 'buffer':
            self.memory = ConversationBufferWindowMemory(
                k=memory_config.get('window_size', 10),
                memory_key="chat_history",
                return_messages=True
            )
        elif memory_type == 'summary':
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True
            )
        else:
            # No memory
            self.memory = None
    
    def _initialize_tools(self, config: Dict[str, Any]) -> None:
        """Initialize tools for the agent."""
        # Standard FBA tools
        self.tools = [
            fba_pricing_analysis,
            fba_inventory_analysis,
            fba_market_analysis
        ]
        
        # Add custom tools from config
        custom_tools = config.get('custom_tools', [])
        self.tools.extend(custom_tools)
    
    def _create_prompt_template(self, config: Dict[str, Any]) -> None:
        """Create the prompt template for the agent."""
        system_message = config.get('system_message', """
You are an expert Amazon FBA business agent. Your role is to analyze market conditions, 
product performance, and competitive landscape to make optimal business decisions.

You have access to specialized tools for:
- Pricing analysis and optimization
- Inventory management and forecasting  
- Market condition assessment

For each decision, consider:
1. Current market conditions and trends
2. Competitive positioning
3. Profit margins and financial impact
4. Risk assessment and mitigation

Always provide clear reasoning for your decisions and use the available tools to gather 
necessary data before making recommendations.
""")
        
        messages = [
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
        
        self.prompt_template = ChatPromptTemplate.from_messages(messages)
    
    async def _create_agent(self, config: Dict[str, Any]) -> None:
        """Create the LangChain agent executor."""
        # Create the OpenAI functions agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt_template
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=config.get('verbose', False),
            max_iterations=config.get('max_iterations', 5),
            early_stopping_method="generate"
        )
    
    async def decide(self, state: SimulationState) -> List[ToolCall]:
        """
        Make decisions using the LangChain agent.
        
        Uses the agent's reasoning capabilities and tool access to analyze
        the simulation state and generate appropriate actions.
        """
        if not self._initialized or not self.agent_executor:
            raise AgentRunnerError(
                "LangChain agent not initialized",
                agent_id=self.agent_id,
                framework="LangChain"
            )
        
        try:
            # Prepare input for the agent
            agent_input = self._prepare_agent_input(state)
            
            # Execute the agent
            result = self.agent_executor.invoke({"input": agent_input})
            
            # Parse the output
            return self._parse_agent_output(result, state)
            
        except Exception as e:
            raise AgentRunnerError(
                f"LangChain decision making failed: {str(e)}",
                agent_id=self.agent_id,
                framework="LangChain"
            ) from e
    
    def _prepare_agent_input(self, state: SimulationState) -> str:
        """Prepare input for the LangChain agent."""
        # Create a comprehensive description of the current situation
        products_summary = []
        for product in state.products:
            competitor_prices = [cp[1].to_float() for cp in getattr(product, 'competitor_prices', [])]
            products_summary.append({
                "asin": product.asin,
                "current_price": product.price.to_float(),
                "cost": product.cost.to_float(),
                "inventory": product.inventory_units,
                "sales_velocity": getattr(product, 'sales_velocity', 0),
                "competitor_prices": competitor_prices
            })
        
        market_data = {
            "product_count": len(state.products),
            "total_events": len(state.recent_events),
            "market_conditions": state.market_conditions,
            "financial_position": state.financial_position
        }
        
        return f"""
Current FBA Business Situation Analysis - Tick {state.tick}

TIME: {state.simulation_time.isoformat()}

PRODUCT PORTFOLIO:
{json.dumps(products_summary, indent=2)}

RECENT EVENTS:
{len(state.recent_events)} events in recent history

MARKET DATA:
{json.dumps(market_data, indent=2)}

Please analyze this situation and provide specific actionable recommendations. 
Use the available tools to gather additional insights and make informed decisions.

Focus on:
1. Pricing optimization opportunities
2. Inventory management needs
3. Market condition responses
4. Strategic positioning

Provide your final recommendations as a clear action plan.
"""
    
    def _parse_agent_output(self, result: Dict[str, Any], state: SimulationState) -> List[ToolCall]:
        """Parse LangChain agent output into ToolCalls."""
        tool_calls = []
        
        try:
            # Extract the agent's output
            output = result.get('output', '')
            
            # Look for tool call results in the intermediate steps
            intermediate_steps = result.get('intermediate_steps', [])
            
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                        # Parse tool results
                        try:
                            if isinstance(observation, str):
                                tool_result = json.loads(observation)
                                if 'tool' in tool_result:
                                    tool_calls.append(ToolCall(
                                        tool_name=tool_result['tool'],
                                        parameters=self._extract_tool_parameters(tool_result),
                                        confidence=tool_result.get('confidence', 0.8),
                                        reasoning=tool_result.get('reasoning', 'LangChain agent decision')
                                    ))
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            # If no tool calls found in intermediate steps, try to extract from final output
            if not tool_calls:
                tool_calls = self._extract_tool_calls_from_text(output)
            
            # If still no tool calls, create a default wait action
            if not tool_calls:
                tool_calls.append(ToolCall(
                    tool_name="wait",
                    parameters={},
                    confidence=0.5,
                    reasoning="LangChain agent completed analysis but provided no specific actions"
                ))
        
        except Exception as e:
            logger.warning(f"Error parsing LangChain output: {e}")
            tool_calls.append(ToolCall(
                tool_name="wait",
                parameters={},
                confidence=0.3,
                reasoning=f"Failed to parse LangChain output: {str(e)}"
            ))
        
        return tool_calls
    
    def _extract_tool_parameters(self, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for tool call from result."""
        if tool_result.get('tool') == 'set_price':
            return {
                'asin': tool_result.get('asin'),
                'price': tool_result.get('recommended_price')
            }
        elif tool_result.get('tool') == 'manage_inventory':
            return {
                'asin': tool_result.get('asin'),
                'action': tool_result.get('recommended_action')
            }
        else:
            return {}
    
    def _extract_tool_calls_from_text(self, text: str) -> List[ToolCall]:
        """Extract tool calls from agent text output."""
        tool_calls = []
        
        # Simple pattern matching for common actions
        import re
        
        # Look for price setting patterns
        price_patterns = re.findall(r'set.*price.*(\w+).*\$?(\d+\.?\d*)', text, re.IGNORECASE)
        for asin_match, price_match in price_patterns:
            try:
                tool_calls.append(ToolCall(
                    tool_name="set_price",
                    parameters={"asin": asin_match, "price": float(price_match)},
                    confidence=0.6,
                    reasoning="Extracted from LangChain text output"
                ))
            except ValueError:
                continue
        
        return tool_calls
    
    async def cleanup(self) -> None:
        """Cleanup LangChain resources."""
        try:
            # Clear memory if it exists
            if self.memory:
                self.memory.clear()
            
            # Clear references
            self.agent_executor = None
            self.memory = None
            self.llm = None
            self.tools.clear()
            self.prompt_template = None
            
            logger.info(f"LangChain runner cleaned up for agent {self.agent_id}")
            
        except Exception as e:
            logger.warning(f"Error during LangChain runner cleanup for {self.agent_id}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for LangChain runner."""
        health = await super().health_check()
        health.update({
            "langchain_available": LANGCHAIN_AVAILABLE,
            "agent_executor_initialized": self.agent_executor is not None,
            "memory_configured": self.memory is not None,
            "tools_count": len(self.tools),
            "llm_configured": self.llm is not None
        })
        return health