"""
CrewAI Agent Runner - Multi-agent framework integration.

This runner integrates with CrewAI's multi-agent framework, enabling
crews of specialized agents to collaborate on FBA business decisions.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_runner import AgentRunner, SimulationState, ToolCall, AgentRunnerError

logger = logging.getLogger(__name__)

# CrewAI imports with graceful fallback
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from langchain.llms.base import LLM
    from pydantic import BaseModel, Field
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("CrewAI not available. Install with: pip install crewai")
    # Create mock classes for development
    class Agent: pass
    class Task: pass
    class Crew: pass
    class Process: pass
    class BaseTool: pass
    class LLM: pass
    class BaseModel: pass
    def Field(*args, **kwargs): pass


class FBAPricingTool(BaseTool):
    """Tool for FBA pricing decisions."""
    name: str = "fba_pricing_tool"
    description: str = "Analyze market conditions and recommend pricing actions for FBA products"
    
    def _run(self, asin: str, current_price: float, competitor_prices: List[float],
             reasoning: str = "") -> Dict[str, Any]:
        """Execute pricing analysis."""
        if not competitor_prices:
            return {
                "tool": "set_price",
                "asin": asin,
                "recommended_price": current_price,
                "confidence": 0.6,
                "reasoning": reasoning or "No competitor data available, maintaining current price"
            }
        
        # Calculate average competitor price
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices)
        min_competitor_price = min(competitor_prices)
        max_competitor_price = max(competitor_prices)
        
        # Pricing strategy logic
        if current_price > max_competitor_price * 1.1:
            # We're significantly more expensive than all competitors
            recommended_price = max_competitor_price * 0.95  # Slightly undercut highest competitor
            confidence = 0.9
            reasoning = reasoning or f"Price is {((current_price/max_competitor_price)-1)*100:.1f}% above highest competitor, reducing to {recommended_price:.2f}"
        elif current_price < min_competitor_price * 0.9:
            # We're significantly cheaper than all competitors
            recommended_price = min_competitor_price * 0.98  # Slightly increase but remain competitive
            confidence = 0.8
            reasoning = reasoning or f"Price is {(1-current_price/min_competitor_price)*100:.1f}% below lowest competitor, increasing to {recommended_price:.2f}"
        else:
            # We're in a competitive range
            if current_price > avg_competitor_price:
                recommended_price = avg_competitor_price * 0.99  # Slightly undercut average
                confidence = 0.7
                reasoning = reasoning or f"Price is {((current_price/avg_competitor_price)-1)*100:.1f}% above average competitor, adjusting to {recommended_price:.2f}"
            else:
                recommended_price = current_price  # Maintain current price
                confidence = 0.6
                reasoning = reasoning or f"Price is competitive at {(1-current_price/avg_competitor_price)*100:.1f}% below average, maintaining current price"
        
        return {
            "tool": "set_price",
            "asin": asin,
            "recommended_price": round(recommended_price, 2),
            "confidence": confidence,
            "reasoning": reasoning
        }


class FBAInventoryTool(BaseTool):
    """Tool for FBA inventory management."""
    name: str = "fba_inventory_tool"
    description: str = "Analyze inventory levels and recommend restocking actions"
    
    def _run(self, asin: str, inventory_level: int, sales_velocity: float) -> Dict[str, Any]:
        """Execute inventory analysis."""
        if sales_velocity <= 0:
            return {
                "tool": "manage_inventory",
                "asin": asin,
                "recommended_action": "hold",
                "confidence": 0.9,
                "reasoning": "No sales velocity detected, maintaining current inventory"
            }
        
        # Calculate days of inventory remaining
        days_of_inventory = inventory_level / sales_velocity if sales_velocity > 0 else float('inf')
        
        # Inventory management logic
        if days_of_inventory < 7:
            # Low inventory - need to restock
            recommended_quantity = int(sales_velocity * 14)  # 2 weeks of inventory
            confidence = 0.95
            reasoning = f"Low inventory: {days_of_inventory:.1f} days remaining. Recommend restocking {recommended_quantity} units."
            recommended_action = "restock"
        elif days_of_inventory > 90:
            # Excess inventory - reduce orders
            recommended_quantity = 0
            confidence = 0.85
            reasoning = f"Excess inventory: {days_of_inventory:.1f} days on hand. Recommend pausing restocking."
            recommended_action = "reduce_orders"
        elif days_of_inventory > 30:
            # High but not excessive inventory
            recommended_quantity = int(sales_velocity * 7)  # 1 week of inventory
            confidence = 0.7
            reasoning = f"Adequate inventory: {days_of_inventory:.1f} days on hand. Recommend light restocking of {recommended_quantity} units."
            recommended_action = "light_restock"
        else:
            # Optimal inventory range
            recommended_quantity = 0
            confidence = 0.6
            reasoning = f"Optimal inventory: {days_of_inventory:.1f} days on hand. Maintain current levels."
            recommended_action = "hold"
        
        return {
            "tool": "manage_inventory",
            "asin": asin,
            "recommended_action": recommended_action,
            "recommended_quantity": recommended_quantity,
            "confidence": confidence,
            "reasoning": reasoning
        }


class FBAMarketTool(BaseTool):
    """Tool for FBA market analysis."""
    name: str = "fba_market_tool"
    description: str = "Analyze market conditions and competitive landscape"
    
    def _run(self, products_data: str) -> Dict[str, Any]:
        """Execute market analysis."""
        try:
            import json
            products = json.loads(products_data)
            
            if not products:
                return {
                    "market_condition": "unknown",
                    "recommendation": "gather_more_data",
                    "confidence": 0.3,
                    "reasoning": "No product data available for market analysis"
                }
            
            # Analyze market conditions
            total_products = len(products)
            avg_price = sum(p.get('current_price', 0) for p in products) / total_products
            avg_inventory = sum(p.get('inventory', 0) for p in products) / total_products
            
            # Calculate market competitiveness
            competitor_count = sum(len(p.get('competitor_prices', [])) for p in products)
            avg_competitors_per_product = competitor_count / total_products if total_products > 0 else 0
            
            # Determine market condition
            if avg_competitors_per_product < 2:
                market_condition = "low_competition"
                recommendation = "opportunity_for_growth"
                confidence = 0.8
                reasoning = f"Low competition market with {avg_competitors_per_product:.1f} competitors per product on average"
            elif avg_competitors_per_product < 5:
                market_condition = "moderate_competition"
                recommendation = "competitive_positioning"
                confidence = 0.7
                reasoning = f"Moderate competition with {avg_competitors_per_product:.1f} competitors per product on average"
            else:
                market_condition = "high_competition"
                recommendation = "differentiation_strategy"
                confidence = 0.9
                reasoning = f"Highly competitive market with {avg_competitors_per_product:.1f} competitors per product on average"
            
            # Adjust for inventory levels
            if avg_inventory > 1000:
                recommendation += "_with_inventory_management"
                confidence = min(confidence + 0.1, 1.0)
                reasoning += f". High average inventory ({avg_inventory:.0f} units) requires careful management"
            
            return {
                "market_condition": market_condition,
                "recommendation": recommendation,
                "confidence": confidence,
                "reasoning": reasoning,
                "market_metrics": {
                    "total_products": total_products,
                    "average_price": round(avg_price, 2),
                    "average_inventory": round(avg_inventory, 0),
                    "average_competitors": round(avg_competitors_per_product, 1)
                }
            }
            
        except (json.JSONDecodeError, Exception) as e:
            return {
                "market_condition": "unknown",
                "recommendation": "data_error",
                "confidence": 0.2,
                "reasoning": f"Error analyzing market data: {str(e)}"
            }


class CrewAIRunner(AgentRunner):
    """
    CrewAI runner implementation for multi-agent FBA decision making.
    
    Creates a crew of specialized agents:
    - Pricing Specialist: Handles price optimization
    - Inventory Manager: Manages stock levels  
    - Market Analyst: Analyzes competitive landscape
    - Strategy Coordinator: Combines insights into final decisions
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.crew: Optional[Crew] = None
        self.agents: Dict[str, Agent] = {}
        self.tools: List[BaseTool] = []
        self.llm: Optional[LLM] = None
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the CrewAI crew and agents."""
        if not CREWAI_AVAILABLE:
            raise AgentRunnerError(
                "CrewAI not available. Install with: pip install crewai",
                agent_id=self.agent_id,
                framework="CrewAI"
            )
        
        try:
            # Initialize LLM
            await self._initialize_llm(config)
            
            # Initialize tools
            self._initialize_tools(config)
            
            # Create specialized agents
            await self._create_agents(config)
            
            # Create and configure crew
            await self._create_crew(config)
            
            self._initialized = True
            logger.info(f"CrewAI runner initialized for agent {self.agent_id}")
            
        except Exception as e:
            raise AgentRunnerError(
                f"Failed to initialize CrewAI agent: {str(e)}",
                agent_id=self.agent_id,
                framework="CrewAI"
            ) from e
    
    async def _initialize_llm(self, config: Dict[str, Any]) -> None:
        """Initialize the LLM for CrewAI agents."""
        llm_config = config.get('llm_config', {})
        llm_type = llm_config.get('provider', 'openai')
        
        if llm_type == 'openai':
            from langchain.llms import OpenAI
            self.llm = OpenAI(
                model_name=llm_config.get('model', 'gpt-4'),
                temperature=llm_config.get('temperature', 0.1),
                openai_api_key=llm_config.get('api_key')
            )
        elif llm_type == 'anthropic':
            from langchain.llms import Anthropic
            self.llm = Anthropic(
                model=llm_config.get('model', 'claude-3-sonnet'),
                temperature=llm_config.get('temperature', 0.1),
                anthropic_api_key=llm_config.get('api_key')
            )
        else:
            raise AgentRunnerError(
                f"Unsupported LLM provider: {llm_type}",
                agent_id=self.agent_id,
                framework="CrewAI"
            )
    
    def _initialize_tools(self, config: Dict[str, Any]) -> None:
        """Initialize tools for the crew."""
        self.tools = [
            FBAPricingTool(),
            FBAInventoryTool(), 
            FBAMarketTool()
        ]
        
        # Add any custom tools from config
        custom_tools = config.get('custom_tools', [])
        self.tools.extend(custom_tools)
    
    async def _create_agents(self, config: Dict[str, Any]) -> None:
        """Create specialized agents for the crew."""
        agent_config = config.get('agent_config', {})
        
        # Pricing Specialist
        self.agents['pricing_specialist'] = Agent(
            role='FBA Pricing Specialist',
            goal='Optimize product pricing to maximize profit while remaining competitive',
            backstory="""You are an expert in Amazon FBA pricing strategies. 
            You analyze competitor prices, market conditions, and profit margins to make 
            optimal pricing decisions.""",
            tools=[tool for tool in self.tools if 'pricing' in tool.name],
            llm=self.llm,
            verbose=config.get('verbose', False),
            allow_delegation=agent_config.get('allow_delegation', True)
        )
        
        # Inventory Manager
        self.agents['inventory_manager'] = Agent(
            role='FBA Inventory Manager',
            goal='Maintain optimal inventory levels to prevent stockouts while minimizing costs',
            backstory="""You are an expert in inventory management for Amazon FBA. 
            You analyze sales velocity, lead times, and storage costs to optimize 
            inventory levels.""",
            tools=[tool for tool in self.tools if 'inventory' in tool.name],
            llm=self.llm,
            verbose=config.get('verbose', False),
            allow_delegation=agent_config.get('allow_delegation', True)
        )
        
        # Market Analyst
        self.agents['market_analyst'] = Agent(
            role='FBA Market Analyst', 
            goal='Analyze market conditions and competitive landscape',
            backstory="""You are an expert market analyst specializing in Amazon FBA. 
            You monitor competitor behavior, market trends, and external factors 
            that impact business performance.""",
            tools=[tool for tool in self.tools if 'market' in tool.name],
            llm=self.llm,
            verbose=config.get('verbose', False),
            allow_delegation=agent_config.get('allow_delegation', True)
        )
        
        # Strategy Coordinator
        self.agents['strategy_coordinator'] = Agent(
            role='FBA Strategy Coordinator',
            goal='Coordinate insights from specialists and make final business decisions',
            backstory="""You are a strategic business coordinator for Amazon FBA operations.
            You synthesize insights from pricing, inventory, and market specialists to 
            make cohesive business decisions.""",
            tools=self.tools,  # Access to all tools
            llm=self.llm,
            verbose=config.get('verbose', False),
            allow_delegation=agent_config.get('allow_delegation', True)
        )
    
    async def _create_crew(self, config: Dict[str, Any]) -> None:
        """Create and configure the CrewAI crew."""
        crew_config = config.get('crew_config', {})
        
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=[],  # Tasks will be created dynamically per decision
            process=getattr(Process, crew_config.get('process', 'sequential')),
            verbose=config.get('verbose', False),
            manager_llm=self.llm
        )
    
    async def decide(self, state: SimulationState) -> List[ToolCall]:
        """
        Make decisions using the CrewAI crew.
        
        Creates a collaborative task for the crew based on the current simulation state.
        """
        if not self._initialized or not self.crew:
            raise AgentRunnerError(
                "CrewAI crew not initialized",
                agent_id=self.agent_id,
                framework="CrewAI"
            )
        
        try:
            # Create task based on simulation state
            task = await self._create_fba_task(state)
            
            # Execute crew decision-making
            result = await self._execute_crew_task(task)
            
            # Parse crew output to tool calls
            return self._parse_crew_output(result, state)
            
        except Exception as e:
            raise AgentRunnerError(
                f"CrewAI decision making failed: {str(e)}",
                agent_id=self.agent_id,
                framework="CrewAI"
            ) from e
    
    async def _create_fba_task(self, state: SimulationState) -> Task:
        """Create a collaborative FBA task for the crew."""
        # Prepare context for the crew
        context = {
            "tick": state.tick,
            "simulation_time": state.simulation_time.isoformat(),
            "products": [
                {
                    "asin": p.asin,
                    "current_price": p.price.to_float(),
                    "cost": p.cost.to_float(),
                    "inventory": p.inventory_units,
                    "sales_velocity": getattr(p, 'sales_velocity', 0),
                    "competitor_prices": [cp[1].to_float() for cp in getattr(p, 'competitor_prices', [])]
                }
                for p in state.products
            ],
            "recent_events": state.recent_events[-10:] if state.recent_events else [],
            "market_conditions": state.market_conditions,
            "financial_position": state.financial_position
        }
        
        task_description = f"""
        Analyze the current FBA business situation and provide actionable recommendations.
        
        Current Situation:
        - Simulation tick: {state.tick}
        - Products in portfolio: {len(state.products)}
        - Recent events: {len(state.recent_events)} events to consider
        
        Your team should collaborate to:
        1. Pricing Specialist: Analyze current pricing vs competitors and recommend price adjustments
        2. Inventory Manager: Review inventory levels and recommend any restocking actions
        3. Market Analyst: Assess overall market conditions and trends
        4. Strategy Coordinator: Synthesize all insights into final actionable decisions
        
        Context: {json.dumps(context, indent=2)}
        
        Provide your final recommendations as a JSON list of actions, where each action has:
        - tool: The tool name (e.g., "set_price", "manage_inventory")
        - parameters: Dictionary of tool parameters
        - confidence: Confidence level (0.0-1.0)
        - reasoning: Brief explanation of the decision
        """
        
        return Task(
            description=task_description,
            agent=self.agents['strategy_coordinator'],  # Coordinator leads the task
            expected_output="JSON list of recommended actions with reasoning"
        )
    
    async def _execute_crew_task(self, task: Task) -> Any:
        """Execute the crew task and return results."""
        # Update crew with the current task
        self.crew.tasks = [task]
        
        # Execute the crew
        result = self.crew.kickoff()
        
        return result
    
    def _parse_crew_output(self, result: Any, state: SimulationState) -> List[ToolCall]:
        """Parse CrewAI output into standardized ToolCalls."""
        tool_calls = []
        
        try:
            # Try to parse as JSON first
            if isinstance(result, str):
                # Extract JSON from the result string
                import re
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    result = json_match.group()
                
                actions = json.loads(result)
            elif isinstance(result, list):
                actions = result
            else:
                # Fallback: create a default action
                actions = [{
                    "tool": "wait",
                    "parameters": {},
                    "confidence": 0.5,
                    "reasoning": "Could not parse CrewAI output, defaulting to wait"
                }]
            
            # Convert actions to ToolCalls
            for action in actions:
                if isinstance(action, dict):
                    tool_calls.append(ToolCall(
                        tool_name=action.get('tool', 'wait'),
                        parameters=action.get('parameters', {}),
                        confidence=action.get('confidence', 0.8),
                        reasoning=action.get('reasoning', 'CrewAI team decision')
                    ))
        
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse CrewAI output: {e}")
            # Fallback: return a wait action
            tool_calls.append(ToolCall(
                tool_name="wait",
                parameters={},
                confidence=0.3,
                reasoning=f"Failed to parse CrewAI output: {str(e)}"
            ))
        
        return tool_calls
    
    async def cleanup(self) -> None:
        """Cleanup CrewAI resources."""
        try:
            # CrewAI doesn't require explicit cleanup in most cases
            # but we can clean up our references
            self.crew = None
            self.agents.clear()
            self.tools.clear()
            self.llm = None
            
            logger.info(f"CrewAI runner cleaned up for agent {self.agent_id}")
            
        except Exception as e:
            logger.warning(f"Error during CrewAI runner cleanup for {self.agent_id}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for CrewAI runner."""
        health = await super().health_check()
        health.update({
            "crewai_available": CREWAI_AVAILABLE,
            "crew_initialized": self.crew is not None,
            "agents_count": len(self.agents),
            "tools_count": len(self.tools),
            "llm_configured": self.llm is not None
        })
        return health