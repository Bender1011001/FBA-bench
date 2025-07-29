"""
DIY Agent Runner - Direct implementation for simple agents.

This runner wraps existing FBA-Bench agent patterns (AdvancedAgent, baseline bots)
into the standardized AgentRunner interface without external dependencies.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from .base_runner import AgentRunner, SimulationState, ToolCall, AgentRunnerError
from agents.advanced_agent import AdvancedAgent, AgentConfig
from events import BaseEvent, SetPriceCommand, TickEvent
from money import Money

logger = logging.getLogger(__name__)


class DIYRunner(AgentRunner):
    """
    DIY runner implementation for wrapping existing FBA-Bench agents.
    
    Supports two agent patterns:
    1. Event-driven agents (AdvancedAgent) - Subscribe to events, publish commands
    2. Functional agents (baseline bots) - decide() method returning commands
    
    This runner bridges these patterns to the standardized AgentRunner interface.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.agent: Optional[Union[AdvancedAgent, Any]] = None
        self.agent_type: Optional[str] = None
        self.event_bus = None
        self.gateway = None
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the wrapped agent based on configuration."""
        try:
            agent_type = config.get('agent_type', 'advanced')
            
            if agent_type == 'advanced':
                await self._initialize_advanced_agent(config)
            elif agent_type == 'baseline':
                await self._initialize_baseline_agent(config)
            elif agent_type == 'llm':
                await self._initialize_llm_agent(config)
            else:
                raise AgentRunnerError(
                    f"Unknown agent type: {agent_type}", 
                    agent_id=self.agent_id, 
                    framework="DIY"
                )
            
            self.agent_type = agent_type
            self._initialized = True
            logger.info(f"DIY runner initialized for agent {self.agent_id} (type: {agent_type})")
            
        except Exception as e:
            raise AgentRunnerError(
                f"Failed to initialize DIY agent: {str(e)}",
                agent_id=self.agent_id,
                framework="DIY"
            ) from e
    
    async def _initialize_advanced_agent(self, config: Dict[str, Any]) -> None:
        """Initialize an AdvancedAgent instance."""
        from event_bus import get_event_bus
        from constraints.agent_gateway import AgentGateway
        
        # Extract agent configuration
        agent_config = AgentConfig(
            agent_id=self.agent_id,
            target_asin=config.get('target_asin', 'B0DEFAULT'),
            strategy=config.get('strategy', 'profit_maximizer'),
            price_sensitivity=config.get('price_sensitivity', 0.1),
            reaction_speed=config.get('reaction_speed', 1),
            min_price=Money(config.get('min_price_cents', 500)),
            max_price=Money(config.get('max_price_cents', 5000))
        )
        
        # Get event bus and gateway if available
        self.event_bus = config.get('event_bus') or get_event_bus()
        self.gateway = config.get('agent_gateway')
        
        # Create and start the agent
        self.agent = AdvancedAgent(agent_config, self.event_bus, self.gateway)
        await self.agent.start()
    
    async def _initialize_baseline_agent(self, config: Dict[str, Any]) -> None:
        """Initialize a baseline bot agent."""
        bot_type = config.get('bot_type', 'greedy')
        
        if bot_type == 'greedy':
            from baseline_bots.greedy_script_bot import GreedyScriptBot
            self.agent = GreedyScriptBot(
                reorder_threshold=config.get('reorder_threshold', 10),
                reorder_quantity=config.get('reorder_quantity', 50)
            )
        else:
            raise AgentRunnerError(
                f"Unknown baseline bot type: {bot_type}",
                agent_id=self.agent_id,
                framework="DIY"
            )
    
    async def _initialize_llm_agent(self, config: Dict[str, Any]) -> None:
        """Initialize an LLM-based agent."""
        llm_type = config.get('llm_type', 'claude')
        
        if llm_type == 'claude':
            from baseline_bots.claude_sonnet_bot import ClaudeSonnetBot
            from llm_interface.openrouter_client import OpenRouterClient
            from llm_interface.prompt_adapter import PromptAdapter
            from llm_interface.response_parser import ResponseParser
            
            # Initialize LLM client and supporting components
            llm_client = OpenRouterClient(
                model_name=config.get('model_name', 'anthropic/claude-3-sonnet:beta'),
                api_key=config.get('api_key')
            )
            
            prompt_adapter = PromptAdapter()
            response_parser = ResponseParser()
            
            self.agent = ClaudeSonnetBot(
                agent_id=self.agent_id,
                llm_client=llm_client,
                prompt_adapter=prompt_adapter,
                response_parser=response_parser,
                agent_gateway=config.get('agent_gateway'),
                model_params=config.get('model_params', {})
            )
        elif llm_type == 'gpt':
            from baseline_bots.gpt_4o_mini_bot import GPT4oMiniBot
            # Similar initialization for GPT
            # ... (implementation similar to claude)
        else:
            raise AgentRunnerError(
                f"Unknown LLM agent type: {llm_type}",
                agent_id=self.agent_id,
                framework="DIY"
            )
    
    async def decide(self, state: SimulationState) -> List[ToolCall]:
        """
        Make decisions based on the wrapped agent type.
        
        This method bridges different agent patterns to the unified ToolCall interface.
        """
        if not self._initialized or not self.agent:
            raise AgentRunnerError(
                "Agent not initialized",
                agent_id=self.agent_id,
                framework="DIY"
            )
        
        try:
            if self.agent_type == 'advanced':
                return await self._decide_advanced_agent(state)
            elif self.agent_type in ['baseline', 'llm']:
                return await self._decide_functional_agent(state)
            else:
                raise AgentRunnerError(
                    f"Unknown agent type for decision: {self.agent_type}",
                    agent_id=self.agent_id,
                    framework="DIY"
                )
                
        except Exception as e:
            raise AgentRunnerError(
                f"Decision making failed: {str(e)}",
                agent_id=self.agent_id,
                framework="DIY"
            ) from e
    
    async def _decide_advanced_agent(self, state: SimulationState) -> List[ToolCall]:
        """Handle decision making for event-driven AdvancedAgent."""
        # For AdvancedAgent, we need to trigger its decision-making through events
        # and capture the commands it publishes
        
        commands_captured = []
        
        # Create a temporary event capture mechanism
        original_publish = self.event_bus.publish if self.event_bus else None
        
        async def capture_commands(event_type: str, event: BaseEvent):
            if isinstance(event, SetPriceCommand) and event.agent_id == self.agent_id:
                commands_captured.append(event)
            # Call original publish to maintain event flow
            if original_publish:
                await original_publish(event_type, event)
        
        # Temporarily override publish to capture commands
        if self.event_bus:
            self.event_bus.publish = capture_commands
        
        try:
            # Trigger agent decision making with a tick event
            tick_event = TickEvent(
                event_id=f"tick_{state.tick}",
                timestamp=state.simulation_time,
                tick_number=state.tick
            )
            
            await self.agent.handle_tick_event(tick_event)
            
            # Convert captured commands to ToolCalls
            tool_calls = []
            for command in commands_captured:
                if isinstance(command, SetPriceCommand):
                    tool_calls.append(ToolCall(
                        tool_name="set_price",
                        parameters={
                            "asin": command.asin,
                            "price": command.new_price.to_float()
                        },
                        confidence=0.9,  # Default confidence
                        reasoning=command.reason or "Advanced agent pricing decision"
                    ))
            
            return tool_calls
            
        finally:
            # Restore original publish method
            if self.event_bus and original_publish:
                self.event_bus.publish = original_publish
    
    async def _decide_functional_agent(self, state: SimulationState) -> List[ToolCall]:
        """Handle decision making for functional agents (baseline, LLM bots)."""
        # Convert SimulationState to the format expected by baseline bots
        if self.agent_type == 'baseline':
            from baseline_bots.greedy_script_bot import SimulationState as GreedySimState
            
            bot_state = GreedySimState(
                products=state.products,
                current_tick=state.tick,
                simulation_time=state.simulation_time
            )
            
            # Call the bot's decide method
            actions = self.agent.decide(bot_state)
            
        elif self.agent_type == 'llm':
            from baseline_bots.claude_sonnet_bot import SimulationState as LLMSimState
            
            # Convert to LLM bot state format
            llm_state = LLMSimState(
                products=state.products,
                current_tick=state.tick,
                simulation_time=state.simulation_time,
                recent_events=state.recent_events
            )
            
            # Call the LLM bot's decide method
            actions = await self.agent.decide(llm_state)
        
        else:
            actions = []
        
        # Convert actions to ToolCalls
        tool_calls = []
        for action in actions:
            if isinstance(action, SetPriceCommand):
                tool_calls.append(ToolCall(
                    tool_name="set_price",
                    parameters={
                        "asin": action.asin,
                        "price": action.new_price.to_float()
                    },
                    confidence=0.8,  # Default confidence for baseline/LLM bots
                    reasoning=action.reason or f"{self.agent_type} agent decision"
                ))
            # Add other action types as needed
        
        return tool_calls
    
    async def cleanup(self) -> None:
        """Cleanup the wrapped agent."""
        try:
            if self.agent and self.agent_type == 'advanced':
                await self.agent.stop()
            
            logger.info(f"DIY runner cleaned up for agent {self.agent_id}")
            
        except Exception as e:
            logger.warning(f"Error during DIY runner cleanup for {self.agent_id}: {e}")
            # Don't raise during cleanup - just log the warning


class DIYAgentAdapter:
    """
    Adapter class to wrap any existing agent into the AgentRunner interface.
    
    This provides a more flexible way to integrate custom agents that don't
    fit the standard patterns.
    """
    
    def __init__(self, agent_class, agent_config: Dict[str, Any]):
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.agent_instance = None
    
    async def create_runner(self, agent_id: str) -> DIYRunner:
        """Create a DIY runner wrapping the custom agent."""
        config = {
            'agent_type': 'custom',
            'agent_class': self.agent_class,
            'agent_config': self.agent_config
        }
        
        runner = DIYRunner(agent_id, config)
        await runner.initialize(config)
        return runner