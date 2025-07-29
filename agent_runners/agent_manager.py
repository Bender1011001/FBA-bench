"""
Agent Manager - Integration layer between simulation and AgentRunner framework.

This manager bridges the existing simulation architecture with the new
framework-agnostic AgentRunner interface, handling agent lifecycle,
state conversion, and command execution.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass

from .base_runner import AgentRunner, SimulationState, ToolCall, AgentRunnerError
from .runner_factory import RunnerFactory
from events import BaseEvent, SetPriceCommand, TickEvent
from event_bus import EventBus
from services.world_store import WorldStore
from money import Money

logger = logging.getLogger(__name__)


@dataclass
class AgentRegistration:
    """Registration information for an agent."""
    agent_id: str
    runner: AgentRunner
    framework: str
    config: Dict[str, Any]
    active: bool = True
    last_decision_tick: int = 0
    total_decisions: int = 0
    total_tool_calls: int = 0
    errors: int = 0


class AgentManager:
    """
    Manager for framework-agnostic agents in FBA-Bench simulation.
    
    Responsibilities:
    - Agent lifecycle management (create, initialize, cleanup)
    - State conversion between simulation and AgentRunner formats
    - Tool call execution and command publishing
    - Agent monitoring and health checks
    - Framework-agnostic agent orchestration
    """
    
    def __init__(self, event_bus: EventBus, world_store: Optional[WorldStore] = None):
        self.event_bus = event_bus
        self.world_store = world_store
        self.agents: Dict[str, AgentRegistration] = {}
        self.current_tick = 0
        self.simulation_time = datetime.utcnow()
        self._initialized = False
        
        # Statistics
        self.stats = {
            'total_agents': 0,
            'active_agents': 0,
            'total_decisions': 0,
            'total_tool_calls': 0,
            'total_errors': 0
        }
    
    async def initialize(self) -> None:
        """Initialize the agent manager."""
        if self._initialized:
            return
        
        # Subscribe to simulation events
        await self.event_bus.subscribe('TickEvent', self._handle_tick_event)
        
        self._initialized = True
        logger.info("AgentManager initialized")
    
    async def register_agent(self, agent_id: str, framework: str, 
                           config: Dict[str, Any]) -> AgentRunner:
        """
        Register a new agent with the manager.
        
        Args:
            agent_id: Unique identifier for the agent
            framework: Framework name (diy, crewai, langchain)
            config: Framework-specific configuration
            
        Returns:
            Initialized AgentRunner instance
        """
        if agent_id in self.agents:
            raise AgentRunnerError(
                f"Agent {agent_id} already registered",
                agent_id=agent_id,
                framework=framework
            )
        
        try:
            # Create and initialize the agent runner
            runner = await RunnerFactory.create_and_initialize_runner(
                framework, agent_id, config
            )
            
            # Register the agent
            registration = AgentRegistration(
                agent_id=agent_id,
                runner=runner,
                framework=framework,
                config=config
            )
            
            self.agents[agent_id] = registration
            self._update_stats()
            
            logger.info(f"Registered agent {agent_id} with framework {framework}")
            return runner
            
        except Exception as e:
            raise AgentRunnerError(
                f"Failed to register agent {agent_id}: {str(e)}",
                agent_id=agent_id,
                framework=framework
            ) from e
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent and cleanup resources."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found for unregistration")
            return
        
        registration = self.agents[agent_id]
        
        try:
            # Cleanup the agent runner
            await registration.runner.cleanup()
            
            # Remove from registry
            del self.agents[agent_id]
            self._update_stats()
            
            logger.info(f"Unregistered agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error during agent {agent_id} cleanup: {e}")
    
    async def _handle_tick_event(self, event: TickEvent) -> None:
        """Handle tick events and trigger agent decisions."""
        self.current_tick = event.tick_number
        self.simulation_time = event.timestamp
        
        # Get current simulation state
        simulation_state = await self._build_simulation_state()
        
        # Process all active agents concurrently
        agent_tasks = []
        for agent_id, registration in self.agents.items():
            if registration.active:
                task = asyncio.create_task(
                    self._process_agent_decision(registration, simulation_state)
                )
                agent_tasks.append(task)
        
        # Wait for all agent decisions
        if agent_tasks:
            await asyncio.gather(*agent_tasks, return_exceptions=True)
    
    async def _process_agent_decision(self, registration: AgentRegistration, 
                                    state: SimulationState) -> None:
        """Process decision making for a single agent."""
        agent_id = registration.agent_id
        
        try:
            # Get agent decision
            tool_calls = await registration.runner.decide(state)
            
            # Update registration stats
            registration.last_decision_tick = self.current_tick
            registration.total_decisions += 1
            registration.total_tool_calls += len(tool_calls)
            
            # Execute tool calls
            await self._execute_tool_calls(agent_id, tool_calls)
            
            logger.debug(f"Agent {agent_id} made {len(tool_calls)} tool calls")
            
        except Exception as e:
            registration.errors += 1
            logger.error(f"Agent {agent_id} decision error: {e}")
            
            # Optionally deactivate agent after too many errors
            if registration.errors > 10:
                registration.active = False
                logger.warning(f"Deactivated agent {agent_id} due to too many errors")
    
    async def _execute_tool_calls(self, agent_id: str, tool_calls: List[ToolCall]) -> None:
        """Execute tool calls from an agent."""
        for tool_call in tool_calls:
            try:
                await self._execute_single_tool_call(agent_id, tool_call)
            except Exception as e:
                logger.error(f"Tool call execution failed for agent {agent_id}: {e}")
    
    async def _execute_single_tool_call(self, agent_id: str, tool_call: ToolCall) -> None:
        """Execute a single tool call."""
        if tool_call.tool_name == "set_price":
            await self._execute_set_price(agent_id, tool_call)
        elif tool_call.tool_name == "wait":
            # No action needed for wait
            pass
        else:
            logger.warning(f"Unknown tool call: {tool_call.tool_name}")
    
    async def _execute_set_price(self, agent_id: str, tool_call: ToolCall) -> None:
        """Execute a set_price tool call."""
        params = tool_call.parameters
        asin = params.get('asin')
        price = params.get('price')
        
        if not asin or price is None:
            raise ValueError(f"Invalid set_price parameters: {params}")
        
        # Convert price to Money
        if isinstance(price, (int, float)):
            price_money = Money.from_dollars(price)
        else:
            price_money = Money.from_dollars(float(price))
        
        # Create and publish SetPriceCommand
        command = SetPriceCommand(
            event_id=f"price_cmd_{agent_id}_{self.current_tick}",
            timestamp=self.simulation_time,
            agent_id=agent_id,
            asin=asin,
            new_price=price_money,
            reason=tool_call.reasoning or "Agent decision"
        )
        
        await self.event_bus.publish('SetPriceCommand', command)
        logger.debug(f"Published SetPriceCommand for {agent_id}: {asin} -> ${price}")
    
    async def _build_simulation_state(self) -> SimulationState:
        """Build simulation state for agent decision making."""
        # Get products from world store if available
        products = []
        financial_position = {}
        market_conditions = {}
        recent_events = []
        
        if self.world_store:
            try:
                # Get products from world store
                # This would need to be implemented based on WorldStore interface
                # For now, create a minimal state
                pass
            except Exception as e:
                logger.warning(f"Failed to get state from world store: {e}")
        
        # Create simulation state
        state = SimulationState(
            tick=self.current_tick,
            simulation_time=self.simulation_time,
            products=products,
            recent_events=recent_events,
            financial_position=financial_position,
            market_conditions=market_conditions,
            agent_state={}
        )
        
        return state
    
    def _update_stats(self) -> None:
        """Update agent manager statistics."""
        self.stats['total_agents'] = len(self.agents)
        self.stats['active_agents'] = sum(1 for r in self.agents.values() if r.active)
        self.stats['total_decisions'] = sum(r.total_decisions for r in self.agents.values())
        self.stats['total_tool_calls'] = sum(r.total_tool_calls for r in self.agents.values())
        self.stats['total_errors'] = sum(r.errors for r in self.agents.values())
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status of all agents."""
        agent_health = {}
        
        for agent_id, registration in self.agents.items():
            try:
                health = await registration.runner.health_check()
                agent_health[agent_id] = {
                    'framework': registration.framework,
                    'active': registration.active,
                    'last_decision_tick': registration.last_decision_tick,
                    'total_decisions': registration.total_decisions,
                    'total_tool_calls': registration.total_tool_calls,
                    'errors': registration.errors,
                    'runner_health': health
                }
            except Exception as e:
                agent_health[agent_id] = {
                    'framework': registration.framework,
                    'active': False,
                    'error': str(e)
                }
        
        return {
            'manager_stats': self.stats,
            'agents': agent_health,
            'available_frameworks': RunnerFactory.get_available_frameworks()
        }
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        registration = self.agents[agent_id]
        health = await registration.runner.health_check()
        
        return {
            'agent_id': agent_id,
            'framework': registration.framework,
            'config': registration.config,
            'active': registration.active,
            'last_decision_tick': registration.last_decision_tick,
            'total_decisions': registration.total_decisions,
            'total_tool_calls': registration.total_tool_calls,
            'errors': registration.errors,
            'health': health
        }
    
    async def cleanup(self) -> None:
        """Cleanup all agents and resources."""
        cleanup_tasks = []
        
        for agent_id in list(self.agents.keys()):
            task = asyncio.create_task(self.unregister_agent(agent_id))
            cleanup_tasks.append(task)
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._initialized = False
        logger.info("AgentManager cleanup completed")
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent IDs."""
        return list(self.agents.keys())
    
    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs."""
        return [agent_id for agent_id, reg in self.agents.items() if reg.active]
    
    def get_framework_usage(self) -> Dict[str, int]:
        """Get usage statistics by framework."""
        framework_counts = {}
        for registration in self.agents.values():
            framework = registration.framework
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
        return framework_counts