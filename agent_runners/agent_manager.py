import logging
import asyncio
import os
import yaml
import inspect # Added for Callable type inspection
from typing import Dict, Any, Type, Optional, List, Callable, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass

from .base_runner import AgentRunner, SimulationState, ToolCall, AgentRunnerError
from .runner_factory import RunnerFactory
from events import BaseEvent, SetPriceCommand, TickEvent # Assuming these are common events
from services.world_store import WorldStore
from money import Money 

# For type hinting cycles
if TYPE_CHECKING:
    from event_bus import EventBus # For type hinting only
    from constraints.budget_enforcer import BudgetEnforcer
    from metrics.trust_metrics import TrustMetrics
    from constraints.agent_gateway import AgentGateway


logger = logging.getLogger(__name__)


@dataclass
class AgentRegistration:
    """Registration information for an agent."""
    agent_id: str
    runner: AgentRunner
    framework: str
    is_active: bool = True
    failure_reason: Optional[str] = None
    last_tick_processed: int = -1 # Track the last tick this agent processed

class AgentManager:
    """
    Manages the lifecycle and interaction of multiple agent runners.
    
    The AgentManager is responsible for:
    - Registering different agent instances, potentially using different frameworks.
    - Orchestrating decision-making cycles for all active agents.
    - Passing canonical simulation state to agents.
    - Collecting and validating tool calls (actions) from agents.
    - Publishing agent-related events (e.g., AgentDecisionEvent).
    - Handling agent-specific constraints (e.g., budget enforcement via AgentGateway).
    """
    
    def __init__(self, 
                 event_bus: Optional['EventBus'] = None, # Use forward reference
                 world_store: Optional[WorldStore] = None,
                 budget_enforcer: Optional['BudgetEnforcer'] = None, # Use forward reference
                 trust_metrics: Optional['TrustMetrics'] = None, # Use forward reference
                 agent_gateway: Optional['AgentGateway'] = None, # Use forward reference
                 bot_config_dir: str = "baseline_bots/configs",
                 openrouter_api_key: Optional[str] = None):
        
        self.event_bus = event_bus if event_bus else get_event_bus() # Ensure event bus is always available
        self.world_store = world_store
        self.budget_enforcer = budget_enforcer
        self.trust_metrics = trust_metrics
        self.agent_gateway = agent_gateway
        self.bot_config_dir = bot_config_dir
        self.openrouter_api_key = openrouter_api_key

        self.agent_registry: Dict[str, AgentRegistration] = {} # agent_id -> AgentRegistration
        self.bot_factory: Optional[BotFactory] = None # Will be initialized in start()
        self.last_global_state: Optional[SimulationState] = None # Last state provided to agents
        
        # Statistics
        self.decision_cycles_completed = 0
        self.total_tool_calls = 0
        self.total_decisions_skipped = 0
        self.total_errors = 0
        
        logger.info("AgentManager initialized.")

    async def start(self) -> None:
        """Start the AgentManager and its agents."""
        logger.info(f"AgentManager for {self.agent_registry.agent_count()} agents starting.")
        
        # Initialize BotFactory here so it has access to live service instances
        self.bot_factory = BotFactory(
            config_dir=self.bot_config_dir,
            world_store=self.world_store,
            budget_enforcer=self.budget_enforcer,
            trust_metrics=self.trust_metrics,
            agent_gateway=self.agent_gateway,
            openrouter_api_key=self.openrouter_api_key
        )

        # AgentManager needs to subscribe to relevant events for its operations
        if self.event_bus:
            # Example subscriptions - agent manager might listen to ticks to trigger decisions
            await self.event_bus.subscribe(TickEvent, self._handle_tick_event_for_agents)
            await self.event_bus.subscribe(SetPriceCommand, self._handle_agent_command_acknowledgement) # Monitor agent actions
            logger.info("AgentManager subscribed to core events.")

        # Perform setup for each agent runner
        for agent_id, agent_reg in self.agent_registry.all_agents().items():
            try:
                await agent_reg.runner.initialize(agent_reg.config)
            except AgentRunnerInitializationError as e:
                logger.error(f"Failed to initialize agent {agent_id}: {e}")
                self.agent_registry.mark_agent_as_failed(agent_id, str(e))
        logger.info(f"AgentManager started. {self.agent_registry.active_agent_count()} agents active.")

    async def stop(self) -> None:
        """Stop the AgentManager and its agents."""
        logger.info("AgentManager stopping.")

        # Unsubscribe from events (optional, for explicit cleanup)
        if self.event_bus:
            # EventBus typically doesn't offer unsubscribe directly, but in a full system you would.
            pass

        for agent_id, agent_reg in self.agent_registry.all_agents().items():
            try:
                await agent_reg.runner.cleanup()
            except AgentRunnerCleanupError as e:
                logger.warning(f"Failed to cleanup agent {agent_id}: {e}")
        logger.info("AgentManager stopped.")

    def register_agent(self, agent_id: str, framework: str, config: Dict[str, Any]) -> None:
        """
        Registers a new agent runner with the manager.
        
        Args:
            agent_id: Unique identifier for the agent
            framework: The framework this agent uses (e.g., "diy", "crewai", "langchain")
            config: Framework-specific configuration for this agent
        """
        if agent_id in self.agent_registry:
            logger.warning(f"Agent {agent_id} already registered. Skipping.")
            return

        # Attempt to create the agent runner using the factory
        # Create a temporary BotFactory specifically for agent creation if not already provided
        # This is a fallback, ideally a shared factory instance from orchestrator level should be passed
        if not self.bot_factory:
            self.bot_factory = BotFactory(
                config_dir=self.bot_config_dir,
                world_store=self.world_store,
                budget_enforcer=self.budget_enforcer,
                trust_metrics=self.trust_metrics,
                agent_gateway=self.agent_gateway,
                openrouter_api_key=self.openrouter_api_key
            )

        try:
            # Assuming framework string maps to bot_name in BotFactory
            # This is a simplification; a more robust mapping might be needed
            # For now, if framework is "diy" and bot type is "greedy_script", map it directly
            if framework == "diy" and config.get("agent_type") == "baseline" and config.get("custom_config", {}).get("bot_type") == "greedy":
                runner = self.bot_factory.create_bot(bot_name="GreedyScript", tier="T1") # Hardcode tier for initial test compatibility
                # Note: The tier should ideally come from the global simulation config or agent config.
                # Hardcoding for now to fix bot creation.
            else:
                # For other frameworks or more complex DIY agents, use RunnerFactory
                runner = RunnerFactory.create_runner(
                    framework, 
                    agent_id, 
                    config, 
                    world_store=self.world_store,
                    budget_enforcer=self.budget_enforcer,
                    trust_metrics=self.trust_metrics,
                    agent_gateway=self.agent_gateway
                )
            
            self.agent_registry[agent_id] = AgentRegistration(agent_id, runner, framework, config=config)
            logger.info(f"Agent {agent_id} ({framework}) registered successfully.")
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id} ({framework}): {e}")
            self.agent_registry[agent_id] = AgentRegistration(agent_id, None, framework, is_active=False, failure_reason=str(e))
            self.total_errors += 1

    def deregister_agent(self, agent_id: str) -> None:
        """Deregisters an agent from the manager."""
        if agent_id not in self.agent_registry:
            logger.warning(f"Agent {agent_id} not found, cannot deregister.")
            return
        
        del self.agent_registry[agent_id]
        logger.info(f"Agent {agent_id} deregistered.")

    async def run_decision_cycle(self) -> None:
        """
        Executes a decision-making cycle for all active agents.
        
        This method retrieves the current simulation state, passes it to each
        active agent, collects their tool calls, and publishes them.
        """
        if not self.agent_registry.active_agent_count() > 0:
            logger.debug("No active agents to run decision cycle.")
            self.total_decisions_skipped += 1
            return
        
        # Get the latest global simulation state (from WorldStore, etc.)
        # This is a simplified state; in a real scenario, this would aggregate more data.
        current_state = SimulationState(
            tick=self.event_bus.get_current_tick() if self.event_bus else 0, # Assuming event bus tracks tick
            simulation_time=datetime.now(),
            products=self.world_store.get_all_product_states().values() if self.world_store else [],
            recent_events=self.event_bus.get_recorded_events() if self.event_bus and self.event_bus.is_recording else []
        )
        self.last_global_state = current_state

        logger.debug(f"Running decision cycle for {self.agent_registry.active_agent_count()} agents at tick {current_state.tick}...")
        
        # Concurrently get decisions from all active agents
        decision_tasks = []
        for agent_id, agent_reg in self.agent_registry.active_agents().items():
            if agent_reg.runner:
                task = asyncio.create_task(self._get_agent_decision(agent_reg.runner, current_state))
                decision_tasks.append(task)
            else:
                logger.warning(f"Skipping inactive agent {agent_id} in decision cycle.")
                self.total_decisions_skipped += 1

        agent_decisions = await asyncio.gather(*decision_tasks, return_exceptions=True)
        
        # Process decisions (tool calls)
        for i, result in enumerate(agent_decisions):
            agent_id = list(self.agent_registry.active_agents().keys())[i] # Get corresponding agent_id
            if isinstance(result, Exception):
                logger.error(f"Error getting decision from agent {agent_id}: {result}")
                self.agent_registry.mark_agent_as_failed(agent_id, str(result))
                self.total_errors += 1
            else:
                for tool_call in result: # Each agent returns a list of ToolCall objects
                    await self._process_tool_call(agent_id, tool_call)
                    self.total_tool_calls += 1
        
        self.decision_cycles_completed += 1
        logger.debug(f"Decision cycle completed for tick {current_state.tick}.")

    async def _get_agent_decision(self, runner: AgentRunner, state: SimulationState) -> List[ToolCall]:
        """Get decision (tool calls) from a single agent runner."""
        try:
            return await runner.decide(state)
        except AgentRunnerDecisionError as e:
            logger.error(f"Agent '{runner.agent_id}' decision failed: {e}")
            return [] # Return empty list on failure
        except Exception as e:
            logger.error(f"Unexpected error during agent '{runner.agent_id}' decision: {e}")
            return []

    async def _process_tool_call(self, agent_id: str, tool_call: ToolCall) -> None:
        """Processes a tool call from an agent."""
        logger.info(f"Agent {agent_id} proposes ToolCall: {tool_call.tool_name} with {tool_call.parameters}")
        
        # In a real system, this would involve routing to the actual tool implementation
        # For now, we'll just log and acknowledge
        if self.agent_gateway:
            # The agent gateway would validate and execute the tool call, publishing events
            logger.debug(f"Tool call '{tool_call.tool_name}' for agent {agent_id} submitted to AgentGateway.")
            await self.agent_gateway.process_tool_call(agent_id, tool_call, self.world_store, self.event_bus)
        else:
            logger.warning(f"No AgentGateway configured, cannot process tool call: {tool_call.tool_name}")

    async def _handle_tick_event_for_decision_cycle(self, event: TickEvent) -> None:
        """Handle TickEvent to trigger agent decision cycles."""
        logger.debug(f"AgentManager received TickEvent for tick {event.tick_number}. Triggering decision cycle.")
        await self.run_decision_cycle()

    async def _handle_agent_command_acknowledgement(self, event: SetPriceCommand) -> None:
        """Handle agent commands being acknowledged or processed by WorldStore."""
        logger.debug(f"Agent manager received acknowledgement for command: {event.event_id} from agent {event.agent_id}")
        # In a real system, this might update internal agent state on command status


class AgentRegistry:
    """Manages the registration and state of individual agents for easy lookup."""
    
    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}
        
    def add_agent(self, agent_id: str, runner: AgentRunner, framework: str, config: Dict[str, Any]):
        if agent_id in self._agents:
            logger.warning(f"Agent {agent_id} already exists in registry, overwriting.")
        self._agents[agent_id] = AgentRegistration(agent_id, runner, framework, config)
        logger.debug(f"Agent {agent_id} added to registry.")

    def get_agent(self, agent_id: str) -> Optional[AgentRunner]:
        if agent_id in self._agents and self._agents[agent_id].is_active:
            return self._agents[agent_id].runner
        return None

    def all_agents(self) -> Dict[str, AgentRegistration]:
        return self._agents.copy()

    def active_agents(self) -> Dict[str, AgentRegistration]:
        return {agent_id: reg for agent_id, reg in self._agents.items() if reg.is_active}

    def agent_count(self) -> int:
        return len(self._agents)

    def active_agent_count(self) -> int:
        return len(self.active_agents())

    def mark_agent_as_failed(self, agent_id: str, reason: str):
        if agent_id in self._agents:
            self._agents[agent_id].is_active = False
            self._agents[agent_id].failure_reason = reason
            logger.error(f"Agent {agent_id} marked as failed: {reason}")