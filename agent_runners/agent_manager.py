from __future__ import annotations
import logging
import asyncio
import os
import yaml
import inspect # Added for Callable type inspection
from typing import Dict, Any, Type, Optional, List, Callable, TYPE_CHECKING, Union
from datetime import datetime

# Import AgentRegistration, AgentRunnerDecisionError, AgentRunnerCleanupError from base_runner
from .base_runner import (
    AgentRunner, AgentRunnerStatus, AgentRunnerError,
    AgentRunnerInitializationError, AgentRunnerDecisionError, AgentRunnerCleanupError, AgentRunnerTimeoutError,
)
from fba_bench.core.types import SimulationState, ToolCall, TickEvent, SetPriceCommand, AgentObservation
from agents.multi_domain_controller import MultiDomainController
from agents.skill_coordinator import SkillCoordinator
from agents.skill_modules.base_skill import SkillAction


class AgentRegistration:
    """Registration information for an agent."""
    
    def __init__(self, agent_id: str, runner: AgentRunner, framework: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.runner = runner
        self.framework = framework
        self.config = config
        self.is_active = True
        self.created_at = datetime.now()
# Removed legacy RunnerFactory import; unified AgentFactory is used exclusively
# from .runner_factory import RunnerFactory
# TYPE_CHECKING already imported at module top; no need to import again

if TYPE_CHECKING:
    # Type-only imports to avoid circular dependencies at runtime
    from benchmarking.agents.unified_agent import (
        AgentFactory, UnifiedAgentRunner, AgentContext, AgentAction, AgentState, PydanticAgentConfig
    )
    from event_bus import EventBus
    from constraints.budget_enforcer import BudgetEnforcer
    from metrics.trust_metrics import TrustMetrics
    from constraints.agent_gateway import AgentGateway
    from services.world_store import WorldStore


logger = logging.getLogger(__name__)


# Moved AgentRegistry class definition to the top, before AgentManager
class AgentRegistry:
    """Manages the registration and state of individual agents for easy lookup."""
    
    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}
        
    def add_agent(self, agent_id: str, runner: AgentRunner, framework: str, config: Dict[str, Any]):
        if agent_id in self._agents:
            logger.warning(f"Agent {agent_id} already exists in registry, overwriting.")
        self._agents[agent_id] = AgentRegistration(agent_id, runner, framework, config)
        logger.debug(f"Agent {agent_id} added to registry.")

    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]: # Changed return type to AgentRegistration
        if agent_id in self._agents: # Removed .is_active check from here, get_agent returns registration regardless
            return self._agents[agent_id]
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
        else:
            logger.warning(f"Attempted to mark non-existent agent {agent_id} as failed.")


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
                 event_bus: Optional['EventBus'] = None,  # Use forward reference
                 world_store: Optional['WorldStore'] = None,  # Use forward reference
                 budget_enforcer: Optional['BudgetEnforcer'] = None,  # Use forward reference
                 trust_metrics: Optional['TrustMetrics'] = None,  # Use forward reference
                 agent_gateway: Optional['AgentGateway'] = None,  # Use forward reference
                 bot_config_dir: str = "baseline_bots/configs",
                 openrouter_api_key: Optional[str] = None) -> None:
        
        # Lazy import to avoid import-time cycles
        if event_bus is None:
            from event_bus import get_event_bus as _get_event_bus
            self.event_bus = _get_event_bus()
        else:
            self.event_bus = event_bus
        self.world_store = world_store
        self.budget_enforcer = budget_enforcer
        self.trust_metrics = trust_metrics
        self.agent_gateway = agent_gateway
        self.bot_config_dir = bot_config_dir
        self.openrouter_api_key = openrouter_api_key
 
        self.agent_registry: AgentRegistry = AgentRegistry()  # agent_id -> AgentRegistration
        self.last_global_state: Optional[SimulationState] = None  # Last state provided to agents
        # CEO-level controller per agent for action arbitration across skills/tools
        self.multi_domain_controllers: Dict[str, MultiDomainController] = {}
 
        # Unified agent system components (lazy import to avoid circulars)
        self.unified_agent_factory = None
        self.unified_agent_runners: Dict[str, Any] = {}
        try:
            from benchmarking.agents.unified_agent import AgentFactory, UnifiedAgentRunner  # runtime import
            self.unified_agent_factory = AgentFactory()
        except Exception as e:
            logger.error(f"Unified agent system unavailable: {e}")
            raise
        
        # Statistics
        self.decision_cycles_completed = 0
        self.total_tool_calls = 0
        self.total_decisions_skipped = 0
        self.total_errors = 0
        
        logger.info("AgentManager initialized with unified agent system.")

    async def start(self) -> None:
        """Start the AgentManager and its agents."""
        logger.info(f"AgentManager for {self.agent_registry.agent_count()} agents starting.")

        # AgentManager needs to subscribe to relevant events for its operations
        if self.event_bus:
            # Example subscriptions - agent manager might listen to ticks to trigger decisions
            await self.event_bus.subscribe(TickEvent, self._handle_tick_event_for_decision_cycle)
            await self.event_bus.subscribe(SetPriceCommand, self._handle_agent_command_acknowledgement) # Monitor agent actions
            logger.info("AgentManager subscribed to core events.")

        # Perform setup for each agent runner
        for agent_id, agent_reg in self.agent_registry.all_agents().items():
            try:
                # Ensure runner exists before initializing
                if agent_reg.runner:
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
                if agent_reg.runner: # Ensure runner exists before calling cleanup
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
        # Changed get_agent to return AgentRegistration, so check if runner exists
        if self.agent_registry.get_agent(agent_id) and self.agent_registry.get_agent(agent_id).runner:
            logger.warning(f"Agent {agent_id} already registered. Skipping.")
            return

        try:
            # Always use the unified agent system
            pydantic_config = self._create_pydantic_config_from_dict(agent_id, framework, config)
 
            unified_agent = self.unified_agent_factory.create_agent(agent_id, pydantic_config)
 
            unified_runner = UnifiedAgentRunner(unified_agent)
 
            runner_wrapper = UnifiedAgentRunnerWrapper(unified_runner, agent_id)
 
            self.agent_registry.add_agent(agent_id, runner_wrapper, framework, config)
            self.unified_agent_runners[agent_id] = unified_runner
 
            logger.info(f"Unified agent {agent_id} ({framework}) registered successfully.")
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id} ({framework}): {e}")
            # Add agent as failed
            self.agent_registry.add_agent(agent_id, None, framework, config) # Add with None runner
            self.agent_registry.mark_agent_as_failed(agent_id, str(e))
            self.total_errors += 1

    def deregister_agent(self, agent_id: str) -> None:
        """Deregisters an agent from the manager."""
        # AgentRegistry doesn't have a direct deregister or delete method.
        # We can mark it as failed or implement a remove method in AgentRegistry.
        # For now, let's mark it as failed if it exists.
        if self.agent_registry.get_agent(agent_id) is not None:
            self.agent_registry.mark_agent_as_failed(agent_id, "Deregistered by user")
            logger.info(f"Agent {agent_id} deregistered (marked as failed).")
        else:
            logger.warning(f"Agent {agent_id} not found, cannot deregister.")

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
            products=list(self.world_store.get_all_product_states().values()) if self.world_store else [], # Ensure products is a list
            recent_events=list(self.event_bus.get_recorded_events()) if self.event_bus and self.event_bus.is_recording else []
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
        
        # Process decisions (tool calls) via CEO-level arbitration
        for i, result in enumerate(agent_decisions):
            agent_id = list(self.agent_registry.active_agents().keys())[i]  # Get corresponding agent_id
            if isinstance(result, Exception):
                logger.error(f"Error getting decision from agent {agent_id}: {result}")
                self.agent_registry.mark_agent_as_failed(agent_id, str(result))
                self.total_errors += 1
            else:
                processed = await self._arbitrate_and_dispatch(agent_id, result)  # result: List[ToolCall]
                self.total_tool_calls += processed
        
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
    
    def _get_multi_domain_controller(self, agent_id: str) -> MultiDomainController:
        """
        Lazily create or fetch the MultiDomainController for an agent.
        """
        controller = self.multi_domain_controllers.get(agent_id)
        if controller is None:
            coordinator = SkillCoordinator(agent_id, self.event_bus, config={})
            controller = MultiDomainController(agent_id=agent_id, skill_coordinator=coordinator, config={})
            self.multi_domain_controllers[agent_id] = controller
        return controller

    def _toolcall_to_skillaction(self, tool_call: ToolCall) -> SkillAction:
        """
        Adapt a ToolCall (from an agent runner) into a SkillAction for arbitration.
        """
        try:
            pr = getattr(tool_call, "priority", 0)
            priority = max(0.0, min(1.0, float(pr) / 100.0))
        except Exception:
            priority = 0.5
        reasoning = getattr(tool_call, "reasoning", "") or ""
        confidence = getattr(tool_call, "confidence", 0.5)
        return SkillAction(
            action_type=tool_call.tool_name,
            parameters=tool_call.parameters or {},
            confidence=float(confidence),
            reasoning=reasoning,
            priority=priority,
            resource_requirements={},
            expected_outcome={},
            skill_source="agent_runner"
        )

    def _skillaction_to_toolcall(self, action: SkillAction) -> ToolCall:
        """
        Convert an approved SkillAction back into a ToolCall for execution.
        """
        try:
            pr_int = int(round(max(0.0, min(1.0, float(action.priority))) * 100))
        except Exception:
            pr_int = 50
        return ToolCall(
            tool_name=action.action_type,
            parameters=action.parameters,
            confidence=float(action.confidence),
            reasoning=action.reasoning,
            priority=pr_int
        )

    async def _arbitrate_and_dispatch(self, agent_id: str, tool_calls: List[ToolCall]) -> int:
        """
        Convert tool calls to skill actions, run CEO-level arbitration, and dispatch approved actions.
        Returns the number of tool calls processed.
        """
        if not tool_calls:
            return 0
        try:
            controller = self._get_multi_domain_controller(agent_id)
            skill_actions = [self._toolcall_to_skillaction(tc) for tc in tool_calls]
            approved_actions = await controller.arbitrate_actions(skill_actions)
            count = 0
            for action in approved_actions:
                tc = self._skillaction_to_toolcall(action)
                await self._process_tool_call(agent_id, tc)
                count += 1
            return count
        except Exception as e:
            logger.error(f"Arbitration/dispatch failed for agent {agent_id}: {e}", exc_info=True)
            # Fallback: process original tool calls without arbitration
            count = 0
            for tc in tool_calls:
                await self._process_tool_call(agent_id, tc)
                count += 1
            return count

    def _create_pydantic_config_from_dict(self, agent_id: str, framework: str, config: Dict[str, Any]) -> PydanticAgentConfig:
        """Create a PydanticAgentConfig instance from a raw dictionary and inject core services for DIY/LLM bots."""
        llm_config_dict = config.get('llm_config', {}) or {}
        llm_config = {
            'model': llm_config_dict.get('model', 'gpt-3.5-turbo'),
            'temperature': llm_config_dict.get('temperature', 0.1),
            'max_tokens': llm_config_dict.get('max_tokens', 1000),
            'api_key': llm_config_dict.get('api_key', self.openrouter_api_key),
            'base_url': llm_config_dict.get('base_url'),
            'timeout': llm_config_dict.get('timeout', 60),
            'top_p': llm_config_dict.get('top_p', 1.0),
        }
 
        # Extract general parameters
        agent_params = config.get('parameters', {}) or {}
        custom_config = (config.get('custom_config', {}) or {}).copy()
 
        # Inject core services to enable unified factory to build DIY/LLM baseline bots
        # These objects are used by baseline LLM bots (PromptAdapter, ResponseParser, AgentGateway)
        custom_config.setdefault('_services', {})
        custom_config['_services'].update({
            'world_store': self.world_store,
            'budget_enforcer': self.budget_enforcer,
            'trust_metrics': self.trust_metrics,
            'agent_gateway': self.agent_gateway,
            'openrouter_api_key': self.openrouter_api_key,
        })
 
        return PydanticAgentConfig(
            agent_id=agent_id,
            framework=framework,
            llm_config=llm_config,
            parameters=agent_params,
            custom_config=custom_config,
        )

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


class UnifiedAgentRunnerWrapper(AgentRunner):
    """
    Wrapper class that adapts the UnifiedAgentRunner to the AgentRunner interface.
    This allows the unified agent system to work with the existing AgentManager.
    """
    
    def __init__(self, unified_runner: UnifiedAgentRunner, agent_id: str):
        """Initialize the wrapper."""
        super().__init__(agent_id)
        self.unified_runner = unified_runner
        self.agent_id = agent_id
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the unified agent runner."""
        try:
            await self.unified_runner.initialize()
            logger.info(f"Unified agent runner wrapper {self.agent_id} initialized")
        except Exception as e:
            raise AgentRunnerInitializationError(f"Failed to initialize unified agent runner {self.agent_id}: {e}")
    
    async def decide(self, state: SimulationState) -> List[ToolCall]:
        """Get decision from the unified agent runner."""
        try:
            # Convert SimulationState to AgentContext
            agent_context = self._convert_to_agent_context(state)
            
            # Get actions from the unified agent
            actions = await self.unified_runner.decide(agent_context)
            
            # Convert AgentAction to ToolCall
            tool_calls = []
            for action in actions:
                tool_calls.append(ToolCall(
                    tool_name=action.action_type,
                    parameters=action.parameters,
                    confidence=action.confidence,
                    reasoning=action.reasoning
                ))
            
            return tool_calls
            
        except Exception as e:
            raise AgentRunnerDecisionError(f"Failed to get decision from unified agent {self.agent_id}: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup the unified agent runner."""
        try:
            await self.unified_runner.cleanup()
            logger.info(f"Unified agent runner wrapper {self.agent_id} cleaned up")
        except Exception as e:
            raise AgentRunnerCleanupError(f"Failed to cleanup unified agent runner {self.agent_id}: {e}")
    
    def _convert_to_agent_context(self, state: SimulationState) -> AgentContext:
        """Convert a SimulationState to an AgentContext."""
        # Convert recent events to observations
        observations = []
        for event in state.recent_events:
            if isinstance(event, dict):
                observations.append(AgentObservation(
                    observation_type="event",
                    data=event,
                    source="event_bus"
                ))
        
        # Create the agent context
        return AgentContext(
            agent_id=self.agent_id,
            scenario_id=(
                (state.agent_state.get("scenario_id") if isinstance(state.agent_state, dict) else None)
                or (state.agent_state.get("scenario") if isinstance(state.agent_state, dict) else None)
                or "default"
            ),
            tick=state.tick,
            world_state={
                "products": state.products,
                "simulation_time": state.simulation_time.isoformat() if state.simulation_time else None
            },
            observations=observations,
            messages=[],  # No messages in the current SimulationState
            previous_actions=[],  # No previous actions in the current SimulationState
            metadata={
                "simulation_time": state.simulation_time
            }
        )
