"""
AdvancedAgent for FBA-Bench v3 multi-agent platform.

Demonstrates the sandboxed agent pattern where agents can only interact
with the world through commands published to the EventBus.
"""

import os
import asyncio
import logging
import uuid
import random
import json
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from money import Money
from events import BaseEvent, SetPriceCommand, ProductPriceUpdated, TickEvent, PlaceOrderCommand, RespondToCustomerMessageCommand, RunMarketingCampaignCommand
from event_bus import EventBus, get_event_bus
from constraints.agent_gateway import AgentGateway
from typing import Literal

# OpenTelemetry Imports
from instrumentation.tracer import setup_tracing
from instrumentation.agent_tracer import AgentTracer
from opentelemetry import trace

# Cognitive Architecture Imports
from .hierarchical_planner import StrategicPlanner, TacticalPlanner, StrategicObjective, TacticalAction
from .cognitive_config import CognitiveConfig, get_cognitive_config
from memory_experiments.reflection_module import StructuredReflectionLoop, ReflectionTrigger
from memory_experiments.memory_validator import MemoryConsistencyChecker, MemoryIntegrationGateway
from memory_experiments.memory_enforcer import MemoryEnforcer
from memory_experiments.memory_config import MemoryConfig

# Multi-Skill Architecture Imports
from .skill_coordinator import SkillCoordinator
from .multi_domain_controller import MultiDomainController
from .skill_config import SkillConfigurationManager, get_skill_config
from .skill_modules import (
    SupplyManagerSkill, MarketingManagerSkill,
    CustomerServiceSkill, FinancialAnalystSkill
)
from .skill_modules.base_skill import SkillContext

# Reproducibility System Imports
from reproducibility.sim_seed import SimSeed
from reproducibility.simulation_modes import SimulationMode, get_mode_controller
from reproducibility.reproducibility_config import get_global_config
from reproducibility.event_snapshots import EventSnapshot
from llm_interface.deterministic_client import DeterministicLLMClient, create_deterministic_client, create_hybrid_client
from llm_interface.openrouter_client import OpenRouterClient
from infrastructure.llm_batcher import LLMBatcher # Import LLMBatcher
from infrastructure.resource_manager import ResourceManager # For token allocation


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for AdvancedAgent behavior."""
    agent_id: str
    target_asin: str
    strategy: str = "profit_maximizer"
    price_sensitivity: float = 0.1  # How much to adjust prices (0.0 to 1.0)
    reaction_speed: int = 1  # How often to react (every N ticks)
    min_price: Money = Money(500)  # Minimum price $5.00
    max_price: Money = Money(5000)  # Maximum price $50.00
    
    # Cognitive capabilities configuration
    cognitive_enabled: bool = True
    cognitive_config_template: str = "production"  # "development", "production", "research"
    enable_hierarchical_planning: bool = True
    enable_structured_reflection: bool = True
    enable_memory_validation: bool = True
    
    # Multi-skill system configuration
    enable_multi_skill_system: bool = True
    skill_config_template: str = "balanced"  # "balanced", "crisis", "growth", "optimization"
    
    # Reproducibility and Scalability configuration
    enable_reproducibility: bool = True
    simulation_mode: SimulationMode = SimulationMode.DETERMINISTIC
    component_seed_salt: Optional[str] = None  # If None, uses agent_id
    llm_cache_file: str = "agent_llm_responses.cache"
    enable_llm_interaction_logging: bool = True
    enable_deterministic_validation: bool = True
    enable_supply_management: bool = True
    enable_marketing_management: bool = True
    enable_customer_service: bool = True
    enable_financial_analysis: bool = True
    enable_llm_batching: bool = False # New: Enable LLM request batching by agent


class AdvancedAgent:
    """
    Sandboxed AI agent for FBA-Bench v3 multi-agent platform.
    
    Demonstrates the core multi-agent principles:
    1. No direct access to world state - all perception through events
    2. No direct actions - all intents expressed as commands
    3. Command-arbitration-event loop for all interactions
    4. Complete isolation from other services and agents
    
    The agent subscribes to relevant events to build its world model,
    then publishes SetPriceCommand events to express pricing intentions.
    The WorldStore arbitrates these commands and publishes canonical updates.
    """
    
    def __init__(self, config: AgentConfig, event_bus: Optional[EventBus] = None, gateway: Optional[AgentGateway] = None, 
                 llm_batcher: Optional[LLMBatcher] = None, resource_manager: Optional[ResourceManager] = None):
        """
        Initialize the AdvancedAgent with enhanced cognitive capabilities.
        
        Args:
            config: Agent configuration including strategy and constraints
            event_bus: EventBus for communication (sandboxed interface)
            gateway: AgentGateway instance for budget enforcement
            llm_batcher: Optional LLMBatcher instance for batched LLM requests.
            resource_manager: Optional ResourceManager instance for token/cost tracking.
        """
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        self.gateway = gateway
        self.llm_batcher = llm_batcher # New: LLM batcher instance
        self.resource_manager = resource_manager # New: Resource manager instance
        
        # OpenTelemetry setup for this agent instance
        # Each agent gets its own tracer to better isolate spans by service/agent.
        self.tracer = setup_tracing(service_name=f"fba-bench-agent-{config.agent_id}")
        self.agent_tracer = AgentTracer(self.tracer)

        if self.gateway is None:
            logger.warning(f"Agent {self.config.agent_id} initialized without AgentGateway. Budget enforcement will be skipped.")

        # Agent state (built from events only)
        self.current_tick = 0
        self.last_action_tick = 0
        self.last_reflection_tick = 0
        self.current_price: Optional[Money] = None
        self.price_history: List[Money] = []
        self.inventory_level: int = 0
        self.customer_messages: List[Dict[str, Any]] = []
        self.campaign_status: Dict[str, Any] = {}
        self.financial_data: Dict[str, Any] = {}
        
        # Market perception (from events)
        self.market_data: Dict[str, Any] = {}
        self.competitor_prices: List[Money] = []
        
        # Agent decision-making state
        self.decision_history: List[Dict[str, Any]] = []
        self.commands_sent = 0
        self.commands_accepted = 0
        
        # Reproducibility system integration
        self._reproducibility_initialized = False
        self.llm_client: Optional[DeterministicLLMClient] = None
        self.component_seed: Optional[int] = None
        self._mode_controller = None
        
        # Dictionary to store pending LLM responses from the batcher
        self._pending_llm_responses: Dict[str, asyncio.Future] = {}
        
        if config.enable_reproducibility:
            self._initialize_reproducibility()
        
        # Performance tracking for cognitive systems
        self.performance_metrics: Dict[str, float] = {
            "decision_success_rate": 0.5,
            "response_time": 2.0,
            "profit_margin": 0.1,
            "market_share": 0.05
        }
        self.major_events: List[Dict[str, Any]] = []
        
        # Initialize cognitive architecture
        self.cognitive_config: Optional[CognitiveConfig] = None
        self.strategic_planner: Optional[StrategicPlanner] = None
        self.tactical_planner: Optional[TacticalPlanner] = None
        self.reflection_loop: Optional[StructuredReflectionLoop] = None
        self.memory_consistency_checker: Optional[MemoryConsistencyChecker] = None
        self.memory_integration_gateway: Optional[MemoryIntegrationGateway] = None
        
        # Initialize multi-skill system
        self.skill_coordinator: Optional[SkillCoordinator] = None
        self.multi_domain_controller: Optional[MultiDomainController] = None
        self.skill_config_manager: Optional[SkillConfigurationManager] = None
        self.skills: Dict[str, Any] = {}
        
        if config.cognitive_enabled:
            self._initialize_cognitive_architecture()
            
        if config.enable_multi_skill_system:
            self._initialize_skill_system()
        
        logger.info(f"AdvancedAgent initialized: id={config.agent_id}, target={config.target_asin}, "
                   f"strategy={config.strategy}, cognitive_enabled={config.cognitive_enabled}, "
                   f"multi_skill_enabled={config.enable_multi_skill_system}, "
                   f"llm_batching_enabled={config.enable_llm_batching}")
    
    def _initialize_cognitive_architecture(self):
        """Initialize all cognitive components."""
        try:
            # Initialize cognitive configuration
            self.cognitive_config = get_cognitive_config(
                self.config.cognitive_config_template,
                self.config.agent_id
            )
            
            # Initialize hierarchical planning system
            if self.config.enable_hierarchical_planning:
                self.strategic_planner = StrategicPlanner(
                    self.config.agent_id,
                    self.event_bus
                )
                self.tactical_planner = TacticalPlanner(
                    self.config.agent_id,
                    self.strategic_planner,
                    self.event_bus
                )
                logger.info(f"Hierarchical planning system initialized for agent {self.config.agent_id}")
            
            # Initialize memory validation system
            if self.config.enable_memory_validation and self.gateway and hasattr(self.gateway, 'memory_enforcer'):
                memory_config = MemoryConfig()  # Use default or get from gateway
                
                self.memory_consistency_checker = MemoryConsistencyChecker(
                    self.config.agent_id,
                    memory_config
                )
                
                self.memory_integration_gateway = MemoryIntegrationGateway(
                    self.config.agent_id,
                    self.gateway.memory_enforcer.memory_manager,
                    self.memory_consistency_checker,
                    memory_config,
                    self.event_bus
                )
                logger.info(f"Memory validation system initialized for agent {self.config.agent_id}")
            
            # Initialize structured reflection system
            if self.config.enable_structured_reflection and self.gateway and hasattr(self.gateway, 'memory_enforcer'):
                self.reflection_loop = StructuredReflectionLoop(
                    self.config.agent_id,
                    self.gateway.memory_enforcer.memory_manager,
                    self.gateway.memory_enforcer.config,
                    self.event_bus
                )
                logger.info(f"Structured reflection system initialized for agent {self.config.agent_id}")
            
            logger.info(f"Cognitive architecture fully initialized for agent {self.config.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize cognitive architecture for agent {self.config.agent_id}: {e}")
            # Graceful degradation - continue without cognitive enhancements
            self.config.cognitive_enabled = False
    
    def _initialize_skill_system(self):
        """Initialize the multi-skill coordination system."""
        try:
            # Initialize skill configuration manager
            self.skill_config_manager = SkillConfigurationManager()
            skill_config = get_skill_config(
                self.config.skill_config_template,
                self.config.agent_id
            )
            
            # Initialize skill coordinator
            self.skill_coordinator = SkillCoordinator(
                self.config.agent_id,
                self.event_bus
            )
            
            # Initialize multi-domain controller
            self.multi_domain_controller = MultiDomainController(
                self.config.agent_id,
                self.skill_coordinator
            )
            
            # Initialize and register skills
            if self.config.enable_supply_management:
                supply_skill = SupplyManagerSkill(self.config.agent_id, skill_config)
                self.skills["supply_management"] = supply_skill
                self.skill_coordinator.register_skill(supply_skill)
                logger.info(f"Supply management skill registered for agent {self.config.agent_id}")
            
            if self.config.enable_marketing_management:
                marketing_skill = MarketingManagerSkill(self.config.agent_id, skill_config)
                self.skills["marketing_management"] = marketing_skill
                self.skill_coordinator.register_skill(marketing_skill)
                logger.info(f"Marketing management skill registered for agent {self.config.agent_id}")
            
            if self.config.enable_customer_service:
                customer_skill = CustomerServiceSkill(self.config.agent_id, skill_config)
                self.skills["customer_service"] = customer_skill
                self.skill_coordinator.register_skill(customer_skill)
                logger.info(f"Customer service skill registered for agent {self.config.agent_id}")
            
            if self.config.enable_financial_analysis:
                financial_skill = FinancialAnalystSkill(self.config.agent_id, skill_config)
                self.skills["financial_analysis"] = financial_skill
                self.skill_coordinator.register_skill(financial_skill)
                logger.info(f"Financial analysis skill registered for agent {self.config.agent_id}")
            
            logger.info(f"Multi-skill system initialized for agent {self.config.agent_id} with {len(self.skills)} skills")
            
        except Exception as e:
            logger.error(f"Failed to initialize skill system for agent {self.config.agent_id}: {e}")
            # Graceful degradation - continue without skill system
            self.config.enable_multi_skill_system = False
    
    def _initialize_reproducibility(self):
        """
        Initialize reproducibility features for the agent.
        
        Sets up deterministic seeding, LLM client, and registers with mode controller.
        """
        try:
            # Get global reproducibility configuration
            repro_config = get_global_config()
            
            # Get mode controller instance
            self._mode_controller = get_mode_controller()
            
            # Set up component-specific seeding
            component_salt = self.config.component_seed_salt or self.config.agent_id
            self.component_seed = SimSeed.get_component_seed(f"agent_{component_salt}")
            
            # Register this agent as a component with the mode controller
            self._mode_controller.register_component(
                name=f"agent_{self.config.agent_id}",
                component=self,
                mode_handlers={
                    SimulationMode.DETERMINISTIC: self._handle_deterministic_mode,
                    SimulationMode.STOCHASTIC: self._handle_stochastic_mode,
                    SimulationMode.RESEARCH: self._handle_research_mode
                }
            )
            
            # Initialize LLM client based on current mode and batching preference
            self._setup_llm_client(repro_config)
            
            # Set up event snapshot integration
            if self.config.enable_llm_interaction_logging:
                EventSnapshot.set_snapshot_metadata(
                    simulation_mode=repro_config.simulation_mode.value,
                    master_seed=repro_config.seed_management.master_seed,
                    reproducibility_features_enabled={
                        "deterministic_llm": True,
                        "component_seeding": True,
                        "interaction_logging": True,
                        "llm_batching": self.config.enable_llm_batching
                    }
                )
            
            self._reproducibility_initialized = True
            logger.info(f"Agent {self.config.agent_id} reproducibility initialized: mode={repro_config.simulation_mode.value}, seed={self.component_seed}, llm_batching={self.config.enable_llm_batching}")
            
        except Exception as e:
            logger.error(f"Failed to initialize reproducibility for agent {self.config.agent_id}: {e}")
            self._reproducibility_initialized = False
    
    def _setup_llm_client(self, repro_config):
        """
        Set up the appropriate LLM client based on configuration.
        
        Args:
            repro_config: Global reproducibility configuration
        """
        try:
            # If LLM batching is enabled, we don't directly create a DeterministicLLMClient here.
            # Instead, the _call_llm_deterministic method will route through the LLMBatcher.
            if self.config.enable_llm_batching and self.llm_batcher:
                self.llm_client = None # Agent will use batcher instead of direct client
                logger.debug(f"Agent {self.config.agent_id} configured to use LLMBatcher for LLM calls.")
                return

            # Create underlying OpenRouter client (or other LLM client)
            underlying_client = OpenRouterClient(
                model_name="openai/gpt-4o-mini",  # Default model
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            
            # Wrap with deterministic client based on simulation mode
            if repro_config.simulation_mode == SimulationMode.DETERMINISTIC:
                self.llm_client = create_deterministic_client(
                    underlying_client=underlying_client,
                    cache_file=self.config.llm_cache_file
                )
            elif repro_config.simulation_mode == SimulationMode.RESEARCH: # Research mode is hybrid
                self.llm_client = create_hybrid_client(
                    underlying_client=underlying_client,
                    cache_file=self.config.llm_cache_file
                )
            else: # Stochastic mode - use underlying client directly
                self.llm_client = underlying_client
            
            logger.debug(f"Agent {self.config.agent_id} LLM client configured: {type(self.llm_client).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to setup LLM client for agent {self.config.agent_id}: {e}")
            self.llm_client = None
    
    def _handle_deterministic_mode(self, component, config):
        """Handle switch to deterministic mode."""
        if hasattr(self.llm_client, 'set_deterministic_mode'):
            self.llm_client.set_deterministic_mode(True, self.config.llm_cache_file)
        logger.debug(f"Agent {self.config.agent_id} switched to deterministic mode")
    
    def _handle_stochastic_mode(self, component, config):
        """Handle switch to stochastic mode."""
        if hasattr(self.llm_client, 'set_deterministic_mode'):
            self.llm_client.set_deterministic_mode(False)
        if hasattr(self.llm_client, 'record_responses'):
            self.llm_client.record_responses(True, self.config.llm_cache_file)
        logger.debug(f"Agent {self.config.agent_id} switched to stochastic mode")
    
    def _handle_research_mode(self, component, config):
        """Handle switch to research mode."""
        # Research mode allows both cached and live responses
        if hasattr(self.llm_client, 'set_deterministic_mode'):
            self.llm_client.set_deterministic_mode(False)
        if hasattr(self.llm_client, 'record_responses'):
            self.llm_client.record_responses(True, self.config.llm_cache_file)
        logger.debug(f"Agent {self.config.agent_id} switched to research mode")

    async def _handle_batched_llm_response(self, request_id: str, response: Optional[Dict[str, Any]], error: Optional[Exception]):
        """
        Callback method for when an LLM batcher request completes.
        Resolves the Future associated with the original request.
        """
        if request_id in self._pending_llm_responses:
            future = self._pending_llm_responses.pop(request_id)
            if error:
                future.set_exception(error)
            else:
                future.set_result(response)
        else:
            logger.warning(f"Received batched LLM response for unknown request_id: {request_id}. Response: {response}, Error: {error}")
    
    async def _call_llm_deterministic(
        self,
        prompt: str,
        temperature: float = 0.0,
        model_name: str = "openai/gpt-4o-mini", # Add model_name param
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = {"type": "json_object"}
    ) -> Dict[str, Any]:
        """
        Make a deterministic LLM call with proper logging and validation,
        potentially routing through LLMBatcher.
        
        Args:
            prompt: Input prompt for the LLM
            temperature: Sampling temperature
            model_name: The name of the LLM model to use.
            max_tokens: Maximum tokens to generate
            response_format: Expected response format
            
        Returns:
            LLM response with metadata
        """
        import time
        
        # If LLM batching is enabled, add request to batcher and wait for response
        if self.config.enable_llm_batching and self.llm_batcher:
            request_id = str(uuid.uuid4())
            future = asyncio.Future()
            self._pending_llm_responses[request_id] = future
            
            self.llm_batcher.add_request(
                request_id=request_id,
                prompt=prompt,
                model=model_name,
                callback=self._handle_batched_llm_response
            )
            
            logger.debug(f"Added LLM request {request_id} to batcher for model {model_name}.")
            
            try:
                # Wait for the future to be set by the batcher's callback
                response = await future
                # Simulate cost and allocate tokens now that response is received
                if self.resource_manager:
                    # Very rough estimate of tokens and cost if not provided by batcher response
                    # In a real system, batcher should return this info
                    estimated_tokens = len(prompt) / 4 + len(str(response)) / 4
                    estimated_cost = (estimated_tokens / 1000) * 0.002 
                    self.resource_manager.record_llm_cost(model_name, estimated_cost, int(estimated_tokens))
                return response
            except Exception as e:
                logger.error(f"Agent {self.config.agent_id} failed to get batched LLM response for {request_id}: {e}")
                # Fallback to simulation on batched call failure
                return self._simulate_llm_response_structured(prompt)


        if not self._reproducibility_initialized or not self.llm_client:
            # Fallback to simulation if reproducibility not available (and no batcher)
            logger.warning(f"Agent {self.config.agent_id} falling back to LLM simulation (no client or batcher).")
            return self._simulate_llm_response_structured(prompt)
        
        start_time = time.time()
        
        try:
            # Use deterministic LLM client if available
            if hasattr(self.llm_client, 'call_llm'):
                response = await self.llm_client.call_llm(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
            else: # For compatibility with older generate_response interface
                response = await self.llm_client.generate_response(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
            
            response_time = (time.time() - start_time) * 1000
            
            # Extract metadata if available
            metadata = response.get('_fba_metadata', {})
            cache_hit = metadata.get('cache_hit', False)
            validation_passed = metadata.get('validation_passed', True)
            
            # Log interaction for snapshot inclusion
            if self.config.enable_llm_interaction_logging:
                # Ensure prompt_hash generation is consistent
                if hasattr(self.llm_client, 'cache') and hasattr(self.llm_client.cache, 'generate_prompt_hash'):
                    prompt_hash = self.llm_client.cache.generate_prompt_hash(prompt, temperature=temperature, model=model_name)
                else:
                    prompt_hash = hashlib.sha256(f"{prompt}{temperature}{model_name}".encode()).hexdigest()[:16]
                
                response_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                response_hash = hashlib.sha256(response_content.encode()).hexdigest()[:16]
                
                EventSnapshot.log_llm_interaction(
                    prompt_hash=prompt_hash,
                    model=model_name,
                    temperature=temperature,
                    cache_hit=cache_hit,
                    response_hash=response_hash,
                    deterministic_mode=self._mode_controller._current_mode == SimulationMode.DETERMINISTIC,
                    validation_passed=validation_passed,
                    response_time_ms=response_time
                )

            # Record LLM cost if resource manager is available
            if self.resource_manager:
                prompt_tokens = response.get('usage', {}).get('prompt_tokens', 0)
                completion_tokens = response.get('usage', {}).get('completion_tokens', 0)
                total_tokens = response.get('usage', {}).get('total_tokens', prompt_tokens + completion_tokens)
                # Assuming a cost per 1k tokens for the model. This needs to be more dynamic.
                # For now, a placeholder cost.
                cost_per_token_model = 0.002 / 1000 # Example: $0.002 per 1k tokens
                estimated_cost = total_tokens * cost_per_token_model
                self.resource_manager.record_llm_cost(model_name, estimated_cost, total_tokens)
            
            logger.debug(f"Agent {self.config.agent_id} LLM call completed: cache_hit={cache_hit}, time={response_time:.1f}ms, model={model_name}")
            
            return response
            
        except Exception as e:
            logger.error(f"Agent {self.config.agent_id} LLM call failed: {e}", exc_info=True)
            # Fallback to simulation on error
            return self._simulate_llm_response_structured(prompt)
    
    def _simulate_llm_multi_action_output(self) -> str:
        """
        Simulates an LLM output with multiple action types for advanced agent.
        Enhanced with tactical planning integration.
        """
        actions = []
        overall_reasoning = "Multi-domain strategy execution based on current business needs."

        # Get ready tactical actions if tactical planner is available
        tactical_actions = []
        if self.config.cognitive_enabled and self.tactical_planner:
            try:
                tactical_actions = self.tactical_planner.get_ready_actions(datetime.now())
                if tactical_actions:
                    overall_reasoning += " Incorporating tactical planning recommendations."
            except Exception as e:
                logger.error(f"Error getting tactical actions: {e}")

        # Priority 1: Execute ready tactical actions
        for tactical_action in tactical_actions[:2]:  # Limit to 2 tactical actions per tick
            action_dict = {
                "action_id": tactical_action.action_id,
                "type": tactical_action.action_type,
                "parameters": tactical_action.parameters.copy(),
                "reasoning": f"Tactical plan: {tactical_action.description}",
                "confidence": 0.9,
                "strategic_objective_id": tactical_action.strategic_objective_id
            }
            actions.append(action_dict)

        # Priority 2: Standard reactive actions if no tactical actions or as supplementary
        if len(actions) < 2:  # Allow some reactive actions if tactical planner isn't providing enough
            
            # Pricing action with enhanced logic
            desired_price = self._calculate_desired_price()
            if desired_price and self._should_change_price(desired_price):
                # Enhanced reasoning with strategic context
                price_reasoning = f"Adjusting price based on {self.config.strategy} strategy"
                if self.strategic_planner and hasattr(self.strategic_planner, 'strategic_objectives'):
                    try:
                        # Simple heuristic alignment check without async call
                        active_objectives = [obj for obj in self.strategic_planner.strategic_objectives.values()
                                           if obj.status.value == "active"]
                        if active_objectives:
                            price_reasoning += f" (Supporting {len(active_objectives)} strategic objectives)"
                    except Exception as e:
                        logger.error(f"Error checking strategic context: {e}")
                
                actions.append({
                    "type": "set_price",
                    "parameters": {"asin": self.config.target_asin, "price": desired_price.to_float()},
                    "reasoning": price_reasoning,
                    "confidence": 0.95
                })

            # Inventory action with performance-based logic
            inventory_threshold = 20
            if self.performance_metrics.get("inventory_turnover", 5) > 8:
                inventory_threshold = 30  # Higher threshold for fast-moving inventory
            
            if self.inventory_level < inventory_threshold:
                order_quantity = 100
                # Adjust order quantity based on recent sales performance
                if self.performance_metrics.get("decision_success_rate", 0.5) > 0.8:
                    order_quantity = 150  # Order more if performing well
                    
                actions.append({
                    "type": "place_order",
                    "parameters": {"supplier_id": "supplier_A", "asin": self.config.target_asin, "quantity": order_quantity, "max_price": 10.00},
                    "reasoning": f"Low inventory ({self.inventory_level} < {inventory_threshold}), placing restock order.",
                    "confidence": 0.9
                })
            
            # Customer service action with prioritization
            if self.customer_messages:
                # Prioritize messages based on age and content
                urgent_messages = [msg for msg in self.customer_messages if "urgent" in msg.get("content", "").lower()]
                latest_message = urgent_messages[0] if urgent_messages else self.customer_messages[-1]
                
                response_content = "Thank you for your inquiry. We are looking into this for you."
                if urgent_messages:
                    response_content = "Thank you for your urgent inquiry. We are prioritizing your request and will respond within 24 hours."
                
                actions.append({
                    "type": "respond_to_customer",
                    "parameters": {"message_id": latest_message["message_id"], "response_content": response_content},
                    "reasoning": "Responding to customer query with appropriate priority.",
                    "confidence": 0.8
                })

            # Marketing action with performance-based timing
            marketing_interval = 100  # Default
            if self.performance_metrics.get("profit_margin", 0.1) > 0.2:
                marketing_interval = 50  # More frequent marketing if profitable
            elif self.performance_metrics.get("profit_margin", 0.1) < 0.05:
                marketing_interval = 200  # Less frequent if low margins
                
            if self.current_tick % marketing_interval == 0:
                # Adjust budget based on performance
                base_budget = 500.0
                performance_multiplier = self.performance_metrics.get("decision_success_rate", 0.5)
                adjusted_budget = base_budget * (0.5 + performance_multiplier)
                
                actions.append({
                    "type": "run_marketing_campaign",
                    "parameters": {"campaign_type": "social_media_ads", "budget": adjusted_budget, "duration_days": 7},
                    "reasoning": f"Initiating marketing campaign with performance-adjusted budget (${adjusted_budget:.0f}).",
                    "confidence": 0.75
                })

        return json.dumps({"actions": actions, "overall_reasoning": overall_reasoning})

    def _simulate_llm_response_structured(self, prompt: str) -> Dict[str, Any]:
        """
        Create a structured response that mimics real LLM output format.
        
        Args:
            prompt: Input prompt (used for deterministic simulation)
            
        Returns:
            Structured response in OpenAI format
        """
        # Use component seed for deterministic simulation
        if self.component_seed is not None:
            with SimSeed.component_context(f"agent_{self.config.agent_id}"):
                simulated_content = self._simulate_llm_multi_action_output()
        else:
            simulated_content = self._simulate_llm_multi_action_output()
        
        return {
            "choices": [
                {
                    "message": {
                        "content": simulated_content,
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()) * 1.3,  # Rough approximation
                "completion_tokens": len(simulated_content.split()) * 1.3,
                "total_tokens": (len(prompt) + len(simulated_content)) * 1.3
            },
            "_fba_metadata": {
                "simulation_mode": "fallback",
                "cache_hit": False,
                "validation_passed": True,
                "response_time_ms": random.uniform(100, 500) if self.component_seed is None else 200.0
            }
        }
    
    def get_reproducibility_status(self) -> Dict[str, Any]:
        """
        Get detailed status of reproducibility features for this agent.
        
        Returns:
            Dictionary with reproducibility status information
        """
        status = {
            "agent_id": self.config.agent_id,
            "reproducibility_enabled": self.config.enable_reproducibility,
            "reproducibility_initialized": self._reproducibility_initialized,
            "component_seed": self.component_seed,
            "llm_client_type": type(self.llm_client).__name__ if self.llm_client else "LLMBatcher" if self.config.enable_llm_batching else None,
            "simulation_mode": self._mode_controller._current_mode.value if self._mode_controller else None,
            "llm_interaction_logging": self.config.enable_llm_interaction_logging,
            "llm_batching_enabled": self.config.enable_llm_batching
        }
        
        # Add LLM client statistics if available
        if hasattr(self.llm_client, 'get_cache_statistics'):
            status["llm_cache_stats"] = self.llm_client.get_cache_statistics()
        
        # If using batcher, provide its stats (conceptual)
        if self.config.enable_llm_batching and self.llm_batcher:
            status["llm_batcher_stats"] = self.llm_batcher.stats.copy()
        
        return status
            

    def _initialize_skill_system(self):
        """Initialize the multi-skill coordination system."""
        try:
            # Initialize skill configuration manager
            self.skill_config_manager = SkillConfigurationManager()
            skill_config = get_skill_config(
                self.config.skill_config_template,
                self.config.agent_id
            )
            
            # Initialize skill coordinator
            self.skill_coordinator = SkillCoordinator(
                self.config.agent_id,
                self.event_bus
            )
            
            # Initialize multi-domain controller
            self.multi_domain_controller = MultiDomainController(
                self.config.agent_id,
                self.skill_coordinator
            )
            
            # Initialize and register skills
            if self.config.enable_supply_management:
                supply_skill = SupplyManagerSkill(self.config.agent_id, skill_config)
                self.skills["supply_management"] = supply_skill
                self.skill_coordinator.register_skill(supply_skill)
                logger.info(f"Supply management skill registered for agent {self.config.agent_id}")
            
            if self.config.enable_marketing_management:
                marketing_skill = MarketingManagerSkill(self.config.agent_id, skill_config)
                self.skills["marketing_management"] = marketing_skill
                self.skill_coordinator.register_skill(marketing_skill)
                logger.info(f"Marketing management skill registered for agent {self.config.agent_id}")
            
            if self.config.enable_customer_service:
                customer_skill = CustomerServiceSkill(self.config.agent_id, skill_config)
                self.skills["customer_service"] = customer_skill
                self.skill_coordinator.register_skill(customer_skill)
                logger.info(f"Customer service skill registered for agent {self.config.agent_id}")
            
            if self.config.enable_financial_analysis:
                financial_skill = FinancialAnalystSkill(self.config.agent_id, skill_config)
                self.skills["financial_analysis"] = financial_skill
                self.skill_coordinator.register_skill(financial_skill)
                logger.info(f"Financial analysis skill registered for agent {self.config.agent_id}")
            
            logger.info(f"Multi-skill system initialized for agent {self.config.agent_id} with {len(self.skills)} skills")
            
        except Exception as e:
            logger.error(f"Failed to initialize skill system for agent {self.config.agent_id}: {e}")
            # Graceful degradation - continue without skill system
            self.config.enable_multi_skill_system = False
    
    async def start(self):
        """Start the agent and subscribe to relevant events."""
        # Subscribe to events for world perception
        await self.event_bus.subscribe('TickEvent', self.handle_tick_event)
        await self.event_bus.subscribe('ProductPriceUpdated', self.handle_price_updated)
        await self.event_bus.subscribe('InventoryLevelUpdated', self.handle_inventory_updated)
        await self.event_bus.subscribe('CustomerMessageReceived', self.handle_customer_message)
        await self.event_bus.subscribe('MarketingCampaignStatus', self.handle_marketing_status)
        await self.event_bus.subscribe('ProductSalesEvent', self.handle_sales_event)
        await self.event_bus.subscribe('FinancialStatementReport', self.handle_financial_report)
        
        # Start skill coordination system
        if self.skill_coordinator:
            await self.skill_coordinator.start()
            logger.info(f"Skill coordination system started for agent {self.config.agent_id}")
        
        logger.info(f"AdvancedAgent {self.config.agent_id} started - subscribed to events")
    
    async def stop(self):
        """Stop the agent and clean up skill coordination system."""
        # Stop skill coordination system
        if self.skill_coordinator:
            await self.skill_coordinator.stop()
            logger.info(f"Skill coordination system stopped for agent {self.config.agent_id}")
        
        logger.info(f"AdvancedAgent {self.config.agent_id} stopped")
    
    # Event Handlers (World Perception)
    
    async def handle_tick_event(self, event: TickEvent):
        """
        Process tick events to trigger agent decision-making with cognitive enhancements.
        
        This is the main decision loop where the agent evaluates its current state,
        performs cognitive processing, and decides whether to take action.
        """
        self.current_tick = event.tick_number
        
        # Wrap the whole agent's turn in a span
        with self.agent_tracer.trace_agent_turn(
            agent_id=self.config.agent_id,
            tick=self.current_tick
        ):
            # Enhanced cognitive processing
            if self.config.cognitive_enabled:
                await self._cognitive_processing(event.timestamp)
            
            # Check if it's time to make a decision or reflection
            if self.gateway and \
               hasattr(self.gateway, 'memory_enforcer') and \
               self.gateway.memory_enforcer and \
               await self.gateway.memory_enforcer.check_reflection_trigger(event.timestamp):
                logger.debug(f"Agent {self.config.agent_id} triggered memory reflection at tick {self.current_tick}")
            
            if self._should_act():
                if self.config.enable_multi_skill_system and self.multi_domain_controller:
                    await self._make_skill_based_decision()
                else:
                    await self._make_multi_domain_decision()
    
    async def _cognitive_processing(self, current_time: datetime):
        """Perform comprehensive cognitive processing each tick."""
        try:
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Trigger structured reflection if needed
            await self._check_and_trigger_reflection(current_time)
            
            # Update strategic and tactical plans
            await self._update_planning_systems(current_time)
            
            # Process major events for cognitive systems
            await self._process_major_events_for_cognition()
            
        except Exception as e:
            logger.error(f"Error in cognitive processing for agent {self.config.agent_id}: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics for cognitive analysis."""
        if not self.config.cognitive_enabled:
            return
        
        # Calculate decision success rate
        if self.commands_sent > 0:
            self.performance_metrics["decision_success_rate"] = self.commands_accepted / self.commands_sent
        
        # Update other metrics based on recent events
        if self.price_history:
            # Simple profit margin estimation (placeholder)
            current_price = self.price_history[-1].to_float() if self.price_history else 20.0
            estimated_cost = 15.0  # Placeholder
            self.performance_metrics["profit_margin"] = max(0, (current_price - estimated_cost) / current_price)
        
        # Update market share estimation (placeholder)
        if self.competitor_prices:
            avg_competitor_price = sum(p.to_float() for p in self.competitor_prices) / len(self.competitor_prices)
            current_price = self.current_price.to_float() if self.current_price else 20.0
            if avg_competitor_price > 0:
                price_competitiveness = current_price / avg_competitor_price
                self.performance_metrics["market_share"] = max(0.01, min(0.2, 0.1 / price_competitiveness))
    
    async def _check_and_trigger_reflection(self, current_time: datetime):
        """Check if structured reflection should be triggered."""
        if not self.reflection_loop:
            return
        
        try:
            # Calculate tick interval for reflection trigger
            tick_interval = self.current_tick - self.last_reflection_tick if self.last_reflection_tick > 0 else None
            
            # Trigger reflection with current major events
            reflection_result = await self.reflection_loop.trigger_reflection(
                tick_interval=tick_interval,
                major_events=self.major_events[-10:]  # Last 10 major events
            )
            
            if reflection_result:
                logger.info(f"Structured reflection completed for agent {self.config.agent_id} - "
                           f"insights: {len(reflection_result.insights)}, "
                           f"adjustments: {len(reflection_result.policy_adjustments)}")
                
                # Apply policy adjustments
                await self._apply_policy_adjustments(reflection_result.policy_adjustments)
                
                self.last_reflection_tick = self.current_tick
                
        except Exception as e:
            logger.error(f"Error in reflection processing: {e}")
    
    async def _update_planning_systems(self, current_time: datetime):
        """Update strategic and tactical planning systems."""
        if not self.strategic_planner or not self.tactical_planner:
            return
        
        try:
            # Update strategic plan periodically
            if self.current_tick % 168 == 0:  # Weekly strategic review (assuming 24 ticks/day)
                strategic_context = {
                    "current_metrics": self.performance_metrics,
                    "market_conditions": {
                        "competitive_pressure": 0.5,  # Placeholder
                        "volatility": 0.3  # Placeholder
                    }
                }
                
                await self.strategic_planner.update_strategic_plan(
                    self.performance_metrics,
                    self.major_events[-5:]  # Recent major events
                )
            
            # Generate tactical actions daily
            if self.current_tick % 24 == 0:  # Daily tactical planning
                current_state = {
                    "current_metrics": self.performance_metrics,
                    "inventory_level": self.inventory_level,
                    "customer_messages": self.customer_messages,
                    "available_budget": 1000.0  # Placeholder
                }
                
                strategic_objectives = self.strategic_planner.strategic_objectives
                if strategic_objectives:
                    await self.tactical_planner.generate_tactical_actions(
                        strategic_objectives,
                        current_state
                    )
                    
        except Exception as e:
            logger.error(f"Error in planning systems update: {e}")
    
    async def _process_major_events_for_cognition(self):
        """Process and categorize major events for cognitive systems."""
        # For now, consider any significant event as a major event
        if self.commands_accepted > 0 and self.current_tick % 10 == 0:
            major_event = {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "type": "decision_outcome",
                "severity": 0.5,
                "description": f"Command acceptance rate: {self.commands_accepted}/{self.commands_sent}",
                "impact_level": "medium" if self.commands_accepted / max(1, self.commands_sent) < 0.7 else "low"
            }
            self.major_events.append(major_event)
            
            # Keep only recent major events
            if len(self.major_events) > 50:
                self.major_events = self.major_events[-50:]
    
    async def _apply_policy_adjustments(self, policy_adjustments):
        """Apply policy adjustments from reflection results."""
        for adjustment in policy_adjustments:
            try:
                if adjustment.policy_area == "operations":
                    # Apply operational adjustments
                    if "decision_confidence_threshold" in adjustment.recommended_parameters:
                        # Adjust decision-making parameters
                        logger.info(f"Applying operational adjustment: {adjustment.rationale}")
                
                elif adjustment.policy_area == "risk_management":
                    # Apply risk management adjustments
                    logger.info(f"Applying risk management adjustment: {adjustment.rationale}")
                
                elif adjustment.policy_area == "pricing":
                    # Apply pricing strategy adjustments
                    if "sensitivity" in adjustment.recommended_parameters:
                        new_sensitivity = adjustment.recommended_parameters["sensitivity"]
                        self.config.price_sensitivity = new_sensitivity
                        logger.info(f"Updated price sensitivity to {new_sensitivity}")
                        
            except Exception as e:
                logger.error(f"Error applying policy adjustment {adjustment.adjustment_id}: {e}")
    
    async def _periodic_reflection(self):
        """Perform periodic reflection triggered every N simulation days."""
        if not self.config.cognitive_enabled or not self.reflection_loop:
            return
        
        # This method is called by _cognitive_processing when needed
        # The actual reflection is handled by _check_and_trigger_reflection
        pass
    
    async def handle_price_updated(self, event: ProductPriceUpdated):
        """
        Process ProductPriceUpdated events to maintain world model.
        
        This is how the agent learns about price changes in the market,
        including its own accepted commands and competitor actions.
        """
        if event.asin == self.config.target_asin:
            # Our product price was updated
            previous_price = self.current_price
            self.current_price = event.new_price
            self.price_history.append(event.new_price)
            
            # Track if this was our command that was accepted
            if event.agent_id == self.config.agent_id:
                self.commands_accepted += 1
                logger.info(f"Agent {self.config.agent_id} command accepted: price={event.new_price}")
            else:
                logger.info(f"Agent {self.config.agent_id} observed external price change: {previous_price} -> {event.new_price}")
        
        else:
            # Competitor price change - update market perception
            self.competitor_prices.append(event.new_price)
            if len(self.competitor_prices) > 10:  # Keep last 10 competitor prices
                self.competitor_prices.pop(0)

    async def handle_inventory_updated(self, event: BaseEvent): # Replace BaseEvent with actual InventoryLevelUpdated if defined
        """Process inventory level updates."""
        if event.asin == self.config.target_asin:
            self.inventory_level = event.new_level # Assuming event has new_level attr
            logger.info(f"Agent {self.config.agent_id} observed inventory update: {self.inventory_level}")
        await self._store_event_in_memory(event, "operations")

    async def handle_customer_message(self, event: BaseEvent): # Replace BaseEvent with actual CustomerMessageReceived if defined
        """Process customer messages."""
        self.customer_messages.append({"message_id": event.message_id, "content": event.content, "timestamp": event.timestamp})
        # Keep only recent messages
        if len(self.customer_messages) > 20:
            self.customer_messages.pop(0)
        logger.info(f"Agent {self.config.agent_id} received customer message: {event.message_id}")
        await self._store_event_in_memory(event, "customer_service")
    
    async def handle_marketing_status(self, event: BaseEvent): # Replace BaseEvent with actual MarketingCampaignStatus if defined
        """Process marketing campaign status updates."""
        self.campaign_status = {"campaign_id": event.campaign_id, "status": event.status, "spend": event.spend}
        logger.info(f"Agent {self.config.agent_id} observed marketing campaign status: {event.campaign_id} - {event.status}")
        await self._store_event_in_memory(event, "marketing")

    async def handle_sales_event(self, event: BaseEvent): # Replace BaseEvent with actual ProductSalesEvent if defined
        """Process sales events."""
        # Update sales data, e.g., for sales velocity calculation
        logger.info(f"Agent {self.config.agent_id} observed sales event: {event.quantity} units of {event.asin}")
        await self._store_event_in_memory(event, "sales")

    async def handle_financial_report(self, event: BaseEvent): # Replace BaseEvent with actual FinancialStatementReport if defined
        """Process financial reports."""
        self.financial_data = event.report_data # Assuming event has report_data attr
        logger.info(f"Agent {self.config.agent_id} received financial report.")
        await self._store_event_in_memory(event, "finance")
    
    async def handle_price_updated(self, event: ProductPriceUpdated):
        """
        Process ProductPriceUpdated events to maintain world model.
        
        This is how the agent learns about price changes in the market,
        including its own accepted commands and competitor actions.
        """
        if event.asin == self.config.target_asin:
            # Our product price was updated
            previous_price = self.current_price
            self.current_price = event.new_price
            self.price_history.append(event.new_price)
            
            # Track if this was our command that was accepted
            if event.agent_id == self.config.agent_id:
                self.commands_accepted += 1
                logger.info(f"Agent {self.config.agent_id} command accepted: price={event.new_price}")
            else:
                logger.info(f"Agent {self.config.agent_id} observed external price change: {previous_price} -> {event.new_price}")
        
        else:
            # Competitor price change - update market perception
            self.competitor_prices.append(event.new_price)
            if len(self.competitor_prices) > 10:  # Keep last 10 competitor prices
                self.competitor_prices.pop(0)
    
    # Decision Making
    
    def _should_act(self) -> bool:
        """Determine if the agent should take action this tick."""
        ticks_since_last_action = self.current_tick - self.last_action_tick
        return ticks_since_last_action >= self.config.reaction_speed
    
    async def _make_skill_based_decision(self):
        """
        Make skill-coordinated decisions using the multi-domain controller.
        This leverages specialized skills for different business domains.
        """
        try:
            # Build comprehensive business context
            business_context = self._build_business_context()
            
            # Create skill context for coordination
            skill_context = SkillContext(
                current_state=business_context,
                available_budget=business_context.get("available_budget", 1000.0),
                time_constraints={"urgency": "normal", "deadline": None},
                business_priorities=self._get_current_business_priorities()
            )
            
            # Use multi-domain controller to coordinate skill decisions
            coordinated_actions = await self.multi_domain_controller.coordinate_business_decisions(
                business_context,
                skill_context
            )
            
            # Execute coordinated actions
            for action in coordinated_actions:
                await self._execute_skill_action(action)
                
            self.last_action_tick = self.current_tick
            logger.info(f"Agent {self.config.agent_id} executed {len(coordinated_actions)} skill-coordinated actions")
            
        except Exception as e:
            logger.error(f"Error in skill-based decision making for agent {self.config.agent_id}: {e}")
            # Fallback to traditional decision making
            await self._make_multi_domain_decision()
    
    def _build_business_context(self) -> Dict[str, Any]:
        """Build comprehensive business context for skill coordination."""
        return {
            "agent_id": self.config.agent_id,
            "target_asin": self.config.target_asin,
            "current_tick": self.current_tick,
            "current_price": self.current_price.to_float() if self.current_price else None,
            "inventory_level": self.inventory_level,
            "competitor_prices": [p.to_float() for p in self.competitor_prices],
            "customer_messages": self.customer_messages,
            "campaign_status": self.campaign_status,
            "financial_data": self.financial_data,
            "performance_metrics": self.performance_metrics,
            "strategy": self.config.strategy,
            "available_budget": self._estimate_available_budget(),
            "market_conditions": self._assess_market_conditions(),
            "business_health": self._assess_business_health()
        }
    
    def _get_current_business_priorities(self) -> Dict[str, float]:
        """Determine current business priorities based on performance and conditions."""
        priorities = {
            "survival": 0.1,
            "stabilization": 0.3,
            "growth": 0.4,
            "optimization": 0.2,
            "innovation": 0.0
        }
        
        # Adjust based on performance metrics
        profit_margin = self.performance_metrics.get("profit_margin", 0.1)
        market_share = self.performance_metrics.get("market_share", 0.05)
        decision_success_rate = self.performance_metrics.get("decision_success_rate", 0.5)
        
        if profit_margin < 0.05:  # Low profit margins - focus on survival
            priorities.update({"survival": 0.6, "stabilization": 0.3, "growth": 0.1})
        elif profit_margin > 0.2 and market_share > 0.1:  # Strong performance - focus on growth
            priorities.update({"growth": 0.5, "optimization": 0.3, "innovation": 0.2})
        elif decision_success_rate < 0.3:  # Poor decisions - focus on stabilization
            priorities.update({"stabilization": 0.5, "survival": 0.3, "optimization": 0.2})
        
        return priorities
    
    def _estimate_available_budget(self) -> float:
        """Estimate available budget for actions."""
        # Simplified budget estimation based on performance
        base_budget = 1000.0
        profit_margin = self.performance_metrics.get("profit_margin", 0.1)
        decision_success_rate = self.performance_metrics.get("decision_success_rate", 0.5)
        
        budget_multiplier = (profit_margin * 5) + decision_success_rate
        return base_budget * max(0.1, min(2.0, budget_multiplier))
    
    def _assess_market_conditions(self) -> Dict[str, Any]:
        """Assess current market conditions."""
        competitive_pressure = 0.5  # Default moderate pressure
        
        if self.competitor_prices and self.current_price:
            avg_competitor_price = sum(p.to_float() for p in self.competitor_prices) / len(self.competitor_prices)
            current_price = self.current_price.to_float()
            
            if current_price > avg_competitor_price * 1.2:
                competitive_pressure = 0.8  # High pressure if significantly above market
            elif current_price < avg_competitor_price * 0.8:
                competitive_pressure = 0.2  # Low pressure if significantly below market
        
        return {
            "competitive_pressure": competitive_pressure,
            "volatility": 0.3,  # Placeholder
            "growth_potential": 0.6,  # Placeholder
            "regulatory_risk": 0.1   # Placeholder
        }
    
    def _assess_business_health(self) -> Dict[str, Any]:
        """Assess overall business health metrics."""
        return {
            "financial_stability": min(1.0, self.performance_metrics.get("profit_margin", 0.1) * 10),
            "operational_efficiency": self.performance_metrics.get("decision_success_rate", 0.5),
            "customer_satisfaction": 0.7,  # Placeholder - could be calculated from customer messages
            "market_position": self.performance_metrics.get("market_share", 0.05) * 20
        }
    
    async def _execute_skill_action(self, action: Dict[str, Any]):
        """Execute an action generated by the skill coordination system."""
        try:
            # Extract action details
            action_type = action.get("type")
            parameters = action.get("parameters", {})
            skill_id = action.get("skill_id")
            confidence = action.get("confidence", 0.8)
            reasoning = action.get("reasoning", f"Skill-coordinated action from {skill_id}")
            
            # Execute using the existing action execution framework
            await self._execute_action({
                "type": action_type,
                "parameters": parameters,
                "confidence": confidence,
                "reasoning": reasoning,
                "skill_source": skill_id
            })
            
            logger.debug(f"Executed skill action {action_type} from {skill_id} with confidence {confidence}")
            
        except Exception as e:
            logger.error(f"Error executing skill action: {e}")

    async def _make_multi_domain_decision(self):
        """
        Make decisions across multiple business domains (pricing, inventory, marketing, customer service).
        This method will coordinate the agent's strategy based on current state and goals.
        """
        if self.gateway:
            # 1. Observe and build comprehensive input context
            llm_input_context = self._get_llm_input_context()

            try:
                with self.agent_tracer.trace_observe_phase(current_tick=self.current_tick, event_count=0): # event_count can be dynamic
                    # Use the gateway to preprocess the "request" (context for decision-making)
                    processed_request = await self.gateway.preprocess_request(
                        agent_id=self.config.agent_id,
                        prompt=llm_input_context,
                        action_type="multi_domain_decision",
                        model_name="gpt-4" # Assuming a default model name
                    )
                    modified_input_context = processed_request["modified_prompt"]

                # 2. Simulate LLM's "thinking" process (or actual LLM call)
                llm_tokens_used = 200 # Placeholder for actual LLM token usage
                decision_confidence = 0.90 # Placeholder
                
                # --- Deterministic LLM Call ---
                # Call the deterministic LLM client with proper reproducibility support
                llm_response = await self._call_llm_deterministic(
                    prompt=modified_input_context,
                    temperature=0.0,  # Deterministic temperature for reproducibility
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                # Extract content from LLM response
                llm_content = llm_response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                parsed_llm_response = json.loads(llm_content)
                # --- End Deterministic LLM Call ---

                with self.agent_tracer.trace_think_phase(
                    current_tick=self.current_tick,
                    llm_model="gpt-4", # Placeholder
                    llm_tokens_used=llm_tokens_used,
                    decision_confidence=decision_confidence,
                    parsed_response=parsed_llm_response
                ):
                    # Decisions are already "made" by the simulated LLM output, now validate and publish them
                    pass # The logic below will process `parsed_llm_response`


                # 3. Postprocess and execute actions
                await self.gateway.postprocess_response(
                    agent_id=self.config.agent_id,
                    action_type="multi_domain_decision",
                    raw_prompt=llm_input_context, # Original prompt context
                    llm_response=llm_response, # Use actual LLM response not simulated output
                    model_name="gpt-4"
                )

                # Execute parsed actions
                for action in parsed_llm_response.get("actions", []):
                    await self._execute_action(action)
                
                self.last_action_tick = self.current_tick # Regardless of action, mark tick processed

            except SystemExit as e:
                logger.error(f"Agent {self.config.agent_id} budget hard-fail during multi-domain decision: {e}")
                raise # Re-raise to terminate simulation if hard-fail is triggered
            except json.JSONDecodeError as e:
                logger.error(f"Agent {self.config.agent_id} LLM response was not valid JSON: {llm_content}. Error: {e}")
            except Exception as e:
                logger.error(f"Agent {self.config.agent_id} multi-domain decision error: {e}", exc_info=True)
        else: # No gateway means no budget enforcement or centralized control
            try:
                with self.agent_tracer.trace_observe_phase(current_tick=self.current_tick, event_count=0):
                    llm_input_context = self._get_llm_input_context()
                
                llm_tokens_used = 100 # Placeholder
                decision_confidence = 0.8 # Placeholder
                
                # Call deterministic LLM without gateway constraints
                llm_response = await self._call_llm_deterministic(
                    prompt=llm_input_context,
                    temperature=0.0,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                llm_content = llm_response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                parsed_llm_response = json.loads(llm_content)

                with self.agent_tracer.trace_think_phase(
                    current_tick=self.current_tick,
                    llm_model="script_bot",
                    llm_tokens_used=llm_tokens_used,
                    decision_confidence=decision_confidence,
                    parsed_response=parsed_llm_response
                ):
                    pass # Decisions are simulated here

                for action in parsed_llm_response.get("actions", []):
                    await self._execute_action(action)
                
                self.last_action_tick = self.current_tick

            except json.JSONDecodeError as e:
                logger.error(f"Agent {self.config.agent_id} LLM response was not valid JSON without gateway: {llm_content}. Error: {e}")
            except Exception as e:
                logger.error(f"Agent {self.config.agent_id} decision error without gateway: {e}", exc_info=True)

    async def _execute_action(self, action: Dict[str, Any]):
        """Executes a single action with cognitive validation and learning."""
        action_type = action.get("type")
        parameters = action.get("parameters", {})
        confidence = action.get("confidence", 0.8)
        reasoning = action.get("reasoning", "Agent decision")

        # Enhanced pre-action validation using memory integration gateway
        if self.config.cognitive_enabled and self.memory_integration_gateway:
            try:
                should_proceed, validation_result = await self.memory_integration_gateway.pre_action_validation(action)
                
                if not should_proceed:
                    logger.warning(f"Action {action_type} blocked by validation: {validation_result.recommendations}")
                    return
                    
                if validation_result.confidence_score < 0.6:
                    logger.warning(f"Action {action_type} has low validation confidence: {validation_result.confidence_score:.2f}")
                    
            except Exception as e:
                logger.error(f"Error in pre-action validation: {e}")
                # Continue with action if validation fails gracefully

        # Validate action alignment with strategic objectives
        if self.config.cognitive_enabled and self.strategic_planner:
            try:
                is_aligned, alignment_score, alignment_reasoning = await self.strategic_planner.validate_action_alignment(action)
                
                if not is_aligned:
                    logger.warning(f"Action {action_type} not aligned with strategy: {alignment_reasoning}")
                    # Consider whether to proceed or not based on configuration
                    if alignment_score < 0.3:
                        logger.warning(f"Action {action_type} has very low strategic alignment, skipping")
                        return
                        
            except Exception as e:
                logger.error(f"Error in strategic alignment validation: {e}")

        # Trace the individual tool call
        with self.agent_tracer.trace_tool_call(
            tool_name=action_type,
            current_tick=self.current_tick,
            tool_args=parameters,
            result="initiated" # Result will be confirmed by response event
        ):
            try:
                action_outcome = {"success": False, "impact": {}}
                
                if action_type == "set_price":
                    new_price = Money(int(parameters["price"] * 100)) # Convert float price to cents
                    if self._should_change_price(new_price):
                        await self._publish_price_command(new_price, reasoning)
                        action_outcome = {"success": True, "impact": {"price_change": new_price.to_float()}}

                elif action_type == "place_order":
                    await self._publish_order_command(
                        parameters.get("supplier_id"),
                        parameters.get("asin", self.config.target_asin),
                        parameters.get("quantity"),
                        parameters.get("max_price", 0), # Optional, can be Money obj
                        reasoning
                    )
                    action_outcome = {"success": True, "impact": {"inventory_ordered": parameters.get("quantity", 0)}}
                    
                elif action_type == "respond_to_customer":
                    await self._publish_customer_response_command(
                        parameters.get("message_id"),
                        parameters.get("response_content"),
                        reasoning
                    )
                    action_outcome = {"success": True, "impact": {"customer_satisfaction": 0.1}}
                    
                elif action_type == "run_marketing_campaign":
                    await self._publish_marketing_command(
                        parameters.get("campaign_type"),
                        parameters.get("budget"),
                        parameters.get("duration_days"),
                        reasoning
                    )
                    action_outcome = {"success": True, "impact": {"marketing_spend": parameters.get("budget", 0)}}
                    
                else:
                    logger.warning(f"Agent {self.config.agent_id} attempted unsupported action type: {action_type}")
                    action_outcome = {"success": False, "impact": {}}

                # Post-action learning
                if self.config.cognitive_enabled and self.memory_integration_gateway:
                    try:
                        await self.memory_integration_gateway.post_action_learning(action, action_outcome)
                    except Exception as e:
                        logger.error(f"Error in post-action learning: {e}")

                # Update tactical planner if action was completed
                if action_outcome["success"] and self.tactical_planner:
                    try:
                        await self.tactical_planner.mark_action_completed(
                            action.get("action_id", str(uuid.uuid4())),
                            action_outcome
                        )
                    except Exception as e:
                        logger.error(f"Error updating tactical planner: {e}")
                        
            except KeyError as e:
                logger.error(f"Missing parameter for action {action_type}: {e}")
                # Mark action as failed in tactical planner
                if self.tactical_planner:
                    try:
                        await self.tactical_planner.mark_action_failed(
                            action.get("action_id", str(uuid.uuid4())),
                            f"Missing parameter: {e}"
                        )
                    except Exception as e:
                        logger.error(f"Error marking action as failed: {e}")
                        
            except Exception as e:
                logger.error(f"Error executing action {action_type} for agent {self.config.agent_id}: {e}")
                # Mark action as failed in tactical planner
                if self.tactical_planner:
                    try:
                        await self.tactical_planner.mark_action_failed(
                            action.get("action_id", str(uuid.uuid4())),
                            str(e)
                        )
                    except Exception as e:
                        logger.error(f"Error marking action as failed: {e}")

    def _simulate_llm_multi_action_output(self) -> str:
        """
        Simulates an LLM output with multiple action types for advanced agent.
        Enhanced with tactical planning integration.
        """
        actions = []
        overall_reasoning = "Multi-domain strategy execution based on current business needs."

        # Get ready tactical actions if tactical planner is available
        tactical_actions = []
        if self.config.cognitive_enabled and self.tactical_planner:
            try:
                tactical_actions = self.tactical_planner.get_ready_actions(datetime.now())
                if tactical_actions:
                    overall_reasoning += " Incorporating tactical planning recommendations."
            except Exception as e:
                logger.error(f"Error getting tactical actions: {e}")

        # Priority 1: Execute ready tactical actions
        for tactical_action in tactical_actions[:2]:  # Limit to 2 tactical actions per tick
            action_dict = {
                "action_id": tactical_action.action_id,
                "type": tactical_action.action_type,
                "parameters": tactical_action.parameters.copy(),
                "reasoning": f"Tactical plan: {tactical_action.description}",
                "confidence": 0.9,
                "strategic_objective_id": tactical_action.strategic_objective_id
            }
            actions.append(action_dict)

        # Priority 2: Standard reactive actions if no tactical actions or as supplementary
        if len(actions) < 2:  # Allow some reactive actions if tactical planner isn't providing enough
            
            # Pricing action with enhanced logic
            desired_price = self._calculate_desired_price()
            if desired_price and self._should_change_price(desired_price):
                # Enhanced reasoning with strategic context
                price_reasoning = f"Adjusting price based on {self.config.strategy} strategy"
                if self.strategic_planner and hasattr(self.strategic_planner, 'strategic_objectives'):
                    try:
                        # Simple heuristic alignment check without async call
                        active_objectives = [obj for obj in self.strategic_planner.strategic_objectives.values()
                                           if obj.status.value == "active"]
                        if active_objectives:
                            price_reasoning += f" (Supporting {len(active_objectives)} strategic objectives)"
                    except Exception as e:
                        logger.error(f"Error checking strategic context: {e}")
                
                actions.append({
                    "type": "set_price",
                    "parameters": {"asin": self.config.target_asin, "price": desired_price.to_float()},
                    "reasoning": price_reasoning,
                    "confidence": 0.95
                })

            # Inventory action with performance-based logic
            inventory_threshold = 20
            if self.performance_metrics.get("inventory_turnover", 5) > 8:
                inventory_threshold = 30  # Higher threshold for fast-moving inventory
            
            if self.inventory_level < inventory_threshold:
                order_quantity = 100
                # Adjust order quantity based on recent sales performance
                if self.performance_metrics.get("decision_success_rate", 0.5) > 0.8:
                    order_quantity = 150  # Order more if performing well
                    
                actions.append({
                    "type": "place_order",
                    "parameters": {"supplier_id": "supplier_A", "asin": self.config.target_asin, "quantity": order_quantity, "max_price": 10.00},
                    "reasoning": f"Low inventory ({self.inventory_level} < {inventory_threshold}), placing restock order.",
                    "confidence": 0.9
                })
            
            # Customer service action with prioritization
            if self.customer_messages:
                # Prioritize messages based on age and content
                urgent_messages = [msg for msg in self.customer_messages if "urgent" in msg.get("content", "").lower()]
                latest_message = urgent_messages[0] if urgent_messages else self.customer_messages[-1]
                
                response_content = "Thank you for your inquiry. We are looking into this for you."
                if urgent_messages:
                    response_content = "Thank you for your urgent inquiry. We are prioritizing your request and will respond within 24 hours."
                
                actions.append({
                    "type": "respond_to_customer",
                    "parameters": {"message_id": latest_message["message_id"], "response_content": response_content},
                    "reasoning": "Responding to customer query with appropriate priority.",
                    "confidence": 0.8
                })

            # Marketing action with performance-based timing
            marketing_interval = 100  # Default
            if self.performance_metrics.get("profit_margin", 0.1) > 0.2:
                marketing_interval = 50  # More frequent marketing if profitable
            elif self.performance_metrics.get("profit_margin", 0.1) < 0.05:
                marketing_interval = 200  # Less frequent if low margins
                
            if self.current_tick % marketing_interval == 0:
                # Adjust budget based on performance
                base_budget = 500.0
                performance_multiplier = self.performance_metrics.get("decision_success_rate", 0.5)
                adjusted_budget = base_budget * (0.5 + performance_multiplier)
                
                actions.append({
                    "type": "run_marketing_campaign",
                    "parameters": {"campaign_type": "social_media_ads", "budget": adjusted_budget, "duration_days": 7},
                    "reasoning": f"Initiating marketing campaign with performance-adjusted budget (${adjusted_budget:.0f}).",
                    "confidence": 0.75
                })

        return json.dumps({"actions": actions, "overall_reasoning": overall_reasoning})
    
    def _get_llm_input_context(self) -> str:
        """Generates a text summary of current state for conceptual LLM input."""
        with trace.get_current_span().start_as_current_span("build_observation_context"):
            context = (
                f"Current Tick: {self.current_tick}\n"
                f"Your Product ASIN: {self.config.target_asin}\n"
                f"Your Current Price: {self.current_price or 'N/A'}\n"
                f"Market Data: {self.market_data}\n"
                f"Competitor Prices: {[str(p) for p in self.competitor_prices]}\n"
                f"Current Strategy: {self.config.strategy}\n"
                f"Your Inventory Level: {self.inventory_level}\n"
                f"Your Customer Messages (last {len(self.customer_messages)}): {self.customer_messages}\n"
                f"Your Marketing Campaign Status: {self.campaign_status}\n"
                f"Your Financial Data: {self.financial_data}\n"
                f"Your Goal: Manage all aspects of your FBA business for {self.config.target_asin} to maximize profit/achieve strategic objectives, considering all available information."
            )
            return context
    
    # Helper for storing events in memory if memory_enforcer is available
    async def _store_event_in_memory(self, event: BaseEvent, domain: str):
        if self.gateway and \
           hasattr(self.gateway, 'memory_enforcer') and \
           self.gateway.memory_enforcer:
            await self.gateway.memory_enforcer.store_event_in_memory(event, domain)
            
    def _calculate_desired_price(self) -> Optional[Money]:
        """
        Calculate desired price based on agent's strategy.
        
        This demonstrates different agent strategies and how they
        can lead to diverse market behaviors.
        """
        with trace.get_current_span().start_as_current_span("calculate_price_strategy"):
            if self.config.strategy == "profit_maximizer":
                return self._profit_maximizer_strategy()
            elif self.config.strategy == "market_follower":
                return self._market_follower_strategy()
            elif self.config.strategy == "aggressive_pricer":
                return self._aggressive_pricer_strategy()
            else:
                return self._random_strategy()
    
    def _profit_maximizer_strategy(self) -> Optional[Money]:
        """Strategy that tries to maximize profit through gradual price increases."""
        with trace.get_current_span().start_as_current_span("profit_maximizer_strategy"):
            if not self.current_price:
                return Money(2000)  # Start at $20.00
            
            # Gradually increase price to test market elasticity
            increase_factor = 1.0 + (self.config.price_sensitivity * 0.5)
            new_price = Money(int(self.current_price.cents * increase_factor))
            
            # Bound within agent's constraints
            return min(max(new_price, self.config.min_price), self.config.max_price)
    
    def _market_follower_strategy(self) -> Optional[Money]:
        """Strategy that follows competitor pricing with slight undercut."""
        with trace.get_current_span().start_as_current_span("market_follower_strategy"):
            if not self.competitor_prices:
                return Money(2000)  # Default price
            
            # Calculate average competitor price
            avg_competitor_price = Money(sum(p.cents for p in self.competitor_prices) // len(self.competitor_prices))
            
            # Undercut by small amount
            undercut_factor = 1.0 - (self.config.price_sensitivity * 0.2)
            new_price = Money(int(avg_competitor_price.cents * undercut_factor))
            
            return min(max(new_price, self.config.min_price), self.config.max_price)
    
    def _aggressive_pricer_strategy(self) -> Optional[Money]:
        """Strategy that aggressively undercuts competition."""
        with trace.get_current_span().start_as_current_span("aggressive_pricer_strategy"):
            if not self.competitor_prices:
                return Money(1800)  # Start low at $18.00
            
            # Find minimum competitor price and undercut significantly
            min_competitor_price = min(self.competitor_prices)
            undercut_factor = 1.0 - (self.config.price_sensitivity * 0.3)
            new_price = Money(int(min_competitor_price.cents * undercut_factor))
            
            return min(max(new_price, self.config.min_price), self.config.max_price)
    
    def _random_strategy(self) -> Optional[Money]:
        """Random pricing strategy for chaos testing."""
        with trace.get_current_span().start_as_current_span("random_strategy"):
            price_range = self.config.max_price.cents - self.config.min_price.cents
            random_cents = self.config.min_price.cents + random.randint(0, price_range)
            return Money(random_cents)
    
    def _should_change_price(self, desired_price: Money) -> bool:
        """Determine if the price change is significant enough to warrant a command."""
        with trace.get_current_span().start_as_current_span("should_change_price_check"):
            if not self.current_price:
                return True  # Always set initial price
            
            # Only change if difference is meaningful (e.g., more than 1%)
            if self.current_price.cents == 0: # Avoid division by zero
                return True if desired_price.cents != 0 else False

            price_diff_ratio = abs(desired_price.cents - self.current_price.cents) / self.current_price.cents
            return price_diff_ratio >= 0.01  # 1% minimum change
    
    async def _publish_price_command(self, new_price: Money, reason: str = "Pricing adjustment"):
        """Publish SetPriceCommand to express pricing intention."""
        with trace.get_current_span().start_as_current_span("publish_price_command"):
            command = SetPriceCommand(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                agent_id=self.config.agent_id,
                asin=self.config.target_asin,
                new_price=new_price,
                reason=reason
            )
            await self.event_bus.publish(command)
            self.commands_sent += 1
            self.decision_history.append({
                'tick': self.current_tick, 'command_id': command.event_id,
                'type': 'set_price', 'value': str(new_price), 'reason': reason
            })
            logger.info(f"Agent {self.config.agent_id} published SetPriceCommand: price={new_price}, reason={reason}")

    async def _publish_order_command(self, supplier_id: str, asin: str, quantity: int, max_price: float, reason: str = "Restocking inventory"):
        """Publish PlaceOrderCommand for inventory."""
        with trace.get_current_span().start_as_current_span("publish_order_command"):
            command = PlaceOrderCommand(
                event_id=str(uuid.uuid4()), timestamp=datetime.now(), agent_id=self.config.agent_id,
                supplier_id=supplier_id, asin=asin, quantity=quantity, max_price=Money(int(max_price * 100)),
                reason=reason
            )
            await self.event_bus.publish(command)
            self.commands_sent += 1
            self.decision_history.append({
                'tick': self.current_tick, 'command_id': command.event_id,
                'type': 'place_order', 'quantity': quantity, 'asin': asin, 'reason': reason
            })
            logger.info(f"Agent {self.config.agent_id} published PlaceOrderCommand: {quantity} units of {asin} from {supplier_id}")

    async def _publish_customer_response_command(self, message_id: str, response_content: str, reason: str = "Responding to customer message"):
        """Publish RespondToCustomerMessageCommand."""
        with trace.get_current_span().start_as_current_span("publish_customer_response_command"):
            command = RespondToCustomerMessageCommand(
                event_id=str(uuid.uuid4()), timestamp=datetime.now(), agent_id=self.config.agent_id,
                message_id=message_id, response_content=response_content, reason=reason
            )
            await self.event_bus.publish(command)
            self.commands_sent += 1
            self.decision_history.append({
                'tick': self.current_tick, 'command_id': command.event_id,
                'type': 'respond_to_customer', 'message_id': message_id, 'reason': reason
            })
            logger.info(f"Agent {self.config.agent_id} published RespondToCustomerMessageCommand for message {message_id}")

    async def _publish_marketing_command(self, campaign_type: str, budget: float, duration_days: int, reason: str = "Launching marketing campaign"):
        """Publish RunMarketingCampaignCommand."""
        with trace.get_current_span().start_as_current_span("publish_marketing_command"):
            command = RunMarketingCampaignCommand(
                event_id=str(uuid.uuid4()), timestamp=datetime.now(), agent_id=self.config.agent_id,
                campaign_type=campaign_type, budget=Money(int(budget * 100)), duration_days=duration_days, reason=reason
            )
            await self.event_bus.publish(command)
            self.commands_sent += 1
            self.decision_history.append({
                'tick': self.current_tick, 'command_id': command.event_id,
                'type': 'run_marketing_campaign', 'campaign_type': campaign_type, 'budget': budget, 'reason': reason
            })
            logger.info(f"Agent {self.config.agent_id} published RunMarketingCampaignCommand for {campaign_type} with budget {budget}")
    
    # Status and Analytics
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status including cognitive capabilities."""
        status = {
            'agent_id': self.config.agent_id,
            'target_asin': self.config.target_asin,
            'strategy': self.config.strategy,
            'current_tick': self.current_tick,
            'current_price': str(self.current_price) if self.current_price else None,
            'inventory_level': self.inventory_level,
            'commands_sent': self.commands_sent,
            'commands_accepted': self.commands_accepted,
            'acceptance_rate': self.commands_accepted / max(1, self.commands_sent),
            'competitor_prices_observed': len(self.competitor_prices),
            'decisions_made': len(self.decision_history),
            'last_customer_message_count': len(self.customer_messages),
            'marketing_campaign_active': 'campaign_id' in self.campaign_status,
            'cognitive_enabled': self.config.cognitive_enabled,
            'multi_skill_enabled': self.config.enable_multi_skill_system
        }
        
        # Add multi-skill system status
        if self.config.enable_multi_skill_system:
            status['skill_systems'] = {}
            
            # Skill coordinator status
            if self.skill_coordinator:
                status['skill_systems']['coordinator'] = {
                    'active_skills': len(self.skills),
                    'registered_skills': list(self.skills.keys()),
                    'coordination_strategy': getattr(self.skill_coordinator, 'coordination_strategy', 'priority_based')
                }
            
            # Multi-domain controller status
            if self.multi_domain_controller:
                status['skill_systems']['controller'] = {
                    'operational_mode': getattr(self.multi_domain_controller, 'operational_mode', 'balanced'),
                    'business_priorities': self._get_current_business_priorities()
                }
            
            # Individual skill status
            status['skill_systems']['skills'] = {}
            for skill_name, skill in self.skills.items():
                if hasattr(skill, 'get_skill_status'):
                    status['skill_systems']['skills'][skill_name] = skill.get_skill_status()
                else:
                    status['skill_systems']['skills'][skill_name] = {
                        'enabled': True,
                        'skill_type': type(skill).__name__
                    }
        
        # Add cognitive system status
        if self.config.cognitive_enabled:
            status['cognitive_systems'] = {}
            
            # Performance metrics
            status['performance_metrics'] = self.performance_metrics.copy()
            
            # Strategic planning status
            if self.strategic_planner:
                status['cognitive_systems']['strategic_planning'] = self.strategic_planner.get_strategic_status()
            
            # Tactical planning status
            if self.tactical_planner:
                status['cognitive_systems']['tactical_planning'] = self.tactical_planner.get_tactical_status()
            
            # Reflection system status
            if self.reflection_loop:
                status['cognitive_systems']['reflection'] = self.reflection_loop.get_reflection_status()
            
            # Memory validation status
            if self.memory_integration_gateway:
                status['cognitive_systems']['memory_validation'] = self.memory_integration_gateway.get_gateway_status()
            
            # Cognitive configuration
            if self.cognitive_config:
                status['cognitive_config'] = {
                    'mode': self.cognitive_config.cognitive_mode.value,
                    'template': self.config.cognitive_config_template,
                    'reflection_interval': self.cognitive_config.reflection.reflection_interval,
                    'strategic_horizon': self.cognitive_config.strategic_planning.strategic_planning_horizon,
                    'validation_enabled': self.cognitive_config.memory_validation.memory_validation_enabled
                }
            
            # Recent cognitive milestones
            status['cognitive_milestones'] = {
                'last_reflection_tick': self.last_reflection_tick,
                'major_events_count': len(self.major_events),
                'ticks_since_reflection': self.current_tick - self.last_reflection_tick if self.last_reflection_tick > 0 else 0
            }
        
        return status
    
    async def create_strategic_plan(self, context: Dict[str, Any], timeframe: int = 90) -> bool:
        """Create or update strategic plan based on current context."""
        if not self.config.cognitive_enabled or not self.strategic_planner:
            logger.warning(f"Strategic planning not available for agent {self.config.agent_id}")
            return False
        
        try:
            objectives = await self.strategic_planner.create_strategic_plan(context, timeframe)
            
            # Publish strategic plan created event
            await self._publish_cognitive_milestone_event("StrategicPlanCreated", {
                "objectives_count": len(objectives),
                "timeframe_days": timeframe,
                "context_summary": str(context)[:200]
            })
            
            logger.info(f"Strategic plan created for agent {self.config.agent_id} with {len(objectives)} objectives")
            return True
            
        except Exception as e:
            logger.error(f"Error creating strategic plan: {e}")
            return False
    
    async def trigger_cognitive_reflection(self, force: bool = False) -> bool:
        """Manually trigger cognitive reflection."""
        if not self.config.cognitive_enabled or not self.reflection_loop:
            logger.warning(f"Cognitive reflection not available for agent {self.config.agent_id}")
            return False
        
        try:
            # Force reflection by providing major events
            major_events = self.major_events[-10:] if not force else self.major_events
            
            reflection_result = await self.reflection_loop.trigger_reflection(
                tick_interval=1,  # Force immediate reflection
                major_events=major_events
            )
            
            if reflection_result:
                # Apply policy adjustments
                await self._apply_policy_adjustments(reflection_result.policy_adjustments)
                
                # Publish reflection completed event
                await self._publish_cognitive_milestone_event("ReflectionCompleted", {
                    "insights_count": len(reflection_result.insights),
                    "critical_insights": reflection_result.critical_insights_count,
                    "policy_adjustments": len(reflection_result.policy_adjustments),
                    "analysis_depth": reflection_result.analysis_depth_score
                })
                
                self.last_reflection_tick = self.current_tick
                logger.info(f"Cognitive reflection completed for agent {self.config.agent_id}")
                return True
            else:
                logger.info(f"Cognitive reflection was not triggered for agent {self.config.agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error in cognitive reflection: {e}")
            return False
    
    async def validate_memory_consistency(self) -> Dict[str, Any]:
        """Perform comprehensive memory consistency validation."""
        if not self.config.cognitive_enabled or not self.memory_consistency_checker:
            return {"validation_available": False}
        
        try:
            # Get recent memories for validation
            if self.gateway and \
               hasattr(self.gateway, 'memory_enforcer') and \
               self.gateway.memory_enforcer:
                recent_memories = await self.gateway.memory_enforcer.memory_manager.retrieve_memories(
                    query="recent activities", max_memories=20
                )
                
                # Create a dummy action for validation
                dummy_action = {
                    "type": "validation_check",
                    "parameters": {},
                    "expected_impact": {}
                }
                
                validation_result = await self.memory_consistency_checker.validate_memory_retrieval(
                    recent_memories, dummy_action
                )
                
                # Publish memory inconsistency detected event if needed
                if not validation_result.validation_passed:
                    await self._publish_cognitive_milestone_event("MemoryInconsistencyDetected", {
                        "inconsistencies_count": len(validation_result.inconsistencies_found),
                        "confidence_score": validation_result.confidence_score,
                        "memories_checked": validation_result.memories_checked
                    })
                
                return {
                    "validation_available": True,
                    "validation_passed": validation_result.validation_passed,
                    "inconsistencies_found": len(validation_result.inconsistencies_found),
                    "confidence_score": validation_result.confidence_score,
                    "recommendations": validation_result.recommendations
                }
            else:
                return {"validation_available": False, "reason": "Memory enforcer not available"}
                
        except Exception as e:
            logger.error(f"Error in memory consistency validation: {e}")
            return {"validation_available": False, "error": str(e)}
    
    async def _publish_cognitive_milestone_event(self, event_type: str, event_data: Dict[str, Any]):
        """Publish cognitive milestone events."""
        try:
            milestone_event = BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now()
            )
            
            # Add event-specific data
            milestone_event.event_type = event_type
            milestone_event.agent_id = self.config.agent_id
            milestone_event.tick = self.current_tick
            milestone_event.data = event_data
            
            await self.event_bus.publish(milestone_event)
            logger.debug(f"Published cognitive milestone event: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to publish cognitive milestone event {event_type}: {e}")
    
    def configure_cognitive_system(self, new_config: Dict[str, Any]):
        """Update cognitive system configuration at runtime."""
        if not self.config.cognitive_enabled:
            logger.warning("Cannot configure cognitive system - not enabled")
            return
        
        try:
            # Update cognitive configuration
            if self.cognitive_config:
                for key, value in new_config.items():
                    if hasattr(self.cognitive_config, key):
                        setattr(self.cognitive_config, key, value)
                        logger.info(f"Updated cognitive config {key} = {value}")
            
            # Update specific component configurations
            if 'reflection_interval' in new_config and self.reflection_loop:
                self.reflection_loop.periodic_interval_hours = new_config['reflection_interval'] * 24
            
            if 'validation_mode' in new_config and self.memory_integration_gateway:
                self.memory_integration_gateway.configure_gateway(
                    validation_enabled=new_config.get('validation_enabled', True),
                    blocking_mode=new_config.get('blocking_mode', False)
                )
            
            logger.info(f"Cognitive system configuration updated for agent {self.config.agent_id}")
            
        except Exception as e:
            logger.error(f"Error updating cognitive configuration: {e}")
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get agent's decision history for analysis."""
        return self.decision_history.copy()


# Factory functions for common agent types

def create_profit_maximizer_agent(agent_id: str, target_asin: str, llm_batcher: Optional[LLMBatcher] = None, resource_manager: Optional[ResourceManager] = None) -> AdvancedAgent:
    """Create a profit-maximizing agent."""
    config = AgentConfig(
        agent_id=agent_id,
        target_asin=target_asin,
        strategy="profit_maximizer",
        price_sensitivity=0.15,
        reaction_speed=2
    )
    return AdvancedAgent(config, llm_batcher=llm_batcher, resource_manager=resource_manager)


def create_market_follower_agent(agent_id: str, target_asin: str, llm_batcher: Optional[LLMBatcher] = None, resource_manager: Optional[ResourceManager] = None) -> AdvancedAgent:
    """Create a market-following agent."""
    config = AgentConfig(
        agent_id=agent_id,
        target_asin=target_asin,
        strategy="market_follower",
        price_sensitivity=0.1,
        reaction_speed=3
    )
    return AdvancedAgent(config, llm_batcher=llm_batcher, resource_manager=resource_manager)


def create_aggressive_agent(agent_id: str, target_asin: str, llm_batcher: Optional[LLMBatcher] = None, resource_manager: Optional[ResourceManager] = None) -> AdvancedAgent:
    """Create an aggressive pricing agent."""
    config = AgentConfig(
        agent_id=agent_id,
        target_asin=target_asin,
        strategy="aggressive_pricer",
        price_sensitivity=0.25,
        reaction_speed=1
    )
    return AdvancedAgent(config, llm_batcher=llm_batcher, resource_manager=resource_manager)