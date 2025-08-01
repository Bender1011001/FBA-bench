#!/usr/bin/env python3
"""
FBA-Bench v3 Experiment CLI - High-Throughput Research Platform

Command-line tool for defining, running, and aggregating large-scale 
automated simulation experiments with parameter sweeps.

Usage:
    python experiment_cli.py run sweep.yaml
    python experiment_cli.py run sweep.yaml --parallel 4
    python experiment_cli.py analyze results/price_strategy_test_20250728-123456/
"""

import argparse
import asyncio
import json
import os
import sys
import time
import hashlib
import itertools
import multiprocessing
import logging
import zipfile
import numpy as np # Added for RL agent
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple, Union
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
from money import Money

# Import FBA-Bench components
try:
    from simulation_orchestrator import SimulationOrchestrator, SimulationConfig
    from event_bus import EventBus, DistributedBackend, AsyncioQueueBackend # Import specific backends
    from services.competitor_manager import CompetitorManager
    from services.dashboard_api_service import DashboardAPIService
    from personas import IrrationalSlasher, SlowFollower, CompetitorPersona
    from events import TickEvent, SaleOccurred, CompetitorPricesUpdated
    
    # Import reproducibility components
    from reproducibility.reproducibility_config import (
        ReproducibilityConfig, create_deterministic_config, create_research_config,
        get_global_config, set_global_config, load_config_from_env
    )
    from reproducibility.simulation_modes import (
        SimulationMode, get_mode_controller, set_global_mode
    )
    from reproducibility.sim_seed import SimSeed
    from reproducibility.golden_master import GoldenMasterTester, ToleranceConfig
    from reproducibility.event_snapshots import EventSnapshot
    from reproducibility.llm_cache import LLMResponseCache
    
    # Import new infrastructure components
    from infrastructure.llm_batcher import LLMBatcher
    from infrastructure.distributed_event_bus import DistributedEventBus, MockRedisBroker
    from infrastructure.resource_manager import ResourceManager
    from infrastructure.fast_forward_engine import FastForwardEngine
    from infrastructure.distributed_coordinator import DistributedCoordinator
    from infrastructure.performance_monitor import PerformanceMonitor
    from infrastructure.scalability_config import ScalabilityConfig
    
    # Import scenario components
    from scenarios.scenario_engine import ScenarioEngine
    from scenarios.curriculum_validator import CurriculumValidator
    from scenarios.dynamic_generator import DynamicScenarioGenerator
    from scenarios.scenario_config import ScenarioConfigManager

    # Import new features
    from learning.episodic_learning import EpisodicLearningManager
    from learning.rl_environment import FBABenchRLEnvironment, FBABenchSimulator # FBABenchSimulator to mock environment for RL
    from learning.learning_config import LearningConfig
    from integration.real_world_adapter import RealWorldAdapter
    from integration.marketplace_apis.marketplace_factory import MarketplaceFactory
    from integration.integration_validator import IntegrationValidator
    from plugins.plugin_framework import PluginManager
    from plugins.scenario_plugins.base_scenario_plugin import ScenarioPlugin # For type hinting plugins
    from plugins.agent_plugins.base_agent_plugin import AgentPlugin # For type hinting plugins
    from community.contribution_tools import ContributionManager

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Details: {e.args[0]}") # Access the message directly from args
    print("This CLI requires FBA-Bench v3 components. Make sure you're running from the project root.")
    sys.exit(1)

logger = logging.getLogger(__name__)
# Configure logging for CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ExperimentConfig:
    """Configuration for an experiment loaded from sweep.yaml."""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.experiment_name = config_data['experiment_name']
        self.description = config_data.get('description', '')
        self.base_parameters = config_data['base_parameters']
        self.parameter_sweep = config_data['parameter_sweep']
        self.output_config = config_data.get('output', {})
        
    def generate_parameter_combinations(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Generate all parameter combinations from the sweep configuration.
        
        Returns:
            Iterator of (run_number, parameters) tuples
        """
        # Extract sweep parameters
        sweep_params = {}
        for param_name, param_values in self.parameter_sweep.items():
            if param_name == 'competitor_persona_distribution':
                # Special handling for persona distribution configs
                sweep_params[param_name] = param_values
            else:
                sweep_params[param_name] = param_values
        
        # Generate all combinations using itertools.product
        param_names = list(sweep_params.keys())
        param_value_lists = [sweep_params[name] for name in param_names]
        
        run_number = 1
        for combination in itertools.product(*param_value_lists):
            parameters = dict(zip(param_names, combination))
            
            # Merge with base parameters
            final_params = self.base_parameters.copy()
            final_params.update(parameters)
            
            yield run_number, final_params
            run_number += 1
    
    def get_total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        total = 1
        for param_values in self.parameter_sweep.values():
            total *= len(param_values)
        return total


class SimulationRunner:
    """Executes individual simulations with specified parameters."""
    
    def __init__(self, experiment_config: ExperimentConfig, reproducibility_config: Optional[ReproducibilityConfig] = None):
        self.experiment_config = experiment_config
        self.reproducibility_config = reproducibility_config or get_global_config()
        # Scalability config will be passed per-run from ExperimentManager
        self.scalability_config: Optional[ScalabilityConfig] = None
        self.learning_config: LearningConfig = LearningConfig() # Always initialize with a default
        self.real_world_adapter: RealWorldAdapter = RealWorldAdapter(mode="simulation") # Always initialize with a default
        
    def set_scalability_config(self, config: ScalabilityConfig): # Reverted to public
        self.scalability_config = config
    
    def set_learning_config(self, config: LearningConfig): # Reverted to public
        self.learning_config = config

    def set_real_world_adapter(self, adapter: RealWorldAdapter): # Reverted to public
        self.real_world_adapter = adapter


    async def run_single_simulation(self, run_number: int, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single simulation with the given parameters and scalability configurations.
        
        Args:
            run_number: Unique identifier for this run
            parameters: Complete parameter set for this simulation
            
        Returns:
            Dict containing complete simulation results and metrics
        """
        logger.info(f"🚀 Starting simulation run {run_number}")
        logger.info(f"   Parameters: {self._format_key_parameters(parameters)}")
        
        # Initialize reproducibility features for this run
        if self.reproducibility_config:
            # Set up deterministic seeding for this run
            run_seed = self.reproducibility_config.master_seed + run_number
            sim_seed = SimSeed(
                master_seed=run_seed,
                component_seeds={
                    'simulation': run_seed + 1,
                    'competitors': run_seed + 2,
                    'events': run_seed + 3,
                    'llm': run_seed + 4
                }
            )
            SimSeed.set_master_seed(run_seed) # Ensure seed is set for all components

            # Apply simulation mode
            mode_controller = get_mode_controller()
            if self.reproducibility_config.simulation_mode:
                set_global_mode(self.reproducibility_config.simulation_mode)
                logger.info(f"🎯 Simulation mode: {self.reproducibility_config.simulation_mode.value}")
            
            # Initialize LLM cache if needed
            if self.reproducibility_config.enable_llm_caching:
                cache = LLMResponseCache(self.reproducibility_config.cache_file)
                await cache.initialize()
                logger.info(f"💾 LLM caching enabled: {self.reproducibility_config.cache_file}")
            
        # Convert duration_hours to max_ticks (assuming 1 tick per second)
        duration_hours = parameters.get('duration_hours', 72)
        max_ticks = int(duration_hours * 3600 / parameters.get('tick_interval_seconds', 1.0))
        
        # Create simulation configuration
        sim_config = SimulationConfig(
            tick_interval_seconds=parameters.get('tick_interval_seconds', 1.0),
            max_ticks=max_ticks,
            time_acceleration=parameters.get('time_acceleration', 1.0),
            auto_start=False,
            seed=SimSeed.get_master_seed() # Pass the run-specific master seed
        )
        
        # --- Scalability Component Initialization ---
        llm_batcher: Optional[LLMBatcher] = None
        resource_manager: Optional[ResourceManager] = None
        fast_forward_engine: Optional[FastForwardEngine] = None
        distributed_coordinator: Optional[DistributedCoordinator] = None
        performance_monitor: Optional[PerformanceMonitor] = None
        
        # Decide EventBus backend based on distributed mode
        event_bus: EventBus
        current_backend: Union[AsyncioQueueBackend, DistributedBackend]
        
        if self.scalability_config and self.scalability_config.enable_distributed_mode:
            logger.info("⚙️ Distributed mode enabled. Initializing DistributedEventBus backend.")
            # For local testing with MockRedisBroker, otherwise connect to actual Redis/Kafka
            distributed_message_broker = MockRedisBroker()
            distributed_event_bus_instance = DistributedEventBus(broker=distributed_message_broker)
            current_backend = DistributedBackend(distributed_event_bus_instance)
            event_bus = EventBus(backend=current_backend)
            
            distributed_coordinator = DistributedCoordinator(distributed_event_bus_instance, self.scalability_config)
            await distributed_coordinator.start()
            logger.info("DistributedCoordinator started.")

        else:
            logger.info("⚙️ Running in single-process mode (AsyncioQueueBackend).")
            current_backend = AsyncioQueueBackend()
            event_bus = EventBus(backend=current_backend)

        # Initialize common infrastructure components
        resource_manager = ResourceManager()
        if self.scalability_config and self.scalability_config.cost_limit_per_run is not None:
            resource_manager.enforce_cost_limits(self.scalability_config.cost_limit_per_run)
            logger.info(f"LLM cost limit set to ${self.scalability_config.cost_limit_per_run:.2f}")

        performance_monitor = PerformanceMonitor(resource_manager=resource_manager)
        await performance_monitor.start() # Start monitoring in background

        if self.scalability_config and self.scalability_config.enable_llm_batching:
            llm_batcher = LLMBatcher()
            # Set batching parameters from config
            llm_batcher.set_batch_parameters(
                max_size=self.scalability_config.max_batch_size,
                timeout_ms=self.scalability_config.batch_timeout_ms
            )
            await llm_batcher.start()
            logger.info("LLM Batching System started.")

        if self.scalability_config and self.scalability_config.enable_fast_forward:
            # Pass event_bus and orchestrator for internal use
            orchestrator_for_ff = SimulationOrchestrator(sim_config) # Temp instance for type hinting
            fast_forward_engine = FastForwardEngine(event_bus, orchestrator_for_ff)
            # Configure fast-forward threshold
            fast_forward_engine.activity_level_threshold = self.scalability_config.fast_forward_threshold
            await fast_forward_engine.start()
            logger.info("Fast-Forward Simulation Engine started.")


        orchestrator = SimulationOrchestrator(sim_config)
        
        # Initialize learning system components if enabled
        # Need to ensure self.learning_config is available here
        learning_manager = self.episodic_learning_manager if hasattr(self, 'episodic_learning_manager') else EpisodicLearningManager()
        rl_env = None
        if self.learning_config and self.learning_config.enable_episodic_learning:
            logger.info("Initializing RL Environment for agent learning within simulation runner.")
            # Pass a reference to the actual orchestrator for a proper RL environment if needed
            rl_env = FBABenchRLEnvironment(simulator=FBABenchSimulator(), reward_objective=self.learning_config.reward_function)
            rl_env.configure_reward_function(self.learning_config.reward_function)
            
        # Execute simulation with event-driven architecture
        start_time = time.time()
        
        try:
            # Start event bus
            await event_bus.start()
            
            # Configure services with parameters
            competitor_manager = self._setup_competitor_manager(parameters, event_bus)
            dashboard_service = DashboardAPIService(event_bus)
            
            # Subscribe services to event bus
            await self._setup_event_subscriptions(event_bus, competitor_manager, dashboard_service)
            
            # Start orchestrator
            await orchestrator.start(event_bus)

            # Integrate RealWorldAdapter into the simulation loop / agent decision making
            # This is a conceptual integration point. In a real system, the orchestrator
            # or agents would receive the adapter instance.
            if self.real_world_adapter:
                logger.info(f"Simulation configured with RealWorldAdapter in {self.real_world_adapter.mode} mode.")
                # If in sandbox/live mode, initial real-world sync might happen here
                if self.real_world_adapter.mode != "simulation":
                    real_state = await self.real_world_adapter.sync_state_from_real_world()
                    logger.info(f"Synchronized initial state from real world: {real_state}")

            # Additional integration for distributed mode (if enabled)
            if self.scalability_config and self.scalability_config.enable_distributed_mode and distributed_coordinator:
                # Spawn workers based on max_workers config
                for i in range(self.scalability_config.max_workers):
                    # Placeholder partition config for demonstration
                    await distributed_coordinator.spawn_worker({"default_partition": [f"agent_{i*10 + j}" for j in range(10)]})
                
                # In a real distributed setup, the orchestrator might wait for coordinator
                # to advance ticks, or coordinator drives the orchestrator.
                # For now, orchestrator continues its loop, but events route via distributed bus.

            # Main simulation loop controlled by orchestrator instance
            max_duration = parameters.get('duration_hours', 72) * 3600
            start_time_check = time.time()
            
            # Placeholder for agent: get it from `parameters` or a global agent manager
            # For RL training, the RL agent operates on the RL environment
            mock_agent = None
            if rl_env:
                # This mock agent interacts with the RL environment
                class MockRLAgent:
                    def __init__(self, action_space):
                        self.action_space = action_space
                    def choose_action(self, observation):
                        return self.action_space.sample() # Random action for simulation
                    async def learn(self, experience):
                        pass # Learning happens in _train_agent_rl
                mock_agent = MockRLAgent(rl_env.action_space)


            while orchestrator.is_running:
                # Simulation advances by one tick
                # In a full RL loop, the agent would interact with rl_env.step() here
                # and then that would update the simulator (orchestrator)
                
                # For this general simulation loop, just call orchestrator's tick directly
                # If RL is enabled and actively training, this part would be replaced by the RL agent's loop
                
                # Example: If there's an agent in this simulation run, let it interact
                if mock_agent and self.learning_config.enable_episodic_learning and not (hasattr(self.learning_config, 'training_agent_id') and self.learning_config.training_agent_id):
                    # If episodic learning is ON but we are not in dedicated `train-agent` mode,
                    # still simulate agent interaction with RL env to generate data.
                    current_observation = rl_env._get_obs() # Get observation from simulated state
                    action = mock_agent.choose_action(current_observation)
                    _ , reward, terminated, truncated, info = rl_env.step(action)
                    # Use real_world_adapter to execute action if configured
                    if self.real_world_adapter and self.real_world_adapter.mode != "simulation":
                        final_action = await self.real_world_adapter.translate_simulation_action({"type": "rl_action", "value": action})
                        await self.real_world_adapter.execute_action(final_action)
                    
                    # Store episodic data after each step if needed for batch learning
                    # For simplicity, we just log in episodic_learning.py at end of episode.
                    if terminated or truncated:
                        break # Exit simulation loop if RL episode ends


                # Check for memory cleanup
                if resource_manager and resource_manager.monitor_memory_usage()["system_percent"] > self.scalability_config.memory_cleanup_threshold * 100:
                    resource_manager.trigger_garbage_collection(self.scalability_config.memory_cleanup_threshold)

                # Check for fast-forward if enabled
                if fast_forward_engine and self.scalability_config.enable_fast_forward:
                    current_sim_time = orchestrator._calculate_simulation_time()
                    # A more robust idle detection would depend on agent activity *events*
                    is_idle = fast_forward_engine.detect_idle_period(fast_forward_engine._last_agent_activity, fast_forward_engine.activity_level_threshold)
                    
                    if is_idle and (orchestrator.current_tick + fast_forward_engine.min_fast_forward_duration_ticks) <= orchestrator.config.max_ticks:
                        target_tick = orchestrator.current_tick + fast_forward_engine.min_fast_forward_duration_ticks
                        logger.info(f"Attempting fast-forward to tick {target_tick}")
                        await fast_forward_engine.fast_forward_to_tick(target_tick)
                        # After fast-forward, orchestrator.current_tick is updated, loop continues
                
                await asyncio.sleep(0.1)
                
                # Add timeout to prevent infinite loops in testing
                if time.time() - start_time_check > min(max_duration, 120):  # Max 120 seconds for CLI testing with overhead
                    logger.info(f"    ⏰ Simulation timeout reached (120s), stopping...")
                    break
                
        except Exception as e:
            logger.exception(f"❌ Simulation run {run_number} failed: {e}")
            raise
        finally:
            # Ensure orchestrator is stopped
            if orchestrator.is_running:
                await orchestrator.stop()
            
            # Stop event bus and other components
            await event_bus.stop()
            if llm_batcher:
                await llm_batcher.stop()
                logger.info("LLMBatcher stopped.")
            if fast_forward_engine:
                await fast_forward_engine.stop()
                logger.info("FastForwardEngine stopped.")
            if distributed_coordinator:
                await distributed_coordinator.stop()
                logger.info("DistributedCoordinator stopped.")
            if performance_monitor:
                await performance_monitor.stop()
                logger.info("PerformanceMonitor stopped.")
            
            # Log final resource usage/cost
            if resource_manager:
                logger.info(f"Final LLM cost: ${resource_manager.get_total_api_cost():.6f}")
                logger.debug(f"Final memory usage: {resource_manager.stats.get('current_memory_mb', 0):.2f} MB")
                performance_monitor.generate_performance_report() # Generate final report

                # Store any learning experiences if enabled and not in dedicated training mode
                if self.learning_config.enable_episodic_learning and not (hasattr(self.learning_config, 'training_agent_id') and self.learning_config.training_agent_id):
                    simplified_episode_data = {"simulation_id": run_number, "final_state": orchestrator.get_current_state(), "parameters": parameters}
                    simplified_outcomes = {"final_profit": result['metrics'].get('total_profit', 0)}
                    if hasattr(self, 'episodic_learning_manager'): # Check if it's set in ExperimentManager
                        await self.episodic_learning_manager.store_episode_experience("default_agent", simplified_episode_data, simplified_outcomes)
                        await self.episodic_learning_manager.track_learning_progress("default_agent", {"run": run_number, "profit": simplified_outcomes['final_profit']})
                        logger.info("Episodic learning data stored for this run.")
                    else:
                        logger.warning("EpisodicLearningManager not available in SimulationRunner context for storing experience.")

            SimSeed.reset_master_seed() # Clean up seed after run
            
        execution_time = time.time() - start_time
        logger.info(f"✅ Simulation run {run_number} completed in {execution_time:.1f}s")
        
        # Create simplified audit record from simulation results
        result = self._create_simulation_result(run_number, parameters, dashboard_service, execution_time)
        if resource_manager:
            result['resource_metrics'] = resource_manager.stats.copy()
            result['llm_costs'] = resource_manager._llm_api_costs.copy() # Expose detailed costs
            result['total_llm_cost'] = resource_manager.get_total_api_cost()
        return result
    
    def _setup_competitor_manager(self, parameters: Dict[str, Any], event_bus: EventBus) -> CompetitorManager:
        """Set up CompetitorManager with experiment parameters."""
        # Extract competitor manager config
        cm_config = parameters.get('competitor_manager', {})
        
        # Handle persona distribution
        persona_dist = parameters.get('competitor_persona_distribution', {})
        if isinstance(persona_dist, dict) and 'distribution' in persona_dist:
            cm_config['persona_distribution'] = persona_dist['distribution']
        
        # Create competitor manager with event bus
        cm = CompetitorManager(cm_config)
        cm.event_bus = event_bus  # Set event bus for publishing events
        
        return cm
    
    async def _setup_event_subscriptions(self, event_bus: EventBus, 
                                       competitor_manager: CompetitorManager,
                                       dashboard_service: DashboardAPIService) -> None:
        """Subscribe all services to relevant events."""
        # Subscribe services to events they need
        # Note: This would normally be done in a proper initialization routine
        # For now, we'll set up the basic subscriptions manually
        
        from events import TickEvent, SaleOccurred, CompetitorPricesUpdated
        
        # CompetitorManager subscribes to TickEvents
        await event_bus.subscribe(TickEvent, competitor_manager._handle_tick_event)
        
        # DashboardAPIService subscribes to all events for state tracking
        # Note: Using simplified event handlers for CLI integration
        await event_bus.subscribe(TickEvent, self._dashboard_tick_handler(dashboard_service))
        await event_bus.subscribe(SaleOccurred, self._dashboard_sales_handler(dashboard_service))
        await event_bus.subscribe(CompetitorPricesUpdated, self._dashboard_competitor_handler(dashboard_service))
    
    def _create_simulation_result(self, run_number: int, parameters: Dict[str, Any],
                                dashboard_service: DashboardAPIService, execution_time: float) -> Dict[str, Any]:
        """Create simulation result record from dashboard data."""
        # Get final simulation state
        snapshot = dashboard_service.get_simulation_snapshot()
        
        # Create simplified result record
        return {
            'run_number': run_number,
            'parameters': parameters,
            'execution_time_seconds': execution_time,
            'config_hash': self._hash_dict(parameters),
            'timestamp': datetime.now().isoformat(),
            'simulation_data': {
                'final_snapshot': snapshot,
                'total_ticks': snapshot.get('current_tick', 0),
                'simulation_duration_hours': parameters.get('duration_hours', 72),
                'cost_basis': parameters.get('cost_basis', 0),
                'initial_price': parameters.get('initial_price', 0)
            },
            'metrics': {
                'total_revenue': snapshot.get('sales_summary', {}).get('total_revenue', 0),
                'total_profit': snapshot.get('financial_summary', {}).get('total_profit', 0),
                'final_price': snapshot.get('current_price', 0),
                'competitor_count': len(snapshot.get('competitors', [])),
                'sales_count': len(snapshot.get('recent_sales', []))
            }
        }
    
    def _format_key_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format key parameters for display."""
        key_params = []
        
        if 'initial_price' in parameters:
            key_params.append(f"price=${parameters['initial_price']}")
        
        if 'competitor_persona_distribution' in parameters:
            dist = parameters['competitor_persona_distribution']
            if isinstance(dist, dict) and 'name' in dist:
                key_params.append(f"market={dist['name']}")
        
        if 'market_sensitivity' in parameters:
            key_params.append(f"sensitivity={parameters['market_sensitivity']}")
        
        return ", ".join(key_params)
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create deterministic hash of dictionary."""
        return hashlib.sha256(str(sorted(data.items())).encode()).hexdigest()[:16]
    
    def _dashboard_tick_handler(self, dashboard_service: DashboardAPIService):
        """Create tick event handler for dashboard service."""
        async def handler(event):
            # Update dashboard state with tick information
            dashboard_service.simulation_state['current_tick'] = event.tick_number
            dashboard_service.simulation_state['simulation_time'] = event.timestamp
            dashboard_service.simulation_state['last_update'] = datetime.now()
        return handler
    
    def _dashboard_sales_handler(self, dashboard_service: DashboardAPIService):
        """Create sales event handler for dashboard service."""
        async def handler(event):
            # Update dashboard state with sales information
            if 'recent_sales' not in dashboard_service.simulation_state:
                dashboard_service.simulation_state['recent_sales'] = []
            dashboard_service.simulation_state['recent_sales'].append({
                'timestamp': event.timestamp,
                'amount': getattr(event, 'amount', 0),
                'quantity': getattr(event, 'quantity', 1)
            })
            # Keep only recent sales (last 100)
            dashboard_service.simulation_state['recent_sales'] = dashboard_service.simulation_state['recent_sales'][-100:]
        return handler
    
    def _dashboard_competitor_handler(self, dashboard_service: DashboardAPIService):
        """Create competitor prices updated handler for dashboard service."""
        async def handler(event):
            # Update dashboard state with competitor information
            dashboard_service.simulation_state['competitors'] = event.competitors
            dashboard_service.simulation_state['competitor_update_time'] = datetime.now()
        return handler


class ExperimentManager:
    """Manages experiment execution and result aggregation."""
    
    def __init__(self, config_file: str, reproducibility_config: Optional[ReproducibilityConfig] = None):
        self.config_file = config_file
        self.reproducibility_config = reproducibility_config or get_global_config()
        self.experiment_config = self._load_config()
        self.runner = SimulationRunner(self.experiment_config, self.reproducibility_config)
        self.results_dir = self._create_results_directory()
        
    def _load_config(self) -> ExperimentConfig:
        """Load and validate experiment configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            return ExperimentConfig(config_data)
        except Exception as e:
            print(f"❌ Failed to load config file {self.config_file}: {e}")
            sys.exit(1)
    
    def _create_results_directory(self) -> Path:
        """Create timestamped results directory."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(f"results/{self.experiment_config.experiment_name}_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment configuration
        config_path = results_dir / "experiment_config.yaml"
        with open(config_path, 'w') as f:
            with open(self.config_file, 'r') as source:
                f.write(source.read())
        
        print(f"📁 Results directory: {results_dir}")
        return results_dir
    
    async def run_experiment(self, max_runs: Optional[int] = None, parallel_workers: int = 1, args=None) -> None:
        """
        Execute the full experiment with optional parallel processing.
        
        Args:
            max_runs: Maximum number of runs to execute (for testing)
            parallel_workers: Number of parallel processes to use
            args: CLI arguments for reproducibility features
        """
        print(f"🧪 Starting experiment: {self.experiment_config.experiment_name}")
        print(f"📝 Description: {self.experiment_config.description}")
        
        # Display reproducibility configuration
        if self.reproducibility_config:
            print(f"🔬 Reproducibility enabled:")
            print(f"   Mode: {self.reproducibility_config.simulation_mode.value if self.reproducibility_config.simulation_mode else 'default'}")
            print(f"   Master seed: {self.reproducibility_config.master_seed}")
            if self.reproducibility_config.enable_llm_caching:
                print(f"   LLM caching: {self.reproducibility_config.cache_file}")
    
            # Learning and Real-world Integration configurations
            self.learning_config = self._load_learning_config(args)
            print(f"🧠 Learning enabled: {self.learning_config.enable_episodic_learning}")
            print(f"🌐 Real-world mode: {self.learning_config.real_world_mode}")
        else: # Initialize default configs even if reproducibility is not explicitly enabled
            self.learning_config = self._load_learning_config(args)
        
        # Initialize real-world adapter
        self.real_world_adapter = RealWorldAdapter(mode=self.learning_config.real_world_mode)

        # Learning component initialization
        self.episodic_learning_manager = EpisodicLearningManager() # Persistent storage for agent experiences

        total_combinations = self.experiment_config.get_total_combinations()
        if max_runs:
            total_combinations = min(total_combinations, max_runs)
        
        print(f"🔢 Total parameter combinations: {total_combinations}")
        
        # Pass LearningConfig and RealWorldAdapter to the runner
        self.runner.set_learning_config(self.learning_config) # New method
        self.runner.set_real_world_adapter(self.real_world_adapter) # New method
        
        # Pass episodic learning manager to runner for experience storage
        self.runner.episodic_learning_manager = self.episodic_learning_manager
        
        # Scenario: Train agent via RL
        if hasattr(self.learning_config, 'training_agent_id') and self.learning_config.training_agent_id:
            agent_id = self.learning_config.training_agent_id
            episodes_to_train = self.learning_config.training_episodes
            print(f"🧠 Initiating RL training for agent '{agent_id}' over {episodes_to_train} episodes...")
            await self._train_agent_rl(agent_id, episodes_to_train, self.real_world_adapter, self.episodic_learning_manager)
            sys.exit(0) # Exit after training

        # Scenario: Export learned agent
        if args.export_agent:
            agent_id, version = args.export_agent
            print(f"💾 Exporting learned agent '{agent_id}' version '{version}'...")
            export_path = await self.episodic_learning_manager.export_learned_agent(agent_id, version)
            print(f"✅ Agent exported to: {export_path}")
            sys.exit(0) # Exit after export

        # Initialize Plugin Manager and Contribution Manager
        self.plugin_manager = PluginManager()
        self.contribution_manager = ContributionManager(plugin_manager=self.plugin_manager)

        # Load initial plugins from CLI args if provided
        if args.load_plugin:
            print(f"📦 Loading plugins from: {args.load_plugin}")
            for plugin_path in args.load_plugin:
                # Ensure plugin_path is treated as a file path, load from its directory
                try:
                    full_plugin_path = Path(plugin_path)
                    if full_plugin_path.is_file():
                        await self.plugin_manager.load_plugins(str(full_plugin_path.parent))
                        print(f"Loaded plugin from file: {plugin_path}")
                    elif full_plugin_path.is_dir():
                        await self.plugin_manager.load_plugins(str(full_plugin_path))
                        print(f"Loaded plugins from directory: {plugin_path}")
                    else:
                        print(f"⚠️ Warning: Plugin path '{plugin_path}' is neither a file nor a directory. Skipping.")
                except Exception as e:
                    print(f"❌ Error loading plugin {plugin_path}: {e}")
        
        # Load from default plugin dir
        await self.plugin_manager.load_plugins(args.plugin_dir)


        
        if parallel_workers > 1:
            print(f"⚡ Parallel processing enabled: {parallel_workers} workers")
            await self._run_experiment_parallel(max_runs, parallel_workers)
        else:
            print(f"🔄 Sequential processing")
            await self._run_experiment_sequential(max_runs)
        
        # Run post-experiment reproducibility validation if requested
        if args and getattr(args, 'validate_reproducibility', False):
            print(f"🔍 Running post-experiment reproducibility validation...")
            try:
                analysis_results = analyze_experiment_reproducibility(str(self.results_dir), args)
                if analysis_results.get('reproducibility_summary', {}).get('reproducibility_rate', 0) == 1.0:
                    print("✅ Reproducibility validation: PASSED")
                else:
                    print("⚠️  Reproducibility validation: Some issues detected")
            except Exception as e:
                print(f"❌ Reproducibility validation failed: {e}")
    
    async def _run_experiment_sequential(self, max_runs: Optional[int] = None) -> None:
        """Execute experiment runs sequentially."""
        run_count = 0
        for run_number, parameters in self.experiment_config.generate_parameter_combinations():
            if max_runs and run_count >= max_runs:
                break
                
            try:
                # Execute simulation
                result = await self.runner.run_single_simulation(run_number, parameters)
                
                # Save result
                self._save_run_result(run_number, parameters, result)
                
                run_count += 1
                
            except Exception as e:
                print(f"❌ Run {run_number} failed: {e}")
                continue
        
        logger.info(f"🎉 Experiment completed! {run_count} successful runs saved to {self.results_dir}")
    
    def _save_run_result(self, run_number: int, parameters: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Save individual run result to JSON file."""
        result_file = self.results_dir / f"run_{run_number}.json"
        try:
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.info(f"💾 Saved run {run_number} results to {result_file}")
            
            # Print key metrics for immediate feedback
            metrics = result.get('metrics', {})
            logger.info(f"    📊 Key metrics: Revenue=${metrics.get('total_revenue', 0):.2f}, "
                        f"Profit=${metrics.get('total_profit', 0):.2f}, "
                        f"Final Price=${metrics.get('final_price', 0):.2f}")
            if 'total_llm_cost' in result:
                logger.info(f"    💸 LLM Cost: ${result['total_llm_cost']:.6f}")

        except Exception as e:
            logger.error(f"Failed to save run {run_number} result to {result_file}: {e}")
    
    def _load_learning_config(self, args) -> LearningConfig:
        """Load learning configuration from CLI arguments and/or config files."""
        learning_config = LearningConfig()
        
        # Override with CLI arguments
        if hasattr(args, 'enable_learning') and args.enable_learning:
            learning_config.enable_episodic_learning = True
            
        if hasattr(args, 'real_world_mode') and args.real_world_mode:
            learning_config.real_world_mode = args.real_world_mode
            
        if hasattr(args, 'train_agent') and args.train_agent:
            learning_config.training_agent_id = args.train_agent
            learning_config.training_episodes = 50  # Default episodes
            
        # Load from config file if provided
        if hasattr(args, 'learning_config') and args.learning_config:
            try:
                with open(args.learning_config, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update learning config with file data
                for key, value in config_data.items():
                    if hasattr(learning_config, key):
                        setattr(learning_config, key, value)
                        
                logger.info(f"📚 Loaded learning configuration from: {args.learning_config}")
                
            except Exception as e:
                logger.warning(f"Failed to load learning config from {args.learning_config}: {e}")
                
        return learning_config
    
    async def _train_agent_rl(self, agent_id: str, episodes: int, real_world_adapter: RealWorldAdapter, learning_manager: EpisodicLearningManager) -> None:
        """Train an agent using reinforcement learning."""
        logger.info(f"🎯 Starting RL training for agent '{agent_id}' over {episodes} episodes")
        
        # Create RL environment
        rl_env = FBABenchRLEnvironment(simulator=FBABenchSimulator(), reward_objective="profit_maximization")
        
        # Simple training loop
        for episode in range(episodes):
            logger.info(f"📈 Training episode {episode + 1}/{episodes}")
            
            # Reset environment for new episode
            observation, info = rl_env.reset()
            episode_data = {"episode": episode + 1, "agent_id": agent_id, "observations": []}
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 100:  # Max 100 steps per episode
                # Simple random action for demonstration
                action = rl_env.action_space.sample()
                
                # Take step in environment
                next_observation, reward, terminated, truncated, info = rl_env.step(action)
                
                # Store experience
                episode_data["observations"].append({
                    "step": step,
                    "observation": observation.tolist() if hasattr(observation, 'tolist') else observation,
                    "action": action.tolist() if hasattr(action, 'tolist') else action,
                    "reward": reward,
                    "next_observation": next_observation.tolist() if hasattr(next_observation, 'tolist') else next_observation
                })
                
                total_reward += reward
                observation = next_observation
                done = terminated or truncated
                step += 1
                
                # Optional: Execute action in real world if configured
                if real_world_adapter.mode != "simulation":
                    try:
                        sim_action = {"type": "rl_action", "value": action}
                        real_action = await real_world_adapter.translate_simulation_action(sim_action)
                        await real_world_adapter.execute_action(real_action)
                    except Exception as e:
                        logger.warning(f"Failed to execute real-world action: {e}")
            
            # Store episode experience
            episode_outcomes = {"total_reward": total_reward, "steps": step}
            await learning_manager.store_episode_experience(agent_id, episode_data, episode_outcomes)
            await learning_manager.track_learning_progress(agent_id, {"episode": episode + 1, "total_reward": total_reward})
            
            logger.info(f"   Episode {episode + 1} completed: {step} steps, total reward: {total_reward:.2f}")
        
        logger.info(f"✅ RL training completed for agent '{agent_id}'")
            
    async def _run_experiment_parallel(self, max_runs: Optional[int] = None, parallel_workers: int = 4) -> None:
        """Execute experiment runs in parallel using multiprocessing."""
        # Collect all parameter combinations
        all_combinations = []
        for run_number, parameters in self.experiment_config.generate_parameter_combinations():
            if max_runs and len(all_combinations) >= max_runs:
                break
            all_combinations.append((run_number, parameters))
        
        # Create worker arguments. Need to pass scalability config to workers too.
        # Note: ScalabilityConfig dataclass needs to be serializable for multiprocessing.
        scalability_config_for_workers = self.runner.scalability_config # Get the config set for this runner
        worker_args = [
            (run_number, parameters, str(self.results_dir), self.config_file, asdict(scalability_config_for_workers))
            for run_number, parameters in all_combinations
        ]
        
        logger.info(f"🚀 Launching {len(worker_args)} simulations across {parallel_workers} local processes...")
        
        # Execute in parallel
        successful_runs = 0
        with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            # Submit all jobs
            future_to_run = {
                executor.submit(_run_single_simulation_worker, args): args[0]
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_run):
                run_number = future_to_run[future]
                try:
                    success = future.result()
                    if success:
                        successful_runs += 1
                        logger.info(f"✅ Run {run_number} completed successfully by worker")
                    else:
                        logger.error(f"❌ Run {run_number} failed by worker")
                except Exception as e:
                    logger.exception(f"❌ Run {run_number} crashed in worker: {e}")
        
        logger.info(f"🎉 Parallel experiment completed! {successful_runs}/{len(worker_args)} runs successful")


def _run_single_simulation_worker(args: tuple) -> bool:
    """
    Worker function for parallel processing of individual simulation runs.
    
    This function runs in a separate process and cannot use shared state.
    """
    run_number, parameters, results_dir, config_file, scalability_config_data = args
    
    try:
        # Configure logging for the worker process
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - Worker %(process)d - %(levelname)s - %(message)s')
        worker_logger = logging.getLogger(f"Worker-{os.getpid()}")
        worker_logger.info(f"🔄 Worker {os.getpid()} starting simulation run {run_number}")
        
        # Create a new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Load experiment config
            with open(config_file, 'r') as f:
                import yaml
                config_data = yaml.safe_load(f)
            
            exp_config = ExperimentConfig(config_data)
            runner = SimulationRunner(exp_config)

            # Recreate ScalabilityConfig from dict (must be done in worker process)
            from infrastructure.scalability_config import ScalabilityConfig
            worker_scalability_config = ScalabilityConfig(**scalability_config_data)
            runner.set_scalability_config(worker_scalability_config)

            # Execute the simulation
            result = loop.run_until_complete(
                runner.run_single_simulation(run_number, parameters)
            )
            
            # Save result
            result_file = Path(results_dir) / f"run_{run_number}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            worker_logger.info(f"💾 Worker {os.getpid()} saved run {run_number} to {result_file}")
            return True
            
        finally:
            loop.close()
            
    except Exception as e:
        worker_logger.exception(f"❌ Worker {os.getpid()} failed for run {run_number}: {e}")
        return False


def setup_reproducibility_from_args(args) -> ReproducibilityConfig:
    """
    Set up reproducibility configuration from CLI arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured ReproducibilityConfig instance
    """
    # Start with default or file-based config
    if hasattr(args, 'config_reproducibility') and args.config_reproducibility:
        if os.path.exists(args.config_reproducibility):
            config = ReproducibilityConfig.load_from_file(args.config_reproducibility)
            if config is None:
                print(f"⚠️ Failed to load reproducibility config from {args.config_reproducibility}, using defaults")
                config = load_config_from_env()
        else:
            print(f"⚠️ Reproducibility config file not found: {args.config_reproducibility}, using defaults")
            config = load_config_from_env()
    else:
        config = load_config_from_env()
    
    # Override with CLI arguments
    if hasattr(args, 'deterministic') and args.deterministic:
        config.simulation_mode = SimulationMode.DETERMINISTIC
        config.llm_cache.allow_cache_misses = False
        config.seed_management.strict_validation = True
    
    if hasattr(args, 'simulation_mode') and args.simulation_mode:
        mode_map = {
            'deterministic': SimulationMode.DETERMINISTIC,
            'stochastic': SimulationMode.STOCHASTIC,
            'research': SimulationMode.RESEARCH
        }
        config.simulation_mode = mode_map[args.simulation_mode]
    
    if hasattr(args, 'master_seed') and args.master_seed is not None:
        config.seed_management.master_seed = args.master_seed
    
    if hasattr(args, 'cache_file') and args.cache_file:
        config.llm_cache.cache_file = args.cache_file
    
    if hasattr(args, 'record_responses') and args.record_responses:
        config.llm_cache.enabled = True
        if config.simulation_mode != SimulationMode.DETERMINISTIC:
            config.simulation_mode = SimulationMode.RESEARCH
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("⚠️ Reproducibility configuration issues:")
        for issue in issues:
            print(f"  • {issue}")
        if config.fail_fast_on_validation_error:
            print("❌ Exiting due to configuration errors")
            sys.exit(1)
    
    # Apply global configuration
    set_global_config(config)
    
    print(f"🔬 Reproducibility mode: {config.simulation_mode.value}")
    if config.seed_management.master_seed:
        print(f"🎲 Master seed: {config.seed_management.master_seed}")
    
    return config

async def analyze_experiment_reproducibility(results_dir: str, args) -> Dict[str, Any]:
    """
    Analyze experiment results for reproducibility validation.
    
    Args:
        results_dir: Path to experiment results directory
        args: CLI arguments with analysis options
        
    Returns:
        Analysis results dictionary
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"📊 Analyzing reproducibility for: {results_dir}")
    
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(results_path),
        "analysis_type": "reproducibility_validation"
    }

    # Initialize Learning Manager and RealWorldAdapter for analysis if needed
    # (This is a simplified approach, a more robust solution would pass these objects around)
    learning_manager = EpisodicLearningManager()
    # Mock RealWorldAdapter if it's not actually running in a simulation env
    temp_real_world_adapter = RealWorldAdapter(mode="simulation")
    integration_validator = IntegrationValidator(temp_real_world_adapter)

    # Handle integration validation if argument is present
    if hasattr(args, 'validate_integration') and args.validate_integration:
        logger.info("Running integration validation tests...")
        # Define dummy test actions for CLI-triggered validation
        mock_sim_action = {"type": "set_price", "value": 20.0}
        mock_real_expected_action = {"api_call": "update_product_price", "parameters": {"product_sku": "FBA-SKU-123", "new_price": 20.0}}
        
        consistency_check_result,_ = await integration_validator.validate_action_consistency(mock_sim_action, mock_real_expected_action)

        dangerous_actions_for_test = [
            {"type": "set_price", "value": 9999.0}, # Arbitrarily high price
            {"type": "adjust_inventory", "value": -5000} # Arbitrarily large inventory drop
        ]
        safety_test_results = await integration_validator.test_safety_constraints(dangerous_actions_for_test)
        
        # For simplicity, we directly add results here. A full system would have more structured reporting.
        analysis_results["integration_validation_summary"] = {
            "action_consistency_passed": consistency_check_result,
            "safety_tests": safety_test_results
        }
        logger.info(f"Integration validation summary: {analysis_results['integration_validation_summary']}")

    # Handle plugin benchmarking if argument is present
    if hasattr(args, 'benchmark_community_plugin') and args.benchmark_community_plugin:
        plugin_path_to_benchmark = args.benchmark_community_plugin
        logger.info(f"Running benchmark for community plugin: {plugin_path_to_benchmark}")
        
        # Need a temporary PluginManager and ContributionManager
        temp_plugin_manager = PluginManager()
        temp_contribution_manager = ContributionManager(plugin_manager=temp_plugin_manager)

        # Define dummy scenarios for benchmarking
        mock_scenarios = [
            {"name": "Benchmark Scenario 1", "duration": 10},
            {"name": "Benchmark Scenario 2", "duration": 5}
        ]
        
        benchmark_result = await temp_contribution_manager.benchmark_plugin_performance(
            plugin_path_to_benchmark, mock_scenarios
        )
        analysis_results["community_plugin_benchmark"] = benchmark_result
        logger.info(f"Community plugin benchmark results: {analysis_results['community_plugin_benchmark']}")

    # Find snapshot files
    snapshot_files = list(results_path.glob("**/*_enhanced.json"))
    parquet_files = list(results_path.glob("**/*.parquet"))
    
    print(f"📁 Found {len(snapshot_files)} enhanced snapshots, {len(parquet_files)} event files")
    
    if hasattr(args, 'validate_snapshots') and args.validate_snapshots and len(snapshot_files) >= 2:
        print("🔍 Validating snapshot reproducibility...")
        
        # Compare pairs of snapshots
        reproducibility_results = []
        for i in range(len(snapshot_files) - 1):
            snapshot1 = snapshot_files[i]
            snapshot2 = snapshot_files[i + 1]
            
            try:
                validation_result = EventSnapshot.validate_snapshot_reproducibility(
                    snapshot1, snapshot2
                )
                reproducibility_results.append({
                    "snapshot1": snapshot1.name,
                    "snapshot2": snapshot2.name,
                    "is_reproducible": validation_result["is_reproducible"],
                    "issues": validation_result.get("issues", []),
                    "statistics": validation_result.get("statistics", {})
                })
                
                if validation_result["is_reproducible"]:
                    print(f"  ✅ {snapshot1.name} ↔ {snapshot2.name}: Reproducible")
                else:
                    print(f"  ❌ {snapshot1.name} ↔ {snapshot2.name}: Issues found")
                    for issue in validation_result.get("issues", []):
                        print(f"    • {issue}")
                        
            except Exception as e:
                print(f"  ❌ Error comparing {snapshot1.name} ↔ {snapshot2.name}: {e}")
                reproducibility_results.append({
                    "snapshot1": snapshot1.name,
                    "snapshot2": snapshot2.name,
                    "error": str(e)
                })
        
        analysis_results["snapshot_validation"] = reproducibility_results
    
    # Golden master comparison
    if hasattr(args, 'compare_golden') and args.compare_golden:
        print(f"🏆 Comparing against golden master: {args.compare_golden}")
        
        try:
            # Set up golden master tester
            golden_master = GoldenMasterTester(storage_dir=str(results_path.parent))
            
            # Load tolerance config if provided
            tolerance_config = None
            if hasattr(args, 'tolerance_config') and args.tolerance_config:
                if os.path.exists(args.tolerance_config):
                    with open(args.tolerance_config, 'r') as f:
                        tolerance_data = yaml.safe_load(f)
                    tolerance_config = ToleranceConfig(**tolerance_data)
            
            # Find the most recent simulation data
            simulation_data_files = list(results_path.glob("**/simulation_results.json"))
            if simulation_data_files:
                latest_results = simulation_data_files[-1]  # Assume last is latest
                
                with open(latest_results, 'r') as f:
                    simulation_data = json.load(f)
                
                comparison_result = golden_master.compare_against_golden(
                    simulation_data, args.compare_golden, tolerance_config
                )
                
                analysis_results["golden_master_comparison"] = {
                    "baseline_label": args.compare_golden,
                    "is_identical": comparison_result.is_identical,
                    "is_within_tolerance": comparison_result.is_within_tolerance,
                    "critical_differences": len(comparison_result.critical_differences),
                    "total_differences": len(comparison_result.differences),
                    "summary": comparison_result.summary()
                }
                
                print(f"  {comparison_result.summary()}")
                
            else:
                print("  ⚠️ No simulation results found for comparison")
                
        except Exception as e:
            print(f"  ❌ Golden master comparison failed: {e}")
            analysis_results["golden_master_comparison"] = {"error": str(e)}
    
    # Generate comprehensive report
    if hasattr(args, 'generate_report') and args.generate_report:
        print("📋 Generating comprehensive reproducibility report...")
        
        # Collect all reproducibility data
        reproducibility_summary = {
            "total_snapshots": len(snapshot_files),
            "total_event_files": len(parquet_files),
            "reproducible_pairs": 0,
            "failed_pairs": 0,
            "reproducibility_rate": 0.0
        }
        
        if "snapshot_validation" in analysis_results:
            for result in analysis_results["snapshot_validation"]:
                if result.get("is_reproducible", False):
                    reproducibility_summary["reproducible_pairs"] += 1
                else:
                    reproducibility_summary["failed_pairs"] += 1
            
            total_pairs = reproducibility_summary["reproducible_pairs"] + reproducibility_summary["failed_pairs"]
            if total_pairs > 0:
                reproducibility_summary["reproducibility_rate"] = reproducibility_summary["reproducible_pairs"] / total_pairs
        
        analysis_results["reproducibility_summary"] = reproducibility_summary
        
        # Generate recommendations
        recommendations = []
        if reproducibility_summary["reproducibility_rate"] < 1.0:
            recommendations.append("Some simulation runs are not perfectly reproducible")
            recommendations.append("Check for uncontrolled randomness sources")
            recommendations.append("Verify LLM cache consistency")
            recommendations.append("Consider stricter deterministic mode settings")
        else:
            recommendations.append("All tested runs show perfect reproducibility")
            recommendations.append("Current setup maintains scientific validity")
        
        analysis_results["recommendations"] = recommendations
        
        print(f"  📈 Reproducibility rate: {reproducibility_summary['reproducibility_rate']:.1%}")
        print("  💡 Recommendations:")
        for rec in recommendations:
            print(f"    • {rec}")
    
    return analysis_results

def export_analysis_results(results: Dict[str, Any], output_path: str, format_type: str):
    """
    Export analysis results to specified format.
    
    Args:
        results: Analysis results dictionary
        output_path: Output file path
        format_type: Export format (json, yaml, html)
    """
    output_file = Path(output_path)
    
    try:
        if format_type == 'json':
            with open(output_file.with_suffix('.json'), 'w') as f:
                json.dump(results, f, indent=2, separators=(',', ': '))
        
        elif format_type == 'yaml':
            with open(output_file.with_suffix('.yaml'), 'w') as f:
                yaml.dump(results, f, default_flow_style=False, indent=2)
        
        elif format_type == 'html':
            # Generate HTML report
            html_content = generate_html_report(results)
            with open(output_file.with_suffix('.html'), 'w') as f:
                f.write(html_content)
        
        print(f"📄 Analysis results exported to: {output_file.with_suffix('.' + format_type)}")
        
    except Exception as e:
        print(f"❌ Failed to export results: {e}")

def generate_html_report(results: Dict[str, Any]) -> str:
    """
    Generate HTML report from analysis results.
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        HTML content string
    """
    timestamp = results.get("timestamp", "Unknown")
    results_dir = results.get("results_dir", "Unknown")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FBA-Bench Reproducibility Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .success {{ color: green; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
            .stats {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 FBA-Bench Reproducibility Analysis Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Results Directory:</strong> {results_dir}</p>
        </div>
    """
    
    # Add reproducibility summary
    if "reproducibility_summary" in results:
        summary = results["reproducibility_summary"]
        rate = summary["reproducibility_rate"]
        status_class = "success" if rate == 1.0 else "warning" if rate > 0.5 else "error"
        
        html += f"""
        <div class="section">
            <h2>📊 Reproducibility Summary</h2>
            <div class="stats">
                <p><strong>Reproducibility Rate:</strong> <span class="{status_class}">{rate:.1%}</span></p>
                <p><strong>Total Snapshots:</strong> {summary['total_snapshots']}</p>
                <p><strong>Reproducible Pairs:</strong> {summary['reproducible_pairs']}</p>
                <p><strong>Failed Pairs:</strong> {summary['failed_pairs']}</p>
            </div>
        </div>
        """
    
    # Add snapshot validation results
    if "snapshot_validation" in results:
        html += """
        <div class="section">
            <h2>🔍 Snapshot Validation Results</h2>
            <table>
                <tr><th>Snapshot 1</th><th>Snapshot 2</th><th>Status</th><th>Issues</th></tr>
        """
        
        for validation in results["snapshot_validation"]:
            status = "✅ Reproducible" if validation.get("is_reproducible", False) else "❌ Issues Found"
            issues = ", ".join(validation.get("issues", []))
            if not issues:
                issues = "None"
            
            html += f"""
                <tr>
                    <td>{validation.get('snapshot1', 'Unknown')}</td>
                    <td>{validation.get('snapshot2', 'Unknown')}</td>
                    <td>{status}</td>
                    <td>{issues}</td>
                </tr>
            """
        
        html += "</table></div>"
    
    # Add recommendations
    if "recommendations" in results:
        html += """
        <div class="section">
            <h2>💡 Recommendations</h2>
            <ul>
        """
        
        for rec in results["recommendations"]:
            html += f"<li>{rec}</li>"
        
        html += "</ul></div>"
    
    html += """
    </body>
    </html>
    """
    
    return html


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FBA-Bench v3 Experiment CLI - High-Throughput Research Platform"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Execute experiment from configuration file')
    run_parser.add_argument('config_file', nargs='?', default='sweep.yaml',
                           help='Path to sweep.yaml configuration file (optional, defaults to sweep.yaml)')
    run_parser.add_argument('--max-runs', type=int, help='Maximum number of runs to execute (for testing)')
    run_parser.add_argument('--parallel', type=int, default=1, help='Number of local parallel processes')
    
    # Scalability options
    run_parser.add_argument('--distributed', action='store_true',
                           help='Enable distributed simulation mode (requires external message broker setup)')
    run_parser.add_argument('--workers', type=int, default=1,
                           help='Number of distributed worker processes (only applicable with --distributed)')
    run_parser.add_argument('--enable-batching', action='store_true',
                           help='Activate LLM request batching for cost optimization')
    run_parser.add_argument('--fast-forward', action='store_true',
                           help='Enable time compression for idle simulation periods')
    run_parser.add_argument('--cost-limit', type=float, default=None,
                           help='Set maximum LLM API spending limit per simulation run (USD)')

    # Reproducibility options
    run_parser.add_argument('--deterministic', action='store_true',
                           help='Force deterministic mode for bit-perfect reproducibility')
    run_parser.add_argument('--record-responses', action='store_true',
                           help='Enable LLM response recording for later deterministic replay')
    run_parser.add_argument('--validate-reproducibility', action='store_true',
                           help='Run golden master validation after experiments')
    run_parser.add_argument('--cache-file', type=str, default='llm_responses.cache',
                           help='Specify LLM cache file path')
    run_parser.add_argument('--simulation-mode', type=str,
                           choices=['deterministic', 'stochastic', 'research'],
                           help='Set simulation mode (overrides config)')
    run_parser.add_argument('--master-seed', type=int,
                           help='Set master seed for deterministic runs')
    run_parser.add_argument('--golden-master-label', type=str,
                           help='Label for golden master baseline recording')
    run_parser.add_argument('--config-reproducibility', type=str,
                           help='Path to reproducibility configuration file')

    # Scenario options
    run_parser.add_argument('--scenario', type=str, help='Select a specific scenario by name to run (e.g., "international_expansion")')
    run_parser.add_argument('--tier', type=int, choices=[0, 1, 2, 3], help='Run all predefined scenarios for a specific difficulty tier (0-3)')
    run_parser.add_argument('--validate-curriculum', action='store_true',
                           help='Run curriculum validation analysis after a set of scenario runs')
    run_parser.add_argument('--generate-scenario', type=str,
                           help='Generate a procedural scenario based on a base template (e.g., "tier_0_baseline")')
    run_parser.add_argument('--dynamic-scenario-output', type=str, default='generated_scenario.yaml',
                           help='Output file path for generated scenario (with --generate-scenario)')
    run_parser.add_argument('--benchmark-scenarios', action='store_true',
                           help='Test agent across all available scenarios and collect performance metrics')
    run_parser.add_argument('--agents', type=str, nargs='+', help='List of agent models to use for scenario runs or benchmarking (e.g., "ClaudeSonnetBot" "GPT4oMiniBot")')
    run_parser.add_argument('--dynamic-randomization-config', type=str,
                           help='Path to a YAML file containing randomization configuration for dynamic scenario generation')

    # Agent learning and adaptation options
    run_parser.add_argument('--enable-learning', action='store_true',
                           help='Activate agent learning mode across simulation runs')
    run_parser.add_argument('--train-agent', type=str, nargs='?', const='default_agent_id',
                           help='Train a specific agent for X episodes (--train-agent [AGENT_ID] [EPISODES]). If EPISODES is not provided, defaults to 1.')
    run_parser.add_argument('--export-agent', type=str, nargs=2, metavar=('AGENT_ID', 'VERSION'),
                           help='Export a trained agent model with a specific version for evaluation (e.g., --export-agent AgentA v1.0)')
    run_parser.add_argument('--learning-config', type=str,
                           help='Path to a YAML file for advanced learning configuration (overrides CLI flags)')

    # Real-world integration options
    run_parser.add_argument('--real-world-mode', type=str,
                           choices=['simulation', 'sandbox', 'live'], default='simulation',
                           help='Set real-world integration mode (simulation, sandbox, or live)')
    run_parser.add_argument('--validate-integration', action='store_true',
                           help='Run comprehensive simulation-to-real integration validation tests')

    # Community and Extensibility options
    run_parser.add_argument('--load-plugin', type=str, nargs='+',
                           help='Load one or more community plugins from specified file paths (e.g., --load-plugin path/to/plugin1.py)')
    run_parser.add_argument('--benchmark-community-plugin', type=str,
                           help='Benchmark a specific community plugin (provide plugin file path)')
    run_parser.add_argument('--plugin-dir', type=str, default='plugins',
                           help='Directory to search for community plugins (defaults to ./plugins)')
    
    # Analyze command with reproducibility features
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results and validate reproducibility')
    analyze_parser.add_argument('results_dir', help='Path to experiment results directory')
    analyze_parser.add_argument('--compare-golden', type=str,
                               help='Compare results against golden master baseline')
    analyze_parser.add_argument('--validate-snapshots', action='store_true',
                               help='Validate event snapshot reproducibility')
    analyze_parser.add_argument('--generate-report', action='store_true',
                               help='Generate comprehensive reproducibility report')
    analyze_parser.add_argument('--tolerance-config', type=str,
                               help='Path to tolerance configuration for comparisons')
    analyze_parser.add_argument('--export-format', type=str,
                               choices=['json', 'yaml', 'html'], default='json',
                               help='Format for exported analysis results')
    
    # Observability options for analyze command
    analyze_parser.add_argument('--analyze-traces', action='store_true',
                               help='Run post-simulation trace analysis using TraceAnalyzer')
    analyze_parser.add_argument('--generate-insights', action='store_true',
                               help='Create automated insight reports from trace analysis')
    analyze_parser.add_argument('--monitor-performance', action='store_true',
                               help='Enable real-time performance monitoring during analysis')
    analyze_parser.add_argument('--error-analysis', action='store_true',
                               help='Analyze agent error patterns using AgentErrorHandler')
    analyze_parser.add_argument('--export-traces', type=str, choices=['json', 'zip'],
                               help='Export raw traces in various formats (json, zip)')
    
    # Golden Master command
    golden_parser = subparsers.add_parser('golden', help='Golden master operations')
    golden_subparsers = golden_parser.add_subparsers(dest='golden_command', help='Golden master sub-commands')
    
    # Record golden master baseline
    record_parser = golden_subparsers.add_parser('record', help='Record golden master baseline')
    record_parser.add_argument('results_dir', help='Path to experiment results directory')
    record_parser.add_argument('--label', type=str, required=True,
                               help='Label for the golden master baseline')
    record_parser.add_argument('--config', type=str,
                               help='Path to tolerance configuration file')
    record_parser.add_argument('--description', type=str,
                               help='Description of the baseline')
    
    # Compare against golden master
    compare_parser = golden_subparsers.add_parser('compare', help='Compare results against golden master')
    compare_parser.add_argument('results_dir', help='Path to experiment results directory')
    compare_parser.add_argument('--baseline', type=str, required=True,
                                help='Golden master baseline label to compare against')
    compare_parser.add_argument('--config', type=str,
                                help='Path to tolerance configuration file')
    compare_parser.add_argument('--output', type=str,
                                help='Output file for comparison report')
    
    # List golden master baselines
    list_parser = golden_subparsers.add_parser('list', help='List available golden master baselines')
    list_parser.add_argument('--storage-path', type=str,
                             help='Path to golden master storage directory')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        # Conditional validation for config_file for sweep vs. single scenario
        if not args.scenario and not args.tier and not args.generate_scenario and not os.path.exists(args.config_file):
            print(f"❌ Configuration file not found: {args.config_file}. Required for sweep experiments.")
            sys.exit(1)
            
        if (args.scenario or args.tier or args.generate_scenario) and args.config_file != 'sweep.yaml':
             logger.warning("Using --scenario, --tier, or --generate-scenario. 'config_file' (sweep.yaml) will be ignored for experiment definition.")
        
        repro_config = setup_reproducibility_from_args(args)
        
        # Instantiate managers for scenario operations
        scenario_engine = ScenarioEngine()
        curriculum_validator = CurriculumValidator()
        scenario_config_manager = ScenarioConfigManager()
        dynamic_generator = DynamicScenarioGenerator()
        
        if args.generate_scenario:
            if not args.dynamic_randomization_config:
                print("❌ --dynamic-randomization-config is required when using --generate-scenario.")
                sys.exit(1)
            try:
                with open(args.dynamic_randomization_config, 'r') as f:
                    dynamic_rand_config = yaml.safe_load(f)
                
                logger.info(f"Generating dynamic scenario based on template: {args.generate_scenario}")
                generated_scenario = scenario_config_manager.generate_dynamic_scenario(
                    args.generate_scenario, dynamic_rand_config, target_tier=args.tier
                )
                generated_scenario.to_yaml(args.dynamic_scenario_output)
                print(f"✅ Dynamic scenario generated and saved to: {args.dynamic_scenario_output}")
            except Exception as e:
                print(f"❌ Failed to generate dynamic scenario: {e}")
                sys.exit(1)
            sys.exit(0) # Exit after generation
            
        if args.scenario or args.tier or args.benchmark_scenarios:
            # Handle scenario-based runs (single scenario or all scenarios in a tier)
            agent_models_to_test = {}
            if args.agents:
                # Placeholder for actual agent loading using agent_runners
                # For now, just a dummy object
                from baseline_bots.bot_factory import BotFactory
                for agent_name in args.agents:
                    try:
                        agent_models_to_test[agent_name] = BotFactory.create_bot(agent_name)
                    except Exception as e:
                        print(f"❌ Could not load agent '{agent_name}': {e}. Skipping this agent.")
            if not agent_models_to_test:
                print("❌ No agent models specified or loaded for scenario run. Using a DummyAgent for testing.")
                # Fallback to a default dummy agent if none provided for scenario/tier runs
                class DummyAgent:
                    def name(self): return "DummyAgent"
                agent_models_to_test = {"DummyAgent": DummyAgent()}
                
            scenarios_to_run: List[str] = []
            if args.scenario:
                scenarios_to_run.append(args.scenario)
            elif args.tier is not None:
                tier_configs = scenario_config_manager.get_scenarios_by_tier(args.tier)
                if not tier_configs:
                    print(f"❌ No scenarios found for tier {args.tier}.")
                    sys.exit(1)
                scenarios_to_run.extend([cfg.config_data['scenario_name'] for cfg in tier_configs])
            elif args.benchmark_scenarios:
                 all_available_scenarios_info = scenario_config_manager.available_scenarios
                 scenarios_to_run.extend(all_available_scenarios_info.keys()) # Get all scenario names

            all_scenario_results = []
            for s_name in scenarios_to_run:
                print(f"\n--- Running scenario: {s_name} ---")
                scenario_filepath = scenario_config_manager.get_scenario_path(s_name)
                if not scenario_filepath:
                    print(f"❌ Scenario '{s_name}' not found. Skipping.")
                    continue
                    
                for agent_name, agent_instance in agent_models_to_test.items():
                    print(f"  Agent: {agent_name}")
                    try:
                        run_results = await scenario_engine.run_simulation(scenario_filepath, {agent_name: agent_instance})
                        all_scenario_results.append({
                            **run_results,
                            'cli_run_type': 'scenario_run',
                            'agent_name': agent_name
                        })
                        curriculum_validator.benchmark_agent_performance(
                            agent_name,
                            run_results.get('tier'),
                            run_results.get('scenario_name'),
                            run_results
                        )
                    except Exception as e:
                        print(f"❌ Error running simulation for scenario '{s_name}' with agent '{agent_name}': {e}")
                        all_scenario_results.append({
                            'scenario_name': s_name,
                            'agent_name': agent_name,
                            'success_status': 'error',
                            'error_message': str(e),
                            'cli_run_type': 'scenario_run'
                        })
            
            # Save all scenario run results
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = Path(f"scenario_results/{'benchmark' if args.benchmark_scenarios else 'scenario'}_run_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "scenario_summary.json", 'w') as f:
                json.dump(all_scenario_results, f, indent=2)
            print(f"\n✅ All scenario results saved to {output_dir}")

            if args.validate_curriculum:
                print("\n--- Running Curriculum Validation ---")
                curriculum_report = curriculum_validator.generate_curriculum_report()
                report_file = output_dir / "curriculum_validation_report.json"
                with open(report_file, 'w') as f:
                    json.dump(curriculum_report, f, indent=2)
                print(f"✅ Curriculum validation report saved to: {report_file}")
                curriculum_validator.recommend_tier_adjustments(curriculum_report.get('tier_progression_validation', {}))
                
            sys.exit(0) # Exit after scenario/tier/benchmark run
            
        else:
            # Existing sweep logic
            manager = ExperimentManager(args.config_file, reproducibility_config=repro_config)
            # Pass args to manager for learning/integration/plugin flags
            parallel_workers = getattr(args, 'parallel', 1)
            manager = ExperimentManager(args.config_file, reproducibility_config=repro_config)
            # Pass args to manager for learning/integration/plugin flags
            parallel_workers = getattr(args, 'parallel', 1)
            await manager.run_experiment(
                max_runs=args.max_runs,
                parallel_workers=parallel_workers,
                args=args
            )

    elif args.command == 'analyze':
        # Handles '--validate-integration', '--benchmark-community-plugin' directly in analyze_experiment_reproducibility
        try:
            # Import observability components dynamically to avoid circular dependencies and for CLI entry
            from observability.trace_analyzer import TraceAnalyzer
            from observability.alert_system import ObservabilityAlertSystem
            from tools.documentation_generator import ToolDocGenerator # For error analysis documentation
            from tools.error_handler import AgentErrorHandler
            from instrumentation.simulation_tracer import SimulationTracer # For trace data access
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
            from opentelemetry import trace # Re-import trace here for local scope if needed

            # Setup minimal tracer if needed for direct trace operations within CLI analysis
            # In a full system, OTel setup would be global
            provider = TracerProvider()
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            otel_tracer = trace.get_tracer(__name__)
            simulation_tracer_instance = SimulationTracer(otel_tracer) # Used to access buffer, if data were collected live

            trace_analyzer = TraceAnalyzer()
            alert_system = ObservabilityAlertSystem()
            error_handler = AgentErrorHandler()
            doc_generator = ToolDocGenerator() # For generating error documentation

            # Run reproducibility analysis (existing functionality)
            analysis_results = await analyze_experiment_reproducibility(args.results_dir, args)

            # New Observability Analysis Features
            collected_traces = []
            # For CLI analysis, we expect traces to be saved to disk, not from live memory.
            # Look for exported trace files (e.g., from SimulationTracer's _record_event_data if exported)
            trace_files = list(Path(args.results_dir).glob("**/*_traces.json")) # Assuming traces are exported as JSON files
            if not trace_files:
                logger.warning(f"No trace files found in {args.results_dir} matching pattern '*_traces.json'. Observability analysis will be limited.")
            for t_file in trace_files:
                try:
                    with open(t_file, 'r') as f:
                        collected_traces.extend(json.load(f))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to load trace file {t_file}: {e}")

            if args.analyze_traces and collected_traces:
                logger.info("Running trace analysis...")
                # Placeholder for determining failure_point. In a real scenario, this would come from logs.
                # For basic CLI analysis, we can infer from last error or just analyze the whole trace.
                # A robust solution might iterate over all runs and their trace files.
                # For this task, we'll assume a single, stitched trace for simplicity.
                failure_point = {"event_type": "end_of_sim", "timestamp": time.time(), "reason": "CLI initiated analysis"}
                failure_summary = trace_analyzer.analyze_simulation_failure(collected_traces, failure_point)
                analysis_results["trace_failure_analysis"] = failure_summary
                
                # Behavioral patterns require a specific structure for agent traces/decisions
                # We'll need to parse 'llm.reasoning' and 'tool_call.error_full_details' from generic traces
                agent_decisions_from_traces = []
                for event in collected_traces:
                    if event.get("span_name", "").startswith("agent_turn_"):
                        # Attempt to reconstruct simplified agent decision from spans
                        agent_id = event["attributes"].get("agent.id")
                        actions_in_turn = []
                        # Find child spans for 'think' and 'tool_call'
                        # This is a simplified stand-in. Proper span-linking or direct decision logging is better.
                        if "think.action_type" in event["attributes"]: # Simplified from agent_tracer change
                            actions_in_turn.append({"action": event["attributes"]["think.action_type"], "status": "success"}) # Assuming success for now
                        if "tool_call.name" in event["attributes"]:
                            status = "failure" if event["attributes"].get("tool_call.success") is False else "success"
                            error_details = json.loads(event["attributes"]["tool_call.error_full_details"]) if "tool_call.error_full_details" in event["attributes"] else {}
                            actions_in_turn.append({"action": event["attributes"]["tool_call.name"], "status": status, "error_details": error_details})
                        if agent_id and actions_in_turn:
                            agent_decisions_from_traces.append({"agent_id": agent_id, "decisions": actions_in_turn})

                agent_behavior_patterns = trace_analyzer.detect_behavioral_patterns(agent_decisions_from_traces)
                analysis_results["agent_behavior_patterns"] = agent_behavior_patterns

                # Extract timing data for performance bottlenecks
                timing_data = [{"operation": t.get("span_name"), "duration_ms": t.get("attributes", {}).get("span.duration_ms", 0), "timestamp": t.get("timestamp")} for t in collected_traces if "span.duration_ms" in t.get("attributes", {})]
                performance_bottlenecks = trace_analyzer.identify_performance_bottlenecks(timing_data)
                analysis_results["performance_bottlenecks"] = performance_bottlenecks

                logger.info("Trace analysis completed.")

            if args.generate_insights and (args.analyze_traces or "trace_failure_analysis" in analysis_results):
                logger.info("Generating automated insight report...")
                insight_report = trace_analyzer.generate_insight_report(analysis_results)
                report_output_path = Path(args.results_dir) / "observability_insight_report.md"
                with open(report_output_path, 'w') as f:
                    f.write(insight_report)
                logger.info(f"Insight report saved to: {report_output_path}")

            if args.error_analysis:
                logger.info("Analyzing agent error patterns and generating feedback...")
                # Get common errors, assuming behavioral patterns were detected
                captured_errors = analysis_results.get("agent_behavior_patterns", {}).get("common_errors", {})
                if captured_errors:
                    logger.info(f"Detected common errors: {captured_errors}")
                    error_recommendations = doc_generator.document_error_scenarios(captured_errors)
                    analysis_results["error_analysis_recommendations"] = error_recommendations
                    logger.info(f"Error analysis recommendations: {json.dumps(error_recommendations, indent=2)}")
                else:
                    logger.info("No common agent error patterns detected from available traces for error analysis.")
                
            if args.export_traces and collected_traces:
                output_trace_path = Path(args.results_dir) / f"exported_traces.{args.export_traces}"
                if args.export_traces == 'json':
                    with open(output_trace_path, 'w') as f:
                        json.dump(collected_traces, f, indent=2)
                    logger.info(f"Raw traces exported to {output_trace_path}")
                elif args.export_traces == 'zip':
                    # Export multiple trace files into a single zip archive
                    with zipfile.ZipFile(output_trace_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for idx, trace_obj in enumerate(collected_traces):
                            trace_filename = f"trace_event_{idx}.json"
                            zipf.writestr(trace_filename, json.dumps(trace_obj, indent=2))
                    logger.info(f"Raw traces exported to {output_trace_path}")


            # Export results (original functionality)
            if hasattr(args, 'export_format') and args.export_format:
                output_path = Path(args.results_dir) / "reproducibility_analysis"
                export_analysis_results(analysis_results, str(output_path), args.export_format)
            
            logger.info("✅ Analysis completed successfully")
            
        except Exception as e:
            logger.exception(f"❌ Analysis failed: {e}")
            sys.exit(1)
            
    elif args.command == 'golden':
        try:
            if args.golden_command == 'record':
                # Record golden master baseline
                tolerance_config = None
                if args.config:
                    tolerance_config = ToleranceConfig.from_file(args.config)
                
                tester = GoldenMasterTester(tolerance_config=tolerance_config)
                success = tester.record_baseline(
                    args.results_dir,
                    args.label,
                    description=getattr(args, 'description', None)
                )
                
                if success:
                    print(f"✅ Golden master baseline '{args.label}' recorded successfully")
                else:
                    print(f"❌ Failed to record golden master baseline")
                    sys.exit(1)
                    
            elif args.golden_command == 'compare':
                # Compare against golden master
                tolerance_config = None
                if args.config:
                    tolerance_config = ToleranceConfig.from_file(args.config)
                
                tester = GoldenMasterTester(tolerance_config=tolerance_config)
                comparison_result = tester.compare_with_baseline(args.results_dir, args.baseline)
                
                # Output results
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(asdict(comparison_result), f, indent=2)
                    print(f"📄 Comparison report saved to: {args.output}")
                else:
                    print(f"📊 Comparison Results:")
                    print(f"   Overall Match: {'✅ PASS' if comparison_result.overall_match else '❌ FAIL'}")
                    print(f"   Confidence Score: {comparison_result.confidence_score:.3f}")
                    print(f"   Differences: {len(comparison_result.differences)}")
                    
            elif args.golden_command == 'list':
                # List available baselines
                storage_path = getattr(args, 'storage_path', './golden_masters')
                tester = GoldenMasterTester(storage_path=storage_path)
                baselines = tester.list_baselines()
                
                if baselines:
                    print("📋 Available Golden Master Baselines:")
                    for baseline in baselines:
                        print(f"   • {baseline['label']} (recorded: {baseline['timestamp']})")
                        if 'description' in baseline and baseline['description']:
                            print(f"     Description: {baseline['description']}")
                else:
                    print("📋 No golden master baselines found")
                        
            else:
                golden_parser.print_help()
                    
        except Exception as e:
            print(f"❌ Golden master operation failed: {e}")
            sys.exit(1)
            
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("CLI operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during CLI execution: {e}")
        sys.exit(1)