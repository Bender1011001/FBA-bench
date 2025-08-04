"""
Core benchmarking engine.

This module provides the main BenchmarkEngine class that orchestrates the entire
benchmarking process, including agent lifecycle management, metrics collection,
and reproducible execution.
"""

import asyncio
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path

from .config import BenchmarkConfig, AgentConfig, ScenarioConfig
from .results import BenchmarkResult, ScenarioResult, AgentRunResult, MetricResult

# Import registry components
from ..agents.registry import agent_registry
from ..scenarios.registry import scenario_registry
from ..metrics.registry import metrics_registry

# Import existing FBA-Bench components
from agent_runners.agent_manager import AgentManager
from agent_runners.base_runner import SimulationState
from scenarios.base_scenario import BaseScenario
from metrics.metric_suite import MetricSuite
from event_bus import EventBus
from services.world_store import WorldStore
from constraints.budget_enforcer import BudgetEnforcer
from constraints.agent_gateway import AgentGateway
from metrics.trust_metrics import TrustMetrics

logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """
    Core benchmarking engine that orchestrates the entire benchmarking process.
    
    The BenchmarkEngine is responsible for:
    - Loading and validating benchmark configurations
    - Managing agent lifecycle (initialization, execution, cleanup)
    - Collecting and aggregating metrics across multiple runs
    - Ensuring reproducible execution with deterministic random seeds
    - Parallel execution capabilities for scalability
    """
    
    def __init__(self, config: Union[BenchmarkConfig, str, Path]):
        """
        Initialize the benchmark engine.
        
        Args:
            config: Benchmark configuration (either as BenchmarkConfig object or path to config file)
        """
        if isinstance(config, (str, Path)):
            self.config = BenchmarkConfig.from_file(config)
        else:
            self.config = config
        
        # Validate configuration
        validation_errors = self.config.validate()
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")
        
        # Initialize components
        self.event_bus = EventBus()
        self.world_store = WorldStore()
        self.budget_enforcer = BudgetEnforcer()
        self.trust_metrics = TrustMetrics()
        self.agent_gateway = AgentGateway()
        
        # Initialize agent manager
        self.agent_manager = AgentManager(
            event_bus=self.event_bus,
            world_store=self.world_store,
            budget_enforcer=self.budget_enforcer,
            trust_metrics=self.trust_metrics,
            agent_gateway=self.agent_gateway
        )
        
        # Initialize metric suite
        self.metric_suite = MetricSuite(
            tier=self.config.environment,
            financial_audit_service=None,  # Will be set during initialization
            sales_service=None,  # Will be set during initialization
            trust_score_service=self.trust_metrics
        )
        
        # Scenario registry
        self.scenarios: Dict[str, BaseScenario] = {}
        
        # Results storage
        self.current_result: Optional[BenchmarkResult] = None
        
        # Set random seed for reproducibility
        if self.config.execution.random_seed is not None:
            import random
            import numpy as np
            random.seed(self.config.execution.random_seed)
            np.random.seed(self.config.execution.random_seed)
        
        logger.info(f"BenchmarkEngine initialized for '{self.config.name}'")
    
    def register_scenario(self, scenario: BaseScenario) -> None:
        """
        Register a scenario with the benchmark engine.
        
        Args:
            scenario: Scenario instance to register
        """
        self.scenarios[scenario.name] = scenario
        logger.info(f"Registered scenario: {scenario.name}")
    
    def get_config_hash(self) -> str:
        """
        Generate a hash of the configuration for reproducibility tracking.
        
        Returns:
            SHA256 hash of the configuration
        """
        config_dict = self.config.to_dict()
        config_str = str(sorted(config_dict.items()))
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    async def initialize(self) -> None:
        """
        Initialize the benchmark engine and all components.
        
        This method sets up the event bus, world store, agents, and scenarios.
        """
        logger.info("Initializing benchmark engine...")
        
        # Start event bus
        await self.event_bus.start()
        
        # Initialize world store
        await self.world_store.initialize()
        
        # Register agents
        for agent_config in self.config.agents:
            if agent_config.enabled:
                self.agent_manager.register_agent(
                    agent_id=agent_config.agent_id,
                    framework=agent_config.framework,
                    config=agent_config.config
                )
        
        # Start agent manager
        await self.agent_manager.start()
        
        # Subscribe metric suite to events
        await self.metric_suite.subscribe_to_events(self.event_bus)
        
        logger.info("Benchmark engine initialized successfully")
    
    async def run_benchmark(self) -> BenchmarkResult:
        """
        Run the complete benchmark with all configured scenarios and agents.
        
        Returns:
            Complete benchmark result
        """
        start_time = datetime.now()
        logger.info(f"Starting benchmark: {self.config.name}")
        
        # Initialize if not already done
        if not self.agent_manager.bot_factory:
            await self.initialize()
        
        # Create benchmark result
        self.current_result = BenchmarkResult(
            benchmark_name=self.config.name,
            config_hash=self.get_config_hash(),
            start_time=start_time,
            end_time=start_time,  # Will be updated at the end
            duration_seconds=0.0,  # Will be updated at the end
            metadata={
                'config': self.config.to_dict(),
                'environment': self.config.environment,
                'num_agents': len([a for a in self.config.agents if a.enabled]),
                'num_scenarios': len([s for s in self.config.scenarios if s.enabled])
            }
        )
        
        # Run scenarios
        for scenario_config in self.config.scenarios:
            if scenario_config.enabled:
                scenario_result = await self.run_scenario(scenario_config)
                self.current_result.scenario_results.append(scenario_result)
        
        # Finalize benchmark result
        end_time = datetime.now()
        self.current_result.end_time = end_time
        self.current_result.duration_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"Benchmark completed in {self.current_result.duration_seconds:.2f} seconds")
        
        return self.current_result
    
    async def run_scenario(self, scenario_config: ScenarioConfig) -> ScenarioResult:
        """
        Run a single scenario with all configured agents.
        
        Args:
            scenario_config: Configuration for the scenario to run
            
        Returns:
            Scenario result
        """
        start_time = datetime.now()
        logger.info(f"Running scenario: {scenario_config.name}")
        
        # Get scenario instance
        if scenario_config.name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_config.name}' not registered")
        
        scenario = self.scenarios[scenario_config.name]
        
        # Initialize scenario
        await scenario.initialize(scenario_config.parameters)
        
        # Create scenario result
        scenario_result = ScenarioResult(
            scenario_name=scenario_config.name,
            start_time=start_time,
            end_time=start_time,  # Will be updated at the end
            duration_seconds=0.0  # Will be updated at the end
        )
        
        # Run multiple iterations
        for run_number in range(1, self.config.execution.num_runs + 1):
            logger.info(f"Running scenario '{scenario_config.name}' - iteration {run_number}/{self.config.execution.num_runs}")
            
            # Reset world state for each run
            await self.world_store.reset()
            
            # Run scenario iteration
            iteration_results = await self.run_scenario_iteration(
                scenario, scenario_config, run_number
            )
            
            # Add to scenario results
            scenario_result.agent_results.extend(iteration_results)
        
        # Finalize scenario result
        end_time = datetime.now()
        scenario_result.end_time = end_time
        scenario_result.duration_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"Scenario '{scenario_config.name}' completed in {scenario_result.duration_seconds:.2f} seconds")
        
        return scenario_result
    
    async def run_scenario_iteration(
        self, 
        scenario: BaseScenario, 
        scenario_config: ScenarioConfig, 
        run_number: int
    ) -> List[AgentRunResult]:
        """
        Run a single iteration of a scenario with all agents.
        
        Args:
            scenario: Scenario instance
            scenario_config: Scenario configuration
            run_number: Current run number
            
        Returns:
            List of agent run results
        """
        start_time = datetime.now()
        agent_results = []
        
        # Get enabled agents
        enabled_agents = [a for a in self.config.agents if a.enabled]
        
        # Run agents in parallel if configured
        if self.config.execution.parallel_execution and len(enabled_agents) > 1:
            agent_results = await self._run_agents_parallel(
                scenario, scenario_config, run_number, enabled_agents
            )
        else:
            agent_results = await self._run_agents_sequential(
                scenario, scenario_config, run_number, enabled_agents
            )
        
        return agent_results
    
    async def _run_agents_parallel(
        self,
        scenario: BaseScenario,
        scenario_config: ScenarioConfig,
        run_number: int,
        agents: List[AgentConfig]
    ) -> List[AgentRunResult]:
        """
        Run agents in parallel for a scenario iteration.
        
        Args:
            scenario: Scenario instance
            scenario_config: Scenario configuration
            run_number: Current run number
            agents: List of agent configurations
            
        Returns:
            List of agent run results
        """
        async def run_single_agent(agent_config: AgentConfig) -> AgentRunResult:
            return await self._run_single_agent(
                scenario, scenario_config, run_number, agent_config
            )
        
        # Create tasks for all agents
        tasks = [run_single_agent(agent) for agent in agents]
        
        # Execute tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.execution.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Scenario iteration {run_number} timed out")
            # Create timeout results for all agents
            return [
                AgentRunResult(
                    agent_id=agent.agent_id,
                    scenario_name=scenario_config.name,
                    run_number=run_number,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0.0,
                    errors=["Execution timed out"],
                    success=False
                )
                for agent in agents
            ]
        
        # Process results
        agent_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                agent_results.append(AgentRunResult(
                    agent_id=agents[i].agent_id,
                    scenario_name=scenario_config.name,
                    run_number=run_number,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0.0,
                    errors=[str(result)],
                    success=False
                ))
            else:
                agent_results.append(result)
        
        return agent_results
    
    async def _run_agents_sequential(
        self,
        scenario: BaseScenario,
        scenario_config: ScenarioConfig,
        run_number: int,
        agents: List[AgentConfig]
    ) -> List[AgentRunResult]:
        """
        Run agents sequentially for a scenario iteration.
        
        Args:
            scenario: Scenario instance
            scenario_config: Scenario configuration
            run_number: Current run number
            agents: List of agent configurations
            
        Returns:
            List of agent run results
        """
        agent_results = []
        
        for agent_config in agents:
            try:
                result = await self._run_single_agent(
                    scenario, scenario_config, run_number, agent_config
                )
                agent_results.append(result)
            except Exception as e:
                logger.error(f"Error running agent {agent_config.agent_id}: {e}")
                agent_results.append(AgentRunResult(
                    agent_id=agent_config.agent_id,
                    scenario_name=scenario_config.name,
                    run_number=run_number,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0.0,
                    errors=[str(e)],
                    success=False
                ))
        
        return agent_results
    
    async def _run_single_agent(
        self,
        scenario: BaseScenario,
        scenario_config: ScenarioConfig,
        run_number: int,
        agent_config: AgentConfig
    ) -> AgentRunResult:
        """
        Run a single agent for a scenario iteration.
        
        Args:
            scenario: Scenario instance
            scenario_config: Scenario configuration
            run_number: Current run number
            agent_config: Agent configuration
            
        Returns:
            Agent run result
        """
        start_time = datetime.now()
        
        # Create agent run result
        agent_result = AgentRunResult(
            agent_id=agent_config.agent_id,
            scenario_name=scenario_config.name,
            run_number=run_number,
            start_time=start_time,
            end_time=start_time,  # Will be updated at the end
            duration_seconds=0.0,  # Will be updated at the end
            success=True
        )
        
        try:
            # Setup scenario for agent
            await scenario.setup_for_agent(agent_config.agent_id)
            
            # Run scenario for the specified duration
            for tick in range(scenario_config.duration_ticks):
                # Create simulation state
                simulation_state = SimulationState(
                    tick=tick,
                    simulation_time=start_time,
                    products=self.world_store.get_all_product_states().values(),
                    recent_events=self.event_bus.get_recorded_events()[-10:],  # Last 10 events
                    financial_position=self.world_store.get_financial_state(agent_config.agent_id),
                    market_conditions=self.world_store.get_market_conditions(),
                    agent_state=self.world_store.get_agent_state(agent_config.agent_id)
                )
                
                # Run agent decision cycle
                await self.agent_manager.run_decision_cycle()
                
                # Update scenario
                await scenario.update_tick(tick, simulation_state)
                
                # Collect metrics at specified intervals
                if tick % self.config.metrics.collection_interval == 0:
                    metrics = await self._collect_metrics(agent_config.agent_id, tick)
                    agent_result.metrics.extend(metrics)
            
            # Collect final metrics
            final_metrics = await self._collect_metrics(agent_config.agent_id, scenario_config.duration_ticks)
            agent_result.metrics.extend(final_metrics)
            
        except Exception as e:
            logger.error(f"Error in agent {agent_config.agent_id} run: {e}")
            agent_result.errors.append(str(e))
            agent_result.success = False
        
        # Finalize agent result
        end_time = datetime.now()
        agent_result.end_time = end_time
        agent_result.duration_seconds = (end_time - start_time).total_seconds()
        
        return agent_result
    
    async def _collect_metrics(self, agent_id: str, tick: int) -> List[MetricResult]:
        """
        Collect metrics for a specific agent at a given tick.
        
        Args:
            agent_id: Agent ID
            tick: Current tick
            
        Returns:
            List of metric results
        """
        metrics = []
        
        # Calculate KPIs from metric suite
        kpis = self.metric_suite.calculate_kpis(tick)
        
        # Convert KPIs to metric results
        for metric_name, metric_data in kpis.get('breakdown', {}).items():
            if isinstance(metric_data, dict) and 'score' in metric_data:
                metrics.append(MetricResult(
                    name=f"{metric_name}_score",
                    value=metric_data['score'],
                    unit="score",
                    timestamp=datetime.now(),
                    metadata={'tick': tick, 'agent_id': agent_id}
                ))
        
        return metrics
    
    async def cleanup(self) -> None:
        """
        Clean up resources after benchmark completion.
        """
        logger.info("Cleaning up benchmark engine...")
        
        # Stop agent manager
        if self.agent_manager:
            await self.agent_manager.stop()
        
        # Stop event bus
        if self.event_bus:
            await self.event_bus.stop()
        
        # Clean up world store
        if self.world_store:
            await self.world_store.cleanup()
        
        logger.info("Benchmark engine cleaned up")
    
    async def save_results(self, output_dir: Union[str, Path] = None) -> Path:
        """
        Save benchmark results to files.
        
        Args:
            output_dir: Output directory (uses config if not specified)
            
        Returns:
            Path to the saved results file
        """
        if output_dir is None:
            output_dir = self.config.execution.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self.config.name}_{timestamp}.json"
        output_path = output_dir / filename
        
        # Save results
        if self.current_result:
            self.current_result.save_to_file(output_path)
            logger.info(f"Results saved to: {output_path}")
        else:
            logger.warning("No results to save")
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the benchmark results.
        
        Returns:
            Summary statistics
        """
        if self.current_result is None:
            return {"error": "No benchmark results available"}
        
        return self.current_result.calculate_summary_statistics()


# Global benchmark engine instance (initialized lazily)
benchmark_engine = None

def get_benchmark_engine():
    """
    Get the global benchmark engine instance.
    
    Returns:
        BenchmarkEngine: The global benchmark engine instance
    """
    global benchmark_engine
    if benchmark_engine is None:
        from benchmarking.config import config_manager
        from benchmarking.integration import integration_manager
        benchmark_engine = BenchmarkEngine(config_manager, integration_manager)
    return benchmark_engine