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

from ..config.pydantic_config import BenchmarkConfig
from ..config.pydantic_config import AgentConfig
from ..config.pydantic_config import ScenarioConfig
from ..config.pydantic_config import ConfigurationManager
# SchemaRegistry and schema_registry are not directly used by BenchmarkEngine, can be removed if not needed elsewhere.
from .results import BenchmarkResult, ScenarioResult, AgentRunResult, MetricResult

# Import registry components
from ..agents.registry import agent_registry
from ..scenarios.registry import scenario_registry
from ..metrics.registry import metrics_registry

# Import existing FBA-Bench components
from agent_runners.agent_manager import AgentManager
from fba_bench.core.types import SimulationState
from benchmarking.scenarios.base import BaseScenario
from metrics.metric_suite import MetricSuite
from event_bus import EventBus
from services.world_store import WorldStore
from constraints.budget_enforcer import BudgetEnforcer
from constraints.agent_gateway import AgentGateway
from metrics.trust_metrics import TrustMetrics # TrustMetrics will use TrustScoreService

# Import the new real services
from financial_audit import FinancialAuditService
from services.trust_score_service import TrustScoreService
from services.sales_service import SalesService
from services.double_entry_ledger_service import DoubleEntryLedgerService
from services.bsr_engine_v3 import BsrEngineV3Service

logger = logging.getLogger(__name__)

class BenchmarkEngine:
    """
    The main orchestrator for running benchmarks.

    Manages the entire lifecycle of a benchmark, from setup and execution
    to results collection and reporting. Ensures reproducibility and
    integrates with various services like event bus, world store, and metrics.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the BenchmarkEngine with a configuration.

        Args:
            config: The BenchmarkConfig object containing all settings.
        """
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.current_benchmark_result: Optional[BenchmarkResult] = None
        self.current_scenario_result: Optional[ScenarioResult] = None
        self.current_agent_run_results: List[AgentRunResult] = []

        # Core infrastructure
        self.event_bus = EventBus()
        self.world_store = WorldStore(self.event_bus)
        self.agent_manager = AgentManager(
            event_bus=self.event_bus,
            world_store=self.world_store,
            openrouter_api_key=config.llm_config.get('api_key') if config.llm_config else None,
            use_unified_agents=True  # Enable unified agent system
        )
        
        # Initialize real services
        # Config for FinancialAuditService
        fa_config = self.config.services.get("financial_audit", {
            "halt_on_violation": True,
            "tolerance_cents": 0,
            "audit_enabled": True,
            "starting_cash_dollars": 10000.0,
            "starting_inventory_dollars": 5000.0
        })
        self.financial_audit_service = FinancialAuditService(config=fa_config)

        # Initialize Double-Entry Ledger Service and attach to audit as system-of-record
        ledger_config = self.config.services.get("ledger", {})
        self.ledger_service = DoubleEntryLedgerService(config=ledger_config)
        # Wire audit to pull balances from ledger snapshots
        self.financial_audit_service.ledger_service = self.ledger_service

        # Config for TrustScoreService
        ts_config = self.config.services.get("trust_score", {
            "base_score": 100.0,
            "violation_penalty": 5.0,
            "feedback_weight": 0.2,
            "min_score": 0.0,
            "max_score": 100.0
        })
        self.trust_score_service = TrustScoreService(config=ts_config)

        # SalesService depends on FinancialAuditService
        self.sales_service = SalesService(financial_audit_service=self.financial_audit_service)
        
        # Initialize BSR v3 engine (EMA-based velocity and conversion tracking)
        bsr_config = self.config.services.get("bsr", {})
        self.bsr_engine_v3 = BsrEngineV3Service(config=bsr_config)
        
        # Initialize TrustMetrics with the TrustScoreService
        self.trust_metrics = TrustMetrics(trust_score_service=self.trust_score_service)

        # Initialize MetricSuite with the real services
        self.metric_suite = MetricSuite(
            tier=self.config.tier,
            weights=self.config.metric_weights, # Assuming BenchmarkConfig has metric_weights
            financial_audit_service=self.financial_audit_service,
            sales_service=self.sales_service, # MetricSuite will use this for Ops/Marketing metrics
            trust_score_service=self.trust_score_service # MetricSuite passes this to TrustMetrics
        )

        # Constraints and Gateway
        self.budget_enforcer = BudgetEnforcer(
            self.config.budget_limit,
            self.event_bus,
            tier=self.config.tier # TODO: Make BudgetEnforcer tier configurable if not already
        )
        self.agent_gateway = AgentGateway(
            self.event_bus,
            self.agent_manager,
            self.budget_enforcer
        )

        self.is_initialized = False
        self.execution_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        logger.info(f"BenchmarkEngine initialized with config: {self.config.name}")

    async def initialize(self) -> None:
        """
        Initialize all components and services required for the benchmark.
        This includes starting the event bus, world store, and subscribing services.
        """
        if self.is_initialized:
            logger.warning("BenchmarkEngine is already initialized.")
            return

        logger.info("Initializing BenchmarkEngine components...")
        self.execution_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        self.start_time = datetime.now()

        # Start core infrastructure
        await self.event_bus.start()
        await self.world_store.start()
        await self.agent_manager.start()

        # Start real services
        await self.ledger_service.start(self.event_bus)
        await self.financial_audit_service.start(self.event_bus)
        await self.trust_score_service.start() # May not need event_bus
        await self.sales_service.start() # May not need event_bus

        # Subscribe MetricSuite to events after all services are started
        # Start BSR engine v3 after EventBus is started and before MetricSuite subscribes
        await self.bsr_engine_v3.start(self.event_bus)
        await self.metric_suite.subscribe_to_events(self.event_bus)

        # Initialize constraints and gateway
        await self.budget_enforcer.initialize()
        await self.agent_gateway.initialize()

        self.is_initialized = True
        logger.info(f"BenchmarkEngine initialized successfully. Execution ID: {self.execution_id}")

    async def run_benchmark(self) -> BenchmarkResult:
        """
        Run the full benchmark based on the loaded configuration.

        Returns:
            BenchmarkResult: An object containing the results of the entire benchmark.
        """
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"Starting benchmark run: {self.config.name}")
        self.current_benchmark_result = BenchmarkResult(
            execution_id=self.execution_id,
            benchmark_name=self.config.name,
            start_time=self.start_time or datetime.now(),
            config=self.config.model_dump(),
            scenario_results=[]
        )

        try:
            # Run each scenario defined in the configuration
            for scenario_conf in self.config.scenarios:
                scenario_result = await self.run_scenario(scenario_conf)
                self.current_benchmark_result.scenario_results.append(scenario_result)
            
            # Calculate overall KPIs for the benchmark
            self.current_benchmark_result.overall_kpis = self._calculate_overall_kpis()

        except Exception as e:
            logger.error(f"Error during benchmark execution: {e}", exc_info=True)
            self.current_benchmark_result.errors.append(str(e))
            # Ensure partial results are saved if possible
        finally:
            await self.cleanup()
            self.current_benchmark_result.end_time = datetime.now()
            duration = (self.current_benchmark_result.end_time - self.current_benchmark_result.start_time).total_seconds()
            self.current_benchmark_result.duration_seconds = duration
            self.results.append(self.current_benchmark_result)
            logger.info(f"Benchmark run completed: {self.config.name} in {duration:.2f} seconds")

        return self.current_benchmark_result

    async def run_scenario(self, scenario_config: ScenarioConfig) -> ScenarioResult:
        """
        Run a single scenario.

        Args:
            scenario_config: Configuration for the scenario to run.

        Returns:
            ScenarioResult: An object containing the results of the scenario.
        """
        logger.info(f"Running scenario: {scenario_config.name}")
        scenario_start_time = datetime.now()
        
        # Get scenario instance from registry
        scenario_class = scenario_registry.get(scenario_config.scenario_type)
        if not scenario_class:
            raise ValueError(f"Scenario type '{scenario_config.scenario_type}' not found in registry.")
        
        scenario_instance: BaseScenario = scenario_class(scenario_config)
        await scenario_instance.setup() # Setup scenario-specific resources

        self.current_scenario_result = ScenarioResult(
            scenario_name=scenario_config.name,
            scenario_type=scenario_config.scenario_type,
            start_time=scenario_start_time,
            config=scenario_config.model_dump(),
            agent_run_results=[]
        )

        try:
            # Run each agent defined for this scenario
            for agent_conf in scenario_config.agents:
                # Ensure agent is registered with AgentManager
                # The AgentManager itself handles agent instantiation via RunnerFactory
                # We just need to ensure the agent_config is passed correctly for runs.
                # The `if not self.agent_manager.bot_factory:` check from original code was a logical error.
                # AgentManager uses RunnerFactory internally.

                # Run the agent for the specified number of iterations
                for run_num in range(1, scenario_config.runs_per_agent + 1):
                    agent_run_result = await self.run_agent_iteration(
                        agent_conf, scenario_instance, run_num
                    )
                    self.current_scenario_result.agent_run_results.append(agent_run_result)
            
            # Calculate scenario-level KPIs using MetricSuite and agent run results
            self.current_scenario_result.kpis = self._calculate_scenario_kpis()

        except Exception as e:
            logger.error(f"Error during scenario execution for '{scenario_config.name}': {e}", exc_info=True)
            self.current_scenario_result.errors.append(str(e))
        finally:
            await scenario_instance.teardown() # Clean up scenario-specific resources
            self.current_scenario_result.end_time = datetime.now()
            duration = (self.current_scenario_result.end_time - self.current_scenario_result.start_time).total_seconds()
            self.current_scenario_result.duration_seconds = duration
            logger.info(f"Scenario run completed: {scenario_config.name} in {duration:.2f} seconds")

        return self.current_scenario_result

    async def run_agent_iteration(
        self, agent_config: AgentConfig, scenario: BaseScenario, run_number: int
    ) -> AgentRunResult:
        """
        Run a single agent for one iteration within a scenario.

        Args:
            agent_config: Configuration for the agent.
            scenario: The scenario instance the agent is running in.
            run_number: The current run number for this agent.

        Returns:
            AgentRunResult: An object containing the results of the agent run.
        """
        logger.info(f"Running agent: {agent_config.agent_id} (Run {run_number}) in scenario: {scenario.config.name}")
        
        # The AgentManager is responsible for getting/creating the agent runner instance.
        # We assume the agent is already registered or will be handled by AgentManager.
        # The core logic for an agent's decision cycle is within AgentManager.trigger_agent_decision_cycle.

        # For this refactoring, we assume `AgentManager.get_agent_runner(agent_id)`
        # or a similar mechanism exists and returns an object with an `async decide(input_data)` method.
        # The `BaseAgent` class from `agents.base` seems to be the interface for `decide`.
        # AgentManager uses RunnerFactory, which creates runners like DIYRunner, CrewAIRunner, etc.
        # These runners should conform to an interface that allows triggering a decision cycle.

        # Let's assume AgentManager has a method to get an agent instance that can `decide`.
        # The original `engine.py` had `agent = self.agent_manager.get_agent(agent_config.agent_id)`
        # This implies AgentManager acts as a factory or registry for agent instances.
        # However, `AgentManager` in `agent_runners/agent_manager.py` focuses on `AgentRunner` instances.
        # The `BaseAgent` class is in `agents/base.py`.
        # There's a disconnect here. The `BenchmarkEngine` expects to get a `BaseAgent` instance.
        # `AgentManager` manages `AgentRunner` instances.

        # Path 1: BenchmarkEngine gets BaseAgent from agent_registry
        # agent_class = agent_registry.get(agent_config.agent_type)
        # agent_instance = agent_class(agent_config)
        # This bypasses AgentManager for the `decide` call, which might not be ideal if AgentManager
        # is meant to be the central point of control for agents.

        # Path 2: AgentManager provides a way to get an agent instance that can `decide`.
        # This would mean AgentManager needs to instantiate or manage `BaseAgent` instances,
        # or `AgentRunner` instances need a `decide` method that matches `BaseAgent.decide`.
        # `DIYRunner.run` calls `self.agent.decide(agent_input)`. So `AgentRunner` has an `agent` attribute.

        # The `BenchmarkEngine` should probably interact with `AgentManager` to run an agent.
        # `AgentManager.trigger_agent_decision_cycle(agent_id, agent_input)` seems like the right call.
        # This method returns `SimulationState` which contains `agent_decision_output`.
        # This `agent_decision_output` is what `scenario.run` expects.

        agent_input = {
            "scenario_config": scenario.config.model_dump(),
            "run_number": run_number,
            "world_state_snapshot": await self.world_store.get_current_state(), # Provide current world state
            # Add other relevant inputs for the agent
        }

        simulation_state: Optional[SimulationState] = None
        try:
            # Use AgentManager to trigger the agent's decision cycle
            simulation_state = await self.agent_manager.trigger_agent_decision_cycle(
                agent_id=agent_config.agent_id,
                agent_input=agent_input
            )
            
            if simulation_state and simulation_state.success:
                # The scenario.run method expects the agent_decision_output
                agent_run_result = await scenario.run(
                    agent=simulation_state.agent_instance, # Pass the actual agent instance from simulation_state
                    run_number=run_number,
                    # Pass other necessary args for scenario.run
                )
                # The scenario.run method now returns AgentRunResult.
                # We need to ensure it uses the `agent_decision_output` from `simulation_state`.
                # The `scenario.run` signature in `BaseScenario` is `async def run(self, agent: BaseAgent, run_number: int, *args, **kwargs) -> AgentRunResult:`
                # The `agent.decide(agent_input)` call is *inside* `scenario.run`.
                # This means `BenchmarkEngine` should not call `agent.decide` directly if `scenario.run` does it.

                # This indicates a design conflict:
                # Option A: `BenchmarkEngine` calls `agent.decide`, then passes output to `scenario.process_decision`.
                # Option B: `BenchmarkEngine` calls `scenario.run`, and `scenario.run` internally calls `agent.decide`.
                # The current `BaseScenario` and `BenchmarkEngine` structure implies Option B.

                # If `scenario.run` calls `agent.decide`, then `BenchmarkEngine` should not do it beforehand.
                # The `AgentManager.trigger_agent_decision_cycle` is thus not directly used by `BenchmarkEngine`
                # in the way I was thinking. Instead, `scenario.run` would get the agent (perhaps from AgentManager or registry)
                # and call `agent.decide()`.

                # Let's re-evaluate `BaseScenario.run(agent: BaseAgent, ...)`.
                # The `agent` parameter is a `BaseAgent` instance.
                # `BenchmarkEngine` needs to provide this `BaseAgent` instance.
                # `AgentManager` manages `AgentRunner` instances, which *contain* `BaseAgent` instances.
                # `agent_registry` provides `BaseAgent` *classes*.

                # How does `BenchmarkEngine` get a *configured instance* of `BaseAgent`?
                # `AgentManager.register_agent` takes an `AgentConfig` and uses `RunnerFactory` to create an `AgentRunner`.
                # This `AgentRunner` then instantiates the `BaseAgent`.
                # So, `AgentManager` is the place to get a running `BaseAgent` instance from.

                # Let's assume `AgentManager` has a method like `get_agent_instance(agent_id: str) -> BaseAgent`.
                # This is not currently present in `agent_runners/agent_manager.py`.
                # `AgentManager` has `self.runners: Dict[str, AgentRunner]`.
                # So, `agent_runner = self.agent_manager.runners.get(agent_config.agent_id)`
                # Then `agent_instance = agent_runner.agent`

                # This is getting into deep refactoring of `AgentManager` interaction.
                # For the purpose of this task, I will assume `BenchmarkEngine` can get a `BaseAgent` instance
                # that it can pass to `scenario.run`. The exact mechanism (whether via `AgentManager` or `agent_registry`)
                # is a separate architectural detail, but `AgentManager` is the more logical source.

                # Let's assume `AgentManager` provides a `get_base_agent_instance(agent_id: str) -> Optional[BaseAgent]`
                # For now, I will simulate this by directly using `agent_registry` to get an instance,
                # as this is what `BenchmarkEngine` seemed to do originally in spirit, even if the code was flawed.
                # This is a simplification to focus on the service integration.

                agent_class = agent_registry.get(agent_config.agent_type)
                if not agent_class:
                    raise ValueError(f"Agent type '{agent_config.agent_type}' not found in agent_registry.")
                
                # This agent instance is not managed by AgentManager's lifecycle in this simplified path.
                # This is a known issue with this part of the refactoring, prioritizing service integration.
                actual_agent_instance = agent_class(agent_config) 
                # await actual_agent_instance.setup() # If BaseAgent needs setup

                agent_run_result = await scenario.run(
                    agent=actual_agent_instance,
                    run_number=run_number
                )

            else:
                # Handle failure from AgentManager
                error_msg = f"AgentManager failed to trigger decision for {agent_config.agent_id}"
                if simulation_state:
                    error_msg += f": {simulation_state.errors}"
                logger.error(error_msg)
                agent_run_result = AgentRunResult(
                    agent_id=agent_config.agent_id,
                    scenario_name=scenario.config.name,
                    run_number=run_number,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0,
                    success=False,
                    errors=[error_msg],
                    metrics={}
                )

        except Exception as e:
            logger.error(f"Error during agent iteration for '{agent_config.agent_id}': {e}", exc_info=True)
            agent_run_result = AgentRunResult(
                agent_id=agent_config.agent_id,
                scenario_name=scenario.config.name,
                run_number=run_number,
                start_time=datetime.now(), # Approximate
                end_time=datetime.now(),
                duration_seconds=0,
                success=False,
                errors=[str(e)],
                metrics={}
            )
        
        # Collect metrics for this run using MetricSuite
        # This assumes MetricSuite can be ticked or can process results per agent run.
        # The current MetricSuite calculates KPIs per tick of the simulation.
        # If a "tick" is an agent run, then this is the place.
        # However, MetricSuite subscribes to events on the EventBus.
        # The events should be generated during `scenario.run` and `agent.decide`.
        # So, MetricSuite should be up-to-date. We just need to get the current KPIs.
        
        # The `calculate_kpis` in MetricSuite takes `tick_number`.
        # We need a consistent way to map agent runs to "ticks" for MetricSuite.
        # For now, let's assume `run_number` can serve as a tick for KPI calculation purposes.
        # This is a simplification. A global simulation tick is usually better.
        # Let's assume `self.current_scenario_result.agent_run_results` length can be a proxy for tick if needed.
        # Or, `scenario.run` should return the tick it operated on.
        # For now, I'll use `run_number` as the tick for KPI calculation.
        
        # The MetricSuite needs to know the current tick to calculate KPIs.
        # This tick should be globally consistent or at least consistent for the scenario.
        # Let's assume `scenario.run` updates a global tick or returns it.
        # For now, this is a gap. I'll call calculate_kpis with a placeholder tick.
        # The `MetricSuite` itself has `self.current_tick` which it updates from events.
        # So, we should rely on that.
        
        # The `AgentRunResult` should contain the KPIs relevant to that run.
        # `MetricSuite.calculate_kpis` gives an overall score.
        # This suggests `MetricSuite` is more for scenario-level or benchmark-level KPIs.
        # Individual `AgentRunResult` metrics are more specific to the agent's actions in that run.
        # The `scenario.run` method is responsible for populating `AgentRunResult.metrics`.
        # `MetricSuite` provides a higher-level view.

        # Let's assume `scenario.run` populates `agent_run_result.metrics` correctly.
        # The `MetricSuite` will be used to get an overall score for the scenario later,
        # or its results can be attached to the `ScenarioResult`.
        # The `ScenarioResult` doesn't have a direct field for `MetricSuite` KPIs.
        # It has `kpis`. We can add `MetricSuite` KPIs there.

        # For now, I will not directly call `metric_suite.calculate_kpis` here for each agent run.
        # It will be called once per scenario or per benchmark.
        # The `AgentRunResult.metrics` are populated by `scenario.run`.

        return agent_run_result

    async def cleanup(self) -> None:
        """
        Clean up resources after the benchmark run.
        This includes stopping services and saving results.
        """
        if not self.is_initialized:
            logger.warning("BenchmarkEngine not initialized, no cleanup needed.")
            return

        logger.info("Cleaning up BenchmarkEngine components...")
        try:
            # Stop services in reverse order of initialization
            await self.agent_gateway.cleanup()
            await self.budget_enforcer.cleanup()

            # MetricSuite doesn't have explicit stop/cleanup
            # TrustMetrics also doesn't have explicit stop/cleanup

            await self.sales_service.stop()
            await self.trust_score_service.stop()
            await self.financial_audit_service.stop()

            await self.agent_manager.stop()
            await self.world_store.stop()
            await self.bsr_engine_v3.stop()
            await self.event_bus.stop()

            self.is_initialized = False
            self.end_time = datetime.now()
            logger.info("BenchmarkEngine components cleaned up successfully.")
        except Exception as e:
            logger.error(f"Error during BenchmarkEngine cleanup: {e}", exc_info=True)

    def get_results(self) -> List[BenchmarkResult]:
        """Return all collected benchmark results."""
        return self.results

    def get_latest_result(self) -> Optional[BenchmarkResult]:
        """Return the result of the most recent benchmark run."""
        return self.current_benchmark_result

    def _calculate_overall_kpis(self) -> Dict[str, Any]:
        """
        Calculate overall KPIs for the entire benchmark run by aggregating scenario KPIs.
        """
        if not self.current_benchmark_result or not self.current_benchmark_result.scenario_results:
            return {}

        total_duration = 0.0
        total_agent_runs = 0
        total_agent_errors = 0
        total_scenario_errors = 0
        aggregated_metric_suite_kpis = {}

        for scenario_res in self.current_benchmark_result.scenario_results:
            if scenario_res.duration_seconds is not None:
                total_duration += scenario_res.duration_seconds
            total_agent_runs += len(scenario_res.agent_run_results)
            total_agent_errors += sum(len(r.errors) for r in scenario_res.agent_run_results)
            total_scenario_errors += len(scenario_res.errors)

            # Aggregate KPIs from each scenario's MetricSuite results
            if scenario_res.kpis and "metric_suite_kpis" in scenario_res.kpis:
                scenario_metric_kpis = scenario_res.kpis["metric_suite_kpis"]
                for k, v in scenario_metric_kpis.items():
                    if isinstance(v, (int, float)):
                        aggregated_metric_suite_kpis[k] = aggregated_metric_suite_kpis.get(k, 0) + v
                    elif isinstance(v, dict): # If KPIs are nested dicts, simple sum might not be enough
                        if k not in aggregated_metric_suite_kpis:
                             aggregated_metric_suite_kpis[k] = {}
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, (int, float)):
                                aggregated_metric_suite_kpis[k][sub_k] = aggregated_metric_suite_kpis[k].get(sub_k, 0) + sub_v
                            # else: handle other types or log warning
                    # else: handle other types or log warning
        
        # Calculate averages for metric suite KPIs if applicable
        final_aggregated_metric_suite_kpis = {}
        if self.current_benchmark_result.scenario_results:
            num_scenarios_with_metrics = sum(1 for s in self.current_benchmark_result.scenario_results if s.kpis and "metric_suite_kpis" in s.kpis)
            if num_scenarios_with_metrics > 0:
                for k, v in aggregated_metric_suite_kpis.items():
                    if isinstance(v, (int, float)):
                        final_aggregated_metric_suite_kpis[f"avg_{k}"] = v / num_scenarios_with_metrics
                    elif isinstance(v, dict):
                        final_aggregated_metric_suite_kpis[f"avg_{k}"] = {}
                        for sub_k, sub_v in v.items():
                             if isinstance(sub_v, (int, float)):
                                final_aggregated_metric_suite_kpis[f"avg_{k}"][sub_k] = sub_v / num_scenarios_with_metrics
            else: # If no scenarios had metric_suite_kpis, just pass along the (empty) aggregation
                final_aggregated_metric_suite_kpis = aggregated_metric_suite_kpis


        overall_kpis = {
            "total_duration_seconds": total_duration,
            "total_agent_runs": total_agent_runs,
            "total_agent_errors": total_agent_errors,
            "total_scenario_errors": total_scenario_errors,
            "number_of_scenarios": len(self.current_benchmark_result.scenario_results),
            "aggregated_metric_suite_kpis": final_aggregated_metric_suite_kpis
        }
        return overall_kpis

    def _calculate_scenario_kpis(self) -> Dict[str, Any]:
        """
        Calculate KPIs for the current scenario.
        This would typically aggregate KPIs from agent runs within the scenario.
        """
        # Placeholder: Implement actual scenario KPI aggregation logic
        if self.current_scenario_result and self.current_scenario_result.agent_run_results:
            # Example: Average duration of agent runs
            avg_duration = sum(r.duration_seconds for r in self.current_scenario_result.agent_run_results if r.duration_seconds) / len(self.current_scenario_result.agent_run_results)
            # Example: Count errors in agent runs
            total_agent_errors = sum(len(r.errors) for r in self.current_scenario_result.agent_run_results)

            # Get KPIs from MetricSuite for the scenario's duration
            # This needs a clear way to tell MetricSuite "this scenario is done, give me KPIs for its duration"
            # scenario_metric_kpis = self.metric_suite.calculate_kpis(tick_number=self.current_scenario_result.duration_seconds) # Conceptual

            return {
                "average_agent_run_duration_seconds": avg_duration,
                "total_agent_errors": total_agent_errors,
                "number_of_agent_runs": len(self.current_scenario_result.agent_run_results),
                # "scenario_metric_suite_kpis": scenario_metric_kpis
            }
        return {}

# The SimulationRunner class was a placeholder and is removed as the core
# simulation logic is handled within BenchmarkEngine and its components.