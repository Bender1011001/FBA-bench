"""
Core benchmarking engine.

This module provides the main BenchmarkEngine class that orchestrates the entire
benchmarking process, including agent lifecycle management, metrics collection,
and reproducible execution.
"""

from __future__ import annotations

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
# Import only the types we still use from results; BenchmarkResult is redefined below for strict v2 schema
from .results import ScenarioResult, AgentRunResult, MetricResult

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
from services.customer_reputation_service import CustomerReputationService
from services.market_simulator import MarketSimulationService
from services.marketing_service import MarketingService
from services.supply_chain_service import SupplyChainService

logger = logging.getLogger(__name__)

class BenchmarkError(Exception):
    """Engine-level benchmark error."""
    pass

# Provide strict Pydantic v2 models and enums expected by unit tests
from enum import Enum as _Enum
from pydantic import BaseModel as _PydBaseModel, Field as _Field, ConfigDict as _ConfigDict  # v2
from datetime import datetime as _dt


class RunStatus(str, _Enum):
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"
    stopped = "stopped"
    timeout = "timeout"


class BenchmarkResult(_PydBaseModel):
    # Strict schema: forbid unknown fields; strip whitespace in strings
    model_config = _ConfigDict(extra="forbid", str_strip_whitespace=True)

    benchmark_id: str
    run_id: str
    status: RunStatus
    metrics: Dict[str, float] = _Field(default_factory=dict)
    warnings: List[str] = _Field(default_factory=list)
    errors: List[str] = _Field(default_factory=list)
    started_at: Optional[_dt] = None
    finished_at: Optional[_dt] = None


class BenchmarkRun(_PydBaseModel):
    model_config = _ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    config: Dict[str, object] = _Field(default_factory=dict)
    created_at: _dt = _Field(default_factory=_dt.utcnow)
    updated_at: Optional[_dt] = None

    def mark(self, status: RunStatus) -> None:
        self.status = status
        self.updated_at = _dt.utcnow()


# Re-export for direct import in tests
__all__ = [
    "BenchmarkEngine",
    "BenchmarkError",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRun",
    "RunStatus",
]
class BenchmarkEngine:
    """
    The main orchestrator for running benchmarks.

    Manages the entire lifecycle of a benchmark, from setup and execution
    to results collection and reporting. Ensures reproducibility and
    integrates with various services like event bus, world store, and metrics.
    """

    def __init__(self, config: BenchmarkConfig, agent_registry_override: Optional["AgentRegistry"] = None):
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
        # Agent registry (module-level import preserved for test patching)
        self._agent_registry = agent_registry_override or agent_registry
        self._agent_registry_override_used = agent_registry_override is not None

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

        # Customer reputation service (reviews -> reputation -> BSR impact)
        self.customer_reputation_service = CustomerReputationService()
        
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
        # Prefer explicit budget_overrides when provided; otherwise use tier-configured BudgetEnforcer.
        if getattr(self.config, "budget_overrides", None):
            self.budget_enforcer = BudgetEnforcer(self.config.budget_overrides, self.event_bus)
        else:
            self.budget_enforcer = BudgetEnforcer.from_tier_config(
                tier=self.config.tier,
                event_bus=self.event_bus
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
        # Record full event stream for outcome analysis and reproducibility
        self.event_bus.start_recording()

        # Start real services
        await self.ledger_service.start(self.event_bus)
        await self.financial_audit_service.start(self.event_bus)
        await self.trust_score_service.start() # May not need event_bus
        await self.sales_service.start() # May not need event_bus

        # Subscribe MetricSuite to events after all services are started
        # Start BSR engine v3 after EventBus is started and before MetricSuite subscribes
        await self.bsr_engine_v3.start(self.event_bus)
        # Wire BSR engine to read reputation from WorldStore
        self.bsr_engine_v3.set_reputation_provider(self.world_store.get_reputation_score)
        # Start customer reputation service
        await self.customer_reputation_service.start(self.event_bus, self.world_store)
        await self.metric_suite.subscribe_to_events(self.event_bus)

        # Start OutcomeAnalysisService
        try:
            from services.outcome_analysis_service import OutcomeAnalysisService
            self.outcome_analysis_service = OutcomeAnalysisService()
            await self.outcome_analysis_service.start(self.event_bus)
        except Exception as e:
            logger.warning(f"OutcomeAnalysisService unavailable: {e}")
            self.outcome_analysis_service = None

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
    
        caught_exc: Optional[BaseException] = None
        try:
            # Run each scenario defined in the configuration
            for scenario_conf in self.config.scenarios:
                if not getattr(scenario_conf, "enabled", True):
                    logger.info(f"Skipping disabled scenario: {getattr(scenario_conf, 'name', 'unknown')}")
                    continue
                scenario_result = await self.run_scenario(scenario_conf)
                self.current_benchmark_result.scenario_results.append(scenario_result)
    
        except Exception as e:
            logger.error(f"Error during benchmark execution: {e}", exc_info=True)
            self.current_benchmark_result.errors.append(str(e))
            caught_exc = e
        finally:
            # Calculate overall KPIs in finally to guarantee availability on failure.
            try:
                self.current_benchmark_result.overall_kpis = self._calculate_overall_kpis()
            except Exception as kpi_err:
                logger.warning(f"Failed to calculate overall KPIs: {kpi_err}", exc_info=True)
    
            await self.cleanup()
            self.current_benchmark_result.end_time = datetime.now()
            duration = (self.current_benchmark_result.end_time - self.current_benchmark_result.start_time).total_seconds()
            self.current_benchmark_result.duration_seconds = duration
            self.results.append(self.current_benchmark_result)
            logger.info(f"Benchmark run completed: {self.config.name} in {duration:.2f} seconds")
    
        if caught_exc is not None:
            # Preserve original exception semantics after KPI calculation
            raise caught_exc
    
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

        # Start per-scenario world-model services
        market_simulator = MarketSimulationService(world_store=self.world_store, event_bus=self.event_bus)
        marketing_service = MarketingService(world_store=self.world_store, event_bus=self.event_bus)
        supply_chain_service = SupplyChainService(world_store=self.world_store, event_bus=self.event_bus)
        await marketing_service.start()
        await supply_chain_service.start()
        await market_simulator.start()

        # Setup scenario with core services
        await scenario_instance.setup(
            event_bus=self.event_bus,
            world_store=self.world_store,
            market_simulator=market_simulator,
            supply_chain_service=supply_chain_service,
            marketing_service=marketing_service,
        )

        self.current_scenario_result = ScenarioResult(
            scenario_name=scenario_config.name,
            scenario_type=getattr(scenario_config, "scenario_type", scenario_config.name),
            start_time=scenario_start_time,
            config=scenario_config.model_dump() if hasattr(scenario_config, "model_dump") else scenario_config.__dict__,
            agent_run_results=[]
        )

        try:
            # Ensure agents from benchmark config are registered and active
            for agent_cfg in self.config.agents:
                if getattr(agent_cfg, "enabled", True):
                    # Register if not present
                    reg = self.agent_manager.agent_registry.get_agent(agent_cfg.agent_id)
                    if reg is None or reg.runner is None:
                        self.agent_manager.register_agent(agent_cfg.agent_id, agent_cfg.framework, agent_cfg.config or {})

            # Execute scenario for each enabled agent in scenario (fallback to all config agents if scenario doesn't list)
            scenario_agents: List[AgentConfig] = []
            if getattr(scenario_config, "agents", None):
                scenario_agents = [a for a in scenario_config.agents if getattr(a, "enabled", True)]
            else:
                scenario_agents = [a for a in self.config.agents if getattr(a, "enabled", True)]

            total_ticks = getattr(scenario_config, "runs_per_agent", None) or getattr(scenario_config, "duration_ticks", 1)
            total_ticks = int(total_ticks) if total_ticks and total_ticks > 0 else 1

            for agent_conf in scenario_agents:
                for run_num in range(1, total_ticks + 1):
                    agent_run_result = await self.run_agent_iteration(
                        agent_conf,
                        scenario_instance,
                        run_num,
                        event_bus=self.event_bus,
                        world_store=self.world_store,
                        market_simulator=market_simulator,
                        supply_chain_service=supply_chain_service,
                        marketing_service=marketing_service,
                    )
                    self.current_scenario_result.agent_run_results.append(agent_run_result)

            # Calculate scenario-level KPIs using MetricSuite and agent run results
            self.current_scenario_result.kpis = self._calculate_scenario_kpis()

        except Exception as e:
            logger.error(f"Error during scenario execution for '{scenario_config.name}': {e}", exc_info=True)
            self.current_scenario_result.errors.append(str(e))
        finally:
            # Stop per-scenario services
            try:
                await marketing_service.stop()
            except Exception:
                pass
            try:
                await supply_chain_service.stop()
            except Exception:
                pass

            await scenario_instance.teardown() # Clean up scenario-specific resources
            self.current_scenario_result.end_time = datetime.now()
            duration = (self.current_scenario_result.end_time - self.current_scenario_result.start_time).total_seconds()
            self.current_scenario_result.duration_seconds = duration
            logger.info(f"Scenario run completed: {scenario_config.name} in {duration:.2f} seconds")

        return self.current_scenario_result

    async def run_agent_iteration(
        self,
        agent_config: AgentConfig,
        scenario: BaseScenario,
        run_number: int,
        **kwargs: Any,
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

        try:
            # Retrieve the managed runner from AgentManager and delegate execution to scenario
            agent_runner = self.agent_manager.get_agent_runner(agent_config.agent_id)
            if not agent_runner:
                raise RuntimeError(f"Agent '{agent_config.agent_id}' is not registered or initialized.")

            # Scenarios call agent.decide(...) internally, so pass the runner (it exposes decide)
            agent_run_result = await scenario.run(
                agent=agent_runner,
                run_number=run_number
            )

        except Exception as e:
            logger.error(f"Error during agent iteration for '{agent_config.agent_id}': {e}", exc_info=True)
            agent_run_result = AgentRunResult(
                agent_id=agent_config.agent_id,
                scenario_name=scenario.config.name,
                run_number=run_number,
                start_time=datetime.now(),  # Approximate
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
        # scenario.run should be responsible for advancing ticks; MetricSuite tracks ticks via events.
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

        # Learning loop: analyze outcome and inform agent runner
        try:
            if getattr(self, "outcome_analysis_service", None) is not None:
                # Analyze all events since last tick/run for this agent
                outcome = await self.outcome_analysis_service.analyze_tick_outcome(
                    agent_id=agent_config.agent_id
                )
                if outcome:
                    try:
                        agent_runner = self.agent_manager.get_agent_runner(agent_config.agent_id)
                        if agent_runner and hasattr(agent_runner, "learn"):
                            await agent_runner.learn(outcome)
                    except Exception as lrn_err:
                        logger.warning(f"Agent {agent_config.agent_id} learn() failed: {lrn_err}")
        except Exception as e:
            logger.warning(f"Outcome analysis failed for agent {agent_config.agent_id}: {e}")

        return agent_run_result

    def _load_agent(self, slug: str, version: Optional[str] = None, config: Any = None) -> Any:
        """
        Resolve and create or retrieve an agent instance using the agent registry.

        Errors:
          - When agent not found -> raises BenchmarkError("Agent {slug} not found")
          - When creation fails -> raises BenchmarkError("Agent {slug} creation failed: {e}")
        """
        # Important: if no override was supplied at construction time, use the module-level
        # agent_registry to respect tests that patch benchmarking.core.engine.agent_registry.
        reg = self._agent_registry if getattr(self, "_agent_registry_override_used", False) else agent_registry
        try:
            descriptor = reg.get_agent(slug, version)
            ctor = descriptor.constructor
            if callable(ctor):
                return reg.create_agent(slug, version=version, config=config)
            # pre-instantiated
            return ctor
        except KeyError as e:
            raise BenchmarkError(f"Agent {slug} not found") from e
        except ValueError as e:
            raise BenchmarkError(f"Agent {slug} creation failed: {e}") from e

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
        # Aggregate scenario-level metrics across agent runs
        if self.current_scenario_result and self.current_scenario_result.agent_run_results:
            durations = [r.duration_seconds for r in self.current_scenario_result.agent_run_results if r.duration_seconds]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
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
# -------------------- Lightweight, async Benchmarking Engine (new API) --------------------
# This section implements a clean, self-contained benchmarking engine that coexists with the
# existing BenchmarkEngine above. It follows the spec described in the task.
#
# Example EngineConfig (see models for full schema and examples):
# {
#   "scenarios":[{"key":"example_scenario","params":{"difficulty":"easy"},"repetitions":2,"seeds":[1,2],"timeout_seconds":5}],
#   "runners":[{"key":"diy","config":{"agent_id":"baseline-1"}}],
#   "metrics":["technical_performance"],
#   "validators":["basic_consistency"],
#   "parallelism":2,
#   "retries":1
# }

import asyncio
import contextlib
import importlib
import json
import math
import time as _time
from dataclasses import dataclass
from hashlib import sha256
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    # Use FBA centralized logging if available
    from fba_bench.core.logging import setup_logging  # type: ignore
    setup_logging()
except Exception:
    pass

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
except Exception as e:  # pragma: no cover
    # pydantic is a dependency in pyproject; this guard avoids import-time crash in exotic envs
    raise

# Registries and helpers
from benchmarking.scenarios.registry import scenario_registry
from benchmarking.metrics.registry import MetricRegistry
from benchmarking.validators.registry import ValidatorRegistry
from agent_runners.registry import create_runner
from agent_runners.base_runner import AgentRunnerInitializationError  # type: ignore

# Optional Redis pubsub
with contextlib.suppress(Exception):
    from fba_bench_api.core.redis_client import get_redis  # type: ignore

# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------

class RunnerSpec(BaseModel):
    key: str = Field(..., description="Runner registry key (e.g., 'diy','crewai','langchain').")
    config: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"key": "diy", "config": {"agent_id": "baseline-1"}}
            ]
        }
    }


class ScenarioSpec(BaseModel):
    key: str = Field(..., description="Scenario key in registry or dotted import path module:function or module:Class")
    params: Optional[Dict[str, Any]] = Field(default=None)
    repetitions: int = Field(default=1, ge=1)
    seeds: Optional[List[int]] = Field(default=None)
    timeout_seconds: Optional[int] = Field(default=None, ge=1)
    # Per the spec
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "key": "tests.simple_scenario",
                    "params": {"difficulty": "easy"},
                    "repetitions": 2,
                    "seeds": [1, 2],
                    "timeout_seconds": 5,
                }
            ]
        }
    }


class EngineConfig(BaseModel):
    scenarios: List[ScenarioSpec]
    runners: List[RunnerSpec]
    metrics: List[str] = Field(default_factory=list)
    validators: List[str] = Field(default_factory=list)
    parallelism: int = Field(default=1, ge=1, description="Maximum concurrent run tasks")
    retries: int = Field(default=0, ge=0, description="Retry attempts for failed/error runs")
    observation_topic_prefix: str = Field(default="benchmark")

    @model_validator(mode="after")
    def _validate_lists(self) -> "EngineConfig":
        if not self.scenarios:
            raise ValueError("At least one scenario must be provided")
        if not self.runners:
            raise ValueError("At least one runner must be provided")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scenarios": [
                        {
                            "key": "example_scenario",
                            "params": {"difficulty": "easy"},
                            "repetitions": 2,
                            "seeds": [1, 2],
                            "timeout_seconds": 5,
                        }
                    ],
                    "runners": [{"key": "diy", "config": {"agent_id": "baseline-1"}}],
                    "metrics": ["technical_performance"],
                    "validators": ["basic_consistency"],
                    "parallelism": 2,
                    "retries": 1,
                    "observation_topic_prefix": "benchmark",
                }
            ]
        }
    }


class RunResult(BaseModel):
    scenario_key: str
    runner_key: str
    seed: Optional[int] = None
    status: str = Field(description="success|failed|timeout|error")
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scenario_key": "example_scenario",
                    "runner_key": "diy",
                    "seed": 1,
                    "status": "success",
                    "output": {"value": 42},
                    "error": None,
                    "duration_ms": 120,
                    "metrics": {"score": 0.95},
                    "artifacts": {"log": "s3://..."},
                }
            ]
        }
    }


class ScenarioReport(BaseModel):
    scenario_key: str
    runs: List[RunResult]
    aggregates: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scenario_key": "example_scenario",
                    "runs": [],
                    "aggregates": {"pass_count": 1, "fail_count": 0, "duration_ms": {"avg": 120}},
                }
            ]
        }
    }


class EngineReport(BaseModel):
    started_at: float
    finished_at: float
    config_digest: str
    scenario_reports: List[ScenarioReport]
    totals: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "started_at": 1723948123.123,
                    "finished_at": 1723948125.223,
                    "config_digest": "abc123...",
                    "scenario_reports": [],
                    "totals": {"runs": 4, "success": 4, "failed": 0, "duration_ms": {"sum": 480}},
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Engine implementation
# ---------------------------------------------------------------------------

class Engine:
    """
    Orchestrates: load scenario -> run agents/runs -> collect raw results -> apply metrics
                   -> run validators -> aggregate/report.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._metrics_registry = MetricRegistry()
        self._validators_registry = ValidatorRegistry()
        self._sema = asyncio.Semaphore(self.config.parallelism)
        self._started_at: float = 0.0
        self._redis_available: bool = False
        # Detect if redis client import succeeded
        self._redis_available = "get_redis" in globals()

    async def run(self) -> EngineReport:
        self._started_at = _time.time()
        digest = _digest_config(self.config)
        scenario_reports: List[ScenarioReport] = []

        # Execute each scenario
        for sc in self.config.scenarios:
            runs: List[RunResult] = []
            tasks: List[asyncio.Task] = []

            # Determine seeds/repetitions
            seeds = sc.seeds if sc.seeds else [None] * sc.repetitions

            for runner_spec in self.config.runners:
                for seed in seeds:
                    tasks.append(asyncio.create_task(self._guarded_run(sc, runner_spec, seed)))

            # Concurrency bound
            for chunk in _as_completed_bounded(tasks, self._sema):
                run_result: RunResult = await chunk
                runs.append(run_result)
                # Pub/sub per-run finished
                await self._publish_event(
                    topic=f"{self.config.observation_topic_prefix}:scenario:{sc.key}",
                    event={
                        "type": "run_finished",
                        "runner": run_result.runner_key,
                        "seed": run_result.seed,
                        "status": run_result.status,
                    },
                )

            # After runs: apply scenario metrics aggregation and validators
            aggregates = summarize_scenario(ScenarioReport(scenario_key=sc.key, runs=runs, aggregates={}))
            # Apply validators via function-style registry first; fallback to legacy class-based
            try:
                # Provide a full ScenarioReport-like dict to validators
                validations = await self._apply_validators(
                    scenario_report={
                        "scenario_key": sc.key,
                        "runs": [r.model_dump() for r in runs],
                        "aggregates": aggregates,
                    },
                    context={
                        "scenario_key": sc.key,
                        "expected_seeds": sc.seeds,
                        "config_digest": digest,
                        **(sc.params or {}),
                    },
                )
            except Exception as e:
                logger.error(f"_apply_validators failed: {e}")
                validations = [{"validator": "engine_apply_validators", "error": _short_error(str(e))}]
            if validations:
                aggregates.setdefault("validations", validations)

            scenario_reports.append(ScenarioReport(scenario_key=sc.key, runs=runs, aggregates=aggregates))

        finished_at = _time.time()
        report = EngineReport(
            started_at=self._started_at,
            finished_at=finished_at,
            config_digest=digest,
            scenario_reports=scenario_reports,
            totals=compute_totals(scenario_reports),
        )
        return report

    async def _guarded_run(self, scenario_spec: ScenarioSpec, runner_spec: RunnerSpec, seed: Optional[int]) -> RunResult:
        # Create runner with error handling
        try:
            runner = await _maybe_async(create_runner, runner_spec.key, runner_spec.config)
        except (ValueError, AgentRunnerInitializationError) as e:
            return RunResult(
                scenario_key=scenario_spec.key,
                runner_key=runner_spec.key,
                seed=seed,
                status="error",
                error=_short_error(str(e)),
                output=None,
                duration_ms=0,
                metrics={},
                artifacts=None,
            )
        except Exception as e:
            return RunResult(
                scenario_key=scenario_spec.key,
                runner_key=runner_spec.key,
                seed=seed,
                status="error",
                error=_short_error(f"runner_create_failed: {e}"),
                output=None,
                duration_ms=0,
                metrics={},
                artifacts=None,
            )

        # Resolve scenario target
        try:
            scenario_target = _resolve_scenario(scenario_spec.key)
        except Exception as e:
            return RunResult(
                scenario_key=scenario_spec.key,
                runner_key=runner_spec.key,
                seed=seed,
                status="error",
                error=_short_error(f"scenario_not_found: {e}"),
                output=None,
                duration_ms=0,
                metrics={},
                artifacts=None,
            )

        payload = _build_payload(scenario_spec.params or {}, seed)

        # Optional pub/sub started
        await self._publish_event(
            topic=f"{self.config.observation_topic_prefix}:scenario:{scenario_spec.key}",
            event={"type": "run_started", "runner": runner_spec.key, "seed": seed},
        )

        # Attempts with retries
        attempts = 0
        last_error: Optional[str] = None
        t0 = _time.perf_counter()
        while True:
            attempts += 1
            try:
                coro = _execute_scenario(scenario_target, runner, payload)
                if scenario_spec.timeout_seconds:
                    output = await asyncio.wait_for(coro, timeout=scenario_spec.timeout_seconds)
                else:
                    output = await coro
                duration_ms = int((_time.perf_counter() - t0) * 1000)
                # Apply metrics non-fatal
                run_for_metrics = {
                    "scenario_key": scenario_spec.key,
                    "runner_key": runner_spec.key,
                    "seed": seed,
                    "status": "success",
                    "output": _safe_jsonable(output),
                    "error": None,
                    "duration_ms": duration_ms,
                    "metrics": {},
                    "artifacts": None,
                }
                # Merge scenario params at top-level for metric contexts so metrics can
                # directly access expected keys (e.g., expected_output, keywords).
                metrics_context = {
                    "scenario_key": scenario_spec.key,
                    **(scenario_spec.params or {}),
                }
                metrics_out = await self._apply_metrics(run_for_metrics, metrics_context)
                return RunResult(
                    scenario_key=scenario_spec.key,
                    runner_key=runner_spec.key,
                    seed=seed,
                    status="success",
                    error=None,
                    output=_safe_jsonable(output),
                    duration_ms=duration_ms,
                    metrics=metrics_out,
                    artifacts=None,
                )
            except asyncio.TimeoutError:
                duration_ms = int((_time.perf_counter() - t0) * 1000)
                return RunResult(
                    scenario_key=scenario_spec.key,
                    runner_key=runner_spec.key,
                    seed=seed,
                    status="timeout",
                    error="timeout",
                    output=None,
                    duration_ms=duration_ms,
                    metrics={},
                    artifacts=None,
                )
            except Exception as e:
                last_error = _short_error(str(e))
                # Only retry for failed/error (not timeout per spec)
                if attempts <= self.config.retries:
                    # Deterministic backoff (no sleep to keep tests fast)
                    continue
                duration_ms = int((_time.perf_counter() - t0) * 1000)
                return RunResult(
                    scenario_key=scenario_spec.key,
                    runner_key=runner_spec.key,
                    seed=seed,
                    status="error",
                    error=last_error,
                    output=None,
                    duration_ms=duration_ms,
                    metrics={},
                    artifacts=None,
                )

    async def _apply_metrics(self, run: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply configured metrics to a single run.

        Supports two metric systems:
        1) Legacy class-based MetricRegistry where metric.calculate(data: dict) -> number/dict
        2) New function-style registry: evaluate(run: dict, context: dict|None=None) -> dict

        The function-style registry is attempted first by key; if not found, falls back to the legacy registry.
        Any exceptions are caught and reported as non-fatal errors per metric.

        Implementation detail: we pass the metrics computed so far in this loop to subsequent metrics
        via run['metrics'] to enable composite metrics (e.g., custom_scriptable) to reference prior results.
        """
        # Local import to avoid import cycles at module import time
        try:
            from ..metrics.registry import get_metric as _get_fn_metric  # function-style registry
        except Exception:
            _get_fn_metric = None  # soft-fail; we still attempt legacy

        result: Dict[str, Any] = {}
        for mkey in self.config.metrics:
            try:
                # Prepare a view of the run including metrics computed so far
                if isinstance(run, dict):
                    run_with_partial = dict(run)
                    # ensure nested dict
                    existing = run_with_partial.get("metrics") or {}
                    # do not mutate caller
                    merged = dict(existing)
                    merged.update(result)
                    run_with_partial["metrics"] = merged
                else:
                    run_with_partial = {"output": run, "metrics": dict(result)}

                # Prefer function-style metrics if available
                if _get_fn_metric is not None:
                    try:
                        fn = _get_fn_metric(mkey)
                    except KeyError:
                        fn = None
                    if callable(fn):
                        # New interface: evaluate(run: dict, context: dict|None=None) -> dict
                        val = fn(run_with_partial, context)
                        result[mkey] = val
                        continue

                # Fallback to legacy class-based MetricRegistry
                metric = self._metrics_registry.create_metric(mkey)
                if metric is None:
                    # Graceful: not found
                    result[mkey] = {"error": "metric_not_found"}
                    continue
                # Legacy interface calculate(data: dict) -> float|dict
                payload = run_with_partial if isinstance(run_with_partial, dict) else {"output": run_with_partial}
                # For legacy metrics, they typically expect "output" structure; provide both for compatibility
                if "output" not in payload and isinstance(run_with_partial, dict):
                    payload = {"output": run_with_partial.get("output", run_with_partial)}
                val = metric.calculate(payload)
                result[mkey] = val
            except Exception as e:
                logger.error(f"Metric '{mkey}' failed: {e}")
                result[mkey] = {"error": _short_error(str(e))}
        return result

    async def _apply_validators(self, scenario_report: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Apply configured validators to a ScenarioReport-like dict.

        Supports two validator systems:
        1) Function-style registry: validate(report: dict, context: dict|None=None) -> dict
           via benchmarking.validators.registry.get_validator
        2) Legacy class-based ValidatorRegistry (self._validators_registry)

        All exceptions are caught and converted to non-fatal errors per validator.
        """
        # Local import to avoid import cycles
        try:
            from ..validators.registry import get_validator as _get_fn_validator  # function-style
        except Exception:
            _get_fn_validator = None

        results: List[Dict[str, Any]] = []
        for vkey in self.config.validators:
            try:
                # Prefer function-style validator
                if _get_fn_validator is not None:
                    try:
                        vfn = _get_fn_validator(vkey)
                    except KeyError:
                        vfn = None
                    if callable(vfn):
                        out = vfn(scenario_report if isinstance(scenario_report, dict) else dict(scenario_report), context or {})
                        results.append({"validator": vkey, "issues": out.get("issues", []), "summary": out.get("summary", {})})
                        continue

                # Fallback: legacy class-based validator
                validator = self._validators_registry.create_validator(vkey)
                if validator is None:
                    results.append({"validator": vkey, "error": "validator_not_found"})
                    continue
                # legacy .validate(...) may expect various shapes; pass scenario_report
                legacy_out = validator.validate(scenario_report)
                try:
                    normalized = legacy_out.to_dict() if hasattr(legacy_out, "to_dict") else dict(legacy_out)
                except Exception:
                    normalized = {"result": str(legacy_out)}
                # Normalize to standard container
                if "issues" in normalized or "summary" in normalized:
                    results.append({"validator": vkey, "issues": normalized.get("issues", []), "summary": normalized.get("summary", {})})
                else:
                    results.append({"validator": vkey, "result": normalized})
            except Exception as e:
                logger.error(f"Validator '{vkey}' failed: {e}")
                results.append({"validator": vkey, "error": _short_error(str(e))})
        return results

    async def _publish_event(self, topic: str, event: Dict[str, Any]) -> None:
        if not self._redis_available:
            return
        try:
            client = await get_redis()  # type: ignore
            await client.publish(topic, json.dumps(event))
        except Exception:
            # Silent per spec
            return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _digest_config(cfg: EngineConfig) -> str:
    try:
        data = cfg.model_dump()
    except Exception:
        data = dict(cfg)  # type: ignore
    return sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _resolve_scenario(key: str) -> Any:
    """
    Resolve a scenario target by key:
    - Try scenario_registry.get(key) which returns a class; instantiate with params if __init__ accepts it
    - Else treat key as 'module:attr' and import (function or class). If class, instantiate.
    The returned object must be either:
      - an object with async def run(self, runner, payload) -> dict
      - or an async callable like async def fn(runner, payload) -> dict
    """
    with contextlib.suppress(Exception):
        cls = scenario_registry.get(key)  # type: ignore
        # Return class (instantiate later in _execute_scenario to pass params)
        return cls

    # Dotted import fallback: "path.to.module:callable_or_Class"
    if ":" not in key:
        raise ValueError(f"Scenario '{key}' not found in registry and not a dotted path")
    mod_name, attr = key.split(":", 1)
    module = importlib.import_module(mod_name)
    target = getattr(module, attr)
    return target


async def _execute_scenario(target: Any, runner: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the resolved scenario target uniformly.
    If target is a class with optional generate_input(params, seed) and async run(runner, payload).
    If target is a function async def f(runner, payload).
    """
    # Instantiate if class type
    if isinstance(target, type):
        instance = target(payload.get("params"))
        gen = getattr(instance, "generate_input", None)
        if callable(gen):
            try:
                payload = {"params": payload.get("params"), "seed": payload.get("seed"), "input": gen(payload.get("seed"), payload.get("params"))}
            except Exception:
                # fall back to original payload
                pass
        run = getattr(instance, "run", None)
        if not callable(run):
            raise TypeError("Scenario class missing run(...)")
        out = run(runner=runner, payload=payload)
        return await _maybe_await(out)

    # If target has generate_input in module-level attr, apply (rare)
    gen = getattr(target, "generate_input", None)
    if callable(gen):
        try:
            payload = {"params": payload.get("params"), "seed": payload.get("seed"), "input": gen(payload.get("seed"), payload.get("params"))}  # type: ignore
        except Exception:
            pass

    # Callable scenario function
    if callable(target):
        out = target(runner=runner, payload=payload)
        return await _maybe_await(out)

    raise TypeError("Unsupported scenario target type")


def _build_payload(params: Dict[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    return {"params": json.loads(json.dumps(params)), "seed": seed}


async def _maybe_async(fn: Callable, *a: Any, **kw: Any) -> Any:
    res = fn(*a, **kw)
    return await _maybe_await(res)


async def _maybe_await(x: Any) -> Any:
    if asyncio.iscoroutine(x):
        return await x
    return x


def _short_error(msg: str, max_len: int = 300) -> str:
    m = msg.strip().replace("\n", " ")[:max_len]
    return m


def _as_completed_bounded(tasks: List[asyncio.Task], sema: asyncio.Semaphore):
    """
    Consume tasks with a concurrency bound using a semaphore. The tasks
    are created already; we acquire before awaiting each to avoid over-parallelism spikes.
    """
    async def _consume(t: asyncio.Task):
        async with sema:
            return await t
    return [_consume(t) for t in tasks]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def summarize_scenario(report: ScenarioReport) -> Dict[str, Any]:
    """
    Compute scenario aggregates:
    - pass/fail counts by status
    - duration stats
    - aggregated numeric metrics means per metric key
    """
    statuses = [r.status for r in report.runs]
    success = sum(1 for s in statuses if s == "success")
    failed = sum(1 for s in statuses if s in ("failed", "error"))
    timeouts = sum(1 for s in statuses if s == "timeout")
    durations = [r.duration_ms for r in report.runs if r.duration_ms is not None]
    dur_stats = {
        "count": len(durations),
        "sum": sum(durations) if durations else 0,
        "avg": mean(durations) if durations else 0.0,
        "min": min(durations) if durations else 0,
        "max": max(durations) if durations else 0,
    }

    # Aggregate metrics: if metric value is numeric, compute mean; else count occurrences
    metric_keys: Dict[str, List[float]] = {}
    categorical_counts: Dict[str, Dict[str, int]] = {}
    for r in report.runs:
        for k, v in (r.metrics or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                metric_keys.setdefault(k, []).append(float(v))
            else:
                # stringify non-numeric for counting
                categorical_counts.setdefault(k, {})
                sval = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
                categorical_counts[k][sval] = categorical_counts[k].get(sval, 0) + 1

    metric_means = {k: (mean(vals) if vals else 0.0) for k, vals in metric_keys.items()}
    aggregates = {
        "pass_count": success,
        "fail_count": failed,
        "timeout_count": timeouts,
        "runs": len(report.runs),
        "duration_ms": dur_stats,
        "metrics": {"mean": metric_means, "categorical_counts": categorical_counts},
    }
    return aggregates


def compute_totals(scenario_reports: List[ScenarioReport]) -> Dict[str, Any]:
    total_runs = sum(len(sr.runs) for sr in scenario_reports)
    success = sum(1 for sr in scenario_reports for r in sr.runs if r.status == "success")
    failed = sum(1 for sr in scenario_reports for r in sr.runs if r.status in ("failed", "error"))
    timeouts = sum(1 for sr in scenario_reports for r in sr.runs if r.status == "timeout")
    durations = [r.duration_ms for sr in scenario_reports for r in sr.runs if r.duration_ms is not None]
    dur_stats = {
        "count": len(durations),
        "sum": sum(durations) if durations else 0,
        "avg": mean(durations) if durations else 0.0,
        "min": min(durations) if durations else 0,
        "max": max(durations) if durations else 0,
    }

    # Per-metric aggregates (mean across all runs for numeric)
    metric_values: Dict[str, List[float]] = {}
    for sr in scenario_reports:
        for r in sr.runs:
            for k, v in (r.metrics or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                    metric_values.setdefault(k, []).append(float(v))
    metric_means = {k: (mean(vals) if vals else 0.0) for k, vals in metric_values.items()}

    return {
        "runs": total_runs,
        "success": success,
        "failed": failed,
        "timeout": timeouts,
        "duration_ms": dur_stats,
        "metrics": {"mean": metric_means},
    }


# ---------------------------------------------------------------------------
# Sync convenience wrapper
# ---------------------------------------------------------------------------

def run_benchmark(config: Union[Dict[str, Any], EngineConfig]) -> EngineReport:
    """
    Synchronous convenience wrapper.
    - Accepts dict or EngineConfig.
    - Runs event loop safely (loop-aware).
    """
    cfg = config if isinstance(config, EngineConfig) else EngineConfig.model_validate(config)  # type: ignore
    eng = Engine(cfg)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # In a running loop context; run via task and wait
        return asyncio.run_coroutine_threadsafe(eng.run(), loop).result()
    return asyncio.run(eng.run())
# Fallback logger for lightweight engine section (defined late is fine for runtime use)
import logging as _logging
logger = _logging.getLogger(__name__)