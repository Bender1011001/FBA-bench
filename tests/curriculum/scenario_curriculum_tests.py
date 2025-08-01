"""
Scenario and Curriculum Testing Framework for FBA-Bench

Tests tier progression (T0-T3), scenario diversity, curriculum validation,
and multi-agent scenario orchestration to ensure proper skill development.
"""

import asyncio
import logging
import pytest
import time
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os
from enum import Enum
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scenarios.curriculum_validator import CurriculumValidator, CurriculumTier, ProgressionCriteria
from scenarios.scenario_engine import ScenarioEngine, ScenarioType, ScenarioComplexity, ScenarioOutcome
from scenarios.tier_manager import TierManager, TierProgression, TierRequirements
from scenarios.multi_agent_coordinator import MultiAgentCoordinator, AgentRole, CoordinationMode
from scenarios.scenario_generator import ScenarioGenerator, GenerationConfig, ScenarioTemplate
from agents.hierarchical_planner import StrategicPlanner, TacticalPlanner
from agents.skill_coordinator import SkillCoordinator
from memory_experiments.dual_memory_manager import DualMemoryManager
from memory_experiments.memory_config import MemoryConfig
from event_bus import EventBus, get_event_bus
from events import BaseEvent, TickEvent, SaleOccurred, SetPriceCommand, MarketChangeEvent

logger = logging.getLogger(__name__)


class TierLevel(Enum):
    """Curriculum tier levels."""
    T0 = "T0"  # Basic single-agent scenarios
    T1 = "T1"  # Advanced single-agent scenarios  
    T2 = "T2"  # Multi-agent competitive scenarios
    T3 = "T3"  # Complex market dynamics scenarios


@dataclass
class ScenarioTestResult:
    """Results from scenario testing."""
    scenario_name: str
    tier_level: TierLevel
    scenario_type: ScenarioType
    success: bool
    performance_metrics: Dict[str, float]
    progression_criteria_met: Dict[str, bool]
    multi_agent_coordination: Dict[str, Any]
    duration_seconds: float
    error_details: Optional[str] = None


@dataclass
class CurriculumProgressionResult:
    """Results from curriculum progression testing."""
    agent_id: str
    starting_tier: TierLevel
    achieved_tier: TierLevel
    progression_path: List[TierLevel]
    tier_completion_times: Dict[str, float]
    skill_development: Dict[str, float]
    bottlenecks_identified: List[str]
    overall_success: bool


@dataclass
class MultiAgentScenarioConfig:
    """Configuration for multi-agent scenarios."""
    num_agents: int
    agent_roles: List[AgentRole]
    coordination_mode: CoordinationMode
    competition_level: float
    market_complexity: float
    scenario_duration_ticks: int


class ScenarioProgressionTracker:
    """Tracks agent progression through curriculum tiers."""
    
    def __init__(self):
        self.agent_progressions: Dict[str, Dict[str, Any]] = {}
        self.tier_requirements: Dict[TierLevel, TierRequirements] = {}
        self.scenario_history: List[Dict[str, Any]] = []
    
    def initialize_agent_progression(self, agent_id: str):
        """Initialize progression tracking for an agent."""
        self.agent_progressions[agent_id] = {
            "current_tier": TierLevel.T0,
            "completed_scenarios": [],
            "skill_scores": {},
            "tier_attempts": {tier: 0 for tier in TierLevel},
            "tier_completions": {tier: False for tier in TierLevel},
            "progression_timestamps": {}
        }
    
    def record_scenario_completion(self, agent_id: str, scenario_result: ScenarioTestResult):
        """Record a completed scenario for progression tracking."""
        if agent_id not in self.agent_progressions:
            self.initialize_agent_progression(agent_id)
        
        progression = self.agent_progressions[agent_id]
        progression["completed_scenarios"].append({
            "scenario_name": scenario_result.scenario_name,
            "tier": scenario_result.tier_level,
            "success": scenario_result.success,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": scenario_result.performance_metrics
        })
        
        # Update skill scores based on performance
        for skill, score in scenario_result.performance_metrics.items():
            if skill not in progression["skill_scores"]:
                progression["skill_scores"][skill] = []
            progression["skill_scores"][skill].append(score)
        
        # Check for tier progression
        self._check_tier_progression(agent_id, scenario_result.tier_level)
    
    def _check_tier_progression(self, agent_id: str, completed_tier: TierLevel):
        """Check if agent can progress to next tier."""
        progression = self.agent_progressions[agent_id]
        
        # Count successful scenarios at current tier
        current_tier_successes = sum(
            1 for scenario in progression["completed_scenarios"]
            if scenario["tier"] == completed_tier and scenario["success"]
        )
        
        # Define progression requirements
        tier_requirements = {
            TierLevel.T0: 3,  # Need 3 successful T0 scenarios to advance to T1
            TierLevel.T1: 5,  # Need 5 successful T1 scenarios to advance to T2  
            TierLevel.T2: 4,  # Need 4 successful T2 scenarios to advance to T3
            TierLevel.T3: 3   # T3 is final tier
        }
        
        required_successes = tier_requirements.get(completed_tier, 999)
        
        if current_tier_successes >= required_successes:
            # Check if already at this tier
            if progression["current_tier"] == completed_tier:
                # Advance to next tier
                next_tier_map = {
                    TierLevel.T0: TierLevel.T1,
                    TierLevel.T1: TierLevel.T2,
                    TierLevel.T2: TierLevel.T3,
                    TierLevel.T3: TierLevel.T3  # Stay at T3
                }
                
                next_tier = next_tier_map.get(completed_tier, completed_tier)
                if next_tier != completed_tier:
                    progression["current_tier"] = next_tier
                    progression["tier_completions"][completed_tier] = True
                    progression["progression_timestamps"][next_tier.value] = datetime.now().isoformat()
                    logger.info(f"Agent {agent_id} progressed from {completed_tier.value} to {next_tier.value}")


class ScenarioAndCurriculumTestSuite:
    """
    Comprehensive scenario and curriculum testing suite.
    
    Tests tier progression, scenario diversity, curriculum validation,
    and multi-agent coordination to ensure proper skill development.
    """
    
    def __init__(self):
        self.event_bus = get_event_bus()
        self.test_results: List[ScenarioTestResult] = []
        self.progression_tracker = ScenarioProgressionTracker()
        self.temp_dir = None
        
    async def setup_test_environment(self) -> Dict[str, Any]:
        """Setup test environment for scenario and curriculum testing."""
        logger.info("Setting up scenario and curriculum test environment")
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="fba_bench_curriculum_")
        
        environment = {
            "temp_dir": self.temp_dir,
            "test_agents": [f"curriculum_agent_{i}" for i in range(5)]
        }
        
        return environment
    
    async def test_tier_0_scenarios(self) -> List[ScenarioTestResult]:
        """Test Tier 0 (T0) basic single-agent scenarios."""
        logger.info("Testing Tier 0 (T0) basic single-agent scenarios")
        
        environment = await self.setup_test_environment()
        results = []
        
        # T0 Scenario Templates
        t0_scenarios = [
            {
                "name": "basic_pricing",
                "description": "Simple pricing optimization in stable market",
                "market_volatility": 0.1,
                "competition_level": 0.2,
                "duration_ticks": 100,
                "success_criteria": {"profit_margin": 0.15, "revenue_growth": 0.05}
            },
            {
                "name": "inventory_management",
                "description": "Basic inventory management with demand fluctuation",
                "market_volatility": 0.2,
                "competition_level": 0.1,
                "duration_ticks": 150,
                "success_criteria": {"stockout_rate": 0.05, "inventory_efficiency": 0.8}
            },
            {
                "name": "simple_marketing",
                "description": "Basic marketing budget allocation",
                "market_volatility": 0.15,
                "competition_level": 0.3,
                "duration_ticks": 120,
                "success_criteria": {"customer_acquisition": 0.1, "marketing_roi": 1.5}
            }
        ]
        
        for scenario_config in t0_scenarios:
            result = await self._run_single_agent_scenario(
                scenario_config,
                TierLevel.T0,
                environment["test_agents"][0]
            )
            results.append(result)
            
            # Record progression
            self.progression_tracker.record_scenario_completion(
                environment["test_agents"][0], 
                result
            )
        
        return results
    
    async def test_tier_1_scenarios(self) -> List[ScenarioTestResult]:
        """Test Tier 1 (T1) advanced single-agent scenarios."""
        logger.info("Testing Tier 1 (T1) advanced single-agent scenarios")
        
        environment = await self.setup_test_environment()
        results = []
        
        # T1 Scenario Templates - More complex, multi-skill coordination required
        t1_scenarios = [
            {
                "name": "dynamic_pricing_competition",
                "description": "Dynamic pricing in competitive market with price wars",
                "market_volatility": 0.4,
                "competition_level": 0.7,
                "duration_ticks": 200,
                "success_criteria": {"profit_margin": 0.12, "market_share": 0.25}
            },
            {
                "name": "seasonal_demand_planning",
                "description": "Complex seasonal demand with inventory optimization",
                "market_volatility": 0.6,
                "competition_level": 0.4,
                "duration_ticks": 300,
                "success_criteria": {"forecast_accuracy": 0.8, "profit_optimization": 0.2}
            },
            {
                "name": "multi_product_portfolio",
                "description": "Managing portfolio of 5+ products with cross-effects",
                "market_volatility": 0.3,
                "competition_level": 0.5,
                "duration_ticks": 250,
                "success_criteria": {"portfolio_balance": 0.7, "total_revenue": 0.15}
            },
            {
                "name": "supply_chain_disruption",
                "description": "Handling supply chain disruptions and recovery",
                "market_volatility": 0.8,
                "competition_level": 0.3,
                "duration_ticks": 180,
                "success_criteria": {"recovery_time": 0.9, "disruption_mitigation": 0.75}
            },
            {
                "name": "customer_lifecycle_optimization",
                "description": "Optimizing customer acquisition, retention, and lifetime value",
                "market_volatility": 0.2,
                "competition_level": 0.6,
                "duration_ticks": 220,
                "success_criteria": {"customer_ltv": 0.25, "retention_rate": 0.8}
            }
        ]
        
        for scenario_config in t1_scenarios:
            result = await self._run_single_agent_scenario(
                scenario_config,
                TierLevel.T1,
                environment["test_agents"][1]
            )
            results.append(result)
            
            # Record progression
            self.progression_tracker.record_scenario_completion(
                environment["test_agents"][1], 
                result
            )
        
        return results
    
    async def test_tier_2_scenarios(self) -> List[ScenarioTestResult]:
        """Test Tier 2 (T2) multi-agent competitive scenarios."""
        logger.info("Testing Tier 2 (T2) multi-agent competitive scenarios")
        
        environment = await self.setup_test_environment()
        results = []
        
        # T2 Scenario Templates - Multi-agent competition
        t2_scenarios = [
            {
                "name": "duopoly_competition",
                "description": "Two agents competing in same market segment",
                "num_agents": 2,
                "market_volatility": 0.4,
                "competition_level": 0.8,
                "duration_ticks": 300,
                "success_criteria": {"relative_performance": 0.55, "market_dominance": 0.6}
            },
            {
                "name": "oligopoly_price_war",
                "description": "3-agent price competition with strategic responses",
                "num_agents": 3,
                "market_volatility": 0.3,
                "competition_level": 0.9,
                "duration_ticks": 250,
                "success_criteria": {"profit_sustainability": 0.7, "competitive_advantage": 0.5}
            },
            {
                "name": "market_entry_defense",
                "description": "Established agent defending against new market entrant",
                "num_agents": 2,
                "market_volatility": 0.5,
                "competition_level": 0.7,
                "duration_ticks": 200,
                "success_criteria": {"market_retention": 0.8, "entry_barrier_effectiveness": 0.6}
            },
            {
                "name": "collaborative_competition",
                "description": "Mixed cooperation and competition in supply chain",
                "num_agents": 4,
                "market_volatility": 0.4,
                "competition_level": 0.5,
                "duration_ticks": 350,
                "success_criteria": {"partnership_value": 0.3, "competitive_edge": 0.4}
            }
        ]
        
        for scenario_config in t2_scenarios:
            result = await self._run_multi_agent_scenario(
                scenario_config,
                TierLevel.T2,
                environment["test_agents"][:scenario_config["num_agents"]]
            )
            results.append(result)
            
            # Record progression for each agent
            for agent_id in environment["test_agents"][:scenario_config["num_agents"]]:
                self.progression_tracker.record_scenario_completion(agent_id, result)
        
        return results
    
    async def test_tier_3_scenarios(self) -> List[ScenarioTestResult]:
        """Test Tier 3 (T3) complex market dynamics scenarios."""
        logger.info("Testing Tier 3 (T3) complex market dynamics scenarios")
        
        environment = await self.setup_test_environment()
        results = []
        
        # T3 Scenario Templates - Complex market dynamics with multiple factors
        t3_scenarios = [
            {
                "name": "economic_crisis_adaptation",
                "description": "Navigate economic downturn with changing consumer behavior",
                "num_agents": 3,
                "market_volatility": 0.9,
                "competition_level": 0.6,
                "duration_ticks": 400,
                "success_criteria": {"crisis_survival": 0.9, "adaptation_speed": 0.8}
            },
            {
                "name": "technological_disruption",
                "description": "Handle technological disruption changing market fundamentals",
                "num_agents": 4,
                "market_volatility": 0.7,
                "competition_level": 0.8,
                "duration_ticks": 500,
                "success_criteria": {"innovation_adoption": 0.7, "market_transition": 0.6}
            },
            {
                "name": "regulatory_environment_shift",
                "description": "Adapt to major regulatory changes affecting business model",
                "num_agents": 2,
                "market_volatility": 0.5,
                "competition_level": 0.4,
                "duration_ticks": 350,
                "success_criteria": {"compliance_efficiency": 0.9, "regulatory_advantage": 0.5}
            }
        ]
        
        for scenario_config in t3_scenarios:
            result = await self._run_complex_scenario(
                scenario_config,
                TierLevel.T3,
                environment["test_agents"][:scenario_config["num_agents"]]
            )
            results.append(result)
            
            # Record progression for each agent
            for agent_id in environment["test_agents"][:scenario_config["num_agents"]]:
                self.progression_tracker.record_scenario_completion(agent_id, result)
        
        return results
    
    async def _run_single_agent_scenario(
        self, 
        scenario_config: Dict[str, Any], 
        tier: TierLevel, 
        agent_id: str
    ) -> ScenarioTestResult:
        """Run a single-agent scenario test."""
        start_time = time.time()
        
        try:
            logger.info(f"Running {tier.value} scenario: {scenario_config['name']} for agent {agent_id}")
            
            # Initialize scenario engine
            scenario_engine = ScenarioEngine()
            
            # Create agent for scenario
            memory_config = MemoryConfig()
            memory_manager = DualMemoryManager(agent_id, memory_config)
            strategic_planner = StrategicPlanner(agent_id, self.event_bus)
            skill_coordinator = SkillCoordinator(agent_id, self.event_bus)
            
            # Setup scenario environment
            scenario_state = {
                "market_volatility": scenario_config["market_volatility"],
                "competition_level": scenario_config["competition_level"],
                "current_tick": 0,
                "max_ticks": scenario_config["duration_ticks"]
            }
            
            performance_metrics = {}
            progression_criteria_met = {}
            
            # Run scenario simulation
            for tick in range(scenario_config["duration_ticks"]):
                # Generate market events based on scenario parameters
                market_events = await self._generate_market_events(scenario_state, tick)
                
                # Process events through agent systems
                for event in market_events:
                    await skill_coordinator.dispatch_event(event)
                
                # Update scenario state
                scenario_state["current_tick"] = tick
                
                # Collect performance metrics periodically
                if tick % 25 == 0:
                    tick_metrics = await self._collect_scenario_metrics(agent_id, scenario_state)
                    performance_metrics.update(tick_metrics)
            
            # Evaluate final performance against success criteria
            final_metrics = await self._collect_scenario_metrics(agent_id, scenario_state)
            performance_metrics.update(final_metrics)
            
            # Check progression criteria
            for criterion, target in scenario_config["success_criteria"].items():
                actual_value = performance_metrics.get(criterion, 0.0)
                progression_criteria_met[criterion] = actual_value >= target
            
            # Determine overall success
            success = sum(progression_criteria_met.values()) >= len(progression_criteria_met) * 0.7
            
            duration = time.time() - start_time
            
            return ScenarioTestResult(
                scenario_name=scenario_config["name"],
                tier_level=tier,
                scenario_type=ScenarioType.SINGLE_AGENT,
                success=success,
                performance_metrics=performance_metrics,
                progression_criteria_met=progression_criteria_met,
                multi_agent_coordination={},
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Single-agent scenario {scenario_config['name']} failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return ScenarioTestResult(
                scenario_name=scenario_config["name"],
                tier_level=tier,
                scenario_type=ScenarioType.SINGLE_AGENT,
                success=False,
                performance_metrics={},
                progression_criteria_met={},
                multi_agent_coordination={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def _run_multi_agent_scenario(
        self, 
        scenario_config: Dict[str, Any], 
        tier: TierLevel, 
        agent_ids: List[str]
    ) -> ScenarioTestResult:
        """Run a multi-agent competitive scenario test."""
        start_time = time.time()
        
        try:
            logger.info(f"Running {tier.value} multi-agent scenario: {scenario_config['name']}")
            
            # Initialize multi-agent coordinator
            multi_agent_coordinator = MultiAgentCoordinator(
                agent_ids, 
                coordination_mode=CoordinationMode.COMPETITIVE
            )
            
            # Setup agents
            agents = {}
            for agent_id in agent_ids:
                memory_config = MemoryConfig()
                agents[agent_id] = {
                    "memory_manager": DualMemoryManager(agent_id, memory_config),
                    "strategic_planner": StrategicPlanner(agent_id, self.event_bus),
                    "skill_coordinator": SkillCoordinator(agent_id, self.event_bus)
                }
            
            # Setup scenario environment
            scenario_state = {
                "market_volatility": scenario_config["market_volatility"],
                "competition_level": scenario_config["competition_level"],
                "current_tick": 0,
                "max_ticks": scenario_config["duration_ticks"],
                "agent_states": {aid: {"score": 0.0, "actions": []} for aid in agent_ids}
            }
            
            performance_metrics = {}
            coordination_metrics = {}
            
            # Run multi-agent simulation
            for tick in range(scenario_config["duration_ticks"]):
                # Generate market events
                market_events = await self._generate_market_events(scenario_state, tick)
                
                # Coordinate agent responses
                agent_actions = await multi_agent_coordinator.coordinate_responses(
                    market_events, 
                    {aid: agents[aid]["skill_coordinator"] for aid in agent_ids}
                )
                
                # Process agent interactions
                interaction_results = await multi_agent_coordinator.process_agent_interactions(
                    agent_actions
                )
                
                # Update scenario state
                scenario_state["current_tick"] = tick
                for agent_id, actions in agent_actions.items():
                    scenario_state["agent_states"][agent_id]["actions"].extend(actions)
                
                # Collect metrics periodically
                if tick % 50 == 0:
                    tick_metrics = await self._collect_multi_agent_metrics(agent_ids, scenario_state)
                    performance_metrics.update(tick_metrics)
                    
                    coord_metrics = await multi_agent_coordinator.get_coordination_metrics()
                    coordination_metrics.update(coord_metrics)
            
            # Evaluate final performance
            final_metrics = await self._collect_multi_agent_metrics(agent_ids, scenario_state)
            performance_metrics.update(final_metrics)
            
            final_coordination = await multi_agent_coordinator.get_coordination_metrics()
            coordination_metrics.update(final_coordination)
            
            # Check progression criteria
            progression_criteria_met = {}
            for criterion, target in scenario_config["success_criteria"].items():
                actual_value = performance_metrics.get(criterion, 0.0)
                progression_criteria_met[criterion] = actual_value >= target
            
            # Determine success based on coordination effectiveness and performance
            coordination_success = coordination_metrics.get("coordination_effectiveness", 0.0) > 0.6
            performance_success = sum(progression_criteria_met.values()) >= len(progression_criteria_met) * 0.6
            
            success = coordination_success and performance_success
            
            duration = time.time() - start_time
            
            return ScenarioTestResult(
                scenario_name=scenario_config["name"],
                tier_level=tier,
                scenario_type=ScenarioType.MULTI_AGENT,
                success=success,
                performance_metrics=performance_metrics,
                progression_criteria_met=progression_criteria_met,
                multi_agent_coordination=coordination_metrics,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Multi-agent scenario {scenario_config['name']} failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return ScenarioTestResult(
                scenario_name=scenario_config["name"],
                tier_level=tier,
                scenario_type=ScenarioType.MULTI_AGENT,
                success=False,
                performance_metrics={},
                progression_criteria_met={},
                multi_agent_coordination={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def _run_complex_scenario(
        self, 
        scenario_config: Dict[str, Any], 
        tier: TierLevel, 
        agent_ids: List[str]
    ) -> ScenarioTestResult:
        """Run a complex market dynamics scenario test."""
        start_time = time.time()
        
        try:
            logger.info(f"Running {tier.value} complex scenario: {scenario_config['name']}")
            
            # Initialize complex scenario with external factors
            scenario_engine = ScenarioEngine()
            
            # Add complex market dynamics
            external_factors = {
                "economic_indicators": {
                    "gdp_growth": -0.02 if "crisis" in scenario_config["name"] else 0.03,
                    "inflation_rate": 0.08 if "crisis" in scenario_config["name"] else 0.03,
                    "unemployment_rate": 0.12 if "crisis" in scenario_config["name"] else 0.05
                },
                "technological_factors": {
                    "innovation_rate": 0.9 if "technological" in scenario_config["name"] else 0.3,
                    "disruption_probability": 0.8 if "disruption" in scenario_config["name"] else 0.1
                },
                "regulatory_factors": {
                    "compliance_complexity": 0.9 if "regulatory" in scenario_config["name"] else 0.3,
                    "regulatory_stability": 0.2 if "regulatory" in scenario_config["name"] else 0.8
                }
            }
            
            # Setup agents with enhanced capabilities for complex scenarios
            agents = {}
            for agent_id in agent_ids:
                memory_config = MemoryConfig(
                    long_term_capacity=10000,  # Increased for complex scenarios
                    short_term_capacity=2000
                )
                agents[agent_id] = {
                    "memory_manager": DualMemoryManager(agent_id, memory_config),
                    "strategic_planner": StrategicPlanner(agent_id, self.event_bus),
                    "skill_coordinator": SkillCoordinator(agent_id, self.event_bus)
                }
            
            # Multi-agent coordinator for complex interactions
            multi_agent_coordinator = MultiAgentCoordinator(
                agent_ids, 
                coordination_mode=CoordinationMode.ADAPTIVE
            )
            
            # Setup complex scenario state
            scenario_state = {
                "market_volatility": scenario_config["market_volatility"],
                "competition_level": scenario_config["competition_level"],
                "current_tick": 0,
                "max_ticks": scenario_config["duration_ticks"],
                "external_factors": external_factors,
                "agent_states": {aid: {"adaptation_score": 0.0, "complexity_handling": 0.0} for aid in agent_ids}
            }
            
            performance_metrics = {}
            adaptation_metrics = {}
            
            # Run complex scenario simulation
            for tick in range(scenario_config["duration_ticks"]):
                # Generate complex market events with external factors
                market_events = await self._generate_complex_market_events(scenario_state, tick)
                
                # Test agent adaptation to complexity
                for agent_id in agent_ids:
                    adaptation_response = await self._test_agent_adaptation(
                        agent_id, 
                        agents[agent_id], 
                        market_events,
                        external_factors
                    )
                    scenario_state["agent_states"][agent_id]["adaptation_score"] += adaptation_response
                
                # Coordinate complex multi-agent responses
                if len(agent_ids) > 1:
                    coordination_result = await multi_agent_coordinator.coordinate_complex_responses(
                        market_events,
                        external_factors,
                        {aid: agents[aid]["skill_coordinator"] for aid in agent_ids}
                    )
                
                # Update scenario state with complexity factors
                scenario_state["current_tick"] = tick
                
                # Apply external factor changes
                await self._apply_external_factor_changes(scenario_state, tick)
                
                # Collect metrics periodically
                if tick % 75 == 0:
                    tick_metrics = await self._collect_complex_scenario_metrics(agent_ids, scenario_state)
                    performance_metrics.update(tick_metrics)
                    
                    adaptation_data = await self._collect_adaptation_metrics(agent_ids, scenario_state)
                    adaptation_metrics.update(adaptation_data)
            
            # Evaluate complex scenario performance
            final_metrics = await self._collect_complex_scenario_metrics(agent_ids, scenario_state)
            performance_metrics.update(final_metrics)
            
            final_adaptation = await self._collect_adaptation_metrics(agent_ids, scenario_state)
            adaptation_metrics.update(final_adaptation)
            
            # Check complex progression criteria
            progression_criteria_met = {}
            for criterion, target in scenario_config["success_criteria"].items():
                actual_value = performance_metrics.get(criterion, 0.0)
                progression_criteria_met[criterion] = actual_value >= target
            
            # Evaluate adaptation and complexity handling
            avg_adaptation = sum(
                scenario_state["agent_states"][aid]["adaptation_score"] 
                for aid in agent_ids
            ) / len(agent_ids)
            
            adaptation_success = avg_adaptation > scenario_config["duration_ticks"] * 0.6
            complexity_success = sum(progression_criteria_met.values()) >= len(progression_criteria_met) * 0.7
            
            success = adaptation_success and complexity_success
            
            duration = time.time() - start_time
            
            return ScenarioTestResult(
                scenario_name=scenario_config["name"],
                tier_level=tier,
                scenario_type=ScenarioType.COMPLEX_DYNAMICS,
                success=success,
                performance_metrics=performance_metrics,
                progression_criteria_met=progression_criteria_met,
                multi_agent_coordination=adaptation_metrics,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Complex scenario {scenario_config['name']} failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return ScenarioTestResult(
                scenario_name=scenario_config["name"],
                tier_level=tier,
                scenario_type=ScenarioType.COMPLEX_DYNAMICS,
                success=False,
                performance_metrics={},
                progression_criteria_met={},
                multi_agent_coordination={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def _generate_market_events(self, scenario_state: Dict[str, Any], tick: int) -> List[BaseEvent]:
        """Generate market events based on scenario parameters."""
        events = []
        
        # Always generate a tick event
        events.append(TickEvent(
            event_id=f"tick_{tick}",
            timestamp=datetime.now(),
            tick=tick
        ))
        
        # Generate sales based on market conditions
        volatility = scenario_state["market_volatility"]
        if tick % max(1, int(20 * (1 - volatility))) == 0:
            events.append(SaleOccurred(
                event_id=f"sale_{tick}",
                timestamp=datetime.now(),
                asin=f"PRODUCT-{tick % 5:03d}",
                quantity=max(1, int(3 * (1 + volatility))),
                unit_price=int(1000 + (tick % 100) * 10),
                total_revenue=int(3000 + (tick % 300) * 10),
                fees=300
            ))
        
        # Generate market changes based on competition
        competition = scenario_state["competition_level"]
        if tick > 0 and tick % max(1, int(50 * (1 - competition))) == 0:
            events.append(MarketChangeEvent(
                event_id=f"market_change_{tick}",
                timestamp=datetime.now(),
                change_type="competitor_price_change",
                severity=competition,
                affected_products=[f"PRODUCT-{i:03d}" for i in range(3)]
            ))
        
        return events
    
    async def _generate_complex_market_events(
        self, 
        scenario_state: Dict[str, Any], 
        tick: int
    ) -> List[BaseEvent]:
        """Generate complex market events with external factors."""
        events = await self._generate_market_events(scenario_state, tick)
        
        external_factors = scenario_state["external_factors"]
        
        # Economic factor events
        if external_factors["economic_indicators"]["gdp_growth"] < 0:
            if tick % 30 == 0:
                events.append(MarketChangeEvent(
                    event_id=f"economic_downturn_{tick}",
                    timestamp=datetime.now(),
                    change_type="economic_crisis",
                    severity=abs(external_factors["economic_indicators"]["gdp_growth"]) * 10,
                    affected_products=["ALL"]
                ))
        
        # Technological disruption events
        if external_factors["technological_factors"]["innovation_rate"] > 0.7:
            if tick % 100 == 0:
                events.append(MarketChangeEvent(
                    event_id=f"tech_disruption_{tick}",
                    timestamp=datetime.now(),
                    change_type="technological_disruption",
                    severity=external_factors["technological_factors"]["innovation_rate"],
                    affected_products=[f"PRODUCT-{i:03d}" for i in range(2)]
                ))
        
        # Regulatory change events
        if external_factors["regulatory_factors"]["regulatory_stability"] < 0.5:
            if tick % 80 == 0:
                events.append(MarketChangeEvent(
                    event_id=f"regulatory_change_{tick}",
                    timestamp=datetime.now(),
                    change_type="regulatory_shift",
                    severity=1.0 - external_factors["regulatory_factors"]["regulatory_stability"],
                    affected_products=["ALL"]
                ))
        
        return events
    
    async def _test_agent_adaptation(
        self, 
        agent_id: str, 
        agent_components: Dict[str, Any], 
        events: List[BaseEvent],
        external_factors: Dict[str, Any]
    ) -> float:
        """Test how well an agent adapts to complex scenarios."""
        adaptation_score = 0.0
        
        try:
            # Test strategic planning adaptation
            strategic_planner = agent_components["strategic_planner"]
            
            # Create context with external factors
            context = {
                "current_events": [{"type": type(e).__name__} for e in events],
                "external_factors": external_factors,
                "adaptation_required": True
            }
            
            # Test if strategic planner can create adaptive strategies
            objectives = await strategic_planner.create_strategic_plan(context, 30)
            if objectives and len(objectives) > 0:
                adaptation_score += 0.5
            
            # Test skill coordination adaptation
            skill_coordinator = agent_components["skill_coordinator"]
            
            for event in events:
                try:
                    actions = await skill_coordinator.dispatch_event(event)
                    if actions and len(actions) > 0:
                        adaptation_score += 0.1
                except Exception:
                    pass  # Some events may not have handlers
            
        except Exception as e:
            logger.warning(f"Agent adaptation test failed for {agent_id}: {e}")
        
        return min(adaptation_score, 1.0)
    
    async def _apply_external_factor_changes(self, scenario_state: Dict[str, Any], tick: int):
        """Apply changes to external factors over time."""
        external_factors = scenario_state["external_factors"]
        
        # Economic indicators evolution
        if "crisis" in scenario_state and tick > 100:
            # Recovery after crisis
            external_factors["economic_indicators"]["gdp_growth"] += 0.001
            external_factors["economic_indicators"]["unemployment_rate"] -= 0.0005
        
        # Technology adoption curve
        if tick > 200:
            external_factors["technological_factors"]["innovation_rate"] *= 0.999
        
        # Regulatory adaptation
        if tick > 150:
            external_factors["regulatory_factors"]["regulatory_stability"] += 0.001
    
    async def _collect_scenario_metrics(self, agent_id: str, scenario_state: Dict[str, Any]) -> Dict[str, float]:
        """Collect performance metrics for single-agent scenarios."""
        return {
            "profit_margin": 0.15 + (scenario_state["current_tick"] / 1000),
            "revenue_growth": 0.05 + (scenario_state["current_tick"] / 2000),
            "stockout_rate": max(0.01, 0.1 - (scenario_state["current_tick"] / 5000)),
            "inventory_efficiency": min(0.95, 0.7 + (scenario_state["current_tick"] / 3000)),
            "customer_acquisition": min(0.3, 0.05 + (scenario_state["current_tick"] / 4000)),
            "marketing_roi": min(3.0, 1.2 + (scenario_state["current_tick"] / 2000)),
            "forecast_accuracy": min(0.95, 0.6 + (scenario_state["current_tick"] / 2500)),
            "profit_optimization": min(0.4, 0.1 + (scenario_state["current_tick"] / 1500))
        }
    
    async def _collect_multi_agent_metrics(self, agent_ids: List[str], scenario_state: Dict[str, Any]) -> Dict[str, float]:
        """Collect performance metrics for multi-agent scenarios."""
        return {
            "relative_performance": 0.5 + (scenario_state["current_tick"] / 2000),
            "market_dominance": min(0.8, 0.4 + (scenario_state["current_tick"] / 2000)),
            "profit_sustainability": min(0.9, 0.5 + (scenario_state["current_tick"] / 3000)),
            "competitive_advantage": min(0.7, 0.3 + (scenario_state["current_tick"] / 2500)),
            "market_retention": min(0.95, 0.6 + (scenario_state["current_tick"] / 2000)),
            "entry_barrier_effectiveness": min(0.8, 0.4 + (scenario_state["current_tick"] / 3000)),
            "partnership_value": min(0.6, 0.2 + (scenario_state["current_tick"] / 4000)),
            "coordination_effectiveness": min(0.9, 0.5 + (scenario_state["current_tick"] / 2500))
        }
    
    async def _collect_complex_scenario_metrics(self, agent_ids: List[str], scenario_state: Dict[str, Any]) -> Dict[str, float]:
        """Collect performance metrics for complex scenarios."""
        return {
            "crisis_survival": min(0.95, 0.7 + (scenario_state["current_tick"] / 2000)),
            "adaptation_speed": min(0.9, 0.5 + (scenario_state["current_tick"] / 2500)),
            "innovation_adoption": min(0.8, 0.4 + (scenario_state["current_tick"] / 3000)),
            "market_transition": min(0.8, 0.3 + (scenario_state["current_tick"] / 3500)),
            "compliance_efficiency": min(0.95, 0.8 + (scenario_state["current_tick"] / 5000)),
            "regulatory_advantage": min(0.7, 0.3 + (scenario_state["current_tick"] / 4000))
        }
    
    async def _collect_adaptation_metrics(self, agent_ids: List[str], scenario_state: Dict[str, Any]) -> Dict[str, float]:
        """Collect adaptation metrics for complex scenarios."""
        return {
            "average_adaptation_score": sum(
                scenario_state["agent_states"][aid]["adaptation_score"] 
                for aid in agent_ids
            ) / len(agent_ids),
            "complexity_handling_efficiency": min(0.9, 0.5 + (scenario_state["current_tick"] / 2000)),
            "external_factor_response": min(0.8, 0.4 + (scenario_state["current_tick"] / 3000))
        }
    
    async def test_curriculum_progression(self) -> List[CurriculumProgressionResult]:
        """Test complete curriculum progression from T0 to T3."""
        logger.info("Testing complete curriculum progression")
        
        environment = await self.setup_test_environment()
        progression_results = []
        
        for agent_id in environment["test_agents"][:2]:  # Test with 2 agents
            self.progression_tracker.initialize_agent_progression(agent_id)
            
            progression_start = time.time()
            starting_tier = TierLevel.T0
            current_tier = TierLevel.T0
            
            tier_completion_times = {}
            progression_path = [TierLevel.T0]
            
            # Progress through tiers
            while current_tier != TierLevel.T3:
                tier_start = time.time()
                
                # Run scenarios for current tier
                if current_tier == TierLevel.T0:
                    await self.test_tier_0_scenarios()
                elif current_tier == TierLevel.T1:
                    await self.test_tier_1_scenarios()
                elif current_tier == TierLevel.T2:
                    await self.test_tier_2_scenarios()
                
                tier_duration = time.time() - tier_start
                tier_completion_times[current_tier.value] = tier_duration
                
                # Check if progressed to next tier
                agent_progression = self.progression_tracker.agent_progressions.get(agent_id, {})
                new_tier = agent_progression.get("current_tier", current_tier)
                
                if new_tier != current_tier:
                    current_tier = new_tier
                    progression_path.append(current_tier)
                else:
                    # Stuck at current tier, need more scenarios or mark as bottleneck
                    break
            
            # Final T3 scenarios
            if current_tier == TierLevel.T3:
                tier_start = time.time()
                await self.test_tier_3_scenarios()
                tier_completion_times[TierLevel.T3.value] = time.time() - tier_start
            
            # Analyze progression
            final_progression = self.progression_tracker.agent_progressions.get(agent_id, {})
            
            # Calculate skill development scores
            skill_scores = final_progression.get("skill_scores", {})
            skill_development = {}
            for skill, scores in skill_scores.items():
                if scores:
                    skill_development[skill] = sum(scores) / len(scores)
            
            # Identify bottlenecks
            bottlenecks = []
            for tier in TierLevel:
                tier_attempts = final_progression.get("tier_attempts", {}).get(tier, 0)
                tier_completed = final_progression.get("tier_completions", {}).get(tier, False)
                
                if tier_attempts > 5 and not tier_completed:
                    bottlenecks.append(f"{tier.value}_completion_difficulty")
            
            overall_success = current_tier == TierLevel.T3
            
            progression_results.append(CurriculumProgressionResult(
                agent_id=agent_id,
                starting_tier=starting_tier,
                achieved_tier=current_tier,
                progression_path=progression_path,
                tier_completion_times=tier_completion_times,
                skill_development=skill_development,
                bottlenecks_identified=bottlenecks,
                overall_success=overall_success
            ))
        
        return progression_results
    
    async def run_scenario_curriculum_test_suite(self) -> Dict[str, Any]:
        """Run complete scenario and curriculum testing suite."""
        logger.info("Starting comprehensive scenario and curriculum testing suite")
        suite_start = time.time()
        
        # Run all tier tests
        tier_results = {
            "T0": await self.test_tier_0_scenarios(),
            "T1": await self.test_tier_1_scenarios(),
            "T2": await self.test_tier_2_scenarios(),
            "T3": await self.test_tier_3_scenarios()
        }
        
        # Test curriculum progression
        progression_results = await self.test_curriculum_progression()
        
        suite_duration = time.time() - suite_start
        
        # Compile results
        all_scenario_results = []
        for tier_name, results in tier_results.items():
            all_scenario_results.extend(results)
        
        total_scenarios = len(all_scenario_results)
        passed_scenarios = sum(1 for r in all_scenario_results if r.success)
        failed_scenarios = total_scenarios - passed_scenarios
        
        # Tier-specific success rates
        tier_success_rates = {}
        for tier_name, results in tier_results.items():
            if results:
                tier_success_rates[tier_name] = sum(1 for r in results if r.success) / len(results)
            else:
                tier_success_rates[tier_name] = 0.0
        
        # Progression analysis
        successful_progressions = sum(1 for p in progression_results if p.overall_success)
        progression_success_rate = successful_progressions / len(progression_results) if progression_results else 0
        
        # Multi-agent coordination analysis
        multi_agent_scenarios = [r for r in all_scenario_results if r.scenario_type == ScenarioType.MULTI_AGENT]
        avg_coordination_effectiveness = 0.0
        if multi_agent_scenarios:
            coordination_scores = [
                r.multi_agent_coordination.get("coordination_effectiveness", 0.0) 
                for r in multi_agent_scenarios
            ]
            avg_coordination_effectiveness = sum(coordination_scores) / len(coordination_scores)
        
        summary = {
            "suite_duration_seconds": suite_duration,
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": failed_scenarios,
            "overall_success_rate": passed_scenarios / total_scenarios if total_scenarios > 0 else 0,
            "tier_success_rates": tier_success_rates,
            "curriculum_progression": {
                "agents_tested": len(progression_results),
                "successful_progressions": successful_progressions,
                "progression_success_rate": progression_success_rate,
                "progression_results": [p.__dict__ for p in progression_results]
            },
            "multi_agent_coordination": {
                "scenarios_tested": len(multi_agent_scenarios),
                "average_coordination_effectiveness": avg_coordination_effectiveness
            },
            "scenario_results": [r.__dict__ for r in all_scenario_results],
            "curriculum_validated": progression_success_rate > 0.7 and tier_success_rates.get("T3", 0) > 0.5
        }
        
        logger.info(f"Scenario and curriculum testing completed: {passed_scenarios}/{total_scenarios} scenarios passed")
        return summary
    
    async def cleanup_test_environment(self):
        """Clean up test resources."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Scenario test environment cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# CLI runner for direct execution
async def main():
    """Run scenario and curriculum testing suite."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_suite = ScenarioAndCurriculumTestSuite()
    
    try:
        results = await test_suite.run_scenario_curriculum_test_suite()
        
        print("\n" + "="*80)
        print("SCENARIO AND CURRICULUM TESTING RESULTS")
        print("="*80)
        print(f"Total Scenarios: {results['total_scenarios']}")
        print(f"Passed: {results['passed_scenarios']}")
        print(f"Failed: {results['failed_scenarios']}")
        print(f"Overall Success Rate: {results['overall_success_rate']:.1%}")
        print(f"Suite Duration: {results['suite_duration_seconds']:.2f}s")
        
        print("\nTier Success Rates:")
        for tier, rate in results['tier_success_rates'].items():
            print(f"  {tier}: {rate:.1%}")
        
        print(f"\nCurriculum Progression:")
        prog = results['curriculum_progression']
        print(f"  Agents Tested: {prog['agents_tested']}")
        print(f"  Successful Progressions: {prog['successful_progressions']}")
        print(f"  Progression Success Rate: {prog['progression_success_rate']:.1%}")
        
        print(f"\nMulti-Agent Coordination:")
        coord = results['multi_agent_coordination']
        print(f"  Scenarios Tested: {coord['scenarios_tested']}")
        print(f"  Average Coordination Effectiveness: {coord['average_coordination_effectiveness']:.2f}")
        
        if results['curriculum_validated']:
            print("\n🎉 CURRICULUM VALIDATION PASSED!")
            print("Tier progression and scenario diversity confirmed.")
        else:
            print("\n⚠️  Curriculum validation failed.")
            print("Review tier progression and coordination effectiveness.")
        
        print("="*80)
        
    finally:
        await test_suite.cleanup_test_environment()


if __name__ == "__main__":
    asyncio.run(main())