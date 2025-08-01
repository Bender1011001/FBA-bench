"""
Hierarchical Planning System for FBA-Bench Cognitive Architecture

Implements strategic and tactical planning capabilities to enable agents
to formulate long-term strategies and break them down into executable actions.
"""

import asyncio
import logging
import uuid
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from event_bus import EventBus, get_event_bus
from events import BaseEvent

logger = logging.getLogger(__name__)


class PlanPriority(Enum):
    """Priority levels for strategic and tactical plans."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PlanStatus(Enum):
    """Status of plans throughout their lifecycle."""
    DRAFT = "draft"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanType(Enum):
    """Types of strategic plans."""
    GROWTH = "growth"
    OPTIMIZATION = "optimization"
    DEFENSIVE = "defensive"
    EXPLORATORY = "exploratory"
    RECOVERY = "recovery"


@dataclass
class StrategicObjective:
    """A high-level strategic objective with measurable outcomes."""
    objective_id: str
    title: str
    description: str
    target_metrics: Dict[str, float]  # e.g., {"profit_margin": 0.25, "market_share": 0.15}
    timeframe_days: int
    priority: PlanPriority
    status: PlanStatus
    created_at: datetime
    target_completion: datetime
    progress_indicators: Dict[str, float] = None
    dependencies: List[str] = None  # Other objective IDs this depends on
    
    def __post_init__(self):
        if self.progress_indicators is None:
            self.progress_indicators = {}
        if self.dependencies is None:
            self.dependencies = []
    
    def calculate_progress(self, current_metrics: Dict[str, float]) -> float:
        """Calculate overall progress towards objective completion."""
        if not self.target_metrics:
            return 0.0
        
        progress_scores = []
        for metric, target_value in self.target_metrics.items():
            current_value = current_metrics.get(metric, 0.0)
            if target_value == 0:
                continue
            
            # Calculate progress as percentage toward target
            progress = min(1.0, current_value / target_value)
            progress_scores.append(progress)
        
        return sum(progress_scores) / len(progress_scores) if progress_scores else 0.0
    
    def is_overdue(self, current_time: datetime) -> bool:
        """Check if objective is overdue."""
        return current_time > self.target_completion and self.status not in [PlanStatus.COMPLETED, PlanStatus.CANCELLED]


@dataclass
class TacticalAction:
    """A specific action that serves strategic objectives."""
    action_id: str
    title: str
    description: str
    action_type: str  # e.g., "set_price", "place_order", "run_marketing_campaign"
    parameters: Dict[str, Any]
    strategic_objective_id: str
    priority: PlanPriority
    status: PlanStatus
    created_at: datetime
    scheduled_execution: datetime
    estimated_duration_hours: float
    expected_impact: Dict[str, float]  # Expected impact on metrics
    prerequisites: List[str] = None  # Other action IDs that must complete first
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
    
    def is_ready_for_execution(self, current_time: datetime, completed_actions: List[str]) -> bool:
        """Check if action is ready for execution."""
        if self.status != PlanStatus.ACTIVE:
            return False
        
        # Check if scheduled time has arrived
        if current_time < self.scheduled_execution:
            return False
        
        # Check if all prerequisites are completed
        for prereq in self.prerequisites:
            if prereq not in completed_actions:
                return False
        
        return True


class StrategicPlanner:
    """
    Manages high-level quarterly/phase goals and strategic direction.
    
    Responsible for creating, updating, and validating strategic plans
    that guide the agent's long-term decision-making.
    """
    
    def __init__(self, agent_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the Strategic Planner.
        
        Args:
            agent_id: Unique identifier for the agent
            event_bus: Event bus for publishing strategic events
        """
        self.agent_id = agent_id
        self.event_bus = event_bus or get_event_bus()
        
        # Strategic state
        self.strategic_objectives: Dict[str, StrategicObjective] = {}
        self.current_strategy_type: Optional[PlanType] = None
        self.strategy_created_at: Optional[datetime] = None
        self.last_strategy_review: Optional[datetime] = None
        
        # Performance tracking
        self.strategy_performance_history: List[Dict[str, Any]] = []
        self.external_events_impact: List[Dict[str, Any]] = []
        
        logger.info(f"StrategicPlanner initialized for agent {agent_id}")
    
    async def create_strategic_plan(self, context: Dict[str, Any], timeframe: int) -> Dict[str, StrategicObjective]:
        """
        Generate strategic objectives based on current context and timeframe.
        
        Args:
            context: Current business context including metrics, market conditions, etc.
            timeframe: Planning horizon in days
            
        Returns:
            Dictionary of strategic objectives with their IDs as keys
        """
        logger.info(f"Creating strategic plan for agent {self.agent_id} with {timeframe}-day horizon")
        
        current_time = datetime.now()
        strategy_type = self._determine_strategy_type(context)
        
        # Clear existing objectives if starting fresh strategic cycle
        if self._should_create_new_strategy(current_time, strategy_type):
            await self._archive_completed_objectives()
            self.strategic_objectives.clear()
            self.current_strategy_type = strategy_type
            self.strategy_created_at = current_time
        
        # Generate strategic objectives based on strategy type
        objectives = await self._generate_objectives_for_strategy(
            strategy_type, context, timeframe, current_time
        )
        
        # Validate objective dependencies and feasibility
        validated_objectives = await self._validate_strategic_objectives(objectives, context)
        
        # Store objectives and publish strategic plan event
        for obj_id, objective in validated_objectives.items():
            self.strategic_objectives[obj_id] = objective
        
        await self._publish_strategic_plan_created_event(validated_objectives)
        
        logger.info(f"Created strategic plan with {len(validated_objectives)} objectives")
        return validated_objectives
    
    async def update_strategic_plan(self, current_performance: Dict[str, float], 
                                   external_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adapt strategy based on performance and external events.
        
        Args:
            current_performance: Current metrics and KPIs
            external_events: List of external events that may impact strategy
            
        Returns:
            Dictionary containing update results and any new/modified objectives
        """
        logger.info(f"Updating strategic plan for agent {self.agent_id}")
        
        current_time = datetime.now()
        update_results = {
            "objectives_modified": [],
            "objectives_added": [],
            "objectives_cancelled": [],
            "strategy_changes": [],
            "performance_assessment": {}
        }
        
        # Assess performance against current objectives
        performance_assessment = await self._assess_strategic_performance(current_performance)
        update_results["performance_assessment"] = performance_assessment
        
        # Analyze impact of external events
        event_impact = await self._analyze_external_events_impact(external_events)
        self.external_events_impact.extend(event_impact)
        
        # Update objective priorities and status based on performance
        for obj_id, objective in self.strategic_objectives.items():
            progress = objective.calculate_progress(current_performance)
            objective.progress_indicators = {"overall_progress": progress}
            
            # Modify objectives based on performance and events
            modifications = await self._determine_objective_modifications(
                objective, performance_assessment, event_impact
            )
            
            if modifications:
                update_results["objectives_modified"].append({
                    "objective_id": obj_id,
                    "modifications": modifications
                })
                await self._apply_objective_modifications(objective, modifications)
        
        # Add new objectives if needed based on external events
        new_objectives = await self._identify_new_objectives_needed(
            external_events, current_performance
        )
        
        for new_obj in new_objectives:
            obj_id = new_obj.objective_id
            self.strategic_objectives[obj_id] = new_obj
            update_results["objectives_added"].append(obj_id)
        
        # Cancel objectives that are no longer viable
        cancelled_objectives = await self._identify_objectives_to_cancel(
            current_performance, external_events
        )
        
        for obj_id in cancelled_objectives:
            if obj_id in self.strategic_objectives:
                self.strategic_objectives[obj_id].status = PlanStatus.CANCELLED
                update_results["objectives_cancelled"].append(obj_id)
        
        # Update strategy performance history
        self.strategy_performance_history.append({
            "timestamp": current_time.isoformat(),
            "performance_metrics": current_performance,
            "objectives_count": len(self.strategic_objectives),
            "avg_progress": performance_assessment.get("average_progress", 0.0)
        })
        
        self.last_strategy_review = current_time
        await self._publish_strategic_plan_updated_event(update_results)
        
        logger.info(f"Strategic plan updated: {len(update_results['objectives_modified'])} modified, "
                   f"{len(update_results['objectives_added'])} added, "
                   f"{len(update_results['objectives_cancelled'])} cancelled")
        
        return update_results
    
    async def validate_action_alignment(self, proposed_action: Dict[str, Any], 
                                       current_strategy: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, str]:
        """
        Check if a proposed action aligns with current strategic objectives.
        
        Args:
            proposed_action: Action to validate (contains type, parameters, expected_impact)
            current_strategy: Optional strategy context for validation
            
        Returns:
            Tuple of (is_aligned, alignment_score, reasoning)
        """
        if not self.strategic_objectives:
            return True, 0.5, "No strategic objectives defined - action allowed by default"
        
        action_type = proposed_action.get("type", "unknown")
        expected_impact = proposed_action.get("expected_impact", {})
        
        alignment_scores = []
        supporting_objectives = []
        
        # Check alignment against each active strategic objective
        for obj_id, objective in self.strategic_objectives.items():
            if objective.status != PlanStatus.ACTIVE:
                continue
            
            alignment_score = await self._calculate_action_objective_alignment(
                proposed_action, objective
            )
            
            if alignment_score > 0.3:  # Threshold for meaningful alignment
                alignment_scores.append(alignment_score)
                supporting_objectives.append(obj_id)
        
        if not alignment_scores:
            return False, 0.0, f"Action '{action_type}' does not align with any active strategic objectives"
        
        overall_alignment = max(alignment_scores)  # Use best alignment score
        
        is_aligned = overall_alignment >= 0.6  # Alignment threshold
        
        reasoning = f"Action '{action_type}' alignment score: {overall_alignment:.2f}. "
        if is_aligned:
            reasoning += f"Supports objectives: {supporting_objectives}"
        else:
            reasoning += "Insufficient alignment with strategic goals"
        
        return is_aligned, overall_alignment, reasoning
    
    def get_strategic_status(self) -> Dict[str, Any]:
        """Get comprehensive status of strategic planning."""
        current_time = datetime.now()
        
        active_objectives = [obj for obj in self.strategic_objectives.values() 
                           if obj.status == PlanStatus.ACTIVE]
        
        overdue_objectives = [obj for obj in active_objectives 
                            if obj.is_overdue(current_time)]
        
        status = {
            "agent_id": self.agent_id,
            "current_strategy_type": self.current_strategy_type.value if self.current_strategy_type else None,
            "strategy_age_days": (current_time - self.strategy_created_at).days if self.strategy_created_at else 0,
            "total_objectives": len(self.strategic_objectives),
            "active_objectives": len(active_objectives),
            "overdue_objectives": len(overdue_objectives),
            "last_review": self.last_strategy_review.isoformat() if self.last_strategy_review else None,
            "performance_history_entries": len(self.strategy_performance_history)
        }
        
        if active_objectives:
            # Calculate average progress
            progress_scores = []
            for obj in active_objectives:
                if obj.progress_indicators:
                    progress_scores.append(obj.progress_indicators.get("overall_progress", 0.0))
            
            status["average_progress"] = sum(progress_scores) / len(progress_scores) if progress_scores else 0.0
            
            # Identify highest priority objectives
            high_priority_objectives = [obj for obj in active_objectives 
                                      if obj.priority in [PlanPriority.HIGH, PlanPriority.CRITICAL]]
            status["high_priority_objectives"] = len(high_priority_objectives)
        
        return status
    
    # Private helper methods
    
    def _determine_strategy_type(self, context: Dict[str, Any]) -> PlanType:
        """Determine appropriate strategy type based on context."""
        current_metrics = context.get("current_metrics", {})
        market_conditions = context.get("market_conditions", {})
        
        profit_margin = current_metrics.get("profit_margin", 0.0)
        revenue_growth = current_metrics.get("revenue_growth", 0.0)
        competitive_pressure = market_conditions.get("competitive_pressure", 0.5)
        market_volatility = market_conditions.get("volatility", 0.5)
        
        # Decision logic for strategy type
        if profit_margin < 0.1 and revenue_growth < 0.05:
            return PlanType.RECOVERY
        elif competitive_pressure > 0.7:
            return PlanType.DEFENSIVE
        elif revenue_growth > 0.15 and profit_margin > 0.15:
            return PlanType.GROWTH
        elif market_volatility > 0.6:
            return PlanType.EXPLORATORY
        else:
            return PlanType.OPTIMIZATION
    
    def _should_create_new_strategy(self, current_time: datetime, strategy_type: PlanType) -> bool:
        """Determine if a new strategy should be created."""
        if not self.strategy_created_at:
            return True
        
        strategy_age_days = (current_time - self.strategy_created_at).days
        
        # Create new strategy if type changed or strategy is old
        if self.current_strategy_type != strategy_type:
            return True
        
        if strategy_age_days > 90:  # Quarterly refresh
            return True
        
        return False
    
    async def _generate_objectives_for_strategy(self, strategy_type: PlanType, context: Dict[str, Any], 
                                               timeframe: int, current_time: datetime) -> Dict[str, StrategicObjective]:
        """Generate strategic objectives based on strategy type."""
        objectives = {}
        target_completion = current_time + timedelta(days=timeframe)
        
        current_metrics = context.get("current_metrics", {})
        
        if strategy_type == PlanType.GROWTH:
            # Revenue growth objective
            obj_id = str(uuid.uuid4())
            objectives[obj_id] = StrategicObjective(
                objective_id=obj_id,
                title="Revenue Growth",
                description="Increase revenue by expanding market share and optimizing pricing",
                target_metrics={"revenue_growth": 0.25, "market_share": current_metrics.get("market_share", 0.1) * 1.2},
                timeframe_days=timeframe,
                priority=PlanPriority.HIGH,
                status=PlanStatus.ACTIVE,
                created_at=current_time,
                target_completion=target_completion
            )
            
            # Operational efficiency objective
            obj_id = str(uuid.uuid4())
            objectives[obj_id] = StrategicObjective(
                objective_id=obj_id,
                title="Operational Efficiency",
                description="Improve operational efficiency to support growth",
                target_metrics={"cost_reduction": 0.1, "inventory_turnover": 8.0},
                timeframe_days=timeframe,
                priority=PlanPriority.MEDIUM,
                status=PlanStatus.ACTIVE,
                created_at=current_time,
                target_completion=target_completion
            )
        
        elif strategy_type == PlanType.DEFENSIVE:
            # Market position defense
            obj_id = str(uuid.uuid4())
            objectives[obj_id] = StrategicObjective(
                objective_id=obj_id,
                title="Market Position Defense",
                description="Defend market position against competitive threats",
                target_metrics={"market_share_retention": 0.95, "customer_retention": 0.85},
                timeframe_days=timeframe,
                priority=PlanPriority.CRITICAL,
                status=PlanStatus.ACTIVE,
                created_at=current_time,
                target_completion=target_completion
            )
        
        elif strategy_type == PlanType.RECOVERY:
            # Profitability recovery
            obj_id = str(uuid.uuid4())
            objectives[obj_id] = StrategicObjective(
                objective_id=obj_id,
                title="Profitability Recovery",
                description="Restore profitability through cost optimization and strategic pricing",
                target_metrics={"profit_margin": 0.15, "break_even_days": 30},
                timeframe_days=min(timeframe, 60),  # Shorter timeframe for recovery
                priority=PlanPriority.CRITICAL,
                status=PlanStatus.ACTIVE,
                created_at=current_time,
                target_completion=current_time + timedelta(days=min(timeframe, 60))
            )
        
        # Add common objectives for all strategies
        obj_id = str(uuid.uuid4())
        objectives[obj_id] = StrategicObjective(
            objective_id=obj_id,
            title="Risk Management",
            description="Maintain acceptable risk levels and ensure business continuity",
            target_metrics={"risk_score": 0.3, "cash_flow_positive_days": timeframe * 0.8},
            timeframe_days=timeframe,
            priority=PlanPriority.MEDIUM,
            status=PlanStatus.ACTIVE,
            created_at=current_time,
            target_completion=target_completion
        )
        
        return objectives
    
    async def _validate_strategic_objectives(self, objectives: Dict[str, StrategicObjective], 
                                           context: Dict[str, Any]) -> Dict[str, StrategicObjective]:
        """Validate and potentially modify objectives based on constraints."""
        validated = {}
        
        for obj_id, objective in objectives.items():
            # Validate target metrics are achievable
            if self._are_targets_realistic(objective, context):
                validated[obj_id] = objective
            else:
                # Adjust targets to be more realistic
                adjusted_objective = self._adjust_objective_targets(objective, context)
                validated[obj_id] = adjusted_objective
                logger.warning(f"Adjusted targets for objective '{objective.title}' to be more realistic")
        
        return validated
    
    def _are_targets_realistic(self, objective: StrategicObjective, context: Dict[str, Any]) -> bool:
        """Check if objective targets are realistic given current context."""
        current_metrics = context.get("current_metrics", {})
        
        for metric, target_value in objective.target_metrics.items():
            current_value = current_metrics.get(metric, 0.0)
            
            # Check if growth rate is reasonable (e.g., not more than 100% improvement)
            if current_value > 0:
                growth_rate = (target_value - current_value) / current_value
                if growth_rate > 1.0:  # More than 100% improvement
                    return False
        
        return True
    
    def _adjust_objective_targets(self, objective: StrategicObjective, context: Dict[str, Any]) -> StrategicObjective:
        """Adjust objective targets to be more realistic."""
        current_metrics = context.get("current_metrics", {})
        adjusted_targets = {}
        
        for metric, target_value in objective.target_metrics.items():
            current_value = current_metrics.get(metric, 0.0)
            
            if current_value > 0:
                # Limit growth to maximum 50% improvement
                max_improvement = current_value * 1.5
                adjusted_targets[metric] = min(target_value, max_improvement)
            else:
                # For metrics starting from zero, use conservative targets
                adjusted_targets[metric] = target_value * 0.5
        
        objective.target_metrics = adjusted_targets
        return objective
    
    async def _assess_strategic_performance(self, current_performance: Dict[str, float]) -> Dict[str, Any]:
        """Assess performance against strategic objectives."""
        assessment = {
            "average_progress": 0.0,
            "objectives_on_track": 0,
            "objectives_behind": 0,
            "objectives_ahead": 0,
            "performance_trends": {}
        }
        
        if not self.strategic_objectives:
            return assessment
        
        progress_scores = []
        
        for objective in self.strategic_objectives.values():
            if objective.status != PlanStatus.ACTIVE:
                continue
            
            progress = objective.calculate_progress(current_performance)
            progress_scores.append(progress)
            
            # Categorize progress
            if progress >= 0.8:
                assessment["objectives_ahead"] += 1
            elif progress >= 0.5:
                assessment["objectives_on_track"] += 1
            else:
                assessment["objectives_behind"] += 1
        
        if progress_scores:
            assessment["average_progress"] = sum(progress_scores) / len(progress_scores)
        
        return assessment
    
    async def _analyze_external_events_impact(self, external_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze impact of external events on strategic planning."""
        event_impacts = []
        
        for event in external_events:
            impact = {
                "event_id": event.get("event_id", str(uuid.uuid4())),
                "event_type": event.get("type", "unknown"),
                "impact_level": self._assess_event_impact_level(event),
                "affected_objectives": self._identify_affected_objectives(event),
                "recommended_actions": self._suggest_responses_to_event(event)
            }
            event_impacts.append(impact)
        
        return event_impacts
    
    def _assess_event_impact_level(self, event: Dict[str, Any]) -> str:
        """Assess the impact level of an external event."""
        event_type = event.get("type", "")
        severity = event.get("severity", 0.5)
        
        high_impact_events = ["fee_hike", "supply_disruption", "competitor_major_action", "market_crash"]
        medium_impact_events = ["demand_change", "seasonal_shift", "new_competitor"]
        
        if event_type in high_impact_events or severity > 0.7:
            return "high"
        elif event_type in medium_impact_events or severity > 0.4:
            return "medium"
        else:
            return "low"
    
    def _identify_affected_objectives(self, event: Dict[str, Any]) -> List[str]:
        """Identify which objectives are affected by an external event."""
        affected = []
        event_type = event.get("type", "")
        
        for obj_id, objective in self.strategic_objectives.items():
            if objective.status != PlanStatus.ACTIVE:
                continue
            
            # Simple mapping of event types to affected objectives
            if event_type in ["fee_hike", "cost_increase"] and "cost" in objective.description.lower():
                affected.append(obj_id)
            elif event_type in ["demand_change", "market_shift"] and "revenue" in objective.description.lower():
                affected.append(obj_id)
            elif event_type in ["competitor_action"] and "market" in objective.description.lower():
                affected.append(obj_id)
        
        return affected
    
    def _suggest_responses_to_event(self, event: Dict[str, Any]) -> List[str]:
        """Suggest strategic responses to external events."""
        event_type = event.get("type", "")
        responses = []
        
        if event_type == "fee_hike":
            responses.extend(["review_pricing_strategy", "optimize_costs", "diversify_revenue_streams"])
        elif event_type == "competitor_action":
            responses.extend(["competitive_analysis", "differentiation_strategy", "market_positioning"])
        elif event_type == "demand_change":
            responses.extend(["demand_forecasting", "inventory_adjustment", "marketing_strategy"])
        
        return responses
    
    async def _determine_objective_modifications(self, objective: StrategicObjective, 
                                                performance_assessment: Dict[str, Any],
                                                event_impacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine what modifications are needed for an objective."""
        modifications = {}
        
        # Check if objective is significantly behind schedule
        progress = objective.progress_indicators.get("overall_progress", 0.0)
        if progress < 0.3:
            modifications["priority"] = PlanPriority.HIGH
            modifications["reason"] = "Behind schedule - increasing priority"
        
        # Check for event impacts
        for impact in event_impacts:
            if objective.objective_id in impact.get("affected_objectives", []):
                if impact["impact_level"] == "high":
                    modifications["status"] = PlanStatus.ON_HOLD
                    modifications["reason"] = f"High impact event: {impact['event_type']}"
                    break
        
        return modifications
    
    async def _apply_objective_modifications(self, objective: StrategicObjective, modifications: Dict[str, Any]):
        """Apply modifications to an objective."""
        for key, value in modifications.items():
            if key == "reason":
                continue  # Skip reason field
            if hasattr(objective, key):
                setattr(objective, key, value)
    
    async def _identify_new_objectives_needed(self, external_events: List[Dict[str, Any]], 
                                            current_performance: Dict[str, float]) -> List[StrategicObjective]:
        """Identify new objectives needed based on external events."""
        new_objectives = []
        current_time = datetime.now()
        
        for event in external_events:
            event_type = event.get("type", "")
            severity = event.get("severity", 0.5)
            
            if event_type == "competitor_major_action" and severity > 0.7:
                # Create competitive response objective
                obj_id = str(uuid.uuid4())
                new_obj = StrategicObjective(
                    objective_id=obj_id,
                    title="Competitive Response",
                    description=f"Respond to major competitive threat: {event.get('description', '')}",
                    target_metrics={"competitive_position_score": 0.7},
                    timeframe_days=30,
                    priority=PlanPriority.HIGH,
                    status=PlanStatus.ACTIVE,
                    created_at=current_time,
                    target_completion=current_time + timedelta(days=30)
                )
                new_objectives.append(new_obj)
        
        return new_objectives
    
    async def _identify_objectives_to_cancel(self, current_performance: Dict[str, float], 
                                           external_events: List[Dict[str, Any]]) -> List[str]:
        """Identify objectives that should be cancelled."""
        to_cancel = []
        
        for obj_id, objective in self.strategic_objectives.items():
            if objective.status != PlanStatus.ACTIVE:
                continue
            
            # Cancel if objective is significantly overdue and not progressing
            if objective.is_overdue(datetime.now()):
                progress = objective.progress_indicators.get("overall_progress", 0.0)
                if progress < 0.2:  # Less than 20% progress when overdue
                    to_cancel.append(obj_id)
        
        return to_cancel
    
    async def _calculate_action_objective_alignment(self, action: Dict[str, Any], 
                                                   objective: StrategicObjective) -> float:
        """Calculate how well an action aligns with a strategic objective."""
        action_type = action.get("type", "")
        expected_impact = action.get("expected_impact", {})
        
        alignment_score = 0.0
        
        # Check if action's expected impact aligns with objective's target metrics
        for metric, impact_value in expected_impact.items():
            if metric in objective.target_metrics:
                target_value = objective.target_metrics[metric]
                
                # Positive alignment if action impact moves toward target
                if target_value > 0 and impact_value > 0:
                    alignment_score += min(0.5, impact_value / target_value)
                elif target_value < 0 and impact_value < 0:
                    alignment_score += min(0.5, abs(impact_value / target_value))
        
        # Bonus alignment for action types that naturally support objective
        action_objective_synergies = {
            "set_price": ["revenue", "profit", "market_share"],
            "place_order": ["inventory", "cost", "operational_efficiency"],
            "run_marketing_campaign": ["market_share", "revenue", "brand_awareness"],
            "respond_to_customer": ["customer_satisfaction", "retention"]
        }
        
        relevant_metrics = action_objective_synergies.get(action_type, [])
        for metric in relevant_metrics:
            if any(metric in target_metric for target_metric in objective.target_metrics.keys()):
                alignment_score += 0.2
        
        return min(1.0, alignment_score)
    
    async def _archive_completed_objectives(self):
        """Archive completed objectives to keep active list manageable."""
        completed_objectives = [
            obj for obj in self.strategic_objectives.values() 
            if obj.status in [PlanStatus.COMPLETED, PlanStatus.CANCELLED, PlanStatus.FAILED]
        ]
        
        # In a real implementation, this would store to persistent storage
        logger.info(f"Archived {len(completed_objectives)} completed objectives")
    
    async def _publish_strategic_plan_created_event(self, objectives: Dict[str, StrategicObjective]):
        """Publish event when strategic plan is created."""
        event_data = {
            "agent_id": self.agent_id,
            "event_type": "StrategicPlanCreated",
            "timestamp": datetime.now().isoformat(),
            "strategy_type": self.current_strategy_type.value if self.current_strategy_type else None,
            "objectives_count": len(objectives),
            "objective_summaries": [
                {
                    "id": obj.objective_id,
                    "title": obj.title,
                    "priority": obj.priority.value,
                    "timeframe_days": obj.timeframe_days
                }
                for obj in objectives.values()
            ]
        }
        
        try:
            await self.event_bus.publish(BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                **event_data
            ))
        except Exception as e:
            logger.error(f"Failed to publish strategic plan created event: {e}")
    
    async def _publish_strategic_plan_updated_event(self, update_results: Dict[str, Any]):
        """Publish event when strategic plan is updated."""
        event_data = {
            "agent_id": self.agent_id,
            "event_type": "StrategicPlanUpdated",
            "timestamp": datetime.now().isoformat(),
            "update_results": update_results
        }
        
        try:
            await self.event_bus.publish(BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                **event_data
            ))
        except Exception as e:
            logger.error(f"Failed to publish strategic plan updated event: {e}")


class TacticalPlanner:
    """
    Manages short-term actions that serve strategic goals.
    
    Responsible for generating, prioritizing, and scheduling tactical actions
    that implement strategic objectives.
    """
    
    def __init__(self, agent_id: str, strategic_planner: StrategicPlanner, event_bus: Optional[EventBus] = None):
        """
        Initialize the Tactical Planner.
        
        Args:
            agent_id: Unique identifier for the agent
            strategic_planner: Reference to strategic planner for alignment
            event_bus: Event bus for publishing tactical events
        """
        self.agent_id = agent_id
        self.strategic_planner = strategic_planner
        self.event_bus = event_bus or get_event_bus()
        
        # Tactical state
        self.tactical_actions: Dict[str, TacticalAction] = {}
        self.completed_actions: List[str] = []
        self.action_execution_history: List[Dict[str, Any]] = []
        
        # Planning parameters
        self.planning_horizon_hours = 168  # 1 week default
        self.max_concurrent_actions = 5
        
        logger.info(f"TacticalPlanner initialized for agent {agent_id}")
    
    async def generate_tactical_actions(self, strategic_goals: Dict[str, StrategicObjective], 
                                       current_state: Dict[str, Any]) -> List[TacticalAction]:
        """
        Create action plans that serve strategic goals.
        
        Args:
            strategic_goals: Active strategic objectives to serve
            current_state: Current business state and context
            
        Returns:
            List of tactical actions ordered by priority and dependencies
        """
        logger.info(f"Generating tactical actions for {len(strategic_goals)} strategic goals")
        
        current_time = datetime.now()
        new_actions = []
        
        # Generate actions for each strategic objective
        for objective in strategic_goals.values():
            if objective.status != PlanStatus.ACTIVE:
                continue
            
            objective_actions = await self._generate_actions_for_objective(
                objective, current_state, current_time
            )
            new_actions.extend(objective_actions)
        
        # Generate immediate response actions based on current state
        immediate_actions = await self._generate_immediate_response_actions(
            current_state, current_time
        )
        new_actions.extend(immediate_actions)
        
        # Validate and schedule actions
        validated_actions = await self._validate_and_schedule_actions(new_actions, current_state)
        
        # Add to tactical actions registry
        for action in validated_actions:
            self.tactical_actions[action.action_id] = action
        
        # Clean up old completed actions
        await self._cleanup_old_actions()
        
        await self._publish_tactical_actions_generated_event(validated_actions)
        
        logger.info(f"Generated {len(validated_actions)} tactical actions")
        return validated_actions
    
    async def prioritize_actions(self, action_list: List[TacticalAction], 
                                constraints: Dict[str, Any]) -> List[TacticalAction]:
        """
        Rank actions by urgency, importance, and resource constraints.
        
        Args:
            action_list: List of actions to prioritize
            constraints: Resource and timing constraints
            
        Returns:
            Prioritized list of actions
        """
        logger.info(f"Prioritizing {len(action_list)} tactical actions")
        
        # Calculate priority scores for each action
        action_scores = []
        
        for action in action_list:
            score = await self._calculate_action_priority_score(action, constraints)
            action_scores.append((action, score))
        
        # Sort by priority score (descending)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply resource constraints and dependencies
        prioritized_actions = await self._apply_constraints_to_prioritization(
            action_scores, constraints
        )
        
        await self._publish_tactical_actions_prioritized_event(prioritized_actions)
        
        logger.info(f"Prioritized actions - top priority: {prioritized_actions[0].title if prioritized_actions else 'None'}")
        return prioritized_actions
    
    def get_ready_actions(self, current_time: Optional[datetime] = None) -> List[TacticalAction]:
        """Get actions that are ready for execution."""
        current_time = current_time or datetime.now()
        
        ready_actions = []
        for action in self.tactical_actions.values():
            if action.is_ready_for_execution(current_time, self.completed_actions):
                ready_actions.append(action)
        
        return ready_actions
    
    async def mark_action_completed(self, action_id: str, execution_result: Dict[str, Any]):
        """Mark an action as completed and record results."""
        if action_id in self.tactical_actions:
            action = self.tactical_actions[action_id]
            action.status = PlanStatus.COMPLETED
            
            self.completed_actions.append(action_id)
            
            # Record execution history
            execution_record = {
                "action_id": action_id,
                "completed_at": datetime.now().isoformat(),
                "execution_result": execution_result,
                "strategic_objective_id": action.strategic_objective_id
            }
            self.action_execution_history.append(execution_record)
            
            await self._publish_tactical_action_completed_event(action, execution_result)
            
            logger.info(f"Marked action {action_id} as completed")
    
    async def mark_action_failed(self, action_id: str, failure_reason: str):
        """Mark an action as failed and optionally reschedule."""
        if action_id in self.tactical_actions:
            action = self.tactical_actions[action_id]
            action.status = PlanStatus.FAILED
            
            # Record failure
            failure_record = {
                "action_id": action_id,
                "failed_at": datetime.now().isoformat(),
                "failure_reason": failure_reason,
                "strategic_objective_id": action.strategic_objective_id
            }
            self.action_execution_history.append(failure_record)
            
            # Determine if action should be rescheduled
            should_reschedule = await self._should_reschedule_failed_action(action, failure_reason)
            
            if should_reschedule:
                await self._reschedule_failed_action(action, failure_reason)
            
            logger.warning(f"Marked action {action_id} as failed: {failure_reason}")
    
    def get_tactical_status(self) -> Dict[str, Any]:
        """Get comprehensive status of tactical planning."""
        current_time = datetime.now()
        
        active_actions = [a for a in self.tactical_actions.values() if a.status == PlanStatus.ACTIVE]
        ready_actions = self.get_ready_actions(current_time)
        overdue_actions = [a for a in active_actions if a.scheduled_execution < current_time]
        
        status = {
            "agent_id": self.agent_id,
            "total_actions": len(self.tactical_actions),
            "active_actions": len(active_actions),
            "ready_actions": len(ready_actions),
            "overdue_actions": len(overdue_actions),
            "completed_actions": len(self.completed_actions),
            "execution_history_entries": len(self.action_execution_history),
            "planning_horizon_hours": self.planning_horizon_hours
        }
        
        if active_actions:
            # Calculate priority distribution
            priority_counts = {}
            for action in active_actions:
                priority = action.priority.value
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            status["priority_distribution"] = priority_counts
            
            # Next scheduled action
            next_action = min(active_actions, key=lambda a: a.scheduled_execution)
            status["next_scheduled_action"] = {
                "action_id": next_action.action_id,
                "title": next_action.title,
                "scheduled_at": next_action.scheduled_execution.isoformat()
            }
        
        return status
    
    # Private helper methods
    
    async def _generate_actions_for_objective(self, objective: StrategicObjective, 
                                             current_state: Dict[str, Any], 
                                             current_time: datetime) -> List[TacticalAction]:
        """Generate tactical actions to achieve a strategic objective."""
        actions = []
        
        # Analyze what actions are needed based on objective's target metrics
        for metric, target_value in objective.target_metrics.items():
            current_value = current_state.get("current_metrics", {}).get(metric, 0.0)
            
            if metric in ["revenue", "profit", "market_share"]:
                # Revenue-focused actions
                if current_value < target_value:
                    actions.extend(await self._generate_revenue_actions(
                        objective, current_state, current_time
                    ))
            
            elif metric in ["cost_reduction", "operational_efficiency"]:
                # Cost optimization actions
                actions.extend(await self._generate_cost_optimization_actions(
                    objective, current_state, current_time
                ))
            
            elif metric in ["inventory_turnover", "stock_levels"]:
                # Inventory management actions
                actions.extend(await self._generate_inventory_actions(
                    objective, current_state, current_time
                ))
        
        return actions
    
    async def _generate_revenue_actions(self, objective: StrategicObjective, 
                                       current_state: Dict[str, Any], 
                                       current_time: datetime) -> List[TacticalAction]:
        """Generate actions focused on revenue improvement."""
        actions = []
        
        # Price optimization action
        action_id = str(uuid.uuid4())
        actions.append(TacticalAction(
            action_id=action_id,
            title="Price Optimization Review",
            description="Analyze and optimize pricing strategy for revenue growth",
            action_type="set_price",
            parameters={
                "analysis_type": "revenue_optimization",
                "market_analysis": True,
                "competitor_analysis": True
            },
            strategic_objective_id=objective.objective_id,
            priority=PlanPriority.HIGH,
            status=PlanStatus.ACTIVE,
            created_at=current_time,
            scheduled_execution=current_time + timedelta(hours=2),
            estimated_duration_hours=1.0,
            expected_impact={"revenue": 0.1, "profit_margin": 0.05}
        ))
        
        # Marketing campaign action
        action_id = str(uuid.uuid4())
        actions.append(TacticalAction(
            action_id=action_id,
            title="Revenue-Focused Marketing Campaign",
            description="Launch targeted marketing campaign to boost sales",
            action_type="run_marketing_campaign",
            parameters={
                "campaign_type": "revenue_boost",
                "budget": 1000.0,
                "duration_days": 7,
                "target_demographics": "high_value_customers"
            },
            strategic_objective_id=objective.objective_id,
            priority=PlanPriority.MEDIUM,
            status=PlanStatus.ACTIVE,
            created_at=current_time,
            scheduled_execution=current_time + timedelta(hours=24),
            estimated_duration_hours=168.0,  # 1 week campaign
            expected_impact={"revenue": 0.15, "market_share": 0.05}
        ))
        
        return actions
    
    async def _generate_cost_optimization_actions(self, objective: StrategicObjective, 
                                                 current_state: Dict[str, Any], 
                                                 current_time: datetime) -> List[TacticalAction]:
        """Generate actions focused on cost optimization."""
        actions = []
        
        # Supplier negotiation action
        action_id = str(uuid.uuid4())
        actions.append(TacticalAction(
            action_id=action_id,
            title="Supplier Cost Negotiation",
            description="Negotiate better terms with suppliers to reduce costs",
            action_type="place_order",
            parameters={
                "negotiation_type": "cost_reduction",
                "target_reduction": 0.1,
                "renegotiate_existing": True
            },
            strategic_objective_id=objective.objective_id,
            priority=PlanPriority.MEDIUM,
            status=PlanStatus.ACTIVE,
            created_at=current_time,
            scheduled_execution=current_time + timedelta(hours=8),
            estimated_duration_hours=4.0,
            expected_impact={"cost_reduction": 0.1, "profit_margin": 0.05}
        ))
        
        return actions
    
    async def _generate_inventory_actions(self, objective: StrategicObjective, 
                                         current_state: Dict[str, Any], 
                                         current_time: datetime) -> List[TacticalAction]:
        """Generate actions focused on inventory management."""
        actions = []
        
        inventory_level = current_state.get("inventory_level", 0)
        
        if inventory_level < 50:  # Low inventory threshold
            action_id = str(uuid.uuid4())
            actions.append(TacticalAction(
                action_id=action_id,
                title="Inventory Restocking",
                description="Restock inventory to maintain service levels",
                action_type="place_order",
                parameters={
                    "quantity": 100,
                    "urgency": "medium",
                    "optimize_costs": True
                },
                strategic_objective_id=objective.objective_id,
                priority=PlanPriority.HIGH,
                status=PlanStatus.ACTIVE,
                created_at=current_time,
                scheduled_execution=current_time + timedelta(hours=1),
                estimated_duration_hours=0.5,
                expected_impact={"inventory_turnover": 0.2, "service_level": 0.1}
            ))
        
        return actions
    
    async def _generate_immediate_response_actions(self, current_state: Dict[str, Any], 
                                                  current_time: datetime) -> List[TacticalAction]:
        """Generate actions for immediate response to current state."""
        actions = []
        
        # Check for customer messages requiring response
        customer_messages = current_state.get("customer_messages", [])
        unresponded_messages = [msg for msg in customer_messages if not msg.get("responded", False)]
        
        for message in unresponded_messages[-3:]:  # Respond to latest 3 messages
            action_id = str(uuid.uuid4())
            actions.append(TacticalAction(
                action_id=action_id,
                title=f"Respond to Customer Message",
                description=f"Respond to customer inquiry: {message.get('content', '')[:50]}...",
                action_type="respond_to_customer",
                parameters={
                    "message_id": message.get("message_id"),
                    "priority": "standard",
                    "personalized": True
                },
                strategic_objective_id="",  # Not tied to specific strategic objective
                priority=PlanPriority.MEDIUM,
                status=PlanStatus.ACTIVE,
                created_at=current_time,
                scheduled_execution=current_time + timedelta(minutes=30),
                estimated_duration_hours=0.25,
                expected_impact={"customer_satisfaction": 0.05}
            ))
        
        return actions
    
    async def _validate_and_schedule_actions(self, actions: List[TacticalAction], 
                                           current_state: Dict[str, Any]) -> List[TacticalAction]:
        """Validate actions and optimize their scheduling."""
        validated_actions = []
        
        for action in actions:
            # Validate action parameters
            if await self._validate_action_parameters(action, current_state):
                # Optimize scheduling based on dependencies and resources
                optimized_action = await self._optimize_action_scheduling(action, validated_actions)
                validated_actions.append(optimized_action)
            else:
                logger.warning(f"Action {action.title} failed validation and was excluded")
        
        return validated_actions
    
    async def _validate_action_parameters(self, action: TacticalAction, 
                                         current_state: Dict[str, Any]) -> bool:
        """Validate that action parameters are feasible."""
        # Check budget constraints for actions with costs
        if action.action_type == "run_marketing_campaign":
            budget = action.parameters.get("budget", 0)
            available_budget = current_state.get("available_budget", 0)
            if budget > available_budget:
                return False
        
        # Check inventory constraints for ordering actions
        if action.action_type == "place_order":
            quantity = action.parameters.get("quantity", 0)
            if quantity <= 0:
                return False
        
        return True
    
    async def _optimize_action_scheduling(self, action: TacticalAction, 
                                         existing_actions: List[TacticalAction]) -> TacticalAction:
        """Optimize action scheduling to avoid conflicts and respect dependencies."""
        # Check for scheduling conflicts
        conflicting_actions = [
            a for a in existing_actions 
            if a.action_type == action.action_type and 
            abs((a.scheduled_execution - action.scheduled_execution).total_seconds()) < 3600
        ]
        
        if conflicting_actions:
            # Reschedule to avoid conflict
            latest_conflict = max(conflicting_actions, key=lambda a: a.scheduled_execution)
            action.scheduled_execution = latest_conflict.scheduled_execution + timedelta(hours=1)
        
        return action
    
    async def _calculate_action_priority_score(self, action: TacticalAction, 
                                              constraints: Dict[str, Any]) -> float:
        """Calculate priority score for an action."""
        score = 0.0
        
        # Base score from action priority
        priority_scores = {
            PlanPriority.CRITICAL: 1.0,
            PlanPriority.HIGH: 0.8,
            PlanPriority.MEDIUM: 0.6,
            PlanPriority.LOW: 0.4
        }
        score += priority_scores.get(action.priority, 0.5)
        
        # Strategic alignment bonus
        if action.strategic_objective_id:
            strategic_objective = self.strategic_planner.strategic_objectives.get(action.strategic_objective_id)
            if strategic_objective and strategic_objective.priority in [PlanPriority.HIGH, PlanPriority.CRITICAL]:
                score += 0.3
        
        # Urgency bonus based on scheduling
        current_time = datetime.now()
        hours_until_execution = (action.scheduled_execution - current_time).total_seconds() / 3600
        if hours_until_execution < 1:
            score += 0.4  # Very urgent
        elif hours_until_execution < 4:
            score += 0.2  # Moderately urgent
        
        # Expected impact bonus
        total_expected_impact = sum(action.expected_impact.values())
        score += min(0.3, total_expected_impact)
        
        return score
    
    async def _apply_constraints_to_prioritization(self, action_scores: List[Tuple[TacticalAction, float]], 
                                                  constraints: Dict[str, Any]) -> List[TacticalAction]:
        """Apply resource and dependency constraints to prioritized actions."""
        prioritized_actions = []
        used_resources = {}
        
        max_concurrent = constraints.get("max_concurrent_actions", self.max_concurrent_actions)
        
        for action, score in action_scores:
            # Check resource constraints
            if len(prioritized_actions) >= max_concurrent:
                break
            
            # Check if action conflicts with resource usage
            resource_conflict = False
            if action.action_type in used_resources:
                if used_resources[action.action_type] >= 1:  # Limit one action per type
                    resource_conflict = True
            
            if not resource_conflict:
                prioritized_actions.append(action)
                used_resources[action.action_type] = used_resources.get(action.action_type, 0) + 1
        
        return prioritized_actions
    
    async def _cleanup_old_actions(self):
        """Remove old completed actions to keep registry manageable."""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        old_actions = [
            action_id for action_id, action in self.tactical_actions.items()
            if action.status in [PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED] and
            action.created_at < cutoff_time
        ]
        
        for action_id in old_actions:
            del self.tactical_actions[action_id]
        
        logger.info(f"Cleaned up {len(old_actions)} old tactical actions")
    
    async def _should_reschedule_failed_action(self, action: TacticalAction, failure_reason: str) -> bool:
        """Determine if a failed action should be rescheduled."""
        # Reschedule if failure was due to temporary issues
        temporary_failures = ["resource_unavailable", "network_error", "temporary_constraint"]
        
        return any(temp_failure in failure_reason.lower() for temp_failure in temporary_failures)
    
    async def _reschedule_failed_action(self, action: TacticalAction, failure_reason: str):
        """Reschedule a failed action."""
        # Create new action with updated schedule
        new_action_id = str(uuid.uuid4())
        rescheduled_action = TacticalAction(
            action_id=new_action_id,
            title=f"[RETRY] {action.title}",
            description=f"Retrying failed action: {action.description}",
            action_type=action.action_type,
            parameters=action.parameters.copy(),
            strategic_objective_id=action.strategic_objective_id,
            priority=action.priority,
            status=PlanStatus.ACTIVE,
            created_at=datetime.now(),
            scheduled_execution=datetime.now() + timedelta(hours=1),  # Reschedule for 1 hour later
            estimated_duration_hours=action.estimated_duration_hours,
            expected_impact=action.expected_impact.copy()
        )
        
        self.tactical_actions[new_action_id] = rescheduled_action
        logger.info(f"Rescheduled failed action {action.action_id} as {new_action_id}")
    
    async def _publish_tactical_actions_generated_event(self, actions: List[TacticalAction]):
        """Publish event when tactical actions are generated."""
        event_data = {
            "agent_id": self.agent_id,
            "event_type": "TacticalActionsGenerated",
            "timestamp": datetime.now().isoformat(),
            "actions_count": len(actions),
            "action_summaries": [
                {
                    "id": action.action_id,
                    "title": action.title,
                    "type": action.action_type,
                    "priority": action.priority.value,
                    "scheduled_at": action.scheduled_execution.isoformat()
                }
                for action in actions
            ]
        }
        
        try:
            await self.event_bus.publish(BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                **event_data
            ))
        except Exception as e:
            logger.error(f"Failed to publish tactical actions generated event: {e}")
    
    async def _publish_tactical_actions_prioritized_event(self, actions: List[TacticalAction]):
        """Publish event when tactical actions are prioritized."""
        event_data = {
            "agent_id": self.agent_id,
            "event_type": "TacticalActionsPrioritized",
            "timestamp": datetime.now().isoformat(),
            "prioritized_actions": [action.action_id for action in actions]
        }
        
        try:
            await self.event_bus.publish(BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                **event_data
            ))
        except Exception as e:
            logger.error(f"Failed to publish tactical actions prioritized event: {e}")
    
    async def _publish_tactical_action_completed_event(self, action: TacticalAction, 
                                                      execution_result: Dict[str, Any]):
        """Publish event when a tactical action is completed."""
        event_data = {
            "agent_id": self.agent_id,
            "event_type": "TacticalActionCompleted",
            "timestamp": datetime.now().isoformat(),
            "action_id": action.action_id,
            "action_type": action.action_type,
            "strategic_objective_id": action.strategic_objective_id,
            "execution_result": execution_result
        }
        
        try:
            await self.event_bus.publish(BaseEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                **event_data
            ))
        except Exception as e:
            logger.error(f"Failed to publish tactical action completed event: {e}")