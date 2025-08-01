"""
Skill Coordinator for FBA-Bench Multi-Domain Agent Architecture.

This module provides event-driven coordination for skill modules, managing
event subscription, priority-based dispatch, resource allocation, and
concurrent execution of multiple skills with performance tracking.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Type
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from .skill_modules.base_skill import BaseSkill, SkillAction, SkillContext, SkillOutcome, SkillStatus
from events import BaseEvent
from event_bus import EventBus

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """Coordination strategies for handling multiple skill actions."""
    PRIORITY_BASED = "priority_based"
    ROUND_ROBIN = "round_robin"
    RESOURCE_OPTIMAL = "resource_optimal"
    CONSENSUS_BASED = "consensus_based"


@dataclass
class SkillSubscription:
    """
    Skill subscription to event types with priority and filters.
    
    Attributes:
        skill: The skill instance
        event_types: Event types this skill subscribes to
        priority_multiplier: Multiplier for skill priority calculations
        filters: Optional filters for event processing
        max_concurrent_events: Maximum concurrent events this skill can handle
        current_load: Current processing load
    """
    skill: BaseSkill
    event_types: Set[str]
    priority_multiplier: float = 1.0
    filters: Dict[str, Any] = field(default_factory=dict)
    max_concurrent_events: int = 3
    current_load: int = 0


@dataclass
class ResourceAllocation:
    """
    Resource allocation tracking for skill coordination.
    
    Attributes:
        total_budget: Total budget available
        allocated_budget: Budget allocated to skills
        remaining_budget: Remaining available budget
        token_budget: Total token budget for LLM calls
        allocated_tokens: Tokens allocated to skills
        remaining_tokens: Remaining token budget
        concurrent_slots: Available concurrent execution slots
        used_slots: Currently used execution slots
    """
    total_budget: int = 1000000  # $10,000 in cents
    allocated_budget: int = 0
    remaining_budget: int = 1000000
    token_budget: int = 100000
    allocated_tokens: int = 0
    remaining_tokens: int = 100000
    concurrent_slots: int = 5
    used_slots: int = 0


@dataclass
class SkillPerformanceMetrics:
    """
    Performance metrics for skill coordination analysis.
    
    Attributes:
        skill_name: Name of the skill
        total_events_processed: Total events processed by skill
        total_actions_generated: Total actions generated
        average_response_time: Average response time in seconds
        success_rate: Success rate of skill actions
        resource_efficiency: Resource utilization efficiency
        conflict_rate: Rate of conflicts with other skills
        last_update: Last metrics update timestamp
    """
    skill_name: str
    total_events_processed: int = 0
    total_actions_generated: int = 0
    average_response_time: float = 0.0
    success_rate: float = 0.0
    resource_efficiency: float = 1.0
    conflict_rate: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class SkillCoordinator:
    """
    Event-driven coordination system for skill modules.
    
    Manages skill registration, event routing, resource allocation, 
    and concurrent execution with performance monitoring and conflict resolution.
    """
    
    def __init__(self, agent_id: str, event_bus: EventBus, config: Dict[str, Any] = None):
        """
        Initialize the Skill Coordinator.
        
        Args:
            agent_id: ID of the agent this coordinator serves
            event_bus: Event bus for communication
            config: Configuration parameters for coordination
        """
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.config = config or {}
        
        # Configuration parameters
        self.coordination_strategy = CoordinationStrategy(
            self.config.get('coordination_strategy', 'priority_based')
        )
        self.max_concurrent_skills = self.config.get('max_concurrent_skills', 3)
        self.conflict_resolution_timeout = self.config.get('conflict_resolution_timeout', 5.0)
        self.performance_tracking_enabled = self.config.get('performance_tracking_enabled', True)
        
        # Skill management
        self.skill_subscriptions: Dict[str, SkillSubscription] = {}
        self.event_skill_mapping: Dict[str, List[str]] = defaultdict(list)
        
        # Resource management
        self.resource_allocation = ResourceAllocation(
            total_budget=self.config.get('total_budget_cents', 1000000),
            token_budget=self.config.get('token_budget', 100000),
            concurrent_slots=self.max_concurrent_skills
        )
        
        # Performance tracking
        self.skill_metrics: Dict[str, SkillPerformanceMetrics] = {}
        self.coordination_history: List[Dict[str, Any]] = []
        self.conflict_log: List[Dict[str, Any]] = []
        
        # Execution management
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.pending_actions: List[Tuple[str, SkillAction]] = []
        self.execution_lock = asyncio.Lock()
        
        logger.info(f"SkillCoordinator initialized for agent {agent_id}")
    
    async def register_skill(self, skill: BaseSkill, event_types: List[str], 
                           priority_multiplier: float = 1.0,
                           filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a skill with the coordinator for event processing.
        
        Args:
            skill: Skill instance to register
            event_types: List of event types the skill should handle
            priority_multiplier: Multiplier for skill priority calculations
            filters: Optional filters for event processing
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            skill_name = skill.skill_name
            
            # Create subscription
            subscription = SkillSubscription(
                skill=skill,
                event_types=set(event_types),
                priority_multiplier=priority_multiplier,
                filters=filters or {},
                max_concurrent_events=self.config.get(f'{skill_name}_max_concurrent', 3)
            )
            
            self.skill_subscriptions[skill_name] = subscription
            
            # Update event-skill mapping
            for event_type in event_types:
                if skill_name not in self.event_skill_mapping[event_type]:
                    self.event_skill_mapping[event_type].append(skill_name)
            
            # Initialize performance metrics
            self.skill_metrics[skill_name] = SkillPerformanceMetrics(skill_name=skill_name)
            
            # Subscribe skill to event bus
            await skill.subscribe_to_events(event_types)
            
            logger.info(f"Registered skill {skill_name} for events: {event_types}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering skill {skill.skill_name}: {e}")
            return False
    
    async def dispatch_event(self, event: BaseEvent) -> List[SkillAction]:
        """
        Dispatch event to relevant skills based on priority and generate coordinated actions.
        
        Args:
            event: Event to dispatch
            
        Returns:
            List of coordinated actions from skills
        """
        event_type = type(event).__name__
        
        # Find skills interested in this event type
        interested_skills = self.event_skill_mapping.get(event_type, [])
        
        if not interested_skills:
            return []
        
        try:
            # Calculate skill priorities for this event
            skill_priorities = await self._calculate_skill_priorities(event, interested_skills)
            
            # Filter skills based on resource availability and load
            available_skills = self._filter_available_skills(skill_priorities)
            
            # Dispatch to skills concurrently
            skill_actions = await self._dispatch_to_skills(event, available_skills)
            
            # Coordinate and resolve conflicts
            coordinated_actions = await self._coordinate_actions(skill_actions)
            
            # Update performance metrics
            if self.performance_tracking_enabled:
                await self._update_performance_metrics(event, skill_actions)
            
            # Log coordination decision
            self._log_coordination_decision(event, skill_actions, coordinated_actions)
            
            return coordinated_actions
            
        except Exception as e:
            logger.error(f"Error dispatching event {event_type}: {e}")
            return []
    
    async def _calculate_skill_priorities(self, event: BaseEvent, interested_skills: List[str]) -> List[Tuple[str, float]]:
        """Calculate priority scores for skills handling this event."""
        skill_priorities = []
        
        for skill_name in interested_skills:
            subscription = self.skill_subscriptions.get(skill_name)
            if not subscription:
                continue
            
            try:
                # Get base priority from skill
                base_priority = subscription.skill.get_priority_score(event)
                
                # Apply priority multiplier
                adjusted_priority = base_priority * subscription.priority_multiplier
                
                # Factor in current load
                load_factor = 1.0 - (subscription.current_load / subscription.max_concurrent_events)
                final_priority = adjusted_priority * load_factor
                
                skill_priorities.append((skill_name, final_priority))
                
            except Exception as e:
                logger.error(f"Error calculating priority for skill {skill_name}: {e}")
        
        # Sort by priority (highest first)
        skill_priorities.sort(key=lambda x: x[1], reverse=True)
        return skill_priorities
    
    def _filter_available_skills(self, skill_priorities: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Filter skills based on resource availability and load."""
        available_skills = []
        
        for skill_name, priority in skill_priorities:
            subscription = self.skill_subscriptions.get(skill_name)
            if not subscription:
                continue
            
            # Check if skill has capacity
            if subscription.current_load >= subscription.max_concurrent_events:
                continue
            
            # Check if we have available execution slots
            if self.resource_allocation.used_slots >= self.resource_allocation.concurrent_slots:
                break  # No more concurrent slots available
            
            available_skills.append((skill_name, priority))
        
        return available_skills
    
    async def _dispatch_to_skills(self, event: BaseEvent, available_skills: List[Tuple[str, float]]) -> List[Tuple[str, List[SkillAction]]]:
        """Dispatch event to available skills concurrently."""
        dispatch_tasks = []
        skill_actions = []
        
        for skill_name, priority in available_skills:
            subscription = self.skill_subscriptions.get(skill_name)
            if not subscription:
                continue
            
            # Create dispatch task
            task = asyncio.create_task(
                self._process_skill_event(subscription.skill, event, skill_name)
            )
            dispatch_tasks.append((skill_name, task))
            
            # Update load tracking
            subscription.current_load += 1
            self.resource_allocation.used_slots += 1
        
        # Wait for all skills to process the event
        for skill_name, task in dispatch_tasks:
            try:
                actions = await asyncio.wait_for(task, timeout=self.conflict_resolution_timeout)
                if actions:
                    skill_actions.append((skill_name, actions))
            except asyncio.TimeoutError:
                logger.warning(f"Skill {skill_name} timed out processing event")
            except Exception as e:
                logger.error(f"Error in skill {skill_name} processing: {e}")
            finally:
                # Update load tracking
                subscription = self.skill_subscriptions.get(skill_name)
                if subscription:
                    subscription.current_load = max(0, subscription.current_load - 1)
                self.resource_allocation.used_slots = max(0, self.resource_allocation.used_slots - 1)
        
        return skill_actions
    
    async def _process_skill_event(self, skill: BaseSkill, event: BaseEvent, skill_name: str) -> Optional[List[SkillAction]]:
        """Process event with a single skill and return actions."""
        try:
            start_time = datetime.now()
            
            # Process event through skill
            actions = await skill.process_event(event)
            
            # Update response time metric
            if self.performance_tracking_enabled:
                response_time = (datetime.now() - start_time).total_seconds()
                metrics = self.skill_metrics.get(skill_name)
                if metrics:
                    metrics.total_events_processed += 1
                    metrics.average_response_time = (
                        (metrics.average_response_time * (metrics.total_events_processed - 1) + response_time) /
                        metrics.total_events_processed
                    )
                    if actions:
                        metrics.total_actions_generated += len(actions)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error processing event in skill {skill_name}: {e}")
            return None
    
    async def _coordinate_actions(self, skill_actions: List[Tuple[str, List[SkillAction]]]) -> List[SkillAction]:
        """Coordinate actions from multiple skills and resolve conflicts."""
        if not skill_actions:
            return []
        
        # Flatten all actions with skill attribution
        all_actions = []
        for skill_name, actions in skill_actions:
            for action in actions:
                action.skill_source = skill_name  # Ensure skill source is set
                all_actions.append(action)
        
        if len(all_actions) <= 1:
            return all_actions
        
        # Apply coordination strategy
        if self.coordination_strategy == CoordinationStrategy.PRIORITY_BASED:
            return await self._coordinate_by_priority(all_actions)
        elif self.coordination_strategy == CoordinationStrategy.RESOURCE_OPTIMAL:
            return await self._coordinate_by_resources(all_actions)
        elif self.coordination_strategy == CoordinationStrategy.CONSENSUS_BASED:
            return await self._coordinate_by_consensus(all_actions)
        else:  # ROUND_ROBIN
            return await self._coordinate_round_robin(all_actions)
    
    async def _coordinate_by_priority(self, actions: List[SkillAction]) -> List[SkillAction]:
        """Coordinate actions based on priority scores."""
        # Sort by priority and confidence
        sorted_actions = sorted(actions, key=lambda a: a.priority * a.confidence, reverse=True)
        
        # Check for conflicts and resolve
        coordinated_actions = []
        resource_usage = {"budget": 0, "tokens": 0}
        
        for action in sorted_actions:
            # Check resource constraints
            action_budget = action.resource_requirements.get("budget", 0)
            action_tokens = action.resource_requirements.get("tokens", 0)
            
            if (resource_usage["budget"] + action_budget <= self.resource_allocation.remaining_budget and
                resource_usage["tokens"] + action_tokens <= self.resource_allocation.remaining_tokens):
                
                # Check for conflicts with already selected actions
                if not self._has_conflict(action, coordinated_actions):
                    coordinated_actions.append(action)
                    resource_usage["budget"] += action_budget
                    resource_usage["tokens"] += action_tokens
                else:
                    # Log conflict
                    self._log_conflict(action, coordinated_actions)
        
        return coordinated_actions
    
    async def _coordinate_by_resources(self, actions: List[SkillAction]) -> List[SkillAction]:
        """Coordinate actions to optimize resource utilization."""
        # Calculate resource efficiency for each action
        efficient_actions = []
        
        for action in actions:
            budget_req = action.resource_requirements.get("budget", 0)
            tokens_req = action.resource_requirements.get("tokens", 0)
            
            # Calculate efficiency score (expected outcome / resource cost)
            expected_value = sum(action.expected_outcome.values()) if action.expected_outcome else 1.0
            resource_cost = budget_req + tokens_req + 1  # Avoid division by zero
            efficiency = (expected_value * action.confidence) / resource_cost
            
            efficient_actions.append((action, efficiency))
        
        # Sort by efficiency and select within resource constraints
        efficient_actions.sort(key=lambda x: x[1], reverse=True)
        
        coordinated_actions = []
        resource_usage = {"budget": 0, "tokens": 0}
        
        for action, efficiency in efficient_actions:
            action_budget = action.resource_requirements.get("budget", 0)
            action_tokens = action.resource_requirements.get("tokens", 0)
            
            if (resource_usage["budget"] + action_budget <= self.resource_allocation.remaining_budget and
                resource_usage["tokens"] + action_tokens <= self.resource_allocation.remaining_tokens and
                not self._has_conflict(action, coordinated_actions)):
                
                coordinated_actions.append(action)
                resource_usage["budget"] += action_budget
                resource_usage["tokens"] += action_tokens
        
        return coordinated_actions
    
    async def _coordinate_by_consensus(self, actions: List[SkillAction]) -> List[SkillAction]:
        """Coordinate actions based on consensus among skills."""
        # Group actions by type
        action_groups = defaultdict(list)
        for action in actions:
            action_groups[action.action_type].append(action)
        
        coordinated_actions = []
        
        for action_type, group_actions in action_groups.items():
            if len(group_actions) == 1:
                # No conflict, include the action
                coordinated_actions.append(group_actions[0])
            else:
                # Multiple skills suggest same action type - find consensus
                consensus_action = await self._find_consensus_action(group_actions)
                if consensus_action:
                    coordinated_actions.append(consensus_action)
        
        return coordinated_actions
    
    async def _coordinate_round_robin(self, actions: List[SkillAction]) -> List[SkillAction]:
        """Coordinate actions using round-robin selection."""
        # Group actions by skill
        skill_actions = defaultdict(list)
        for action in actions:
            skill_actions[action.skill_source].append(action)
        
        coordinated_actions = []
        skill_list = list(skill_actions.keys())
        skill_index = 0
        
        while any(skill_actions.values()):
            current_skill = skill_list[skill_index]
            if skill_actions[current_skill]:
                action = skill_actions[current_skill].pop(0)
                if not self._has_conflict(action, coordinated_actions):
                    coordinated_actions.append(action)
            
            skill_index = (skill_index + 1) % len(skill_list)
            
            # Prevent infinite loop
            if not any(skill_actions.values()):
                break
        
        return coordinated_actions
    
    def _has_conflict(self, action: SkillAction, existing_actions: List[SkillAction]) -> bool:
        """Check if action conflicts with existing actions."""
        for existing in existing_actions:
            # Check for same action type targeting same resource
            if (action.action_type == existing.action_type and
                action.parameters.get("asin") == existing.parameters.get("asin")):
                return True
            
            # Check for mutually exclusive actions
            if self._are_mutually_exclusive(action.action_type, existing.action_type):
                return True
        
        return False
    
    def _are_mutually_exclusive(self, action_type1: str, action_type2: str) -> bool:
        """Check if two action types are mutually exclusive."""
        exclusive_pairs = [
            ("set_price", "adjust_pricing_strategy"),  # Can't set price and adjust strategy simultaneously
            ("place_order", "implement_cost_reduction"),  # Can't order inventory while cutting costs
            ("run_marketing_campaign", "reduce_campaign_spend"),  # Conflicting marketing actions
        ]
        
        return (action_type1, action_type2) in exclusive_pairs or (action_type2, action_type1) in exclusive_pairs
    
    async def _find_consensus_action(self, actions: List[SkillAction]) -> Optional[SkillAction]:
        """Find consensus action from multiple similar actions."""
        if not actions:
            return None
        
        # Calculate weighted average of parameters based on confidence
        total_confidence = sum(action.confidence for action in actions)
        if total_confidence == 0:
            return actions[0]  # Fallback to first action
        
        # Use the action with highest confidence as base
        base_action = max(actions, key=lambda a: a.confidence)
        
        # Average numeric parameters weighted by confidence
        numeric_params = {}
        for param_name in base_action.parameters:
            if isinstance(base_action.parameters[param_name], (int, float)):
                weighted_sum = sum(
                    action.parameters.get(param_name, 0) * action.confidence 
                    for action in actions 
                    if param_name in action.parameters
                )
                numeric_params[param_name] = weighted_sum / total_confidence
        
        # Create consensus action
        consensus_action = SkillAction(
            action_type=base_action.action_type,
            parameters={**base_action.parameters, **numeric_params},
            confidence=total_confidence / len(actions),  # Average confidence
            reasoning=f"Consensus from {len(actions)} skills: {base_action.reasoning}",
            priority=max(action.priority for action in actions),
            resource_requirements=base_action.resource_requirements,
            expected_outcome=base_action.expected_outcome,
            skill_source="consensus"
        )
        
        return consensus_action
    
    def _log_conflict(self, conflicting_action: SkillAction, existing_actions: List[SkillAction]):
        """Log action conflict for analysis."""
        conflict_entry = {
            "timestamp": datetime.now(),
            "conflicting_action": {
                "type": conflicting_action.action_type,
                "skill": conflicting_action.skill_source,
                "priority": conflicting_action.priority
            },
            "existing_actions": [
                {
                    "type": action.action_type,
                    "skill": action.skill_source,
                    "priority": action.priority
                }
                for action in existing_actions
            ],
            "resolution": "priority_override"
        }
        
        self.conflict_log.append(conflict_entry)
        
        # Update conflict rate metrics
        if conflicting_action.skill_source in self.skill_metrics:
            metrics = self.skill_metrics[conflicting_action.skill_source]
            metrics.conflict_rate = len([c for c in self.conflict_log 
                                       if c["conflicting_action"]["skill"] == conflicting_action.skill_source]) / max(1, metrics.total_actions_generated)
    
    def _log_coordination_decision(self, event: BaseEvent, skill_actions: List[Tuple[str, List[SkillAction]]], 
                                 coordinated_actions: List[SkillAction]):
        """Log coordination decision for analysis."""
        coordination_entry = {
            "timestamp": datetime.now(),
            "event_type": type(event).__name__,
            "participating_skills": [skill_name for skill_name, _ in skill_actions],
            "total_actions_generated": sum(len(actions) for _, actions in skill_actions),
            "coordinated_actions_count": len(coordinated_actions),
            "coordination_strategy": self.coordination_strategy.value,
            "resource_usage": {
                "budget": sum(action.resource_requirements.get("budget", 0) for action in coordinated_actions),
                "tokens": sum(action.resource_requirements.get("tokens", 0) for action in coordinated_actions)
            }
        }
        
        self.coordination_history.append(coordination_entry)
        
        # Keep history size manageable
        if len(self.coordination_history) > 1000:
            self.coordination_history = self.coordination_history[-500:]
    
    async def _update_performance_metrics(self, event: BaseEvent, skill_actions: List[Tuple[str, List[SkillAction]]]):
        """Update performance metrics for skills."""
        for skill_name, actions in skill_actions:
            if skill_name not in self.skill_metrics:
                continue
            
            metrics = self.skill_metrics[skill_name]
            metrics.last_update = datetime.now()
            
            # Update success rate based on action confidence
            if actions:
                avg_confidence = sum(action.confidence for action in actions) / len(actions)
                metrics.success_rate = (metrics.success_rate * 0.9) + (avg_confidence * 0.1)
    
    async def coordinate_actions(self, skill_actions: List[SkillAction]) -> List[SkillAction]:
        """
        Public method to coordinate pre-generated skill actions.
        
        Args:
            skill_actions: List of actions from various skills
            
        Returns:
            Coordinated list of actions
        """
        return await self._coordinate_actions([("external", skill_actions)])
    
    def get_skill_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all registered skills.
        
        Returns:
            Dictionary of skill names to their performance metrics
        """
        metrics_dict = {}
        
        for skill_name, metrics in self.skill_metrics.items():
            metrics_dict[skill_name] = {
                "total_events_processed": metrics.total_events_processed,
                "total_actions_generated": metrics.total_actions_generated,
                "average_response_time": round(metrics.average_response_time, 3),
                "success_rate": round(metrics.success_rate, 3),
                "resource_efficiency": round(metrics.resource_efficiency, 3),
                "conflict_rate": round(metrics.conflict_rate, 3),
                "last_update": metrics.last_update.isoformat()
            }
        
        return metrics_dict
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics and analytics."""
        if not self.coordination_history:
            return {}
        
        recent_history = self.coordination_history[-100:]  # Last 100 coordinations
        
        return {
            "total_coordinations": len(self.coordination_history),
            "recent_coordinations": len(recent_history),
            "average_actions_per_coordination": sum(
                entry["coordinated_actions_count"] for entry in recent_history
            ) / len(recent_history),
            "coordination_strategy": self.coordination_strategy.value,
            "total_conflicts": len(self.conflict_log),
            "resource_utilization": {
                "budget_utilization": (self.resource_allocation.allocated_budget / 
                                     self.resource_allocation.total_budget) if self.resource_allocation.total_budget > 0 else 0,
                "token_utilization": (self.resource_allocation.allocated_tokens / 
                                    self.resource_allocation.token_budget) if self.resource_allocation.token_budget > 0 else 0,
                "concurrent_slots_used": self.resource_allocation.used_slots,
                "max_concurrent_slots": self.resource_allocation.concurrent_slots
            },
            "skill_participation": {
                skill_name: sum(1 for entry in recent_history if skill_name in entry["participating_skills"])
                for skill_name in self.skill_subscriptions.keys()
            }
        }
    
    async def update_resource_allocation(self, budget_delta: int = 0, token_delta: int = 0) -> bool:
        """
        Update resource allocation for the coordinator.
        
        Args:
            budget_delta: Change in budget allocation (cents)
            token_delta: Change in token allocation
            
        Returns:
            True if update successful, False if insufficient resources
        """
        new_budget = self.resource_allocation.remaining_budget + budget_delta
        new_tokens = self.resource_allocation.remaining_tokens + token_delta
        
        if new_budget < 0 or new_tokens < 0:
            return False
        
        self.resource_allocation.remaining_budget = new_budget
        self.resource_allocation.remaining_tokens = new_tokens
        
        if budget_delta < 0:
            self.resource_allocation.allocated_budget += abs(budget_delta)
        if token_delta < 0:
            self.resource_allocation.allocated_tokens += abs(token_delta)
        
        return True
    
    async def shutdown(self):
        """Shutdown the coordinator and clean up resources."""
        # Cancel any active execution tasks
        for task_id, task in self.active_executions.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.active_executions.clear()
        
        # Reset skill loads
        for subscription in self.skill_subscriptions.values():
            subscription.current_load = 0
        
        self.resource_allocation.used_slots = 0
        
        logger.info(f"SkillCoordinator for agent {self.agent_id} shutdown complete")