"""
Multi-Domain Controller for FBA-Bench Agent Architecture.

This module provides CEO-level coordination for multi-skill agents, handling
strategic decision making, resource allocation, action arbitration, and
ensuring alignment with business objectives across all skill domains.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .skill_coordinator import SkillCoordinator
from .skill_modules.base_skill import SkillAction, SkillContext
from money import Money

logger = logging.getLogger(__name__)


class BusinessPriority(Enum):
    """Business priority levels for resource allocation."""
    SURVIVAL = "survival"        # Cash flow crisis, immediate threats
    STABILIZATION = "stabilization"  # Addressing critical issues
    GROWTH = "growth"           # Scaling and expansion
    OPTIMIZATION = "optimization"    # Efficiency improvements
    INNOVATION = "innovation"    # New opportunities


class StrategicObjective(Enum):
    """Strategic business objectives."""
    PROFITABILITY = "profitability"
    MARKET_SHARE = "market_share"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    FINANCIAL_STABILITY = "financial_stability"
    BRAND_REPUTATION = "brand_reputation"


@dataclass
class BusinessState:
    """
    Current business state for strategic decision making.
    
    Attributes:
        financial_health: Financial health score (0.0 to 1.0)
        cash_position: Current cash position status
        market_position: Competitive market position
        customer_satisfaction: Customer satisfaction level
        operational_efficiency: Operational efficiency score
        growth_trajectory: Business growth direction
        risk_level: Overall business risk assessment
        strategic_focus: Current strategic focus areas
    """
    financial_health: float = 0.5
    cash_position: str = "stable"
    market_position: str = "competitive"
    customer_satisfaction: float = 0.8
    operational_efficiency: float = 0.7
    growth_trajectory: str = "stable"
    risk_level: str = "moderate"
    strategic_focus: List[StrategicObjective] = field(default_factory=list)


@dataclass
class ResourceAllocationPlan:
    """
    Resource allocation plan across business domains.
    
    Attributes:
        total_budget: Total available budget
        allocations: Budget allocations by domain
        constraints: Resource constraints and limits
        priority_multipliers: Priority multipliers by domain
        reallocation_triggers: Conditions for budget reallocation
        approval_thresholds: Thresholds requiring approval
    """
    total_budget: Money
    allocations: Dict[str, Money] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority_multipliers: Dict[str, float] = field(default_factory=dict)
    reallocation_triggers: Dict[str, float] = field(default_factory=dict)
    approval_thresholds: Dict[str, Money] = field(default_factory=dict)


@dataclass
class StrategicDecision:
    """
    Strategic decision record for tracking and analysis.
    
    Attributes:
        decision_id: Unique identifier for the decision
        timestamp: When decision was made
        decision_type: Type of strategic decision
        context: Business context at time of decision
        actions_approved: Actions that were approved
        actions_rejected: Actions that were rejected
        reasoning: Strategic reasoning for the decision
        expected_impact: Expected business impact
        success_metrics: Metrics to track decision success
    """
    decision_id: str
    timestamp: datetime
    decision_type: str
    context: BusinessState
    actions_approved: List[SkillAction]
    actions_rejected: List[SkillAction]
    reasoning: str
    expected_impact: Dict[str, float]
    success_metrics: Dict[str, str] = field(default_factory=dict)


class MultiDomainController:
    """
    CEO-level coordination controller for multi-skill agents.
    
    Manages strategic resource allocation, action arbitration, and ensures
    all skill activities align with business objectives and priorities.
    """
    
    def __init__(self, agent_id: str, skill_coordinator: SkillCoordinator, 
                 config: Dict[str, Any] = None):
        """
        Initialize the Multi-Domain Controller.
        
        Args:
            agent_id: ID of the agent this controller manages
            skill_coordinator: Skill coordinator for event routing
            config: Configuration parameters for strategic management
        """
        self.agent_id = agent_id
        self.skill_coordinator = skill_coordinator
        self.config = config or {}
        
        # Strategic configuration
        self.business_objectives = [
            StrategicObjective.PROFITABILITY,
            StrategicObjective.FINANCIAL_STABILITY,
            StrategicObjective.CUSTOMER_SATISFACTION
        ]
        self.current_priority = BusinessPriority.STABILIZATION
        self.strategic_planning_horizon = self.config.get('planning_horizon_days', 30)
        
        # Resource management
        self.resource_plan = ResourceAllocationPlan(
            total_budget=Money(self.config.get('total_budget_cents', 1000000))  # $10,000
        )
        self._initialize_resource_allocations()
        
        # Business state tracking
        self.current_business_state = BusinessState()
        self.state_history: List[Tuple[datetime, BusinessState]] = []
        
        # Decision tracking
        self.strategic_decisions: List[StrategicDecision] = []
        self.pending_approvals: List[Tuple[SkillAction, datetime]] = []
        
        # Performance tracking
        self.decision_success_rate = 0.7
        self.resource_utilization_efficiency = 0.8
        self.strategic_alignment_score = 0.8
        self.last_strategic_review = datetime.now()
        
        logger.info(f"MultiDomainController initialized for agent {agent_id}")
    
    def _initialize_resource_allocations(self):
        """Initialize default resource allocations across domains."""
        # Default allocation percentages based on business priority
        default_allocations = {
            "inventory_management": 0.35,   # 35% for inventory/supply chain
            "marketing": 0.25,              # 25% for marketing and growth
            "customer_service": 0.15,       # 15% for customer satisfaction
            "financial_operations": 0.10,   # 10% for financial management
            "strategic_reserve": 0.15       # 15% reserve for opportunities
        }
        
        for domain, percentage in default_allocations.items():
            allocation = Money(int(self.resource_plan.total_budget.cents * percentage))
            self.resource_plan.allocations[domain] = allocation
        
        # Set priority multipliers based on current business priority
        self._update_priority_multipliers()
        
        # Set approval thresholds
        self.resource_plan.approval_thresholds = {
            "inventory_management": Money(50000),  # $500
            "marketing": Money(25000),             # $250
            "customer_service": Money(10000),      # $100
            "financial_operations": Money(100000), # $1000
            "emergency": Money(20000)              # $200
        }
    
    def _update_priority_multipliers(self):
        """Update priority multipliers based on current business priority."""
        if self.current_priority == BusinessPriority.SURVIVAL:
            # Focus on cash flow and cost reduction
            self.resource_plan.priority_multipliers = {
                "inventory_management": 1.2,
                "marketing": 0.5,
                "customer_service": 0.8,
                "financial_operations": 1.5
            }
        elif self.current_priority == BusinessPriority.GROWTH:
            # Focus on marketing and expansion
            self.resource_plan.priority_multipliers = {
                "inventory_management": 1.1,
                "marketing": 1.5,
                "customer_service": 1.2,
                "financial_operations": 0.9
            }
        else:  # STABILIZATION, OPTIMIZATION, INNOVATION
            # Balanced approach
            self.resource_plan.priority_multipliers = {
                "inventory_management": 1.0,
                "marketing": 1.0,
                "customer_service": 1.0,
                "financial_operations": 1.0
            }
    
    async def evaluate_business_priorities(self, current_state: Dict[str, Any]) -> BusinessPriority:
        """
        Evaluate current business priorities based on state and context.
        
        Args:
            current_state: Current business state information
            
        Returns:
            Determined business priority level
        """
        # Update business state
        await self._update_business_state(current_state)
        
        # Analyze financial health
        financial_health = current_state.get('financial_health', 0.5)
        cash_position = current_state.get('cash_position', 'stable')
        
        # Determine priority based on critical factors
        if financial_health < 0.3 or cash_position == 'critical':
            priority = BusinessPriority.SURVIVAL
        elif financial_health < 0.5 or cash_position == 'warning':
            priority = BusinessPriority.STABILIZATION
        elif financial_health > 0.8 and cash_position == 'healthy':
            priority = BusinessPriority.GROWTH
        elif financial_health > 0.6:
            priority = BusinessPriority.OPTIMIZATION
        else:
            priority = BusinessPriority.STABILIZATION
        
        # Update if priority changed
        if priority != self.current_priority:
            logger.info(f"Business priority changed from {self.current_priority.value} to {priority.value}")
            self.current_priority = priority
            self._update_priority_multipliers()
            await self._reallocate_resources_for_priority()
        
        return priority
    
    async def _update_business_state(self, current_state: Dict[str, Any]):
        """Update business state tracking."""
        self.current_business_state.financial_health = current_state.get('financial_health', 0.5)
        self.current_business_state.cash_position = current_state.get('cash_position', 'stable')
        self.current_business_state.customer_satisfaction = current_state.get('customer_satisfaction', 0.8)
        self.current_business_state.operational_efficiency = current_state.get('operational_efficiency', 0.7)
        
        # Determine strategic focus based on weakest areas
        focus_areas = []
        if self.current_business_state.financial_health < 0.6:
            focus_areas.append(StrategicObjective.FINANCIAL_STABILITY)
        if self.current_business_state.customer_satisfaction < 0.7:
            focus_areas.append(StrategicObjective.CUSTOMER_SATISFACTION)
        if self.current_business_state.operational_efficiency < 0.6:
            focus_areas.append(StrategicObjective.OPERATIONAL_EFFICIENCY)
        
        if not focus_areas:  # All areas healthy, focus on growth
            focus_areas.append(StrategicObjective.PROFITABILITY)
        
        self.current_business_state.strategic_focus = focus_areas
        
        # Store state history
        self.state_history.append((datetime.now(), self.current_business_state))
        if len(self.state_history) > 100:  # Keep last 100 entries
            self.state_history = self.state_history[-50:]
    
    async def arbitrate_actions(self, competing_actions: List[SkillAction]) -> List[SkillAction]:
        """
        Arbitrate between competing actions from different skills.
        
        Args:
            competing_actions: List of competing skill actions
            
        Returns:
            Arbitrated list of approved actions
        """
        if not competing_actions:
            return []
        
        # Group actions by domain/skill
        domain_actions = self._group_actions_by_domain(competing_actions)
        
        # Apply strategic filtering
        strategic_actions = await self._apply_strategic_filter(competing_actions)
        
        # Apply resource constraints
        resource_approved = await self._apply_resource_constraints(strategic_actions)
        
        # Apply business rules and approval thresholds
        final_approved = await self._apply_business_rules(resource_approved)
        
        # Log strategic decision
        await self._log_strategic_decision("action_arbitration", competing_actions, final_approved)
        
        return final_approved
    
    def _group_actions_by_domain(self, actions: List[SkillAction]) -> Dict[str, List[SkillAction]]:
        """Group actions by business domain."""
        domain_mapping = {
            "SupplyManager": "inventory_management",
            "MarketingManager": "marketing", 
            "CustomerService": "customer_service",
            "FinancialAnalyst": "financial_operations"
        }
        
        domain_actions = {}
        for action in actions:
            domain = domain_mapping.get(action.skill_source, "other")
            if domain not in domain_actions:
                domain_actions[domain] = []
            domain_actions[domain].append(action)
        
        return domain_actions
    
    async def _apply_strategic_filter(self, actions: List[SkillAction]) -> List[SkillAction]:
        """Filter actions based on strategic alignment."""
        aligned_actions = []
        
        for action in actions:
            alignment_score = await self._calculate_strategic_alignment(action)
            
            # Only approve actions with sufficient strategic alignment
            if alignment_score >= 0.6:  # 60% alignment threshold
                aligned_actions.append(action)
            else:
                logger.debug(f"Action {action.action_type} filtered out due to low strategic alignment: {alignment_score:.2f}")
        
        return aligned_actions
    
    async def _calculate_strategic_alignment(self, action: SkillAction) -> float:
        """Calculate how well an action aligns with strategic objectives."""
        alignment_score = 0.5  # Base alignment
        
        # Check alignment with current strategic focus
        for objective in self.current_business_state.strategic_focus:
            if self._action_supports_objective(action, objective):
                alignment_score += 0.2
        
        # Check alignment with business priority
        priority_alignment = self._calculate_priority_alignment(action)
        alignment_score += priority_alignment * 0.3
        
        # Factor in expected outcomes
        outcome_alignment = self._calculate_outcome_alignment(action)
        alignment_score += outcome_alignment * 0.2
        
        return min(1.0, alignment_score)
    
    def _action_supports_objective(self, action: SkillAction, objective: StrategicObjective) -> bool:
        """Check if an action supports a strategic objective."""
        action_objective_mapping = {
            StrategicObjective.PROFITABILITY: [
                "set_price", "optimize_costs", "place_order", "run_marketing_campaign"
            ],
            StrategicObjective.FINANCIAL_STABILITY: [
                "budget_alert", "cashflow_alert", "optimize_costs", "assess_financial_health"
            ],
            StrategicObjective.CUSTOMER_SATISFACTION: [
                "respond_to_customer_message", "respond_to_review", "improve_customer_satisfaction"
            ],
            StrategicObjective.OPERATIONAL_EFFICIENCY: [
                "optimize_costs", "improve_response_time", "place_order"
            ],
            StrategicObjective.MARKET_SHARE: [
                "run_marketing_campaign", "set_price", "adjust_pricing_strategy"
            ],
            StrategicObjective.BRAND_REPUTATION: [
                "respond_to_review", "respond_to_customer_message", "improve_customer_satisfaction"
            ]
        }
        
        supported_actions = action_objective_mapping.get(objective, [])
        return action.action_type in supported_actions
    
    def _calculate_priority_alignment(self, action: SkillAction) -> float:
        """Calculate how well action aligns with current business priority."""
        priority_action_weights = {
            BusinessPriority.SURVIVAL: {
                "optimize_costs": 1.0,
                "cashflow_alert": 1.0,
                "budget_alert": 0.9,
                "assess_financial_health": 0.8,
                "place_order": 0.3,  # Lower priority during survival
                "run_marketing_campaign": 0.2
            },
            BusinessPriority.GROWTH: {
                "run_marketing_campaign": 1.0,
                "set_price": 0.8,
                "place_order": 0.9,
                "respond_to_customer_message": 0.7,
                "optimize_costs": 0.4
            }
        }
        
        # Default weights for other priorities
        default_weights = {action_type: 0.6 for action_type in [
            "set_price", "place_order", "run_marketing_campaign", 
            "respond_to_customer_message", "optimize_costs"
        ]}
        
        weights = priority_action_weights.get(self.current_priority, default_weights)
        return weights.get(action.action_type, 0.5)
    
    def _calculate_outcome_alignment(self, action: SkillAction) -> float:
        """Calculate alignment based on expected outcomes."""
        if not action.expected_outcome:
            return 0.5
        
        # Weight outcomes based on current strategic focus
        outcome_weights = {
            "profit_improvement": 0.8 if StrategicObjective.PROFITABILITY in self.current_business_state.strategic_focus else 0.5,
            "customer_satisfaction_improvement": 0.8 if StrategicObjective.CUSTOMER_SATISFACTION in self.current_business_state.strategic_focus else 0.5,
            "cost_savings": 0.9 if self.current_priority == BusinessPriority.SURVIVAL else 0.6,
            "revenue_growth": 0.9 if self.current_priority == BusinessPriority.GROWTH else 0.6
        }
        
        alignment = 0.0
        outcome_count = 0
        
        for outcome_key, outcome_value in action.expected_outcome.items():
            if outcome_key in outcome_weights and isinstance(outcome_value, (int, float)):
                weight = outcome_weights[outcome_key]
                # Normalize outcome value and apply weight
                normalized_value = min(1.0, abs(outcome_value))
                alignment += weight * normalized_value
                outcome_count += 1
        
        return alignment / max(1, outcome_count)
    
    async def _apply_resource_constraints(self, actions: List[SkillAction]) -> List[SkillAction]:
        """Apply resource constraints and budget limits."""
        approved_actions = []
        remaining_budgets = {
            domain: allocation for domain, allocation in self.resource_plan.allocations.items()
        }
        
        # Sort actions by priority and strategic alignment
        sorted_actions = sorted(actions, 
                              key=lambda a: a.priority * self._calculate_priority_alignment(a),
                              reverse=True)
        
        for action in sorted_actions:
            # Determine domain for resource allocation
            domain = self._get_action_domain(action)
            
            # Check budget requirements
            budget_required = action.resource_requirements.get("budget", 0)
            if isinstance(budget_required, int):
                budget_required = Money(budget_required)
            
            # Check if we have sufficient budget
            if domain in remaining_budgets and remaining_budgets[domain].cents >= budget_required.cents:
                approved_actions.append(action)
                remaining_budgets[domain] = Money(remaining_budgets[domain].cents - budget_required.cents)
            elif budget_required.cents <= self.resource_plan.allocations.get("strategic_reserve", Money(0)).cents:
                # Use strategic reserve for high-priority actions
                approved_actions.append(action)
                reserve = self.resource_plan.allocations.get("strategic_reserve", Money(0))
                self.resource_plan.allocations["strategic_reserve"] = Money(reserve.cents - budget_required.cents)
            else:
                logger.debug(f"Action {action.action_type} rejected due to insufficient budget: required {budget_required}, available {remaining_budgets.get(domain, Money(0))}")
        
        return approved_actions
    
    def _get_action_domain(self, action: SkillAction) -> str:
        """Get the business domain for an action."""
        skill_domain_mapping = {
            "SupplyManager": "inventory_management",
            "MarketingManager": "marketing",
            "CustomerService": "customer_service", 
            "FinancialAnalyst": "financial_operations"
        }
        return skill_domain_mapping.get(action.skill_source, "other")
    
    async def _apply_business_rules(self, actions: List[SkillAction]) -> List[SkillAction]:
        """Apply business rules and approval thresholds."""
        approved_actions = []
        
        for action in actions:
            # Check approval thresholds
            budget_required = action.resource_requirements.get("budget", 0)
            if isinstance(budget_required, int):
                budget_required = Money(budget_required)
            
            domain = self._get_action_domain(action)
            threshold = self.resource_plan.approval_thresholds.get(domain, Money(0))
            
            # Auto-approve if under threshold
            if budget_required.cents <= threshold.cents:
                approved_actions.append(action)
            else:
                # Add to pending approvals for manual review
                self.pending_approvals.append((action, datetime.now()))
                logger.info(f"Action {action.action_type} requires approval: budget ${budget_required.to_float():.2f} exceeds threshold ${threshold.to_float():.2f}")
        
        return approved_actions
    
    async def allocate_resources(self, skill_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Allocate resources across skill requests based on strategic priorities.
        
        Args:
            skill_requests: List of resource requests from skills
            
        Returns:
            Resource allocation decisions
        """
        allocation_decisions = {
            "approved_allocations": {},
            "rejected_requests": [],
            "pending_requests": [],
            "total_allocated": Money(0),
            "remaining_budget": self.resource_plan.total_budget
        }
        
        # Sort requests by strategic priority
        prioritized_requests = await self._prioritize_resource_requests(skill_requests)
        
        total_allocated = Money(0)
        
        for request in prioritized_requests:
            skill_name = request.get("skill_name")
            requested_amount = Money(request.get("amount_cents", 0))
            purpose = request.get("purpose", "general")
            
            # Check if allocation is strategically justified
            if await self._validate_resource_request(request):
                # Check if we have sufficient budget
                if total_allocated.cents + requested_amount.cents <= self.resource_plan.total_budget.cents:
                    allocation_decisions["approved_allocations"][skill_name] = {
                        "amount": requested_amount.cents,
                        "purpose": purpose,
                        "approval_reason": "strategic_alignment"
                    }
                    total_allocated = Money(total_allocated.cents + requested_amount.cents)
                else:
                    allocation_decisions["rejected_requests"].append({
                        "skill_name": skill_name,
                        "amount": requested_amount.cents,
                        "rejection_reason": "insufficient_budget"
                    })
            else:
                allocation_decisions["rejected_requests"].append({
                    "skill_name": skill_name,
                    "amount": requested_amount.cents,
                    "rejection_reason": "poor_strategic_alignment"
                })
        
        allocation_decisions["total_allocated"] = total_allocated
        allocation_decisions["remaining_budget"] = Money(
            self.resource_plan.total_budget.cents - total_allocated.cents
        )
        
        return allocation_decisions
    
    async def _prioritize_resource_requests(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize resource requests based on strategic value."""
        scored_requests = []
        
        for request in requests:
            priority_score = await self._calculate_request_priority(request)
            scored_requests.append((request, priority_score))
        
        # Sort by priority score (highest first)
        scored_requests.sort(key=lambda x: x[1], reverse=True)
        
        return [request for request, score in scored_requests]
    
    async def _calculate_request_priority(self, request: Dict[str, Any]) -> float:
        """Calculate priority score for resource request."""
        base_score = 0.5
        
        # Factor in skill type and current business priority
        skill_name = request.get("skill_name", "")
        domain = self._skill_to_domain(skill_name)
        priority_multiplier = self.resource_plan.priority_multipliers.get(domain, 1.0)
        
        # Factor in expected ROI
        expected_roi = request.get("expected_roi", 1.0)
        roi_score = min(1.0, expected_roi / 2.0)  # Normalize to 0-1
        
        # Factor in urgency
        urgency = request.get("urgency", "medium")
        urgency_multiplier = {"low": 0.8, "medium": 1.0, "high": 1.3, "critical": 1.5}[urgency]
        
        final_score = base_score * priority_multiplier * roi_score * urgency_multiplier
        return min(1.0, final_score)
    
    def _skill_to_domain(self, skill_name: str) -> str:
        """Map skill name to business domain."""
        mapping = {
            "SupplyManager": "inventory_management",
            "MarketingManager": "marketing",
            "CustomerService": "customer_service",
            "FinancialAnalyst": "financial_operations"
        }
        return mapping.get(skill_name, "other")
    
    async def _validate_resource_request(self, request: Dict[str, Any]) -> bool:
        """Validate if resource request aligns with strategic objectives."""
        # Check strategic alignment
        purpose = request.get("purpose", "general")
        skill_name = request.get("skill_name", "")
        
        # Strategic purposes get higher validation scores
        strategic_purposes = [
            "growth_investment", "crisis_response", "customer_retention",
            "cost_optimization", "market_expansion"
        ]
        
        if purpose in strategic_purposes:
            return True
        
        # Check if skill is aligned with current strategic focus
        domain = self._skill_to_domain(skill_name)
        priority_multiplier = self.resource_plan.priority_multipliers.get(domain, 1.0)
        
        return priority_multiplier >= 1.0
    
    def validate_strategic_alignment(self, action: SkillAction, strategic_plan: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Validate if action aligns with strategic plan.
        
        Args:
            action: Action to validate
            strategic_plan: Current strategic plan
            
        Returns:
            Tuple of (is_aligned, alignment_score, reasoning)
        """
        try:
            # Calculate strategic alignment score
            alignment_score = 0.0
            reasoning_parts = []
            
            # Check objective alignment
            action_objectives = self._get_action_objectives(action)
            plan_objectives = strategic_plan.get("objectives", [])
            
            objective_overlap = len(set(action_objectives) & set(plan_objectives))
            if objective_overlap > 0:
                alignment_score += 0.4
                reasoning_parts.append(f"Supports {objective_overlap} strategic objectives")
            
            # Check resource allocation alignment
            domain = self._get_action_domain(action)
            domain_priority = self.resource_plan.priority_multipliers.get(domain, 1.0)
            
            if domain_priority >= 1.0:
                alignment_score += 0.3
                reasoning_parts.append(f"Domain {domain} is strategically prioritized")
            
            # Check business priority alignment
            priority_alignment = self._calculate_priority_alignment(action)
            alignment_score += priority_alignment * 0.3
            reasoning_parts.append(f"Priority alignment: {priority_alignment:.2f}")
            
            # Determine if aligned (threshold: 0.6)
            is_aligned = alignment_score >= 0.6
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No clear strategic alignment"
            
            return is_aligned, alignment_score, reasoning
            
        except Exception as e:
            logger.error(f"Error validating strategic alignment: {e}")
            return False, 0.0, f"Validation error: {e}"
    
    def _get_action_objectives(self, action: SkillAction) -> List[str]:
        """Get strategic objectives that an action supports."""
        action_objective_mapping = {
            "set_price": ["profitability", "market_share"],
            "place_order": ["operational_efficiency", "profitability"],
            "run_marketing_campaign": ["market_share", "brand_reputation"],
            "respond_to_customer_message": ["customer_satisfaction", "brand_reputation"],
            "optimize_costs": ["profitability", "financial_stability"],
            "assess_financial_health": ["financial_stability"],
            "budget_alert": ["financial_stability"],
            "cashflow_alert": ["financial_stability"]
        }
        
        return action_objective_mapping.get(action.action_type, [])
    
    async def _reallocate_resources_for_priority(self):
        """Reallocate resources based on changed business priority."""
        logger.info(f"Reallocating resources for priority: {self.current_priority.value}")
        
        # Calculate new allocation percentages based on priority
        if self.current_priority == BusinessPriority.SURVIVAL:
            new_percentages = {
                "inventory_management": 0.30,   # Reduce inventory investment
                "marketing": 0.15,              # Cut marketing spend
                "customer_service": 0.20,       # Maintain customer service
                "financial_operations": 0.15,   # Increase financial focus
                "strategic_reserve": 0.20       # Increase reserve for crisis
            }
        elif self.current_priority == BusinessPriority.GROWTH:
            new_percentages = {
                "inventory_management": 0.40,   # Increase inventory for growth
                "marketing": 0.35,              # Boost marketing spend
                "customer_service": 0.15,       # Maintain service levels
                "financial_operations": 0.05,   # Minimal financial overhead
                "strategic_reserve": 0.05       # Minimal reserve, invest in growth
            }
        else:  # Balanced allocation for other priorities
            new_percentages = {
                "inventory_management": 0.35,
                "marketing": 0.25,
                "customer_service": 0.15,
                "financial_operations": 0.10,
                "strategic_reserve": 0.15
            }
        
        # Update allocations
        for domain, percentage in new_percentages.items():
            allocation = Money(int(self.resource_plan.total_budget.cents * percentage))
            self.resource_plan.allocations[domain] = allocation
        
        logger.info(f"Resource reallocation completed for {self.current_priority.value} priority")
    
    async def _log_strategic_decision(self, decision_type: str, all_actions: List[SkillAction],
                                    approved_actions: List[SkillAction]):
        """Log strategic decision for tracking and analysis."""
        rejected_actions = [action for action in all_actions if action not in approved_actions]
        
        # Calculate expected impact
        expected_impact = {}
        for action in approved_actions:
            for outcome, value in action.expected_outcome.items():
                if outcome not in expected_impact:
                    expected_impact[outcome] = 0
                if isinstance(value, (int, float)):
                    expected_impact[outcome] += value
        
        decision = StrategicDecision(
            decision_id=f"{decision_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            decision_type=decision_type,
            context=self.current_business_state,
            actions_approved=approved_actions,
            actions_rejected=rejected_actions,
            reasoning=f"Arbitration based on {self.current_priority.value} priority and strategic alignment",
            expected_impact=expected_impact
        )
        
        self.strategic_decisions.append(decision)
        
        # Keep decision history manageable
        if len(self.strategic_decisions) > 1000:
            self.strategic_decisions = self.strategic_decisions[-500:]
        
        # Update decision success tracking
        self._update_decision_tracking(decision)
    
    def _update_decision_tracking(self, decision: StrategicDecision):
        """Update decision tracking metrics."""
        # Simple success rate calculation based on action confidence
        if decision.actions_approved:
            avg_confidence = sum(action.confidence for action in decision.actions_approved) / len(decision.actions_approved)
            # Update running average
            self.decision_success_rate = (self.decision_success_rate * 0.9) + (avg_confidence * 0.1)
        
        # Update strategic alignment score
        alignment_scores = []
        for action in decision.actions_approved:
            is_aligned, alignment_score, _ = self.validate_strategic_alignment(action, {"objectives": [obj.value for obj in self.current_business_state.strategic_focus]})
            alignment_scores.append(alignment_score)
        
        if alignment_scores:
            avg_alignment = sum(alignment_scores) / len(alignment_scores)
            self.strategic_alignment_score = (self.strategic_alignment_score * 0.9) + (avg_alignment * 0.1)
    
    async def get_strategic_dashboard(self) -> Dict[str, Any]:
        """Get strategic dashboard with key metrics and status."""
        return {
            "business_state": {
                "financial_health": self.current_business_state.financial_health,
                "cash_position": self.current_business_state.cash_position,
                "customer_satisfaction": self.current_business_state.customer_satisfaction,
                "operational_efficiency": self.current_business_state.operational_efficiency,
                "current_priority": self.current_priority.value,
                "strategic_focus": [obj.value for obj in self.current_business_state.strategic_focus]
            },
            "resource_allocation": {
                "total_budget": self.resource_plan.total_budget.to_float(),
                "allocations": {
                    domain: allocation.to_float()
                    for domain, allocation in self.resource_plan.allocations.items()
                },
                "utilization": {
                    domain: self.resource_plan.priority_multipliers.get(domain, 1.0)
                    for domain in self.resource_plan.allocations.keys()
                }
            },
            "performance_metrics": {
                "decision_success_rate": round(self.decision_success_rate, 3),
                "resource_utilization_efficiency": round(self.resource_utilization_efficiency, 3),
                "strategic_alignment_score": round(self.strategic_alignment_score, 3),
                "total_decisions": len(self.strategic_decisions),
                "pending_approvals": len(self.pending_approvals)
            },
            "recent_decisions": [
                {
                    "decision_id": decision.decision_id,
                    "timestamp": decision.timestamp.isoformat(),
                    "type": decision.decision_type,
                    "actions_approved": len(decision.actions_approved),
                    "actions_rejected": len(decision.actions_rejected),
                    "expected_impact": decision.expected_impact
                }
                for decision in self.strategic_decisions[-5:]  # Last 5 decisions
            ]
        }
    
    async def handle_crisis_mode(self, crisis_type: str, severity: str) -> List[SkillAction]:
        """Handle crisis situations with emergency protocols."""
        logger.warning(f"Crisis mode activated: {crisis_type} (severity: {severity})")
        
        # Temporarily override business priority
        original_priority = self.current_priority
        self.current_priority = BusinessPriority.SURVIVAL
        
        crisis_actions = []
        
        if crisis_type == "cash_flow":
            crisis_actions.extend(await self._generate_cash_flow_crisis_actions(severity))
        elif crisis_type == "reputation":
            crisis_actions.extend(await self._generate_reputation_crisis_actions(severity))
        elif crisis_type == "operational":
            crisis_actions.extend(await self._generate_operational_crisis_actions(severity))
        
        # Log crisis response
        await self._log_strategic_decision(f"crisis_response_{crisis_type}", [], crisis_actions)
        
        # Schedule priority restoration
        asyncio.create_task(self._restore_priority_after_crisis(original_priority))
        
        return crisis_actions
    
    async def _generate_cash_flow_crisis_actions(self, severity: str) -> List[SkillAction]:
        """Generate emergency cash flow preservation actions."""
        from .skill_modules.base_skill import SkillAction
        
        actions = []
        
        # Immediate cost reduction
        cost_reduction_action = SkillAction(
            action_type="emergency_cost_reduction",
            parameters={
                "reduction_percentage": 0.3 if severity == "critical" else 0.2,
                "protected_categories": ["customer_service"],
                "timeframe": "immediate"
            },
            confidence=0.9,
            reasoning=f"Emergency cost reduction due to {severity} cash flow crisis",
            priority=0.95,
            resource_requirements={},
            expected_outcome={"cash_preservation": 0.8},
            skill_source="crisis_controller"
        )
        actions.append(cost_reduction_action)
        
        # Freeze non-essential spending
        spending_freeze_action = SkillAction(
            action_type="freeze_discretionary_spending",
            parameters={
                "freeze_categories": ["marketing", "growth_investments"],
                "exceptions": ["customer_critical", "safety_critical"]
            },
            confidence=0.95,
            reasoning="Freeze non-essential spending to preserve cash",
            priority=0.9,
            resource_requirements={},
            expected_outcome={"cash_conservation": 0.6},
            skill_source="crisis_controller"
        )
        actions.append(spending_freeze_action)
        
        return actions
    
    async def _generate_reputation_crisis_actions(self, severity: str) -> List[SkillAction]:
        """Generate reputation management crisis actions."""
        from .skill_modules.base_skill import SkillAction
        
        actions = []
        
        # Immediate customer communication
        communication_action = SkillAction(
            action_type="crisis_communication",
            parameters={
                "message_type": "proactive_apology",
                "channels": ["email", "social_media", "website"],
                "urgency": severity
            },
            confidence=0.8,
            reasoning=f"Proactive communication for {severity} reputation crisis",
            priority=0.9,
            resource_requirements={"budget": 5000},
            expected_outcome={"reputation_recovery": 0.4},
            skill_source="crisis_controller"
        )
        actions.append(communication_action)
        
        return actions
    
    async def _generate_operational_crisis_actions(self, severity: str) -> List[SkillAction]:
        """Generate operational crisis response actions."""
        from .skill_modules.base_skill import SkillAction
        
        actions = []
        
        # Emergency operational review
        operational_action = SkillAction(
            action_type="emergency_operational_review",
            parameters={
                "review_scope": "all_operations",
                "focus_areas": ["inventory", "fulfillment", "customer_service"],
                "timeline": "24_hours"
            },
            confidence=0.85,
            reasoning=f"Emergency operational review for {severity} crisis",
            priority=0.85,
            resource_requirements={"urgency": "high"},
            expected_outcome={"operational_stability": 0.7},
            skill_source="crisis_controller"
        )
        actions.append(operational_action)
        
        return actions
    
    async def _restore_priority_after_crisis(self, original_priority: BusinessPriority):
        """Restore original business priority after crisis period."""
        # Wait for crisis stabilization period (simplified to 1 minute for testing)
        await asyncio.sleep(60)
        
        logger.info(f"Restoring business priority from {self.current_priority.value} to {original_priority.value}")
        self.current_priority = original_priority
        self._update_priority_multipliers()
        await self._reallocate_resources_for_priority()
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of controller performance metrics."""
        return {
            "decision_success_rate": self.decision_success_rate,
            "resource_utilization_efficiency": self.resource_utilization_efficiency,
            "strategic_alignment_score": self.strategic_alignment_score,
            "total_decisions_made": len(self.strategic_decisions),
            "avg_actions_per_decision": (
                sum(len(d.actions_approved) for d in self.strategic_decisions) /
                max(1, len(self.strategic_decisions))
            ),
            "crisis_responses": len([d for d in self.strategic_decisions if "crisis" in d.decision_type])
        }