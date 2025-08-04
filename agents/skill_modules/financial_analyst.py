"""
Financial Analyst Skill Module for FBA-Bench Multi-Domain Agent Architecture.

This module handles budget management and financial planning, monitoring financial health,
assessing budget constraints, recommending cost optimizations, and forecasting cashflow
through LLM-driven decision making.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .base_skill import BaseSkill, SkillAction, SkillContext, SkillOutcome
from events import BaseEvent, TickEvent, SaleOccurred
from money import Money

logger = logging.getLogger(__name__)


class BudgetStatus(Enum):
    """Budget status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


class ExpenseCategory(Enum):
    """Expense category types."""
    INVENTORY = "inventory"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    CUSTOMER_SERVICE = "customer_service"
    FEES = "fees"
    OTHER = "other"


@dataclass
class BudgetAllocation:
    """
    Budget allocation for different business areas.
    
    Attributes:
        category: Expense category
        allocated_amount: Total allocated budget
        spent_amount: Amount already spent
        remaining_amount: Remaining budget
        period_start: Budget period start date
        period_end: Budget period end date
        utilization_rate: Percentage of budget utilized
        burn_rate: Daily spend rate
        projected_overspend: Projected amount over budget
        status: Current budget status
    """
    category: ExpenseCategory
    allocated_amount: Money
    spent_amount: Money
    remaining_amount: Money
    period_start: datetime
    period_end: datetime
    utilization_rate: float = 0.0
    burn_rate: Money = field(default_factory=lambda: Money(0))
    projected_overspend: Money = field(default_factory=lambda: Money(0))
    status: BudgetStatus = BudgetStatus.HEALTHY


@dataclass
class FinancialForecast:
    """
    Financial forecast analysis for planning and decision making.
    
    Attributes:
        forecast_period: Time period for forecast
        revenue_projection: Projected revenue
        expense_projection: Projected expenses
        profit_projection: Projected profit
        cashflow_projection: Projected cashflow
        confidence_level: Confidence in forecast accuracy
        key_assumptions: Key assumptions underlying forecast
        risk_factors: Identified risk factors
        recommended_actions: Recommended financial actions
    """
    forecast_period: str
    revenue_projection: Money
    expense_projection: Money
    profit_projection: Money
    cashflow_projection: Money
    confidence_level: float
    key_assumptions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class CostOptimization:
    """
    Cost optimization recommendation.
    
    Attributes:
        optimization_id: Unique identifier
        category: Expense category to optimize
        current_spend: Current spending level
        optimized_spend: Recommended spending level
        potential_savings: Potential cost savings
        implementation_effort: Required effort level
        risk_level: Risk associated with optimization
        timeline: Implementation timeline
        description: Description of optimization
        success_probability: Probability of successful implementation
    """
    optimization_id: str
    category: ExpenseCategory
    current_spend: Money
    optimized_spend: Money
    potential_savings: Money
    implementation_effort: str
    risk_level: str
    timeline: str
    description: str
    success_probability: float


class FinancialAnalystSkill(BaseSkill):
    """
    Financial Analyst Skill for budget management and financial planning.
    
    Handles budget monitoring, cost optimization, financial forecasting,
    and strategic financial decision making to maintain profitability
    and optimize resource allocation.
    """
    
    def __init__(self, agent_id: str, event_bus, config: Dict[str, Any] = None):
        """
        Initialize the Financial Analyst Skill.
        
        Args:
            agent_id: ID of the agent this skill belongs to
            event_bus: Event bus for communication
            config: Configuration parameters for financial analysis
        """
        super().__init__("FinancialAnalyst", agent_id, event_bus)
        
        # Configuration parameters
        self.config = config or {}
        self.total_budget = Money(self.config.get('total_budget_cents', 1000000))  # $10,000
        self.budget_warning_threshold = self.config.get('warning_threshold', 0.8)  # 80%
        self.budget_critical_threshold = self.config.get('critical_threshold', 0.95)  # 95%
        self.min_cash_reserve = Money(self.config.get('min_cash_reserve_cents', 100000))  # $1,000
        
        # Budget tracking
        self.budget_allocations: Dict[ExpenseCategory, BudgetAllocation] = {}
        self.expense_history: List[Dict[str, Any]] = []
        self.revenue_history: List[Dict[str, Any]] = []
        
        # Financial state
        self.current_cash: Money = Money(self.config.get('starting_cash_cents', 500000))  # $5,000
        self.total_revenue: Money = Money(0)
        self.total_expenses: Money = Money(0)
        self.profit_margin: float = 0.0
        
        # Analysis and forecasting
        self.financial_forecasts: List[FinancialForecast] = []
        self.cost_optimizations: List[CostOptimization] = []
        self.risk_alerts: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.budget_accuracy: float = 0.8
        self.forecast_accuracy: float = 0.7
        self.cost_savings_achieved: Money = Money(0)
        self.last_analysis_date = datetime.now()
        
        # Initialize budget allocations
        self._initialize_budget_allocations()
        
        logger.info(f"FinancialAnalystSkill initialized for agent {agent_id}")
    
    def _initialize_budget_allocations(self):
        """Initialize budget allocations for different categories."""
        # Default budget allocation percentages
        allocation_percentages = {
            ExpenseCategory.INVENTORY: 0.40,  # 40% for inventory
            ExpenseCategory.MARKETING: 0.25,  # 25% for marketing
            ExpenseCategory.OPERATIONS: 0.20,  # 20% for operations
            ExpenseCategory.CUSTOMER_SERVICE: 0.10,  # 10% for customer service
            ExpenseCategory.FEES: 0.05  # 5% for fees and other expenses
        }
        
        period_start = datetime.now()
        period_end = period_start + timedelta(days=30)  # Monthly budgets
        
        for category, percentage in allocation_percentages.items():
            allocated_amount = Money(int(self.total_budget.cents * percentage))
            
            self.budget_allocations[category] = BudgetAllocation(
                category=category,
                allocated_amount=allocated_amount,
                spent_amount=Money(0),
                remaining_amount=allocated_amount,
                period_start=period_start,
                period_end=period_end
            )
    
    async def process_event(self, event: BaseEvent) -> Optional[List[SkillAction]]:
        """
        Process events relevant to financial analysis and generate actions.
        
        Args:
            event: Event to process
            
        Returns:
            List of financial analysis actions or None
        """
        actions = []
        
        try:
            if isinstance(event, SaleOccurred):
                actions.extend(await self._handle_sale_occurred(event))
            elif isinstance(event, TickEvent):
                actions.extend(await self._handle_tick_event(event))
            
            # Filter actions by confidence threshold
            confidence_threshold = self.adaptation_parameters.get('confidence_threshold', 0.6)
            filtered_actions = [action for action in actions if action.confidence >= confidence_threshold]
            
            return filtered_actions if filtered_actions else None
            
        except Exception as e:
            logger.error(f"Error processing event in FinancialAnalystSkill: {e}")
            return None
    
    async def _handle_sale_occurred(self, event: SaleOccurred) -> List[SkillAction]:
        """Handle sales events for revenue tracking and financial analysis."""
        actions = []
        
        # Update revenue tracking
        self.total_revenue = Money(self.total_revenue.cents + event.total_revenue.cents)
        self.revenue_history.append({
            "timestamp": datetime.now(),
            "revenue": event.total_revenue,
            "profit": event.total_profit,
            "asin": event.asin,
            "units": event.units_sold
        })
        
        # Update current cash (simplified)
        self.current_cash = Money(self.current_cash.cents + event.total_profit.cents)
        
        # Analyze profitability
        profit_analysis_action = await self._analyze_profitability(event)
        if profit_analysis_action:
            actions.append(profit_analysis_action)
        
        # Check for budget impact from fees
        if event.total_fees.cents > 0:
            fee_impact_action = await self._analyze_fee_impact(event)
            if fee_impact_action:
                actions.append(fee_impact_action)
        
        return actions
    
    async def _handle_tick_event(self, event: TickEvent) -> List[SkillAction]:
        """Handle periodic tick events for financial monitoring and analysis."""
        actions = []
        
        # Budget monitoring every few ticks
        if event.tick_number % 3 == 0:
            budget_actions = await self._monitor_budgets()
            actions.extend(budget_actions)
        
        # Financial health analysis every 5 ticks
        if event.tick_number % 5 == 0:
            health_actions = await self._assess_financial_health()
            actions.extend(health_actions)
        
        # Cost optimization analysis every 10 ticks
        if event.tick_number % 10 == 0:
            optimization_actions = await self._identify_cost_optimizations()
            actions.extend(optimization_actions)
        
        # Cashflow forecasting every 15 ticks
        if event.tick_number % 15 == 0:
            forecast_actions = await self._forecast_cashflow()
            actions.extend(forecast_actions)
        
        return actions
    
    async def generate_actions(self, context: SkillContext, constraints: Dict[str, Any]) -> List[SkillAction]:
        """
        Generate financial analysis actions based on current context.
        
        Args:
            context: Current context information
            constraints: Active constraints and limits
            
        Returns:
            List of recommended financial actions
        """
        actions = []
        
        try:
            # Analyze current financial position
            current_financial_state = await self._analyze_current_financial_state(context)
            
            # Generate budget recommendations
            budget_actions = await self._generate_budget_recommendations(current_financial_state, constraints)
            actions.extend(budget_actions)
            
            # Generate cost optimization recommendations
            if current_financial_state.get('cash_position', 'healthy') in ['warning', 'critical']:
                cost_actions = await self._generate_cost_reduction_actions(constraints)
                actions.extend(cost_actions)
            
            # Generate investment recommendations
            if current_financial_state.get('cash_position', 'healthy') == 'healthy':
                investment_actions = await self._generate_investment_recommendations(context, constraints)
                actions.extend(investment_actions)
            
            # Sort by financial impact and urgency
            actions.sort(key=lambda x: x.priority * self._calculate_financial_impact_score(x), reverse=True)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating financial analysis actions: {e}")
            return []
    
    def get_priority_score(self, event: BaseEvent) -> float:
        """Calculate priority score for financial analysis events."""
        if isinstance(event, SaleOccurred):
            # Higher priority for high-value sales or loss-making sales
            priority = 0.4  # Base priority
            if event.total_revenue.cents > 20000:  # $200+ sale
                priority += 0.2
            if event.total_profit.cents < 0:  # Loss-making sale
                priority += 0.4
            return min(0.9, priority)
        
        elif isinstance(event, TickEvent):
            # Regular financial monitoring
            return 0.3
        
        return 0.2
    
    async def _monitor_budgets(self) -> List[SkillAction]:
        """Monitor budget utilization and generate alerts."""
        actions = []
        
        for category, allocation in self.budget_allocations.items():
            # Update budget status
            utilization = allocation.spent_amount.cents / allocation.allocated_amount.cents if allocation.allocated_amount.cents > 0 else 0
            allocation.utilization_rate = utilization
            
            # Determine status
            if utilization >= 1.0:
                allocation.status = BudgetStatus.EXCEEDED
            elif utilization >= self.budget_critical_threshold:
                allocation.status = BudgetStatus.CRITICAL
            elif utilization >= self.budget_warning_threshold:
                allocation.status = BudgetStatus.WARNING
            else:
                allocation.status = BudgetStatus.HEALTHY
            
            # Generate alerts for problematic budgets
            if allocation.status in [BudgetStatus.WARNING, BudgetStatus.CRITICAL, BudgetStatus.EXCEEDED]:
                alert_action = await self._create_budget_alert_action(allocation)
                if alert_action:
                    actions.append(alert_action)
        
        return actions
    
    async def _create_budget_alert_action(self, allocation: BudgetAllocation) -> Optional[SkillAction]:
        """Create budget alert action for budget issues."""
        try:
            severity = {
                BudgetStatus.WARNING: "warning",
                BudgetStatus.CRITICAL: "critical", 
                BudgetStatus.EXCEEDED: "exceeded"
            }[allocation.status]
            
            # Calculate recommended action based on severity
            if allocation.status == BudgetStatus.EXCEEDED:
                recommended_action = "immediate_spend_freeze"
                priority = 0.9
            elif allocation.status == BudgetStatus.CRITICAL:
                recommended_action = "reduce_spending_immediately"
                priority = 0.8
            else:  # WARNING
                recommended_action = "monitor_and_optimize"
                priority = 0.6
            
            return SkillAction(
                action_type="budget_alert",
                parameters={
                    "category": allocation.category.value,
                    "severity": severity,
                    "utilization_rate": allocation.utilization_rate,
                    "allocated_amount": allocation.allocated_amount.cents,
                    "spent_amount": allocation.spent_amount.cents,
                    "remaining_amount": allocation.remaining_amount.cents,
                    "recommended_action": recommended_action,
                    "days_remaining": (allocation.period_end - datetime.now()).days
                },
                confidence=0.9,  # High confidence in budget data
                reasoning=f"Budget {severity} for {allocation.category.value}: {allocation.utilization_rate:.1%} utilized",
                priority=priority,
                resource_requirements={
                    "budget_management_authority": True
                },
                expected_outcome={
                    "budget_compliance": 0.8 if recommended_action != "immediate_spend_freeze" else 1.0,
                    "cost_control": 0.6
                },
                skill_source=self.skill_name
            )
            
        except Exception as e:
            logger.error(f"Error creating budget alert action: {e}")
            return None
    
    async def _assess_financial_health(self) -> List[SkillAction]:
        """Assess overall financial health and generate recommendations."""
        actions = []
        
        # Calculate key financial metrics
        profit_margin = self._calculate_profit_margin()
        cash_runway = self._calculate_cash_runway()
        burn_rate = self._calculate_burn_rate()
        
        # Generate health assessment
        health_score = self._calculate_financial_health_score(profit_margin, cash_runway, burn_rate)
        
        # Generate recommendations based on health score
        if health_score < 0.5:  # Poor financial health
            health_action = await self._create_financial_health_action(health_score, "poor")
            if health_action:
                actions.append(health_action)
        elif health_score < 0.7:  # Moderate financial health
            health_action = await self._create_financial_health_action(health_score, "moderate")
            if health_action:
                actions.append(health_action)
        
        return actions
    
    async def _create_financial_health_action(self, health_score: float, health_level: str) -> Optional[SkillAction]:
        """Create financial health assessment action."""
        try:
            # Determine recommendations based on health level
            recommendations = []
            priority = 0.5
            
            if health_level == "poor":
                recommendations = [
                    "immediate_cost_reduction",
                    "revenue_optimization",
                    "cash_conservation"
                ]
                priority = 0.8
            elif health_level == "moderate":
                recommendations = [
                    "profit_margin_improvement",
                    "expense_optimization",
                    "revenue_growth_focus"
                ]
                priority = 0.6
            
            return SkillAction(
                action_type="assess_financial_health",
                parameters={
                    "health_score": health_score,
                    "health_level": health_level,
                    "recommendations": recommendations,
                    "profit_margin": self.profit_margin,
                    "cash_runway_days": self._calculate_cash_runway(),
                    "burn_rate": self._calculate_burn_rate().cents,
                    "total_cash": self.current_cash.cents
                },
                confidence=0.8,
                reasoning=f"Financial health assessment: {health_level} (score: {health_score:.2f})",
                priority=priority,
                resource_requirements={
                    "financial_planning_authority": True
                },
                expected_outcome={
                    "financial_stability_improvement": 0.3,
                    "risk_mitigation": 0.4
                },
                skill_source=self.skill_name
            )
            
        except Exception as e:
            logger.error(f"Error creating financial health action: {e}")
            return None
    
    async def _identify_cost_optimizations(self) -> List[SkillAction]:
        """Identify cost optimization opportunities."""
        actions = []
        
        # Analyze each expense category for optimization potential
        for category, allocation in self.budget_allocations.items():
            optimization = await self._analyze_category_optimization(category, allocation)
            if optimization and optimization.potential_savings.cents > 0:
                optimization_action = await self._create_cost_optimization_action(optimization)
                if optimization_action:
                    actions.append(optimization_action)
        
        return actions
    
    async def _analyze_category_optimization(self, category: ExpenseCategory, allocation: BudgetAllocation) -> Optional[CostOptimization]:
        """Analyze cost optimization potential for a category."""
        try:
            # Calculate current utilization and efficiency
            if allocation.spent_amount.cents == 0:
                return None
            
            # Simplified optimization analysis
            optimization_potential = self._calculate_optimization_potential(category, allocation)
            
            if optimization_potential > 0.1:  # At least 10% savings potential
                potential_savings = Money(int(allocation.spent_amount.cents * optimization_potential))
                optimized_spend = Money(allocation.spent_amount.cents - potential_savings.cents)
                
                return CostOptimization(
                    optimization_id=f"{category.value}_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    category=category,
                    current_spend=allocation.spent_amount,
                    optimized_spend=optimized_spend,
                    potential_savings=potential_savings,
                    implementation_effort="medium",
                    risk_level="low" if optimization_potential < 0.2 else "medium",
                    timeline="1-2 weeks",
                    description=f"Optimize {category.value} spending through efficiency improvements",
                    success_probability=0.8 if optimization_potential < 0.3 else 0.6
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing category optimization: {e}")
            return None
    
    def _calculate_optimization_potential(self, category: ExpenseCategory, allocation: BudgetAllocation) -> float:
        """Calculate optimization potential for a category."""
        # Simplified calculation based on category type and utilization
        base_potential = {
            ExpenseCategory.INVENTORY: 0.15,  # 15% potential savings
            ExpenseCategory.MARKETING: 0.20,  # 20% potential savings
            ExpenseCategory.OPERATIONS: 0.10,  # 10% potential savings
            ExpenseCategory.CUSTOMER_SERVICE: 0.12,  # 12% potential savings
            ExpenseCategory.FEES: 0.05   # 5% potential savings (less control)
        }
        
        potential = base_potential.get(category, 0.1)
        
        # Adjust based on utilization rate
        if allocation.utilization_rate > 0.9:  # High utilization suggests inefficiency
            potential *= 1.3
        elif allocation.utilization_rate < 0.5:  # Low utilization might mean waste
            potential *= 1.2
        
        return min(0.3, potential)  # Cap at 30% potential savings
    
    async def _create_cost_optimization_action(self, optimization: CostOptimization) -> Optional[SkillAction]:
        """Create cost optimization action."""
        try:
            return SkillAction(
                action_type="optimize_costs",
                parameters={
                    "optimization_id": optimization.optimization_id,
                    "category": optimization.category.value,
                    "current_spend": optimization.current_spend.cents,
                    "target_spend": optimization.optimized_spend.cents,
                    "potential_savings": optimization.potential_savings.cents,
                    "implementation_plan": [
                        "analyze_current_efficiency",
                        "identify_waste_areas",
                        "implement_optimizations",
                        "monitor_results"
                    ],
                    "timeline": optimization.timeline,
                    "risk_mitigation": "gradual_implementation"
                },
                confidence=optimization.success_probability,
                reasoning=f"Cost optimization opportunity: ${optimization.potential_savings.to_float():.2f} savings in {optimization.category.value}",
                priority=min(0.8, optimization.potential_savings.cents / 10000.0),  # Higher priority for larger savings
                resource_requirements={
                    "implementation_effort": optimization.implementation_effort,
                    "budget_adjustment_authority": True
                },
                expected_outcome={
                    "cost_savings": optimization.potential_savings.cents,
                    "efficiency_improvement": 0.2,
                    "roi_estimate": optimization.potential_savings.cents / max(1, optimization.current_spend.cents)
                },
                skill_source=self.skill_name
            )
            
        except Exception as e:
            logger.error(f"Error creating cost optimization action: {e}")
            return None
    
    async def _forecast_cashflow(self) -> List[SkillAction]:
        """Forecast cashflow and generate planning actions."""
        actions = []
        
        # Generate cashflow forecast
        forecast = await self._generate_financial_forecast("30_days")
        
        if forecast:
            self.financial_forecasts.append(forecast)
            
            # Generate actions based on forecast
            if forecast.cashflow_projection.cents < self.min_cash_reserve.cents:
                cashflow_action = await self._create_cashflow_alert_action(forecast)
                if cashflow_action:
                    actions.append(cashflow_action)
        
        return actions
    
    async def _generate_financial_forecast(self, period: str) -> Optional[FinancialForecast]:
        """Generate financial forecast for specified period."""
        try:
            # Calculate historical averages for projection
            if len(self.revenue_history) < 3:
                return None  # Need sufficient data
            
            recent_revenue = [r["revenue"] for r in self.revenue_history[-7:]]  # Last 7 sales
            avg_revenue_per_sale = Money(sum(r.cents for r in recent_revenue) // len(recent_revenue))
            
            # Project based on period
            if period == "30_days":
                projected_sales = 15  # Assume 15 sales per month
                days = 30
            elif period == "7_days":
                projected_sales = 4   # Assume 4 sales per week
                days = 7
            else:
                projected_sales = 10
                days = 30
            
            revenue_projection = Money(avg_revenue_per_sale.cents * projected_sales)
            
            # Calculate expense projection based on current burn rate
            daily_burn = self._calculate_burn_rate()
            expense_projection = Money(daily_burn.cents * days)
            
            # Calculate profit and cashflow
            profit_projection = Money(revenue_projection.cents - expense_projection.cents)
            current_cash_projection = Money(self.current_cash.cents + profit_projection.cents)
            
            return FinancialForecast(
                forecast_period=period,
                revenue_projection=revenue_projection,
                expense_projection=expense_projection,
                profit_projection=profit_projection,
                cashflow_projection=current_cash_projection,
                confidence_level=0.7,  # Moderate confidence
                key_assumptions=[
                    f"Average revenue per sale: ${avg_revenue_per_sale.to_float():.2f}",
                    f"Projected sales: {projected_sales}",
                    f"Daily burn rate: ${daily_burn.to_float():.2f}"
                ],
                risk_factors=[
                    "Market volatility",
                    "Unexpected expenses",
                    "Sales volume variation"
                ],
                recommended_actions=[
                    "Monitor cash position closely",
                    "Maintain expense discipline",
                    "Focus on profitable products"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error generating financial forecast: {e}")
            return None
    
    async def _create_cashflow_alert_action(self, forecast: FinancialForecast) -> Optional[SkillAction]:
        """Create cashflow alert action for low cash projections."""
        try:
            shortage_amount = Money(self.min_cash_reserve.cents - forecast.cashflow_projection.cents)
            
            return SkillAction(
                action_type="cashflow_alert",
                parameters={
                    "forecast_period": forecast.forecast_period,
                    "projected_cash": forecast.cashflow_projection.cents,
                    "minimum_required": self.min_cash_reserve.cents,
                    "shortage_amount": shortage_amount.cents,
                    "recommended_actions": [
                        "accelerate_collections",
                        "defer_non_critical_expenses",
                        "explore_short_term_financing",
                        "optimize_working_capital"
                    ],
                    "urgency": "high" if shortage_amount.cents > 50000 else "medium"
                },
                confidence=forecast.confidence_level,
                reasoning=f"Cashflow forecast shows potential shortage of ${shortage_amount.to_float():.2f}",
                priority=0.9,
                resource_requirements={
                    "cash_management_authority": True,
                    "financial_planning": True
                },
                expected_outcome={
                    "cash_position_improvement": shortage_amount.cents,
                    "financial_stability": 0.6
                },
                skill_source=self.skill_name
            )
            
        except Exception as e:
            logger.error(f"Error creating cashflow alert action: {e}")
            return None
    
    def _calculate_profit_margin(self) -> float:
        """Calculate current profit margin."""
        if self.total_revenue.cents == 0:
            return 0.0
        
        profit = self.total_revenue.cents - self.total_expenses.cents
        margin = profit / self.total_revenue.cents
        self.profit_margin = margin
        return margin
    
    def _calculate_cash_runway(self) -> int:
        """Calculate cash runway in days."""
        daily_burn = self._calculate_burn_rate()
        if daily_burn.cents <= 0:
            return 365  # If no burn or positive cash generation
        
        return max(0, self.current_cash.cents // daily_burn.cents)
    
    def _calculate_burn_rate(self) -> Money:
        """Calculate daily cash burn rate."""
        if len(self.expense_history) < 2:
            return Money(100)  # Default $1/day burn rate
        
        # Calculate average daily expenses from recent history
        recent_expenses = self.expense_history[-7:]  # Last 7 expense records
        total_expenses = sum(exp.get("amount", 0) for exp in recent_expenses)
        total_days = len(recent_expenses)
        
        daily_burn = total_expenses // max(1, total_days)
        return Money(int(daily_burn))
    
    def _calculate_financial_health_score(self, profit_margin: float, cash_runway: int, burn_rate: Money) -> float:
        """Calculate overall financial health score (0.0 to 1.0)."""
        # Profit margin component (0-0.4)
        margin_score = min(0.4, max(0.0, profit_margin * 2))  # 20% margin = 0.4 score
        
        # Cash runway component (0-0.4)
        runway_score = min(0.4, cash_runway / 90.0)  # 90 days = 0.4 score
        
        # Burn rate efficiency component (0-0.2)
        # Lower burn rate relative to revenue is better
        if self.total_revenue.cents > 0:
            burn_efficiency = 1.0 - min(1.0, (burn_rate.cents * 30) / self.total_revenue.cents)
            burn_score = burn_efficiency * 0.2
        else:
            burn_score = 0.1  # Neutral score if no revenue yet
        
        total_score = margin_score + runway_score + burn_score
        return min(1.0, total_score)
    
    def _calculate_financial_impact_score(self, action: SkillAction) -> float:
        """Calculate financial impact score for action prioritization."""
        impact_score = 0.5  # Base score
        
        # Adjust based on action type
        if action.action_type == "budget_alert":
            impact_score += 0.3
        elif action.action_type == "optimize_costs":
            potential_savings = action.parameters.get("potential_savings", 0)
            impact_score += min(0.4, potential_savings / 50000.0)  # $500 = 0.4 boost
        elif action.action_type == "cashflow_alert":
            impact_score += 0.4
        
        return min(1.0, impact_score)
    
    async def _analyze_current_financial_state(self, context: SkillContext) -> Dict[str, Any]:
        """Analyze current financial state for planning."""
        profit_margin = self._calculate_profit_margin()
        cash_runway = self._calculate_cash_runway()
        health_score = self._calculate_financial_health_score(profit_margin, cash_runway, self._calculate_burn_rate())
        
        # Determine cash position status
        if self.current_cash.cents < self.min_cash_reserve.cents:
            cash_position = "critical"
        elif self.current_cash.cents < self.min_cash_reserve.cents * 2:
            cash_position = "warning"
        else:
            cash_position = "healthy"
        
        return {
            "profit_margin": profit_margin,
            "cash_runway_days": cash_runway,
            "health_score": health_score,
            "cash_position": cash_position,
            "current_cash": self.current_cash.cents,
            "total_revenue": self.total_revenue.cents,
            "total_expenses": self.total_expenses.cents
        }
    
    async def _generate_budget_recommendations(self, financial_state: Dict[str, Any], constraints: Dict[str, Any]) -> List[SkillAction]:
        """Generate budget-related recommendations."""
        actions = []
        
        # Check for budget reallocation opportunities
        if financial_state["cash_position"] in ["warning", "critical"]:
            reallocation_action = await self._create_budget_reallocation_action(financial_state)
            if reallocation_action:
                actions.append(reallocation_action)
        
        return actions
    
    async def _create_budget_reallocation_action(self, financial_state: Dict[str, Any]) -> Optional[SkillAction]:
        """Create budget reallocation action."""
        try:
            # Identify categories with underutilization
            underutilized = []
            overutilized = []
            
            for category, allocation in self.budget_allocations.items():
                if allocation.utilization_rate < 0.7:  # Less than 70% used
                    underutilized.append((category, allocation))
                elif allocation.utilization_rate > 0.9:  # More than 90% used
                    overutilized.append((category, allocation))
            
            if underutilized and overutilized:
                return SkillAction(
                    action_type="reallocate_budget",
                    parameters={
                        "underutilized_categories": [cat.value for cat, _ in underutilized],
                        "overutilized_categories": [cat.value for cat, _ in overutilized],
                        "reallocation_amount": min(10000, sum(alloc.remaining_amount.cents for _, alloc in underutilized) // 2),
                        "justification": "Optimize budget allocation based on actual utilization patterns"
                    },
                    confidence=0.8,
                    reasoning="Budget reallocation needed to optimize spending efficiency",
                    priority=0.7,
                    resource_requirements={
                        "budget_reallocation_authority": True
                    },
                    expected_outcome={
                        "budget_efficiency_improvement": 0.2,
                        "spending_optimization": 0.15
                    },
                    skill_source=self.skill_name
                )
        
        except Exception as e:
            logger.error(f"Error creating budget reallocation action: {e}")
        
        return None
    
    async def _generate_cost_reduction_actions(self, constraints: Dict[str, Any]) -> List[SkillAction]:
        """Generate cost reduction actions for poor financial health."""
        actions = []
        
        # Generate immediate cost reduction action
        reduction_action = SkillAction(
            action_type="implement_cost_reduction",
            parameters={
                "reduction_target": 0.2,  # 20% cost reduction
                "priority_categories": ["marketing", "operations"],
                "timeframe": "immediate",
                "preservation_categories": ["inventory"],  # Don't cut inventory
                "reduction_strategies": [
                    "defer_non_critical_expenses",
                    "renegotiate_supplier_terms",
                    "optimize_operational_efficiency"
                ]
            },
            confidence=0.8,
            reasoning="Immediate cost reduction needed due to poor financial health",
            priority=0.9,
            resource_requirements={
                "cost_reduction_authority": True
            },
            expected_outcome={
                "monthly_savings": self.total_expenses.cents * 0.2,
                "cash_runway_extension": 30  # days
            },
            skill_source=self.skill_name
        )
        
        actions.append(reduction_action)
        return actions
    
    async def _generate_investment_recommendations(self, context: SkillContext, constraints: Dict[str, Any]) -> List[SkillAction]:
        """Generate investment recommendations for healthy financial position."""
        actions = []
        
        # Recommend growth investments if financially healthy
        if self.current_cash.cents > self.min_cash_reserve.cents * 3:  # 3x cash reserve
            investment_action = SkillAction(
                action_type="recommend_growth_investment",
                parameters={
                    "investment_amount": min(50000, (self.current_cash.cents - self.min_cash_reserve.cents) // 2),
                    "investment_categories": ["marketing", "inventory"],
                    "expected_roi": 1.5,  # 150% ROI
                    "risk_level": "moderate",
                    "timeframe": "3_months"
                },
                confidence=0.7,
                reasoning="Strong cash position allows for growth investments",
                priority=0.5,
                resource_requirements={
                    "investment_authority": True
                },
                expected_outcome={
                    "revenue_growth": 0.3,
                    "market_share_increase": 0.1
                },
                skill_source=self.skill_name
            )
            
            actions.append(investment_action)
        
        return actions
    
    async def _analyze_profitability(self, event: SaleOccurred) -> Optional[SkillAction]:
        """Analyze profitability of sale and generate recommendations."""
        try:
            # Calculate profitability metrics
            profit_margin = event.total_profit.cents / event.total_revenue.cents if event.total_revenue.cents > 0 else 0
            
            # Generate action if profitability is concerning
            if profit_margin < 0.1:  # Less than 10% margin
                return SkillAction(
                    action_type="analyze_profitability",
                    parameters={
                        "asin": event.asin,
                        "profit_margin": profit_margin,
                        "unit_profit": event.total_profit.cents // max(1, event.units_sold),
                        "concern_level": "low_margin",
                        "recommended_actions": [
                            "review_pricing_strategy",
                            "analyze_cost_structure",
                            "consider_product_mix_optimization"
                        ]
                    },
                    confidence=0.8,
                    reasoning=f"Low profit margin detected: {profit_margin:.1%} for {event.asin}",
                    priority=0.6,
                    resource_requirements={
                        "profitability_analysis": True
                    },
                    expected_outcome={
                        "margin_improvement": 0.05,
                        "cost_optimization": 0.1
                    },
                    skill_source=self.skill_name
                )
        
        except Exception as e:
            logger.error(f"Error analyzing profitability: {e}")
        
        return None
    
    async def _analyze_fee_impact(self, event: SaleOccurred) -> Optional[SkillAction]:
        """Analyze impact of fees on profitability."""
        try:
            fee_percentage = event.total_fees.cents / event.total_revenue.cents if event.total_revenue.cents > 0 else 0
            
            # Track fee in expenses
            self.total_expenses = Money(self.total_expenses.cents + event.total_fees.cents)
            self.expense_history.append({
                "timestamp": datetime.now(),
                "amount": event.total_fees.cents,
                "category": "fees",
                "description": f"Fees for sale {event.event_id}"
            })
            
            # Update budget allocation for fees
            if ExpenseCategory.FEES in self.budget_allocations:
                allocation = self.budget_allocations[ExpenseCategory.FEES]
                allocation.spent_amount = Money(allocation.spent_amount.cents + event.total_fees.cents)
                allocation.remaining_amount = Money(allocation.allocated_amount.cents - allocation.spent_amount.cents)
            
            # Generate action if fee percentage is high
            if fee_percentage > 0.2:  # More than 20% fees
                return SkillAction(
                    action_type="analyze_fee_impact",
                    parameters={
                        "asin": event.asin,
                        "fee_percentage": fee_percentage,
                        "fee_amount": event.total_fees.cents,
                        "fee_types": list(event.fee_breakdown.keys()) if event.fee_breakdown else ["unknown"],
                        "mitigation_strategies": [
                            "optimize_product_dimensions",
                            "review_fulfillment_method",
                            "analyze_fee_structure"
                        ]
                    },
                    confidence=0.9,
                    reasoning=f"High fee percentage detected: {fee_percentage:.1%} for {event.asin}",
                    priority=0.5,
                    resource_requirements={
                        "fee_optimization": True
                    },
                    expected_outcome={
                        "fee_reduction": event.total_fees.cents * 0.1,
                        "margin_improvement": 0.02
                    },
                    skill_source=self.skill_name
                )
        
        except Exception as e:
            logger.error(f"Error analyzing fee impact: {e}")
        
        return None

    # Public methods for external access
    async def assess_financial_health(self) -> Dict[str, float]:
        """Public method to assess financial health for external use."""
        profit_margin = self._calculate_profit_margin()
        cash_runway = self._calculate_cash_runway()
        burn_rate = self._calculate_burn_rate()
        health_score = self._calculate_financial_health_score(profit_margin, cash_runway, burn_rate)
        
        return {
            "health_score": health_score,
            "profit_margin": profit_margin,
            "cash_runway_days": cash_runway,
            "daily_burn_rate": burn_rate.to_float(),
            "current_cash": self.current_cash.to_float()
        }
    
    async def recommend_cost_cuts(self, target_reduction: float) -> List[Dict[str, Any]]:
        """Public method to recommend cost cuts for external use."""
        recommendations = []
        
        for category, allocation in self.budget_allocations.items():
            if category == ExpenseCategory.INVENTORY:
                continue  # Don't recommend cutting inventory
            
            potential_cut = allocation.spent_amount.cents * target_reduction
            if potential_cut > 1000:  # Only recommend cuts > $10
                recommendations.append({
                    "category": category.value,
                    "current_spend": allocation.spent_amount.to_float(),
                    "recommended_cut": potential_cut / 100.0,  # Convert to dollars
                    "impact": "low" if potential_cut < 5000 else "medium",
                    "implementation": "immediate"
                })
        
        return recommendations
    
    async def forecast_cashflow(self, period_days: int) -> Dict[str, float]:
        """Public method to forecast cashflow for external use."""
        period_str = f"{period_days}_days"
        forecast = await self._generate_financial_forecast(period_str)
        
        if forecast:
            return {
                "period_days": period_days,
                "revenue_projection": forecast.revenue_projection.to_float(),
                "expense_projection": forecast.expense_projection.to_float(),
                "profit_projection": forecast.profit_projection.to_float(),
                "cashflow_projection": forecast.cashflow_projection.to_float(),
                "confidence_level": forecast.confidence_level
            }
        
        return {
            "period_days": period_days,
            "revenue_projection": 0.0,
            "expense_projection": 0.0,
            "profit_projection": 0.0,
            "cashflow_projection": self.current_cash.to_float(),
            "confidence_level": 0.0
        }

# Alias for backward compatibility
FinancialAnalyst = FinancialAnalystSkill