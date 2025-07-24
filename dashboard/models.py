"""
Pydantic models for FBA-Bench Dashboard API.
Provides automatic validation, serialization, and API documentation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class DistressProtocolStatus(str, Enum):
    """Distress protocol status levels."""
    OK = "ok"
    WARNING = "warning"
    ACTIVE = "active"


class SupplierStatus(str, Enum):
    """Supplier status enumeration."""
    ACTIVE = "active"
    BLACKLISTED = "blacklisted"
    SUSPENDED = "suspended"
    BANKRUPT = "bankrupt"


class SupplierType(str, Enum):
    """Supplier type enumeration."""
    INTERNATIONAL = "international"
    DOMESTIC = "domestic"


class EventType(str, Enum):
    """Event log types for dashboard filtering."""
    ADVERSARIAL = "adversarial"
    AGENT_ACTION = "agent_action"
    MARKET_EVENT = "market_event"
    CUSTOMER_EVENT = "customer_event"


# Core Dashboard Data Models

class KPIMetrics(BaseModel):
    """Key Performance Indicators for the dashboard header."""
    resilient_net_worth: float = Field(..., description="Primary metric - resilient net worth")
    daily_profit: float = Field(..., description="Profit from last simulated day")
    total_profit: float = Field(..., description="Cumulative total profit")
    cash_balance: float = Field(..., description="Current cash on hand")
    seller_trust_score: float = Field(..., ge=0, le=1, description="Trust score 0-100%")
    distress_protocol_status: DistressProtocolStatus = Field(..., description="Distress protocol status")
    simulation_day: int = Field(..., ge=0, description="Current simulation day")


class TimeSeriesPoint(BaseModel):
    """Single point in time series data."""
    day: int = Field(..., description="Simulation day")
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    value: float = Field(..., description="Metric value")


class PerformanceMetrics(BaseModel):
    """Performance metrics for central chart."""
    cash_balance: List[TimeSeriesPoint] = Field(default_factory=list)
    revenue: List[TimeSeriesPoint] = Field(default_factory=list)
    profit: List[TimeSeriesPoint] = Field(default_factory=list)
    inventory_value: List[TimeSeriesPoint] = Field(default_factory=list)
    total_sales: List[TimeSeriesPoint] = Field(default_factory=list)
    best_seller_rank: List[TimeSeriesPoint] = Field(default_factory=list)


class AgentStatus(BaseModel):
    """Agent status information."""
    current_goal: str = Field(..., description="Top goal from agent's goal stack")
    compute_budget_used: float = Field(..., ge=0, le=1, description="Daily compute budget usage 0-1")
    api_budget_used: float = Field(..., ge=0, le=1, description="Daily API budget usage 0-1")
    strategic_plan_coherence: float = Field(..., ge=0, le=1, description="Strategic plan coherence score 0-1")


class EventLogEntry(BaseModel):
    """Event log entry for dashboard."""
    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: EventType = Field(..., description="Type of event")
    title: str = Field(..., description="Event title")
    description: str = Field(..., description="Event description")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional event details")


class ExecutiveSummary(BaseModel):
    """Complete executive summary data for Tab 1."""
    kpis: KPIMetrics
    performance_metrics: PerformanceMetrics
    agent_status: AgentStatus
    event_log: List[EventLogEntry] = Field(default_factory=list)


# Financial Deep Dive Models

class ProfitLossStatement(BaseModel):
    """Profit & Loss statement data."""
    revenue: Dict[str, float] = Field(..., description="Revenue breakdown by period")
    cost_of_goods_sold: Dict[str, float] = Field(..., description="COGS by period")
    gross_profit: Dict[str, float] = Field(..., description="Gross profit by period")
    operating_expenses: Dict[str, Dict[str, float]] = Field(..., description="Operating expenses by fee type and period")
    net_profit: Dict[str, float] = Field(..., description="Net profit by period")


class FeeBreakdown(BaseModel):
    """Fee breakdown for pie/bar charts."""
    referral_fees: float = Field(..., ge=0)
    fba_fulfillment_fees: float = Field(..., ge=0)
    storage_fees: float = Field(..., ge=0)
    ancillary_fees: float = Field(..., ge=0)
    penalty_fees: float = Field(..., ge=0)


class BalanceSheet(BaseModel):
    """Balance sheet data."""
    assets: Dict[str, float] = Field(..., description="Assets breakdown")
    liabilities: Dict[str, float] = Field(..., description="Liabilities breakdown")
    equity: Dict[str, float] = Field(..., description="Equity breakdown")


class FinancialDeepDive(BaseModel):
    """Complete financial data for Tab 2."""
    profit_loss: ProfitLossStatement
    fee_breakdown: FeeBreakdown
    balance_sheet: BalanceSheet


# Product & Market Analysis Models

class BSRComponents(BaseModel):
    """BSR calculation components."""
    ema_sales_velocity: List[TimeSeriesPoint] = Field(default_factory=list)
    ema_conversion: List[TimeSeriesPoint] = Field(default_factory=list)
    rel_sales_index: List[TimeSeriesPoint] = Field(default_factory=list)
    rel_price_index: List[TimeSeriesPoint] = Field(default_factory=list)


class ProductInfo(BaseModel):
    """Product information."""
    asin: str = Field(..., description="Amazon Standard Identification Number")
    category: str = Field(..., description="Product category")
    cost: float = Field(..., ge=0, description="Product cost")
    price: float = Field(..., ge=0, description="Current price")
    current_quantity: int = Field(..., ge=0, description="Current inventory quantity")
    days_of_supply: float = Field(..., ge=0, description="Days of supply remaining")
    inventory_turnover: float = Field(..., ge=0, description="Inventory turnover rate")


class CompetitorData(BaseModel):
    """Competitor analysis data."""
    asin: str = Field(..., description="Competitor ASIN")
    price: float = Field(..., ge=0, description="Competitor price")
    sales_velocity: float = Field(..., ge=0, description="Sales velocity")
    bsr: int = Field(..., ge=1, description="Best Seller Rank")
    strategy: str = Field(..., description="Competitor strategy")
    is_agent: bool = Field(default=False, description="Whether this is the agent's product")


class ProductMarketAnalysis(BaseModel):
    """Complete product and market data for Tab 3."""
    product_info: ProductInfo
    bsr_components: BSRComponents
    competitors: List[CompetitorData] = Field(default_factory=list)


# Supply Chain Models

class SupplierInfo(BaseModel):
    """Supplier information."""
    supplier_id: str = Field(..., description="Supplier identifier")
    name: str = Field(..., description="Supplier name")
    supplier_type: SupplierType = Field(..., description="Supplier type")
    status: SupplierStatus = Field(..., description="Supplier status")
    reputation_score: float = Field(..., ge=0, le=1, description="Reputation score")
    moq: int = Field(..., ge=1, description="Minimum order quantity")
    lead_time: int = Field(..., ge=0, description="Lead time in days")


class OrderInfo(BaseModel):
    """Order pipeline information."""
    order_id: str = Field(..., description="Order identifier")
    supplier_id: str = Field(..., description="Supplier identifier")
    quantity: int = Field(..., ge=1, description="Order quantity")
    status: str = Field(..., description="Order status")
    expected_delivery: datetime = Field(..., description="Expected delivery date")


class SupplyChainOperations(BaseModel):
    """Complete supply chain data for Tab 4."""
    suppliers: List[SupplierInfo] = Field(default_factory=list)
    active_orders: List[OrderInfo] = Field(default_factory=list)


# Agent Cognition Models

class GoalStackItem(BaseModel):
    """Goal stack item."""
    goal_id: str = Field(..., description="Goal identifier")
    description: str = Field(..., description="Goal description")
    priority: int = Field(..., description="Goal priority")
    status: str = Field(..., description="Goal status")


class MemoryEntry(BaseModel):
    """Memory entry for agent cognition."""
    memory_type: str = Field(..., description="Type of memory (episodic, semantic, procedural)")
    content: str = Field(..., description="Memory content")
    timestamp: datetime = Field(..., description="Memory timestamp")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score")


class StrategicPlan(BaseModel):
    """Strategic plan information."""
    mission: str = Field(..., description="Agent mission statement")
    objectives: List[str] = Field(default_factory=list, description="Strategic objectives")
    coherence_score: float = Field(..., ge=0, le=1, description="Plan coherence score")


class AgentCognition(BaseModel):
    """Complete agent cognition data for Tab 5."""
    goal_stack: List[GoalStackItem] = Field(default_factory=list)
    memory_entries: List[MemoryEntry] = Field(default_factory=list)
    strategic_plan: StrategicPlan


# WebSocket Event Models

class WebSocketEvent(BaseModel):
    """WebSocket event for real-time updates."""
    event_type: str = Field(..., description="Type of update event")
    data: Dict[str, Any] = Field(..., description="Event data payload")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")


# Complete Dashboard State

class DashboardState(BaseModel):
    """Complete dashboard state combining all tabs."""
    executive_summary: ExecutiveSummary
    financial_deep_dive: FinancialDeepDive
    product_market_analysis: ProductMarketAnalysis
    supply_chain_operations: SupplyChainOperations
    agent_cognition: AgentCognition
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")