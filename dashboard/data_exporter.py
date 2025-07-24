"""
Data export layer for FBA-Bench Dashboard.
Extracts data from simulation components and converts to Pydantic models.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .models import (
    # Core models
    KPIMetrics, PerformanceMetrics, AgentStatus, EventLogEntry, ExecutiveSummary,
    ProfitLossStatement, FeeBreakdown, BalanceSheet, FinancialDeepDive,
    ProductInfo, BSRComponents, CompetitorData, ProductMarketAnalysis,
    SupplierInfo, OrderInfo, SupplyChainOperations,
    GoalStackItem, MemoryEntry, StrategicPlan, AgentCognition,
    DashboardState, TimeSeriesPoint,
    # Enums
    DistressProtocolStatus, EventType, SupplierStatus, SupplierType
)


class DashboardDataExporter:
    """
    Exports data from FBA-Bench simulation components to dashboard models.
    Non-invasive integration that doesn't modify core simulation logic.
    """
    
    def __init__(self, simulation, agent=None):
        """
        Initialize the data exporter.
        
        Args:
            simulation: FBA-Bench Simulation instance
            agent: AdvancedAgent instance (optional)
        """
        self.simulation = simulation
        self.agent = agent
        self._time_series_cache = defaultdict(list)
    
    def extract_kpi_metrics(self) -> KPIMetrics:
        """Extract KPI metrics for dashboard header."""
        # Calculate resilient net worth (cash + inventory value - liabilities)
        cash_balance = self.simulation.ledger.balance("Cash")
        inventory_value = self._calculate_inventory_value()
        resilient_net_worth = cash_balance + inventory_value
        
        # Calculate daily and total profit
        daily_profit = self._calculate_daily_profit()
        total_profit = self._calculate_total_profit()
        
        # Get seller trust score
        trust_score = getattr(self.simulation, 'trust_score', 1.0)
        
        # Determine distress protocol status
        distress_status = self._get_distress_protocol_status()
        
        # Get current simulation day
        simulation_day = getattr(self.simulation, 'day', 0)
        
        return KPIMetrics(
            resilient_net_worth=resilient_net_worth,
            daily_profit=daily_profit,
            total_profit=total_profit,
            cash_balance=cash_balance,
            seller_trust_score=trust_score,
            distress_protocol_status=distress_status,
            simulation_day=simulation_day
        )
    
    def extract_performance_metrics(self) -> PerformanceMetrics:
        """Extract time-series performance metrics."""
        current_day = getattr(self.simulation, 'day', 0)
        current_time = datetime.now()
        
        # Update time series cache with current values
        self._update_time_series_cache(current_day, current_time)
        
        return PerformanceMetrics(
            cash_balance=self._time_series_cache['cash_balance'],
            revenue=self._time_series_cache['revenue'],
            profit=self._time_series_cache['profit'],
            inventory_value=self._time_series_cache['inventory_value'],
            total_sales=self._time_series_cache['total_sales'],
            best_seller_rank=self._time_series_cache['best_seller_rank']
        )
    
    def extract_agent_status(self) -> AgentStatus:
        """Extract agent status information."""
        if not self.agent:
            return AgentStatus(
                current_goal="No agent connected",
                compute_budget_used=0.0,
                api_budget_used=0.0,
                strategic_plan_coherence=0.0
            )
        
        # Get current goal from goal stack
        current_goal = "No active goals"
        if hasattr(self.agent, 'goal_stack') and self.agent.goal_stack:
            current_goal = str(self.agent.goal_stack[0]) if self.agent.goal_stack else "No active goals"
        
        # Calculate budget usage
        compute_budget_used = getattr(self.agent, 'cpu_units_used', 0) / getattr(self.agent, 'cpu_budget', 100)
        api_budget_used = getattr(self.agent, 'api_cost', 0) / getattr(self.agent, 'api_budget', 100)
        
        # Calculate strategic plan coherence (simplified)
        coherence = self._calculate_strategic_plan_coherence()
        
        return AgentStatus(
            current_goal=current_goal,
            compute_budget_used=min(1.0, compute_budget_used),
            api_budget_used=min(1.0, api_budget_used),
            strategic_plan_coherence=coherence
        )
    
    def extract_event_log(self, limit: int = 50) -> List[EventLogEntry]:
        """Extract recent event log entries."""
        events = []
        
        # Extract from simulation event log
        if hasattr(self.simulation, 'event_log'):
            for event in self.simulation.event_log[-limit:]:
                event_type = self._classify_event_type(event)
                events.append(EventLogEntry(
                    timestamp=event.get('timestamp', datetime.now()),
                    event_type=event_type,
                    title=event.get('title', 'Unknown Event'),
                    description=event.get('description', ''),
                    details=event.get('details')
                ))
        
        return events
    
    def extract_executive_summary(self) -> ExecutiveSummary:
        """Extract complete executive summary data."""
        return ExecutiveSummary(
            kpis=self.extract_kpi_metrics(),
            performance_metrics=self.extract_performance_metrics(),
            agent_status=self.extract_agent_status(),
            event_log=self.extract_event_log()
        )
    
    def extract_financial_deep_dive(self) -> FinancialDeepDive:
        """Extract financial deep dive data."""
        return FinancialDeepDive(
            profit_loss=self._extract_profit_loss(),
            fee_breakdown=self._extract_fee_breakdown(),
            balance_sheet=self._extract_balance_sheet()
        )
    
    def extract_product_market_analysis(self) -> ProductMarketAnalysis:
        """Extract product and market analysis data."""
        # Get primary product (first product or default)
        primary_asin = list(self.simulation.products.keys())[0] if self.simulation.products else "B000TEST"
        product = self.simulation.products.get(primary_asin)
        
        if not product:
            # Create default product info if no products exist
            product_info = ProductInfo(
                asin=primary_asin,
                category="DEFAULT",
                cost=5.0,
                price=19.99,
                current_quantity=0,
                days_of_supply=0.0,
                inventory_turnover=0.0
            )
            bsr_components = BSRComponents()
            competitors = []
        else:
            product_info = self._extract_product_info(product)
            bsr_components = self._extract_bsr_components(product)
            competitors = self._extract_competitor_data(product)
        
        return ProductMarketAnalysis(
            product_info=product_info,
            bsr_components=bsr_components,
            competitors=competitors
        )
    
    def extract_supply_chain_operations(self) -> SupplyChainOperations:
        """Extract supply chain operations data."""
        suppliers = []
        active_orders = []
        
        # Extract supplier data if supply chain exists
        if hasattr(self.simulation, 'supply_chain'):
            suppliers = self._extract_supplier_info()
            active_orders = self._extract_active_orders()
        
        return SupplyChainOperations(
            suppliers=suppliers,
            active_orders=active_orders
        )
    
    def extract_agent_cognition(self) -> AgentCognition:
        """Extract agent cognition and strategy data."""
        if not self.agent:
            return AgentCognition(
                goal_stack=[],
                memory_entries=[],
                strategic_plan=StrategicPlan(
                    mission="No agent connected",
                    objectives=[],
                    coherence_score=0.0
                )
            )
        
        return AgentCognition(
            goal_stack=self._extract_goal_stack(),
            memory_entries=self._extract_memory_entries(),
            strategic_plan=self._extract_strategic_plan()
        )
    
    def extract_complete_dashboard_state(self) -> DashboardState:
        """Extract complete dashboard state for all tabs."""
        return DashboardState(
            executive_summary=self.extract_executive_summary(),
            financial_deep_dive=self.extract_financial_deep_dive(),
            product_market_analysis=self.extract_product_market_analysis(),
            supply_chain_operations=self.extract_supply_chain_operations(),
            agent_cognition=self.extract_agent_cognition()
        )
    
    # Private helper methods
    
    def _calculate_inventory_value(self) -> float:
        """Calculate total inventory value."""
        total_value = 0.0
        for asin, product in self.simulation.products.items():
            quantity = product.qty if hasattr(product, 'qty') else 0
            cost = getattr(product, 'cost', 0.0)
            total_value += quantity * cost
        return total_value
    
    def _calculate_daily_profit(self) -> float:
        """Calculate profit for the current day."""
        # Simplified calculation - would need more sophisticated tracking
        revenue = self.simulation.ledger.balance("Revenue")
        expenses = abs(self.simulation.ledger.balance("Expenses"))
        return revenue - expenses
    
    def _calculate_total_profit(self) -> float:
        """Calculate total cumulative profit."""
        return self.simulation.ledger.balance("Cash") - 10000  # Assuming 10k seed capital
    
    def _get_distress_protocol_status(self) -> DistressProtocolStatus:
        """Determine distress protocol status."""
        cash_balance = self.simulation.ledger.balance("Cash")
        
        if cash_balance < -1000:
            return DistressProtocolStatus.ACTIVE
        elif cash_balance < 1000:
            return DistressProtocolStatus.WARNING
        else:
            return DistressProtocolStatus.OK
    
    def _update_time_series_cache(self, day: int, timestamp: datetime):
        """Update time series cache with current values."""
        # Cash balance
        cash_balance = self.simulation.ledger.balance("Cash")
        self._time_series_cache['cash_balance'].append(
            TimeSeriesPoint(day=day, timestamp=timestamp, value=cash_balance)
        )
        
        # Revenue (daily)
        revenue = self._calculate_daily_revenue()
        self._time_series_cache['revenue'].append(
            TimeSeriesPoint(day=day, timestamp=timestamp, value=revenue)
        )
        
        # Profit (daily)
        profit = self._calculate_daily_profit()
        self._time_series_cache['profit'].append(
            TimeSeriesPoint(day=day, timestamp=timestamp, value=profit)
        )
        
        # Inventory value
        inventory_value = self._calculate_inventory_value()
        self._time_series_cache['inventory_value'].append(
            TimeSeriesPoint(day=day, timestamp=timestamp, value=inventory_value)
        )
        
        # Total sales (units)
        total_sales = self._calculate_daily_sales_units()
        self._time_series_cache['total_sales'].append(
            TimeSeriesPoint(day=day, timestamp=timestamp, value=total_sales)
        )
        
        # Best Seller Rank (primary product)
        bsr = self._get_primary_product_bsr()
        self._time_series_cache['best_seller_rank'].append(
            TimeSeriesPoint(day=day, timestamp=timestamp, value=bsr)
        )
        
        # Keep only last 100 points to prevent memory bloat
        for key in self._time_series_cache:
            if len(self._time_series_cache[key]) > 100:
                self._time_series_cache[key] = self._time_series_cache[key][-100:]
    
    def _calculate_daily_revenue(self) -> float:
        """Calculate daily revenue."""
        # Simplified - would need proper daily tracking
        return self.simulation.ledger.balance("Revenue") / max(1, getattr(self.simulation, 'day', 1))
    
    def _calculate_daily_sales_units(self) -> float:
        """Calculate daily sales in units."""
        total_units = 0.0
        for product in self.simulation.products.values():
            if hasattr(product, 'sales_velocity'):
                total_units += product.sales_velocity
        return total_units
    
    def _get_primary_product_bsr(self) -> float:
        """Get BSR of primary product."""
        if not self.simulation.products:
            return 100000.0
        
        primary_product = list(self.simulation.products.values())[0]
        return float(getattr(primary_product, 'bsr', 100000))
    
    def _calculate_strategic_plan_coherence(self) -> float:
        """Calculate strategic plan coherence score."""
        # Simplified coherence calculation
        if not self.agent or not hasattr(self.agent, 'strategic_plan'):
            return 0.0
        
        # Basic coherence based on goal alignment
        return 0.85  # Placeholder - would implement proper coherence analysis
    
    def _classify_event_type(self, event: Dict[str, Any]) -> EventType:
        """Classify event type for filtering."""
        event_desc = event.get('description', '').lower()
        
        if any(word in event_desc for word in ['supplier', 'bankruptcy', 'shock']):
            return EventType.ADVERSARIAL
        elif any(word in event_desc for word in ['agent', 'price', 'adjust']):
            return EventType.AGENT_ACTION
        elif any(word in event_desc for word in ['market', 'prime', 'demand']):
            return EventType.MARKET_EVENT
        else:
            return EventType.CUSTOMER_EVENT
    
    def _extract_profit_loss(self) -> ProfitLossStatement:
        """Extract P&L statement data."""
        # Simplified P&L extraction
        current_day = getattr(self.simulation, 'day', 0)
        
        return ProfitLossStatement(
            revenue={
                "current_day": self._calculate_daily_revenue(),
                "week_to_date": self._calculate_daily_revenue() * min(7, current_day),
                "month_to_date": self._calculate_daily_revenue() * min(30, current_day),
                "total": self.simulation.ledger.balance("Revenue")
            },
            cost_of_goods_sold={
                "current_day": 0.0,  # Would need proper COGS tracking
                "week_to_date": 0.0,
                "month_to_date": 0.0,
                "total": 0.0
            },
            gross_profit={
                "current_day": self._calculate_daily_revenue(),
                "week_to_date": self._calculate_daily_revenue() * min(7, current_day),
                "month_to_date": self._calculate_daily_revenue() * min(30, current_day),
                "total": self.simulation.ledger.balance("Revenue")
            },
            operating_expenses={
                "referral_fees": {"total": abs(self.simulation.ledger.balance("Referral Fees"))},
                "fba_fees": {"total": abs(self.simulation.ledger.balance("FBA Fees"))},
                "storage_fees": {"total": abs(self.simulation.ledger.balance("Storage Fees"))}
            },
            net_profit={
                "current_day": self._calculate_daily_profit(),
                "week_to_date": self._calculate_daily_profit() * min(7, current_day),
                "month_to_date": self._calculate_daily_profit() * min(30, current_day),
                "total": self._calculate_total_profit()
            }
        )
    
    def _extract_fee_breakdown(self) -> FeeBreakdown:
        """Extract fee breakdown data."""
        return FeeBreakdown(
            referral_fees=abs(self.simulation.ledger.balance("Referral Fees")),
            fba_fulfillment_fees=abs(self.simulation.ledger.balance("FBA Fees")),
            storage_fees=abs(self.simulation.ledger.balance("Storage Fees")),
            ancillary_fees=abs(self.simulation.ledger.balance("Ancillary Fees")),
            penalty_fees=abs(self.simulation.ledger.balance("Penalty Fees"))
        )
    
    def _extract_balance_sheet(self) -> BalanceSheet:
        """Extract balance sheet data."""
        cash_balance = self.simulation.ledger.balance("Cash")
        inventory_value = self._calculate_inventory_value()
        
        return BalanceSheet(
            assets={
                "cash": cash_balance,
                "inventory": inventory_value,
                "total": cash_balance + inventory_value
            },
            liabilities={
                "accounts_payable": 0.0,  # Would need proper liability tracking
                "total": 0.0
            },
            equity={
                "seed_capital": 10000.0,  # Assuming 10k seed capital
                "retained_earnings": self._calculate_total_profit(),
                "total": 10000.0 + self._calculate_total_profit()
            }
        )
    
    def _extract_product_info(self, product) -> ProductInfo:
        """Extract product information."""
        return ProductInfo(
            asin=product.asin,
            category=product.category,
            cost=product.cost,
            price=product.price,
            current_quantity=product.qty,
            days_of_supply=self._calculate_days_of_supply(product),
            inventory_turnover=self._calculate_inventory_turnover(product)
        )
    
    def _extract_bsr_components(self, product) -> BSRComponents:
        """Extract BSR components for charting."""
        current_day = getattr(self.simulation, 'day', 0)
        current_time = datetime.now()
        
        # Create time series points for BSR components
        return BSRComponents(
            ema_sales_velocity=[TimeSeriesPoint(
                day=current_day,
                timestamp=current_time,
                value=getattr(product, 'ema_sales_velocity', 0.0)
            )],
            ema_conversion=[TimeSeriesPoint(
                day=current_day,
                timestamp=current_time,
                value=getattr(product, 'ema_conversion', 0.0)
            )],
            rel_sales_index=[TimeSeriesPoint(
                day=current_day,
                timestamp=current_time,
                value=1.0  # Would need proper calculation
            )],
            rel_price_index=[TimeSeriesPoint(
                day=current_day,
                timestamp=current_time,
                value=1.0  # Would need proper calculation
            )]
        )
    
    def _extract_competitor_data(self, product) -> List[CompetitorData]:
        """Extract competitor analysis data."""
        competitors = []
        
        # Add agent's product first
        competitors.append(CompetitorData(
            asin=product.asin,
            price=product.price,
            sales_velocity=getattr(product, 'sales_velocity', 0.0),
            bsr=getattr(product, 'bsr', 100000),
            strategy="agent",
            is_agent=True
        ))
        
        # Add market competitors if available
        if hasattr(self.simulation, 'competitors'):
            for competitor in self.simulation.competitors[:10]:  # Limit to 10 competitors
                competitors.append(CompetitorData(
                    asin=competitor.asin,
                    price=competitor.price,
                    sales_velocity=competitor.sales_velocity,
                    bsr=getattr(competitor, 'bsr', 100000),
                    strategy=competitor.strategy,
                    is_agent=False
                ))
        
        return competitors
    
    def _extract_supplier_info(self) -> List[SupplierInfo]:
        """Extract supplier information."""
        suppliers = []
        
        if hasattr(self.simulation, 'supply_chain') and hasattr(self.simulation.supply_chain, 'suppliers'):
            for supplier in self.simulation.supply_chain.suppliers.values():
                suppliers.append(SupplierInfo(
                    supplier_id=supplier.supplier_id,
                    name=supplier.name,
                    supplier_type=SupplierType(supplier.supplier_type.value),
                    status=SupplierStatus(supplier.status.value),
                    reputation_score=supplier.reputation_score,
                    moq=supplier.moq_min,
                    lead_time=supplier.calculate_total_lead_time()
                ))
        
        return suppliers
    
    def _extract_active_orders(self) -> List[OrderInfo]:
        """Extract active order information."""
        orders = []
        
        if hasattr(self.simulation, 'supply_chain') and hasattr(self.simulation.supply_chain, 'active_orders'):
            for order in self.simulation.supply_chain.active_orders:
                orders.append(OrderInfo(
                    order_id=order.get('order_id', 'unknown'),
                    supplier_id=order.get('supplier_id', 'unknown'),
                    quantity=order.get('quantity', 0),
                    status=order.get('status', 'unknown'),
                    expected_delivery=order.get('expected_delivery', datetime.now())
                ))
        
        return orders
    
    def _extract_goal_stack(self) -> List[GoalStackItem]:
        """Extract agent goal stack."""
        goals = []
        
        if hasattr(self.agent, 'goal_stack'):
            for i, goal in enumerate(self.agent.goal_stack):
                goals.append(GoalStackItem(
                    goal_id=f"goal_{i}",
                    description=str(goal),
                    priority=i,
                    status="active" if i == 0 else "pending"
                ))
        
        return goals
    
    def _extract_memory_entries(self, limit: int = 20) -> List[MemoryEntry]:
        """Extract agent memory entries."""
        entries = []
        
        if hasattr(self.agent, 'long_term_memory'):
            memory = self.agent.long_term_memory
            
            # Extract episodic memories
            for episode in memory.get('episodic', [])[-limit//3:]:
                entries.append(MemoryEntry(
                    memory_type="episodic",
                    content=str(episode),
                    timestamp=datetime.now(),
                    relevance_score=0.8
                ))
            
            # Extract semantic memories
            for key, value in list(memory.get('semantic', {}).items())[-limit//3:]:
                entries.append(MemoryEntry(
                    memory_type="semantic",
                    content=f"{key}: {value}",
                    timestamp=datetime.now(),
                    relevance_score=0.9
                ))
            
            # Extract procedural memories
            for procedure in memory.get('procedural', [])[-limit//3:]:
                entries.append(MemoryEntry(
                    memory_type="procedural",
                    content=str(procedure),
                    timestamp=datetime.now(),
                    relevance_score=0.7
                ))
        
        return entries
    
    def _extract_strategic_plan(self) -> StrategicPlan:
        """Extract agent strategic plan."""
        if hasattr(self.agent, 'strategic_plan'):
            plan = self.agent.strategic_plan
            return StrategicPlan(
                mission=plan.get('mission', 'No mission defined'),
                objectives=plan.get('objectives', []),
                coherence_score=self._calculate_strategic_plan_coherence()
            )
        
        return StrategicPlan(
            mission="No strategic plan available",
            objectives=[],
            coherence_score=0.0
        )
    
    def _calculate_days_of_supply(self, product) -> float:
        """Calculate days of supply for a product."""
        sales_velocity = getattr(product, 'sales_velocity', 0.0)
        if sales_velocity <= 0:
            return float('inf')
        return product.qty / sales_velocity
    
    def _calculate_inventory_turnover(self, product) -> float:
        """Calculate inventory turnover rate."""
        # Simplified calculation - would need historical data
        sales_velocity = getattr(product, 'sales_velocity', 0.0)
        avg_inventory = max(1, product.qty)
        return (sales_velocity * 365) / avg_inventory