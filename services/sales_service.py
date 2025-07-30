"""SalesService for FBA-Bench v3 event-driven architecture."""

import asyncio
import logging
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from money import Money
from models.product import Product
from models.sales_result import SalesResult
from events import TickEvent, SaleOccurred, CompetitorPricesUpdated, CompetitorState, ProductPriceUpdated
from event_bus import EventBus, get_event_bus
from services.fee_calculation_service import FeeCalculationService
from services.world_store import WorldStore # Added this import

logger = logging.getLogger(__name__)


class DemandModel(Enum):
    """Types of demand models for sales calculation."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic"
    SEASONAL = "seasonal"


@dataclass
class MarketConditions:
    """Current market conditions affecting sales."""
    timestamp: datetime
    base_demand_multiplier: float = 1.0
    price_elasticity: float = -1.5
    competition_factor: float = 1.0
    seasonal_adjustment: float = 1.0
    external_shock_factor: float = 1.0
    trust_impact_factor: float = 0.3
    inventory_pressure_factor: float = 1.0
    
    def get_effective_demand_multiplier(self) -> float:
        """Calculate effective demand multiplier from all factors."""
        return (
            self.base_demand_multiplier *
            self.competition_factor *
            self.seasonal_adjustment *
            self.external_shock_factor *
            (1.0 + self.inventory_pressure_factor - 1.0)
        )


@dataclass
class SalesCalculationResult:
    """Result of sales calculation for a single product."""
    product_id: str
    base_demand: float
    adjusted_demand: float
    units_sold: int
    conversion_rate: float
    revenue: Money
    total_fees: Money
    profit: Money
    cost_basis: Money
    market_conditions: MarketConditions
    fee_breakdown: Dict[str, Money] = field(default_factory=dict)
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)


class SalesService:
    """
    Event-driven sales calculation service for FBA-Bench v3.
    
    Handles all sales-related calculations including demand modeling,
    unit sales, revenue, fees, and profit calculations. Integrates with
    the FeeCalculationService for comprehensive fee calculations.
    
    Publishes SaleOccurred events for each transaction.
    """
    
    def __init__(self, config: Dict, fee_service: FeeCalculationService, world_store: WorldStore):
        """
        Initialize the SalesService.
        
        Args:
            config: Service configuration
            fee_service: Fee calculation service instance
            world_store: WorldStore instance for product state
        """
        self.config = config
        self.fee_service = fee_service
        self.world_store = world_store # Store the world_store instance
        self.event_bus: Optional[EventBus] = None
        
        # Sales calculation parameters
        self.demand_model = DemandModel(config.get('demand_model', 'exponential'))
        self.base_conversion_rate = config.get('base_conversion_rate', 0.15)
        self.price_elasticity = config.get('price_elasticity', -1.5)
        self.trust_score_impact = config.get('trust_score_impact', 0.3)
        self.inventory_impact_threshold = config.get('inventory_impact_threshold', 10)
        
        # Random factors for market simulation
        self.demand_volatility = config.get('demand_volatility', 0.1)
        self.conversion_volatility = config.get('conversion_volatility', 0.05)
        
        # Performance tracking
        self.sales_history: List[SalesCalculationResult] = []
        self.max_history_size = config.get('max_history_size', 1000)
        
        # Market conditions
        self.current_market_conditions = MarketConditions(timestamp=datetime.now())
        
        # Competitor data (updated via events)
        self.current_competitor_states: List[CompetitorState] = []
        self.competitor_market_summary: Dict[str, Any] = {}
        
        # Product price data (updated via ProductPriceUpdated events from WorldStore)
        self.canonical_product_prices: Dict[str, Money] = {}
        self.product_price_history: Dict[str, List[Tuple[datetime, Money]]] = {}
        
        logger.info(f"SalesService initialized with {self.demand_model.value} demand model")

    async def start(self, event_bus: EventBus) -> None:
        """Start the sales service and subscribe to events."""
        self.event_bus = event_bus
        await self.event_bus.subscribe(TickEvent, self._handle_tick_event)
        await self.event_bus.subscribe(CompetitorPricesUpdated, self._handle_competitor_update)
        await self.event_bus.subscribe(ProductPriceUpdated, self._handle_product_price_update)
        logger.info("SalesService started and subscribed to TickEvent, CompetitorPricesUpdated, and ProductPriceUpdated")
    
    async def stop(self) -> None:
        """Stop the sales service."""
        logger.info("SalesService stopped")
    
    async def _handle_tick_event(self, event: TickEvent) -> None:
        """Handle tick events by processing sales for all active products."""
        try:
            # Update market conditions based on tick metadata
            self._update_market_conditions(event)
            
            # Get active products (this would normally come from a product registry)
            active_products = self._get_active_products(event)
            
            # Process sales for each product
            for product in active_products:
                if product.is_in_stock():
                    sales_result = await self._calculate_sales(product)
                    if sales_result.units_sold > 0:
                        await self._publish_sale_occurred(sales_result, event)
                        
        except Exception as e:
            logger.error(f"Error handling tick event {event.event_id}: {e}")
    
    async def _handle_competitor_update(self, event: CompetitorPricesUpdated) -> None:
        """Handle competitor price update events."""
        try:
            # Update current competitor state
            self.current_competitor_states = event.competitors
            self.competitor_market_summary = event.market_summary
            
            logger.debug(f"Updated competitor data: {len(self.current_competitor_states)} competitors")
            
        except Exception as e:
            logger.error(f"Error handling competitor update event {event.event_id}: {e}")
    
    async def _handle_product_price_update(self, event: ProductPriceUpdated) -> None:
        """
        Handle ProductPriceUpdated events from WorldStore.
        
        Maintains canonical product price data for sales calculations.
        This ensures SalesService uses the authoritative price data
        managed by WorldStore instead of potentially stale product state.
        """
        try:
            asin = event.asin
            new_price = event.new_price
            
            # Update canonical price
            self.canonical_product_prices[asin] = new_price
            
            # Track price history for trend analysis
            if asin not in self.product_price_history:
                self.product_price_history[asin] = []
            
            self.product_price_history[asin].append((event.timestamp, new_price))
            
            # Keep only last 50 price changes per product
            if len(self.product_price_history[asin]) > 50:
                self.product_price_history[asin] = self.product_price_history[asin][-50:]
            
            logger.debug(f"Updated canonical price for {asin}: {event.previous_price} -> {new_price} (agent: {event.agent_id})")
            
        except Exception as e:
            logger.error(f"Error handling product price update event {event.event_id}: {e}")
    
    def _update_market_conditions(self, tick_event: TickEvent) -> None:
        """Update market conditions based on tick event metadata."""
        metadata = tick_event.metadata
        
        # Update market conditions from tick metadata
        self.current_market_conditions.timestamp = tick_event.simulation_time
        
        if 'demand_multiplier' in metadata:
            self.current_market_conditions.base_demand_multiplier = metadata['demand_multiplier']
        
        if 'seasonal_factor' in metadata:
            self.current_market_conditions.seasonal_adjustment = metadata['seasonal_factor']
        
        if 'external_shock' in metadata:
            self.current_market_conditions.external_shock_factor = metadata['external_shock']
        
        # Add some random market volatility
        volatility = random.uniform(-self.demand_volatility, self.demand_volatility)
        self.current_market_conditions.base_demand_multiplier *= (1.0 + volatility)
    
    def _get_active_products(self, tick_event: TickEvent) -> List[Product]:
        """Get list of active products for this tick."""
        # This is a placeholder - in a real implementation, this would fetch
        # from a product registry or database
        # For now, we'll return an empty list since we don't have product data
        return []
    
    async def _calculate_sales(self, product: Product) -> SalesCalculationResult:
        """Calculate sales for a single product."""
        try:
            # Calculate base demand
            base_demand = self._calculate_base_demand(product)
            
            # Apply market conditions to get adjusted demand
            adjusted_demand = self._apply_market_conditions(product, base_demand)
            
            # Calculate conversion rate
            conversion_rate = self._calculate_conversion_rate(product)
            
            # Calculate units sold
            units_sold = self._calculate_units_sold(adjusted_demand, conversion_rate, product)
            
            # Calculate financial results
            revenue, fees, profit, cost_basis, fee_breakdown = await self._calculate_financial_results(
                product, units_sold
            )
            
            # Create sales result
            sales_result = SalesCalculationResult(
                product_id=product.asin,
                base_demand=base_demand,
                adjusted_demand=adjusted_demand,
                units_sold=units_sold,
                conversion_rate=conversion_rate,
                revenue=revenue,
                total_fees=fees,
                profit=profit,
                cost_basis=cost_basis,
                market_conditions=self.current_market_conditions,
                fee_breakdown=fee_breakdown,
                calculation_metadata={
                    'demand_model': self.demand_model.value,
                    'timestamp': datetime.now().isoformat(),
                    'trust_score': product.trust_score,
                    'current_price': str(product.price),
                    'inventory_available': product.get_available_units()
                }
            )
            
            # Add to history
            self._add_to_history(sales_result)
            
            return sales_result
            
        except Exception as e:
            logger.error(f"Error calculating sales for product {product.asin}: {e}")
            # Return zero sales result
            return SalesCalculationResult(
                product_id=product.asin,
                base_demand=0.0,
                adjusted_demand=0.0,
                units_sold=0,
                conversion_rate=0.0,
                revenue=Money.zero(),
                total_fees=Money.zero(),
                profit=Money.zero(),
                cost_basis=Money.zero(),
                market_conditions=self.current_market_conditions
            )
    
    def _calculate_base_demand(self, product: Product) -> float:
        """Calculate base demand for a product."""
        base_demand = product.base_demand
        
        if self.demand_model == DemandModel.LINEAR:
            # Simple linear demand based on BSR
            bsr_factor = max(0.1, 1.0 - (product.bsr / 1_000_000))
            return base_demand * bsr_factor
            
        elif self.demand_model == DemandModel.EXPONENTIAL:
            # Exponential demand model based on BSR
            bsr_factor = 1.0 / (1.0 + (product.bsr / 100_000) ** 0.5)
            return base_demand * bsr_factor
            
        elif self.demand_model == DemandModel.LOGISTIC:
            # Logistic demand model
            bsr_normalized = product.bsr / 1_000_000
            bsr_factor = 1.0 / (1.0 + (bsr_normalized / 0.1) ** 2)
            return base_demand * bsr_factor
            
        else:  # SEASONAL
            # Seasonal demand with time-based variations
            month = datetime.now().month
            seasonal_factors = {
                1: 0.8, 2: 0.85, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.05,
                7: 1.1, 8: 1.05, 9: 1.0, 10: 1.1, 11: 1.3, 12: 1.4
            }
            seasonal_factor = seasonal_factors.get(month, 1.0)
            bsr_factor = 1.0 / (1.0 + (product.bsr / 50_000) ** 0.3)
            return base_demand * bsr_factor * seasonal_factor
    
    def _apply_market_conditions(self, product: Product, base_demand: float) -> float:
        """Apply market conditions to base demand."""
        # Start with base demand
        adjusted_demand = base_demand
        
        # Apply market conditions multiplier
        adjusted_demand *= self.current_market_conditions.get_effective_demand_multiplier()
        
        # Apply price elasticity using real competitor data
        competitor_avg_price = self._get_competitor_average_price()
        if competitor_avg_price and competitor_avg_price.cents != 0:
            price_ratio = float(product.price / competitor_avg_price)
            # Apply price elasticity
            price_adjustment = price_ratio ** self.price_elasticity
            adjusted_demand *= price_adjustment
            
            # Apply competition factor based on our position
            competition_factor = self._calculate_competition_factor(product.price)
            adjusted_demand *= competition_factor
        else:
            # No competitor data available, use base pricing logic
            logger.debug("No competitor data available for price elasticity calculation")
        
        # Apply trust score impact
        trust_adjustment = 1.0 + (product.trust_score - 0.7) * self.trust_score_impact
        adjusted_demand *= trust_adjustment
        
        # Apply inventory pressure
        available_units = product.get_available_units()
        if available_units <= self.inventory_impact_threshold:
            # Reduce demand if inventory is low (stockout risk)
            inventory_factor = max(0.1, available_units / self.inventory_impact_threshold)
            adjusted_demand *= inventory_factor
        
        return max(0.0, adjusted_demand)
    
    def _calculate_conversion_rate(self, product: Product) -> float:
        """Calculate conversion rate for a product."""
        # Base conversion rate
        conversion_rate = self.base_conversion_rate
        
        # Trust score impact on conversion
        trust_adjustment = 1.0 + (product.trust_score - 0.7) * 0.5
        conversion_rate *= trust_adjustment
        
        # Use EMA conversion if available
        if product.ema_conversion > 0:
            # Blend historical and calculated conversion
            conversion_rate = 0.7 * product.ema_conversion + 0.3 * conversion_rate
        
        # Add random volatility
        volatility = random.uniform(-self.conversion_volatility, self.conversion_volatility)
        conversion_rate *= (1.0 + volatility)
        
        # Ensure bounds
        return max(0.01, min(1.0, conversion_rate))
    
    def _calculate_units_sold(self, demand: float, conversion_rate: float, product: Product) -> int:
        """Calculate actual units sold based on demand and conversion."""
        # Expected sales
        expected_sales = demand * conversion_rate
        
        # Add Poisson noise for realistic variation
        units_sold = max(0, np.random.poisson(expected_sales)) if expected_sales > 0 else 0
        
        # Constrain by available inventory
        available_units = product.get_available_units()
        units_sold = min(units_sold, available_units)
        return int(units_sold)

    def get_current_inventory_value(self) -> float:
        """Calculate the total current inventory value from WorldStore."""
        total_value = Money.zero()
        # Iterate through all products managed by WorldStore and sum their inventory value
        for asin, product_state in self.world_store.get_all_product_states().items():
            if product_state.inventory_quantity > 0 and product_state.cost_basis is not None:
                total_value += product_state.cost_basis * product_state.inventory_quantity
        return total_value.dollars

    async def _calculate_financial_results(
        self,
        product: Product,
        units_sold: int
    ) -> Tuple[Money, Money, Money, Money, Dict[str, Money]]:
        """Calculate financial results for a sale."""
        if units_sold == 0:
            return Money.zero(), Money.zero(), Money.zero(), Money.zero(), {}
        
        # Calculate revenue
        revenue = product.price * units_sold
        
        # Calculate cost basis
        cost_basis = product.cost * units_sold
        
        # Calculate fees using fee calculation service
        fee_breakdown_obj = self.fee_service.calculate_comprehensive_fees(
            product, 
            product.price,
            {
                'units_sold': units_sold,
                'storage_duration_days': 30,  # Assume 30 days storage
                'requires_prep': False,
                'requires_labeling': False
            }
        )
        
        # Extract fee information
        total_fees = fee_breakdown_obj.total_fees * units_sold
        fee_breakdown = {}
        from decimal import Decimal # Needed for Decimal conversion
        for fee in fee_breakdown_obj.individual_fees:
            logger.debug(f"DEBUG fee.calculated_amount={fee.calculated_amount} type={type(fee.calculated_amount)}, units_sold={units_sold} type={type(units_sold)} for fee {fee.fee_type.value}")
            if isinstance(fee.calculated_amount, Money):
                fee_breakdown[fee.fee_type.value] = fee.calculated_amount * int(units_sold)
            else:
                sale_amount = Decimal(str(fee.calculated_amount)) * Decimal(int(units_sold))
                fee_breakdown[fee.fee_type.value] = Money(sale_amount) # Ensure Decimal for multiplication
            logger.debug(f"DEBUG fee.calculated_amount type: {type(fee.calculated_amount)} for fee {fee.fee_type.value}")
        
        # Total fees from fee_breakdown_obj.total_fees is already for all units, so no need to multiply by units_sold again here.
        # This was potentially a bug in the previous iteration.
        total_fees = fee_breakdown_obj.total_fees # Use the aggregated total fees
        
        # Calculate profit
        profit = revenue - total_fees - cost_basis
        
        return revenue, total_fees, profit, cost_basis, fee_breakdown
    
    async def _publish_sale_occurred(self, sales_result: SalesCalculationResult, tick_event: TickEvent) -> None:
        """Publish a SaleOccurred event for the sales result."""
        try:
            # Generate unique event ID
            event_id = f"sale_{sales_result.product_id}_{tick_event.tick_number}_{int(datetime.now().timestamp())}"
            
            # Create SaleOccurred event
            sale_event = SaleOccurred(
                event_id=event_id,
                timestamp=datetime.now(),
                asin=sales_result.product_id,
                units_sold=sales_result.units_sold,
                units_demanded=int(sales_result.adjusted_demand),
                unit_price=sales_result.revenue / sales_result.units_sold if sales_result.units_sold > 0 else Money.zero(),
                total_revenue=sales_result.revenue,
                total_fees=sales_result.total_fees,
                total_profit=sales_result.profit,
                cost_basis=sales_result.cost_basis,
                trust_score_at_sale=sales_result.calculation_metadata.get('trust_score', 0.7),
                bsr_at_sale=1000000,  # Would get from product in real implementation
                conversion_rate=sales_result.conversion_rate,
                fee_breakdown=sales_result.fee_breakdown,
                market_conditions={
                    'demand_multiplier': sales_result.market_conditions.base_demand_multiplier,
                    'price_elasticity': self.price_elasticity,
                    'seasonal_adjustment': sales_result.market_conditions.seasonal_adjustment,
                    'external_shock_factor': sales_result.market_conditions.external_shock_factor
                },
                customer_segment=None  # Would be determined by customer analysis
            )
            
            # Publish event
            if self.event_bus:
                await self.event_bus.publish(sale_event)
                logger.debug(f"Published SaleOccurred event: {event_id}")
            else:
                logger.warning("EventBus not available, cannot publish SaleOccurred event")
                
        except Exception as e:
            logger.error(f"Error publishing SaleOccurred event: {e}")
    
    def _add_to_history(self, sales_result: SalesCalculationResult) -> None:
        """Add sales result to history with size limit."""
        self.sales_history.append(sales_result)
        
        # Trim history if it exceeds max size
        if len(self.sales_history) > self.max_history_size:
            self.sales_history = self.sales_history[-self.max_history_size:]
    
    def calculate_sales_for_product(self, product: Product) -> SalesCalculationResult:
        """Synchronous method to calculate sales for a specific product."""
        # This is a sync wrapper for direct sales calculation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._calculate_sales(product))
        finally:
            loop.close()
    
    def get_sales_statistics(self) -> Dict[str, Any]:
        """Get sales service statistics."""
        if not self.sales_history:
            return {
                'total_transactions': 0,
                'total_revenue': Money.zero(),
                'total_profit': Money.zero(),
                'average_units_per_sale': 0.0,
                'average_conversion_rate': 0.0
            }
        
        total_transactions = len(self.sales_history)
        total_revenue = sum((result.revenue for result in self.sales_history), Money.zero())
        total_profit = sum((result.profit for result in self.sales_history), Money.zero())
        total_units = sum(result.units_sold for result in self.sales_history)
        avg_conversion = sum(result.conversion_rate for result in self.sales_history) / total_transactions
        
        return {
            'total_transactions': total_transactions,
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'total_units_sold': total_units,
            'average_units_per_sale': total_units / total_transactions if total_transactions > 0 else 0.0,
            'average_conversion_rate': avg_conversion,
            'demand_model': self.demand_model.value,
            'current_market_conditions': {
                'base_demand_multiplier': self.current_market_conditions.base_demand_multiplier,
                'seasonal_adjustment': self.current_market_conditions.seasonal_adjustment,
                'external_shock_factor': self.current_market_conditions.external_shock_factor
            }
        }
    
    def _get_competitor_average_price(self) -> Optional[Money]:
        """Calculate average competitor price from current competitor data."""
        if not self.current_competitor_states:
            return None
            
        total_cents = sum(comp.price.cents for comp in self.current_competitor_states)
        return Money(total_cents // len(self.current_competitor_states))
    
    def _calculate_competition_factor(self, our_price: Money) -> float:
        """Calculate competition factor based on our position relative to competitors."""
        if not self.current_competitor_states:
            return 1.0  # No competition data, neutral factor
            
        competitor_prices = [comp.price for comp in self.current_competitor_states]
        if not competitor_prices:
            return 1.0
            
        # Sort prices to determine our position
        all_prices = competitor_prices + [our_price]
        all_prices.sort()
        
        our_rank = all_prices.index(our_price) + 1
        total_sellers = len(all_prices)
        
        # Calculate percentile position (higher = more expensive)
        percentile = (total_sellers - our_rank + 1) / total_sellers
        
        # Competition factor based on price position
        if percentile >= 0.8:  # Premium pricing (top 20%)
            return 0.7  # Lower demand due to high price
        elif percentile >= 0.6:  # Above average pricing
            return 0.85
        elif percentile >= 0.4:  # Average pricing
            return 1.0  # Neutral
        elif percentile >= 0.2:  # Below average pricing
            return 1.15  # Increased demand due to competitive pricing
        else:  # Discount pricing (bottom 20%)
            return 1.3  # Higher demand due to low price
    
    def get_competitor_market_summary(self) -> Dict[str, Any]:
        """Get current competitor market summary."""
        if not self.current_competitor_states:
            return {
                'competitor_count': 0,
                'average_price': None,
                'price_range': None,
                'market_updated': False
            }
            
        prices = [comp.price for comp in self.current_competitor_states]
        return {
            'competitor_count': len(self.current_competitor_states),
            'average_price': str(self._get_competitor_average_price()),
            'price_range': [str(min(prices)), str(max(prices))],
            'average_bsr': sum(comp.bsr for comp in self.current_competitor_states) / len(self.current_competitor_states),
            'average_sales_velocity': sum(comp.sales_velocity for comp in self.current_competitor_states) / len(self.current_competitor_states),
            'market_updated': True,
            'raw_summary': self.competitor_market_summary
        }