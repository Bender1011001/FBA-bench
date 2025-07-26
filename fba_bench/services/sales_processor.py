"""Sales processing service for FBA-Bench simulation."""
from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, List, Protocol
from decimal import Decimal

from fba_bench.money import Money
from fba_bench.config_loader import FBABenchConfig
from fba_bench.ledger import Ledger, Transaction, Entry
from fba_bench.inventory import InventoryManager
from fba_bench.fee_engine import FeeEngine
from fba_bench.models.sales_result import SalesResult
from fba_bench.models import Competitor

# Load configuration constants
from fba_bench.config_loader import load_config
_config = load_config()
CUBIC_FEET_PER_UNIT = _config.simulation.cubic_feet_per_unit
MONTHS_STORAGE_DEFAULT = _config.simulation.months_storage_default
REMOVAL_UNITS_DEFAULT = _config.simulation.removal_units_default
RETURN_FEES_DEFAULT = _config.simulation.return_fees_default
AGED_DAYS_DEFAULT = _config.simulation.aged_days_default
AGED_CUBIC_FEET_PER_UNIT = _config.simulation.aged_cubic_feet_per_unit
LOW_INVENTORY_UNITS_DEFAULT = _config.simulation.low_inventory_units_default
TRAILING_DAYS_SUPPLY_DEFAULT = _config.simulation.trailing_days_supply_default
WEEKS_SUPPLY_DEFAULT = _config.simulation.weeks_supply_default
EMA_DECAY = _config.simulation.ema_decay
BSR_BASE = _config.market_dynamics.bsr_base
BSR_SMOOTHING_FACTOR = _config.market_dynamics.bsr_smoothing_factor
BSR_MIN_VALUE = _config.market_dynamics.bsr_min_value
BSR_MAX_VALUE = _config.market_dynamics.bsr_max_value
INDIVIDUAL_PER_ITEM = _config.fees.individual_per_item


class MarketDynamics(Protocol):
    """Protocol for market dynamics calculations."""
    
    def calculate_demand(self, base_demand: float, price: Money, elasticity: float,
                        seasonality_multiplier: float, competitors: List[Competitor],
                        trust_score: float) -> int:
        """Calculate demand for a product."""
        ...
    
    def calculate_trust_score(self, cancellations: int, policy_violations: int,
                             review_manipulation: int, customer_issues: int,
                             base_score: float) -> float:
        """Calculate seller trust score."""
        ...
    
    def calculate_dynamic_elasticity(self, bsr: int) -> float:
        """Calculate dynamic price elasticity based on BSR."""
        ...
    
    def get_seasonality_multiplier(self, date: datetime, category: str) -> float:
        """Get seasonality multiplier for given date and category."""
        ...


class SalesProcessor:
    """Processes sales for products in the FBA-Bench simulation."""
    
    def __init__(self, config: FBABenchConfig, ledger: Ledger, 
                 inventory: InventoryManager, market_dynamics: MarketDynamics, 
                 rng: random.Random):
        """Initialize the SalesProcessor.
        
        Args:
            config: FBA-Bench configuration
            ledger: Ledger for posting transactions
            inventory: Inventory manager
            market_dynamics: Market dynamics calculator
            rng: Random number generator for deterministic behavior
        """
        self.config = config
        self.ledger = ledger
        self.inventory = inventory
        self.market_dynamics = market_dynamics
        self.rng = rng
    
    def process_product_sales(self, asin: str, product, competitors: List[Competitor],
                             customer_events: Dict, current_date: datetime,
                             fee_engine: FeeEngine, selling_plan: str = "Professional",
                             event_log: List = None) -> SalesResult:
        """Process sales for a single product and post ledger entries.
        
        Args:
            asin: Product ASIN
            product: Product object
            competitors: List of competitor products
            customer_events: Customer events by ASIN
            current_date: Current simulation date
            fee_engine: Fee calculation engine
            selling_plan: Selling plan type ("Professional" or "Individual")
            event_log: Event log for recording suppression events
            
        Returns:
            SalesResult with sales processing details
        """
        # 1. Calculate trust score
        trust_score = self._calculate_trust_score(asin, customer_events)
        product.trust_score = trust_score
        
        # 2. Apply trust score effects on fees and listing suppression
        trust_fee_multiplier = self._calculate_trust_fee_multiplier(trust_score)
        suppression_info = self._check_listing_suppression(trust_score)
        
        # 3. Calculate demand with market dynamics
        current_elasticity = self.market_dynamics.calculate_dynamic_elasticity(product.bsr)
        seasonality_multiplier = self.market_dynamics.get_seasonality_multiplier(current_date, product.category)
        demand = self.market_dynamics.calculate_demand(
            base_demand=product.base_demand,
            price=product.price,
            elasticity=current_elasticity,
            seasonality_multiplier=seasonality_multiplier,
            competitors=competitors,
            trust_score=trust_score
        )
        
        # 4. Apply graduated listing suppression based on trust score
        if suppression_info["suppressed"]:
            # Apply demand reduction based on suppression level
            demand = int(demand * suppression_info["demand_multiplier"])
            
            # Apply search ranking penalty (affects BSR negatively)
            search_penalty = suppression_info["search_penalty"]
            if search_penalty > 0:
                # Worsen BSR by the penalty factor (higher BSR = worse ranking)
                product.bsr = min(BSR_MAX_VALUE, int(product.bsr * (1 + search_penalty)))
            
            # Log suppression with severity level
            if event_log is not None:
                level = suppression_info["level"]
                multiplier = suppression_info["demand_multiplier"]
                event_log.append(
                    f"Day {current_date.day}: Listing {asin} suppressed ({level} level) - "
                    f"trust score {trust_score:.2f}, demand reduced to {multiplier*100:.0f}%"
                )
        
        # 5. Process inventory and sales
        units_sold = self.inventory.remove(asin, demand)
        
        # 6. Update BSR tracking and history
        self._update_product_bsr(product, units_sold, demand, competitors)
        
        # 7. Calculate fees and revenue
        revenue, total_fees = self._calculate_revenue_and_fees(
            product, units_sold, fee_engine, current_date, trust_fee_multiplier, selling_plan
        )
        
        # 8. Post ledger entries (embedded in service)
        if units_sold > 0:
            self._post_sales_transaction(asin, revenue, total_fees, product.cost * units_sold, current_date)
        
        # 9. Generate customer events
        self._generate_customer_events(asin, units_sold, customer_events, current_date, product, competitors)
        
        return SalesResult(
            units_sold=units_sold,
            revenue=revenue,
            total_fees=total_fees,
            demand=demand,
            trust_score=trust_score,
            bsr_change=0  # Could track BSR changes if needed
        )
    
    def _calculate_trust_score(self, asin: str, customer_events: Dict) -> float:
        """Calculate seller trust score from customer events."""
        trust_score = 1.0
        if asin in customer_events:
            events = customer_events[asin]
            cancellations = sum(1 for e in events if e.get("type") == "a_to_z_claim")
            policy_violations = getattr(self, 'policy_violations', 0)  # Track policy violations
            review_manipulation = getattr(self, 'review_manipulation_count', 0)  # Track review manipulation
            customer_issues = sum(1 for e in events if e.get("type") in ["negative_review", "negative_feedback"])
            trust_score = self.market_dynamics.calculate_trust_score(
                cancellations=cancellations,
                policy_violations=policy_violations,
                review_manipulation=review_manipulation,
                customer_issues=customer_issues,
                base_score=1.0
            )
        return trust_score
    
    def _calculate_trust_fee_multiplier(self, trust_score: float) -> float:
        """Calculate fee multiplier based on seller trust score."""
        if trust_score >= 0.9:
            return 1.0  # No penalty for high trust
        elif trust_score >= 0.7:
            return 1.1  # 10% penalty for medium trust
        elif trust_score >= 0.5:
            return 1.25  # 25% penalty for low trust
        else:
            return 1.5  # 50% penalty for very low trust
    
    def _check_listing_suppression(self, trust_score: float) -> dict:
        """Check if listing should be suppressed based on trust score."""
        if trust_score >= 0.7:
            return {"suppressed": False, "level": "none", "demand_multiplier": 1.0, "search_penalty": 0.0}
        elif trust_score >= 0.5:
            return {"suppressed": True, "level": "warning", "demand_multiplier": 0.8, "search_penalty": 0.1}
        elif trust_score >= 0.3:
            return {"suppressed": True, "level": "moderate", "demand_multiplier": 0.5, "search_penalty": 0.3}
        elif trust_score >= 0.1:
            return {"suppressed": True, "level": "severe", "demand_multiplier": 0.2, "search_penalty": 0.6}
        else:
            return {"suppressed": True, "level": "critical", "demand_multiplier": 0.05, "search_penalty": 0.9}
    
    def _update_product_bsr(self, product, units_sold: int, demand: int, competitors: List[Competitor]):
        """Update product BSR tracking and history."""
        # Track sales and demand history
        product.sales_history.append(units_sold)
        product.demand_history.append(demand)
        conversion = units_sold / demand if demand > 0 else 0.0
        product.conversion_history.append(conversion)
        
        # Update EMA for sales velocity and conversion
        if len(product.sales_history) == 1:
            product.ema_sales_velocity = units_sold
            product.ema_conversion = conversion
        else:
            product.ema_sales_velocity = (1 - EMA_DECAY) * product.ema_sales_velocity + EMA_DECAY * units_sold
            product.ema_conversion = (1 - EMA_DECAY) * product.ema_conversion + EMA_DECAY * conversion
        
        # Update BSR based on blueprint formula
        # BSR = base / (ema_sales_velocity * ema_conversion * rel_sales_index * rel_price_index)
        # Compute competitor averages for relative indices
        avg_comp_sales = BSR_SMOOTHING_FACTOR  # Use config smoothing factor instead of hardcoded 1.0
        avg_comp_price = product.price
        
        if competitors:
            avg_comp_sales = max(BSR_SMOOTHING_FACTOR, sum(c.sales_velocity for c in competitors) / len(competitors))
            total_comp_price = Money.zero()
            for c in competitors:
                total_comp_price += c.price
            avg_comp_price = total_comp_price / len(competitors)
        
        # Relative sales index: agent's sales velocity vs. competitors
        rel_sales_index = max(BSR_SMOOTHING_FACTOR, product.ema_sales_velocity) / avg_comp_sales
        
        # Relative price index: agent's price vs. competitors (lower price = better index)
        rel_price_index = avg_comp_price / max(Money.from_dollars(BSR_SMOOTHING_FACTOR), product.price)
        
        # Apply blueprint BSR formula with bounds checking
        if product.ema_sales_velocity > BSR_SMOOTHING_FACTOR and product.ema_conversion > BSR_SMOOTHING_FACTOR:
            denominator = (
                product.ema_sales_velocity *
                product.ema_conversion *
                rel_sales_index *
                float(rel_price_index)
            )
            calculated_bsr = BSR_BASE / max(BSR_SMOOTHING_FACTOR, denominator)
            product.bsr = max(BSR_MIN_VALUE, min(BSR_MAX_VALUE, int(calculated_bsr)))
        else:
            product.bsr = BSR_BASE
    
    def _calculate_revenue_and_fees(self, product, units_sold: int, fee_engine: FeeEngine,
                                   current_date: datetime, trust_fee_multiplier: float,
                                   selling_plan: str) -> tuple[Money, Money]:
        """Calculate revenue and total fees for the sale."""
        if units_sold == 0:
            return Money.zero(), Money.zero()
        
        revenue = units_sold * product.price
        
        # Calculate dynamic size tier and dim weight
        dims = product.dimensions
        weight = product.weight
        cubic_inches = dims["L"] * dims["W"] * dims["H"]
        dim_weight = cubic_inches / 139
        dim_weight_applies = dim_weight > weight
        size_tier = "standard"
        if max(dims["L"], dims["W"], dims["H"]) > 18 or weight > 20:
            size_tier = "oversize"
        
        # Calculate fee parameters
        if hasattr(product, "dimensions") and product.dimensions:
            dims = product.dimensions
            cubic_feet_per_unit = (dims.get("L", 10.0) * dims.get("W", 6.0) * dims.get("H", 2.0)) / 1728.0
        else:
            cubic_feet_per_unit = CUBIC_FEET_PER_UNIT
        cubic_feet = cubic_feet_per_unit * units_sold

        months_storage = MONTHS_STORAGE_DEFAULT
        removal_units = REMOVAL_UNITS_DEFAULT
        return_applicable_fees = RETURN_FEES_DEFAULT
        aged_days = AGED_DAYS_DEFAULT
        if hasattr(product, "dimensions") and product.dimensions:
            aged_cubic_feet_per_unit = cubic_feet_per_unit * 0.4  # Example: 40% aged
        else:
            aged_cubic_feet_per_unit = AGED_CUBIC_FEET_PER_UNIT
        aged_cubic_feet = aged_cubic_feet_per_unit * units_sold

        low_inventory_units = LOW_INVENTORY_UNITS_DEFAULT
        trailing_days_supply = TRAILING_DAYS_SUPPLY_DEFAULT
        weeks_supply = WEEKS_SUPPLY_DEFAULT
        unplanned_units = 0  # No unplanned service in this tick

        # Calculate ancillary and penalty fees based on current state
        ancillary_fee = Money.zero()  # Placeholder - would need access to simulation state
        penalty_fee = Money.zero()    # Placeholder - would need access to simulation state
        
        # Add selling plan per-item fee for Individual plan
        selling_plan_fee = Money.zero()
        if selling_plan == "Individual":
            selling_plan_fee = Money.from_dollars(INDIVIDUAL_PER_ITEM) * units_sold

        # BUGFIX: Convert total ancillary/penalty fees to per-unit amounts before passing to fee engine
        # The fee engine expects per-unit fees, but ancillary_fee and penalty_fee are calculated as totals
        ancillary_fee_per_unit = ancillary_fee.to_float() / max(1, units_sold) if ancillary_fee > Money.zero() else 0.0
        penalty_fee_per_unit = penalty_fee.to_float() / max(1, units_sold) if penalty_fee > Money.zero() else 0.0
        
        # All fee calculations are now handled by FeeEngine.total_fees
        fees = fee_engine.total_fees(
            category=product.category,
            price=product.price,
            size_tier=size_tier,
            size="small" if size_tier == "standard" else "large",
            is_holiday_season=(current_date.month in [11, 12]),
            dim_weight_applies=dim_weight_applies,
            cubic_feet=cubic_feet,
            months_storage=months_storage,
            removal_units=removal_units,
            return_applicable_fees=return_applicable_fees,
            aged_days=aged_days,
            aged_cubic_feet=aged_cubic_feet,
            low_inventory_units=low_inventory_units,
            trailing_days_supply=trailing_days_supply,
            weeks_supply=weeks_supply,
            unplanned_units=unplanned_units,
            penalty_fee=penalty_fee_per_unit,
            ancillary_fee=ancillary_fee_per_unit
        )
        # BUGFIX: fee_engine.total_fees() returns per-unit fees, so multiply by units_sold only once
        total_fees = (Money.from_dollars(fees["total"]) * units_sold * Decimal(str(trust_fee_multiplier))) + selling_plan_fee
        
        return revenue, total_fees
    
    def _post_sales_transaction(self, asin: str, revenue: Money, total_fees: Money,
                               cogs: Money, current_date: datetime):
        """Post sales transaction to the ledger."""
        # All amounts are positive Money objects; accounting effect determined by debit/credit side
        ts = current_date
        debits = [
            Entry("COGS", cogs, ts),        # Cost of goods sold expense
            Entry("Fees", total_fees, ts),  # Fees expense
        ]
        credits = [
            Entry("Revenue", revenue, ts),   # Gross revenue earned
            Entry("Inventory", cogs, ts),    # Inventory asset reduced
        ]
        
        # Handle cash flow based on whether net is positive or negative
        net_cash = revenue - total_fees
        if net_cash >= Money.zero():
            # Positive cash flow: debit Cash (asset increase)
            debits.insert(0, Entry("Cash", net_cash, ts))
        else:
            # Negative cash flow: credit Cash (asset decrease) with positive amount
            credits.insert(0, Entry("Cash", -net_cash, ts))
        
        # Post balanced transaction with all positive amounts
        self.ledger.post(Transaction(f"Sales and fees for {asin}",
            debits=debits,
            credits=credits
        ))
    
    def _generate_customer_events(self, asin: str, units_sold: int, customer_events: Dict,
                                 current_date: datetime, product, competitors: List[Competitor]):
        """Generate customer events after sale."""
        if asin not in customer_events:
            customer_events[asin] = []
        
        for _ in range(units_sold):
            event_type = self._generate_customer_event(asin, product, competitors)
            if event_type:
                customer_events[asin].append({
                    "type": event_type,
                    "date": current_date,
                })
    
    def _generate_customer_event(self, asin: str, product, competitors: List[Competitor]):
        """Generate a single customer event based on product and market conditions."""
        if not product:
            return None
        
        # Calculate average competitor price for price sensitivity
        avg_comp_price = product.price
        if competitors:
            total_comp_price = Money.zero()
            for c in competitors:
                total_comp_price += c.price
            avg_comp_price = total_comp_price / len(competitors)
        
        trust_score = getattr(product, 'trust_score', 1.0)
        
        # Customer segmentation (affects behavior patterns)
        customer_segments = {
            "price_sensitive": {"weight": Decimal('0.4'), "price_factor": Decimal('2.0'), "review_rate": Decimal('0.8')},
            "quality_focused": {"weight": Decimal('0.3'), "price_factor": Decimal('0.5'), "review_rate": Decimal('1.5')},
            "convenience_focused": {"weight": Decimal('0.2'), "price_factor": Decimal('1.0'), "review_rate": Decimal('0.6')},
            "brand_loyal": {"weight": Decimal('0.1'), "price_factor": Decimal('0.3'), "review_rate": Decimal('1.2')}
        }
        
        # Select customer segment
        segment_roll = self.rng.random()
        cumulative = 0
        selected_segment = "price_sensitive"
        for segment, props in customer_segments.items():
            cumulative += props["weight"]
            if segment_roll <= cumulative:
                selected_segment = segment
                break
        
        segment_props = customer_segments[selected_segment]
        
        # Enhanced probability calculations
        base_price_penalty = 0.0
        if product.price > avg_comp_price:
            price_diff_ratio = (product.price - avg_comp_price) / avg_comp_price
            base_price_penalty = min(Decimal('0.15'), price_diff_ratio * Decimal('0.3'))
            # Apply customer segment price sensitivity
            price_penalty = base_price_penalty * Decimal(str(segment_props["price_factor"]))
        else:
            price_penalty = Decimal('0.0')
        
        # Category-specific adjustments
        category_factors = {
            "Electronics": {"defect_rate": Decimal('1.2'), "return_rate": Decimal('1.3'), "review_rate": Decimal('1.1')},
            "Health": {"defect_rate": Decimal('0.8'), "return_rate": Decimal('1.5'), "review_rate": Decimal('1.3')},
            "Beauty": {"defect_rate": Decimal('0.9'), "return_rate": Decimal('1.4'), "review_rate": Decimal('1.2')},
            "Toys": {"defect_rate": Decimal('1.1'), "return_rate": Decimal('1.2'), "review_rate": Decimal('0.9')},
            "Books": {"defect_rate": Decimal('0.7'), "return_rate": Decimal('0.8'), "review_rate": Decimal('1.4')},
            "DEFAULT": {"defect_rate": Decimal('1.0'), "return_rate": Decimal('1.0'), "review_rate": Decimal('1.0')}
        }
        
        category_factor = category_factors.get(product.category, category_factors["DEFAULT"])
        
        # Calculate enhanced probabilities
        # Fix: Convert trust_factor to Decimal to prevent float/Decimal mixing
        trust_factor = Decimal(str(max(0.1, trust_score)))
        review_multiplier = segment_props["review_rate"] * category_factor["review_rate"]
        
        # Positive review probability
        prob_pos_review = max(Decimal('0.02'), min(Decimal('0.25'),
            Decimal('0.12') * trust_factor * review_multiplier - price_penalty * Decimal('0.5')))
        
        # Negative review probability
        prob_neg_review = max(Decimal('0.01'), min(Decimal('0.15'),
            (Decimal('0.08') - Decimal('0.06') * trust_factor) * category_factor["defect_rate"] + price_penalty))
        
        # A-to-Z claim probability
        prob_a_to_z = max(Decimal('0.001'), min(Decimal('0.05'),
            (Decimal('0.015') - Decimal('0.012') * Decimal(str(trust_factor))) * Decimal(str(category_factor["return_rate"])) + price_penalty * Decimal('0.3')))
        
        # Return request probability
        prob_return = max(Decimal('0.005'), min(Decimal('0.08'),
            Decimal('0.03') * category_factor["return_rate"] + price_penalty * Decimal('0.2')))
        
        # Customer message probability
        prob_message = Decimal('0.02') + (Decimal('1.0') - Decimal(str(trust_factor))) * Decimal('0.02')

        # Seller feedback probability
        prob_feedback = max(Decimal('0.01'), Decimal('0.015') * review_multiplier)
        
        # Generate event based on probabilities
        roll = self.rng.random()
        
        if roll < prob_pos_review:
            return "positive_review"
        elif roll < prob_pos_review + prob_neg_review:
            return "negative_review"
        elif roll < prob_pos_review + prob_neg_review + prob_a_to_z:
            return "a_to_z_claim"
        elif roll < prob_pos_review + prob_neg_review + prob_a_to_z + prob_return:
            return "return_request"
        elif roll < prob_pos_review + prob_neg_review + prob_a_to_z + prob_return + prob_message:
            return "customer_message"
        elif roll < prob_pos_review + prob_neg_review + prob_a_to_z + prob_return + prob_message + prob_feedback:
            return "seller_feedback"
        
        return None