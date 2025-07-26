"""FBA‑Bench core simulation engine (condensed)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List
from decimal import Decimal

from . import market_dynamics
from .adversarial_events import AdversarialEventCatalog
from .fee_engine import FeeEngine
from .inventory import InventoryManager
from .ledger import Ledger, Transaction, Entry
from .supply_chain import GlobalSupplyChain
from .audit import run_and_audit, RunAudit
from .money import Money
from .services import CompetitorManager, SalesProcessor
from .models import Competitor, SalesResult
from .config_loader import load_config

from fba_bench.config import (
    # Product defaults
    DEFAULT_CATEGORY, DEFAULT_COST, DEFAULT_PRICE, DEFAULT_QTY,
    
    # Fee calculation constants
    CUBIC_FEET_PER_UNIT, MONTHS_STORAGE_DEFAULT, REMOVAL_UNITS_DEFAULT,
    RETURN_FEES_DEFAULT, AGED_DAYS_DEFAULT, AGED_CUBIC_FEET_PER_UNIT,
    LOW_INVENTORY_UNITS_DEFAULT, TRAILING_DAYS_SUPPLY_DEFAULT,
    WEEKS_SUPPLY_DEFAULT, EMA_DECAY,
    
    # BSR calculation constants
    BSR_BASE, BSR_SMOOTHING_FACTOR, BSR_MIN_VALUE, BSR_MAX_VALUE,
    
    # Competitor behavior constants
    COMPETITOR_PRICE_CHANGE_BASE, COMPETITOR_SALES_CHANGE_BASE,
    COMPETITOR_STRATEGIES, AGGRESSIVE_UNDERCUT_THRESHOLD,
    AGGRESSIVE_UNDERCUT_AMOUNT, FOLLOWER_PRICE_SENSITIVITY,
    PREMIUM_PRICE_MAINTENANCE, VALUE_COMPETITIVE_THRESHOLD,
)

@dataclass
class Product:
    """
    Represents a product in the FBA-Bench simulation.

    Fields:
        asin (str): Amazon Standard Identification Number.
        category (str): Product category, influences fees and seasonality.
        cost (Money): Wholesale cost per unit.
        price (Money): Retail price set by the agent.
        bsr (int): Best Seller Rank, dynamically changes (default: 100000).
        base_demand (float): Baseline average demand if price were $1 (default: 20.0).
        sales_velocity (float): Current sales velocity (default: 0.0).
        sales_history (List[float]): List of daily sales.
        demand_history (List[float]): List of daily demand.
        conversion_history (List[float]): List of daily conversion rates.
        ema_sales_velocity (float): Exponential moving average of sales velocity.
        ema_conversion (float): Exponential moving average of conversion rate.
        dimensions (Dict[str, float]): Physical dimensions (L, W, H) in inches.
        weight (float): Product weight in pounds.
        _inventory_manager: InventoryManager = field(default=None, init=False, repr=False)
    """
    asin: str
    category: str
    cost: Money
    price: Money
    base_demand: float = DEFAULT_QTY  # Use config for base_demand if available, else set in config
    bsr: int = 100000
    sales_velocity: float = 0.0
    sales_history: List[float] = field(default_factory=list)
    demand_history: List[float] = field(default_factory=list)
    conversion_history: List[float] = field(default_factory=list)
    ema_sales_velocity: float = 0.0
    ema_conversion: float = 0.0
    dimensions: Dict[str, float] = field(default_factory=lambda: {"L": 10.0, "W": 6.0, "H": 2.0})
    weight: float = 1.0
    _inventory_manager: InventoryManager = field(default=None, init=False, repr=False)
    
    @property
    def qty(self) -> int:
        """Get current inventory quantity for this product."""
        if self._inventory_manager is None:
            return 0
        return self._inventory_manager.quantity(self.asin)
    
    def _set_inventory_manager(self, inventory_manager: InventoryManager):
        """Internal method to set the inventory manager reference."""
        self._inventory_manager = inventory_manager

class Simulation:
    """
    Runs a simplified FBA market simulation.

    Attributes:
        rng (random.Random): Random number generator for reproducibility.
        ledger (Ledger): Double-entry ledger for all transactions.
        fees (FeeEngine): Fee calculation engine.
        inventory (InventoryManager): Manages product inventory.
        now (datetime): Current simulation date.
        products (Dict[str, Product]): All products in the simulation.
        adversarial_events (AdversarialEventCatalog): Catalog of adversarial events.
        event_log (list): Log of simulation events.
        customer_events (dict): Customer events by ASIN.
        competitors (list): List of competitor products.
    """

    def __init__(self, seed: int = 42, adversarial_events: AdversarialEventCatalog = None, num_competitors: int = 3, selling_plan: str = "Professional"):
        """
        Initialize the Simulation.

        Args:
            seed (int): Random seed for reproducibility.
            adversarial_events (AdversarialEventCatalog, optional): Catalog of adversarial events.
            num_competitors (int): Number of competitors to simulate.
            selling_plan (str): Selling plan type - "Professional" or "Individual"
        """
        self.rng = random.Random(seed)
        self.ledger = Ledger()
        self.fees = FeeEngine()
        self.inventory = InventoryManager()
        self.supply_chain = GlobalSupplyChain()  # Initialize Global Supply Chain
        self.now = datetime(2025, 1, 1)
        self.products: Dict[str, Product] = {}
        self.adversarial_events = adversarial_events
        self.event_log = []
        self.selling_plan = selling_plan  # Professional or Individual
        self.monthly_plan_fee_charged = False  # Track if monthly fee has been charged
        self.disputes = []  # List of filed disputes (for Dispute.file Tool)
        # Customer systems: reviews, feedback, messages, claims
        self.customer_events = {}  # asin -> list of events
        
        # Load configuration and initialize services
        self.config = load_config()
        self.competitor_manager = CompetitorManager(self.config, self.rng)
        self.sales_processor = SalesProcessor(
            config=self.config,
            ledger=self.ledger,
            inventory=self.inventory,
            market_dynamics=market_dynamics,
            rng=self.rng
        )
        
        # seed capital
        seed_capital = Money.from_dollars(10_000)
        self.ledger.post(Transaction("Seed capital", [Entry("Cash", seed_capital, self.now)], [Entry("Equity", seed_capital, self.now)]))
        # Initialize competitors
        # Competitors are initialized after the first product is launched (see launch_product)
        self.competitors: List[Competitor] = []

    def _init_competitors(self, num_competitors: int, asin: str = None, category: str = None):
        """
        Initializes the list of competitor products for the simulation.

        Args:
            num_competitors (int): Number of competitors to create.
            asin (str, optional): ASIN of the agent's product.
            category (str, optional): Product category.

        Populates self.competitors with mock competitor products for BSR v3 and demand calculations.
        """
        if asin is None or category is None or asin not in self.products:
            self.competitors = []
            return
            
        agent_prod = self.products[asin]
        self.competitors = self.competitor_manager.initialize_competitors(
            num_competitors=num_competitors,
            agent_asin=asin,
            agent_category=category,
            agent_price=agent_prod.price,
            agent_cost=agent_prod.cost,
            agent_base_demand=agent_prod.base_demand,
            agent_bsr=agent_prod.bsr
        )

    def update_competitors(self):
        """
        Update competitor behavior using the CompetitorManager service.
        
        This method delegates all competitor update logic to the CompetitorManager,
        maintaining the same external interface while using the new service architecture.
        """
        if not self.products or not self.competitors:
            return
        
        # Get agent product (assume single product for now)
        agent_prod = list(self.products.values())[0]
        agent_price = agent_prod.price
        agent_sales = getattr(agent_prod, 'ema_sales_velocity', 0.0)
        agent_bsr = agent_prod.bsr
        
        # Market conditions
        is_holiday_season = self.now.month in [11, 12]
        market_demand_multiplier = 1.5 if is_holiday_season else 1.0
        
        # Delegate to CompetitorManager
        self.competitor_manager.update_competitors(
            competitors=self.competitors,
            agent_price=agent_price,
            agent_sales=int(agent_sales),
            agent_bsr=agent_bsr,
            market_demand_multiplier=market_demand_multiplier,
            current_date=self.now
        )
        
        # Update market share calculations
        self.competitor_manager.calculate_market_share(self.competitors, agent_sales)


    # Tool‑like API
    def launch_product(self, asin: str, category: str, cost, price, qty: int, base_demand: float = DEFAULT_QTY, bsr: int = 100000, dimensions=None, weight=None):
        """
        Launch a new product in the simulation.

        Args:
            asin (str): Amazon Standard Identification Number.
            category (str): Product category.
            cost (Money or float): Wholesale cost per unit.
            price (Money or float): Retail price.
            qty (int): Initial inventory quantity.
            base_demand (float, optional): Baseline demand.
            bsr (int, optional): Initial Best Seller Rank.
            dimensions (dict, optional): Product dimensions.
            weight (float, optional): Product weight.
        """
        # Convert float inputs to Money for backward compatibility
        if isinstance(cost, (int, float)):
            cost = Money.from_dollars(cost)
        if isinstance(price, (int, float)):
            price = Money.from_dollars(price)
            
        prod = Product(
            asin, category, cost, price, base_demand=base_demand, bsr=bsr,
            dimensions=dimensions if dimensions is not None else {"L": 10.0, "W": 6.0, "H": 2.0},
            weight=weight if weight is not None else 1.0
        )
        # Connect the product to the inventory manager
        prod._set_inventory_manager(self.inventory)
        self.products[asin] = prod
        self.inventory.add(asin, qty, cost, self.now)
        amt = cost * qty  # Money arithmetic
        self.ledger.post(Transaction("Inventory purchase",
            [Entry("Inventory", amt, self.now)],
            [Entry("Cash", amt, self.now)]))  # Credit entry (cash outflow)
        # Initialize competitors for this product (default 3 competitors)
        if len(self.competitors) == 0:
            self._init_competitors(3, asin=asin, category=category)

    def set_price(self, asin: str, price):
        """
        Set the price for a given product.

        Args:
            asin (str): Amazon Standard Identification Number.
            price (Money or float): New price to set.
        """
        # Convert float to Money for backward compatibility
        if isinstance(price, (int, float)):
            price = Money.from_dollars(price)
        self.products[asin].price = price

    def tick_day(self):
        """
        Advance simulation by one day, process sales & fees.

        Updates product sales, demand, BSR, processes fees, and generates customer events.
        """
        self.update_competitors()
        self.now += timedelta(days=1)
        
        # Charge monthly Professional plan fee on the first day of each month
        if self.selling_plan == "Professional" and self.now.day == 1 and not self.monthly_plan_fee_charged:
            self._charge_monthly_plan_fee()
        elif self.now.day != 1:
            self.monthly_plan_fee_charged = False  # Reset for next month
        
        # --- Adversarial Events ---
        if self.adversarial_events:
            self.adversarial_events.run_events(self, self.now.timetuple().tm_yday)
        # Process sales for each product using the SalesProcessor
        for asin, prod in self.products.items():
            # --- Competitor Set for this product ---
            competitors = [c for c in self.competitors if c.asin != asin]
            
            # Process sales using the dedicated service
            sales_result = self.sales_processor.process_product_sales(
                asin=asin,
                product=prod,
                competitors=competitors,
                customer_events=self.customer_events,
                current_date=self.now,
                fee_engine=self.fees,
                selling_plan=self.selling_plan,
                event_log=self.event_log
            )
            sold = sales_result.units_sold
            demand = sales_result.demand
            # --- BSR v2: Track sales and demand history ---
            prod.sales_history.append(sold)
            prod.demand_history.append(demand)
            conversion = sold / demand if demand > 0 else 0.0
            prod.conversion_history.append(conversion)
            # --- BSR v2: Update EMA for sales velocity and conversion ---
            if len(prod.sales_history) == 1:
                prod.ema_sales_velocity = sold
                prod.ema_conversion = conversion
            else:
                prod.ema_sales_velocity = (1 - EMA_DECAY) * prod.ema_sales_velocity + EMA_DECAY * sold
                prod.ema_conversion = (1 - EMA_DECAY) * prod.ema_conversion + EMA_DECAY * conversion
            
            # --- BSR v3: Update BSR based on blueprint formula ---
            # BSR = base / (ema_sales_velocity * ema_conversion * rel_sales_index * rel_price_index)
            # Compute competitor averages for relative indices
            avg_comp_sales = BSR_SMOOTHING_FACTOR  # Use config smoothing factor instead of hardcoded 1.0
            avg_comp_price = prod.price
            
            if competitors:
                avg_comp_sales = max(BSR_SMOOTHING_FACTOR, sum(c.sales_velocity for c in competitors) / len(competitors))
                total_comp_price = Money.zero()
                for c in competitors:
                    total_comp_price += c.price
                avg_comp_price = total_comp_price / len(competitors)
            
            # Relative sales index: agent's sales velocity vs. competitors
            rel_sales_index = max(BSR_SMOOTHING_FACTOR, prod.ema_sales_velocity) / avg_comp_sales
            
            # Relative price index: agent's price vs. competitors (lower price = better index)
            rel_price_index = avg_comp_price / max(Money.from_dollars(BSR_SMOOTHING_FACTOR), prod.price)
            
            # Apply blueprint BSR formula with bounds checking
            if prod.ema_sales_velocity > BSR_SMOOTHING_FACTOR and prod.ema_conversion > BSR_SMOOTHING_FACTOR:
                denominator = (
                    prod.ema_sales_velocity *
                    prod.ema_conversion *
                    rel_sales_index *
                    float(rel_price_index)
                )
                calculated_bsr = BSR_BASE / max(BSR_SMOOTHING_FACTOR, denominator)
                prod.bsr = max(BSR_MIN_VALUE, min(BSR_MAX_VALUE, int(calculated_bsr)))
            else:
                prod.bsr = BSR_BASE
            if sold:
                revenue = sold * prod.price
                # Calculate all fees using the high-fidelity fee engine
                # --- Advanced Fee Parameter Calculation (stubs/estimates for now) ---
                # In a real implementation, these would be calculated from inventory/product state or loaded from config
                # Use product dimensions if available, else config default
                if hasattr(prod, "dimensions") and prod.dimensions:
                    dims = prod.dimensions
                    cubic_feet_per_unit = (dims.get("L", 10.0) * dims.get("W", 6.0) * dims.get("H", 2.0)) / 1728.0
                else:
                    cubic_feet_per_unit = CUBIC_FEET_PER_UNIT
                cubic_feet = cubic_feet_per_unit * sold

                months_storage = MONTHS_STORAGE_DEFAULT
                removal_units = REMOVAL_UNITS_DEFAULT
                return_applicable_fees = RETURN_FEES_DEFAULT
                aged_days = AGED_DAYS_DEFAULT
                if hasattr(prod, "dimensions") and prod.dimensions:
                    aged_cubic_feet_per_unit = cubic_feet_per_unit * 0.4  # Example: 40% aged
                else:
                    aged_cubic_feet_per_unit = AGED_CUBIC_FEET_PER_UNIT
                aged_cubic_feet = aged_cubic_feet_per_unit * sold

                low_inventory_units = LOW_INVENTORY_UNITS_DEFAULT
                trailing_days_supply = TRAILING_DAYS_SUPPLY_DEFAULT
                weeks_supply = WEEKS_SUPPLY_DEFAULT
                unplanned_units = 0  # No unplanned service in this tick

                # Determine size tier and dimensional weight
                size_tier = "standard"  # Default size tier
                dim_weight_applies = False  # Default dimensional weight
                
                # Calculate ancillary and penalty fees based on current state
                ancillary_fee = self._calculate_ancillary_fees(asin, sold)
                penalty_fee = self._calculate_penalty_fees(asin, sold)
                
                # Add selling plan per-item fee for Individual plan
                selling_plan_fee = Money.zero()
                if self.selling_plan == "Individual":
                    from fba_bench.config import INDIVIDUAL_PER_ITEM
                    selling_plan_fee = Money.from_dollars(INDIVIDUAL_PER_ITEM) * sold

                # Calculate trust score fee multiplier
                trust_score = getattr(prod, 'trust_score', 1.0)
                trust_fee_multiplier = max(1.0, 1.0 + (1.0 - trust_score) * 0.5)  # Higher fees for lower trust
                
                # BUGFIX: Convert total ancillary/penalty fees to per-unit amounts before passing to fee engine
                # The fee engine expects per-unit fees, but ancillary_fee and penalty_fee are calculated as totals
                ancillary_fee_per_unit = ancillary_fee.to_float() / max(1, sold) if ancillary_fee > Money.zero() else 0.0
                penalty_fee_per_unit = penalty_fee.to_float() / max(1, sold) if penalty_fee > Money.zero() else 0.0
                
                # All fee calculations are now handled by FeeEngine.total_fees
                fees = self.fees.total_fees(
                    category=prod.category,
                    price=prod.price,
                    size_tier=size_tier,
                    size="small" if size_tier == "standard" else "large",
                    is_holiday_season=(self.now.month in [11, 12]),
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
                # BUGFIX: fee_engine.total_fees() returns per-unit fees, so multiply by sold only once
                total_fees = (Money.from_dollars(fees["total"]) * sold * Decimal(str(trust_fee_multiplier))) + selling_plan_fee
                referral = fees["referral_fee"] * sold
                cogs = sold * prod.cost
                # ledger entries for sales and all fees - corrected accounting structure
                # All amounts are positive Money objects; accounting effect determined by debit/credit side
                ts = self.now
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
                # --- Customer Systems: Generate customer events after sale ---
                if asin not in self.customer_events:
                    self.customer_events[asin] = []
                for _ in range(sold):
                    event_type = self._generate_customer_event()
                    if event_type:
                        self.customer_events[asin].append({
                            "type": event_type,
                            "date": self.now,
                        })
    def _calculate_trust_fee_multiplier(self, trust_score: float) -> float:
        """
        Calculate fee multiplier based on seller trust score.
        Lower trust = higher fees as penalty.
        """
        if trust_score >= 0.9:
            return 1.0  # No penalty for high trust
        elif trust_score >= 0.7:
            return 1.1  # 10% penalty for medium trust
        elif trust_score >= 0.5:
            return 1.25  # 25% penalty for low trust
        else:
            return 1.5  # 50% penalty for very low trust

    def _check_listing_suppression(self, trust_score: float) -> dict:
        """
        Check if listing should be suppressed based on trust score.
        Returns a dict with suppression details including severity level.
        """
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

    def _generate_customer_event(self, asin=None, price=None, trust_score=1.0, avg_comp_price=None):
        """
        Enhanced customer event generation with realistic customer behavior patterns.
        Returns a dict describing the event, or None if no event occurs.
        
        Enhanced features:
        - Customer segments with different behaviors
        - Product category influences
        - Seasonal effects
        - Price sensitivity variations
        - Review authenticity factors
        """
        # Use the simulation's seeded RNG for determinism
        if not asin or asin not in self.products:
            return None
            
        product = self.products[asin]
        
        # Customer segmentation (affects behavior patterns)
        customer_segments = {
            "price_sensitive": {"weight": 0.4, "price_factor": 2.0, "review_rate": 0.8},
            "quality_focused": {"weight": 0.3, "price_factor": 0.5, "review_rate": 1.5},
            "convenience_focused": {"weight": 0.2, "price_factor": 1.0, "review_rate": 0.6},
            "brand_loyal": {"weight": 0.1, "price_factor": 0.3, "review_rate": 1.2}
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
        if price is not None and avg_comp_price is not None and price > avg_comp_price:
            price_diff_ratio = (price - avg_comp_price) / avg_comp_price
            base_price_penalty = min(0.15, price_diff_ratio * 0.3)
            # Apply customer segment price sensitivity
            price_penalty = base_price_penalty * segment_props["price_factor"]
        else:
            price_penalty = 0.0
        
        # Category-specific adjustments
        category_factors = {
            "Electronics": {"defect_rate": 1.2, "return_rate": 1.3, "review_rate": 1.1},
            "Health": {"defect_rate": 0.8, "return_rate": 1.5, "review_rate": 1.3},
            "Beauty": {"defect_rate": 0.9, "return_rate": 1.4, "review_rate": 1.2},
            "Toys": {"defect_rate": 1.1, "return_rate": 1.2, "review_rate": 0.9},
            "Books": {"defect_rate": 0.7, "return_rate": 0.8, "review_rate": 1.4},
            "DEFAULT": {"defect_rate": 1.0, "return_rate": 1.0, "review_rate": 1.0}
        }
        
        category_factor = category_factors.get(product.category, category_factors["DEFAULT"])
        
        # Seasonal adjustments
        season_factor = 1.0
        if hasattr(self, 'now') and self.now:
            month = self.now.month
            if month in [11, 12]:  # Holiday season
                season_factor = 1.2  # More reviews and issues during holidays
            elif month in [6, 7, 8]:  # Summer
                season_factor = 0.9  # Slightly less activity
        
        # Calculate enhanced probabilities
        trust_factor = max(0.1, trust_score)
        review_multiplier = segment_props["review_rate"] * category_factor["review_rate"] * season_factor
        
        # Positive review probability
        prob_pos_review = max(0.02, min(0.25,
            0.12 * trust_factor * review_multiplier - price_penalty * 0.5))
        
        # Negative review probability
        prob_neg_review = max(0.01, min(0.15,
            (0.08 - 0.06 * trust_factor) * category_factor["defect_rate"] + price_penalty))
        
        # A-to-Z claim probability
        prob_a_to_z = max(0.001, min(0.05,
            (0.015 - 0.012 * trust_factor) * category_factor["return_rate"] + price_penalty * 0.3))
        
        # Return request probability
        prob_return = max(0.005, min(0.08,
            0.03 * category_factor["return_rate"] + price_penalty * 0.2))
        
        # Customer message probability
        prob_message = 0.02 + (1.0 - trust_factor) * 0.02
        
        # Seller feedback probability
        prob_feedback = max(0.01, 0.015 * review_multiplier)
        
        # Product question probability
        prob_question = 0.025 if selected_segment == "quality_focused" else 0.01
        
        # Generate event based on probabilities
        r = self.rng.random()
        
        if r < prob_pos_review:
            # Generate realistic positive review
            positive_messages = [
                "Great product, exactly as described!",
                "Fast shipping and excellent quality.",
                "Highly recommend this item.",
                "Perfect for my needs, very satisfied.",
                "Good value for the price.",
                "Works as expected, no issues.",
                "Quick delivery and well packaged."
            ]
            score = self.rng.choices([4, 5], weights=[0.3, 0.7])[0]
            return {
                "type": "positive_review",
                "asin": asin,
                "score": score,
                "text": self.rng.choice(positive_messages),
                "customer_segment": selected_segment,
                "date": self.now
            }
        
        r -= prob_pos_review
        if r < prob_neg_review:
            # Generate realistic negative review
            negative_messages = [
                "Product did not meet expectations.",
                "Poor quality, broke after a few uses.",
                "Not as described in the listing.",
                "Overpriced for what you get.",
                "Shipping was delayed significantly.",
                "Product arrived damaged.",
                "Doesn't work as advertised."
            ]
            score = self.rng.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            return {
                "type": "negative_review",
                "asin": asin,
                "score": score,
                "text": self.rng.choice(negative_messages),
                "customer_segment": selected_segment,
                "date": self.now
            }
        
        r -= prob_neg_review
        if r < prob_a_to_z:
            # Generate A-to-Z claim
            claim_reasons = [
                "Item not as described",
                "Product defective",
                "Never received item",
                "Item significantly different from listing",
                "Product stopped working"
            ]
            return {
                "type": "a_to_z_claim",
                "asin": asin,
                "reason": self.rng.choice(claim_reasons),
                "customer_segment": selected_segment,
                "date": self.now
            }
        
        r -= prob_a_to_z
        if r < prob_return:
            # Generate return request
            return_reasons = [
                "Changed mind",
                "Found better price elsewhere",
                "Product not needed",
                "Size/fit issues",
                "Quality concerns"
            ]
            return {
                "type": "return_request",
                "asin": asin,
                "reason": self.rng.choice(return_reasons),
                "customer_segment": selected_segment,
                "date": self.now
            }
        
        r -= prob_return
        if r < prob_message:
            # Generate customer message
            message_types = [
                "Can you provide more details about this product?",
                "When will this item ship?",
                "Is this compatible with [other product]?",
                "What is your return policy?",
                "Can you expedite shipping?"
            ]
            return {
                "type": "message",
                "asin": asin,
                "content": self.rng.choice(message_types),
                "customer_segment": selected_segment,
                "date": self.now
            }
        
        r -= prob_message
        if r < prob_feedback:
            # Generate seller feedback
            feedback_scores = [1, 2, 3, 4, 5]
            feedback_weights = [0.05, 0.1, 0.15, 0.3, 0.4] if trust_score > 0.7 else [0.2, 0.2, 0.2, 0.2, 0.2]
            score = self.rng.choices(feedback_scores, weights=feedback_weights)[0]
            
            feedback_messages = {
                1: "Poor communication and slow shipping.",
                2: "Below average service.",
                3: "Average seller experience.",
                4: "Good seller, fast shipping.",
                5: "Excellent seller, highly recommend!"
            }
            
            return {
                "type": "seller_feedback",
                "asin": asin,
                "score": score,
                "text": feedback_messages[score],
                "customer_segment": selected_segment,
                "date": self.now
            }
        
        r -= prob_feedback
        if r < prob_question:
            # Generate product question
            questions = [
                "What are the exact dimensions?",
                "Is this item in stock?",
                "What materials is this made from?",
                "How long is the warranty?",
                "Are there any color options?"
            ]
            return {
                "type": "product_question",
                "asin": asin,
                "question": self.rng.choice(questions),
                "customer_segment": selected_segment,
                "date": self.now
            }
        
        return None  # No event generated

    def _charge_monthly_plan_fee(self):
        """Charge the monthly Professional selling plan fee."""
        from fba_bench.config import PROFESSIONAL_MONTHLY
        fee = Money.from_dollars(PROFESSIONAL_MONTHLY)
        self.ledger.post(Transaction("Professional Plan Monthly Fee",
            debits=[Entry("Fees", fee, self.now)],    # Fee expense (debit increases expense)
            credits=[Entry("Cash", fee, self.now)]))  # Cash decrease (credit decreases asset)
        self.monthly_plan_fee_charged = True
        self.event_log.append(f"Day {self.now.day}: Charged Professional plan monthly fee: ${PROFESSIONAL_MONTHLY}")

    def _calculate_ancillary_fees(self, asin: str, units_sold: int) -> Money:
        """
        Calculate ancillary fees based on current product state and recent events.
        Enhanced with realistic Amazon FBA ancillary fee scenarios.
        """
        ancillary_fee = Money.zero()
        prod = self.products.get(asin)
        if not prod:
            return Money.zero()
        
        # 1. Return processing fees (based on actual returns)
        if asin in self.customer_events:
            recent_events = [e for e in self.customer_events[asin]
                           if (self.now - e["date"]).days <= 30]  # Last 30 days
            
            # Returns and refunds
            returns = [e for e in recent_events if e["type"] in ["a_to_z_claim", "negative_review"]]
            if returns:
                from fba_bench.config import RETURN_PROCESSING_FEE_PCT
                # Return processing fee based on returned units
                estimated_return_rate = min(0.3, len(returns) / max(1, units_sold * 4))  # Max 30% return rate
                returned_units = int(units_sold * estimated_return_rate)
                ancillary_fee += Money.from_dollars(returned_units * prod.price.to_float() * RETURN_PROCESSING_FEE_PCT)
        
        # 2. Prep service fees (based on product characteristics and category)
        prep_fee_probability = 0.02  # Base 2% chance
        
        # Higher prep fees for certain categories
        high_prep_categories = ["Electronics", "Toys", "Health", "Beauty"]
        if prod.category in high_prep_categories:
            prep_fee_probability = 0.08  # 8% for high-prep categories
        
        # Size-based prep fee adjustments
        if prod.weight > 5.0:  # Heavy items need more prep
            prep_fee_probability *= 1.5
        
        if self.rng.random() < prep_fee_probability:
            from fba_bench.config import UNPLANNED_SERVICE_FEE_PER_UNIT
            prep_units = max(1, int(units_sold * 0.1))  # 10% of units need prep
            ancillary_fee += Money.from_dollars(UNPLANNED_SERVICE_FEE_PER_UNIT) * prep_units
        
        # 3. Labeling and packaging fees
        if self.rng.random() < 0.03:  # 3% chance of labeling issues
            labeling_fee = Money.from_dollars(0.55) * units_sold  # $0.55 per unit for labeling
            ancillary_fee += labeling_fee
        
        # 4. Disposal fees for damaged/unsellable inventory
        if asin in self.customer_events:
            damage_events = [e for e in self.customer_events[asin]
                           if e.get("type") == "negative_review" and "damaged" in e.get("message", "").lower()]
            if damage_events:
                # Estimate disposal needs based on damage reports
                disposal_rate = min(0.05, len(damage_events) / max(1, units_sold * 10))
                disposal_units = int(units_sold * disposal_rate)
                disposal_fee = Money.from_dollars(0.15) * disposal_units  # $0.15 per unit disposal
                ancillary_fee += disposal_fee
        
        # 5. Repackaging fees for customer returns
        if asin in self.customer_events:
            return_events = [e for e in self.customer_events[asin]
                           if e.get("type") == "a_to_z_claim"]
            if return_events:
                repackaging_rate = min(0.15, len(return_events) / max(1, units_sold * 5))
                repackaging_units = int(units_sold * repackaging_rate)
                repackaging_fee = Money.from_dollars(1.00) * repackaging_units  # $1.00 per unit repackaging
                ancillary_fee += repackaging_fee
        
        # 6. Photography and content fees (occasional)
        if self.rng.random() < 0.01:  # 1% chance per day
            content_fee = Money.from_dollars(50.0)  # One-time content update fee
            ancillary_fee += content_fee
        
        return ancillary_fee

    def _calculate_penalty_fees(self, asin: str, units_sold: int) -> Money:
        """
        Calculate penalty fees based on policy violations and performance issues.
        Enhanced with realistic Amazon FBA penalty scenarios.
        """
        penalty_fee = Money.zero()
        prod = self.products.get(asin)
        if not prod:
            return Money.zero()
        
        # 1. Performance-based penalties
        if asin in self.customer_events:
            events = self.customer_events[asin]
            recent_events = [e for e in events if (self.now - e["date"]).days <= 30]
            
            # Customer satisfaction penalties
            negative_events = [e for e in recent_events if e["type"] in ["negative_review", "a_to_z_claim"]]
            total_interactions = len(recent_events)
            
            if total_interactions > 0:
                negative_rate = len(negative_events) / total_interactions
                
                # Graduated penalty system based on negative feedback rate
                if negative_rate > 0.3:  # >30% negative feedback
                    penalty_fee += Money.from_dollars(200.0)  # Severe performance penalty
                elif negative_rate > 0.2:  # >20% negative feedback
                    penalty_fee += Money.from_dollars(100.0)  # Moderate performance penalty
                elif negative_rate > 0.1:  # >10% negative feedback
                    penalty_fee += Money.from_dollars(50.0)   # Warning penalty
            
            # A-to-Z claim penalties (more severe)
            a_to_z_claims = [e for e in recent_events if e["type"] == "a_to_z_claim"]
            if len(a_to_z_claims) > 0:
                # Escalating penalty for multiple claims
                claim_penalty = Money.from_dollars(75.0) * len(a_to_z_claims)  # $75 per claim
                if len(a_to_z_claims) > 3:
                    claim_penalty = claim_penalty * Money.from_dollars(1.5)  # 50% increase for excessive claims
                penalty_fee += claim_penalty
        
        # 2. Inventory performance penalties
        batches = self.inventory._batches.get(asin, [])
        current_inventory = sum(getattr(batch, "quantity", 0) for batch in batches)
        
        # Stockout penalty (impacts customer experience)
        if current_inventory == 0 and units_sold > 0:
            stockout_penalty = Money.from_dollars(min(100.0, units_sold * 2.0))  # $2 per lost sale, max $100
            penalty_fee += stockout_penalty
        
        # Excess inventory penalty (storage utilization)
        if current_inventory > units_sold * 10:  # More than 10x daily sales
            excess_penalty = Money.from_dollars((current_inventory - units_sold * 10) * 0.10)  # $0.10 per excess unit
            penalty_fee += Money.from_dollars(min(50.0, excess_penalty.to_decimal()))  # Cap at $50
        
        # 3. Trust score penalties
        if asin in self.customer_events:
            events = self.customer_events[asin]
            cancellations = sum(1 for e in events if e.get("type") == "a_to_z_claim")
            negative_reviews = sum(1 for e in events if e.get("type") == "negative_review")
            customer_issues = sum(1 for e in events if e.get("type") in ["negative_review", "negative_feedback"])
            
            trust_score = market_dynamics.calculate_trust_score(
                cancellations=cancellations,
                negative_reviews=negative_reviews,
                customer_issues=customer_issues,
                total_orders=max(1, units_sold * 10)  # Estimate total orders
            )
            
            if trust_score < 0.5:  # Low trust score
                trust_penalty = Money.from_dollars((0.5 - trust_score) * 200.0)  # Up to $100 penalty
                penalty_fee += trust_penalty
        else:
            trust_score = 1.0  # Default high trust score for new products
        
        # 4. Policy violation penalties
        recent_violations = [e for e in self.event_log
                           if "policy" in e.lower() and asin in e and
                           "Day " + str(max(1, self.now.day - 30)) <= e <= "Day " + str(self.now.day)]
        
        # Escalating penalties for repeated violations
        for i, violation in enumerate(recent_violations):
            base_penalty = Money.from_dollars(150.0)  # Base penalty per violation
            escalation_multiplier = 1.0 + (i * 0.5)  # 50% increase per additional violation
            penalty_fee += base_penalty * Money.from_dollars(escalation_multiplier)
        
        # 5. Category-specific penalties
        high_risk_categories = ["Health", "Beauty", "Baby", "Electronics"]
        if prod.category in high_risk_categories:
            # Higher penalties for regulated categories
            if penalty_fee > Money.zero():
                penalty_fee = penalty_fee * Money.from_dollars(1.3)  # 30% increase for high-risk categories
        
        # 6. Pricing policy penalties
        if hasattr(self, 'competitors') and asin in self.competitors:
            competitors = self.competitors[asin]
            if competitors:
                avg_competitor_price = sum(c.price for c in competitors) / len(competitors)
                
                # Penalty for excessive pricing (potential price gouging)
                if prod.price > Money.from_dollars(avg_competitor_price * 2.0):
                    pricing_penalty = Money.from_dollars(min(75.0, (prod.price.to_decimal() - avg_competitor_price) * 0.1))
                    penalty_fee += pricing_penalty
        
        # 7. Late shipment penalties (simulated)
        if self.rng.random() < 0.02:  # 2% chance of late shipment issues
            late_shipment_penalty = Money.from_dollars(25.0) * units_sold  # $25 per unit for late shipment
            penalty_fee += Money.from_dollars(min(200.0, late_shipment_penalty.to_decimal()))  # Cap at $200
        
        # 8. Account health penalties (cumulative effect)
        total_recent_penalties = sum(1 for e in self.event_log
                                   if "penalty" in e.lower() and
                                   "Day " + str(max(1, self.now.day - 90)) <= e <= "Day " + str(self.now.day))
        
        if total_recent_penalties > 5:  # More than 5 penalties in 90 days
            account_health_penalty = Money.from_dollars((total_recent_penalties - 5) * 50.0)  # $50 per excess penalty
            penalty_fee += Money.from_dollars(min(300.0, account_health_penalty.to_decimal()))  # Cap at $300
        
        return penalty_fee


    def run(self, days: int = 30):
        for _ in range(days):
            self.tick_day()
    
    def run_and_audit(self, days: int = 30) -> RunAudit:
        """Run simulation and return comprehensive audit data."""
        return run_and_audit(self, days)
    
    def enable_fee_audits(self, enabled: bool = True):
        """Enable or disable fee auditing for closure checking."""
        self._fee_audits_enabled = enabled

def bootstrap_example():
    sim = Simulation()
    sim.launch_product("B000TEST", "DEFAULT", cost=Money.from_dollars(5.0), price=Money.from_dollars(19.99), qty=100)
    sim.run(7)
    return sim
if __name__ == "__main__":
    sim = bootstrap_example()
    print("Ending cash:", sim.ledger.balance("Cash"))