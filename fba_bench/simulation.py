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

# Load configuration constants
_config = load_config()

# Product defaults
DEFAULT_CATEGORY = _config.agent_defaults.default_category
DEFAULT_COST = _config.agent_defaults.default_cost
DEFAULT_PRICE = _config.agent_defaults.default_price
DEFAULT_QTY = _config.agent_defaults.default_qty

# Fee calculation constants
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

# BSR calculation constants
BSR_BASE = _config.market_dynamics.bsr_base
BSR_SMOOTHING_FACTOR = _config.market_dynamics.bsr_smoothing_factor
BSR_MIN_VALUE = _config.market_dynamics.bsr_min_value
BSR_MAX_VALUE = _config.market_dynamics.bsr_max_value

# Competitor behavior constants
COMPETITOR_PRICE_CHANGE_BASE = _config.competitor_model.price_change_base
COMPETITOR_SALES_CHANGE_BASE = _config.competitor_model.sales_change_base
COMPETITOR_STRATEGIES = _config.competitor_model.strategies
AGGRESSIVE_UNDERCUT_THRESHOLD = _config.competitor_model.aggressive_undercut_threshold
AGGRESSIVE_UNDERCUT_AMOUNT = _config.competitor_model.aggressive_undercut_amount
FOLLOWER_PRICE_SENSITIVITY = _config.competitor_model.follower_price_sensitivity
PREMIUM_PRICE_MAINTENANCE = _config.competitor_model.premium_price_maintenance
VALUE_COMPETITIVE_THRESHOLD = _config.competitor_model.value_competitive_threshold

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
        Advance simulation by one day using modular service orchestration.
        
        This method replaces the previous 686-line monolithic implementation
        with a clean, testable, and maintainable service-oriented approach.
        """
        if not hasattr(self, '_orchestrator'):
            self._initialize_orchestrator()
        
        self._orchestrator.tick_day(self)
    
    def _initialize_orchestrator(self):
        """Initialize the simulation orchestrator with all required services."""
        from fba_bench.services.simulation_orchestrator import SimulationOrchestrator
        from fba_bench.services.bsr_calculation_service import BSRCalculationService
        from fba_bench.services.customer_event_service import CustomerEventService
        from fba_bench.services.penalty_fee_service import PenaltyFeeService
        
        # Load configuration for services
        config = {
            'ema_decay': EMA_DECAY,
            'bsr_smoothing_factor': BSR_SMOOTHING_FACTOR,
            'bsr_base': BSR_BASE,
            'bsr_min_value': BSR_MIN_VALUE,
            'bsr_max_value': BSR_MAX_VALUE,
            'return_processing_fee_pct': 0.15,
            'unplanned_service_fee_per_unit': 3.00
        }
        
        # Initialize services
        from fba_bench.services.demand_service import DemandService
        from fba_bench.services.inventory_service import InventoryService
        from fba_bench.services.trust_score_service import TrustScoreService
        from fba_bench.services.listing_manager import ListingManagerService
        
        demand_service = DemandService()
        inventory_service = InventoryService()
        customer_event_service = CustomerEventService(self.rng, config)
        penalty_fee_service = PenaltyFeeService(self.rng, config)
        trust_score_service = TrustScoreService()
        listing_manager_service = ListingManagerService()
        
        # Initialize orchestrator with available services
        # Some services may not exist yet, so we'll use None for optional ones
        self._orchestrator = SimulationOrchestrator(
            sales_processor=getattr(self, 'sales_processor', None),
            competitor_manager=getattr(self, 'competitor_manager', None),
            demand_service=demand_service,
            inventory_service=inventory_service,
            customer_event_service=customer_event_service,
            penalty_fee_service=penalty_fee_service,
            fee_calculation_service=getattr(self, 'fee_calculation_service', None),
            event_management_service=getattr(self, 'event_management_service', None),
            trust_score_service=trust_score_service,
            listing_manager_service=listing_manager_service,
            config=config
        )


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

    # Legacy methods moved to specialized services for better modularity and testability
    # These methods are kept as stubs for backward compatibility
    
        return Money.zero()


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