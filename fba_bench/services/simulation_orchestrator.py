"""
Simulation Orchestrator Service.

Coordinates all simulation services to replace the monolithic tick_day() method
with a clean, modular, and testable architecture.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from fba_bench.money import Money
from fba_bench.ledger import Entry, Transaction
from fba_bench.config_loader import load_config

# Load configuration constants
_config = load_config()
PROFESSIONAL_MONTHLY = _config.fees.professional_monthly
CUBIC_FEET_PER_UNIT = _config.simulation.cubic_feet_per_unit
MONTHS_STORAGE_DEFAULT = _config.simulation.months_storage_default
REMOVAL_UNITS_DEFAULT = _config.simulation.removal_units_default
RETURN_FEES_DEFAULT = _config.simulation.return_fees_default
AGED_DAYS_DEFAULT = _config.simulation.aged_days_default
AGED_CUBIC_FEET_PER_UNIT = _config.simulation.aged_cubic_feet_per_unit
LOW_INVENTORY_UNITS_DEFAULT = _config.simulation.low_inventory_units_default
TRAILING_DAYS_SUPPLY_DEFAULT = _config.simulation.trailing_days_supply_default
WEEKS_SUPPLY_DEFAULT = _config.simulation.weeks_supply_default
INDIVIDUAL_PER_ITEM = _config.fees.individual_per_item
from fba_bench.services.sales_processor import SalesProcessor
from fba_bench.services.competitor_manager import CompetitorManager
from fba_bench.services.demand_service import DemandService
from fba_bench.services.inventory_service import InventoryService
from fba_bench.services.customer_event_service import CustomerEventService
from fba_bench.services.penalty_fee_service import PenaltyFeeService
from fba_bench.services.fee_calculation_service import FeeCalculationService
from fba_bench.services.event_management_service import EventManagementService
from fba_bench.services.trust_score_service import TrustScoreService
from fba_bench.services.listing_manager import ListingManagerService


class SimulationOrchestrator:
    """
    Orchestrates all simulation services to provide a clean, modular tick_day() implementation.
    
    This service replaces the 686-line monolithic tick_day() method with a clean
    orchestration pattern that delegates to specialized services.
    """
    
    def __init__(
        self,
        sales_processor: Optional[SalesProcessor],
        competitor_manager: Optional[CompetitorManager],
        demand_service: DemandService,
        inventory_service: InventoryService,
        customer_event_service: CustomerEventService,
        penalty_fee_service: PenaltyFeeService,
        fee_calculation_service: Optional[FeeCalculationService],
        event_management_service: Optional[EventManagementService],
        trust_score_service: Optional[TrustScoreService],
        listing_manager_service: Optional[ListingManagerService],
        config: Dict[str, Any]
    ):
        """
        Initialize simulation orchestrator with services.
        
        Args:
            sales_processor: Service for processing sales transactions (optional)
            competitor_manager: Service for managing competitor behavior (optional)
            bsr_service: Service for BSR calculations
            customer_event_service: Service for customer event generation
            penalty_fee_service: Service for penalty and ancillary fee calculations
            fee_calculation_service: Service for fee calculations and ledger entries (optional)
            event_management_service: Service for event management (optional)
            trust_score_service: Service for trust score updates (optional)
            config: Configuration dictionary
        """
        self.sales_processor = sales_processor
        self.competitor_manager = competitor_manager
        self.demand_service = demand_service
        self.inventory_service = inventory_service
        self.customer_event_service = customer_event_service
        self.penalty_fee_service = penalty_fee_service
        self.fee_calculation_service = fee_calculation_service
        self.event_management_service = event_management_service
        self.trust_score_service = trust_score_service
        self.listing_manager_service = listing_manager_service
        self.config = config
    
    def tick_day(self, simulation: Any) -> None:
        """
        Execute a single simulation day with clean service orchestration.
        
        This method replaces the 686-line monolithic tick_day() method with
        a clean, modular approach that delegates to specialized services.
        
        Args:
            simulation: The simulation object containing all state
        """
        # 1. Update simulation time
        self._advance_simulation_time(simulation)
        
        # 2. Handle monthly fees
        self._process_monthly_fees(simulation)
        
        # 3. Process adversarial events
        self._process_adversarial_events(simulation)
        
        # 4. Update competitors
        self._update_competitors(simulation)
        
        # 5. Process sales for each product
        self._process_product_sales(simulation)
    
    def _advance_simulation_time(self, simulation: Any) -> None:
        """Advance simulation by one day."""
        simulation.now += timedelta(days=1)
    
    def _process_monthly_fees(self, simulation: Any) -> None:
        """Handle monthly Professional plan fees."""
        if (simulation.selling_plan == "Professional" and 
            simulation.now.day == 1 and 
            not simulation.monthly_plan_fee_charged):
            
            self._charge_monthly_plan_fee(simulation)
        elif simulation.now.day != 1:
            simulation.monthly_plan_fee_charged = False
    
    def _charge_monthly_plan_fee(self, simulation: Any) -> None:
        """Charge the monthly Professional selling plan fee."""
        fee = Money.from_dollars(PROFESSIONAL_MONTHLY)
        
        simulation.ledger.post(Transaction(
            "Professional Plan Monthly Fee",
            debits=[Entry("Fees", fee, simulation.now)],
            credits=[Entry("Cash", fee, simulation.now)]
        ))
        
        simulation.monthly_plan_fee_charged = True
        simulation.event_log.append(
            f"Day {simulation.now.day}: Charged Professional plan monthly fee: ${PROFESSIONAL_MONTHLY}"
        )
    
    def _process_adversarial_events(self, simulation: Any) -> None:
        """Process adversarial events if enabled."""
        if simulation.adversarial_events:
            simulation.adversarial_events.run_events(
                simulation, simulation.now.timetuple().tm_yday
            )
    
    def _update_competitors(self, simulation: Any) -> None:
        """Update competitor behavior and market dynamics."""
        simulation.update_competitors()
    
    def _process_product_sales(self, simulation: Any) -> None:
        """Process sales for each product using modular services."""
        for asin, product in simulation.products.items():
            self._process_single_product_sales(simulation, asin, product)
    
    def _process_single_product_sales(
        self, simulation: Any, asin: str, product: Any
    ) -> None:
        """Process sales for a single product with full service orchestration."""
        # Get competitors for this product
        competitors = [c for c in simulation.competitors if c.asin != asin]

        # 0. Update trust score using TrustScoreService
        if self.trust_score_service is not None:
            # Gather customer events for this product
            product_events = [
                e for e in simulation.customer_events if e.get("asin") == asin
            ]
            total_orders = getattr(product, "total_orders", 0)
            new_trust_score = self.trust_score_service.update_score(
                getattr(product, "trust_score", 1.0),
                product_events,
                total_orders
            )
            if hasattr(product, "set_trust_score"):
                product.set_trust_score(new_trust_score)
            else:
                product.trust_score = new_trust_score

        # 1. Manage Listings (suppression/reinstatement)
        suppression_info = None
        if self.listing_manager_service is not None:
            suppression_info = self.listing_manager_service.update_listings(
                getattr(product, "trust_score", 1.0),
                product,
                simulation.event_log,
                simulation.now
            )
        else:
            suppression_info = {"suppressed": False, "demand_multiplier": 1.0, "search_penalty": 0.0}

        # 2. Process sales using SalesProcessor
        sales_result = self.sales_processor.process_product_sales(
            asin=asin,
            product=product,
            competitors=competitors,
            customer_events=simulation.customer_events,
            current_date=simulation.now,
            fee_engine=simulation.fees,
            selling_plan=simulation.selling_plan,
            event_log=simulation.event_log
        )

        units_sold = sales_result.units_sold
        demand = sales_result.demand

        # Apply suppression effects to demand and BSR
        if suppression_info and suppression_info.get("suppressed", False):
            demand = int(demand * suppression_info.get("demand_multiplier", 1.0))
            search_penalty = suppression_info.get("search_penalty", 0.0)
            if search_penalty > 0 and hasattr(product, "bsr"):
                # Worsen BSR by the penalty factor (higher BSR = worse ranking)
                BSR_MAX_VALUE = getattr(product, "BSR_MAX_VALUE", 1_000_000)
                product.bsr = min(BSR_MAX_VALUE, int(product.bsr * (1 + search_penalty)))

        # 3. Update inventory after sales and capture COGS
        cogs = Money.zero()
        if self.inventory_service is not None:
            cogs = self.inventory_service.update_inventory(product, units_sold)
            
            # Record COGS transaction in ledger if we have sales
            if units_sold > 0 and cogs > Money.zero():
                asin_str = getattr(product, "asin", "UNKNOWN")
                simulation.ledger.post(Transaction(
                    f"COGS for {units_sold} units of {asin_str}",
                    [Entry("Cost of Goods Sold", cogs, simulation.now)],
                    [Entry("Inventory", cogs, simulation.now)]
                ))

        # 4. Update BSR metrics
        self.demand_service.update_bsr(
            product=product,
            units_sold=units_sold,
            demand=demand,
            competitors=competitors
        )

        # 5. Generate customer events if sales occurred
        if units_sold > 0:
            self._generate_customer_events(
                simulation, asin, product, units_sold, competitors
            )

    def _prepare_fee_parameters(self, product: Any, units_sold: int) -> Dict[str, Any]:
        """Prepare parameters for fee engine calculation."""
        # Calculate product dimensions
        if hasattr(product, "dimensions") and product.dimensions:
            dims = product.dimensions
            cubic_feet_per_unit = (
                dims.get("L", 10.0) * dims.get("W", 6.0) * dims.get("H", 2.0)
            ) / 1728.0
        else:
            cubic_feet_per_unit = CUBIC_FEET_PER_UNIT
        
        cubic_feet = cubic_feet_per_unit * units_sold
        
        # Use default values from config
        
        aged_cubic_feet_per_unit = (
            cubic_feet_per_unit * 0.4 if hasattr(product, "dimensions") and product.dimensions
            else AGED_CUBIC_FEET_PER_UNIT
        )
        aged_cubic_feet = aged_cubic_feet_per_unit * units_sold
        
        return {
            "category": product.category,
            "price": product.price,
            "size_tier": "standard",  # Default size tier
            "size": "small",
            "is_holiday_season": (datetime.now().month in [11, 12]),
            "dim_weight_applies": False,
            "cubic_feet": cubic_feet,
            "months_storage": MONTHS_STORAGE_DEFAULT,
            "removal_units": REMOVAL_UNITS_DEFAULT,
            "return_applicable_fees": RETURN_FEES_DEFAULT,
            "aged_days": AGED_DAYS_DEFAULT,
            "aged_cubic_feet": aged_cubic_feet,
            "low_inventory_units": LOW_INVENTORY_UNITS_DEFAULT,
            "trailing_days_supply": TRAILING_DAYS_SUPPLY_DEFAULT,
            "weeks_supply": WEEKS_SUPPLY_DEFAULT,
            "unplanned_units": 0,
            "penalty_fee": 0.0,
            "ancillary_fee": 0.0
        }
    
    def _calculate_selling_plan_fee(self, selling_plan: str, units_sold: int) -> Money:
        """Calculate selling plan per-item fees."""
        if selling_plan == "Individual":
            return Money.from_dollars(INDIVIDUAL_PER_ITEM) * units_sold
        return Money.zero()
    
    def _post_sales_transaction(
        self,
        simulation: Any,
        asin: str,
        revenue: Money,
        total_fees: Money,
        cogs: Money
    ) -> None:
        """Post sales transaction to ledger with proper accounting."""
        ts = simulation.now
        
        # Prepare debit and credit entries
        debits = [
            Entry("COGS", cogs, ts),
            Entry("Fees", total_fees, ts),
        ]
        
        credits = [
            Entry("Revenue", revenue, ts),
            Entry("Inventory", cogs, ts),
        ]
        
        # Handle cash flow
        net_cash = revenue - total_fees
        if net_cash >= Money.zero():
            debits.insert(0, Entry("Cash", net_cash, ts))
        else:
            credits.insert(0, Entry("Cash", -net_cash, ts))
        
        # Post balanced transaction
        simulation.ledger.post(Transaction(
            f"Sales and fees for {asin}",
            debits=debits,
            credits=credits
        ))
    
    def _generate_customer_events(
        self,
        simulation: Any,
        asin: str,
        product: Any,
        units_sold: int,
        competitors: List[Any]
    ) -> None:
        """Generate customer events for the product."""
        # Calculate average competitor price
        avg_comp_price = None
        if competitors:
            total_comp_price = Money.zero()
            for competitor in competitors:
                total_comp_price += competitor.price
            avg_comp_price = total_comp_price / len(competitors)
        
        # Generate events
        events = self.customer_event_service.generate_customer_events(
            asin=asin,
            product=product,
            units_sold=units_sold,
            current_date=simulation.now,
            avg_comp_price=avg_comp_price
        )
        
        # Store events
        if asin not in simulation.customer_events:
            simulation.customer_events[asin] = []
        
        simulation.customer_events[asin].extend(events)
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get metrics about the orchestration performance."""
        return {
            "services_count": 7,
            "modular_architecture": True,
            "testable_components": True,
            "separation_of_concerns": True,
            "single_responsibility": True
        }