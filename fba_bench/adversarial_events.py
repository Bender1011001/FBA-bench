"""
Adversarial Events and Exploit Catalog for FBA-Bench.

Defines a system for red-team testing, agent hardening, and stress-testing via adversarial events.
"""

from typing import Callable, List, Dict, Any
import random

from fba_bench.config_loader import load_config

# Load configuration
_config = load_config()
DEFAULT_SUPPLY_SHOCK_FACTOR = _config.adversarial_events.default_supply_shock_factor
DEFAULT_SUPPLY_SHOCK_DURATION = _config.adversarial_events.default_supply_shock_duration
DEFAULT_REVIEW_ATTACK_COUNT = _config.adversarial_events.default_review_attack_count
DEFAULT_PRICE_WAR_DROP_PCT = _config.adversarial_events.default_price_war_drop_pct
DEFAULT_PRICE_WAR_DURATION = _config.adversarial_events.default_price_war_duration
DEFAULT_FAKE_REVIEW_COUNT = _config.adversarial_events.default_fake_review_count
DEFAULT_INVENTORY_FREEZE_DURATION = _config.adversarial_events.default_inventory_freeze_duration
DEFAULT_BUYBOX_HIJACK_DURATION = _config.adversarial_events.default_buybox_hijack_duration
DEFAULT_POLICY_PENALTY_AMOUNT = _config.adversarial_events.default_policy_penalty_amount

class AdversarialEvent:
    """
    Base class for adversarial events.

    Attributes:
        name (str): Name of the event.
        description (str): Description of the event.
        apply_fn (Callable): Function to apply the event to the simulation.
    """
    def __init__(self, name: str, description: str, apply_fn: Callable[['Simulation', int], None]):
        """
        Initialize an AdversarialEvent.

        Args:
            name (str): Name of the event.
            description (str): Description of the event.
            apply_fn (Callable): Function to apply the event to the simulation.
        """
        self.name = name
        self.description = description
        self.apply_fn = apply_fn

    def apply(self, simulation, day: int):
        """
        Apply the event to the simulation.

        Args:
            simulation (Simulation): The simulation instance.
            day (int): The current simulation day.
        """
        self.apply_fn(simulation, day)

class SupplyShockEvent(AdversarialEvent):
    """
    Event that reduces base demand for all products to simulate a supply shock.

    Args:
        shock_factor (float): Factor by which to reduce base demand.
        duration (int): Duration of the supply shock in days.
    """
    def __init__(self, shock_factor: float = DEFAULT_SUPPLY_SHOCK_FACTOR, duration: int = DEFAULT_SUPPLY_SHOCK_DURATION):
        def apply_fn(sim, day):
            for prod in sim.products.values():
                prod.base_demand = max(1, prod.base_demand * shock_factor)
            if hasattr(sim, "event_log"):
                sim.event_log.append(
                    f"Day {day}: Supply shock! Base demand reduced by {shock_factor*100:.0f}% for {duration} days."
                )
        super().__init__(
            name="Supply Shock",
            description=f"Reduces base demand for all products by {shock_factor*100:.0f}% for {duration} days.",
            apply_fn=apply_fn
        )
        self.duration = duration

class ReviewAttackEvent(AdversarialEvent):
    """
    Event that injects negative reviews to worsen a product's BSR.

    Args:
        asin (str): ASIN of the product to attack.
        negative_reviews (int): Number of negative reviews to inject.
    """
    def __init__(self, asin: str, negative_reviews: int = DEFAULT_REVIEW_ATTACK_COUNT):
        def apply_fn(sim, day):
            if asin in sim.products:
                prod = sim.products[asin]
                prod.bsr = int(prod.bsr * 1.5)
                if hasattr(sim, "event_log"):
                    sim.event_log.append(
                        f"Day {day}: Review attack on {asin}! BSR worsened."
                    )
        super().__init__(
            name="Review Attack",
            description=f"Injects {negative_reviews} negative reviews to worsen BSR.",
            apply_fn=apply_fn
        )
class ListingHijackEvent(AdversarialEvent):
    """
    Event that corrupts a product's listing to simulate a hijack.

    Args:
        asin (str): ASIN of the product to hijack.
        hijack_type (str): Type of hijack ("title", "description", etc.).
    """
    def __init__(self, asin: str, hijack_type: str = "title"):
        def apply_fn(sim, day):
            if asin in sim.products:
                prod = sim.products[asin]
                if hijack_type == "title":
                    prod.category = "HIJACKED"
                    if hasattr(prod, "name"):
                        prod.name = "FAKE PRODUCT"
                    if hasattr(prod, "description"):
                        prod.description = "This listing has been hijacked and now contains misleading information."
                elif hijack_type == "description":
                    if hasattr(prod, "description"):
                        prod.description = "This listing has been hijacked and now contains misleading information."
                # Extend for other hijack types (e.g., images)
                if hasattr(sim, "event_log"):
                    sim.event_log.append(
                        f"Day {day}: Listing hijack on {asin}! {hijack_type.capitalize()} corrupted."
                    )
        super().__init__(
            name="Listing Hijack",
            description=f"Corrupts the {hijack_type} of product {asin} to simulate a hijack.",
            apply_fn=apply_fn
        )

class SupplierBankruptcyEvent(AdversarialEvent):
    def __init__(self, supplier_id: str):
        def apply_fn(sim, day):
            # Mark the supplier as bankrupt in the simulation's supply chain
            if hasattr(sim, "supply_chain") and supplier_id in sim.supply_chain.suppliers:
                from fba_bench.supply_chain import SupplierStatus
                supplier = sim.supply_chain.suppliers[supplier_id]
                supplier.status = SupplierStatus.BANKRUPT
                
                # Cancel all active orders with this supplier
                orders_to_cancel = [order_id for order_id, order in sim.supply_chain.active_orders.items()
                                  if order["supplier_id"] == supplier_id]
                
                for order_id in orders_to_cancel:
                    sim.supply_chain.complete_order(order_id, successful=False)
                
                if hasattr(sim, "event_log"):
                    sim.event_log.append(
                        f"Day {day}: Supplier bankruptcy! Supplier {supplier_id} ({supplier.name}) is now bankrupt. "
                        f"{len(orders_to_cancel)} orders cancelled."
                    )
            elif hasattr(sim, "event_log"):
                sim.event_log.append(
                    f"Day {day}: Supplier bankruptcy event triggered but supplier {supplier_id} not found."
                )
        super().__init__(
            name="Supplier Bankruptcy",
            description=f"Supplier {supplier_id} becomes bankrupt, forcing supply chain adaptation.",
            apply_fn=apply_fn
        )

class PriceWarEvent(AdversarialEvent):
    def __init__(self, asin: str, price_drop_pct: float = 0.2, duration: int = 2):
        def apply_fn(sim, day):
            for comp in getattr(sim, "competitors", []):
                if hasattr(comp, "price"):
                    comp.price = max(0.5, comp.price * (1 - price_drop_pct))
            if hasattr(sim, "event_log"):
                sim.event_log.append(
                    f"Day {day}: Price war! Competitor prices for {asin} dropped by {price_drop_pct*100:.0f}% for {duration} days."
                )
        super().__init__(
            name="Price War",
            description=f"Competitors drop prices by {price_drop_pct*100:.0f}% for {duration} days.",
            apply_fn=apply_fn
        )
        self.duration = duration

class FakeReviewSurgeEvent(AdversarialEvent):
    def __init__(self, asin: str, positive: bool = False, count: int = 20):
        def apply_fn(sim, day):
            if asin in sim.products:
                prod = sim.products[asin]
                # For simplicity, just adjust BSR and log
                if positive:
                    prod.bsr = int(prod.bsr * 0.8)
                else:
                    prod.bsr = int(prod.bsr * 1.2)
                if hasattr(sim, "event_log"):
                    sim.event_log.append(
                        f"Day {day}: {'Positive' if positive else 'Negative'} fake review surge on {asin}! BSR {'improved' if positive else 'worsened'}."
                    )
        super().__init__(
            name="Fake Review Surge",
            description=f"Injects {count} {'positive' if positive else 'negative'} fake reviews.",
            apply_fn=apply_fn
        )

class InventoryFreezeEvent(AdversarialEvent):
    def __init__(self, asin: str, duration: int = 2):
        def apply_fn(sim, day):
            if hasattr(sim.inventory, "freeze"):
                sim.inventory.freeze(asin, duration)
            if hasattr(sim, "event_log"):
                sim.event_log.append(
                    f"Day {day}: Inventory freeze! {asin} cannot be sold for {duration} days."
                )
        super().__init__(
            name="Inventory Freeze",
            description=f"Freezes inventory for {asin} for {duration} days.",
            apply_fn=apply_fn
        )
        self.duration = duration

class BuyboxHijackEvent(AdversarialEvent):
    def __init__(self, asin: str, duration: int = 1):
        def apply_fn(sim, day):
            if asin in sim.products:
                prod = sim.products[asin]
                prod.sales_velocity = 0
                if hasattr(sim, "event_log"):
                    sim.event_log.append(
                        f"Day {day}: Buybox hijack! {asin} loses buybox, sales drop to zero for {duration} days."
                    )
        super().__init__(
            name="Buybox Hijack",
            description=f"Removes buybox from {asin}, sales drop to zero for {duration} days.",
            apply_fn=apply_fn
        )
        self.duration = duration

class PolicyPenaltyEvent(AdversarialEvent):
    def __init__(self, asin: str, penalty_amount: float = 100.0):
        def apply_fn(sim, day):
            if hasattr(sim, "ledger"):
                sim.ledger.post(
                    sim.ledger.Transaction(
                        f"Policy penalty for {asin}",
                        debits=[sim.ledger.Entry("Fees", penalty_amount, sim.now)],   # Fee expense (debit increases expense)
                        credits=[sim.ledger.Entry("Cash", penalty_amount, sim.now)]   # Cash decrease (credit decreases asset)
                    )
                )
            if hasattr(sim, "event_log"):
                sim.event_log.append(
                    f"Day {day}: Policy penalty! {asin} fined ${penalty_amount:.2f}."
                )
        super().__init__(
            name="Policy Penalty",
            description=f"Applies a policy violation penalty of ${penalty_amount:.2f} to {asin}.",
            apply_fn=apply_fn
        )

class PolicyChangeEvent(AdversarialEvent):
    def __init__(self, fee_increase: float = 1.1):
        def apply_fn(sim, day):
            if hasattr(sim, "fee_engine"):
                sim.fee_engine.FBA_FEES = {
                    k: {kk: vv * fee_increase for kk, vv in v.items()}
                    for k, v in sim.fee_engine.FBA_FEES.items()
                }
            if hasattr(sim, "event_log"):
                sim.event_log.append(
                    f"Day {day}: Policy change! FBA fees increased by {int((fee_increase-1)*100)}%."
                )
        super().__init__(
            name="Policy Change",
            description=f"Increases all FBA fees by {int((fee_increase-1)*100)}%.",
            apply_fn=apply_fn
        )

class AdversarialEventCatalog:
    """
    Catalog and manager for adversarial events.
    """
    def __init__(self, events: List[AdversarialEvent] = None):
        self.events = events or []

    def add_event(self, event: AdversarialEvent):
        self.events.append(event)

    def get_event(self, name: str) -> AdversarialEvent:
        for event in self.events:
            if event.name == name:
                return event
        raise ValueError(f"Event '{name}' not found.")

    def random_event(self) -> AdversarialEvent:
        return random.choice(self.events)

    def run_events(self, simulation, day: int):
        """
        Apply all events for the day.
        """
        for event in self.events:
            event.apply(simulation, day)