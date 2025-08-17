"""
MarketSimulationService

Initial coherent world model step for FBA-Bench.
- Listens to SetPriceCommand and CompetitorPricesUpdated events
- Computes demand/sales using a simple, stateful model
- Publishes SaleOccurred and updates inventory via InventoryUpdate
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from event_bus import EventBus, get_event_bus
from money import Money
from services.world_store import WorldStore
from fba_events.pricing import SetPriceCommand
from fba_events.sales import SaleOccurred
from events import CompetitorPricesUpdated, InventoryUpdate  # via shim (fba_events re-export)

logger = logging.getLogger(__name__)


@dataclass
class CompetitorSnapshot:
    asin: str
    price: Money
    bsr: Optional[int] = None
    sales_velocity: Optional[float] = None


class MarketSimulationService:
    """
    Stateful demand/sales simulation.

    Demand model (initial step):
      demand = base_demand * (p / p_ref)^(-elasticity)
      where p_ref = min(average competitor price, previous known price), if available
      If no competitor data is available, use the last known canonical price as p_ref.

    - units_sold = min(demand, inventory)
    - revenue = units_sold * price
    - fees: initially set to $0.00 (can be integrated with FeeCalculationService later)
    - profit = revenue - fees - cost_basis * units_sold
    - publishes SaleOccurred and updates inventory via InventoryUpdate
    """

    def __init__(
        self,
        world_store: WorldStore,
        event_bus: Optional[EventBus] = None,
        base_demand: int = 100,
        demand_elasticity: float = 1.5,
    ) -> None:
        self.world_store = world_store
        self.event_bus = event_bus or get_event_bus()
        self.base_demand = base_demand
        self.demand_elasticity = demand_elasticity

        # Internal caches
        self._competitors_by_asin: Dict[str, List[CompetitorSnapshot]] = {}
        self._price_reference_by_asin: Dict[str, Money] = {}

        # Control flags
        self._started = False

    async def start(self) -> None:
        """Subscribe to relevant events."""
        if self._started:
            return
        await self.event_bus.subscribe(SetPriceCommand, self._on_set_price_command)
        await self.event_bus.subscribe(CompetitorPricesUpdated, self._on_competitor_prices_updated)
        self._started = True
        logger.info("MarketSimulationService started and subscribed to events.")

    async def _on_competitor_prices_updated(self, event: CompetitorPricesUpdated) -> None:
        """Update competitor cache on incoming updates."""
        try:
            updated: List[CompetitorSnapshot] = []
            for comp in getattr(event, "competitors", []):
                # comp has fields: asin, price (Money), bsr, sales_velocity
                updated.append(
                    CompetitorSnapshot(
                        asin=comp.asin,
                        price=comp.price,
                        bsr=getattr(comp, "bsr", None),
                        sales_velocity=getattr(comp, "sales_velocity", None),
                    )
                )
            # We don't know mapping from product ASIN -> competitor list here; this event likely contains many competitor ASINs.
            # For simplicity, we bucket by each competitor's ASIN (downstream can map as needed).
            # If the target product ASIN equals competitor.asin, it represents alternate sellers on same listing;
            # otherwise it represents related SKUs/close substitutes (future refinement).
            for comp in updated:
                self._competitors_by_asin.setdefault(comp.asin, [])
                # Keep last N competitor snapshots (bounded)
                self._competitors_by_asin[comp.asin].append(comp)
                if len(self._competitors_by_asin[comp.asin]) > 50:
                    self._competitors_by_asin[comp.asin] = self._competitors_by_asin[comp.asin][-25:]
        except Exception as e:
            logger.error(f"Error handling CompetitorPricesUpdated: {e}", exc_info=True)

    async def _on_set_price_command(self, event: SetPriceCommand) -> None:
        """
        Hook for awareness. WorldStore arbitrates and applies prices.
        We don't compute sales directly here because WorldStore must first update the canonical price.
        Scenarios can explicitly call process_for_asin after publishing to sequence the tick deterministically.
        """
        # No-op here; orchestration via scenario.tick calls process_for_asin
        pass

    def _compute_price_reference(self, asin: str, current_price: Money) -> Money:
        """
        Determine reference price p_ref for demand calculation:
        - If competitors known for this ASIN: use min(avg competitor price, current reference)
        - Else use cached reference or fall back to current canonical price
        """
        ref = self._price_reference_by_asin.get(asin)
        if ref is None:
            ref = current_price

        comps = self._competitors_by_asin.get(asin, [])
        if comps:
            # Use average competitor price across most recent snapshots for this ASIN
            # Consider last up to 10 snapshots for smoothing
            window = comps[-10:] if len(comps) > 10 else comps
            if window:
                avg_cents = sum(c.price.cents for c in window) / len(window)
                avg_price = Money(int(round(avg_cents)))
                # Reference is min of prior ref and avg competitor (aggressive market pressure)
                ref = avg_price if avg_price.cents < ref.cents else ref

        # Cache and return
        self._price_reference_by_asin[asin] = ref
        return ref

    def _safe_div(self, a: float, b: float, default: float = 1.0) -> float:
        try:
            if b <= 0.0:
                return default
            return a / b
        except Exception:
            return default

    def _demand(self, price: Money, ref_price: Money) -> int:
        """
        Compute integer units demanded with elasticity.
        demand = base_demand * (p / p_ref)^(-elasticity)
        """
        p = float(price.cents) / 100.0
        p_ref = float(ref_price.cents) / 100.0
        ratio = self._safe_div(p, p_ref, default=1.0)
        # clamp ratio to avoid extremes
        if ratio <= 0.0:
            ratio = 0.01
        quantity = self.base_demand * (ratio ** (-self.demand_elasticity))
        # integer demand
        return max(0, int(round(quantity)))

    async def process_for_asin(self, asin: str) -> None:
        """
        Execute market simulation for a single ASIN for the current tick:
        - Read canonical price and inventory from WorldStore
        - Compute demand and realized sales
        - Publish SaleOccurred
        - Update inventory via InventoryUpdate
        """
        try:
            product = self.world_store.get_product_state(asin)
            if not product:
                logger.debug(f"No product state for ASIN {asin}, skipping market processing.")
                return

            current_price = product.price
            ref_price = self._compute_price_reference(asin, current_price)

            units_demanded = self._demand(current_price, ref_price)
            inventory_qty = self.world_store.get_product_inventory_quantity(asin)
            units_sold = min(units_demanded, max(0, inventory_qty))

            revenue = current_price * units_sold
            total_fees = Money.zero()  # integrate FeeCalculationService later
            cost_basis = self.world_store.get_product_cost_ratio_safe(asin) if hasattr(self.world_store, "get_product_cost_ratio_safe") else self.world_store.get_product_cost_basis(asin)
            # cost_basis above represents unit or total? In WorldStore it is average cost basis per unit.
            # Convert to total cost for units sold:
            if hasattr(cost_basis, "cents"):
                total_cost = cost_basis * units_sold
            else:
                # Fallback if external helper returned number
                total_cost = Money.zero()

            total_profit = Money(revenue.cents - total_fees.cents - total_cost.cents)

            # Trust score / BSR not yet integrated in WorldStore; use placeholders suitable for initial step
            trust_score = float(product.metadata.get("trust_score", 0.9)) if hasattr(product, "metadata") and isinstance(product.metadata, dict) else 0.9
            bsr = int(product.metadata.get("bsr", 1000)) if hasattr(product, "metadata") and isinstance(product.metadata, dict) else 1000

            sale = SaleOccurred(
                event_id=f"sale_{asin}_{int(datetime.now().timestamp()*1000)}",
                timestamp=datetime.now(),
                asin=asin,
                units_sold=units_sold,
                units_demanded=units_demanded,
                unit_price=current_price,
                total_revenue=revenue,
                total_fees=total_fees,
                total_profit=total_profit,
                cost_basis=total_cost,
                trust_score_at_sale=trust_score,
                bsr_at_sale=bsr,
                conversion_rate=(units_sold / units_demanded) if units_demanded > 0 else 0.0,
                fee_breakdown={},
                market_conditions={
                    "reference_price": str(ref_price),
                    "elasticity": self.demand_elasticity,
                    "base_demand": self.base_demand,
                },
                customer_segment=None,
            )
            await self.event_bus.publish(sale)

            # Update inventory after sale
            new_qty = max(0, inventory_qty - units_sold)
            inv_update = InventoryUpdate(
                event_id=f"inv_{asin}_{int(datetime.now().timestamp()*1000)}",
                timestamp=datetime.now(),
                asin=asin,
                new_quantity=new_qty,
                cost_basis=self.world_store.get_product_cost_basis(asin),
            )
            await self.event_bus.publish(inv_update)

            logger.info(
                f"MarketSimulationService processed ASIN {asin}: price={current_price}, "
                f"demand={units_demanded}, sold={units_sold}, revenue={revenue}"
            )

        except Exception as e:
            logger.error(f"Error in MarketSimulationService.process_for_asin for {asin}: {e}", exc_info=True)