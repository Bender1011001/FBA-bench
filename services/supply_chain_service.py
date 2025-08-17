"""
SupplyChainService

Manages supplier state and pending purchase orders. Integrates with EventBus to:
- Accept PlaceOrderCommand from agents
- Schedule deliveries at future ticks (current_tick + lead_time)
- On each tick, process arriving orders and publish InventoryUpdate events
- Support disruption controls (lead time increase, fulfillment rate reduction)

Contract:
- Subscribe to PlaceOrderCommand and TickEvent
- Methods:
    - set_disruption(active: bool, lead_time_increase: int = 0, fulfillment_rate: float = 1.0)
    - process_tick() -> processes any arrivals for the current tick
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from event_bus import EventBus, get_event_bus
from services.world_store import WorldStore
from events import TickEvent, InventoryUpdate  # via shim to fba_events
from fba_events.supplier import PlaceOrderCommand
from money import Money

logger = logging.getLogger(__name__)


@dataclass
class PendingOrder:
    order_id: str
    supplier_id: str
    asin: str
    quantity: int
    max_price: Money
    arrival_tick: int


class SupplyChainService:
    """
    Stateful supply chain/order management.

    - Receives PlaceOrderCommand and schedules a PendingOrder.
    - On each TickEvent, processes arrivals and publishes InventoryUpdate.
    - Disruption controls:
        active -> whether disruption parameters should apply
        lead_time_increase -> additional ticks to add to base lead time
        fulfillment_rate -> portion of ordered units actually delivered (0.0-1.0)
    """

    def __init__(
        self,
        world_store: WorldStore,
        event_bus: Optional[EventBus] = None,
        base_lead_time: int = 2,
    ) -> None:
        self.world_store = world_store
        self.event_bus = event_bus or get_event_bus()
        self.base_lead_time = max(0, int(base_lead_time))

        self._started: bool = False
        self._current_tick: int = 0

        # Disruption controls
        self._disruption_active: bool = False
        self._lead_time_increase: int = 0
        self._fulfillment_rate: float = 1.0

        # Pending orders keyed by asin for quick grouping
        self._pending: List[PendingOrder] = []

    async def start(self) -> None:
        if self._started:
            return
        await self.event_bus.subscribe(PlaceOrderCommand, self._on_place_order)
        await self.event_bus.subscribe(TickEvent, self._on_tick)
        self._started = True
        logger.info("SupplyChainService started and subscribed to PlaceOrderCommand and TickEvent.")

    async def stop(self) -> None:
        # EventBus doesn't expose unsubscribe; rely on _started flag if needed
        self._started = False
        logger.info("SupplyChainService stopped.")

    def set_disruption(
        self,
        active: bool,
        lead_time_increase: int = 0,
        fulfillment_rate: float = 1.0,
    ) -> None:
        """
        Configure disruption parameters.

        - lead_time_increase: extra ticks added to base lead time (non-negative)
        - fulfillment_rate: delivered quantity ratio [0.0, 1.0]
        """
        self._disruption_active = bool(active)
        self._lead_time_increase = max(0, int(lead_time_increase))
        self._fulfillment_rate = max(0.0, min(1.0, float(fulfillment_rate)))
        logger.debug(
            "SupplyChainService disruption set: active=%s, lead_time_increase=%d, fulfillment_rate=%.2f",
            self._disruption_active, self._lead_time_increase, self._fulfillment_rate
        )

    async def _on_place_order(self, event: PlaceOrderCommand) -> None:
        """
        Handle incoming PlaceOrderCommand by scheduling a pending delivery.
        """
        try:
            extra_lead = self._lead_time_increase if self._disruption_active else 0
            arrival_tick = self._current_tick + self.base_lead_time + extra_lead
            pending = PendingOrder(
                order_id=event.event_id or f"order_{uuid.uuid4()}",
                supplier_id=event.supplier_id,
                asin=event.asin,
                quantity=event.quantity,
                max_price=event.max_price,
                arrival_tick=arrival_tick,
            )
            self._pending.append(pending)
            logger.info(
                "SupplyChainService scheduled order: asin=%s qty=%d arrival_tick=%d supplier=%s",
                pending.asin, pending.quantity, pending.arrival_tick, pending.supplier_id
            )
        except Exception as e:
            logger.error(f"Error handling PlaceOrderCommand {getattr(event, 'event_id', 'unknown')}: {e}", exc_info=True)

    async def _on_tick(self, event: TickEvent) -> None:
        """
        Update current tick and process arrivals.
        """
        try:
            self._current_tick = int(getattr(event, "tick_number", self._current_tick))
            await self.process_tick()
        except Exception as e:
            logger.error(f"Error processing TickEvent in SupplyChainService: {e}", exc_info=True)

    async def process_tick(self) -> None:
        """
        Process any pending orders whose arrival_tick <= current tick.
        Applies fulfillment_rate and publishes InventoryUpdate for delivered units.
        """
        if not self._pending:
            return

        remaining: List[PendingOrder] = []
        for po in self._pending:
            if po.arrival_tick <= self._current_tick:
                # Determine delivered quantity under disruption
                delivered = int(po.quantity * (self._fulfillment_rate if self._disruption_active else 1.0))
                delivered = max(0, min(po.quantity, delivered))

                # Deliver arrived units
                if delivered > 0:
                    try:
                        current_qty = self.world_store.get_product_inventory_quantity(po.asin)
                    except Exception:
                        current_qty = 0

                    new_qty = current_qty + delivered
                    try:
                        cost_basis = self.world_store.get_product_cost_basis(po.asin)
                    except Exception:
                        cost_basis = Money.zero()

                    inv_event = InventoryUpdate(
                        event_id=f"inv_supply_{po.asin}_{uuid.uuid4()}",
                        timestamp=datetime.now(),
                        asin=po.asin,
                        new_quantity=new_qty,
                        cost_basis=cost_basis,
                    )
                    await self.event_bus.publish(inv_event)
                    logger.info(
                        "SupplyChainService delivered: asin=%s delivered=%d new_inventory=%d",
                        po.asin, delivered, new_qty
                    )

                remainder = po.quantity - delivered
                # If not fully fulfilled and disruption active, re-queue remainder for next tick
                if remainder > 0:
                    remaining.append(
                        PendingOrder(
                            order_id=po.order_id,
                            supplier_id=po.supplier_id,
                            asin=po.asin,
                            quantity=remainder,
                            max_price=po.max_price,
                            arrival_tick=self._current_tick + 1,  # next tick attempt
                        )
                    )
            else:
                remaining.append(po)

        self._pending = remaining

    # Introspection helpers (useful for tests/metrics)
    def get_pending_orders(self) -> List[Dict[str, object]]:
        return [
            {
                "order_id": po.order_id,
                "supplier_id": po.supplier_id,
                "asin": po.asin,
                "quantity": po.quantity,
                "arrival_tick": po.arrival_tick,
            }
            for po in self._pending
        ]

    def get_current_tick(self) -> int:
        return self._current_tick