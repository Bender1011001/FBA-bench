"""
CustomerReputationService

Tracks customer sentiment and reviews to maintain a per-ASIN reputation score in the WorldStore.
Integrates with BSR calculation via BsrEngineV3Service through WorldStore.get_reputation_score.

Responsibilities:
- Subscribe to SaleOccurred events
  * With a configured probability, emit a CustomerReviewEvent derived from sale context
  * Update reputation score based on generated review rating
  * Penalize reputation for stockouts/unmet demand
- Subscribe to RespondToReviewCommand (from CustomerServiceSkill)
  * Slightly improve reputation as an effect of good customer service response

Configuration (optional):
- review_probability: float in [0,1], default 0.10 (10% chance per sale to generate a review)
- ema_alpha: float in (0,1], default 0.10 (EMA update factor towards rating/5.0)
- stockout_penalty: float in [0,1], default 0.05 (max penalty applied when unmet_demand == units_demanded)
"""
from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

from event_bus import EventBus
from services.world_store import WorldStore
from fba_events.sales import SaleOccurred
from fba_events.customer import CustomerReviewEvent, RespondToReviewCommand

logger = logging.getLogger(__name__)


@dataclass
class ReputationConfig:
    review_probability: float = 0.10
    ema_alpha: float = 0.10
    stockout_penalty: float = 0.05  # Maximum downward adjustment for full stockout


class CustomerReputationService:
    """
    Event-driven reputation tracker service.

    Lifecycle:
      await start(event_bus, world_store)
      await stop()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.config = ReputationConfig(
            review_probability=float(cfg.get("review_probability", 0.10)),
            ema_alpha=float(cfg.get("ema_alpha", 0.10)),
            stockout_penalty=float(cfg.get("stockout_penalty", 0.05)),
        )
        self._started: bool = False
        self._event_bus: Optional[EventBus] = None
        self._world_store: Optional[WorldStore] = None

    async def start(self, event_bus: EventBus, world_store: WorldStore) -> None:
        if self._started:
            return
        self._event_bus = event_bus
        self._world_store = world_store

        # Subscribe to sales and customer service responses
        await self._event_bus.subscribe(SaleOccurred, self._on_sale_occurred)
        await self._event_bus.subscribe(RespondToReviewCommand, self._on_respond_to_review)

        self._started = True
        logger.info("CustomerReputationService started and subscribed to SaleOccurred and RespondToReviewCommand")

    async def stop(self) -> None:
        # EventBus does not currently provide unsubscribe; flip running flag
        self._started = False
        logger.info("CustomerReputationService stopped")

    # ========== Event Handlers ==========

    async def _on_sale_occurred(self, event: SaleOccurred) -> None:
        """
        Handle sales events:
          - With probability p, generate a CustomerReviewEvent
          - Update reputation via EMA towards rating/5.0 if review emitted
          - Penalize reputation for unmet demand (stockout)
        """
        if not self._started or self._world_store is None or self._event_bus is None:
            return

        asin = event.asin
        units_demanded = max(0, int(getattr(event, "units_demanded", 0)))
        units_sold = max(0, int(getattr(event, "units_sold", 0)))
        unmet = max(0, units_demanded - units_sold)

        # Deterministic pseudo-random selection using event_id hash for reproducibility
        h = int(hashlib.sha256((event.event_id or f"{asin}-{datetime.now().isoformat()}").encode()).hexdigest(), 16)
        # Map to [0,1)
        u = (h % 10_000_000) / 10_000_000.0

        emitted_rating: Optional[int] = None
        if u < self.config.review_probability:
            # Derive rating from sale context:
            # - Base around 4 stars with bounded adjustments
            # - If conversion is high and price likely attractive => slightly higher
            # - Penalize for any unmet demand (customers can be annoyed)
            conversion = 0.0
            try:
                conversion = float(getattr(event, "conversion_rate", 0.0))
            except Exception:
                conversion = 0.0

            # Start with base rating
            base = 4.0

            # Conversion lift: up to +0.5 stars if conversion is strong
            conv_lift = min(0.5, max(0.0, conversion) * 0.5)

            # Stockout penalty: proportional to unmet_demand ratio
            penalty = 0.0
            if units_demanded > 0 and unmet > 0:
                unmet_ratio = min(1.0, unmet / float(units_demanded))
                penalty = self.config.stockout_penalty * 5.0 * unmet_ratio  # convert to "stars" scale (0..0.25 star default)

            raw_rating = base + conv_lift - penalty
            bounded = max(1.0, min(5.0, raw_rating))
            emitted_rating = int(round(bounded))

            review = CustomerReviewEvent(
                event_id=f"review_{asin}_{event.event_id}",
                timestamp=datetime.now(),
                asin=asin,
                rating=emitted_rating,
                comment=f"Auto-generated review (conversion={conversion:.2f}, unmet={unmet})"
            )
            await self._event_bus.publish(review)
            logger.info(f"CustomerReputationService emitted review for {asin}: rating={emitted_rating}")

            # Update reputation towards rating/5 via EMA
            self._update_reputation_towards(asin, target=emitted_rating / 5.0, alpha=self.config.ema_alpha)

        # Apply additional penalty for unmet demand regardless of review emission
        if units_demanded > 0 and unmet > 0:
            unmet_ratio = min(1.0, unmet / float(units_demanded))
            self._apply_stockout_penalty(asin, unmet_ratio)

    async def _on_respond_to_review(self, event: RespondToReviewCommand) -> None:
        """
        Handle customer service response command by providing a small positive nudge to reputation.
        """
        if not self._started or self._world_store is None:
            return

        asin = getattr(event, "asin", None)
        if not asin:
            return

        current = self._world_store.get_reputation_score(asin)
        # Gentle improvement capped in [0,1], magnitude smaller than review impact
        boost = 0.01
        new_score = max(0.0, min(1.0, current + boost))
        self._world_store.set_reputation_score(asin, new_score)
        logger.info(f"CustomerReputationService reputation boost for {asin} due to RespondToReviewCommand: {current:.3f} -> {new_score:.3f}")

    # ========== Internal Helpers ==========

    def _update_reputation_towards(self, asin: str, target: float, alpha: float) -> None:
        """
        EMA update of reputation towards target in [0,1].
        rep = (1 - alpha) * rep + alpha * target
        """
        if self._world_store is None:
            return
        try:
            rep = float(self._world_store.get_reputation_score(asin))
        except Exception:
            rep = 0.7
        t = max(0.0, min(1.0, float(target)))
        a = max(0.0, min(1.0, float(alpha)))
        new_rep = (1.0 - a) * rep + a * t
        new_rep = max(0.0, min(1.0, new_rep))
        self._world_store.set_reputation_score(asin, new_rep)
        logger.debug(f"CustomerReputationService EMA rep update for {asin}: rep={rep:.3f} target={t:.3f} alpha={a:.2f} -> {new_rep:.3f}")

    def _apply_stockout_penalty(self, asin: str, unmet_ratio: float) -> None:
        """
        Apply a penalty proportional to unmet demand ratio.
        Max penalty reduces reputation by stockout_penalty when unmet_ratio == 1.0.
        """
        if self._world_store is None:
            return
        try:
            rep = float(self._world_store.get_reputation_score(asin))
        except Exception:
            rep = 0.7
        penalty = max(0.0, min(1.0, float(unmet_ratio))) * max(0.0, min(1.0, self.config.stockout_penalty))
        new_rep = max(0.0, min(1.0, rep - penalty))
        if new_rep != rep:
            self._world_store.set_reputation_score(asin, new_rep)
            logger.info(f"CustomerReputationService stockout penalty for {asin} (ratio={unmet_ratio:.2f}): {rep:.3f} -> {new_rep:.3f}")