"""
Advanced heuristic-based agent optimized for FBA-Bench scenarios.

This agent implements robust, production-grade pricing logic designed to perform
well across a variety of benchmark scenarios without relying on external LLMs.

It consumes a lightweight configuration dict (as created by the Pydantic
AgentConfig used in the unified agent factory) and exposes an async `decide`
method compatible with DIYAdapter in the unified agent system.

Key behaviors:
- Competitive pricing: undercut or match nearest competitor within safe bounds
- Margin protection: never price below cost floor (configurable margin)
- Demand-aware adjustments: react to recent demand signals and inventory levels
- Smoothing: limit per-tick price changes to avoid oscillation

Integration points:
- Unified agent factory creates this agent when DIY `agent_type` is "advanced"
  [AgentFactory._create_diy_agent()](benchmarking/agents/unified_agent.py:911)
- DIYAdapter converts returned ToolCall objects to AgentAction
  [DIYAdapter.decide()](benchmarking/agents/unified_agent.py:657)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import math
import logging

# Core runner protocol types (canonical for DIY agents)
from fba_bench.core.types import SimulationState, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class PriceMemory:
    """Tracks recent demand and price decisions for smoothing and trend detection."""
    demand_history: deque = field(default_factory=lambda: deque(maxlen=30))
    price_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_price: Optional[float] = None

    def push(self, demand: float, price: float) -> None:
        self.demand_history.append(float(max(demand, 0.0)))
        self.price_history.append(float(max(price, 0.0)))
        self.last_price = price

    def avg_demand(self, window: int = 7) -> float:
        if not self.demand_history:
            return 0.0
        if window <= 0:
            window = len(self.demand_history)
        items = list(self.demand_history)[-window:]
        return sum(items) / max(len(items), 1)

    def avg_price(self, window: int = 7) -> float:
        if not self.price_history:
            return 0.0
        if window <= 0:
            window = len(self.price_history)
        items = list(self.price_history)[-window:]
        return sum(items) / max(len(items), 1)


class AdvancedAgent:
    """
    Advanced heuristic agent for FBA-Bench DIY framework.

    Expected construction path:
      agent = AdvancedAgent(config_dict)

    Where config_dict typically comes from PydanticAgentConfig.dict():
      {
        "framework": "diy",
        "parameters": {
          "agent_type": "advanced",
          "target_asin": "B0DEFAULT",
          "min_margin": 0.12,
          "undercut": 0.01,
          "max_change_pct": 0.15,
          "price_sensitivity": 0.10,
          "reaction_speed": 1,
          "inventory_low_threshold": 10,
          "inventory_target": 100
        },
        ...
      }

    The agent returns a list[ToolCall] where each ToolCall has:
      - tool_name: "set_price"
      - parameters: {"asin": str, "price": float}
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # Entire PydanticAgentConfig as dict is passed in; we care mainly about parameters.
        self._raw_config = config or {}
        params = (self._raw_config.get("parameters") or {}) if isinstance(self._raw_config, dict) else {}

        # Core parameters with robust defaults
        self.agent_id: str = self._raw_config.get("id") or self._raw_config.get("agent_id") or "advanced_agent"
        self.target_asin: str = params.get("target_asin", "B0DEFAULT")

        # Pricing controls
        self.min_margin: float = float(params.get("min_margin", 0.12))          # minimum margin over cost
        self.undercut: float = float(params.get("undercut", 0.01))              # undercut competitor by 1%
        self.max_change_pct: float = float(params.get("max_change_pct", 0.15))  # limit per-tick price change to 15%
        self.price_sensitivity: float = float(params.get("price_sensitivity", 0.10))  # demand elasticity heuristic

        # Inventory-aware behavior
        self.reaction_speed: float = float(params.get("reaction_speed", 1.0))   # amplify dampen adjustments
        self.inventory_low_threshold: int = int(params.get("inventory_low_threshold", 10))
        self.inventory_target: int = int(params.get("inventory_target", 100))

        # Internal memory per ASIN
        self._mem: Dict[str, PriceMemory] = {}

        # Lifecycle flags
        self._initialized: bool = False
        self._shutdown: bool = False

        # Validate parameters sanity
        if self.min_margin < 0.0:
            self.min_margin = 0.0
        if self.undercut < 0.0:
            self.undercut = 0.0
        if self.max_change_pct <= 0.0:
            self.max_change_pct = 0.10

        logger.info(f"AdvancedAgent[{self.agent_id}] configured for ASIN={self.target_asin} "
                    f"min_margin={self.min_margin} undercut={self.undercut} "
                    f"max_change_pct={self.max_change_pct} price_sensitivity={self.price_sensitivity}")

    # Optional sync initialize for DIYAdapter compatibility
    def initialize(self) -> None:
        self._initialized = True
        self._shutdown = False
        logger.info(f"AdvancedAgent[{self.agent_id}] initialized")

    # Optional async initialize if called by adapter
    async def _async_initialize(self) -> None:
        self.initialize()

    # Optional reset hook
    def reset(self) -> None:
        self._mem.clear()
        logger.info(f"AdvancedAgent[{self.agent_id}] reset state")

    # Optional shutdown hook
    def shutdown(self) -> None:
        self._shutdown = True
        logger.info(f"AdvancedAgent[{self.agent_id}] shutdown")

    # Core decision API expected by DIYAdapter
    async def decide(self, state: SimulationState) -> List[ToolCall]:
        """
        Compute next action(s) as ToolCalls. Primary action is "set_price".

        Robustness: Works with partial state information. If required data is missing,
        falls back to safe, conservative adjustments and never prices below a reasonable floor.
        """
        if not self._initialized:
            # Support both sync and async init patterns
            if asyncio.iscoroutinefunction(self._async_initialize):
                await self._async_initialize()
            else:
                self.initialize()

        asin = self._resolve_target_asin(state)
        product = self._extract_product(state, asin)
        current_price = self._get_current_price(product)
        cost = self._get_cost(product, fallback_ratio=0.6, fallback_price=current_price)
        floor_price = max(cost * (1.0 + self.min_margin), 0.01)

        competitor_price = self._estimate_competitor_price(product, default=current_price or max(floor_price, 1.0))
        demand = self._estimate_recent_demand(state, asin, product)
        inventory = self._get_inventory(product)
        inventory_ratio = self._compute_inventory_ratio(inventory)

        # Initialize memory for ASIN
        mem = self._mem.setdefault(asin, PriceMemory())

        # Derive target price using a composite heuristic:
        # 1) Start from competitive anchor (competitor - undercut) but respect floor
        # 2) Adjust by demand and inventory pressure
        # 3) Smooth by limiting per-tick change from last known price
        anchor_price = max(competitor_price * (1.0 - self.undercut), floor_price)

        demand_factor = self._compute_demand_factor(demand, mem.avg_demand(7))
        inventory_factor = self._compute_inventory_factor(inventory_ratio)

        # Reaction scaling
        adjustment_multiplier = 1.0 + (self.price_sensitivity * self.reaction_speed) * (demand_factor + inventory_factor)
        raw_target = max(anchor_price * adjustment_multiplier, floor_price)

        # Smooth against last price when available
        target_price = self._smooth_price(raw_target, reference_price=(mem.last_price or current_price or raw_target))

        # Book-keeping
        mem.push(demand=demand, price=target_price)

        # Ensure final sanity
        final_price = max(round(float(target_price), 2), round(floor_price, 2))

        logger.debug(
            f"AdvancedAgent[{self.agent_id}] asin={asin} "
            f"current={current_price} competitor={competitor_price} cost={cost} floor={floor_price} "
            f"demand={demand} inv={inventory} inv_ratio={inventory_ratio:.2f} "
            f"anchor={anchor_price} adj_mult={adjustment_multiplier:.3f} "
            f"raw_target={raw_target} smoothed={target_price} final={final_price}"
        )

        # If price is unchanged and we lack sufficient signal to adjust, we still emit the command
        # to keep the control loop explicit.
        return [
            ToolCall(
                tool_name="set_price",
                parameters={"asin": asin, "price": final_price},
                confidence=self._compute_confidence(current_price, final_price, demand),
                reasoning=self._build_reasoning(
                    asin=asin,
                    current_price=current_price,
                    competitor_price=competitor_price,
                    floor=floor_price,
                    demand=demand,
                    inventory=inventory,
                    target=final_price,
                ),
                priority=1,
            )
        ]

    # -----------------------
    # Heuristic subroutines
    # -----------------------

    def _resolve_target_asin(self, state: SimulationState) -> str:
        if self.target_asin and self.target_asin != "B0DEFAULT":
            return self.target_asin
        # Fallback to first product asin if available
        if state.products:
            first = state.products[0]
            asin = first.get("asin") or first.get("ASIN") or self.target_asin
            return asin or self.target_asin
        return self.target_asin

    def _extract_product(self, state: SimulationState, asin: str) -> Dict[str, Any]:
        # Attempt to find the product matching ASIN
        for p in state.products or []:
            if (p.get("asin") or p.get("ASIN")) == asin:
                return p
        # If not found, return an empty dict with minimal defaults
        return {"asin": asin}

    def _get_current_price(self, product: Dict[str, Any]) -> Optional[float]:
        price = product.get("price") or product.get("current_price") or product.get("our_price")
        try:
            return float(price) if price is not None else None
        except Exception:
            return None

    def _get_cost(self, product: Dict[str, Any], fallback_ratio: float, fallback_price: Optional[float]) -> float:
        cost = product.get("cost") or product.get("unit_cost") or product.get("COGS")
        try:
            return float(cost) if cost is not None else float(fallback_price or 10.0) * float(fallback_ratio)
        except Exception:
            return float(fallback_price or 10.0) * float(fallback_ratio)

    def _estimate_competitor_price(self, product: Dict[str, Any], default: float) -> float:
        # Search common structures for competitor pricing
        competitors = product.get("competitors") or product.get("offers") or []
        prices: List[float] = []
        for c in competitors:
            # Common keys: price, offer_price, listing_price
            for key in ("price", "offer_price", "listing_price"):
                v = c.get(key)
                if v is not None:
                    try:
                        prices.append(float(v))
                    except Exception:
                        continue
        if prices:
            # Return the lowest visible competitor price
            return max(min(prices), 0.01)
        # Fallback to any known market reference on the product
        market_price = product.get("market_price") or product.get("avg_market_price")
        try:
            if market_price is not None:
                return float(market_price)
        except Exception:
            pass
        return max(float(default), 0.01)

    def _estimate_recent_demand(self, state: SimulationState, asin: str, product: Dict[str, Any]) -> float:
        """
        Estimate recent demand using available signals:
        - Sum of units_sold in recent_events for the ASIN
        - product-level demand field if present
        - else small baseline to avoid zero-division
        """
        # Try recent events
        total = 0.0
        for evt in state.recent_events or []:
            try:
                evt_asin = evt.get("asin") or evt.get("ASIN")
                if evt_asin and evt_asin != asin:
                    continue
                # Common keys that may indicate demand/sales
                for key in ("units_sold", "sales", "demand", "quantity"):
                    if key in evt and evt[key] is not None:
                        total += float(evt[key])
            except Exception:
                continue

        if total > 0.0:
            return total

        # Try product-level demand
        for key in ("demand", "recent_demand", "avg_daily_demand", "sales_velocity"):
            v = product.get(key)
            if v is not None:
                try:
                    v_f = float(v)
                    if v_f > 0:
                        return v_f
                except Exception:
                    continue

        # Baseline small demand to keep heuristic stable
        return 1.0

    def _get_inventory(self, product: Dict[str, Any]) -> int:
        inv = product.get("inventory") or product.get("stock") or product.get("qty_on_hand")
        try:
            return int(inv) if inv is not None else self.inventory_target
        except Exception:
            return self.inventory_target

    def _compute_inventory_ratio(self, inventory: int) -> float:
        # Ratio < 1.0 indicates below target, > 1.0 above target
        target = max(self.inventory_target, 1)
        return float(inventory) / float(target)

    def _compute_demand_factor(self, demand_now: float, demand_avg: float) -> float:
        """
        Positive when demand_now > demand_avg (increase price), negative otherwise.
        Scales modestly to avoid overreaction.
        """
        if demand_avg <= 0:
            return 0.0
        delta = (demand_now - demand_avg) / max(demand_avg, 1e-6)
        # Clamp to avoid runaway
        return float(max(min(delta, 1.0), -1.0)) * 0.5

    def _compute_inventory_factor(self, inventory_ratio: float) -> float:
        """
        If inventory is low (< 0.5 of target), nudge price upward.
        If inventory is high (> 1.2 of target), nudge price downward.
        """
        if inventory_ratio < 0.5:
            return 0.2  # raise price to slow sales
        if inventory_ratio > 1.2:
            return -0.15  # lower price to accelerate sales
        return 0.0

    def _smooth_price(self, target: float, reference_price: float) -> float:
        """
        Limit the change relative to reference_price to +/- max_change_pct.
        """
        if reference_price is None or reference_price <= 0:
            return target
        max_up = reference_price * (1.0 + self.max_change_pct)
        max_down = reference_price * (1.0 - self.max_change_pct)
        return float(min(max(target, max_down), max_up))

    def _compute_confidence(self, current: Optional[float], new: float, demand: float) -> float:
        """
        Higher confidence when demand is strong and adjustment is modest; lower when large swings.
        """
        if new <= 0:
            return 0.5
        if current is None or current <= 0:
            base = 0.7
            swing = 0.0
        else:
            rel_change = abs(new - current) / max(current, 1e-6)
            swing = max(0.0, 1.0 - min(rel_change / (self.max_change_pct + 1e-6), 1.0))
            base = 0.6 + 0.2 * swing
        demand_boost = min(math.log1p(max(demand, 0.0)) / 3.0, 0.3)  # cap
        return float(max(min(base + demand_boost, 0.99), 0.5))

    def _build_reasoning(
        self,
        asin: str,
        current_price: Optional[float],
        competitor_price: float,
        floor: float,
        demand: float,
        inventory: int,
        target: float,
    ) -> str:
        return (
            f"ASIN={asin}; current={current_price}; competitor={competitor_price}; floor={floor}; "
            f"demand={demand}; inventory={inventory}; target={target}. "
            f"Price set to balance competition, margin floor, demand signal, and inventory pressure "
            f"with smoothing constraints."
        )