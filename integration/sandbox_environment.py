from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Dict, List, Optional

from .real_world_adapter import SafetyLevel  # Reuse safety level enum for parity

logger = logging.getLogger(__name__)


class IsolationLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class SandboxConfig:
    isolation_level: IsolationLevel = IsolationLevel.MEDIUM
    enable_api_mocking: bool = True
    enable_real_world_validation: bool = True
    safety_level: SafetyLevel = SafetyLevel.MEDIUM
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_api_calls_per_minute": 100,
        "max_price_change_percent": 0.15,
        "max_order_value": 50000,
    })


class SandboxEnvironment:
    """
    Deterministic sandbox that mirrors the RealWorldAdapter surface required by tests.
    Provides:
      - async initialize()
      - async import_state(state) -> bool
      - async export_state() -> dict (parity for consistency checks)
      - async make_api_call(method, **params) with methods: set_price, get_inventory, place_order
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config: SandboxConfig = config or SandboxConfig()
        self._lock = threading.RLock()
        self._initialized: bool = False

        # Simple in-memory state similar to adapter
        self._catalog: Dict[str, Dict[str, Any]] = {}
        self._api_call_window_ts: float = monotonic()
        self._api_calls_in_window: int = 0

    # ---------- Lifecycle ----------
    async def initialize(self) -> bool:
        with self._lock:
            self._initialized = True
            self._api_call_window_ts = monotonic()
            self._api_calls_in_window = 0
        return True

    # ---------- State sync ----------
    async def import_state(self, state: Dict[str, Any]) -> bool:
        """
        Accepts export_state() output from RealWorldAdapter and hydrates local mirrors.
        """
        try:
            catalog = state.get("catalog", [])
            with self._lock:
                self._catalog.clear()
                for rec in catalog:
                    asin = rec.get("asin")
                    if not asin:
                        continue
                    self._catalog[asin] = dict(rec)
            return True
        except Exception as e:
            logger.error("Sandbox import_state failed: %s", e, exc_info=True)
            return False

    async def export_state(self) -> Dict[str, Any]:
        """
        Provide a state snapshot compatible with consistency checker.
        """
        with self._lock:
            return {
                "mode": "sandbox",
                "catalog": list(self._catalog.values()),
                "products_cached": len(self._catalog),
            }

    # ---------- API facade ----------
    async def make_api_call(self, method: str, **params: Any) -> Dict[str, Any]:
        """
        Supported: set_price, get_inventory, place_order
        Enforces simple resource limits when configured.
        """
        self._rate_limit_check()

        method_lc = (method or "").lower()
        if method_lc == "set_price":
            asin = params.get("asin")
            raw_price = params.get("price")
            if asin is None or raw_price is None:
                raise ValueError("asin and price are required for set_price")

            # Accept cents or dollars like adapter
            price = float(raw_price)
            if price > 1000:  # likely cents
                price = price / 100.0

            # Enforce max price change percent if we have a baseline
            with self._lock:
                prev = self._catalog.get(asin)
                if prev:
                    old_price = float(prev.get("price", 0.0) or 0.0)
                    if old_price > 0:
                        pct_change = abs((price - old_price) / old_price)
                        max_pct = float(self.config.resource_limits.get("max_price_change_percent", 0.15))
                        if pct_change > max_pct:
                            # In sandbox, we simulate blocking by returning failure
                            return {
                                "success": False,
                                "error": "price_change_limit_exceeded",
                                "allowed_max_change": max_pct,
                            }
                # Apply update
                self._catalog[asin] = {
                    "asin": asin,
                    "price": round(price, 2),
                    "inventory": int((prev or {}).get("inventory", 100)),
                    "bsr": int((prev or {}).get("bsr", 5000)),
                    "conversion_rate": float((prev or {}).get("conversion_rate", 0.1)),
                }
            return {"success": True, "asin": asin, "price": round(price, 2)}

        if method_lc == "get_inventory":
            asin = params.get("asin")
            if not asin:
                raise ValueError("asin is required for get_inventory")
            with self._lock:
                rec = self._catalog.get(asin) or {
                    "asin": asin,
                    "price": 25.0,
                    "inventory": 100,
                    "bsr": 5000,
                    "conversion_rate": 0.1,
                }
                self._catalog[asin] = rec
                inv = int(rec.get("inventory", 100))
                return {
                    "asin": asin,
                    "available_quantity": inv,
                    "reserved_quantity": 0,
                    "fulfillable_quantity": inv,
                }

        if method_lc == "place_order":
            asin = params.get("asin")
            quantity = int(params.get("quantity", 0))
            if not asin or quantity <= 0:
                raise ValueError("asin and positive quantity required")
            max_value = float(self.config.resource_limits.get("max_order_value", 50000))
            est_cost = quantity * 15.0
            if est_cost > max_value:
                return {
                    "success": False,
                    "error": "order_value_limit_exceeded",
                    "max_order_value": max_value,
                }
            return {
                "success": True,
                "order_id": f"sbx_order_{asin}_{quantity}",
                "asin": asin,
                "quantity": quantity,
                "estimated_cost": est_cost,
            }

        raise ValueError(f"Unsupported API method: {method}")

    # ---------- Helpers ----------
    def _rate_limit_check(self) -> None:
        max_per_min = int(self.config.resource_limits.get("max_api_calls_per_minute", 100))
        now = monotonic()
        with self._lock:
            window = now - self._api_call_window_ts
            if window >= 60.0:
                self._api_call_window_ts = now
                self._api_calls_in_window = 0
            self._api_calls_in_window += 1
            if self._api_calls_in_window > max_per_min:
                # In sandbox, emulate rate limit exceeded as an error
                raise RuntimeError("sandbox_rate_limit_exceeded")


__all__ = [
    "SandboxEnvironment",
    "SandboxConfig",
    "IsolationLevel",
    "SafetyLevel",
]