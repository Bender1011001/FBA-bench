from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from time import monotonic
from typing import Any, Dict, Optional, List


class OperationalMode(str, Enum):
    SIMULATION = "simulation"
    SANDBOX = "sandbox"
    STAGING = "staging"
    PRODUCTION = "production"


# Backwards-compat aliases expected by tests
# tests import: from integration.real_world_adapter import AdapterMode, SafetyLevel
AdapterMode = OperationalMode


class SafetyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class SafetyConfig:
    # Simple guardrails for price changes and actions
    max_daily_price_change_percent: float = 20.0
    manual_approval_threshold_usd: float = 10000.0
    dry_run_mode: bool = True


@dataclass(frozen=True)
class SyncConfig:
    # Synchronization settings (no-op in our in-memory adapter)
    sync_frequency_seconds: int = 300
    sync_on_actions: bool = True
    conflict_resolution: str = "real_world_wins"


@dataclass(frozen=True)
class MarketplaceConfig:
    # Minimal marketplace configuration stub
    platform: str = "amazon"
    credentials: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegrationConfig:
    mode: OperationalMode = OperationalMode.SIMULATION
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: float = 5.0
    max_retries: int = 0

    # Extended configs referenced by tests
    safety_config: SafetyConfig = field(default_factory=SafetyConfig)
    sync_config: SyncConfig = field(default_factory=SyncConfig)
    marketplace_config: MarketplaceConfig = field(default_factory=MarketplaceConfig)

    def __post_init__(self) -> None:
        # Validate that timeout and retries are sensible
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

    @classmethod
    def from_yaml(cls, path: str) -> "IntegrationConfig":
        # Lightweight loader to satisfy docs/tests that may reference this API.
        # We accept JSON superset for simplicity if provided; otherwise return defaults.
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                # Fallback to defaults if file missing/invalid
                data = {}

        return cls(
            mode=_coerce_mode(data.get("mode", cls.mode)),
            endpoint=data.get("endpoint"),
            api_key=data.get("api_key"),
            timeout_seconds=float(data.get("timeout_seconds", 5.0)),
            max_retries=int(data.get("max_retries", 0)),
            safety_config=SafetyConfig(**(data.get("safety_config") or {})),
            sync_config=SyncConfig(**(data.get("sync_config") or {})),
            marketplace_config=MarketplaceConfig(**(data.get("marketplace_config") or {})),
        )


def _coerce_mode(value: Any) -> OperationalMode:
    if isinstance(value, OperationalMode):
        return value
    if isinstance(value, str):
        try:
            return OperationalMode(value.lower())
        except Exception as ex:
            raise ValueError(f"Invalid OperationalMode: {value}") from ex
    raise ValueError(f"Invalid OperationalMode type: {type(value).__name__}")


@dataclass
class ProductRecord:
    asin: str
    price: float
    inventory: int
    bsr: int
    conversion_rate: float
    last_updated_ts: float = field(default_factory=monotonic)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "asin": self.asin,
            "price": round(self.price, 2),
            "inventory": int(self.inventory),
            "bsr": int(self.bsr),
            "conversion_rate": round(self.conversion_rate, 4),
            "last_updated_ts": self.last_updated_ts,
        }


class RealWorldAdapter:
    """Deterministic, production-safe adapter abstraction for integration tests.

    Features:
    - In-memory deterministic product catalog generation seeded by ASIN.
    - Thread-safe price updates and reads.
    - Clear lifecycle: async initialize()/cleanup() and sync connect()/close() helpers.
    - Safety checks for price updates (basic guardrails only).
    """

    def __init__(self, config: Optional[IntegrationConfig | Dict[str, Any]] = None) -> None:
        # Accept dict-like config for test convenience
        if config is None:
            parsed = IntegrationConfig()
        elif isinstance(config, IntegrationConfig):
            parsed = IntegrationConfig(
                mode=_coerce_mode(config.mode),
                endpoint=config.endpoint,
                api_key=config.api_key,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
                safety_config=config.safety_config,
                sync_config=config.sync_config,
                marketplace_config=config.marketplace_config,
            )
        elif isinstance(config, dict):
            safety_raw = config.get("safety_config") or {}
            sync_raw = config.get("sync_config") or {}
            market_raw = config.get("marketplace_config") or {}
            parsed = IntegrationConfig(
                mode=_coerce_mode(config.get("mode", OperationalMode.SIMULATION)),
                endpoint=config.get("endpoint"),
                api_key=config.get("api_key"),
                timeout_seconds=float(config.get("timeout_seconds", 5.0)),
                max_retries=int(config.get("max_retries", 0)),
                safety_config=SafetyConfig(**safety_raw) if isinstance(safety_raw, dict) else SafetyConfig(),
                sync_config=SyncConfig(**sync_raw) if isinstance(sync_raw, dict) else SyncConfig(),
                marketplace_config=MarketplaceConfig(**market_raw) if isinstance(market_raw, dict) else MarketplaceConfig(),
            )
        else:
            raise TypeError("config must be None, IntegrationConfig, or dict")

        self.config: IntegrationConfig = parsed

        self._connected: bool = False
        self._initialized: bool = False
        self._lock = threading.RLock()
        self._catalog: Dict[str, ProductRecord] = {}
        self._price_history: Dict[str, List[Dict[str, Any]]] = {}

    # ---- Async lifecycle (preferred by tests) --------------------------------

    async def initialize(self) -> bool:
        with self._lock:
            self._connected = True
            self._initialized = True
        return True

    async def cleanup(self) -> None:
        with self._lock:
            self._connected = False
            self._initialized = False

    # ---- Sync helpers (for compatibility) ------------------------------------

    def connect(self) -> None:
        with self._lock:
            self._connected = True
            self._initialized = True

    def close(self) -> None:
        with self._lock:
            self._connected = False
            self._initialized = False

    # ---- Status / Health ------------------------------------------------------

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "connected": self._connected,
                "initialized": self._initialized,
                "mode": self.config.mode.value,
                "endpoint": self.config.endpoint,
                "products_cached": len(self._catalog),
            }

    def health(self) -> Dict[str, Any]:
        with self._lock:
            status = "ok" if self._connected else "disconnected"
            return {
                "status": status,
                "connected": self._connected,
                "initialized": self._initialized,
                "mode": self.config.mode.value,
                "timeout_seconds": self.config.timeout_seconds,
                "dry_run_mode": bool(self.config.safety_config.dry_run_mode),
            }

    # ---- Catalog Operations ---------------------------------------------------

    def fetch_product_data(self, asin: str) -> Dict[str, Any]:
        """Fetch product data deterministically for a given ASIN.

        If the ASIN is not present, create a deterministic record seeded by the ASIN.
        """
        if not asin or not isinstance(asin, str):
            raise ValueError("asin must be a non-empty string")

        with self._lock:
            self._ensure_connected()

            if asin not in self._catalog:
                self._catalog[asin] = self._create_seeded_record(asin)

            record = self._catalog[asin]
            return record.to_public_dict()

    def submit_price(self, agent_id: str, asin: str, new_price: float) -> Dict[str, Any]:
        """Submit a new price for an ASIN, updating the catalog and recording history."""
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")
        if not asin or not isinstance(asin, str):
            raise ValueError("asin must be a non-empty string")
        if not isinstance(new_price, (int, float)) or new_price <= 0:
            raise ValueError("new_price must be a positive number")

        with self._lock:
            self._ensure_connected()
            if asin not in self._catalog:
                self._catalog[asin] = self._create_seeded_record(asin)

            record = self._catalog[asin]
            old_price = record.price

            # Safety: limit daily price change percent
            max_pct = float(self.config.safety_config.max_daily_price_change_percent)
            if old_price > 0:
                pct_change = abs((float(new_price) - old_price) / old_price) * 100.0
                if pct_change > max_pct and not self.config.safety_config.dry_run_mode:
                    raise ValueError(
                        f"Price change {pct_change:.2f}% exceeds safety limit {max_pct:.2f}%"
                    )

            # Apply update
            record.price = float(new_price)
            record.last_updated_ts = monotonic()

            hist = self._price_history.setdefault(asin, [])
            hist.append(
                {
                    "ts": record.last_updated_ts,
                    "agent_id": agent_id,
                    "old_price": round(old_price, 2),
                    "new_price": round(record.price, 2),
                }
            )

            return {
                "ok": True,
                "asin": asin,
                "agent_id": agent_id,
                "old_price": round(old_price, 2),
                "new_price": round(record.price, 2),
                "last_updated_ts": record.last_updated_ts,
            }

    def get_price_history(self, asin: str) -> List[Dict[str, Any]]:
        if not asin or not isinstance(asin, str):
            raise ValueError("asin must be a non-empty string")
        with self._lock:
            return list(self._price_history.get(asin, []))

    # ---- Async utilities used by tests -----------------------------------------
    async def export_state(self) -> Dict[str, Any]:
        """
        Export a lightweight, deterministic state snapshot for synchronization tests.
        """
        with self._lock:
            catalog_state = [rec.to_public_dict() for rec in self._catalog.values()]
            price_events = sum(len(v) for v in self._price_history.values())
            return {
                "mode": self.config.mode.value,
                "catalog": catalog_state,
                "price_event_count": price_events,
                "products_cached": len(self._catalog),
            }

    async def make_api_call(self, method: str, **params: Any) -> Dict[str, Any]:
        """
        Minimal API facade to align with tests' expectations.
        Supported methods: set_price, get_inventory, place_order
        """
        method_lc = (method or "").lower()
        if method_lc == "set_price":
            asin = params.get("asin")
            price = params.get("price")
            agent_id = params.get("agent_id", "adapter")
            # Accept cents or dollars
            if isinstance(price, (int, float)) and price > 1000:  # likely cents
                price = float(price) / 100.0
            result = self.submit_price(agent_id=agent_id, asin=asin, new_price=float(price))
            return {"success": True, "asin": result["asin"], "price": result["new_price"]}
        if method_lc == "get_inventory":
            asin = params.get("asin")
            rec = self.fetch_product_data(asin)
            return {
                "asin": asin,
                "available_quantity": rec["inventory"],
                "reserved_quantity": 0,
                "fulfillable_quantity": rec["inventory"],
            }
        if method_lc == "place_order":
            asin = params.get("asin")
            quantity = int(params.get("quantity", 0))
            if quantity <= 0:
                raise ValueError("quantity must be positive")
            return {
                "success": True,
                "order_id": f"order_{asin}_{quantity}",
                "asin": asin,
                "quantity": quantity,
                "estimated_cost": quantity * 15.0,
            }
        raise ValueError(f"Unsupported API method: {method}")

    # ---- Internal Utilities ---------------------------------------------------

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("RealWorldAdapter is not connected. Call initialize()/connect() first.")

    @staticmethod
    def _create_seeded_record(asin: str) -> ProductRecord:
        """Create a deterministic ProductRecord based on ASIN."""
        # Deterministic hash -> pseudo-random but stable parameters
        seed_bytes = hashlib.sha256(asin.encode("utf-8")).digest()
        # Map bytes to reasonable ranges
        price_base = 5 + (seed_bytes[0] / 255) * 95  # $5 .. $100
        inventory = 10 + int((seed_bytes[1] / 255) * 490)  # 10 .. 500
        bsr = 1 + int((seed_bytes[2] / 255) * 99999)  # rank 1 .. 100000
        conversion_rate = 0.01 + (seed_bytes[3] / 255) * 0.39  # 0.01 .. 0.40

        return ProductRecord(
            asin=asin,
            price=round(price_base, 2),
            inventory=inventory,
            bsr=bsr,
            conversion_rate=round(conversion_rate, 4),
        )


__all__ = [
    "OperationalMode",
    "AdapterMode",
    "SafetyLevel",
    "IntegrationConfig",
    "RealWorldAdapter",
    "ProductRecord",
    "SafetyConfig",
    "SyncConfig",
    "MarketplaceConfig",
]
