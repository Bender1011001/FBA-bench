from __future__ import annotations

"""
Complex Marketplace Scenario

Implements a multi-step order workflow with inventory, pricing, and fulfillment logic.

Public interface exposed by this module:
- [python.class ComplexMarketplaceConfig(BaseModel)](benchmarking/scenarios/complex_marketplace.py:1)
- [python.def generate_input(seed: int|None, params: dict|None) -> dict](benchmarking/scenarios/complex_marketplace.py:1)
- [python.async def run(input_payload: dict, runner_callable: Callable[[dict], Awaitable[dict]], timeout_seconds: int|None=None) -> dict](benchmarking/scenarios/complex_marketplace.py:1)
- [python.def postprocess(raw_output: dict) -> dict](benchmarking/scenarios/complex_marketplace.py:1)

Registration:
- Registers itself under key "complex_marketplace" via the global ScenarioRegistry.
"""

import math
import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

from .registry import scenario_registry

# Ensure stable decimal arithmetic
getcontext().prec = 28
ROUND_CTX = ROUND_HALF_UP


def _rnd(seed: Optional[int]) -> random.Random:
    r = random.Random()
    if seed is not None:
        r.seed(seed)
    return r


def _decimal(value: float | int | str, q: str = "0.01") -> Decimal:
    return (Decimal(str(value))).quantize(Decimal(q), rounding=ROUND_CTX)


def _safe_round(value: Decimal | float, ndigits: int = 2) -> float:
    if isinstance(value, Decimal):
        return float(value.quantize(Decimal("1." + "0" * ndigits), rounding=ROUND_CTX))
    return round(float(value), ndigits)


class ComplexMarketplaceConfig(BaseModel):
    """
    Configuration schema (Pydantic v2) for Complex Marketplace Scenario.
    """

    num_products: int = Field(20, ge=1, le=500, description="Number of unique products in the catalog")
    num_orders: int = Field(50, ge=1, le=5000, description="Number of orders to synthesize")
    max_quantity: int = Field(5, ge=1, le=100, description="Maximum quantity per order line")
    price_variance: float = Field(
        0.1, ge=0.0, le=1.0, description="Max fractional deviation around base price for price perturbation"
    )
    allow_backorder: bool = Field(False, description="Whether orders can be accepted beyond available stock")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "num_products": 10,
                    "num_orders": 25,
                    "max_quantity": 4,
                    "price_variance": 0.15,
                    "allow_backorder": False,
                }
            ]
        }
    }

    @field_validator("price_variance")
    @classmethod
    def _check_price_variance(cls, v: float) -> float:
        # Already bounded by Field, but keep explicit validator for clarity and error messaging
        if not (0.0 <= v <= 1.0):
            raise ValueError("price_variance must be within [0.0, 1.0]")
        return v


def _synthesize_catalog(
    r: random.Random, num_products: int, price_variance: float
) -> List[Dict[str, Any]]:
    """
    Create a deterministic product catalog with base prices, price deltas and stock.
    """
    catalog: List[Dict[str, Any]] = []
    for i in range(num_products):
        base_price = _decimal(5 + r.random() * 95)  # $5 - $100
        # Apply a small deterministic price variance using a symmetric multiplier
        variance = 1 + (r.uniform(-price_variance, price_variance))
        price = _decimal(float(base_price) * variance)
        stock = int(r.randint(10, 200))
        catalog.append(
            {
                "sku": f"P{i+1:04d}",
                "base_price": float(base_price),
                "price": float(price),
                "stock": stock,
            }
        )
    return catalog


def _synthesize_orders(
    r: random.Random,
    catalog: List[Dict[str, Any]],
    num_orders: int,
    max_quantity: int,
) -> List[Dict[str, Any]]:
    """
    Create deterministic batch of orders with realistic noise.
    - Randomly pick products
    - Random quantities
    - Occasional invalid SKU or excessive quantity to test policy handling
    """
    orders: List[Dict[str, Any]] = []
    skus = [p["sku"] for p in catalog]

    for i in range(num_orders):
        # Sometimes create multi-line orders (1-3 lines)
        lines = []
        for _ in range(1, r.choice([1, 1, 2, 3]) + 1):
            if r.random() < 0.05:
                # 5% invalid SKU to test runner validation
                sku = f"X{r.randint(1000, 9999)}"
            else:
                sku = r.choice(skus)

            # 10% generate quantity above max to test policy handling
            if r.random() < 0.1:
                qty = max_quantity + r.randint(1, 5)
            else:
                qty = r.randint(1, max_quantity)

            # Optional client price hint; sometimes slightly off to test final pricing logic
            if r.random() < 0.3:
                # pick near product's listed price, +/- up to 5%
                ref_price = None
                for p in catalog:
                    if p["sku"] == sku:
                        ref_price = p["price"]
                        break
                if ref_price is None:
                    # invalid SKU => random price hint
                    price_hint = _safe_round(_decimal(1 + r.random() * 150))
                else:
                    price_hint = _safe_round(_decimal(float(ref_price) * (1 + r.uniform(-0.05, 0.05))))
            else:
                price_hint = None

            lines.append({"sku": sku, "quantity": qty, "price_hint": price_hint})

        orders.append({"order_id": f"O{i+1:06d}", "lines": lines})

    return orders


def generate_input(seed: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Deterministically generate the input payload for the scenario based on seed and params.
    """
    params = params or {}
    try:
        cfg = ComplexMarketplaceConfig(**params)
    except ValidationError as e:
        # Re-raise as ValueError for engines that expect standard exception
        raise ValueError(str(e)) from e

    r = _rnd(seed)
    catalog = _synthesize_catalog(r, cfg.num_products, cfg.price_variance)
    orders = _synthesize_orders(r, catalog, cfg.num_orders, cfg.max_quantity)

    payload = {
        "config": cfg.model_dump(),
        "seed": seed,
        "catalog": catalog,
        "orders": orders,
        "policies": {
            "allow_backorder": cfg.allow_backorder,
            "max_quantity_per_line": cfg.max_quantity,
        },
        "expected_outputs": {
            # The scenario is runner-agnostic; expected outputs are left for evaluation logic,
            # not to enforce exact behavior, but to provide deterministic parameters.
        },
        "task": "Compute accepted orders, final prices, and fulfillment plan respecting stock and policies.",
    }
    return payload


async def run(
    input_payload: Dict[str, Any],
    runner_callable: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute the scenario by calling the runner and evaluating its response.

    Runner contract:
    - Input: dict with keys 'catalog', 'orders', 'policies'
    - Output expected fields (dict):
      {
        "accepted_orders": list[{"order_id": str, "lines": [{"sku": str, "quantity": int, "unit_price": float}]}],
        "rejections": list[{"order_id": str, "reason": str}],
        "fulfillment": dict[sku -> {"allocated": int}],
        "policy_violations": int
      }
    The scenario will compute derived KPIs: accepted count, revenue, fulfilled rate, policy_violations.
    """
    # Engine will enforce timeout via asyncio.wait_for around this call; we only pass through timeout hint
    runner_input = {
        "catalog": input_payload.get("catalog", []),
        "orders": input_payload.get("orders", []),
        "policies": input_payload.get("policies", {}),
        "seed": input_payload.get("seed"),
        "instructions": (
            "You are an order processing system. Validate SKUs, enforce quantity limits, "
            "apply product prices to each accepted line (override price_hint if needed), "
            "respect allow_backorder policy, generate fulfillment allocations by SKU, "
            "and count policy_violations for invalid SKUs or quantity breaches."
        ),
    }

    # Invoke runner
    raw = await runner_callable(runner_input)

    # Validate/normalize runner output shape
    accepted_orders: List[Dict[str, Any]] = list(raw.get("accepted_orders", []))
    rejections: List[Dict[str, Any]] = list(raw.get("rejections", []))
    fulfillment: Dict[str, Dict[str, Any]] = dict(raw.get("fulfillment", {}))
    policy_violations: int = int(raw.get("policy_violations", 0))

    # Compute revenue and fulfillment KPIs deterministically
    revenue = Decimal("0.00")
    total_requested_by_sku: Dict[str, int] = {}
    allocated_by_sku: Dict[str, int] = {}

    # Sum revenue from accepted orders
    for order in accepted_orders:
        for line in order.get("lines", []):
            qty = int(line.get("quantity", 0))
            unit_price = _decimal(line.get("unit_price", 0.0))
            revenue += unit_price * qty

            sku = str(line.get("sku"))
            total_requested_by_sku[sku] = total_requested_by_sku.get(sku, 0) + qty

    # Aggregate allocated units from fulfillment
    for sku, alloc in fulfillment.items():
        allocated_by_sku[sku] = int(alloc.get("allocated", 0))

    # Compute fulfilled rate across all SKUs present in accepted orders
    total_requested = sum(total_requested_by_sku.values())
    total_allocated = 0
    for sku, req in total_requested_by_sku.items():
        total_allocated += min(req, allocated_by_sku.get(sku, 0))

    fulfilled_rate = 1.0 if total_requested == 0 else (total_allocated / total_requested)

    result = {
        "accepted": int(len(accepted_orders)),
        "revenue": _safe_round(revenue, 2),
        "fulfilled_rate": float(_decimal(fulfilled_rate).quantize(Decimal("0.0001"), rounding=ROUND_CTX)),
        "policy_violations": int(policy_violations),
        "details": {
            "accepted_orders": accepted_orders,
            "rejections": rejections,
            "fulfillment": fulfillment,
            "totals": {
                "total_requested": int(total_requested),
                "total_allocated": int(total_allocated),
            },
        },
    }
    return result


def postprocess(raw_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize floats/roundings to ensure stable outputs across platforms.
    """
    out = dict(raw_output)
    # Ensure revenue rounded to 2 decimals, fulfilled_rate to 4 decimals
    if "revenue" in out:
        out["revenue"] = _safe_round(Decimal(str(out["revenue"])), 2)
    if "fulfilled_rate" in out:
        fr = Decimal(str(out["fulfilled_rate"])).quantize(Decimal("0.0001"), rounding=ROUND_CTX)
        out["fulfilled_rate"] = float(fr)
    return out


# Register with the scenario registry under the key "complex_marketplace".
# The registry stores classes/callables; we register the module-level API via a lightweight adapter class.
@dataclass
class ComplexMarketplaceScenarioAdapter:
    """
    Adapter to present module-level functions as a class-like callable for registries/engines
    that expect a class or callable. Engines can introspect attributes or call methods directly.
    """

    Config = ComplexMarketplaceConfig

    @staticmethod
    def generate_input(seed: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return generate_input(seed=seed, params=params)

    @staticmethod
    async def run(
        input_payload: Dict[str, Any],
        runner_callable: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await globals()["run"](input_payload, runner_callable, timeout_seconds)

    @staticmethod
    def postprocess(raw_output: Dict[str, Any]) -> Dict[str, Any]:
        return postprocess(raw_output)


# Perform registration at import time.
scenario_registry.register("complex_marketplace", ComplexMarketplaceScenarioAdapter)