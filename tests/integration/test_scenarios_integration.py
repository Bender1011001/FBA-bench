import asyncio
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from benchmarking.scenarios.complex_marketplace import generate_input as cm_generate_input
from benchmarking.scenarios.research_summarization import generate_input as rs_generate_input
from benchmarking.scenarios.multiturn_tool_use import generate_input as mt_generate_input


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complex_marketplace_run_integration():
    # Generate deterministic input
    params = {"num_products": 6, "num_orders": 8, "max_quantity": 3, "price_variance": 0.05, "allow_backorder": False}
    seed = 42
    payload = cm_generate_input(seed=seed, params=params)

    catalog = {p["sku"]: p for p in payload["catalog"]}

    # Define a deterministic runner that:
    # - Accepts only valid SKUs
    # - Enforces quantity <= max_quantity
    # - No backorders: allocate min(stock_available, requested)
    # - Unit price equals catalog price (ignore price_hint)
    def _runner_logic(inp: Dict[str, Any]) -> Dict[str, Any]:
        policies = inp.get("policies", {})
        allow_backorder = bool(policies.get("allow_backorder", False))
        max_q = int(policies.get("max_quantity_per_line", 999999))

        accepted_orders: List[Dict[str, Any]] = []
        rejections: List[Dict[str, Any]] = []
        fulfillment_alloc: Dict[str, Dict[str, int]] = {}
        policy_violations = 0

        # Track remaining stock
        stock = {p["sku"]: int(p["stock"]) for p in inp.get("catalog", [])}

        for order in inp.get("orders", []):
            out_lines = []
            reject_reasons = []
            for line in order.get("lines", []):
                sku = line.get("sku")
                qty = int(line.get("quantity", 0))
                if sku not in stock:
                    policy_violations += 1
                    reject_reasons.append("invalid_sku")
                    continue
                if qty > max_q:
                    policy_violations += 1
                    # Clip to max_q or reject; choose to clip and accept remaining
                    qty = max_q
                # Allocate from stock
                if allow_backorder:
                    alloc = qty
                else:
                    alloc = min(qty, stock[sku])
                if alloc <= 0:
                    reject_reasons.append("out_of_stock")
                    continue
                stock[sku] -= alloc
                price = catalog[sku]["price"]
                out_lines.append({"sku": sku, "quantity": alloc, "unit_price": price})
                fulfillment_alloc[sku] = {"allocated": fulfillment_alloc.get(sku, {}).get("allocated", 0) + alloc}
            if out_lines and not reject_reasons:
                accepted_orders.append({"order_id": order["order_id"], "lines": out_lines})
            elif out_lines:
                # Mixed case: accept the valid subset; still count as accepted
                accepted_orders.append({"order_id": order["order_id"], "lines": out_lines})
            else:
                rejections.append({"order_id": order["order_id"], "reason": ",".join(sorted(set(reject_reasons))) or "invalid"})
        return {
            "accepted_orders": accepted_orders,
            "rejections": rejections,
            "fulfillment": fulfillment_alloc,
            "policy_violations": policy_violations,
        }

    async def runner_callable(inp: Dict[str, Any]) -> Dict[str, Any]:
        return _runner_logic(inp)

    # Import scenario run function
    from benchmarking.scenarios.complex_marketplace import run as cm_run, postprocess as cm_post

    result = await cm_run(payload, runner_callable, timeout_seconds=5)
    result = cm_post(result)

    assert "accepted" in result and isinstance(result["accepted"], int)
    assert "revenue" in result and isinstance(result["revenue"], float)
    assert "fulfilled_rate" in result and 0.0 <= result["fulfilled_rate"] <= 1.0
    assert "policy_violations" in result and isinstance(result["policy_violations"], int)
    # With seed determinism, re-running yields same metrics
    result2 = await cm_run(payload, runner_callable, timeout_seconds=5)
    result2 = cm_post(result2)
    assert result == result2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_summarization_run_integration():
    params = {"num_docs": 4, "max_tokens": 60, "focus_keywords": ["Q3", "revenue"], "noise_probability": 0.1}
    seed = 101
    payload = rs_generate_input(seed=seed, params=params)

    # Deterministic runner that:
    # - extracts key numeric statements and composes a short summary
    # - explicitly mentions focus keywords
    async def runner_callable(inp: Dict[str, Any]) -> Dict[str, Any]:
        docs = inp.get("documents", [])
        fk = inp.get("focus_keywords", [])
        # Build summary by taking first sentences mentioning numbers and keywords
        parts: List[str] = []
        for d in docs:
            abstract = d.get("abstract", "")
            # naive extraction: split; pick sentences with % or 'Q'
            for s in abstract.split(". "):
                if "%" in s or "Q" in s:
                    parts.append(s.strip().rstrip("."))
                    break
        # Ensure keywords are mentioned
        if fk:
            parts.append("Focus: " + ", ".join(fk))
        summary = ". ".join(parts)
        # Enforce max_tokens by truncation
        max_toks = int(inp.get("max_tokens", 200))
        words = summary.split()
        if len(words) > max_toks:
            words = words[:max_toks]
        return {"summary": " ".join(words).strip()}

    from benchmarking.scenarios.research_summarization import run as rs_run

    result = await rs_run(payload, runner_callable, timeout_seconds=5)

    assert isinstance(result["summary"], str) and len(result["summary"]) > 0
    assert isinstance(result["coverage_score"], float)
    assert isinstance(result["length_ok"], bool) and result["length_ok"] is True
    assert isinstance(result["keyword_hits"], int) and result["keyword_hits"] >= 0

    # Deterministic across runs
    result2 = await rs_run(payload, runner_callable, timeout_seconds=5)
    assert result == result2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiturn_tool_use_run_integration_variants():
    # Variant A: math + extraction
    params_a = {"steps": 2, "include_math": True, "include_extraction": True, "include_transform": False}
    seed_a = 7
    payload_a = mt_generate_input(seed=seed_a, params=params_a)

    # Variant B: all three steps
    params_b = {"steps": 3, "include_math": True, "include_extraction": True, "include_transform": True}
    seed_b = 9
    payload_b = mt_generate_input(seed=seed_b, params=params_b)

    async def runner_callable(inp: Dict[str, Any]) -> Dict[str, Any]:
        tasks = inp.get("tasks", [])
        results: List[Dict[str, Any]] = []
        for t in tasks:
            if t["type"] == "math":
                expr = t["expression"]
                a, op, b = expr.split()
                a, b = int(a), int(b)
                if op == "+":
                    res = a + b
                elif op == "-":
                    res = a - b
                else:
                    res = a * b
                results.append({"type": "math", "result": res})
            elif t["type"] == "extraction":
                text = t["text"]
                # Simple parse based on the deterministic template used in generation
                # On {date}, {name} paid ${amount} for invoice {invoice} from account {account} in {region}.
                try:
                    before, after = text.split(" paid $", 1)
                    date = before.split("On ")[1].strip()
                    amount = after.split(" for invoice ")[0].strip()
                    invoice = after.split(" for invoice ")[1].split(" from account ")[0].strip()
                    account = after.split(" from account ")[1].split(" in ")[0].strip()
                    region = after.split(" in ")[1].split(".")[0].strip()
                    name = before.split(", ")[1].strip()
                except Exception:
                    name = date = amount = invoice = account = region = ""
                results.append(
                    {
                        "type": "extraction",
                        "result": {
                            "name": name,
                            "date": date,
                            "amount": amount,
                            "account": account,
                            "invoice": invoice,
                            "region": region,
                        },
                    }
                )
            elif t["type"] == "transform":
                data = t["data"]
                items = list(data["items"])
                multiplier = int(data["multiplier"])
                offset = int(data["offset"])
                transformed = [i * multiplier + offset for i in items]
                results.append({"type": "transform", "result": {"sum": sum(items), "transformed": transformed}})
        return {"results": results}

    from benchmarking.scenarios.multiturn_tool_use import run as mt_run

    res_a = await mt_run(payload_a, runner_callable, timeout_seconds=5)
    assert res_a["steps_completed"] == res_a["total"]
    assert res_a["correct"] == res_a["total"]
    assert abs(res_a["score"] - 1.0) < 1e-9

    res_b = await mt_run(payload_b, runner_callable, timeout_seconds=5)
    assert res_b["steps_completed"] == res_b["total"]
    assert res_b["correct"] == res_b["total"]
    assert abs(res_b["score"] - 1.0) < 1e-9