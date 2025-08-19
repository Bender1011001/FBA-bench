import json
import pytest

from typing import Any, Dict

from fba_bench.core.llm_outputs import FbaDecision, PriceDecision, TaskPlan, ToolCall, AgentResponse
from fba_bench.core.llm_validation import (
    get_schema,
    validate_output,
    validate_by_name,
    validate_with_jsonschema,
    CONTRACT_REGISTRY,
)


def _mk_fba_decision_payload(price: Any = 23.5, extra: bool = False) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "asin": "B07XAMPLE",
        "new_price": price,
        "reasoning": "Adjusting price to remain competitive.",
    }
    if extra:
        item["unexpected"] = "xyz"
    return {
        "pricing_decisions": [item],
        "meta": {"confidence": 0.8},
    }


def test_registry_contains_expected_contracts():
    assert "fba_decision" in CONTRACT_REGISTRY
    assert "task_plan" in CONTRACT_REGISTRY
    assert "tool_call" in CONTRACT_REGISTRY
    assert "agent_response" in CONTRACT_REGISTRY
    assert CONTRACT_REGISTRY["fba_decision"] is FbaDecision


def test_get_schema_returns_dict_with_properties():
    schema = get_schema(FbaDecision)
    assert isinstance(schema, dict)
    assert "properties" in schema or "$schema" in schema


# ----------------------- Positive cases ---------------------------------------


def test_validate_fba_decision_strict_success_from_dict():
    ok, instance, errors = validate_output(FbaDecision, _mk_fba_decision_payload(), strict=True)
    assert ok is True
    assert instance is not None
    assert errors == []
    assert isinstance(instance, FbaDecision)
    assert isinstance(instance.pricing_decisions[0], PriceDecision)


def test_validate_fba_decision_strict_success_from_json_string():
    payload = json.dumps(_mk_fba_decision_payload())
    ok, instance, errors = validate_output(FbaDecision, payload, strict=True)
    assert ok is True
    assert instance is not None
    assert errors == []
    assert instance.pricing_decisions[0].new_price == 23.5


def test_validate_task_plan_positive():
    payload = {
        "objective": "Increase market share",
        "steps": ["Analyze competitors", "Lower price", "Monitor outcome"],
        "constraints": ["Budget <= 1000"],
        "metadata": {"priority": "high"},
    }
    ok, instance, errors = validate_output(TaskPlan, payload, strict=True)
    assert ok is True
    assert instance is not None
    assert errors == []
    assert instance.objective == "Increase market share"
    assert instance.steps and isinstance(instance.steps[0], str)


def test_validate_tool_call_positive():
    payload = {"tool_name": "set_price", "arguments": {"asin": "B07XAMPLE", "price": 23.1}}
    ok, instance, errors = validate_output(ToolCall, payload, strict=True)
    assert ok and instance is not None and errors == []
    assert instance.tool_name == "set_price"
    assert "asin" in instance.arguments


def test_validate_agent_response_positive():
    payload = {
        "content": "Lowering price slightly to improve competitiveness.",
        "citations": ["https://example.com/market-report"],
        "tool_calls": [{"tool_name": "set_price", "arguments": {"asin": "B07XAMPLE", "price": 23.47}}],
        "plan": {"objective": "Improve competitiveness", "steps": ["Analyze", "Adjust", "Observe"]},
    }
    ok, instance, errors = validate_output(AgentResponse, payload, strict=True)
    assert ok and instance is not None and errors == []
    assert instance.plan and instance.plan.objective.startswith("Improve")


# ----------------------- Negative cases ---------------------------------------


def test_missing_required_field_fails():
    bad = {"pricing_decisions": [{"asin": "B07XAMPLE", "reasoning": "missing price"}]}
    ok, instance, errors = validate_output(FbaDecision, bad, strict=True)
    assert not ok and instance is None
    # Expect an error pointing at pricing_decisions/0/new_price
    assert any("pricing_decisions" in e.get("loc", "") for e in errors)


def test_extra_fields_in_strict_mode_fail():
    payload = _mk_fba_decision_payload(extra=True)
    ok, instance, errors = validate_output(FbaDecision, payload, strict=True)
    assert not ok and instance is None
    # Ensure "unexpected" field flagged
    assert any("unexpected" in e.get("msg", "") or "extra" in e.get("type", "") for e in errors)


def test_type_mismatch_steps_not_list_fails():
    bad = {"objective": "Improve", "steps": "not a list"}
    ok, instance, errors = validate_output(TaskPlan, bad, strict=True)
    assert not ok and instance is None
    assert any("steps" in e.get("loc", "") for e in errors)


# ----------------------- Non-strict coercion ----------------------------------


def test_non_strict_coercion_numeric_string_to_float_succeeds():
    payload = _mk_fba_decision_payload(price="23.47")
    ok, instance, errors = validate_output(FbaDecision, payload, strict=False)
    assert ok and instance is not None and errors == []
    assert isinstance(instance.pricing_decisions[0].new_price, float)
    assert instance.pricing_decisions[0].new_price == pytest.approx(23.47, rel=1e-6)


def test_non_strict_ignores_unknown_fields():
    payload = _mk_fba_decision_payload(extra=True)
    ok, instance, errors = validate_output(FbaDecision, payload, strict=False)
    assert ok and instance is not None and errors == []
    # unknown field should be stripped; model dump should not contain it
    dumped = instance.model_dump()
    assert "unexpected" not in dumped["pricing_decisions"][0]


# ----------------------- validate_by_name -------------------------------------


def test_validate_by_name_success_and_dump():
    payload = _mk_fba_decision_payload()
    ok, data, errors = validate_by_name("fba_decision", payload, strict=True)
    assert ok and data is not None and errors == []
    assert isinstance(data, dict)
    assert "pricing_decisions" in data


def test_validate_by_name_unknown_contract():
    ok, data, errors = validate_by_name("nonexistent_contract", {}, strict=True)
    assert not ok and data is None
    assert errors and errors[0]["type"] == "unknown_contract"


# ----------------------- jsonschema path (optional) ---------------------------


def test_validate_with_jsonschema_optional():
    try:
        import jsonschema  # noqa: F401
        has_jsonschema = True
    except Exception:
        has_jsonschema = False

    # Simple schema: require a field 'a' number >= 0
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number", "minimum": 0}},
        "required": ["a"],
        "additionalProperties": False,
    }

    if not has_jsonschema:
        errors = validate_with_jsonschema(schema, {"a": 1})
        assert errors and "not available" in errors[0]["message"]
        pytest.skip("jsonschema not installed; skipping strict jsonschema validation tests")

    # Positive
    errors = validate_with_jsonschema(schema, {"a": 1})
    assert errors == []

    # Negative: missing required
    errors = validate_with_jsonschema(schema, {})
    assert errors and errors[0]["validator"] in ("required", "internal", "unknown")

    # Negative: type mismatch
    errors = validate_with_jsonschema(schema, {"a": -1})
    assert errors and errors[0]["validator"] in ("minimum", "internal", "unknown")