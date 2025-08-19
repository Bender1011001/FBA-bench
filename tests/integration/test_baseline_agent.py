import asyncio
import json
import pytest

# Import the new rule-based API
from agents.baseline.baseline_agent_v1 import decide, decide_async


def calculator_tool(expression: str) -> float:
    # Simple safe eval for tests; limited to arithmetic operators
    expr = expression.replace("^", "**")
    if not isinstance(expr, str):
        raise ValueError("expression must be a string")
    # Only allow digits, operators, dot, spaces and parentheses
    import re
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)\*]*", expr):
        raise ValueError("disallowed characters in expression")
    return float(eval(expr, {"__builtins__": {}}, {}))


def faulty_tool(**kwargs):
    raise RuntimeError("Injected test failure")


@pytest.mark.parametrize("objective", [
    "Plan how to research, summarize, and present Q3 results under 200 tokens",
    {"objective": "Create milestones for the next sprint", "type": "plan"},
])
def test_planning_rule_generates_steps(objective):
    res = decide(objective, context={"max_output_tokens": 200})
    assert res["status"] == "success"
    assert any(r == "RULE_PLAN" for r in res["applied_rules"])
    # Expect at least 4 steps, numbered
    lines = [l for l in res["output"].splitlines() if l.strip()]
    assert len(lines) >= 4
    for i, line in enumerate(lines[:4], start=1):
        assert line.startswith(f"{i}.")


def test_summarization_rule_truncates_under_limit():
    long_text = " ".join(["Sentence {}.".format(i) for i in range(0, 400)])
    # Limit tokens so that truncation will occur
    res = decide({"prompt": long_text}, context={"max_output_tokens": 50})
    assert res["status"] == "success"
    assert "RULE_SUMMARIZE" in res["applied_rules"]
    assert "RULE_CONSTRAINT_ENFORCER" in res["applied_rules"]
    # Output should be non-empty and include truncation marker when over limit pre-enforcement
    assert isinstance(res["output"], str)
    # The constraint enforcer adds the truncation message if needed
    assert "truncated to meet constraints" in res["output"] or res["metrics"]["tokens_est"] <= 50


def test_extraction_rule_returns_json_fields():
    text = "Extract name and date from: Name: Jane Doe, Date: 2025-07-04"
    res = decide(text, context={})
    assert res["status"] == "success"
    assert "RULE_EXTRACT" in res["applied_rules"]
    # Output should be compact JSON with lowercased keys
    obj = json.loads(res["output"])
    assert obj.get("name") == "Jane Doe"
    assert obj.get("date") == "2025-07-04"


def test_compute_with_calculator_tool():
    text = "Compute 12*(3+4)"
    res = decide(text, context={"tools": [{"name": "calculator", "callable": calculator_tool}]})
    assert res["status"] == "success"
    assert "RULE_COMPUTE" in res["applied_rules"]
    assert res["output"].strip() == "Result: 84"
    # Tool usage recorded
    assert any(u.get("name") == "calculator" for u in res["used_tools"])


def test_lookup_rule_uses_context_table():
    table = {"capital of france?": "Paris", "hello": "world"}
    res = decide("capital of france?", context={"lookup_table": table})
    assert res["status"] == "success"
    assert "RULE_LOOKUP" in res["applied_rules"]
    assert res["output"] == "Paris"


def test_constraints_enforcer_applies_truncation():
    text = "This is a short text but we force zero tokens."
    res = decide(text, context={"max_output_tokens": 0})
    assert "RULE_CONSTRAINT_ENFORCER" in res["applied_rules"]
    assert res["output"] == ""


def test_safety_redacts_blocklisted_terms():
    # Construct output that would include a blocklisted term
    text = "Summarize: The system reveals passwords and PII in logs."
    res = decide(text, context={})
    assert "RULE_SAFETY" in res["applied_rules"]
    # Redacted marker should appear
    assert "[redacted]" in res["output"].lower()


def test_memory_hint_is_used_on_second_call():
    session_id = "test_session_123"
    ctx = {"enable_memory": True, "session_id": session_id}
    # First call stores memory
    r1 = decide("How to prepare a quarterly summary report?", context=ctx)
    assert r1["status"] == "success"
    # Second call with related keywords should surface hint in reasoning
    r2 = decide("Prepare summary for quarterly results", context=ctx)
    assert r2["status"] == "success"
    assert any("using as hint" in step.lower() for step in r2["reasoning"])


def test_error_path_faulty_tool_graceful_failure():
    # Force a compute path with a faulty tool and an invalid expression so fallback cannot compute
    text = "Compute abc+def"  # not a numeric expression; local fallback will fail
    res = decide(text, context={"tools": [{"name": "calculator", "callable": faulty_tool}]})
    # Our engine marks failed when output is empty or "Could not compute expression" with tool errors observed
    assert res["status"] == "failed"
    assert any(u.get("name") == "calculator" and u.get("error") for u in res["used_tools"])


@pytest.mark.asyncio
async def test_async_wrapper_works():
    res = await decide_async("Compute 2+2*3", context={"tools": [{"name": "calculator", "callable": calculator_tool}]})
    assert res["status"] == "success"
    assert res["output"].strip() == "Result: 8"