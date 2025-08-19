# Baseline Agent V1 — Deterministic Rule-Based Reference

This module provides a lightweight, deterministic, explainable baseline agent suitable for benchmarking and as a reference implementation. It includes a rule engine, optional tools adapter, short-term memory, constraints enforcement, and safety redaction.

Key entrypoints:
- Sync: [`python.def decide()`](agents/baseline/baseline_agent_v1.py:467)
- Async: [`python.async def decide_async()`](agents/baseline/baseline_agent_v1.py:626)

The legacy pricing class is preserved for backward-compat:
- [`python.class BaselineAgentV1`](agents/baseline/baseline_agent_v1.py:22)
  - [`python.def BaselineAgentV1.decide()`](agents/baseline/baseline_agent_v1.py:49)

## Public API

- [`python.def decide(task, context=None)`](agents/baseline/baseline_agent_v1.py:467)
  - task: dict | str
  - context: dict | None

- [`python.async def decide_async(task, context=None)`](agents/baseline/baseline_agent_v1.py:626)
  - Awaitable wrapper around decide for async environments.

Return structure:
```json
{
  "status": "success | failed",
  "output": "<final string>",
  "reasoning": ["...steps..."],
  "applied_rules": ["RULE_X", "RULE_Y"],
  "used_tools": [{"name":"...", "args":{...}, "result":"...", "error":"..."}],
  "metrics": {"duration_ms": 12, "tokens_est": 42},
  "memory": {"recent": [{"task":"...", "output":"..."}]}
}
```

## Capabilities

- Rule engine (priority-ordered, deterministic):
  - RULE_PLAN: multi-step/milestone cues → numbered plan
  - RULE_SUMMARIZE: long or “summarize” prompts → extractive-lite summary
  - RULE_EXTRACT: parse key:value pairs → JSON
  - RULE_COMPUTE: arithmetic expressions → calculator tool or safe local eval
  - RULE_LOOKUP: context lookup_table hit → direct answer
  - RULE_CONSTRAINT_ENFORCER: respects max_output_tokens
  - RULE_SAFETY: redacts disallowed terms

See rule definitions around:
- [`python.list _RULES`](agents/baseline/baseline_agent_v1.py:454)

- Tools adapter:
  - Pass tools via context["tools"] as a list of {"name": str, "callable": callable, ...} or dict name→callable/spec
  - Invocation recording and error capture: see [`python.def _invoke_tool()`](agents/baseline/baseline_agent_v1.py:209)

- Memory:
  - Session-scoped ring buffer (last N=10 by default)
  - Enable with context["enable_memory"]=True and optional context["session_id"]
  - Memory hint retrieval influences reasoning

- Constraints and Safety:
  - context["max_output_tokens"] or constraints["max_output_tokens"]
  - Token estimate heuristic: see [`python.def _estimate_tokens()`](agents/baseline/baseline_agent_v1.py:148)
  - Safety blocklist and redaction: see [`python.def _apply_safety_redaction()`](agents/baseline/baseline_agent_v1.py:247)

- Logging:
  - Uses project logging when available: [`python.def setup_logging()`](fba_bench/core/logging.py:110)

## Context Parameters

- tools: list[{"name": str, "callable": callable, "description": str, "schema": dict|None}] | dict[str, callable|spec]
- max_output_tokens: int
- enable_memory: bool
- session_id: str
- lookup_table: dict[str, str]
- summary_len: int (characters), default ~1200
- max_rule_steps: int, default 6

## Usage Examples

1) Planning
```python
from agents.baseline.baseline_agent_v1 import decide
res = decide("Plan how to research, summarize, and present Q3 results under 200 tokens",
             context={"max_output_tokens": 200})
print(res["output"])
```

2) Extraction
```python
res = decide("Extract name and date from: Name: Jane Doe, Date: 2025-07-04")
# => {"name":"Jane Doe","date":"2025-07-04"}
```

3) Compute with a tool
```python
def calculator(expression: str) -> float:
    expr = expression.replace("^", "**")
    import re
    if not re.fullmatch(r"[0-9\\.\\s\\+\\-\\*\\/\\(\\)\\*]*", expr):
        raise ValueError("bad expr")
    return float(eval(expr, {"__builtins__": {}}, {}))

res = decide("Compute 12*(3+4)",
             context={"tools": [{"name": "calculator", "callable": calculator}]})
# => output: "Result: 84"
```

4) Lookup table
```python
res = decide("capital of france?",
             context={"lookup_table": {"capital of france?": "Paris"}})
# => output: "Paris"
```

5) Memory
```python
ctx = {"enable_memory": True, "session_id": "s1"}
decide("How to prepare a quarterly summary report?", context=ctx)
res = decide("Prepare summary for quarterly results", context=ctx)
# res["reasoning"] includes a hint from recent memory
```

## Determinism and Limits

- Same input/context → identical outputs/reasoning. Rule order is fixed: PLAN → SUMMARIZE → EXTRACT → COMPUTE → LOOKUP → CONSTRAINT_ENFORCER → SAFETY.
- Max rule applications: 6 (configurable via context).
- Conflicting constraints (e.g., max_output_tokens=0) enforce empty output rationally.
- Safety blocklist currently includes: passwords, malware, PII (case-insensitive redaction).

## Testing

Integration tests cover common scenarios:
- Planning, Summarization, Extraction, Compute (with tool), Lookup, Constraints, Safety, Memory, Error paths.

See:
- [`filename tests/integration/test_baseline_agent.py`](tests/integration/test_baseline_agent.py)