# Runner Integrations: CrewAI and LangChain

This document describes the production-ready runner implementations for CrewAI and LangChain. Both runners conform to the unified runner API and expose a standard async run(task_input: dict) method that returns a normalized, framework-agnostic result.

Key features:
- Soft dependencies (optional imports) with clear NotInstalled guidance
- Pydantic v2 config schemas and per-run input validation
- Normalized results: status, output, steps, tool_calls, metrics
- Tool adapters: accept callables or dict descriptors, wrapped for framework use
- Robust logging and optional Redis progress events (guarded by REDIS_URL)

Files:
- [agent_runners/crewai_runner.py](agent_runners/crewai_runner.py)
- [agent_runners/langchain_runner.py](agent_runners/langchain_runner.py)
- [agent_runners/registry.py](agent_runners/registry.py)

Base interfaces:
- [agent_runners/base_runner.py](agent_runners/base_runner.py)

Optional Redis client used for progress publishing:
- [fba_bench_api/core/redis_client.py](fba_bench_api/core/redis_client.py)

## 1. Unified Result Shape

Both runners return the following structure from run(task_input):

{
  "status": "success" | "failed",
  "output": str,                       # Final agent message or JSON
  "steps": [                           # Conversational trace
    {"role": "system"|"user"|"assistant", "content": str, "tool_call": optional_dict}
  ],
  "tool_calls": [                      # If available
    {"name": str, "args": any, "result": any}
  ],
  "metrics": {
    "duration_ms": int,
    "token_usage": optional_dict
  }
}

## 2. CrewAI Runner

Class: CrewAIRunner
File: [agent_runners/crewai_runner.py](agent_runners/crewai_runner.py)

Soft dependency behavior:
- On import failure, initialization raises AgentRunnerInitializationError with install hint:
  pip install "crewai>=0.28"

Configuration schema (Pydantic v2):

class CrewAIRunnerConfig(BaseModel):
  model: Optional[str]
  temperature: Optional[float] = 0.3
  max_steps: Optional[int] = 5
  tools: Optional[List[ToolSpec|callable|dict]]
  memory: Optional[bool] = False
  system_prompt: Optional[str] = "You are an FBA pricing expert. Provide JSON-only outputs."
  agent_name: Optional[str] = "CrewAI Pricing Agent"
  allow_delegation: Optional[bool] = False

Per-run input schema:

class CrewAITaskInput(BaseModel):
  prompt: Optional[str]
  products: Optional[List[dict]]
  market_conditions: Optional[dict]
  recent_events: Optional[List[dict]]
  tick: Optional[int] = 0
  tools: Optional[List[ToolSpec|callable|dict]]
  extra: Optional[dict]

ToolSpec (accepted for both runners):

class ToolSpec(BaseModel):
  name: Optional[str]
  description: Optional[str]
  schema: Optional[dict]
  callable: Optional[callable]

Example usage:

from agent_runners.crewai_runner import CrewAIRunner

runner = CrewAIRunner("agent-crew", {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "system_prompt": "Only JSON.",
})
result = await runner.run({
    "prompt": "Make pricing decisions for these products.",
    "products": [{"asin": "B0TEST", "current_price": 19.99, "cost": 10.0}],
    "tick": 1
})
print(result["status"], result["output"])

## 3. LangChain Runner

Class: LangChainRunner
File: [agent_runners/langchain_runner.py](agent_runners/langchain_runner.py)

Soft dependency behavior:
- On import failure, initialization raises AgentRunnerInitializationError with install hint:
  pip install "langchain>=0.3" "langchain-openai>=0.2"

Configuration schema (Pydantic v2):

class LangChainRunnerConfig(BaseModel):
  model: Optional[str] = "gpt-4o-mini"
  temperature: Optional[float] = 0.3
  max_tokens: Optional[int] = 2048
  tools: Optional[List[ToolSpec|callable|dict]]
  memory: Optional[bool] = False
  system_prompt: Optional[str] = "You are an FBA pricing expert. Provide JSON-only outputs."
  agent_name: Optional[str] = "LangChain Pricing Agent"

Per-run input schema:

class LangChainTaskInput(BaseModel):
  prompt: Optional[str]
  products: Optional[List[dict]]
  market_conditions: Optional[dict]
  recent_events: Optional[List[dict]]
  tick: Optional[int] = 0
  tools: Optional[List[ToolSpec|callable|dict]]
  extra: Optional[dict]

Example usage:

from agent_runners.langchain_runner import LangChainRunner

runner = LangChainRunner("agent-lc", {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "system_prompt": "Only JSON.",
})
result = await runner.run({
    "prompt": "Make pricing decisions for these products.",
    "products": [{"asin": "B0TEST", "current_price": 19.99, "cost": 10.0}],
    "tick": 1
})
print(result["status"], result["output"])

## 4. Tools

Both runners accept tools as:
- Callables: def my_tool(payload: dict) -> dict or str
- Dict descriptors: {"name","description","callable","schema"?}
- ToolSpec instances

The runners wrap tools into framework-specific adapters. If the framework does not support tools or the adapter is not available, the run proceeds without tools.

Example:

def set_price(payload: dict):
    """Set product price."""
    # payload: {"asin": "...", "price": 19.99}
    return {"ok": True, "asin": payload.get("asin"), "price": payload.get("price")}

runner = CrewAIRunner("agent-crew", {"tools": [set_price]})
# or LangChainRunner(...)

## 5. Logging and Progress

- Logging follows centralized configuration in [fba_bench/core/logging.py](fba_bench/core/logging.py)
- If REDIS_URL is set, progress events are published on topics:
  - runner:crewai:<agent_id>
  - runner:langchain:<agent_id>

Events:
- {"phase":"start"| "inference_start" | "inference_end" | "complete" | "error", ...}

## 6. Tests

Integration tests added:
- [tests/integration/runners/test_crewai_runner.py](tests/integration/runners/test_crewai_runner.py)
- [tests/integration/runners/test_langchain_runner.py](tests/integration/runners/test_langchain_runner.py)

Tests mock external frameworks when not installed and validate:
- NotInstalled errors
- Successful run path with fake frameworks
- Config validation failures

Run tests:
poetry run pytest -q

## 7. Installation Notes

- CrewAI:
  pip install "crewai>=0.28"

- LangChain:
  pip install "langchain>=0.3" "langchain-openai>=0.2"

- Optional Redis events:
  export REDIS_URL="redis://localhost:6379/0"