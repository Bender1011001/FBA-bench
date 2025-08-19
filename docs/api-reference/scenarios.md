# Scenario Modules: Complex, Realistic Benchmarks

This document describes three production-ready, deterministic scenarios added to the benchmarking suite. Each scenario is runner-agnostic and communicates exclusively via a strict, documented interface. All scenarios expose:

- [python.class ScenarioConfig(BaseModel)](docs/api-reference/scenarios.md:1)
- [python.def generate_input(seed: int|None, params: dict|None) -> dict](docs/api-reference/scenarios.md:1)
- [python.async def run(input_payload: dict, runner_callable: Callable[[dict], Awaitable[dict]], timeout_seconds: int|None=None) -> dict](docs/api-reference/scenarios.md:1)
- Optionally [python.def postprocess(raw_output: dict) -> dict](docs/api-reference/scenarios.md:1)

Registration keys:
- complex_marketplace
- research_summarization
- multiturn_tool_use

Direct code links:
- [benchmarking/scenarios/complex_marketplace.py](benchmarking/scenarios/complex_marketplace.py:1)
- [benchmarking/scenarios/research_summarization.py](benchmarking/scenarios/research_summarization.py:1)
- [benchmarking/scenarios/multiturn_tool_use.py](benchmarking/scenarios/multiturn_tool_use.py:1)

All scenarios are deterministic by default with `seed` and are parameterizable via `params` validated by Pydantic v2 models. Floating point outputs are stabilized via explicit rounding where applicable.

## 1) ComplexMarketplaceScenario

Module: [benchmarking/scenarios/complex_marketplace.py](benchmarking/scenarios/complex_marketplace.py:1)  
Config: [python.class ComplexMarketplaceConfig(BaseModel)](benchmarking/scenarios/complex_marketplace.py:1)

Parameters (defaults):
- num_products: int=20
- num_orders: int=50
- max_quantity: int=5
- price_variance: float=0.1
- allow_backorder: bool=False

Behavior:
- Deterministically synthesizes a product catalog with seeded prices and stocks.
- Generates a batch of multi-line orders with realistic noise (invalid SKUs, excessive quantities, optional price hints).
- Runner is tasked to validate orders, price accepted lines, and produce a fulfillment plan respecting stock and `allow_backorder`.

Runner input contract (provided to runner_callable):
- catalog: list of products with fields: sku, base_price, price, stock
- orders: list of orders with order_id and lines: sku, quantity, price_hint?
- policies: { allow_backorder: bool, max_quantity_per_line: int }
- seed, instructions

Runner output expectation:
- accepted_orders: list with lines {sku, quantity, unit_price}
- rejections: list with reasons
- fulfillment: map sku -> {allocated: int}
- policy_violations: int

Scenario output:
- {"accepted": int, "revenue": float, "fulfilled_rate": float, "policy_violations": int, "details": {...}}

Postprocessing:
- [python.def postprocess(raw_output: dict) -> dict](benchmarking/scenarios/complex_marketplace.py:1) normalizes revenue and ratios.

## 2) ResearchSummarizationScenario

Module: [benchmarking/scenarios/research_summarization.py](benchmarking/scenarios/research_summarization.py:1)  
Config: [python.class ResearchSummarizationConfig(BaseModel)](benchmarking/scenarios/research_summarization.py:1)

Parameters (defaults):
- num_docs: int=5
- max_tokens: int=200
- focus_keywords: list[str]|None=None
- noise_probability: float=0.0

Behavior:
- Builds a deterministic corpus of abstracts across several themes with overlapping keywords.
- Optionally injects noise sentences deterministically.
- Runner must produce a focused summary addressing `focus_keywords` under `max_tokens`.

Runner input contract:
- documents: list[{id,title,abstract,theme,keywords}]
- focus_keywords: list[str]
- max_tokens: int
- prompt/instructions
- seed

Runner output expectation:
- {"summary": str}

Scenario output:
- {"summary": str, "coverage_score": float, "length_ok": bool, "keyword_hits": int}

Scoring:
- Coverage is proportion of focus keywords found in the summary (case-insensitive).
- Brevity/length constraint is also factored.

## 3) MultiTurnToolUseScenario

Module: [benchmarking/scenarios/multiturn_tool_use.py](benchmarking/scenarios/multiturn_tool_use.py:1)  
Config: [python.class MultiTurnToolUseConfig(BaseModel)](benchmarking/scenarios/multiturn_tool_use.py:1)

Parameters (defaults):
- steps: int=3
- include_math: bool=True
- include_extraction: bool=True
- include_transform: bool=False

Behavior:
- Emits a sequence of tool-like tasks derived deterministically from `seed`.
- Supported step types:
  - math: compute a deterministic arithmetic expression
  - extraction: parse structured fields from a deterministic sentence template
  - transform: sum a list and apply an element-wise transform using multiplier/offset

Runner input contract:
- tasks: list of step dicts
- instructions, seed

Runner output expectation:
- {"results": list[dict]} with entries per task, e.g. {"type": "math", "result": 42}

Scenario output:
- {"steps_completed": int, "correct": int, "total": int, "score": float, "details": {"steps": [...]}}
- Score is exact-match based on the deterministic expected results.

## Using Scenarios via EngineConfig

Engine and models:
- [python.class EngineConfig(BaseModel)](benchmarking/core/engine.py:1)
- [python.class ScenarioSpec(BaseModel)](benchmarking/core/engine.py:1)
- [python.def run_benchmark(config: dict|EngineConfig) -> EngineReport](benchmarking/core/engine.py:1)

Minimal example:
```python
from benchmarking.core.engine import EngineConfig, ScenarioSpec, RunnerSpec, run_benchmark

config = EngineConfig(
    scenarios=[
        ScenarioSpec(key="complex_marketplace", params={"num_products": 10, "num_orders": 25}, repetitions=1, seeds=[42], timeout_seconds=5),
        ScenarioSpec(key="research_summarization", params={"num_docs": 5, "max_tokens": 180, "focus_keywords": ["Q3","revenue"]}, repetitions=1, seeds=[101]),
        ScenarioSpec(key="multiturn_tool_use", params={"steps": 3, "include_transform": True}, repetitions=1, seeds=[7]),
    ],
    runners=[RunnerSpec(key="diy", config={"agent_id": "baseline-1"})],
    metrics=[],
    validators=[],
    parallelism=1,
    retries=0,
)
report = run_benchmark(config)
print(report.model_dump())
```

## Determinism and Validation

- All scenarios accept `seed` to ensure deterministic generation and evaluation.
- Parameter `params` is parsed by Pydantic v2 models with json_schema_extra examples.
- Numeric outputs are stabilized via Decimal rounding where appropriate (e.g., revenue, rates).
- Scenarios are runner-agnostic; the engine supplies the async `runner_callable`.

## Registry Integration

Registration happens on module import using the global registry:
- [python.class ScenarioRegistry](benchmarking/scenarios/registry.py:1)
- Keys:
  - "complex_marketplace" - [benchmarking/scenarios/complex_marketplace.py](benchmarking/scenarios/complex_marketplace.py:1)
  - "research_summarization" - [benchmarking/scenarios/research_summarization.py](benchmarking/scenarios/research_summarization.py:1)
  - "multiturn_tool_use" - [benchmarking/scenarios/multiturn_tool_use.py](benchmarking/scenarios/multiturn_tool_use.py:1)