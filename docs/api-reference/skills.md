# Skills System (Lightweight, Deterministic)

This module provides a minimal, deterministic skill system that agents and runners can call uniformly, with strict validation and a central registry for discovery.

Key goals:
- Minimal dependencies (stdlib + Pydantic v2).
- Deterministic behavior and explicit errors.
- Uniform call contract:
  - Sync: `run(params: dict) -> dict`
  - Optional Async: `arun(params: dict) -> dict` (defaults to wrapping `run` in a threadpool).

Package location: `agents/skill_modules/`

- Core: [`agents/skill_modules/base.py`](agents/skill_modules/base.py:1)
- Registry: [`agents/skill_modules/registry.py`](agents/skill_modules/registry.py:1)
- Built-in skills:
  - Calculator: [`agents/skill_modules/calculator.py`](agents/skill_modules/calculator.py:1)
  - Summarize: [`agents/skill_modules/summarize.py`](agents/skill_modules/summarize.py:1)
  - ExtractFields: [`agents/skill_modules/extract_fields.py`](agents/skill_modules/extract_fields.py:1)
  - Lookup: [`agents/skill_modules/lookup.py`](agents/skill_modules/lookup.py:1)
  - TransformText: [`agents/skill_modules/transform_text.py`](agents/skill_modules/transform_text.py:1)

Logging integrates with [`fba_bench/core/logging.py`](fba_bench/core/logging.py:1) when present.

## Quick Start

List available skills and create by name:

```python
from agents.skill_modules.registry import list_skills, create

skills = list_skills()
# [{'name': 'calculator', 'description': '...', 'input_schema': {...}, ...}, ...]

calc = create("calculator")
out = calc.run({"expression": "12*(3+4)"})
# {'result': 84.0, 'steps': ['parsed AST', 'evaluated nodes safely']}

# Async form (optional)
import asyncio
async def demo():
    res = await calc.arun({"expression": "2+2*3"})
    # res -> {'result': 8.0, 'steps': [...]}
asyncio.run(demo())
```

## Uniform Interface

All skills implement the `Skill` ABC:

- `name: str` – unique registry key
- `description: str`
- `input_model: type[BaseModel]`
- `output_model: type[BaseModel]`
- `run(self, params: dict) -> dict`
- `arun(self, params: dict) -> dict` (default wraps `run`)

Errors are raised as `SkillExecutionError`.

See definition in [`agents/skill_modules/base.py`](agents/skill_modules/base.py:1).

## Built-in Skills

### Calculator

- Input model: `{"expression": str}`
- Output model: `{"result": float, "steps": list[str]}`
- Features:
  - Safe evaluation via Python AST whitelist
  - Supports: `+ - * / // % **` and parentheses
  - Guards exponent size, nesting, node count, and magnitude
  - Disallows names, calls, attributes, subscripts
- Example:
  ```python
  from agents.skill_modules.registry import create
  calc = create("calculator")
  calc.run({"expression": "2+2*3"})  # -> {'result': 8.0, 'steps': [...]}
  ```

Implementation: [`agents/skill_modules/calculator.py`](agents/skill_modules/calculator.py:1)

### Summarize

- Input: `{"text": str, "max_tokens": int=128}`
- Output: `{"summary": str}`
- Strategy: Simple extractive – split into sentences, take in order until budget reached (whitespace token heuristic).
- Deterministic splitting and joining.

Implementation: [`agents/skill_modules/summarize.py`](agents/skill_modules/summarize.py:1)

### ExtractFields

- Input: `{"text": str, "fields": list[str]}`
- Output: `{"extracted": dict[str, str]}`
- Strategy:
  - Parse lines like `Field: value` (case-insensitive keys)
  - Parse embedded JSON-like objects for primitives
  - Only return requested fields; missing fields are omitted

Implementation: [`agents/skill_modules/extract_fields.py`](agents/skill_modules/extract_fields.py:1)

### Lookup

- Input: `{"key": str, "table": dict[str, str]}`
- Output: `{"value": str|None, "found": bool}`
- Behavior: Exact match, case-sensitive by default.

Implementation: [`agents/skill_modules/lookup.py`](agents/skill_modules/lookup.py:1)

### TransformText

- Input: `{"text": str, "operations": list[str]}`
- Output: `{"text": str}`
- Supported ops: `lower`, `upper`, `strip`, `dedupe_whitespace`, `title`
- Applies operations in order; unknown operation raises `SkillExecutionError`.

Implementation: [`agents/skill_modules/transform_text.py`](agents/skill_modules/transform_text.py:1)

## Registry API

Use [`agents/skill_modules/registry.py`](agents/skill_modules/registry.py:1):

- `register(name: str, cls: type[Skill]) -> None` – register a new skill
- `create(name: str, config: dict|None = None) -> Skill` – instantiate by name
- `list_skills() -> list[dict]` – metadata including JSON Schemas

Built-in skills are auto-registered on import.

Example:

```python
from agents.skill_modules.registry import register, create
from agents.skill_modules.base import Skill, SkillInputModel, SkillOutputModel

class MyInput(SkillInputModel):
    ...

class MyOutput(SkillOutputModel):
    ...

class MySkill(Skill):
    name = "my_skill"
    description = "Example"
    input_model = MyInput
    output_model = MyOutput

    def run(self, params: dict) -> dict:
        data = self.input_model.model_validate(params)
        return MyOutput(...).model_dump()

register(MySkill.name, MySkill)
inst = create("my_skill")
```

## Determinism and Safety

- All inputs/outputs validated by Pydantic v2 models.
- Calculator uses strict AST whitelist; exponent and magnitude guards.
- Summarization is extractive and ordering-deterministic.
- TransformText only allows a documented set of pure operations.
- Extraction patterns are simple and bounded; JSON-like parsing ignores nested complex types.
- Registry errors include supported skill names.

## Testing

Run tests focused on skills:

```bash
poetry run pytest -q tests/integration/skills
```

Tests include:
- Registry listing and creation
- Calculator valid/invalid expressions
- Summarize deterministic budget handling
- ExtractFields mixed sources and missing fields
- Lookup case-sensitive behavior
- TransformText ordered ops and error on unknown ops

## Using in Agents/Runners

The following import is stable:

```python
from agents.skill_modules.registry import create, list_skills

skill = create("transform_text")
out = skill.run({"text": "  Hello   WORLD ", "operations": ["strip", "lower", "dedupe_whitespace", "title"]})
# -> {"text": "Hello World"}
```

## Relationship to Legacy Multi-Skill System

This lightweight skill system is independent of the event-driven domain skills used by the Multi-Skill Agent architecture (e.g., `SupplyManager`, `MarketingManager`). Those continue to live under the same package for compatibility. See:
- Legacy base: [`agents/skill_modules/base_skill.py`](agents/skill_modules/base_skill.py:1)
- Coordinator: [`agents/skill_coordinator.py`](agents/skill_coordinator.py:1)
- Overview: [`docs/multi-skill-agents/skill-system-overview.md`](docs/multi-skill-agents/skill-system-overview.md:1)
