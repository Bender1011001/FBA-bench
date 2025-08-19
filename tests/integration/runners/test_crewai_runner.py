import json
import sys
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from agent_runners.crewai_runner import CrewAIRunner, CrewAIRunnerConfig
from agent_runners.base_runner import AgentRunnerInitializationError


@pytest.mark.integration
def test_crewai_runner_not_installed(monkeypatch):
    # Ensure crewai import fails
    sys.modules.pop("crewai", None)

    with pytest.raises(AgentRunnerInitializationError) as exc:
        CrewAIRunner("agent-x", {"model": "gpt-4o-mini"})  # base __init__ triggers _do_initialize

    assert "CrewAI is not installed" in str(exc.value) or "CrewAI not installed" in str(exc.value)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crewai_runner_success_with_fake_lib(monkeypatch):
    # Fake CrewAI minimal surface
    class FakeAgent:
        def __init__(self, **kwargs): ...

    class FakeTask:
        def __init__(self, description: str, agent: Any, expected_output: str, **kwargs):
            self.description = description
            self.agent = agent
            self.expected_output = expected_output

    class FakeCrew:
        def __init__(self, agents, tasks, verbose=False):
            self.agents = agents
            self.tasks = tasks
            self.verbose = verbose

        def kickoff(self):
            # Return valid JSON string
            return json.dumps(
                {
                    "decisions": [{"asin": "B0TEST", "new_price": 19.99, "reasoning": "ok"}],
                    "meta": {"tick": 1},
                }
            )

    fake_module = SimpleNamespace(Agent=FakeAgent, Task=FakeTask, Crew=FakeCrew, Tool=None)
    monkeypatch.setitem(sys.modules, "crewai", fake_module)

    runner = CrewAIRunner(
        "agent-1",
        {
            "model": "gpt-4o-mini",
            "system_prompt": "Only JSON outputs.",
        },
    )

    result = await runner.run(
        {
            "prompt": "make pricing decision",
            "products": [{"asin": "B0TEST", "current_price": 20.50, "cost": 10.0}],
            "tick": 1,
        }
    )

    assert result["status"] == "success"
    assert isinstance(result["output"], str)
    assert result["steps"] and result["steps"][-1]["role"] == "assistant"
    assert "metrics" in result and "duration_ms" in result["metrics"]


@pytest.mark.integration
def test_crewai_runner_config_validation_error():
    # invalid temperature
    with pytest.raises(Exception):
        CrewAIRunner("agent-v", {"temperature": 99.0})