import importlib
import sys
import types
import pytest

from agent_runners.registry import create_runner, supported_runners
from agent_runners.base_runner import AgentRunnerInitializationError, AgentRunner


def test_supported_runners_contains_expected_keys():
    keys = supported_runners()
    # DIY should always be present
    assert "diy" in keys
    # Optional frameworks should be registered by key (soft deps on instantiation)
    assert "crewai" in keys
    assert "langchain" in keys


@pytest.mark.integration
def test_create_runner_unknown_key_raises_value_error():
    with pytest.raises(ValueError) as exc:
        create_runner("unknown_runner_key_xyz", {"agent_id": "a1"})
    msg = str(exc.value).lower()
    assert "unknown runner key" in msg
    assert "supported keys" in msg


@pytest.mark.integration
def test_deprecated_runner_factory_import_raises_runtimeerror():
    # Ensure a fresh import attempt hits the deprecation shim
    sys.modules.pop("agent_runners.runner_factory", None)
    with pytest.raises(RuntimeError) as exc:
        importlib.import_module("agent_runners.runner_factory")
    msg = str(exc.value)
    assert "runner_factory is deprecated" in msg.lower()
    assert "agent_runners.registry" in msg
    assert "create_runner" in msg


@pytest.mark.integration
def test_crewai_runner_instantiation_path(monkeypatch):
    """
    Behavior:
    - If CrewAI dependency is missing, the runner should raise AgentRunnerInitializationError
      during construction (base __init__ triggers _do_initialize).
    - If CrewAI is installed, instance should be created and be an AgentRunner.
    """
    # Try to instantiate; if missing dependency, expect AgentRunnerInitializationError
    try:
        runner = create_runner("crewai", {"agent_id": "crew-1", "model": "gpt-4o-mini"})
        assert isinstance(runner, AgentRunner)
    except AgentRunnerInitializationError as exc:
        # Validate missing dependency path has clear guidance
        assert "crewai" in str(exc).lower()
        assert "install" in str(exc).lower()


@pytest.mark.integration
def test_langchain_runner_instantiation_path(monkeypatch):
    """
    Behavior:
    - If LangChain deps are missing, expect AgentRunnerInitializationError on construct.
    - If installed, create a valid runner instance.
    """
    try:
        runner = create_runner("langchain", {"agent_id": "lc-1", "model": "gpt-4o-mini"})
        assert isinstance(runner, AgentRunner)
    except AgentRunnerInitializationError as exc:
        txt = str(exc).lower()
        assert "langchain" in txt or "langchain-openai" in txt
        assert "install" in txt