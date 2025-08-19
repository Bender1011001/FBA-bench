import sys
from types import SimpleNamespace
import pytest

from agent_runners.langchain_runner import LangChainRunner
from agent_runners.base_runner import AgentRunnerInitializationError


@pytest.mark.integration
def test_langchain_runner_not_installed(monkeypatch):
    # Ensure langchain_openai import fails
    sys.modules.pop("langchain_openai", None)
    sys.modules.pop("langchain", None)
    sys.modules.pop("langchain_core", None)
    sys.modules.pop("langchain.tools", None)
    sys.modules.pop("langchain.agents", None)

    with pytest.raises(AgentRunnerInitializationError) as exc:
        LangChainRunner("lc-agent-x", {"model": "gpt-4o-mini"})

    assert "LangChain" in str(exc.value)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_langchain_runner_success_with_fake_lib(monkeypatch):
    # Fake ChatOpenAI that returns a message-like object with content
    class FakeResp:
        def __init__(self, content: str):
            self.content = content

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, x):
            # x may be list of messages or string
            return FakeResp('{"decisions":[{"asin":"B0TEST","new_price":19.99,"reasoning":"ok"}],"meta":{"tick":1}}')

    # Fake tools and agents surfaces
    class FakeTool:
        def __init__(self, name, func, description=""):
            self.name = name
            self.func = func
            self.description = description

    def fake_initialize_agent(tools, llm, agent, verbose=False):
        class AgentWrapper:
            def __init__(self, tools):
                self._tools = tools

            def run(self, prompt: str):
                # Always return valid JSON string
                return '{"decisions":[{"asin":"B0TOOL","new_price":18.75,"reasoning":"tool"}],"meta":{"tick":2}}'

        return AgentWrapper(tools)

    # Fake messages
    class FakeSystemMessage:
        def __init__(self, content): self.content = content

    class FakeHumanMessage:
        def __init__(self, content): self.content = content

    # Install fakes into sys.modules
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    monkeypatch.setitem(sys.modules, "langchain.tools", SimpleNamespace(Tool=FakeTool))
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(AgentType=SimpleNamespace(STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION="S"),
                        initialize_agent=fake_initialize_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.messages",
        SimpleNamespace(SystemMessage=FakeSystemMessage, HumanMessage=FakeHumanMessage),
    )

    # With tools in config
    def sample_tool(payload):
        """set price"""
        return {"ok": True, "input": payload}

    runner = LangChainRunner(
        "lc-agent-1",
        {
            "model": "gpt-4o-mini",
            "system_prompt": "Only JSON outputs.",
            "tools": [sample_tool],
        },
    )

    result = await runner.run(
        {
            "prompt": "make pricing decision",
            "products": [{"asin": "B0TEST", "current_price": 21.00, "cost": 10.5}],
            "tick": 2,
        }
    )

    assert result["status"] == "success"
    assert isinstance(result["output"], str)
    assert result["steps"] and result["steps"][-1]["role"] == "assistant"
    assert "metrics" in result and "duration_ms" in result["metrics"]