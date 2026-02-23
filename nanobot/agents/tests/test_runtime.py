"""Tests for runtime engine: ToolFactory, AgentInstance, Orchestrator."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agents.spec import AgentSpec
from nanobot.agents.registry import AgentRegistry
from nanobot.agents.factory import ToolFactory, TOOL_CATALOG
from nanobot.agents.instance import AgentInstance, AgentResult
from nanobot.agents.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    AgentSession,
)
from nanobot.agents.tools.transfer import HANDOFF_SIGNAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    name="test_agent",
    team="test_team",
    tools=(),
    handoffs=(),
    **kw,
) -> AgentSpec:
    return AgentSpec(
        name=name,
        team=team,
        description=f"{name} agent",
        tools=tools,
        handoffs=handoffs,
        **kw,
    )


def _make_registry(*specs: AgentSpec) -> AgentRegistry:
    reg = AgentRegistry()
    for s in specs:
        reg.register(s)
    return reg


class FakeToolCallRequest:
    """Mimics nanobot.providers.base.ToolCallRequest."""

    def __init__(self, id: str, name: str, arguments: dict[str, Any]):
        self.id = id
        self.name = name
        self.arguments = arguments


class FakeLLMResponse:
    """Mimics nanobot.providers.base.LLMResponse."""

    def __init__(
        self,
        content: str = "",
        tool_calls: list | None = None,
        usage: dict[str, int] | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage or {}

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# ToolFactory tests
# ---------------------------------------------------------------------------


class TestToolFactory:
    def test_build_tools_empty_spec(self):
        """Agent with no tools or handoffs gets an empty registry."""
        spec = _make_spec()
        reg = _make_registry(spec)
        factory = ToolFactory(reg)
        tools = factory.build_tools(spec)
        assert len(tools) == 0

    def test_build_tools_with_handoffs(self):
        """Handoff targets become transfer_to_<name> tools."""
        ceo = _make_spec("ceo", handoffs=("dev_manager",))
        dev = _make_spec("dev_manager")
        reg = _make_registry(ceo, dev)
        factory = ToolFactory(reg)
        tools = factory.build_tools(ceo)
        assert tools.has("transfer_to_dev_manager")

    def test_build_tools_unknown_tool_skipped(self):
        """Unknown tools in spec are silently skipped."""
        spec = _make_spec(tools=("nonexistent_tool",))
        reg = _make_registry(spec)
        factory = ToolFactory(reg)
        tools = factory.build_tools(spec)
        # Should have 0 tools (unknown one is skipped)
        assert len(tools) == 0

    def test_build_tools_multiple_handoffs(self):
        """Multiple handoff targets each get their own transfer tool."""
        spec = _make_spec(handoffs=("agent_a", "agent_b", "agent_c"))
        a = _make_spec("agent_a")
        b = _make_spec("agent_b")
        c = _make_spec("agent_c")
        reg = _make_registry(spec, a, b, c)
        factory = ToolFactory(reg)
        tools = factory.build_tools(spec)
        assert tools.has("transfer_to_agent_a")
        assert tools.has("transfer_to_agent_b")
        assert tools.has("transfer_to_agent_c")
        assert len(tools) == 3

    def test_tool_catalog_keys(self):
        """Verify expected tools exist in the catalog."""
        expected = {"web_search", "web_fetch", "file_read", "file_write", "shell"}
        assert expected.issubset(set(TOOL_CATALOG.keys()))


# ---------------------------------------------------------------------------
# AgentResult tests
# ---------------------------------------------------------------------------


class TestAgentResult:
    def test_stop_result(self):
        r = AgentResult("bot", "hello", "stop")
        assert not r.is_handoff
        assert not r.is_error
        assert r.agent_name == "bot"

    def test_handoff_result(self):
        r = AgentResult("bot", "__HANDOFF__:target:reason:ctx", "handoff")
        assert r.is_handoff
        assert not r.is_error

    def test_error_result(self):
        r = AgentResult("bot", "oops", "error")
        assert r.is_error
        assert not r.is_handoff

    def test_repr(self):
        r = AgentResult("bot", "some content here", "stop")
        assert "bot" in repr(r)
        assert "stop" in repr(r)


# ---------------------------------------------------------------------------
# AgentInstance tests
# ---------------------------------------------------------------------------


class TestAgentInstance:
    def _make_instance(self, spec=None, provider=None, tools=None):
        spec = spec or _make_spec()
        provider = provider or AsyncMock()
        if tools is None:
            tools = MagicMock()
            tools.__len__ = MagicMock(return_value=0)
            tools.get_definitions = MagicMock(return_value=[])
        return AgentInstance(spec, provider, tools)

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Agent returns text on first iteration."""
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=FakeLLMResponse(
                content="Hello!",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        )
        instance = self._make_instance(provider=provider)
        result = await instance.run([{"role": "user", "content": "hi"}])
        assert result.content == "Hello!"
        assert result.finish_reason == "stop"
        assert not result.is_handoff

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        """Agent calls a tool then responds with text."""
        tool_call = FakeToolCallRequest("tc1", "web_search", {"query": "test"})

        provider = AsyncMock()
        # First call: tool call, second call: text response
        provider.chat = AsyncMock(
            side_effect=[
                FakeLLMResponse(tool_calls=[tool_call]),
                FakeLLMResponse(content="Found results!"),
            ]
        )

        tools = MagicMock()
        tools.__len__ = MagicMock(return_value=1)
        tools.get_definitions = MagicMock(return_value=[{"type": "function", "function": {"name": "web_search"}}])
        tools.execute = AsyncMock(return_value="search results here")

        instance = self._make_instance(provider=provider, tools=tools)
        result = await instance.run([{"role": "user", "content": "search for test"}])
        assert result.content == "Found results!"
        assert result.finish_reason == "stop"
        tools.execute.assert_called_once_with("web_search", {"query": "test"})

    @pytest.mark.asyncio
    async def test_handoff_detected(self):
        """Agent triggers a handoff via tool call."""
        handoff_signal = f"{HANDOFF_SIGNAL}:dev_manager:need code help:user wants a script"
        tool_call = FakeToolCallRequest(
            "tc1", "transfer_to_dev_manager", {"reason": "need code help", "context": "user wants a script"}
        )

        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=FakeLLMResponse(tool_calls=[tool_call])
        )

        tools = MagicMock()
        tools.__len__ = MagicMock(return_value=1)
        tools.get_definitions = MagicMock(return_value=[])
        tools.execute = AsyncMock(return_value=handoff_signal)

        instance = self._make_instance(provider=provider, tools=tools)
        result = await instance.run([{"role": "user", "content": "write me code"}])
        assert result.is_handoff
        assert result.finish_reason == "handoff"
        assert "dev_manager" in result.content

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        """Agent hits max iterations limit."""
        tool_call = FakeToolCallRequest("tc1", "web_search", {"query": "loop"})

        provider = AsyncMock()
        # Always return tool calls — never finishes
        provider.chat = AsyncMock(
            return_value=FakeLLMResponse(tool_calls=[tool_call])
        )

        tools = MagicMock()
        tools.__len__ = MagicMock(return_value=1)
        tools.get_definitions = MagicMock(return_value=[])
        tools.execute = AsyncMock(return_value="results")

        spec = _make_spec(max_iterations=3)
        instance = self._make_instance(spec=spec, provider=provider, tools=tools)
        result = await instance.run([{"role": "user", "content": "go"}])
        assert result.finish_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_llm_error(self):
        """Agent handles LLM errors gracefully."""
        provider = AsyncMock()
        provider.chat = AsyncMock(side_effect=Exception("API timeout"))

        instance = self._make_instance(provider=provider)
        result = await instance.run([{"role": "user", "content": "hi"}])
        assert result.is_error
        assert "API timeout" in result.content


# ---------------------------------------------------------------------------
# AgentSession tests
# ---------------------------------------------------------------------------


class TestAgentSession:
    def test_creation(self):
        s = AgentSession("sess1", "ceo")
        assert s.session_id == "sess1"
        assert s.current_agent == "ceo"
        assert s.handoff_chain == ["ceo"]

    def test_record_handoff(self):
        s = AgentSession("sess1", "ceo")
        s.record_handoff("dev_manager")
        assert s.current_agent == "dev_manager"
        assert s.handoff_chain == ["ceo", "dev_manager"]

    def test_accumulate_usage(self):
        s = AgentSession("sess1", "ceo")
        s.accumulate_usage({"prompt_tokens": 10, "completion_tokens": 5})
        s.accumulate_usage({"prompt_tokens": 20, "completion_tokens": 10})
        assert s.total_usage["prompt_tokens"] == 30
        assert s.total_usage["completion_tokens"] == 15


# ---------------------------------------------------------------------------
# OrchestratorResult tests
# ---------------------------------------------------------------------------


class TestOrchestratorResult:
    def test_to_chat_completion(self):
        r = OrchestratorResult(
            content="Hello!",
            agent_name="ceo",
            session_id="s1",
            finish_reason="stop",
            handoff_chain=["ceo"],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        cc = r.to_chat_completion()
        assert cc["model"] == "agent/ceo"
        assert cc["choices"][0]["message"]["content"] == "Hello!"
        assert cc["choices"][0]["finish_reason"] == "stop"
        assert cc["nanobot_metadata"]["responding_agent"] == "ceo"
        assert cc["usage"]["total_tokens"] == 15

    def test_loop_detected_maps_to_stop(self):
        r = OrchestratorResult(
            content="loop!", agent_name="ceo", session_id="s1",
            finish_reason="loop_detected",
        )
        cc = r.to_chat_completion()
        assert cc["choices"][0]["finish_reason"] == "stop"

    def test_to_stream_chunk(self):
        r = OrchestratorResult(content="Hi", agent_name="ceo", session_id="s1")
        chunk = r.to_stream_chunk("partial")
        assert chunk["choices"][0]["delta"]["content"] == "partial"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_to_stream_chunk_final(self):
        r = OrchestratorResult(content="Hi", agent_name="ceo", session_id="s1")
        chunk = r.to_stream_chunk("")
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_repr(self):
        r = OrchestratorResult(
            content="x", agent_name="ceo", session_id="s1",
            handoff_chain=["ceo", "dev"],
        )
        assert "ceo" in repr(r)
        assert "dev" in repr(r)


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def _make_orchestrator(self, agents: list[AgentSpec] | None = None):
        """Create an Orchestrator with mock provider and real registry."""
        agents = agents or [_make_spec("ceo", "exec", role="lead")]
        reg = _make_registry(*agents)
        provider = AsyncMock()
        factory = MagicMock(spec=ToolFactory)
        factory.build_tools = MagicMock(return_value=MagicMock())
        # Make the mock tool registry have 0 tools
        factory.build_tools.return_value.__len__ = MagicMock(return_value=0)
        factory.build_tools.return_value.get_definitions = MagicMock(return_value=[])
        return Orchestrator(provider, reg, factory), provider, factory

    @pytest.mark.asyncio
    async def test_simple_response(self):
        """Orchestrator returns response from a single agent."""
        orch, provider, factory = self._make_orchestrator()

        # Mock provider to return simple text
        provider.chat = AsyncMock(
            return_value=FakeLLMResponse(
                content="I'm the CEO!",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        )

        result = await orch.run("ceo", [{"role": "user", "content": "hello"}])
        assert result.content == "I'm the CEO!"
        assert result.agent_name == "ceo"
        assert result.finish_reason == "stop"
        assert result.handoff_chain == ["ceo"]

    @pytest.mark.asyncio
    async def test_agent_not_found(self):
        """Orchestrator handles missing agent gracefully."""
        orch, _, _ = self._make_orchestrator()
        result = await orch.run("nonexistent", [{"role": "user", "content": "hi"}])
        assert result.finish_reason == "error"
        assert "nonexistent" in result.content

    @pytest.mark.asyncio
    async def test_handoff_flow(self):
        """Orchestrator handles a single handoff between agents."""
        ceo = _make_spec("ceo", "exec", handoffs=("dev_manager",))
        dev = _make_spec("dev_manager", "dev")
        orch, provider, factory = self._make_orchestrator([ceo, dev])

        handoff_signal = f"{HANDOFF_SIGNAL}:dev_manager:need code:user wants python"

        # First agent call: tool call that triggers handoff
        tool_call = FakeToolCallRequest(
            "tc1", "transfer_to_dev_manager",
            {"reason": "need code", "context": "user wants python"},
        )
        handoff_response = FakeLLMResponse(tool_calls=[tool_call])

        # Second agent call: direct text response
        final_response = FakeLLMResponse(
            content="Here's your Python code!",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )

        # Setup mock tool registry that returns handoff signal
        mock_tools = MagicMock()
        mock_tools.__len__ = MagicMock(return_value=1)
        mock_tools.get_definitions = MagicMock(return_value=[])
        mock_tools.execute = AsyncMock(return_value=handoff_signal)
        factory.build_tools = MagicMock(return_value=mock_tools)

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return handoff_response
            return final_response

        provider.chat = mock_chat

        result = await orch.run("ceo", [{"role": "user", "content": "write python"}])
        assert result.agent_name == "dev_manager"
        assert result.content == "Here's your Python code!"
        assert result.handoff_chain == ["ceo", "dev_manager"]

    @pytest.mark.asyncio
    async def test_loop_detection(self):
        """Orchestrator detects handoff loops."""
        orch, _, _ = self._make_orchestrator()
        # Simulate a chain with repeated agents
        assert orch._detect_loop(["ceo", "dev", "ceo", "dev"], "ceo") is True
        assert orch._detect_loop(["ceo", "dev"], "ceo") is False
        assert orch._detect_loop(["a", "b", "a", "b"], "a") is True

    @pytest.mark.asyncio
    async def test_session_tracking(self):
        """Orchestrator tracks sessions properly."""
        orch, provider, _ = self._make_orchestrator()
        provider.chat = AsyncMock(
            return_value=FakeLLMResponse(content="done")
        )

        result = await orch.run(
            "ceo",
            [{"role": "user", "content": "hi"}],
            session_id="test-session",
        )
        assert result.session_id == "test-session"
        session = orch.get_session("test-session")
        assert session is not None
        assert session.current_agent == "ceo"

    @pytest.mark.asyncio
    async def test_max_handoffs_limit(self):
        """Orchestrator stops after MAX_HANDOFFS."""
        # Create agents that always hand off to each other
        a = _make_spec("agent_a", handoffs=("agent_b",))
        b = _make_spec("agent_b", handoffs=("agent_a",))
        orch, provider, factory = self._make_orchestrator([a, b])
        orch.MAX_HANDOFFS = 3  # Lower for test speed

        # Always trigger a handoff tool call
        tool_call_a = FakeToolCallRequest("tc1", "transfer_to_agent_b", {"reason": "pass"})
        tool_call_b = FakeToolCallRequest("tc2", "transfer_to_agent_a", {"reason": "pass"})

        # Override loop detection to not trigger (test max_handoffs specifically)
        orch._detect_loop = lambda chain, target, window=4: False

        mock_tools = MagicMock()
        mock_tools.__len__ = MagicMock(return_value=1)
        mock_tools.get_definitions = MagicMock(return_value=[])

        call_count = 0

        async def mock_execute(name, args):
            if "agent_b" in name:
                return f"{HANDOFF_SIGNAL}:agent_b:pass:"
            return f"{HANDOFF_SIGNAL}:agent_a:pass:"

        mock_tools.execute = mock_execute
        factory.build_tools = MagicMock(return_value=mock_tools)

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return FakeLLMResponse(tool_calls=[tool_call_a])
            return FakeLLMResponse(tool_calls=[tool_call_b])

        provider.chat = mock_chat

        result = await orch.run("agent_a", [{"role": "user", "content": "go"}])
        assert result.finish_reason == "max_handoffs"

    def test_list_agents(self):
        """Orchestrator delegates to registry for agent listing."""
        orch, _, _ = self._make_orchestrator()
        models = orch.list_agents()
        assert len(models) == 1
        assert models[0]["id"] == "agent/ceo"

    @pytest.mark.asyncio
    async def test_prepare_messages_injects_system_prompt(self):
        """_prepare_messages adds agent system prompt."""
        spec = _make_spec(system_prompt="You are a helpful CEO.")
        orch, _, _ = self._make_orchestrator([spec])
        session = AgentSession("s1", "test_agent")

        msgs = orch._prepare_messages(
            spec,
            [{"role": "user", "content": "hi"}],
            session,
        )
        # First message should be system
        assert msgs[0]["role"] == "system"
        assert "You are a helpful CEO." in msgs[0]["content"]
        # Second should be user (system from input stripped)
        assert msgs[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_prepare_messages_strips_existing_system(self):
        """_prepare_messages removes existing system messages from input."""
        spec = _make_spec()
        orch, _, _ = self._make_orchestrator([spec])
        session = AgentSession("s1", "test_agent")

        msgs = orch._prepare_messages(
            spec,
            [
                {"role": "system", "content": "old system prompt"},
                {"role": "user", "content": "hi"},
            ],
            session,
        )
        # Should only have our system + user, not old system
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "old system prompt" not in msgs[0]["content"]
