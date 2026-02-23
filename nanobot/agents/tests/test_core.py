"""Tests for agent core framework: spec, registry, transfer tool, loader."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from nanobot.agents.spec import AgentSpec
from nanobot.agents.registry import AgentRegistry
from nanobot.agents.tools.transfer import (
    TransferToAgentTool,
    is_handoff_signal,
    parse_handoff_signal,
    HANDOFF_SIGNAL,
)
from nanobot.agents.loader import load_agents


# --- AgentSpec tests ---


class TestAgentSpec:
    def test_basic_creation(self):
        spec = AgentSpec(
            name="ceo",
            team="executive",
            description="The CEO agent",
        )
        assert spec.name == "ceo"
        assert spec.team == "executive"
        assert spec.role == "member"
        assert spec.model == "default"
        assert spec.tools == ()
        assert spec.handoffs == ()

    def test_lead_role(self):
        spec = AgentSpec(name="ceo", team="executive", description="CEO", role="lead")
        assert spec.is_lead is True

    def test_member_role(self):
        spec = AgentSpec(name="worker", team="dev", description="Worker")
        assert spec.is_lead is False

    def test_qualified_name(self):
        spec = AgentSpec(name="ceo", team="executive", description="CEO")
        assert spec.qualified_name == "executive/ceo"

    def test_model_id(self):
        spec = AgentSpec(name="ceo", team="executive", description="CEO")
        assert spec.model_id == "agent/ceo"

    def test_to_model_entry(self):
        spec = AgentSpec(
            name="ceo", team="executive", description="CEO", role="lead"
        )
        entry = spec.to_model_entry()
        assert entry["id"] == "agent/ceo"
        assert entry["object"] == "model"
        assert entry["owned_by"] == "nanobot"
        assert entry["metadata"]["team"] == "executive"
        assert entry["metadata"]["role"] == "lead"

    def test_frozen(self):
        spec = AgentSpec(name="ceo", team="executive", description="CEO")
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore


# --- AgentRegistry tests ---


class TestAgentRegistry:
    def _make_spec(self, name="test", team="team", role="member"):
        return AgentSpec(name=name, team=team, description=f"{name} agent", role=role)

    def test_register_and_get(self):
        reg = AgentRegistry()
        spec = self._make_spec("ceo", "executive")
        reg.register(spec)
        assert reg.get("ceo") is spec
        assert reg.has("ceo")
        assert "ceo" in reg

    def test_get_missing(self):
        reg = AgentRegistry()
        assert reg.get("nonexistent") is None
        assert not reg.has("nonexistent")

    def test_unregister(self):
        reg = AgentRegistry()
        reg.register(self._make_spec("ceo"))
        reg.unregister("ceo")
        assert not reg.has("ceo")

    def test_get_team(self):
        reg = AgentRegistry()
        reg.register(self._make_spec("a1", "team_a"))
        reg.register(self._make_spec("a2", "team_a"))
        reg.register(self._make_spec("b1", "team_b"))
        team_a = reg.get_team("team_a")
        assert len(team_a) == 2
        assert all(a.team == "team_a" for a in team_a)

    def test_get_lead(self):
        reg = AgentRegistry()
        reg.register(self._make_spec("mgr", "dev", "lead"))
        reg.register(self._make_spec("worker", "dev", "member"))
        lead = reg.get_lead("dev")
        assert lead is not None
        assert lead.name == "mgr"

    def test_get_lead_none(self):
        reg = AgentRegistry()
        reg.register(self._make_spec("worker", "dev"))
        assert reg.get_lead("dev") is None

    def test_get_teams(self):
        reg = AgentRegistry()
        reg.register(self._make_spec("a", "alpha"))
        reg.register(self._make_spec("b", "beta"))
        reg.register(self._make_spec("c", "alpha"))
        assert reg.get_teams() == ["alpha", "beta"]

    def test_list_as_models(self):
        reg = AgentRegistry()
        reg.register(self._make_spec("ceo", "exec"))
        reg.register(self._make_spec("dev", "dev"))
        models = reg.list_as_models()
        assert len(models) == 2
        ids = [m["id"] for m in models]
        assert "agent/ceo" in ids
        assert "agent/dev" in ids

    def test_len_and_iter(self):
        reg = AgentRegistry()
        reg.register(self._make_spec("a"))
        reg.register(self._make_spec("b"))
        assert len(reg) == 2
        names = [s.name for s in reg]
        assert "a" in names and "b" in names


# --- TransferToAgentTool tests ---


class TestTransferTool:
    def test_name_and_description(self):
        tool = TransferToAgentTool("dev_manager", "Handle development tasks")
        assert tool.name == "transfer_to_dev_manager"
        assert "dev_manager" in tool.description

    def test_parameters_schema(self):
        tool = TransferToAgentTool("ceo")
        params = tool.parameters
        assert params["type"] == "object"
        assert "reason" in params["properties"]
        assert "context" in params["properties"]
        assert "reason" in params["required"]

    def test_execute_returns_handoff_signal(self):
        tool = TransferToAgentTool("dev_manager")
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(reason="needs code review", context="PR #42")
        )
        assert result.startswith(HANDOFF_SIGNAL)
        assert "dev_manager" in result
        assert "needs code review" in result

    def test_to_schema(self):
        tool = TransferToAgentTool("ceo")
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "transfer_to_ceo"


class TestHandoffSignalParsing:
    def test_is_handoff_signal(self):
        assert is_handoff_signal("__HANDOFF__:ceo:reason:ctx")
        assert not is_handoff_signal("normal text")
        assert not is_handoff_signal("")

    def test_parse_handoff_signal(self):
        target, reason, ctx = parse_handoff_signal(
            "__HANDOFF__:dev_manager:needs help:user wants code"
        )
        assert target == "dev_manager"
        assert reason == "needs help"
        assert ctx == "user wants code"

    def test_parse_minimal_signal(self):
        target, reason, ctx = parse_handoff_signal("__HANDOFF__:ceo")
        assert target == "ceo"
        assert reason == ""
        assert ctx == ""


# --- Loader tests ---


class TestLoader:
    def _create_team_dir(self, base: Path, team: str, agents_yaml: str, prompts: dict[str, str] | None = None):
        team_dir = base / team
        team_dir.mkdir(parents=True)
        (team_dir / "agents.yaml").write_text(agents_yaml)
        if prompts:
            for fname, content in prompts.items():
                (team_dir / fname).write_text(content)

    def test_load_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_team_dir(
                base,
                "executive",
                """
agents:
  - name: ceo
    role: lead
    description: "The CEO"
    tools: [web_search]
    handoffs: [dev_manager]
    system_prompt: ceo.md
""",
                {"ceo.md": "You are the CEO."},
            )

            reg = load_agents(base)
            assert len(reg) == 1
            spec = reg.get("ceo")
            assert spec is not None
            assert spec.team == "executive"
            assert spec.role == "lead"
            assert spec.tools == ("web_search",)
            assert spec.handoffs == ("dev_manager",)
            assert spec.system_prompt == "You are the CEO."

    def test_load_multiple_teams(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_team_dir(
                base,
                "executive",
                'agents:\n  - name: ceo\n    description: "CEO"\n    role: lead\n',
            )
            self._create_team_dir(
                base,
                "development",
                'agents:\n  - name: dev_manager\n    description: "Dev Lead"\n    role: lead\n',
            )
            reg = load_agents(base)
            assert len(reg) == 2
            assert reg.has("ceo")
            assert reg.has("dev_manager")

    def test_load_missing_dir(self):
        reg = load_agents(Path("/nonexistent/path"))
        assert len(reg) == 0

    def test_load_missing_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_team_dir(
                base,
                "team",
                'agents:\n  - name: bot\n    description: "Bot"\n    system_prompt: missing.md\n',
            )
            reg = load_agents(base)
            spec = reg.get("bot")
            assert spec is not None
            assert spec.system_prompt == ""
