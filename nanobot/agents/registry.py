"""Agent registry for managing agent specifications."""

from typing import Any

from nanobot.agents.spec import AgentSpec


class AgentRegistry:
    """
    Registry for agent specifications.

    Modeled after ToolRegistry — allows dynamic registration,
    lookup by name/team, and API model listing.
    """

    def __init__(self):
        self._agents: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        """Register an agent specification."""
        self._agents[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Unregister an agent by name."""
        self._agents.pop(name, None)

    def get(self, name: str) -> AgentSpec | None:
        """Get an agent spec by name."""
        return self._agents.get(name)

    def has(self, name: str) -> bool:
        """Check if an agent is registered."""
        return name in self._agents

    def get_team(self, team: str) -> list[AgentSpec]:
        """Get all agents in a team."""
        return [a for a in self._agents.values() if a.team == team]

    def get_lead(self, team: str) -> AgentSpec | None:
        """Get the lead agent for a team."""
        for a in self._agents.values():
            if a.team == team and a.is_lead:
                return a
        return None

    def get_teams(self) -> list[str]:
        """Get list of unique team names."""
        return sorted(set(a.team for a in self._agents.values()))

    def list_as_models(self) -> list[dict[str, Any]]:
        """Get all agents as OpenAI /v1/models entries."""
        return [a.to_model_entry() for a in self._agents.values()]

    @property
    def agent_names(self) -> list[str]:
        """Get list of registered agent names."""
        return list(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __iter__(self):
        return iter(self._agents.values())
