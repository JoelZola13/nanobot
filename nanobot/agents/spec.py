"""Agent specification dataclass."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    """
    Immutable specification for an agent.

    Loaded from YAML definitions and used by the orchestrator
    to instantiate agent runtime loops.
    """

    name: str
    team: str
    description: str
    role: str = "member"  # "lead" or "member"
    model: str = "default"  # LLM model or "default" to use system default
    tools: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()  # Agent names this agent can hand off to
    system_prompt: str = ""  # Loaded from .md file
    max_iterations: int = 25
    temperature: float = 0.7
    max_tokens: int = 4096

    @property
    def is_lead(self) -> bool:
        """Whether this agent is a team lead."""
        return self.role == "lead"

    @property
    def qualified_name(self) -> str:
        """Fully qualified name: team/agent_name."""
        return f"{self.team}/{self.name}"

    @property
    def model_id(self) -> str:
        """Model ID for API exposure: agent/name."""
        return f"agent/{self.name}"

    def to_model_entry(self) -> dict:
        """Convert to OpenAI /v1/models list entry."""
        return {
            "id": self.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "nanobot",
            "permission": [],
            "root": self.model_id,
            "parent": None,
            "metadata": {
                "team": self.team,
                "role": self.role,
                "description": self.description,
            },
        }
