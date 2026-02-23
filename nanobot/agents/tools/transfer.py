"""Transfer tool for agent handoffs."""

from typing import Any

from nanobot.agent.tools.base import Tool


# Handoff signal prefix — intercepted by the Orchestrator
HANDOFF_SIGNAL = "__HANDOFF__"


class TransferToAgentTool(Tool):
    """
    Tool that enables agent-to-agent handoffs.

    Following the OpenAI Agents SDK pattern, each agent gets
    transfer_to_<target> tools for its allowed handoff targets.
    When called, it returns a signal string that the Orchestrator
    intercepts to perform the actual handoff.
    """

    def __init__(self, target_name: str, target_description: str = ""):
        self._target_name = target_name
        self._target_description = target_description or f"Transfer to {target_name}"

    @property
    def name(self) -> str:
        return f"transfer_to_{self._target_name}"

    @property
    def description(self) -> str:
        return (
            f"Hand off the conversation to the '{self._target_name}' agent. "
            f"{self._target_description}. "
            "Use this when the user's request falls within this agent's expertise."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you're handing off to this agent.",
                },
                "context": {
                    "type": "string",
                    "description": "Summary of the conversation so far and what the target agent needs to know.",
                },
            },
            "required": ["reason"],
        }

    async def execute(self, **kwargs: Any) -> str:
        reason = kwargs.get("reason", "")
        context = kwargs.get("context", "")
        return f"{HANDOFF_SIGNAL}:{self._target_name}:{reason}:{context}"


def is_handoff_signal(text: str) -> bool:
    """Check if a tool result is a handoff signal."""
    return text.startswith(f"{HANDOFF_SIGNAL}:")


def parse_handoff_signal(text: str) -> tuple[str, str, str]:
    """
    Parse a handoff signal into (target_agent, reason, context).

    Returns:
        Tuple of (target_name, reason, context).
    """
    parts = text.split(":", 3)
    # parts[0] = "__HANDOFF__"
    target = parts[1] if len(parts) > 1 else ""
    reason = parts[2] if len(parts) > 2 else ""
    context = parts[3] if len(parts) > 3 else ""
    return target, reason, context
