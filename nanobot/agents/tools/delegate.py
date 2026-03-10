"""Delegate tool — runs a sub-agent internally and returns its response."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agents.instance import AgentInstance

if TYPE_CHECKING:
    from nanobot.agents.registry import AgentRegistry
    from nanobot.agents.factory import ToolFactory
    from nanobot.providers.base import LLMProvider


class DelegateToAgentTool(Tool):
    """
    Runs a sub-agent's full execution loop and returns its text response.

    Unlike TransferToAgentTool (which signals a handoff to the Orchestrator
    and the calling agent loses control), DelegateToAgentTool keeps the
    calling agent active. The sub-agent runs internally — including its own
    tool calls — and the final text response comes back as this tool's result.

    This lets lead agents (CEO, managers) gather results from their teams.
    """

    def __init__(
        self,
        target_name: str,
        target_description: str,
        agent_registry: AgentRegistry,
        provider: LLMProvider,
        tool_factory: ToolFactory,
    ):
        self._target_name = target_name
        self._target_description = target_description or f"Delegate to {target_name}"
        self._agent_registry = agent_registry
        self._provider = provider
        self._tool_factory = tool_factory
        # Set per-request by AgentInstance.run() before execute() is called
        self._on_progress = None

    @property
    def name(self) -> str:
        return f"delegate_to_{self._target_name}"

    @property
    def description(self) -> str:
        return (
            f"Delegate a task to {self._target_name} and get their response. "
            f"{self._target_description}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Clear description of what you need this agent to do. "
                        "Be specific about the desired output."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Additional context, background information, or data "
                        "the agent needs to complete the task."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(self, **kwargs: Any) -> str:
        task = kwargs.get("task", "")
        context = kwargs.get("context", "")

        if not task:
            return "Error: 'task' parameter is required."

        # Resolve the target agent spec
        spec = self._agent_registry.get(self._target_name)
        if spec is None:
            return f"Error: Agent '{self._target_name}' not found in registry."

        # Build tools for the sub-agent
        tools = self._tool_factory.build_tools(spec)

        # Build messages with system prompt + the delegated task
        user_content = task
        if context:
            user_content += f"\n\nContext:\n{context}"

        system_prompt = spec.system_prompt or (
            f"You are the {spec.name} agent. {spec.description}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Resolve model
        model = None if spec.model == "default" else spec.model

        # Run the sub-agent
        instance = AgentInstance(spec, self._provider, tools)

        logger.info(
            f"Delegate: running sub-agent '{self._target_name}' "
            f"(task: {task[:80]}...)"
        )

        # Stream delegation progress
        on_progress = self._on_progress
        if on_progress:
            await on_progress(
                f"📋 Delegating to **{self._target_name}**..."
            )

        try:
            result = await instance.run(messages, model=model, on_progress=on_progress)
        except Exception as e:
            logger.error(
                f"Delegate to '{self._target_name}' failed: {e}"
            )
            return f"Error: Sub-agent '{self._target_name}' failed — {e}"

        if result.is_error:
            return f"Sub-agent '{self._target_name}' encountered an error: {result.content}"

        if result.is_handoff:
            # Sub-agent wants to hand off further — return the content
            # and let the calling agent decide what to do
            return (
                f"Sub-agent '{self._target_name}' requested a handoff "
                f"instead of completing the task. Response: {result.content}"
            )

        return result.content
