"""Tool factory for building per-agent tool sets."""

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agents.spec import AgentSpec
from nanobot.agents.registry import AgentRegistry
from nanobot.agents.tools.transfer import TransferToAgentTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.base import Tool


# Map of tool name → (module_path, class_name, required_kwargs)
# These are the tools from nanobot.agent.tools that agents can reference by name
TOOL_CATALOG: dict[str, tuple[str, str]] = {
    "web_search": ("nanobot.agent.tools.web", "WebSearchTool"),
    "web_fetch": ("nanobot.agent.tools.web", "WebFetchTool"),
    "file_read": ("nanobot.agent.tools.filesystem", "ReadFileTool"),
    "file_write": ("nanobot.agent.tools.filesystem", "WriteFileTool"),
    "file_edit": ("nanobot.agent.tools.filesystem", "EditFileTool"),
    "list_dir": ("nanobot.agent.tools.filesystem", "ListDirTool"),
    "shell": ("nanobot.agent.tools.shell", "ExecTool"),
    "exec": ("nanobot.agent.tools.shell", "ExecTool"),  # alias for shell
    "email_send": ("nanobot.agent.tools.email_tools", "EmailSendTool"),
    "email_read": ("nanobot.agent.tools.email_tools", "EmailReadTool"),
    "image_gen": ("nanobot.agent.tools.image_gen", "ImageGenTool"),
    "tts": ("nanobot.agent.tools.tts", "TTSTool"),
    "postiz": ("nanobot.agent.tools.postiz", "PostizTool"),
    "article_image": ("nanobot.agent.tools.article_image", "ArticleImageTool"),
}


class ToolFactory:
    """
    Creates per-agent tool sets from AgentSpec definitions.

    Each agent gets:
    1. The tools listed in its spec (from TOOL_CATALOG)
    2. TransferToAgentTool instances for each allowed handoff target
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        workspace: Path | None = None,
        tool_config: dict[str, Any] | None = None,
    ):
        self._agent_registry = agent_registry
        self._workspace = workspace or Path.cwd()
        self._tool_config = tool_config or {}

    def build_tools(self, spec: AgentSpec) -> ToolRegistry:
        """
        Build a complete ToolRegistry for an agent based on its spec.

        Args:
            spec: The agent specification.

        Returns:
            ToolRegistry with all tools the agent is allowed to use.
        """
        registry = ToolRegistry()

        # 1. Add capability tools from spec
        for tool_name in spec.tools:
            tool = self._create_tool(tool_name)
            if tool:
                registry.register(tool)
            else:
                logger.warning(
                    f"Unknown tool '{tool_name}' for agent '{spec.name}', skipping"
                )

        # 2. Add transfer tools for each handoff target
        for target_name in spec.handoffs:
            target_spec = self._agent_registry.get(target_name)
            description = (
                target_spec.description if target_spec else f"Agent: {target_name}"
            )
            transfer_tool = TransferToAgentTool(target_name, description)
            registry.register(transfer_tool)

        logger.debug(
            f"Built {len(registry)} tools for agent '{spec.name}': "
            f"{registry.tool_names}"
        )
        return registry

    def _create_tool(self, tool_name: str) -> Tool | None:
        """
        Create a tool instance by name from the catalog.

        Uses lazy importing to avoid importing unused tool modules.
        """
        if tool_name not in TOOL_CATALOG:
            return None

        module_path, class_name = TOOL_CATALOG[tool_name]

        try:
            import importlib

            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)

            # Build constructor kwargs based on tool type and config
            kwargs = self._get_tool_kwargs(tool_name)
            return tool_class(**kwargs)

        except Exception as e:
            logger.error(f"Failed to create tool '{tool_name}': {e}")
            return None

    def _get_tool_kwargs(self, tool_name: str) -> dict[str, Any]:
        """Get constructor kwargs for a tool based on config."""
        kwargs: dict[str, Any] = {}

        # Workspace-aware tools
        if tool_name in ("file_read", "file_write", "file_edit", "list_dir"):
            if self._tool_config.get("restrict_to_workspace"):
                kwargs["allowed_dir"] = self._workspace

        if tool_name in ("shell", "exec"):
            kwargs["working_dir"] = str(self._workspace)
            shell_config = self._tool_config.get("shell", {})
            if "timeout" in shell_config:
                kwargs["timeout"] = shell_config["timeout"]

        if tool_name == "web_search":
            api_key = self._tool_config.get("brave_api_key")
            if api_key:
                kwargs["api_key"] = api_key

        return kwargs
