"""Tool factory for building per-agent tool sets."""

import fnmatch
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.agents.spec import AgentSpec
from nanobot.agents.registry import AgentRegistry
from nanobot.agents.tools.transfer import TransferToAgentTool
from nanobot.agents.tools.delegate import DelegateToAgentTool
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
    "exec": ("nanobot.agent.tools.shell", "ExecTool"),  # alias used in agent YAML specs
    "email_send": ("nanobot.agent.tools.email_tools", "EmailSendTool"),
    "email_read": ("nanobot.agent.tools.email_tools", "EmailReadTool"),
    "image_gen": ("nanobot.agent.tools.image_gen", "QwenImageGenTool"),
    "tts": ("nanobot.agent.tools.tts", "QwenTTSTool"),
    "postiz": ("nanobot.agent.tools.postiz", "PostizPublishTool"),
    "article_image": ("nanobot.agent.tools.article_image", "ArticleImageTool"),
    # ── Search & research ──
    "news_search": ("nanobot.agent.tools.search", "NewsSearchTool"),
    "academic_search": ("nanobot.agent.tools.search", "AcademicSearchTool"),
    "image_search": ("nanobot.agent.tools.image_search", "ImageSearchTool"),
    "grants_database": ("nanobot.agent.tools.grants", "GrantsDatabaseTool"),
    # ── Scraping & data extraction ──
    "web_scrape": ("nanobot.agent.tools.scraping", "WebScrapeTool"),
    "html_parser": ("nanobot.agent.tools.scraping", "HtmlParserTool"),
    "data_extractor": ("nanobot.agent.tools.scraping", "DataExtractorTool"),
    # ── Utilities ──
    "calculator": ("nanobot.agent.tools.calculator", "CalculatorTool"),
    "crypto_api": ("nanobot.agent.tools.crypto", "CryptoApiTool"),
    "invoice_generator": ("nanobot.agent.tools.finance", "InvoiceGeneratorTool"),
    # ── Documents & spreadsheets ──
    "spreadsheet_read": ("nanobot.agent.tools.spreadsheet", "SpreadsheetReadTool"),
    "spreadsheet_write": ("nanobot.agent.tools.spreadsheet", "SpreadsheetWriteTool"),
    "document_editor": ("nanobot.agent.tools.documents", "DocumentEditorTool"),
    "calendar_read": ("nanobot.agent.tools.documents", "CalendarReadTool"),
    "edit_file": ("nanobot.agent.tools.filesystem", "EditFileTool"),
}


class ToolFactory:
    """
    Creates per-agent tool sets from AgentSpec definitions.

    Each agent gets:
    1. The tools listed in its spec (from TOOL_CATALOG)
    2. DelegateToAgentTool (for lead agents delegating down) or
       TransferToAgentTool (for member agents handing up) for each handoff target
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        workspace: Path | None = None,
        tool_config: dict[str, Any] | None = None,
        provider: LLMProvider | None = None,
        mcp_tools: dict[str, Tool] | None = None,
    ):
        self._agent_registry = agent_registry
        self._workspace = workspace or Path.cwd()
        self._tool_config = tool_config or {}
        self._provider = provider
        # MCP tools keyed by their full name (e.g. "mcp_airtable_list_records")
        self._mcp_tools: dict[str, Tool] = mcp_tools or {}

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
            # Wildcard MCP patterns like "mcp_airtable_*" or "mcp_google-calendar_*"
            if "*" in tool_name and tool_name.startswith("mcp_"):
                matched = 0
                for mcp_name, mcp_tool in self._mcp_tools.items():
                    if fnmatch.fnmatch(mcp_name, tool_name):
                        registry.register(mcp_tool)
                        matched += 1
                if matched == 0:
                    logger.warning(
                        f"MCP wildcard '{tool_name}' matched 0 tools for agent '{spec.name}'"
                    )
                else:
                    logger.debug(
                        f"MCP wildcard '{tool_name}' matched {matched} tools for '{spec.name}'"
                    )
                continue

            tool = self._create_tool(tool_name)
            if tool:
                registry.register(tool)
            else:
                logger.warning(
                    f"Unknown tool '{tool_name}' for agent '{spec.name}', skipping"
                )

        # 2. Add handoff tools for each target
        # Lead agents get delegate_to_* (runs sub-agent, returns result)
        # for all downward handoffs. Upward handoffs (to ceo) stay as transfer_to_*.
        # Member agents always get transfer_to_* (Orchestrator handles the switch).
        for target_name in spec.handoffs:
            target_spec = self._agent_registry.get(target_name)
            description = (
                target_spec.description if target_spec else f"Agent: {target_name}"
            )
            if spec.is_lead and target_name != "ceo" and self._provider is not None:
                delegate_tool = DelegateToAgentTool(
                    target_name,
                    description,
                    self._agent_registry,
                    self._provider,
                    self,
                )
                registry.register(delegate_tool)
            else:
                transfer_tool = TransferToAgentTool(target_name, description)
                registry.register(transfer_tool)

        logger.debug(
            f"Built {len(registry)} tools for agent '{spec.name}': "
            f"{registry.tool_names}"
        )
        return registry

    def _create_tool(self, tool_name: str) -> Tool | None:
        """
        Create a tool instance by name from the catalog or MCP pool.

        Uses lazy importing to avoid importing unused tool modules.
        Falls back to MCP tools if not found in the static catalog.
        """
        # Check MCP tools first for exact matches (e.g. "mcp_airtable_list_records")
        if tool_name in self._mcp_tools:
            return self._mcp_tools[tool_name]

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

        if tool_name == "shell":
            kwargs["working_dir"] = str(self._workspace)
            shell_config = self._tool_config.get("shell", {})
            if "timeout" in shell_config:
                kwargs["timeout"] = shell_config["timeout"]

        if tool_name == "web_search":
            api_key = self._tool_config.get("brave_api_key")
            if api_key:
                kwargs["api_key"] = api_key

        if tool_name == "postiz":
            from nanobot.config.schema import PostizConfig
            raw = self._tool_config.get("postiz", {})
            kwargs["config"] = PostizConfig(**raw) if isinstance(raw, dict) else raw

        if tool_name in ("news_search", "image_search"):
            api_key = self._tool_config.get("brave_api_key")
            if api_key:
                kwargs["api_key"] = api_key

        if tool_name in ("spreadsheet_read", "spreadsheet_write", "document_editor"):
            if self._tool_config.get("restrict_to_workspace"):
                kwargs["allowed_dir"] = self._workspace

        if tool_name == "tts":
            kwargs["audio_dir"] = self._workspace / "remotion" / "public" / "audio"

        # Email tools need IMAP/SMTP credentials from config
        if tool_name == "email_read":
            email_cfg = self._tool_config.get("email", {})
            kwargs["imap_host"] = email_cfg.get("imap_host", "")
            kwargs["imap_port"] = email_cfg.get("imap_port", 993)
            kwargs["username"] = email_cfg.get("imap_username", "")
            kwargs["password"] = email_cfg.get("imap_password", "")

        if tool_name == "email_send":
            email_cfg = self._tool_config.get("email", {})
            kwargs["smtp_host"] = email_cfg.get("smtp_host", "")
            kwargs["smtp_port"] = email_cfg.get("smtp_port", 587)
            kwargs["username"] = email_cfg.get("smtp_username", "")
            kwargs["password"] = email_cfg.get("smtp_password", "")
            kwargs["from_addr"] = email_cfg.get("from_addr", email_cfg.get("smtp_username", ""))

        return kwargs
