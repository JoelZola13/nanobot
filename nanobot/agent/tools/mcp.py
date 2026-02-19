"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


class MCPServerConnection:
    """Manages a single MCP server connection with auto-reconnect."""

    def __init__(self, name: str, cfg):
        self.name = name
        self.cfg = cfg
        self.session = None
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        if self.cfg.command:
            params = StdioServerParameters(
                command=self.cfg.command, args=self.cfg.args, env=self.cfg.env or None
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif self.cfg.url:
            from mcp.client.streamable_http import streamable_http_client
            read, write, _ = await self._stack.enter_async_context(
                streamable_http_client(self.cfg.url)
            )
        else:
            raise ValueError(f"MCP server '{self.name}': no command or url configured")

        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        logger.info(f"MCP server '{self.name}': connected")

    async def reconnect(self) -> None:
        """Tear down and re-establish the connection."""
        async with self._lock:
            logger.warning(f"MCP server '{self.name}': reconnecting...")
            await self.close()
            await self.connect()

    async def close(self) -> None:
        """Close the connection."""
        if self._stack:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None
            self.session = None

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool with auto-reconnect on failure."""
        for attempt in range(2):
            try:
                if self.session is None:
                    await self.reconnect()
                return await self.session.call_tool(tool_name, arguments=arguments)
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"MCP server '{self.name}': call to '{tool_name}' failed ({e}), reconnecting...")
                    await self.reconnect()
                else:
                    raise


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a nanobot Tool."""

    def __init__(self, conn: MCPServerConnection, server_name: str, tool_def):
        self._conn = conn
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types
        result = await self._conn.call_tool(self._original_name, arguments=kwargs)
        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts) or "(no output)"


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry, stack: AsyncExitStack
) -> None:
    """Connect to configured MCP servers and register their tools."""
    for name, cfg in mcp_servers.items():
        try:
            conn = MCPServerConnection(name, cfg)
            await conn.connect()

            tools = await conn.session.list_tools()
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(conn, name, tool_def)
                registry.register(wrapper)
                logger.debug(f"MCP: registered tool '{wrapper.name}' from server '{name}'")

            logger.info(f"MCP server '{name}': connected, {len(tools.tools)} tools registered")
        except Exception as e:
            logger.error(f"MCP server '{name}': failed to connect: {e}")
