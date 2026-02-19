"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

import asyncio
import atexit
import base64
import os
import platform
import re
import signal
import subprocess
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

SCREENSHOTS_DIR = Path.home() / ".nanobot" / "workspace" / "screenshots"

# Playwright tools that interact with the page (should activate Chromium window)
_INTERACTIVE_TOOLS = frozenset({
    "browser_navigate", "browser_click", "browser_type", "browser_hover",
    "browser_drag", "browser_fill_form", "browser_select_option", "browser_press_key",
    "browser_snapshot", "browser_take_screenshot", "browser_scroll",
})

# Patterns that indicate stale DOM references
_STALE_PATTERNS = re.compile(
    r"element not found|stale|target closed|no element found|frame was detached|"
    r"execution context was destroyed|node is detached|element is not attached",
    re.IGNORECASE,
)


def _get_all_descendant_pids(parent_pid: int) -> list[int]:
    """Get all descendant PIDs of a process (children, grandchildren, etc.)."""
    descendants = []
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for pid_str in result.stdout.strip().split('\n'):
                if pid_str:
                    pid = int(pid_str)
                    descendants.append(pid)
                    descendants.extend(_get_all_descendant_pids(pid))
    except Exception:
        pass
    return descendants


def _kill_all_children():
    """Kill all child processes of the current process. Called on exit/signal."""
    my_pid = os.getpid()
    children = _get_all_descendant_pids(my_pid)
    if not children:
        return
    logger.info(f"Killing {len(children)} MCP child processes...")
    for pid in reversed(children):  # Kill deepest children first
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT by killing all children before exiting."""
    _kill_all_children()
    # Re-raise the signal with default handler so the process actually exits
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup handlers — ensures MCP subprocesses die with the parent
atexit.register(_kill_all_children)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


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
                command=self.cfg.command, args=self.cfg.args, env=self.cfg.env or None,
                cwd=self.cfg.cwd or None,
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
        self._server_name = server_name
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._pending_images: list[str] = []  # base64 data URLs for vision injection

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def _activate_chromium(self) -> None:
        """Bring Chromium to the front on macOS (fire-and-forget)."""
        if platform.system() != "Darwin":
            return
        try:
            subprocess.Popen(
                ["osascript", "-e", 'tell application "Chromium" to activate'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    async def _execute_inner(self, **kwargs: Any) -> str:
        """Execute the MCP tool and process results including images."""
        from mcp import types

        result = await self._conn.call_tool(self._original_name, arguments=kwargs)
        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            elif isinstance(block, types.ImageContent):
                # Save screenshot to disk and add markdown link
                try:
                    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time() * 1000)
                    filename = f"{ts}.png"
                    filepath = SCREENSHOTS_DIR / filename
                    img_bytes = base64.b64decode(block.data)
                    filepath.write_bytes(img_bytes)
                    parts.append(f"![screenshot](http://localhost:18790/screenshots/{filename})")
                    # Store base64 data URL for vision injection
                    data_url = f"data:{block.mimeType};base64,{block.data}"
                    self._pending_images.append(data_url)
                    logger.info(f"Screenshot saved: {filepath} ({len(img_bytes)} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to save screenshot: {e}")
                    parts.append("(screenshot captured but failed to save)")
            else:
                parts.append(str(block))
        return "\n".join(parts) or "(no output)"

    async def execute(self, **kwargs: Any) -> str:
        self._pending_images.clear()

        # Auto-activate Chromium window for interactive Playwright tools
        if self._server_name == "playwright" and self._original_name in _INTERACTIVE_TOOLS:
            self._activate_chromium()

        result = await self._execute_inner(**kwargs)

        # Stale DOM retry: if a Playwright tool (not snapshot/navigate) returns stale error,
        # take a fresh snapshot then retry once
        if (
            self._server_name == "playwright"
            and self._original_name not in ("browser_snapshot", "browser_navigate")
            and _STALE_PATTERNS.search(result)
        ):
            logger.warning(f"Stale DOM detected for {self._original_name}, refreshing snapshot and retrying...")
            try:
                await self._conn.call_tool("browser_snapshot", arguments={})
            except Exception as e:
                logger.warning(f"Snapshot refresh failed: {e}")
            self._pending_images.clear()
            result = await self._execute_inner(**kwargs)

        return result


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
