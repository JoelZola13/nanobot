"""Bridge nanobot tools → LangChain BaseTool instances.

Every nanobot tool (MCP, email, web, filesystem, shell, etc.) gets wrapped
as a LangChain StructuredTool so deepagents can invoke them natively.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import StructuredTool
from loguru import logger

from nanobot.agent.tools.base import Tool as NanobotTool
from nanobot.agent.tools.registry import ToolRegistry


def _json_schema_to_langchain_args(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema parameters dict to a flat field description dict.

    LangChain StructuredTool.from_function uses pydantic under the hood,
    but we can also pass raw JSON Schema as `args_schema` via a dynamic model.
    """
    return schema


def _create_langchain_tool(nanobot_tool: NanobotTool) -> StructuredTool:
    """Wrap a single nanobot Tool as a LangChain StructuredTool."""

    # Get the JSON Schema for the tool's parameters
    tool_def = nanobot_tool.to_schema()
    params_schema = tool_def.get("function", {}).get("parameters", {})

    # Build a dynamic pydantic model from the JSON Schema
    from pydantic import create_model, Field
    from typing import Optional

    properties = params_schema.get("properties", {})
    required = set(params_schema.get("required", []))

    # Build field definitions for create_model
    field_definitions: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        description = prop_schema.get("description", "")
        default = ... if prop_name in required else None

        # Map JSON Schema types to Python types
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        python_type = type_map.get(prop_type, str)

        if prop_name in required:
            field_definitions[prop_name] = (python_type, Field(description=description))
        else:
            field_definitions[prop_name] = (
                Optional[python_type],
                Field(default=None, description=description),
            )

    # Create dynamic pydantic model
    if field_definitions:
        ArgsModel = create_model(
            f"{nanobot_tool.name}_Args",
            **field_definitions,
        )
    else:
        ArgsModel = None

    tool_name = nanobot_tool.name
    tool_description = tool_def.get("function", {}).get("description", nanobot_tool.description or f"Tool: {tool_name}")

    # Truncate description if too long (LangChain has limits)
    if len(tool_description) > 1024:
        tool_description = tool_description[:1020] + "..."

    async def _arun(**kwargs: Any) -> str:
        """Async execution — calls the nanobot tool's execute method."""
        try:
            result = await nanobot_tool.execute(**kwargs)
            if isinstance(result, dict):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' error: {e}")
            return f"[TOOL_ERROR] {tool_name}: {e}"

    def _run(**kwargs: Any) -> str:
        """Sync execution — wraps async in event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, nanobot_tool.execute(**kwargs)
                    ).result()
            else:
                result = asyncio.run(nanobot_tool.execute(**kwargs))

            if isinstance(result, dict):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' error: {e}")
            return f"[TOOL_ERROR] {tool_name}: {e}"

    tool = StructuredTool(
        name=tool_name,
        description=tool_description,
        func=_run,
        coroutine=_arun,
        args_schema=ArgsModel,
    )
    return tool


def bridge_registry(registry: ToolRegistry) -> list[StructuredTool]:
    """Convert an entire nanobot ToolRegistry into LangChain tools.

    Args:
        registry: The nanobot ToolRegistry containing all registered tools.

    Returns:
        List of LangChain StructuredTool instances.
    """
    tools: list[StructuredTool] = []
    for name in registry.tool_names:
        nanobot_tool = registry.get(name)
        if nanobot_tool is None:
            continue
        try:
            lc_tool = _create_langchain_tool(nanobot_tool)
            tools.append(lc_tool)
            logger.debug(f"Bridged nanobot tool → LangChain: {name}")
        except Exception as e:
            logger.warning(f"Failed to bridge tool '{name}': {e}")
    logger.info(f"Bridged {len(tools)} nanobot tools to LangChain")
    return tools


def bridge_tools(nanobot_tools: dict[str, NanobotTool]) -> list[StructuredTool]:
    """Convert a dict of nanobot tools to LangChain tools.

    Args:
        nanobot_tools: Dict mapping tool name → NanobotTool instance.

    Returns:
        List of LangChain StructuredTool instances.
    """
    tools: list[StructuredTool] = []
    for name, tool in nanobot_tools.items():
        try:
            lc_tool = _create_langchain_tool(tool)
            tools.append(lc_tool)
        except Exception as e:
            logger.warning(f"Failed to bridge tool '{name}': {e}")
    return tools
