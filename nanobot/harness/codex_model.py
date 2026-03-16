"""LangChain BaseChatModel wrapper for OpenAI Codex OAuth provider.

This wraps Joel's codex OAuth flow (chatgpt.com/backend-api/codex/responses)
as a LangChain-compatible chat model so deepagents can use it natively.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from loguru import logger


class CodexChatModel(BaseChatModel):
    """LangChain chat model backed by OpenAI Codex OAuth."""

    model_name: str = "gpt-5.1-codex"

    @property
    def _llm_type(self) -> str:
        return "codex-oauth"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync wrapper — runs async codex call in event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — use a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self._agenerate(messages, stop=stop, **kwargs),
                )
                return future.result(timeout=120)
        else:
            return asyncio.run(
                self._agenerate(messages, stop=stop, **kwargs)
            )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate via codex OAuth endpoint."""
        from nanobot.providers.openai_codex_provider import (
            OpenAICodexProvider,
        )

        # Convert LangChain messages to OpenAI format
        oai_messages = _langchain_to_openai(messages)

        # Extract tool definitions from kwargs (deepagents passes them)
        tools = kwargs.get("tools")
        oai_tools = None
        if tools:
            oai_tools = _convert_lc_tools(tools)

        provider = OpenAICodexProvider(default_model=f"openai-codex/{self.model_name}")
        response = await provider.chat(
            messages=oai_messages,
            tools=oai_tools,
            model=f"openai-codex/{self.model_name}",
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
        )

        # Convert response to LangChain format
        tool_calls = []
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.name,
                        args=tc.arguments if isinstance(tc.arguments, dict) else {},
                        id=tc.id or "call_0",
                    )
                )

        ai_message = AIMessage(
            content=response.content or "",
            tool_calls=tool_calls,
        )

        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Bind tools to this model for function calling."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tool_defs = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_defs.append(tool)
            else:
                try:
                    tool_defs.append(convert_to_openai_tool(tool))
                except Exception:
                    # Fallback: manually extract what we can
                    tool_defs.append({
                        "type": "function",
                        "function": {
                            "name": getattr(tool, "name", "unknown"),
                            "description": getattr(tool, "description", "") or "",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    })

        return self.bind(tools=tool_defs, **kwargs)


def _langchain_to_openai(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to OpenAI chat format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", "call_0"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id or "call_0",
                "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
            })
        else:
            # Fallback
            result.append({"role": "user", "content": str(msg.content)})
    return result


def _convert_lc_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert LangChain tool format to OpenAI function calling format."""
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            result.append(tool)
        elif "function" in tool:
            result.append({"type": "function", **tool})
        elif "name" in tool:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
    return result
