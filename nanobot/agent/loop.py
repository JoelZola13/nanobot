"""Agent loop: the core processing engine."""

import asyncio
import base64
import mimetypes
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
import re
import tempfile
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.postiz import PostizPublishTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager

_API_HOST = "http://localhost:18790"
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        postiz_config: "PostizConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.config.schema import PostizConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self.mem0 = None  # Mem0Store, set externally by api_server
        self.orchestrator = None  # Orchestrator, set externally by api_server for multi-agent routing

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connect_lock = asyncio.Lock()
        self.postiz_config = postiz_config or PostizConfig()
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Social posting tool (Postiz)
        if self.postiz_config.enabled:
            self.tools.register(PostizPublishTool(self.postiz_config))
            logger.info("Postiz social publishing tool registered")
    
    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return

        async with self._mcp_connect_lock:
            if self._mcp_connected or not self._mcp_servers:
                return

            from nanobot.agent.tools.mcp import connect_mcp_servers

            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            try:
                await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

                # Register computer_use tool if Playwright MCP is available
                if any("playwright" in name for name in self.tools.tool_names):
                    try:
                        from nanobot.agent.tools.computer_use import ComputerUseManager, ComputerUseTool
                        manager = ComputerUseManager(self.tools, self.provider, self.model)
                        self.tools.register(ComputerUseTool(manager))
                        logger.info("Computer use tool registered (Playwright MCP available)")
                    except Exception as e:
                        logger.debug(f"Computer use tool not available: {e}")

                self._mcp_connected = True
            except BaseException:
                if self._mcp_stack is not None:
                    try:
                        await self._mcp_stack.aclose()
                    except Exception:
                        pass
                    self._mcp_stack = None
                raise

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _normalize_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize chat/completions messages to a robust internal format."""
        normalized: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant", "tool"}:
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"text", "input_text"}:
                        text = part.get("text") or ""
                        if isinstance(text, str):
                            parts.append({"type": "text", "text": text})
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url")
                        if isinstance(image_url, dict):
                            url = image_url.get("url")
                        else:
                            url = image_url
                        if isinstance(url, str):
                            parts.append({"type": "image_url", "image_url": {"url": url}})
                    elif part.get("type") in {"input_image", "image"}:
                        image_url = part.get("image_url") or part.get("url") or part.get("input_image")
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url")
                        if isinstance(image_url, str):
                            parts.append({"type": "image_url", "image_url": {"url": image_url}})
                    elif part.get("type") in {"input_audio", "audio"}:
                        parts.append(part)
                content = parts
            elif not isinstance(content, str):
                content = json.dumps(content)

            entry: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant" and msg.get("tool_calls"):
                entry["tool_calls"] = msg.get("tool_calls")
            if role == "tool" and msg.get("tool_call_id"):
                entry["tool_call_id"] = msg.get("tool_call_id")

            normalized.append(entry)

        return normalized

    @staticmethod
    def _normalize_audio_extension_from_mime(mime: str | None) -> str:
        """Map MIME type to a temp-file extension for transcription."""
        if not mime:
            return ".wav"
        extension = mimetypes.guess_extension(mime)
        return extension or ".wav"

    @staticmethod
    async def _transcribe_api_audio_part(part: dict[str, Any]) -> str:
        """Transcribe an API audio attachment using the Groq transcription provider."""
        try:
            from nanobot.providers.transcription import GroqTranscriptionProvider
        except Exception:
            logger.exception("Failed to import Groq transcription provider")
            return ""

        provider = GroqTranscriptionProvider()
        if not provider.api_key:
            return ""

        source: str | None = None
        source_is_base64 = False
        fmt: str | None = None
        mime: str | None = None

        audio_data = part.get("input_audio")
        if isinstance(audio_data, dict):
            fmt = audio_data.get("format")
            mime = audio_data.get("mime_type")
            if isinstance(audio_data.get("data"), str):
                source = audio_data["data"]
                source_is_base64 = True
            elif isinstance(audio_data.get("url"), str):
                source = audio_data["url"]

        if source is None:
            audio_url = part.get("audio_url")
            if isinstance(audio_url, dict) and isinstance(audio_url.get("url"), str):
                source = audio_url["url"]
            elif isinstance(audio_url, str):
                source = audio_url

        if source is None and isinstance(part.get("url"), str):
            source = part["url"]

        if source is None:
            return ""

        extension = (
            f".{fmt.lstrip('.')}"
            if isinstance(fmt, str) and fmt
            else AgentLoop._normalize_audio_extension_from_mime(mime)
        )

        if source_is_base64 and source.startswith("data:"):
            if "," in source:
                _, source = source.split(",", 1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp_path = tmp.name

        try:
            if source_is_base64:
                try:
                    audio_bytes = base64.b64decode(source)
                except Exception:
                    logger.warning("Could not decode base64 audio attachment")
                    return ""
                Path(tmp_path).write_bytes(audio_bytes)
            else:
                import httpx

                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.get(source, follow_redirects=True)
                    response.raise_for_status()
                Path(tmp_path).write_bytes(response.content)

            return await provider.transcribe(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def _resolve_audio_attachments(self, messages: list[dict[str, Any]]) -> None:
        """Transcribe inline/audio attachments in API message content."""
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            changed = False
            next_parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"input_audio", "audio"}:
                    changed = True
                    transcript = await self._transcribe_api_audio_part(part)
                    if transcript:
                        next_parts.append({
                            "type": "text",
                            "text": f"[audio attachment: {transcript}]",
                        })
                    else:
                        next_parts.append({
                            "type": "text",
                            "text": "[audio attachment]",
                        })
                    continue

                next_parts.append(part)

            if changed:
                msg["content"] = next_parts

    @staticmethod
    def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
        """Return the last user message text from OpenAI-style messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    texts = [
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
                    ]
                    if texts:
                        return "\n".join(texts)
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                            return "[image attachment]"
                        if isinstance(part, dict) and part.get("type") in {"audio", "input_audio"}:
                            return "[audio attachment]"
                return ""
        return ""

    async def process_openai_messages(
        self,
        messages: list[dict[str, Any]],
        session_key: str = "cli:direct",
        channel: str = "api",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        tool_choice: str = "auto",
        required_tool_name: str | None = None,
        model_override: str | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        reset_session: bool = False,
        max_iterations_override: int | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, list[str], dict[str, int], str]:
        """
        Process a full OpenAI-style message array (chat/completions format).

        This is used by the API server and keeps LibreChat-style clients
        from feeling stateless and truncated.
        """
        await self._connect_mcp()
        normalized = self._normalize_api_messages(messages)
        await self._resolve_audio_attachments(normalized)
        if not normalized:
            if return_usage:
                return "", [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "stop"
            return ""

        user_content = self._extract_last_user_text(normalized)
        if not user_content:
            if return_usage:
                return "", [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "stop"
            return ""

        current_user_message = ""
        for msg in reversed(normalized):
            if msg.get("role") == "user":
                current_user_message = msg.get("content", "")
                break

        session = self.sessions.get_or_create(session_key)
        if reset_session:
            session.clear()
            self.sessions.save(session)

        command = user_content.strip().lower()
        if command == "/new":
            session.clear()
            self.sessions.save(session)
            message = "New session started. Memory consolidation in progress."
            if return_usage:
                return message, [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "stop"
            return message

        if command == "/help":
            message = "🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands"
            if return_usage:
                return message, [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "stop"
            return message

        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        self._set_tool_context(channel, chat_id)

        semantic_memories = None
        if self.mem0:
            try:
                semantic_memories = await self.mem0.search(user_content) or None
            except Exception as e:
                logger.warning(f"Mem0 search error: {e}")

        history = normalized[:-1]
        initial_messages = self.context.build_messages(
            history=history,
            current_message=current_user_message or user_content,
            channel=channel,
            chat_id=chat_id,
            semantic_memories=semantic_memories,
        )

        async def _no_progress(_: str) -> None:
            return

        final_content, tools_used, usage, final_finish_reason = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _no_progress,
            tool_choice=tool_choice,
            required_tool_name=required_tool_name,
            model_override=model_override,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
            max_iterations_override=max_iterations_override,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        session.add_message("user", user_content)
        session.add_message("assistant", final_content, tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        if self.mem0:
            exchange = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": final_content},
            ]
            asyncio.create_task(self.mem0.add(exchange))

        if return_usage:
            return final_content, tools_used, usage, final_finish_reason
        return final_content

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        tool_choice: str = "auto",
        required_tool_name: str | None = None,
        model_override: str | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        max_iterations_override: int | None = None,
    ) -> tuple[str | None, list[str], dict[str, int], str]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.
            on_progress: Optional callback to push intermediate content to the user.

        Returns:
            Tuple of (final_content, tools_used, usage, finish_reason).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        artifact_links: list[str] = []  # Collect media URLs that should be surfaced to users
        effective_model = model_override or self.model
        effective_temperature = self.temperature if temperature_override is None else temperature_override
        effective_max_tokens = self.max_tokens if max_tokens_override is None else max_tokens_override
        iteration_limit = max_iterations_override or self.max_iterations
        final_finish_reason = "stop"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        tools_enabled = tool_choice != "none"
        tool_defs = self.tools.get_definitions() if tools_enabled else None
        if tools_enabled and required_tool_name and tool_defs:
            required_defs = []
            for tool_def in tool_defs:
                fn = tool_def.get("function") if isinstance(tool_def, dict) else None
                if isinstance(fn, dict) and fn.get("name") == required_tool_name:
                    required_defs.append(tool_def)
                    break
            if required_defs:
                tool_defs = required_defs
            else:
                logger.warning(f"Required tool '{required_tool_name}' not available in tool definitions.")
                return (
                    f"Required tool '{required_tool_name}' is not available.",
                    tools_used,
                    usage,
                    "stop",
                )

        if tool_choice == "required" and not tool_defs:
            return (
                "A tool call was required, but no tools are available.",
                tools_used,
                usage,
                "stop",
            )

        while iteration < iteration_limit:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=tool_defs,
                model=effective_model,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
            if response.finish_reason:
                final_finish_reason = response.finish_reason

            for key, value in (response.usage or {}).items():
                try:
                    usage[key] = usage.get(key, 0) + int(value)
                except (TypeError, ValueError):
                    continue

            if response.has_tool_calls:
                if not tools_enabled or not tool_defs:
                    final_content = self._strip_think(response.content) or response.content or ""
                    break

                selected_tool_calls = list(response.tool_calls)
                if required_tool_name:
                    selected_tool_calls = [
                        tc for tc in response.tool_calls if tc.name == required_tool_name
                    ]
                    if not selected_tool_calls:
                        final_content = (
                            f"Required tool '{required_tool_name}' was requested but was not used."
                        )
                        final_finish_reason = "stop"
                        break

                if on_progress:
                    clean = self._strip_think(response.content)
                    await on_progress(clean or self._tool_hint(selected_tool_calls))

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in selected_tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in selected_tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

                    # Collect media links from tool results so responses stay clickable in chat UI.
                    artifact_links.extend(self._extract_artifact_links(result))

                    # Vision injection: if the tool captured screenshots, pipe them to the LLM
                    tool_obj = self.tools.get(tool_call.name)
                    if tool_obj and hasattr(tool_obj, "_pending_images") and tool_obj._pending_images:
                        image_parts = []
                        for data_url in tool_obj._pending_images:
                            image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                        image_parts.append({
                            "type": "text",
                            "text": "Browser screenshot — analyze to understand current page state.",
                        })
                        messages.append({"role": "user", "content": image_parts})
                        logger.info(f"Injected {len(tool_obj._pending_images)} screenshot(s) into LLM context")
                        tool_obj._pending_images.clear()
            else:
                if tool_choice == "required":
                    final_content = (
                        self._strip_think(response.content)
                        or "A tool call was required, but the model returned text."
                    )
                    final_finish_reason = "stop"
                    break
                final_content = self._strip_think(response.content)
                break

        # Append media links to final response so they are discoverable in chat clients.
        if artifact_links:
            if final_content:
                final_content = final_content + "\n\n" + "\n".join(artifact_links)
            else:
                final_content = "\n".join(artifact_links)

        return final_content, tools_used, usage, final_finish_reason

    @classmethod
    def _extract_artifact_links(cls, tool_result: str) -> list[str]:
        """Extract markdown/web URLs for files generated by tools."""
        if not tool_result:
            return []

        links: list[str] = []
        seen: set[str] = set()

        def _add(link: str) -> None:
            if link not in seen:
                seen.add(link)
                links.append(link)

        for line in tool_result.splitlines():
            text = line.strip()
            if not text:
                continue

            # Already-rendered markdown image links (screenshots)
            if text.startswith(f"![screenshot]({ _API_HOST }/screenshots/"):
                _add(text)
                continue

            # Common output fields produced by file-based tools
            if text.startswith("File:"):
                path_text = text.split(":", 1)[1].strip()
                path = Path(path_text)
                ext = path.suffix.lower()
                if ext in _AUDIO_EXTS:
                    _add(f"[Audio output]({ _API_HOST }/audio/{path.name})")
                elif ext in _VIDEO_EXTS:
                    _add(f"[Video output]({ _API_HOST }/videos/{path.name})")
                elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    _add(f"![Screenshot/output]({ _API_HOST }/screenshots/{path.name})")
                continue

            # Remotion-style reference in tts output: <Audio src={staticFile('audio/name.wav')}/>
            m = re.search(r"staticFile\(\s*['\"]audio/([^'\"]+)['\"]\s*\)", text)
            if m:
                _add(f"[Audio output]({ _API_HOST }/audio/{m.group(1)})")
                continue

            # Skip lines that are already markdown image/link syntax (avoid duplicates)
            if text.startswith("![") or text.startswith("["):
                continue

            # Tool output that already emits the public URL
            m = re.search(rf"({re.escape(_API_HOST)}/(?:audio|videos|screenshots)/[^\s)]+)", text)
            if m:
                url = m.group(1)
                if "/audio/" in url:
                    _add(f"[Audio output]({url})")
                elif "/videos/" in url:
                    _add(f"[Video output]({url})")
                elif "/screenshots/" in url:
                    _add(f"![Screenshot/output]({url})")
                continue

        return links

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        try:
            await self._run_inner()
        except Exception as e:
            logger.critical(f"Agent loop crashed: {e}", exc_info=True)
            raise

    async def _run_inner(self) -> None:
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    logger.debug(f"Agent loop: processing message from {msg.channel}:{msg.sender_id}")
                    response = await self._process_message(msg)
                    if response:
                        logger.debug(f"Agent loop: publishing outbound to {response.channel}:{response.chat_id} ({len(response.content)} chars)")
                        await self.bus.publish_outbound(response)
                    else:
                        logger.warning(f"Agent loop: _process_message returned None for {msg.channel}:{msg.sender_id}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}",
                        metadata=msg.metadata or {},
                    ))
            except asyncio.TimeoutError:
                continue
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        model_override: str | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        max_iterations_override: int | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
            on_progress: Optional callback for intermediate output (defaults to bus publish).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        self._set_tool_context(msg.channel, msg.chat_id)

        # Search Mem0 for relevant semantic memories
        semantic_memories = None
        if self.mem0:
            try:
                semantic_memories = await self.mem0.search(msg.content) or None
                if semantic_memories:
                    logger.info(f"Mem0 recalled {len(semantic_memories)} memories")
            except Exception as e:
                logger.warning(f"Mem0 search error: {e}")

        async def _bus_progress(content: str) -> None:
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content,
                metadata=msg.metadata or {},
            ))

        # Multi-agent path: route through the Orchestrator (CEO as entry point)
        if self.orchestrator is not None:
            # Build messages in OpenAI format for the Orchestrator
            history = session.get_history(max_messages=self.memory_window)
            orchestrator_messages = list(history)
            # Build user content (with optional media)
            user_content = self.context._build_user_content(
                msg.content, msg.media if msg.media else None
            )
            orchestrator_messages.append({"role": "user", "content": user_content})

            result = await self.orchestrator.run(
                agent_name="ceo",
                messages=orchestrator_messages,
                session_id=key,
                model_override=model_override,
            )
            final_content = result.content
            tools_used: list[str] = []
            logger.info(
                f"Multi-agent response via {' → '.join(result.handoff_chain)} "
                f"(finish={result.finish_reason})"
            )
        else:
            # Single-agent path: existing behavior
            initial_messages = self.context.build_messages(
                history=session.get_history(max_messages=self.memory_window),
                current_message=msg.content,
                media=msg.media if msg.media else None,
                channel=msg.channel,
                chat_id=msg.chat_id,
                semantic_memories=semantic_memories,
            )

            final_content, tools_used, _, _ = await self._run_agent_loop(
                initial_messages, on_progress=on_progress or _bus_progress,
                model_override=model_override,
                temperature_override=temperature_override,
                max_tokens_override=max_tokens_override,
                max_iterations_override=max_iterations_override,
            )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        # Store exchange in Mem0 (background, non-blocking)
        if self.mem0:
            exchange = [
                {"role": "user", "content": msg.content},
                {"role": "assistant", "content": final_content},
            ]
            asyncio.create_task(self.mem0.add(exchange))

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _, _, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        model_override: str | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        max_iterations_override: int | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
            on_progress: Optional callback for intermediate output.
        
        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            model_override=model_override,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
            max_iterations_override=max_iterations_override,
        )
        return response.content if response else ""
