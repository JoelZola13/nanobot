"""OpenAI-compatible API server that wraps nanobot's AgentLoop."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

SCREENSHOTS_DIR = Path.home() / ".nanobot" / "workspace" / "screenshots"
VIDEOS_DIR = Path.home() / ".nanobot" / "workspace" / "remotion" / "out"
AUDIO_DIR = Path.home() / ".nanobot" / "workspace" / "remotion" / "public" / "audio"
ARTICLE_IMAGES_DIR = Path.home() / ".nanobot" / "workspace" / "article-images"
AVATARS_DIR = Path(__file__).parent.parent / "static" / "avatars"

from nanobot.config.loader import load_config
from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.session.manager import SessionManager
from nanobot.cron.service import CronService


_agent: AgentLoop | None = None
_cron: CronService | None = None
_orchestrator: Any = None  # Orchestrator | None — lazy import to avoid circular deps


def _make_provider(config):
    """Create the appropriate LLM provider from config (mirrors cli/commands.py)."""
    from loguru import logger
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.custom_provider import CustomProvider

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)

    if provider_name == "custom":
        return CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    # Warn if no API key is configured — the LLM call will fail or hang
    if not p or not p.api_key:
        logger.warning(
            f"No API key found for model '{model}' (provider: {provider_name}). "
            f"LLM calls will fail. Configure the provider API key in config.yaml "
            f"or set the appropriate environment variable."
        )

    from nanobot.providers.registry import find_by_name
    spec = find_by_name(provider_name)

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=provider_name,
    )


@asynccontextmanager
async def lifespan(app):
    global _agent, _cron, _orchestrator
    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron_service = CronService(store_path=cron_store_path)
    _cron = cron_service

    defaults = config.agents.defaults
    _agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=defaults.model,
        max_iterations=defaults.max_tool_iterations,
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        memory_window=defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron_service,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        postiz_config=config.tools.postiz,
    )
    # Initialize Mem0 semantic memory (optional — degrades gracefully)
    try:
        from nanobot.agent.mem0_memory import Mem0Store
        _agent.mem0 = Mem0Store(
            workspace=config.workspace_path,
            provider=provider,
            model=defaults.model,
            loop=asyncio.get_event_loop(),
        )
    except Exception as e:
        logger.warning(f"Mem0 semantic memory not available: {e}")

    # Wire up cron job callback to process messages through agent
    async def _on_cron_job(job):
        logger.info(f"Cron job '{job.name}' firing: {job.payload.message}")
        response = await _agent.process_direct(
            content=job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cron",
            chat_id=job.payload.to or "system",
        )
        return response

    cron_service.on_job = _on_cron_job

    # Register email tools if email channel is configured
    email_cfg = config.channels.email
    if email_cfg.enabled and email_cfg.imap_host and email_cfg.imap_username:
        from nanobot.agent.tools.email_tools import EmailReadTool, EmailSendTool
        _agent.tools.register(EmailReadTool(
            imap_host=email_cfg.imap_host,
            imap_port=email_cfg.imap_port,
            username=email_cfg.imap_username,
            password=email_cfg.imap_password,
            use_ssl=email_cfg.imap_use_ssl,
            mailbox=email_cfg.imap_mailbox,
        ))
        _agent.tools.register(EmailSendTool(
            smtp_host=email_cfg.smtp_host,
            smtp_port=email_cfg.smtp_port,
            username=email_cfg.smtp_username,
            password=email_cfg.smtp_password,
            from_addr=email_cfg.from_address or email_cfg.smtp_username,
            use_tls=email_cfg.smtp_use_tls,
            use_ssl=email_cfg.smtp_use_ssl,
        ))
        logger.info("Email tools registered (read + send)")

    # Register Remotion video tools
    remotion_dir = config.workspace_path / "remotion"
    if remotion_dir.exists():
        from nanobot.agent.tools.remotion import RemotionComposeTool, RemotionRenderTool
        _agent.tools.register(RemotionComposeTool(remotion_dir=remotion_dir))
        _agent.tools.register(RemotionRenderTool(
            remotion_dir=remotion_dir,
            base_url="http://localhost:18790",
        ))
        from nanobot.agent.tools.tts import QwenTTSTool
        audio_dir = remotion_dir / "public" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        _agent.tools.register(QwenTTSTool(audio_dir=audio_dir))
        logger.info("Remotion tools registered (compose + render + tts)")

    # Register Qwen-Image generation tool (always available, connects to local server)
    from nanobot.agent.tools.image_gen import QwenImageGenTool
    _agent.tools.register(QwenImageGenTool())
    logger.info("Qwen-Image generation tool registered (local server at :18791)")

    # Register article image generation tool
    from nanobot.agent.tools.article_image import ArticleImageTool
    _agent.tools.register(ArticleImageTool(base_url="http://localhost:18790"))
    logger.info("Article image generation tool registered")

    # Start Slack channel if enabled
    channel_manager = None
    slack_cfg = config.channels.slack
    if slack_cfg.enabled and slack_cfg.bot_token and slack_cfg.app_token:
        from nanobot.channels.manager import ChannelManager
        channel_manager = ChannelManager(config, bus)
        # Start the agent loop (consumes inbound from bus, publishes outbound)
        agent_task = asyncio.create_task(_agent.run())

        def _on_agent_task_done(task: asyncio.Task) -> None:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                logger.warning("Agent task was cancelled")
                return
            if exc:
                logger.critical(f"Agent task died with exception: {exc}", exc_info=exc)
            else:
                logger.warning("Agent task finished unexpectedly (no exception)")

        agent_task.add_done_callback(_on_agent_task_done)

        # Start the channel manager (starts Slack + outbound dispatcher)
        channels_task = asyncio.create_task(channel_manager.start_all())
        logger.info("Slack channel started")

    # Ensure screenshots directory exists
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Start cron service
    try:
        await cron_service.start()
        logger.info(f"Cron service started ({len(cron_service.list_jobs(include_disabled=True))} jobs)")
    except Exception as exc:
        _cron = None
        logger.warning(f"Cron service failed to start: {exc}. Continuing without cron scheduling.")

    # ── Multi-agent system initialization ──
    try:
        from nanobot.agents.loader import load_agents
        from nanobot.agents.factory import ToolFactory
        from nanobot.agents.orchestrator import Orchestrator

        teams_dir = Path(__file__).parent / "agents" / "teams"
        agent_registry = load_agents(teams_dir)

        if len(agent_registry) > 0:
            tool_config: dict[str, Any] = {
                "brave_api_key": config.tools.web.search.api_key or None,
                "restrict_to_workspace": config.tools.restrict_to_workspace,
                "shell": {
                    "timeout": getattr(config.tools.exec, "timeout", 120),
                },
            }
            tool_factory = ToolFactory(
                agent_registry,
                workspace=config.workspace_path,
                tool_config=tool_config,
                provider=provider,
            )
            _orchestrator = Orchestrator(
                provider=provider,
                agent_registry=agent_registry,
                tool_factory=tool_factory,
                default_model=defaults.model,
            )
            # Wire orchestrator into the agent loop so all channels
            # (Slack, Telegram, etc.) route through the multi-agent system
            _agent.orchestrator = _orchestrator
            logger.info(
                f"Multi-agent system ready: {len(agent_registry)} agents across "
                f"{len(agent_registry.get_teams())} teams (all channels enabled)"
            )
        else:
            logger.info("No agent team definitions found — multi-agent system disabled")
    except Exception as exc:
        logger.warning(f"Multi-agent system not available: {exc}")

    logger.info("Nanobot API server ready")
    yield
    cron_service.stop()
    if channel_manager:
        await channel_manager.stop_all()
        _agent.stop()
    await _agent.close_mcp()
    _agent = None
    _cron = None


def _as_bool(value: Any) -> bool:
    """Best-effort cast for header/body boolean values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _as_int(value: Any, *, field: str, min_value: int = 1) -> int | None:
    """Best-effort int parser used for request compatibility."""
    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{field} must be an integer")
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field} must be an integer")
        parsed = int(stripped)
    else:
        raise ValueError(f"{field} must be an integer")
    if parsed < min_value:
        raise ValueError(f"{field} must be >= {min_value}")
    return parsed


def _as_float(
    value: Any,
    *,
    field: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Best-effort float parser used for request compatibility."""
    if value is None:
        return None
    if isinstance(value, bool):
        value = float(int(value))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{field} must be <= {max_value}")
    return parsed


def _parse_tool_choice(value: Any) -> tuple[str, str | None, str | None]:
    """Return (tool_choice, required_tool_name, error)."""
    if value is None:
        return "auto", None, None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "auto", "required"}:
            return lowered, None, None
        return "", None, "tool_choice must be one of: none, auto, required"

    if isinstance(value, dict):
        type_name = str(value.get("type", "")).strip().lower()
        if type_name in {"none", "auto"}:
            return type_name, None, None
        if type_name == "function":
            fn_name = value.get("function", {}).get("name")
            if isinstance(fn_name, str) and fn_name.strip():
                return "required", fn_name.strip(), None
            return "auto", None, "tool_choice.function requires a function name"
        return "auto", None, "tool_choice.type must be one of: none, auto, function"

    return "auto", None, "tool_choice must be a string or object"


def _parse_stream_options(value: Any) -> tuple[bool, str | None]:
    """Return include_usage flag and optional validation error."""
    if value is None:
        return False, None
    if not isinstance(value, dict):
        return False, "stream_options must be an object"
    include_usage = value.get("include_usage")
    if include_usage is None:
        return False, None
    if isinstance(include_usage, bool):
        return include_usage, None
    if isinstance(include_usage, str):
        return include_usage.strip().lower() in {"1", "true", "yes", "on"}, None
    return False, "stream_options.include_usage must be a boolean"


def _openai_error(
    message: str,
    *,
    status_code: int = 400,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Format errors in OpenAI-compatible structure."""
    return JSONResponse({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code or "invalid_request_error",
        },
    }, status_code=status_code)


def _resolve_session_key(request: Request, body: dict[str, Any]) -> str:
    """Build a session key that keeps LibreChat conversations separate."""
    session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("x-conversation-id")
        or request.headers.get("x-chat-id")
    )
    if session_id:
        return f"librechat:{session_id}"

    conv_id = (
        body.get("conversation_id")
        or body.get("session_id")
        or request.headers.get("x-user-id")
    )
    if conv_id:
        return f"librechat:{conv_id}"

    user = body.get("user")
    if isinstance(user, str) and user.strip():
        return f"librechat:{user.strip()}"
    return "librechat:default"


def _resolve_model(request_model: Any, default_model: str) -> tuple[str, str]:
    """Return `(requested_model_for_client, resolved_model_for_provider)`.

    Keep custom model names from clients untouched, but map Codex-style clients
    to prefixed model IDs when needed.
    """
    requested = str(request_model).strip() if request_model else ""
    if not requested:
        requested = default_model

    if requested.lower() == "openai-codex":
        requested = default_model

    if requested.lower().startswith("openai-codex/"):
        return requested, requested

    default_model = str(default_model)
    if "openai-codex/" in default_model and "/" not in requested and not requested.startswith("openai-codex"):
        return requested, f"openai-codex/{requested}"

    return requested, requested


async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Handle POST /v1/chat/completions."""
    if not _agent:
        return _openai_error("Agent not initialized", status_code=503, code="unavailable")

    body = await request.json()
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return _openai_error("messages must be a list", param="messages")
    if not messages:
        return _openai_error("No messages provided", code="empty_request")
    stream = _as_bool(body.get("stream", False))

    requested_model, provider_model = _resolve_model(
        body.get("model", _agent.model),
        _agent.model,
    )

    temperature = body.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            return _openai_error("temperature must be a number", param="temperature")

    max_tokens = (
        body.get("max_tokens")
        if body.get("max_tokens") is not None
        else body.get("max_output_tokens", body.get("max_completion_tokens"))
    )
    try:
        max_tokens = _as_int(max_tokens, field="max_tokens")
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="max_tokens")

    try:
        top_p = _as_float(body.get("top_p"), field="top_p", min_value=0.0, max_value=1.0)
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="top_p")

    try:
        _as_float(
            body.get("presence_penalty"),
            field="presence_penalty",
            min_value=-2.0,
            max_value=2.0,
        )
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="presence_penalty")
    try:
        _as_float(
            body.get("frequency_penalty"),
            field="frequency_penalty",
            min_value=-2.0,
            max_value=2.0,
        )
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="frequency_penalty")
    stop = body.get("stop")
    if stop is not None:
        if isinstance(stop, str):
            if not stop:
                return _openai_error("stop must not be empty", param="stop")
        elif isinstance(stop, list):
            if not stop:
                return _openai_error("stop must not be empty", param="stop")
            if not all(isinstance(item, str) and item for item in stop):
                return _openai_error("stop entries must be non-empty strings", param="stop")
        else:
            return _openai_error("stop must be a string or list of strings", param="stop")

    tool_choice, required_tool_name, err = _parse_tool_choice(body.get("tool_choice"))
    if err:
        return _openai_error(err, param="tool_choice")

    include_usage, err = _parse_stream_options(body.get("stream_options"))
    if err:
        return _openai_error(err, param="stream_options")

    n = body.get("n")
    if n is not None:
        try:
            n = _as_int(n, field="n", min_value=1)
        except (TypeError, ValueError) as err:
            return _openai_error(str(err), param="n")
        if n != 1:
            return _openai_error("Only n=1 is supported", param="n")

    try:
        max_iterations = _as_int(body.get("max_iterations"), field="max_iterations")
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="max_iterations")
    if max_iterations is None:
        try:
            max_iterations = _as_int(body.get("max_tool_iterations"), field="max_tool_iterations")
        except (TypeError, ValueError) as err:
            return _openai_error(str(err), param="max_tool_iterations")

    session_key = _resolve_session_key(request, body)
    reset_session = _as_bool(
        request.headers.get("x-session-reset") or body.get("session_reset")
    )

    if not messages or not all(isinstance(msg, dict) for msg in messages):
        return _openai_error("Invalid message format", code="invalid_request_error")

    # ── Multi-agent routing ──
    # If the requested model starts with "agent/", route to the Orchestrator
    # instead of the single-agent AgentLoop.
    if requested_model.startswith("agent/") and _orchestrator is not None:
        agent_name = requested_model.removeprefix("agent/")

        # Don't pass provider_model as override — each agent has its own
        # configured model.  The "agent/auto" or "agent/ceo" name is for
        # routing only, not a real LLM model identifier.
        if stream:
            return StreamingResponse(
                _stream_agent_response(
                    agent_name=agent_name,
                    messages=messages,
                    model_override=None,
                    requested_model=requested_model,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
            )

        result = await _orchestrator.run(
            agent_name=agent_name,
            messages=messages,
            model_override=None,
        )
        if result.trace:
            logger.info(f"Multi-agent trace:\n{result.trace.log_summary()}")
        return JSONResponse(result.to_chat_completion(model=requested_model))

    if stream:
        return StreamingResponse(
            _stream_response(
                messages=messages,
                session_key=session_key,
                requested_model=requested_model,
                provider_model=provider_model,
                tool_choice=tool_choice,
                required_tool_name=required_tool_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_iterations,
                reset_session=reset_session,
                include_usage=include_usage,
            ),
            media_type="text/event-stream",
        )

    response_text, tools_used, usage, finish_reason = await _agent.process_openai_messages(
        messages=messages,
        session_key=session_key,
        channel="api",
        chat_id="librechat",
        model_override=provider_model,
        tool_choice=tool_choice,
        required_tool_name=required_tool_name,
        temperature_override=temperature,
        max_tokens_override=max_tokens,
        max_iterations_override=max_iterations,
        reset_session=reset_session,
        return_usage=True,
    )
    if not response_text:
        return _openai_error("No user message found", code="empty_request")

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
                "tool_calls": [
                    {"type": "function", "function": {"name": tool}} for tool in tools_used
                ] if tools_used else [],
            },
            "finish_reason": finish_reason or "stop",
        }],
        "usage": usage,
    })


async def _stream_response(
    messages: list[dict[str, Any]],
    session_key: str,
    requested_model: str,
    provider_model: str,
    tool_choice: str,
    required_tool_name: str | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_iterations: int | None = None,
    reset_session: bool = False,
    include_usage: bool = False,
):
    """Stream response as SSE events with intermediate progress."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _on_progress(text: str) -> None:
        """Push intermediate progress (tool hints) to the SSE stream."""
        if text:
            await progress_queue.put(text)

    async def _run_agent():
        """Run the agent and signal completion."""
        try:
            result = await _agent.process_openai_messages(
                messages=messages,
                session_key=session_key,
                channel="api",
                chat_id="librechat",
                on_progress=_on_progress,
                tool_choice=tool_choice,
                required_tool_name=required_tool_name,
                model_override=provider_model,
                temperature_override=temperature,
                max_tokens_override=max_tokens,
                max_iterations_override=max_iterations,
                reset_session=reset_session,
                return_usage=True,
            )
            await progress_queue.put(None)  # Signal done
            return result
        except Exception as e:
            await progress_queue.put(None)
            return f"Error: {e}"

    # Start agent processing in background
    agent_task = asyncio.create_task(_run_agent())

    # OpenAI-compatible stream starts with role-only assistant delta.
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    # Stream intermediate progress as it arrives
    while True:
        item = await progress_queue.get()
        if item is None:
            break
        # Send progress as a chunk (shown as "thinking" text)
        progress_text = f"{item}\n\n"
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{
                "index": 0,
                "delta": {"content": progress_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Get final response
    final_result = await agent_task
    tools_used: list[str] = []
    if isinstance(final_result, tuple):
        response_text = final_result[0] or ""
        if len(final_result) == 4:
            tools_used = final_result[1] if isinstance(final_result[1], list) else []
            usage = final_result[2] if isinstance(final_result[2], dict) else {}
            finish_reason = final_result[3] if isinstance(final_result[3], str) else None
        else:
            tools_used = []
            usage = {}
            finish_reason = None
    else:
        response_text = final_result
        usage = {}
        finish_reason = None

    # Send the final response
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": response_text,
                "tool_calls": [
                    {"type": "function", "function": {"name": tool}} for tool in tools_used
                ] if tools_used else [],
            },
            "finish_reason": finish_reason or "stop",
        }],
    }
    if include_usage and isinstance(usage, dict):
        usage_payload = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    yield f"data: {json.dumps(chunk)}\n\n"

    # OpenAI-compatible [DONE] usage packet when requested.
    if include_usage and isinstance(usage, dict):
        usage_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [],
            "usage": usage_payload,
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_agent_response(
    agent_name: str,
    messages: list[dict],
    model_override: str | None,
    requested_model: str,
    include_usage: bool,
):
    """Stream a multi-agent orchestrator response as SSE chunks.

    Uses an asyncio.Queue so that on_progress callbacks from the
    orchestrator / agent instances / delegate tools stream progress
    to the client in real-time (routing decisions, tool calls,
    delegation events, handoffs) instead of blocking until the
    entire multi-agent pipeline completes.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _on_progress(text: str) -> None:
        """Push intermediate progress to the SSE stream."""
        if text:
            await progress_queue.put(text)

    async def _run_orchestrator():
        """Run the orchestrator in background; signal done when finished."""
        try:
            result = await _orchestrator.run(
                agent_name=agent_name,
                messages=messages,
                model_override=model_override,
                on_progress=_on_progress,
            )
            await progress_queue.put(None)  # Signal done
            return result
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            await progress_queue.put(None)
            return e

    # Start orchestrator in background
    agent_task = asyncio.create_task(_run_orchestrator())

    # OpenAI-compatible stream starts with role-only assistant delta.
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    # Stream intermediate progress as it arrives
    while True:
        item = await progress_queue.get()
        if item is None:
            break
        progress_text = f"{item}\n\n"
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{
                "index": 0,
                "delta": {"content": progress_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Get final result
    result = await agent_task

    if isinstance(result, Exception):
        content = f"Error: {result}"
        usage = {}
        session_id = "error"
    else:
        # Log the trace
        if result.trace:
            logger.info(f"Multi-agent trace:\n{result.trace.log_summary()}")
        content = result.content
        usage = result.usage or {}
        session_id = result.session_id

    # Send the final response content
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Usage packet if requested
    if include_usage and usage:
        usage_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


async def list_models(request: Request) -> JSONResponse:
    """Handle GET /v1/models."""
    now = int(time.time())
    default_model = _agent.model if _agent else "gpt-5.4"
    model_ids = [
        default_model,
        default_model.removeprefix("openai-codex/"),
    ]
    # Keep popular aliases for clients that request openai-style names.
    model_ids.extend(["gpt-5.4", "gpt-5"])
    deduped = []
    for model_id in model_ids:
        if model_id and model_id not in deduped:
            deduped.append(model_id)

    models = [
        {"id": model_id, "object": "model", "created": now, "owned_by": "nanobot"}
        for model_id in deduped
    ]

    # Append multi-agent models if the orchestrator is available
    if _orchestrator is not None:
        # Auto-router model — routes by intent, no CEO bottleneck
        models.insert(0, {
            "id": "agent/auto",
            "object": "model",
            "created": now,
            "owned_by": "nanobot",
        })
        models.extend(_orchestrator.list_agents())

    return JSONResponse({"object": "list", "data": models})


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def serve_screenshot(request: Request) -> FileResponse | JSONResponse:
    """Serve a saved browser screenshot image."""
    filename = request.path_params["filename"]
    filepath = SCREENSHOTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/png")


async def serve_avatar(request: Request) -> FileResponse | JSONResponse:
    """Serve a generated agent avatar SVG."""
    filename = request.path_params["filename"]
    filepath = AVATARS_DIR / filename
    if ".." in filename or not filepath.resolve().is_relative_to(AVATARS_DIR.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/svg+xml")


async def serve_audio(request: Request) -> FileResponse | JSONResponse:
    """Serve a TTS-generated audio file."""
    filename = request.path_params["filename"]
    filepath = AUDIO_DIR / filename
    if ".." in filename or not filepath.resolve().is_relative_to(AUDIO_DIR.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(filepath, media_type="audio/wav")


MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}


async def serve_video(request: Request) -> FileResponse | JSONResponse:
    """Serve a rendered Remotion video or still."""
    filename = request.path_params["filename"]
    filepath = VIDEOS_DIR / filename
    # Prevent path traversal
    if ".." in filename or not filepath.resolve().is_relative_to(VIDEOS_DIR.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    media_type = MIME_TYPES.get(filepath.suffix.lower(), "application/octet-stream")
    return FileResponse(filepath, media_type=media_type)


def _job_to_dict(job) -> dict:
    """Serialize a CronJob to a JSON-safe dict."""
    return {
        "id": job.id,
        "name": job.name,
        "enabled": job.enabled,
        "schedule": {
            "kind": job.schedule.kind,
            "atMs": job.schedule.at_ms,
            "everyMs": job.schedule.every_ms,
            "expr": job.schedule.expr,
            "tz": job.schedule.tz,
        },
        "payload": {
            "kind": job.payload.kind,
            "message": job.payload.message,
            "deliver": job.payload.deliver,
            "channel": job.payload.channel,
            "to": job.payload.to,
        },
        "state": {
            "nextRunAtMs": job.state.next_run_at_ms,
            "lastRunAtMs": job.state.last_run_at_ms,
            "lastStatus": job.state.last_status,
            "lastError": job.state.last_error,
        },
        "createdAtMs": job.created_at_ms,
    }


async def cron_list_jobs(request: Request) -> JSONResponse:
    """GET /api/cron/jobs — list all cron jobs."""
    if not _cron:
        return JSONResponse({"error": "Cron service not initialized"}, status_code=503)
    jobs = _cron.list_jobs(include_disabled=True)
    return JSONResponse({"jobs": [_job_to_dict(j) for j in jobs]})


async def cron_toggle_job(request: Request) -> JSONResponse:
    """POST /api/cron/jobs/{job_id}/toggle — enable/disable a job."""
    if not _cron:
        return JSONResponse({"error": "Cron service not initialized"}, status_code=503)
    job_id = request.path_params["job_id"]
    body = await request.json()
    enabled = body.get("enabled", True)
    job = _cron.enable_job(job_id, enabled=enabled)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse({"ok": True, "job": _job_to_dict(job)})


async def cron_run_job(request: Request) -> JSONResponse:
    """POST /api/cron/jobs/{job_id}/run — force-run a job now."""
    if not _cron:
        return JSONResponse({"error": "Cron service not initialized"}, status_code=503)
    job_id = request.path_params["job_id"]
    ok = await _cron.run_job(job_id, force=True)
    if not ok:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse({"ok": True})


async def cron_delete_job(request: Request) -> JSONResponse:
    """DELETE /api/cron/jobs/{job_id} — delete a job."""
    if not _cron:
        return JSONResponse({"error": "Cron service not initialized"}, status_code=503)
    job_id = request.path_params["job_id"]
    removed = _cron.remove_job(job_id)
    if not removed:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse({"ok": True})


async def generate_article_image(request: Request) -> JSONResponse:
    """POST /api/article-image/generate — create cover + body PNGs."""
    from nanobot.services.article_image import generate_article_images

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    headline = body.get("headline")
    body_text = body.get("body_text")
    hero_url = body.get("hero_image_url")
    if not headline or not body_text or not hero_url:
        return JSONResponse(
            {"error": "headline, body_text, and hero_image_url are required"},
            status_code=400,
        )

    try:
        result = await generate_article_images(
            headline=headline,
            body_text=body_text,
            hero_image_url=hero_url,
            category=body.get("category", "ARTICLE"),
        )
    except Exception as exc:
        logger.exception("Failed to generate article images")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse(result)


async def serve_article_image(request: Request) -> FileResponse | JSONResponse:
    """Serve a generated article image."""
    filename = request.path_params["filename"]
    filepath = ARTICLE_IMAGES_DIR / filename
    if ".." in filename or not filepath.resolve().is_relative_to(ARTICLE_IMAGES_DIR.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    media_type = MIME_TYPES.get(filepath.suffix.lower(), "image/png")
    return FileResponse(filepath, media_type=media_type)


async def audio_transcriptions(request: Request) -> JSONResponse:
    """OpenAI-compatible STT endpoint — proxies to Groq Whisper."""
    try:
        form = await request.form()
    except Exception:
        return _openai_error("Invalid multipart form data", status_code=400)

    upload = form.get("file")
    if upload is None:
        return _openai_error("Missing required field: file", param="file")

    response_format = form.get("response_format", "json")

    # Save uploaded file to a temp path
    suffix = Path(upload.filename).suffix if upload.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await upload.read())
        tmp_path = tmp.name

    try:
        from nanobot.providers.transcription import GroqTranscriptionProvider
        provider = GroqTranscriptionProvider()
        text = await provider.transcribe(tmp_path)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return _openai_error(f"Transcription failed: {e}", status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if response_format == "text":
        from starlette.responses import Response
        return Response(text, media_type="text/plain")

    return JSONResponse({"text": text})


_QWEN_SPEAKERS = {"Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"}
_OPENAI_VOICE_MAP = {
    "alloy": "Ryan", "echo": "Ryan", "fable": "Ryan", "onyx": "Ryan", "shimmer": "Ryan",
    "nova": "Vivian",
}


async def audio_speech(request: Request) -> StreamingResponse | JSONResponse:
    """OpenAI-compatible TTS endpoint — generates speech locally via Qwen3-TTS."""
    try:
        body = await request.json()
    except Exception:
        return _openai_error("Invalid JSON body", status_code=400)

    text = body.get("input", "")
    if not text:
        return _openai_error("Missing required field: input", param="input")

    voice = body.get("voice", "Ryan")
    # Map OpenAI voice names → Qwen speaker; pass through if already a Qwen speaker name
    speaker = voice if voice in _QWEN_SPEAKERS else _OPENAI_VOICE_MAP.get(voice, "Ryan")

    tmp_name = f"tts_{uuid.uuid4().hex[:12]}"
    wav_path = AUDIO_DIR / f"{tmp_name}.wav"

    try:
        from nanobot.agent.tools.tts import QwenTTSTool
        tool = QwenTTSTool(AUDIO_DIR)
        result = await tool.execute(text=text, output_name=tmp_name, speaker=speaker)
        if result.startswith("Error"):
            return _openai_error(result, status_code=500)
    except Exception as e:
        logger.error(f"Qwen TTS failed: {e}")
        return _openai_error(f"TTS failed: {e}", status_code=500)

    if not wav_path.exists():
        return _openai_error("TTS produced no output file", status_code=500)

    audio_bytes = wav_path.read_bytes()

    # Convert WAV to MP3 for LobeHub compatibility
    response_format = body.get("response_format", "mp3")
    if response_format == "mp3":
        import subprocess
        mp3_path = wav_path.with_suffix(".mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-q:a", "2", str(mp3_path)],
                capture_output=True, check=True,
            )
            audio_bytes = mp3_path.read_bytes()
            mp3_path.unlink(missing_ok=True)
            media_type = "audio/mpeg"
            filename = "speech.mp3"
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"ffmpeg WAV→MP3 conversion failed, returning WAV: {e}")
            media_type = "audio/wav"
            filename = "speech.wav"
    else:
        media_type = "audio/wav"
        filename = "speech.wav"

    wav_path.unlink(missing_ok=True)

    return StreamingResponse(
        content=iter([audio_bytes]),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ─── Embeddings endpoint (for LobeHub Knowledge Base / RAG) ──────────────
_embedding_model = None


async def embeddings(request: Request) -> JSONResponse:
    """OpenAI-compatible embeddings endpoint using local sentence-transformers."""
    global _embedding_model
    try:
        body = await request.json()
    except Exception:
        return _openai_error("Invalid JSON body", status_code=400)

    texts = body.get("input", "")
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return _openai_error("Missing required field: input", param="input")

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")

    vectors = _embedding_model.encode(texts, normalize_embeddings=True)

    data = [
        {"object": "embedding", "embedding": v.tolist(), "index": i}
        for i, v in enumerate(vectors)
    ]
    return JSONResponse({
        "object": "list",
        "data": data,
        "model": body.get("model", "all-MiniLM-L6-v2"),
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts),
        },
    })


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/v1/audio/transcriptions", audio_transcriptions, methods=["POST"]),
        Route("/v1/audio/speech", audio_speech, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/screenshots/{filename:path}", serve_screenshot, methods=["GET"]),
        Route("/videos/{filename:path}", serve_video, methods=["GET"]),
        Route("/audio/{filename:path}", serve_audio, methods=["GET"]),
        Route("/avatars/{filename:path}", serve_avatar, methods=["GET"]),
        Route("/api/cron/jobs", cron_list_jobs, methods=["GET"]),
        Route("/api/cron/jobs/{job_id}/toggle", cron_toggle_job, methods=["POST"]),
        Route("/api/cron/jobs/{job_id}/run", cron_run_job, methods=["POST"]),
        Route("/api/cron/jobs/{job_id}", cron_delete_job, methods=["DELETE"]),
        Route("/api/article-image/generate", generate_article_image, methods=["POST"]),
        Route("/article-images/{filename:path}", serve_article_image, methods=["GET"]),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        ),
    ],
    lifespan=lifespan,
)
