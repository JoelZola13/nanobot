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
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles

SCREENSHOTS_DIR = Path.home() / ".nanobot" / "workspace" / "screenshots"
VIDEOS_DIR = Path.home() / ".nanobot" / "workspace" / "remotion" / "out"
AUDIO_DIR = Path.home() / ".nanobot" / "workspace" / "remotion" / "public" / "audio"
ARTICLE_IMAGES_DIR = Path.home() / ".nanobot" / "workspace" / "article-images"
AVATARS_DIR = Path(__file__).parent.parent / "static" / "avatars"
SHARED_ASSETS_DIR = Path(__file__).parent.parent / "LibreChat" / "client" / "public"
GATEWAY_STATIC_DIR = Path(__file__).parent / "gateway" / "static"

from nanobot.config.loader import load_config
from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.session.manager import SessionManager
from nanobot.cron.service import CronService


_agent: AgentLoop | None = None
_cron: CronService | None = None
_orchestrator: Any = None  # Orchestrator | None — lazy import to avoid circular deps
_harness: Any = None  # DeepAgentHarness | None — deepagents-powered engine
_gateway: Any = None  # GatewayServer | None — WS mission control
_redis_bus: Any = None  # RedisBus | None — cross-service event bus


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


def _install_mcp_error_suppressor():
    """Suppress the MCP cancel-scope RuntimeError that crashes the process.

    The MCP stdio_client async generator raises RuntimeError during cleanup
    due to an anyio cancel-scope bug.  This fires outside any try/except,
    so we catch it at the event loop level.
    """
    loop = asyncio.get_event_loop()
    _orig = loop.get_exception_handler()

    def _handler(loop, context):
        exc = context.get("exception")
        msg = context.get("message", "")
        if exc and "cancel scope" in str(exc):
            logger.debug(f"Suppressed MCP cancel-scope error: {exc}")
            return
        if "cancel scope" in msg:
            logger.debug(f"Suppressed MCP cancel-scope message: {msg}")
            return
        if _orig:
            _orig(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


@asynccontextmanager
async def lifespan(app):
    global _agent, _cron, _orchestrator
    _install_mcp_error_suppressor()
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

    # Register SV Social tools (direct PostgreSQL access)
    _social_pool = None
    try:
        import asyncpg
        _social_pool = await asyncpg.create_pool(
            "postgresql://lobehub:lobehub_password@localhost:5433/social",
            min_size=1,
            max_size=5,
        )
        from nanobot.agent.tools.social_tools import ALL_SOCIAL_TOOLS
        for tool_cls in ALL_SOCIAL_TOOLS:
            _agent.tools.register(tool_cls(pool=_social_pool))
        logger.info(f"SV Social tools registered ({len(ALL_SOCIAL_TOOLS)} tools)")

        # Register unified search tool (spans Social + MeiliSearch)
        from nanobot.agent.tools.unified_search import UnifiedSearchTool
        _agent.tools.register(UnifiedSearchTool(
            pool=_social_pool,
            meili_url="http://localhost:7700",
            meili_key="DrhYf7zENyR6AlUCKmnz0eYASOQdl6zxH7s7MKFSfFCt",
        ))
        logger.info("Unified search tool registered (social + chat + directory)")
    except Exception as e:
        logger.warning(f"SV Social tools not available: {e}")

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
            # Collect pre-instantiated tools (social, unified_search) for sub-agent access
            _extra_tools = {}
            for t_name in _agent.tools.tool_names:
                if t_name.startswith("social_") or t_name == "unified_search":
                    t = _agent.tools.get(t_name)
                    if t:
                        _extra_tools[t_name] = t

            tool_factory = ToolFactory(
                agent_registry,
                workspace=config.workspace_path,
                tool_config=tool_config,
                provider=provider,
                mcp_tools=_extra_tools,
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

    # ── Deep Agent Harness initialization ──
    try:
        from nanobot.harness import DeepAgentHarness
        global _harness

        _harness = DeepAgentHarness(
            workspace=config.workspace_path,
            config=config.to_dict() if hasattr(config, 'to_dict') else {},
        )
        # Collect MCP tools from the agent loop for bridging
        mcp_tool_dict = {}
        for tool_name in _agent.tools.tool_names:
            tool = _agent.tools.get(tool_name)
            if tool:
                mcp_tool_dict[tool_name] = tool

        await _harness.initialize(
            tool_registry=_agent.tools,
            mcp_tools=mcp_tool_dict,
            teams_dir=Path(__file__).parent / "agents" / "teams",
        )
        logger.info(
            f"Deep Agent Harness ready: {_harness.agent_count} agents, "
            f"universal memory enabled"
        )
    except Exception as exc:
        _harness = None
        logger.warning(f"Deep Agent Harness not available: {exc}")
        import traceback
        traceback.print_exc()

    # ── Gateway (WebSocket mission control) ──
    global _gateway
    try:
        from nanobot.gateway.server import GatewayServer
        from nanobot.gateway.auth import GatewayAuth

        _gateway = GatewayServer(agent=_agent, auth=GatewayAuth(), config=config)
        logger.info("Gateway mission control ready at /ws (dashboard at /dashboard)")
    except Exception as exc:
        _gateway = None
        logger.warning(f"Gateway not available: {exc}")

    # ── Redis event bus for cross-service communication ──
    global _redis_bus
    try:
        from nanobot.bus.redis_bus import RedisBus

        _redis_bus = RedisBus(url="redis://localhost:6380")

        # Example subscriber: log social messages for awareness
        async def _on_social_message(event):
            logger.debug(f"[RedisBus] Social message in #{event.get('channelName', '?')}: {event.get('content', '')[:80]}")

        _redis_bus.subscribe("social.message.new", _on_social_message)
        await _redis_bus.start()
        logger.info("Redis event bus started (redis://localhost:6380)")
    except Exception as exc:
        _redis_bus = None
        logger.warning(f"Redis event bus not available: {exc}")

    # ── Platform awareness: refresh context with Social status ──
    _platform_task = None
    if _social_pool and _agent:
        async def _refresh_platform_status():
            """Periodically query Social DB and update agent context."""
            while True:
                try:
                    async with _social_pool.acquire() as conn:
                        # Online users
                        online = await conn.fetch(
                            """SELECT display_name, is_agent FROM users
                               WHERE status != 'offline'
                                  OR last_seen_at > NOW() - INTERVAL '5 minutes'
                               ORDER BY is_agent ASC LIMIT 15"""
                        )
                        people = [r["display_name"] for r in online if not r["is_agent"]]
                        agents = [r["display_name"] for r in online if r["is_agent"]]

                        # Recent activity
                        recent = await conn.fetch(
                            """SELECT c.name, COUNT(*) as cnt
                               FROM messages m JOIN channels c ON m.channel_id = c.id
                               WHERE m.created_at > NOW() - INTERVAL '1 hour'
                                 AND m.deleted_at IS NULL AND c.name IS NOT NULL
                               GROUP BY c.name ORDER BY cnt DESC LIMIT 5"""
                        )

                    parts = ["\n## Platform Status (auto-refreshed)"]
                    if people:
                        parts.append(f"Online people: {', '.join(people)}")
                    if agents:
                        parts.append(f"Online agents: {', '.join(agents)}")
                    if not people and not agents:
                        parts.append("No users currently online")
                    if recent:
                        activity = "; ".join(f"#{r['name']} ({r['cnt']} msgs)" for r in recent)
                        parts.append(f"Recent Social activity (1h): {activity}")

                    _agent.context._platform_status = "\n".join(parts)
                except Exception as e:
                    logger.debug(f"Platform status refresh failed: {e}")
                await asyncio.sleep(60)

        _platform_task = asyncio.create_task(_refresh_platform_status())
        logger.info("Platform awareness task started (60s refresh)")

    logger.info("Nanobot API server ready")
    yield
    try:
        cron_service.stop()
        if channel_manager:
            await channel_manager.stop_all()
            _agent.stop()
        await _agent.close_mcp()
    except (RuntimeError, BaseExceptionGroup) as shutdown_err:
        logger.warning(f"Ignoring MCP shutdown error (harmless): {shutdown_err}")
    except Exception as shutdown_err:
        logger.warning(f"Unexpected shutdown error (continuing): {shutdown_err}")
    if _platform_task:
        _platform_task.cancel()
    if _redis_bus:
        await _redis_bus.stop()
    if _social_pool:
        await _social_pool.close()
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

    # Publish agent task completion to Redis event bus
    if _redis_bus and tools_used:
        try:
            await _redis_bus.publish("agent.task.complete", {
                "type": "agent.task.complete",
                "agentName": requested_model,
                "taskSummary": (response_text or "")[:150],
                "toolsUsed": tools_used[:5],
                "channel": "librechat",
            })
        except Exception:
            pass

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

    # Publish agent task completion to Redis event bus
    if _redis_bus and not isinstance(result, Exception):
        try:
            agents_involved = []
            if hasattr(result, 'trace') and result.trace:
                agents_involved = [s.agent for s in getattr(result.trace, 'steps', [])][:5]
            await _redis_bus.publish("agent.task.complete", {
                "type": "agent.task.complete",
                "agentName": agent_name,
                "taskSummary": (content or "")[:150],
                "agentsInvolved": agents_involved,
                "channel": "librechat",
            })
        except Exception:
            pass

    # Send the final response content (separate from finish_reason per OpenAI spec)
    if content:
        content_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

    # Send finish_reason in a separate chunk (LibreChat requires this split)
    stop_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"

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
    model_ids.extend(["gpt-5.4", "gpt-5.1", "gpt-5"])
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
    agent_count = 0
    teams = []
    if _orchestrator is not None:
        try:
            agent_count = len(_orchestrator.registry.agent_names)
            teams = list(_orchestrator.registry.get_teams())
        except Exception:
            pass
    harness_info = {}
    if _harness is not None:
        harness_info = {
            "engine": "deepagents",
            "agents": _harness.agent_count,
            "initialized": _harness.is_initialized,
            "universal_memory": True,
        }
    return JSONResponse({
        "status": "ok",
        "agents": agent_count,
        "teams": len(teams),
        "orchestrator": _orchestrator is not None,
        "harness": harness_info if harness_info else None,
    })


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


async def agents_status(request: Request) -> JSONResponse:
    """Return registered agents and their availability (for Paperclip integration)."""
    if _orchestrator is None:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=503)
    try:
        agents = []
        for name in _orchestrator.registry.agent_names:
            agent_cfg = _orchestrator.registry.get(name)
            agents.append({
                "name": name,
                "team": getattr(agent_cfg, "team", "unknown"),
                "role": getattr(agent_cfg, "role", "member"),
                "model": getattr(agent_cfg, "model", "unknown"),
                "status": "available",
            })
        return JSONResponse({"agents": agents, "count": len(agents)})
    except Exception as e:
        logger.error(f"agents_status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def serve_shared_asset(request: Request) -> FileResponse | JSONResponse:
    """Serve shared JS/CSS assets that need to be loaded across all apps."""
    filename = request.path_params["filename"]
    # Only allow .js and .css files
    if not filename.endswith((".js", ".css")):
        return JSONResponse({"error": "Not found"}, status_code=404)
    filepath = SHARED_ASSETS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    media_type = "application/javascript" if filename.endswith(".js") else "text/css"
    return FileResponse(filepath, media_type=media_type)


async def agents_list(request: Request) -> JSONResponse:
    """GET /v1/agents/list — Cross-app agent discovery endpoint.

    Returns detailed agent info for all registered agents.
    Used by SV Social (@mention autocomplete), LobeHub (agent catalog),
    and Mission Control (dispatch agent picker).
    """
    if _orchestrator is None:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=503)
    try:
        agents = []
        teams_set: set[str] = set()
        for name in _orchestrator.registry.agent_names:
            spec = _orchestrator.registry.get(name)
            team = getattr(spec, "team", "unknown")
            role = getattr(spec, "role", "member")
            description = getattr(spec, "description", "")
            handoffs = list(getattr(spec, "handoffs", []) or [])
            tools = list(getattr(spec, "tools", []) or [])
            teams_set.add(team)

            # Build chat URL for deep linking from any app
            chat_url = f"http://localhost:3180/?agentModel=agent/{name}"

            agents.append({
                "name": name,
                "displayName": name.replace("_", " ").title(),
                "team": team,
                "role": role,
                "description": description,
                "model": f"agent/{name}",
                "status": "available",
                "chatUrl": chat_url,
                "handoffs": handoffs,
                "toolCount": len(tools),
            })

        # Sort: leads first, then alphabetically by team then name
        role_order = {"lead": 0, "memory": 2}
        agents.sort(key=lambda a: (role_order.get(a["role"], 1), a["team"], a["name"]))

        return JSONResponse({
            "agents": agents,
            "count": len(agents),
            "teams": sorted(teams_set),
            "teamCount": len(teams_set),
        })
    except Exception as e:
        logger.error(f"agents_list error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)



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




async def groups_api(request: Request) -> JSONResponse:
    """GET /groups — Return agent teams as community groups."""
    teams = [
        {
            "id": 1,
            "name": "Executive",
            "description": "Strategic leadership and cross-team coordination. The CEO oversees all operations and delegates work across the organization.",
            "avatar_url": None,
            "member_count": 3,
            "tags": ["Leadership", "Strategy", "Operations"],
            "category": "leadership",
            "is_public": True,
            "is_member": True,
            "channel_slug": "executive",
            "channel_id": "channel-executive",
            "agents": ["ceo", "auto-router"],
        },
        {
            "id": 2,
            "name": "Communication",
            "description": "Email, Slack, WhatsApp, calendar, and social outreach. Manages all inbound and outbound messaging across platforms.",
            "avatar_url": None,
            "member_count": 7,
            "tags": ["Email", "Slack", "WhatsApp", "Calendar"],
            "category": "operations",
            "is_public": True,
            "is_member": True,
            "channel_slug": "communication",
            "channel_id": "channel-communication",
            "agents": ["communication-manager", "email-agent", "slack-agent", "social-agent", "whatsapp-agent", "calendar-agent"],
        },
        {
            "id": 3,
            "name": "Content",
            "description": "Article research, writing, social media management, and editorial workflow for Street Voices publications.",
            "avatar_url": None,
            "member_count": 4,
            "tags": ["Articles", "Writing", "Social Media"],
            "category": "creative",
            "is_public": True,
            "is_member": True,
            "channel_slug": "content",
            "channel_id": "channel-content",
            "agents": ["content-manager", "article-researcher", "article-writer", "social-media"],
        },
        {
            "id": 4,
            "name": "Development",
            "description": "Full-stack engineering, database administration, DevOps, and infrastructure management.",
            "avatar_url": None,
            "member_count": 5,
            "tags": ["Engineering", "Backend", "Frontend", "DevOps"],
            "category": "technical",
            "is_public": True,
            "is_member": True,
            "channel_slug": "development",
            "channel_id": "channel-development",
            "agents": ["dev-manager", "backend-dev", "frontend-dev", "database-admin", "devops"],
        },
        {
            "id": 5,
            "name": "Finance",
            "description": "Financial management, accounting, crypto operations, and budget tracking for Street Voices.",
            "avatar_url": None,
            "member_count": 3,
            "tags": ["Finance", "Accounting", "Crypto"],
            "category": "operations",
            "is_public": True,
            "is_member": True,
            "channel_slug": "finance",
            "channel_id": "channel-finance",
            "agents": ["finance-manager", "accounting-agent", "crypto-agent"],
        },
        {
            "id": 6,
            "name": "Grant Writing",
            "description": "Grant research, proposal writing, budget planning, and project management for funding applications.",
            "avatar_url": None,
            "member_count": 5,
            "tags": ["Grants", "Proposals", "Budgets"],
            "category": "operations",
            "is_public": True,
            "is_member": True,
            "channel_slug": "grant",
            "channel_id": "channel-grant",
            "agents": ["grant-manager", "grant-writer", "budget-manager", "project-manager"],
        },
        {
            "id": 7,
            "name": "Research",
            "description": "Media platform analysis, program research, and strategic insights for Street Voices initiatives.",
            "avatar_url": None,
            "member_count": 4,
            "tags": ["Research", "Analysis", "Insights"],
            "category": "research",
            "is_public": True,
            "is_member": True,
            "channel_slug": "research",
            "channel_id": "channel-research",
            "agents": ["research-manager", "media-platform-researcher", "media-program-researcher", "street-bot-researcher"],
        },
        {
            "id": 8,
            "name": "Scraping",
            "description": "Web scraping, data collection, and automated information gathering from online sources.",
            "avatar_url": None,
            "member_count": 3,
            "tags": ["Data", "Scraping", "Automation"],
            "category": "technical",
            "is_public": True,
            "is_member": True,
            "channel_slug": "scraping",
            "channel_id": "channel-scraping",
            "agents": ["scraping-manager", "scraping-agent"],
        },
    ]
    # Enrich groups with latest message from Social DB
    try:
        pool = await _get_social_db()
        channel_ids = [t["channel_id"] for t in teams]
        # Fetch latest message + unread count per channel in one query
        rows = await pool.fetch("""
            SELECT DISTINCT ON (m.channel_id)
                m.channel_id,
                m.content,
                m.created_at,
                u.username AS author_username,
                u.display_name AS author_display_name,
                u.is_agent AS author_is_agent
            FROM messages m
            JOIN users u ON m.author_id = u.id
            WHERE m.channel_id = ANY($1::text[])
            ORDER BY m.channel_id, m.created_at DESC
        """, channel_ids)
        latest_map = {}
        for row in rows:
            latest_map[row["channel_id"]] = {
                "content": (row["content"][:120] + "…") if len(row["content"]) > 120 else row["content"],
                "timestamp": row["created_at"].isoformat(),
                "author": row["author_display_name"] or row["author_username"],
                "is_agent": row["author_is_agent"],
            }
        # Also get message counts per channel
        count_rows = await pool.fetch("""
            SELECT channel_id, COUNT(*) as msg_count
            FROM messages
            WHERE channel_id = ANY($1::text[])
            GROUP BY channel_id
        """, channel_ids)
        count_map = {r["channel_id"]: r["msg_count"] for r in count_rows}

        for team in teams:
            cid = team["channel_id"]
            team["last_message"] = latest_map.get(cid)
            team["message_count"] = count_map.get(cid, 0)
    except Exception as e:
        logger.warning(f"Failed to enrich groups with messages: {e}")
        for team in teams:
            team["last_message"] = None
            team["message_count"] = 0

    # Support single group lookup via ?id=N
    group_id = request.query_params.get("id")
    if group_id:
        try:
            gid = int(group_id)
            group = next((g for g in teams if g["id"] == gid), None)
            if group:
                return JSONResponse(group)
            return JSONResponse({"error": "Group not found"}, status_code=404)
        except ValueError:
            pass
    return JSONResponse(teams)


# ── Group → Channel ID mapping ──────────────────────────────────
_GROUP_CHANNEL_MAP = {
    1: "channel-executive",
    2: "channel-communication",
    3: "channel-content",
    4: "channel-development",
    5: "channel-finance",
    6: "channel-grant",
    7: "channel-research",
    8: "channel-scraping",
}

SOCIAL_DB_URL = "postgresql://lobehub:lobehub_password@localhost:5433/social"

async def _get_social_db():
    """Get or create asyncpg connection pool for social DB."""
    import asyncpg
    if not hasattr(_get_social_db, "_pool") or _get_social_db._pool is None:
        _get_social_db._pool = await asyncpg.create_pool(SOCIAL_DB_URL, min_size=1, max_size=5)
    return _get_social_db._pool

_get_social_db._pool = None


async def group_messages(request: Request) -> JSONResponse:
    """GET /groups/{group_id}/messages — Fetch messages for a group's channel."""
    group_id = int(request.path_params["group_id"])
    channel_id = _GROUP_CHANNEL_MAP.get(group_id)
    if not channel_id:
        return JSONResponse({"error": "Group not found"}, status_code=404)

    pool = await _get_social_db()
    rows = await pool.fetch("""
        SELECT m.id, m.content, m.created_at, m.is_edited, m.is_pinned,
               u.id as author_id, u.username, u.display_name, u.avatar_url, u.is_agent
        FROM messages m
        JOIN users u ON m.author_id = u.id
        WHERE m.channel_id = $1 AND m.deleted_at IS NULL AND m.parent_id IS NULL
        ORDER BY m.created_at ASC
        LIMIT 100
    """, channel_id)

    messages = []
    for row in rows:
        messages.append({
            "id": row["id"],
            "content": row["content"],
            "createdAt": row["created_at"].isoformat(),
            "isEdited": row["is_edited"],
            "isPinned": row["is_pinned"],
            "author": {
                "id": row["author_id"],
                "username": row["username"],
                "displayName": row["display_name"],
                "avatarUrl": row["avatar_url"],
                "isAgent": row["is_agent"],
            },
        })
    return JSONResponse({"messages": messages})


async def group_send_message(request: Request) -> JSONResponse:
    """POST /groups/{group_id}/messages — Send a message to a group channel."""
    group_id = int(request.path_params["group_id"])
    channel_id = _GROUP_CHANNEL_MAP.get(group_id)
    if not channel_id:
        return JSONResponse({"error": "Group not found"}, status_code=404)

    body = await request.json()
    content = body.get("content", "").strip()
    if not content:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # Use Joel's user ID
    user_id = "cmmq4buiv0000qrrtsv31r71l"

    pool = await _get_social_db()

    # Ensure Joel is a member of this channel
    member = await pool.fetchrow(
        "SELECT id FROM channel_members WHERE channel_id = $1 AND user_id = $2",
        channel_id, user_id,
    )
    if not member:
        import uuid
        member_id = f"member-{channel_id}-joel"
        await pool.execute(
            "INSERT INTO channel_members (id, channel_id, user_id, role, joined_at) VALUES ($1, $2, $3, 'member', NOW()) ON CONFLICT (channel_id, user_id) DO NOTHING",
            member_id, channel_id, user_id,
        )

    # Insert message
    import uuid
    msg_id = str(uuid.uuid4())
    await pool.execute("""
        INSERT INTO messages (id, channel_id, author_id, content, created_at, updated_at, is_edited, is_pinned)
        VALUES ($1, $2, $3, $4, NOW(), NOW(), false, false)
    """, msg_id, channel_id, user_id, content)

    # Fetch the inserted message back
    row = await pool.fetchrow("""
        SELECT m.id, m.content, m.created_at,
               u.id as author_id, u.username, u.display_name, u.avatar_url, u.is_agent
        FROM messages m JOIN users u ON m.author_id = u.id
        WHERE m.id = $1
    """, msg_id)

    msg = {
        "id": row["id"],
        "content": row["content"],
        "createdAt": row["created_at"].isoformat(),
        "isEdited": False,
        "isPinned": False,
        "author": {
            "id": row["author_id"],
            "username": row["username"],
            "displayName": row["display_name"],
            "avatarUrl": row["avatar_url"],
            "isAgent": row["is_agent"],
        },
    }

    # Check for @agent mentions and trigger agent response
    mentioned_agents = []
    for agent_name in body.get("agents", []):
        if f"@{agent_name}" in content:
            mentioned_agents.append(agent_name)

    # If any agent was mentioned, trigger a response asynchronously
    if mentioned_agents:
        async def _trigger_agent(agent_username: str):
            """Send the message to nanobot for agent to respond, then save the reply."""
            try:
                agent_user = await pool.fetchrow(
                    "SELECT id, username, display_name FROM users WHERE username = $1 AND is_agent = true",
                    agent_username,
                )
                if not agent_user:
                    return

                # Process through the shared nanobot agent
                if _agent is None:
                    logger.error("Agent not initialized, cannot process group mention")
                    return
                result = await _agent.process_direct(
                    f"[Group chat #{group_id}] {content}",
                    channel=f"group-{group_id}",
                    session_key=f"group:{group_id}",
                )

                # Save agent reply
                reply_id = str(uuid.uuid4())
                await pool.execute("""
                    INSERT INTO messages (id, channel_id, author_id, content, created_at, updated_at, is_edited, is_pinned)
                    VALUES ($1, $2, $3, $4, NOW(), NOW(), false, false)
                """, reply_id, channel_id, agent_user["id"], result or "I couldn't process that request.")
            except Exception as e:
                logger.error(f"Agent response error: {e}")

        for agent in mentioned_agents:
            asyncio.create_task(_trigger_agent(agent))

    return JSONResponse(msg, status_code=201)


async def group_members(request: Request) -> JSONResponse:
    """GET /groups/{group_id}/members — List members of a group channel."""
    group_id = int(request.path_params["group_id"])
    channel_id = _GROUP_CHANNEL_MAP.get(group_id)
    if not channel_id:
        return JSONResponse({"error": "Group not found"}, status_code=404)

    pool = await _get_social_db()
    rows = await pool.fetch("""
        SELECT u.id, u.username, u.display_name, u.avatar_url, u.is_agent, u.status,
               cm.role, cm.joined_at
        FROM channel_members cm
        JOIN users u ON cm.user_id = u.id
        WHERE cm.channel_id = $1
        ORDER BY u.is_agent ASC, u.display_name ASC
    """, channel_id)

    members = []
    for row in rows:
        members.append({
            "id": row["id"],
            "username": row["username"],
            "displayName": row["display_name"],
            "avatarUrl": row["avatar_url"],
            "isAgent": row["is_agent"],
            "status": row["status"] or "offline",
            "role": row["role"],
            "joinedAt": row["joined_at"].isoformat() if row["joined_at"] else None,
        })
    return JSONResponse({"members": members})


async def memory_api(request: Request) -> JSONResponse:
    """GET/POST /v1/memory — Universal shared memory API.

    GET: Returns full memory snapshot (shared, contacts, decisions, projects, agents, topics)
    POST: Write to memory (target, content, name, mode)
    """
    if not _harness:
        return JSONResponse({"error": "Deep Agent Harness not initialized"}, status_code=503)

    from nanobot.harness.api import handle_memory
    return await handle_memory(request, _harness)


async def memory_context_api(request: Request) -> JSONResponse:
    """GET /v1/memory/context/{agent_name} — Get the scoped context an agent sees."""
    if not _harness:
        return JSONResponse({"error": "Deep Agent Harness not initialized"}, status_code=503)

    agent_name = request.path_params.get("agent_name", "ceo")
    # Resolve team from agent spec for scoped context
    agent_spec = _harness._agents.get(agent_name, {})
    agent_team = agent_spec.get("team")

    context = _harness.memory.build_context_for_agent(agent_name, team=agent_team)
    return JSONResponse({
        "agent": agent_name,
        "team": agent_team,
        "context": context,
        "context_length": len(context),
    })


async def agent_sessions_api(request: Request) -> JSONResponse:
    """GET /v1/agents/{agent_name}/sessions — Get an agent's recent conversation summaries.

    This is how frontends can show conversation history per agent.
    The CEO gets all agents' sessions; individual agents get only their own.
    """
    if not _harness:
        return JSONResponse({"error": "Deep Agent Harness not initialized"}, status_code=503)

    agent_name = request.path_params.get("agent_name", "ceo")
    limit = int(request.query_params.get("limit", "10"))

    if agent_name == "ceo":
        sessions = _harness.memory._get_recent_sessions(limit=limit, agent_name=None)
    else:
        sessions = _harness.memory._get_recent_sessions(limit=limit, agent_name=agent_name)

    return JSONResponse({
        "agent": agent_name,
        "sessions": sessions,
        "count": len(sessions),
    })


async def harness_status(request: Request) -> JSONResponse:
    """GET /v1/harness/status — Deep Agent Harness status and diagnostics."""
    if not _harness:
        return JSONResponse({
            "status": "not_initialized",
            "engine": "deepagents",
            "agents": 0,
        })

    return JSONResponse({
        "status": "ready" if _harness.is_initialized else "initializing",
        "engine": "deepagents",
        "agents": _harness.agent_count,
        "memory": {
            "shared_size": len(_harness.memory.get_shared_context()),
            "contacts_size": len(_harness.memory.get_contacts()),
            "decisions_size": len(_harness.memory.get_decisions()),
            "projects_size": len(_harness.memory.get_projects()),
        },
    })


async def serve_dashboard(request: Request):
    """Serve the mission control dashboard SPA."""
    index_path = GATEWAY_STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "Dashboard not found"}, status_code=404)
    return HTMLResponse(index_path.read_text())


async def gateway_ws(websocket):
    """WebSocket endpoint — delegates to GatewayServer if available."""
    if _gateway is None:
        await websocket.close(1013, "Gateway not initialized")
        return
    await _gateway._ws_endpoint(websocket)


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
        Route("/v1/agents/status", agents_status, methods=["GET"]),
        Route("/v1/agents/list", agents_list, methods=["GET"]),
        Route("/groups", groups_api, methods=["GET"]),
        Route("/groups/{group_id:int}/messages", group_messages, methods=["GET"]),
        Route("/groups/{group_id:int}/messages", group_send_message, methods=["POST"]),
        Route("/groups/{group_id:int}/members", group_members, methods=["GET"]),
        Route("/shared/{filename:path}", serve_shared_asset, methods=["GET"]),
        Route("/api/article-image/generate", generate_article_image, methods=["POST"]),
        Route("/article-images/{filename:path}", serve_article_image, methods=["GET"]),
        # ── Deep Agent Harness endpoints ──
        Route("/v1/memory", memory_api, methods=["GET", "POST"]),
        Route("/v1/memory/context/{agent_name}", memory_context_api, methods=["GET"]),
        Route("/v1/agents/{agent_name}/sessions", agent_sessions_api, methods=["GET"]),
        Route("/v1/harness/status", harness_status, methods=["GET"]),
        # ── Mission Control dashboard ──
        Route("/dashboard", serve_dashboard, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=str(GATEWAY_STATIC_DIR)), name="gateway-static"),
        # ── WebSocket gateway ──
        WebSocketRoute("/ws", gateway_ws),
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
