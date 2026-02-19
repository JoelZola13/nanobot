"""OpenAI-compatible API server that wraps nanobot's AgentLoop."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

SCREENSHOTS_DIR = Path.home() / ".nanobot" / "workspace" / "screenshots"

from nanobot.config.loader import load_config
from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.session.manager import SessionManager
from nanobot.cron.service import CronService


_agent: AgentLoop | None = None


def _make_provider(config):
    """Create the appropriate LLM provider from config (mirrors cli/commands.py)."""
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
    global _agent
    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron_service = CronService(store_path=cron_store_path)

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
    )
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

    # Start Slack channel if enabled
    channel_manager = None
    slack_cfg = config.channels.slack
    if slack_cfg.enabled and slack_cfg.bot_token and slack_cfg.app_token:
        from nanobot.channels.manager import ChannelManager
        channel_manager = ChannelManager(config, bus)
        # Start the agent loop (consumes inbound from bus, publishes outbound)
        agent_task = asyncio.create_task(_agent.run())
        # Start the channel manager (starts Slack + outbound dispatcher)
        channels_task = asyncio.create_task(channel_manager.start_all())
        logger.info("Slack channel started")

    # Ensure screenshots directory exists
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Start cron service
    await cron_service.start()
    logger.info(f"Cron service started ({len(cron_service.list_jobs(include_disabled=True))} jobs)")

    logger.info("Nanobot API server ready")
    yield
    cron_service.stop()
    if channel_manager:
        await channel_manager.stop_all()
        _agent.stop()
    await _agent.close_mcp()
    _agent = None


def _extract_user_content(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message content from OpenAI-format messages."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part["text"])
                return "\n".join(parts)
            return content
    return ""


async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Handle POST /v1/chat/completions."""
    if not _agent:
        return JSONResponse({"error": "Agent not initialized"}, status_code=503)

    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = body.get("model", "nanobot")

    user_content = _extract_user_content(messages)
    if not user_content:
        return JSONResponse({"error": "No user message found"}, status_code=400)

    session_key = request.headers.get("x-session-id", "librechat:default")

    if stream:
        return StreamingResponse(
            _stream_response(user_content, session_key, model),
            media_type="text/event-stream",
        )

    response_text = await _agent.process_direct(
        content=user_content,
        session_key=session_key,
        channel="api",
        chat_id="librechat",
    )

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


async def _stream_response(content: str, session_key: str, model: str):
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
            result = await _agent.process_direct(
                content=content,
                session_key=session_key,
                channel="api",
                chat_id="librechat",
                on_progress=_on_progress,
            )
            await progress_queue.put(None)  # Signal done
            return result
        except Exception as e:
            await progress_queue.put(None)
            return f"Error: {e}"

    # Start agent processing in background
    agent_task = asyncio.create_task(_run_agent())

    # Stream intermediate progress as it arrives
    progress_parts = []
    while True:
        item = await progress_queue.get()
        if item is None:
            break
        progress_parts.append(item)
        # Send progress as a chunk (shown as "thinking" text)
        progress_text = f"⏳ {item}\n\n"
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": progress_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Get final response
    response_text = await agent_task

    # Send the final response
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": response_text},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Send done
    done_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def list_models(request: Request) -> JSONResponse:
    """Handle GET /v1/models."""
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "nanobot",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nanobot",
        }],
    })


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def serve_screenshot(request: Request) -> FileResponse | JSONResponse:
    """Serve a saved browser screenshot image."""
    filename = request.path_params["filename"]
    filepath = SCREENSHOTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/png")


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/screenshots/{filename:path}", serve_screenshot, methods=["GET"]),
    ],
    lifespan=lifespan,
)
