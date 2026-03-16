"""API integration — wire the Deep Agent Harness into nanobot's API server.

This module provides the Starlette route handlers that replace/augment
the existing orchestrator with the deepagents-powered harness.

The harness serves the same OpenAI-compatible API at /v1/chat/completions
and /v1/models, but uses deepagents' LangGraph execution engine with
universal shared memory.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from nanobot.harness.harness import DeepAgentHarness


async def handle_chat_completions(
    request: Request,
    harness: DeepAgentHarness,
) -> JSONResponse | StreamingResponse:
    """Handle POST /v1/chat/completions using the deep agent harness.

    This is a drop-in replacement for the existing handler in api_server.py.
    It accepts the same OpenAI-compatible request format and returns the same
    response format, but routes through deepagents' LangGraph engine.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON body"}},
            status_code=400,
        )

    messages = body.get("messages", [])
    model = body.get("model", "agent/ceo")
    stream = body.get("stream", False)

    # Extract agent name from model field (e.g., "agent/ceo" → "ceo")
    agent_name = model
    if agent_name.startswith("agent/"):
        agent_name = agent_name[6:]
    if agent_name == "auto" or not agent_name:
        agent_name = "ceo"

    # Session ID from headers or generate
    session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("x-conversation-id")
        or body.get("session_id")
    )

    if stream:
        return StreamingResponse(
            _stream_response(harness, agent_name, messages, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await harness.run(
            agent_name=agent_name,
            messages=messages,
            session_id=session_id,
        )
        return JSONResponse(result.to_chat_completion(model=model))


async def _stream_response(
    harness: DeepAgentHarness,
    agent_name: str,
    messages: list[dict[str, Any]],
    session_id: str | None,
) -> Any:
    """Generate SSE stream from harness."""
    try:
        async for chunk in harness.run_stream(
            agent_name=agent_name,
            messages=messages,
            session_id=session_id,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_chunk = {
            "id": f"chatcmpl-error",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": f"agent/{agent_name}",
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\nError: {e}"},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


async def handle_models(harness: DeepAgentHarness) -> JSONResponse:
    """Handle GET /v1/models using the harness agent registry."""
    models = harness.list_agents()

    # Also add a meta-model for "auto" routing
    models.insert(0, {
        "id": "agent/auto",
        "object": "model",
        "created": 0,
        "owned_by": "nanobot-deepagents",
        "root": "agent/auto",
        "parent": None,
        "metadata": {
            "description": "Auto-route to the best agent for your request",
            "engine": "deepagents",
        },
    })

    return JSONResponse({
        "object": "list",
        "data": models,
    })


async def handle_memory(request: Request, harness: DeepAgentHarness) -> JSONResponse:
    """Handle GET/POST /v1/memory — universal memory API.

    GET: Returns the full memory snapshot
    POST: Write to specific memory targets
    """
    if request.method == "GET":
        return JSONResponse(harness.get_memory_snapshot())

    elif request.method == "POST":
        try:
            body = await request.json()
            memory = harness.get_memory()

            target = body.get("target", "shared")
            content = body.get("content", "")
            name = body.get("name")
            mode = body.get("mode", "append")

            if target == "shared":
                if mode == "replace":
                    memory.update_shared(content)
                else:
                    memory.append_shared(content)
            elif target == "contacts":
                if name:
                    memory.append_contact(name, content)
                else:
                    return JSONResponse({"error": "name required for contacts"}, status_code=400)
            elif target == "decisions":
                memory.log_decision(content)
            elif target == "projects":
                memory.update_projects(content)
            else:
                return JSONResponse({"error": f"Unknown target: {target}"}, status_code=400)

            return JSONResponse({"status": "ok", "target": target, "mode": mode})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"error": "Method not allowed"}, status_code=405)


async def handle_health(harness: DeepAgentHarness) -> JSONResponse:
    """Enhanced health check with harness status."""
    return JSONResponse({
        "status": "ok",
        "engine": "deepagents",
        "agents": harness.agent_count,
        "initialized": harness.is_initialized,
        "memory": {
            "has_shared": bool(harness.memory.get_shared_context().strip()),
            "has_contacts": bool(harness.memory.get_contacts().strip()),
        },
    })
