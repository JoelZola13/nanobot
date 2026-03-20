"""Bidirectional sync between Nanobot API and Paperclip.

Posts activity events and cost reports back to Paperclip after
every chat completion, keeping the two systems connected as one.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

PAPERCLIP_API = "http://127.0.0.1:3100/api"
COMPANY_ID = "78940514-fbb0-4c2d-8cee-09bcfd5399a4"

# Agent name -> Paperclip agent ID cache (populated on first use)
_agent_id_cache: dict[str, str] = {}
_cache_loaded = False
_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(timeout=10.0)
    return _http


async def _load_agent_cache() -> None:
    """Load Paperclip agent list and build name->ID mapping."""
    global _cache_loaded
    if _cache_loaded:
        return
    try:
        resp = await _client().get(f"{PAPERCLIP_API}/companies/{COMPANY_ID}/agents")
        resp.raise_for_status()
        agents = resp.json()
        for agent in agents:
            name = (agent.get("metadata") or {}).get("nanobotName", "")
            if name:
                _agent_id_cache[name] = agent["id"]
            # Also map by display name (lowercased, underscored)
            display = agent.get("name", "")
            key = display.lower().replace(" ", "_")
            if key:
                _agent_id_cache[key] = agent["id"]
        _cache_loaded = True
        logger.info(f"[paperclip-sync] Cached {len(_agent_id_cache)} agent mappings")
    except Exception as e:
        logger.warning(f"[paperclip-sync] Failed to load agent cache: {e}")


def _resolve_agent_id(agent_name: str) -> str | None:
    """Resolve a nanobot agent name to a Paperclip agent UUID."""
    if not agent_name:
        return None
    # Direct match
    if agent_name in _agent_id_cache:
        return _agent_id_cache[agent_name]
    # Normalize
    normalized = agent_name.lower().replace("-", "_").replace(" ", "_")
    return _agent_id_cache.get(normalized)


async def post_completion_event(
    *,
    agent_name: str | None = None,
    session_key: str = "",
    model: str = "",
    usage: dict[str, Any] | None = None,
    response_length: int = 0,
    tools_used: list[str] | None = None,
) -> None:
    """Post a completion event to Paperclip activity log and report costs.

    Called after every successful chat completion. Fire-and-forget —
    failures are logged but never block the response.
    """
    asyncio.create_task(_post_event(
        agent_name=agent_name,
        session_key=session_key,
        model=model,
        usage=usage or {},
        response_length=response_length,
        tools_used=tools_used or [],
    ))


async def _post_event(
    *,
    agent_name: str | None,
    session_key: str,
    model: str,
    usage: dict[str, Any],
    response_length: int,
    tools_used: list[str],
) -> None:
    """Internal: post activity + cost to Paperclip."""
    try:
        await _load_agent_cache()
        client = _client()

        total_tokens = usage.get("total_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Post activity event
        agent_id = _resolve_agent_id(agent_name or "")
        activity_payload = {
            "action": "agent.chat_completion",
            "actorType": "agent" if agent_id else "user",
            "actorId": agent_id or "nanobot-api",
            "entityType": "agent",
            "entityId": agent_id or "nanobot-api",
            "details": {
                "session": session_key,
                "model": model,
                "tokens": total_tokens,
                "responseLength": response_length,
                "tools": tools_used[:5] if tools_used else [],
                "agentName": agent_name or "default",
            },
        }
        try:
            resp = await client.post(
                f"{PAPERCLIP_API}/companies/{COMPANY_ID}/activity",
                json=activity_payload,
            )
            if resp.status_code >= 400:
                logger.debug(f"[paperclip-sync] Activity post returned {resp.status_code}")
        except Exception as e:
            logger.debug(f"[paperclip-sync] Activity post failed: {e}")

        # Report cost if we have token usage and an agent ID
        if agent_id and total_tokens > 0:
            # Estimate cost: input ~$0.003/1K, output ~$0.015/1K (codex pricing)
            cost_cents = max(1, round(
                (prompt_tokens / 1000 * 0.3) + (completion_tokens / 1000 * 1.5)
            ))
            cost_payload = {
                "agentId": agent_id,
                "provider": "openai-codex",
                "model": model or "gpt-5.1-codex",
                "inputTokens": prompt_tokens,
                "outputTokens": completion_tokens,
                "costCents": cost_cents,
                "occurredAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            }
            try:
                resp = await client.post(
                    f"{PAPERCLIP_API}/companies/{COMPANY_ID}/cost-events",
                    json=cost_payload,
                )
                if resp.status_code < 400:
                    logger.debug(f"[paperclip-sync] Cost reported: {cost_cents}¢ for {agent_name}")
            except Exception as e:
                logger.debug(f"[paperclip-sync] Cost report failed: {e}")

    except Exception as e:
        logger.warning(f"[paperclip-sync] Event post failed: {e}")


async def update_agent_status(agent_name: str, status: str = "active") -> None:
    """Update an agent's status in Paperclip (active/idle/error)."""
    try:
        await _load_agent_cache()
        agent_id = _resolve_agent_id(agent_name)
        if not agent_id:
            return
        # Paperclip uses lastHeartbeatAt to track agent activity
        # Posting a heartbeat-like update marks the agent as recently active
        await _client().patch(
            f"{PAPERCLIP_API}/agents/{agent_id}",
            json={"status": status},
        )
    except Exception as e:
        logger.debug(f"[paperclip-sync] Status update failed for {agent_name}: {e}")


async def shutdown() -> None:
    """Close the HTTP client."""
    global _http
    if _http and not _http.is_closed:
        await _http.aclose()
        _http = None
