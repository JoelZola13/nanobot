"""Redis pub/sub bus for the Live tab. We publish ONLY safe summaries — never
the full event — to keep the channel cheap and unimportant if leaked."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import redis.asyncio as redis_async


CHANNEL_LIVE = "analytics:live"


_client: redis_async.Redis | None = None


async def get_client(redis_url: str) -> redis_async.Redis:
    global _client
    if _client is None:
        _client = redis_async.from_url(redis_url, decode_responses=True)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def publish_live(redis_url: str, summary: dict[str, Any]) -> None:
    client = await get_client(redis_url)
    await client.publish(CHANNEL_LIVE, json.dumps(summary, default=str))


async def subscribe_live(redis_url: str) -> AsyncIterator[dict[str, Any]]:
    client = await get_client(redis_url)
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    await pubsub.subscribe(CHANNEL_LIVE)
    try:
        async for msg in pubsub.listen():
            data = msg.get("data")
            if not data:
                continue
            try:
                yield json.loads(data)
            except (TypeError, json.JSONDecodeError):
                continue
    finally:
        await pubsub.unsubscribe(CHANNEL_LIVE)
        await pubsub.aclose()


def safe_summary(event: dict[str, Any]) -> dict[str, Any]:
    """Project a captured event down to a tiny stream-safe shape."""
    return {
        "event_name":      event.get("event_name"),
        "product_area":    event.get("product_area"),
        "route_pattern":   event.get("route_pattern"),
        "user_role":       event.get("user_role"),
        "street_profile_id": event.get("street_profile_id"),
        "occurred_at":     event.get("timestamp"),
    }
