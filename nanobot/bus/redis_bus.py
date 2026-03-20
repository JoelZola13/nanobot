"""Redis-backed event bus for cross-service communication."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from loguru import logger

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore


class RedisBus:
    """
    Pub/sub event bus backed by Redis.

    Enables cross-service event flow between nanobot, SV Social, and LibreChat.
    Services publish events to named channels; subscribers receive them in real-time.
    """

    def __init__(self, url: str = "redis://localhost:6380"):
        if aioredis is None:
            raise ImportError("redis package required: pip install redis[hiredis]")
        self._url = url
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._subscribers: dict[str, list[Callable[[dict], Awaitable[None]]]] = {}
        self._listener_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Connect to Redis and start the subscriber listener."""
        self._redis = aioredis.from_url(self._url, decode_responses=True)
        self._pubsub = self._redis.pubsub()
        self._running = True

        # Subscribe to any channels that were registered before start
        if self._subscribers:
            await self._pubsub.subscribe(*self._subscribers.keys())

        self._listener_task = asyncio.create_task(self._listen())
        logger.info(f"RedisBus started (url={self._url}, channels={list(self._subscribers.keys())})")

    async def stop(self) -> None:
        """Disconnect from Redis and stop the listener."""
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        logger.info("RedisBus stopped")

    async def publish(self, channel: str, event: dict[str, Any]) -> None:
        """Publish an event to a Redis channel."""
        if not self._redis:
            logger.warning("RedisBus not connected, dropping event")
            return
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(event, default=str)
        await self._redis.publish(channel, payload)

    def subscribe(self, channel: str, callback: Callable[[dict], Awaitable[None]]) -> None:
        """Register a callback for events on a channel. Call before start() or dynamically."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(callback)

        # If already running, subscribe dynamically
        if self._pubsub and self._running:
            asyncio.create_task(self._pubsub.subscribe(channel))

    async def _listen(self) -> None:
        """Background task: read messages from Redis pub/sub and dispatch to callbacks."""
        try:
            while self._running:
                msg = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg["type"] == "message":
                    channel = msg["channel"]
                    try:
                        data = json.loads(msg["data"])
                    except (json.JSONDecodeError, TypeError):
                        data = {"raw": msg["data"]}

                    callbacks = self._subscribers.get(channel, [])
                    for cb in callbacks:
                        try:
                            await cb(data)
                        except Exception as e:
                            logger.error(f"RedisBus callback error on {channel}: {e}")
                else:
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"RedisBus listener error: {e}")
