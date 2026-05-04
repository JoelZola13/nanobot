"""GET /api/analytics/live — Server-Sent Events stream of safe event summaries.

The dashboard's Live tab opens this stream and renders summaries as they
arrive. Backfills the last N items from analytics_live_events on connect."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from starlette.requests import Request
from starlette.responses import StreamingResponse

from ..auth import require_admin
from ..config import load_config
from ..db import get_pool, serialise
from ..redis_bus import subscribe_live


async def live_stream(request: Request) -> StreamingResponse:
    _, err = await require_admin(request)
    if err is not None:
        return err

    cfg = load_config()

    async def gen() -> AsyncGenerator[bytes, None]:
        # Backfill the last 100 events.
        pool = await get_pool(cfg.db_url)
        rows = await pool.fetch(
            """
            SELECT event_name, product_area, route_pattern, user_role,
                   street_profile_id, occurred_at
              FROM analytics_events
             WHERE occurred_at > NOW() - INTERVAL '5 minutes'
             ORDER BY occurred_at ASC
             LIMIT 100;
            """
        )
        for row in rows:
            yield _sse(serialise(row))

        # Then subscribe to the live channel.
        async for summary in subscribe_live(cfg.redis_url):
            if await request.is_disconnected():
                break
            yield _sse(summary)

        # Heartbeat in case nothing is happening.
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(15)
            yield b": heartbeat\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data, default=str)}\n\n".encode("utf-8")
