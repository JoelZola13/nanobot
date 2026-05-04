"""GET /api/analytics/health — collector self-check."""

from __future__ import annotations

from datetime import datetime, timezone

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import load_config
from ..db import get_pool
from ..redis_bus import get_client


async def health(request: Request) -> JSONResponse:
    cfg = load_config()
    db_ok = True
    redis_ok = True
    try:
        pool = await get_pool(cfg.db_url)
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1;")
    except Exception:
        db_ok = False
    try:
        client = await get_client(cfg.redis_url)
        await client.ping()
    except Exception:
        redis_ok = False
    return JSONResponse({
        "status":  "ok" if (db_ok and redis_ok) else "degraded",
        "db":      db_ok,
        "redis":   redis_ok,
        "now":     datetime.now(timezone.utc).isoformat(),
    })
