"""POST /api/analytics/sessions — open a session row, return session_id."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import load_config
from ..db import get_pool, serialise


async def create_session(request: Request) -> JSONResponse:
    cfg = load_config()
    try:
        payload: dict[str, Any] = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    pool = await get_pool(cfg.db_url)
    user_agent = (payload.get("user_agent") or "")[:512]
    viewport_width = int(payload.get("viewport_width") or 0)

    row = await pool.fetchrow(
        """
        INSERT INTO analytics_sessions (
            anonymous_id, user_id, street_profile_id, app_variant, app_version,
            environment, consent_state, user_agent_family, os_family, device_type,
            viewport_bucket, referrer_host, entry_route_pattern
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING id, started_at;
        """,
        payload.get("anonymous_id"),
        payload.get("user_id"),
        payload.get("street_profile_id"),
        payload.get("app_variant") or "streetbot",
        payload.get("app_version"),
        payload.get("environment") or "local",
        payload.get("consent_state") or "full",
        _ua_family(user_agent),
        _os_family(user_agent),
        _device_type(user_agent),
        _viewport_bucket(viewport_width),
        _referrer_host(payload.get("referrer")),
        payload.get("entry_route") or "/",
    )

    if row is None:
        return JSONResponse({"error": "session_create_failed"}, status_code=500)

    return JSONResponse({
        "session_id": str(row["id"]),
        "started_at": row["started_at"].astimezone(timezone.utc).isoformat(),
    })


def _ua_family(ua: str) -> str:
    u = ua.lower()
    if "edg/" in u: return "edge"
    if "firefox/" in u: return "firefox"
    if "chrome/" in u: return "chrome"
    if "safari/" in u: return "safari"
    return "other"


def _os_family(ua: str) -> str:
    u = ua.lower()
    if "windows" in u: return "windows"
    if "mac os" in u or "macintosh" in u: return "mac"
    if "android" in u: return "android"
    if "iphone" in u or "ipad" in u or "ipod" in u: return "ios"
    if "linux" in u: return "linux"
    return "other"


def _device_type(ua: str) -> str:
    u = ua.lower()
    if "ipad" in u or "tablet" in u: return "tablet"
    if "iphone" in u or "android" in u or "mobile" in u: return "mobile"
    return "desktop"


def _viewport_bucket(width: int) -> str:
    if width < 640: return "sm"
    if width < 1024: return "md"
    if width < 1440: return "lg"
    return "xl"


def _referrer_host(referrer: str | None) -> str | None:
    if not referrer:
        return None
    try:
        from urllib.parse import urlparse
        return urlparse(referrer).netloc or None
    except Exception:
        return None
