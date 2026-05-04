"""POST /api/analytics/server-event — backend services emit trusted truth events.

Auth: shared HMAC secret via X-Analytics-Server-Secret header.

If the same event_id was previously inserted by the client, the row's source
flips to 'both' (server wins on truth — the client's optimistic capture is
preserved by event_id but the server payload is canonical).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..auth import verify_server_event_secret
from ..config import load_config
from ..db import get_pool
from ..privacy import scrub
from ..redis_bus import publish_live, safe_summary
from ..schema_validator import validate_event


async def server_capture(request: Request) -> JSONResponse:
    cfg = load_config()
    if not verify_server_event_secret(request, cfg.server_event_secret):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    event = _build_envelope(payload)
    issues = validate_event(event)
    if any(i.type in ("missing_envelope", "unknown_event") for i in issues):
        return JSONResponse({"error": "invalid_event", "issues": [i.detail for i in issues]}, status_code=400)

    cleaned_props, _ = scrub(event.get("properties") or {})
    cleaned_ctx,   _ = scrub(event.get("context") or {})

    pool = await get_pool(cfg.db_url)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO analytics_events (
                event_id, event_name, product_area, session_id, distinct_id,
                user_id, anonymous_id, street_profile_id, route, route_pattern,
                page_title, entry_point, auth_state, user_role, device_type,
                viewport_bucket, environment, app_variant, app_version,
                feature_flags, consent_state, properties, context, source,
                occurred_at
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15,
                $16, $17, $18, $19,
                $20, $21, $22, $23, 'server',
                $24
            )
            ON CONFLICT (event_id) DO UPDATE
               SET source = 'both',
                   properties = EXCLUDED.properties,
                   user_id = COALESCE(EXCLUDED.user_id, analytics_events.user_id),
                   street_profile_id = COALESCE(EXCLUDED.street_profile_id, analytics_events.street_profile_id);
            """,
            _coerce_uuid(event.get("event_id")) or uuid4(),
            event.get("event_name"),
            event.get("product_area"),
            _coerce_uuid(event.get("session_id")),
            event.get("distinct_id") or event.get("user_id") or "server",
            event.get("user_id"),
            event.get("anonymous_id"),
            event.get("street_profile_id"),
            event.get("route"),
            event.get("route_pattern"),
            event.get("page_title"),
            event.get("entry_point") or "direct_url",
            event.get("auth_state") or ("authenticated" if event.get("user_id") else "anonymous"),
            event.get("user_role") or "unknown",
            event.get("device_type") or "desktop",
            event.get("viewport_bucket") or "lg",
            event.get("environment") or "local",
            event.get("app_variant") or "streetbot",
            event.get("app_version"),
            event.get("feature_flags") or [],
            event.get("consent_state") or "full",
            json.dumps(cleaned_props),
            json.dumps(cleaned_ctx),
            _coerce_ts(event.get("timestamp")),
        )

    try:
        await publish_live(cfg.redis_url, safe_summary(event))
    except Exception:
        pass

    return JSONResponse({"accepted": 1})


def _build_envelope(p: dict[str, Any]) -> dict[str, Any]:
    """Server callers can supply a partial envelope; we fill in safe defaults."""
    out = dict(p)
    out.setdefault("event_id", str(uuid4()))
    out.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    out.setdefault("environment", "production")
    out.setdefault("app_variant", "streetbot")
    out.setdefault("consent_state", "full")
    out.setdefault("auth_state", "authenticated" if p.get("user_id") else "anonymous")
    out.setdefault("device_type", "desktop")
    out.setdefault("viewport_bucket", "lg")
    out.setdefault("feature_flags", [])
    out.setdefault("entry_point", "direct_url")
    out.setdefault("user_role", "unknown")
    out.setdefault("distinct_id", p.get("user_id") or "server")
    out.setdefault("route", p.get("route_pattern") or "/")
    out.setdefault("route_pattern", p.get("route") or "/")
    return out


def _coerce_uuid(v: Any) -> UUID | None:
    if v is None: return None
    if isinstance(v, UUID): return v
    try: return UUID(str(v))
    except (ValueError, TypeError): return None


def _coerce_ts(v: Any) -> datetime:
    if isinstance(v, datetime): return v
    try:
        s = str(v).rstrip("Z")
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)
