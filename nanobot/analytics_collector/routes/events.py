"""POST /api/analytics/events/batch — ingest a batch of client events.

Pipeline per event:
  1. envelope + per-event prop validation (catalog)
  2. server-side privacy scrub (defense in depth)
  3. INSERT into analytics_events (ON CONFLICT (event_id) DO UPDATE WHERE source = 'client')
  4. Side-effect projections: page_views (page_entered), page_durations (page_exited), clicks
  5. Publish safe summary to Redis live channel
  6. Bump session counters
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import load_config
from ..db import get_pool
from ..privacy import scrub
from ..redis_bus import publish_live, safe_summary
from ..schema_validator import (
    is_replay_on_sensitive_route,
    validate_event,
)


async def ingest_batch(request: Request) -> JSONResponse:
    cfg = load_config()
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    events = payload.get("events") or []
    if not isinstance(events, list):
        return JSONResponse({"error": "events must be a list"}, status_code=400)

    pool = await get_pool(cfg.db_url)
    accepted = 0
    rejected = 0
    violations: list[dict[str, Any]] = []

    async with pool.acquire() as conn:
        async with conn.transaction():
            for raw_event in events:
                if not isinstance(raw_event, dict):
                    rejected += 1
                    continue

                # 1. catalog validation -----------------------------------
                issues = validate_event(raw_event)
                if any(i.type in ("missing_envelope", "unknown_event") for i in issues):
                    rejected += 1
                    for issue in issues:
                        violations.append({
                            "event_name":   raw_event.get("event_name"),
                            "received_id":  raw_event.get("event_id"),
                            "type":         issue.type,
                            "detail":       issue.detail,
                            "user_role":    raw_event.get("user_role"),
                            "app_version":  raw_event.get("app_version"),
                            "environment":  raw_event.get("environment"),
                            "payload":      raw_event,
                        })
                    continue

                # missing_required is a soft warning — we still store the event.
                for issue in issues:
                    if issue.type == "missing_required":
                        violations.append({
                            "event_name":   raw_event.get("event_name"),
                            "received_id":  raw_event.get("event_id"),
                            "type":         issue.type,
                            "detail":       issue.detail,
                            "user_role":    raw_event.get("user_role"),
                            "app_version":  raw_event.get("app_version"),
                            "environment":  raw_event.get("environment"),
                            "payload":      None,  # no need to store the whole thing
                        })

                if is_replay_on_sensitive_route(raw_event):
                    violations.append({
                        "event_name":   raw_event.get("event_name"),
                        "received_id":  raw_event.get("event_id"),
                        "type":         "replay_on_sensitive_route",
                        "detail":       f"replay capture on sensitive route {raw_event.get('route')}",
                        "user_role":    raw_event.get("user_role"),
                        "app_version":  raw_event.get("app_version"),
                        "environment":  raw_event.get("environment"),
                        "payload":      None,
                    })
                    # We still drop the event itself.
                    rejected += 1
                    continue

                # 2. privacy scrub ----------------------------------------
                cleaned_props, redacted_paths = scrub(raw_event.get("properties") or {})
                cleaned_ctx,   _              = scrub(raw_event.get("context") or {})
                if redacted_paths:
                    violations.append({
                        "event_name":   raw_event.get("event_name"),
                        "received_id":  raw_event.get("event_id"),
                        "type":         "disallowed_pii",
                        "detail":       "redacted: " + ",".join(redacted_paths)[:500],
                        "user_role":    raw_event.get("user_role"),
                        "app_version":  raw_event.get("app_version"),
                        "environment":  raw_event.get("environment"),
                        "payload":      None,
                    })

                # 3. insert event -----------------------------------------
                event_id = _coerce_uuid(raw_event.get("event_id"))
                if event_id is None:
                    rejected += 1
                    continue
                session_id = _coerce_uuid(raw_event.get("session_id"))
                occurred_at = _coerce_ts(raw_event.get("timestamp"))

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
                        $20, $21, $22, $23, 'client',
                        $24
                    )
                    ON CONFLICT (event_id) DO UPDATE
                       SET source = CASE WHEN analytics_events.source = 'server' THEN 'both' ELSE 'client' END
                       WHERE analytics_events.source <> 'server';
                    """,
                    event_id,
                    raw_event.get("event_name"),
                    raw_event.get("product_area"),
                    session_id,
                    raw_event.get("distinct_id"),
                    raw_event.get("user_id"),
                    raw_event.get("anonymous_id"),
                    raw_event.get("street_profile_id"),
                    raw_event.get("route"),
                    raw_event.get("route_pattern"),
                    raw_event.get("page_title"),
                    raw_event.get("entry_point"),
                    raw_event.get("auth_state"),
                    raw_event.get("user_role"),
                    raw_event.get("device_type"),
                    raw_event.get("viewport_bucket"),
                    raw_event.get("environment") or "local",
                    raw_event.get("app_variant"),
                    raw_event.get("app_version"),
                    raw_event.get("feature_flags") or [],
                    raw_event.get("consent_state") or "full",
                    json.dumps(cleaned_props),
                    json.dumps(cleaned_ctx),
                    occurred_at,
                )
                accepted += 1

                # 4. side-effect projections -------------------------------
                await _project_event(conn, raw_event, event_id, session_id, cleaned_props, occurred_at)

    # 5. publish live summaries (outside the txn so failures don't roll back) -
    for raw_event in events:
        if isinstance(raw_event, dict):
            try:
                await publish_live(cfg.redis_url, safe_summary(raw_event))
            except Exception:
                pass  # live stream is best-effort

    # 6. log violations -----------------------------------------------------
    if violations:
        await _log_violations(cfg.db_url, violations)

    return JSONResponse({"accepted": accepted, "rejected": rejected, "violations": len(violations)})


async def _project_event(
    conn: Any,
    raw_event: dict[str, Any],
    event_id: UUID,
    session_id: UUID | None,
    cleaned_props: dict[str, Any],
    occurred_at: datetime,
) -> None:
    name = raw_event.get("event_name")

    if name == "page_entered":
        await conn.execute(
            """
            INSERT INTO analytics_page_views (
                event_id, session_id, user_id, street_profile_id, product_area,
                route, route_pattern, page_title, entry_point, referrer_route_pattern, entered_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (event_id) DO NOTHING;
            """,
            event_id, session_id,
            raw_event.get("user_id"), raw_event.get("street_profile_id"),
            raw_event.get("product_area"),
            raw_event.get("route"), raw_event.get("route_pattern"),
            raw_event.get("page_title"), raw_event.get("entry_point"),
            cleaned_props.get("referrer_route_pattern"),
            occurred_at,
        )

    elif name == "page_exited":
        # Update the latest open page-view for this session+route, then snapshot
        # to analytics_page_durations.
        active_ms = int(cleaned_props.get("active_time_ms") or 0)
        idle_ms   = int(cleaned_props.get("idle_time_ms") or 0)
        scroll    = int(cleaned_props.get("scroll_depth_percent") or 0)
        click_n   = int(cleaned_props.get("click_count") or 0)
        rage_n    = int(cleaned_props.get("rage_click_count") or 0)
        bucket    = _duration_bucket_ms(active_ms)
        page_view = await conn.fetchrow(
            """
            UPDATE analytics_page_views
               SET exited_at = $1, active_time_ms = $2, idle_time_ms = $3,
                   scroll_depth_percent = $4, click_count = $5, rage_click_count = $6
             WHERE id = (
                 SELECT id FROM analytics_page_views
                  WHERE session_id = $7 AND route_pattern = $8 AND exited_at IS NULL
                  ORDER BY entered_at DESC LIMIT 1
             )
            RETURNING id, user_id, street_profile_id, product_area, route_pattern;
            """,
            occurred_at, active_ms, idle_ms, scroll, click_n, rage_n,
            session_id, raw_event.get("route_pattern"),
        )
        if page_view:
            await conn.execute(
                """
                INSERT INTO analytics_page_durations (
                    page_view_id, session_id, user_id, street_profile_id, product_area,
                    route_pattern, active_time_ms, idle_time_ms, scroll_depth_percent, bucket, occurred_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11);
                """,
                page_view["id"], session_id,
                page_view["user_id"], page_view["street_profile_id"],
                page_view["product_area"], page_view["route_pattern"],
                active_ms, idle_ms, scroll, bucket, occurred_at,
            )

    elif name in ("element_clicked", "cta_clicked", "navigation_clicked", "dead_click_detected", "rage_click_detected"):
        click_type = {
            "element_clicked":     "element",
            "cta_clicked":         "cta",
            "navigation_clicked":  "navigation",
            "dead_click_detected": "dead",
            "rage_click_detected": "rage",
        }[name]
        await conn.execute(
            """
            INSERT INTO analytics_clicks (
                event_id, session_id, user_id, street_profile_id, product_area,
                route_pattern, click_type, label_key, element_role, destination,
                is_dead, rage_count, occurred_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (event_id) DO NOTHING;
            """,
            event_id, session_id,
            raw_event.get("user_id"), raw_event.get("street_profile_id"),
            raw_event.get("product_area"), raw_event.get("route_pattern"),
            click_type,
            cleaned_props.get("cta") or cleaned_props.get("label_key"),
            cleaned_props.get("element_role"),
            cleaned_props.get("destination") or cleaned_props.get("to_route"),
            click_type == "dead",
            int(cleaned_props.get("count") or 0) if click_type == "rage" else 0,
            occurred_at,
        )

    # Bump session counters lightly.
    if session_id is not None:
        await conn.execute(
            """
            UPDATE analytics_sessions
               SET last_seen_at = $1,
                   event_count  = event_count + 1,
                   page_view_count = page_view_count + CASE WHEN $2 = 'page_entered' THEN 1 ELSE 0 END,
                   active_time_ms  = active_time_ms + COALESCE($3, 0),
                   idle_time_ms    = idle_time_ms   + COALESCE($4, 0),
                   rage_click_count = rage_click_count + CASE WHEN $2 = 'rage_click_detected' THEN 1 ELSE 0 END,
                   error_count = error_count + CASE WHEN $2 IN ('page_error_occurred','platform_api_request_failed','ai_error_seen') THEN 1 ELSE 0 END,
                   last_product_area = COALESCE($5, last_product_area),
                   exit_route_pattern = COALESCE($6, exit_route_pattern),
                   user_id = COALESCE(analytics_sessions.user_id, $7),
                   street_profile_id = COALESCE(analytics_sessions.street_profile_id, $8)
             WHERE id = $9;
            """,
            occurred_at,
            name,
            int(cleaned_props.get("active_time_ms") or 0) if name == "page_exited" else None,
            int(cleaned_props.get("idle_time_ms") or 0)   if name == "page_exited" else None,
            raw_event.get("product_area"),
            raw_event.get("route_pattern") if name == "page_exited" else None,
            raw_event.get("user_id"),
            raw_event.get("street_profile_id"),
            session_id,
        )


async def _log_violations(db_url: str, violations: list[dict[str, Any]]) -> None:
    pool = await get_pool(db_url)
    async with pool.acquire() as conn:
        for v in violations:
            await conn.execute(
                """
                INSERT INTO analytics_event_schema_violations (
                    event_name, received_event_id, violation_type, violation_detail,
                    received_payload, user_role, app_version, environment
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
                """,
                v["event_name"],
                _coerce_uuid(v.get("received_id")),
                v["type"],
                v["detail"],
                json.dumps(v["payload"]) if v.get("payload") is not None else None,
                v.get("user_role"),
                v.get("app_version"),
                v.get("environment") or "local",
            )


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


def _coerce_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        # ISO 8601, e.g. 2026-04-26T20:13:00.000Z
        s = str(value).rstrip("Z")
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _duration_bucket_ms(ms: int) -> str:
    s = ms / 1000
    if s < 5:    return "<5s"
    if s < 30:   return "5-30s"
    if s < 120:  return "30s-2m"
    if s < 600:  return "2-10m"
    return ">10m"
