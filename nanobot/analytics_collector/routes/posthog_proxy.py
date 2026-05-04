"""PostHog proxy. Browser never sees the personal API key — it asks the
collector for funnels / retention / replay deeplinks, and the collector forwards
to PostHog with the server-only key.

Endpoints:
  POST /api/analytics/posthog/insight   { query: <PostHog HogQL> }
  POST /api/analytics/posthog/replay    { recording_id, reason } → audited deeplink
  GET  /api/analytics/posthog/replays   ?user_id=&street_profile_id=&since=
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..auth import require_admin
from ..config import load_config
from ..db import get_pool


async def insight_query(request: Request) -> JSONResponse:
    user_id, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    if not cfg.posthog_api_key or not cfg.posthog_project_id:
        return JSONResponse({"error": "posthog_not_configured"}, status_code=503)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    query = body.get("query")
    if not isinstance(query, dict):
        return JSONResponse({"error": "missing query object"}, status_code=400)

    url = f"{cfg.posthog_host.rstrip('/')}/api/projects/{cfg.posthog_project_id}/query/"
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.post(
                url,
                json={"query": query},
                headers={"Authorization": f"Bearer {cfg.posthog_api_key}"},
            )
        except httpx.HTTPError as exc:
            return JSONResponse({"error": "posthog_unreachable", "detail": str(exc)}, status_code=502)

    if resp.status_code >= 400:
        return JSONResponse({"error": "posthog_error", "status": resp.status_code, "body": resp.text[:500]}, status_code=502)
    return JSONResponse(resp.json())


async def replay_deeplink(request: Request) -> JSONResponse:
    user_id, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    recording_id = body.get("recording_id")
    if not recording_id:
        return JSONResponse({"error": "missing recording_id"}, status_code=400)

    reason = (body.get("reason") or "").strip()
    target_user_id        = body.get("target_user_id")
    target_profile_id     = body.get("target_street_profile_id")

    pool = await get_pool(cfg.db_url)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO analytics_replay_audit (
                admin_user_id, posthog_recording_id, target_user_id,
                target_street_profile_id, reason
            ) VALUES ($1, $2, $3, $4, $5);
            """,
            user_id, recording_id, target_user_id, target_profile_id, reason or None,
        )

    deeplink = f"{cfg.posthog_host.rstrip('/')}/replay/{recording_id}"
    return JSONResponse({"recording_id": recording_id, "url": deeplink})


async def list_replays(request: Request) -> JSONResponse:
    user_id, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    if not cfg.posthog_api_key or not cfg.posthog_project_id:
        return JSONResponse({"error": "posthog_not_configured"}, status_code=503)

    params: dict[str, Any] = {}
    target_user_id    = request.query_params.get("user_id")
    target_profile_id = request.query_params.get("street_profile_id")
    since             = request.query_params.get("since")
    if target_user_id:    params["distinct_id"] = target_user_id
    if target_profile_id: params["properties"]  = json.dumps([{"key": "street_profile_id", "value": target_profile_id}])
    if since:             params["date_from"]   = since

    url = f"{cfg.posthog_host.rstrip('/')}/api/projects/{cfg.posthog_project_id}/session_recordings/"
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(
                url, params=params,
                headers={"Authorization": f"Bearer {cfg.posthog_api_key}"},
            )
        except httpx.HTTPError as exc:
            return JSONResponse({"error": "posthog_unreachable", "detail": str(exc)}, status_code=502)
    if resp.status_code >= 400:
        return JSONResponse({"error": "posthog_error", "status": resp.status_code, "body": resp.text[:500]}, status_code=502)
    return JSONResponse(resp.json())
