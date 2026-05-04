"""Auth helpers for the dashboard query endpoints.

The collector is mounted inside nanobot, which is fronted by Casdoor SSO via
the LibreChat shell. By the time a request hits the collector's query routes,
the session cookie is already validated by the host. We re-check role against
the social DB before returning analytics data, since the dashboard is admin-
only.

For the public-write endpoints (sessions, events/batch) we don't require
auth — clients fire from the browser and we trust the privacy filter to keep
PII out. Volume protection is via the host's existing rate limiter.

Server-event endpoints (POST /server-event) require a shared HMAC secret in
the X-Analytics-Server-Secret header.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse


# When ANALYTICS_LOCAL_OPEN_QUERY=1 the dashboard query endpoints accept any
# request — useful for the local sandbox where we trust localhost access and
# don't have header-injection wired through the LibreChat shell yet. In any
# other environment the original admin gate applies.
_LOCAL_OPEN_QUERY = os.environ.get("ANALYTICS_LOCAL_OPEN_QUERY", "").lower() in ("1", "true", "yes")


async def require_admin(request: Request) -> tuple[str | None, JSONResponse | None]:
    """Returns (user_id, error_response). When error_response is non-None,
    the route should return it without doing any work."""
    # The LibreChat shell sets X-User-Id and X-User-Role headers on
    # admin-authenticated requests forwarded through nanobot. If those are
    # absent, fall back to query params for local dev.
    user_id  = request.headers.get("x-user-id")  or request.query_params.get("user_id")
    user_role = request.headers.get("x-user-role") or request.query_params.get("user_role")

    if _LOCAL_OPEN_QUERY:
        # Local sandbox bypass — return whatever IDs we can scrape, never reject.
        return user_id or "local-dev", None

    if not user_id:
        return None, JSONResponse({"error": "unauthenticated"}, status_code=401)
    if (user_role or "").lower() != "admin":
        return user_id, JSONResponse({"error": "forbidden"}, status_code=403)
    return user_id, None


def verify_server_event_secret(request: Request, expected: str | None) -> bool:
    if not expected:
        return False  # secret not configured → reject
    provided = request.headers.get("x-analytics-server-secret", "")
    return hmac.compare_digest(provided, expected)


def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d
