"""Helper for backend services to emit trusted truth events.

Use from any Python service mounted in the same process tree as the collector,
or any other backend that can reach the collector over HTTP and shares the
ANALYTICS_SERVER_EVENT_SECRET.

    from analytics_collector.server_event import capture as capture_server_event

    await capture_server_event(
        event_name="jobs_application_submitted",
        event_id=client_event_id,            # for dedup
        user_id=user.id,
        street_profile_id=profile.id,
        properties={"job_id": job.id, "submission_type": "internal", "docs_count": 2},
    )
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx


log = logging.getLogger("analytics_collector.server_event")


async def capture(
    *,
    event_name:           str,
    event_id:             str | None = None,
    user_id:              str | None = None,
    street_profile_id:    str | None = None,
    product_area:         str | None = None,
    route_pattern:        str | None = None,
    properties:           dict[str, Any] | None = None,
    timestamp:            datetime | None = None,
    collector_url:        str | None = None,
    secret:               str | None = None,
) -> bool:
    """POST to /api/analytics/server-event. Returns True on success."""
    url = (collector_url or os.environ.get("ANALYTICS_COLLECTOR_URL", "")).rstrip("/")
    secret = secret or os.environ.get("ANALYTICS_SERVER_EVENT_SECRET", "")
    if not url or not secret:
        log.debug("server-event skipped — no collector_url/secret configured")
        return False

    body = {
        "event_id":          event_id,
        "event_name":        event_name,
        "user_id":           user_id,
        "street_profile_id": street_profile_id,
        "product_area":      product_area,
        "route_pattern":     route_pattern,
        "properties":        properties or {},
        "timestamp":         (timestamp or datetime.now(timezone.utc)).isoformat(),
    }
    headers = {"x-analytics-server-secret": secret}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{url}/server-event", json=body, headers=headers)
        if r.status_code >= 400:
            log.warning("server-event rejected: %s %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as exc:
        log.warning("server-event failed: %s", exc)
        return False
