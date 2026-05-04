"""Configuration loaded from environment variables. No secrets are accepted via
HTTP. PostHog API keys live here only; the dashboard reaches them through this
process and never directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CollectorConfig:
    db_url:                 str
    redis_url:              str
    posthog_api_key:        str | None
    posthog_project_id:     str | None
    posthog_host:           str
    server_event_secret:    str | None
    rollup_interval_sec:    int
    live_buffer_max:        int
    debug:                  bool


def load_config() -> CollectorConfig:
    return CollectorConfig(
        db_url               = os.environ["ANALYTICS_DB_URL"],
        redis_url            = os.environ.get("ANALYTICS_REDIS_URL", "redis://sv-redis:6379"),
        posthog_api_key      = os.environ.get("POSTHOG_PROJECT_API_KEY"),
        posthog_project_id   = os.environ.get("POSTHOG_PROJECT_ID"),
        posthog_host         = os.environ.get("POSTHOG_HOST", "https://app.posthog.com"),
        server_event_secret  = os.environ.get("ANALYTICS_SERVER_EVENT_SECRET"),
        rollup_interval_sec  = int(os.environ.get("ANALYTICS_ROLLUP_INTERVAL_SEC", "60")),
        live_buffer_max      = int(os.environ.get("ANALYTICS_LIVE_BUFFER_MAX", "1000")),
        debug                = os.environ.get("ANALYTICS_DEBUG", "0") == "1",
    )
