"""Wires routes, lifespan, and rollup loop. Exposed via:

  build_routes()        — list of Starlette Route objects to mount in nanobot
  lifespan()            — async context manager — pools open at startup, close at shutdown
  attach_to_lifespan()  — wraps the host's existing lifespan and merges ours
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Iterable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .config import load_config
from .db import close_pool, get_pool
from .redis_bus import close_client, get_client
from .rollups import run_loop

from .routes.events import ingest_batch
from .routes.health import health
from .routes.live import live_stream
from .routes.posthog_proxy import insight_query, list_replays, replay_deeplink
from .routes import query as q
from .routes.server_event import server_capture
from .routes.sessions import create_session


def build_routes(prefix: str = "/api/analytics") -> list[Route | Mount]:
    """Return the Starlette routes for the collector. Mount in api_server.py."""
    return [
        Mount(prefix, routes=[
            Route("/sessions",        create_session, methods=["POST"]),
            Route("/events/batch",    ingest_batch,   methods=["POST"]),
            Route("/server-event",    server_capture, methods=["POST"]),
            Route("/live",            live_stream,    methods=["GET"]),
            Route("/health",          health,         methods=["GET"]),

            # Dashboard reads (admin-only)
            Route("/query/overview",         q.overview,       methods=["GET"]),
            Route("/query/product-areas",    q.product_areas,  methods=["GET"]),
            Route("/query/retention",        q.retention,      methods=["GET"]),
            Route("/query/profiles",         q.profiles,       methods=["GET"]),
            Route("/query/profile/{profile_id}", q.profile_detail, methods=["GET"]),
            Route("/query/funnels",          q.funnels,        methods=["GET"]),
            Route("/query/funnel/{funnel_key}", q.funnel_detail, methods=["GET"]),
            Route("/query/journeys",         q.journeys,       methods=["GET"]),
            Route("/query/clicks/top",       q.clicks_top,     methods=["GET"]),
            Route("/query/clicks/dead",      q.clicks_dead,    methods=["GET"]),
            Route("/query/pages/top",        q.pages_top,      methods=["GET"]),
            Route("/query/pages/longest",    q.pages_longest,  methods=["GET"]),
            Route("/query/pages/exits",      q.pages_exits,    methods=["GET"]),
            Route("/query/data-quality",     q.data_quality,   methods=["GET"]),
            Route("/query/platform/health",  q.platform_health,methods=["GET"]),
            Route("/query/platform/api",     q.platform_api,   methods=["GET"]),
            Route("/query/alerts",           q.alerts,         methods=["GET"]),
            Route("/query/events/recent",    q.events_recent,  methods=["GET"]),
            Route("/query/section/home",     q.section_home,     methods=["GET"]),
            Route("/query/section/new-chat",      q.section_new_chat,      methods=["GET"]),
            Route("/query/section/notifications", q.section_notifications, methods=["GET"]),
            Route("/query/section/search",        q.section_search,        methods=["GET"]),
            Route("/query/section/email",         q.section_email,         methods=["GET"]),
            Route("/query/section/street-profile",q.section_street_profile,methods=["GET"]),
            Route("/query/section/gallery",       q.section_gallery,       methods=["GET"]),
            Route("/query/section/news",          q.section_news,          methods=["GET"]),
            Route("/query/section/directory",     q.section_directory,     methods=["GET"]),
            Route("/query/section/jobs",          q.section_jobs,          methods=["GET"]),
            Route("/query/section/academy",       q.section_academy,       methods=["GET"]),
            Route("/query/section/calendar",      q.section_calendar,      methods=["GET"]),
            Route("/query/section/case-management",q.section_case_management,methods=["GET"]),
            Route("/query/replay/sessions",                    q.replay_sessions,        methods=["GET"]),
            Route("/query/replay/session/{session_id}",        q.replay_session_detail,  methods=["GET"]),

            # PostHog proxy
            Route("/posthog/insight",  insight_query,   methods=["POST"]),
            Route("/posthog/replay",   replay_deeplink, methods=["POST"]),
            Route("/posthog/replays",  list_replays,    methods=["GET"]),
        ])
    ]


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    cfg = load_config()
    logging.getLogger("analytics_collector").setLevel(
        logging.DEBUG if cfg.debug else logging.INFO,
    )
    await get_pool(cfg.db_url)
    await get_client(cfg.redis_url)

    rollup_task = asyncio.create_task(run_loop(cfg), name="analytics-rollups")
    try:
        yield
    finally:
        rollup_task.cancel()
        try:
            await rollup_task
        except (asyncio.CancelledError, Exception):
            pass
        await close_client()
        await close_pool()


def attach_to_lifespan(host_lifespan: Callable[[Starlette], AsyncIterator[None]]) -> Callable[[Starlette], AsyncIterator[None]]:
    """Compose with an existing host lifespan. Both run; we wrap inside the host's.

        from analytics_collector import attach_to_lifespan
        app = Starlette(routes=routes, lifespan=attach_to_lifespan(my_lifespan))
    """
    @asynccontextmanager
    async def merged(app: Starlette) -> AsyncIterator[None]:
        async with host_lifespan(app):
            async with lifespan(app):
                yield
    return merged


# Allow running standalone for development:
#   uvicorn analytics_collector.app:standalone --host 0.0.0.0 --port 18890
standalone = Starlette(routes=build_routes(), lifespan=lifespan)
