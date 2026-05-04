"""Local 3180 analytics collector — Starlette mount.

Usage in nanobot/api_server.py:

    from analytics_collector import build_routes as build_analytics_routes

    routes = [
        # ... existing routes ...
        *build_analytics_routes(),
    ]

The mount lifecycle (DB pool, Redis pubsub, rollup loop) is owned by an async
context manager exposed by `lifespan`. If the host application has its own
`lifespan` (nanobot does), import `attach_to_lifespan` and merge.
"""

from .app import build_routes, lifespan, attach_to_lifespan

__all__ = ["build_routes", "lifespan", "attach_to_lifespan"]
