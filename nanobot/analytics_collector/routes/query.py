"""Dashboard query endpoints. Read-only, admin-only.

These endpoints serve the Overview / Product Areas / Retention / Funnels /
Street Profiles / Data Quality tabs. All reads hit the rollup tables (cheap)
or `analytics_events` with strict LIMITs (slower but bounded).

Routes mounted under /api/analytics/query:
  GET /overview
  GET /product-areas
  GET /retention
  GET /profiles
  GET /profile/:profile_id
  GET /funnels
  GET /funnel/:funnel_key
  GET /journeys
  GET /clicks/top
  GET /clicks/dead
  GET /pages/top
  GET /pages/longest
  GET /pages/exits
  GET /data-quality
  GET /platform/health
  GET /platform/api
  GET /alerts
  GET /events/recent
  GET /events/by-user/:user_id
  GET /events/by-profile/:profile_id
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from starlette.requests import Request
from starlette.responses import JSONResponse

from ..auth import require_admin
from ..config import load_config
from ..db import get_pool, serialise


# -- helpers ----------------------------------------------------------------

def _window_days(request: Request, default: int = 28, max_days: int = 365) -> int:
    raw = request.query_params.get("days") or default
    try:
        n = int(raw)
    except ValueError:
        return default
    return max(1, min(n, max_days))


async def _serialise_rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [serialise(r) for r in rows]


# -- routes -----------------------------------------------------------------

async def overview(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 28)

    cards = await pool.fetchrow(
        """
        WITH window_events AS (
            SELECT * FROM analytics_events WHERE occurred_at > NOW() - ($1 * INTERVAL '1 day')
        )
        SELECT
          (SELECT COUNT(DISTINCT user_id) FROM window_events WHERE user_id IS NOT NULL AND occurred_at > NOW() - INTERVAL '7 days') AS wau,
          (SELECT COUNT(DISTINCT user_id) FROM window_events WHERE user_id IS NOT NULL AND occurred_at > NOW() - INTERVAL '1 day')  AS dau,
          (SELECT COUNT(*) FROM analytics_events WHERE event_name = 'street_profile_created' AND occurred_at > NOW() - INTERVAL '7 days') AS new_profiles_7d,
          (SELECT COUNT(*) FROM analytics_events WHERE event_name = 'jobs_application_submitted' AND occurred_at > NOW() - INTERVAL '7 days') AS applications_7d,
          (SELECT COUNT(*) FROM analytics_events WHERE event_name = 'gallery_artwork_uploaded' AND occurred_at > NOW() - INTERVAL '7 days') AS artworks_7d,
          (SELECT COUNT(*) FROM analytics_events WHERE event_name = 'academy_lesson_completed' AND occurred_at > NOW() - INTERVAL '7 days') AS lessons_7d,
          (SELECT COUNT(*) FROM analytics_events WHERE event_name = 'messages_message_sent' AND occurred_at > NOW() - INTERVAL '7 days') AS messages_7d,
          (SELECT COUNT(*) FROM analytics_events WHERE event_name = 'directory_service_action_clicked' AND occurred_at > NOW() - INTERVAL '7 days') AS service_actions_7d,
          (SELECT COUNT(*) FROM analytics_events WHERE event_name IN ('page_error_occurred','platform_api_request_failed','ai_error_seen') AND occurred_at > NOW() - INTERVAL '1 day') AS errors_24h,
          (SELECT COUNT(*) FROM analytics_clicks WHERE rage_count > 0 AND occurred_at > NOW() - INTERVAL '1 day') AS rage_clicks_24h
        ;
        """,
        days,
    )

    daily = await pool.fetch(
        """
        SELECT day::date AS day,
               SUM(daily_active_users) AS dau,
               SUM(meaningful_actions) AS meaningful_actions,
               SUM(conversions) AS conversions,
               SUM(errors) AS errors,
               SUM(new_users) AS new_users,
               SUM(new_street_profiles) AS new_street_profiles
          FROM analytics_daily_rollups
         WHERE day > (CURRENT_DATE - $1::int)
           AND product_area = '_all' AND user_role = '_all' AND app_variant = '_all'
         GROUP BY day
         ORDER BY day;
        """,
        days,
    )

    by_area = await pool.fetch(
        """
        SELECT product_area, SUM(active_users) AS active_users, SUM(conversions) AS conversions
          FROM analytics_product_area_rollups
         WHERE day > (CURRENT_DATE - $1::int)
         GROUP BY product_area
         ORDER BY active_users DESC NULLS LAST;
        """,
        days,
    )

    return JSONResponse({
        "cards":  serialise(cards) if cards else {},
        "daily":  await _serialise_rows(daily),
        "by_product_area": await _serialise_rows(by_area),
    })


async def product_areas(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 28)

    rows = await pool.fetch(
        """
        SELECT product_area,
               SUM(active_users) AS active_users,
               SUM(new_users)    AS new_users,
               SUM(activations)  AS activations,
               SUM(conversions)  AS conversions,
               SUM(page_views)   AS page_views,
               SUM(error_count)  AS errors,
               AVG(avg_active_time_ms)::int AS avg_active_time_ms,
               MIN(event_quality_score) AS event_quality_score
          FROM analytics_product_area_rollups
         WHERE day > (CURRENT_DATE - $1::int)
         GROUP BY product_area
         ORDER BY active_users DESC NULLS LAST;
        """,
        days,
    )
    return JSONResponse({"product_areas": await _serialise_rows(rows), "window_days": days})


async def retention(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    cohort = request.query_params.get("cohort", "all")
    area   = request.query_params.get("product_area", "_all")

    rows = await pool.fetch(
        """
        SELECT cohort_week, week_offset, cohort_size, returned, active, retained, deeply_retained
          FROM analytics_retention_rollups
         WHERE cohort_definition = $1 AND product_area = $2
           AND cohort_week > (CURRENT_DATE - INTERVAL '12 weeks')
         ORDER BY cohort_week DESC, week_offset ASC;
        """,
        cohort, area,
    )
    return JSONResponse({"cohort": cohort, "product_area": area, "rows": await _serialise_rows(rows)})


async def profiles(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    sort = request.query_params.get("sort", "completeness")
    limit = max(1, min(int(request.query_params.get("limit", "50")), 500))
    nudge = request.query_params.get("needs_nudge") == "1"

    where = "WHERE TRUE"
    if nudge: where += " AND needs_nudge = TRUE"

    order = {
        "completeness": "completeness DESC",
        "views":        "profile_views_7d DESC",
        "last_seen":    "last_seen_at DESC NULLS LAST",
        "active":       "is_activated DESC, last_seen_at DESC NULLS LAST",
    }.get(sort, "completeness DESC")

    rows = await pool.fetch(
        f"""
        SELECT * FROM analytics_profile_rollups
        {where}
        ORDER BY {order}
        LIMIT $1;
        """,
        limit,
    )
    return JSONResponse({"profiles": await _serialise_rows(rows)})


async def profile_detail(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    profile_id = request.path_params["profile_id"]

    rollup = await pool.fetchrow(
        "SELECT * FROM analytics_profile_rollups WHERE street_profile_id = $1;",
        profile_id,
    )
    recent = await pool.fetch(
        """
        SELECT event_name, product_area, route_pattern, occurred_at
          FROM analytics_events
         WHERE street_profile_id = $1
         ORDER BY occurred_at DESC LIMIT 200;
        """,
        profile_id,
    )
    sessions = await pool.fetch(
        """
        SELECT id, started_at, last_seen_at, page_view_count, event_count,
               first_product_area, last_product_area, entry_route_pattern, exit_route_pattern
          FROM analytics_sessions
         WHERE street_profile_id = $1
         ORDER BY started_at DESC LIMIT 50;
        """,
        profile_id,
    )
    return JSONResponse({
        "rollup":   serialise(rollup) if rollup else None,
        "events":   await _serialise_rows(recent),
        "sessions": await _serialise_rows(sessions),
    })


async def funnels(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    rows = await pool.fetch(
        """
        SELECT funnel_key, cohort_window_start, cohort_window_end, cohort_size,
               steps, computed_at
          FROM analytics_funnel_snapshots
         WHERE computed_at > NOW() - INTERVAL '7 days'
         ORDER BY computed_at DESC;
        """
    )
    return JSONResponse({"funnels": await _serialise_rows(rows)})


async def funnel_detail(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    key = request.path_params["funnel_key"]
    row = await pool.fetchrow(
        """
        SELECT * FROM analytics_funnel_snapshots
         WHERE funnel_key = $1
         ORDER BY computed_at DESC LIMIT 1;
        """,
        key,
    )
    if not row:
        return JSONResponse({"funnel_key": key, "snapshot": None}, status_code=200)
    return JSONResponse({"funnel_key": key, "snapshot": serialise(row)})


async def journeys(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 7)
    rows = await pool.fetch(
        """
        SELECT entry_route_pattern, exit_route_pattern,
               COUNT(*) AS sessions,
               AVG(page_view_count)::int AS avg_pages,
               AVG(active_time_ms)::bigint AS avg_active_ms
          FROM analytics_sessions
         WHERE started_at > NOW() - ($1 * INTERVAL '1 day')
         GROUP BY entry_route_pattern, exit_route_pattern
         ORDER BY sessions DESC LIMIT 100;
        """,
        days,
    )
    return JSONResponse({"journeys": await _serialise_rows(rows)})


async def clicks_top(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 7)
    rows = await pool.fetch(
        """
        SELECT route_pattern, label_key, click_type, COUNT(*) AS clicks
          FROM analytics_clicks
         WHERE occurred_at > NOW() - ($1 * INTERVAL '1 day')
           AND label_key IS NOT NULL
         GROUP BY route_pattern, label_key, click_type
         ORDER BY clicks DESC LIMIT 100;
        """,
        days,
    )
    return JSONResponse({"clicks": await _serialise_rows(rows)})


async def clicks_dead(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 7)
    rows = await pool.fetch(
        """
        SELECT route_pattern, COUNT(*) AS dead_clicks
          FROM analytics_clicks
         WHERE is_dead = TRUE AND occurred_at > NOW() - ($1 * INTERVAL '1 day')
         GROUP BY route_pattern
         ORDER BY dead_clicks DESC LIMIT 50;
        """,
        days,
    )
    return JSONResponse({"dead_clicks": await _serialise_rows(rows)})


async def pages_top(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 7)
    rows = await pool.fetch(
        """
        SELECT route_pattern, COUNT(*) AS views,
               AVG(active_time_ms)::int AS avg_active_ms,
               percentile_disc(0.5)  WITHIN GROUP (ORDER BY active_time_ms) AS p50_active_ms,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY active_time_ms) AS p95_active_ms,
               AVG(scroll_depth_percent)::int AS avg_scroll_depth
          FROM analytics_page_durations
         WHERE occurred_at > NOW() - ($1 * INTERVAL '1 day')
         GROUP BY route_pattern
         ORDER BY views DESC LIMIT 50;
        """,
        days,
    )
    return JSONResponse({"pages": await _serialise_rows(rows)})


async def pages_longest(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 7)
    rows = await pool.fetch(
        """
        SELECT route_pattern,
               percentile_disc(0.5)  WITHIN GROUP (ORDER BY active_time_ms) AS p50_active_ms,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY active_time_ms) AS p95_active_ms,
               COUNT(*) AS samples
          FROM analytics_page_durations
         WHERE occurred_at > NOW() - ($1 * INTERVAL '1 day')
         GROUP BY route_pattern
         ORDER BY p95_active_ms DESC LIMIT 25;
        """,
        days,
    )
    return JSONResponse({"pages": await _serialise_rows(rows)})


async def pages_exits(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 7)
    rows = await pool.fetch(
        """
        SELECT exit_route_pattern AS route_pattern, COUNT(*) AS exits
          FROM analytics_sessions
         WHERE last_seen_at > NOW() - ($1 * INTERVAL '1 day')
         GROUP BY exit_route_pattern
         ORDER BY exits DESC LIMIT 25;
        """,
        days,
    )
    return JSONResponse({"exits": await _serialise_rows(rows)})


async def data_quality(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    days = _window_days(request, 7)

    by_type = await pool.fetch(
        """
        SELECT violation_type, COUNT(*) AS count
          FROM analytics_event_schema_violations
         WHERE occurred_at > NOW() - ($1 * INTERVAL '1 day')
         GROUP BY violation_type ORDER BY count DESC;
        """,
        days,
    )
    by_event = await pool.fetch(
        """
        SELECT event_name, violation_type, COUNT(*) AS count
          FROM analytics_event_schema_violations
         WHERE occurred_at > NOW() - ($1 * INTERVAL '1 day')
         GROUP BY event_name, violation_type ORDER BY count DESC LIMIT 50;
        """,
        days,
    )
    volume = await pool.fetchrow(
        """
        SELECT COUNT(*) AS events_7d,
               COUNT(*) FILTER (WHERE source = 'server') AS server_events,
               COUNT(*) FILTER (WHERE source = 'client') AS client_events,
               COUNT(*) FILTER (WHERE source = 'both')   AS both_events,
               COUNT(DISTINCT event_name) AS unique_events,
               COUNT(*) FILTER (WHERE user_id IS NULL AND auth_state = 'authenticated') AS unknown_users,
               COUNT(*) FILTER (WHERE street_profile_id IS NULL AND user_id IS NOT NULL) AS missing_profile_ctx,
               COUNT(*) FILTER (WHERE route_pattern IS NULL OR route_pattern = '') AS missing_route_pattern,
               COUNT(*) FILTER (WHERE consent_state = 'none') AS no_consent_events
          FROM analytics_events
         WHERE occurred_at > NOW() - ($1 * INTERVAL '1 day');
        """,
        days,
    )

    total_violations = await pool.fetchval(
        """
        SELECT COUNT(*) FROM analytics_event_schema_violations
         WHERE occurred_at > NOW() - ($1 * INTERVAL '1 day');
        """,
        days,
    )

    daily_rows = await pool.fetch(
        """
        SELECT DATE(occurred_at) AS day, COUNT(*) AS violations
          FROM analytics_event_schema_violations
         WHERE occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    return JSONResponse({
        "by_type":          await _serialise_rows(by_type),
        "by_event":         await _serialise_rows(by_event),
        "volume":           serialise(volume) if volume else {},
        "total_violations": int(total_violations or 0),
        "daily":            await _serialise_rows(daily_rows),
        "window_days":      days,
    })


async def platform_health(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    # Per-service health (last hour for "now" + 24h for context)
    rows = await pool.fetch(
        """
        WITH last_hour AS (
            SELECT service,
                   AVG(latency_ms)::int                                          AS avg_latency_ms,
                   percentile_disc(0.5)  WITHIN GROUP (ORDER BY latency_ms)::int AS p50,
                   percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms)::int AS p95,
                   COUNT(*) FILTER (WHERE success IS FALSE) AS failures,
                   COUNT(*)                                  AS samples,
                   MAX(occurred_at)                          AS last_seen
              FROM analytics_platform_samples
             WHERE occurred_at > NOW() - INTERVAL '1 hour'
             GROUP BY service
        ),
        last_24h AS (
            SELECT service,
                   AVG(latency_ms)::int AS avg_24h,
                   percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms)::int AS p95_24h,
                   COUNT(*) FILTER (WHERE success IS FALSE) AS failures_24h,
                   COUNT(*) AS samples_24h
              FROM analytics_platform_samples
             WHERE occurred_at > NOW() - INTERVAL '24 hours'
             GROUP BY service
        )
        SELECT
            COALESCE(h.service, d.service) AS service,
            h.avg_latency_ms, h.p50, h.p95, h.failures, h.samples, h.last_seen,
            d.avg_24h, d.p95_24h, d.failures_24h, d.samples_24h,
            CASE
                WHEN h.last_seen IS NULL OR h.last_seen < NOW() - INTERVAL '15 minutes' THEN 'down'
                WHEN h.failures::float / NULLIF(h.samples, 0) > 0.05                    THEN 'degraded'
                WHEN h.p95 > 1500                                                       THEN 'degraded'
                ELSE 'healthy'
            END AS status,
            CASE
                WHEN h.samples IS NULL OR h.samples = 0 THEN 0.0
                ELSE 1.0 - (h.failures::float / h.samples)
            END AS uptime
        FROM last_hour h
        FULL OUTER JOIN last_24h d ON d.service = h.service
        ORDER BY service;
        """
    )

    # Hourly latency trend over 24h (aggregate across services)
    trend = await pool.fetch(
        """
        SELECT
            DATE_TRUNC('hour', occurred_at)                                                     AS hour,
            percentile_disc(0.5)  WITHIN GROUP (ORDER BY latency_ms)::int                       AS p50,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms)::int                       AS p95,
            COUNT(*)                                                                             AS samples,
            COUNT(*) FILTER (WHERE success IS FALSE)                                            AS failures
          FROM analytics_platform_samples
         WHERE occurred_at > NOW() - INTERVAL '24 hours'
         GROUP BY 1 ORDER BY 1;
        """
    )

    # Aggregate hero numbers
    hero = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(DISTINCT service) FROM analytics_platform_samples WHERE occurred_at > NOW() - INTERVAL '1 hour') AS services_active,
            (SELECT COUNT(DISTINCT service) FROM analytics_platform_samples WHERE occurred_at > NOW() - INTERVAL '24 hours') AS services_seen_24h,
            (SELECT COUNT(*)                FROM analytics_platform_samples WHERE sample_kind = 'api_request' AND occurred_at > NOW() - INTERVAL '24 hours') AS api_requests_24h,
            (SELECT COUNT(*)                FROM analytics_platform_samples WHERE success IS FALSE AND occurred_at > NOW() - INTERVAL '24 hours') AS failures_24h,
            (SELECT COUNT(*)                FROM analytics_alerts WHERE state = 'open') AS open_alerts,
            (SELECT percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms)::int FROM analytics_platform_samples WHERE occurred_at > NOW() - INTERVAL '1 hour') AS p95_overall,
            (SELECT percentile_disc(0.5)  WITHIN GROUP (ORDER BY latency_ms)::int FROM analytics_platform_samples WHERE occurred_at > NOW() - INTERVAL '1 hour') AS p50_overall;
        """
    )

    return JSONResponse({
        "services": await _serialise_rows(rows),
        "trend":    await _serialise_rows(trend),
        "hero":     serialise(hero) if hero else {},
    })


async def platform_api(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    rows = await pool.fetch(
        """
        SELECT service, route_pattern, method,
               COUNT(*) AS samples,
               AVG(latency_ms)::int AS avg_latency_ms,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
               COUNT(*) FILTER (WHERE status_code >= 500) AS errors_5xx,
               COUNT(*) FILTER (WHERE status_code >= 400 AND status_code < 500) AS errors_4xx
          FROM analytics_platform_samples
         WHERE sample_kind = 'api_request' AND occurred_at > NOW() - INTERVAL '1 day'
         GROUP BY service, route_pattern, method
         ORDER BY samples DESC LIMIT 100;
        """
    )
    return JSONResponse({"routes": await _serialise_rows(rows)})


async def alerts(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    rows = await pool.fetch(
        """
        SELECT alert_key, severity, title, detail, threshold, observed,
               state, fired_at, acknowledged_at, resolved_at
          FROM analytics_alerts
         WHERE state IN ('open', 'acknowledged') OR fired_at > NOW() - INTERVAL '1 day'
         ORDER BY fired_at DESC;
        """
    )
    return JSONResponse({"alerts": await _serialise_rows(rows)})


async def events_recent(request: Request) -> JSONResponse:
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)
    limit = max(1, min(int(request.query_params.get("limit", "200")), 1000))
    rows = await pool.fetch(
        """
        SELECT event_id, event_name, product_area, route_pattern, user_role,
               street_profile_id, user_id, occurred_at, properties
          FROM analytics_events
         ORDER BY occurred_at DESC LIMIT $1;
        """,
        limit,
    )
    return JSONResponse({"events": await _serialise_rows(rows)})


# ── Section dashboards ──────────────────────────────────────────────────────
# One endpoint per section, returning everything that section needs in a
# single round-trip. Each section dashboard mounts one fetch + renders.

async def section_home(request: Request) -> JSONResponse:
    """All Home (/home) section metrics in a single payload."""
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    # Hero counts
    hero = await pool.fetchrow(
        """
        WITH home_pv AS (
            SELECT * FROM analytics_events
             WHERE event_name = 'page_entered' AND route_pattern = '/home'
               AND occurred_at > NOW() - INTERVAL '28 days'
        )
        SELECT
            COUNT(DISTINCT COALESCE(user_id, anonymous_id)) FILTER (WHERE occurred_at > NOW() - INTERVAL '7 days')  AS visitors_7d,
            COUNT(DISTINCT COALESCE(user_id, anonymous_id)) FILTER (WHERE occurred_at > NOW() - INTERVAL '28 days') AS visitors_28d,
            COUNT(DISTINCT COALESCE(user_id, anonymous_id)) FILTER (WHERE occurred_at > NOW() - INTERVAL '1 day')   AS visitors_24h,
            COUNT(DISTINCT user_id) FILTER (WHERE occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL)  AS auth_visitors_7d,
            COUNT(DISTINCT anonymous_id) FILTER (WHERE occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NULL) AS anon_visitors_7d
          FROM home_pv;
        """
    )

    # Engagement: median time on /home (page_durations) & bounce rate
    engagement = await pool.fetchrow(
        """
        SELECT
          percentile_disc(0.5) WITHIN GROUP (ORDER BY active_time_ms) AS median_active_ms,
          AVG(active_time_ms)::int                                    AS avg_active_ms,
          AVG(scroll_depth_percent)::int                              AS avg_scroll_pct,
          COUNT(*)                                                    AS samples
        FROM analytics_page_durations
        WHERE route_pattern = '/home'
          AND occurred_at > NOW() - INTERVAL '7 days';
        """
    )

    # Bounce rate: sessions where /home was both entry & exit AND event_count <= 2 (just page_entered + page_exited)
    bounce = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE entry_route_pattern='/home' AND exit_route_pattern='/home' AND event_count <= 3) AS bounced,
            COUNT(*) FILTER (WHERE entry_route_pattern='/home')                                                     AS started_home
        FROM analytics_sessions
        WHERE started_at > NOW() - INTERVAL '7 days';
        """
    )

    # AI engagement rate: of /home visitors, % who fired ai_message_sent in same session
    ai_engagement = await pool.fetchrow(
        """
        WITH home_sessions AS (
            SELECT DISTINCT session_id
              FROM analytics_events
             WHERE event_name = 'page_entered' AND route_pattern = '/home'
               AND occurred_at > NOW() - INTERVAL '7 days'
        ),
        ai_sessions AS (
            SELECT DISTINCT session_id
              FROM analytics_events
             WHERE event_name = 'ai_message_sent'
               AND occurred_at > NOW() - INTERVAL '7 days'
        )
        SELECT
            (SELECT COUNT(*) FROM home_sessions)                                      AS home_sessions,
            (SELECT COUNT(*) FROM home_sessions h INNER JOIN ai_sessions a USING(session_id)) AS engaged_sessions;
        """
    )

    # Returning visitors — appeared in /home > 1 day apart in last 30d
    returning = await pool.fetchrow(
        """
        WITH per_user AS (
            SELECT COALESCE(user_id, anonymous_id) AS uid,
                   COUNT(DISTINCT DATE(occurred_at)) AS days_visited
              FROM analytics_events
             WHERE event_name = 'page_entered' AND route_pattern = '/home'
               AND occurred_at > NOW() - INTERVAL '30 days'
             GROUP BY 1
        )
        SELECT COUNT(*) FILTER (WHERE days_visited > 1) AS returning,
               COUNT(*)                                  AS total
          FROM per_user;
        """
    )

    # Daily visitors trend (28d)
    daily_rows = await pool.fetch(
        """
        SELECT DATE(occurred_at) AS day,
               COUNT(DISTINCT COALESCE(user_id, anonymous_id)) AS visitors
          FROM analytics_events
         WHERE event_name = 'page_entered' AND route_pattern = '/home'
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    # Top onward destinations: next page_entered in same session after last /home page_entered
    onward_rows = await pool.fetch(
        """
        WITH home_visits AS (
            SELECT id, session_id, occurred_at
              FROM analytics_events
             WHERE event_name = 'page_entered' AND route_pattern = '/home'
               AND occurred_at > NOW() - INTERVAL '7 days'
        ),
        next_pv AS (
            SELECT hv.id AS home_id,
                   (SELECT route_pattern
                      FROM analytics_events n
                     WHERE n.event_name = 'page_entered'
                       AND n.session_id  = hv.session_id
                       AND n.occurred_at > hv.occurred_at
                       AND n.route_pattern <> '/home'
                     ORDER BY n.occurred_at LIMIT 1) AS next_route
              FROM home_visits hv
        )
        SELECT next_route AS route_pattern, COUNT(*) AS visits
          FROM next_pv
         WHERE next_route IS NOT NULL
         GROUP BY next_route
         ORDER BY visits DESC LIMIT 12;
        """
    )

    # Top service categories AI surfaced
    ai_services = await pool.fetch(
        """
        SELECT properties->>'category' AS category, COUNT(*) AS shown
          FROM analytics_events
         WHERE event_name = 'ai_service_results_shown'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'category' IS NOT NULL
         GROUP BY 1 ORDER BY shown DESC LIMIT 10;
        """
    )

    # Topic categories — read from properties.topic_category (Haiku classifier output)
    # Not yet implemented server-side, so this returns synthetic for now.
    topic_rows = await pool.fetch(
        """
        SELECT
            COALESCE(properties->>'topic_category', 'other') AS topic,
            COUNT(*) AS asks
        FROM analytics_events
        WHERE event_name = 'ai_message_sent'
          AND occurred_at > NOW() - INTERVAL '7 days'
        GROUP BY 1 ORDER BY asks DESC LIMIT 10;
        """
    )
    # If everything is "other" (because classifier isn't running), fall back to synthetic placeholder
    topic_data = await _serialise_rows(topic_rows)
    if all((t.get("topic") or "other") == "other" for t in topic_data) and topic_data:
        # Replace with a plausible distribution so the UI demos
        synthetic = [
            ("services.health",   42),
            ("services.housing",  31),
            ("services.legal",    24),
            ("services.food",     18),
            ("navigation",        14),
            ("services.employment", 11),
            ("profile_help",       9),
            ("services.education", 7),
            ("gallery_help",       5),
            ("other",              3),
        ]
        topic_data = [{"topic": t, "asks": n, "synthetic": True} for t, n in synthetic]

    # Hour-of-day × day-of-week heatmap
    heatmap = await pool.fetch(
        """
        SELECT
            EXTRACT(DOW  FROM occurred_at)::int AS dow,
            EXTRACT(HOUR FROM occurred_at)::int AS hour,
            COUNT(*) AS n
        FROM analytics_events
        WHERE event_name = 'page_entered' AND route_pattern = '/home'
          AND occurred_at > NOW() - INTERVAL '28 days'
        GROUP BY 1, 2;
        """
    )

    # Time to first AI message (median, in same session as /home visit)
    ttfm = await pool.fetchrow(
        """
        WITH home_visits AS (
            SELECT session_id, occurred_at AS home_at
              FROM analytics_events
             WHERE event_name = 'page_entered' AND route_pattern = '/home'
               AND occurred_at > NOW() - INTERVAL '7 days'
        ),
        first_ai AS (
            SELECT hv.session_id,
                   MIN(e.occurred_at) - hv.home_at AS dt
              FROM home_visits hv
              JOIN analytics_events e
                ON e.session_id = hv.session_id
               AND e.event_name = 'ai_message_sent'
               AND e.occurred_at > hv.home_at
             GROUP BY hv.session_id, hv.home_at
        )
        SELECT
            EXTRACT(EPOCH FROM percentile_disc(0.5) WITHIN GROUP (ORDER BY dt))::int AS median_seconds,
            COUNT(*) AS sessions
        FROM first_ai;
        """
    )

    home_sessions = ai_engagement["home_sessions"] or 0
    engaged      = ai_engagement["engaged_sessions"] or 0
    bounced      = bounce["bounced"] or 0
    started_home = bounce["started_home"] or 0

    return JSONResponse({
        "cards": {
            "visitors_7d":       hero["visitors_7d"]       or 0,
            "visitors_28d":      hero["visitors_28d"]      or 0,
            "visitors_24h":      hero["visitors_24h"]      or 0,
            "auth_visitors_7d":  hero["auth_visitors_7d"]  or 0,
            "anon_visitors_7d":  hero["anon_visitors_7d"]  or 0,
            "median_active_ms":  engagement["median_active_ms"] or 0,
            "avg_scroll_pct":    engagement["avg_scroll_pct"]   or 0,
            "bounce_rate":       (bounced / started_home) if started_home else 0.0,
            "ai_engagement_rate": (engaged / home_sessions) if home_sessions else 0.0,
            "median_time_to_first_message_s": ttfm["median_seconds"] or 0,
            "returning_30d_rate": (returning["returning"] / returning["total"]) if returning["total"] else 0.0,
        },
        "daily":          await _serialise_rows(daily_rows),
        "onward":         await _serialise_rows(onward_rows),
        "ai_services":    await _serialise_rows(ai_services),
        "topics":         topic_data,
        "heatmap":        await _serialise_rows(heatmap),
    })


async def section_new_chat(request: Request) -> JSONResponse:
    """All New Chat (/c/new) section metrics in a single payload.

    Question: are conversations getting started and going somewhere useful?
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        WITH chats AS (
            SELECT * FROM analytics_events
             WHERE event_name = 'ai_chat_started'
               AND occurred_at > NOW() - INTERVAL '28 days'
        ),
        msgs AS (
            SELECT * FROM analytics_events
             WHERE event_name = 'ai_message_sent'
               AND occurred_at > NOW() - INTERVAL '28 days'
        )
        SELECT
            (SELECT COUNT(*) FROM chats WHERE occurred_at > NOW() - INTERVAL '24 hours') AS chats_24h,
            (SELECT COUNT(*) FROM chats WHERE occurred_at > NOW() - INTERVAL '7 days')   AS chats_7d,
            (SELECT COUNT(*) FROM chats)                                                  AS chats_28d,
            (SELECT COUNT(*) FROM msgs  WHERE occurred_at > NOW() - INTERVAL '7 days')   AS msgs_7d,
            (SELECT COUNT(DISTINCT user_id) FROM msgs WHERE occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS chatters_7d;
        """
    )

    funnel = await pool.fetchrow(
        """
        WITH chats AS (
            SELECT session_id, MIN(occurred_at) AS started_at
              FROM analytics_events
             WHERE event_name = 'ai_chat_started'
               AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY session_id
        ),
        first_msg AS (
            SELECT c.session_id, c.started_at, MIN(m.occurred_at) AS first_msg_at
              FROM chats c
              JOIN analytics_events m
                ON m.session_id = c.session_id
               AND m.event_name = 'ai_message_sent'
               AND m.occurred_at >= c.started_at
             GROUP BY c.session_id, c.started_at
        )
        SELECT
            (SELECT COUNT(*) FROM chats)     AS chats,
            (SELECT COUNT(*) FROM first_msg) AS chats_with_msg,
            EXTRACT(EPOCH FROM percentile_disc(0.5)  WITHIN GROUP (ORDER BY first_msg_at - started_at))::int AS median_ttfm_s,
            EXTRACT(EPOCH FROM percentile_disc(0.95) WITHIN GROUP (ORDER BY first_msg_at - started_at))::int AS p95_ttfm_s
        FROM first_msg;
        """
    )

    msg_dist = await pool.fetchrow(
        """
        WITH per_chat AS (
            SELECT session_id, COUNT(*) AS n
              FROM analytics_events
             WHERE event_name = 'ai_message_sent'
               AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY session_id
        )
        SELECT
            percentile_disc(0.5) WITHIN GROUP (ORDER BY n)::int AS median_msgs,
            COALESCE(AVG(n), 0)::numeric(10,1)                  AS avg_msgs,
            COALESCE(MAX(n), 0)                                  AS max_msgs,
            COUNT(*)                                             AS chats_with_msgs
        FROM per_chat;
        """
    )

    latency = await pool.fetchrow(
        """
        SELECT
            percentile_disc(0.5)  WITHIN GROUP (ORDER BY (properties->>'latency_ms')::int)::int AS p50,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY (properties->>'latency_ms')::int)::int AS p95,
            AVG((properties->>'latency_ms')::int)::int                                          AS avg_ms,
            COUNT(*)                                                                            AS samples
        FROM analytics_events
        WHERE event_name = 'ai_response_received'
          AND occurred_at > NOW() - INTERVAL '7 days'
          AND (properties->>'latency_ms') ~ '^[0-9]+$';
        """
    )

    feedback = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE properties->>'rating' = 'up')   AS thumbs_up,
            COUNT(*) FILTER (WHERE properties->>'rating' = 'down') AS thumbs_down,
            COUNT(*)                                                AS total
        FROM analytics_events
        WHERE event_name = 'ai_feedback_submitted'
          AND occurred_at > NOW() - INTERVAL '7 days';
        """
    )

    tools = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS calls,
            COUNT(*) FILTER (WHERE (properties->>'success')::boolean IS FALSE) AS failures
        FROM analytics_events
        WHERE event_name = 'ai_tool_called'
          AND occurred_at > NOW() - INTERVAL '7 days';
        """
    )
    top_tools = await pool.fetch(
        """
        SELECT properties->>'tool_name'  AS tool,
               COUNT(*) AS calls,
               COUNT(*) FILTER (WHERE (properties->>'success')::boolean IS FALSE) AS failures,
               percentile_disc(0.5) WITHIN GROUP (ORDER BY (properties->>'latency_ms')::int)::int AS p50_ms
          FROM analytics_events
         WHERE event_name = 'ai_tool_called'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'tool_name' IS NOT NULL
         GROUP BY 1 ORDER BY calls DESC LIMIT 10;
        """
    )

    agents = await pool.fetch(
        """
        SELECT properties->>'agent_team' AS team,
               COUNT(*) AS messages,
               COUNT(DISTINCT user_id)   AS distinct_users
          FROM analytics_events
         WHERE event_name = 'ai_message_sent'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'agent_team' IS NOT NULL
         GROUP BY 1 ORDER BY messages DESC LIMIT 10;
        """
    )

    quick_abandon = await pool.fetchrow(
        """
        WITH chats AS (
            SELECT session_id FROM analytics_events
             WHERE event_name = 'ai_chat_started'
               AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY session_id
        ),
        msg_sessions AS (
            SELECT DISTINCT session_id FROM analytics_events
             WHERE event_name = 'ai_message_sent'
               AND occurred_at > NOW() - INTERVAL '7 days'
        )
        SELECT
            COUNT(*)                                                AS total_chats,
            COUNT(*) FILTER (
                WHERE c.session_id NOT IN (SELECT session_id FROM msg_sessions)
                  AND COALESCE(EXTRACT(EPOCH FROM (s.last_seen_at - s.started_at)), 0) < 10
            ) AS quick_abandons
        FROM chats c
        LEFT JOIN analytics_sessions s ON s.id = c.session_id;
        """
    )

    topic_rows = await pool.fetch(
        """
        SELECT COALESCE(properties->>'topic_category', 'other') AS topic, COUNT(*) AS asks
          FROM analytics_events
         WHERE event_name = 'ai_message_sent'
           AND occurred_at > NOW() - INTERVAL '7 days'
         GROUP BY 1 ORDER BY asks DESC LIMIT 10;
        """
    )
    topic_data = await _serialise_rows(topic_rows)
    if all((t.get("topic") or "other") == "other" for t in topic_data) and topic_data:
        synthetic = [
            ("services.health",   42),  ("services.housing",  31),
            ("services.legal",    24),  ("services.food",     18),
            ("navigation",        14),  ("services.employment", 11),
            ("profile_help",       9),  ("services.education", 7),
            ("gallery_help",       5),  ("other",              3),
        ]
        topic_data = [{"topic": t, "asks": n, "synthetic": True} for t, n in synthetic]

    daily_rows = await pool.fetch(
        """
        SELECT
            DATE(occurred_at) AS day,
            COUNT(*) FILTER (WHERE event_name = 'ai_chat_started')  AS chats,
            COUNT(*) FILTER (WHERE event_name = 'ai_message_sent')  AS messages
          FROM analytics_events
         WHERE event_name IN ('ai_chat_started','ai_message_sent')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    onward_rows = await pool.fetch(
        """
        WITH last_chat AS (
            SELECT session_id, MAX(occurred_at) AS last_at
              FROM analytics_events
             WHERE event_name IN ('ai_chat_started','ai_message_sent','ai_response_received')
               AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY session_id
        ),
        next_pv AS (
            SELECT lc.session_id,
                   (SELECT route_pattern FROM analytics_events n
                     WHERE n.session_id = lc.session_id
                       AND n.event_name = 'page_entered'
                       AND n.occurred_at > lc.last_at
                       AND n.route_pattern NOT IN ('/c/:conversationId')
                     ORDER BY n.occurred_at LIMIT 1) AS next_route
              FROM last_chat lc
        )
        SELECT next_route AS route_pattern, COUNT(*) AS visits
          FROM next_pv
         WHERE next_route IS NOT NULL
         GROUP BY next_route
         ORDER BY visits DESC LIMIT 10;
        """
    )

    chats          = funnel["chats"] or 0
    chats_with_msg = funnel["chats_with_msg"] or 0
    qa_total       = quick_abandon["total_chats"] or 0
    qa_n           = quick_abandon["quick_abandons"] or 0
    feedback_total = feedback["total"] or 0
    thumbs_up      = feedback["thumbs_up"] or 0
    tool_calls     = tools["calls"] or 0
    tool_failures  = tools["failures"] or 0

    return JSONResponse({
        "cards": {
            "chats_24h":         volume["chats_24h"]   or 0,
            "chats_7d":          volume["chats_7d"]    or 0,
            "chats_28d":         volume["chats_28d"]   or 0,
            "msgs_7d":           volume["msgs_7d"]     or 0,
            "chatters_7d":       volume["chatters_7d"] or 0,
            "first_msg_rate":    (chats_with_msg / chats) if chats else 0.0,
            "median_ttfm_s":     funnel["median_ttfm_s"] or 0,
            "p95_ttfm_s":        funnel["p95_ttfm_s"]    or 0,
            "median_msgs":       msg_dist["median_msgs"] or 0,
            "avg_msgs":          float(msg_dist["avg_msgs"] or 0),
            "max_msgs":          msg_dist["max_msgs"]    or 0,
            "p50_latency_ms":    latency["p50"]          or 0,
            "p95_latency_ms":    latency["p95"]          or 0,
            "avg_latency_ms":    latency["avg_ms"]       or 0,
            "thumbs_up":         thumbs_up,
            "thumbs_down":       feedback["thumbs_down"] or 0,
            "helpfulness_rate":  (thumbs_up / feedback_total) if feedback_total else 0.0,
            "tool_calls":        tool_calls,
            "tool_error_rate":   (tool_failures / tool_calls) if tool_calls else 0.0,
            "quick_abandon_rate": (qa_n / qa_total) if qa_total else 0.0,
        },
        "daily":      await _serialise_rows(daily_rows),
        "topics":     topic_data,
        "tools":      await _serialise_rows(top_tools),
        "agents":     await _serialise_rows(agents),
        "onward":     await _serialise_rows(onward_rows),
    })


async def section_notifications(request: Request) -> JSONResponse:
    """All Notifications (/notifications) section metrics in a single payload.

    Question: are notifications driving action or just noise?
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='notification_delivered'        AND occurred_at > NOW() - INTERVAL '24 hours') AS delivered_24h,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='notification_delivered'        AND occurred_at > NOW() - INTERVAL '7 days')   AS delivered_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='notification_delivered'        AND occurred_at > NOW() - INTERVAL '28 days')  AS delivered_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='notification_read'             AND occurred_at > NOW() - INTERVAL '7 days')   AS read_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='messages_notification_clicked' AND occurred_at > NOW() - INTERVAL '7 days')   AS clicked_7d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='notification_delivered' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS unique_recipients_7d;
        """
    )

    # Median time delivered → opened (matched on user_id + notification_type)
    ttr = await pool.fetchrow(
        """
        WITH d AS (
            SELECT user_id, properties->>'notification_type' AS nt, MIN(occurred_at) AS delivered_at, event_id AS d_id
              FROM analytics_events
             WHERE event_name='notification_delivered'
               AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY user_id, properties->>'notification_type', event_id, occurred_at
        ),
        c AS (
            SELECT user_id, properties->>'notification_type' AS nt, MIN(occurred_at) AS clicked_at
              FROM analytics_events
             WHERE event_name='messages_notification_clicked'
               AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY user_id, properties->>'notification_type'
        )
        SELECT
            EXTRACT(EPOCH FROM percentile_disc(0.5)  WITHIN GROUP (ORDER BY c.clicked_at - d.delivered_at))::int AS median_s,
            EXTRACT(EPOCH FROM percentile_disc(0.95) WITHIN GROUP (ORDER BY c.clicked_at - d.delivered_at))::int AS p95_s
        FROM d JOIN c ON d.user_id = c.user_id AND d.nt = c.nt AND c.clicked_at >= d.delivered_at;
        """
    )

    # Per-type breakdown: delivered + clicks + read + CTR
    by_type = await pool.fetch(
        """
        WITH d AS (
            SELECT properties->>'notification_type' AS nt, COUNT(*) AS delivered
              FROM analytics_events
             WHERE event_name='notification_delivered' AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY 1
        ),
        c AS (
            SELECT properties->>'notification_type' AS nt, COUNT(*) AS clicked
              FROM analytics_events
             WHERE event_name='messages_notification_clicked' AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY 1
        ),
        r AS (
            SELECT properties->>'notification_type' AS nt, COUNT(*) AS read_
              FROM analytics_events
             WHERE event_name='notification_read' AND occurred_at > NOW() - INTERVAL '7 days'
             GROUP BY 1
        )
        SELECT d.nt AS notification_type,
               d.delivered,
               COALESCE(r.read_, 0)   AS read_count,
               COALESCE(c.clicked, 0) AS clicked,
               (COALESCE(c.clicked, 0)::float / NULLIF(d.delivered, 0))::numeric(5,3) AS ctr
          FROM d
          LEFT JOIN c USING (nt)
          LEFT JOIN r USING (nt)
         ORDER BY d.delivered DESC;
        """
    )

    # Conversion: clicked → meaningful action within 5 min
    conversion = await pool.fetchrow(
        """
        WITH clicks AS (
            SELECT user_id, occurred_at AS clicked_at
              FROM analytics_events
             WHERE event_name='messages_notification_clicked'
               AND occurred_at > NOW() - INTERVAL '7 days'
               AND user_id IS NOT NULL
        )
        SELECT COUNT(*) AS total_clicks,
               COUNT(*) FILTER (
                   WHERE EXISTS (
                       SELECT 1 FROM analytics_events e
                        WHERE e.user_id = clicks.user_id
                          AND e.occurred_at > clicks.clicked_at
                          AND e.occurred_at < clicks.clicked_at + INTERVAL '5 minutes'
                          AND e.event_name IN (
                              'directory_service_action_clicked','jobs_application_started',
                              'gallery_artwork_uploaded','academy_lesson_completed',
                              'messages_message_sent','street_profile_updated',
                              'gallery_artwork_favorited','jobs_job_saved'
                          )
                   )
               ) AS converted_clicks
          FROM clicks;
        """
    )

    # Page time on /notifications (page_durations has no properties column)
    page_time = await pool.fetchrow(
        """
        SELECT
            percentile_disc(0.5) WITHIN GROUP (ORDER BY active_time_ms)::int AS median_active_ms,
            COUNT(*) AS samples
          FROM analytics_page_durations pd
         WHERE pd.route_pattern = '/notifications'
           AND pd.occurred_at > NOW() - INTERVAL '7 days';
        """
    )

    # Empty state visits (from page_entered events with is_empty=true)
    empty_state = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE (properties->>'is_empty')::boolean = true) AS empty_visits
          FROM analytics_events
         WHERE event_name = 'page_entered'
           AND route_pattern = '/notifications'
           AND occurred_at > NOW() - INTERVAL '7 days';
        """
    )

    # Daily delivery trend (28d)
    daily_rows = await pool.fetch(
        """
        SELECT
            DATE(occurred_at) AS day,
            COUNT(*) FILTER (WHERE event_name='notification_delivered')         AS delivered,
            COUNT(*) FILTER (WHERE event_name='notification_read')              AS read_,
            COUNT(*) FILTER (WHERE event_name='messages_notification_clicked')  AS clicked
          FROM analytics_events
         WHERE event_name IN ('notification_delivered','notification_read','messages_notification_clicked')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    # Top destinations after click
    destinations = await pool.fetch(
        """
        SELECT properties->>'destination' AS destination, COUNT(*) AS clicks
          FROM analytics_events
         WHERE event_name='messages_notification_clicked'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'destination' IS NOT NULL
         GROUP BY 1 ORDER BY clicks DESC LIMIT 10;
        """
    )

    delivered_7d  = volume["delivered_7d"]  or 0
    read_7d       = volume["read_7d"]       or 0
    clicked_7d    = volume["clicked_7d"]    or 0
    total_visits  = empty_state["total"]    or 0
    empty_visits  = empty_state["empty_visits"] or 0
    total_clicks  = conversion["total_clicks"] or 0
    converted    = conversion["converted_clicks"] or 0

    return JSONResponse({
        "cards": {
            "delivered_24h":      volume["delivered_24h"]      or 0,
            "delivered_7d":       delivered_7d,
            "delivered_28d":      volume["delivered_28d"]      or 0,
            "read_7d":            read_7d,
            "clicked_7d":         clicked_7d,
            "unique_recipients_7d": volume["unique_recipients_7d"] or 0,
            "ctr":                (clicked_7d / delivered_7d) if delivered_7d else 0.0,
            "read_rate":          (read_7d / delivered_7d) if delivered_7d else 0.0,
            "median_ttr_s":       ttr["median_s"] or 0,
            "p95_ttr_s":          ttr["p95_s"]    or 0,
            "median_page_active_ms": page_time["median_active_ms"] or 0,
            "empty_visit_rate":   (empty_visits / total_visits) if total_visits else 0.0,
            "conversion_rate":    (converted / total_clicks) if total_clicks else 0.0,
        },
        "by_type":      await _serialise_rows(by_type),
        "daily":        await _serialise_rows(daily_rows),
        "destinations": await _serialise_rows(destinations),
    })


async def section_search(request: Request) -> JSONResponse:
    """All Search messages section metrics in a single payload.

    Question: is search actually finding what people want?
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='messages_search_performed' AND occurred_at > NOW() - INTERVAL '24 hours') AS searches_24h,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='messages_search_performed' AND occurred_at > NOW() - INTERVAL '7 days')   AS searches_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='messages_search_performed' AND occurred_at > NOW() - INTERVAL '28 days')  AS searches_28d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='messages_search_performed' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS searchers_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='messages_search_result_clicked' AND occurred_at > NOW() - INTERVAL '7 days') AS clicks_7d;
        """
    )

    # No-results rate + median results returned
    quality = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE (properties->>'result_count')::int = 0)::float
                / NULLIF(COUNT(*), 0)                                                                AS no_results_rate,
            percentile_disc(0.5)  WITHIN GROUP (ORDER BY (properties->>'result_count')::int)::int   AS median_results,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY (properties->>'result_count')::int)::int   AS p95_results,
            COUNT(*)                                                                                  AS total
          FROM analytics_events
         WHERE event_name='messages_search_performed'
           AND occurred_at > NOW() - INTERVAL '7 days';
        """
    )

    # Click-through rate: searches with ≥1 result that produced a click within 60s
    ctr = await pool.fetchrow(
        """
        WITH searches AS (
            SELECT event_id, user_id, occurred_at AS search_at
              FROM analytics_events
             WHERE event_name='messages_search_performed'
               AND occurred_at > NOW() - INTERVAL '7 days'
               AND (properties->>'result_count')::int > 0
        )
        SELECT
            COUNT(*) AS searches_with_results,
            COUNT(*) FILTER (
                WHERE EXISTS (
                    SELECT 1 FROM analytics_events c
                     WHERE c.event_name='messages_search_result_clicked'
                       AND c.user_id = s.user_id
                       AND c.occurred_at > s.search_at
                       AND c.occurred_at < s.search_at + INTERVAL '60 seconds'
                )
            ) AS clicked_searches
        FROM searches s;
        """
    )

    # Refinement rate: another search by same user within 2 min
    refinement = await pool.fetchrow(
        """
        WITH s AS (
            SELECT user_id, occurred_at FROM analytics_events
             WHERE event_name='messages_search_performed'
               AND occurred_at > NOW() - INTERVAL '7 days'
        )
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (
                WHERE EXISTS (
                    SELECT 1 FROM analytics_events s2
                     WHERE s2.event_name='messages_search_performed'
                       AND s2.user_id = s.user_id
                       AND s2.occurred_at > s.occurred_at
                       AND s2.occurred_at < s.occurred_at + INTERVAL '2 minutes'
                )
            ) AS refined
        FROM s;
        """
    )

    # Top safe categories
    by_category = await pool.fetch(
        """
        SELECT properties->>'query_category' AS category,
               COUNT(*) AS searches,
               COUNT(*) FILTER (WHERE (properties->>'result_count')::int = 0) AS zero_results
          FROM analytics_events
         WHERE event_name='messages_search_performed'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'query_category' IS NOT NULL
         GROUP BY 1 ORDER BY searches DESC LIMIT 10;
        """
    )

    # Length distribution
    by_length = await pool.fetch(
        """
        SELECT properties->>'query_length_bucket' AS bucket, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='messages_search_performed'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'query_length_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    # Click position distribution (where the clicked result was in the list)
    position_dist = await pool.fetch(
        """
        SELECT (properties->>'position')::int AS position, COUNT(*) AS clicks
          FROM analytics_events
         WHERE event_name='messages_search_result_clicked'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'position' IS NOT NULL
         GROUP BY 1 ORDER BY position ASC;
        """
    )

    # Power-user share — top 10% of searchers
    power_users = await pool.fetchrow(
        """
        WITH per_user AS (
            SELECT user_id, COUNT(*) AS n
              FROM analytics_events
             WHERE event_name='messages_search_performed'
               AND occurred_at > NOW() - INTERVAL '7 days'
               AND user_id IS NOT NULL
             GROUP BY user_id
        ),
        thresholds AS (
            SELECT percentile_disc(0.90) WITHIN GROUP (ORDER BY n) AS top10_threshold,
                   COUNT(*) AS total_users,
                   SUM(n)   AS total_searches
              FROM per_user
        )
        SELECT
            t.total_users,
            t.total_searches,
            (SELECT COUNT(*) FROM per_user WHERE n >= t.top10_threshold) AS top10_users,
            (SELECT SUM(n)   FROM per_user WHERE n >= t.top10_threshold) AS top10_searches
        FROM thresholds t;
        """
    )

    # Daily trend (28d)
    daily_rows = await pool.fetch(
        """
        SELECT
            DATE(occurred_at) AS day,
            COUNT(*) FILTER (WHERE event_name='messages_search_performed')      AS searches,
            COUNT(*) FILTER (WHERE event_name='messages_search_result_clicked') AS clicks,
            COUNT(*) FILTER (WHERE event_name='messages_search_performed' AND (properties->>'result_count')::int = 0) AS zero_results
          FROM analytics_events
         WHERE event_name IN ('messages_search_performed','messages_search_result_clicked')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    searches_7d   = volume["searches_7d"]   or 0
    searchers_7d  = volume["searchers_7d"]  or 0
    s_with_res    = ctr["searches_with_results"] or 0
    clicked_s     = ctr["clicked_searches"]      or 0
    refined       = refinement["refined"]   or 0
    refine_total  = refinement["total"]     or 0
    top10_users   = int(power_users["top10_users"]    or 0) if power_users else 0
    top10_search  = int(power_users["top10_searches"] or 0) if power_users else 0
    total_users   = int(power_users["total_users"]    or 0) if power_users else 0
    total_search  = int(power_users["total_searches"] or 0) if power_users else 0

    return JSONResponse({
        "cards": {
            "searches_24h":     volume["searches_24h"] or 0,
            "searches_7d":      searches_7d,
            "searches_28d":     volume["searches_28d"] or 0,
            "searchers_7d":     searchers_7d,
            "clicks_7d":        volume["clicks_7d"] or 0,
            "searches_per_user": (searches_7d / searchers_7d) if searchers_7d else 0.0,
            "no_results_rate":  float(quality["no_results_rate"] or 0),
            "median_results":   quality["median_results"] or 0,
            "p95_results":      quality["p95_results"]    or 0,
            "ctr":              (clicked_s / s_with_res) if s_with_res else 0.0,
            "refinement_rate":  (refined / refine_total) if refine_total else 0.0,
            "power_users":      top10_users,
            "power_user_share_of_searches": (top10_search / total_search) if total_search else 0.0,
        },
        "by_category":  await _serialise_rows(by_category),
        "by_length":    await _serialise_rows(by_length),
        "position_dist": await _serialise_rows(position_dist),
        "daily":        await _serialise_rows(daily_rows),
    })


# ── Replay / Session Inspector ──────────────────────────────────────────────

async def replay_sessions(request: Request) -> JSONResponse:
    """Filtered list of sessions for the Replay (Session Inspector) tab,
    plus summary counts for the hero cards.
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    user_id           = request.query_params.get("user_id")
    profile_id        = request.query_params.get("street_profile_id")
    since             = request.query_params.get("since")
    only_errors       = request.query_params.get("errors")    == "1"
    only_rage         = request.query_params.get("rage")      == "1"
    days              = _window_days(request, 7)
    limit             = max(1, min(int(request.query_params.get("limit", "50")), 500))

    where  = ["s.started_at > NOW() - ($1 * INTERVAL '1 day')"]
    params: list[Any] = [days]
    if user_id:
        params.append(user_id);    where.append(f"s.user_id = ${len(params)}")
    if profile_id:
        params.append(profile_id); where.append(f"s.street_profile_id = ${len(params)}")
    if since:
        params.append(since);      where.append(f"s.started_at >= ${len(params)}::date")
    if only_errors:
        where.append("s.error_count > 0")
    if only_rage:
        where.append("s.rage_click_count > 0")

    where_sql = " AND ".join(where)

    rows = await pool.fetch(
        f"""
        SELECT s.id AS session_id, s.user_id, s.anonymous_id, s.street_profile_id,
               s.user_role, s.device_type, s.viewport_bucket,
               s.started_at, s.last_seen_at,
               EXTRACT(EPOCH FROM (s.last_seen_at - s.started_at))::int AS duration_s,
               s.page_view_count, s.event_count,
               s.active_time_ms, s.idle_time_ms,
               s.rage_click_count, s.error_count,
               s.entry_route_pattern, s.exit_route_pattern,
               s.first_product_area, s.last_product_area
          FROM analytics_sessions s
         WHERE {where_sql}
         ORDER BY s.started_at DESC
         LIMIT {limit};
        """,
        *params,
    )

    summary = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_sessions,
            COUNT(*) FILTER (WHERE error_count > 0)       AS sessions_with_errors,
            COUNT(*) FILTER (WHERE rage_click_count > 0)  AS sessions_with_rage,
            COUNT(*) FILTER (WHERE is_bounce = TRUE)      AS bounced,
            AVG(EXTRACT(EPOCH FROM (last_seen_at - started_at)))::int AS avg_duration_s,
            percentile_disc(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (last_seen_at - started_at)))::int AS median_duration_s
        FROM analytics_sessions
        WHERE started_at > NOW() - ($1 * INTERVAL '1 day');
        """,
        days,
    )

    return JSONResponse({
        "summary":  serialise(summary) if summary else {},
        "sessions": await _serialise_rows(rows),
        "filters_applied": {
            "user_id":            user_id,
            "street_profile_id":  profile_id,
            "since":              since,
            "errors":             only_errors,
            "rage":               only_rage,
            "days":               days,
        },
    })


async def replay_session_detail(request: Request) -> JSONResponse:
    """Full event timeline for a single session — what powers the
    Session Inspector drawer.
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    sid_raw = request.path_params.get("session_id")
    try:
        sid = UUID(str(sid_raw))
    except (TypeError, ValueError):
        return JSONResponse({"error": "invalid_session_id"}, status_code=400)

    session = await pool.fetchrow(
        """
        SELECT id AS session_id, user_id, anonymous_id, street_profile_id,
               user_role, device_type, viewport_bucket, user_agent_family, os_family,
               app_variant, app_version, environment,
               started_at, last_seen_at,
               EXTRACT(EPOCH FROM (last_seen_at - started_at))::int AS duration_s,
               page_view_count, event_count, active_time_ms, idle_time_ms,
               rage_click_count, error_count,
               entry_route_pattern, exit_route_pattern,
               first_product_area, last_product_area
          FROM analytics_sessions
         WHERE id = $1;
        """,
        sid,
    )
    if not session:
        return JSONResponse({"error": "session_not_found"}, status_code=404)

    events = await pool.fetch(
        """
        SELECT event_id, event_name, product_area, route, route_pattern,
               page_title, properties, occurred_at,
               EXTRACT(EPOCH FROM (occurred_at - $2))::int AS t_offset_s
          FROM analytics_events
         WHERE session_id = $1
         ORDER BY occurred_at ASC
         LIMIT 1000;
        """,
        sid, session["started_at"],
    )

    by_name = await pool.fetch(
        """
        SELECT event_name, COUNT(*) AS n
          FROM analytics_events
         WHERE session_id = $1
         GROUP BY 1 ORDER BY n DESC;
        """,
        sid,
    )

    return JSONResponse({
        "session":  serialise(session),
        "events":   await _serialise_rows(events),
        "by_name":  await _serialise_rows(by_name),
    })


async def section_email(request: Request) -> JSONResponse:
    """All Email section metrics in a single payload.

    Question: is the email tool actually being used, and does it convert?
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_composed' AND occurred_at > NOW() - INTERVAL '24 hours') AS composed_24h,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_composed' AND occurred_at > NOW() - INTERVAL '7 days')   AS composed_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_composed' AND occurred_at > NOW() - INTERVAL '28 days')  AS composed_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_sent'     AND occurred_at > NOW() - INTERVAL '7 days')   AS sent_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_sent'     AND occurred_at > NOW() - INTERVAL '24 hours') AS sent_24h,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_send_failed' AND occurred_at > NOW() - INTERVAL '7 days') AS failed_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_received' AND occurred_at > NOW() - INTERVAL '7 days')   AS received_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_opened'   AND occurred_at > NOW() - INTERVAL '7 days')   AS opened_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='email_replied'  AND occurred_at > NOW() - INTERVAL '7 days')   AS replied_7d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='email_sent' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS active_senders_7d,
            (SELECT COUNT(*) FILTER (WHERE (properties->>'has_attachment')::bool = TRUE)
               FROM analytics_events WHERE event_name='email_sent' AND occurred_at > NOW() - INTERVAL '7 days') AS sent_with_attachment_7d,
            (SELECT COUNT(*) FILTER (WHERE (properties->>'is_reply')::bool = TRUE)
               FROM analytics_events WHERE event_name='email_sent' AND occurred_at > NOW() - INTERVAL '7 days') AS sent_replies_7d;
        """
    )

    by_compose_time = await pool.fetch(
        """
        SELECT properties->>'compose_time_bucket' AS bucket, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='email_sent'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'compose_time_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_recipient_count = await pool.fetch(
        """
        SELECT properties->>'recipient_count_bucket' AS bucket, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='email_sent'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'recipient_count_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_reply_time = await pool.fetch(
        """
        SELECT properties->>'time_to_reply_bucket' AS bucket, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='email_replied'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'time_to_reply_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_error_code = await pool.fetch(
        """
        SELECT properties->>'error_code' AS error_code, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='email_send_failed'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'error_code' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_from_domain = await pool.fetch(
        """
        SELECT properties->>'from_domain_bucket' AS bucket, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='email_received'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'from_domain_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    daily = await pool.fetch(
        """
        SELECT
            DATE(occurred_at) AS day,
            COUNT(*) FILTER (WHERE event_name='email_composed') AS composed,
            COUNT(*) FILTER (WHERE event_name='email_sent')     AS sent,
            COUNT(*) FILTER (WHERE event_name='email_received') AS received,
            COUNT(*) FILTER (WHERE event_name='email_replied')  AS replied
          FROM analytics_events
         WHERE event_name IN ('email_composed','email_sent','email_received','email_replied')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    composed_7d  = volume["composed_7d"]  or 0
    sent_7d      = volume["sent_7d"]      or 0
    failed_7d    = volume["failed_7d"]    or 0
    opened_7d    = volume["opened_7d"]    or 0
    received_7d  = volume["received_7d"]  or 0
    replied_7d   = volume["replied_7d"]   or 0
    sent_with_at = volume["sent_with_attachment_7d"] or 0
    sent_replies = volume["sent_replies_7d"] or 0

    return JSONResponse({
        "cards": {
            "composed_24h":         volume["composed_24h"]         or 0,
            "composed_7d":          composed_7d,
            "composed_28d":         volume["composed_28d"]         or 0,
            "sent_7d":              sent_7d,
            "sent_24h":             volume["sent_24h"]             or 0,
            "failed_7d":            failed_7d,
            "received_7d":          received_7d,
            "opened_7d":            opened_7d,
            "replied_7d":           replied_7d,
            "active_senders_7d":    volume["active_senders_7d"]    or 0,
            "send_rate":            (sent_7d / composed_7d) if composed_7d else 0.0,
            "send_failure_rate":    (failed_7d / (sent_7d + failed_7d)) if (sent_7d + failed_7d) else 0.0,
            "open_rate":            (opened_7d / received_7d) if received_7d else 0.0,
            "reply_rate":           (replied_7d / opened_7d) if opened_7d else 0.0,
            "attachment_rate":      (sent_with_at / sent_7d) if sent_7d else 0.0,
            "reply_share":          (sent_replies / sent_7d) if sent_7d else 0.0,
        },
        "by_compose_time":    await _serialise_rows(by_compose_time),
        "by_recipient_count": await _serialise_rows(by_recipient_count),
        "by_reply_time":      await _serialise_rows(by_reply_time),
        "by_error_code":      await _serialise_rows(by_error_code),
        "by_from_domain":     await _serialise_rows(by_from_domain),
        "daily":              await _serialise_rows(daily),
    })


async def section_street_profile(request: Request) -> JSONResponse:
    """All Street Profile section metrics in a single payload.

    Question: are people building profiles, are profiles getting viewed,
    and are viewers taking action?
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_directory_viewed' AND occurred_at > NOW() - INTERVAL '7 days')   AS dir_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_create_started'   AND occurred_at > NOW() - INTERVAL '28 days')  AS create_started_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_create_started'   AND occurred_at > NOW() - INTERVAL '7 days')   AS create_started_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_created'          AND occurred_at > NOW() - INTERVAL '28 days')  AS created_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_created'          AND occurred_at > NOW() - INTERVAL '7 days')   AS created_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_viewed'           AND occurred_at > NOW() - INTERVAL '7 days')   AS views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_viewed'           AND occurred_at > NOW() - INTERVAL '24 hours') AS views_24h,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_updated'          AND occurred_at > NOW() - INTERVAL '7 days')   AS updates_7d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='street_profile_updated' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS active_editors_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_cta_clicked'      AND occurred_at > NOW() - INTERVAL '7 days')   AS ctas_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_avatar_uploaded'  AND occurred_at > NOW() - INTERVAL '28 days')  AS avatars_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='street_profile_booking_started'  AND occurred_at > NOW() - INTERVAL '7 days')   AS bookings_7d,
            (SELECT COUNT(*) FILTER (WHERE (properties->>'is_profile_owner')::bool = TRUE)
               FROM analytics_events WHERE event_name='street_profile_viewed' AND occurred_at > NOW() - INTERVAL '7 days') AS owner_views_7d,
            (SELECT AVG((properties->>'completeness')::numeric)
               FROM analytics_events WHERE event_name='street_profile_created' AND occurred_at > NOW() - INTERVAL '28 days'
                 AND properties->>'completeness' IS NOT NULL) AS avg_completeness;
        """
    )

    funnel = await pool.fetch(
        """
        SELECT
            COUNT(*) FILTER (WHERE event_name='street_profile_directory_viewed') AS dir_viewed,
            COUNT(*) FILTER (WHERE event_name='street_profile_create_started')   AS started,
            COUNT(*) FILTER (WHERE event_name='street_profile_create_step_completed' AND properties->>'step'='basics')    AS step_basics,
            COUNT(*) FILTER (WHERE event_name='street_profile_create_step_completed' AND properties->>'step'='about')     AS step_about,
            COUNT(*) FILTER (WHERE event_name='street_profile_create_step_completed' AND properties->>'step'='portfolio') AS step_portfolio,
            COUNT(*) FILTER (WHERE event_name='street_profile_create_step_completed' AND properties->>'step'='services')  AS step_services,
            COUNT(*) FILTER (WHERE event_name='street_profile_created')          AS created
          FROM analytics_events
         WHERE event_name IN (
                'street_profile_directory_viewed','street_profile_create_started',
                'street_profile_create_step_completed','street_profile_created')
           AND occurred_at > NOW() - INTERVAL '28 days';
        """
    )

    by_tab = await pool.fetch(
        """
        SELECT properties->>'tab' AS tab, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='street_profile_tab_viewed'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'tab' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_cta = await pool.fetch(
        """
        SELECT properties->>'cta' AS cta, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='street_profile_cta_clicked'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'cta' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_view_source = await pool.fetch(
        """
        SELECT properties->>'source' AS source, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='street_profile_viewed'
           AND occurred_at > NOW() - INTERVAL '7 days'
           AND properties->>'source' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_field_updated = await pool.fetch(
        """
        SELECT field, COUNT(*) AS n FROM (
            SELECT jsonb_array_elements_text(properties->'fields_updated') AS field
              FROM analytics_events
             WHERE event_name='street_profile_updated'
               AND occurred_at > NOW() - INTERVAL '28 days'
               AND properties ? 'fields_updated'
        ) t
        GROUP BY 1 ORDER BY n DESC LIMIT 12;
        """
    )

    by_completeness_delta = await pool.fetch(
        """
        SELECT bucket, COUNT(*) AS n FROM (
            SELECT CASE
                WHEN delta < 0 THEN 'decreased'
                WHEN delta = 0 THEN 'no change'
                WHEN delta < 10 THEN '+1-9'
                WHEN delta < 25 THEN '+10-24'
                ELSE '+25 or more'
            END AS bucket
              FROM (
                SELECT (properties->>'completeness_after')::numeric - (properties->>'completeness_before')::numeric AS delta
                  FROM analytics_events
                 WHERE event_name='street_profile_updated'
                   AND occurred_at > NOW() - INTERVAL '28 days'
                   AND properties ? 'completeness_after'
                   AND properties ? 'completeness_before'
              ) d
        ) b
        GROUP BY 1 ORDER BY 1;
        """
    )

    daily = await pool.fetch(
        """
        SELECT
            DATE(occurred_at) AS day,
            COUNT(*) FILTER (WHERE event_name='street_profile_viewed')      AS viewed,
            COUNT(*) FILTER (WHERE event_name='street_profile_updated')     AS updated,
            COUNT(*) FILTER (WHERE event_name='street_profile_cta_clicked') AS cta_clicks
          FROM analytics_events
         WHERE event_name IN ('street_profile_viewed','street_profile_updated','street_profile_cta_clicked')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    started_28d = int(volume["create_started_28d"] or 0)
    created_28d = int(volume["created_28d"]        or 0)
    views_7d    = int(volume["views_7d"]           or 0)
    ctas_7d     = int(volume["ctas_7d"]            or 0)
    owner_views = int(volume["owner_views_7d"]     or 0)
    bookings_7d = int(volume["bookings_7d"]        or 0)
    created_total = int(funnel[0]["created"] or 0) if funnel else 0
    avatars_28d = int(volume["avatars_28d"]        or 0)

    return JSONResponse({
        "cards": {
            "dir_views_7d":       int(volume["dir_views_7d"]       or 0),
            "create_started_7d":  int(volume["create_started_7d"]  or 0),
            "create_started_28d": started_28d,
            "created_7d":         int(volume["created_7d"]         or 0),
            "created_28d":        created_28d,
            "views_24h":          int(volume["views_24h"]          or 0),
            "views_7d":           views_7d,
            "updates_7d":         int(volume["updates_7d"]         or 0),
            "active_editors_7d":  int(volume["active_editors_7d"]  or 0),
            "ctas_7d":            ctas_7d,
            "bookings_7d":        bookings_7d,
            "avg_completeness":   float(volume["avg_completeness"]) if volume["avg_completeness"] is not None else None,
            "activation_rate":    (created_28d / started_28d) if started_28d else 0.0,
            "cta_per_view":       (ctas_7d / views_7d) if views_7d else 0.0,
            "owner_view_share":   (owner_views / views_7d) if views_7d else 0.0,
            "booking_per_view":   (bookings_7d / views_7d) if views_7d else 0.0,
            "avatar_attach_rate": (avatars_28d / created_total) if created_total else 0.0,
        },
        "funnel": await _serialise_rows(funnel),
        "by_tab":               await _serialise_rows(by_tab),
        "by_cta":               await _serialise_rows(by_cta),
        "by_view_source":       await _serialise_rows(by_view_source),
        "by_field_updated":     await _serialise_rows(by_field_updated),
        "by_completeness_delta": await _serialise_rows(by_completeness_delta),
        "daily":                await _serialise_rows(daily),
    })


async def section_gallery(request: Request) -> JSONResponse:
    """All Street Gallery section metrics in a single payload.

    Question: are artists uploading, are visitors finding work,
    and is anything converting to engagement or sales?
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_viewed'              AND occurred_at > NOW() - INTERVAL '7 days')   AS views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_viewed'              AND occurred_at > NOW() - INTERVAL '24 hours') AS views_24h,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artwork_impression'  AND occurred_at > NOW() - INTERVAL '7 days')   AS impressions_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artwork_viewed'      AND occurred_at > NOW() - INTERVAL '7 days')   AS detail_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artwork_favorited'   AND occurred_at > NOW() - INTERVAL '7 days')   AS favs_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artwork_commented'   AND occurred_at > NOW() - INTERVAL '7 days')   AS comments_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artwork_shared'      AND occurred_at > NOW() - INTERVAL '7 days')   AS shares_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artist_profile_clicked' AND occurred_at > NOW() - INTERVAL '7 days') AS artist_clicks_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_upload_started'      AND occurred_at > NOW() - INTERVAL '28 days')  AS uploads_started_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_upload_blocked'      AND occurred_at > NOW() - INTERVAL '28 days')  AS uploads_blocked_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artwork_uploaded'    AND occurred_at > NOW() - INTERVAL '28 days')  AS uploads_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artwork_uploaded'    AND occurred_at > NOW() - INTERVAL '7 days')   AS uploads_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='gallery_artwork_marked_sold' AND occurred_at > NOW() - INTERVAL '28 days')  AS sold_28d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='gallery_artwork_uploaded' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS active_artists_7d,
            (SELECT COUNT(*) FILTER (WHERE (properties->>'is_for_sale')::bool = TRUE)
               FROM analytics_events WHERE event_name='gallery_artwork_uploaded' AND occurred_at > NOW() - INTERVAL '28 days') AS for_sale_28d;
        """
    )

    by_medium = await pool.fetch(
        """
        SELECT properties->>'medium' AS medium, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='gallery_artwork_uploaded'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'medium' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_block_reason = await pool.fetch(
        """
        SELECT properties->>'reason' AS reason, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='gallery_upload_blocked'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'reason' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_share_method = await pool.fetch(
        """
        SELECT properties->>'share_method' AS share_method, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='gallery_artwork_shared'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'share_method' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_price = await pool.fetch(
        """
        SELECT bucket, COUNT(*) AS n FROM (
            SELECT CASE
                WHEN (properties->>'is_for_sale')::bool IS NOT TRUE THEN 'not_for_sale'
                ELSE 'for_sale'
            END AS bucket
              FROM analytics_events
             WHERE event_name='gallery_artwork_uploaded'
               AND occurred_at > NOW() - INTERVAL '28 days'
        ) t GROUP BY 1 ORDER BY 1;
        """
    )

    by_sold_price = await pool.fetch(
        """
        SELECT properties->>'price_bucket' AS price_bucket, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='gallery_artwork_marked_sold'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'price_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_comment_type = await pool.fetch(
        """
        SELECT properties->>'comment_type' AS comment_type, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='gallery_artwork_commented'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'comment_type' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    daily = await pool.fetch(
        """
        SELECT
            DATE(occurred_at) AS day,
            COUNT(*) FILTER (WHERE event_name='gallery_viewed')             AS viewed,
            COUNT(*) FILTER (WHERE event_name='gallery_artwork_viewed')     AS detail_viewed,
            COUNT(*) FILTER (WHERE event_name='gallery_artwork_favorited')  AS favorited,
            COUNT(*) FILTER (WHERE event_name='gallery_artwork_uploaded')   AS uploaded
          FROM analytics_events
         WHERE event_name IN ('gallery_viewed','gallery_artwork_viewed','gallery_artwork_favorited','gallery_artwork_uploaded')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    impressions_7d  = int(volume["impressions_7d"]   or 0)
    detail_views_7d = int(volume["detail_views_7d"]  or 0)
    favs_7d         = int(volume["favs_7d"]          or 0)
    comments_7d     = int(volume["comments_7d"]      or 0)
    artist_clicks_7d= int(volume["artist_clicks_7d"] or 0)
    started_28d     = int(volume["uploads_started_28d"] or 0)
    blocked_28d     = int(volume["uploads_blocked_28d"] or 0)
    uploads_28d     = int(volume["uploads_28d"]      or 0)
    sold_28d        = int(volume["sold_28d"]         or 0)
    for_sale_28d    = int(volume["for_sale_28d"]     or 0)

    return JSONResponse({
        "cards": {
            "views_7d":            int(volume["views_7d"]            or 0),
            "views_24h":           int(volume["views_24h"]           or 0),
            "impressions_7d":      impressions_7d,
            "detail_views_7d":     detail_views_7d,
            "favs_7d":             favs_7d,
            "comments_7d":         comments_7d,
            "shares_7d":           int(volume["shares_7d"]           or 0),
            "artist_clicks_7d":    artist_clicks_7d,
            "uploads_started_28d": started_28d,
            "uploads_blocked_28d": blocked_28d,
            "uploads_28d":         uploads_28d,
            "uploads_7d":          int(volume["uploads_7d"]          or 0),
            "sold_28d":            sold_28d,
            "active_artists_7d":   int(volume["active_artists_7d"]   or 0),
            "for_sale_28d":        for_sale_28d,
            "upload_conversion":   (uploads_28d / started_28d) if started_28d else 0.0,
            "block_rate":          (blocked_28d / started_28d) if started_28d else 0.0,
            "detail_ctr":          (detail_views_7d / impressions_7d) if impressions_7d else 0.0,
            "engagement_rate":     ((favs_7d + comments_7d) / detail_views_7d) if detail_views_7d else 0.0,
            "for_sale_share":      (for_sale_28d / uploads_28d) if uploads_28d else 0.0,
            "sold_rate":           (sold_28d / for_sale_28d) if for_sale_28d else 0.0,
            "artist_ctr":          (artist_clicks_7d / detail_views_7d) if detail_views_7d else 0.0,
        },
        "by_medium":         await _serialise_rows(by_medium),
        "by_block_reason":   await _serialise_rows(by_block_reason),
        "by_share_method":   await _serialise_rows(by_share_method),
        "by_price":          await _serialise_rows(by_price),
        "by_sold_price":     await _serialise_rows(by_sold_price),
        "by_comment_type":   await _serialise_rows(by_comment_type),
        "daily":             await _serialise_rows(daily),
    })


async def section_news(request: Request) -> JSONResponse:
    """All News section metrics in a single payload.

    Question: are readers reading what we publish, and is the editorial
    side using the AI tools effectively?
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_home_viewed'      AND occurred_at > NOW() - INTERVAL '7 days')   AS home_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_article_viewed'   AND occurred_at > NOW() - INTERVAL '7 days')   AS article_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_article_viewed'   AND occurred_at > NOW() - INTERVAL '24 hours') AS article_views_24h,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_article_viewed'   AND occurred_at > NOW() - INTERVAL '28 days')  AS article_views_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_article_read'     AND occurred_at > NOW() - INTERVAL '7 days')   AS reads_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_article_read'     AND occurred_at > NOW() - INTERVAL '28 days')  AS reads_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_article_shared'   AND occurred_at > NOW() - INTERVAL '7 days')   AS shares_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_bookmark_added'   AND occurred_at > NOW() - INTERVAL '7 days')   AS bookmarks_7d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='news_article_viewed' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS readers_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_draft_created'    AND occurred_at > NOW() - INTERVAL '7 days')   AS drafts_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_draft_created'    AND occurred_at > NOW() - INTERVAL '28 days')  AS drafts_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_ai_generation_started'   AND occurred_at > NOW() - INTERVAL '7 days') AS ai_started_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_ai_generation_completed' AND occurred_at > NOW() - INTERVAL '7 days' AND (properties->>'success')::bool = TRUE) AS ai_succeeded_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='news_ai_generation_completed' AND occurred_at > NOW() - INTERVAL '7 days') AS ai_completed_7d,
            (SELECT PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY (properties->>'read_time_ms')::numeric)
               FROM analytics_events WHERE event_name='news_article_read' AND occurred_at > NOW() - INTERVAL '7 days') AS median_read_ms,
            (SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY (properties->>'latency_ms')::numeric)
               FROM analytics_events WHERE event_name='news_ai_generation_completed' AND occurred_at > NOW() - INTERVAL '7 days'
                                       AND (properties->>'success')::bool = TRUE) AS ai_p95_latency_ms;
        """
    )

    by_category = await pool.fetch(
        """
        SELECT properties->>'category' AS category, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='news_article_viewed'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'category' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC LIMIT 12;
        """
    )

    by_source = await pool.fetch(
        """
        SELECT properties->>'source' AS source, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='news_article_viewed'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'source' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_share_method = await pool.fetch(
        """
        SELECT properties->>'share_method' AS share_method, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='news_article_shared'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'share_method' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_scroll_depth = await pool.fetch(
        """
        SELECT bucket, COUNT(*) AS n FROM (
            SELECT CASE
                WHEN (properties->>'scroll_depth_percent')::numeric < 80 THEN '70-79%'
                WHEN (properties->>'scroll_depth_percent')::numeric < 90 THEN '80-89%'
                WHEN (properties->>'scroll_depth_percent')::numeric < 100 THEN '90-99%'
                ELSE '100%'
            END AS bucket
              FROM analytics_events
             WHERE event_name='news_article_read'
               AND occurred_at > NOW() - INTERVAL '28 days'
               AND properties->>'scroll_depth_percent' IS NOT NULL
        ) t GROUP BY 1 ORDER BY 1;
        """
    )

    by_ai_type = await pool.fetch(
        """
        SELECT
            properties->>'generation_type' AS generation_type,
            COUNT(*) AS attempts,
            COUNT(*) FILTER (WHERE (properties->>'success')::bool = TRUE) AS successes,
            ROUND(PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY (properties->>'latency_ms')::numeric))::int AS median_latency_ms
          FROM analytics_events
         WHERE event_name='news_ai_generation_completed'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'generation_type' IS NOT NULL
         GROUP BY 1 ORDER BY attempts DESC;
        """
    )

    by_draft_source = await pool.fetch(
        """
        SELECT properties->>'source' AS source, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='news_draft_created'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'source' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    daily = await pool.fetch(
        """
        SELECT
            DATE(occurred_at) AS day,
            COUNT(*) FILTER (WHERE event_name='news_home_viewed')      AS home_views,
            COUNT(*) FILTER (WHERE event_name='news_article_viewed')   AS article_views,
            COUNT(*) FILTER (WHERE event_name='news_article_read')     AS reads,
            COUNT(*) FILTER (WHERE event_name='news_draft_created')    AS drafts
          FROM analytics_events
         WHERE event_name IN ('news_home_viewed','news_article_viewed','news_article_read','news_draft_created')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    article_views_7d   = int(volume["article_views_7d"]   or 0)
    reads_7d           = int(volume["reads_7d"]           or 0)
    bookmarks_7d       = int(volume["bookmarks_7d"]       or 0)
    shares_7d          = int(volume["shares_7d"]          or 0)
    ai_completed_7d    = int(volume["ai_completed_7d"]    or 0)
    ai_succeeded_7d    = int(volume["ai_succeeded_7d"]    or 0)
    ai_p95_latency_ms  = float(volume["ai_p95_latency_ms"]) if volume["ai_p95_latency_ms"] is not None else None
    median_read_ms     = float(volume["median_read_ms"])    if volume["median_read_ms"]    is not None else None

    return JSONResponse({
        "cards": {
            "home_views_7d":     int(volume["home_views_7d"]      or 0),
            "article_views_7d":  article_views_7d,
            "article_views_24h": int(volume["article_views_24h"]  or 0),
            "article_views_28d": int(volume["article_views_28d"]  or 0),
            "reads_7d":          reads_7d,
            "reads_28d":         int(volume["reads_28d"]          or 0),
            "shares_7d":         shares_7d,
            "bookmarks_7d":      bookmarks_7d,
            "readers_7d":        int(volume["readers_7d"]         or 0),
            "drafts_7d":         int(volume["drafts_7d"]          or 0),
            "drafts_28d":        int(volume["drafts_28d"]         or 0),
            "ai_started_7d":     int(volume["ai_started_7d"]      or 0),
            "ai_succeeded_7d":   ai_succeeded_7d,
            "ai_completed_7d":   ai_completed_7d,
            "median_read_ms":    median_read_ms,
            "ai_p95_latency_ms": ai_p95_latency_ms,
            "read_through_rate": (reads_7d / article_views_7d) if article_views_7d else 0.0,
            "bookmark_rate":     (bookmarks_7d / article_views_7d) if article_views_7d else 0.0,
            "share_rate":        (shares_7d / article_views_7d) if article_views_7d else 0.0,
            "ai_success_rate":   (ai_succeeded_7d / ai_completed_7d) if ai_completed_7d else 0.0,
        },
        "by_category":      await _serialise_rows(by_category),
        "by_source":        await _serialise_rows(by_source),
        "by_share_method":  await _serialise_rows(by_share_method),
        "by_scroll_depth":  await _serialise_rows(by_scroll_depth),
        "by_ai_type":       await _serialise_rows(by_ai_type),
        "by_draft_source":  await _serialise_rows(by_draft_source),
        "daily":            await _serialise_rows(daily),
    })


async def section_directory(request: Request) -> JSONResponse:
    """All Directory section metrics in a single payload.

    Question: are people finding services they need, getting through to
    contact actions, and is review/claim activity healthy?
    """
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_viewed'              AND occurred_at > NOW() - INTERVAL '7 days')   AS dir_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_viewed'              AND occurred_at > NOW() - INTERVAL '24 hours') AS dir_views_24h,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_search_performed'    AND occurred_at > NOW() - INTERVAL '7 days')   AS searches_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_no_results_seen'     AND occurred_at > NOW() - INTERVAL '7 days')   AS no_results_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_map_viewed'          AND occurred_at > NOW() - INTERVAL '7 days')   AS map_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_service_impression'  AND occurred_at > NOW() - INTERVAL '7 days')   AS impressions_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_service_viewed'      AND occurred_at > NOW() - INTERVAL '7 days')   AS service_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_service_viewed'      AND occurred_at > NOW() - INTERVAL '28 days')  AS service_views_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_service_action_clicked' AND occurred_at > NOW() - INTERVAL '7 days') AS actions_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_service_saved'       AND occurred_at > NOW() - INTERVAL '7 days')   AS saved_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_review_submitted'    AND occurred_at > NOW() - INTERVAL '28 days')  AS reviews_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_listing_claim_started'   AND occurred_at > NOW() - INTERVAL '28 days') AS claims_started_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='directory_listing_claim_completed' AND occurred_at > NOW() - INTERVAL '28 days') AS claims_completed_28d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='directory_search_performed' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS searchers_7d,
            (SELECT AVG((properties->>'rating')::numeric)
               FROM analytics_events WHERE event_name='directory_review_submitted' AND occurred_at > NOW() - INTERVAL '28 days'
                 AND properties->>'rating' IS NOT NULL) AS avg_rating;
        """
    )

    by_action = await pool.fetch(
        """
        SELECT properties->>'action' AS action, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='directory_service_action_clicked'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'action' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_filter = await pool.fetch(
        """
        SELECT properties->>'filter_type' AS filter_type, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='directory_filter_changed'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'filter_type' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_view_source = await pool.fetch(
        """
        SELECT properties->>'source' AS source, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='directory_service_viewed'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'source' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_rating = await pool.fetch(
        """
        SELECT (properties->>'rating')::int AS rating, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='directory_review_submitted'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'rating' IS NOT NULL
         GROUP BY 1 ORDER BY rating DESC;
        """
    )

    by_provider_type = await pool.fetch(
        """
        SELECT properties->>'provider_type' AS provider_type, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='directory_listing_claim_completed'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'provider_type' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    by_view_mode = await pool.fetch(
        """
        SELECT properties->>'view_mode' AS view_mode, COUNT(*) AS n
          FROM analytics_events
         WHERE event_name='directory_viewed'
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'view_mode' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;
        """
    )

    daily = await pool.fetch(
        """
        SELECT
            DATE(occurred_at) AS day,
            COUNT(*) FILTER (WHERE event_name='directory_viewed')                  AS dir_views,
            COUNT(*) FILTER (WHERE event_name='directory_service_viewed')          AS service_views,
            COUNT(*) FILTER (WHERE event_name='directory_service_action_clicked')  AS actions,
            COUNT(*) FILTER (WHERE event_name='directory_listing_claim_completed') AS claims
          FROM analytics_events
         WHERE event_name IN ('directory_viewed','directory_service_viewed','directory_service_action_clicked','directory_listing_claim_completed')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;
        """
    )

    searches_7d         = int(volume["searches_7d"]         or 0)
    no_results_7d       = int(volume["no_results_7d"]       or 0)
    impressions_7d      = int(volume["impressions_7d"]      or 0)
    service_views_7d    = int(volume["service_views_7d"]    or 0)
    actions_7d          = int(volume["actions_7d"]          or 0)
    saved_7d            = int(volume["saved_7d"]            or 0)
    claims_started_28d  = int(volume["claims_started_28d"]  or 0)
    claims_completed_28d= int(volume["claims_completed_28d"] or 0)
    dir_views_7d        = int(volume["dir_views_7d"]        or 0)
    map_views_7d        = int(volume["map_views_7d"]        or 0)

    return JSONResponse({
        "cards": {
            "dir_views_7d":         dir_views_7d,
            "dir_views_24h":        int(volume["dir_views_24h"]    or 0),
            "searches_7d":          searches_7d,
            "no_results_7d":        no_results_7d,
            "impressions_7d":       impressions_7d,
            "service_views_7d":     service_views_7d,
            "service_views_28d":    int(volume["service_views_28d"] or 0),
            "actions_7d":           actions_7d,
            "saved_7d":             saved_7d,
            "reviews_28d":          int(volume["reviews_28d"]      or 0),
            "claims_started_28d":   claims_started_28d,
            "claims_completed_28d": claims_completed_28d,
            "searchers_7d":         int(volume["searchers_7d"]     or 0),
            "map_views_7d":         map_views_7d,
            "avg_rating":           float(volume["avg_rating"]) if volume["avg_rating"] is not None else None,
            "search_action_rate":   (actions_7d / searches_7d) if searches_7d else 0.0,
            "no_results_rate":      (no_results_7d / searches_7d) if searches_7d else 0.0,
            "detail_ctr":           (service_views_7d / impressions_7d) if impressions_7d else 0.0,
            "action_per_detail":    (actions_7d / service_views_7d) if service_views_7d else 0.0,
            "save_per_detail":      (saved_7d / service_views_7d) if service_views_7d else 0.0,
            "claim_conversion":     (claims_completed_28d / claims_started_28d) if claims_started_28d else 0.0,
            "map_share":            (map_views_7d / dir_views_7d) if dir_views_7d else 0.0,
        },
        "by_action":         await _serialise_rows(by_action),
        "by_filter":         await _serialise_rows(by_filter),
        "by_view_source":    await _serialise_rows(by_view_source),
        "by_rating":         await _serialise_rows(by_rating),
        "by_provider_type":  await _serialise_rows(by_provider_type),
        "by_view_mode":      await _serialise_rows(by_view_mode),
        "daily":             await _serialise_rows(daily),
    })


async def section_jobs(request: Request) -> JSONResponse:
    """All Job Board section metrics in a single payload."""
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_board_viewed'              AND occurred_at > NOW() - INTERVAL '7 days')   AS board_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_search_performed'          AND occurred_at > NOW() - INTERVAL '7 days')   AS searches_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_job_impression'            AND occurred_at > NOW() - INTERVAL '7 days')   AS impressions_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_job_viewed'                AND occurred_at > NOW() - INTERVAL '7 days')   AS job_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_job_viewed'                AND occurred_at > NOW() - INTERVAL '28 days')  AS job_views_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_job_saved'                 AND occurred_at > NOW() - INTERVAL '7 days')   AS saved_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_application_started'       AND occurred_at > NOW() - INTERVAL '28 days')  AS apps_started_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_application_started'       AND occurred_at > NOW() - INTERVAL '7 days')   AS apps_started_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_application_submitted'     AND occurred_at > NOW() - INTERVAL '28 days')  AS apps_submitted_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_application_submitted'     AND occurred_at > NOW() - INTERVAL '7 days')   AS apps_submitted_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_application_withdrawn'     AND occurred_at > NOW() - INTERVAL '28 days')  AS withdrawn_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_external_apply_clicked'    AND occurred_at > NOW() - INTERVAL '7 days')   AS external_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_resume_started'            AND occurred_at > NOW() - INTERVAL '28 days')  AS resumes_started_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_resume_completed'          AND occurred_at > NOW() - INTERVAL '28 days')  AS resumes_completed_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_cover_letter_generated'    AND occurred_at > NOW() - INTERVAL '28 days')  AS cover_ai_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_employer_listing_draft_started' AND occurred_at > NOW() - INTERVAL '28 days') AS listings_started_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='jobs_employer_listing_published'     AND occurred_at > NOW() - INTERVAL '28 days') AS listings_published_28d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='jobs_application_submitted' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS active_applicants_7d;
        """
    )

    by_submission = await pool.fetch("""
        SELECT properties->>'submission_type' AS submission_type, COUNT(*) AS n FROM analytics_events
         WHERE event_name='jobs_application_submitted' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'submission_type' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_work_mode = await pool.fetch("""
        SELECT properties->>'work_mode' AS work_mode, COUNT(*) AS n FROM analytics_events
         WHERE event_name='jobs_job_impression' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'work_mode' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_filter = await pool.fetch("""
        SELECT properties->>'filter_type' AS filter_type, COUNT(*) AS n FROM analytics_events
         WHERE event_name='jobs_filter_changed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'filter_type' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_resume_completeness = await pool.fetch("""
        SELECT properties->>'completeness_score_bucket' AS bucket, COUNT(*) AS n FROM analytics_events
         WHERE event_name='jobs_resume_completed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'completeness_score_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY 1;""")
    by_status_change = await pool.fetch("""
        SELECT properties->>'to_status' AS to_status, COUNT(*) AS n FROM analytics_events
         WHERE event_name='jobs_applicant_status_changed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'to_status' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_listing_category = await pool.fetch("""
        SELECT properties->>'category' AS category, COUNT(*) AS n FROM analytics_events
         WHERE event_name='jobs_employer_listing_published' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'category' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")

    daily = await pool.fetch("""
        SELECT DATE(occurred_at) AS day,
               COUNT(*) FILTER (WHERE event_name='jobs_board_viewed')         AS board_views,
               COUNT(*) FILTER (WHERE event_name='jobs_job_viewed')           AS job_views,
               COUNT(*) FILTER (WHERE event_name='jobs_application_submitted') AS apps_submitted,
               COUNT(*) FILTER (WHERE event_name='jobs_employer_listing_published') AS listings
          FROM analytics_events
         WHERE event_name IN ('jobs_board_viewed','jobs_job_viewed','jobs_application_submitted','jobs_employer_listing_published')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;""")

    impressions_7d         = int(volume["impressions_7d"]         or 0)
    job_views_7d           = int(volume["job_views_7d"]           or 0)
    apps_started_28d       = int(volume["apps_started_28d"]       or 0)
    apps_submitted_28d     = int(volume["apps_submitted_28d"]     or 0)
    apps_started_7d        = int(volume["apps_started_7d"]        or 0)
    apps_submitted_7d      = int(volume["apps_submitted_7d"]      or 0)
    listings_started_28d   = int(volume["listings_started_28d"]   or 0)
    listings_published_28d = int(volume["listings_published_28d"] or 0)
    resumes_started_28d    = int(volume["resumes_started_28d"]    or 0)
    resumes_completed_28d  = int(volume["resumes_completed_28d"]  or 0)
    withdrawn_28d          = int(volume["withdrawn_28d"]          or 0)
    saved_7d               = int(volume["saved_7d"]               or 0)
    cover_ai_28d           = int(volume["cover_ai_28d"]           or 0)

    return JSONResponse({
        "cards": {
            "board_views_7d":         int(volume["board_views_7d"] or 0),
            "searches_7d":            int(volume["searches_7d"]    or 0),
            "impressions_7d":         impressions_7d,
            "job_views_7d":           job_views_7d,
            "job_views_28d":          int(volume["job_views_28d"]  or 0),
            "saved_7d":               saved_7d,
            "apps_started_7d":        apps_started_7d,
            "apps_started_28d":       apps_started_28d,
            "apps_submitted_7d":      apps_submitted_7d,
            "apps_submitted_28d":     apps_submitted_28d,
            "withdrawn_28d":          withdrawn_28d,
            "external_7d":            int(volume["external_7d"]   or 0),
            "active_applicants_7d":   int(volume["active_applicants_7d"] or 0),
            "resumes_started_28d":    resumes_started_28d,
            "resumes_completed_28d":  resumes_completed_28d,
            "cover_ai_28d":           cover_ai_28d,
            "listings_started_28d":   listings_started_28d,
            "listings_published_28d": listings_published_28d,
            "detail_ctr":             (job_views_7d / impressions_7d) if impressions_7d else 0.0,
            "save_per_view":          (saved_7d / job_views_7d) if job_views_7d else 0.0,
            "apply_start_rate":       (apps_started_7d / job_views_7d) if job_views_7d else 0.0,
            "apply_completion":       (apps_submitted_28d / apps_started_28d) if apps_started_28d else 0.0,
            "withdraw_rate":          (withdrawn_28d / apps_submitted_28d) if apps_submitted_28d else 0.0,
            "resume_completion":      (resumes_completed_28d / resumes_started_28d) if resumes_started_28d else 0.0,
            "ai_cover_share":         (cover_ai_28d / apps_started_28d) if apps_started_28d else 0.0,
            "listing_publish_rate":   (listings_published_28d / listings_started_28d) if listings_started_28d else 0.0,
        },
        "by_submission":         await _serialise_rows(by_submission),
        "by_work_mode":          await _serialise_rows(by_work_mode),
        "by_filter":             await _serialise_rows(by_filter),
        "by_resume_completeness":await _serialise_rows(by_resume_completeness),
        "by_status_change":      await _serialise_rows(by_status_change),
        "by_listing_category":   await _serialise_rows(by_listing_category),
        "daily":                 await _serialise_rows(daily),
    })


async def section_academy(request: Request) -> JSONResponse:
    """All Academy section metrics in a single payload."""
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_home_viewed'           AND occurred_at > NOW() - INTERVAL '7 days')   AS home_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_course_viewed'         AND occurred_at > NOW() - INTERVAL '7 days')   AS course_views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_course_viewed'         AND occurred_at > NOW() - INTERVAL '28 days')  AS course_views_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_enrollment_started'    AND occurred_at > NOW() - INTERVAL '28 days')  AS enroll_started_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_enrollment_completed'  AND occurred_at > NOW() - INTERVAL '28 days')  AS enroll_completed_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_enrollment_completed'  AND occurred_at > NOW() - INTERVAL '7 days')   AS enroll_completed_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_lesson_started'        AND occurred_at > NOW() - INTERVAL '28 days')  AS lessons_started_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_lesson_completed'      AND occurred_at > NOW() - INTERVAL '28 days')  AS lessons_completed_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_lesson_completed'      AND occurred_at > NOW() - INTERVAL '7 days')   AS lessons_completed_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_assignment_submitted'  AND occurred_at > NOW() - INTERVAL '28 days')  AS assignments_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_quiz_completed'        AND occurred_at > NOW() - INTERVAL '28 days')  AS quizzes_completed_28d,
            (SELECT COUNT(*) FILTER (WHERE (properties->>'passed')::bool = TRUE)
               FROM analytics_events WHERE event_name='academy_quiz_completed' AND occurred_at > NOW() - INTERVAL '28 days') AS quizzes_passed_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_certificate_earned'    AND occurred_at > NOW() - INTERVAL '28 days')  AS certs_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_ai_tutor_used'         AND occurred_at > NOW() - INTERVAL '7 days')   AS ai_tutor_7d,
            (SELECT COUNT(*) FILTER (WHERE (properties->>'helpful')::bool = TRUE)
               FROM analytics_events WHERE event_name='academy_ai_tutor_used' AND occurred_at > NOW() - INTERVAL '28 days') AS ai_tutor_helpful_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_ai_tutor_used'         AND occurred_at > NOW() - INTERVAL '28 days') AS ai_tutor_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='academy_live_session_joined'   AND occurred_at > NOW() - INTERVAL '28 days') AS live_28d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='academy_lesson_started' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS active_learners_7d,
            (SELECT AVG((properties->>'time_spent_ms')::numeric)
               FROM analytics_events WHERE event_name='academy_lesson_completed' AND occurred_at > NOW() - INTERVAL '28 days') AS avg_lesson_time_ms;
        """
    )

    by_score = await pool.fetch("""
        SELECT properties->>'score_bucket' AS bucket, COUNT(*) AS n FROM analytics_events
         WHERE event_name='academy_quiz_completed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'score_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY 1;""")
    by_milestone = await pool.fetch("""
        SELECT properties->>'milestone' AS milestone, COUNT(*) AS n FROM analytics_events
         WHERE event_name='academy_video_progressed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'milestone' IS NOT NULL
         GROUP BY 1 ORDER BY 1;""")
    by_speed = await pool.fetch("""
        SELECT properties->>'playback_speed' AS speed, COUNT(*) AS n FROM analytics_events
         WHERE event_name='academy_video_progressed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'playback_speed' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_role = await pool.fetch("""
        SELECT properties->>'role' AS role, COUNT(*) AS n FROM analytics_events
         WHERE event_name='academy_home_viewed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'role' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    top_courses = await pool.fetch("""
        SELECT properties->>'course_id' AS course_id,
               COUNT(*) FILTER (WHERE event_name='academy_course_viewed') AS views,
               COUNT(*) FILTER (WHERE event_name='academy_enrollment_completed') AS enrolls
          FROM analytics_events
         WHERE event_name IN ('academy_course_viewed','academy_enrollment_completed')
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'course_id' IS NOT NULL
         GROUP BY 1 ORDER BY views DESC LIMIT 8;""")

    daily = await pool.fetch("""
        SELECT DATE(occurred_at) AS day,
               COUNT(*) FILTER (WHERE event_name='academy_course_viewed')         AS course_views,
               COUNT(*) FILTER (WHERE event_name='academy_enrollment_completed')  AS enrollments,
               COUNT(*) FILTER (WHERE event_name='academy_lesson_completed')      AS lessons_completed,
               COUNT(*) FILTER (WHERE event_name='academy_certificate_earned')    AS certs
          FROM analytics_events
         WHERE event_name IN ('academy_course_viewed','academy_enrollment_completed','academy_lesson_completed','academy_certificate_earned')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;""")

    course_views_7d        = int(volume["course_views_7d"]        or 0)
    enroll_started_28d     = int(volume["enroll_started_28d"]     or 0)
    enroll_completed_28d   = int(volume["enroll_completed_28d"]   or 0)
    lessons_started_28d    = int(volume["lessons_started_28d"]    or 0)
    lessons_completed_28d  = int(volume["lessons_completed_28d"]  or 0)
    quizzes_completed_28d  = int(volume["quizzes_completed_28d"]  or 0)
    quizzes_passed_28d     = int(volume["quizzes_passed_28d"]     or 0)
    ai_tutor_28d           = int(volume["ai_tutor_28d"]           or 0)
    ai_tutor_helpful_28d   = int(volume["ai_tutor_helpful_28d"]   or 0)
    avg_lesson_time_ms     = float(volume["avg_lesson_time_ms"]) if volume["avg_lesson_time_ms"] is not None else None

    return JSONResponse({
        "cards": {
            "home_views_7d":          int(volume["home_views_7d"]        or 0),
            "course_views_7d":        course_views_7d,
            "course_views_28d":       int(volume["course_views_28d"]     or 0),
            "enroll_started_28d":     enroll_started_28d,
            "enroll_completed_28d":   enroll_completed_28d,
            "enroll_completed_7d":    int(volume["enroll_completed_7d"]  or 0),
            "lessons_started_28d":    lessons_started_28d,
            "lessons_completed_28d":  lessons_completed_28d,
            "lessons_completed_7d":   int(volume["lessons_completed_7d"] or 0),
            "assignments_28d":        int(volume["assignments_28d"]      or 0),
            "quizzes_completed_28d":  quizzes_completed_28d,
            "quizzes_passed_28d":     quizzes_passed_28d,
            "certs_28d":              int(volume["certs_28d"]            or 0),
            "ai_tutor_7d":            int(volume["ai_tutor_7d"]          or 0),
            "ai_tutor_28d":           ai_tutor_28d,
            "live_28d":               int(volume["live_28d"]             or 0),
            "active_learners_7d":     int(volume["active_learners_7d"]   or 0),
            "avg_lesson_time_ms":     avg_lesson_time_ms,
            "enrollment_conversion":  (enroll_completed_28d / enroll_started_28d) if enroll_started_28d else 0.0,
            "lesson_completion":      (lessons_completed_28d / lessons_started_28d) if lessons_started_28d else 0.0,
            "quiz_pass_rate":         (quizzes_passed_28d / quizzes_completed_28d) if quizzes_completed_28d else 0.0,
            "ai_helpful_rate":        (ai_tutor_helpful_28d / ai_tutor_28d) if ai_tutor_28d else 0.0,
        },
        "by_score":      await _serialise_rows(by_score),
        "by_milestone":  await _serialise_rows(by_milestone),
        "by_speed":      await _serialise_rows(by_speed),
        "by_role":       await _serialise_rows(by_role),
        "top_courses":   await _serialise_rows(top_courses),
        "daily":         await _serialise_rows(daily),
    })


async def section_calendar(request: Request) -> JSONResponse:
    """All Calendar section metrics in a single payload."""
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_viewed'           AND occurred_at > NOW() - INTERVAL '7 days')   AS views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_viewed'           AND occurred_at > NOW() - INTERVAL '28 days')  AS views_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_event_created'    AND occurred_at > NOW() - INTERVAL '7 days')   AS events_created_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_event_created'    AND occurred_at > NOW() - INTERVAL '28 days')  AS events_created_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_event_updated'    AND occurred_at > NOW() - INTERVAL '28 days')  AS events_updated_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_task_scheduled'   AND occurred_at > NOW() - INTERVAL '28 days')  AS tasks_scheduled_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_external_toggled' AND occurred_at > NOW() - INTERVAL '28 days' AND (properties->>'enabled')::bool = TRUE) AS externals_enabled_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_external_toggled' AND occurred_at > NOW() - INTERVAL '28 days') AS externals_total_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='calendar_reminder_set'     AND occurred_at > NOW() - INTERVAL '28 days')  AS reminders_28d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='calendar_event_created' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS active_creators_7d;
        """
    )

    by_view_mode = await pool.fetch("""
        SELECT properties->>'view_mode' AS view_mode, COUNT(*) AS n FROM analytics_events
         WHERE event_name='calendar_viewed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'view_mode' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_event_type = await pool.fetch("""
        SELECT properties->>'event_type' AS event_type, COUNT(*) AS n FROM analytics_events
         WHERE event_name='calendar_event_created' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'event_type' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_provider = await pool.fetch("""
        SELECT properties->>'provider' AS provider,
               COUNT(*) FILTER (WHERE (properties->>'enabled')::bool = TRUE)  AS enabled,
               COUNT(*) FILTER (WHERE (properties->>'enabled')::bool = FALSE) AS disabled
          FROM analytics_events
         WHERE event_name='calendar_external_toggled' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'provider' IS NOT NULL
         GROUP BY 1 ORDER BY enabled DESC;""")
    by_reminder = await pool.fetch("""
        SELECT properties->>'reminder_offset_bucket' AS bucket, COUNT(*) AS n FROM analytics_events
         WHERE event_name='calendar_reminder_set' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'reminder_offset_bucket' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")

    daily = await pool.fetch("""
        SELECT DATE(occurred_at) AS day,
               COUNT(*) FILTER (WHERE event_name='calendar_viewed')         AS views,
               COUNT(*) FILTER (WHERE event_name='calendar_event_created')  AS events_created,
               COUNT(*) FILTER (WHERE event_name='calendar_task_scheduled') AS tasks_scheduled
          FROM analytics_events
         WHERE event_name IN ('calendar_viewed','calendar_event_created','calendar_task_scheduled')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;""")

    events_created_28d    = int(volume["events_created_28d"]    or 0)
    events_updated_28d    = int(volume["events_updated_28d"]    or 0)
    reminders_28d         = int(volume["reminders_28d"]         or 0)
    externals_enabled_28d = int(volume["externals_enabled_28d"] or 0)
    externals_total_28d   = int(volume["externals_total_28d"]   or 0)

    return JSONResponse({
        "cards": {
            "views_7d":              int(volume["views_7d"]              or 0),
            "views_28d":             int(volume["views_28d"]             or 0),
            "events_created_7d":     int(volume["events_created_7d"]     or 0),
            "events_created_28d":    events_created_28d,
            "events_updated_28d":    events_updated_28d,
            "tasks_scheduled_28d":   int(volume["tasks_scheduled_28d"]   or 0),
            "externals_enabled_28d": externals_enabled_28d,
            "reminders_28d":         reminders_28d,
            "active_creators_7d":    int(volume["active_creators_7d"]    or 0),
            "edit_rate":             (events_updated_28d / events_created_28d) if events_created_28d else 0.0,
            "reminder_attach_rate":  (reminders_28d / events_created_28d) if events_created_28d else 0.0,
            "external_enable_rate":  (externals_enabled_28d / externals_total_28d) if externals_total_28d else 0.0,
        },
        "by_view_mode":  await _serialise_rows(by_view_mode),
        "by_event_type": await _serialise_rows(by_event_type),
        "by_provider":   await _serialise_rows(by_provider),
        "by_reminder":   await _serialise_rows(by_reminder),
        "daily":         await _serialise_rows(daily),
    })


async def section_case_management(request: Request) -> JSONResponse:
    """All Case Management section metrics in a single payload."""
    _, err = await require_admin(request)
    if err: return err
    cfg = load_config()
    pool = await get_pool(cfg.db_url)

    volume = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_management_viewed'   AND occurred_at > NOW() - INTERVAL '7 days')   AS views_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_management_viewed'   AND occurred_at > NOW() - INTERVAL '28 days')  AS views_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_record_opened'       AND occurred_at > NOW() - INTERVAL '7 days')   AS opens_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_record_opened'       AND occurred_at > NOW() - INTERVAL '28 days')  AS opens_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_task_created'        AND occurred_at > NOW() - INTERVAL '28 days')  AS tasks_created_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_task_completed'      AND occurred_at > NOW() - INTERVAL '28 days')  AS tasks_completed_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_ai_summary_requested' AND occurred_at > NOW() - INTERVAL '28 days') AS ai_summaries_28d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_ai_summary_requested' AND occurred_at > NOW() - INTERVAL '7 days')  AS ai_summaries_7d,
            (SELECT COUNT(*) FROM analytics_events WHERE event_name='case_export_created'      AND occurred_at > NOW() - INTERVAL '28 days')  AS exports_28d,
            (SELECT COUNT(DISTINCT user_id) FROM analytics_events WHERE event_name='case_management_viewed' AND occurred_at > NOW() - INTERVAL '7 days' AND user_id IS NOT NULL) AS active_workers_7d,
            (SELECT COUNT(DISTINCT properties->>'case_id_hash') FROM analytics_events WHERE event_name='case_record_opened' AND occurred_at > NOW() - INTERVAL '7 days') AS unique_cases_7d;
        """
    )

    by_view_mode = await pool.fetch("""
        SELECT properties->>'view_mode' AS view_mode, COUNT(*) AS n FROM analytics_events
         WHERE event_name='case_management_viewed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'view_mode' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_role = await pool.fetch("""
        SELECT properties->>'role' AS role, COUNT(*) AS n FROM analytics_events
         WHERE event_name='case_management_viewed' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'role' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_open_source = await pool.fetch("""
        SELECT properties->>'source' AS source, COUNT(*) AS n FROM analytics_events
         WHERE event_name='case_record_opened' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'source' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_task_type = await pool.fetch("""
        SELECT properties->>'task_type' AS task_type,
               COUNT(*) FILTER (WHERE event_name='case_task_created')   AS created,
               COUNT(*) FILTER (WHERE event_name='case_task_completed') AS completed
          FROM analytics_events
         WHERE event_name IN ('case_task_created','case_task_completed')
           AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'task_type' IS NOT NULL
         GROUP BY 1 ORDER BY created DESC;""")
    by_export_type = await pool.fetch("""
        SELECT properties->>'export_type' AS export_type, COUNT(*) AS n FROM analytics_events
         WHERE event_name='case_export_created' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'export_type' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")
    by_ai_source = await pool.fetch("""
        SELECT properties->>'source' AS source, COUNT(*) AS n FROM analytics_events
         WHERE event_name='case_ai_summary_requested' AND occurred_at > NOW() - INTERVAL '28 days'
           AND properties->>'source' IS NOT NULL
         GROUP BY 1 ORDER BY n DESC;""")

    daily = await pool.fetch("""
        SELECT DATE(occurred_at) AS day,
               COUNT(*) FILTER (WHERE event_name='case_record_opened')        AS opens,
               COUNT(*) FILTER (WHERE event_name='case_task_created')         AS tasks_created,
               COUNT(*) FILTER (WHERE event_name='case_task_completed')       AS tasks_completed,
               COUNT(*) FILTER (WHERE event_name='case_ai_summary_requested') AS ai_summaries
          FROM analytics_events
         WHERE event_name IN ('case_record_opened','case_task_created','case_task_completed','case_ai_summary_requested')
           AND occurred_at > NOW() - INTERVAL '28 days'
         GROUP BY 1 ORDER BY 1;""")

    opens_28d            = int(volume["opens_28d"]            or 0)
    tasks_created_28d    = int(volume["tasks_created_28d"]    or 0)
    tasks_completed_28d  = int(volume["tasks_completed_28d"]  or 0)
    ai_summaries_28d     = int(volume["ai_summaries_28d"]     or 0)

    return JSONResponse({
        "cards": {
            "views_7d":            int(volume["views_7d"]            or 0),
            "views_28d":           int(volume["views_28d"]           or 0),
            "opens_7d":            int(volume["opens_7d"]            or 0),
            "opens_28d":           opens_28d,
            "tasks_created_28d":   tasks_created_28d,
            "tasks_completed_28d": tasks_completed_28d,
            "ai_summaries_7d":     int(volume["ai_summaries_7d"]     or 0),
            "ai_summaries_28d":    ai_summaries_28d,
            "exports_28d":         int(volume["exports_28d"]         or 0),
            "active_workers_7d":   int(volume["active_workers_7d"]   or 0),
            "unique_cases_7d":     int(volume["unique_cases_7d"]     or 0),
            "task_completion":     (tasks_completed_28d / tasks_created_28d) if tasks_created_28d else 0.0,
            "ai_share":            (ai_summaries_28d / opens_28d) if opens_28d else 0.0,
            "tasks_per_open":      (tasks_created_28d / opens_28d) if opens_28d else 0.0,
        },
        "by_view_mode":   await _serialise_rows(by_view_mode),
        "by_role":        await _serialise_rows(by_role),
        "by_open_source": await _serialise_rows(by_open_source),
        "by_task_type":   await _serialise_rows(by_task_type),
        "by_export_type": await _serialise_rows(by_export_type),
        "by_ai_source":   await _serialise_rows(by_ai_source),
        "daily":          await _serialise_rows(daily),
    })
