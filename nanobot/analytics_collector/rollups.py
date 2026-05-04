"""Rollup loop. Runs every minute. Computes:

  - analytics_daily_rollups (today, yesterday)
  - analytics_product_area_rollups (today, yesterday)
  - analytics_profile_rollups (incremental, profiles with activity in last hour)
  - trims analytics_live_events to the last 1000 rows

Funnel snapshots and retention rollups run nightly (separate cron).
"""

from __future__ import annotations

import asyncio
import logging

from .config import CollectorConfig
from .db import get_pool

log = logging.getLogger("analytics_collector.rollups")


# Events that count toward "meaningful_actions" / "conversions" in rollups.
MEANINGFUL = (
    "directory_service_action_clicked",
    "directory_service_saved",
    "directory_review_submitted",
    "jobs_application_submitted",
    "jobs_resume_completed",
    "jobs_employer_listing_published",
    "gallery_artwork_uploaded",
    "gallery_artwork_favorited",
    "gallery_artwork_commented",
    "academy_lesson_completed",
    "academy_assignment_submitted",
    "academy_quiz_completed",
    "academy_certificate_earned",
    "messages_message_sent",
    "messages_call_completed",
    "tasks_task_completed",
    "documents_document_shared",
    "documents_document_exported",
    "ai_feedback_submitted",
    "street_profile_created",
    "street_profile_updated",
)

CONVERSIONS = (
    "jobs_application_submitted",
    "gallery_artwork_uploaded",
    "academy_certificate_earned",
    "directory_listing_claim_completed",
    "messages_dm_started",
    "street_profile_created",
)


async def run_loop(cfg: CollectorConfig) -> None:
    """Fire-and-forget background task."""
    while True:
        try:
            await _refresh_daily(cfg)
            await _refresh_product_area(cfg)
            await _refresh_profile_rollups(cfg)
            await _trim_live(cfg)
        except Exception as exc:
            log.warning("rollup tick failed: %s", exc)
        await asyncio.sleep(cfg.rollup_interval_sec)


async def _refresh_daily(cfg: CollectorConfig) -> None:
    pool = await get_pool(cfg.db_url)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO analytics_daily_rollups (
                day, product_area, user_role, app_variant,
                sessions, daily_active_users, new_users, new_street_profiles,
                page_views, meaningful_actions, conversions, errors,
                rage_clicks, dead_clicks, avg_active_time_ms
            )
            SELECT
                day, product_area, user_role, app_variant,
                sessions, daily_active_users, new_users, new_street_profiles,
                page_views, meaningful_actions, conversions, errors,
                rage_clicks, dead_clicks, avg_active_time_ms
            FROM (
                SELECT
                    DATE(occurred_at) AS day,
                    COALESCE(product_area, '_all') AS product_area,
                    COALESCE(user_role, '_all')    AS user_role,
                    COALESCE(app_variant, '_all')  AS app_variant,
                    COUNT(DISTINCT session_id) AS sessions,
                    COUNT(DISTINCT user_id)    AS daily_active_users,
                    COUNT(*) FILTER (WHERE event_name = 'auth_signed_up')                                       AS new_users,
                    COUNT(*) FILTER (WHERE event_name = 'street_profile_created')                                AS new_street_profiles,
                    COUNT(*) FILTER (WHERE event_name = 'page_entered')                                          AS page_views,
                    COUNT(*) FILTER (WHERE event_name = ANY($1::text[]))                                         AS meaningful_actions,
                    COUNT(*) FILTER (WHERE event_name = ANY($2::text[]))                                         AS conversions,
                    COUNT(*) FILTER (WHERE event_name IN ('page_error_occurred','platform_api_request_failed','ai_error_seen')) AS errors,
                    COUNT(*) FILTER (WHERE event_name = 'rage_click_detected')                                   AS rage_clicks,
                    COUNT(*) FILTER (WHERE event_name = 'dead_click_detected')                                   AS dead_clicks,
                    0::int AS avg_active_time_ms
                FROM analytics_events
                WHERE occurred_at > (CURRENT_DATE - INTERVAL '2 days')
                GROUP BY 1, 2, 3, 4
            ) g
            ON CONFLICT (day, product_area, user_role, app_variant) DO UPDATE SET
                sessions             = EXCLUDED.sessions,
                daily_active_users   = EXCLUDED.daily_active_users,
                new_users            = EXCLUDED.new_users,
                new_street_profiles  = EXCLUDED.new_street_profiles,
                page_views           = EXCLUDED.page_views,
                meaningful_actions   = EXCLUDED.meaningful_actions,
                conversions          = EXCLUDED.conversions,
                errors               = EXCLUDED.errors,
                rage_clicks          = EXCLUDED.rage_clicks,
                dead_clicks          = EXCLUDED.dead_clicks,
                avg_active_time_ms   = EXCLUDED.avg_active_time_ms;
            """,
            list(MEANINGFUL),
            list(CONVERSIONS),
        )


async def _refresh_product_area(cfg: CollectorConfig) -> None:
    pool = await get_pool(cfg.db_url)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            WITH window_events AS (
                SELECT DATE(occurred_at) AS day, product_area, event_name,
                       user_id, properties
                  FROM analytics_events
                 WHERE occurred_at > (CURRENT_DATE - INTERVAL '2 days')
                   AND product_area IS NOT NULL
            ),
            base AS (
                SELECT
                    day, product_area,
                    COUNT(DISTINCT user_id) AS active_users,
                    COUNT(*) FILTER (WHERE event_name = 'auth_signed_up') AS new_users,
                    COUNT(*) FILTER (WHERE event_name IN ('street_profile_created','jobs_resume_completed','gallery_artwork_uploaded','academy_enrollment_completed','messages_dm_started')) AS activations,
                    COUNT(*) FILTER (WHERE event_name = ANY($1::text[])) AS conversions,
                    COUNT(*) FILTER (WHERE event_name = 'page_entered') AS page_views,
                    COUNT(*) FILTER (WHERE event_name IN ('page_error_occurred','platform_api_request_failed','ai_error_seen')) AS error_count,
                    COUNT(*) FILTER (WHERE event_name = 'platform_api_request_completed' AND (properties->>'latency_ms') ~ '^[0-9]+$' AND (properties->>'latency_ms')::int > 1500) AS slow_route_count
                  FROM window_events
                 GROUP BY day, product_area
            ),
            durations AS (
                SELECT DATE(occurred_at) AS day, product_area,
                       COALESCE(AVG(active_time_ms)::int, 0) AS avg_active_time_ms
                  FROM analytics_page_durations
                 WHERE occurred_at > (CURRENT_DATE - INTERVAL '2 days') AND product_area IS NOT NULL
                 GROUP BY DATE(occurred_at), product_area
            ),
            top_routes AS (
                SELECT day, product_area, route_pattern,
                       ROW_NUMBER() OVER (PARTITION BY day, product_area ORDER BY n DESC) AS rn
                  FROM (
                      SELECT DATE(occurred_at) AS day, product_area, route_pattern, COUNT(*) AS n
                        FROM analytics_events
                       WHERE occurred_at > (CURRENT_DATE - INTERVAL '2 days')
                         AND event_name = 'page_entered' AND product_area IS NOT NULL
                       GROUP BY 1, 2, 3
                  ) g
            ),
            top_actions AS (
                SELECT day, product_area, event_name,
                       ROW_NUMBER() OVER (PARTITION BY day, product_area ORDER BY n DESC) AS rn
                  FROM (
                      SELECT DATE(occurred_at) AS day, product_area, event_name, COUNT(*) AS n
                        FROM analytics_events
                       WHERE occurred_at > (CURRENT_DATE - INTERVAL '2 days')
                         AND event_name = ANY($2::text[]) AND product_area IS NOT NULL
                       GROUP BY 1, 2, 3
                  ) g
            ),
            violations AS (
                SELECT DATE(occurred_at) AS day, COUNT(*) AS n
                  FROM analytics_event_schema_violations
                 WHERE occurred_at > (CURRENT_DATE - INTERVAL '2 days')
                 GROUP BY DATE(occurred_at)
            )
            INSERT INTO analytics_product_area_rollups (
                day, product_area, active_users, new_users, activations,
                conversions, page_views, avg_active_time_ms, error_count,
                slow_route_count, top_route_pattern, top_action, event_quality_score
            )
            SELECT
                base.day, base.product_area, base.active_users, base.new_users, base.activations,
                base.conversions, base.page_views, COALESCE(durations.avg_active_time_ms, 0),
                base.error_count, base.slow_route_count,
                top_routes.route_pattern,
                top_actions.event_name,
                (100 - LEAST(100, COALESCE(violations.n, 0) / 5))::int
              FROM base
              LEFT JOIN durations    USING (day, product_area)
              LEFT JOIN top_routes   ON top_routes.day = base.day AND top_routes.product_area = base.product_area AND top_routes.rn = 1
              LEFT JOIN top_actions  ON top_actions.day = base.day AND top_actions.product_area = base.product_area AND top_actions.rn = 1
              LEFT JOIN violations   ON violations.day = base.day
            ON CONFLICT (day, product_area) DO UPDATE SET
                active_users        = EXCLUDED.active_users,
                new_users           = EXCLUDED.new_users,
                activations         = EXCLUDED.activations,
                conversions         = EXCLUDED.conversions,
                page_views          = EXCLUDED.page_views,
                avg_active_time_ms  = EXCLUDED.avg_active_time_ms,
                error_count         = EXCLUDED.error_count,
                slow_route_count    = EXCLUDED.slow_route_count,
                top_route_pattern   = EXCLUDED.top_route_pattern,
                top_action          = EXCLUDED.top_action,
                event_quality_score = EXCLUDED.event_quality_score;
            """,
            list(CONVERSIONS),
            list(MEANINGFUL),
        )


async def _refresh_profile_rollups(cfg: CollectorConfig) -> None:
    """Recompute profile rollups for any profile with activity in the last hour."""
    pool = await get_pool(cfg.db_url)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO analytics_profile_rollups (
                street_profile_id, user_id, profile_views, profile_views_7d,
                cta_clicks, artworks_uploaded, job_applications, lessons_completed,
                messages_sent, services_contacted, last_seen_at, last_active_product_area,
                first_activated_area, is_activated, activated_at,
                cross_area_count_7d, needs_nudge, nudge_reason, refreshed_at
            )
            SELECT
                e.street_profile_id,
                MAX(e.user_id) AS user_id,
                COUNT(*) FILTER (WHERE e.event_name = 'street_profile_viewed' AND (e.properties->>'is_profile_owner')::bool = false) AS profile_views,
                COUNT(*) FILTER (WHERE e.event_name = 'street_profile_viewed' AND e.occurred_at > NOW() - INTERVAL '7 days' AND (e.properties->>'is_profile_owner')::bool = false) AS profile_views_7d,
                COUNT(*) FILTER (WHERE e.event_name = 'street_profile_cta_clicked')      AS cta_clicks,
                COUNT(*) FILTER (WHERE e.event_name = 'gallery_artwork_uploaded')        AS artworks_uploaded,
                COUNT(*) FILTER (WHERE e.event_name = 'jobs_application_submitted')      AS job_applications,
                COUNT(*) FILTER (WHERE e.event_name = 'academy_lesson_completed')        AS lessons_completed,
                COUNT(*) FILTER (WHERE e.event_name = 'messages_message_sent')           AS messages_sent,
                COUNT(*) FILTER (WHERE e.event_name = 'directory_service_action_clicked') AS services_contacted,
                MAX(e.occurred_at) AS last_seen_at,
                (ARRAY_AGG(e.product_area ORDER BY e.occurred_at DESC) FILTER (WHERE e.product_area IS NOT NULL))[1] AS last_active_product_area,
                (ARRAY_AGG(e.product_area ORDER BY e.occurred_at ASC)  FILTER (WHERE e.event_name = ANY($1::text[])))[1] AS first_activated_area,
                BOOL_OR(e.event_name = ANY($1::text[])) AS is_activated,
                MIN(e.occurred_at) FILTER (WHERE e.event_name = ANY($1::text[])) AS activated_at,
                COUNT(DISTINCT e.product_area) FILTER (WHERE e.occurred_at > NOW() - INTERVAL '7 days' AND e.product_area IS NOT NULL) AS cross_area_count_7d,
                (
                    BOOL_AND(e.event_name <> 'street_profile_created' OR e.occurred_at < NOW() - INTERVAL '14 days')
                    AND BOOL_AND(e.event_name <> ANY($1::text[]))
                ) AS needs_nudge,
                CASE
                    WHEN COUNT(*) FILTER (WHERE e.event_name = 'gallery_upload_blocked') > 0 THEN 'gallery_upload_blocked'
                    WHEN COUNT(*) FILTER (WHERE e.event_name = ANY($1::text[])) = 0 THEN 'no_activation_yet'
                    ELSE NULL
                END AS nudge_reason,
                NOW() AS refreshed_at
            FROM analytics_events e
            WHERE e.street_profile_id IS NOT NULL
              AND e.street_profile_id IN (
                  SELECT DISTINCT street_profile_id
                    FROM analytics_events
                   WHERE street_profile_id IS NOT NULL
                     AND occurred_at > NOW() - INTERVAL '1 hour'
              )
            GROUP BY e.street_profile_id
            ON CONFLICT (street_profile_id) DO UPDATE SET
                user_id              = COALESCE(EXCLUDED.user_id, analytics_profile_rollups.user_id),
                profile_views        = EXCLUDED.profile_views,
                profile_views_7d     = EXCLUDED.profile_views_7d,
                cta_clicks           = EXCLUDED.cta_clicks,
                artworks_uploaded    = EXCLUDED.artworks_uploaded,
                job_applications     = EXCLUDED.job_applications,
                lessons_completed    = EXCLUDED.lessons_completed,
                messages_sent        = EXCLUDED.messages_sent,
                services_contacted   = EXCLUDED.services_contacted,
                last_seen_at         = EXCLUDED.last_seen_at,
                last_active_product_area = EXCLUDED.last_active_product_area,
                first_activated_area = COALESCE(analytics_profile_rollups.first_activated_area, EXCLUDED.first_activated_area),
                is_activated         = analytics_profile_rollups.is_activated OR EXCLUDED.is_activated,
                activated_at         = COALESCE(analytics_profile_rollups.activated_at, EXCLUDED.activated_at),
                cross_area_count_7d  = EXCLUDED.cross_area_count_7d,
                needs_nudge          = EXCLUDED.needs_nudge,
                nudge_reason         = EXCLUDED.nudge_reason,
                refreshed_at         = NOW();
            """,
            list(MEANINGFUL),
        )


async def _trim_live(cfg: CollectorConfig) -> None:
    pool = await get_pool(cfg.db_url)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM analytics_live_events
             WHERE id < COALESCE(
                 (SELECT id FROM analytics_live_events ORDER BY id DESC OFFSET $1 LIMIT 1),
                 0
             );
            """,
            cfg.live_buffer_max,
        )
