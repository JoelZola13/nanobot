"""OpenAI-compatible API server that wraps nanobot's AgentLoop."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles

# ── Gallery API (lazy import to avoid circular deps) ──
from nanobot.gallery_api import (
    gallery_list_artworks as _gallery_list_artworks,
    gallery_list_mediums as _gallery_list_mediums,
    gallery_get_artwork as _gallery_get_artwork,
    gallery_add_favorite as _gallery_add_favorite,
    gallery_remove_favorite as _gallery_remove_favorite,
    gallery_user_favorites as _gallery_user_favorites,
    gallery_user_favorites_legacy as _gallery_user_favorites_legacy,
    gallery_tags as _gallery_tags,
    gallery_uploads as _gallery_uploads,
    gallery_create_artwork as _gallery_create_artwork,
    street_profiles_batch_lookup as _street_profiles_batch_lookup,
    gallery_saved_collections as _gallery_saved_collections,
    gallery_save_collection as _gallery_save_collection,
    gallery_unsave_collection as _gallery_unsave_collection,
    gallery_list_comments as _gallery_list_comments,
    gallery_post_comment as _gallery_post_comment,
    gallery_edit_comment as _gallery_edit_comment,
    gallery_delete_comment as _gallery_delete_comment,
)

SCREENSHOTS_DIR = Path.home() / ".nanobot" / "workspace" / "screenshots"
GALLERY_DIR = Path.home() / ".nanobot" / "workspace" / "gallery"
GALLERY_DB_FILE = Path.home() / ".nanobot" / "workspace" / "gallery" / "artworks.json"
ACADEMY_DATA_DIR = Path.home() / ".nanobot" / "workspace" / "academy"
ACADEMY_RUNTIME_FILE = ACADEMY_DATA_DIR / "runtime.json"
VIDEOS_DIR = Path.home() / ".nanobot" / "workspace" / "remotion" / "out"
AUDIO_DIR = Path.home() / ".nanobot" / "workspace" / "remotion" / "public" / "audio"
ARTICLE_IMAGES_DIR = Path.home() / ".nanobot" / "workspace" / "article-images"
NEWS_DIR = Path.home() / ".nanobot" / "workspace" / "news"
NEWS_DB_FILE = Path.home() / ".nanobot" / "workspace" / "news" / "articles.json"
AVATARS_DIR = Path(__file__).parent.parent / "static" / "avatars"
SHARED_ASSETS_DIR = Path(__file__).parent.parent / "LibreChat" / "client" / "public"
GATEWAY_STATIC_DIR = Path(__file__).parent / "gateway" / "static"

from nanobot.config.loader import load_config
from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.session.manager import SessionManager
from nanobot.cron.service import CronService


_agent: AgentLoop | None = None
_cron: CronService | None = None
_orchestrator: Any = None  # Orchestrator | None — lazy import to avoid circular deps
_harness: Any = None  # DeepAgentHarness | None — deepagents-powered engine
_gateway: Any = None  # GatewayServer | None — WS mission control
_redis_bus: Any = None  # RedisBus | None — cross-service event bus


def _academy_default_state() -> dict[str, list[dict[str, Any]]]:
    return {
        "enrollments": [],
        "attendance_records": [],
        "learning_paths": [],
        "live_sessions": [],
        "session_registrations": [],
        "session_polls": [],
        "poll_responses": [],
        "session_questions": [],
        "session_feedback": [],
        "cohorts": [],
        "cohort_enrollments": [],
        "cohort_deadlines": [],
        "cohort_announcements": [],
        "reviews": [],
        "certificates": [],
        "course_materials": [],
        "course_schedule_items": [],
        "assignments": [],
        "submissions": [],
        "rubrics": [],
        "rubric_grades": [],
        "video_progress": [],
        "video_bookmarks": [],
        "video_notes": [],
        "forum_discussions": [],
    }


def _load_academy_state() -> dict[str, list[dict[str, Any]]]:
    defaults = _academy_default_state()
    ACADEMY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ACADEMY_RUNTIME_FILE.exists():
        return defaults

    try:
        stored = json.loads(ACADEMY_RUNTIME_FILE.read_text())
    except Exception:
        return defaults

    merged: dict[str, list[dict[str, Any]]] = {}
    for key, fallback in defaults.items():
        value = stored.get(key, fallback)
        merged[key] = list(value) if isinstance(value, list) else list(fallback)
    return merged


def _save_academy_state(state: dict[str, list[dict[str, Any]]]) -> None:
    ACADEMY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ACADEMY_RUNTIME_FILE.write_text(json.dumps(state, indent=2))


def _academy_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _academy_parse_iso(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def _academy_make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


_ACADEMY_RESERVED_PATH_SLUGS = {
    "job-ready",
    "digital-basics",
    "housing-stability",
}


def _academy_slugify(value: str | None) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return base or f"path-{uuid.uuid4().hex[:6]}"


def _academy_unique_path_slug(state: dict[str, list[dict[str, Any]]], title: str, explicit_slug: str | None = None) -> str:
    taken = {
        str(item.get("slug") or "").strip().lower()
        for item in state["learning_paths"]
        if item.get("slug")
    } | _ACADEMY_RESERVED_PATH_SLUGS

    base = _academy_slugify(explicit_slug or title)
    slug = base
    index = 2
    while slug in taken:
        slug = f"{base}-{index}"
        index += 1
    return slug


def _academy_find(items: list[dict[str, Any]], item_id: str) -> dict[str, Any] | None:
    return next((item for item in items if item.get("id") == item_id), None)


def _academy_ensure_course_enrollment(
    state: dict[str, list[dict[str, Any]]],
    *,
    user_id: str,
    course_id: str,
    progress_percent: int = 0,
    status: str = "active",
    enrolled_at: str | None = None,
) -> dict[str, Any]:
    enrollment = next(
        (
            item
            for item in state["enrollments"]
            if item.get("user_id") == user_id and item.get("course_id") == course_id
        ),
        None,
    )
    now_iso = _academy_now_iso()
    requested_status = status or "active"
    requested_progress = int(progress_percent or 0)

    if enrollment is not None:
        if enrollment.get("status") == "dropped" and requested_status != "dropped":
            enrollment["status"] = requested_status
            enrollment["enrolled_at"] = enrolled_at or enrollment.get("enrolled_at") or now_iso
            enrollment["completed_at"] = None
        if requested_progress > int(enrollment.get("progress_percent") or 0):
            enrollment["progress_percent"] = requested_progress
        enrollment["updated_at"] = now_iso
        if int(enrollment.get("progress_percent") or 0) >= 100:
            enrollment["status"] = "completed"
            enrollment["completed_at"] = enrollment.get("completed_at") or now_iso
        return enrollment

    enrollment = {
        "id": _academy_make_id("enrollment"),
        "user_id": user_id,
        "course_id": course_id,
        "status": requested_status,
        "progress_percent": requested_progress,
        "last_accessed_at": None,
        "enrolled_at": enrolled_at or now_iso,
        "created_at": now_iso,
        "updated_at": now_iso,
        "completed_at": now_iso if requested_progress >= 100 else None,
    }
    if enrollment["completed_at"]:
        enrollment["status"] = "completed"
    state["enrollments"].append(enrollment)
    return enrollment


def _academy_sync_course_enrollments_from_cohorts(
    state: dict[str, list[dict[str, Any]]],
    *,
    user_id: str | None = None,
) -> bool:
    changed = False
    cohorts_by_id = {item.get("id"): item for item in state["cohorts"]}

    for cohort_enrollment in state["cohort_enrollments"]:
        cohort_user_id = cohort_enrollment.get("user_id")
        if user_id and cohort_user_id != user_id:
            continue

        cohort = cohorts_by_id.get(cohort_enrollment.get("cohort_id"))
        course_id = cohort.get("course_id") if cohort else None
        if not cohort_user_id or not course_id:
            continue

        existing = next(
            (
                item
                for item in state["enrollments"]
                if item.get("user_id") == cohort_user_id and item.get("course_id") == course_id
            ),
            None,
        )
        if existing is not None:
            continue

        _academy_ensure_course_enrollment(
            state,
            user_id=cohort_user_id,
            course_id=course_id,
            progress_percent=int(cohort_enrollment.get("progress_percent") or 0),
            status=str(cohort_enrollment.get("status") or "active"),
            enrolled_at=cohort_enrollment.get("enrolled_at"),
        )
        changed = True

    return changed


def _academy_session_list_response(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "sessions": sessions,
        "total": len(sessions),
        "upcoming_count": len(
            [s for s in sessions if s.get("status") == "scheduled" and _academy_parse_iso(s.get("scheduled_end")) >= now]
        ),
        "live_count": len([s for s in sessions if s.get("status") == "live"]),
    }


def _academy_session_with_registration(
    state: dict[str, list[dict[str, Any]]],
    session: dict[str, Any],
    user_id: str | None,
) -> dict[str, Any]:
    registration = None
    if user_id:
        registration = next(
            (
                item
                for item in state["session_registrations"]
                if item.get("session_id") == session.get("id") and item.get("user_id") == user_id
            ),
            None,
        )
    return {
        "session": session,
        "registration": registration,
        "is_registered": registration is not None,
        "can_join": session.get("status") == "live" and registration is not None,
    }



def _academy_seed_forum(state: dict[str, list[dict[str, Any]]], forum_id: int, course_id: str) -> None:
    if any(item.get("forum_id") == forum_id for item in state["forum_discussions"]):
        return

    state["forum_discussions"].append(
        {
            "id": forum_id * 1000 + 1,
            "forum_id": forum_id,
            "course_id": course_id,
            "name": "Welcome discussion",
            "subject": "Introduce yourself",
            "message": "Share your goals for this Academy course and what you want to build.",
            "userfullname": "Street Voices Academy",
            "userid": 0,
            "author_role": "instructor",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "modified": int(datetime.now(timezone.utc).timestamp()),
            "numreplies": 0,
            "pinned": True,
            "timemodified": int(datetime.now(timezone.utc).timestamp()),
            "replies": [],
            "reactions": {"up": [], "down": []},
        }
    )


def _academy_normalize_forum_discussion(discussion: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(discussion)
    normalized.setdefault("author_role", "instructor" if str(normalized.get("userid")) == "0" else "student")

    normalized_replies: list[dict[str, Any]] = []
    for reply in normalized.get("replies", []) or []:
        reply_payload = dict(reply)
        reply_payload.setdefault("author_role", "student")
        reply_payload.setdefault("created", normalized.get("created", int(time.time())))
        normalized_replies.append(reply_payload)

    reactions = normalized.get("reactions") or {}
    normalized["replies"] = normalized_replies
    normalized["reactions"] = {
        "up": list(reactions.get("up", [])),
        "down": list(reactions.get("down", [])),
    }
    normalized["numreplies"] = len(normalized_replies)
    return normalized


def _academy_stable_id(prefix: str, *parts: str) -> str:
    seed = "::".join([prefix, *parts])
    return f"{prefix}-{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:10]}"


async def _academy_fetch_course_meta(course_id: str) -> dict[str, Any] | None:
    import httpx

    base = f"{_SUPABASE_URL}/rest/v1/academy_courses"
    headers = _supabase_headers()
    qs = f"select=id,title,instructor_name&id=eq.{course_id}&limit=1"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{base}?{qs}", headers=headers)
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
    except Exception:
        return None

    return None


def _academy_assignment_stats(
    state: dict[str, list[dict[str, Any]]],
    assignment_id: str,
) -> tuple[int, int, float]:
    submissions = [row for row in state["submissions"] if row.get("assignment_id") == assignment_id]
    graded = [
        row
        for row in submissions
        if row.get("status") in {"graded", "returned"} and row.get("adjusted_score", row.get("score")) is not None
    ]
    average = (
        round(
            sum(float(row.get("adjusted_score", row.get("score")) or 0) for row in graded) / len(graded),
            2,
        )
        if graded
        else 0
    )
    return len(submissions), len(graded), average


def _academy_assignment_payload(
    state: dict[str, list[dict[str, Any]]],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    submission_count, graded_count, average_score = _academy_assignment_stats(state, str(assignment.get("id")))
    return {
        **assignment,
        "submission_count": submission_count,
        "graded_count": graded_count,
        "average_score": average_score,
    }


def _academy_normalize_file_attachment(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    url = str(raw.get("url") or "").strip()
    if url == "":
        return None

    return {
        "url": url,
        "filename": raw.get("filename") or raw.get("fileName") or "street-voices-academy-file",
        "size_bytes": int(raw.get("size_bytes") or raw.get("sizeBytes") or 0),
        "mime_type": raw.get("mime_type") or raw.get("mimeType") or "application/octet-stream",
        "uploaded_at": raw.get("uploaded_at") or raw.get("uploadedAt") or _academy_now_iso(),
    }


def _academy_find_assignment(
    state: dict[str, list[dict[str, Any]]],
    assignment_id: str,
) -> dict[str, Any] | None:
    return next((row for row in state["assignments"] if row.get("id") == assignment_id), None)


def _academy_get_latest_submission(
    state: dict[str, list[dict[str, Any]]],
    assignment_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    submissions = [
        row
        for row in state["submissions"]
        if row.get("assignment_id") == assignment_id and row.get("user_id") == user_id
    ]
    if not submissions:
        return None
    submissions.sort(
        key=lambda row: (
            int(row.get("attempt_number") or 0),
            row.get("updated_at") or row.get("created_at") or "",
        ),
        reverse=True,
    )
    return submissions[0]


def _academy_submission_sort_key(submission: dict[str, Any]) -> tuple[int, str]:
    return (
        int(submission.get("attempt_number") or 0),
        str(
            submission.get("updated_at")
            or submission.get("submitted_at")
            or submission.get("created_at")
            or ""
        ),
    )


def _academy_display_name_for_submission(submission: dict[str, Any]) -> str:
    explicit_name = str(submission.get("user_name") or "").strip()
    if explicit_name:
        return explicit_name

    explicit_email = str(submission.get("user_email") or "").strip()
    if explicit_email:
        return explicit_email.split("@")[0] or explicit_email

    raw_user_id = str(submission.get("user_id") or "").strip()
    if raw_user_id == "":
        return "Academy learner"
    if raw_user_id.startswith("academy-local-user"):
        return "Academy learner"
    if "@" in raw_user_id:
        return raw_user_id.split("@")[0] or raw_user_id
    if len(raw_user_id) > 20 and "-" not in raw_user_id and "_" not in raw_user_id:
        return f"Learner {raw_user_id[:8]}"

    cleaned = raw_user_id.replace("-", " ").replace("_", " ").strip()
    if cleaned and any(character.isalpha() for character in cleaned):
        return " ".join(part.capitalize() for part in cleaned.split())[:48]

    return f"Learner {raw_user_id[:8]}"


def _academy_humanize_user_id(raw_user_id: str) -> str:
    raw_user_id = str(raw_user_id or "").strip()
    if raw_user_id == "":
        return "Academy learner"
    if raw_user_id.startswith("academy-local-user"):
        return "Academy learner"
    if "@" in raw_user_id:
        return raw_user_id.split("@")[0] or raw_user_id
    if len(raw_user_id) > 20 and "-" not in raw_user_id and "_" not in raw_user_id:
        return f"Learner {raw_user_id[:8]}"

    cleaned = raw_user_id.replace("-", " ").replace("_", " ").strip()
    if cleaned and any(character.isalpha() for character in cleaned):
        return " ".join(part.capitalize() for part in cleaned.split())[:48]

    return f"Learner {raw_user_id[:8]}"


def _academy_display_name_for_user(
    state: dict[str, list[dict[str, Any]]],
    user_id: str,
    *,
    course_id: str | None = None,
) -> str:
    raw_user_id = str(user_id or "").strip()
    if raw_user_id == "":
        return "Academy learner"

    course_submissions = [
        row
        for row in state["submissions"]
        if row.get("user_id") == raw_user_id and (course_id is None or row.get("course_id") == course_id)
    ]
    course_submissions.sort(key=_academy_submission_sort_key, reverse=True)
    if course_submissions:
        return _academy_display_name_for_submission(course_submissions[0])

    for certificate in state["certificates"]:
        if certificate.get("user_id") != raw_user_id:
            continue
        recipient_name = str(certificate.get("recipient_name") or "").strip()
        if recipient_name:
            return recipient_name

    for discussion in state["forum_discussions"]:
        if course_id and discussion.get("course_id") != course_id:
            continue

        if str(discussion.get("userid") or "") == raw_user_id:
            discussion_name = str(discussion.get("userfullname") or "").strip()
            if discussion_name:
                return discussion_name

        for reply in discussion.get("replies") or []:
            if str(reply.get("userid") or "") != raw_user_id:
                continue
            reply_name = str(reply.get("userfullname") or "").strip()
            if reply_name:
                return reply_name

    return _academy_humanize_user_id(raw_user_id)


def _academy_create_submission(
    state: dict[str, list[dict[str, Any]]],
    assignment: dict[str, Any],
    user_id: str,
    *,
    status: str = "draft",
    text_content: str | None = None,
    file_urls: list[dict[str, Any]] | None = None,
    quiz_answers: list[str] | None = None,
) -> dict[str, Any]:
    attempts = [
        row
        for row in state["submissions"]
        if row.get("assignment_id") == assignment.get("id") and row.get("user_id") == user_id
    ]
    now_iso = _academy_now_iso()
    submission = {
        "id": _academy_make_id("submission"),
        "assignment_id": assignment.get("id"),
        "course_id": assignment.get("course_id"),
        "user_id": user_id,
        "attempt_number": len(attempts) + 1,
        "status": status,
        "submission_type": assignment.get("assignment_type") or "text",
        "text_content": text_content or "",
        "quiz_answers": quiz_answers or [],
        "document_id": None,
        "file_urls": file_urls or [],
        "word_count": len((text_content or "").split()),
        "submitted_at": now_iso if status in {"submitted", "grading", "graded", "returned", "regrade_requested"} else None,
        "is_late": False,
        "days_late": 0,
        "late_penalty_applied": 0,
        "graded_at": None,
        "graded_by": None,
        "score": None,
        "adjusted_score": None,
        "letter_grade": None,
        "feedback": None,
        "feedback_attachments": [],
        "regrade_reason": None,
        "grading_locked_by": None,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    state["submissions"].append(submission)
    return submission


def _academy_quiz_answers_to_text(quiz_answers: list[Any] | None) -> str:
    answers = [str(item).strip() for item in (quiz_answers or []) if str(item).strip()]
    if not answers:
        return ""
    return "\n\n".join(f"Question {index + 1}: {answer}" for index, answer in enumerate(answers))


def _academy_create_course_assignment(
    state: dict[str, list[dict[str, Any]]],
    *,
    course_id: str,
    course_title: str,
    instructor_id: str,
    title: str,
    description: str | None,
    instructions: str | None,
    assignment_type: str,
    due_date: str | None,
    available_from: str | None,
    max_points: int | float | None,
    passing_score: int | float | None,
    max_attempts: int | None,
    allow_late_submissions: bool | None,
    late_penalty_percent: int | float | None,
    max_late_days: int | None,
    allowed_file_types: list[str] | None,
    max_file_size_mb: int | float | None,
    max_files: int | None,
    peer_review_enabled: bool | None,
    peer_reviews_required: int | None,
    is_published: bool | None,
    quiz_questions: list[str] | None,
    resource_file_name: str | None,
    resource_attachment: dict[str, Any] | None,
) -> dict[str, Any]:
    safe_type = assignment_type if assignment_type in {"file_upload", "text", "document", "mixed", "quiz"} else "text"
    safe_questions = [str(item).strip() for item in (quiz_questions or []) if str(item).strip()]
    now_iso = _academy_now_iso()
    assignment = {
        "id": _academy_make_id("assign"),
        "course_id": course_id,
        "course_title": course_title,
        "module_id": None,
        "lesson_id": None,
        "title": title,
        "description": description or f"{title} for {course_title}",
        "instructions": instructions or description or f"<p>Complete {title} in your student dashboard.</p>",
        "assignment_type": safe_type,
        "max_points": float(max_points or 100),
        "passing_score": float(passing_score or 70),
        "due_date": due_date,
        "available_from": available_from or now_iso,
        "available_until": None,
        "allow_late_submissions": True if allow_late_submissions is None else bool(allow_late_submissions),
        "late_penalty_percent": float(late_penalty_percent or 0),
        "max_late_days": int(max_late_days or 7),
        "max_attempts": int(max_attempts or 1),
        "peer_review_enabled": bool(peer_review_enabled),
        "peer_reviews_required": int(peer_reviews_required or 0),
        "rubric_id": None,
        "allowed_file_types": allowed_file_types or ["pdf", "docx", "jpg", "jpeg", "png"],
        "max_file_size_mb": float(max_file_size_mb or 15),
        "max_files": int(max_files or 3),
        "calendar_event_id": None,
        "is_published": True if is_published is None else bool(is_published),
        "created_by": instructor_id,
        "created_at": now_iso,
        "updated_at": now_iso,
        "quiz_questions": safe_questions,
        "resource_file_name": resource_file_name,
        "resource_attachment": _academy_normalize_file_attachment(resource_attachment),
    }
    state["assignments"].append(assignment)
    return assignment


def _academy_seed_assignments_for_course(
    state: dict[str, list[dict[str, Any]]],
    course_id: str,
    *,
    course_title: str | None = None,
    instructor_id: str | None = None,
) -> list[dict[str, Any]]:
    existing = [row for row in state["assignments"] if row.get("course_id") == course_id]
    if existing:
        return existing

    now = datetime.now(timezone.utc)
    available_from = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    due_primary = (now + timedelta(days=5)).isoformat().replace("+00:00", "Z")
    due_secondary = (now + timedelta(days=10)).isoformat().replace("+00:00", "Z")
    course_name = course_title or "Street Voices Academy Course"
    rubric_id = _academy_stable_id("rubric", course_id)

    rubric = next((row for row in state["rubrics"] if row.get("id") == rubric_id), None)
    if rubric is None:
        state["rubrics"].append(
            {
                "id": rubric_id,
                "title": f"{course_name} Rubric",
                "description": "A simple rubric for written reflections and final responses.",
                "course_id": course_id,
                "total_points": 100,
                "is_template": False,
                "is_active": True,
                "created_by": instructor_id or "academy-instructor",
                "created_at": _academy_now_iso(),
                "updated_at": _academy_now_iso(),
                "criteria": [
                    {
                        "id": _academy_stable_id("criterion", course_id, "clarity"),
                        "rubric_id": rubric_id,
                        "criterion_name": "Clarity",
                        "description": "Ideas are communicated clearly and confidently.",
                        "max_points": 40,
                        "order_index": 1,
                        "levels": [],
                    },
                    {
                        "id": _academy_stable_id("criterion", course_id, "reflection"),
                        "rubric_id": rubric_id,
                        "criterion_name": "Reflection",
                        "description": "The submission shows thought, detail, and personal reflection.",
                        "max_points": 60,
                        "order_index": 2,
                        "levels": [],
                    },
                ],
            }
        )

    assignments = [
        {
            "id": _academy_stable_id("assign", course_id, "reflection"),
            "course_id": course_id,
            "course_title": course_name,
            "module_id": None,
            "lesson_id": None,
            "title": "Weekly Reflection",
            "description": f"Share what you learned in {course_name} and how you will apply it.",
            "instructions": "<p>Write a short reflection about the biggest idea you learned this week and how you plan to use it.</p>",
            "assignment_type": "text",
            "max_points": 100,
            "passing_score": 70,
            "due_date": due_primary,
            "available_from": available_from,
            "available_until": None,
            "allow_late_submissions": True,
            "late_penalty_percent": 10,
            "max_late_days": 7,
            "max_attempts": 2,
            "peer_review_enabled": False,
            "peer_reviews_required": 0,
            "rubric_id": rubric_id,
            "allowed_file_types": ["pdf", "docx", "png"],
            "max_file_size_mb": 10,
            "max_files": 3,
            "calendar_event_id": None,
            "is_published": True,
            "created_by": instructor_id or "academy-instructor",
            "created_at": _academy_now_iso(),
            "updated_at": _academy_now_iso(),
        },
        {
            "id": _academy_stable_id("assign", course_id, "action-plan"),
            "course_id": course_id,
            "course_title": course_name,
            "module_id": None,
            "lesson_id": None,
            "title": "Action Plan Submission",
            "description": "Create a small action plan that shows how you will use this course in real life.",
            "instructions": "<p>Submit a short action plan with three concrete next steps you will take after this lesson.</p>",
            "assignment_type": "file_upload",
            "max_points": 50,
            "passing_score": 30,
            "due_date": due_secondary,
            "available_from": available_from,
            "available_until": None,
            "allow_late_submissions": True,
            "late_penalty_percent": 5,
            "max_late_days": 5,
            "max_attempts": 1,
            "peer_review_enabled": False,
            "peer_reviews_required": 0,
            "rubric_id": None,
            "allowed_file_types": ["pdf", "docx", "jpg", "jpeg", "png"],
            "max_file_size_mb": 15,
            "max_files": 3,
            "calendar_event_id": None,
            "is_published": True,
            "created_by": instructor_id or "academy-instructor",
            "created_at": _academy_now_iso(),
            "updated_at": _academy_now_iso(),
        },
    ]
    state["assignments"].extend(assignments)
    return assignments


async def _handle_assignments(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    if _academy_sync_course_enrollments_from_cohorts(state):
        _save_academy_state(state)
    method = request.method
    query = request.query_params

    if parts[0] == "courses" and len(parts) >= 3 and parts[2] == "assignments" and method == "GET":
        course_id = parts[1]
        course_meta = await _academy_fetch_course_meta(course_id)
        assignments = _academy_seed_assignments_for_course(
            state,
            course_id,
            course_title=course_meta.get("title") if course_meta else None,
            instructor_id=course_meta.get("instructor_id") if course_meta else None,
        )
        _save_academy_state(state)
        include_unpublished = query.get("include_unpublished") == "true"
        if not include_unpublished:
            assignments = [row for row in assignments if row.get("is_published", True)]
        return JSONResponse([_academy_assignment_payload(state, row) for row in assignments])

    if parts[0] == "courses" and len(parts) >= 3 and parts[2] == "assignments" and method == "POST":
        course_id = parts[1]
        course_meta = await _academy_fetch_course_meta(course_id)
        body = await request.json()
        title = str(body.get("title") or "").strip()
        if title == "":
            return JSONResponse({"detail": "title is required"}, status_code=400)
        assignment = _academy_create_course_assignment(
            state,
            course_id=course_id,
            course_title=course_meta.get("title") if course_meta else "Street Voices Academy Course",
            instructor_id=request.query_params.get("created_by") or body.get("created_by") or body.get("createdBy") or (course_meta.get("instructor_id") if course_meta else None) or "academy-instructor",
            title=title,
            description=body.get("description"),
            instructions=body.get("instructions"),
            assignment_type=str(body.get("assignment_type") or body.get("assignmentType") or "text"),
            due_date=body.get("due_date") or body.get("dueDate"),
            available_from=body.get("available_from") or body.get("availableFrom"),
            max_points=body.get("max_points") or body.get("maxPoints"),
            passing_score=body.get("passing_score") or body.get("passingScore"),
            max_attempts=body.get("max_attempts") or body.get("maxAttempts"),
            allow_late_submissions=body.get("allow_late_submissions") if "allow_late_submissions" in body else body.get("allowLateSubmissions"),
            late_penalty_percent=body.get("late_penalty_percent") or body.get("latePenaltyPercent"),
            max_late_days=body.get("max_late_days") or body.get("maxLateDays"),
            allowed_file_types=body.get("allowed_file_types") or body.get("allowedFileTypes"),
            max_file_size_mb=body.get("max_file_size_mb") or body.get("maxFileSizeMb"),
            max_files=body.get("max_files") or body.get("maxFiles"),
            peer_review_enabled=body.get("peer_review_enabled") if "peer_review_enabled" in body else body.get("peerReviewEnabled"),
            peer_reviews_required=body.get("peer_reviews_required") or body.get("peerReviewsRequired"),
            is_published=body.get("is_published") if "is_published" in body else body.get("isPublished"),
            quiz_questions=body.get("quiz_questions") or body.get("quizQuestions"),
            resource_file_name=body.get("resource_file_name") or body.get("resourceFileName"),
            resource_attachment=body.get("resource_attachment") or body.get("resourceAttachment"),
        )
        _save_academy_state(state)
        return JSONResponse(_academy_assignment_payload(state, assignment), status_code=201)

    if parts[0] == "courses" and len(parts) >= 3 and parts[2] == "submissions" and method == "GET":
        course_id = parts[1]
        assignment_type_filter = str(query.get("assignment_type") or "all").strip().lower()
        active_user_ids = {
            str(item.get("user_id"))
            for item in state["enrollments"]
            if item.get("course_id") == course_id and item.get("status") != "dropped"
        }
        if not active_user_ids:
            return JSONResponse([])

        latest_rows: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        for submission in state["submissions"]:
            if submission.get("status") == "draft":
                continue

            assignment = _academy_find_assignment(state, str(submission.get("assignment_id") or ""))
            if not assignment or assignment.get("course_id") != course_id:
                continue

            if active_user_ids and str(submission.get("user_id")) not in active_user_ids:
                continue

            normalized_assignment_type = "quiz" if assignment.get("assignment_type") == "quiz" else "assignment"
            if assignment_type_filter in {"quiz", "assignment"} and normalized_assignment_type != assignment_type_filter:
                continue

            row_key = (str(assignment.get("id") or ""), str(submission.get("user_id") or ""))
            current = latest_rows.get(row_key)
            if current is None or _academy_submission_sort_key(submission) > _academy_submission_sort_key(current[0]):
                latest_rows[row_key] = (submission, assignment)

        rows = []
        for submission, assignment in latest_rows.values():
            score_value = submission.get("adjusted_score")
            if score_value is None:
                score_value = submission.get("score")

            display_name = _academy_display_name_for_submission(submission)
            if display_name == "Academy learner" or display_name.startswith("Learner "):
                display_name = _academy_display_name_for_user(
                    state,
                    str(submission.get("user_id") or ""),
                    course_id=course_id,
                )

            rows.append(
                {
                    "submission_id": submission.get("id"),
                    "assignment_id": assignment.get("id"),
                    "assignment_title": assignment.get("title"),
                    "assignment_type": "quiz" if assignment.get("assignment_type") == "quiz" else "assignment",
                    "course_id": assignment.get("course_id"),
                    "course_title": assignment.get("course_title") or "Street Voices Academy Course",
                    "user_id": submission.get("user_id"),
                    "user_name": display_name,
                    "user_email": submission.get("user_email"),
                    "attempt_number": submission.get("attempt_number") or 1,
                    "status": submission.get("status"),
                    "submitted_at": submission.get("submitted_at") or submission.get("updated_at") or submission.get("created_at"),
                    "graded_at": submission.get("graded_at"),
                    "score": score_value,
                    "letter_grade": submission.get("letter_grade"),
                    "max_points": assignment.get("max_points") or 100,
                }
            )

        rows.sort(
            key=lambda row: (
                str(row.get("submitted_at") or ""),
                str(row.get("assignment_title") or ""),
                str(row.get("user_name") or ""),
            ),
            reverse=True,
        )
        return JSONResponse(rows)

    if parts[0] == "users" and len(parts) >= 3 and method == "GET":
        user_id = parts[1]
        action = parts[2]
        course_id = query.get("course_id")
        if action in {"available-assignments", "assignment-stats"} and not course_id:
            return JSONResponse({"detail": "course_id is required"}, status_code=400)

        if action == "available-assignments":
            course_meta = await _academy_fetch_course_meta(course_id or "")
            assignments = _academy_seed_assignments_for_course(
                state,
                course_id or "",
                course_title=course_meta.get("title") if course_meta else None,
                instructor_id=course_meta.get("instructor_id") if course_meta else None,
            )
            _save_academy_state(state)
            return JSONResponse([_academy_assignment_payload(state, row) for row in assignments if row.get("is_published", True)])

        if action == "assignment-stats":
            course_meta = await _academy_fetch_course_meta(course_id or "")
            assignments = _academy_seed_assignments_for_course(
                state,
                course_id or "",
                course_title=course_meta.get("title") if course_meta else None,
                instructor_id=course_meta.get("instructor_id") if course_meta else None,
            )
            assignment_ids = {row.get("id") for row in assignments}
            submissions = [
                row
                for row in state["submissions"]
                if row.get("user_id") == user_id and row.get("assignment_id") in assignment_ids
            ]
            submitted_rows = [row for row in submissions if row.get("status") != "draft"]
            graded_rows = [row for row in submissions if row.get("status") in {"graded", "returned"}]
            average_score = (
                round(
                    sum(float(row.get("adjusted_score", row.get("score")) or 0) for row in graded_rows) / len(graded_rows),
                    2,
                )
                if graded_rows
                else 0
            )
            return JSONResponse(
                {
                    "total_assignments": len(assignments),
                    "submitted": len(submitted_rows),
                    "graded": len(graded_rows),
                    "average_score": average_score,
                    "on_time": len([row for row in submitted_rows if not row.get("is_late")]),
                    "late": len([row for row in submitted_rows if row.get("is_late")]),
                }
            )

    if parts[0] == "assignments":
        assignment = _academy_find_assignment(state, parts[1] if len(parts) >= 2 else "")
        if not assignment:
            return JSONResponse({"detail": "Assignment not found"}, status_code=404)

        if len(parts) == 2 and method == "GET":
            return JSONResponse(_academy_assignment_payload(state, assignment))

        if len(parts) == 2 and method == "DELETE":
            assignment_id = assignment.get("id")
            removed_submission_ids = {
                row.get("id")
                for row in state["submissions"]
                if row.get("assignment_id") == assignment_id
            }
            state["assignments"] = [row for row in state["assignments"] if row.get("id") != assignment_id]
            state["submissions"] = [row for row in state["submissions"] if row.get("assignment_id") != assignment_id]
            state["rubric_grades"] = [
                row for row in state["rubric_grades"] if row.get("submission_id") not in removed_submission_ids
            ]
            _save_academy_state(state)
            return JSONResponse({"ok": True})

        if len(parts) >= 3 and parts[2] == "stats" and method == "GET":
            return JSONResponse(_academy_assignment_payload(state, assignment))

        if len(parts) >= 3 and parts[2] == "availability" and method == "GET":
            return JSONResponse(
                {
                    "is_available": True,
                    "reason": None,
                    "available_from": assignment.get("available_from"),
                    "available_until": assignment.get("available_until"),
                    "due_date": assignment.get("due_date"),
                    "attempts_remaining": assignment.get("max_attempts"),
                    "max_attempts": assignment.get("max_attempts"),
                    "current_attempt": 0,
                }
            )

        if len(parts) >= 3 and parts[2] == "submissions" and method == "POST":
            user_id = query.get("user_id")
            if not user_id:
                return JSONResponse({"detail": "user_id is required"}, status_code=400)
            submission = _academy_create_submission(state, assignment, user_id, status="draft")
            _save_academy_state(state)
            return JSONResponse(submission, status_code=201)

        if len(parts) >= 3 and parts[2] == "my-submission" and method == "GET":
            user_id = query.get("user_id")
            if not user_id:
                return JSONResponse({"detail": "user_id is required"}, status_code=400)
            submission = _academy_get_latest_submission(state, assignment["id"], user_id)
            if submission is None:
                return JSONResponse({"detail": "Submission not found"}, status_code=404)
            return JSONResponse(submission)

    if parts[0] == "submissions":
        submission = _academy_find(state["submissions"], parts[1] if len(parts) >= 2 else "")
        if not submission:
            return JSONResponse({"detail": "Submission not found"}, status_code=404)

        if len(parts) == 2 and method == "GET":
            return JSONResponse(submission)

        if len(parts) == 2 and method in {"PATCH", "PUT"}:
            body = await request.json()
            text_content = body.get("text_content")
            if text_content is not None:
                submission["text_content"] = text_content
                submission["word_count"] = len(str(text_content).split())
            if "quiz_answers" in body:
                submission["quiz_answers"] = body.get("quiz_answers") or []
                if not submission.get("text_content"):
                    summary_text = _academy_quiz_answers_to_text(submission.get("quiz_answers"))
                    submission["text_content"] = summary_text
                    submission["word_count"] = len(summary_text.split())
            if "document_id" in body:
                submission["document_id"] = body.get("document_id")
            if "file_urls" in body:
                submission["file_urls"] = body.get("file_urls") or []
            submission["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(submission)

        if len(parts) >= 3 and parts[2] == "submit" and method == "POST":
            submission["status"] = "submitted"
            submission["submitted_at"] = submission.get("submitted_at") or _academy_now_iso()
            submission["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(submission)

        if len(parts) >= 3 and parts[2] == "request-regrade" and method == "POST":
            body = await request.json()
            submission["status"] = "regrade_requested"
            submission["regrade_reason"] = body.get("reason")
            submission["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(submission)

        if len(parts) >= 3 and parts[2] == "start-grading" and method == "POST":
            submission["status"] = "grading"
            submission["grading_locked_by"] = query.get("grader_id")
            submission["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(submission)

        if len(parts) >= 3 and parts[2] == "grade" and method == "POST":
            body = await request.json()
            score = float(body.get("score") or 0)
            assignment = _academy_find_assignment(state, submission.get("assignment_id", ""))
            max_points = float(assignment.get("max_points") or 100) if assignment else 100
            percentage = score / max_points * 100 if max_points else 0
            letter_grade = (
                "A" if percentage >= 90 else
                "B" if percentage >= 80 else
                "C" if percentage >= 70 else
                "D" if percentage >= 60 else
                "F"
            )
            submission.update(
                {
                    "status": "graded",
                    "grading_locked_by": query.get("grader_id"),
                    "graded_by": query.get("grader_id"),
                    "graded_at": _academy_now_iso(),
                    "score": score,
                    "adjusted_score": score,
                    "letter_grade": letter_grade,
                    "feedback": body.get("feedback"),
                    "feedback_attachments": body.get("feedback_attachments") or [],
                    "updated_at": _academy_now_iso(),
                }
            )
            _save_academy_state(state)
            return JSONResponse(submission)

        if len(parts) >= 3 and parts[2] == "rubric-grade" and method == "POST":
            body = await request.json()
            criterion_grades = body.get("criterion_grades") or []
            earned_points = round(sum(float(item.get("points_earned") or 0) for item in criterion_grades), 2)
            rubric_id = body.get("rubric_id")
            summary = {
                "id": _academy_make_id("rubric-grade"),
                "submission_id": submission.get("id"),
                "rubric_id": rubric_id,
                "criterion_grades": criterion_grades,
                "overall_feedback": body.get("overall_feedback"),
                "earned_points": earned_points,
                "updated_at": _academy_now_iso(),
            }
            state["rubric_grades"] = [
                row
                for row in state["rubric_grades"]
                if not (row.get("submission_id") == submission.get("id") and row.get("rubric_id") == rubric_id)
            ]
            state["rubric_grades"].append(summary)
            submission["status"] = "graded"
            submission["graded_by"] = query.get("grader_id")
            submission["graded_at"] = _academy_now_iso()
            submission["score"] = earned_points
            submission["adjusted_score"] = earned_points
            submission["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            rubric = next((row for row in state["rubrics"] if row.get("id") == rubric_id), None)
            total_points = float(rubric.get("total_points") or 100) if rubric else 100
            return JSONResponse(
                {
                    "rubric_id": rubric_id,
                    "rubric_title": rubric.get("title") if rubric else "Course Rubric",
                    "total_points": total_points,
                    "earned_points": earned_points,
                    "percentage": round(earned_points / total_points * 100, 2) if total_points else 0,
                    "criterion_grades": [
                        {
                            "criterion_id": item.get("criterion_id"),
                            "criterion_name": item.get("criterion_name"),
                            "max_points": item.get("max_points"),
                            "level_id": item.get("level_id"),
                            "level_name": item.get("level_name"),
                            "points_earned": item.get("points_earned"),
                            "feedback": item.get("feedback"),
                        }
                        for item in criterion_grades
                    ],
                }
            )

        if len(parts) >= 3 and parts[2] == "rubric-grades" and method == "GET":
            rubric_id = query.get("rubric_id")
            summary = next(
                (
                    row
                    for row in state["rubric_grades"]
                    if row.get("submission_id") == submission.get("id") and row.get("rubric_id") == rubric_id
                ),
                None,
            )
            if summary is None:
                return JSONResponse({"detail": "Rubric grades not found"}, status_code=404)
            rubric = next((row for row in state["rubrics"] if row.get("id") == rubric_id), None)
            total_points = float(rubric.get("total_points") or 100) if rubric else 100
            return JSONResponse(
                {
                    "rubric_id": rubric_id,
                    "rubric_title": rubric.get("title") if rubric else "Course Rubric",
                    "total_points": total_points,
                    "earned_points": summary.get("earned_points"),
                    "percentage": round(float(summary.get("earned_points") or 0) / total_points * 100, 2) if total_points else 0,
                    "criterion_grades": summary.get("criterion_grades") or [],
                }
            )

        if len(parts) >= 3 and parts[2] == "return" and method == "POST":
            submission["status"] = "returned"
            submission["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(submission)

    if parts[0] == "rubrics" and len(parts) >= 2 and method == "GET":
        rubric = next((row for row in state["rubrics"] if row.get("id") == parts[1]), None)
        if rubric is None:
            return JSONResponse({"detail": "Rubric not found"}, status_code=404)
        return JSONResponse(rubric)

    if parts[0] == "grading" and len(parts) >= 2 and parts[1] == "queue" and method == "GET":
        course_id = query.get("course_id")
        queue_rows = []
        for submission in state["submissions"]:
            if submission.get("status") not in {"submitted", "grading", "regrade_requested"}:
                continue
            assignment = _academy_find_assignment(state, submission.get("assignment_id", ""))
            if not assignment:
                continue
            if course_id and assignment.get("course_id") != course_id:
                continue
            queue_rows.append(
                {
                    "submission_id": submission.get("id"),
                    "assignment_id": assignment.get("id"),
                    "assignment_title": assignment.get("title"),
                    "course_id": assignment.get("course_id"),
                    "course_title": assignment.get("course_title") or "Street Voices Academy Course",
                    "user_id": submission.get("user_id"),
                    "user_name": submission.get("user_name") or "Academy learner",
                    "user_email": submission.get("user_email"),
                    "attempt_number": submission.get("attempt_number") or 1,
                    "submitted_at": submission.get("submitted_at") or submission.get("updated_at") or submission.get("created_at"),
                    "is_late": bool(submission.get("is_late")),
                    "days_late": int(submission.get("days_late") or 0),
                    "status": submission.get("status"),
                    "due_date": assignment.get("due_date"),
                    "max_points": assignment.get("max_points") or 100,
                    "rubric_id": assignment.get("rubric_id"),
                    "grading_locked_by": submission.get("grading_locked_by"),
                }
            )
        queue_rows.sort(key=lambda row: row.get("submitted_at") or "", reverse=False)
        return JSONResponse(queue_rows)

    return JSONResponse({"detail": "Unsupported assignment route"}, status_code=404)


async def _handle_live_sessions(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method
    query = request.query_params

    if len(parts) == 1 or not parts[1]:
        if method == "GET":
            sessions = list(state["live_sessions"])
            if not sessions and not query.get("course_id") and not query.get("instructor_id"):
                sample_session = {
                    "id": _academy_make_id("session"),
                    "course_id": "academy-foundations",
                    "module_id": None,
                    "lesson_id": None,
                    "title": "Street Voices Academy Orientation",
                    "description": "Kick off the Academy experience, meet the learning flow, and preview live learning.",
                    "session_type": "webinar",
                    "instructor_id": "academy-instructor",
                    "co_host_ids": [],
                    "scheduled_start": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                    "scheduled_end": (datetime.now(timezone.utc) + timedelta(days=1, hours=1)).isoformat().replace("+00:00", "Z"),
                    "actual_start": None,
                    "actual_end": None,
                    "status": "scheduled",
                    "max_attendees": 50,
                    "platform": "internal",
                    "meeting_id": None,
                    "meeting_url": None,
                    "session_notes": None,
                    "recording_url": None,
                    "recording_available": False,
                    "is_mandatory": False,
                    "points_for_attending": 5,
                    "created_at": _academy_now_iso(),
                    "updated_at": _academy_now_iso(),
                }
                state["live_sessions"].append(sample_session)
                _save_academy_state(state)
                sessions = [sample_session]
            course_id = query.get("course_id")
            instructor_id = query.get("instructor_id")
            status = query.get("status")
            if course_id:
                sessions = [s for s in sessions if s.get("course_id") == course_id]
            if instructor_id:
                sessions = [s for s in sessions if s.get("instructor_id") == instructor_id]
            if status:
                sessions = [s for s in sessions if s.get("status") == status]
            else:
                sessions = [s for s in sessions if s.get("status") not in {"cancelled"}]
            sessions.sort(key=lambda item: item.get("scheduled_start", ""))
            return JSONResponse(_academy_session_list_response(sessions))

        if method == "POST":
            body = await request.json()
            scheduled_start = body.get("scheduled_start") or _academy_now_iso()
            scheduled_end = body.get("scheduled_end") or (
                _academy_parse_iso(scheduled_start) + timedelta(hours=1)
            ).isoformat().replace("+00:00", "Z")
            session = {
                "id": _academy_make_id("session"),
                "course_id": body.get("course_id"),
                "module_id": body.get("module_id"),
                "lesson_id": body.get("lesson_id"),
                "title": body.get("title") or "Untitled Session",
                "description": body.get("description"),
                "session_type": body.get("session_type") or "class",
                "instructor_id": query.get("instructor_id") or body.get("instructor_id") or "academy-instructor",
                "co_host_ids": body.get("co_host_ids") or [],
                "scheduled_start": scheduled_start,
                "scheduled_end": scheduled_end,
                "actual_start": None,
                "actual_end": None,
                "status": body.get("status") or "scheduled",
                "max_attendees": body.get("max_attendees"),
                "platform": body.get("platform") or "internal",
                "meeting_id": body.get("meeting_id"),
                "meeting_url": body.get("meeting_url"),
                "session_notes": body.get("session_notes"),
                "recording_url": None,
                "recording_available": False,
                "is_mandatory": bool(body.get("is_mandatory", False)),
                "points_for_attending": int(body.get("points_for_attending") or 0),
                "created_at": _academy_now_iso(),
                "updated_at": _academy_now_iso(),
            }
            state["live_sessions"].append(session)
            _save_academy_state(state)
            return JSONResponse(session, status_code=201)

    if len(parts) >= 3 and parts[1] == "course" and method == "GET":
        course_id = parts[2]
        sessions = [s for s in state["live_sessions"] if s.get("course_id") == course_id]
        if not sessions:
            sample_session = {
                "id": _academy_make_id("session"),
                "course_id": course_id,
                "module_id": None,
                "lesson_id": None,
                "title": "Academy Live Lab",
                "description": "A guided Academy session for discussion, check-ins, and course support.",
                "session_type": "class",
                "instructor_id": "academy-instructor",
                "co_host_ids": [],
                "scheduled_start": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat().replace("+00:00", "Z"),
                "scheduled_end": (datetime.now(timezone.utc) + timedelta(days=2, hours=1)).isoformat().replace("+00:00", "Z"),
                "actual_start": None,
                "actual_end": None,
                "status": "scheduled",
                "max_attendees": 25,
                "platform": "internal",
                "meeting_id": None,
                "meeting_url": None,
                "session_notes": None,
                "recording_url": None,
                "recording_available": False,
                "is_mandatory": False,
                "points_for_attending": 10,
                "created_at": _academy_now_iso(),
                "updated_at": _academy_now_iso(),
            }
            state["live_sessions"].append(sample_session)
            _save_academy_state(state)
            sessions = [sample_session]
        status = query.get("status")
        if status:
            sessions = [s for s in sessions if s.get("status") == status]
        else:
            sessions = [s for s in sessions if s.get("status") not in {"cancelled"}]
        if query.get("upcoming_only") == "true":
            now = datetime.now(timezone.utc)
            sessions = [s for s in sessions if _academy_parse_iso(s.get("scheduled_end")) >= now]
        sessions.sort(key=lambda item: item.get("scheduled_start", ""))
        return JSONResponse(_academy_session_list_response(sessions))

    if len(parts) >= 4 and parts[1] == "user" and method == "GET":
        user_id = parts[2]
        scope = parts[3]
        rows = []
        now = datetime.now(timezone.utc)
        for registration in state["session_registrations"]:
            if registration.get("user_id") != user_id:
                continue
            session = _academy_find(state["live_sessions"], registration.get("session_id", ""))
            if not session:
                continue
            if scope == "upcoming" and _academy_parse_iso(session.get("scheduled_end")) < now:
                continue
            rows.append({"registration": registration, "session": session})
        rows.sort(key=lambda item: item["session"].get("scheduled_start", ""))
        return JSONResponse({"sessions": rows, "total": len(rows)})

    if len(parts) >= 4 and parts[1] == "polls":
        poll = _academy_find(state["session_polls"], parts[2])
        if not poll:
            return JSONResponse({"detail": "Poll not found"}, status_code=404)
        if parts[3] == "start" and method == "POST":
            poll["status"] = "active"
            poll["started_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(poll)
        if parts[3] == "end" and method == "POST":
            poll["status"] = "closed"
            poll["ended_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(poll)
        if parts[3] == "respond" and method == "POST":
            user_id = query.get("user_id")
            body = await request.json()
            response = next(
                (
                    item
                    for item in state["poll_responses"]
                    if item.get("poll_id") == poll["id"] and item.get("user_id") == user_id
                ),
                None,
            )
            if response is None:
                response = {
                    "id": _academy_make_id("poll-response"),
                    "poll_id": poll["id"],
                    "user_id": user_id,
                    "response": body.get("response"),
                    "created_at": _academy_now_iso(),
                }
                state["poll_responses"].append(response)
            else:
                response["response"] = body.get("response")
            _save_academy_state(state)
            return JSONResponse({"ok": True})
        if parts[3] == "results" and method == "GET":
            responses = [item for item in state["poll_responses"] if item.get("poll_id") == poll["id"]]
            results: dict[str, int] = {}
            for response in responses:
                value = response.get("response")
                if isinstance(value, list):
                    for item in value:
                        results[str(item)] = results.get(str(item), 0) + 1
                else:
                    results[str(value)] = results.get(str(value), 0) + 1
            return JSONResponse(
                {
                    "poll": poll,
                    "total_responses": len(responses),
                    "results": results,
                }
            )

    if len(parts) >= 4 and parts[1] == "questions":
        question = _academy_find(state["session_questions"], parts[2])
        if not question:
            return JSONResponse({"detail": "Question not found"}, status_code=404)
        if parts[3] == "upvote" and method == "POST":
            question["upvotes"] = int(question.get("upvotes") or 0) + 1
            _save_academy_state(state)
            return JSONResponse({"ok": True})
        if parts[3] == "answer" and method == "POST":
            body = await request.json()
            question["answer"] = body.get("answer")
            question["answered_by"] = query.get("answered_by")
            question["status"] = "answered"
            _save_academy_state(state)
            return JSONResponse(question)

    session_id = parts[1] if len(parts) >= 2 else None
    session = _academy_find(state["live_sessions"], session_id or "")
    if not session:
        return JSONResponse({"detail": "Session not found"}, status_code=404)

    if len(parts) == 2 and method == "GET":
        return JSONResponse(_academy_session_with_registration(state, session, query.get("user_id")))

    if len(parts) == 2 and method in ("PATCH", "PUT"):
        body = await request.json()
        for key in [
            "title",
            "description",
            "scheduled_start",
            "scheduled_end",
            "max_attendees",
            "meeting_id",
            "meeting_url",
            "session_notes",
            "status",
            "recording_url",
        ]:
            if key in body:
                session[key] = body.get(key)
        session["updated_at"] = _academy_now_iso()
        _save_academy_state(state)
        return JSONResponse(session)

    if len(parts) >= 3:
        action = parts[2]
        if action == "start" and method == "POST":
            session["status"] = "live"
            session["actual_start"] = session.get("actual_start") or _academy_now_iso()
            session["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(session)
        if action == "end" and method == "POST":
            session["status"] = "ended"
            session["actual_end"] = _academy_now_iso()
            session["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse(session)
        if action == "cancel" and method == "POST":
            session["status"] = "cancelled"
            session["updated_at"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse({"ok": True})
        if action == "register":
            user_id = query.get("user_id")
            registration = next(
                (
                    item
                    for item in state["session_registrations"]
                    if item.get("session_id") == session["id"] and item.get("user_id") == user_id
                ),
                None,
            )
            if method == "DELETE":
                state["session_registrations"] = [
                    item
                    for item in state["session_registrations"]
                    if not (item.get("session_id") == session["id"] and item.get("user_id") == user_id)
                ]
                _save_academy_state(state)
                return JSONResponse({"ok": True})
            if registration is None:
                registration = {
                    "id": _academy_make_id("registration"),
                    "session_id": session["id"],
                    "user_id": user_id,
                    "status": "registered",
                    "joined_at": None,
                    "left_at": None,
                    "attendance_duration": 0,
                    "attendance_percent": 0,
                    "attended_full": False,
                    "points_earned": 0,
                }
                state["session_registrations"].append(registration)
            _save_academy_state(state)
            return JSONResponse(registration)
        if action == "registrations" and method == "GET":
            rows = [item for item in state["session_registrations"] if item.get("session_id") == session["id"]]
            return JSONResponse(rows)
        if action == "join" and method == "POST":
            user_id = query.get("user_id")
            registration = next(
                (
                    item
                    for item in state["session_registrations"]
                    if item.get("session_id") == session["id"] and item.get("user_id") == user_id
                ),
                None,
            )
            if registration is None:
                registration = {
                    "id": _academy_make_id("registration"),
                    "session_id": session["id"],
                    "user_id": user_id,
                    "status": "registered",
                    "joined_at": None,
                    "left_at": None,
                    "attendance_duration": 0,
                    "attendance_percent": 0,
                    "attended_full": False,
                    "points_earned": 0,
                }
                state["session_registrations"].append(registration)
            registration["joined_at"] = _academy_now_iso()
            registration["status"] = "attended"
            _save_academy_state(state)
            return JSONResponse({"status": "joined", "meeting_url": session.get("meeting_url"), "registration": registration})
        if action == "leave" and method == "POST":
            user_id = query.get("user_id")
            registration = next(
                (
                    item
                    for item in state["session_registrations"]
                    if item.get("session_id") == session["id"] and item.get("user_id") == user_id
                ),
                None,
            )
            if registration is None:
                return JSONResponse({"detail": "Registration not found"}, status_code=404)
            left_at = _academy_now_iso()
            registration["left_at"] = left_at
            duration_minutes = max(
                0,
                int(
                    (
                        _academy_parse_iso(left_at) - _academy_parse_iso(registration.get("joined_at"))
                    ).total_seconds()
                    // 60
                ),
            )
            scheduled_minutes = max(
                1,
                int(
                    (
                        _academy_parse_iso(session.get("scheduled_end")) - _academy_parse_iso(session.get("scheduled_start"))
                    ).total_seconds()
                    // 60
                ),
            )
            registration["attendance_duration"] = duration_minutes
            registration["attendance_percent"] = min(100, round(duration_minutes / scheduled_minutes * 100))
            registration["attended_full"] = registration["attendance_percent"] >= 80
            registration["points_earned"] = session.get("points_for_attending", 0) if registration["attended_full"] else 0
            _save_academy_state(state)
            return JSONResponse({"status": "left", "attendance_duration": duration_minutes})
        if action == "polls" and method == "POST":
            body = await request.json()
            poll = {
                "id": _academy_make_id("poll"),
                "session_id": session["id"],
                "question": body.get("question"),
                "poll_type": body.get("poll_type") or "single",
                "options": body.get("options") or [],
                "is_anonymous": bool(body.get("is_anonymous", True)),
                "show_results_live": bool(body.get("show_results_live", True)),
                "status": "draft",
                "started_at": None,
                "ended_at": None,
            }
            state["session_polls"].append(poll)
            _save_academy_state(state)
            return JSONResponse(poll, status_code=201)
        if action == "questions":
            if method == "GET":
                rows = [item for item in state["session_questions"] if item.get("session_id") == session["id"]]
                status = query.get("status")
                if status:
                    rows = [item for item in rows if item.get("status") == status]
                rows.sort(key=lambda item: item.get("created_at", ""), reverse=True)
                return JSONResponse(rows)
            if method == "POST":
                body = await request.json()
                question = {
                    "id": _academy_make_id("question"),
                    "session_id": session["id"],
                    "user_id": query.get("user_id"),
                    "question": body.get("question"),
                    "is_anonymous": bool(body.get("is_anonymous", False)),
                    "status": "pending",
                    "upvotes": 0,
                    "answer": None,
                    "answered_by": None,
                    "created_at": _academy_now_iso(),
                }
                state["session_questions"].append(question)
                _save_academy_state(state)
                return JSONResponse(question, status_code=201)
        if action == "feedback":
            if len(parts) >= 4 and parts[3] == "summary" and method == "GET":
                rows = [item for item in state["session_feedback"] if item.get("session_id") == session["id"]]
                total = len(rows)
                def _avg(key: str) -> float | None:
                    values = [float(item[key]) for item in rows if item.get(key) is not None]
                    return round(sum(values) / len(values), 2) if values else None
                recommend_count = len([item for item in rows if item.get("would_recommend")])
                return JSONResponse(
                    {
                        "total_responses": total,
                        "average_overall": _avg("overall_rating"),
                        "average_content": _avg("content_rating"),
                        "average_presenter": _avg("presenter_rating"),
                        "average_tech": _avg("tech_rating"),
                        "recommend_percent": round(recommend_count / total * 100, 2) if total else 0,
                    }
                )
            if method == "POST":
                body = await request.json()
                existing = next(
                    (
                        item
                        for item in state["session_feedback"]
                        if item.get("session_id") == session["id"] and item.get("user_id") == query.get("user_id")
                    ),
                    None,
                )
                feedback = {
                    "id": existing.get("id") if existing else _academy_make_id("feedback"),
                    "session_id": session["id"],
                    "user_id": query.get("user_id"),
                    **body,
                    "updated_at": _academy_now_iso(),
                }
                state["session_feedback"] = [
                    item
                    for item in state["session_feedback"]
                    if not (item.get("session_id") == session["id"] and item.get("user_id") == query.get("user_id"))
                ]
                state["session_feedback"].append(feedback)
                _save_academy_state(state)
                return JSONResponse({"ok": True})

    return JSONResponse({"detail": "Unsupported live session route"}, status_code=404)


async def _handle_cohorts(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method
    query = request.query_params

    if len(parts) == 1 or not parts[1]:
        if method == "GET":
            rows = list(state["cohorts"])
            if not rows and not query.get("instructor_id"):
                sample_cohort = {
                    "id": _academy_make_id("cohort"),
                    "course_id": "academy-foundations",
                    "name": "Academy Launch Cohort",
                    "description": "A sample Academy cohort for the foundational LMS release.",
                    "start_date": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                    "end_date": (datetime.now(timezone.utc) + timedelta(days=29)).isoformat().replace("+00:00", "Z"),
                    "max_capacity": 25,
                    "current_enrollment": 0,
                    "status": "upcoming",
                    "is_self_paced": False,
                    "enrollment_deadline": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat().replace("+00:00", "Z"),
                    "instructor_id": "academy-instructor",
                    "created_at": _academy_now_iso(),
                    "updated_at": _academy_now_iso(),
                }
                state["cohorts"].append(sample_cohort)
                _save_academy_state(state)
                rows = [sample_cohort]
            instructor_id = query.get("instructor_id")
            if instructor_id:
                rows = [item for item in rows if item.get("instructor_id") == instructor_id]
            return JSONResponse(rows)
        if method == "POST":
            body = await request.json()
            cohort = {
                "id": _academy_make_id("cohort"),
                "course_id": body.get("course_id"),
                "name": body.get("name") or "Academy Cohort",
                "description": body.get("description"),
                "start_date": body.get("start_date") or _academy_now_iso(),
                "end_date": body.get("end_date") or (_academy_parse_iso(_academy_now_iso()) + timedelta(days=30)).isoformat().replace("+00:00", "Z"),
                "max_capacity": body.get("max_capacity"),
                "current_enrollment": 0,
                "status": body.get("status") or "upcoming",
                "is_self_paced": bool(body.get("is_self_paced", False)),
                "enrollment_deadline": body.get("enrollment_deadline"),
                "instructor_id": query.get("instructor_id") or body.get("instructor_id") or "academy-instructor",
                "created_at": _academy_now_iso(),
                "updated_at": _academy_now_iso(),
            }
            state["cohorts"].append(cohort)
            _save_academy_state(state)
            return JSONResponse(cohort, status_code=201)

    if len(parts) >= 3 and parts[1] == "course" and method == "GET":
        course_id = parts[2]
        rows = [item for item in state["cohorts"] if item.get("course_id") == course_id]
        if not rows:
            sample_cohort = {
                "id": _academy_make_id("cohort"),
                "course_id": course_id,
                "name": "Spring Academy Cohort",
                "description": "A paced Academy cohort with accountability, deadlines, and live touchpoints.",
                "start_date": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                "end_date": (datetime.now(timezone.utc) + timedelta(days=31)).isoformat().replace("+00:00", "Z"),
                "max_capacity": 30,
                "current_enrollment": 0,
                "status": "upcoming",
                "is_self_paced": False,
                "enrollment_deadline": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat().replace("+00:00", "Z"),
                "instructor_id": "academy-instructor",
                "created_at": _academy_now_iso(),
                "updated_at": _academy_now_iso(),
            }
            state["cohorts"].append(sample_cohort)
            state["cohort_deadlines"].append(
                {
                    "id": _academy_make_id("deadline"),
                    "cohort_id": sample_cohort["id"],
                    "module_id": None,
                    "lesson_id": None,
                    "assignment_id": None,
                    "deadline": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat().replace("+00:00", "Z"),
                    "description": "First milestone check-in",
                    "created_at": _academy_now_iso(),
                }
            )
            _save_academy_state(state)
            rows = [sample_cohort]
        if query.get("include_past") != "true":
            now = datetime.now(timezone.utc)
            rows = [item for item in rows if _academy_parse_iso(item.get("end_date")) >= now]
        rows.sort(key=lambda item: item.get("start_date", ""))
        return JSONResponse(rows)

    if len(parts) >= 3 and parts[1] == "user" and method == "GET":
        user_id = parts[2]
        if len(parts) >= 4 and parts[3] == "deadlines":
            cohort_ids = {
                item.get("cohort_id")
                for item in state["cohort_enrollments"]
                if item.get("user_id") == user_id
            }
            rows = [item for item in state["cohort_deadlines"] if item.get("cohort_id") in cohort_ids]
            return JSONResponse(rows)
        cohort_ids = {
            item.get("cohort_id")
            for item in state["cohort_enrollments"]
            if item.get("user_id") == user_id
        }
        rows = [item for item in state["cohorts"] if item.get("id") in cohort_ids]
        return JSONResponse(rows)

    cohort_id = parts[1] if len(parts) >= 2 else None
    cohort = _academy_find(state["cohorts"], cohort_id or "")
    if not cohort:
        return JSONResponse({"detail": "Cohort not found"}, status_code=404)

    if len(parts) == 2 and method == "GET":
        return JSONResponse(cohort)
    if len(parts) == 2 and method in ("PUT", "PATCH"):
        body = await request.json()
        for key in [
            "name",
            "description",
            "start_date",
            "end_date",
            "max_capacity",
            "status",
            "is_self_paced",
            "enrollment_deadline",
        ]:
            if key in body:
                cohort[key] = body.get(key)
        cohort["updated_at"] = _academy_now_iso()
        _save_academy_state(state)
        return JSONResponse(cohort)
    if len(parts) == 2 and method == "DELETE":
        state["cohorts"] = [item for item in state["cohorts"] if item.get("id") != cohort["id"]]
        _save_academy_state(state)
        return JSONResponse({"ok": True})

    action = parts[2] if len(parts) >= 3 else None
    if action == "enroll":
        user_id = query.get("user_id")
        if method == "DELETE":
            state["cohort_enrollments"] = [
                item
                for item in state["cohort_enrollments"]
                if not (item.get("cohort_id") == cohort["id"] and item.get("user_id") == user_id)
            ]
            cohort["current_enrollment"] = max(0, int(cohort.get("current_enrollment") or 0) - 1)
            _save_academy_state(state)
            return JSONResponse({"ok": True})
        enrollment = next(
            (
                item
                for item in state["cohort_enrollments"]
                if item.get("cohort_id") == cohort["id"] and item.get("user_id") == user_id
            ),
            None,
        )
        if enrollment is None:
            enrollment = {
                "id": _academy_make_id("cohort-enrollment"),
                "cohort_id": cohort["id"],
                "user_id": user_id,
                "status": "active",
                "progress_percent": 0,
                "enrolled_at": _academy_now_iso(),
                "completed_at": None,
            }
            state["cohort_enrollments"].append(enrollment)
            cohort["current_enrollment"] = int(cohort.get("current_enrollment") or 0) + 1
        course_id = str(cohort.get("course_id") or "")
        if course_id:
            _academy_ensure_course_enrollment(
                state,
                user_id=user_id,
                course_id=course_id,
                progress_percent=int(enrollment.get("progress_percent") or 0),
                status=str(enrollment.get("status") or "active"),
                enrolled_at=enrollment.get("enrolled_at"),
            )
        _save_academy_state(state)
        return JSONResponse({"success": True, "message": "Enrolled", "enrollment": enrollment})
    if action == "enrollments" and method == "GET":
        rows = [item for item in state["cohort_enrollments"] if item.get("cohort_id") == cohort["id"]]
        return JSONResponse(rows)
    if action == "deadlines":
        if method == "GET":
            rows = [item for item in state["cohort_deadlines"] if item.get("cohort_id") == cohort["id"]]
            return JSONResponse(rows)
        if method == "POST":
            body = await request.json()
            deadline = {
                "id": _academy_make_id("deadline"),
                "cohort_id": cohort["id"],
                "module_id": body.get("module_id"),
                "lesson_id": body.get("lesson_id"),
                "assignment_id": body.get("assignment_id"),
                "deadline": body.get("deadline"),
                "description": body.get("description"),
                "created_at": _academy_now_iso(),
            }
            state["cohort_deadlines"].append(deadline)
            _save_academy_state(state)
            return JSONResponse(deadline, status_code=201)
    if action == "announcements":
        if method == "GET":
            rows = [item for item in state["cohort_announcements"] if item.get("cohort_id") == cohort["id"]]
            return JSONResponse(rows[: int(query.get("limit") or 20)])
        if method == "POST":
            body = await request.json()
            announcement = {
                "id": _academy_make_id("announcement"),
                "cohort_id": cohort["id"],
                "author_id": query.get("author_id"),
                "title": body.get("title"),
                "content": body.get("content"),
                "is_pinned": bool(body.get("is_pinned", False)),
                "created_at": _academy_now_iso(),
            }
            state["cohort_announcements"].append(announcement)
            _save_academy_state(state)
            return JSONResponse(announcement, status_code=201)
    if action == "analytics" and method == "GET":
        enrollments = [item for item in state["cohort_enrollments"] if item.get("cohort_id") == cohort["id"]]
        total = len(enrollments)
        completed = len([item for item in enrollments if item.get("status") == "completed"])
        active = len([item for item in enrollments if item.get("status") == "active"])
        avg_progress = round(sum(item.get("progress_percent") or 0 for item in enrollments) / total, 2) if total else 0
        return JSONResponse(
            {
                "total_enrolled": total,
                "active_count": active,
                "completed_count": completed,
                "average_progress": avg_progress,
                "completion_rate": round(completed / total * 100, 2) if total else 0,
                "enrollment_by_date": [],
            }
        )

    return JSONResponse({"detail": "Unsupported cohort route"}, status_code=404)


async def _handle_learning_paths(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method
    query = request.query_params

    if len(parts) == 1 and method == "GET":
        rows = list(state["learning_paths"])
        created_by = query.get("created_by")
        if created_by:
            rows = [item for item in rows if item.get("created_by") == created_by]

        rows.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
        limit = query.get("limit")
        if limit:
            try:
                rows = rows[: max(0, int(limit))]
            except ValueError:
                pass
        return JSONResponse(rows)

    if len(parts) == 1 and method == "POST":
        body = await request.json()
        title = str(body.get("title") or "").strip()
        course_ids = [
            str(course_id).strip()
            for course_id in body.get("course_ids", [])
            if str(course_id).strip()
        ]
        if not title:
            return JSONResponse({"detail": "title is required"}, status_code=400)
        if not course_ids:
            return JSONResponse({"detail": "course_ids is required"}, status_code=400)

        now_iso = _academy_now_iso()
        learning_path = {
            "id": _academy_make_id("path"),
            "slug": _academy_unique_path_slug(state, title, str(body.get("slug") or "").strip() or None),
            "title": title,
            "description": str(body.get("description") or "").strip() or f"{title} is a guided Academy learning path.",
            "courses": len(course_ids),
            "hours": int(body.get("hours") or max(len(course_ids) * 8, 8)),
            "level": str(body.get("level") or "Beginner").strip() or "Beginner",
            "delivery_mode": str(body.get("delivery_mode") or "Online and In person").strip() or "Online and In person",
            "color": str(body.get("color") or "#F97316").strip() or "#F97316",
            "requirements": [str(item).strip() for item in body.get("requirements", []) if str(item).strip()],
            "what_youll_learn": [str(item).strip() for item in body.get("what_youll_learn", []) if str(item).strip()],
            "milestones": [str(item).strip() for item in body.get("milestones", []) if str(item).strip()],
            "outcomes": [str(item).strip() for item in body.get("outcomes", []) if str(item).strip()],
            "preferred_categories": [
                str(item).strip()
                for item in body.get("preferred_categories", [])
                if str(item).strip()
            ],
            "course_ids": course_ids,
            "created_by": body.get("created_by") or query.get("created_by"),
            "source": body.get("source") or "generated",
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        state["learning_paths"].append(learning_path)
        _save_academy_state(state)
        return JSONResponse(learning_path, status_code=201)

    identifier = parts[1] if len(parts) >= 2 else ""
    learning_path = _academy_find(state["learning_paths"], identifier) or next(
        (item for item in state["learning_paths"] if item.get("slug") == identifier),
        None,
    )
    if not learning_path:
        return JSONResponse({"detail": "Learning path not found"}, status_code=404)

    if method == "GET":
        return JSONResponse(learning_path)

    if method in {"PATCH", "PUT"}:
        body = await request.json()
        for key in ("title", "description", "level", "color", "source"):
            if key in body:
                learning_path[key] = body.get(key)
        if "delivery_mode" in body:
            learning_path["delivery_mode"] = body.get("delivery_mode")
        if "hours" in body:
            learning_path["hours"] = int(body.get("hours") or 0)
        if "course_ids" in body:
            course_ids = [
                str(course_id).strip()
                for course_id in body.get("course_ids", [])
                if str(course_id).strip()
            ]
            learning_path["course_ids"] = course_ids
            learning_path["courses"] = len(course_ids)
        for key in ("requirements", "what_youll_learn", "milestones", "outcomes", "preferred_categories"):
            if key in body:
                learning_path[key] = [str(item).strip() for item in body.get(key, []) if str(item).strip()]
        learning_path["updated_at"] = _academy_now_iso()
        _save_academy_state(state)
        return JSONResponse(learning_path)

    if method == "DELETE":
        state["learning_paths"] = [item for item in state["learning_paths"] if item.get("id") != learning_path["id"]]
        _save_academy_state(state)
        return JSONResponse({"ok": True})

    return JSONResponse({"detail": "Unsupported learning path route"}, status_code=404)


async def _handle_reviews(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method

    if len(parts) >= 4 and parts[1] == "course" and parts[3] == "stats" and method == "GET":
        course_id = parts[2]
        rows = [item for item in state["reviews"] if item.get("course_id") == course_id]
        count = len(rows)
        distribution = {key: 0 for key in range(1, 6)}
        for row in rows:
            distribution[int(row.get("rating") or 0)] = distribution.get(int(row.get("rating") or 0), 0) + 1
        average = round(sum((row.get("rating") or 0) for row in rows) / count, 2) if count else 0
        return JSONResponse({"average": average, "count": count, "distribution": distribution})

    if len(parts) >= 3 and parts[1] == "course" and method == "GET":
        course_id = parts[2]
        rows = [item for item in state["reviews"] if item.get("course_id") == course_id]
        rows.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return JSONResponse(rows)

    if len(parts) >= 5 and parts[1] == "user" and parts[3] == "course" and method == "GET":
        user_id = parts[2]
        course_id = parts[4]
        review = next(
            (
                item
                for item in state["reviews"]
                if item.get("user_id") == user_id and item.get("course_id") == course_id
            ),
            None,
        )
        return JSONResponse(review)

    if len(parts) == 1 and method == "POST":
        body = await request.json()
        review = {
            "id": _academy_make_id("review"),
            "user_id": body.get("user_id"),
            "course_id": body.get("course_id"),
            "rating": body.get("rating"),
            "review_text": body.get("review_text"),
            "created_at": _academy_now_iso(),
            "updated_at": _academy_now_iso(),
        }
        state["reviews"] = [
            item
            for item in state["reviews"]
            if not (item.get("user_id") == review["user_id"] and item.get("course_id") == review["course_id"])
        ]
        state["reviews"].append(review)
        _save_academy_state(state)
        return JSONResponse(review, status_code=201)

    review = _academy_find(state["reviews"], parts[1] if len(parts) >= 2 else "")
    if not review:
        return JSONResponse({"detail": "Review not found"}, status_code=404)
    if method in ("PATCH", "PUT"):
        body = await request.json()
        if "rating" in body:
            review["rating"] = body.get("rating")
        if "review_text" in body:
            review["review_text"] = body.get("review_text")
        review["updated_at"] = _academy_now_iso()
        _save_academy_state(state)
        return JSONResponse(review)
    if method == "DELETE":
        state["reviews"] = [item for item in state["reviews"] if item.get("id") != review["id"]]
        _save_academy_state(state)
        return JSONResponse({"ok": True})
    return JSONResponse({"detail": "Unsupported reviews route"}, status_code=404)


async def _handle_certificates(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    query = request.query_params
    method = request.method
    if len(parts) >= 2 and parts[1] == "check-completion" and request.method == "GET":
        user_id = query.get("user_id")
        course_id = query.get("course_id")
        is_completed = any(
            item.get("user_id") == user_id and item.get("course_id") == course_id
            for item in state["session_registrations"]
        ) or any(
            item.get("user_id") == user_id and item.get("course_id") == course_id
            for item in state["certificates"]
        ) or any(
            item.get("user_id") == user_id and item.get("course_id") == course_id
            for item in await _list_academy_enrollments(user_id=user_id, course_id=course_id)
        )
        return JSONResponse({"is_completed": is_completed})

    if len(parts) >= 2 and parts[1] == "auto-issue" and request.method == "POST":
        user_id = query.get("user_id")
        course_id = query.get("course_id")
        cert = next(
            (
                item
                for item in state["certificates"]
                if item.get("user_id") == user_id and item.get("course_id") == course_id
            ),
            None,
        )
        if cert is None:
            course_meta = await _academy_fetch_course_meta(str(course_id or ""))
            cert = {
                "id": _academy_make_id("certificate"),
                "user_id": user_id,
                "recipient_name": None,
                "course_id": course_id,
                "learning_path_id": None,
                "target_type": "course",
                "target_id": course_id,
                "target_title": course_meta.get("title") if course_meta else None,
                "certificate_title": "Certificate of Achievement",
                "issuer_name": "Street Voices Academy",
                "signature_name": "Street Voices Academy",
                "issued_by": "academy-auto-issue",
                "award_date": _academy_now_iso(),
                "certificate_url": None,
                "badge_url": None,
                "verification_code": uuid.uuid4().hex[:12].upper(),
                "issued_at": _academy_now_iso(),
                "expires_at": None,
            }
            state["certificates"].append(cert)
            _save_academy_state(state)
        return JSONResponse(cert)

    if len(parts) == 1 and method == "POST":
        body = await request.json()
        user_id = str(body.get("user_id") or "").strip()
        target_type = str(body.get("target_type") or "course").strip().lower()
        target_id = str(body.get("target_id") or body.get("course_id") or body.get("learning_path_id") or "").strip()
        if target_type not in {"course", "learning_path"}:
            target_type = "course"
        if not user_id or not target_id:
            return JSONResponse({"detail": "user_id and target_id are required"}, status_code=400)

        if target_type == "course":
            course_meta = await _academy_fetch_course_meta(target_id)
            fallback_title = course_meta.get("title") if course_meta else None
        else:
            fallback_title = next(
                (
                    item.get("title")
                    for item in state["learning_paths"]
                    if item.get("id") == target_id or item.get("slug") == target_id
                ),
                None,
            )

        existing = next(
            (
                item
                for item in state["certificates"]
                if item.get("user_id") == user_id
                and str(item.get("target_type") or ("learning_path" if item.get("learning_path_id") else "course")) == target_type
                and str(item.get("target_id") or item.get("learning_path_id") or item.get("course_id") or "") == target_id
            ),
            None,
        )

        cert = {
            "id": existing.get("id") if existing else _academy_make_id("certificate"),
            "user_id": user_id,
            "recipient_name": body.get("recipient_name"),
            "course_id": (body.get("course_id") or target_id) if target_type == "course" else None,
            "learning_path_id": (body.get("learning_path_id") or target_id) if target_type == "learning_path" else None,
            "target_type": target_type,
            "target_id": target_id,
            "target_title": body.get("target_title") or fallback_title,
            "certificate_title": body.get("certificate_title") or "Certificate of Achievement",
            "issuer_name": body.get("issuer_name") or "Street Voices Academy",
            "signature_name": body.get("signature_name") or "Street Voices Academy",
            "issued_by": body.get("issued_by"),
            "award_date": body.get("award_date") or _academy_now_iso(),
            "certificate_url": body.get("certificate_url"),
            "badge_url": body.get("badge_url"),
            "verification_code": existing.get("verification_code") if existing else uuid.uuid4().hex[:12].upper(),
            "issued_at": existing.get("issued_at") if existing else _academy_now_iso(),
            "updated_at": _academy_now_iso(),
            "expires_at": body.get("expires_at"),
        }

        state["certificates"] = [
            item
            for item in state["certificates"]
            if item.get("id") != cert["id"]
        ]
        state["certificates"].append(cert)
        _save_academy_state(state)
        return JSONResponse(cert, status_code=201 if existing is None else 200)

    if len(parts) >= 2 and request.method == "GET":
        user_id = parts[1]
        rows = [item for item in state["certificates"] if item.get("user_id") == user_id]
        rows.sort(key=lambda item: item.get("updated_at") or item.get("issued_at") or "", reverse=True)
        return JSONResponse(rows)

    return JSONResponse({"detail": "Unsupported certificate route"}, status_code=404)


async def _handle_enrollments(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    state_changed = _academy_sync_course_enrollments_from_cohorts(
        state,
        user_id=request.query_params.get("user_id"),
    )
    if state_changed:
        _save_academy_state(state)
    method = request.method
    query = request.query_params

    def _match(item: dict[str, Any]) -> bool:
        user_id = query.get("user_id")
        course_id = query.get("course_id")
        status = query.get("status")
        if user_id and item.get("user_id") != user_id:
            return False
        if course_id and item.get("course_id") != course_id:
            return False
        if status and item.get("status") != status:
            return False
        return True

    if len(parts) == 1 and method == "GET":
        rows = [item for item in state["enrollments"] if _match(item)]
        rows.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return JSONResponse(rows)

    if len(parts) == 1 and method == "POST":
        body = await request.json()
        user_id = body.get("user_id") or query.get("user_id")
        course_id = body.get("course_id") or query.get("course_id")
        if not user_id or not course_id:
            return JSONResponse({"detail": "user_id and course_id are required"}, status_code=400)

        enrollment = next(
            (
                item
                for item in state["enrollments"]
                if item.get("user_id") == user_id and item.get("course_id") == course_id
            ),
            None,
        )
        if enrollment is None:
            enrollment = {
                "id": _academy_make_id("enrollment"),
                "user_id": user_id,
                "course_id": course_id,
                "status": body.get("status") or "active",
                "progress_percent": int(body.get("progress_percent") or 0),
                "last_accessed_at": body.get("last_accessed_at"),
                "enrolled_at": _academy_now_iso(),
                "created_at": _academy_now_iso(),
                "updated_at": _academy_now_iso(),
                "completed_at": None,
            }
            state["enrollments"].append(enrollment)
        else:
            enrollment["status"] = body.get("status") or "active"
            enrollment["progress_percent"] = int(body.get("progress_percent") or enrollment.get("progress_percent") or 0)
            enrollment["last_accessed_at"] = body.get("last_accessed_at") or enrollment.get("last_accessed_at")
            enrollment["updated_at"] = _academy_now_iso()
            if enrollment.get("status") != "completed":
                enrollment["completed_at"] = None

        if int(enrollment.get("progress_percent") or 0) >= 100:
            enrollment["status"] = "completed"
            enrollment["completed_at"] = enrollment.get("completed_at") or _academy_now_iso()

        _save_academy_state(state)
        return JSONResponse(enrollment, status_code=201)

    if len(parts) >= 2 and method == "GET":
        enrollment_id = parts[1]
        enrollment = _academy_find(state["enrollments"], enrollment_id)
        if enrollment:
            return JSONResponse(enrollment)

        rows = [item for item in state["enrollments"] if item.get("user_id") == enrollment_id]
        if rows:
            rows.sort(key=lambda item: item.get("created_at", ""), reverse=True)
            return JSONResponse(rows)
        return JSONResponse({"detail": "Enrollment not found"}, status_code=404)

    if len(parts) >= 2 and method in ("PATCH", "PUT"):
        enrollment = _academy_find(state["enrollments"], parts[1])
        if not enrollment:
            return JSONResponse({"detail": "Enrollment not found"}, status_code=404)

        body = await request.json()
        for key in ["status", "last_accessed_at"]:
            if key in body:
                enrollment[key] = body.get(key)
        if "progress_percent" in body:
            enrollment["progress_percent"] = int(body.get("progress_percent") or 0)
            if enrollment["progress_percent"] >= 100:
                enrollment["status"] = "completed"
                enrollment["completed_at"] = enrollment.get("completed_at") or _academy_now_iso()

        enrollment["updated_at"] = _academy_now_iso()
        _save_academy_state(state)
        return JSONResponse(enrollment)

    if len(parts) >= 2 and method == "DELETE":
        existing = _academy_find(state["enrollments"], parts[1])
        if not existing:
            return JSONResponse({"detail": "Enrollment not found"}, status_code=404)

        now_iso = _academy_now_iso()
        affected_enrollments = [
            item
            for item in state["enrollments"]
            if item.get("user_id") == existing.get("user_id") and item.get("course_id") == existing.get("course_id")
        ]
        for enrollment in affected_enrollments:
            enrollment["status"] = "dropped"
            enrollment["updated_at"] = now_iso
            enrollment["completed_at"] = None

        removed_counts: dict[str, int] = {}
        matching_cohort_ids = {
            item.get("id")
            for item in state["cohorts"]
            if item.get("course_id") == existing.get("course_id")
        }
        remaining_cohort_enrollments = []
        for cohort_enrollment in state["cohort_enrollments"]:
            if cohort_enrollment.get("user_id") == existing.get("user_id") and cohort_enrollment.get("cohort_id") in matching_cohort_ids:
                cohort_id = str(cohort_enrollment.get("cohort_id"))
                removed_counts[cohort_id] = removed_counts.get(cohort_id, 0) + 1
                continue
            remaining_cohort_enrollments.append(cohort_enrollment)
        state["cohort_enrollments"] = remaining_cohort_enrollments

        for cohort in state["cohorts"]:
            cohort_id = str(cohort.get("id"))
            if cohort_id in removed_counts:
                cohort["current_enrollment"] = max(
                    0,
                    int(cohort.get("current_enrollment") or 0) - removed_counts[cohort_id],
                )
                cohort["updated_at"] = now_iso

        _save_academy_state(state)
        return JSONResponse({
            "ok": True,
            "dropped_enrollment_count": len(affected_enrollments),
            "removed_cohort_enrollment_count": sum(removed_counts.values()),
        })

    return JSONResponse({"detail": "Unsupported enrollment route"}, status_code=404)


async def _handle_materials(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method

    if len(parts) >= 2 and parts[1] == "types" and method == "GET":
        return JSONResponse({"types": ["syllabus", "handout", "reading", "worksheet", "reference", "supplementary"]})

    def _entity_key_and_id() -> tuple[str | None, str | None]:
        if len(parts) < 3:
            return None, None
        if parts[1] == "courses":
            return "course_id", parts[2]
        if parts[1] == "modules":
            return "module_id", parts[2]
        if parts[1] == "lessons":
            return "lesson_id", parts[2]
        return None, None

    entity_key, entity_id = _entity_key_and_id()
    if entity_key and entity_id and method == "GET":
        rows = [item for item in state["course_materials"] if item.get(entity_key) == entity_id]
        rows.sort(key=lambda item: item.get("sortOrder", 0))
        return JSONResponse(rows)
    if entity_key and entity_id and method == "POST":
        body = await request.json()
        material = {
            "id": _academy_make_id("material"),
            "linkId": _academy_make_id("material-link"),
            "documentId": body.get("documentId"),
            "title": body.get("title") or "Course document",
            "documentType": body.get("documentType") or "document",
            "status": "ready",
            "materialType": body.get("materialType") or "supplementary",
            "sortOrder": int(body.get("sortOrder") or 0),
            "wordCount": int(body.get("wordCount") or 0),
            "readingTimeMinutes": int(body.get("readingTimeMinutes") or 0),
            "authorId": request.query_params.get("createdBy"),
            "notes": body.get("notes"),
            "fileName": body.get("fileName"),
            "fileUrl": body.get("fileUrl"),
            "mimeType": body.get("mimeType"),
            "sizeBytes": body.get("sizeBytes"),
            "uploadedAt": body.get("uploadedAt"),
            "scheduleItemId": body.get("scheduleItemId"),
            "createdAt": _academy_now_iso(),
            "updatedAt": _academy_now_iso(),
            "course_id": entity_id if entity_key == "course_id" else None,
            "module_id": entity_id if entity_key == "module_id" else None,
            "lesson_id": entity_id if entity_key == "lesson_id" else None,
        }
        state["course_materials"].append(material)
        _save_academy_state(state)
        return JSONResponse({"success": True, "linkId": material["linkId"]}, status_code=201)

    if len(parts) >= 3 and parts[1] == "links":
        material = next((item for item in state["course_materials"] if item.get("linkId") == parts[2]), None)
        if not material:
            return JSONResponse({"detail": "Material link not found"}, status_code=404)
        if method == "DELETE":
            state["course_materials"] = [item for item in state["course_materials"] if item.get("linkId") != material["linkId"]]
            _save_academy_state(state)
            return JSONResponse({"success": True})
        if len(parts) >= 4 and parts[3] == "type" and method == "PATCH":
            body = await request.json()
            material["materialType"] = body.get("materialType") or material.get("materialType")
            material["updatedAt"] = _academy_now_iso()
            _save_academy_state(state)
            return JSONResponse({"success": True})

    if len(parts) >= 4 and parts[1] == "lessons" and parts[3] == "reorder" and method in ("PUT", "PATCH"):
        body = await request.json()
        order = {link_id: idx for idx, link_id in enumerate(body.get("linkIds") or [])}
        for material in state["course_materials"]:
            if material.get("lesson_id") == parts[2] and material.get("linkId") in order:
                material["sortOrder"] = order[material["linkId"]]
        _save_academy_state(state)
        return JSONResponse({"success": True})

    return JSONResponse([], status_code=200)


def _academy_schedule_category_label(category: str) -> str:
    return {
        "assignment": "Assignment",
        "reading": "Reading",
        "material": "Material",
    }.get(category, "Course Item")


def _academy_create_schedule_assignment(
    state: dict[str, list[dict[str, Any]]],
    *,
    course_id: str,
    course_title: str,
    instructor_id: str,
    title: str,
    notes: str | None,
    due_date: str,
) -> dict[str, Any]:
    assignment = {
        "id": _academy_make_id("assign"),
        "course_id": course_id,
        "course_title": course_title,
        "module_id": None,
        "lesson_id": None,
        "title": title,
        "description": notes or f"Assignment for {course_title}",
        "instructions": f"<p>{notes or 'Complete this assignment and submit your work in the student dashboard.'}</p>",
        "assignment_type": "text",
        "max_points": 100,
        "passing_score": 70,
        "due_date": due_date,
        "available_from": _academy_now_iso(),
        "available_until": None,
        "allow_late_submissions": True,
        "late_penalty_percent": 5,
        "max_late_days": 7,
        "max_attempts": 1,
        "peer_review_enabled": False,
        "peer_reviews_required": 0,
        "rubric_id": None,
        "allowed_file_types": ["pdf", "docx", "jpg", "jpeg", "png"],
        "max_file_size_mb": 15,
        "max_files": 3,
        "calendar_event_id": None,
        "is_published": True,
        "created_by": instructor_id,
        "created_at": _academy_now_iso(),
        "updated_at": _academy_now_iso(),
    }
    state["assignments"].append(assignment)
    return assignment


def _academy_create_schedule_material(
    state: dict[str, list[dict[str, Any]]],
    *,
    course_id: str,
    title: str,
    notes: str | None,
    created_by: str,
    material_type: str,
    schedule_item_id: str,
    file_name: str | None,
    document_type: str | None,
    file_url: str | None,
    mime_type: str | None,
    size_bytes: int | None,
    uploaded_at: str | None,
) -> dict[str, Any]:
    material = {
        "id": _academy_make_id("material"),
        "linkId": _academy_make_id("material-link"),
        "documentId": f"upload:{_academy_make_id('document')}",
        "title": title or file_name or "Course material",
        "documentType": document_type or "uploaded-file",
        "status": "ready",
        "materialType": material_type,
        "sortOrder": len([item for item in state["course_materials"] if item.get("course_id") == course_id]),
        "wordCount": 0,
        "readingTimeMinutes": 0,
        "authorId": created_by,
        "notes": notes,
        "fileName": file_name,
        "fileUrl": file_url,
        "mimeType": mime_type,
        "sizeBytes": size_bytes,
        "uploadedAt": uploaded_at,
        "scheduleItemId": schedule_item_id,
        "createdAt": _academy_now_iso(),
        "updatedAt": _academy_now_iso(),
        "course_id": course_id,
        "module_id": None,
        "lesson_id": None,
    }
    state["course_materials"].append(material)
    return material


def _academy_remove_schedule_item_links(
    state: dict[str, list[dict[str, Any]]],
    schedule_item: dict[str, Any],
) -> None:
    linked_assignment_id = schedule_item.get("linked_assignment_id")
    linked_material_link_id = schedule_item.get("linked_material_link_id")

    if linked_assignment_id:
        state["assignments"] = [item for item in state["assignments"] if item.get("id") != linked_assignment_id]
        state["submissions"] = [item for item in state["submissions"] if item.get("assignment_id") != linked_assignment_id]

    if linked_material_link_id:
        state["course_materials"] = [item for item in state["course_materials"] if item.get("linkId") != linked_material_link_id]


async def _handle_schedule_items(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method

    if parts[0] == "courses" and len(parts) >= 3 and parts[2] == "schedule-items":
        course_id = parts[1]
        if method == "GET":
            rows = [item for item in state["course_schedule_items"] if item.get("course_id") == course_id]
            rows.sort(key=lambda item: item.get("scheduled_at") or "")
            return JSONResponse(rows)

        if method == "POST":
            body = await request.json()
            course_meta = await _academy_fetch_course_meta(course_id)
            category = str(body.get("category") or "assignment").lower()
            if category not in {"assignment", "reading", "material"}:
                category = "material"

            schedule_item = {
                "id": _academy_make_id("schedule-item"),
                "course_id": course_id,
                "title": body.get("title") or _academy_schedule_category_label(category),
                "notes": body.get("notes") or body.get("description"),
                "scheduled_at": body.get("scheduled_at") or body.get("scheduledAt") or _academy_now_iso(),
                "category": category,
                "created_by": request.query_params.get("created_by") or body.get("created_by") or body.get("createdBy") or "academy-instructor",
                "linked_assignment_id": None,
                "linked_material_link_id": None,
                "file_name": body.get("file_name") or body.get("fileName"),
                "file_url": body.get("file_url") or body.get("fileUrl"),
                "mime_type": body.get("mime_type") or body.get("mimeType"),
                "size_bytes": body.get("size_bytes") or body.get("sizeBytes"),
                "uploaded_at": body.get("uploaded_at") or body.get("uploadedAt"),
                "created_at": _academy_now_iso(),
                "updated_at": _academy_now_iso(),
            }

            course_title = course_meta.get("title") if course_meta else "Street Voices Academy Course"
            instructor_id = (course_meta.get("instructor_id") if course_meta else None) or schedule_item["created_by"]

            if category == "assignment":
                assignment = _academy_create_schedule_assignment(
                    state,
                    course_id=course_id,
                    course_title=course_title,
                    instructor_id=instructor_id,
                    title=schedule_item["title"],
                    notes=schedule_item["notes"],
                    due_date=schedule_item["scheduled_at"],
                )
                schedule_item["linked_assignment_id"] = assignment.get("id")
            else:
                material = _academy_create_schedule_material(
                    state,
                    course_id=course_id,
                    title=schedule_item["title"],
                    notes=schedule_item["notes"],
                    created_by=schedule_item["created_by"],
                    material_type="reading" if category == "reading" else "supplementary",
                    schedule_item_id=schedule_item["id"],
                    file_name=schedule_item.get("file_name"),
                    document_type=body.get("document_type") or body.get("documentType"),
                    file_url=schedule_item.get("file_url"),
                    mime_type=schedule_item.get("mime_type"),
                    size_bytes=int(schedule_item.get("size_bytes") or 0) or None,
                    uploaded_at=schedule_item.get("uploaded_at"),
                )
                schedule_item["linked_material_link_id"] = material.get("linkId")

            state["course_schedule_items"].append(schedule_item)
            _save_academy_state(state)
            return JSONResponse(schedule_item, status_code=201)

    if parts[0] == "schedule-items" and len(parts) >= 2:
        schedule_item = _academy_find(state["course_schedule_items"], parts[1])
        if not schedule_item:
            return JSONResponse({"detail": "Schedule item not found"}, status_code=404)

        if method == "DELETE":
            _academy_remove_schedule_item_links(state, schedule_item)
            state["course_schedule_items"] = [item for item in state["course_schedule_items"] if item.get("id") != schedule_item.get("id")]
            _save_academy_state(state)
            return JSONResponse({"success": True})

    return JSONResponse({"detail": "Unsupported schedule route"}, status_code=404)


async def _handle_attendance(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method

    if parts[0] == "courses" and len(parts) >= 3 and parts[2] == "attendance":
        course_id = parts[1]
        class_date = str(
            request.query_params.get("class_date")
            or request.query_params.get("date")
            or _academy_now_iso()[:10]
        )[:10]

        if method == "GET":
            records_by_user = {
                str(record.get("user_id") or ""): record
                for record in state["attendance_records"]
                if record.get("course_id") == course_id and str(record.get("class_date") or "")[:10] == class_date
            }

            active_enrollments = [
                enrollment
                for enrollment in state["enrollments"]
                if enrollment.get("course_id") == course_id and enrollment.get("status") != "dropped"
            ]

            rows = []
            for enrollment in active_enrollments:
                user_id = str(enrollment.get("user_id") or "")
                attendance = records_by_user.get(user_id)
                rows.append(
                    {
                        "record_id": attendance.get("id") if attendance else None,
                        "course_id": course_id,
                        "class_date": class_date,
                        "user_id": user_id,
                        "student_id": user_id,
                        "user_name": _academy_display_name_for_user(state, user_id, course_id=course_id),
                        "attendance_status": attendance.get("attendance_status") if attendance else None,
                        "marked_by": attendance.get("marked_by") if attendance else None,
                        "updated_at": attendance.get("updated_at") if attendance else None,
                        "progress_percent": int(enrollment.get("progress_percent") or 0),
                        "enrollment_status": enrollment.get("status") or "active",
                    }
                )

            rows.sort(
                key=lambda item: (
                    str(item.get("user_name") or "").lower(),
                    str(item.get("user_id") or "").lower(),
                )
            )
            present_count = len([row for row in rows if row.get("attendance_status") == "present"])
            absent_count = len([row for row in rows if row.get("attendance_status") == "absent"])
            return JSONResponse(
                {
                    "course_id": course_id,
                    "class_date": class_date,
                    "students": rows,
                    "total": len(rows),
                    "present_count": present_count,
                    "absent_count": absent_count,
                }
            )

        if method in {"POST", "PUT", "PATCH"}:
            body = await request.json()
            user_id = str(body.get("user_id") or body.get("student_id") or "").strip()
            attendance_status = str(body.get("attendance_status") or body.get("status") or "").strip().lower()
            class_date = str(body.get("class_date") or class_date)[:10]
            marked_by = (
                body.get("marked_by")
                or body.get("markedBy")
                or request.query_params.get("marked_by")
                or request.query_params.get("markedBy")
                or "academy-instructor"
            )

            if user_id == "":
                return JSONResponse({"detail": "user_id is required"}, status_code=400)
            if attendance_status not in {"present", "absent"}:
                return JSONResponse({"detail": "attendance_status must be present or absent"}, status_code=400)

            active_enrollment = next(
                (
                    enrollment
                    for enrollment in state["enrollments"]
                    if enrollment.get("course_id") == course_id
                    and enrollment.get("user_id") == user_id
                    and enrollment.get("status") != "dropped"
                ),
                None,
            )
            if active_enrollment is None:
                return JSONResponse({"detail": "Active enrollment not found for this learner"}, status_code=404)

            record = next(
                (
                    item
                    for item in state["attendance_records"]
                    if item.get("course_id") == course_id
                    and item.get("user_id") == user_id
                    and str(item.get("class_date") or "")[:10] == class_date
                ),
                None,
            )

            now_iso = _academy_now_iso()
            if record is None:
                record = {
                    "id": _academy_make_id("attendance"),
                    "course_id": course_id,
                    "user_id": user_id,
                    "class_date": class_date,
                    "attendance_status": attendance_status,
                    "marked_by": marked_by,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
                state["attendance_records"].append(record)
            else:
                record["attendance_status"] = attendance_status
                record["marked_by"] = marked_by
                record["updated_at"] = now_iso

            _save_academy_state(state)
            return JSONResponse(
                {
                    **record,
                    "student_id": user_id,
                    "user_name": _academy_display_name_for_user(state, user_id, course_id=course_id),
                }
            )

    return JSONResponse({"detail": "Unsupported attendance route"}, status_code=404)


async def _handle_video(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method
    query = request.query_params

    if len(parts) >= 3 and parts[1] == "progress":
        lesson_id = parts[2]
        user_id = query.get("user_id")
        progress = next(
            (
                item
                for item in state["video_progress"]
                if item.get("lesson_id") == lesson_id and item.get("user_id") == user_id
            ),
            None,
        )
        if method == "GET":
            if progress is None:
                return JSONResponse({"detail": "Not found"}, status_code=404)
            return JSONResponse(progress)
        if method == "POST":
            body = await request.json()
            duration = float(body.get("duration") or 0)
            current_time = float(body.get("current_time") or 0)
            progress_percent = round(current_time / duration * 100, 2) if duration else 0
            if progress is None:
                progress = {
                    "id": _academy_make_id("video-progress"),
                    "user_id": user_id,
                    "lesson_id": lesson_id,
                    "video_url": query.get("video_url"),
                    "current_time": current_time,
                    "duration": duration,
                    "progress_percent": progress_percent,
                    "watch_count": 1,
                    "total_watch_time": current_time,
                    "completed": progress_percent >= 90,
                    "completed_at": _academy_now_iso() if progress_percent >= 90 else None,
                    "watched_segments": [],
                    "playback_speed": body.get("playback_speed") or 1,
                    "quality": body.get("quality") or "auto",
                    "last_watched_at": _academy_now_iso(),
                }
                state["video_progress"].append(progress)
            else:
                progress.update(
                    {
                        "current_time": current_time,
                        "duration": duration,
                        "progress_percent": progress_percent,
                        "total_watch_time": float(progress.get("total_watch_time") or 0) + current_time,
                        "completed": progress_percent >= 90,
                        "completed_at": _academy_now_iso() if progress_percent >= 90 else progress.get("completed_at"),
                        "playback_speed": body.get("playback_speed") or progress.get("playback_speed") or 1,
                        "quality": body.get("quality") or progress.get("quality") or "auto",
                        "last_watched_at": _academy_now_iso(),
                    }
                )
            _save_academy_state(state)
            return JSONResponse(progress)

    if len(parts) >= 3 and parts[1] == "stats" and method == "GET":
        lesson_id = parts[2]
        rows = [item for item in state["video_progress"] if item.get("lesson_id") == lesson_id]
        total = len(rows)
        completed = len([item for item in rows if item.get("completed")])
        avg_progress = round(sum(float(item.get("progress_percent") or 0) for item in rows) / total, 2) if total else 0
        return JSONResponse(
            {
                "total_views": total,
                "unique_viewers": total,
                "avg_progress": avg_progress,
                "completion_rate": round(completed / total * 100, 2) if total else 0,
                "completed_count": completed,
            }
        )

    if len(parts) >= 3 and parts[1] in {"bookmarks", "notes"}:
        key = "video_bookmarks" if parts[1] == "bookmarks" else "video_notes"
        lesson_id = parts[2]
        user_id = query.get("user_id")
        if method == "GET":
            rows = [item for item in state[key] if item.get("lesson_id") == lesson_id and item.get("user_id") == user_id]
            return JSONResponse(rows)
        if method == "POST":
            body = await request.json()
            item = {
                "id": _academy_make_id(parts[1][:-1]),
                "user_id": user_id,
                "lesson_id": lesson_id,
                "timestamp": body.get("timestamp"),
                "title": body.get("title"),
                "note": body.get("note"),
                "content": body.get("content"),
                "created_at": _academy_now_iso(),
            }
            state[key].append(item)
            _save_academy_state(state)
            return JSONResponse(item, status_code=201)

    if len(parts) >= 3 and parts[1] in {"bookmarks", "notes"} and method in ("DELETE", "PUT"):
        key = "video_bookmarks" if parts[1] == "bookmarks" else "video_notes"
        item = _academy_find(state[key], parts[2])
        if not item:
            return JSONResponse({"detail": "Not found"}, status_code=404)
        if method == "DELETE":
            state[key] = [row for row in state[key] if row.get("id") != item["id"]]
            _save_academy_state(state)
            return JSONResponse({"ok": True})
        body = await request.json()
        item["content"] = body.get("content")
        _save_academy_state(state)
        return JSONResponse(item)

    return JSONResponse({"detail": "Unsupported video route"}, status_code=404)


async def _handle_moodle(request: Request, parts: list[str]) -> Response:
    state = _load_academy_state()
    method = request.method

    if len(parts) == 3 and parts[1] == "forums" and method == "GET":
        course_id = parts[2]
        forum_id = abs(hash(course_id)) % 100000
        _academy_seed_forum(state, forum_id, course_id)
        _save_academy_state(state)
        return JSONResponse(
            {
                "forums": [
                    {
                        "id": forum_id,
                        "course": course_id,
                        "name": "Course Discussions",
                        "intro": "Questions, reflections, and peer discussion for this Academy course.",
                        "type": "general",
                        "numdiscussions": len([item for item in state["forum_discussions"] if item.get("forum_id") == forum_id]),
                    }
                ]
            }
        )



    if len(parts) >= 6 and parts[1] == "forums" and parts[3] == "discussions":
        forum_id = int(parts[2])
        discussion_id = int(parts[4])
        discussion = next(
            (
                item
                for item in state["forum_discussions"]
                if item.get("forum_id") == forum_id and int(item.get("id", 0)) == discussion_id
            ),
            None,
        )

        if discussion is None:
            return JSONResponse({"error": "Discussion not found"}, status_code=404)

        if parts[5] == "replies" and method == "POST":
            body = await request.json()
            created_at = int(time.time())
            author_role = body.get("author_role") or "student"
            fallback_author = "Course Instructor" if author_role == "instructor" else "Academy learner"
            reply = {
                "id": int(time.time() * 1000),
                "message": body.get("message") or "",
                "userfullname": body.get("author_name") or fallback_author,
                "userid": body.get("author_id") or 1,
                "author_role": author_role,
                "created": created_at,
            }
            discussion.setdefault("replies", []).append(reply)
            discussion["modified"] = created_at
            discussion["timemodified"] = created_at
            discussion["numreplies"] = len(discussion.get("replies", []))
            _save_academy_state(state)
            return JSONResponse(_academy_normalize_forum_discussion(discussion), status_code=201)

        if parts[5] == "reactions" and method == "POST":
            body = await request.json()
            reaction = body.get("reaction")
            if reaction not in {"up", "down"}:
                return JSONResponse({"error": "Invalid reaction"}, status_code=400)

            author_id = str(body.get("author_id") or 1)
            created_at = int(time.time())
            reactions = discussion.setdefault("reactions", {"up": [], "down": []})
            was_selected = any(str(entry) == author_id for entry in reactions.get(reaction, []))

            for key in ("up", "down"):
                reactions[key] = [entry for entry in reactions.get(key, []) if str(entry) != author_id]

            if not was_selected:
                reactions[reaction].append(author_id)

            discussion["modified"] = created_at
            discussion["timemodified"] = created_at
            _save_academy_state(state)
            return JSONResponse(_academy_normalize_forum_discussion(discussion), status_code=201)

    if len(parts) >= 5 and parts[1] == "forums" and parts[3] == "discussions" and method == "DELETE":
        forum_id = int(parts[2])
        discussion_id = int(parts[4])
        discussion = next(
            (
                item
                for item in state["forum_discussions"]
                if item.get("forum_id") == forum_id and int(item.get("id", 0)) == discussion_id
            ),
            None,
        )
        if discussion is None:
            return JSONResponse({"error": "Discussion not found"}, status_code=404)

        state["forum_discussions"] = [
            item
            for item in state["forum_discussions"]
            if not (item.get("forum_id") == forum_id and int(item.get("id", 0)) == discussion_id)
        ]
        _save_academy_state(state)
        return JSONResponse({"deleted": True})

    if len(parts) >= 4 and parts[1] == "forums" and parts[3] == "discussions":
        forum_id = int(parts[2])
        if method == "GET":
            rows = [_academy_normalize_forum_discussion(item) for item in state["forum_discussions"] if item.get("forum_id") == forum_id]
            rows.sort(key=lambda item: item.get("created", 0), reverse=True)
            return JSONResponse({"discussions": rows})
        if method == "POST":
            body = await request.json()
            created_at = int(time.time())
            author_role = body.get("author_role") or "student"
            fallback_author = "Course Instructor" if author_role == "instructor" else "Academy learner"
            discussion = _academy_normalize_forum_discussion(
                {
                    "id": int(time.time() * 1000),
                    "forum_id": forum_id,
                    "course_id": body.get("course_id"),
                    "name": body.get("subject") or "New discussion",
                    "subject": body.get("subject") or "New discussion",
                    "message": body.get("message") or "",
                    "userfullname": body.get("author_name") or fallback_author,
                    "userid": body.get("author_id") or 1,
                    "author_role": author_role,
                    "created": created_at,
                    "modified": created_at,
                    "numreplies": 0,
                    "pinned": False,
                    "timemodified": created_at,
                    "replies": [],
                    "reactions": {"up": [], "down": []},
                }
            )
            state["forum_discussions"].append(discussion)
            _save_academy_state(state)
            return JSONResponse(discussion, status_code=201)

    if len(parts) >= 3 and parts[1] == "calendar" and method == "GET":
        user_id = parts[2]
        events = []
        for session in state["live_sessions"]:
            is_participant = any(
                item.get("session_id") == session.get("id") and item.get("user_id") == user_id
                for item in state["session_registrations"]
            )
            if is_participant or session.get("instructor_id") == user_id:
                events.append(
                    {
                        "id": abs(hash(session["id"])) % 100000,
                        "name": session.get("title"),
                        "description": session.get("description") or "",
                        "coursefullname": "Street Voices Academy",
                        "timestart": int(_academy_parse_iso(session.get("scheduled_start")).timestamp()),
                        "timeduration": int(
                            (
                                _academy_parse_iso(session.get("scheduled_end")) - _academy_parse_iso(session.get("scheduled_start"))
                            ).total_seconds()
                        ),
                        "eventtype": "course",
                        "url": f"/academy/live-sessions/{session['id']}",
                    }
                )
        for deadline in state["cohort_deadlines"]:
            events.append(
                {
                    "id": abs(hash(deadline["id"])) % 100000,
                    "name": deadline.get("description") or "Cohort deadline",
                    "description": deadline.get("description") or "",
                    "coursefullname": "Street Voices Academy",
                    "timestart": int(_academy_parse_iso(deadline.get("deadline")).timestamp()),
                    "timeduration": 0,
                    "eventtype": "due",
                    "url": f"/academy/courses/{deadline.get('cohort_id')}",
                }
            )
        events.sort(key=lambda item: item["timestart"])
        return JSONResponse({"events": events})

    if len(parts) >= 4 and parts[1] == "grades" and method == "GET":
        user_id = parts[2]
        course_id = parts[3]
        course_meta = await _academy_fetch_course_meta(course_id)
        _academy_seed_assignments_for_course(
            state,
            course_id,
            course_title=course_meta.get("title") if course_meta else None,
            instructor_id=course_meta.get("instructor_id") if course_meta else None,
        )
        _save_academy_state(state)

        grade_items = []
        for submission in state["submissions"]:
            if submission.get("user_id") != user_id or submission.get("course_id") != course_id:
                continue

            assignment = _academy_find_assignment(state, submission.get("assignment_id", ""))
            if not assignment:
                continue

            score = submission.get("adjusted_score", submission.get("score"))
            max_points = float(assignment.get("max_points") or 100)
            percentage = round((float(score) / max_points) * 100, 2) if score is not None and max_points else None

            grade_items.append(
                {
                    "id": abs(hash(submission["id"])) % 100000,
                    "itemname": assignment.get("title") or "Assignment",
                    "itemtype": "mod",
                    "itemmodule": "assign",
                    "cmid": assignment.get("id"),
                    "graderaw": score,
                    "gradeformatted": f"{score}/{int(max_points)}" if score is not None else None,
                    "grademin": 0,
                    "grademax": max_points,
                    "percentageformatted": f"{percentage}%" if percentage is not None else None,
                    "feedback": submission.get("feedback"),
                }
            )

        return JSONResponse({"grade_items": grade_items})

    if len(parts) >= 3 and parts[1] == "badges" and method == "GET":
        user_id = parts[2]
        badges = [
            {
                "id": abs(hash(cert["id"])) % 100000,
                "name": "Course Completion Badge",
                "description": "Awarded for completing a Street Voices Academy course.",
                "badgeurl": cert.get("badge_url"),
                "issuername": "Street Voices Academy",
                "courseid": cert.get("course_id"),
                "dateissued": int(_academy_parse_iso(cert.get("issued_at")).timestamp()),
            }
            for cert in state["certificates"]
            if cert.get("user_id") == user_id
        ]
        return JSONResponse({"badges": badges})

    if len(parts) >= 3 and parts[1] == "assignments" and method == "GET":
        course_id = parts[2]
        course_meta = await _academy_fetch_course_meta(course_id)
        assignments = _academy_seed_assignments_for_course(
            state,
            course_id,
            course_title=course_meta.get("title") if course_meta else None,
            instructor_id=course_meta.get("instructor_id") if course_meta else None,
        )
        _save_academy_state(state)
        moodle_assignments = [
            {
                "id": row.get("id"),
                "cmid": int(uuid.uuid5(uuid.NAMESPACE_URL, str(row.get("id"))).int % 100000),
                "course": row.get("course_id"),
                "name": row.get("title"),
                "intro": row.get("instructions") or row.get("description") or "",
                "duedate": int(_academy_parse_iso(row.get("due_date")).timestamp()) if row.get("due_date") else 0,
                "allowsubmissionsfromdate": int(_academy_parse_iso(row.get("available_from")).timestamp()) if row.get("available_from") else 0,
                "cutoffdate": int(_academy_parse_iso(row.get("available_until")).timestamp()) if row.get("available_until") else 0,
                "grade": row.get("max_points") or 100,
                "nosubmissions": 0,
                "submissiondrafts": 1,
            }
            for row in assignments
        ]
        return JSONResponse({"assignments": moodle_assignments})

    if len(parts) >= 4 and parts[1] == "assignments" and parts[3] == "submit" and method == "POST":
        assignment_id = parts[2]
        assignment = _academy_find_assignment(state, assignment_id)
        if assignment is None:
            return JSONResponse({"detail": "Assignment not found"}, status_code=404)
        body = await request.json()
        user_id = request.query_params.get("user_id") or body.get("user_id")
        if not user_id:
            return JSONResponse({"detail": "user_id is required"}, status_code=400)

        submission = _academy_get_latest_submission(state, assignment_id, user_id)
        if submission is None:
            quiz_answers = body.get("quiz_answers") or []
            submission = _academy_create_submission(
                state,
                assignment,
                user_id,
                status="submitted",
                text_content=body.get("text") or body.get("text_content") or _academy_quiz_answers_to_text(quiz_answers),
                quiz_answers=quiz_answers,
            )
        else:
            quiz_answers = body.get("quiz_answers") or submission.get("quiz_answers") or []
            submission["quiz_answers"] = quiz_answers
            submission["text_content"] = body.get("text") or body.get("text_content") or submission.get("text_content") or _academy_quiz_answers_to_text(quiz_answers)
            submission["word_count"] = len(str(submission.get("text_content") or "").split())
            submission["status"] = "submitted"
            submission["submitted_at"] = _academy_now_iso()
            submission["updated_at"] = _academy_now_iso()
        _save_academy_state(state)
        return JSONResponse({"ok": True, "submission_id": submission.get("id")})

    return JSONResponse({"detail": "Unsupported moodle route"}, status_code=404)


async def _list_academy_enrollments(user_id: str | None = None, course_id: str | None = None) -> list[dict[str, Any]]:
    state = _load_academy_state()
    if _academy_sync_course_enrollments_from_cohorts(state, user_id=user_id):
        _save_academy_state(state)
    local_rows = [
        item
        for item in state["enrollments"]
        if (not user_id or item.get("user_id") == user_id)
        and (not course_id or item.get("course_id") == course_id)
    ]
    if local_rows:
        return local_rows

    import httpx

    base = f"{_SUPABASE_URL}/rest/v1/academy_enrollments"
    qs = "select=*"
    if user_id:
        qs += f"&user_id=eq.{user_id}"
    if course_id:
        qs += f"&course_id=eq.{course_id}"
    headers = _supabase_headers()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{base}?{qs}", headers=headers)
            data = resp.json()
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _make_provider(config):
    """Create the appropriate LLM provider from config (mirrors cli/commands.py)."""
    from loguru import logger
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.custom_provider import CustomProvider

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)

    if provider_name == "custom":
        return CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    # Warn if no API key is configured — the LLM call will fail or hang
    if not p or not p.api_key:
        logger.warning(
            f"No API key found for model '{model}' (provider: {provider_name}). "
            f"LLM calls will fail. Configure the provider API key in config.yaml "
            f"or set the appropriate environment variable."
        )

    from nanobot.providers.registry import find_by_name
    spec = find_by_name(provider_name)

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=provider_name,
    )


def _install_mcp_error_suppressor():
    """Suppress the MCP cancel-scope RuntimeError that crashes the process.

    The MCP stdio_client async generator raises RuntimeError during cleanup
    due to an anyio cancel-scope bug.  This fires outside any try/except,
    so we catch it at the event loop level.
    """
    loop = asyncio.get_event_loop()
    _orig = loop.get_exception_handler()

    def _handler(loop, context):
        exc = context.get("exception")
        msg = context.get("message", "")
        if exc and "cancel scope" in str(exc):
            logger.debug(f"Suppressed MCP cancel-scope error: {exc}")
            return
        if "cancel scope" in msg:
            logger.debug(f"Suppressed MCP cancel-scope message: {msg}")
            return
        if _orig:
            _orig(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


@asynccontextmanager
async def lifespan(app):
    global _agent, _cron, _orchestrator
    _install_mcp_error_suppressor()
    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron_service = CronService(store_path=cron_store_path)
    _cron = cron_service

    defaults = config.agents.defaults
    _agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=defaults.model,
        max_iterations=defaults.max_tool_iterations,
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        memory_window=defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron_service,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        postiz_config=config.tools.postiz,
    )
    # Initialize Mem0 semantic memory (optional — degrades gracefully)
    try:
        from nanobot.agent.mem0_memory import Mem0Store
        _agent.mem0 = Mem0Store(
            workspace=config.workspace_path,
            provider=provider,
            model=defaults.model,
            loop=asyncio.get_event_loop(),
        )
    except Exception as e:
        logger.warning(f"Mem0 semantic memory not available: {e}")

    # Wire up cron job callback to process messages through agent
    async def _on_cron_job(job):
        logger.info(f"Cron job '{job.name}' firing: {job.payload.message}")
        response = await _agent.process_direct(
            content=job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cron",
            chat_id=job.payload.to or "system",
        )
        return response

    cron_service.on_job = _on_cron_job

    # Register email tools if email channel is configured
    email_cfg = config.channels.email
    if email_cfg.enabled and email_cfg.imap_host and email_cfg.imap_username:
        from nanobot.agent.tools.email_tools import EmailReadTool, EmailSendTool
        _agent.tools.register(EmailReadTool(
            imap_host=email_cfg.imap_host,
            imap_port=email_cfg.imap_port,
            username=email_cfg.imap_username,
            password=email_cfg.imap_password,
            use_ssl=email_cfg.imap_use_ssl,
            mailbox=email_cfg.imap_mailbox,
        ))
        _agent.tools.register(EmailSendTool(
            smtp_host=email_cfg.smtp_host,
            smtp_port=email_cfg.smtp_port,
            username=email_cfg.smtp_username,
            password=email_cfg.smtp_password,
            from_addr=email_cfg.from_address or email_cfg.smtp_username,
            use_tls=email_cfg.smtp_use_tls,
            use_ssl=email_cfg.smtp_use_ssl,
        ))
        logger.info("Email tools registered (read + send)")

    # Register Remotion video tools
    remotion_dir = config.workspace_path / "remotion"
    if remotion_dir.exists():
        try:
            from nanobot.agent.tools.remotion import RemotionComposeTool, RemotionRenderTool
            _agent.tools.register(RemotionComposeTool(remotion_dir=remotion_dir))
            _agent.tools.register(RemotionRenderTool(
                remotion_dir=remotion_dir,
                base_url="http://localhost:18790",
            ))
            from nanobot.agent.tools.tts import QwenTTSTool
            audio_dir = remotion_dir / "public" / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            _agent.tools.register(QwenTTSTool(audio_dir=audio_dir))
            logger.info("Remotion tools registered (compose + render + tts)")
        except ImportError as e:
            logger.warning(f"Remotion/TTS tools not available (missing deps): {e}")

    # Register Qwen-Image generation tool (always available, connects to local server)
    from nanobot.agent.tools.image_gen import QwenImageGenTool
    _agent.tools.register(QwenImageGenTool())
    logger.info("Qwen-Image generation tool registered (local server at :18791)")

    # Register article image generation tool
    from nanobot.agent.tools.article_image import ArticleImageTool
    _agent.tools.register(ArticleImageTool(base_url="http://localhost:18790"))
    logger.info("Article image generation tool registered")

    # Register Academy tools (OpenMAIC + SBP backend)
    try:
        from nanobot.agent.tools.academy import (
            AcademyCreateCourseTool,
            AcademyListCoursesTool,
            AcademyGenerateQuizTool,
            AcademyGradeQuizTool,
            AcademyTutorTool,
        )
        from nanobot.services.openmaic_client import OpenMAICClient

        _openmaic_client = OpenMAICClient(base_url="http://localhost:3001")
        _academy_api_url = "http://localhost:18790"  # Self-proxy (nanobot → Supabase)
        _agent.tools.register(AcademyCreateCourseTool(openmaic=_openmaic_client, sbp_api=_academy_api_url))
        _agent.tools.register(AcademyListCoursesTool(sbp_api=_academy_api_url))
        _agent.tools.register(AcademyGenerateQuizTool(openmaic=_openmaic_client, sbp_api=_academy_api_url))
        _agent.tools.register(AcademyGradeQuizTool(openmaic=_openmaic_client, sbp_api=_academy_api_url))
        _agent.tools.register(AcademyTutorTool(openmaic=_openmaic_client))
        logger.info("Academy tools registered (5 tools, OpenMAIC + Supabase)")
    except Exception as e:
        logger.warning(f"Academy tools not available: {e}")

    # Register SV Social tools (direct PostgreSQL access)
    _social_pool = None
    try:
        import asyncpg
        _social_pool = await asyncpg.create_pool(
            "postgresql://lobehub:lobehub_password@localhost:5433/social",
            min_size=1,
            max_size=5,
        )
        from nanobot.agent.tools.social_tools import ALL_SOCIAL_TOOLS
        for tool_cls in ALL_SOCIAL_TOOLS:
            _agent.tools.register(tool_cls(pool=_social_pool))
        logger.info(f"SV Social tools registered ({len(ALL_SOCIAL_TOOLS)} tools)")

        # Register unified search tool (spans Social + MeiliSearch)
        from nanobot.agent.tools.unified_search import UnifiedSearchTool
        _agent.tools.register(UnifiedSearchTool(
            pool=_social_pool,
            meili_url="http://localhost:7700",
            meili_key="DrhYf7zENyR6AlUCKmnz0eYASOQdl6zxH7s7MKFSfFCt",
        ))
        logger.info("Unified search tool registered (social + chat + directory)")
    except Exception as e:
        logger.warning(f"SV Social tools not available: {e}")

    # Start Slack channel if enabled
    channel_manager = None
    slack_cfg = config.channels.slack
    if slack_cfg.enabled and slack_cfg.bot_token and slack_cfg.app_token:
        from nanobot.channels.manager import ChannelManager
        channel_manager = ChannelManager(config, bus)
        # Start the agent loop (consumes inbound from bus, publishes outbound)
        agent_task = asyncio.create_task(_agent.run())

        def _on_agent_task_done(task: asyncio.Task) -> None:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                logger.warning("Agent task was cancelled")
                return
            if exc:
                logger.critical(f"Agent task died with exception: {exc}", exc_info=exc)
            else:
                logger.warning("Agent task finished unexpectedly (no exception)")

        agent_task.add_done_callback(_on_agent_task_done)

        # Start the channel manager (starts Slack + outbound dispatcher)
        channels_task = asyncio.create_task(channel_manager.start_all())
        logger.info("Slack channel started")

    # Ensure screenshots directory exists
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Start cron service
    try:
        await cron_service.start()
        logger.info(f"Cron service started ({len(cron_service.list_jobs(include_disabled=True))} jobs)")
    except Exception as exc:
        _cron = None
        logger.warning(f"Cron service failed to start: {exc}. Continuing without cron scheduling.")

    # ── Multi-agent system initialization ──
    try:
        from nanobot.agents.loader import load_agents
        from nanobot.agents.factory import ToolFactory
        from nanobot.agents.orchestrator import Orchestrator

        teams_dir = Path(__file__).parent / "agents" / "teams"
        agent_registry = load_agents(teams_dir)

        if len(agent_registry) > 0:
            tool_config: dict[str, Any] = {
                "brave_api_key": config.tools.web.search.api_key or None,
                "restrict_to_workspace": config.tools.restrict_to_workspace,
                "shell": {
                    "timeout": getattr(config.tools.exec, "timeout", 120),
                },
            }
            # MCP tools connect async in the agent loop — pass the agent's
            # tool registry reference so ToolFactory can pull them lazily.
            # We pass an empty dict now; the factory will be patched once MCP connects.
            _extra_tools: dict[str, Any] = {}
            _mcp_ready = asyncio.Event()

            tool_factory = ToolFactory(
                agent_registry,
                workspace=config.workspace_path,
                tool_config=tool_config,
                provider=provider,
                mcp_tools=_extra_tools,
                mcp_ready=_mcp_ready,
            )

            # Schedule a task to inject MCP tools once they're connected
            async def _inject_mcp_tools():
                """Wait for ALL MCP tools to be available, then inject them."""
                # Wait until Playwright is connected (it's one of the slower ones)
                for _ in range(60):  # wait up to 60 seconds
                    await asyncio.sleep(2)
                    pw_names = [n for n in _agent.tools.tool_names if "playwright" in n]
                    if pw_names:
                        break
                # Now collect everything
                for t_name in _agent.tools.tool_names:
                    if (t_name.startswith("mcp_")
                        or t_name.startswith("social_")
                        or t_name == "unified_search"):
                        t = _agent.tools.get(t_name)
                        if t:
                            _extra_tools[t_name] = t
                pw_count = sum(1 for n in _extra_tools if "playwright" in n)
                logger.info(
                    f"Injected {len(_extra_tools)} MCP/social tools into ToolFactory "
                    f"({pw_count} Playwright tools)"
                )
                if pw_count == 0:
                    logger.warning("Playwright MCP tools not found — browser automation won't work for agents")
                _mcp_ready.set()

            asyncio.create_task(_inject_mcp_tools())
            _orchestrator = Orchestrator(
                provider=provider,
                agent_registry=agent_registry,
                tool_factory=tool_factory,
                default_model=defaults.model,
            )
            # Wire orchestrator into the agent loop so all channels
            # (Slack, Telegram, etc.) route through the multi-agent system
            _agent.orchestrator = _orchestrator
            logger.info(
                f"Multi-agent system ready: {len(agent_registry)} agents across "
                f"{len(agent_registry.get_teams())} teams (all channels enabled)"
            )
        else:
            logger.info("No agent team definitions found — multi-agent system disabled")
    except Exception as exc:
        logger.warning(f"Multi-agent system not available: {exc}")

    # ── Deep Agent Harness initialization ──
    try:
        from nanobot.harness import DeepAgentHarness
        global _harness

        _harness = DeepAgentHarness(
            workspace=config.workspace_path,
            config=config.to_dict() if hasattr(config, 'to_dict') else {},
        )
        # Collect MCP tools from the agent loop for bridging
        mcp_tool_dict = {}
        for tool_name in _agent.tools.tool_names:
            tool = _agent.tools.get(tool_name)
            if tool:
                mcp_tool_dict[tool_name] = tool

        await _harness.initialize(
            tool_registry=_agent.tools,
            mcp_tools=mcp_tool_dict,
            teams_dir=Path(__file__).parent / "agents" / "teams",
        )
        logger.info(
            f"Deep Agent Harness ready: {_harness.agent_count} agents, "
            f"universal memory enabled"
        )
    except Exception as exc:
        _harness = None
        logger.warning(f"Deep Agent Harness not available: {exc}")
        import traceback
        traceback.print_exc()

    # ── Gateway (WebSocket mission control) ──
    global _gateway
    try:
        from nanobot.gateway.server import GatewayServer
        from nanobot.gateway.auth import GatewayAuth

        _gateway = GatewayServer(agent=_agent, auth=GatewayAuth(), config=config)
        logger.info("Gateway mission control ready at /ws (dashboard at /dashboard)")
    except Exception as exc:
        _gateway = None
        logger.warning(f"Gateway not available: {exc}")

    # ── Redis event bus for cross-service communication ──
    global _redis_bus
    try:
        from nanobot.bus.redis_bus import RedisBus

        _redis_bus = RedisBus(url="redis://localhost:6380")

        # Example subscriber: log social messages for awareness
        async def _on_social_message(event):
            logger.debug(f"[RedisBus] Social message in #{event.get('channelName', '?')}: {event.get('content', '')[:80]}")

        _redis_bus.subscribe("social.message.new", _on_social_message)
        await _redis_bus.start()
        logger.info("Redis event bus started (redis://localhost:6380)")
    except Exception as exc:
        _redis_bus = None
        logger.warning(f"Redis event bus not available: {exc}")

    # ── Platform awareness: refresh context with Social status ──
    _platform_task = None
    if _social_pool and _agent:
        async def _refresh_platform_status():
            """Periodically query Social DB and update agent context."""
            while True:
                try:
                    async with _social_pool.acquire() as conn:
                        # Online users
                        online = await conn.fetch(
                            """SELECT display_name, is_agent FROM users
                               WHERE status != 'offline'
                                  OR last_seen_at > NOW() - INTERVAL '5 minutes'
                               ORDER BY is_agent ASC LIMIT 15"""
                        )
                        people = [r["display_name"] for r in online if not r["is_agent"]]
                        agents = [r["display_name"] for r in online if r["is_agent"]]

                        # Recent activity
                        recent = await conn.fetch(
                            """SELECT c.name, COUNT(*) as cnt
                               FROM messages m JOIN channels c ON m.channel_id = c.id
                               WHERE m.created_at > NOW() - INTERVAL '1 hour'
                                 AND m.deleted_at IS NULL AND c.name IS NOT NULL
                               GROUP BY c.name ORDER BY cnt DESC LIMIT 5"""
                        )

                    parts = ["\n## Platform Status (auto-refreshed)"]
                    if people:
                        parts.append(f"Online people: {', '.join(people)}")
                    if agents:
                        parts.append(f"Online agents: {', '.join(agents)}")
                    if not people and not agents:
                        parts.append("No users currently online")
                    if recent:
                        activity = "; ".join(f"#{r['name']} ({r['cnt']} msgs)" for r in recent)
                        parts.append(f"Recent Social activity (1h): {activity}")

                    _agent.context._platform_status = "\n".join(parts)
                except Exception as e:
                    logger.debug(f"Platform status refresh failed: {e}")
                await asyncio.sleep(60)

        _platform_task = asyncio.create_task(_refresh_platform_status())
        logger.info("Platform awareness task started (60s refresh)")

    # Register daily news cron job
    await _register_daily_news_cron()

    logger.info("Nanobot API server ready")
    yield
    try:
        cron_service.stop()
        if channel_manager:
            await channel_manager.stop_all()
            _agent.stop()
        await _agent.close_mcp()
    except (RuntimeError, BaseExceptionGroup) as shutdown_err:
        logger.warning(f"Ignoring MCP shutdown error (harmless): {shutdown_err}")
    except Exception as shutdown_err:
        logger.warning(f"Unexpected shutdown error (continuing): {shutdown_err}")
    if _platform_task:
        _platform_task.cancel()
    if _redis_bus:
        await _redis_bus.stop()
    if _social_pool:
        await _social_pool.close()
    _agent = None
    _cron = None


def _as_bool(value: Any) -> bool:
    """Best-effort cast for header/body boolean values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _as_int(value: Any, *, field: str, min_value: int = 1) -> int | None:
    """Best-effort int parser used for request compatibility."""
    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{field} must be an integer")
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field} must be an integer")
        parsed = int(stripped)
    else:
        raise ValueError(f"{field} must be an integer")
    if parsed < min_value:
        raise ValueError(f"{field} must be >= {min_value}")
    return parsed


def _as_float(
    value: Any,
    *,
    field: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Best-effort float parser used for request compatibility."""
    if value is None:
        return None
    if isinstance(value, bool):
        value = float(int(value))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{field} must be <= {max_value}")
    return parsed


def _parse_tool_choice(value: Any) -> tuple[str, str | None, str | None]:
    """Return (tool_choice, required_tool_name, error)."""
    if value is None:
        return "auto", None, None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "auto", "required"}:
            return lowered, None, None
        return "", None, "tool_choice must be one of: none, auto, required"

    if isinstance(value, dict):
        type_name = str(value.get("type", "")).strip().lower()
        if type_name in {"none", "auto"}:
            return type_name, None, None
        if type_name == "function":
            fn_name = value.get("function", {}).get("name")
            if isinstance(fn_name, str) and fn_name.strip():
                return "required", fn_name.strip(), None
            return "auto", None, "tool_choice.function requires a function name"
        return "auto", None, "tool_choice.type must be one of: none, auto, function"

    return "auto", None, "tool_choice must be a string or object"


def _parse_stream_options(value: Any) -> tuple[bool, str | None]:
    """Return include_usage flag and optional validation error."""
    if value is None:
        return False, None
    if not isinstance(value, dict):
        return False, "stream_options must be an object"
    include_usage = value.get("include_usage")
    if include_usage is None:
        return False, None
    if isinstance(include_usage, bool):
        return include_usage, None
    if isinstance(include_usage, str):
        return include_usage.strip().lower() in {"1", "true", "yes", "on"}, None
    return False, "stream_options.include_usage must be a boolean"


def _openai_error(
    message: str,
    *,
    status_code: int = 400,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Format errors in OpenAI-compatible structure."""
    return JSONResponse({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code or "invalid_request_error",
        },
    }, status_code=status_code)


def _resolve_session_key(request: Request, body: dict[str, Any]) -> str:
    """Build a session key that keeps LibreChat conversations separate."""
    session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("x-conversation-id")
        or request.headers.get("x-chat-id")
    )
    if session_id:
        return f"librechat:{session_id}"

    conv_id = (
        body.get("conversation_id")
        or body.get("session_id")
        or request.headers.get("x-user-id")
    )
    if conv_id:
        return f"librechat:{conv_id}"

    user = body.get("user")
    if isinstance(user, str) and user.strip():
        return f"librechat:{user.strip()}"
    return "librechat:default"


def _resolve_model(request_model: Any, default_model: str) -> tuple[str, str]:
    """Return `(requested_model_for_client, resolved_model_for_provider)`.

    Keep custom model names from clients untouched, but map Codex-style clients
    to prefixed model IDs when needed.
    """
    requested = str(request_model).strip() if request_model else ""
    if not requested:
        requested = default_model

    if requested.lower() == "openai-codex":
        requested = default_model

    if requested.lower().startswith("openai-codex/"):
        return requested, requested

    default_model = str(default_model)
    if "openai-codex/" in default_model and "/" not in requested and not requested.startswith("openai-codex"):
        return requested, f"openai-codex/{requested}"

    return requested, requested


async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Handle POST /v1/chat/completions."""
    if not _agent:
        return _openai_error("Agent not initialized", status_code=503, code="unavailable")

    body = await request.json()
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return _openai_error("messages must be a list", param="messages")
    if not messages:
        return _openai_error("No messages provided", code="empty_request")
    stream = _as_bool(body.get("stream", False))

    requested_model, provider_model = _resolve_model(
        body.get("model", _agent.model),
        _agent.model,
    )

    temperature = body.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            return _openai_error("temperature must be a number", param="temperature")

    max_tokens = (
        body.get("max_tokens")
        if body.get("max_tokens") is not None
        else body.get("max_output_tokens", body.get("max_completion_tokens"))
    )
    try:
        max_tokens = _as_int(max_tokens, field="max_tokens")
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="max_tokens")

    try:
        top_p = _as_float(body.get("top_p"), field="top_p", min_value=0.0, max_value=1.0)
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="top_p")

    try:
        _as_float(
            body.get("presence_penalty"),
            field="presence_penalty",
            min_value=-2.0,
            max_value=2.0,
        )
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="presence_penalty")
    try:
        _as_float(
            body.get("frequency_penalty"),
            field="frequency_penalty",
            min_value=-2.0,
            max_value=2.0,
        )
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="frequency_penalty")
    stop = body.get("stop")
    if stop is not None:
        if isinstance(stop, str):
            if not stop:
                return _openai_error("stop must not be empty", param="stop")
        elif isinstance(stop, list):
            if not stop:
                return _openai_error("stop must not be empty", param="stop")
            if not all(isinstance(item, str) and item for item in stop):
                return _openai_error("stop entries must be non-empty strings", param="stop")
        else:
            return _openai_error("stop must be a string or list of strings", param="stop")

    tool_choice, required_tool_name, err = _parse_tool_choice(body.get("tool_choice"))
    if err:
        return _openai_error(err, param="tool_choice")

    include_usage, err = _parse_stream_options(body.get("stream_options"))
    if err:
        return _openai_error(err, param="stream_options")

    n = body.get("n")
    if n is not None:
        try:
            n = _as_int(n, field="n", min_value=1)
        except (TypeError, ValueError) as err:
            return _openai_error(str(err), param="n")
        if n != 1:
            return _openai_error("Only n=1 is supported", param="n")

    try:
        max_iterations = _as_int(body.get("max_iterations"), field="max_iterations")
    except (TypeError, ValueError) as err:
        return _openai_error(str(err), param="max_iterations")
    if max_iterations is None:
        try:
            max_iterations = _as_int(body.get("max_tool_iterations"), field="max_tool_iterations")
        except (TypeError, ValueError) as err:
            return _openai_error(str(err), param="max_tool_iterations")

    session_key = _resolve_session_key(request, body)
    reset_session = _as_bool(
        request.headers.get("x-session-reset") or body.get("session_reset")
    )

    if not messages or not all(isinstance(msg, dict) for msg in messages):
        return _openai_error("Invalid message format", code="invalid_request_error")

    # ── Multi-agent routing ──
    # If the requested model starts with "agent/", route to the Orchestrator
    # instead of the single-agent AgentLoop.
    if requested_model.startswith("agent/") and _orchestrator is not None:
        agent_name = requested_model.removeprefix("agent/")

        # Don't pass provider_model as override — each agent has its own
        # configured model.  The "agent/auto" or "agent/ceo" name is for
        # routing only, not a real LLM model identifier.
        if stream:
            return StreamingResponse(
                _stream_agent_response(
                    agent_name=agent_name,
                    messages=messages,
                    model_override=None,
                    requested_model=requested_model,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
            )

        result = await _orchestrator.run(
            agent_name=agent_name,
            messages=messages,
            model_override=None,
        )
        if result.trace:
            logger.info(f"Multi-agent trace:\n{result.trace.log_summary()}")
        return JSONResponse(result.to_chat_completion(model=requested_model))

    if stream:
        return StreamingResponse(
            _stream_response(
                messages=messages,
                session_key=session_key,
                requested_model=requested_model,
                provider_model=provider_model,
                tool_choice=tool_choice,
                required_tool_name=required_tool_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_iterations,
                reset_session=reset_session,
                include_usage=include_usage,
            ),
            media_type="text/event-stream",
        )

    response_text, tools_used, usage, finish_reason = await _agent.process_openai_messages(
        messages=messages,
        session_key=session_key,
        channel="api",
        chat_id="librechat",
        model_override=provider_model,
        tool_choice=tool_choice,
        required_tool_name=required_tool_name,
        temperature_override=temperature,
        max_tokens_override=max_tokens,
        max_iterations_override=max_iterations,
        reset_session=reset_session,
        return_usage=True,
    )
    if not response_text:
        return _openai_error("No user message found", code="empty_request")

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
                "tool_calls": [
                    {"type": "function", "function": {"name": tool}} for tool in tools_used
                ] if tools_used else [],
            },
            "finish_reason": finish_reason or "stop",
        }],
        "usage": usage,
    })


async def _stream_response(
    messages: list[dict[str, Any]],
    session_key: str,
    requested_model: str,
    provider_model: str,
    tool_choice: str,
    required_tool_name: str | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_iterations: int | None = None,
    reset_session: bool = False,
    include_usage: bool = False,
):
    """Stream response as SSE events with intermediate progress."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _on_progress(text: str) -> None:
        """Push intermediate progress (tool hints) to the SSE stream."""
        if text:
            await progress_queue.put(text)

    async def _run_agent():
        """Run the agent and signal completion."""
        try:
            result = await _agent.process_openai_messages(
                messages=messages,
                session_key=session_key,
                channel="api",
                chat_id="librechat",
                on_progress=_on_progress,
                tool_choice=tool_choice,
                required_tool_name=required_tool_name,
                model_override=provider_model,
                temperature_override=temperature,
                max_tokens_override=max_tokens,
                max_iterations_override=max_iterations,
                reset_session=reset_session,
                return_usage=True,
            )
            await progress_queue.put(None)  # Signal done
            return result
        except Exception as e:
            await progress_queue.put(None)
            return f"Error: {e}"

    # Start agent processing in background
    agent_task = asyncio.create_task(_run_agent())

    # OpenAI-compatible stream starts with role-only assistant delta.
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    # Stream intermediate progress as it arrives
    while True:
        item = await progress_queue.get()
        if item is None:
            break
        # Send progress as a chunk (shown as "thinking" text)
        progress_text = f"{item}\n\n"
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{
                "index": 0,
                "delta": {"content": progress_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Get final response
    final_result = await agent_task
    tools_used: list[str] = []
    if isinstance(final_result, tuple):
        response_text = final_result[0] or ""
        if len(final_result) == 4:
            tools_used = final_result[1] if isinstance(final_result[1], list) else []
            usage = final_result[2] if isinstance(final_result[2], dict) else {}
            finish_reason = final_result[3] if isinstance(final_result[3], str) else None
        else:
            tools_used = []
            usage = {}
            finish_reason = None
    else:
        response_text = final_result
        usage = {}
        finish_reason = None

    # Publish agent task completion to Redis event bus
    if _redis_bus and tools_used:
        try:
            await _redis_bus.publish("agent.task.complete", {
                "type": "agent.task.complete",
                "agentName": requested_model,
                "taskSummary": (response_text or "")[:150],
                "toolsUsed": tools_used[:5],
                "channel": "librechat",
            })
        except Exception:
            pass

    # Send the final response
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": response_text,
                "tool_calls": [
                    {"type": "function", "function": {"name": tool}} for tool in tools_used
                ] if tools_used else [],
            },
            "finish_reason": finish_reason or "stop",
        }],
    }
    if include_usage and isinstance(usage, dict):
        usage_payload = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    yield f"data: {json.dumps(chunk)}\n\n"

    # OpenAI-compatible [DONE] usage packet when requested.
    if include_usage and isinstance(usage, dict):
        usage_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [],
            "usage": usage_payload,
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_agent_response(
    agent_name: str,
    messages: list[dict],
    model_override: str | None,
    requested_model: str,
    include_usage: bool,
):
    """Stream a multi-agent orchestrator response as SSE chunks.

    Uses an asyncio.Queue so that on_progress callbacks from the
    orchestrator / agent instances / delegate tools stream progress
    to the client in real-time (routing decisions, tool calls,
    delegation events, handoffs) instead of blocking until the
    entire multi-agent pipeline completes.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _on_progress(text: str) -> None:
        """Push intermediate progress to the SSE stream."""
        if text:
            await progress_queue.put(text)

    async def _run_orchestrator():
        """Run the orchestrator in background; signal done when finished."""
        try:
            result = await _orchestrator.run(
                agent_name=agent_name,
                messages=messages,
                model_override=model_override,
                on_progress=_on_progress,
            )
            await progress_queue.put(None)  # Signal done
            return result
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            await progress_queue.put(None)
            return e

    # Start orchestrator in background
    agent_task = asyncio.create_task(_run_orchestrator())

    # OpenAI-compatible stream starts with role-only assistant delta.
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    # Stream intermediate progress as it arrives
    while True:
        item = await progress_queue.get()
        if item is None:
            break
        progress_text = f"{item}\n\n"
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{
                "index": 0,
                "delta": {"content": progress_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Get final result
    result = await agent_task

    if isinstance(result, Exception):
        content = f"Error: {result}"
        usage = {}
        session_id = "error"
    else:
        # Log the trace
        if result.trace:
            logger.info(f"Multi-agent trace:\n{result.trace.log_summary()}")
        content = result.content
        usage = result.usage or {}
        session_id = result.session_id

    # Publish agent task completion to Redis event bus
    if _redis_bus and not isinstance(result, Exception):
        try:
            agents_involved = []
            if hasattr(result, 'trace') and result.trace:
                agents_involved = [s.agent for s in getattr(result.trace, 'steps', [])][:5]
            await _redis_bus.publish("agent.task.complete", {
                "type": "agent.task.complete",
                "agentName": agent_name,
                "taskSummary": (content or "")[:150],
                "agentsInvolved": agents_involved,
                "channel": "librechat",
            })
        except Exception:
            pass

    # Send the final response content
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Usage packet if requested
    if include_usage and usage:
        usage_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def list_models(request: Request) -> JSONResponse:
    """Handle GET /v1/models."""
    now = int(time.time())
    default_model = _agent.model if _agent else "gpt-5.4"
    model_ids = [
        default_model,
        default_model.removeprefix("openai-codex/"),
    ]
    # Keep popular aliases for clients that request openai-style names.
    model_ids.extend(["gpt-5.4", "gpt-5.1", "gpt-5"])
    deduped = []
    for model_id in model_ids:
        if model_id and model_id not in deduped:
            deduped.append(model_id)

    models = [
        {"id": model_id, "object": "model", "created": now, "owned_by": "nanobot"}
        for model_id in deduped
    ]

    # Append multi-agent models if the orchestrator is available
    if _orchestrator is not None:
        # Auto-router model — routes by intent, no CEO bottleneck
        models.insert(0, {
            "id": "agent/auto",
            "object": "model",
            "created": now,
            "owned_by": "nanobot",
        })
        models.extend(_orchestrator.list_agents())

    return JSONResponse({"object": "list", "data": models})


async def health(request: Request) -> JSONResponse:
    agent_count = 0
    teams = []
    if _orchestrator is not None:
        try:
            agent_count = len(_orchestrator.registry.agent_names)
            teams = list(_orchestrator.registry.get_teams())
        except Exception:
            pass
    harness_info = {}
    if _harness is not None:
        harness_info = {
            "engine": "deepagents",
            "agents": _harness.agent_count,
            "initialized": _harness.is_initialized,
            "universal_memory": True,
        }
    return JSONResponse({
        "status": "ok",
        "agents": agent_count,
        "teams": len(teams),
        "orchestrator": _orchestrator is not None,
        "harness": harness_info if harness_info else None,
    })


async def serve_screenshot(request: Request) -> FileResponse | JSONResponse:
    """Serve a saved browser screenshot image."""
    filename = request.path_params["filename"]
    filepath = SCREENSHOTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/png")


async def serve_avatar(request: Request) -> FileResponse | JSONResponse:
    """Serve a generated agent avatar SVG."""
    filename = request.path_params["filename"]
    filepath = AVATARS_DIR / filename
    if ".." in filename or not filepath.resolve().is_relative_to(AVATARS_DIR.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/svg+xml")



async def serve_audio(request: Request) -> FileResponse | JSONResponse:
    """Serve a TTS-generated audio file."""
    filename = request.path_params["filename"]
    filepath = AUDIO_DIR / filename
    if ".." in filename or not filepath.resolve().is_relative_to(AUDIO_DIR.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(filepath, media_type="audio/wav")


MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}


async def serve_video(request: Request) -> FileResponse | JSONResponse:
    """Serve a rendered Remotion video or still."""
    filename = request.path_params["filename"]
    filepath = VIDEOS_DIR / filename
    # Prevent path traversal
    if ".." in filename or not filepath.resolve().is_relative_to(VIDEOS_DIR.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    media_type = MIME_TYPES.get(filepath.suffix.lower(), "application/octet-stream")
    return FileResponse(filepath, media_type=media_type)


def _job_to_dict(job) -> dict:
    """Serialize a CronJob to a JSON-safe dict."""
    return {
        "id": job.id,
        "name": job.name,
        "enabled": job.enabled,
        "schedule": {
            "kind": job.schedule.kind,
            "atMs": job.schedule.at_ms,
            "everyMs": job.schedule.every_ms,
            "expr": job.schedule.expr,
            "tz": job.schedule.tz,
        },
        "payload": {
            "kind": job.payload.kind,
            "message": job.payload.message,
            "deliver": job.payload.deliver,
            "channel": job.payload.channel,
            "to": job.payload.to,
        },
        "state": {
            "nextRunAtMs": job.state.next_run_at_ms,
            "lastRunAtMs": job.state.last_run_at_ms,
            "lastStatus": job.state.last_status,
            "lastError": job.state.last_error,
        },
        "createdAtMs": job.created_at_ms,
    }


async def cron_list_jobs(request: Request) -> JSONResponse:
    """GET /api/cron/jobs — list all cron jobs."""
    if not _cron:
        return JSONResponse({"error": "Cron service not initialized"}, status_code=503)
    jobs = _cron.list_jobs(include_disabled=True)
    return JSONResponse({"jobs": [_job_to_dict(j) for j in jobs]})


async def cron_toggle_job(request: Request) -> JSONResponse:
    """POST /api/cron/jobs/{job_id}/toggle — enable/disable a job."""
    if not _cron:
        return JSONResponse({"error": "Cron service not initialized"}, status_code=503)
    job_id = request.path_params["job_id"]
    body = await request.json()
    enabled = body.get("enabled", True)
    job = _cron.enable_job(job_id, enabled=enabled)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse({"ok": True, "job": _job_to_dict(job)})


async def cron_run_job(request: Request) -> JSONResponse:
    """POST /api/cron/jobs/{job_id}/run — force-run a job now."""
    if not _cron:
        return JSONResponse({"error": "Cron service not initialized"}, status_code=503)
    job_id = request.path_params["job_id"]
    ok = await _cron.run_job(job_id, force=True)
    if not ok:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse({"ok": True})


async def cron_delete_job(request: Request) -> JSONResponse:
    """DELETE /api/cron/jobs/{job_id} — delete a job."""
    if not _cron:
        return JSONResponse({"error": "Cron service not initialized"}, status_code=503)
    job_id = request.path_params["job_id"]
    removed = _cron.remove_job(job_id)
    if not removed:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse({"ok": True})


async def agents_status(request: Request) -> JSONResponse:
    """Return registered agents and their availability (for Paperclip integration)."""
    if _orchestrator is None:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=503)
    try:
        agents = []
        for name in _orchestrator.registry.agent_names:
            agent_cfg = _orchestrator.registry.get(name)
            agents.append({
                "name": name,
                "team": getattr(agent_cfg, "team", "unknown"),
                "role": getattr(agent_cfg, "role", "member"),
                "model": getattr(agent_cfg, "model", "unknown"),
                "status": "available",
            })
        return JSONResponse({"agents": agents, "count": len(agents)})
    except Exception as e:
        logger.error(f"agents_status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def serve_shared_asset(request: Request) -> FileResponse | JSONResponse:
    """Serve shared JS/CSS assets that need to be loaded across all apps."""
    filename = request.path_params["filename"]
    # Only allow .js and .css files
    if not filename.endswith((".js", ".css")):
        return JSONResponse({"error": "Not found"}, status_code=404)
    filepath = SHARED_ASSETS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    media_type = "application/javascript" if filename.endswith(".js") else "text/css"
    return FileResponse(filepath, media_type=media_type)


async def agents_list(request: Request) -> JSONResponse:
    """GET /v1/agents/list — Cross-app agent discovery endpoint.

    Returns detailed agent info for all registered agents.
    Used by SV Social (@mention autocomplete), LobeHub (agent catalog),
    and Mission Control (dispatch agent picker).
    """
    if _orchestrator is None:
        return JSONResponse({"error": "Orchestrator not initialized"}, status_code=503)
    try:
        agents = []
        teams_set: set[str] = set()
        for name in _orchestrator.registry.agent_names:
            spec = _orchestrator.registry.get(name)
            team = getattr(spec, "team", "unknown")
            role = getattr(spec, "role", "member")
            description = getattr(spec, "description", "")
            handoffs = list(getattr(spec, "handoffs", []) or [])
            tools = list(getattr(spec, "tools", []) or [])
            teams_set.add(team)

            # Build chat URL for deep linking from any app
            chat_url = f"http://localhost:3180/?agentModel=agent/{name}"

            agents.append({
                "name": name,
                "displayName": name.replace("_", " ").title(),
                "team": team,
                "role": role,
                "description": description,
                "model": f"agent/{name}",
                "status": "available",
                "chatUrl": chat_url,
                "handoffs": handoffs,
                "toolCount": len(tools),
            })

        # Sort: leads first, then alphabetically by team then name
        role_order = {"lead": 0, "memory": 2}
        agents.sort(key=lambda a: (role_order.get(a["role"], 1), a["team"], a["name"]))

        return JSONResponse({
            "agents": agents,
            "count": len(agents),
            "teams": sorted(teams_set),
            "teamCount": len(teams_set),
        })
    except Exception as e:
        logger.error(f"agents_list error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)



async def generate_article_image(request: Request) -> JSONResponse:
    """POST /api/article-image/generate — create cover + body PNGs."""
    from nanobot.services.article_image import generate_article_images

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    headline = body.get("headline")
    body_text = body.get("body_text")
    hero_url = body.get("hero_image_url")
    if not headline or not body_text or not hero_url:
        return JSONResponse(
            {"error": "headline, body_text, and hero_image_url are required"},
            status_code=400,
        )

    try:
        result = await generate_article_images(
            headline=headline,
            body_text=body_text,
            hero_image_url=hero_url,
            category=body.get("category", "ARTICLE"),
        )
    except Exception as exc:
        logger.exception("Failed to generate article images")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse(result)


async def serve_article_image(request: Request) -> FileResponse | JSONResponse:
    """Serve a generated article image."""
    filename = request.path_params["filename"]
    filepath = ARTICLE_IMAGES_DIR / filename
    if ".." in filename or not filepath.resolve().is_relative_to(ARTICLE_IMAGES_DIR.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    media_type = MIME_TYPES.get(filepath.suffix.lower(), "image/png")
    return FileResponse(filepath, media_type=media_type)


async def audio_transcriptions(request: Request) -> JSONResponse:
    """OpenAI-compatible STT endpoint — proxies to Groq Whisper."""
    try:
        form = await request.form()
    except Exception:
        return _openai_error("Invalid multipart form data", status_code=400)

    upload = form.get("file")
    if upload is None:
        return _openai_error("Missing required field: file", param="file")

    response_format = form.get("response_format", "json")

    # Save uploaded file to a temp path
    suffix = Path(upload.filename).suffix if upload.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await upload.read())
        tmp_path = tmp.name

    try:
        from nanobot.providers.transcription import GroqTranscriptionProvider
        provider = GroqTranscriptionProvider()
        text = await provider.transcribe(tmp_path)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return _openai_error(f"Transcription failed: {e}", status_code=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if response_format == "text":
        from starlette.responses import Response
        return Response(text, media_type="text/plain")

    return JSONResponse({"text": text})


_QWEN_SPEAKERS = {"Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"}
_OPENAI_VOICE_MAP = {
    "alloy": "Ryan", "echo": "Ryan", "fable": "Ryan", "onyx": "Ryan", "shimmer": "Ryan",
    "nova": "Vivian",
}


async def audio_speech(request: Request) -> StreamingResponse | JSONResponse:
    """OpenAI-compatible TTS endpoint — generates speech locally via Qwen3-TTS."""
    try:
        body = await request.json()
    except Exception:
        return _openai_error("Invalid JSON body", status_code=400)

    text = body.get("input", "")
    if not text:
        return _openai_error("Missing required field: input", param="input")

    voice = body.get("voice", "Ryan")
    # Map OpenAI voice names → Qwen speaker; pass through if already a Qwen speaker name
    speaker = voice if voice in _QWEN_SPEAKERS else _OPENAI_VOICE_MAP.get(voice, "Ryan")

    tmp_name = f"tts_{uuid.uuid4().hex[:12]}"
    wav_path = AUDIO_DIR / f"{tmp_name}.wav"

    try:
        from nanobot.agent.tools.tts import QwenTTSTool
        tool = QwenTTSTool(AUDIO_DIR)
        result = await tool.execute(text=text, output_name=tmp_name, speaker=speaker)
        if result.startswith("Error"):
            return _openai_error(result, status_code=500)
    except Exception as e:
        logger.error(f"Qwen TTS failed: {e}")
        return _openai_error(f"TTS failed: {e}", status_code=500)

    if not wav_path.exists():
        return _openai_error("TTS produced no output file", status_code=500)

    audio_bytes = wav_path.read_bytes()

    # Convert WAV to MP3 for LobeHub compatibility
    response_format = body.get("response_format", "mp3")
    if response_format == "mp3":
        import subprocess
        mp3_path = wav_path.with_suffix(".mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-q:a", "2", str(mp3_path)],
                capture_output=True, check=True,
            )
            audio_bytes = mp3_path.read_bytes()
            mp3_path.unlink(missing_ok=True)
            media_type = "audio/mpeg"
            filename = "speech.mp3"
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"ffmpeg WAV→MP3 conversion failed, returning WAV: {e}")
            media_type = "audio/wav"
            filename = "speech.wav"
    else:
        media_type = "audio/wav"
        filename = "speech.wav"

    wav_path.unlink(missing_ok=True)

    return StreamingResponse(
        content=iter([audio_bytes]),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ─── Embeddings endpoint (for LobeHub Knowledge Base / RAG) ──────────────
_embedding_model = None


async def embeddings(request: Request) -> JSONResponse:
    """OpenAI-compatible embeddings endpoint using local sentence-transformers."""
    global _embedding_model
    try:
        body = await request.json()
    except Exception:
        return _openai_error("Invalid JSON body", status_code=400)

    texts = body.get("input", "")
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return _openai_error("Missing required field: input", param="input")

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")

    vectors = _embedding_model.encode(texts, normalize_embeddings=True)

    data = [
        {"object": "embedding", "embedding": v.tolist(), "index": i}
        for i, v in enumerate(vectors)
    ]
    return JSONResponse({
        "object": "list",
        "data": data,
        "model": body.get("model", "all-MiniLM-L6-v2"),
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts),
        },
    })




async def memory_api(request: Request) -> JSONResponse:
    """GET/POST /v1/memory — Universal shared memory API.

    GET: Returns full memory snapshot (shared, contacts, decisions, projects, agents, topics)
    POST: Write to memory (target, content, name, mode)
    """
    if not _harness:
        return JSONResponse({"error": "Deep Agent Harness not initialized"}, status_code=503)

    from nanobot.harness.api import handle_memory
    return await handle_memory(request, _harness)


async def memory_context_api(request: Request) -> JSONResponse:
    """GET /v1/memory/context/{agent_name} — Get the scoped context an agent sees."""
    if not _harness:
        return JSONResponse({"error": "Deep Agent Harness not initialized"}, status_code=503)

    agent_name = request.path_params.get("agent_name", "ceo")
    # Resolve team from agent spec for scoped context
    agent_spec = _harness._agents.get(agent_name, {})
    agent_team = agent_spec.get("team")

    context = _harness.memory.build_context_for_agent(agent_name, team=agent_team)
    return JSONResponse({
        "agent": agent_name,
        "team": agent_team,
        "context": context,
        "context_length": len(context),
    })


async def agent_sessions_api(request: Request) -> JSONResponse:
    """GET /v1/agents/{agent_name}/sessions — Get an agent's recent conversation summaries.

    This is how frontends can show conversation history per agent.
    The CEO gets all agents' sessions; individual agents get only their own.
    """
    if not _harness:
        return JSONResponse({"error": "Deep Agent Harness not initialized"}, status_code=503)

    agent_name = request.path_params.get("agent_name", "ceo")
    limit = int(request.query_params.get("limit", "10"))

    if agent_name == "ceo":
        sessions = _harness.memory._get_recent_sessions(limit=limit, agent_name=None)
    else:
        sessions = _harness.memory._get_recent_sessions(limit=limit, agent_name=agent_name)

    return JSONResponse({
        "agent": agent_name,
        "sessions": sessions,
        "count": len(sessions),
    })


async def harness_status(request: Request) -> JSONResponse:
    """GET /v1/harness/status — Deep Agent Harness status and diagnostics."""
    if not _harness:
        return JSONResponse({
            "status": "not_initialized",
            "engine": "deepagents",
            "agents": 0,
        })

    return JSONResponse({
        "status": "ready" if _harness.is_initialized else "initializing",
        "engine": "deepagents",
        "agents": _harness.agent_count,
        "memory": {
            "shared_size": len(_harness.memory.get_shared_context()),
            "contacts_size": len(_harness.memory.get_contacts()),
            "decisions_size": len(_harness.memory.get_decisions()),
            "projects_size": len(_harness.memory.get_projects()),
        },
    })


async def serve_dashboard(request: Request):
    """Serve the mission control dashboard SPA."""
    index_path = GATEWAY_STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "Dashboard not found"}, status_code=404)
    return HTMLResponse(index_path.read_text())


async def gateway_ws(websocket):
    """WebSocket endpoint — delegates to GatewayServer if available."""
    if _gateway is None:
        await websocket.close(1013, "Gateway not initialized")
        return
    await _gateway._ws_endpoint(websocket)


# ── Groups / Agent Teams API ──────────────────────────────────────────────────

_GROUP_CHANNEL_MAP = {
    1: "channel-executive",
    2: "channel-communication",
    3: "channel-content",
    4: "channel-development",
    5: "channel-finance",
    6: "channel-grant",
    7: "channel-research",
    8: "channel-scraping",
}

_GROUP_NAMES = {
    1: "Executive", 2: "Communication", 3: "Content", 4: "Development",
    5: "Finance", 6: "Grant Writing", 7: "Research", 8: "Scraping",
}

_TEAMS = [
    {"id": 1, "name": "Executive", "description": "Strategic leadership and cross-team coordination. The CEO oversees all operations and delegates work across the organization.", "avatar_url": None, "member_count": 3, "tags": ["Leadership", "Strategy", "Operations"], "category": "leadership", "is_public": True, "is_member": True, "channel_slug": "executive", "channel_id": "channel-executive", "agents": ["ceo", "auto-router"]},
    {"id": 2, "name": "Communication", "description": "Email, Slack, WhatsApp, calendar, and social outreach. Manages all inbound and outbound messaging across platforms.", "avatar_url": None, "member_count": 7, "tags": ["Email", "Slack", "WhatsApp", "Calendar"], "category": "operations", "is_public": True, "is_member": True, "channel_slug": "communication", "channel_id": "channel-communication", "agents": ["communication-manager", "email-agent", "slack-agent", "social-agent", "whatsapp-agent", "calendar-agent"]},
    {"id": 3, "name": "Content", "description": "Article research, writing, social media management, and editorial workflow for Street Voices publications.", "avatar_url": None, "member_count": 4, "tags": ["Articles", "Writing", "Social Media"], "category": "creative", "is_public": True, "is_member": True, "channel_slug": "content", "channel_id": "channel-content", "agents": ["content-manager", "article-researcher", "article-writer", "social-media"]},
    {"id": 4, "name": "Development", "description": "Full-stack engineering, database administration, DevOps, and infrastructure management.", "avatar_url": None, "member_count": 5, "tags": ["Engineering", "Backend", "Frontend", "DevOps"], "category": "technical", "is_public": True, "is_member": True, "channel_slug": "development", "channel_id": "channel-development", "agents": ["dev-manager", "backend-dev", "frontend-dev", "database-admin", "devops"]},
    {"id": 5, "name": "Finance", "description": "Financial management, accounting, crypto operations, and budget tracking for Street Voices.", "avatar_url": None, "member_count": 3, "tags": ["Finance", "Accounting", "Crypto"], "category": "operations", "is_public": True, "is_member": True, "channel_slug": "finance", "channel_id": "channel-finance", "agents": ["finance-manager", "accounting-agent", "crypto-agent"]},
    {"id": 6, "name": "Grant Writing", "description": "Grant research, proposal writing, budget planning, and project management for funding applications.", "avatar_url": None, "member_count": 5, "tags": ["Grants", "Proposals", "Budgets"], "category": "operations", "is_public": True, "is_member": True, "channel_slug": "grant", "channel_id": "channel-grant", "agents": ["grant-manager", "grant-writer", "budget-manager", "project-manager"]},
    {"id": 7, "name": "Research", "description": "Media platform analysis, program research, and strategic insights for Street Voices initiatives.", "avatar_url": None, "member_count": 4, "tags": ["Research", "Analysis", "Insights"], "category": "research", "is_public": True, "is_member": True, "channel_slug": "research", "channel_id": "channel-research", "agents": ["research-manager", "media-platform-researcher", "media-program-researcher", "street-bot-researcher"]},
    {"id": 8, "name": "Scraping", "description": "Web scraping, data collection, and automated information gathering from online sources.", "avatar_url": None, "member_count": 3, "tags": ["Data", "Scraping", "Automation"], "category": "technical", "is_public": True, "is_member": True, "channel_slug": "scraping", "channel_id": "channel-scraping", "agents": ["scraping-manager", "scraping-agent"]},
]

_social_pool = None

async def _get_social_db():
    global _social_pool
    if _social_pool is None:
        import asyncpg
        _social_pool = await asyncpg.create_pool(
            "postgresql://lobehub:lobehub_password@localhost:5433/social",
            min_size=1, max_size=5,
        )
    return _social_pool


_SUPABASE_URL = "https://bkxkrjktbqxefgsoavxf.supabase.co"
_SUPABASE_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJreGtyamt0YnF4ZWZnc29hdnhmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjkxNjU1NiwiZXhwIjoyMDc4NDkyNTU2fQ."
    "RXvByoU2sUheesX6VqTbHTlI1HqT7m2W3ZW-EDblGPY"
)


def _supabase_headers() -> dict:
    return {
        "apikey": _SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {_SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


# ── Academy AI Tutor — lightweight LLM chat for the tutor widget ──

# In-memory session store (simple — no persistence needed for tutor chat)
_tutor_sessions: dict[str, dict] = {}


async def _handle_tutor(request: Request, parts: list[str]) -> Response:
    """Handle /api/academy/tutor/* endpoints."""
    # parts: ["tutor", "chat"], ["tutor", "sessions"], etc.
    action = parts[1] if len(parts) > 1 else ""
    method = request.method

    if action == "chat" and method == "POST":
        return await _tutor_chat(request)
    elif action == "sessions" and method == "POST" and len(parts) == 2:
        return await _tutor_start_session(request)
    elif action == "sessions" and method == "GET" and len(parts) == 2:
        return await _tutor_list_sessions(request)
    elif action == "sessions" and len(parts) >= 3:
        session_id = parts[2]
        if len(parts) >= 4 and parts[3] == "end" and method == "POST":
            return _tutor_end_session(session_id, request)
        elif len(parts) >= 4 and parts[3] == "messages":
            return _tutor_get_messages(session_id)
        elif method == "GET":
            return _tutor_get_session(session_id)
    elif action == "explain" and method == "POST":
        return await _tutor_explain(request)
    elif action == "recommendations" and method == "GET":
        return _tutor_recommendations(request)
    elif action == "quick" and len(parts) >= 3:
        if parts[2] == "help" and method == "POST":
            return await _tutor_quick_help(request)
        elif parts[2] == "quiz-prep" and method == "POST":
            return _tutor_quiz_prep(request)

    return JSONResponse({"error": f"Unknown tutor endpoint: {'/'.join(parts)}"}, status_code=404)


async def _tutor_agent_call(user_message: str, session_key: str = "tutor:default") -> str:
    """Route tutor messages through the full AgentLoop (has academy tools)."""
    if not _agent:
        return "The academy agent is not available right now. Please try again later."
    try:
        response = await _agent.process_direct(
            content=user_message,
            session_key=session_key,
            channel="academy-tutor",
            chat_id="tutor",
        )
        return response or "I'm not sure how to help with that. Could you rephrase?"
    except Exception as e:
        logger.error(f"Tutor agent call failed: {e}")
        return "I'm having trouble connecting right now. Please try again in a moment."


async def _tutor_chat(request: Request) -> Response:
    """POST /api/academy/tutor/chat — routes through the full academy agent."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "Message is required"}, status_code=400)

    session_id = body.get("session_id")
    course_id = body.get("course_id")
    lesson_id = body.get("lesson_id")
    user_id = str(request.query_params.get("user_id", "anonymous"))

    # Build the full message with context for the agent
    context_prefix = ""
    if course_id:
        context_prefix += f"[Academy Tutor | Course: {course_id}] "
    if lesson_id:
        context_prefix += f"[Lesson: {lesson_id}] "

    agent_message = f"{context_prefix}{user_message}\n\n(Respond as the Academy Tutor. After your answer, include exactly 3 follow-up suggestions as a JSON array on the last line prefixed with SUGGESTIONS:)"

    # Use session-scoped key so the agent remembers conversation context
    if not session_id:
        session_id = str(uuid.uuid4())
    session_key = f"tutor:{user_id}:{session_id}"

    # Route through the full agent (has academy tools: create course, quiz, etc.)
    raw_response = await _tutor_agent_call(agent_message, session_key=session_key)

    # Parse suggestions from response
    suggestions = []
    message_text = raw_response
    if "SUGGESTIONS:" in raw_response:
        parts = raw_response.rsplit("SUGGESTIONS:", 1)
        message_text = parts[0].strip()
        try:
            suggestions = json.loads(parts[1].strip())
        except Exception:
            pass

    # Track session in memory
    if session_id not in _tutor_sessions:
        _tutor_sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "course_id": course_id,
            "lesson_id": lesson_id,
            "session_type": body.get("session_type", "general"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message_count": 0,
            "messages": [],
        }
    session = _tutor_sessions[session_id]
    session["messages"].append({"role": "user", "content": user_message, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    session["messages"].append({"role": "assistant", "content": message_text, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    session["message_count"] = len(session["messages"])

    return JSONResponse({
        "message": message_text,
        "session_id": session_id,
        "suggestions": suggestions,
    })


async def _tutor_start_session(request: Request) -> Response:
    """POST /api/academy/tutor/sessions — create a new session."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    user_id = request.query_params.get("user_id", "anonymous")
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "user_id": user_id,
        "course_id": body.get("course_id"),
        "lesson_id": body.get("lesson_id"),
        "session_type": body.get("session_type", "general"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message_count": 0,
        "messages": [],
    }
    _tutor_sessions[session_id] = session
    return JSONResponse({k: v for k, v in session.items() if k != "messages"})


async def _tutor_list_sessions(request: Request) -> Response:
    """GET /api/academy/tutor/sessions — list user sessions."""
    user_id = request.query_params.get("user_id", "")
    sessions = [
        {k: v for k, v in s.items() if k != "messages"}
        for s in _tutor_sessions.values()
        if s.get("user_id") == user_id
    ]
    return JSONResponse({"sessions": sessions})


def _tutor_get_session(session_id: str) -> Response:
    """GET /api/academy/tutor/sessions/{id}."""
    session = _tutor_sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse({k: v for k, v in session.items() if k != "messages"})


def _tutor_end_session(session_id: str, request: Request) -> Response:
    """POST /api/academy/tutor/sessions/{id}/end."""
    session = _tutor_sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    session["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return JSONResponse({k: v for k, v in session.items() if k != "messages"})


def _tutor_get_messages(session_id: str) -> Response:
    """GET /api/academy/tutor/sessions/{id}/messages."""
    session = _tutor_sessions.get(session_id)
    if not session:
        return JSONResponse({"messages": []})
    messages = [
        {"id": f"msg-{i}", "session_id": session_id, **m}
        for i, m in enumerate(session.get("messages", []))
    ]
    return JSONResponse({"messages": messages})


async def _tutor_explain(request: Request) -> Response:
    """POST /api/academy/tutor/explain — explain a concept."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    concept = body.get("concept", "").strip()
    if not concept:
        return JSONResponse({"error": "Concept is required"}, status_code=400)

    difficulty = body.get("difficulty_level", "detailed")
    prompt = f"[Academy Tutor] Explain this concept at a {difficulty} level with real-world examples: {concept}"

    explanation = await _tutor_agent_call(prompt)
    if "SUGGESTIONS:" in explanation:
        explanation = explanation.rsplit("SUGGESTIONS:", 1)[0].strip()

    return JSONResponse({"explanation": explanation})


def _tutor_recommendations(request: Request) -> Response:
    """GET /api/academy/tutor/recommendations — static starter recommendations."""
    return JSONResponse({"recommendations": [
        {"recommendation_type": "course", "title": "Speaking Up with Confidence", "description": "Build advocacy skills for any situation", "priority": 1},
        {"recommendation_type": "course", "title": "Know Your Rights", "description": "Understand your fundamental rights", "priority": 2},
        {"recommendation_type": "lesson", "title": "Power Mapping 101", "description": "Learn to identify key decision makers", "priority": 3},
    ]})


async def _tutor_quick_help(request: Request) -> Response:
    """POST /api/academy/tutor/quick/help — quick contextual help."""
    question = request.query_params.get("question", "")
    if not question:
        try:
            body = await request.json()
            question = body.get("question", "")
        except Exception:
            pass
    if not question:
        return JSONResponse({"error": "Question is required"}, status_code=400)

    answer = await _tutor_agent_call(f"[Academy Tutor - Quick Help] Answer briefly in 2-3 sentences: {question}")
    return JSONResponse({"answer": answer, "suggestions": []})


def _tutor_quiz_prep(request: Request) -> Response:
    """POST /api/academy/tutor/quick/quiz-prep — quiz preparation tips."""
    return JSONResponse({
        "tips": [
            "Review the key concepts from each lesson before attempting the quiz",
            "Focus on understanding the 'why' behind each concept, not just memorizing facts",
            "Try explaining concepts in your own words to test your understanding",
        ],
        "struggling_topics": [],
        "progress": {},
    })


async def academy_proxy(request: Request) -> Response:
    """Academy API — direct Supabase REST proxy + AI tutor endpoints."""
    import httpx

    path = request.url.path  # e.g. /api/academy/courses or /api/academy/tutor/chat
    parts = path.replace("/api/academy/", "").strip("/").split("/")

    resource = parts[0] if parts else ""

    # ── AI Tutor endpoints (LLM-powered, not Supabase) ──
    if resource == "tutor":
        return await _handle_tutor(request, parts)

    # ── Academy runtime-backed endpoints ──
    if resource == "live-sessions":
        return await _handle_live_sessions(request, parts)
    if resource in {"assignments", "submissions", "rubrics", "grading"}:
        return await _handle_assignments(request, parts)
    if resource == "users" and len(parts) >= 3 and parts[2] in {"available-assignments", "assignment-stats"}:
        return await _handle_assignments(request, parts)
    if resource == "cohorts":
        return await _handle_cohorts(request, parts)
    if resource == "learning-paths":
        return await _handle_learning_paths(request, parts)
    if resource == "reviews":
        return await _handle_reviews(request, parts)
    if resource == "certificates":
        return await _handle_certificates(request, parts)
    if resource == "enrollments":
        return await _handle_enrollments(request, parts)
    if resource == "materials":
        return await _handle_materials(request, parts)
    if resource == "schedule-items":
        return await _handle_schedule_items(request, parts)
    if resource == "attendance":
        return await _handle_attendance(request, parts)
    if resource == "video":
        return await _handle_video(request, parts)
    if resource == "moodle":
        return await _handle_moodle(request, parts)

    if resource == "courses" and len(parts) >= 3 and parts[2] == "cohorts":
        return await _handle_cohorts(request, ["cohorts", "course", parts[1]])
    if resource == "courses" and len(parts) >= 3 and parts[2] == "live-sessions":
        return await _handle_live_sessions(request, ["live-sessions", "course", parts[1]])
    if resource == "courses" and len(parts) >= 3 and parts[2] == "assignments":
        return await _handle_assignments(request, parts)
    if resource == "courses" and len(parts) >= 3 and parts[2] == "submissions":
        return await _handle_assignments(request, parts)
    if resource == "courses" and len(parts) >= 3 and parts[2] == "schedule-items":
        return await _handle_schedule_items(request, parts)
    if resource == "courses" and len(parts) >= 3 and parts[2] == "attendance":
        return await _handle_attendance(request, parts)

    table_map = {
        "courses": "academy_courses",
        "modules": "academy_modules",
        "lessons": "academy_lessons",
        "quizzes": "academy_quizzes",
        "questions": "academy_quiz_questions",
        "enrollments": "academy_enrollments",
        "submissions": "academy_submissions",
        "discussions": "academy_discussions",
    }
    table = table_map.get(resource)
    if not table:
        return JSONResponse({"error": f"Unknown academy resource: {resource}"}, status_code=404)

    base = f"{_SUPABASE_URL}/rest/v1/{table}"
    headers = _supabase_headers()
    method = request.method

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                # Build Supabase query from request params
                params = dict(request.query_params)
                qs = "select=*"

                # Handle specific resource ID
                if len(parts) >= 2 and parts[1]:
                    qs += f"&id=eq.{parts[1]}"

                # Handle nested resources (e.g. courses/{id}/modules)
                if len(parts) >= 3:
                    nested_resource = parts[2]
                    nested_table = table_map.get(nested_resource)
                    if nested_table:
                        table = nested_table
                        base = f"{_SUPABASE_URL}/rest/v1/{nested_table}"
                        parent_fk = "course_id"
                        if resource == "modules":
                            parent_fk = "module_id"
                        elif resource == "lessons":
                            parent_fk = "lesson_id"
                        qs = f"select=*&{parent_fk}=eq.{parts[1]}"
                        if len(parts) >= 4 and parts[3]:
                            qs += f"&id=eq.{parts[3]}"
                        # Handle deeply nested (courses/{id}/modules/{mid}/lessons)
                        if len(parts) >= 5:
                            deep_resource = parts[4]
                            deep_table = table_map.get(deep_resource)
                            if deep_table:
                                base = f"{_SUPABASE_URL}/rest/v1/{deep_table}"
                                deep_parent_fk = "module_id"
                                if nested_resource == "lessons":
                                    deep_parent_fk = "lesson_id"
                                qs = f"select=*&{deep_parent_fk}=eq.{parts[3]}"

                # Apply filters from query params
                state = params.pop("state", None)
                if state:
                    qs += f"&state=eq.{state}"
                elif resource == "courses" and len(parts) < 2:
                    qs += "&state=eq.published"  # Default to published

                category = params.pop("category", None)
                if category:
                    qs += f"&category=eq.{category}"
                level = params.pop("level", None)
                if level:
                    qs += f"&level=eq.{level}"

                user_id = params.pop("user_id", None)
                if user_id:
                    qs += f"&user_id=eq.{user_id}"
                course_id = params.pop("course_id", None)
                if course_id:
                    qs += f"&course_id=eq.{course_id}"
                instructor_id = params.pop("instructor_id", None)
                if instructor_id:
                    qs += f"&instructor_id=eq.{instructor_id}"

                limit = params.pop("limit", "20")
                skip = params.pop("skip", "0")
                qs += f"&order=created_at.desc&offset={skip}&limit={limit}"

                resp = await client.get(f"{base}?{qs}", headers=headers)
                data = resp.json()

                # Return a single record only for ID-shaped paths such as
                # /resource/:id, /resource/:id/nested/:nestedId, etc.
                is_single_record_request = len(parts) in {2, 4, 6} and bool(parts[-1])
                if is_single_record_request and isinstance(data, list):
                    if not data:
                        return JSONResponse({"detail": "Not found"}, status_code=404)
                    return JSONResponse(data[0])

                return JSONResponse(data)

            elif method == "POST":
                body = await request.body()
                body_json = json.loads(body) if body else {}

                # Inject parent IDs for nested resources
                if len(parts) >= 3:
                    nested_resource = parts[2]
                    nested_table = table_map.get(nested_resource)
                    if nested_table:
                        base = f"{_SUPABASE_URL}/rest/v1/{nested_table}"
                        parent_fk = "course_id"
                        if resource == "modules":
                            parent_fk = "module_id"
                        elif resource == "lessons":
                            parent_fk = "lesson_id"
                        body_json[parent_fk] = parts[1]
                        if len(parts) >= 5:
                            deep_resource = parts[4]
                            deep_table = table_map.get(deep_resource)
                            if deep_table:
                                base = f"{_SUPABASE_URL}/rest/v1/{deep_table}"
                                deep_parent_fk = "module_id"
                                if nested_resource == "lessons":
                                    deep_parent_fk = "lesson_id"
                                body_json[deep_parent_fk] = parts[3]

                resp = await client.post(base, headers=headers, json=body_json)
                data = resp.json()
                if isinstance(data, list) and data:
                    return JSONResponse(data[0], status_code=201)
                return JSONResponse(data, status_code=resp.status_code)

            elif method in ("PATCH", "PUT"):
                if len(parts) < 2:
                    return JSONResponse({"error": "Resource ID required"}, status_code=400)
                resource_id = parts[1]
                body = await request.body()
                url = f"{base}?id=eq.{resource_id}"
                resp = await client.patch(url, headers=headers, content=body)
                data = resp.json()
                if isinstance(data, list) and data:
                    return JSONResponse(data[0])
                return JSONResponse(data, status_code=resp.status_code)

            elif method == "DELETE":
                if len(parts) < 2:
                    return JSONResponse({"error": "Resource ID required"}, status_code=400)
                resource_id = parts[1]
                url = f"{base}?id=eq.{resource_id}"
                resp = await client.delete(url, headers=headers)
                return Response(status_code=204)

    except httpx.ConnectError:
        return JSONResponse({"error": "Supabase not reachable"}, status_code=502)
    except Exception as e:
        logger.error(f"Academy proxy error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def groups_api(request: Request) -> JSONResponse:
    """GET /groups — Return agent teams as community groups."""
    group_id = request.query_params.get("id")
    if group_id:
        team = next((t for t in _TEAMS if t["id"] == int(group_id)), None)
        return JSONResponse([team] if team else [])
    return JSONResponse(_TEAMS)


async def group_messages(request: Request) -> JSONResponse:
    """GET /groups/{group_id}/messages — Fetch messages from a group channel."""
    group_id = int(request.path_params["group_id"])
    channel_id = _GROUP_CHANNEL_MAP.get(group_id)
    if not channel_id:
        return JSONResponse({"error": "Group not found"}, status_code=404)

    pool = await _get_social_db()
    rows = await pool.fetch("""
        SELECT m.id, m.content, m.created_at, m.is_edited, m.is_pinned,
               u.id as author_id, u.username, u.display_name, u.avatar_url, u.is_agent
        FROM messages m JOIN users u ON m.author_id = u.id
        WHERE m.channel_id = $1
        ORDER BY m.created_at ASC
        LIMIT 100
    """, channel_id)

    messages = [{
        "id": r["id"],
        "content": r["content"],
        "createdAt": r["created_at"].isoformat(),
        "isEdited": r["is_edited"],
        "isPinned": r["is_pinned"],
        "author": {
            "id": r["author_id"],
            "username": r["username"],
            "displayName": r["display_name"],
            "avatarUrl": r["avatar_url"],
            "isAgent": r["is_agent"],
        },
    } for r in rows]
    return JSONResponse({"messages": messages})


async def group_send_message(request: Request) -> JSONResponse:
    """POST /groups/{group_id}/messages — Send a message and trigger agent response."""
    group_id = int(request.path_params["group_id"])
    channel_id = _GROUP_CHANNEL_MAP.get(group_id)
    if not channel_id:
        return JSONResponse({"error": "Group not found"}, status_code=404)

    body = await request.json()
    content = body.get("content", "").strip()
    if not content:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    user_id = "cmmq4buiv0000qrrtsv31r71l"
    pool = await _get_social_db()

    # Ensure Joel is a member
    member = await pool.fetchrow(
        "SELECT id FROM channel_members WHERE channel_id = $1 AND user_id = $2",
        channel_id, user_id,
    )
    if not member:
        member_id = f"member-{channel_id}-joel"
        await pool.execute(
            "INSERT INTO channel_members (id, channel_id, user_id, role, joined_at) VALUES ($1, $2, $3, 'member', NOW()) ON CONFLICT (channel_id, user_id) DO NOTHING",
            member_id, channel_id, user_id,
        )

    # Insert message
    msg_id = str(uuid.uuid4())
    await pool.execute("""
        INSERT INTO messages (id, channel_id, author_id, content, created_at, updated_at, is_edited, is_pinned)
        VALUES ($1, $2, $3, $4, NOW(), NOW(), false, false)
    """, msg_id, channel_id, user_id, content)

    # Fetch inserted message
    row = await pool.fetchrow("""
        SELECT m.id, m.content, m.created_at,
               u.id as author_id, u.username, u.display_name, u.avatar_url, u.is_agent
        FROM messages m JOIN users u ON m.author_id = u.id
        WHERE m.id = $1
    """, msg_id)

    msg = {
        "id": row["id"], "content": row["content"],
        "createdAt": row["created_at"].isoformat(),
        "isEdited": False, "isPinned": False,
        "author": {"id": row["author_id"], "username": row["username"],
                    "displayName": row["display_name"], "avatarUrl": row["avatar_url"],
                    "isAgent": row["is_agent"]},
    }

    # Determine which agent(s) should respond
    agents_list = body.get("agents", [])
    mentioned_agents = [a for a in agents_list if f"@{a}" in content]

    # If no specific @mention, the lead agent (first in list) always responds
    if not mentioned_agents and agents_list:
        mentioned_agents = [agents_list[0]]

    group_name = _GROUP_NAMES.get(group_id, f"Group {group_id}")

    async def _trigger_agent(agent_username: str):
        try:
            agent_user = await pool.fetchrow(
                "SELECT id, username, display_name FROM users WHERE username = $1 AND is_agent = true",
                agent_username,
            )
            if not agent_user:
                return
            if _agent is None:
                logger.error("Agent not initialized, cannot process group message")
                return

            prompt = (
                f"You are responding as {agent_user['display_name']} in the {group_name} team group chat. "
                f"Keep your response concise, relevant to your role, and conversational. "
                f"The user said: {content}"
            )
            result = await _agent.process_direct(
                prompt,
                channel=f"group-{group_id}",
                session_key=f"group:{group_id}",
            )

            reply_id = str(uuid.uuid4())
            await pool.execute("""
                INSERT INTO messages (id, channel_id, author_id, content, created_at, updated_at, is_edited, is_pinned)
                VALUES ($1, $2, $3, $4, NOW(), NOW(), false, false)
            """, reply_id, channel_id, agent_user["id"], result or "I couldn't process that request.")
        except Exception as e:
            logger.error(f"Agent response error: {e}")

    for agent in mentioned_agents:
        asyncio.create_task(_trigger_agent(agent))

    return JSONResponse(msg, status_code=201)


async def group_members(request: Request) -> JSONResponse:
    """GET /groups/{group_id}/members — List members of a group channel."""
    group_id = int(request.path_params["group_id"])
    channel_id = _GROUP_CHANNEL_MAP.get(group_id)
    if not channel_id:
        return JSONResponse({"error": "Group not found"}, status_code=404)

    pool = await _get_social_db()
    rows = await pool.fetch("""
        SELECT u.id, u.username, u.display_name, u.avatar_url, u.is_agent
        FROM channel_members cm JOIN users u ON cm.user_id = u.id
        WHERE cm.channel_id = $1
        ORDER BY u.is_agent ASC, u.display_name ASC
    """, channel_id)

    members = [{
        "id": r["id"], "username": r["username"],
        "displayName": r["display_name"], "avatarUrl": r["avatar_url"],
        "isAgent": r["is_agent"], "status": "online",
        "role": "member", "joinedAt": None,
    } for r in rows]
    return JSONResponse({"members": members})


# ── LLM Proxy: share Codex OAuth with external services (e.g. OpenMAIC) ──

async def llm_proxy_completions(request: Request) -> Response:
    """Lightweight OpenAI Chat Completions proxy using Codex OAuth.

    Accepts standard OpenAI /chat/completions requests, forwards them through
    the Codex Responses API, and returns a standard completions response.
    This lets services like OpenMAIC use nanobot's auth without their own key.
    """
    import asyncio as _asyncio
    from nanobot.providers.openai_codex_provider import (
        OpenAICodexProvider,
        _convert_messages,
        _build_headers,
        _strip_model_prefix,
        _request_codex,
        _prompt_cache_key,
        _iter_sse,
    )
    from oauth_cli_kit import get_token as _get_token

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON body"}}, status_code=400)

    messages = body.get("messages", [])
    raw_model = body.get("model", "gpt-4o")
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 4096)
    stream = body.get("stream", False)

    # Map common OpenAI model names to Codex-compatible equivalents
    # Codex OAuth only supports specific model IDs
    _CODEX_MODEL_MAP = {
        "gpt-4o": "gpt-5.4",
        "gpt-4o-mini": "gpt-5.4",
        "gpt-4-turbo": "gpt-5.4",
        "gpt-4": "gpt-5.4",
        "gpt-3.5-turbo": "gpt-5.4",
        "gpt-5.1-codex": "gpt-5.4",
    }
    model = _CODEX_MODEL_MAP.get(raw_model, raw_model)

    # Convert standard messages to Codex Responses format
    system_prompt, input_items = _convert_messages(messages)

    token = await _asyncio.to_thread(_get_token)
    headers = _build_headers(token.account_id, token.access)

    codex_body: dict[str, Any] = {
        "model": _strip_model_prefix(model),
        "store": False,
        "stream": True,
        "instructions": system_prompt,
        "input": input_items,
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": _prompt_cache_key(messages),
    }

    if stream:
        # SSE streaming — translate Codex SSE events to OpenAI streaming format
        async def _stream_proxy():
            import httpx as _httpx
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            try:
                async with _httpx.AsyncClient(timeout=120.0, verify=True) as client:
                    async with client.stream(
                        "POST",
                        "https://chatgpt.com/backend-api/codex/responses",
                        headers=headers,
                        json=codex_body,
                    ) as resp:
                        if resp.status_code != 200:
                            raw = await resp.aread()
                            error_chunk = {
                                "id": request_id, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"content": f"[Error: HTTP {resp.status_code}]"}, "finish_reason": "stop"}],
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        # Emit initial role chunk (required by @ai-sdk/openai)
                        role_chunk = {
                            "id": request_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(role_chunk)}\n\n"

                        async for event in _iter_sse(resp):
                            event_type = event.get("type")
                            if event_type == "response.output_text.delta":
                                delta_text = event.get("delta", "")
                                chunk = {
                                    "id": request_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                            elif event_type == "response.completed":
                                done_chunk = {
                                    "id": request_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                }
                                yield f"data: {json.dumps(done_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return
            except Exception as exc:
                logger.error(f"LLM proxy stream error: {exc}")
                error_chunk = {
                    "id": request_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": f"[Error: {exc}]"}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(_stream_proxy(), media_type="text/event-stream")
    else:
        # Non-streaming — collect full response
        try:
            content, tool_calls, finish_reason = await _request_codex(
                "https://chatgpt.com/backend-api/codex/responses",
                headers, codex_body, verify=True,
            )
        except Exception as e:
            err_str = str(e).upper()
            is_ssl = any(kw in err_str for kw in ("CERTIFICATE_VERIFY_FAILED", "SSL", "TLS"))
            if is_ssl:
                content, tool_calls, finish_reason = await _request_codex(
                    "https://chatgpt.com/backend-api/codex/responses",
                    headers, codex_body, verify=False,
                )
            else:
                return JSONResponse(
                    {"error": {"message": str(e), "type": "server_error"}},
                    status_code=502,
                )

        # Build standard OpenAI response
        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JSONResponse(response_data)


async def llm_proxy_models(request: Request) -> JSONResponse:
    """Return available models for the LLM proxy."""
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": "gpt-5.4", "object": "model", "owned_by": "nanobot-proxy"},
            {"id": "gpt-4o", "object": "model", "owned_by": "nanobot-proxy"},
        ],
    })


# ── Gallery API (local artwork uploads) ──────────────────────────────────────

def _load_gallery_db() -> list[dict]:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    if GALLERY_DB_FILE.exists():
        try:
            return json.loads(GALLERY_DB_FILE.read_text())
        except Exception:
            return []
    return []


def _save_gallery_db(artworks: list[dict]) -> None:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DB_FILE.write_text(json.dumps(artworks, indent=2))


async def gallery_upload(request: Request) -> JSONResponse:
    """POST /gallery/upload — Upload artwork image + metadata."""
    import uuid as _uuid

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = GALLERY_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    form = await request.form()
    image = form.get("image")
    if not image or not hasattr(image, "read"):
        return JSONResponse({"error": "No image file provided"}, status_code=400)

    title = form.get("title", "Untitled")
    description = form.get("description", "")
    artist_name = form.get("artist_name", "Anonymous")
    medium = form.get("medium", "")
    style = form.get("style", "")
    tags = form.get("tags", "")
    user_id = form.get("user_id", "")
    is_for_sale = form.get("is_for_sale", "false").lower() == "true"
    price = form.get("price", "")

    ext = Path(image.filename or "upload.jpg").suffix or ".jpg"
    artwork_id = _uuid.uuid4().hex[:12]
    filename = f"{artwork_id}{ext}"
    filepath = images_dir / filename
    content = await image.read()
    filepath.write_bytes(content)

    artwork = {
        "id": artwork_id,
        "user_id": user_id,
        "artist_name": str(artist_name),
        "title": str(title),
        "description": str(description),
        "medium": str(medium),
        "style": str(style),
        "tags": [t.strip() for t in str(tags).split(",") if t.strip()],
        "image_url": f"/sbapi/gallery/images/{filename}",
        "thumbnail_url": f"/sbapi/gallery/images/{filename}",
        "is_for_sale": is_for_sale,
        "price": float(price) if price else None,
        "currency": "CAD",
        "is_public": True,
        "is_approved": True,
        "view_count": 0,
        "favorite_count": 0,
        "comment_count": 0,
        "share_count": 0,
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }

    artworks = _load_gallery_db()
    artworks.insert(0, artwork)
    _save_gallery_db(artworks)

    logger.info(f"Gallery upload: {title} by {artist_name} ({filename})")
    return JSONResponse(artwork, status_code=201)


async def gallery_list_artworks(request: Request) -> JSONResponse:
    """GET /gallery/artworks — List all artworks."""
    artworks = _load_gallery_db()
    search = request.query_params.get("search", "").lower()
    medium = request.query_params.get("medium", "")
    style = request.query_params.get("style", "")

    filtered = artworks
    if search:
        filtered = [a for a in filtered if search in a.get("title", "").lower()
                     or search in a.get("artist_name", "").lower()
                     or search in a.get("description", "").lower()]
    if medium:
        filtered = [a for a in filtered if a.get("medium", "").lower() == medium.lower()]
    if style:
        filtered = [a for a in filtered if a.get("style", "").lower() == style.lower()]

    return JSONResponse({"artworks": filtered, "total": len(filtered)})


async def gallery_get_artwork(request: Request) -> JSONResponse:
    artwork_id = request.path_params["artwork_id"]
    artworks = _load_gallery_db()
    artwork = next((a for a in artworks if a["id"] == artwork_id), None)
    if not artwork:
        return JSONResponse({"error": "Artwork not found"}, status_code=404)
    return JSONResponse(artwork)


async def gallery_list_uploads(request: Request) -> JSONResponse:
    user_id = request.query_params.get("user_id", "")
    artworks = _load_gallery_db()
    uploads = [a for a in artworks if a.get("user_id") == user_id] if user_id else artworks
    return JSONResponse({"uploads": uploads})


async def gallery_list_mediums(request: Request) -> JSONResponse:
    artworks = _load_gallery_db()
    mediums = sorted(set(a.get("medium", "") for a in artworks if a.get("medium")))
    return JSONResponse({"mediums": mediums})


async def gallery_list_tags(request: Request) -> JSONResponse:
    artworks = _load_gallery_db()
    tag_counts: dict[str, int] = {}
    for a in artworks:
        for tag in a.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:20]
    return JSONResponse({"tags": [{"name": t, "count": c} for t, c in tags]})


async def gallery_serve_image(request: Request) -> FileResponse | JSONResponse:
    filename = request.path_params["filename"]
    images_dir = GALLERY_DIR / "images"
    filepath = images_dir / filename
    if ".." in filename or not filepath.resolve().is_relative_to(images_dir.resolve()):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "Image not found"}, status_code=404)
    suffix = filepath.suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                   ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml"}
    return FileResponse(filepath, media_type=media_types.get(suffix, "application/octet-stream"))


async def gallery_delete_artwork(request: Request) -> JSONResponse:
    artwork_id = request.path_params["artwork_id"]
    artworks = _load_gallery_db()
    artwork = next((a for a in artworks if a["id"] == artwork_id), None)
    if not artwork:
        return JSONResponse({"error": "Not found"}, status_code=404)
    image_url = artwork.get("image_url", "")
    if image_url:
        img_name = image_url.rsplit("/", 1)[-1]
        img_path = GALLERY_DIR / "images" / img_name
        if img_path.exists():
            img_path.unlink()
    artworks = [a for a in artworks if a["id"] != artwork_id]
    _save_gallery_db(artworks)
    return JSONResponse({"ok": True})


# ── News Articles API (JSON-backed CRUD) ─────────────────────────────────────

def _load_news_db() -> list[dict]:
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    if NEWS_DB_FILE.exists():
        try:
            return json.loads(NEWS_DB_FILE.read_text())
        except Exception:
            return []
    return []


def _save_news_db(articles: list[dict]) -> None:
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_DB_FILE.write_text(json.dumps(articles, indent=2))


async def news_create_article(request: Request) -> JSONResponse:
    """POST /news/articles — Create a draft article."""
    body = await request.json()
    now = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    article = {
        "id": uuid.uuid4().hex[:12],
        "title": body.get("title", "Untitled"),
        "slug": body.get("slug", ""),
        "excerpt": body.get("excerpt", ""),
        "content": body.get("content", ""),
        "content_blocks": body.get("content_blocks", []),
        "category": body.get("category", ""),
        "tags": body.get("tags", []),
        "image_url": body.get("image_url", ""),
        "image_credit": body.get("image_credit", ""),
        "image_caption": body.get("image_caption", ""),
        "seo_meta": body.get("seo_meta", ""),
        "seo_keywords": body.get("seo_keywords", ""),
        "seo_hashtags": body.get("seo_hashtags", ""),
        "status": body.get("status", "draft"),
        "source_urls": body.get("source_urls", []),
        "ai_generated": body.get("ai_generated", False),
        "created_at": now,
        "updated_at": now,
        "published_at": None,
    }
    articles = _load_news_db()
    articles.insert(0, article)
    _save_news_db(articles)
    logger.info(f"News: created article '{article['title']}' ({article['id']})")
    return JSONResponse(article, status_code=201)


async def news_list_articles(request: Request) -> JSONResponse:
    """GET /news/articles — List articles with optional status/limit filter."""
    articles = _load_news_db()
    status = request.query_params.get("status", "")
    limit = int(request.query_params.get("limit", "100"))
    if status:
        articles = [a for a in articles if a.get("status") == status]
    return JSONResponse({"articles": articles[:limit], "total": len(articles)})


async def news_get_article(request: Request) -> JSONResponse:
    """GET /news/articles/{article_id} — Get single article."""
    article_id = request.path_params["article_id"]
    articles = _load_news_db()
    article = next((a for a in articles if a["id"] == article_id), None)
    if not article:
        return JSONResponse({"error": "Article not found"}, status_code=404)
    return JSONResponse(article)


async def news_update_article(request: Request) -> JSONResponse:
    """PATCH /news/articles/{article_id} — Update article fields."""
    article_id = request.path_params["article_id"]
    body = await request.json()
    articles = _load_news_db()
    article = next((a for a in articles if a["id"] == article_id), None)
    if not article:
        return JSONResponse({"error": "Article not found"}, status_code=404)
    for key in ("title", "slug", "excerpt", "content", "content_blocks", "category",
                "tags", "image_url", "image_credit", "image_caption",
                "seo_meta", "seo_keywords", "seo_hashtags", "status", "source_urls"):
        if key in body:
            article[key] = body[key]
    article["updated_at"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    if body.get("status") == "published" and not article.get("published_at"):
        article["published_at"] = article["updated_at"]
    _save_news_db(articles)
    return JSONResponse(article)


async def news_delete_article(request: Request) -> JSONResponse:
    """DELETE /news/articles/{article_id} — Delete an article."""
    article_id = request.path_params["article_id"]
    articles = _load_news_db()
    before = len(articles)
    articles = [a for a in articles if a["id"] != article_id]
    if len(articles) == before:
        return JSONResponse({"error": "Not found"}, status_code=404)
    _save_news_db(articles)
    return JSONResponse({"ok": True})


def _parse_article_from_agent_response(text: str, category: str) -> dict:
    """Extract article fields from agent markdown response."""
    import re

    lines = text.strip().split("\n")
    title = ""
    excerpt = ""
    body_lines: list[str] = []
    image_url = ""
    image_caption = ""
    source_urls: list[str] = []
    seo_meta = ""
    seo_keywords = ""
    seo_hashtags = ""
    in_sources = False
    in_body = False
    in_seo = False

    for line in lines:
        stripped = line.strip()

        # Extract hero image with caption
        img_match = re.match(r"!\[([^\]]*)\]\((https?://[^\s)]+)\)", stripped)
        if img_match and not image_url:
            image_caption = img_match.group(1).strip()
            image_url = img_match.group(2)
            continue

        # Extract sources
        if stripped.lower().startswith("**sources visited"):
            in_sources = True
            in_seo = False
            continue
        if in_sources:
            url_match = re.match(r"[-*]\s*(https?://\S+)", stripped)
            if url_match:
                source_urls.append(url_match.group(1))
                continue
            if stripped == "---" or stripped.startswith("#"):
                in_sources = False

        # Extract SEO section
        if stripped.lower().startswith("**seo"):
            in_seo = True
            in_body = False
            continue
        if in_seo:
            meta_match = re.match(r"[-*]\s*Meta:\s*(.+)", stripped, re.IGNORECASE)
            kw_match = re.match(r"[-*]\s*Keywords?:\s*(.+)", stripped, re.IGNORECASE)
            ht_match = re.match(r"[-*]\s*Hashtags?:\s*(.+)", stripped, re.IGNORECASE)
            if meta_match:
                seo_meta = meta_match.group(1).strip()
                continue
            if kw_match:
                seo_keywords = kw_match.group(1).strip()
                continue
            if ht_match:
                seo_hashtags = ht_match.group(1).strip()
                continue
            if stripped.startswith("Saved to:") or stripped == "---" or stripped == "":
                continue

        # Extract title (first # heading after sources)
        if stripped.startswith("# ") and not title:
            title = stripped.lstrip("# ").strip()
            in_body = True
            continue

        # Collect body (stop before SEO section)
        if in_body and not in_seo:
            if stripped.startswith("Saved to:") or stripped.startswith("Want me to publish"):
                continue
            if stripped == "---":
                continue
            body_lines.append(line)

    body = "\n".join(body_lines).strip()
    # Generate excerpt from first paragraph
    if body and not excerpt:
        first_para = body.split("\n\n")[0].replace("\n", " ").strip()
        excerpt = first_para[:200] + ("..." if len(first_para) > 200 else "")

    # Auto-generate SEO meta if AI didn't provide one
    if not seo_meta and excerpt:
        seo_meta = excerpt[:150]

    # Convert markdown body to HTML for the block editor
    html_body = _markdown_to_html(body) if body else ""

    return {
        "title": title or f"Untitled {category} Article",
        "excerpt": excerpt,
        "content": html_body,
        "image_url": image_url,
        "image_caption": image_caption,
        "source_urls": source_urls,
        "seo_meta": seo_meta,
        "seo_keywords": seo_keywords,
        "seo_hashtags": seo_hashtags,
    }


def _markdown_to_html(md: str) -> str:
    """Convert simple markdown to HTML for the block editor."""
    import re as _re

    html_parts: list[str] = []
    paragraphs = md.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Headings
        heading_match = _re.match(r"^(#{1,6})\s+(.+)$", para)
        if heading_match:
            level = len(heading_match.group(1))
            text = _inline_md_to_html(heading_match.group(2).strip())
            html_parts.append(f"<h{level}>{text}</h{level}>")
            continue

        # Unordered lists
        if _re.match(r"^[-*]\s", para):
            items = _re.findall(r"[-*]\s+(.+)", para)
            li_html = "".join(f"<li>{_inline_md_to_html(item)}</li>" for item in items)
            html_parts.append(f"<ul>{li_html}</ul>")
            continue

        # Ordered lists
        if _re.match(r"^\d+\.\s", para):
            items = _re.findall(r"\d+\.\s+(.+)", para)
            li_html = "".join(f"<li>{_inline_md_to_html(item)}</li>" for item in items)
            html_parts.append(f"<ol>{li_html}</ol>")
            continue

        # Blockquote
        if para.startswith(">"):
            quote_text = _re.sub(r"^>\s*", "", para, flags=_re.MULTILINE)
            html_parts.append(f"<blockquote>{_inline_md_to_html(quote_text)}</blockquote>")
            continue

        # Regular paragraph (may span multiple lines)
        text = para.replace("\n", " ")
        html_parts.append(f"<p>{_inline_md_to_html(text)}</p>")

    return "\n".join(html_parts)


def _inline_md_to_html(text: str) -> str:
    """Convert inline markdown (bold, italic, links) to HTML."""
    import re as _re
    # Links: [text](url)
    text = _re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', text)
    # Bold: **text** or __text__
    text = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = _re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    # Italic: *text* or _text_
    text = _re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = _re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", text)
    return text


async def _fetch_pexels_image(query: str) -> tuple[str, str]:
    """Fetch a stock photo from Pexels for the given query.
    Returns (image_url, credit_text) or ("", "") on failure."""
    import httpx
    try:
        config = load_config()
        api_key = config.tools.stock_photos.api_key if hasattr(config.tools, "stock_photos") else ""
        if not api_key:
            # Fallback: read from config.json directly
            import json as _json
            cfg_path = Path.home() / ".nanobot" / "config.json"
            if cfg_path.exists():
                cfg = _json.loads(cfg_path.read_text())
                api_key = cfg.get("tools", {}).get("stockPhotos", {}).get("apiKey", "")
        if not api_key:
            return "", ""

        # Use first few meaningful words from query
        search_terms = " ".join(query.split()[:5])
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.pexels.com/v1/search",
                params={"query": search_terms, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": api_key},
            )
            if resp.status_code != 200:
                logger.warning(f"Pexels API returned {resp.status_code}")
                return "", ""
            data = resp.json()
            photos = data.get("photos", [])
            if not photos:
                return "", ""
            photo = photos[0]
            image_url = photo.get("src", {}).get("landscape", "") or photo.get("src", {}).get("large", "")
            photographer = photo.get("photographer", "Unknown")
            photo_url = photo.get("url", "")
            credit = f"Photo by {photographer} on Pexels"
            return image_url, credit
    except Exception as e:
        logger.warning(f"Pexels image fetch failed: {e}")
        return "", ""


def _is_valid_image_url(url: str) -> bool:
    """Check if a URL is from a known image CDN that allows hotlinking."""
    if not url:
        return False
    lower = url.lower()
    # Only trust known image CDNs that allow embedding
    trusted_cdns = (
        "images.pexels.com", "images.unsplash.com", "i.imgur.com",
        "cloudinary.com", "res.cloudinary.com", "cdn.pixabay.com",
    )
    return any(cdn in lower for cdn in trusted_cdns)


_DAILY_CATEGORY_CONFIG = {
    "local": {
        "label": "Local",
        "focus": "Toronto / Greater Toronto Area (GTA), Ontario, Canada",
        "examples": "city politics, community events, local economy, culture, arts and entertainment, education, health, transit, housing, food, sports, social justice, environment, technology startups, film and music scenes",
    },
    "national": {
        "label": "National",
        "focus": "Canada-wide",
        "examples": "federal politics, economy, immigration, Indigenous rights, healthcare, education policy, culture, Canadian film and music, sports, environment, technology, social issues, cost of living, elections",
    },
    "international": {
        "label": "International",
        "focus": "Global",
        "examples": "world politics, global economy, climate change, technology, culture, film, music, sports, human rights, health, conflict, diplomacy, science, innovation, social movements",
    },
}


def _build_article_prompt(cfg: dict, cat: str, today: str) -> str:
    return (
        f"You are the Article Writer for Street Voices (streetvoices.ca), "
        f"a media platform covering a wide range of topics including politics, economy, culture, "
        f"entertainment, health, social justice, technology, sports, environment, and community stories.\n\n"
        f"CATEGORY: {cfg['label']}\n"
        f"GEOGRAPHIC FOCUS: {cfg['focus']}\n"
        f"POSSIBLE TOPICS: {cfg['examples']}\n"
        f"DATE: {today}\n\n"
        f"IMPORTANT: Choose a topic that is DIFFERENT from housing or homelessness. "
        f"Pick the most interesting and trending story right now from ANY of the possible topics listed above. "
        f"Variety is key — cover politics, economy, culture, entertainment, health, tech, sports, or environment.\n\n"
        f"WRITING STYLE:\n"
        f"- Write like a journalist for a community-focused outlet — clear, engaging, and accessible.\n"
        f"- Headlines should be direct and compelling.\n"
        f"- Open with the most important or interesting angle to hook the reader.\n"
        f"- Keep language conversational but informed — this is journalism, not academic writing.\n"
        f"- Include real data, quotes from sources, and specific details that bring the story to life.\n"
        f"- End with a forward-looking angle — what happens next, or what readers should know.\n"
        f"- Do NOT use '## Community angle' or '## Looking forward' as literal section headers — "
        f"weave these naturally into the article flow.\n"
        f"- 500-800 words total.\n\n"
        f"STEPS:\n"
        f"1. Use web_search to find the most interesting, trending, newsworthy story from the past 48 hours. "
        f"Search broadly — NOT just housing or homelessness.\n"
        f"2. Use web_fetch on the best result to get details.\n"
        f"3. Write the full article.\n"
        f"4. Use image_search ONCE to find a hero image.\n"
        f"5. Use file_write to save to output/articles/{today}-{cat}.md with YAML frontmatter.\n\n"
        f"RESPOND WITH THIS EXACT FORMAT:\n"
        f"**Sources visited:**\n- [list URLs]\n\n"
        f"![One-sentence image caption describing what the image shows](image_url)\n"
        f"*Source: credit*\n\n"
        f"# Headline\n\nFull article body...\n\n"
        f"---\n\n"
        f"**SEO:**\n"
        f"- Meta: [140-150 character description suitable for search engines]\n"
        f"- Keywords: [5-8 relevant SEO keywords, comma separated]\n"
        f"- Hashtags: [3-5 relevant hashtags for social media, e.g. #Toronto #Housing]\n\n"
        f"Saved to: `output/articles/{today}-{cat}.md`"
    )


async def news_generate_daily(request: Request) -> StreamingResponse:
    """POST /news/generate-daily — SSE stream with progress updates."""
    if not _agent:
        return JSONResponse({"error": "Agent not initialized"}, status_code=503)

    async def _stream():
        now = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        today = __import__("datetime").date.today().isoformat()
        categories = list(_DAILY_CATEGORY_CONFIG.items())
        total = len(categories)
        results: list[dict] = []

        if not _agent._mcp_connected:
            _agent._mcp_connected = True

        for idx, (cat, cfg) in enumerate(categories):
            step = idx + 1
            label = cfg["label"]

            # --- Phase 1: Researching ---
            yield f"data: {json.dumps({'step': step, 'total': total, 'category': label, 'phase': 'researching', 'message': f'Researching {label} news stories...'})}\n\n"

            try:
                prompt = _build_article_prompt(cfg, cat, today)
                logger.info(f"News generate-daily: starting {cat} article...")

                # --- Phase 2: Writing ---
                yield f"data: {json.dumps({'step': step, 'total': total, 'category': label, 'phase': 'writing', 'message': f'Writing {label} article...'})}\n\n"

                response = await asyncio.wait_for(
                    _agent.process_direct(
                        content=prompt,
                        session_key=f"news:generate-daily:{cat}:{today}",
                        channel="api",
                        chat_id="news-dashboard",
                    ),
                    timeout=300,
                )
                response_text = response if isinstance(response, str) else (response.content if hasattr(response, "content") else str(response))

                parsed = _parse_article_from_agent_response(response_text, cat)

                # --- Phase 3: Finding image ---
                yield f"data: {json.dumps({'step': step, 'total': total, 'category': label, 'phase': 'image', 'message': f'Finding cover image for {label} article...'})}\n\n"

                image_url = parsed["image_url"]
                image_credit = ""
                if not _is_valid_image_url(image_url):
                    search_q = parsed["title"] or f"{cfg['label']} {cfg['focus']}"
                    image_url, image_credit = await _fetch_pexels_image(search_q)

                # --- Phase 4: Saving ---
                yield f"data: {json.dumps({'step': step, 'total': total, 'category': label, 'phase': 'saving', 'message': f'Saving {label} article...'})}\n\n"

                slug = (
                    parsed["title"].lower()
                    .replace(" ", "-")
                    .replace("'", "")[:60]
                    + "-" + uuid.uuid4().hex[:6]
                )
                article = {
                    "id": uuid.uuid4().hex[:12],
                    "title": parsed["title"],
                    "slug": slug,
                    "excerpt": parsed["excerpt"],
                    "content": parsed["content"],
                    "content_blocks": [],
                    "category": cat.capitalize(),
                    "tags": [cat],
                    "image_url": image_url,
                    "image_credit": image_credit,
                    "image_caption": parsed.get("image_caption", ""),
                    "seo_meta": parsed.get("seo_meta", ""),
                    "seo_keywords": parsed.get("seo_keywords", ""),
                    "seo_hashtags": parsed.get("seo_hashtags", ""),
                    "status": "draft",
                    "source_urls": parsed["source_urls"],
                    "ai_generated": True,
                    "created_at": now,
                    "updated_at": now,
                    "published_at": None,
                }
                db = _load_news_db()
                db.insert(0, article)
                _save_news_db(db)
                results.append(article)
                logger.info(f"News generate-daily: {cat} article saved — '{article['title']}'")

                # --- Phase 5: Done ---
                done_title = parsed["title"][:60]
                done_evt = {"step": step, "total": total, "category": label, "phase": "done", "message": f"{label} article complete: {done_title}", "title": parsed["title"]}
                yield f"data: {json.dumps(done_evt)}\n\n"

            except asyncio.TimeoutError:
                err_evt = {"step": step, "total": total, "category": label, "phase": "error", "message": f"{label} article timed out (5 min limit)"}
                yield f"data: {json.dumps(err_evt)}\n\n"
                results.append({"category": label, "error": "Timed out"})
            except Exception as e:
                logger.exception(f"News generate-daily: failed for {cat}")
                err_msg = str(e)[:60]
                err_evt = {"step": step, "total": total, "category": label, "phase": "error", "message": f"{label} article failed: {err_msg}"}
                yield f"data: {json.dumps(err_evt)}\n\n"
                results.append({"category": label, "error": str(e)})

        # Final event
        yield f"data: {json.dumps({'step': total, 'total': total, 'phase': 'complete', 'message': 'All articles generated!', 'articles': results})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


async def news_generate_custom(request: Request) -> JSONResponse:
    """POST /news/generate-custom — Generate a single custom article by type/topic."""
    if not _agent:
        return JSONResponse({"error": "Agent not initialized"}, status_code=503)

    body = await request.json()
    article_type = body.get("type", "General")
    topic = body.get("topic", "")
    description = body.get("description", "")

    if not topic.strip():
        return JSONResponse({"error": "Topic is required"}, status_code=400)

    now = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    today = __import__("datetime").date.today().isoformat()

    # Skip MCP connection
    if not _agent._mcp_connected:
        _agent._mcp_connected = True

    try:
        desc_line = f"\nADDITIONAL CONTEXT: {description}" if description else ""
        prompt = (
            f"You are the Article Writer for Street Voices (streetvoices.ca), "
            f"a social enterprise and media platform covering housing, social justice, "
            f"community stories, and cultural issues in the Greater Toronto Area and beyond.\n\n"
            f"ARTICLE TYPE: {article_type}\n"
            f"TOPIC: {topic}\n"
            f"DATE: {today}\n"
            f"{desc_line}\n\n"
            f"WRITING STYLE:\n"
            f"- Write like a journalist for a community-focused outlet — clear, engaging, accessible.\n"
            f"- Headlines should be direct and compelling.\n"
            f"- Open with the most important or interesting angle.\n"
            f"- Include real data, quotes, and specific details.\n"
            f"- Use person-first language.\n"
            f"- 500-800 words total.\n\n"
            f"STEPS:\n"
            f"1. Use web_search to find relevant recent information about this topic.\n"
            f"2. Use web_fetch on the best result to get details.\n"
            f"3. Write the full article.\n"
            f"4. Use image_search ONCE to find a hero image.\n"
            f"5. Use file_write to save to output/articles/{today}-custom.md with YAML frontmatter.\n\n"
            f"RESPOND WITH THIS EXACT FORMAT:\n"
            f"**Sources visited:**\n- [list URLs]\n\n"
            f"![One-sentence image caption describing what the image shows](image_url)\n"
            f"*Source: credit*\n\n"
            f"# Headline\n\nFull article body...\n\n"
            f"---\n\n"
            f"**SEO:**\n"
            f"- Meta: [140-150 character description suitable for search engines]\n"
            f"- Keywords: [5-8 relevant SEO keywords, comma separated]\n"
            f"- Hashtags: [3-5 relevant hashtags for social media, e.g. #Toronto #Culture]\n\n"
            f"Saved to: `output/articles/{today}-custom.md`"
        )

        logger.info(f"News generate-custom: {article_type} — {topic[:50]}")
        response = await asyncio.wait_for(
            _agent.process_direct(
                content=prompt,
                session_key=f"news:generate-custom:{today}:{uuid.uuid4().hex[:6]}",
                channel="api",
                chat_id="news-dashboard",
            ),
            timeout=300,
        )
        response_text = response if isinstance(response, str) else (response.content if hasattr(response, "content") else str(response))

        parsed = _parse_article_from_agent_response(response_text, article_type)

        # Ensure a valid image — fallback to Pexels stock photo
        image_url = parsed["image_url"]
        image_credit = ""
        if not _is_valid_image_url(image_url):
            image_url, image_credit = await _fetch_pexels_image(parsed["title"] or topic)
            if image_url:
                logger.info(f"News custom: fetched Pexels image — {image_credit}")

        slug = (
            parsed["title"].lower()
            .replace(" ", "-")
            .replace("'", "")[:60]
            + "-"
            + uuid.uuid4().hex[:6]
        )
        article = {
            "id": uuid.uuid4().hex[:12],
            "title": parsed["title"],
            "slug": slug,
            "excerpt": parsed["excerpt"],
            "content": parsed["content"],
            "content_blocks": [],
            "category": article_type,
            "tags": [article_type.lower()],
            "image_url": image_url,
            "image_credit": image_credit,
            "image_caption": parsed.get("image_caption", ""),
            "seo_meta": parsed.get("seo_meta", ""),
            "seo_keywords": parsed.get("seo_keywords", ""),
            "seo_hashtags": parsed.get("seo_hashtags", ""),
            "status": "draft",
            "source_urls": parsed["source_urls"],
            "ai_generated": True,
            "created_at": now,
            "updated_at": now,
            "published_at": None,
        }
        articles = _load_news_db()
        articles.insert(0, article)
        _save_news_db(articles)
        logger.info(f"News generate-custom: saved '{article['title']}'")
        return JSONResponse(article, status_code=201)

    except asyncio.TimeoutError:
        return JSONResponse({"error": "Generation timed out (5 min limit)"}, status_code=504)
    except Exception as e:
        logger.exception("News generate-custom: failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def news_generate_image(request: Request) -> JSONResponse:
    """GET /news/generate-image?query=... — Fetch a new stock photo from Pexels."""
    import random as _random
    import httpx

    query = request.query_params.get("query", "news article")
    try:
        cfg_path = Path.home() / ".nanobot" / "config.json"
        api_key = ""
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            api_key = cfg.get("tools", {}).get("stockPhotos", {}).get("apiKey", "")
        if not api_key:
            return JSONResponse({"error": "Pexels API key not configured"}, status_code=500)

        search_terms = " ".join(query.split()[:6])
        # Fetch multiple results and pick a random one for variety
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.pexels.com/v1/search",
                params={"query": search_terms, "per_page": 15, "orientation": "landscape"},
                headers={"Authorization": api_key},
            )
            if resp.status_code != 200:
                return JSONResponse({"error": f"Pexels API error: {resp.status_code}"}, status_code=502)
            data = resp.json()
            photos = data.get("photos", [])
            if not photos:
                return JSONResponse({"error": "No images found for this query"}, status_code=404)
            photo = _random.choice(photos)
            image_url = photo.get("src", {}).get("landscape", "") or photo.get("src", {}).get("large", "")
            photographer = photo.get("photographer", "Unknown")
            credit = f"Photo by {photographer} on Pexels"
            return JSONResponse({"image_url": image_url, "credit": credit})
    except Exception as e:
        logger.exception("news_generate_image failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def _register_daily_news_cron() -> None:
    """Register the daily news generation cron job if not already present."""
    if not _cron:
        return
    from nanobot.cron.types import CronSchedule
    store = _cron._load_store()
    for job in store.jobs:
        if job.name == "daily-news-pipeline":
            logger.info("Cron: daily-news-pipeline already registered")
            return
    _cron.add_job(
        name="daily-news-pipeline",
        schedule=CronSchedule(kind="cron", expr="0 7 * * *", tz="America/Toronto"),
        message=(
            "Run the daily news pipeline. Research and write 3 articles: "
            "1 local (Toronto/GTA), 1 national (Canada), 1 international (global). "
            "Save each as a draft."
        ),
        channel="api",
        to="news-dashboard",
    )
    logger.info("Cron: registered daily-news-pipeline (7 AM Toronto)")


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/v1/audio/transcriptions", audio_transcriptions, methods=["POST"]),
        Route("/v1/audio/speech", audio_speech, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/screenshots/{filename:path}", serve_screenshot, methods=["GET"]),
        Route("/videos/{filename:path}", serve_video, methods=["GET"]),
        Route("/audio/{filename:path}", serve_audio, methods=["GET"]),
        Route("/avatars/{filename:path}", serve_avatar, methods=["GET"]),
        Route("/api/cron/jobs", cron_list_jobs, methods=["GET"]),
        Route("/api/cron/jobs/{job_id}/toggle", cron_toggle_job, methods=["POST"]),
        Route("/api/cron/jobs/{job_id}/run", cron_run_job, methods=["POST"]),
        Route("/api/cron/jobs/{job_id}", cron_delete_job, methods=["DELETE"]),
        Route("/v1/agents/status", agents_status, methods=["GET"]),
        Route("/v1/agents/list", agents_list, methods=["GET"]),
        Route("/shared/{filename:path}", serve_shared_asset, methods=["GET"]),
        Route("/api/article-image/generate", generate_article_image, methods=["POST"]),
        Route("/article-images/{filename:path}", serve_article_image, methods=["GET"]),
        # ── LLM Proxy (Codex OAuth passthrough for OpenMAIC etc.) ──
        Route("/v1/llm-proxy/chat/completions", llm_proxy_completions, methods=["POST"]),
        Route("/v1/llm-proxy/models", llm_proxy_models, methods=["GET"]),
        # ── Academy proxy to SBP backend ──
        Route("/api/academy/{path:path}", academy_proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]),
        # ── News Articles CRUD ──
        Route("/news/articles", news_create_article, methods=["POST"]),
        Route("/news/articles", news_list_articles, methods=["GET"]),
        Route("/news/articles/{article_id}", news_get_article, methods=["GET"]),
        Route("/news/articles/{article_id}", news_update_article, methods=["PATCH"]),
        Route("/news/articles/{article_id}", news_delete_article, methods=["DELETE"]),
        Route("/news/generate-daily", news_generate_daily, methods=["POST"]),
        Route("/news/generate-custom", news_generate_custom, methods=["POST"]),
        Route("/news/generate-image", news_generate_image, methods=["GET"]),
        # ── Gallery (local artwork uploads) ──
        Route("/gallery/upload", gallery_upload, methods=["POST"]),
        Route("/gallery/artworks/mediums", gallery_list_mediums, methods=["GET"]),
        Route("/gallery/artworks/{artwork_id}", gallery_get_artwork, methods=["GET"]),
        Route("/gallery/artworks/{artwork_id}", gallery_delete_artwork, methods=["DELETE"]),
        Route("/gallery/artworks", gallery_list_artworks, methods=["GET"]),
        Route("/gallery/uploads", gallery_list_uploads, methods=["GET"]),
        Route("/gallery/tags", gallery_list_tags, methods=["GET"]),
        Route("/gallery/images/{filename:path}", gallery_serve_image, methods=["GET"]),
        # ── Groups / Agent Teams ──
        Route("/groups", groups_api, methods=["GET"]),
        Route("/groups/{group_id:int}/messages", group_messages, methods=["GET"]),
        Route("/groups/{group_id:int}/messages", group_send_message, methods=["POST"]),
        Route("/groups/{group_id:int}/members", group_members, methods=["GET"]),
        # ── Gallery API ──
        Route("/gallery/collections/saved", _gallery_saved_collections, methods=["GET"]),
        Route("/gallery/collections/save", _gallery_save_collection, methods=["POST"]),
        Route("/gallery/collections/save", _gallery_unsave_collection, methods=["DELETE"]),
        Route("/gallery/comments", _gallery_list_comments, methods=["GET"]),
        Route("/gallery/comments", _gallery_post_comment, methods=["POST"]),
        Route("/gallery/comments", _gallery_edit_comment, methods=["PUT"]),
        Route("/gallery/comments", _gallery_delete_comment, methods=["DELETE"]),
        Route("/gallery/artworks/mediums", _gallery_list_mediums, methods=["GET"]),
        Route("/gallery/artworks/{artwork_id:path}/favorites", _gallery_add_favorite, methods=["POST"]),
        Route("/gallery/artworks/{artwork_id:path}/favorites", _gallery_remove_favorite, methods=["DELETE"]),
        Route("/gallery/artworks/{artwork_id:path}", _gallery_get_artwork, methods=["GET"]),
        Route("/gallery/artworks", _gallery_create_artwork, methods=["POST"]),
        Route("/gallery/artworks", _gallery_list_artworks, methods=["GET"]),
        Route("/gallery/user/{user_id:path}/artwork-favorites", _gallery_user_favorites, methods=["GET"]),
        Route("/gallery/users/{user_id:path}/favorites", _gallery_user_favorites_legacy, methods=["GET"]),
        Route("/gallery/tags", _gallery_tags, methods=["GET"]),
        Route("/gallery/uploads", _gallery_uploads, methods=["GET"]),
        Route("/street-profiles/batch-lookup", _street_profiles_batch_lookup, methods=["POST"]),
        # ── Deep Agent Harness endpoints ──
        Route("/v1/memory", memory_api, methods=["GET", "POST"]),
        Route("/v1/memory/context/{agent_name}", memory_context_api, methods=["GET"]),
        Route("/v1/agents/{agent_name}/sessions", agent_sessions_api, methods=["GET"]),
        Route("/v1/harness/status", harness_status, methods=["GET"]),
        # ── Mission Control dashboard ──
        Route("/dashboard", serve_dashboard, methods=["GET"]),
        Mount("/static", app=StaticFiles(directory=str(GATEWAY_STATIC_DIR)), name="gateway-static"),
        # ── WebSocket gateway ──
        WebSocketRoute("/ws", gateway_ws),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        ),
    ],
    lifespan=lifespan,
)
