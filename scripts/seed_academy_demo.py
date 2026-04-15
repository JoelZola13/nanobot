#!/usr/bin/env python3
from __future__ import annotations

import json
import base64
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


RUNTIME_FILE = Path.home() / ".nanobot" / "workspace" / "academy" / "runtime.json"
ACADEMY_KEYS = (
    "enrollments",
    "attendance_records",
    "learning_paths",
    "live_sessions",
    "session_registrations",
    "session_polls",
    "poll_responses",
    "session_questions",
    "session_feedback",
    "cohorts",
    "cohort_enrollments",
    "cohort_deadlines",
    "cohort_announcements",
    "reviews",
    "certificates",
    "course_materials",
    "course_schedule_items",
    "assignments",
    "submissions",
    "rubrics",
    "rubric_grades",
    "video_progress",
    "video_bookmarks",
    "video_notes",
    "forum_discussions",
)

CONTAINER_NAME = "nanobot-nginx"
BACKEND_BASE = "http://host.docker.internal:18790"
CURRENT_USER_ID = "69dd0e215b40c39cc3a47691"
CURRENT_USER_NAME = "Faith Macpherson"
INSTRUCTOR_NAME = "Street Voices Academy"

SYNTHETIC_LEARNERS = [
    {"id": "academy-demo-learner-amara", "name": "Amara Lewis"},
    {"id": "academy-demo-learner-devon", "name": "Devon Carter"},
    {"id": "academy-demo-learner-zuri", "name": "Zuri Bennett"},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_in(days: int = 0, hours: int = 0, minutes: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days, hours=hours, minutes=minutes)).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: str) -> str:
    seed = "::".join([prefix, *parts])
    return f"{prefix}-{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:10]}"


def stable_int_id(*parts: str) -> int:
    seed = "::".join(parts)
    return int(uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:10], 16)


def slugify(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        else:
            allowed.append("-")
    slug = "".join(allowed)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or f"path-{uuid.uuid4().hex[:6]}"


def text_data_url(label: str, content: str) -> dict[str, Any]:
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return {
        "url": f"data:text/plain;base64,{encoded}",
        "filename": label,
        "size_bytes": len(content.encode("utf-8")),
        "mime_type": "text/plain",
        "uploaded_at": now_iso(),
    }


def load_state() -> dict[str, list[dict[str, Any]]]:
    if not RUNTIME_FILE.exists():
        return {key: [] for key in ACADEMY_KEYS}

    state = json.loads(RUNTIME_FILE.read_text())
    return {key: list(state.get(key, [])) for key in ACADEMY_KEYS}


def save_state(state: dict[str, list[dict[str, Any]]]) -> None:
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_FILE.write_text(json.dumps(state, indent=2))


def docker_json(path: str) -> Any:
    command = [
        "docker",
        "exec",
        CONTAINER_NAME,
        "curl",
        "-sS",
        f"{BACKEND_BASE}{path}",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"Failed: {' '.join(command)}")
    output = result.stdout.strip()
    return json.loads(output or "null")


def ensure_enrollment(
    state: dict[str, list[dict[str, Any]]],
    *,
    user_id: str,
    course_id: str,
    progress_percent: int,
    status: str,
) -> dict[str, Any]:
    enrollment = next(
        (
            item
            for item in state["enrollments"]
            if item.get("user_id") == user_id and item.get("course_id") == course_id
        ),
        None,
    )
    timestamp = now_iso()
    final_status = "completed" if progress_percent >= 100 else status
    if enrollment is None:
        enrollment = {
            "id": stable_id("enrollment", user_id, course_id),
            "user_id": user_id,
            "course_id": course_id,
            "status": final_status,
            "progress_percent": progress_percent,
            "last_accessed_at": timestamp,
            "enrolled_at": timestamp,
            "created_at": timestamp,
            "updated_at": timestamp,
            "completed_at": timestamp if final_status == "completed" else None,
        }
        state["enrollments"].append(enrollment)
        return enrollment

    enrollment["status"] = final_status
    enrollment["progress_percent"] = progress_percent
    enrollment["last_accessed_at"] = timestamp
    enrollment["updated_at"] = timestamp
    enrollment["enrolled_at"] = enrollment.get("enrolled_at") or timestamp
    enrollment["completed_at"] = timestamp if final_status == "completed" else None
    return enrollment


def ensure_learning_path(
    state: dict[str, list[dict[str, Any]]],
    *,
    title: str,
    description: str,
    course_ids: list[str],
    level: str,
    color: str,
    delivery_mode: str,
    requirements: list[str],
    what_youll_learn: list[str],
    milestones: list[str],
    outcomes: list[str],
    preferred_categories: list[str],
) -> dict[str, Any]:
    slug = slugify(title)
    now = now_iso()
    path = next((item for item in state["learning_paths"] if item.get("slug") == slug), None)
    payload = {
        "id": stable_id("path", slug),
        "slug": slug,
        "title": title,
        "description": description,
        "courses": len(course_ids),
        "hours": max(len(course_ids) * 8, 8),
        "level": level,
        "delivery_mode": delivery_mode,
        "color": color,
        "requirements": requirements,
        "what_youll_learn": what_youll_learn,
        "milestones": milestones,
        "outcomes": outcomes,
        "preferred_categories": preferred_categories,
        "course_ids": course_ids,
        "created_by": CURRENT_USER_ID,
        "source": "generated",
        "created_at": now,
        "updated_at": now,
    }
    if path is None:
        state["learning_paths"].append(payload)
        return payload
    path.update(payload)
    path["created_at"] = path.get("created_at") or now
    return path


def ensure_review(
    state: dict[str, list[dict[str, Any]]],
    *,
    course_id: str,
    user_id: str,
    rating: int,
    review_text: str,
) -> dict[str, Any]:
    review = next(
        (
            item
            for item in state["reviews"]
            if item.get("course_id") == course_id and item.get("user_id") == user_id
        ),
        None,
    )
    payload = {
        "id": stable_id("review", course_id, user_id),
        "user_id": user_id,
        "course_id": course_id,
        "rating": rating,
        "review_text": review_text,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    if review is None:
        state["reviews"].append(payload)
        return payload
    review.update(payload)
    return review


def ensure_live_session(
    state: dict[str, list[dict[str, Any]]],
    *,
    course_id: str,
    title: str,
    description: str,
    scheduled_start: str,
    scheduled_end: str,
    status: str,
    meeting_id: str,
    meeting_url: str,
    session_notes: str,
    recording_url: str | None = None,
) -> dict[str, Any]:
    session = next(
        (
            item
            for item in state["live_sessions"]
            if item.get("course_id") == course_id and item.get("title") == title
        ),
        None,
    )
    actual_start = scheduled_start if status in {"live", "ended"} else None
    actual_end = scheduled_end if status == "ended" else None
    payload = {
        "id": stable_id("session", course_id, title),
        "course_id": course_id,
        "module_id": None,
        "lesson_id": None,
        "title": title,
        "description": description,
        "session_type": "class",
        "instructor_id": CURRENT_USER_ID,
        "co_host_ids": [],
        "scheduled_start": scheduled_start,
        "scheduled_end": scheduled_end,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "status": status,
        "max_attendees": 40,
        "platform": "zoom",
        "meeting_id": meeting_id,
        "meeting_url": meeting_url,
        "session_notes": session_notes,
        "recording_url": recording_url,
        "recording_available": bool(recording_url),
        "is_mandatory": False,
        "points_for_attending": 10,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    if session is None:
        state["live_sessions"].append(payload)
        return payload
    session.update(payload)
    session["created_at"] = session.get("created_at") or now_iso()
    return session


def ensure_session_registration(
    state: dict[str, list[dict[str, Any]]],
    *,
    session_id: str,
    user_id: str,
    status: str,
    joined_at: str | None = None,
    left_at: str | None = None,
) -> dict[str, Any]:
    registration = next(
        (
            item
            for item in state["session_registrations"]
            if item.get("session_id") == session_id and item.get("user_id") == user_id
        ),
        None,
    )
    payload = {
        "id": stable_id("registration", session_id, user_id),
        "session_id": session_id,
        "user_id": user_id,
        "status": status,
        "joined_at": joined_at,
        "left_at": left_at,
        "attendance_duration": 0,
        "attendance_percent": 0,
        "attended_full": False,
        "points_earned": 0,
    }
    if registration is None:
        state["session_registrations"].append(payload)
        return payload
    registration.update(payload)
    return registration


def ensure_material(
    state: dict[str, list[dict[str, Any]]],
    *,
    course_id: str,
    title: str,
    notes: str,
    material_type: str,
    schedule_item_id: str | None,
    file_name: str | None,
    file_attachment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    material = next(
        (
            item
            for item in state["course_materials"]
            if item.get("course_id") == course_id and item.get("title") == title
        ),
        None,
    )
    payload = {
        "id": stable_id("material", course_id, title),
        "linkId": stable_id("material-link", course_id, title),
        "documentId": f"upload:{stable_id('document', course_id, title)}",
        "title": title,
        "documentType": "uploaded-file",
        "status": "ready",
        "materialType": material_type,
        "sortOrder": len([item for item in state["course_materials"] if item.get("course_id") == course_id]),
        "wordCount": 0,
        "readingTimeMinutes": 0,
        "authorId": CURRENT_USER_ID,
        "notes": notes,
        "fileName": file_name,
        "fileUrl": file_attachment.get("url") if file_attachment else None,
        "mimeType": file_attachment.get("mime_type") if file_attachment else None,
        "sizeBytes": file_attachment.get("size_bytes") if file_attachment else None,
        "uploadedAt": file_attachment.get("uploaded_at") if file_attachment else None,
        "scheduleItemId": schedule_item_id,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "course_id": course_id,
        "module_id": None,
        "lesson_id": None,
    }
    if material is None:
        state["course_materials"].append(payload)
        return payload
    material.update(payload)
    material["createdAt"] = material.get("createdAt") or now_iso()
    return material


def ensure_assignment(
    state: dict[str, list[dict[str, Any]]],
    *,
    course_id: str,
    course_title: str,
    title: str,
    assignment_type: str,
    description: str,
    instructions: str,
    due_date: str,
    quiz_questions: list[str] | None = None,
    resource_file_name: str | None = None,
    resource_attachment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assignment = next(
        (
            item
            for item in state["assignments"]
            if item.get("course_id") == course_id and item.get("title") == title
        ),
        None,
    )
    payload = {
        "id": stable_id("assign", course_id, title),
        "course_id": course_id,
        "course_title": course_title,
        "module_id": None,
        "lesson_id": None,
        "title": title,
        "description": description,
        "instructions": instructions,
        "assignment_type": assignment_type,
        "max_points": 100,
        "passing_score": 70,
        "due_date": due_date,
        "available_from": now_iso(),
        "available_until": None,
        "allow_late_submissions": True,
        "late_penalty_percent": 5,
        "max_late_days": 7,
        "max_attempts": 1,
        "peer_review_enabled": False,
        "peer_reviews_required": 0,
        "rubric_id": None,
        "allowed_file_types": [] if assignment_type == "quiz" else ["pdf", "docx", "jpg", "jpeg", "png"],
        "max_file_size_mb": 0 if assignment_type == "quiz" else 15,
        "max_files": 1 if assignment_type == "quiz" else 3,
        "calendar_event_id": None,
        "is_published": True,
        "created_by": CURRENT_USER_ID,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "quiz_questions": quiz_questions or [],
        "resource_file_name": resource_file_name,
        "resource_attachment": resource_attachment,
    }
    if assignment is None:
        state["assignments"].append(payload)
        return payload
    assignment.update(payload)
    assignment["created_at"] = assignment.get("created_at") or now_iso()
    return assignment


def ensure_schedule_item(
    state: dict[str, list[dict[str, Any]]],
    *,
    course_id: str,
    title: str,
    notes: str,
    category: str,
    scheduled_at: str,
    linked_assignment_id: str | None = None,
    linked_material_link_id: str | None = None,
    file_name: str | None = None,
    file_url: str | None = None,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    uploaded_at: str | None = None,
) -> dict[str, Any]:
    item = next(
        (
            row
            for row in state["course_schedule_items"]
            if row.get("course_id") == course_id and row.get("title") == title and row.get("category") == category
        ),
        None,
    )
    payload = {
        "id": stable_id("schedule-item", course_id, category, title),
        "course_id": course_id,
        "title": title,
        "notes": notes,
        "scheduled_at": scheduled_at,
        "category": category,
        "created_by": CURRENT_USER_ID,
        "linked_assignment_id": linked_assignment_id,
        "linked_material_link_id": linked_material_link_id,
        "file_name": file_name,
        "file_url": file_url,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "uploaded_at": uploaded_at,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    if item is None:
        state["course_schedule_items"].append(payload)
        return payload
    item.update(payload)
    item["created_at"] = item.get("created_at") or now_iso()
    return item


def ensure_submission(
    state: dict[str, list[dict[str, Any]]],
    *,
    assignment_id: str,
    course_id: str,
    user_id: str,
    user_name: str,
    submission_type: str,
    text_content: str,
    quiz_answers: list[str] | None,
    status: str,
    score: float | None,
    feedback: str | None,
) -> dict[str, Any]:
    submission = next(
        (
            item
            for item in state["submissions"]
            if item.get("assignment_id") == assignment_id and item.get("user_id") == user_id
        ),
        None,
    )
    submitted_at = now_iso()
    payload = {
        "id": stable_id("submission", assignment_id, user_id),
        "assignment_id": assignment_id,
        "course_id": course_id,
        "user_id": user_id,
        "user_name": user_name,
        "attempt_number": 1,
        "status": status,
        "submission_type": submission_type,
        "text_content": text_content,
        "document_id": None,
        "file_urls": [],
        "word_count": len(text_content.split()),
        "submitted_at": submitted_at,
        "is_late": False,
        "days_late": 0,
        "late_penalty_applied": 0,
        "graded_at": submitted_at if status in {"graded", "returned"} else None,
        "graded_by": CURRENT_USER_ID if status in {"graded", "returned"} else None,
        "score": score,
        "adjusted_score": score,
        "letter_grade": "A" if score is not None and score >= 90 else "B" if score is not None and score >= 80 else "C" if score is not None else None,
        "feedback": feedback,
        "feedback_attachments": [],
        "regrade_reason": None,
        "grading_locked_by": CURRENT_USER_ID if status in {"graded", "returned"} else None,
        "created_at": submitted_at,
        "updated_at": submitted_at,
        "quiz_answers": quiz_answers or [],
    }
    if submission is None:
        state["submissions"].append(payload)
        return payload
    submission.update(payload)
    submission["created_at"] = submission.get("created_at") or submitted_at
    return submission


def ensure_certificate(
    state: dict[str, list[dict[str, Any]]],
    *,
    user_id: str,
    recipient_name: str,
    target_type: str,
    target_id: str,
    target_title: str,
    course_id: str | None = None,
    learning_path_id: str | None = None,
) -> dict[str, Any]:
    certificate = next(
        (
            item
            for item in state["certificates"]
            if item.get("user_id") == user_id and item.get("target_type") == target_type and item.get("target_id") == target_id
        ),
        None,
    )
    payload = {
        "id": stable_id("certificate", user_id, target_type, target_id),
        "user_id": user_id,
        "recipient_name": recipient_name,
        "course_id": course_id,
        "learning_path_id": learning_path_id,
        "target_type": target_type,
        "target_id": target_id,
        "target_title": target_title,
        "certificate_title": "Certificate of Achievement",
        "issuer_name": INSTRUCTOR_NAME,
        "signature_name": INSTRUCTOR_NAME,
        "issued_by": CURRENT_USER_ID,
        "award_date": datetime.now().date().isoformat(),
        "certificate_url": None,
        "badge_url": None,
        "verification_code": stable_id("verify", user_id, target_id).split("-")[-1].upper(),
        "issued_at": now_iso(),
        "updated_at": now_iso(),
        "expires_at": None,
    }
    if certificate is None:
        state["certificates"].append(payload)
        return payload
    certificate.update(payload)
    certificate["issued_at"] = certificate.get("issued_at") or now_iso()
    return certificate


def ensure_discussion(
    state: dict[str, list[dict[str, Any]]],
    *,
    forum_id: int,
    course_id: str,
    subject: str,
    message: str,
    replies: list[dict[str, Any]],
    reactions: dict[str, list[str]],
) -> dict[str, Any]:
    discussion = next(
        (
            item
            for item in state["forum_discussions"]
            if item.get("forum_id") == forum_id and item.get("subject") == subject
        ),
        None,
    )
    created = int(datetime.now(timezone.utc).timestamp())
    payload = {
        "id": stable_int_id("discussion", course_id, subject),
        "forum_id": forum_id,
        "course_id": course_id,
        "name": subject,
        "subject": subject,
        "message": message,
        "userfullname": INSTRUCTOR_NAME,
        "userid": CURRENT_USER_ID,
        "author_role": "instructor",
        "created": created,
        "modified": created,
        "numreplies": len(replies),
        "pinned": False,
        "timemodified": created,
        "replies": replies,
        "reactions": reactions,
    }
    if discussion is None:
        state["forum_discussions"].append(payload)
        return payload
    discussion.update(payload)
    return discussion


def build_reply(author_id: str, author_name: str, message: str, author_role: str = "student") -> dict[str, Any]:
    created = int(datetime.now(timezone.utc).timestamp())
    return {
        "id": stable_int_id("reply", author_id, message),
        "message": message,
        "userfullname": author_name,
        "userid": author_id,
        "author_role": author_role,
        "created": created,
    }


def find_course(courses: list[dict[str, Any]], title: str) -> dict[str, Any]:
    for course in courses:
        if str(course.get("title")) == title:
            return course
    raise KeyError(f"Missing course title: {title}")


def main() -> int:
    state = load_state()
    courses = docker_json("/api/academy/courses?limit=100")
    course_by_title = {str(course.get("title")): course for course in courses}

    speaking = find_course(courses, "Speaking Up with Confidence")
    rights = find_course(courses, "Know Your Rights")
    systems = find_course(courses, "Navigating Systems")
    resume = find_course(courses, "Resume Writing Workshop")
    interview = find_course(courses, "Interview Skills Mastery")
    communication = find_course(courses, "Workplace Communication")
    computers = find_course(courses, "Getting Started with Computers")
    email = find_course(courses, "Email Essentials")
    smartphone = find_course(courses, "Smartphone Skills")
    ethics = find_course(courses, "Ethics and Boundaries")
    resources = find_course(courses, "Resource Navigation")
    listening = find_course(courses, "Active Listening Skills")

    ensure_learning_path(
        state,
        title="Advocacy Confidence Path",
        description="Build confidence, know your rights, and learn how to move through systems with a clear support plan.",
        course_ids=[speaking["id"], rights["id"], systems["id"]],
        level="Beginner",
        color="#F97316",
        delivery_mode="Online and In person",
        requirements=[
            "Bring your voice, lived experience, and willingness to practice.",
            "Join the live support sessions when you can.",
            "Move through the courses in order for the smoothest learning flow.",
        ],
        what_youll_learn=[
            "How to speak up with more confidence.",
            "How to understand and use your rights in real situations.",
            "How to navigate support systems with less stress.",
        ],
        milestones=[
            "Complete the confidence course.",
            "Finish the rights course and quick quiz.",
            "Use the systems course to map your next real-life step.",
        ],
        outcomes=[
            "Leave with a practical advocacy foundation.",
            "Know where to go next when you need support.",
            "Build momentum with one connected Academy plan.",
        ],
        preferred_categories=["advocacy"],
    )
    career_path = ensure_learning_path(
        state,
        title="Job Ready Communication Path",
        description="Move from preparation to action with courses that strengthen communication, interviews, and your resume.",
        course_ids=[communication["id"], interview["id"], resume["id"]],
        level="Beginner",
        color="#7C3AED",
        delivery_mode="Online and In person",
        requirements=[
            "Set aside time each week to practice.",
            "Be ready to reflect on your goals and experience.",
            "Attend the live job-readiness check-ins when available.",
        ],
        what_youll_learn=[
            "How to communicate clearly in work settings.",
            "How to prepare for interviews with confidence.",
            "How to build a resume that reflects your strengths.",
        ],
        milestones=[
            "Complete your communication basics.",
            "Practice interview answers.",
            "Finish a polished resume draft.",
        ],
        outcomes=[
            "Build a stronger employment foundation.",
            "Feel more prepared for job applications.",
            "Move toward work with a clearer plan.",
        ],
        preferred_categories=["employment"],
    )
    ensure_learning_path(
        state,
        title="Digital Restart Path",
        description="Get comfortable with computers, email, and your phone so online tasks feel easier and more manageable.",
        course_ids=[computers["id"], email["id"], smartphone["id"]],
        level="Beginner",
        color="#0EA5E9",
        delivery_mode="Online and In person",
        requirements=[
            "No previous digital experience is required.",
            "Bring a phone or laptop if you have one.",
            "Practice one small digital skill between lessons.",
        ],
        what_youll_learn=[
            "Core computer basics for everyday tasks.",
            "How to use email with more confidence.",
            "How to use your phone for learning and communication.",
        ],
        milestones=[
            "Complete the computer basics course.",
            "Send and organize email with confidence.",
            "Set up a safer, more useful smartphone workflow.",
        ],
        outcomes=[
            "Feel more independent online.",
            "Handle practical digital tasks with less support.",
            "Build a strong foundation for more advanced courses.",
        ],
        preferred_categories=["digital skills"],
    )
    community_path = ensure_learning_path(
        state,
        title="Community Support Facilitation Path",
        description="Strengthen boundaries, resource sharing, and active listening so you can support others with clarity and care.",
        course_ids=[ethics["id"], resources["id"], listening["id"]],
        level="Intermediate",
        color="#22C55E",
        delivery_mode="Online and In person",
        requirements=[
            "Some community or peer support experience helps, but is not required.",
            "Be ready to participate in reflection and discussion.",
            "Stay open to feedback and practice.",
        ],
        what_youll_learn=[
            "How to hold healthy boundaries in support roles.",
            "How to connect people to the right resources.",
            "How to listen with empathy and intention.",
        ],
        milestones=[
            "Set your support boundaries.",
            "Build a resource navigation toolkit.",
            "Practice active listening in real scenarios.",
        ],
        outcomes=[
            "Lead with stronger facilitation skills.",
            "Support others more confidently.",
            "Build a deeper community practice.",
        ],
        preferred_categories=["advocacy"],
    )

    ensure_enrollment(state, user_id=CURRENT_USER_ID, course_id=speaking["id"], progress_percent=62, status="active")
    ensure_enrollment(state, user_id=CURRENT_USER_ID, course_id=rights["id"], progress_percent=100, status="completed")
    ensure_enrollment(state, user_id=CURRENT_USER_ID, course_id=systems["id"], progress_percent=24, status="active")
    ensure_enrollment(state, user_id=SYNTHETIC_LEARNERS[0]["id"], course_id=speaking["id"], progress_percent=78, status="active")
    ensure_enrollment(state, user_id=SYNTHETIC_LEARNERS[0]["id"], course_id=rights["id"], progress_percent=56, status="active")
    ensure_enrollment(state, user_id=SYNTHETIC_LEARNERS[1]["id"], course_id=speaking["id"], progress_percent=33, status="active")
    ensure_enrollment(state, user_id=SYNTHETIC_LEARNERS[1]["id"], course_id=systems["id"], progress_percent=18, status="active")
    ensure_enrollment(state, user_id=SYNTHETIC_LEARNERS[2]["id"], course_id=rights["id"], progress_percent=82, status="active")
    ensure_enrollment(state, user_id=SYNTHETIC_LEARNERS[2]["id"], course_id=career_path["course_ids"][0], progress_percent=12, status="active")

    ensure_review(
        state,
        course_id=speaking["id"],
        user_id=SYNTHETIC_LEARNERS[0]["id"],
        rating=5,
        review_text="This course felt uplifting and practical. The live coaching made it easier to speak with more confidence right away.",
    )
    ensure_review(
        state,
        course_id=speaking["id"],
        user_id=SYNTHETIC_LEARNERS[1]["id"],
        rating=4,
        review_text="Clear lessons, good pacing, and the assignments helped me use the ideas in real conversations.",
    )
    ensure_review(
        state,
        course_id=rights["id"],
        user_id=SYNTHETIC_LEARNERS[2]["id"],
        rating=5,
        review_text="Helpful and easy to follow. I liked seeing how the rights information connects to everyday situations.",
    )
    ensure_review(
        state,
        course_id=computers["id"],
        user_id=SYNTHETIC_LEARNERS[0]["id"],
        rating=4,
        review_text="A great reset for anyone starting over with digital skills.",
    )

    speaking_upcoming = ensure_live_session(
        state,
        course_id=speaking["id"],
        title="Confidence Coaching Circle",
        description="Practice speaking up, get support, and prepare for the next course activity.",
        scheduled_start=iso_in(days=2, hours=2),
        scheduled_end=iso_in(days=2, hours=3),
        status="scheduled",
        meeting_id="SVA-COACH-204",
        meeting_url="https://zoom.us/j/987650001",
        session_notes="Zoom meeting: Meeting ID SVA-COACH-204. Join early for audio check and bring one advocacy scenario to practice.",
    )
    speaking_replay = ensure_live_session(
        state,
        course_id=speaking["id"],
        title="Community Practice Replay",
        description="Replay available from the last live practice session.",
        scheduled_start=iso_in(days=-3, hours=1),
        scheduled_end=iso_in(days=-3, hours=2),
        status="ended",
        meeting_id="SVA-REPLAY-118",
        meeting_url="https://zoom.us/j/987650002",
        session_notes="Review the replay and take notes on one technique you want to use next.",
        recording_url="https://streetvoicesacademy.local/recordings/community-practice-replay",
    )
    rights_session = ensure_live_session(
        state,
        course_id=rights["id"],
        title="Rights Q&A Session",
        description="Bring your questions and walk through real examples with the instructor.",
        scheduled_start=iso_in(days=4, hours=1),
        scheduled_end=iso_in(days=4, hours=2),
        status="scheduled",
        meeting_id="SVA-RIGHTS-310",
        meeting_url="https://zoom.us/j/987650003",
        session_notes="Zoom meeting: Meeting ID SVA-RIGHTS-310. Bring one rights-related question from your week.",
    )
    systems_session = ensure_live_session(
        state,
        course_id=systems["id"],
        title="Systems Navigation Office Hours",
        description="Office hours to map your next support step and ask questions.",
        scheduled_start=iso_in(days=6, hours=2),
        scheduled_end=iso_in(days=6, hours=3),
        status="scheduled",
        meeting_id="SVA-SYSTEMS-211",
        meeting_url="https://zoom.us/j/987650004",
        session_notes="Use this session to bring one systems challenge and leave with a next action plan.",
    )

    ensure_session_registration(state, session_id=speaking_upcoming["id"], user_id=CURRENT_USER_ID, status="registered")
    ensure_session_registration(state, session_id=speaking_upcoming["id"], user_id=SYNTHETIC_LEARNERS[0]["id"], status="registered")
    ensure_session_registration(state, session_id=speaking_replay["id"], user_id=CURRENT_USER_ID, status="attended", joined_at=iso_in(days=-3, hours=1), left_at=iso_in(days=-3, hours=2))
    ensure_session_registration(state, session_id=rights_session["id"], user_id=CURRENT_USER_ID, status="registered")
    ensure_session_registration(state, session_id=systems_session["id"], user_id=CURRENT_USER_ID, status="registered")

    confidence_quiz_attachment = text_data_url(
        "confidence-check-in.pdf",
        "Street Voices Academy\nConfidence Check-In Quiz\n\n1. What helps you prepare before speaking up?\n2. How does practice build confidence?\n3. What support strategy will you use this week?",
    )
    confidence_toolkit_attachment = text_data_url(
        "confidence-toolkit.pdf",
        "Confidence Toolkit\n\n- Pause and breathe before you speak.\n- Write down your one key point.\n- Practice with a trusted person.\n- End with the action you need.",
    )
    confidence_worksheet_attachment = text_data_url(
        "session-prep-worksheet.pdf",
        "Session Prep Worksheet\n\nGoal:\nMy key message:\nWhat support do I need?\nWhat will I practice in today's session?",
    )
    rights_reference_attachment = text_data_url(
        "rights-quick-reference.pdf",
        "Rights Quick Reference\n\nSituation:\nRelevant right:\nWho can help:\nNext step:",
    )
    systems_mapping_attachment = text_data_url(
        "systems-mapping-template.pdf",
        "Systems Mapping Template\n\nGoal:\nPeople involved:\nOrganizations involved:\nDeadlines:\nNext step:",
    )

    speaking_assignment = ensure_assignment(
        state,
        course_id=speaking["id"],
        course_title=speaking["title"],
        title="Confidence Reflection",
        assignment_type="mixed",
        description="Reflect on one moment where you practiced speaking up and explain what changed.",
        instructions="<p>Write a short reflection or upload notes that show how you used one confidence tool from this course.</p>",
        due_date=iso_in(days=5),
    )
    speaking_quiz = ensure_assignment(
        state,
        course_id=speaking["id"],
        course_title=speaking["title"],
        title="Confidence Check-In Quiz",
        assignment_type="quiz",
        description="A short quiz to help you reflect on the key ideas from the course so far.",
        instructions="<p>Answer each question in your own words. This quiz is visible only to learners enrolled in this course.</p>",
        due_date=iso_in(days=3),
        quiz_questions=[
            "What is one way you can prepare before speaking up in a difficult conversation?",
            "Why does practice help build confidence over time?",
            "What support strategy from this course feels most useful to you right now?",
        ],
        resource_file_name="confidence-check-in.pdf",
        resource_attachment=confidence_quiz_attachment,
    )
    rights_assignment = ensure_assignment(
        state,
        course_id=rights["id"],
        course_title=rights["title"],
        title="Rights Action Plan",
        assignment_type="mixed",
        description="Create a short action plan for using one right or protection in a real-life situation.",
        instructions="<p>Write or upload a short plan with one situation, one right, and one next step you can take.</p>",
        due_date=iso_in(days=7),
    )

    speaking_reading_item = ensure_schedule_item(
        state,
        course_id=speaking["id"],
        title="Read the confidence toolkit",
        notes="Review the toolkit before the next live coaching circle.",
        category="reading",
        scheduled_at=iso_in(days=1),
        file_name="confidence-toolkit.pdf",
        file_url=confidence_toolkit_attachment["url"],
        mime_type=confidence_toolkit_attachment["mime_type"],
        size_bytes=confidence_toolkit_attachment["size_bytes"],
        uploaded_at=confidence_toolkit_attachment["uploaded_at"],
    )
    speaking_reading_material = ensure_material(
        state,
        course_id=speaking["id"],
        title="Confidence Toolkit",
        notes="A quick guide students can keep open during class.",
        material_type="reading",
        schedule_item_id=speaking_reading_item["id"],
        file_name="confidence-toolkit.pdf",
        file_attachment=confidence_toolkit_attachment,
    )
    speaking_reading_item["linked_material_link_id"] = speaking_reading_material["linkId"]

    speaking_assignment_item = ensure_schedule_item(
        state,
        course_id=speaking["id"],
        title=speaking_assignment["title"],
        notes="Post your reflection before the next live session.",
        category="assignment",
        scheduled_at=speaking_assignment["due_date"],
        linked_assignment_id=speaking_assignment["id"],
    )
    speaking_material_item = ensure_schedule_item(
        state,
        course_id=speaking["id"],
        title="Session prep worksheet",
        notes="Use this worksheet to prepare for your confidence practice.",
        category="material",
        scheduled_at=iso_in(days=2),
        file_name="session-prep-worksheet.pdf",
        file_url=confidence_worksheet_attachment["url"],
        mime_type=confidence_worksheet_attachment["mime_type"],
        size_bytes=confidence_worksheet_attachment["size_bytes"],
        uploaded_at=confidence_worksheet_attachment["uploaded_at"],
    )
    speaking_material = ensure_material(
        state,
        course_id=speaking["id"],
        title="Session Prep Worksheet",
        notes="Students can use this worksheet before live coaching.",
        material_type="supplementary",
        schedule_item_id=speaking_material_item["id"],
        file_name="session-prep-worksheet.pdf",
        file_attachment=confidence_worksheet_attachment,
    )
    speaking_material_item["linked_material_link_id"] = speaking_material["linkId"]

    rights_material_item = ensure_schedule_item(
        state,
        course_id=rights["id"],
        title="Rights quick reference",
        notes="Keep this sheet nearby while you work through the course.",
        category="material",
        scheduled_at=iso_in(days=2, hours=2),
        file_name="rights-quick-reference.pdf",
        file_url=rights_reference_attachment["url"],
        mime_type=rights_reference_attachment["mime_type"],
        size_bytes=rights_reference_attachment["size_bytes"],
        uploaded_at=rights_reference_attachment["uploaded_at"],
    )
    rights_material = ensure_material(
        state,
        course_id=rights["id"],
        title="Rights Quick Reference",
        notes="A one-page guide students can revisit during the week.",
        material_type="reference",
        schedule_item_id=rights_material_item["id"],
        file_name="rights-quick-reference.pdf",
        file_attachment=rights_reference_attachment,
    )
    rights_material_item["linked_material_link_id"] = rights_material["linkId"]

    systems_material = ensure_material(
        state,
        course_id=systems["id"],
        title="Systems Mapping Template",
        notes="A printable map for tracking the people, organizations, and steps involved in your next support goal.",
        material_type="worksheet",
        schedule_item_id=None,
        file_name="systems-mapping-template.pdf",
        file_attachment=systems_mapping_attachment,
    )

    ensure_submission(
        state,
        assignment_id=speaking_assignment["id"],
        course_id=speaking["id"],
        user_id=CURRENT_USER_ID,
        user_name=CURRENT_USER_NAME,
        submission_type="mixed",
        text_content="I practiced speaking up with a service provider and felt more prepared because I wrote down my main point first.",
        quiz_answers=None,
        status="graded",
        score=93,
        feedback="Strong reflection. You clearly connected the course to a real situation.",
    )
    ensure_submission(
        state,
        assignment_id=speaking_quiz["id"],
        course_id=speaking["id"],
        user_id=CURRENT_USER_ID,
        user_name=CURRENT_USER_NAME,
        submission_type="text",
        text_content="Completed the confidence quiz.",
        quiz_answers=[
            "I can prepare by writing down my main message and one question.",
            "Practice lowers stress and helps me respond more clearly.",
            "The grounding pause before I answer feels most useful right now.",
        ],
        status="graded",
        score=96,
        feedback="Excellent quiz responses. Clear and practical.",
    )
    ensure_submission(
        state,
        assignment_id=speaking_assignment["id"],
        course_id=speaking["id"],
        user_id=SYNTHETIC_LEARNERS[0]["id"],
        user_name=SYNTHETIC_LEARNERS[0]["name"],
        submission_type="mixed",
        text_content="I used the practice script before calling an agency and it helped me stay calm.",
        quiz_answers=None,
        status="submitted",
        score=None,
        feedback=None,
    )
    ensure_submission(
        state,
        assignment_id=speaking_quiz["id"],
        course_id=speaking["id"],
        user_id=SYNTHETIC_LEARNERS[1]["id"],
        user_name=SYNTHETIC_LEARNERS[1]["name"],
        submission_type="text",
        text_content="Completed the quiz.",
        quiz_answers=[
            "I can breathe first and decide what I need to say.",
            "Practice helps me remember what I want to say.",
            "I liked the planning worksheet the most.",
        ],
        status="graded",
        score=88,
        feedback="Good work. Keep making your responses more specific.",
    )
    ensure_submission(
        state,
        assignment_id=rights_assignment["id"],
        course_id=rights["id"],
        user_id=CURRENT_USER_ID,
        user_name=CURRENT_USER_NAME,
        submission_type="mixed",
        text_content="My action plan focuses on documenting an issue, asking a clear question, and requesting written follow-up.",
        quiz_answers=None,
        status="graded",
        score=91,
        feedback="Well done. Your action steps are clear and realistic.",
    )

    speaking_forum_id = int(docker_json(f"/api/academy/moodle/forums/{speaking['id']}")["forums"][0]["id"])
    rights_forum_id = int(docker_json(f"/api/academy/moodle/forums/{rights['id']}")["forums"][0]["id"])
    systems_forum_id = int(docker_json(f"/api/academy/moodle/forums/{systems['id']}")["forums"][0]["id"])

    ensure_discussion(
        state,
        forum_id=speaking_forum_id,
        course_id=speaking["id"],
        subject="Weekly confidence reminder",
        message="This week, focus on one small moment where you can speak up clearly. Use the worksheet, come to the live session, and post your reflection before Friday.",
        replies=[
            build_reply(SYNTHETIC_LEARNERS[0]["id"], SYNTHETIC_LEARNERS[0]["name"], "I used the worksheet already and it helped me organize what I wanted to say."),
            build_reply(CURRENT_USER_ID, CURRENT_USER_NAME, "The live session replay gave me a good example of how to slow down before responding."),
        ],
        reactions={"up": [CURRENT_USER_ID, SYNTHETIC_LEARNERS[1]["id"]], "down": []},
    )
    ensure_discussion(
        state,
        forum_id=rights_forum_id,
        course_id=rights["id"],
        subject="Bring one rights question to class",
        message="For the next rights session, bring one real-life question so we can work through it together. You do not need to share personal details if you do not want to.",
        replies=[
            build_reply(SYNTHETIC_LEARNERS[2]["id"], SYNTHETIC_LEARNERS[2]["name"], "I have a question about what to document before asking for help."),
        ],
        reactions={"up": [CURRENT_USER_ID], "down": []},
    )
    ensure_discussion(
        state,
        forum_id=systems_forum_id,
        course_id=systems["id"],
        subject="Map one next step this week",
        message="Choose one system you want to navigate this week and map out one clear next action before our office hours.",
        replies=[
            build_reply(CURRENT_USER_ID, CURRENT_USER_NAME, "I am going to use the template to map out my next call and follow-up."),
        ],
        reactions={"up": [SYNTHETIC_LEARNERS[1]["id"]], "down": []},
    )

    ensure_certificate(
        state,
        user_id=CURRENT_USER_ID,
        recipient_name=CURRENT_USER_NAME,
        target_type="course",
        target_id=rights["id"],
        target_title=rights["title"],
        course_id=rights["id"],
    )
    ensure_certificate(
        state,
        user_id=CURRENT_USER_ID,
        recipient_name=CURRENT_USER_NAME,
        target_type="learning_path",
        target_id=career_path["slug"],
        target_title=career_path["title"],
        learning_path_id=career_path["slug"],
    )
    ensure_certificate(
        state,
        user_id=SYNTHETIC_LEARNERS[0]["id"],
        recipient_name=SYNTHETIC_LEARNERS[0]["name"],
        target_type="learning_path",
        target_id=community_path["slug"],
        target_title=community_path["title"],
        learning_path_id=community_path["slug"],
    )

    save_state(state)

    summary = {
        "learning_paths": len(state["learning_paths"]),
        "enrollments": len(state["enrollments"]),
        "live_sessions": len(state["live_sessions"]),
        "course_schedule_items": len(state["course_schedule_items"]),
        "course_materials": len(state["course_materials"]),
        "assignments": len(state["assignments"]),
        "submissions": len(state["submissions"]),
        "reviews": len(state["reviews"]),
        "certificates": len(state["certificates"]),
        "forum_discussions": len(state["forum_discussions"]),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
