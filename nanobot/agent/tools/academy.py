"""Academic agent tools — OpenMAIC classroom generation + SBP academy backend."""

from __future__ import annotations

import json
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.services.openmaic_client import OpenMAICClient

SBP_TIMEOUT = 15  # seconds for SBP backend calls


class AcademyCreateCourseTool(Tool):
    """Generate a full course from a topic using OpenMAIC, then persist via SBP backend."""

    name = "academy_create_course"
    description = (
        "Create a new course from a topic description. Uses AI to generate an interactive "
        "classroom with slides, quizzes, and simulations, then saves it as a structured course "
        "with modules and lessons. Returns the course ID and summary."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The course topic or description (e.g. 'Introduction to Python programming')",
            },
            "level": {
                "type": "string",
                "enum": ["beginner", "intermediate", "advanced"],
                "description": "Difficulty level",
            },
            "category": {
                "type": "string",
                "description": "Course category (e.g. Technology, Business, Marketing)",
            },
            "duration": {
                "type": "string",
                "description": "Estimated duration (e.g. '4 weeks', '2 hours')",
            },
            "instructor_name": {
                "type": "string",
                "description": "Instructor name to attribute the course to",
            },
        },
        "required": ["topic"],
    }

    def __init__(self, *, openmaic: OpenMAICClient, sbp_api: str):
        self._openmaic = openmaic
        self._sbp = sbp_api.rstrip("/")

    async def execute(self, **kwargs: Any) -> str:
        topic = kwargs["topic"]
        level = kwargs.get("level", "beginner")
        category = kwargs.get("category", "")
        duration = kwargs.get("duration", "")
        instructor = kwargs.get("instructor_name", "AI Instructor")

        # Step 1: Try OpenMAIC classroom generation
        classroom_data = None
        openmaic_available = await self._openmaic.health()

        if openmaic_available:
            try:
                logger.info(f"Generating classroom via OpenMAIC: {topic!r}")
                classroom_data = await self._openmaic.generate_and_wait(topic)
            except Exception as e:
                logger.warning(f"OpenMAIC generation failed, creating course manually: {e}")

        # Step 2: Create course in SBP backend
        course_payload = {
            "title": topic,
            "description": f"AI-generated course: {topic}",
            "level": level,
            "category": category,
            "duration": duration,
            "instructor_name": instructor,
            "tags": ["ai-generated"],
        }

        async with httpx.AsyncClient(timeout=SBP_TIMEOUT) as client:
            resp = await client.post(
                f"{self._sbp}/api/academy/courses",
                json=course_payload,
            )
            resp.raise_for_status()
            course = resp.json()
            course_id = course["id"]

        # Step 3: If we have OpenMAIC classroom data, create modules/lessons from scenes
        modules_created = 0
        lessons_created = 0
        quizzes_created = 0

        if classroom_data:
            scenes = classroom_data.get("scenes") or classroom_data.get("data", {}).get("scenes", [])
            outlines = classroom_data.get("outlines", [])

            # Group scenes into modules by outline sections
            module_map: dict[str, str] = {}  # section_name -> module_id

            for i, scene in enumerate(scenes):
                scene_type = scene.get("type", "slide")
                section = scene.get("section", f"Module {i // 3 + 1}")

                # Create module if not exists
                if section not in module_map:
                    async with httpx.AsyncClient(timeout=SBP_TIMEOUT) as client:
                        mod_resp = await client.post(
                            f"{self._sbp}/api/academy/courses/{course_id}/modules",
                            json={
                                "title": section,
                                "order_index": len(module_map),
                            },
                        )
                        if mod_resp.status_code < 300:
                            mod = mod_resp.json()
                            module_map[section] = mod["id"]
                            modules_created += 1

                module_id = module_map.get(section)
                if not module_id:
                    continue

                # Determine lesson type from scene type
                lesson_type_map = {
                    "slide": "article",
                    "quiz": "quiz",
                    "interactive": "embed",
                    "pbl": "embed",
                }
                lesson_type = lesson_type_map.get(scene_type, "article")

                # Create lesson
                async with httpx.AsyncClient(timeout=SBP_TIMEOUT) as client:
                    lesson_resp = await client.post(
                        f"{self._sbp}/api/academy/modules/{module_id}/lessons",
                        json={
                            "title": scene.get("title", f"Lesson {i + 1}"),
                            "lesson_type": lesson_type,
                            "content_text": json.dumps(scene),
                            "order_index": i,
                        },
                    )
                    if lesson_resp.status_code < 300:
                        lesson = lesson_resp.json()
                        lessons_created += 1

                        # Create quiz if scene is a quiz type
                        if scene_type == "quiz" and scene.get("questions"):
                            quiz_resp = await client.post(
                                f"{self._sbp}/api/academy/quizzes",
                                json={
                                    "lesson_id": lesson["id"],
                                    "title": scene.get("title", "Quiz"),
                                    "passing_score": 70.0,
                                },
                            )
                            if quiz_resp.status_code < 300:
                                quiz = quiz_resp.json()
                                quizzes_created += 1
                                # Add questions
                                for q_idx, q in enumerate(scene["questions"]):
                                    await client.post(
                                        f"{self._sbp}/api/academy/quizzes/{quiz['id']}/questions",
                                        json={
                                            "quiz_id": quiz["id"],
                                            "question_text": q.get("question", q.get("text", "")),
                                            "question_type": q.get("type", "multiple_choice"),
                                            "options": q.get("options", []),
                                            "correct_answer": q.get("answer", q.get("correctAnswer", "")),
                                            "points": q.get("points", 1),
                                            "order_index": q_idx,
                                        },
                                    )

        # Step 4: Publish the course
        async with httpx.AsyncClient(timeout=SBP_TIMEOUT) as client:
            await client.patch(
                f"{self._sbp}/api/academy/courses/{course_id}",
                json={"state": "published"},
            )

        parts = [f"Course created: **{topic}** (ID: {course_id})"]
        if classroom_data:
            parts.append(f"Generated {modules_created} modules, {lessons_created} lessons, {quizzes_created} quizzes via OpenMAIC")
        else:
            parts.append("Course shell created (OpenMAIC not available — add modules/lessons manually)")
        parts.append(f"View at: /academy/courses/{course_id}")
        return "\n".join(parts)


class AcademyListCoursesTool(Tool):
    """List available courses from the academy backend."""

    name = "academy_list_courses"
    description = "List available courses in the academy. Can filter by category, level, or state."
    parameters = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Filter by category"},
            "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
            "state": {"type": "string", "enum": ["draft", "published", "archived"], "default": "published"},
            "limit": {"type": "integer", "description": "Max results (1-50)", "default": 20},
        },
    }

    def __init__(self, *, sbp_api: str):
        self._sbp = sbp_api.rstrip("/")

    async def execute(self, **kwargs: Any) -> str:
        params: dict[str, Any] = {"limit": kwargs.get("limit", 20)}
        if kwargs.get("category"):
            params["category"] = kwargs["category"]
        if kwargs.get("level"):
            params["level"] = kwargs["level"]
        if kwargs.get("state"):
            params["state"] = kwargs["state"]

        async with httpx.AsyncClient(timeout=SBP_TIMEOUT) as client:
            resp = await client.get(f"{self._sbp}/api/academy/courses", params=params)
            resp.raise_for_status()
            courses = resp.json()

        if not courses:
            return "No courses found matching your criteria."

        lines = [f"Found {len(courses)} course(s):\n"]
        for c in courses:
            level = c.get("level", "").capitalize()
            cat = c.get("category", "")
            enrolled = c.get("enrolled_count", 0)
            modules = c.get("module_count", 0)
            lessons = c.get("lesson_count", 0)
            lines.append(
                f"- **{c['title']}** (ID: {c['id']})\n"
                f"  {level} | {cat} | {modules} modules, {lessons} lessons | {enrolled} enrolled"
            )
        return "\n".join(lines)


class AcademyGenerateQuizTool(Tool):
    """Generate quiz questions for a lesson/topic using OpenMAIC."""

    name = "academy_generate_quiz"
    description = (
        "Generate quiz questions for a specific topic or lesson using AI. "
        "Creates the quiz and questions in the academy backend."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Quiz topic or lesson content to quiz on"},
            "lesson_id": {"type": "string", "description": "Lesson ID to attach the quiz to"},
            "num_questions": {"type": "integer", "description": "Number of questions (1-20)", "default": 5},
            "question_types": {
                "type": "string",
                "description": "Comma-separated types: multiple_choice, true_false, short_answer",
                "default": "multiple_choice",
            },
        },
        "required": ["topic"],
    }

    def __init__(self, *, openmaic: OpenMAICClient, sbp_api: str):
        self._openmaic = openmaic
        self._sbp = sbp_api.rstrip("/")

    async def execute(self, **kwargs: Any) -> str:
        topic = kwargs["topic"]
        lesson_id = kwargs.get("lesson_id")
        num_q = min(kwargs.get("num_questions", 5), 20)

        # Use OpenMAIC to generate quiz scene
        openmaic_available = await self._openmaic.health()
        if not openmaic_available:
            return "OpenMAIC is not available. Please ensure it's running on port 3001."

        try:
            # Generate a quiz scene via outline + content
            outlines = await self._openmaic.generate_outlines(
                f"Create a quiz with {num_q} questions about: {topic}"
            )
            if not outlines:
                return "Failed to generate quiz outlines."

            # Get the first quiz-type outline
            quiz_outline = next(
                (o for o in outlines if o.get("type") == "quiz"),
                outlines[0] if outlines else None,
            )
            if not quiz_outline:
                return "No quiz outline generated."

            scene = await self._openmaic.generate_scene_content(
                quiz_outline, all_outlines=outlines
            )
        except Exception as e:
            return f"Quiz generation failed: {e}"

        # Extract questions from scene
        questions = scene.get("questions", [])
        if not questions:
            return "Generated scene contained no quiz questions."

        # Create quiz in SBP backend
        if lesson_id:
            async with httpx.AsyncClient(timeout=SBP_TIMEOUT) as client:
                quiz_resp = await client.post(
                    f"{self._sbp}/api/academy/quizzes",
                    json={
                        "lesson_id": lesson_id,
                        "title": f"Quiz: {topic}",
                        "passing_score": 70.0,
                    },
                )
                if quiz_resp.status_code >= 300:
                    return f"Failed to create quiz: {quiz_resp.text}"
                quiz = quiz_resp.json()

                for i, q in enumerate(questions[:num_q]):
                    await client.post(
                        f"{self._sbp}/api/academy/quizzes/{quiz['id']}/questions",
                        json={
                            "quiz_id": quiz["id"],
                            "question_text": q.get("question", q.get("text", "")),
                            "question_type": q.get("type", "multiple_choice"),
                            "options": q.get("options", []),
                            "correct_answer": q.get("answer", q.get("correctAnswer", "")),
                            "points": q.get("points", 1),
                            "order_index": i,
                        },
                    )

            return f"Quiz created with {min(len(questions), num_q)} questions (ID: {quiz['id']}) attached to lesson {lesson_id}"

        # Return questions as text if no lesson_id
        lines = [f"Generated {len(questions)} quiz questions about '{topic}':\n"]
        for i, q in enumerate(questions[:num_q], 1):
            lines.append(f"{i}. {q.get('question', q.get('text', ''))}")
            if q.get("options"):
                for j, opt in enumerate(q["options"]):
                    lines.append(f"   {chr(65+j)}) {opt}")
            lines.append(f"   Answer: {q.get('answer', q.get('correctAnswer', ''))}\n")
        return "\n".join(lines)


class AcademyGradeQuizTool(Tool):
    """Grade quiz answers using OpenMAIC's AI grading."""

    name = "academy_grade_quiz"
    description = (
        "Grade a student's quiz answers using AI. For short-answer questions, "
        "uses AI to evaluate the response quality and provide feedback."
    )
    parameters = {
        "type": "object",
        "properties": {
            "quiz_id": {"type": "string", "description": "Quiz ID to grade"},
            "user_id": {"type": "string", "description": "Student user ID"},
            "answers": {
                "type": "object",
                "description": "Map of question_id -> student's answer",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["quiz_id", "user_id", "answers"],
    }

    def __init__(self, *, openmaic: OpenMAICClient, sbp_api: str):
        self._openmaic = openmaic
        self._sbp = sbp_api.rstrip("/")

    async def execute(self, **kwargs: Any) -> str:
        quiz_id = kwargs["quiz_id"]
        user_id = kwargs["user_id"]
        answers = kwargs["answers"]

        # Get quiz questions from SBP
        async with httpx.AsyncClient(timeout=SBP_TIMEOUT) as client:
            quiz_resp = await client.get(f"{self._sbp}/api/academy/quizzes/{quiz_id}")
            if quiz_resp.status_code != 200:
                return f"Quiz {quiz_id} not found."
            quiz = quiz_resp.json()

        questions = quiz.get("questions", [])
        total_points = 0
        earned_points = 0
        feedback_parts = []

        openmaic_available = await self._openmaic.health()

        for q in questions:
            q_id = q["id"]
            student_answer = answers.get(q_id, "")
            correct_answer = q.get("correct_answer", "")
            points = q.get("points", 1)
            total_points += points

            if q.get("question_type") == "short_answer" and openmaic_available:
                # AI grading for short-answer
                try:
                    result = await self._openmaic.grade_quiz(
                        question=q["question_text"],
                        user_answer=student_answer,
                        points=points,
                    )
                    score = result.get("score", 0)
                    comment = result.get("comment", "")
                    earned_points += score
                    feedback_parts.append(f"Q: {q['question_text']}\n  Score: {score}/{points} — {comment}")
                except Exception:
                    # Fallback to exact match
                    if student_answer.strip().lower() == correct_answer.strip().lower():
                        earned_points += points
                        feedback_parts.append(f"Q: {q['question_text']}\n  Correct! ({points}/{points})")
                    else:
                        feedback_parts.append(f"Q: {q['question_text']}\n  Incorrect (0/{points})")
            else:
                # Exact match for multiple choice / true-false
                if student_answer.strip().lower() == correct_answer.strip().lower():
                    earned_points += points
                    feedback_parts.append(f"Q: {q['question_text']}\n  Correct! ({points}/{points})")
                else:
                    feedback_parts.append(f"Q: {q['question_text']}\n  Incorrect (0/{points}) — Answer: {correct_answer}")

        passed = earned_points >= (quiz.get("passing_score", 70) / 100 * total_points)

        # Submit results to SBP
        async with httpx.AsyncClient(timeout=SBP_TIMEOUT) as client:
            await client.post(
                f"{self._sbp}/api/academy/quizzes/{quiz_id}/submit",
                json={"user_id": user_id, "answers": answers},
            )

        score_pct = round(earned_points / total_points * 100) if total_points else 0
        result_str = "PASSED" if passed else "FAILED"
        return f"**Quiz Results: {result_str}** — {earned_points}/{total_points} points ({score_pct}%)\n\n" + "\n\n".join(feedback_parts)


class AcademyTutorTool(Tool):
    """AI tutor powered by OpenMAIC's multi-agent system."""

    name = "academy_tutor"
    description = (
        "Chat with an AI tutor about course content. The tutor uses OpenMAIC's "
        "multi-agent system with a teacher agent to provide explanations, answer "
        "questions, and guide learning."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Student's question or message"},
            "course_context": {"type": "string", "description": "Course title or topic for context"},
            "lesson_context": {"type": "string", "description": "Current lesson content for context"},
        },
        "required": ["message"],
    }

    def __init__(self, *, openmaic: OpenMAICClient):
        self._openmaic = openmaic

    async def execute(self, **kwargs: Any) -> str:
        message = kwargs["message"]
        course_ctx = kwargs.get("course_context", "")
        lesson_ctx = kwargs.get("lesson_context", "")

        openmaic_available = await self._openmaic.health()
        if not openmaic_available:
            return (
                "The AI tutor (OpenMAIC) is not currently available. "
                "I can still help answer your question directly. What would you like to know?"
            )

        # Build context-aware messages for OpenMAIC
        system_msg = (
            "You are a helpful and encouraging tutor. "
            "Explain concepts clearly, use examples, and check for understanding."
        )
        if course_ctx:
            system_msg += f"\n\nCourse context: {course_ctx}"
        if lesson_ctx:
            system_msg += f"\n\nCurrent lesson: {lesson_ctx}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": message},
        ]

        # Stream response from OpenMAIC teacher agent
        response_parts = []
        try:
            async for event in self._openmaic.chat(
                messages,
                agent_config={"agentIds": ["teacher"]},
            ):
                text = event.get("text") or event.get("content", "")
                if text:
                    response_parts.append(text)
        except Exception as e:
            logger.warning(f"OpenMAIC tutor chat failed: {e}")
            return f"Tutor encountered an error: {e}. Try asking me directly instead."

        if not response_parts:
            return "The tutor didn't generate a response. Please try rephrasing your question."

        return "".join(response_parts)


# All tools for convenient registration
ALL_ACADEMY_TOOLS = [
    AcademyCreateCourseTool,
    AcademyListCoursesTool,
    AcademyGenerateQuizTool,
    AcademyGradeQuizTool,
    AcademyTutorTool,
]
