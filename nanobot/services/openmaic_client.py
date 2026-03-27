"""Async HTTP client for OpenMAIC multi-agent classroom platform."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx
from loguru import logger

DEFAULT_BASE_URL = "http://localhost:3001"
POLL_INTERVAL = 3  # seconds between generation status polls
POLL_TIMEOUT = 300  # max seconds to wait for generation


class OpenMAICClient:
    """Wraps OpenMAIC's stateless REST/SSE API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Classroom generation (async job)
    # ------------------------------------------------------------------

    async def generate_classroom(
        self,
        topic: str,
        *,
        language: str = "en",
        pdf_content: bytes | None = None,
        api_keys: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Start classroom generation. Returns {jobId, pollUrl}."""
        payload: dict[str, Any] = {"requirement": topic, "language": language}
        headers = self._api_key_headers(api_keys)
        headers["x-model"] = "openai:gpt-5.4"

        async with httpx.AsyncClient(timeout=30) as client:
            if pdf_content:
                resp = await client.post(
                    f"{self.base_url}/api/generate-classroom",
                    data={"requirement": topic, "language": language},
                    files={"pdf": ("upload.pdf", pdf_content, "application/pdf")},
                    headers=headers,
                )
            else:
                resp = await client.post(
                    f"{self.base_url}/api/generate-classroom",
                    json=payload,
                    headers=headers,
                )
            resp.raise_for_status()
            return resp.json()

    async def poll_generation(self, job_id: str) -> dict[str, Any]:
        """Poll generation job status. Returns full job object."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/api/generate-classroom/{job_id}"
            )
            resp.raise_for_status()
            return resp.json()

    async def generate_and_wait(
        self,
        topic: str,
        *,
        language: str = "en",
        pdf_content: bytes | None = None,
        api_keys: dict[str, str] | None = None,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """Generate classroom and poll until complete. Returns classroom data."""
        result = await self.generate_classroom(
            topic, language=language, pdf_content=pdf_content, api_keys=api_keys
        )
        job_id = result.get("jobId") or result.get("id")
        if not job_id:
            raise ValueError(f"No jobId in generation response: {result}")

        logger.info(f"OpenMAIC generation started: job={job_id} topic={topic!r}")
        elapsed = 0
        while elapsed < POLL_TIMEOUT:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            status = await self.poll_generation(job_id)
            state = status.get("status") or status.get("state", "unknown")

            if on_progress:
                progress = status.get("progress", {})
                await on_progress(state, progress)

            if state in ("completed", "done", "success", "succeeded"):
                logger.info(f"OpenMAIC generation complete: job={job_id}")
                return status.get("result", status)
            elif state in ("failed", "error"):
                error = status.get("error", "Unknown error")
                raise RuntimeError(f"OpenMAIC generation failed: {error}")

        raise TimeoutError(
            f"OpenMAIC generation timed out after {POLL_TIMEOUT}s (job={job_id})"
        )

    # ------------------------------------------------------------------
    # Chat (stateless SSE streaming)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        *,
        stage_state: dict | None = None,
        agent_config: dict | None = None,
        api_keys: dict[str, str] | None = None,
    ) -> AsyncIterator[dict]:
        """Stream multi-agent chat responses via SSE."""
        payload: dict[str, Any] = {"messages": messages}
        if stage_state:
            payload["storeState"] = stage_state
        if agent_config:
            payload["config"] = agent_config
        headers = self._api_key_headers(api_keys)
        headers["Accept"] = "text/event-stream"
        headers["x-model"] = "openai:gpt-5.4"

        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            return
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue

    # ------------------------------------------------------------------
    # Quiz grading
    # ------------------------------------------------------------------

    async def grade_quiz(
        self,
        question: str,
        user_answer: str,
        points: int = 1,
        *,
        comment_prompt: str | None = None,
        language: str = "en",
        api_keys: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Grade a quiz answer using LLM. Returns {score, comment}."""
        payload: dict[str, Any] = {
            "question": question,
            "userAnswer": user_answer,
            "points": points,
        }
        if comment_prompt:
            payload["commentPrompt"] = comment_prompt
        if language != "en":
            payload["language"] = language

        headers = self._api_key_headers(api_keys)
        headers["x-model"] = "openai:gpt-5.4"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/api/quiz-grade",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Scene generation helpers
    # ------------------------------------------------------------------

    async def generate_outlines(
        self,
        topic: str,
        *,
        language: str = "en-US",
        reference_materials: list[str] | None = None,
        api_keys: dict[str, str] | None = None,
    ) -> list[dict]:
        """Generate scene outlines (non-streaming, collects all)."""
        payload: dict[str, Any] = {
            "requirements": {"requirement": topic, "language": language},
        }
        if reference_materials:
            payload["researchContext"] = "\n".join(reference_materials)
        headers = self._api_key_headers(api_keys)
        headers["Accept"] = "text/event-stream"
        headers["x-model"] = "openai:gpt-5.4"

        outlines: list[dict] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate/scene-outlines-stream",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            event = json.loads(data)
                            if event.get("type") == "outline":
                                outlines.append(event.get("data", event))
                            elif event.get("type") == "done":
                                break
                        except json.JSONDecodeError:
                            continue
        return outlines

    async def generate_scene_content(
        self,
        outline: dict,
        *,
        all_outlines: list[dict] | None = None,
        stage_info: dict | None = None,
        stage_id: str = "default",
        api_keys: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Generate rich content for a single scene outline."""
        payload: dict[str, Any] = {
            "outline": outline,
            "stageId": stage_id,
        }
        if all_outlines:
            payload["allOutlines"] = all_outlines
        if stage_info:
            payload["stageInfo"] = stage_info

        headers = self._api_key_headers(api_keys)
        headers["x-model"] = "openai:gpt-5.4"

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate/scene-content",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # PDF parsing
    # ------------------------------------------------------------------

    async def parse_pdf(
        self,
        pdf_bytes: bytes,
        *,
        api_keys: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Parse PDF and extract content."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/api/parse-pdf",
                files={"pdf": ("upload.pdf", pdf_bytes, "application/pdf")},
                headers=self._api_key_headers(api_keys),
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Check if OpenMAIC is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/health")
                return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _api_key_headers(api_keys: dict[str, str] | None) -> dict[str, str]:
        """Build headers for per-request API key overrides."""
        headers: dict[str, str] = {}
        if not api_keys:
            return headers
        # OpenMAIC accepts provider keys via headers
        for key, value in api_keys.items():
            headers[f"x-{key}"] = value
        return headers
