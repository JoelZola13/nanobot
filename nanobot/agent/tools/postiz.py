"""Postiz publishing tool for social posts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

import httpx

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.web import WebFetchTool


class PostizPublishTool(Tool):
    """Create a post in Postiz."""

    def __init__(self, config):
        self.config = config
        self._fetch_tool = WebFetchTool()

    @property
    def name(self) -> str:
        return "postiz_publish"

    @property
    def description(self) -> str:
        return (
            "Publish a post via Postiz. If no caption is provided, "
            "the tool can draft one from a source URL. Useful for "
            "social workflows into Instagram accounts configured in Postiz."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "caption": {"type": "string", "description": "Post text/copy."},
                "sourceUrl": {
                    "type": "string",
                    "description": "If caption is missing, fetch and draft from this URL.",
                },
                "targetHandle": {
                    "type": "string",
                    "description": "Instagram handle or target name in Postiz.",
                    "default": self.config.default_target_handle,
                },
                "platform": {
                    "type": "string",
                    "description": "Target platform slug (e.g. instagram).",
                    "default": self.config.default_platform,
                },
                "mediaUrls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional image/video URLs.",
                },
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional hashtags, without #.",
                },
                "publish": {"type": "boolean", "description": "Publish immediately (default true).", "default": True},
                "path": {"type": "string", "description": "Override path relative to Postiz base URL."},
                "method": {"type": "string", "enum": ["POST", "PATCH", "PUT"], "default": "POST"},
                "payload": {
                    "type": "object",
                    "description": "Send a custom payload instead of the default auto payload.",
                },
                "dryRun": {"type": "boolean", "description": "Build payload and preview without sending.", "default": False},
                "timeoutSeconds": {
                    "type": "integer",
                    "description": "Request timeout for Postiz (seconds).",
                    "minimum": 1,
                    "maximum": 120,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        caption: str | None = None,
        sourceUrl: str | None = None,
        targetHandle: str | None = None,
        platform: str | None = None,
        mediaUrls: list[str] | None = None,
        hashtags: list[str] | None = None,
        publish: bool = True,
        path: str | None = None,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
        dryRun: bool = False,
        timeoutSeconds: int | None = None,
        **kwargs: Any,
    ) -> str:
        if not self.config.enabled:
            return "Error: Postiz tool is disabled. Enable tools.postiz.enabled in config."

        final_caption = caption or ""
        if not final_caption and sourceUrl:
            source = await self._fetch_tool.execute(url=sourceUrl, extractMode="text", maxChars=5000)
            final_caption = self._extract_text(source)
            if not final_caption:
                return f"Error: Failed to derive caption from source URL: {sourceUrl}"

        if not final_caption:
            return "Error: caption is required when sourceUrl is not provided."

        # Build defaults and normalize
        target_handle = (targetHandle or self.config.default_target_handle or "").strip()
        platform_value = (platform or self.config.default_platform or "instagram").strip().lower()
        max_chars = getattr(self.config, "default_max_caption_chars", 2200)
        final_caption = self._trim_caption(final_caption, max_chars)
        if hashtags:
            tag_block = " ".join(f"#{tag.strip().lstrip('#')}" for tag in hashtags if tag and tag.strip())
            if tag_block:
                final_caption = f"{final_caption}\n\n{tag_block}"

        if payload is not None and isinstance(payload, Mapping):
            request_body = dict(payload)
            if "caption" not in request_body and final_caption:
                request_body["caption"] = final_caption
            request_body.setdefault("platform", platform_value)
            if target_handle:
                request_body.setdefault("targetHandle", target_handle)
            request_body.setdefault("publish", bool(publish))
        else:
            request_body = {
                "caption": final_caption,
                "platform": platform_value,
                "publish": bool(publish),
            }
            if target_handle:
                request_body["targetHandle"] = target_handle
            if mediaUrls:
                request_body["mediaUrls"] = mediaUrls

        final_path = path or self.config.publish_path
        endpoint = self._join_url(self.config.base_url, final_path)
        timeout = timeoutSeconds or getattr(self.config, "request_timeout", 30)
        method = method.upper()
        headers = self._build_headers()

        if dryRun:
            return json.dumps(
                {
                    "status": "dry_run",
                    "method": method,
                    "endpoint": endpoint,
                    "headers": self._redact_headers(headers),
                    "payload": request_body,
                },
                ensure_ascii=False,
                indent=2,
            )

        if method not in {"POST", "PATCH", "PUT"}:
            return "Error: method must be one of POST, PATCH, PUT"

        try:
            async with httpx.AsyncClient(timeout=float(timeout)) as client:
                response = await client.request(method, endpoint, json=request_body, headers=headers)
                response.raise_for_status()
        except httpx.ConnectError as e:
            return (
                "Error: cannot reach Postiz. "
                f"Check that {self.config.base_url!r} is running and reachable. "
                f"Original error: {e}"
            )
        except httpx.HTTPError as e:
            return f"Error: Postiz request failed: {e}"

        try:
            if "application/json" in response.headers.get("content-type", ""):
                data = response.json()
                return json.dumps(data, ensure_ascii=False, indent=2)
            return response.text
        except Exception as e:
            return f"Error parsing Postiz response: {e}\nRaw: {response.text}"

    @staticmethod
    def _trim_caption(text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _extract_text(fetch_result: str) -> str:
        if not fetch_result:
            return ""
        try:
            raw = json.loads(fetch_result)
        except Exception:
            return fetch_result.strip()
        if isinstance(raw, dict):
            return str(raw.get("text") or raw.get("title") or "").strip()
        return str(raw).strip()

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        base = base_url.rstrip("/")
        if not path:
            return base
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = f"/{path}"
        return urljoin(f"{base}/", path.lstrip("/"))

    def _build_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        token = (getattr(self.config, "api_key", "") or "").strip()
        if token:
            key_name = getattr(self.config, "api_key_header", "Authorization")
            key_prefix = getattr(self.config, "api_key_prefix", "Bearer")
            headers[key_name] = f"{key_prefix} {token}".strip() if key_name.lower() == "authorization" else token
        extra_headers = getattr(self.config, "extra_headers", None)
        if isinstance(extra_headers, Mapping):
            headers.update(extra_headers)
        return headers

    @staticmethod
    def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
        safe = dict(headers)
        for key in list(safe.keys()):
            if key.lower() == "authorization":
                safe[key] = "****"
        return safe
