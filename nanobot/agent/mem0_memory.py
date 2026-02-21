"""Mem0 semantic memory integration for nanobot."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger


class _NanobotLLM:
    """Custom mem0 LLM that delegates to the nanobot provider.

    mem0 expects a synchronous generate_response(). Since this runs inside
    asyncio.to_thread(), we schedule the async provider call back on the
    main event loop via run_coroutine_threadsafe().
    """

    def __init__(self, provider: Any, model: str, loop: asyncio.AbstractEventLoop):
        self._provider = provider
        self._model = model
        self._loop = loop

    # mem0 only reads self.config.model in a couple of places
    class _FakeConfig:
        def __init__(self, model: str):
            self.model = model
            self.temperature = 0.1
            self.max_tokens = 1500
            self.top_p = 0.1

    @property
    def config(self):
        return self._FakeConfig(self._model)

    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto"):
        """Synchronous entry point called by mem0 (runs in a worker thread)."""
        coro = self._provider.chat(
            messages=messages,
            model=self._model,
            temperature=0.1,
            max_tokens=1500,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        resp = future.result(timeout=60)

        # mem0 expects either a plain string or {"content": ..., "tool_calls": [...]}
        if tools:
            return {"content": resp.content or "", "tool_calls": []}
        return resp.content or ""


class Mem0Store:
    """Async wrapper around mem0's Memory class.

    Uses ChromaDB (in-process) for vectors and sentence-transformers for
    local embeddings.  The LLM for fact extraction is the same provider
    nanobot already uses.
    """

    def __init__(
        self,
        workspace: Path,
        provider: Any,
        model: str,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        from mem0 import Memory

        self._loop = loop or asyncio.get_event_loop()
        chroma_path = str(workspace / "mem0_chroma")
        history_path = str(workspace / "mem0_history.db")

        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "nanobot_memories",
                    "path": chroma_path,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "all-MiniLM-L6-v2",
                    "model_kwargs": {"device": "cpu"},
                },
            },
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": "openai/gpt-4o-mini",  # placeholder, replaced below
                    "temperature": 0.1,
                    "max_tokens": 1500,
                },
            },
            "history_db_path": history_path,
            "version": "v1.1",
        }

        self._memory = Memory.from_config(config)

        # Replace the LLM with our custom wrapper that uses the nanobot provider
        self._memory.llm = _NanobotLLM(provider, model, self._loop)

        logger.info(f"Mem0 semantic memory initialized (chroma: {chroma_path})")

    async def search(self, query: str, user_id: str = "joel", limit: int = 5) -> list[str]:
        """Search for relevant memories."""
        try:
            results = await asyncio.to_thread(
                self._memory.search, query=query, user_id=user_id, limit=limit
            )
            if not results:
                return []
            # results is a list of dicts with "memory" key
            memories = []
            for r in results:
                if isinstance(r, dict) and r.get("memory"):
                    memories.append(r["memory"])
                elif isinstance(r, str):
                    memories.append(r)
            return memories
        except Exception as e:
            logger.warning(f"Mem0 search failed: {e}")
            return []

    async def add(self, messages: list[dict], user_id: str = "joel") -> None:
        """Store a conversation exchange — mem0 auto-extracts facts."""
        try:
            await asyncio.to_thread(
                self._memory.add, messages=messages, user_id=user_id
            )
        except Exception as e:
            logger.warning(f"Mem0 add failed: {e}")
