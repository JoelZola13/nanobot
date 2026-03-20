"""Unified search tool: searches across SV Social, LibreChat (MeiliSearch), and more."""

import json
from typing import Any

import asyncpg

from nanobot.agent.tools.base import Tool

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


class UnifiedSearchTool(Tool):
    """Search across all platform data sources at once."""

    name = "unified_search"
    description = (
        "Search across the entire platform: SV Social messages, LibreChat conversation history, "
        "and directory/services. Returns results ranked by recency from all sources. "
        "Use this for broad searches that span multiple features."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text",
            },
            "sources": {
                "type": "string",
                "description": "Comma-separated sources to search: 'social,chat,directory' (default: all)",
            },
            "count": {
                "type": "integer",
                "description": "Max results per source (default 5, max 15)",
                "minimum": 1,
                "maximum": 15,
            },
        },
        "required": ["query"],
    }

    def __init__(self, pool: asyncpg.Pool | None = None, meili_url: str = "http://localhost:7700", meili_key: str = ""):
        self._pool = pool
        self._meili_url = meili_url
        self._meili_key = meili_key

    async def execute(self, query: str, sources: str = "social,chat,directory", count: int = 5, **kwargs) -> str:
        count = min(max(1, count), 15)
        source_list = [s.strip().lower() for s in sources.split(",")]
        results = []

        # Search Social messages (PostgreSQL)
        if "social" in source_list and self._pool:
            try:
                social_results = await self._search_social(query, count)
                if social_results:
                    results.append(("SV Social Messages", social_results))
            except Exception as e:
                results.append(("SV Social Messages", [f"Error: {e}"]))

        # Search LibreChat conversations + messages (MeiliSearch)
        if "chat" in source_list and self._meili_key:
            try:
                chat_results = await self._search_meili("messages", query, count)
                if chat_results:
                    results.append(("LibreChat Messages", chat_results))
            except Exception as e:
                results.append(("LibreChat Messages", [f"Error: {e}"]))

        # Search directory/services (MeiliSearch)
        if "directory" in source_list and self._meili_key:
            try:
                dir_results = await self._search_meili("directory_services", query, count)
                if dir_results:
                    results.append(("Directory & Services", dir_results))
            except Exception as e:
                results.append(("Directory & Services", [f"Error: {e}"]))

        if not results:
            return f"No results found for '{query}' across any source."

        # Format output
        lines = [f"Unified search results for '{query}':\n"]
        for source_name, items in results:
            lines.append(f"── {source_name} ({len(items)} results) ──")
            for item in items:
                lines.append(f"  {item}")
            lines.append("")

        return "\n".join(lines)

    async def _search_social(self, query: str, limit: int) -> list[str]:
        """Search SV Social messages via PostgreSQL."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT m.content, m.created_at, u.display_name, c.name AS channel_name
                   FROM messages m
                   JOIN users u ON m.author_id = u.id
                   JOIN channels c ON m.channel_id = c.id
                   WHERE m.deleted_at IS NULL AND m.content ILIKE $1
                   ORDER BY m.created_at DESC LIMIT $2""",
                f"%{query}%", limit,
            )
            results = []
            for r in rows:
                ts = r["created_at"].strftime("%Y-%m-%d %H:%M")
                ch = f"#{r['channel_name']}" if r["channel_name"] else "DM"
                results.append(f"[{ts}] {ch} — {r['display_name']}: {r['content'][:200]}")
            return results

    async def _search_meili(self, index: str, query: str, limit: int) -> list[str]:
        """Search MeiliSearch index."""
        if not httpx:
            return ["httpx not installed"]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._meili_url}/indexes/{index}/search",
                headers={"Authorization": f"Bearer {self._meili_key}"},
                json={"q": query, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for hit in data.get("hits", []):
            if index == "messages":
                text = hit.get("text", hit.get("content", ""))[:200]
                sender = hit.get("sender", "")
                results.append(f"{sender}: {text}")
            elif index == "directory_services":
                name = hit.get("name", hit.get("title", ""))
                desc = hit.get("description", "")[:100]
                results.append(f"{name} — {desc}")
            else:
                results.append(str(hit)[:200])
        return results
