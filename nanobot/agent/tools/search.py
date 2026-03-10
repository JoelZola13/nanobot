"""Search tools: news_search, academic_search."""

import json
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"


class NewsSearchTool(Tool):
    """Search recent news using Brave Search API."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "news_search"

    @property
    def description(self) -> str:
        return "Search recent news articles by topic or keyword. Returns titles, URLs, descriptions, and publication dates."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for news articles"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 5, max 20)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        max_results = min(int(kwargs.get("max_results", 5)), 20)

        if not query:
            return "Error: No search query provided."
        if not self._api_key:
            return "Error: Brave API key not configured. Set brave_api_key in tool config."

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/news/search",
                    params={"q": query, "count": max_results},
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self._api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Error: Brave API returned HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Error: Request failed: {e}"

        results = data.get("results", [])
        if not results:
            return f"No news results found for: {query}"

        lines = [f"News results for: {query}\n"]
        for i, r in enumerate(results[:max_results], 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            desc = r.get("description", "")
            age = r.get("age", "")
            source = r.get("meta_url", {}).get("hostname", "")
            lines.append(f"{i}. {title}")
            if source:
                lines.append(f"   Source: {source}")
            if age:
                lines.append(f"   Published: {age}")
            if desc:
                lines.append(f"   {desc}")
            if url:
                lines.append(f"   {url}")
            lines.append("")

        return "\n".join(lines)


class AcademicSearchTool(Tool):
    """Search academic papers via Semantic Scholar API."""

    @property
    def name(self) -> str:
        return "academic_search"

    @property
    def description(self) -> str:
        return "Search academic papers and research publications. Returns titles, authors, year, citation count, and abstracts."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for academic papers"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 5, max 20)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        max_results = min(int(kwargs.get("max_results", 5)), 20)

        if not query:
            return "Error: No search query provided."

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": max_results,
                        "fields": "title,authors,year,citationCount,abstract,url",
                    },
                    headers={"User-Agent": USER_AGENT},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Error: Semantic Scholar API returned HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Error: Request failed: {e}"

        papers = data.get("data", [])
        if not papers:
            return f"No academic papers found for: {query}"

        lines = [f"Academic papers for: {query}\n"]
        for i, p in enumerate(papers[:max_results], 1):
            title = p.get("title", "Untitled")
            year = p.get("year", "n/a")
            citations = p.get("citationCount", 0)
            authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
            if len(p.get("authors") or []) > 3:
                authors += " et al."
            abstract = p.get("abstract", "")
            url = p.get("url", "")

            lines.append(f"{i}. {title} ({year})")
            if authors:
                lines.append(f"   Authors: {authors}")
            lines.append(f"   Citations: {citations}")
            if abstract:
                lines.append(f"   Abstract: {abstract[:300]}{'...' if len(abstract or '') > 300 else ''}")
            if url:
                lines.append(f"   {url}")
            lines.append("")

        return "\n".join(lines)
