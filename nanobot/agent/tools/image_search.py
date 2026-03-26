"""Image search tool using Brave Image Search API."""

from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


class ImageSearchTool(Tool):
    """Search for images using Brave Image Search API."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "image_search"

    @property
    def description(self) -> str:
        return (
            "Search for images by keyword. Returns image URLs, titles, "
            "dimensions, and source pages. Useful for finding hero photos "
            "for articles."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for images",
                },
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
                    "https://api.search.brave.com/res/v1/images/search",
                    params={"q": query, "count": max_results, "safesearch": "strict"},
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self._api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("error", {}).get("detail", "")
            except Exception:
                detail = e.response.text[:200]
            return f"Error: Brave API returned HTTP {e.response.status_code}: {detail}"
        except httpx.RequestError as e:
            return f"Error: Request failed: {e}"

        results = data.get("results", [])
        if not results:
            return f"No images found for: {query}"

        lines = [f"Image results for: {query}\n"]
        for i, r in enumerate(results[:max_results], 1):
            title = r.get("title", "No title")
            img_url = r.get("properties", {}).get("url", r.get("url", ""))
            source_url = r.get("url", "")
            source = r.get("source", "")
            width = r.get("properties", {}).get("width", "")
            height = r.get("properties", {}).get("height", "")

            lines.append(f"{i}. {title}")
            if source:
                lines.append(f"   Source: {source}")
            if width and height:
                lines.append(f"   Dimensions: {width}x{height}")
            if img_url:
                lines.append(f"   Image URL: {img_url}")
                # Include markdown image so agent can embed it inline
                lines.append(f"   Preview: ![{title}]({img_url})")
            if source_url and source_url != img_url:
                lines.append(f"   Page URL: {source_url}")
            lines.append("")

        lines.append(
            "TIP: To show any image inline in the conversation, include it as "
            "markdown: ![description](image_url)"
        )
        return "\n".join(lines)
