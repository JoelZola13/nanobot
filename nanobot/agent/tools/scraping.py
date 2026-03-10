"""Web scraping tools: web_scrape, html_parser, data_extractor."""

import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"


def _clean_text(html_content: str) -> str:
    """Strip tags and normalize whitespace."""
    text = re.sub(r'<script[\s\S]*?</script>', '', html_content, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', html_content, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class WebScrapeTool(Tool):
    """Scrape content from a web page."""

    @property
    def name(self) -> str:
        return "web_scrape"

    @property
    def description(self) -> str:
        return (
            "Scrape content from a web page URL. Optionally provide a CSS selector "
            "to target specific elements. Returns extracted text content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL of the page to scrape"},
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to target specific content",
                },
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url", "").strip()
        selector = kwargs.get("selector", "").strip()

        if not url:
            return "Error: No URL provided."

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Error: Only http/https URLs are supported, got '{parsed.scheme or 'none'}'"

        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=30, headers={"User-Agent": USER_AGENT}
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} fetching {url}"
        except httpx.RequestError as e:
            return f"Error: Request failed: {e}"

        if selector:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "html.parser")
                elements = soup.select(selector)
                if not elements:
                    return f"No elements found matching selector: {selector}"
                texts = [el.get_text(separator=" ", strip=True) for el in elements]
                return "\n\n---\n\n".join(texts)
            except ImportError:
                return "Error: beautifulsoup4 is required for CSS selector support. Install with: pip install beautifulsoup4"
            except Exception as e:
                return f"Error parsing with selector: {e}"

        # No selector — use readability if available, else strip tags
        try:
            from readability import Document

            doc = Document(html)
            content = doc.summary()
            return _clean_text(content)
        except ImportError:
            return _clean_text(html)[:10000]


class HtmlParserTool(Tool):
    """Parse and extract data from raw HTML."""

    @property
    def name(self) -> str:
        return "html_parser"

    @property
    def description(self) -> str:
        return "Parse HTML content and extract elements matching a CSS selector. Returns text content of matched elements."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "html": {"type": "string", "description": "HTML content to parse"},
                "selector": {
                    "type": "string",
                    "description": "CSS selector to extract elements",
                },
            },
            "required": ["html", "selector"],
        }

    async def execute(self, **kwargs: Any) -> str:
        html_content = kwargs.get("html", "")
        selector = kwargs.get("selector", "").strip()

        if not html_content:
            return "Error: No HTML content provided."
        if not selector:
            return "Error: No CSS selector provided."

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return "Error: beautifulsoup4 is required. Install with: pip install beautifulsoup4"

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            elements = soup.select(selector)

            if not elements:
                return f"No elements found matching selector: {selector}"

            results = []
            for i, el in enumerate(elements, 1):
                tag = el.name
                text = el.get_text(separator=" ", strip=True)
                attrs = dict(el.attrs) if el.attrs else {}
                entry = f"[{i}] <{tag}>"
                if attrs:
                    attr_str = ", ".join(f'{k}="{v}"' for k, v in list(attrs.items())[:5])
                    entry += f" ({attr_str})"
                entry += f"\n{text}"
                results.append(entry)

            return f"Found {len(elements)} element(s):\n\n" + "\n\n".join(results)
        except Exception as e:
            return f"Error parsing HTML: {e}"


class DataExtractorTool(Tool):
    """Extract structured data from text using pattern matching."""

    @property
    def name(self) -> str:
        return "data_extractor"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to extract data from",
                },
                "schema": {
                    "type": "string",
                    "description": "What to extract: 'emails', 'urls', 'phones', 'dates', 'prices', 'numbers', or 'all'",
                },
            },
            "required": ["content"],
        }

    @property
    def description(self) -> str:
        return (
            "Extract structured data from text content. Can extract emails, URLs, "
            "phone numbers, dates, prices, and numbers. Specify schema type or use 'all'."
        )

    _PATTERNS: dict[str, re.Pattern] = {
        "emails": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
        "urls": re.compile(r'https?://[^\s<>"\']+'),
        "phones": re.compile(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
        "dates": re.compile(
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
            r'|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
            r'|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            re.I,
        ),
        "prices": re.compile(r'\$[\d,]+(?:\.\d{2})?'),
        "numbers": re.compile(r'\b\d[\d,]*(?:\.\d+)?\b'),
    }

    async def execute(self, **kwargs: Any) -> str:
        content = kwargs.get("content", "")
        schema = kwargs.get("schema", "all").strip().lower()

        if not content:
            return "Error: No content provided."

        if schema == "all":
            extract_types = list(self._PATTERNS.keys())
        elif schema in self._PATTERNS:
            extract_types = [schema]
        else:
            return f"Error: Unknown schema '{schema}'. Use: {', '.join(self._PATTERNS.keys())} or 'all'"

        results: dict[str, list[str]] = {}
        for dtype in extract_types:
            matches = list(set(self._PATTERNS[dtype].findall(content)))
            if matches:
                results[dtype] = sorted(matches)

        if not results:
            return "No data found matching the requested patterns."

        return json.dumps(results, indent=2)
