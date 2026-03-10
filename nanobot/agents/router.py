"""Intent-based router for direct agent dispatch.

Bypasses the CEO agent by classifying user intent from the message
and routing directly to the appropriate team lead. Zero LLM calls —
pure keyword matching for instant routing.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from nanobot.agents.registry import AgentRegistry


# ── Routing rules ──
# Each entry: (agent_name, keywords, description_for_logging)
# Order matters — first match wins. More specific patterns go first.
ROUTING_RULES: list[tuple[str, list[str], str]] = [
    # ── Content & articles ──
    (
        "content_manager",
        [
            "article", "blog", "write", "publish", "editorial",
            "newsletter", "content", "news pipeline", "daily news",
            "headline", "story", "stories", "press release",
            "social media", "post", "tweet", "instagram",
            "street voices", "streetvoices",
        ],
        "content/editorial",
    ),
    # ── Communication ──
    (
        "communication_manager",
        [
            "email", "slack", "whatsapp", "message", "send",
            "reply", "forward", "inbox", "notification",
            "telegram", "sms", "text message", "call",
            "contact", "outreach", "respond",
        ],
        "communication",
    ),
    # ── Development ──
    (
        "development_manager",
        [
            "code", "develop", "build", "deploy", "bug",
            "feature", "api", "server", "database", "frontend",
            "backend", "git", "github", "pr", "pull request",
            "debug", "test", "ci", "pipeline", "docker",
            "infrastructure", "devops", "website", "app",
            "programming", "software", "script",
        ],
        "development",
    ),
    # ── Finance ──
    (
        "finance_manager",
        [
            "invoice", "budget", "expense", "revenue",
            "financial", "accounting", "bookkeeping", "tax",
            "payment", "payroll", "profit", "loss",
            "crypto", "bitcoin", "ethereum", "wallet",
            "transaction", "bank", "funding",
        ],
        "finance",
    ),
    # ── Grant writing ──
    (
        "grant_manager",
        [
            "grant", "funding application", "proposal",
            "funder", "foundation", "nonprofit funding",
            "grant writing", "letter of intent", "loi",
            "grant report", "grant deadline",
        ],
        "grants",
    ),
    # ── Research ──
    (
        "research_manager",
        [
            "research", "analyze", "analysis", "report",
            "investigate", "study", "data", "statistics",
            "survey", "findings", "literature review",
            "intelligence", "briefing", "competitor",
            "market", "trend",
        ],
        "research",
    ),
    # ── Scraping ──
    (
        "scraping_manager",
        [
            "scrape", "crawl", "extract", "harvest",
            "web scraping", "spider", "parse html",
            "data extraction", "monitor site",
        ],
        "scraping",
    ),
    # ── Security ──
    (
        "security_compliance",
        [
            "security", "compliance", "audit", "vulnerability",
            "access control", "permissions", "threat",
            "privacy", "gdpr", "encryption",
        ],
        "security",
    ),
]

# Pre-compile patterns for performance
_COMPILED_RULES: list[tuple[str, re.Pattern, str]] = []
for _agent, _keywords, _desc in ROUTING_RULES:
    # Build a single regex that matches any keyword as a whole word
    _pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(kw) for kw in _keywords) + r")\b",
        re.IGNORECASE,
    )
    _COMPILED_RULES.append((_agent, _pattern, _desc))


def route_request(
    messages: list[dict[str, Any]],
    registry: AgentRegistry,
    fallback: str = "ceo",
) -> str:
    """
    Classify user intent and return the best agent name to handle it.

    Scans ALL user messages for keyword matches against routing rules.
    This handles multi-turn conversations where the first message has
    the intent ("write an article") but follow-ups are contextual
    ("anyone", "yes", "that one").

    Returns the agent with the most keyword hits across the full
    conversation. Falls back to CEO for ambiguous or unclassifiable
    requests.

    Args:
        messages: Conversation messages (scans all user messages).
        registry: Agent registry to verify agent exists.
        fallback: Agent to use when no rules match.

    Returns:
        Name of the agent to route to.
    """
    # Extract ALL user message text for classification
    user_text = _extract_all_user_text(messages)
    if not user_text:
        logger.debug("Router: no user text found, falling back to CEO")
        return fallback

    # Score each rule by number of keyword matches across all messages
    scores: list[tuple[str, int, str]] = []
    for agent_name, pattern, desc in _COMPILED_RULES:
        matches = pattern.findall(user_text)
        if matches:
            scores.append((agent_name, len(matches), desc))

    if not scores:
        logger.info(f"Router: no keyword matches, routing to '{fallback}'")
        return fallback

    # Sort by score descending, pick the winner
    scores.sort(key=lambda x: x[1], reverse=True)
    winner, score, desc = scores[0]

    # Verify the agent exists in registry
    if not registry.has(winner):
        logger.warning(
            f"Router: matched '{winner}' but not in registry, "
            f"falling back to '{fallback}'"
        )
        return fallback

    logger.info(
        f"Router: '{user_text[:80]}...' → {winner} "
        f"({desc}, score={score})"
    )
    return winner


def _extract_all_user_text(messages: list[dict[str, Any]]) -> str:
    """
    Extract text from ALL user messages for classification.

    Combines all user messages so keywords from earlier turns
    (e.g. "write an article") still contribute to routing even
    when the latest message is just "anyone" or "yes".
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content.strip()
            if text:
                parts.append(text)
        elif isinstance(content, list):
            # Handle multimodal content (list of parts)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "").strip()
                    if text:
                        parts.append(text)
    return " ".join(parts)
