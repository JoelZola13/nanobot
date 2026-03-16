"""Auto-memory middleware — automatically extracts and persists facts.

This runs after every agent conversation turn and extracts:
- Contact information mentioned
- Decisions or preferences expressed
- Project status updates
- Any explicit "remember this" instructions

It writes these to universal memory so they're available to all agents.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from nanobot.harness.memory import UniversalMemory


# Patterns that suggest memory-worthy content
CONTACT_PATTERNS = [
    r"(?:my|his|her|their)\s+(?:email|phone|number)\s+(?:is|:)\s+(.+)",
    r"(?:reach|contact)\s+(?:me|them|him|her)\s+(?:at|via)\s+(.+)",
    r"([A-Za-z]+ [A-Za-z]+)\s+(?:is|works at|from)\s+(.+)",
]

DECISION_PATTERNS = [
    r"(?:always|never|from now on|going forward|remember to)\s+(.+)",
    r"(?:i prefer|i want|i'd like|i need you to)\s+(.+)",
    r"(?:don't|do not|stop)\s+(.+?)(?:\.|$)",
]

PROJECT_PATTERNS = [
    r"(?:the|our)\s+(\w+)\s+project\s+(?:is|has|needs)\s+(.+)",
    r"(?:working on|started|launched|completed)\s+(.+)",
]


async def extract_and_save(
    memory: UniversalMemory,
    messages: list[dict[str, Any]],
    agent_name: str,
) -> None:
    """Extract memory-worthy facts from a conversation and save them.

    This is called after each agent turn completes. It's intentionally
    lightweight — just pattern matching, no LLM calls.

    Args:
        memory: Universal memory instance.
        messages: The conversation messages (user + assistant).
        agent_name: Name of the agent that handled this conversation.
    """
    try:
        # Get the last user message and last assistant response
        user_msg = ""
        assistant_msg = ""
        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                if role == "user" and not user_msg:
                    user_msg = content
                elif role == "assistant" and not assistant_msg:
                    assistant_msg = content
            if user_msg and assistant_msg:
                break

        if not user_msg:
            return

        combined = f"{user_msg}\n{assistant_msg}"

        # Check for explicit memory instructions
        explicit_patterns = [
            r"remember (?:that |this:? )?(.+)",
            r"save (?:this|that):? (.+)",
            r"note (?:that |this:? )?(.+)",
            r"keep in mind:? (.+)",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, user_msg, re.IGNORECASE)
            if match:
                fact = match.group(1).strip().rstrip(".")
                memory.append_shared(f"[User instruction] {fact}")
                logger.info(f"Auto-memory: Saved explicit instruction: {fact[:80]}")
                break

        # Check for contact mentions
        for pattern in CONTACT_PATTERNS:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    memory.append_contact(groups[0].strip(), groups[1].strip())
                    logger.debug(f"Auto-memory: Saved contact info: {groups[0]}")
                break

        # Check for decisions/preferences
        for pattern in DECISION_PATTERNS:
            match = re.search(pattern, user_msg, re.IGNORECASE)
            if match:
                decision = match.group(1).strip().rstrip(".")
                if len(decision) > 10:  # Skip trivially short matches
                    memory.log_decision(decision, context=f"From conversation with {agent_name}")
                    logger.debug(f"Auto-memory: Logged decision: {decision[:80]}")
                break

    except Exception as e:
        # Auto-memory should never crash the main flow
        logger.warning(f"Auto-memory extraction failed: {e}")
