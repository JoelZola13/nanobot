"""Multi-agent system for Nanobot.

Provides agent definitions, registry, orchestration, and handoff support.
"""

from nanobot.agents.spec import AgentSpec
from nanobot.agents.registry import AgentRegistry

__all__ = ["AgentSpec", "AgentRegistry"]
