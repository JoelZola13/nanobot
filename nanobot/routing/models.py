"""Data models for binding-based routing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BindingRule:
    """A rule mapping (channel, account, peer) to a target agent.

    Wildcards ("*") match anything. More-specific fields take priority
    during resolution (peer > account > channel > default).
    """

    channel: str = "*"
    account: str = "*"
    peer: str = "*"
    agent: str = "ceo"
    session_namespace: str = ""


@dataclass(frozen=True)
class ResolvedBinding:
    """Result of resolving a binding for an inbound message."""

    agent_name: str
    session_namespace: str = ""
    matched_rule: BindingRule | None = None
