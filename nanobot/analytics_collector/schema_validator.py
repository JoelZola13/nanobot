"""Validate incoming events against the static catalog. Lenient by design — we
would rather store an event with extra props than reject it — but we DO reject:

  - unknown event names (those need to be added to the catalog first)
  - missing envelope fields
  - missing required per-event props
  - replay-on-sensitive-route flags
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .catalog import event_catalog, event_index


SENSITIVE_ROUTE_PATTERNS = [re.compile(p) for p in [
    r"^/messages(/|$)",
    r"^/c/[^/?]+",
    r"^/jobs/resume(/|$)",
    r"^/jobs/[^/]+\?.*apply=1",
    r"^/case-management(/|$)",
    r"^/documents(/|$)",
    r"^/settings(/|$)",
    r"^/profile/edit(/|$)",
    r"^/grantwriter(/|$)",
]]


@dataclass
class ValidationIssue:
    type:    str       # unknown_event | missing_envelope | missing_required | invalid_type | replay_on_sensitive_route
    detail:  str


def validate_event(event: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    envelope = event_catalog()["envelope"]

    for required in envelope["required"]:
        if event.get(required) in (None, ""):
            issues.append(ValidationIssue("missing_envelope", f"missing envelope field: {required}"))

    name = event.get("event_name")
    if not name:
        issues.append(ValidationIssue("missing_envelope", "missing event_name"))
        return issues

    idx = event_index()
    definition = idx.get(name)
    if definition is None:
        issues.append(ValidationIssue("unknown_event", f"event_name not in catalog: {name}"))
        return issues

    props = event.get("properties") or {}
    for required_prop in definition.get("required", []):
        if required_prop not in props:
            issues.append(ValidationIssue("missing_required", f"{name}: missing required prop {required_prop}"))

    return issues


def is_replay_on_sensitive_route(event: dict[str, Any]) -> bool:
    """Detect the worst kind of misconfiguration — a session_recording capture
    landing while the user is on a route that should never be recorded.
    Replays themselves are stored in PostHog, not here, but PostHog emits
    `$snapshot` capture events to the same pipeline; if any of those reach us
    while route is sensitive, we flag it."""
    if event.get("event_name") not in {"$snapshot", "$session_recording"}:
        return False
    route = event.get("route") or ""
    return any(p.search(route) for p in SENSITIVE_ROUTE_PATTERNS)
