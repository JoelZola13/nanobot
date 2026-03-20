"""Token-based authentication for gateway connections.

Supports both WS handshake auth and HTTP Bearer tokens.
Tokens are configured via `gateway.auth_tokens` in config or
the NANOBOT_GATEWAY__AUTH_TOKENS environment variable.
"""

from __future__ import annotations

import secrets
from typing import Sequence

from loguru import logger


class GatewayAuth:
    """Validates bearer tokens for gateway access.

    If no tokens are configured the gateway runs in open mode
    (all connections accepted). This matches Nanobot's existing
    zero-config-required philosophy.
    """

    def __init__(self, tokens: Sequence[str] | None = None):
        self._tokens: set[str] = set(tokens) if tokens else set()

    @property
    def enabled(self) -> bool:
        return bool(self._tokens)

    def validate(self, token: str) -> bool:
        """Check if *token* is in the allowed set.

        Returns True when auth is disabled (no tokens configured).
        """
        if not self._tokens:
            return True
        return _compare_token(token, self._tokens)

    def validate_bearer(self, header: str) -> bool:
        """Validate an ``Authorization: Bearer <token>`` header value."""
        if not self._tokens:
            return True
        if not header:
            return False
        parts = header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        return self.validate(parts[1])

    @staticmethod
    def generate_token() -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_urlsafe(32)


def _compare_token(candidate: str, allowed: set[str]) -> bool:
    """Constant-time-ish membership check."""
    for t in allowed:
        if secrets.compare_digest(candidate, t):
            return True
    return False
