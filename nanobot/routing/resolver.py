"""BindingResolver — indexes rules into hash maps for O(1) lookup."""

from __future__ import annotations

import logging
from nanobot.routing.models import BindingRule, ResolvedBinding

log = logging.getLogger(__name__)


class BindingResolver:
    """Resolve inbound messages to target agents using binding rules.

    Priority order (highest first):
      1. Exact peer match  (channel + peer)
      2. Exact account match  (channel + account)
      3. Channel-only match
      4. Default agent
    """

    def __init__(self, rules: list[BindingRule], default_agent: str = "ceo") -> None:
        self.default_agent = default_agent

        # Indexed maps for O(1) lookup
        self._peer_map: dict[tuple[str, str], BindingRule] = {}
        self._account_map: dict[tuple[str, str], BindingRule] = {}
        self._channel_map: dict[str, BindingRule] = {}

        for rule in rules:
            has_peer = rule.peer != "*"
            has_account = rule.account != "*"
            has_channel = rule.channel != "*"

            if has_peer and has_channel:
                self._peer_map[(rule.channel, rule.peer)] = rule
            elif has_account and has_channel:
                self._account_map[(rule.channel, rule.account)] = rule
            elif has_channel:
                self._channel_map[rule.channel] = rule
            elif has_peer:
                # peer-only (any channel) — store with wildcard channel
                self._peer_map[("*", rule.peer)] = rule
            elif has_account:
                self._account_map[("*", rule.account)] = rule
            else:
                # Catch-all rule — update default
                self.default_agent = rule.agent

        log.debug(
            "BindingResolver: %d peer, %d account, %d channel rules; default=%s",
            len(self._peer_map),
            len(self._account_map),
            len(self._channel_map),
            self.default_agent,
        )

    def resolve(
        self,
        channel: str,
        peer_id: str | None = None,
        account_id: str | None = None,
        chat_id: str | None = None,
    ) -> ResolvedBinding:
        """Resolve the target agent for an inbound message."""
        # 1. Peer match (channel-specific, then wildcard)
        if peer_id:
            for key in [(channel, peer_id), ("*", peer_id)]:
                rule = self._peer_map.get(key)
                if rule:
                    return self._make_result(rule)

        # Also try chat_id as peer (for group chats)
        if chat_id and chat_id != peer_id:
            for key in [(channel, chat_id), ("*", chat_id)]:
                rule = self._peer_map.get(key)
                if rule:
                    return self._make_result(rule)

        # 2. Account match
        if account_id:
            for key in [(channel, account_id), ("*", account_id)]:
                rule = self._account_map.get(key)
                if rule:
                    return self._make_result(rule)

        # 3. Channel match
        rule = self._channel_map.get(channel)
        if rule:
            return self._make_result(rule)

        # 4. Default
        return ResolvedBinding(agent_name=self.default_agent)

    def _make_result(self, rule: BindingRule) -> ResolvedBinding:
        return ResolvedBinding(
            agent_name=rule.agent,
            session_namespace=rule.session_namespace,
            matched_rule=rule,
        )
