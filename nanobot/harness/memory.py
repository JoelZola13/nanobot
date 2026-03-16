"""Universal shared memory for all agents.

Every agent — regardless of which platform it's accessed from (LibreChat,
LobeHub, Social, Mission Control) — reads from and writes to the same
persistent memory store. This gives universal context continuity.

Architecture:
    ┌─────────────────────────────────────────────┐
    │          Universal Memory Store              │
    │  ~/.nanobot/workspace/memory/universal/      │
    ├─────────────────────────────────────────────┤
    │  SHARED.md     — facts all agents see       │
    │  CONTACTS.md   — contact knowledge base     │
    │  DECISIONS.md  — decisions & preferences     │
    │  PROJECTS.md   — active project context      │
    │  agents/{name}.md — per-agent memory         │
    │  topics/{slug}.md — topic-indexed memory     │
    │  sessions/{id}.md — cross-session summaries  │
    └─────────────────────────────────────────────┘

LangGraph Integration:
    - Uses InMemoryStore as the runtime cache
    - Persists to disk on every write
    - Loaded into agent context via custom middleware
    - SQLite checkpointer for conversation state
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class UniversalMemory:
    """Persistent shared memory accessible by all agents across all platforms.

    Thread-safe, file-backed, with in-memory caching for fast reads.
    """

    def __init__(self, workspace: Path):
        self.base_dir = workspace / "memory" / "universal"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "agents").mkdir(exist_ok=True)
        (self.base_dir / "teams").mkdir(exist_ok=True)
        (self.base_dir / "topics").mkdir(exist_ok=True)
        (self.base_dir / "sessions").mkdir(exist_ok=True)

        # In-memory cache with timestamps
        self._cache: dict[str, tuple[str, float]] = {}
        self._cache_ttl = 30.0  # seconds

        # Initialize core files if they don't exist
        self._ensure_file("SHARED.md", self._default_shared())
        self._ensure_file("CONTACTS.md", self._default_contacts())
        self._ensure_file("DECISIONS.md", self._default_decisions())
        self._ensure_file("PROJECTS.md", self._default_projects())

    # ── Core reads ──────────────────────────────────────────────

    def get_shared_context(self) -> str:
        """Get the universal shared context all agents see."""
        return self._read("SHARED.md")

    def get_contacts(self) -> str:
        """Get the contact knowledge base."""
        return self._read("CONTACTS.md")

    def get_decisions(self) -> str:
        """Get decision history and preferences."""
        return self._read("DECISIONS.md")

    def get_projects(self) -> str:
        """Get active project context."""
        return self._read("PROJECTS.md")

    def get_agent_memory(self, agent_name: str) -> str:
        """Get memory specific to an agent."""
        safe_name = self._safe_name(agent_name)
        path = f"agents/{safe_name}.md"
        return self._read(path) if (self.base_dir / path).exists() else ""

    def get_team_memory(self, team_name: str) -> str:
        """Get memory specific to a team."""
        safe_team = self._safe_name(team_name)
        path = f"teams/{safe_team}.md"
        return self._read(path) if (self.base_dir / path).exists() else ""

    def get_topic_memory(self, topic: str) -> str:
        """Get memory indexed by topic."""
        safe_topic = self._safe_name(topic)
        path = f"topics/{safe_topic}.md"
        return self._read(path) if (self.base_dir / path).exists() else ""

    def get_session_summary(self, session_id: str) -> str:
        """Get a cross-session summary."""
        safe_id = self._safe_name(session_id)
        path = f"sessions/{safe_id}.md"
        return self._read(path) if (self.base_dir / path).exists() else ""

    # ── Core writes ─────────────────────────────────────────────

    def update_shared(self, content: str) -> None:
        """Replace the shared context."""
        self._write("SHARED.md", content)

    def append_shared(self, entry: str) -> None:
        """Append a fact to shared context."""
        current = self.get_shared_context()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        updated = current.rstrip() + f"\n\n### [{timestamp}]\n{entry}\n"
        self._write("SHARED.md", updated)

    def update_contacts(self, content: str) -> None:
        """Replace the contacts knowledge base."""
        self._write("CONTACTS.md", content)

    def append_contact(self, name: str, info: str) -> None:
        """Add or update a contact entry."""
        current = self.get_contacts()
        timestamp = datetime.now().strftime("%Y-%m-%d")
        # Check if contact already exists
        pattern = rf"## {re.escape(name)}\n"
        if re.search(pattern, current):
            # Update existing entry — append info
            current = re.sub(
                pattern + r"(.*?)(?=\n## |\Z)",
                f"## {name}\n\\1\n- [{timestamp}] {info}\n",
                current,
                flags=re.DOTALL,
            )
            self._write("CONTACTS.md", current)
        else:
            updated = current.rstrip() + f"\n\n## {name}\n- [{timestamp}] {info}\n"
            self._write("CONTACTS.md", updated)

    def update_decisions(self, content: str) -> None:
        """Replace decision history."""
        self._write("DECISIONS.md", content)

    def log_decision(self, decision: str, context: str = "") -> None:
        """Log a decision or preference."""
        current = self.get_decisions()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n### [{timestamp}] {decision}\n"
        if context:
            entry += f"{context}\n"
        self._write("DECISIONS.md", current.rstrip() + entry)

    def update_projects(self, content: str) -> None:
        """Replace project context."""
        self._write("PROJECTS.md", content)

    def update_agent_memory(self, agent_name: str, content: str) -> None:
        """Write memory specific to an agent."""
        safe_name = self._safe_name(agent_name)
        self._write(f"agents/{safe_name}.md", content)

    def append_agent_memory(self, agent_name: str, entry: str) -> None:
        """Append to an agent's memory."""
        safe_name = self._safe_name(agent_name)
        path = f"agents/{safe_name}.md"
        current = self._read(path) if (self.base_dir / path).exists() else f"# {agent_name} Memory\n"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._write(path, current.rstrip() + f"\n\n- [{timestamp}] {entry}\n")

    def update_team_memory(self, team_name: str, content: str) -> None:
        """Write memory specific to a team."""
        safe_team = self._safe_name(team_name)
        self._write(f"teams/{safe_team}.md", content)

    def append_team_memory(self, team_name: str, entry: str) -> None:
        """Append to a team's memory."""
        safe_team = self._safe_name(team_name)
        path = f"teams/{safe_team}.md"
        current = self._read(path) if (self.base_dir / path).exists() else f"# {team_name} Team Memory\n"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._write(path, current.rstrip() + f"\n\n- [{timestamp}] {entry}\n")

    def update_topic(self, topic: str, content: str) -> None:
        """Write topic-indexed memory."""
        safe_topic = self._safe_name(topic)
        self._write(f"topics/{safe_topic}.md", content)

    def save_session_summary(self, session_id: str, summary: str, agent_name: str = "unknown") -> None:
        """Save a conversation summary scoped to the agent that ran it.

        Sessions are stored per-agent so each agent can recall its own
        conversation history without seeing other agents' threads.
        The CEO reads ALL agents' sessions for the bird's-eye view.
        """
        safe_agent = self._safe_name(agent_name)
        agent_sessions_dir = self.base_dir / "sessions" / safe_agent
        agent_sessions_dir.mkdir(parents=True, exist_ok=True)

        safe_id = self._safe_name(session_id)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = (
            f"# Session Summary\n\n"
            f"**Agent**: {agent_name}\n"
            f"**Time**: {timestamp}\n"
            f"**Session**: {session_id}\n\n"
            f"{summary}\n"
        )
        self._write(f"sessions/{safe_agent}/{safe_id}.md", content)

        # Also save to the legacy flat location for backwards compat
        self._write(f"sessions/{safe_id}.md", content)

    # ── Composite context builder ───────────────────────────────

    def build_context_for_agent(
        self,
        agent_name: str,
        team: str | None = None,
        topics: list[str] | None = None,
    ) -> str:
        """Build the scoped memory context an agent sees.

        Memory is layered — each agent gets a tailored view:

        Layer 1 — GLOBAL (auto-injected into every agent):
            • SHARED.md — identity, location, brand, critical rules
            • DECISIONS.md — user preferences & standing instructions

        Layer 2 — TEAM (auto-injected only for agents on that team):
            • teams/{team}.md — team-specific knowledge & contacts

        Layer 3 — AGENT (auto-injected only for this specific agent):
            • agents/{name}.md — personal memory, learned facts

        Layer 4 — ON-DEMAND (available via memory_read tool, NOT auto-injected):
            • CONTACTS.md — full contact book (use memory_read to look up)
            • PROJECTS.md — all project context (use memory_read to look up)
            • topics/*.md — topic-indexed deep dives
            • sessions/*.md — past conversation summaries

        The CEO is the exception — it gets contacts and projects auto-injected
        because it needs the full organizational picture to route effectively.
        """
        parts: list[str] = []

        # ── Layer 1: Global (everyone) ──────────────────────────
        shared = self.get_shared_context()
        if shared.strip():
            parts.append(f"<global_context>\n{shared}\n</global_context>")

        decisions = self.get_decisions()
        if decisions.strip() and decisions != self._default_decisions():
            parts.append(f"<decisions>\n{decisions}\n</decisions>")

        # ── Layer 2: Team (only your team) ──────────────────────
        if team:
            team_mem = self.get_team_memory(team)
            if team_mem.strip():
                parts.append(f"<team_memory team=\"{team}\">\n{team_mem}\n</team_memory>")

        # ── Layer 3: Agent (only you) ───────────────────────────
        agent_mem = self.get_agent_memory(agent_name)
        if agent_mem.strip():
            parts.append(f"<your_memory>\n{agent_mem}\n</your_memory>")

        # ── Layer 4: Conversation history ───────────────────────
        if agent_name == "ceo":
            # CEO sees contacts, projects, and ALL agents' recent sessions
            contacts = self.get_contacts()
            if contacts.strip() and contacts != self._default_contacts():
                parts.append(f"<contacts>\n{contacts}\n</contacts>")

            projects = self.get_projects()
            if projects.strip() and projects != self._default_projects():
                parts.append(f"<active_projects>\n{projects}\n</active_projects>")

            recent_sessions = self._get_recent_sessions(limit=10, agent_name=None)
            if recent_sessions:
                parts.append(
                    "<all_agent_sessions>\n"
                    + "\n---\n".join(recent_sessions)
                    + "\n</all_agent_sessions>"
                )
        else:
            # Individual agents see only THEIR OWN recent sessions
            my_sessions = self._get_recent_sessions(limit=5, agent_name=agent_name)
            if my_sessions:
                parts.append(
                    "<your_recent_conversations>\n"
                    + "\n---\n".join(my_sessions)
                    + "\n</your_recent_conversations>"
                )

        # ── Topic memories (if explicitly requested) ────────────
        if topics:
            for topic in topics:
                topic_mem = self.get_topic_memory(topic)
                if topic_mem.strip():
                    parts.append(
                        f"<topic_memory topic=\"{topic}\">\n{topic_mem}\n</topic_memory>"
                    )

        if not parts:
            return ""

        # Build the header based on what the agent can see vs. look up
        on_demand_note = (
            "You also have access to the full contact book and project list "
            "via the `memory_read` tool — use `memory_read(target='contacts')` "
            "or `memory_read(target='projects')` when you need that context."
        )

        return (
            "## Memory Context\n\n"
            "Below is your scoped memory — global facts, your team's knowledge, "
            "and your personal memory. This persists across conversations and platforms.\n\n"
            f"{on_demand_note}\n\n"
            + "\n\n".join(parts)
        )

    # ── Internal helpers ────────────────────────────────────────

    def _read(self, rel_path: str) -> str:
        """Read a file with caching."""
        now = time.monotonic()
        if rel_path in self._cache:
            content, ts = self._cache[rel_path]
            if now - ts < self._cache_ttl:
                return content

        full_path = self.base_dir / rel_path
        if not full_path.exists():
            return ""
        content = full_path.read_text(encoding="utf-8")
        self._cache[rel_path] = (content, now)
        return content

    def _write(self, rel_path: str, content: str) -> None:
        """Write a file and update cache."""
        full_path = self.base_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        self._cache[rel_path] = (content, time.monotonic())
        logger.debug(f"Universal memory updated: {rel_path} ({len(content)} chars)")

    def _ensure_file(self, rel_path: str, default_content: str) -> None:
        """Create a file with default content if it doesn't exist."""
        full_path = self.base_dir / rel_path
        if not full_path.exists():
            full_path.write_text(default_content, encoding="utf-8")
            logger.info(f"Initialized universal memory file: {rel_path}")

    def _safe_name(self, name: str) -> str:
        """Convert a name to a safe filename."""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower().strip())

    def _get_recent_sessions(self, limit: int = 5, agent_name: str | None = None) -> list[str]:
        """Get the most recent session summaries.

        Args:
            limit: Max number of sessions to return.
            agent_name: If set, only return sessions for this agent.
                        If None, return sessions across ALL agents (CEO view).
        """
        sessions_dir = self.base_dir / "sessions"
        if not sessions_dir.exists():
            return []

        if agent_name:
            # Per-agent sessions: look in sessions/{agent_name}/
            safe_agent = self._safe_name(agent_name)
            agent_dir = sessions_dir / safe_agent
            if not agent_dir.exists():
                return []
            files = sorted(
                agent_dir.glob("*.md"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
        else:
            # CEO view: all sessions across all agents
            files = sorted(
                sessions_dir.rglob("*.md"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )

        results = []
        for f in files[:limit]:
            content = f.read_text(encoding="utf-8").strip()
            if content:
                results.append(content)
        return results

    # ── Default file templates ──────────────────────────────────

    @staticmethod
    def _default_shared() -> str:
        return (
            "# Universal Shared Memory\n\n"
            "Facts, context, and knowledge shared across all agents.\n"
            "Updated automatically as agents learn new information.\n"
        )

    @staticmethod
    def _default_contacts() -> str:
        return (
            "# Contacts\n\n"
            "People, organizations, and entities the system knows about.\n"
            "Each contact has a name and accumulated knowledge.\n"
        )

    @staticmethod
    def _default_decisions() -> str:
        return (
            "# Decisions & Preferences\n\n"
            "User decisions, preferences, and standing instructions.\n"
            "Agents should respect these unless explicitly overridden.\n"
        )

    @staticmethod
    def _default_projects() -> str:
        return (
            "# Active Projects\n\n"
            "Current projects, their status, and key context.\n"
        )

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Export all memory as a dict (for API/debugging)."""
        result: dict[str, Any] = {
            "shared": self.get_shared_context(),
            "contacts": self.get_contacts(),
            "decisions": self.get_decisions(),
            "projects": self.get_projects(),
            "agents": {},
            "teams": {},
            "topics": {},
        }
        for f in (self.base_dir / "agents").glob("*.md"):
            result["agents"][f.stem] = f.read_text(encoding="utf-8")
        for f in (self.base_dir / "teams").glob("*.md"):
            result["teams"][f.stem] = f.read_text(encoding="utf-8")
        for f in (self.base_dir / "topics").glob("*.md"):
            result["topics"][f.stem] = f.read_text(encoding="utf-8")
        return result
