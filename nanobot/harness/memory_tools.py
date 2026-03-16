"""LangChain tools for universal memory read/write.

These tools are injected into every agent so they can read from
and write to the shared universal memory store.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from nanobot.harness.memory import UniversalMemory


class MemoryWriteArgs(BaseModel):
    """Arguments for writing to universal memory."""
    target: str = Field(
        description=(
            "Where to write: 'shared' (global facts everyone sees), "
            "'contacts' (people/orgs — accessible via memory_read), "
            "'decisions' (preferences/instructions — auto-injected), "
            "'projects' (project status — accessible via memory_read), "
            "'team' (your team's shared knowledge), "
            "'agent' (your own personal memory), or 'topic' (topic-indexed)."
        )
    )
    content: str = Field(
        description="The content to write or append."
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "Required for 'contacts' (contact name), 'team' (team name), "
            "'agent' (agent name), or 'topic' (topic slug)."
        )
    )
    mode: str = Field(
        default="append",
        description="'append' to add to existing, 'replace' to overwrite entirely."
    )


class MemoryReadArgs(BaseModel):
    """Arguments for reading from universal memory."""
    target: str = Field(
        description=(
            "What to read: 'shared', 'contacts', 'decisions', 'projects', "
            "'team' (team memory), 'agent' (agent memory), "
            "'topic' (specific topic), or 'all' (your full scoped context)."
        )
    )
    name: Optional[str] = Field(
        default=None,
        description="Required for 'team' (team name), 'agent' (agent name), or 'topic' (topic slug)."
    )


class MemorySearchArgs(BaseModel):
    """Arguments for searching universal memory."""
    query: str = Field(description="Search query to find in memory.")
    targets: Optional[list[str]] = Field(
        default=None,
        description="Which memory sections to search. Default: search all."
    )


def create_memory_tools(
    memory: UniversalMemory,
    agent_name: str = "unknown",
    team: str | None = None,
) -> list[StructuredTool]:
    """Create LangChain tools for universal memory access.

    Args:
        memory: The UniversalMemory instance.
        agent_name: Name of the agent these tools are for.
        team: Team this agent belongs to (for scoped team memory).

    Returns:
        List of LangChain tools: [memory_write, memory_read, memory_search]
    """

    def memory_write(
        target: str,
        content: str,
        name: str | None = None,
        mode: str = "append",
    ) -> str:
        """Write to shared memory. Use this to persist facts, contacts, decisions, and project status."""
        try:
            if target == "shared":
                if mode == "replace":
                    memory.update_shared(content)
                else:
                    memory.append_shared(content)
                return f"Updated shared memory ({mode})"

            elif target == "contacts":
                if not name:
                    return "Error: 'name' is required for contacts target"
                if mode == "replace":
                    memory.update_contacts(content)
                else:
                    memory.append_contact(name, content)
                return f"Updated contact: {name}"

            elif target == "decisions":
                if mode == "replace":
                    memory.update_decisions(content)
                else:
                    memory.log_decision(content)
                return f"Logged decision ({mode})"

            elif target == "projects":
                if mode == "replace":
                    memory.update_projects(content)
                else:
                    current = memory.get_projects()
                    memory.update_projects(current.rstrip() + f"\n\n{content}\n")
                return f"Updated projects ({mode})"

            elif target == "agent":
                agent = name or agent_name
                if mode == "replace":
                    memory.update_agent_memory(agent, content)
                else:
                    memory.append_agent_memory(agent, content)
                return f"Updated agent memory for: {agent}"

            elif target == "team":
                team_name = name or team
                if not team_name:
                    return "Error: 'name' is required for team target (or agent must belong to a team)"
                if mode == "replace":
                    memory.update_team_memory(team_name, content)
                else:
                    memory.append_team_memory(team_name, content)
                return f"Updated team memory for: {team_name}"

            elif target == "topic":
                if not name:
                    return "Error: 'name' is required for topic target"
                if mode == "replace":
                    memory.update_topic(name, content)
                else:
                    current = memory.get_topic_memory(name)
                    if not current:
                        current = f"# {name}\n"
                    memory.update_topic(name, current.rstrip() + f"\n\n{content}\n")
                return f"Updated topic memory: {name}"

            else:
                return f"Error: Unknown target '{target}'. Use: shared, contacts, decisions, projects, team, agent, topic"

        except Exception as e:
            return f"Error writing memory: {e}"

    def memory_read(
        target: str,
        name: str | None = None,
    ) -> str:
        """Read from shared memory. Use 'contacts' or 'projects' to look up info not in your auto-injected context."""
        try:
            if target == "shared":
                return memory.get_shared_context() or "(empty)"
            elif target == "contacts":
                return memory.get_contacts() or "(empty)"
            elif target == "decisions":
                return memory.get_decisions() or "(empty)"
            elif target == "projects":
                return memory.get_projects() or "(empty)"
            elif target == "team":
                team_name = name or team
                if not team_name:
                    return "Error: 'name' required for team target"
                return memory.get_team_memory(team_name) or f"(no memory for team '{team_name}')"
            elif target == "agent":
                agent = name or agent_name
                return memory.get_agent_memory(agent) or f"(no memory for agent '{agent}')"
            elif target == "topic":
                if not name:
                    return "Error: 'name' is required for topic target"
                return memory.get_topic_memory(name) or f"(no memory for topic '{name}')"
            elif target == "all":
                return memory.build_context_for_agent(agent_name, team=team) or "(no memory)"
            else:
                return f"Error: Unknown target '{target}'"
        except Exception as e:
            return f"Error reading memory: {e}"

    def memory_search(
        query: str,
        targets: list[str] | None = None,
    ) -> str:
        """Search across universal memory for relevant information."""
        query_lower = query.lower()
        results: list[str] = []
        search_targets = targets or ["shared", "contacts", "decisions", "projects"]

        for target in search_targets:
            content = ""
            if target == "shared":
                content = memory.get_shared_context()
            elif target == "contacts":
                content = memory.get_contacts()
            elif target == "decisions":
                content = memory.get_decisions()
            elif target == "projects":
                content = memory.get_projects()

            if not content:
                continue

            # Simple line-level search
            matching_lines = []
            for line in content.split("\n"):
                if query_lower in line.lower():
                    matching_lines.append(line.strip())

            if matching_lines:
                results.append(f"**{target}**:\n" + "\n".join(matching_lines[:10]))

        # Search team memories
        teams_dir = memory.base_dir / "teams"
        if teams_dir.exists():
            for f in teams_dir.glob("*.md"):
                content = f.read_text(encoding="utf-8")
                matching = [l.strip() for l in content.split("\n") if query_lower in l.lower()]
                if matching:
                    results.append(f"**team/{f.stem}**:\n" + "\n".join(matching[:5]))

        # Search agent memories
        agents_dir = memory.base_dir / "agents"
        if agents_dir.exists():
            for f in agents_dir.glob("*.md"):
                content = f.read_text(encoding="utf-8")
                matching = [l.strip() for l in content.split("\n") if query_lower in l.lower()]
                if matching:
                    results.append(f"**agent/{f.stem}**:\n" + "\n".join(matching[:5]))

        # Search topics
        topics_dir = memory.base_dir / "topics"
        if topics_dir.exists():
            for f in topics_dir.glob("*.md"):
                content = f.read_text(encoding="utf-8")
                matching = [l.strip() for l in content.split("\n") if query_lower in l.lower()]
                if matching:
                    results.append(f"**topic/{f.stem}**:\n" + "\n".join(matching[:5]))

        if not results:
            return f"No results found for: {query}"
        return "\n\n".join(results)

    return [
        StructuredTool(
            name="memory_write",
            description=(
                "Write to universal shared memory. Use this to persist ANY important fact, "
                "contact info, decision, preference, or project status. This memory is shared "
                "across ALL agents and ALL platforms (LibreChat, LobeHub, Social, Mission Control). "
                "Anything you write here will be available in future conversations."
            ),
            func=memory_write,
            args_schema=MemoryWriteArgs,
        ),
        StructuredTool(
            name="memory_read",
            description=(
                "Read from universal shared memory. Check this for context about contacts, "
                "past decisions, active projects, and accumulated knowledge from all agents."
            ),
            func=memory_read,
            args_schema=MemoryReadArgs,
        ),
        StructuredTool(
            name="memory_search",
            description=(
                "Search across all universal memory (shared facts, contacts, decisions, "
                "projects, agent memories, topic memories) for relevant information."
            ),
            func=memory_search,
            args_schema=MemorySearchArgs,
        ),
    ]
