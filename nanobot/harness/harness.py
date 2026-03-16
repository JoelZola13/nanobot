"""Deep Agent Harness — the ultimate multi-agent orchestration engine.

Uses every feature of the deepagents library:

BACKENDS:
  - CompositeBackend routing:
    • /memory/  → FilesystemBackend (persistent AGENTS.md per agent)
    • /skills/  → FilesystemBackend (progressive skill disclosure)
    • /workspace/ → FilesystemBackend (shared persistent workspace)
    • default   → LocalShellBackend (file ops + shell execution)

MIDDLEWARE (full stack on every agent):
  1. TodoListMiddleware — task decomposition and progress tracking
  2. MemoryMiddleware — load AGENTS.md files, self-improving memory
  3. SkillsMiddleware — progressive skill disclosure from 44 skills
  4. FilesystemMiddleware — ls, read, write, edit, grep, glob, execute
  5. SubAgentMiddleware — 37 specialized subagents via `task` tool
  6. SummarizationMiddleware — auto-compact long conversations
  7. PatchToolCallsMiddleware — fix dangling tool calls
  8. AnthropicPromptCachingMiddleware — prompt caching for Claude

FEATURES:
  - Universal shared memory with scoped context (global/team/agent layers)
  - Per-agent session summaries for conversation continuity
  - Self-improving memory (agents can edit their own AGENTS.md)
  - LangGraph checkpointing for conversation persistence
  - All 110+ MCP/nanobot tools available to agents
  - OpenAI-compatible API for all frontends

Usage:
    harness = DeepAgentHarness(workspace, config)
    await harness.initialize(tool_registry, mcp_tools)
    result = await harness.run("ceo", messages, session_id="abc123")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.harness.memory import UniversalMemory
from nanobot.harness.memory_tools import create_memory_tools
from nanobot.harness.tool_bridge import bridge_registry, bridge_tools


# ── Paths ────────────────────────────────────────────────────────
WORKSPACE = Path.home() / ".nanobot" / "workspace"
SKILLS_DIR = WORKSPACE / "skills"
MEMORY_DIR = WORKSPACE / "memory" / "universal"


class DeepAgentHarness:
    """Multi-agent harness powered by deepagents + nanobot.

    Uses EVERY deepagents feature:
    - CompositeBackend (memory + skills + workspace + shell execution)
    - MemoryMiddleware (AGENTS.md self-improving memory)
    - SkillsMiddleware (44 skills, progressive disclosure)
    - FilesystemMiddleware (full filesystem access)
    - SummarizationMiddleware (auto-compact long conversations)
    - PatchToolCallsMiddleware (fix dangling tool calls)
    - SubAgentMiddleware (37 specialized agents with full middleware stacks)
    - LangGraph checkpointing (persistent conversations)
    """

    def __init__(self, workspace: Path, config: dict[str, Any] | None = None):
        self.workspace = workspace
        self.config = config or {}
        self.memory = UniversalMemory(workspace)

        self._agents: dict[str, dict[str, Any]] = {}
        self._agent_graph = None
        self._subagent_specs: list[dict[str, Any]] = []
        self._lc_tools: list[Any] = []
        self._initialized = False
        self._checkpointer = None
        self._sessions: dict[str, dict[str, Any]] = {}

    async def initialize(
        self,
        tool_registry: Any = None,
        mcp_tools: dict[str, Any] | None = None,
        teams_dir: Path | None = None,
    ) -> None:
        """Initialize the harness with full deepagents feature set."""
        from deepagents import create_deep_agent
        from deepagents.backends import (
            CompositeBackend,
            FilesystemBackend,
            LocalShellBackend,
        )
        from deepagents.middleware import (
            MemoryMiddleware,
            SkillsMiddleware,
            SummarizationMiddleware,
        )
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Initializing Deep Agent Harness (full feature mode)...")

        # ── 1. Checkpointer for conversation persistence ──────────
        self._checkpointer = MemorySaver()

        # ── 2. Bridge nanobot tools → LangChain ───────────────────
        self._lc_tools = []
        if tool_registry:
            self._lc_tools = bridge_registry(tool_registry)
        if mcp_tools:
            self._lc_tools.extend(bridge_tools(mcp_tools))
        logger.info(f"Bridged {len(self._lc_tools)} tools to LangChain")

        # ── 3. Load agent definitions from YAML ───────────────────
        if teams_dir is None:
            teams_dir = Path(__file__).parent.parent / "agents" / "teams"
        self._load_agent_specs(teams_dir)
        logger.info(f"Loaded {len(self._agents)} agent specs")

        # ── 4. Ensure per-agent AGENTS.md files exist ─────────────
        self._ensure_agent_memory_files()

        # ── 5. Build backend architecture ─────────────────────────
        #
        # CompositeBackend routes file operations:
        #   /memory/  → persistent agent memory (AGENTS.md files)
        #   /skills/  → skill definitions (SKILL.md files)
        #   /workspace/ → shared persistent workspace
        #   default   → local shell (cwd filesystem + execute)
        #
        memory_backend = FilesystemBackend(
            root_dir=str(MEMORY_DIR),
            virtual_mode=True,
        )
        skills_backend = FilesystemBackend(
            root_dir=str(SKILLS_DIR),
            virtual_mode=True,
        )
        workspace_backend = FilesystemBackend(
            root_dir=str(WORKSPACE),
            virtual_mode=True,
        )
        default_backend = LocalShellBackend(
            root_dir=str(Path.home()),
            timeout=120,
            max_output_bytes=200_000,
            inherit_env=True,
        )

        self._backend = CompositeBackend(
            default=default_backend,
            routes={
                "/memory/": memory_backend,
                "/skills/": skills_backend,
                "/workspace/": workspace_backend,
            },
        )

        logger.info(
            "CompositeBackend ready: "
            "/memory/ → persistent, /skills/ → skills, "
            "/workspace/ → workspace, default → local shell"
        )

        # ── 6. Collect skill sources ──────────────────────────────
        # All skills under ~/.nanobot/workspace/skills/ organized by agent
        skill_sources = []
        if SKILLS_DIR.exists():
            skill_sources.append("/skills/")
        logger.info(f"Skill sources: {skill_sources}")

        # ── 7. Collect memory sources ─────────────────────────────
        # Per-agent AGENTS.md files under /memory/agents/
        memory_sources = ["/memory/AGENTS.md"]
        agents_memory_dir = MEMORY_DIR / "agents"
        if agents_memory_dir.exists():
            for f in agents_memory_dir.glob("*.md"):
                memory_sources.append(f"/memory/agents/{f.name}")
        logger.info(f"Memory sources: {len(memory_sources)} files")

        # ── 8. Build subagent specs ───────────────────────────────
        self._subagent_specs = self._build_subagent_specs()
        logger.info(f"Built {len(self._subagent_specs)} subagent specs")

        # ── 9. CEO memory tools (full access) ─────────────────────
        ceo_memory_tools = create_memory_tools(
            self.memory, agent_name="ceo", team="executive"
        )
        all_ceo_tools = self._lc_tools + ceo_memory_tools

        # ── 10. Build the orchestrator graph ──────────────────────
        #
        # create_deep_agent() automatically adds its own middleware:
        #   TodoListMiddleware, FilesystemMiddleware,
        #   SubAgentMiddleware, SummarizationMiddleware,
        #   AnthropicPromptCachingMiddleware, PatchToolCallsMiddleware
        #
        # We add on top: MemoryMiddleware, SkillsMiddleware
        #
        ceo_prompt = self._build_orchestrator_prompt()

        self._agent_graph = create_deep_agent(
            model=self._resolve_model(),
            tools=all_ceo_tools,
            system_prompt=ceo_prompt,
            subagents=self._subagent_specs,
            checkpointer=self._checkpointer,
            backend=self._backend,
            memory=memory_sources,
            skills=skill_sources if skill_sources else None,
            name="nanobot-orchestrator",
        )

        self._initialized = True
        logger.info(
            f"Deep Agent Harness initialized (FULL FEATURE): "
            f"{len(self._agents)} agents, "
            f"{len(self._lc_tools)} tools, "
            f"{len(self._subagent_specs)} subagents, "
            f"{len(memory_sources)} memory sources, "
            f"{len(skill_sources)} skill sources"
        )

    # ── Agent spec loading ───────────────────────────────────────

    def _load_agent_specs(self, teams_dir: Path) -> None:
        """Load agent specs from YAML team definitions."""
        import yaml

        if not teams_dir.exists():
            logger.warning(f"Teams directory not found: {teams_dir}")
            return

        for team_dir in sorted(teams_dir.iterdir()):
            if not team_dir.is_dir():
                continue

            agents_file = team_dir / "agents.yaml"
            if not agents_file.exists():
                continue

            try:
                with open(agents_file) as f:
                    data = yaml.safe_load(f)

                if not data or "agents" not in data:
                    continue

                team_name = team_dir.name

                for agent_data in data["agents"]:
                    name = agent_data["name"]

                    system_prompt = ""
                    prompt_file = agent_data.get("system_prompt", "")
                    if prompt_file:
                        prompt_path = team_dir / prompt_file
                        if prompt_path.exists():
                            system_prompt = prompt_path.read_text().strip()

                    self._agents[name] = {
                        "name": name,
                        "team": team_name,
                        "description": agent_data.get("description", ""),
                        "role": agent_data.get("role", "member"),
                        "model": agent_data.get("model", "default"),
                        "tools": agent_data.get("tools", []),
                        "handoffs": agent_data.get("handoffs", []),
                        "system_prompt": system_prompt,
                        "max_iterations": agent_data.get("max_iterations", 25),
                        "temperature": agent_data.get("temperature", 0.7),
                    }
            except Exception as e:
                logger.error(f"Failed to load team '{team_dir.name}': {e}")

    def _ensure_agent_memory_files(self) -> None:
        """Ensure every agent has an AGENTS.md file for self-improving memory.

        deepagents' MemoryMiddleware loads these files and the agent can
        edit_file() them to save learnings that persist across conversations.
        """
        agents_dir = MEMORY_DIR / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)

        # Also ensure a root AGENTS.md exists with global instructions
        root_agents_md = MEMORY_DIR / "AGENTS.md"
        if not root_agents_md.exists():
            root_agents_md.write_text(
                "# Street Voices AI — Global Agent Memory\n\n"
                "## Identity\n"
                "- Organization: Street Voices (Toronto-based)\n"
                "- Owner: Joel (joel@streetvoices.ca)\n"
                "- Timezone: Eastern Time (ET)\n"
                "- Communication style: Casual, direct, concise\n\n"
                "## Critical Rules\n"
                "- NEVER send emails or messages without Joel's explicit approval\n"
                "- NEVER auto-reply on any platform (email, WhatsApp, Slack)\n"
                "- Always save important facts to memory\n"
                "- Check memory before making assumptions\n\n"
                "## Learnings\n"
                "<!-- Agents: add learnings below as you discover them -->\n",
                encoding="utf-8",
            )

        for name, spec in self._agents.items():
            if name.endswith("_memory"):
                continue

            agent_md = agents_dir / f"{name}.md"
            if not agent_md.exists():
                team = spec.get("team", "unknown")
                role = spec.get("role", "member")
                desc = spec.get("description", "")
                agent_md.write_text(
                    f"# {name} — Agent Memory\n\n"
                    f"**Team**: {team} | **Role**: {role}\n"
                    f"**Description**: {desc}\n\n"
                    f"## Learnings\n"
                    f"<!-- Add learnings here as you discover them -->\n\n"
                    f"## Known Issues\n"
                    f"<!-- Track bugs, workarounds, and gotchas -->\n",
                    encoding="utf-8",
                )

    # ── Subagent building ────────────────────────────────────────

    def _build_subagent_specs(self) -> list[dict[str, Any]]:
        """Build deepagents SubAgent specs with full middleware stacks.

        Each subagent gets:
        - Its own tools (mapped from YAML)
        - Scoped memory tools (agent + team level)
        - Its own AGENTS.md memory (self-improving via edit_file)
        - Its own skill sources (agent-specific skills)
        - Model override if specified
        """
        specs: list[dict[str, Any]] = []

        for name, agent in self._agents.items():
            if name.endswith("_memory") or name == "ceo":
                continue

            # Build tool set
            agent_tools = self._get_agent_tools(agent)

            # Add scoped memory tools
            agent_team = agent.get("team")
            agent_memory_tools = create_memory_tools(
                self.memory, agent_name=name, team=agent_team
            )
            agent_tools.extend(agent_memory_tools)

            # Build enriched system prompt
            enriched_prompt = self._enrich_agent_prompt(agent)

            # Resolve model (subagents can have different models)
            model = self._resolve_agent_model(agent)

            # Build per-agent skill sources
            # Skill dir names use hyphens per Agent Skills spec
            skill_dir_name = f"agent-{name.replace('_', '-')}"
            agent_skill_dir = SKILLS_DIR / skill_dir_name
            skill_sources = []
            if SKILLS_DIR.exists():
                skill_sources.append("/skills/")
            if agent_skill_dir.exists():
                skill_sources.append(f"/skills/{skill_dir_name}/")

            # Build per-agent memory sources
            memory_sources = ["/memory/AGENTS.md"]
            agent_memory_file = MEMORY_DIR / "agents" / f"{name}.md"
            if agent_memory_file.exists():
                memory_sources.append(f"/memory/agents/{name}.md")

            spec: dict[str, Any] = {
                "name": name,
                "description": agent["description"],
                "system_prompt": enriched_prompt,
                "tools": agent_tools,
                "model": model,
                # deepagents will wire skills into the subagent's middleware
                "skills": skill_sources if skill_sources else None,
            }

            specs.append(spec)
            logger.debug(
                f"Subagent: {name} "
                f"({len(agent_tools)} tools, "
                f"model={model}, "
                f"{len(skill_sources)} skill sources)"
            )

        return specs

    def _get_agent_tools(self, agent_spec: dict[str, Any]) -> list[Any]:
        """Map nanobot tool names to bridged LangChain tools."""
        import fnmatch

        agent_tool_names = agent_spec.get("tools", [])
        matched_tools: list[Any] = []
        tool_name_set: set[str] = set()

        name_aliases = {
            "exec": ["exec", "shell"],
            "shell": ["shell", "exec"],
            "file_read": ["file_read", "read_file"],
            "file_write": ["file_write", "write_file"],
            "edit_file": ["edit_file", "file_edit"],
        }

        for pattern in agent_tool_names:
            if "*" in pattern:
                for lc_tool in self._lc_tools:
                    if fnmatch.fnmatch(lc_tool.name, pattern) and lc_tool.name not in tool_name_set:
                        matched_tools.append(lc_tool)
                        tool_name_set.add(lc_tool.name)
            else:
                names_to_try = name_aliases.get(pattern, [pattern])
                for try_name in names_to_try:
                    for lc_tool in self._lc_tools:
                        if lc_tool.name == try_name and lc_tool.name not in tool_name_set:
                            matched_tools.append(lc_tool)
                            tool_name_set.add(lc_tool.name)
                            break

        return matched_tools

    def _enrich_agent_prompt(self, agent_spec: dict[str, Any]) -> str:
        """Build system prompt with scoped memory context + self-improvement instructions."""
        name = agent_spec["name"]
        team = agent_spec["team"]
        role = agent_spec["role"]
        base_prompt = agent_spec.get("system_prompt", "")

        parts = [
            f"# Agent: {name}",
            f"**Team**: {team} | **Role**: {role}",
            "",
        ]

        if base_prompt:
            parts.append(base_prompt)
            parts.append("")

        # Memory system instructions
        parts.append(
            "## Memory System\n\n"
            "You have a layered memory system that persists across all conversations:\n\n"
            "**Auto-injected** (you see this automatically):\n"
            "- Global facts — identity, location, critical rules\n"
            "- Decisions — user preferences and standing instructions\n"
            f"- Team memory — knowledge shared within the {team} team\n"
            "- Your personal memory — facts you've learned\n"
            "- Your recent past conversations with the user\n\n"
            "**On-demand** (use `memory_read` to look up):\n"
            "- `memory_read(target='contacts')` — full contact book\n"
            "- `memory_read(target='projects')` — all project context\n"
            "- `memory_read(target='team', name='...')` — another team's memory\n\n"
            "**Writing memory** (use `memory_write`):\n"
            "- `target='shared'` — facts everyone should know\n"
            "- `target='contacts'` — people/orgs (name required)\n"
            "- `target='team'` — your team's knowledge\n"
            "- `target='agent'` — your personal notes\n"
            "- `target='decisions'` — user preferences\n"
            "- `target='projects'` — project updates\n\n"
            "## Self-Improving Memory\n\n"
            "Your AGENTS.md file is loaded into context at the start of every conversation. "
            "You can edit it with `edit_file` to save permanent learnings:\n\n"
            "**What to save**: API methods that don't exist, code patterns that work, "
            "known limitations, non-obvious error fixes, user preferences.\n\n"
            "**What NOT to save**: One-off errors, transient issues, speculative improvements.\n\n"
            "**When to update**: IMMEDIATELY after confirming a learning. Don't batch.\n\n"
            "**IMPORTANT**: Save new facts you learn. Check memory before assumptions."
        )

        return "\n".join(parts)

    def _build_orchestrator_prompt(self) -> str:
        """Build the CEO orchestrator system prompt."""
        ceo_spec = self._agents.get("ceo", {})
        base_prompt = ceo_spec.get("system_prompt", "")

        agent_list = []
        for spec in self._subagent_specs:
            agent_list.append(f"- **{spec['name']}**: {spec['description']}")
        agents_section = "\n".join(agent_list) if agent_list else "(no subagents)"

        return (
            f"# Street Voices AI — Central Orchestrator\n\n"
            f"You are the CEO agent of Joel's multi-agent system. You coordinate "
            f"a team of {len(self._subagent_specs)} specialized agents.\n\n"
            f"{base_prompt}\n\n"
            f"## Available Agents\n\n"
            f"Use the `task` tool to delegate work to specialized agents:\n\n"
            f"{agents_section}\n\n"
            f"## Delegation Strategy\n\n"
            f"- **Bias towards single agents**: One comprehensive task beats many narrow ones\n"
            f"- **Parallelize only for comparisons**: 'Compare X vs Y' → one agent per element\n"
            f"- **Max 3 concurrent**: Never more than 3 parallel subagent calls\n"
            f"- **Max 3 delegation rounds**: Stop and synthesize after 3 rounds\n"
            f"- **Simple queries**: Handle directly, don't delegate trivial tasks\n\n"
            f"## Context Handoff\n\n"
            f"When delegating via `task`, **include all relevant context**:\n"
            f"- What the user asked for and why\n"
            f"- Relevant contact info, project context, preferences\n"
            f"- Expected output format\n"
            f"- The subagent starts fresh — it only knows what you tell it "
            f"plus its own persistent memory.\n\n"
            f"## Memory System\n\n"
            f"You have the most complete memory view:\n"
            f"- Global context, decisions, contacts, projects (all auto-injected)\n"
            f"- All agents' recent session summaries\n"
            f"- Individual agents only see their own team/personal memory\n\n"
            f"Use `memory_write` to persist facts across all platforms.\n\n"
            f"## Self-Improving Memory\n\n"
            f"Your AGENTS.md file is loaded every conversation. Use `edit_file` to "
            f"save permanent learnings (API quirks, user preferences, known issues).\n"
            f"Update IMMEDIATELY after confirming a learning.\n\n"
            f"## Guidelines\n\n"
            f"- NEVER send emails/messages without Joel's explicit approval\n"
            f"- Be concise and direct — Joel prefers casual communication\n"
            f"- Use `write_todos` to plan complex tasks\n"
            f"- Save important facts to memory\n"
        )

    def _resolve_model(self) -> Any:
        """Resolve the CEO model — returns a CodexChatModel instance."""
        ceo_model = self._agents.get("ceo", {}).get("model", "default")
        return self._map_model(ceo_model)

    def _resolve_agent_model(self, agent_spec: dict[str, Any]) -> Any:
        """Resolve a specific agent's model."""
        model = agent_spec.get("model", "default")
        return self._map_model(model)

    @staticmethod
    def _map_model(model: str) -> Any:
        """Map nanobot model names to a CodexChatModel instance.

        Since Joel uses Codex OAuth (not standard OpenAI API keys),
        we return a CodexChatModel that wraps the codex provider.
        """
        from nanobot.harness.codex_model import CodexChatModel

        # Extract the actual model name
        if model == "default" or not model:
            model_name = "gpt-5.1-codex"
        elif "/" in model:
            _, model_name = model.split("/", 1)
        else:
            model_name = model

        return CodexChatModel(model_name=model_name)

    # ── Session summarization ────────────────────────────────────

    async def _summarize_session(
        self,
        messages: list[dict[str, Any]],
        final_response: str,
        agent_name: str,
    ) -> str:
        """Summarize a conversation using a fast LLM."""
        transcript_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system" or not content:
                continue
            if len(content) > 300:
                content = content[:300] + "..."
            transcript_parts.append(f"{role.upper()}: {content}")

        resp_preview = final_response[:500] + ("..." if len(final_response) > 500 else "")
        transcript_parts.append(f"ASSISTANT: {resp_preview}")
        transcript = "\n".join(transcript_parts)

        try:
            from langchain_core.messages import HumanMessage
            from nanobot.harness.codex_model import CodexChatModel

            summarizer = CodexChatModel(model_name="gpt-5.1-codex")
            result = await summarizer.ainvoke([HumanMessage(content=(
                "Summarize this conversation in 2-4 bullet points. Focus on:\n"
                "- What the user asked for\n"
                "- What was decided or accomplished\n"
                "- Any follow-ups or open items\n\n"
                f"Conversation with {agent_name}:\n{transcript}\n\n"
                "Summary (bullet points only, no preamble):"
            ))])
            return result.content if hasattr(result, "content") else str(result)

        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            user_msgs = [m.get("content", "")[:200] for m in messages if m.get("role") == "user"]
            return f"User asked: {'; '.join(user_msgs[:3])}\n\nAgent responded: {resp_preview}"

    def _save_session_async(
        self,
        messages: list[dict[str, Any]],
        final_response: str,
        agent_name: str,
        session_id: str,
    ) -> None:
        """Fire-and-forget session summarization."""
        async def _do_save():
            try:
                summary = await self._summarize_session(messages, final_response, agent_name)
                self.memory.save_session_summary(
                    session_id=session_id, summary=summary, agent_name=agent_name
                )
                logger.info(f"Session saved: agent={agent_name}, session={session_id}")
            except Exception as e:
                logger.warning(f"Session save failed: {e}")

        asyncio.create_task(_do_save())

    # ── Public API ───────────────────────────────────────────────

    async def run(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        model_override: str | None = None,
        on_progress: Any = None,
    ) -> HarnessResult:
        """Run a conversation through the harness."""
        if not self._initialized:
            return HarnessResult(
                content="Error: Harness not initialized.",
                agent_name="system", session_id="", finish_reason="error",
            )

        session_id = session_id or str(uuid.uuid4())[:12]

        # Resolve agent and inject scoped memory
        resolved_name = agent_name if agent_name != "auto" else "ceo"
        agent_spec = self._agents.get(resolved_name, {})
        agent_team = agent_spec.get("team")

        memory_context = self.memory.build_context_for_agent(
            resolved_name, team=agent_team
        )

        # Build invocation messages
        invoke_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                continue
            invoke_messages.append({
                "role": msg["role"],
                "content": msg.get("content", ""),
            })

        # Prepend memory context
        if memory_context:
            invoke_messages.insert(0, {
                "role": "user",
                "content": (
                    f"[SYSTEM CONTEXT — Universal Memory]\n\n{memory_context}\n\n"
                    f"[END SYSTEM CONTEXT]\n\n"
                    f"The above is shared context from universal memory. "
                    f"Now respond to the user's actual message below."
                ),
            })

        # Route to specific agent if requested
        if agent_name not in ("ceo", "auto") and agent_name in self._agents:
            user_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
            invoke_messages.append({
                "role": "user",
                "content": (
                    f"Route this request to the {agent_name} agent. "
                    f"User's request: {user_content}"
                ),
            })

        # Convert to LangChain format
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        lc_messages = []
        for msg in invoke_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        config = {"configurable": {"thread_id": session_id}}

        try:
            t0 = time.monotonic()
            if on_progress:
                await on_progress("🧠 Deep Agent Harness processing...")

            result = await self._agent_graph.ainvoke(
                {"messages": lc_messages}, config=config,
            )
            elapsed = time.monotonic() - t0

            final_messages = result.get("messages", [])
            content = ""
            if final_messages:
                final_msg = final_messages[-1]
                content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            else:
                content = "No response generated."

            if content and len(content) > 50:
                self._save_session_async(messages, content, resolved_name, session_id)

            logger.info(
                f"Harness run: agent={resolved_name}, session={session_id}, "
                f"elapsed={elapsed:.1f}s, response={len(content)} chars"
            )

            return HarnessResult(
                content=content, agent_name=agent_name,
                session_id=session_id, finish_reason="stop",
                elapsed_ms=elapsed * 1000,
            )

        except Exception as e:
            logger.error(f"Harness run failed: {e}", exc_info=True)
            return HarnessResult(
                content=f"Error: {e}", agent_name=agent_name,
                session_id=session_id, finish_reason="error",
            )

    async def run_stream(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        on_progress: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a conversation with SSE chunks."""
        if not self._initialized:
            yield _make_chunk(session_id or "", agent_name, "Error: not initialized", finish=True)
            return

        session_id = session_id or str(uuid.uuid4())[:12]

        resolved_name = agent_name if agent_name != "auto" else "ceo"
        agent_spec = self._agents.get(resolved_name, {})
        agent_team = agent_spec.get("team")

        memory_context = self.memory.build_context_for_agent(
            resolved_name, team=agent_team
        )

        from langchain_core.messages import HumanMessage, AIMessage

        lc_messages = []
        if memory_context:
            lc_messages.append(HumanMessage(
                content=f"[SYSTEM CONTEXT]\n{memory_context}\n[END CONTEXT]"
            ))

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        config = {"configurable": {"thread_id": session_id}}

        try:
            collected_content = ""
            prev_len = 0
            async for event in self._agent_graph.astream(
                {"messages": lc_messages}, config=config,
                stream_mode="values",
            ):
                msgs = event.get("messages", []) if isinstance(event, dict) else []
                if not msgs:
                    continue
                # Only look at new messages since last event
                new_msgs = msgs[prev_len:]
                prev_len = len(msgs)
                for msg in new_msgs:
                    if hasattr(msg, "content") and msg.content:
                        from langchain_core.messages import AIMessage as AIM
                        if isinstance(msg, AIM) and not getattr(msg, "tool_calls", None):
                            delta = msg.content
                            collected_content += delta
                            yield _make_chunk(session_id, agent_name, delta)
                            if on_progress:
                                await on_progress(delta)

            yield _make_chunk(session_id, agent_name, "", finish=True)

            if collected_content and len(collected_content) > 50:
                self._save_session_async(
                    messages, collected_content, resolved_name, session_id
                )

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield _make_chunk(session_id, agent_name, f"Error: {e}", finish=True)

    # ── Agent listing ────────────────────────────────────────────

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents as OpenAI /v1/models entries."""
        models = []
        for name, spec in self._agents.items():
            if name.endswith("_memory"):
                continue
            models.append({
                "id": f"agent/{name}",
                "object": "model",
                "created": 0,
                "owned_by": "nanobot-deepagents",
                "permission": [],
                "root": f"agent/{name}",
                "parent": None,
                "metadata": {
                    "team": spec["team"],
                    "role": spec["role"],
                    "description": spec["description"],
                    "engine": "deepagents",
                },
            })
        return models

    def get_agent(self, name: str) -> dict[str, Any] | None:
        return self._agents.get(name)

    @property
    def agent_count(self) -> int:
        return sum(1 for n in self._agents if not n.endswith("_memory"))

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── Memory API ───────────────────────────────────────────────

    def get_memory(self) -> UniversalMemory:
        return self.memory

    def get_memory_snapshot(self) -> dict[str, Any]:
        return self.memory.to_dict()


# ── Result and SSE helpers ───────────────────────────────────────

class HarnessResult:
    """Result from a harness run."""

    __slots__ = (
        "content", "agent_name", "session_id",
        "finish_reason", "handoff_chain", "usage",
        "elapsed_ms",
    )

    def __init__(
        self,
        content: str,
        agent_name: str,
        session_id: str,
        finish_reason: str = "stop",
        handoff_chain: list[str] | None = None,
        usage: dict[str, int] | None = None,
        elapsed_ms: float = 0,
    ):
        self.content = content
        self.agent_name = agent_name
        self.session_id = session_id
        self.finish_reason = finish_reason
        self.handoff_chain = handoff_chain or []
        self.usage = usage or {}
        self.elapsed_ms = elapsed_ms

    def to_chat_completion(self, model: str = "agent/unknown") -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{self.session_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"agent/{self.agent_name}",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self.content},
                "finish_reason": "stop" if self.finish_reason in ("stop", "loop_detected") else self.finish_reason,
            }],
            "usage": {
                "prompt_tokens": self.usage.get("prompt_tokens", 0),
                "completion_tokens": self.usage.get("completion_tokens", 0),
                "total_tokens": self.usage.get("total_tokens", 0),
            },
            "nanobot_metadata": {
                "session_id": self.session_id,
                "responding_agent": self.agent_name,
                "engine": "deepagents",
                "elapsed_ms": self.elapsed_ms,
                "handoff_chain": self.handoff_chain,
            },
        }

    def to_stream_chunk(self, delta_content: str = "") -> dict[str, Any]:
        return _make_chunk(self.session_id, self.agent_name, delta_content, finish=not delta_content)


def _make_chunk(
    session_id: str,
    agent_name: str,
    content: str,
    finish: bool = False,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{session_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": f"agent/{agent_name}",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": "stop" if finish else None,
        }],
    }
