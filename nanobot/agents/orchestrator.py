"""Orchestrator — manages multi-agent handoff loops and session routing."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.agents.spec import AgentSpec
from nanobot.agents.registry import AgentRegistry
from nanobot.agents.factory import ToolFactory
from nanobot.agents.instance import AgentInstance, AgentResult
from nanobot.agents.tools.transfer import parse_handoff_signal
from nanobot.agents.tracing import Trace, emit_handoff_span
from nanobot.agents.router import route_request

# Type alias for the progress callback
ProgressCallback = Callable[[str], Awaitable[None]] | None


class AgentSession:
    """
    Tracks a multi-agent conversation session.

    Each session begins with a target agent and carries messages
    across handoffs so agents can see relevant context.
    """

    __slots__ = (
        "session_id",
        "current_agent",
        "messages",
        "handoff_chain",
        "created_at",
        "total_usage",
    )

    def __init__(self, session_id: str, initial_agent: str):
        self.session_id = session_id
        self.current_agent = initial_agent
        self.messages: list[dict[str, Any]] = []
        self.handoff_chain: list[str] = [initial_agent]
        self.created_at = datetime.now()
        self.total_usage: dict[str, int] = {}

    def record_handoff(self, target: str) -> None:
        """Record an agent handoff in the chain."""
        self.handoff_chain.append(target)
        self.current_agent = target

    def accumulate_usage(self, usage: dict[str, int]) -> None:
        """Add token usage from an agent run."""
        for key, val in usage.items():
            self.total_usage[key] = self.total_usage.get(key, 0) + val


class Orchestrator:
    """
    Central coordinator for multi-agent conversations.

    Responsibilities:
    - Routes incoming requests to the correct agent
    - Manages the handoff loop (agent A → agent B → ... → final response)
    - Builds per-agent tool sets via ToolFactory
    - Tracks sessions across handoffs
    - Prevents infinite handoff loops

    Usage from api_server.py:
        orchestrator = Orchestrator(provider, agent_registry, ...)
        result = await orchestrator.run("ceo", messages)
    """

    # Maximum consecutive handoffs before we force a stop
    MAX_HANDOFFS = 10

    def __init__(
        self,
        provider: LLMProvider,
        agent_registry: AgentRegistry,
        tool_factory: ToolFactory,
        default_model: str | None = None,
    ):
        self.provider = provider
        self.registry = agent_registry
        self.tool_factory = tool_factory
        self.default_model = default_model
        self._sessions: dict[str, AgentSession] = {}

    async def run(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        model_override: str | None = None,
        on_progress: ProgressCallback = None,
    ) -> OrchestratorResult:
        """
        Run a multi-agent conversation starting from the specified agent.

        This is the main entry point called by the API server. It:
        1. Resolves the target agent
        2. Prepends the agent's system prompt
        3. Runs the agent loop
        4. If the agent hands off, switches to the target agent and repeats
        5. Returns the final text response

        Args:
            agent_name: Name of the initial agent (e.g. "ceo").
            messages: User conversation messages (no system prompt — we add it).
            session_id: Optional session ID for tracking.
            model_override: Override the agent's model choice.

        Returns:
            OrchestratorResult with the final response and metadata.
        """
        # ── Auto-routing: resolve "auto" to the best agent ──
        if agent_name == "auto":
            agent_name = route_request(
                messages, self.registry, fallback="ceo"
            )
            logger.info(f"Auto-routed request to '{agent_name}'")
            if on_progress:
                await on_progress(f"🔀 Routed to **{agent_name}**")

        session_id = session_id or str(uuid.uuid4())[:12]
        session = AgentSession(session_id, agent_name)
        self._sessions[session_id] = session

        # Create a trace for this run (context manager emits SDK trace)
        trace = Trace(session_id=session_id, initial_agent=agent_name)
        return await self._run_with_trace(trace, session, agent_name, messages, model_override, on_progress)

    async def _run_with_trace(
        self,
        trace: Trace,
        session: AgentSession,
        agent_name: str,
        messages: list[dict[str, Any]],
        model_override: str | None,
        on_progress: ProgressCallback = None,
    ) -> OrchestratorResult:
        """Inner run loop wrapped in SDK trace context."""
        current_agent = agent_name
        handoff_count = 0
        working_messages = list(messages)
        session_id = session.session_id

        with trace:
          while handoff_count <= self.MAX_HANDOFFS:
            # Resolve the agent spec
            spec = self.registry.get(current_agent)
            if spec is None:
                logger.error(f"Agent '{current_agent}' not found in registry")
                trace.finish()
                return OrchestratorResult(
                    content=f"Error: Agent '{current_agent}' is not available.",
                    agent_name=current_agent,
                    session_id=session_id,
                    finish_reason="error",
                    handoff_chain=session.handoff_chain,
                    trace=trace,
                )

            # Build the agent's tool set
            tools = self.tool_factory.build_tools(spec)

            # Prepare messages with system prompt
            agent_messages = self._prepare_messages(spec, working_messages, session)

            # Determine the model
            model = model_override or self._resolve_model(spec)

            # Create and run the agent instance
            instance = AgentInstance(spec, self.provider, tools)

            logger.info(
                f"[{session_id}] Running agent '{current_agent}' "
                f"(handoff #{handoff_count})"
            )

            if on_progress:
                await on_progress(f"🤖 **{current_agent}** is working...")

            # Start a tracing span for this agent
            span = trace.start_agent_span(current_agent)

            result: AgentResult = await instance.run(agent_messages, model, on_progress=on_progress)
            session.accumulate_usage(result.usage)

            if result.is_handoff:
                # Parse the handoff signal
                target, reason, context = parse_handoff_signal(result.content)

                if not target:
                    logger.error(f"Invalid handoff signal from '{current_agent}'")
                    trace.finish()
                    return OrchestratorResult(
                        content="An internal routing error occurred.",
                        agent_name=current_agent,
                        session_id=session_id,
                        finish_reason="error",
                        handoff_chain=session.handoff_chain,
                        trace=trace,
                    )

                # Check for handoff loops
                if self._detect_loop(session.handoff_chain, target):
                    logger.warning(
                        f"[{session_id}] Handoff loop detected: "
                        f"{' → '.join(session.handoff_chain)} → {target}"
                    )
                    trace.finish()
                    # Force the current agent to respond instead
                    return OrchestratorResult(
                        content=(
                            f"I noticed a routing loop between agents. "
                            f"Let me try to help directly with your request."
                        ),
                        agent_name=current_agent,
                        session_id=session_id,
                        finish_reason="loop_detected",
                        handoff_chain=session.handoff_chain,
                        trace=trace,
                    )

                logger.info(
                    f"[{session_id}] Handoff: {current_agent} → {target} "
                    f"(reason: {reason})"
                )

                if on_progress:
                    await on_progress(
                        f"➡️ **{current_agent}** → **{target}** ({reason})"
                    )

                emit_handoff_span(current_agent, target)
                session.record_handoff(target)
                handoff_count += 1
                current_agent = target

                # Inject handoff context into working messages so the
                # next agent knows why it was called
                working_messages = self._inject_handoff_context(
                    working_messages, current_agent, reason, context, session
                )

            elif result.is_error:
                trace.finish()
                return OrchestratorResult(
                    content=result.content,
                    agent_name=current_agent,
                    session_id=session_id,
                    finish_reason="error",
                    handoff_chain=session.handoff_chain,
                    usage=session.total_usage,
                    trace=trace,
                )
            else:
                # Agent produced a final response
                trace.finish()
                logger.info(f"[{session_id}] {trace.log_summary()}")
                return OrchestratorResult(
                    content=result.content,
                    agent_name=current_agent,
                    session_id=session_id,
                    finish_reason="stop",
                    handoff_chain=session.handoff_chain,
                    usage=session.total_usage,
                    trace=trace,
                )

        # Exceeded max handoffs
        logger.error(
            f"[{session_id}] Exceeded max handoffs ({self.MAX_HANDOFFS})"
        )
        trace.finish()
        return OrchestratorResult(
            content="I've been passed between too many agents. Let me try to help directly.",
            agent_name=current_agent,
            session_id=session_id,
            finish_reason="max_handoffs",
            handoff_chain=session.handoff_chain,
            usage=session.total_usage,
            trace=trace,
        )

    def _prepare_messages(
        self,
        spec: AgentSpec,
        user_messages: list[dict[str, Any]],
        session: AgentSession,
    ) -> list[dict[str, Any]]:
        """
        Prepare messages for an agent, prepending its system prompt.

        If user_messages already contain a system message, we replace it
        with the agent's own system prompt to maintain agent identity.
        """
        from datetime import datetime as dt
        import time as _time

        now = dt.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        # Build the system prompt with agent identity + time context
        system_parts = [
            f"# Agent: {spec.name}",
            f"Team: {spec.team} | Role: {spec.role}",
            f"Current time: {now} ({tz})",
            "",
        ]

        if spec.system_prompt:
            system_parts.append(spec.system_prompt)
        else:
            system_parts.append(
                f"You are the {spec.name} agent. {spec.description}"
            )

        # Add handoff chain context if we've been through multiple agents
        if len(session.handoff_chain) > 1:
            chain_str = " → ".join(session.handoff_chain)
            system_parts.append(
                f"\n## Conversation Routing\n"
                f"This conversation was routed through: {chain_str}\n"
                f"You are the current active agent."
            )

        system_content = "\n".join(system_parts)

        # Build final messages: system + user messages (strip any existing system msgs)
        result: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]
        for msg in user_messages:
            if msg.get("role") != "system":
                result.append(msg)

        return result

    def _inject_handoff_context(
        self,
        messages: list[dict[str, Any]],
        target_agent: str,
        reason: str,
        context: str,
        session: AgentSession,
    ) -> list[dict[str, Any]]:
        """
        Inject handoff context so the receiving agent understands
        why it was called.

        We add a system-level note as the first user-visible context.
        """
        handoff_note = f"[Handoff to {target_agent}] Reason: {reason}"
        if context:
            handoff_note += f"\nContext: {context}"

        # Keep user messages, add handoff context
        result = list(messages)
        result.append(
            {
                "role": "user",
                "content": handoff_note,
            }
        )
        return result

    def _resolve_model(self, spec: AgentSpec) -> str | None:
        """Resolve the model for an agent. Returns None to use provider default."""
        if spec.model == "default":
            return self.default_model
        return spec.model

    def _detect_loop(
        self, chain: list[str], target: str, window: int = 4
    ) -> bool:
        """
        Detect if a handoff would create a loop.

        Checks if the target has appeared in the last `window` handoffs,
        which indicates a ping-pong pattern.
        """
        recent = chain[-window:]
        # If target appeared twice in recent window, it's a loop
        return recent.count(target) >= 2

    def get_session(self, session_id: str) -> AgentSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents as OpenAI /v1/models entries."""
        return self.registry.list_as_models()


class OrchestratorResult:
    """
    Final result from the Orchestrator.

    Suitable for conversion to OpenAI chat completion response format.
    """

    __slots__ = (
        "content",
        "agent_name",
        "session_id",
        "finish_reason",
        "handoff_chain",
        "usage",
        "trace",
    )

    def __init__(
        self,
        content: str,
        agent_name: str,
        session_id: str,
        finish_reason: str = "stop",
        handoff_chain: list[str] | None = None,
        usage: dict[str, int] | None = None,
        trace: Trace | None = None,
    ):
        self.content = content
        self.agent_name = agent_name
        self.session_id = session_id
        self.finish_reason = finish_reason
        self.handoff_chain = handoff_chain or []
        self.usage = usage or {}
        self.trace = trace

    def to_chat_completion(self, model: str = "agent/unknown") -> dict[str, Any]:
        """Convert to OpenAI chat completion response format."""
        import time

        return {
            "id": f"chatcmpl-{self.session_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"agent/{self.agent_name}",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self.content,
                    },
                    "finish_reason": (
                        "stop" if self.finish_reason in ("stop", "loop_detected", "max_handoffs")
                        else self.finish_reason
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": self.usage.get("prompt_tokens", 0),
                "completion_tokens": self.usage.get("completion_tokens", 0),
                "total_tokens": self.usage.get("total_tokens", 0),
            },
            # Custom metadata
            "nanobot_metadata": {
                "session_id": self.session_id,
                "responding_agent": self.agent_name,
                "handoff_chain": self.handoff_chain,
                **({"trace": self.trace.summary()} if self.trace else {}),
            },
        }

    def to_stream_chunk(self, delta_content: str = "") -> dict[str, Any]:
        """Convert to a single SSE stream chunk."""
        import time

        return {
            "id": f"chatcmpl-{self.session_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": f"agent/{self.agent_name}",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta_content} if delta_content else {},
                    "finish_reason": None if delta_content else "stop",
                }
            ],
        }

    def __repr__(self) -> str:
        chain = " → ".join(self.handoff_chain)
        return (
            f"OrchestratorResult(agent={self.agent_name!r}, "
            f"reason={self.finish_reason!r}, chain=[{chain}])"
        )
