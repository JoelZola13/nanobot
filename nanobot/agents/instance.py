"""Agent instance — runtime execution loop for a single agent."""

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

# Type alias for the progress callback
ProgressCallback = Callable[[str], Awaitable[None]] | None

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agents.spec import AgentSpec
from nanobot.agents.tools.transfer import is_handoff_signal
from nanobot.agents.tracing import AgentSpan

# ── Dedup / drift detection constants ────────────────────────────
DEDUP_COOLDOWN_SECS = 5        # suppress identical tool call within N seconds
DRIFT_REPEAT_THRESHOLD = 3     # consecutive identical calls → bail out


class AgentInstance:
    """
    Runtime execution loop for a single agent.

    Modeled after SubagentManager._run_subagent() but designed for
    multi-agent orchestration. Runs the LLM chat loop, handles tool
    calls, and detects handoff signals.

    The instance does NOT perform handoffs itself — it returns the
    handoff signal to the Orchestrator, which manages the actual
    agent transfer.
    """

    def __init__(
        self,
        spec: AgentSpec,
        provider: LLMProvider,
        tools: ToolRegistry,
    ):
        self.spec = spec
        self.provider = provider
        self.tools = tools

    async def run(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        on_progress: ProgressCallback = None,
        span: AgentSpan | None = None,
    ) -> "AgentResult":
        """
        Execute the agent loop until it produces a response or handoff.

        Args:
            messages: Conversation messages (system prompt should already
                      be prepended by the Orchestrator).
            model: Override model (defaults to spec.model resolved by Orchestrator).
            on_progress: Async callback for streaming progress updates.
            span: Optional AgentSpan for structured tracing/observability.

        Returns:
            AgentResult with either a text response or a handoff signal.
        """
        resolved_model = model or (
            None if self.spec.model == "default" else self.spec.model
        )

        tool_defs = self.tools.get_definitions() if len(self.tools) > 0 else None
        iteration = 0

        # ── Dedup / drift state ──────────────────────────────────
        # Maps (tool_name, args_json) → last_call_timestamp
        _recent_calls: dict[tuple[str, str], float] = {}
        # Tracks last N call signatures for drift detection
        _call_history: list[tuple[str, str]] = []

        while iteration < self.spec.max_iterations:
            iteration += 1
            logger.debug(
                f"Agent '{self.spec.name}' iteration {iteration}/{self.spec.max_iterations}"
            )

            try:
                llm_t0 = time.monotonic()
                response: LLMResponse = await self.provider.chat(
                    messages=messages,
                    tools=tool_defs,
                    model=resolved_model,
                    max_tokens=self.spec.max_tokens,
                    temperature=self.spec.temperature,
                )
                llm_ms = (time.monotonic() - llm_t0) * 1000
                logger.info(
                    f"Agent '{self.spec.name}' LLM call [{llm_ms:.0f}ms] "
                    f"iter={iteration} has_tools={response.has_tool_calls} "
                    f"usage={response.usage}"
                )
                if span:
                    span.record_llm_call(
                        model=resolved_model,
                        iteration=iteration,
                        duration_ms=llm_ms,
                        usage=response.usage,
                        has_tool_calls=response.has_tool_calls,
                        tool_call_names=[tc.name for tc in response.tool_calls] if response.has_tool_calls else [],
                    )
            except Exception as e:
                logger.error(f"Agent '{self.spec.name}' LLM error: {e}")
                return AgentResult(
                    agent_name=self.spec.name,
                    content=f"Error: LLM call failed — {e}",
                    finish_reason="error",
                )

            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    }
                )

                # Execute each tool call
                for tc in response.tool_calls:
                    args_json = json.dumps(tc.arguments, sort_keys=True)
                    call_sig = (tc.name, args_json)

                    logger.debug(
                        f"Agent '{self.spec.name}' calling tool: {tc.name}"
                    )
                    logger.debug(
                        f"  Tool args: {args_json[:200]}"
                    )

                    # ── Dedup: skip if identical call within cooldown ─────
                    now = time.monotonic()
                    last_ts = _recent_calls.get(call_sig)
                    if last_ts is not None and (now - last_ts) < DEDUP_COOLDOWN_SECS:
                        logger.warning(
                            f"Agent '{self.spec.name}' dedup: skipping repeated "
                            f"'{tc.name}' call within {DEDUP_COOLDOWN_SECS}s"
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.name,
                                "content": (
                                    "[DEDUP] This identical tool call was just executed. "
                                    "Use the previous result instead of calling again."
                                ),
                            }
                        )
                        continue
                    _recent_calls[call_sig] = now

                    # ── Drift detection: bail if same call repeated N times ─
                    _call_history.append(call_sig)
                    if len(_call_history) >= DRIFT_REPEAT_THRESHOLD:
                        tail = _call_history[-DRIFT_REPEAT_THRESHOLD:]
                        if all(s == tail[0] for s in tail):
                            logger.warning(
                                f"Agent '{self.spec.name}' drift detected: "
                                f"'{tc.name}' called {DRIFT_REPEAT_THRESHOLD}x "
                                f"consecutively with same args — aborting loop"
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "name": tc.name,
                                    "content": (
                                        "[DRIFT] This tool has been called repeatedly "
                                        "with the same arguments. Stopping to avoid "
                                        "an infinite loop."
                                    ),
                                }
                            )
                            return AgentResult(
                                agent_name=self.spec.name,
                                content=(
                                    "I detected a repeating pattern and stopped to "
                                    "avoid wasting resources. Here's what I have so far."
                                ),
                                finish_reason="drift",
                                messages=messages,
                            )

                    # Stream tool call progress
                    if on_progress:
                        await on_progress(
                            f"🔧 **{self.spec.name}** → `{tc.name}`"
                        )

                    # Propagate on_progress to delegate tools so sub-agents
                    # can also stream their progress
                    tool_impl = self.tools.get(tc.name)
                    if tool_impl and hasattr(tool_impl, '_on_progress'):
                        tool_impl._on_progress = on_progress

                    t0 = time.monotonic()
                    try:
                        result = await self.tools.execute(tc.name, tc.arguments)
                    except Exception as tool_err:
                        logger.error(
                            f"Agent '{self.spec.name}' tool '{tc.name}' "
                            f"raised: {tool_err}"
                        )
                        result = (
                            f"[TOOL_ERROR] {tc.name} failed: {tool_err}\n"
                            "The tool encountered an error. You can try a "
                            "different approach or inform the user."
                        )
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    logger.info(
                        f"  Tool '{tc.name}' [{elapsed_ms:.0f}ms] "
                        f"({len(result)} chars): {result[:300]}"
                    )
                    if span:
                        span.record_tool_call(
                            name=tc.name,
                            arguments=tc.arguments,
                            result=result,
                            duration_ms=elapsed_ms,
                            is_handoff=is_handoff_signal(result),
                        )

                    # Check if the tool result is a handoff signal
                    if is_handoff_signal(result):
                        # Still add the tool result to messages for completeness
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.name,
                                "content": result,
                            }
                        )
                        return AgentResult(
                            agent_name=self.spec.name,
                            content=result,
                            finish_reason="handoff",
                            messages=messages,
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result,
                        }
                    )
            else:
                # No tool calls — agent is done, return text response
                return AgentResult(
                    agent_name=self.spec.name,
                    content=response.content or "",
                    finish_reason="stop",
                    messages=messages,
                    usage=response.usage,
                )

        # Hit max iterations
        logger.warning(
            f"Agent '{self.spec.name}' hit max iterations ({self.spec.max_iterations})"
        )
        return AgentResult(
            agent_name=self.spec.name,
            content="I've reached my processing limit. Here's what I have so far.",
            finish_reason="max_iterations",
            messages=messages,
        )


class AgentResult:
    """
    Result from an agent execution.

    Attributes:
        agent_name: Name of the agent that produced this result.
        content: Text content or handoff signal string.
        finish_reason: "stop" | "handoff" | "error" | "max_iterations"
        messages: Updated conversation messages (for handoff continuity).
        usage: Token usage from the LLM.
    """

    __slots__ = ("agent_name", "content", "finish_reason", "messages", "usage")

    def __init__(
        self,
        agent_name: str,
        content: str,
        finish_reason: str = "stop",
        messages: list[dict[str, Any]] | None = None,
        usage: dict[str, int] | None = None,
    ):
        self.agent_name = agent_name
        self.content = content
        self.finish_reason = finish_reason
        self.messages = messages or []
        self.usage = usage or {}

    @property
    def is_handoff(self) -> bool:
        """Whether this result is a handoff signal."""
        return self.finish_reason == "handoff"

    @property
    def is_error(self) -> bool:
        """Whether this result is an error."""
        return self.finish_reason == "error"

    def __repr__(self) -> str:
        return (
            f"AgentResult(agent={self.agent_name!r}, "
            f"reason={self.finish_reason!r}, "
            f"content={self.content[:50]!r}...)"
        )
