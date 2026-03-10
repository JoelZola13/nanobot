"""Tracing module for multi-agent orchestration.

Integrates with the OpenAI Agents SDK tracing system while maintaining
Nanobot's structured trace format. Each orchestrator run creates both:
- A Nanobot Trace (for structured summaries, log output, API responses)
- An Agents SDK trace (for trace processors, OpenTelemetry export)

The SDK integration enables:
- TracingProcessor plugins (log, export, custom analysis)
- Context-propagated spans (agent, function, generation, handoff)
- OpenTelemetry-compatible export via set_trace_processors()

Usage:
    from nanobot.agents.tracing import Trace, setup_tracing

    # Register processors at startup
    setup_tracing(processors=[ConsoleTracingProcessor()])

    # In orchestrator — SDK trace wraps the run
    trace = Trace(session_id="abc123", initial_agent="ceo")
    with trace:
        span = trace.start_agent_span("ceo")
        span.record_llm_call(...)
        span.record_tool_call(...)
    summary = trace.summary()
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Agents SDK tracing imports
from agents.tracing import (
    TracingProcessor,
    add_trace_processor,
    set_trace_processors,
    trace as sdk_trace,
    agent_span as sdk_agent_span,
    function_span as sdk_function_span,
    generation_span as sdk_generation_span,
    handoff_span as sdk_handoff_span,
    custom_span as sdk_custom_span,
    get_current_trace as sdk_get_current_trace,
    get_current_span as sdk_get_current_span,
)
from agents.tracing import (
    Trace as SDKTrace,
    Span as SDKSpan,
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    HandoffSpanData,
    CustomSpanData,
)


# ---------------------------------------------------------------------------
# Nanobot record types (unchanged — used for structured summaries)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """A single tool call within an agent span."""

    name: str
    arguments: dict[str, Any]
    result_preview: str  # truncated result
    duration_ms: float
    is_handoff: bool = False
    handoff_target: str | None = None


@dataclass
class LLMCallRecord:
    """A single LLM call within an agent span."""

    model: str | None
    iteration: int
    duration_ms: float
    usage: dict[str, int] = field(default_factory=dict)
    has_tool_calls: bool = False
    tool_call_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentSpan — tracks one agent's execution, emits SDK spans
# ---------------------------------------------------------------------------


@dataclass
class AgentSpan:
    """
    Tracks the execution of a single agent within the orchestrator.

    Records LLM calls, tool calls, timing, and outcome.
    Also emits corresponding Agents SDK spans for external processors.
    """

    agent_name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None

    # Records
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # Outcome
    finish_reason: str | None = None  # stop, handoff, error, max_iterations
    handoff_target: str | None = None
    handoff_reason: str | None = None
    error: str | None = None

    # SDK span handle (set by Trace.start_agent_span)
    _sdk_span: Any = field(default=None, repr=False)

    def record_llm_call(
        self,
        model: str | None,
        iteration: int,
        duration_ms: float,
        usage: dict[str, int] | None = None,
        has_tool_calls: bool = False,
        tool_call_names: list[str] | None = None,
    ) -> None:
        self.llm_calls.append(
            LLMCallRecord(
                model=model,
                iteration=iteration,
                duration_ms=duration_ms,
                usage=usage or {},
                has_tool_calls=has_tool_calls,
                tool_call_names=tool_call_names or [],
            )
        )

    def record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
        duration_ms: float,
        is_handoff: bool = False,
        handoff_target: str | None = None,
    ) -> None:
        # Truncate result for storage
        preview = result[:200] + "..." if len(result) > 200 else result
        self.tool_calls.append(
            ToolCallRecord(
                name=name,
                arguments=arguments,
                result_preview=preview,
                duration_ms=duration_ms,
                is_handoff=is_handoff,
                handoff_target=handoff_target,
            )
        )

    def record_handoff(self, target: str, reason: str) -> None:
        self.handoff_target = target
        self.handoff_reason = reason
        self.finish_reason = "handoff"

    def finish(self, reason: str, error: str | None = None) -> None:
        self.end_time = time.monotonic()
        self.finish_reason = reason
        self.error = error
        # Close the SDK agent span
        if self._sdk_span is not None:
            try:
                if error:
                    from agents.tracing import SpanError
                    self._sdk_span.set_error(SpanError(message=error))
                self._sdk_span.__exit__(None, None, None)
            except Exception:
                pass  # SDK span already closed or unavailable

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.monotonic()
        return (end - self.start_time) * 1000

    @property
    def total_llm_ms(self) -> float:
        return sum(c.duration_ms for c in self.llm_calls)

    @property
    def total_tool_ms(self) -> float:
        return sum(c.duration_ms for c in self.tool_calls)

    @property
    def total_tokens(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for call in self.llm_calls:
            for k, v in call.usage.items():
                totals[k] = totals.get(k, 0) + v
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "agent": self.agent_name,
            "duration_ms": round(self.duration_ms, 1),
            "finish_reason": self.finish_reason,
            "llm_calls": len(self.llm_calls),
            "tool_calls": [
                {
                    "name": tc.name,
                    "duration_ms": round(tc.duration_ms, 1),
                    "is_handoff": tc.is_handoff,
                    **({"handoff_target": tc.handoff_target} if tc.handoff_target else {}),
                }
                for tc in self.tool_calls
            ],
            "tokens": self.total_tokens,
            **({"handoff_target": self.handoff_target} if self.handoff_target else {}),
            **({"handoff_reason": self.handoff_reason} if self.handoff_reason else {}),
            **({"error": self.error} if self.error else {}),
        }


# ---------------------------------------------------------------------------
# Trace — top-level trace with SDK context manager integration
# ---------------------------------------------------------------------------


class Trace:
    """
    Top-level trace for a multi-agent orchestrator run.

    Acts as a context manager that creates an Agents SDK trace alongside
    the Nanobot structured trace. Use `with trace:` to automatically
    start/finish the SDK trace and propagate context.
    """

    def __init__(self, session_id: str, initial_agent: str):
        self.trace_id = uuid.uuid4().hex[:12]
        self.session_id = session_id
        self.initial_agent = initial_agent
        self.start_time = time.monotonic()
        self.end_time: float | None = None
        self.spans: list[AgentSpan] = []
        self._current_span: AgentSpan | None = None
        self._sdk_trace: Any = None  # SDK TraceCtxManager

    def __enter__(self) -> Trace:
        """Start the SDK trace context."""
        try:
            self._sdk_trace = sdk_trace(
                workflow_name=f"nanobot:{self.initial_agent}",
                trace_id=self.trace_id,
                group_id=self.session_id,
                metadata={"session_id": self.session_id, "initial_agent": self.initial_agent},
            )
            self._sdk_trace.__enter__()
        except Exception as e:
            logger.debug(f"SDK trace start failed (non-fatal): {e}")
            self._sdk_trace = None
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Finish the trace and close SDK context."""
        self.finish()
        if self._sdk_trace is not None:
            try:
                self._sdk_trace.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass

    def start_agent_span(self, agent_name: str) -> AgentSpan:
        """Start a new span for an agent execution."""
        span = AgentSpan(agent_name=agent_name)
        self.spans.append(span)
        self._current_span = span

        # Start an SDK agent span
        try:
            tool_names = []  # populated as tools execute
            sdk_span = sdk_agent_span(
                name=agent_name,
                handoffs=[],
                tools=tool_names,
            )
            sdk_span.__enter__()
            span._sdk_span = sdk_span
        except Exception as e:
            logger.debug(f"SDK agent_span start failed (non-fatal): {e}")

        return span

    @property
    def current_span(self) -> AgentSpan | None:
        return self._current_span

    def finish(self) -> None:
        if self.end_time is not None:
            return  # already finished
        self.end_time = time.monotonic()
        # Finish any open span
        if self._current_span and self._current_span.end_time is None:
            self._current_span.finish("trace_end")

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.monotonic()
        return (end - self.start_time) * 1000

    @property
    def handoff_chain(self) -> list[str]:
        return [s.agent_name for s in self.spans]

    @property
    def responding_agent(self) -> str:
        return self.spans[-1].agent_name if self.spans else self.initial_agent

    @property
    def total_handoffs(self) -> int:
        return max(0, len(self.spans) - 1)

    def summary(self) -> dict[str, Any]:
        """Produce a structured trace summary for API responses and logs."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "duration_ms": round(self.duration_ms, 1),
            "initial_agent": self.initial_agent,
            "responding_agent": self.responding_agent,
            "total_handoffs": self.total_handoffs,
            "handoff_chain": self.handoff_chain,
            "spans": [s.to_dict() for s in self.spans],
        }

    def log_summary(self) -> str:
        """
        Human-readable summary for log output.

        Example:
            Trace abc123 | ceo -> content_manager -> article_writer | 2340ms | 2 handoffs
              +-- ceo (245ms) -> handoff to content_manager
              +-- content_manager (120ms) -> handoff to article_writer
              +-- article_writer (1975ms) -> stop
        """
        chain = " → ".join(self.handoff_chain)
        lines = [
            f"Trace {self.trace_id} | {chain} | "
            f"{self.duration_ms:.0f}ms | {self.total_handoffs} handoff(s)"
        ]

        for i, span in enumerate(self.spans):
            is_last = i == len(self.spans) - 1
            prefix = "  └─ " if is_last else "  ├─ "
            outcome = span.finish_reason or "?"
            if span.handoff_target:
                outcome = f"handoff → {span.handoff_target}"

            tool_info = ""
            if span.tool_calls:
                tool_names = [tc.name for tc in span.tool_calls]
                tool_info = f" tools=[{', '.join(tool_names)}]"

            lines.append(
                f"{prefix}{span.agent_name} ({span.duration_ms:.0f}ms) "
                f"→ {outcome}{tool_info}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SDK span helpers — emit SDK spans from instance.py tool/LLM calls
# ---------------------------------------------------------------------------


def emit_function_span(name: str, input_str: str, output: str, duration_ms: float) -> None:
    """Emit an Agents SDK function span for a tool call."""
    try:
        with sdk_function_span(name=name, input=input_str, output=output):
            pass  # span is opened and immediately closed (already executed)
    except Exception:
        pass  # non-fatal


def emit_generation_span(model: str | None, duration_ms: float, usage: dict[str, int] | None = None) -> None:
    """Emit an Agents SDK generation span for an LLM call."""
    try:
        with sdk_generation_span(
            model=model or "unknown",
            model_config={},
            input=[],
            output=None,
        ):
            pass
    except Exception:
        pass  # non-fatal


def emit_handoff_span(from_agent: str, to_agent: str) -> None:
    """Emit an Agents SDK handoff span."""
    try:
        with sdk_handoff_span(from_agent=from_agent, to_agent=to_agent):
            pass
    except Exception:
        pass  # non-fatal


# ---------------------------------------------------------------------------
# NanobotTracingProcessor — logs trace summaries via loguru
# ---------------------------------------------------------------------------


class NanobotTracingProcessor(TracingProcessor):
    """
    Agents SDK TracingProcessor that logs trace events via loguru.

    Register with setup_tracing() to get structured logs for every
    trace and span in the system.
    """

    def on_trace_start(self, trace: SDKTrace) -> None:
        logger.debug(f"[sdk-trace] Started: {trace.trace_id} name={trace.name}")

    def on_trace_end(self, trace: SDKTrace) -> None:
        logger.debug(f"[sdk-trace] Ended: {trace.trace_id} name={trace.name}")

    def on_span_start(self, span: SDKSpan[Any]) -> None:
        span_type = span.span_data.type if span.span_data else "unknown"
        logger.debug(f"[sdk-span] Started: {span.span_id} type={span_type}")

    def on_span_end(self, span: SDKSpan[Any]) -> None:
        span_type = span.span_data.type if span.span_data else "unknown"
        data = span.span_data.export() if span.span_data else {}
        logger.debug(f"[sdk-span] Ended: {span.span_id} type={span_type} data={data}")

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_tracing(
    processors: list[TracingProcessor] | None = None,
    include_default: bool = True,
) -> None:
    """
    Initialize the Agents SDK tracing system with processors.

    Args:
        processors: Custom TracingProcessor instances to register.
        include_default: If True, include NanobotTracingProcessor for loguru output.
    """
    all_processors: list[TracingProcessor] = []
    if include_default:
        all_processors.append(NanobotTracingProcessor())
    if processors:
        all_processors.extend(processors)

    set_trace_processors(all_processors)
    logger.info(f"Tracing initialized with {len(all_processors)} processor(s)")
