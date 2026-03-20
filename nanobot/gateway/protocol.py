"""Pydantic models for WebSocket gateway frames.

Schema-first protocol: all frames are validated on receive, unknown keys rejected.
Frames use a discriminated union on the `type` field.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ---------------------------------------------------------------------------
# Base frame
# ---------------------------------------------------------------------------

class Frame(BaseModel):
    """Base frame shared by all WS messages."""

    model_config = ConfigDict(extra="forbid")

    type: str
    id: str | None = None  # Request/response correlation ID


# ---------------------------------------------------------------------------
# Client → Server frames
# ---------------------------------------------------------------------------

class ConnectRole(str, Enum):
    operator = "operator"
    node = "node"


class ConnectFrame(Frame):
    """Initial handshake frame sent by the client."""

    type: Literal["connect"] = "connect"
    role: ConnectRole = ConnectRole.operator
    auth: str = ""  # Bearer token
    protocol_version: int = 1


class ChatSendFrame(Frame):
    """Send a chat message to the agent."""

    type: Literal["chat.send"] = "chat.send"
    content: str
    session_key: str = "default"
    stream: bool = True
    channel: str = ""  # Source channel identifier
    chat_id: str = ""  # Source chat/conversation identifier


class ChatHistoryFrame(Frame):
    """Request chat history for a session."""

    type: Literal["chat.history"] = "chat.history"
    session_key: str = "default"
    limit: int = 50


class ToolInvokeFrame(Frame):
    """Directly invoke a registered tool."""

    type: Literal["tool.invoke"] = "tool.invoke"
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class ConfigGetFrame(Frame):
    """Request current (safe) configuration."""

    type: Literal["config.get"] = "config.get"
    path: str = ""  # Optional dot-path filter (empty = full config)


class ConfigSetFrame(Frame):
    """Update a configuration value at runtime."""

    type: Literal["config.set"] = "config.set"
    path: str  # Dot-separated path, e.g. "agents.defaults.model"
    value: Any


class SessionListFrame(Frame):
    """List active sessions."""

    type: Literal["session.list"] = "session.list"


class PingFrame(Frame):
    """Client keepalive."""

    type: Literal["ping"] = "ping"


# ---------------------------------------------------------------------------
# Server → Client frames
# ---------------------------------------------------------------------------

class HelloOkFrame(Frame):
    """Handshake response advertising capabilities."""

    type: Literal["hello-ok"] = "hello-ok"
    protocol_version: int = 1
    methods: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    server_time: datetime = Field(default_factory=datetime.utcnow)


class ErrorFrame(Frame):
    """Error response."""

    type: Literal["error"] = "error"
    code: str  # Machine-readable error code
    message: str  # Human-readable detail
    ref_id: str | None = None  # ID of the frame that caused the error


class ChatTokenFrame(Frame):
    """Streaming token from agent response."""

    type: Literal["chat.token"] = "chat.token"
    session_key: str
    token: str
    finish_reason: str | None = None


class ChatResponseFrame(Frame):
    """Complete (non-streaming) agent response."""

    type: Literal["chat.response"] = "chat.response"
    session_key: str
    content: str
    model: str | None = None
    usage: dict[str, int] | None = None


class ChatHistoryResponseFrame(Frame):
    """Response to chat.history request."""

    type: Literal["chat.history.response"] = "chat.history.response"
    session_key: str
    messages: list[dict[str, Any]] = Field(default_factory=list)


class ToolResultFrame(Frame):
    """Result of a tool invocation."""

    type: Literal["tool.result"] = "tool.result"
    tool: str
    result: str
    success: bool = True


class ToolCallFrame(Frame):
    """Notification that the agent is calling a tool."""

    type: Literal["tool.call"] = "tool.call"
    session_key: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class ConfigValueFrame(Frame):
    """Response to config.get."""

    type: Literal["config.value"] = "config.value"
    config: dict[str, Any] = Field(default_factory=dict)


class SessionListResponseFrame(Frame):
    """Response to session.list."""

    type: Literal["session.list.response"] = "session.list.response"
    sessions: list[dict[str, Any]] = Field(default_factory=list)


class PongFrame(Frame):
    """Server keepalive response."""

    type: Literal["pong"] = "pong"


# ---------------------------------------------------------------------------
# Frame routing
# ---------------------------------------------------------------------------

# Map of type string → model class for inbound (client → server) frames
INBOUND_FRAMES: dict[str, type[Frame]] = {
    "connect": ConnectFrame,
    "chat.send": ChatSendFrame,
    "chat.history": ChatHistoryFrame,
    "tool.invoke": ToolInvokeFrame,
    "config.get": ConfigGetFrame,
    "config.set": ConfigSetFrame,
    "session.list": SessionListFrame,
    "ping": PingFrame,
}


def parse_inbound(raw: dict[str, Any]) -> Frame:
    """Parse and validate an inbound frame dict.

    Raises ValueError if the frame type is unknown or validation fails.
    """
    frame_type = raw.get("type")
    if not frame_type or frame_type not in INBOUND_FRAMES:
        raise ValueError(f"Unknown frame type: {frame_type!r}")
    try:
        return INBOUND_FRAMES[frame_type].model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Frame validation failed for {frame_type!r}: {e}") from e
