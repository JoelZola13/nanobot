"""Pydantic models for device node WS frames.

Extends the gateway protocol with node-specific frames for pairing,
capability advertisement, command invocation, and status reporting.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nanobot.gateway.protocol import Frame


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeCapability(str, Enum):
    """Standard capabilities a node can advertise."""

    camera = "camera"
    screen = "screen"
    location = "location"
    system = "system"
    voice = "voice"
    clipboard = "clipboard"
    notifications = "notifications"


class PairingState(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


# ---------------------------------------------------------------------------
# Node info (sent in pairing / status frames)
# ---------------------------------------------------------------------------

class NodeInfo(BaseModel):
    """Metadata describing a connected device node."""

    model_config = ConfigDict(extra="forbid")

    device_id: str  # Stable identifier for the physical device
    name: str = ""  # Human-friendly label (e.g. "Joel's iPhone")
    platform: str = ""  # "ios", "android", "macos", "linux", "windows"
    capabilities: list[NodeCapability] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Client (node) → Server frames
# ---------------------------------------------------------------------------

class NodePairRequestFrame(Frame):
    """Node requests pairing with the gateway."""

    type: Literal["node.pair.request"] = "node.pair.request"
    node: NodeInfo


class NodeStatusFrame(Frame):
    """Node reports its current status / capabilities."""

    type: Literal["node.status"] = "node.status"
    online: bool = True
    battery: float | None = None  # 0.0 – 1.0
    capabilities: list[NodeCapability] = Field(default_factory=list)


class NodeCommandResultFrame(Frame):
    """Node returns the result of a command invocation."""

    type: Literal["node.command.result"] = "node.command.result"
    command: str  # The command that was invoked (e.g. "camera.capture")
    success: bool = True
    result: Any = None  # Command-specific payload (base64 image, text, JSON)
    error: str | None = None
    ref_id: str | None = None  # Correlation to the invoke frame


# ---------------------------------------------------------------------------
# Server → Node frames
# ---------------------------------------------------------------------------

class NodePairResponseFrame(Frame):
    """Server responds to a pairing request."""

    type: Literal["node.pair.response"] = "node.pair.response"
    state: PairingState
    message: str = ""


class NodeCommandInvokeFrame(Frame):
    """Server asks a node to execute a command."""

    type: Literal["node.command.invoke"] = "node.command.invoke"
    command: str  # Dot-namespaced command (e.g. "camera.capture", "screen.capture")
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 30000


# ---------------------------------------------------------------------------
# Server → Operator frames (notifications about nodes)
# ---------------------------------------------------------------------------

class NodeConnectedFrame(Frame):
    """Notification that a node connected."""

    type: Literal["node.connected"] = "node.connected"
    node: NodeInfo


class NodeDisconnectedFrame(Frame):
    """Notification that a node disconnected."""

    type: Literal["node.disconnected"] = "node.disconnected"
    device_id: str


class NodePairPendingFrame(Frame):
    """Notification that a node is requesting pairing approval."""

    type: Literal["node.pair.pending"] = "node.pair.pending"
    node: NodeInfo


class NodeListResponseFrame(Frame):
    """Response to node.list request."""

    type: Literal["node.list.response"] = "node.list.response"
    nodes: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Operator → Server frames (node management)
# ---------------------------------------------------------------------------

class NodePairApproveFrame(Frame):
    """Operator approves or rejects a pending pairing request."""

    type: Literal["node.pair.approve"] = "node.pair.approve"
    device_id: str
    approve: bool = True


class NodeInvokeFrame(Frame):
    """Operator (or agent) requests a command on a specific node."""

    type: Literal["node.invoke"] = "node.invoke"
    device_id: str  # Target node device ID (or "*" for first available)
    command: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 30000


class NodeListFrame(Frame):
    """Request list of connected nodes."""

    type: Literal["node.list"] = "node.list"


# ---------------------------------------------------------------------------
# Frame routing
# ---------------------------------------------------------------------------

# Frames that can be received from nodes
NODE_INBOUND_FRAMES: dict[str, type[Frame]] = {
    "node.pair.request": NodePairRequestFrame,
    "node.status": NodeStatusFrame,
    "node.command.result": NodeCommandResultFrame,
}

# Frames that can be received from operators (node management)
NODE_OPERATOR_FRAMES: dict[str, type[Frame]] = {
    "node.pair.approve": NodePairApproveFrame,
    "node.invoke": NodeInvokeFrame,
    "node.list": NodeListFrame,
}

# All node-related inbound frames (for gateway integration)
ALL_NODE_INBOUND: dict[str, type[Frame]] = {**NODE_INBOUND_FRAMES, **NODE_OPERATOR_FRAMES}


def parse_node_frame(raw: dict[str, Any]) -> Frame | None:
    """Parse a node-related inbound frame, or return None if not a node frame."""
    frame_type = raw.get("type", "")
    cls = ALL_NODE_INBOUND.get(frame_type)
    if cls is None:
        return None
    try:
        return cls.model_validate(raw)
    except ValidationError:
        return None
