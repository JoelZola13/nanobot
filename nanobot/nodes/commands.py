"""Standard device node command definitions.

Each command has a name, description, parameter schema, and the set of
capabilities it requires.  The NodeManager uses these definitions to
validate invocations before forwarding them to nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nanobot.nodes.protocol import NodeCapability


@dataclass(frozen=True)
class NodeCommand:
    """Defines a single command that can be sent to a device node."""

    name: str  # Dot-namespaced, e.g. "camera.capture"
    description: str
    requires: list[NodeCapability]  # Node must have ALL of these
    params_schema: dict[str, Any] = field(default_factory=dict)  # JSON Schema


# ---------------------------------------------------------------------------
# Standard command catalogue
# ---------------------------------------------------------------------------

COMMANDS: dict[str, NodeCommand] = {}


def _register(cmd: NodeCommand) -> NodeCommand:
    COMMANDS[cmd.name] = cmd
    return cmd


# Camera commands
_register(NodeCommand(
    name="camera.capture",
    description="Capture a photo from the device camera.",
    requires=[NodeCapability.camera],
    params_schema={
        "type": "object",
        "properties": {
            "camera": {
                "type": "string",
                "description": "Which camera to use.",
                "enum": ["front", "back"],
                "default": "back",
            },
            "quality": {
                "type": "integer",
                "description": "JPEG quality 1-100.",
                "minimum": 1,
                "maximum": 100,
                "default": 80,
            },
        },
    },
))

# Screen commands
_register(NodeCommand(
    name="screen.capture",
    description="Capture a screenshot of the device screen.",
    requires=[NodeCapability.screen],
    params_schema={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["png", "jpeg"],
                "default": "png",
            },
        },
    },
))

# Location commands
_register(NodeCommand(
    name="location.get",
    description="Get the device's current GPS location.",
    requires=[NodeCapability.location],
    params_schema={
        "type": "object",
        "properties": {
            "high_accuracy": {
                "type": "boolean",
                "description": "Request high-accuracy GPS fix.",
                "default": False,
            },
        },
    },
))

# System commands
_register(NodeCommand(
    name="system.run",
    description="Run a shell command on the device.",
    requires=[NodeCapability.system],
    params_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds.",
                "default": 30,
            },
        },
        "required": ["command"],
    },
))

_register(NodeCommand(
    name="system.info",
    description="Get device system information (OS, memory, battery, etc.).",
    requires=[NodeCapability.system],
    params_schema={"type": "object", "properties": {}},
))

# Voice commands
_register(NodeCommand(
    name="voice.start",
    description="Start recording audio from the device microphone.",
    requires=[NodeCapability.voice],
    params_schema={
        "type": "object",
        "properties": {
            "max_duration_seconds": {
                "type": "integer",
                "description": "Maximum recording duration.",
                "default": 60,
                "minimum": 1,
                "maximum": 300,
            },
        },
    },
))

_register(NodeCommand(
    name="voice.stop",
    description="Stop audio recording and return the audio data.",
    requires=[NodeCapability.voice],
    params_schema={"type": "object", "properties": {}},
))

_register(NodeCommand(
    name="voice.play",
    description="Play audio on the device speaker.",
    requires=[NodeCapability.voice],
    params_schema={
        "type": "object",
        "properties": {
            "audio_base64": {
                "type": "string",
                "description": "Base64-encoded audio data.",
            },
            "format": {
                "type": "string",
                "enum": ["wav", "mp3", "ogg"],
                "default": "wav",
            },
        },
        "required": ["audio_base64"],
    },
))

# Clipboard commands
_register(NodeCommand(
    name="clipboard.get",
    description="Read the device clipboard contents.",
    requires=[NodeCapability.clipboard],
    params_schema={"type": "object", "properties": {}},
))

_register(NodeCommand(
    name="clipboard.set",
    description="Write text to the device clipboard.",
    requires=[NodeCapability.clipboard],
    params_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to copy to clipboard.",
            },
        },
        "required": ["text"],
    },
))

# Notification commands
_register(NodeCommand(
    name="notifications.send",
    description="Show a local notification on the device.",
    requires=[NodeCapability.notifications],
    params_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["title", "body"],
    },
))


def get_command(name: str) -> NodeCommand | None:
    """Look up a command by name."""
    return COMMANDS.get(name)


def list_commands() -> list[NodeCommand]:
    """Return all registered commands."""
    return list(COMMANDS.values())


def commands_for_capabilities(caps: list[NodeCapability]) -> list[NodeCommand]:
    """Return commands that a node with *caps* can execute."""
    cap_set = set(caps)
    return [cmd for cmd in COMMANDS.values() if set(cmd.requires) <= cap_set]
