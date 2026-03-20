"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from a chat channel."""
    
    channel: str  # telegram, discord, slack, whatsapp
    sender_id: str  # User identifier
    chat_id: str  # Chat/channel identifier
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    
    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Cross-service events (for Redis event bus) ──────────────────

@dataclass
class SocialMessageEvent:
    """New message posted in SV Social."""
    channel_id: str
    channel_name: str
    author_id: str
    author_name: str
    content: str
    message_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "social.message.new",
            "channelId": self.channel_id,
            "channelName": self.channel_name,
            "authorId": self.author_id,
            "authorName": self.author_name,
            "content": self.content,
            "messageId": self.message_id,
        }


@dataclass
class PresenceEvent:
    """User came online/offline/away on SV Social."""
    user_id: str
    username: str
    status: str  # "online" | "away" | "offline"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "social.user.status",
            "userId": self.user_id,
            "username": self.username,
            "status": self.status,
        }


@dataclass
class AgentTaskCompleteEvent:
    """Agent finished processing a task."""
    agent_name: str
    task_summary: str
    channel: str = "librechat"  # originating channel
    session_key: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "agent.task.complete",
            "agentName": self.agent_name,
            "taskSummary": self.task_summary,
            "channel": self.channel,
            "sessionKey": self.session_key,
        }


