"""SV Social tools: read/write messages, channels, presence via direct PostgreSQL access."""

import json
from datetime import datetime, timezone
from typing import Any

import asyncpg

from nanobot.agent.tools.base import Tool


class SocialReadMessagesTool(Tool):
    """Read recent messages from an SV Social channel."""

    name = "social_read_messages"
    description = (
        "Read recent messages from an SV Social channel. "
        "Specify a channel name (e.g. 'general', 'development') or channel ID. "
        "Returns author, content, and timestamp for each message."
    )
    parameters = {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Channel name or ID to read from",
            },
            "count": {
                "type": "integer",
                "description": "Number of recent messages to fetch (default 20, max 50)",
                "minimum": 1,
                "maximum": 50,
            },
            "search": {
                "type": "string",
                "description": "Optional text to filter messages by content",
            },
        },
        "required": ["channel"],
    }

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def execute(self, channel: str, count: int = 20, search: str = "", **kwargs) -> str:
        count = min(max(1, count), 50)
        async with self._pool.acquire() as conn:
            # Resolve channel by name or ID
            ch = await conn.fetchrow(
                "SELECT id, name, type FROM channels WHERE id = $1 OR slug = $2 OR name ILIKE $3 LIMIT 1",
                channel, channel, channel,
            )
            if not ch:
                return f"Channel '{channel}' not found. Use social_list_channels to see available channels."

            if search:
                rows = await conn.fetch(
                    """SELECT m.content, m.created_at, u.display_name, u.username, u.is_agent
                       FROM messages m JOIN users u ON m.author_id = u.id
                       WHERE m.channel_id = $1 AND m.deleted_at IS NULL
                         AND m.content ILIKE $2
                       ORDER BY m.created_at DESC LIMIT $3""",
                    ch["id"], f"%{search}%", count,
                )
            else:
                rows = await conn.fetch(
                    """SELECT m.content, m.created_at, u.display_name, u.username, u.is_agent
                       FROM messages m JOIN users u ON m.author_id = u.id
                       WHERE m.channel_id = $1 AND m.deleted_at IS NULL
                       ORDER BY m.created_at DESC LIMIT $2""",
                    ch["id"], count,
                )

            if not rows:
                return f"No messages found in #{ch['name'] or channel}."

            lines = [f"Messages in #{ch['name'] or channel} ({ch['type']}):\n"]
            for r in reversed(rows):  # Oldest first
                ts = r["created_at"].strftime("%Y-%m-%d %H:%M")
                agent_tag = " [agent]" if r["is_agent"] else ""
                lines.append(f"[{ts}] {r['display_name']}{agent_tag}: {r['content'][:500]}")

            return "\n".join(lines)


class SocialSendMessageTool(Tool):
    """Send a message to an SV Social channel as an agent user."""

    name = "social_send_message"
    description = (
        "Send a message to an SV Social channel. "
        "Only use when the user explicitly asks you to send a message on Social. "
        "The message will appear in real-time for all channel members."
    )
    parameters = {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Channel name or ID to send to",
            },
            "content": {
                "type": "string",
                "description": "Message text to send",
            },
            "agent_username": {
                "type": "string",
                "description": "Agent username to send as (default: 'ceo'). Must be an agent user.",
            },
        },
        "required": ["channel", "content"],
    }

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def execute(self, channel: str, content: str, agent_username: str = "ceo", **kwargs) -> str:
        async with self._pool.acquire() as conn:
            # Resolve channel
            ch = await conn.fetchrow(
                "SELECT id, name FROM channels WHERE id = $1 OR slug = $2 OR name ILIKE $3 LIMIT 1",
                channel, channel, channel,
            )
            if not ch:
                return f"Channel '{channel}' not found."

            # Resolve agent user
            agent = await conn.fetchrow(
                "SELECT id, display_name FROM users WHERE username = $1 AND is_agent = true",
                agent_username,
            )
            if not agent:
                return f"Agent user '{agent_username}' not found. Must be an agent user (is_agent=true)."

            # Insert message
            msg_id = await conn.fetchval(
                """INSERT INTO messages (id, channel_id, author_id, content, created_at, updated_at)
                   VALUES (gen_random_uuid()::text, $1, $2, $3, NOW(), NOW())
                   RETURNING id""",
                ch["id"], agent["id"], content,
            )

            # Notify via PG NOTIFY for real-time broadcast
            notify_payload = json.dumps({
                "id": msg_id,
                "channelId": ch["id"],
                "authorId": agent["id"],
                "authorName": agent["display_name"],
                "content": content,
            })
            await conn.execute("SELECT pg_notify('social_messages', $1)", notify_payload)

            return f"Message sent to #{ch['name']} as {agent['display_name']}: {content[:100]}"


class SocialListChannelsTool(Tool):
    """List all SV Social channels."""

    name = "social_list_channels"
    description = (
        "List all channels on SV Social with member counts and last activity. "
        "Useful for discovering which channels exist before reading messages."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def execute(self, **kwargs) -> str:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT c.id, c.name, c.slug, c.type, c.description,
                          COUNT(DISTINCT cm.user_id) AS member_count,
                          MAX(m.created_at) AS last_message_at
                   FROM channels c
                   LEFT JOIN channel_members cm ON cm.channel_id = c.id
                   LEFT JOIN messages m ON m.channel_id = c.id AND m.deleted_at IS NULL
                   WHERE c.is_archived = false
                   GROUP BY c.id
                   ORDER BY last_message_at DESC NULLS LAST"""
            )

            if not rows:
                return "No channels found."

            lines = ["SV Social Channels:\n"]
            for r in rows:
                name = r["name"] or r["slug"] or r["id"]
                last = r["last_message_at"].strftime("%Y-%m-%d %H:%M") if r["last_message_at"] else "no messages"
                desc = f" — {r['description'][:60]}" if r["description"] else ""
                lines.append(f"  #{name} ({r['type']}, {r['member_count']} members, last: {last}){desc}")

            return "\n".join(lines)


class SocialWhoOnlineTool(Tool):
    """Check who is currently online on SV Social."""

    name = "social_who_online"
    description = (
        "Check who is currently online on SV Social. "
        "Returns users who are online, idle, or away (active in last 5 minutes)."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def execute(self, **kwargs) -> str:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT display_name, username, status, is_agent, last_seen_at
                   FROM users
                   WHERE status != 'offline'
                      OR last_seen_at > NOW() - INTERVAL '5 minutes'
                   ORDER BY is_agent ASC, last_seen_at DESC NULLS LAST"""
            )

            if not rows:
                return "No users are currently online on SV Social."

            humans = []
            agents = []
            for r in rows:
                seen = ""
                if r["last_seen_at"]:
                    seen = f", last seen {r['last_seen_at'].strftime('%H:%M')}"
                entry = f"  {r['display_name']} (@{r['username']}) — {r['status']}{seen}"
                if r["is_agent"]:
                    agents.append(entry)
                else:
                    humans.append(entry)

            parts = ["Online on SV Social:\n"]
            if humans:
                parts.append("People:")
                parts.extend(humans)
            if agents:
                parts.append("\nAgents:")
                parts.extend(agents)

            return "\n".join(parts)


class SocialSearchTool(Tool):
    """Search across all SV Social messages."""

    name = "social_search"
    description = (
        "Search across all SV Social messages by keyword. "
        "Returns matching messages with channel, author, and timestamp."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for in messages",
            },
            "channel": {
                "type": "string",
                "description": "Optional: limit search to a specific channel name or ID",
            },
            "count": {
                "type": "integer",
                "description": "Max results to return (default 10, max 30)",
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": ["query"],
    }

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def execute(self, query: str, channel: str = "", count: int = 10, **kwargs) -> str:
        count = min(max(1, count), 30)
        async with self._pool.acquire() as conn:
            if channel:
                ch = await conn.fetchrow(
                    "SELECT id, name FROM channels WHERE id = $1 OR slug = $2 OR name ILIKE $3 LIMIT 1",
                    channel, channel, channel,
                )
                if not ch:
                    return f"Channel '{channel}' not found."
                rows = await conn.fetch(
                    """SELECT m.content, m.created_at, u.display_name, c.name AS channel_name
                       FROM messages m
                       JOIN users u ON m.author_id = u.id
                       JOIN channels c ON m.channel_id = c.id
                       WHERE m.channel_id = $1 AND m.deleted_at IS NULL
                         AND m.content ILIKE $2
                       ORDER BY m.created_at DESC LIMIT $3""",
                    ch["id"], f"%{query}%", count,
                )
            else:
                rows = await conn.fetch(
                    """SELECT m.content, m.created_at, u.display_name, c.name AS channel_name
                       FROM messages m
                       JOIN users u ON m.author_id = u.id
                       JOIN channels c ON m.channel_id = c.id
                       WHERE m.deleted_at IS NULL AND m.content ILIKE $1
                       ORDER BY m.created_at DESC LIMIT $2""",
                    f"%{query}%", count,
                )

            if not rows:
                return f"No messages matching '{query}'."

            lines = [f"Search results for '{query}' ({len(rows)} matches):\n"]
            for r in rows:
                ts = r["created_at"].strftime("%Y-%m-%d %H:%M")
                lines.append(f"[{ts}] #{r['channel_name']} — {r['display_name']}: {r['content'][:300]}")

            return "\n".join(lines)


class SocialUserProfileTool(Tool):
    """Get a user's profile from SV Social."""

    name = "social_user_profile"
    description = (
        "Get a user's profile from SV Social including bio, status, and connections. "
        "Search by username or display name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "description": "Username or display name to look up",
            },
        },
        "required": ["username"],
    }

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def execute(self, username: str, **kwargs) -> str:
        async with self._pool.acquire() as conn:
            user = await conn.fetchrow(
                """SELECT id, username, display_name, email, bio, location, website,
                          status, is_agent, agent_model, last_seen_at, created_at
                   FROM users
                   WHERE username = $1 OR display_name ILIKE $2
                   LIMIT 1""",
                username, f"%{username}%",
            )
            if not user:
                return f"User '{username}' not found on SV Social."

            # Get connection count
            conn_count = await conn.fetchval(
                """SELECT COUNT(*) FROM connections
                   WHERE (requester_id = $1 OR addressee_id = $1) AND status = 'ACCEPTED'""",
                user["id"],
            )

            # Get message count
            msg_count = await conn.fetchval(
                "SELECT COUNT(*) FROM messages WHERE author_id = $1 AND deleted_at IS NULL",
                user["id"],
            )

            # Get channel memberships
            channels = await conn.fetch(
                """SELECT c.name, c.type FROM channels c
                   JOIN channel_members cm ON cm.channel_id = c.id
                   WHERE cm.user_id = $1 AND c.is_archived = false
                   ORDER BY c.name""",
                user["id"],
            )

            lines = [f"Profile: {user['display_name']} (@{user['username']})"]
            if user["is_agent"]:
                lines[0] += f" [Agent: {user['agent_model'] or 'unknown'}]"
            lines.append(f"  Status: {user['status']}")
            if user["bio"]:
                lines.append(f"  Bio: {user['bio']}")
            if user["location"]:
                lines.append(f"  Location: {user['location']}")
            if user["website"]:
                lines.append(f"  Website: {user['website']}")
            lines.append(f"  Connections: {conn_count}")
            lines.append(f"  Messages sent: {msg_count}")
            if user["last_seen_at"]:
                lines.append(f"  Last seen: {user['last_seen_at'].strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"  Joined: {user['created_at'].strftime('%Y-%m-%d')}")

            if channels:
                ch_names = [f"#{c['name']}" for c in channels if c["name"]]
                lines.append(f"  Channels: {', '.join(ch_names)}")

            return "\n".join(lines)


# Convenience: all tool classes for registration
ALL_SOCIAL_TOOLS = [
    SocialReadMessagesTool,
    SocialSendMessageTool,
    SocialListChannelsTool,
    SocialWhoOnlineTool,
    SocialSearchTool,
    SocialUserProfileTool,
]
