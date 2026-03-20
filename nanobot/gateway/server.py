"""WebSocket gateway server.

Adds WS routes to the existing Starlette application.
All existing HTTP routes remain untouched.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from loguru import logger
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from nanobot.gateway.auth import GatewayAuth, _compare_token
from nanobot.gateway.protocol import (
    ChatHistoryFrame,
    ChatHistoryResponseFrame,
    ChatResponseFrame,
    ChatSendFrame,
    ChatTokenFrame,
    ConfigGetFrame,
    ConfigSetFrame,
    ConfigValueFrame,
    ConnectFrame,
    ErrorFrame,
    HelloOkFrame,
    PingFrame,
    PongFrame,
    SessionListFrame,
    SessionListResponseFrame,
    ToolCallFrame,
    ToolInvokeFrame,
    ToolResultFrame,
    parse_inbound,
)
from nanobot.nodes.protocol import (
    NodeCommandResultFrame,
    NodeInvokeFrame,
    NodeListFrame,
    NodePairApproveFrame,
    NodePairRequestFrame,
    NodeStatusFrame,
    parse_node_frame,
)


# Capability advertisement
METHODS = [
    "chat.send",
    "chat.history",
    "tool.invoke",
    "config.get",
    "config.set",
    "session.list",
    "ping",
    # Node management (operator)
    "node.pair.approve",
    "node.invoke",
    "node.list",
    # Node inbound (device nodes)
    "node.pair.request",
    "node.status",
    "node.command.result",
]
EVENTS = [
    "chat.token",
    "chat.response",
    "tool.call",
    "tool.result",
    "error",
    "pong",
    # Node events
    "node.connected",
    "node.disconnected",
    "node.pair.pending",
    "node.list.response",
    "node.pair.response",
    "node.command.invoke",
    "node.command.result",
]


class GatewayConnection:
    """Represents a single authenticated WS connection."""

    def __init__(self, ws: WebSocket, conn_id: str, role: str = "operator"):
        self.ws = ws
        self.id = conn_id
        self.role = role
        self.authenticated = False

    async def send_frame(self, frame) -> None:
        """Send a protocol frame as JSON."""
        await self.ws.send_json(frame.model_dump(mode="json"))


class GatewayServer:
    """Manages WebSocket connections and frame dispatch.

    Usage in api_server.py lifespan::

        gateway = GatewayServer(agent=agent_loop, auth=auth)
        # Add gateway.routes to the Starlette app routes list
    """

    def __init__(self, agent=None, auth: GatewayAuth | None = None, config=None, harness=None):
        self.agent = agent  # AgentLoop instance
        self.harness = harness  # DeepAgentHarness instance (optional)
        self.auth = auth or GatewayAuth()
        self.config = config
        self._connections: dict[str, GatewayConnection] = {}

        # Lazy import to avoid circular dependency
        from nanobot.nodes.manager import NodeManager
        self.node_manager = NodeManager(self)

    @property
    def routes(self) -> list[WebSocketRoute]:
        """Starlette WebSocket routes to add to the app."""
        return [
            WebSocketRoute("/ws", self._ws_endpoint),
        ]

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    # ------------------------------------------------------------------
    # WebSocket endpoint
    # ------------------------------------------------------------------

    async def _ws_endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        conn_id = uuid.uuid4().hex[:12]
        conn = GatewayConnection(websocket, conn_id)
        self._connections[conn_id] = conn
        logger.info(f"WS connection {conn_id} opened")

        try:
            await self._connection_loop(conn)
        except WebSocketDisconnect:
            logger.info(f"WS connection {conn_id} closed")
        except Exception as e:
            logger.error(f"WS connection {conn_id} error: {e}")
        finally:
            self._connections.pop(conn_id, None)
            self.node_manager.on_disconnect(conn_id)

    async def _connection_loop(self, conn: GatewayConnection) -> None:
        """Read frames in a loop, dispatch to handlers."""
        while True:
            raw_text = await conn.ws.receive_text()
            try:
                raw = json.loads(raw_text)
            except json.JSONDecodeError:
                await conn.send_frame(ErrorFrame(
                    code="invalid_json",
                    message="Could not parse JSON",
                ))
                continue

            try:
                frame = parse_inbound(raw)
            except ValueError:
                # Try node-specific frames as fallback
                frame = parse_node_frame(raw)
                if frame is None:
                    await conn.send_frame(ErrorFrame(
                        code="invalid_frame",
                        message=f"Unknown frame type: {raw.get('type', '?')}",
                        ref_id=raw.get("id"),
                    ))
                    continue

            await self._dispatch(conn, frame)

    # ------------------------------------------------------------------
    # Frame dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, conn: GatewayConnection, frame) -> None:
        """Route a validated frame to its handler."""

        # connect must be first frame (or auth disabled)
        if isinstance(frame, ConnectFrame):
            await self._handle_connect(conn, frame)
            return

        # All other frames require authentication (if enabled)
        if self.auth.enabled and not conn.authenticated:
            await conn.send_frame(ErrorFrame(
                id=frame.id,
                code="auth_required",
                message="Send a connect frame with auth token first",
                ref_id=frame.id,
            ))
            return

        handlers = {
            ChatSendFrame: self._handle_chat_send,
            ChatHistoryFrame: self._handle_chat_history,
            ToolInvokeFrame: self._handle_tool_invoke,
            ConfigGetFrame: self._handle_config_get,
            ConfigSetFrame: self._handle_config_set,
            SessionListFrame: self._handle_session_list,
            PingFrame: self._handle_ping,
            # Node frames from device nodes
            NodePairRequestFrame: self.node_manager.handle_pair_request,
            NodeStatusFrame: self.node_manager.handle_status,
            NodeCommandResultFrame: self.node_manager.handle_command_result,
            # Node frames from operators
            NodePairApproveFrame: self.node_manager.handle_pair_approve,
            NodeInvokeFrame: self.node_manager.handle_node_invoke,
            NodeListFrame: self.node_manager.handle_node_list,
        }

        handler = handlers.get(type(frame))
        if handler:
            try:
                await handler(conn, frame)
            except Exception as e:
                logger.error(f"Handler error for {frame.type}: {e}")
                await conn.send_frame(ErrorFrame(
                    id=frame.id,
                    code="internal_error",
                    message=str(e),
                    ref_id=frame.id,
                ))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_connect(self, conn: GatewayConnection, frame: ConnectFrame) -> None:
        if self.auth.enabled:
            if not self.auth.validate(frame.auth):
                await conn.send_frame(ErrorFrame(
                    id=frame.id,
                    code="auth_failed",
                    message="Invalid authentication token",
                    ref_id=frame.id,
                ))
                return

        conn.authenticated = True
        conn.role = frame.role.value
        logger.info(f"WS {conn.id} authenticated as {conn.role}")

        # Build capability lists including harness info
        methods = list(METHODS)
        events = list(EVENTS)
        from nanobot import api_server
        harness = getattr(api_server, '_harness', None)
        if harness and harness.is_initialized:
            methods.append("harness.status")
            events.append("harness.ready")

        await conn.send_frame(HelloOkFrame(
            id=frame.id,
            methods=methods,
            events=events,
        ))

    async def _handle_chat_send(self, conn: GatewayConnection, frame: ChatSendFrame) -> None:
        # Check if either agent or harness is available
        from nanobot import api_server
        harness = getattr(api_server, '_harness', None)
        has_engine = self.agent is not None or (harness is not None and harness.is_initialized)

        if not has_engine:
            await conn.send_frame(ErrorFrame(
                id=frame.id,
                code="no_agent",
                message="Agent not initialized",
                ref_id=frame.id,
            ))
            return

        if frame.stream:
            await self._handle_chat_stream(conn, frame)
        else:
            await self._handle_chat_blocking(conn, frame)

    async def _handle_chat_stream(self, conn: GatewayConnection, frame: ChatSendFrame) -> None:
        """Process chat with streaming — uses deepagents harness if available."""
        try:
            # Prefer deepagents harness if available
            harness = self.harness
            if harness is None:
                # Try to get it from api_server global
                from nanobot import api_server
                harness = getattr(api_server, '_harness', None)

            if harness is not None and harness.is_initialized:
                messages = [{"role": "user", "content": frame.content}]
                full_response = ""
                async for chunk in harness.run_stream(
                    agent_name="ceo",
                    messages=messages,
                    session_id=frame.session_key,
                ):
                    # Extract content from SSE chunk
                    delta = ""
                    if isinstance(chunk, dict):
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        full_response += delta
                        await conn.send_frame(ChatTokenFrame(
                            id=frame.id,
                            session_key=frame.session_key,
                            token=delta,
                        ))

                await conn.send_frame(ChatResponseFrame(
                    id=frame.id,
                    session_key=frame.session_key,
                    content=full_response,
                    model="deepagents",
                ))
                return

            # Fallback to direct agent loop
            response = await self.agent.process_direct(
                content=frame.content,
                session_key=frame.session_key,
                channel="gateway",
                chat_id=conn.id,
            )

            await conn.send_frame(ChatResponseFrame(
                id=frame.id,
                session_key=frame.session_key,
                content=response or "",
                model=self.agent.model if self.agent else None,
            ))
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            await conn.send_frame(ErrorFrame(
                id=frame.id,
                code="chat_error",
                message=str(e),
                ref_id=frame.id,
            ))

    async def _handle_chat_blocking(self, conn: GatewayConnection, frame: ChatSendFrame) -> None:
        """Process chat and return complete response — uses deepagents harness if available."""
        try:
            # Prefer deepagents harness
            harness = self.harness
            if harness is None:
                from nanobot import api_server
                harness = getattr(api_server, '_harness', None)

            if harness is not None and harness.is_initialized:
                messages = [{"role": "user", "content": frame.content}]
                result = await harness.run(
                    agent_name="ceo",
                    messages=messages,
                    session_id=frame.session_key,
                )
                content = result.get("content", "") if isinstance(result, dict) else str(result)
                await conn.send_frame(ChatResponseFrame(
                    id=frame.id,
                    session_key=frame.session_key,
                    content=content,
                    model="deepagents",
                ))
                return

            # Fallback
            response = await self.agent.process_direct(
                content=frame.content,
                session_key=frame.session_key,
                channel="gateway",
                chat_id=conn.id,
            )

            await conn.send_frame(ChatResponseFrame(
                id=frame.id,
                session_key=frame.session_key,
                content=response or "",
                model=self.agent.model if self.agent else None,
            ))
        except Exception as e:
            await conn.send_frame(ErrorFrame(
                id=frame.id,
                code="chat_error",
                message=str(e),
                ref_id=frame.id,
            ))

    async def _handle_chat_history(self, conn: GatewayConnection, frame: ChatHistoryFrame) -> None:
        if not self.agent:
            await conn.send_frame(ErrorFrame(
                id=frame.id,
                code="no_agent",
                message="Agent not initialized",
                ref_id=frame.id,
            ))
            return

        messages = []
        if hasattr(self.agent, "session_manager") and self.agent.session_manager:
            try:
                history = self.agent.session_manager.load(frame.session_key)
                messages = history[-frame.limit:] if history else []
            except Exception:
                pass

        await conn.send_frame(ChatHistoryResponseFrame(
            id=frame.id,
            session_key=frame.session_key,
            messages=messages,
        ))

    async def _handle_tool_invoke(self, conn: GatewayConnection, frame: ToolInvokeFrame) -> None:
        if not self.agent:
            await conn.send_frame(ErrorFrame(
                id=frame.id,
                code="no_agent",
                message="Agent not initialized",
                ref_id=frame.id,
            ))
            return

        result = await self.agent.tools.execute(frame.tool, frame.params)

        await conn.send_frame(ToolResultFrame(
            id=frame.id,
            tool=frame.tool,
            result=result,
            success=not result.startswith("Error"),
        ))

    async def _handle_config_get(self, conn: GatewayConnection, frame: ConfigGetFrame) -> None:
        safe_config: dict[str, Any] = {}
        if self.config:
            # Expose safe config subset (no secrets)
            safe_config = {
                "agents": {
                    "defaults": {
                        "model": self.config.agents.defaults.model,
                        "max_tokens": self.config.agents.defaults.max_tokens,
                        "temperature": self.config.agents.defaults.temperature,
                        "max_tool_iterations": self.config.agents.defaults.max_tool_iterations,
                        "memory_window": self.config.agents.defaults.memory_window,
                    }
                },
                "gateway": {
                    "host": self.config.gateway.host,
                    "port": self.config.gateway.port,
                },
            }

        # Include harness/orchestrator info
        from nanobot import api_server
        harness = getattr(api_server, '_harness', None)
        orchestrator = getattr(api_server, '_orchestrator', None)
        if harness and harness.is_initialized:
            safe_config["harness"] = {
                "engine": "deepagents",
                "agents": harness.agent_count,
                "universal_memory": True,
            }
        if orchestrator:
            try:
                safe_config["orchestrator"] = {
                    "agents": len(orchestrator.registry.agent_names),
                    "teams": list(orchestrator.registry.get_teams()),
                }
            except Exception:
                pass

        # Include enabled channels
        if self.config and hasattr(self.config, 'channels'):
            channels = {}
            for name in ('email', 'slack', 'whatsapp'):
                ch = getattr(self.config.channels, name, None)
                if ch and getattr(ch, 'enabled', False):
                    channels[name] = {"enabled": True}
            if channels:
                safe_config["channels"] = channels

        # Include MCP servers
        if self.config and hasattr(self.config, 'tools'):
            mcp = getattr(self.config.tools, 'mcp_servers', None)
            if mcp:
                safe_config.setdefault("tools", {})["mcpServers"] = {
                    name: True for name in mcp
                }

        await conn.send_frame(ConfigValueFrame(
            id=frame.id,
            config=safe_config,
        ))

    async def _handle_config_set(self, conn: GatewayConnection, frame: ConfigSetFrame) -> None:
        # Only allow safe runtime config changes
        allowed_paths = {
            "agents.defaults.model",
            "agents.defaults.temperature",
            "agents.defaults.max_tokens",
            "agents.defaults.max_tool_iterations",
            "agents.defaults.memory_window",
        }

        if frame.path not in allowed_paths:
            await conn.send_frame(ErrorFrame(
                id=frame.id,
                code="config_denied",
                message=f"Cannot set '{frame.path}' at runtime",
                ref_id=frame.id,
            ))
            return

        # Apply to live config
        if self.config:
            parts = frame.path.split(".")
            obj = self.config
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                setattr(obj, parts[-1], frame.value)
                # Also update the agent if applicable
                if self.agent and parts[-1] in ("model", "temperature", "max_tokens"):
                    setattr(self.agent, parts[-1], frame.value)

        await conn.send_frame(ConfigValueFrame(
            id=frame.id,
            config={frame.path: frame.value},
        ))

    async def _handle_session_list(self, conn: GatewayConnection, frame: SessionListFrame) -> None:
        sessions: list[dict[str, Any]] = []

        # Pull from agent loop's session manager
        if self.agent and hasattr(self.agent, "session_manager") and self.agent.session_manager:
            try:
                session_keys = self.agent.session_manager.list_sessions()
                sessions = [{"key": k} for k in session_keys]
            except Exception:
                pass

        # Also pull from deepagents harness memory
        from nanobot import api_server
        harness = getattr(api_server, '_harness', None)
        if harness and harness.is_initialized and hasattr(harness, 'memory'):
            try:
                recent = harness.memory._get_recent_sessions(limit=20, agent_name=None)
                for s in recent:
                    key = s.get("session_id", s.get("id", ""))
                    if key and not any(existing.get("key") == key for existing in sessions):
                        sessions.append({
                            "key": key,
                            "agent": s.get("agent", ""),
                            "summary": s.get("summary", ""),
                            "source": "harness",
                        })
            except Exception:
                pass

        await conn.send_frame(SessionListResponseFrame(
            id=frame.id,
            sessions=sessions,
        ))

    async def _handle_ping(self, conn: GatewayConnection, frame: PingFrame) -> None:
        await conn.send_frame(PongFrame(id=frame.id))

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    async def broadcast(self, frame, role: str | None = None) -> None:
        """Send a frame to all connected clients (optionally filtered by role)."""
        for conn in list(self._connections.values()):
            if role and conn.role != role:
                continue
            try:
                await conn.send_frame(frame)
            except Exception:
                pass
