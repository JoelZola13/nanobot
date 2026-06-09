"""Node manager — tracks connected device nodes, handles pairing, routes commands."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.nodes.commands import commands_for_capabilities, get_command
from nanobot.nodes.protocol import (
    NodeCapability,
    NodeCommandInvokeFrame,
    NodeCommandResultFrame,
    NodeConnectedFrame,
    NodeDisconnectedFrame,
    NodeInfo,
    NodeListResponseFrame,
    NodePairPendingFrame,
    NodePairRequestFrame,
    NodePairResponseFrame,
    NodeStatusFrame,
    PairingState,
)

if TYPE_CHECKING:
    from nanobot.gateway.server import GatewayConnection, GatewayServer


# ---------------------------------------------------------------------------
# Connected node record
# ---------------------------------------------------------------------------

@dataclass
class ConnectedNode:
    """A paired and connected device node."""

    info: NodeInfo
    conn_id: str  # GatewayConnection.id — used to look up the live WS
    online: bool = True
    battery: float | None = None
    _capabilities: list[NodeCapability] = field(default_factory=list)

    @property
    def device_id(self) -> str:
        return self.info.device_id

    @property
    def capabilities(self) -> list[NodeCapability]:
        return self._capabilities or self.info.capabilities

    def update_status(self, status: NodeStatusFrame) -> None:
        self.online = status.online
        self.battery = status.battery
        if status.capabilities:
            self._capabilities = list(status.capabilities)


# ---------------------------------------------------------------------------
# Pending command (waiting for result from node)
# ---------------------------------------------------------------------------

@dataclass
class _PendingCommand:
    ref_id: str
    future: asyncio.Future[NodeCommandResultFrame]
    device_id: str
    command: str


# ---------------------------------------------------------------------------
# NodeManager
# ---------------------------------------------------------------------------

class NodeManager:
    """Manages device node connections, pairing, and command routing.

    Designed to be owned by :class:`GatewayServer`.  The gateway calls into
    ``handle_*`` methods when it receives node-related frames.
    """

    def __init__(self, gateway: GatewayServer) -> None:
        self.gateway = gateway

        # Paired & connected nodes keyed by device_id
        self._nodes: dict[str, ConnectedNode] = {}

        # Pairing requests awaiting operator approval (device_id → NodeInfo)
        self._pending_pairs: dict[str, NodeInfo] = {}

        # Auto-approve pairing (for development / single-user setups)
        self.auto_approve_pairing: bool = True

        # Outstanding command invocations keyed by ref_id
        self._pending_cmds: dict[str, _PendingCommand] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_nodes(self) -> list[dict[str, Any]]:
        """Return serialisable info for all connected nodes."""
        return [
            {
                "device_id": n.device_id,
                "name": n.info.name,
                "platform": n.info.platform,
                "capabilities": [c.value for c in n.capabilities],
                "online": n.online,
                "battery": n.battery,
            }
            for n in self._nodes.values()
        ]

    def get_node(self, device_id: str) -> ConnectedNode | None:
        return self._nodes.get(device_id)

    def find_node_for_command(self, command: str, device_id: str = "*") -> ConnectedNode | None:
        """Find a node capable of running *command*.

        If *device_id* is ``"*"`` pick the first online node that supports the
        command's required capabilities.
        """
        cmd_def = get_command(command)
        if cmd_def is None:
            return None

        required = set(cmd_def.requires)

        if device_id != "*":
            node = self._nodes.get(device_id)
            if node and node.online and required <= set(node.capabilities):
                return node
            return None

        for node in self._nodes.values():
            if node.online and required <= set(node.capabilities):
                return node
        return None

    # ------------------------------------------------------------------
    # Command invocation
    # ------------------------------------------------------------------

    async def invoke_command(
        self,
        device_id: str,
        command: str,
        params: dict[str, Any] | None = None,
        timeout_ms: int = 30_000,
    ) -> NodeCommandResultFrame:
        """Send a command to a node and wait for the result.

        Raises ``TimeoutError`` if the node doesn't respond in time, or
        ``ValueError`` if no suitable node is found.
        """
        node = self.find_node_for_command(command, device_id)
        if node is None:
            raise ValueError(
                f"No online node available for command {command!r}"
                + (f" (device_id={device_id!r})" if device_id != "*" else "")
            )

        conn = self.gateway._connections.get(node.conn_id)
        if conn is None:
            raise ValueError(f"Node {node.device_id!r} connection lost")

        ref_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[NodeCommandResultFrame] = loop.create_future()

        pending = _PendingCommand(
            ref_id=ref_id,
            future=future,
            device_id=node.device_id,
            command=command,
        )
        self._pending_cmds[ref_id] = pending

        # Send the invoke frame to the node
        invoke_frame = NodeCommandInvokeFrame(
            id=ref_id,
            command=command,
            params=params or {},
            timeout_ms=timeout_ms,
        )
        await conn.send_frame(invoke_frame)
        logger.debug(f"Sent command {command!r} to node {node.device_id!r} (ref={ref_id})")

        try:
            result = await asyncio.wait_for(future, timeout=timeout_ms / 1000)
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Node {node.device_id!r} did not respond to {command!r} within {timeout_ms}ms"
            )
        finally:
            self._pending_cmds.pop(ref_id, None)

    # ------------------------------------------------------------------
    # Frame handlers (called by GatewayServer._dispatch)
    # ------------------------------------------------------------------

    async def handle_pair_request(self, conn: GatewayConnection, frame: NodePairRequestFrame) -> None:
        """Node is requesting to pair."""
        info = frame.node
        device_id = info.device_id
        logger.info(f"Pairing request from {info.name!r} ({device_id}) platform={info.platform}")

        if self.auto_approve_pairing:
            await self._approve_node(conn, info)
            return

        # Queue for manual approval
        self._pending_pairs[device_id] = info

        # Notify operators
        await self.gateway.broadcast(
            NodePairPendingFrame(node=info),
            role="operator",
        )
        logger.info(f"Node {device_id} awaiting operator approval")

    async def handle_pair_approve(self, conn: GatewayConnection, frame: Any) -> None:
        """Operator approves or rejects a pending pair."""
        device_id = frame.device_id
        info = self._pending_pairs.pop(device_id, None)

        if info is None:
            from nanobot.gateway.protocol import ErrorFrame
            await conn.send_frame(ErrorFrame(
                code="not_found",
                message=f"No pending pairing for device {device_id!r}",
                ref_id=frame.id,
            ))
            return

        # Find the node's connection
        node_conn = self._find_conn_for_pending(device_id)
        if node_conn is None:
            from nanobot.gateway.protocol import ErrorFrame
            await conn.send_frame(ErrorFrame(
                code="not_found",
                message=f"Node {device_id!r} is no longer connected",
                ref_id=frame.id,
            ))
            return

        if frame.approve:
            await self._approve_node(node_conn, info)
        else:
            await node_conn.send_frame(NodePairResponseFrame(
                state=PairingState.rejected,
                message="Pairing rejected by operator",
            ))
            logger.info(f"Node {device_id} pairing rejected")

    async def handle_status(self, conn: GatewayConnection, frame: NodeStatusFrame) -> None:
        """Node reports its status."""
        node = self._node_by_conn(conn.id)
        if node is None:
            return
        node.update_status(frame)
        logger.debug(f"Node {node.device_id} status: online={node.online} battery={node.battery}")

    async def handle_command_result(self, conn: GatewayConnection, frame: NodeCommandResultFrame) -> None:
        """Node returns a command result."""
        ref_id = frame.ref_id or frame.id
        if not ref_id:
            logger.warning("Received command result with no ref_id — dropping")
            return

        pending = self._pending_cmds.get(ref_id)
        if pending is None:
            logger.warning(f"Received command result for unknown ref_id={ref_id!r} — dropping")
            return

        if not pending.future.done():
            pending.future.set_result(frame)

    async def handle_node_invoke(self, conn: GatewayConnection, frame: Any) -> None:
        """Operator (or agent) requests a command on a node.

        This is an async bridge — we invoke the command and relay the result
        back to the requesting operator connection.
        """
        try:
            result = await self.invoke_command(
                device_id=frame.device_id,
                command=frame.command,
                params=frame.params,
                timeout_ms=frame.timeout_ms,
            )
            # Forward the result with the operator's original frame id
            result_copy = result.model_copy(update={"ref_id": frame.id})
            await conn.send_frame(result_copy)
        except (ValueError, TimeoutError) as e:
            from nanobot.gateway.protocol import ErrorFrame
            await conn.send_frame(ErrorFrame(
                code="node_error",
                message=str(e),
                ref_id=frame.id,
            ))

    async def handle_node_list(self, conn: GatewayConnection, frame: Any) -> None:
        """Operator requests list of connected nodes."""
        await conn.send_frame(NodeListResponseFrame(
            id=frame.id,
            nodes=self.list_nodes(),
        ))

    # ------------------------------------------------------------------
    # Connection lifecycle (called by GatewayServer)
    # ------------------------------------------------------------------

    def on_disconnect(self, conn_id: str) -> None:
        """Called when a WS connection drops.  Cleans up any node state."""
        node = self._node_by_conn(conn_id)
        if node is None:
            return

        logger.info(f"Node {node.device_id!r} disconnected")
        device_id = node.device_id
        self._nodes.pop(device_id, None)

        # Cancel any pending commands for this node
        for ref_id, pending in list(self._pending_cmds.items()):
            if pending.device_id == device_id and not pending.future.done():
                pending.future.set_exception(
                    ConnectionError(f"Node {device_id!r} disconnected")
                )
                self._pending_cmds.pop(ref_id, None)

        # Notify operators (fire-and-forget, don't block disconnect cleanup)
        asyncio.ensure_future(
            self.gateway.broadcast(
                NodeDisconnectedFrame(device_id=device_id),
                role="operator",
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _approve_node(self, conn: GatewayConnection, info: NodeInfo) -> None:
        """Complete the pairing handshake and register the node."""
        device_id = info.device_id

        # Send approval to node
        await conn.send_frame(NodePairResponseFrame(
            state=PairingState.approved,
            message="Pairing approved",
        ))

        # Register
        self._nodes[device_id] = ConnectedNode(
            info=info,
            conn_id=conn.id,
        )
        logger.info(
            f"Node paired: {info.name!r} ({device_id}) "
            f"caps={[c.value for c in info.capabilities]}"
        )

        # Notify operators
        await self.gateway.broadcast(
            NodeConnectedFrame(node=info),
            role="operator",
        )

    def _node_by_conn(self, conn_id: str) -> ConnectedNode | None:
        """Find a node by its gateway connection ID."""
        for node in self._nodes.values():
            if node.conn_id == conn_id:
                return node
        return None

    def _find_conn_for_pending(self, device_id: str) -> GatewayConnection | None:
        """Find the GatewayConnection for a node that sent a pair request."""
        for conn in self.gateway._connections.values():
            if conn.role == "node":
                # Best-effort: match by checking if this connection has no
                # paired node yet (it's the one waiting for approval).
                # In practice, device_id is sent in the pair request frame,
                # so we can match by checking pending pairs.
                existing = self._node_by_conn(conn.id)
                if existing is None:
                    return conn
        return None
