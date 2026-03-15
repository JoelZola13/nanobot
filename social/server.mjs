import { createServer } from "http";
import { parse } from "url";
import next from "next";
import { Server as SocketIO } from "socket.io";

const dev = process.env.NODE_ENV !== "production";
const port = parseInt(process.env.PORT || "3182", 10);
const app = next({ dev });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  const httpServer = createServer((req, res) => {
    const parsedUrl = parse(req.url, true);
    handle(req, res, parsedUrl);
  });

  const io = new SocketIO(httpServer, {
    path: "/api/socketio",
    addTrailingSlash: false,
    cors: { origin: "*" },
  });

  // Track online users: userId -> Set<socketId>
  const onlineUsers = new Map();
  // Track presence status: userId -> "online" | "away"
  const userStatus = new Map();
  // Track last heartbeat: userId -> timestamp
  const lastHeartbeat = new Map();
  // Track userId -> socketId mapping for call signaling
  const userSockets = new Map();

  // Auto-away after 60s of no heartbeat
  setInterval(() => {
    const now = Date.now();
    for (const [uid, lastBeat] of lastHeartbeat.entries()) {
      if (now - lastBeat > 60000 && userStatus.get(uid) === "online") {
        userStatus.set(uid, "away");
        io.emit("presence:update", { userId: uid, status: "away" });
      }
    }
  }, 15000);

  io.on("connection", (socket) => {
    const userId = socket.handshake.auth?.userId;
    if (!userId) return socket.disconnect();

    // Track online status
    if (!onlineUsers.has(userId)) {
      onlineUsers.set(userId, new Set());
    }
    onlineUsers.get(userId).add(socket.id);
    userSockets.set(userId, socket.id);
    userStatus.set(userId, "online");
    lastHeartbeat.set(userId, Date.now());

    // Broadcast online status
    io.emit("presence:update", { userId, status: "online" });

    // Send current online users to newly connected client
    socket.emit("presence:list", Array.from(userStatus.entries()).map(([id, status]) => ({ userId: id, status })));

    // Join user's channels
    socket.on("join:channels", (channelIds) => {
      if (Array.isArray(channelIds)) {
        channelIds.forEach((id) => socket.join(`channel:${id}`));
      }
    });

    // Join a specific channel room
    socket.on("join:channel", (channelId) => {
      socket.join(`channel:${channelId}`);
    });

    // Leave a channel room
    socket.on("leave:channel", (channelId) => {
      socket.leave(`channel:${channelId}`);
    });

    // Heartbeat for presence
    socket.on("heartbeat", () => {
      lastHeartbeat.set(userId, Date.now());
      if (userStatus.get(userId) !== "online") {
        userStatus.set(userId, "online");
        io.emit("presence:update", { userId, status: "online" });
      }
    });

    // Focus/blur for away detection
    socket.on("presence:active", () => {
      lastHeartbeat.set(userId, Date.now());
      if (userStatus.get(userId) !== "online") {
        userStatus.set(userId, "online");
        io.emit("presence:update", { userId, status: "online" });
      }
    });

    socket.on("presence:away", () => {
      userStatus.set(userId, "away");
      io.emit("presence:update", { userId, status: "away" });
    });

    // New message — broadcast to channel members
    socket.on("message:send", (data) => {
      socket.to(`channel:${data.channelId}`).emit("message:new", data);
    });

    // Message edit — broadcast to channel
    socket.on("message:edit", (data) => {
      socket.to(`channel:${data.channelId}`).emit("message:edit", data);
    });

    // Message delete — broadcast to channel
    socket.on("message:delete", (data) => {
      socket.to(`channel:${data.channelId}`).emit("message:delete", data);
    });

    // Reaction update — broadcast to channel
    socket.on("reaction:update", (data) => {
      socket.to(`channel:${data.channelId}`).emit("reaction:update", data);
    });

    // Pin update — broadcast to channel
    socket.on("message:pin", (data) => {
      socket.to(`channel:${data.channelId}`).emit("message:pin", data);
    });

    // Typing indicator
    socket.on("typing:start", ({ channelId, user }) => {
      socket.to(`channel:${channelId}`).emit("typing:start", { channelId, user });
    });

    socket.on("typing:stop", ({ channelId, userId: uid }) => {
      socket.to(`channel:${channelId}`).emit("typing:stop", { channelId, userId: uid });
    });

    // ── WebRTC Call Signaling ──────────────────────────────────────
    socket.on("call:initiate", ({ targetUserId, callType, channelId }) => {
      const targetSocketId = userSockets.get(targetUserId);
      if (targetSocketId) {
        io.to(targetSocketId).emit("call:incoming", {
          callerId: userId,
          callerName: socket.handshake.auth?.userName || "Unknown",
          callType, // "audio" | "video"
          channelId,
        });
      }
    });

    socket.on("call:offer", ({ targetUserId, offer }) => {
      const targetSocketId = userSockets.get(targetUserId);
      if (targetSocketId) {
        io.to(targetSocketId).emit("call:offer", { callerId: userId, offer });
      }
    });

    socket.on("call:answer", ({ targetUserId, answer }) => {
      const targetSocketId = userSockets.get(targetUserId);
      if (targetSocketId) {
        io.to(targetSocketId).emit("call:answer", { callerId: userId, answer });
      }
    });

    socket.on("call:ice-candidate", ({ targetUserId, candidate }) => {
      const targetSocketId = userSockets.get(targetUserId);
      if (targetSocketId) {
        io.to(targetSocketId).emit("call:ice-candidate", { callerId: userId, candidate });
      }
    });

    socket.on("call:end", ({ targetUserId }) => {
      const targetSocketId = userSockets.get(targetUserId);
      if (targetSocketId) {
        io.to(targetSocketId).emit("call:ended", { callerId: userId });
      }
    });

    socket.on("call:reject", ({ targetUserId }) => {
      const targetSocketId = userSockets.get(targetUserId);
      if (targetSocketId) {
        io.to(targetSocketId).emit("call:rejected", { callerId: userId });
      }
    });

    // Disconnect
    socket.on("disconnect", () => {
      if (onlineUsers.has(userId)) {
        onlineUsers.get(userId).delete(socket.id);
        if (onlineUsers.get(userId).size === 0) {
          onlineUsers.delete(userId);
          userStatus.delete(userId);
          lastHeartbeat.delete(userId);
          userSockets.delete(userId);
          io.emit("presence:update", { userId, status: "offline" });
        }
      }
    });
  });

  // Make io accessible for API routes if needed
  globalThis.__socketio = io;

  httpServer.listen(port, () => {
    console.log(`> Street Voices Social ready on http://localhost:${port}`);
    console.log(`> Socket.IO listening on /api/socketio`);
  });
});
