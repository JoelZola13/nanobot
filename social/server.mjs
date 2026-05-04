import { createServer } from "http";
import { parse } from "url";
import next from "next";
import { Server as SocketIO } from "socket.io";
import pg from "pg";
import Redis from "ioredis";

const dev = process.env.NODE_ENV !== "production";
const port = parseInt(process.env.PORT || "3182", 10);
const app = next({ dev });
const handle = app.getRequestHandler();

function isLocalRequest(req) {
  return ["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(
    req.socket.remoteAddress || "",
  );
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error("Payload too large"));
        req.destroy();
      }
    });
    req.on("end", () => resolve(JSON.parse(body || "{}")));
    req.on("error", reject);
  });
}

async function handleBroadcast(req, res, io) {
  if (req.method !== "POST") {
    res.statusCode = 405;
    res.end("Method Not Allowed");
    return;
  }

  if (!isLocalRequest(req)) {
    res.statusCode = 403;
    res.end("Forbidden");
    return;
  }

  if (!io) {
    res.statusCode = 503;
    res.end("Socket server not ready");
    return;
  }

  try {
    const { room, event, data } = await readJsonBody(req);
    if (typeof room !== "string" || typeof event !== "string") {
      res.statusCode = 400;
      res.end("Invalid broadcast payload");
      return;
    }

    io.to(room).emit(event, data);
    res.statusCode = 204;
    res.end();
  } catch (err) {
    res.statusCode = err.message === "Payload too large" ? 413 : 400;
    res.end(err.message || "Invalid request");
  }
}

app.prepare().then(() => {
  let io;
  const httpServer = createServer((req, res) => {
    const parsedUrl = parse(req.url, true);
    if (parsedUrl.pathname === "/broadcast") {
      handleBroadcast(req, res, io);
      return;
    }
    handle(req, res, parsedUrl);
  });

  io = new SocketIO(httpServer, {
    path: "/ws-social",
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

    // New message — broadcast to channel members + publish to Redis
    socket.on("message:send", (data) => {
      socket.to(`channel:${data.channelId}`).emit("message:new", data);
      // Publish to Redis event bus for cross-service awareness
      if (redisPub) {
        redisPub.publish("social.message.new", JSON.stringify({
          type: "social.message.new",
          channelId: data.channelId,
          channelName: data.channelName || "",
          authorId: data.author?.id || userId,
          authorName: data.author?.displayName || "Unknown",
          content: (data.content || "").slice(0, 200),
          messageId: data.id || "",
          timestamp: new Date().toISOString(),
        })).catch(() => {});
      }
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

  // ── Redis event bus for cross-service communication ─────────────
  const redisUrl = process.env.REDIS_URL || "redis://localhost:6380";
  const redisPub = new Redis(redisUrl);  // For publishing events
  const redisSub = new Redis(redisUrl);  // For subscribing to events

  redisPub.on("connect", () => console.log("[Redis] Publisher connected"));
  redisSub.on("connect", () => console.log("[Redis] Subscriber connected"));
  redisPub.on("error", (err) => console.error("[Redis] Pub error:", err.message));
  redisSub.on("error", (err) => console.error("[Redis] Sub error:", err.message));

  // Subscribe to agent task completion events
  redisSub.subscribe("agent.task.complete", (err) => {
    if (err) console.error("[Redis] Subscribe error:", err.message);
    else console.log("[Redis] Subscribed to agent.task.complete");
  });

  redisSub.on("message", (channel, message) => {
    try {
      const data = JSON.parse(message);
      if (channel === "agent.task.complete") {
        const notification = {
          type: "agent.task.complete",
          title: `Agent ${data.agentName || "Agent"} completed a task`,
          body: data.taskSummary || "",
          sourceService: "nanobot",
          createdAt: data.timestamp || new Date().toISOString(),
        };
        // Broadcast to all connected clients
        io.emit("notification:new", notification);
        // Persist to database for all online human users
        for (const uid of onlineUsers.keys()) {
          pgPool.query(
            `INSERT INTO notifications (id, user_id, type, title, body, source_service, created_at)
             VALUES (gen_random_uuid()::text, $1, $2, $3, $4, $5, NOW())`,
            [uid, notification.type, notification.title, notification.body, notification.sourceService]
          ).catch((err) => console.error("[Notification] DB write error:", err.message));
        }
        console.log(`[Redis] Broadcast agent task complete: ${data.taskSummary?.slice(0, 60)}`);
      }
    } catch (err) {
      console.error("[Redis] Message parse error:", err.message);
    }
  });

  // ── PostgreSQL connections ──────────────────────────────────────
  const pgConnStr = process.env.DATABASE_URL || "postgresql://social:social_password@social-postgres:5432/social";
  // Pool for writes (notifications, etc.)
  const pgPool = new pg.Pool({ connectionString: pgConnStr, max: 3 });
  pgPool.on("error", (err) => console.error("[PG-Pool] Error:", err.message));

  // ── PG NOTIFY listener for agent-sent messages ─────────────────
  // When nanobot agents insert messages via social_send_message tool,
  // they fire pg_notify('social_messages', payload). We listen here
  // and broadcast to the correct Socket.IO channel room.
  const pgClient = new pg.Client({ connectionString: pgConnStr });
  pgClient.connect()
    .then(() => {
      pgClient.query("LISTEN social_messages");
      console.log("[PG-NOTIFY] Listening on social_messages channel");
      pgClient.on("notification", (msg) => {
        try {
          const data = JSON.parse(msg.payload);
          // Broadcast to the channel room so all connected clients see the message
          io.to(`channel:${data.channelId}`).emit("message:new", {
            id: data.id,
            channelId: data.channelId,
            author: {
              id: data.authorId,
              displayName: data.authorName,
            },
            content: data.content,
            createdAt: new Date().toISOString(),
          });
          console.log(`[PG-NOTIFY] Broadcasted agent message to channel:${data.channelId}`);
        } catch (err) {
          console.error("[PG-NOTIFY] Error parsing notification:", err.message);
        }
      });
    })
    .catch((err) => {
      console.error("[PG-NOTIFY] Failed to connect:", err.message);
    });

  httpServer.listen(port, () => {
    console.log(`> Street Voices Social ready on http://localhost:${port}`);
    console.log(`> Socket.IO listening on /ws-social`);
  });
});
