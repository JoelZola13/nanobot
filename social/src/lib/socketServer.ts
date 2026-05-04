/**
 * Get the Socket.IO broadcast interface for API route handlers.
 *
 * Next.js webpack bundles have a different `globalThis` AND `process`
 * than server.mjs, so we can't access the Socket.IO instance directly.
 * Instead, we send an HTTP POST to server.mjs in the same container, which has
 * a /broadcast endpoint that forwards the emit to the real Socket.IO.
 */
type IOInstance = {
  to: (room: string) => { emit: (event: string, data: unknown) => void };
};

const SOCKET_BRIDGE_PORT = process.env.SOCKET_BRIDGE_PORT || process.env.PORT || "3182";

const ioBridge: IOInstance = {
  to: (room: string) => ({
    emit: (event: string, data: unknown) => {
      // Fire-and-forget HTTP POST to server.mjs inside this container.
      fetch(`http://127.0.0.1:${SOCKET_BRIDGE_PORT}/broadcast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room, event, data }),
      }).catch((err) => {
        console.error("[socketServer] broadcast failed:", err.message);
      });
    },
  }),
};

export function getIO(): IOInstance {
  return ioBridge;
}
