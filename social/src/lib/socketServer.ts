/**
 * Get the Socket.IO broadcast interface for API route handlers.
 *
 * Next.js webpack bundles have a different `globalThis` AND `process`
 * than server.mjs, so we can't access the Socket.IO instance directly.
 * Instead, we send an HTTP POST to the wsServer (port 3183) which has
 * a /broadcast endpoint that forwards the emit to the real Socket.IO.
 */
type IOInstance = {
  to: (room: string) => { emit: (event: string, data: unknown) => void };
};

const WS_PORT = process.env.WS_PORT || "3183";

const ioBridge: IOInstance = {
  to: (room: string) => ({
    emit: (event: string, data: unknown) => {
      // Fire-and-forget HTTP POST to the ws server's broadcast endpoint
      // Use 127.0.0.1 not localhost — the Docker entrypoint remaps localhost
      // to the host machine, but wsServer runs inside this container
      fetch(`http://127.0.0.1:${WS_PORT}/broadcast`, {
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
