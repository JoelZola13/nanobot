/**
 * Paperclip ↔ Nanobot Heartbeat Relay
 *
 * Receives heartbeat webhooks from Paperclip (HTTP adapter),
 * translates them into nanobot /v1/chat/completions calls,
 * and reports results + cost events back to Paperclip.
 *
 * Port: 3050
 */

import { createServer, IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Config ──────────────────────────────────────────────────────────
const RELAY_PORT = 3050;
const NANOBOT_API = "http://127.0.0.1:18790";
const PAPERCLIP_API = "http://127.0.0.1:3100/api";

// ── Agent Mapping ───────────────────────────────────────────────────
interface AgentMapping {
  [paperclipId: string]: {
    nanobotName: string;
    team: string;
    role: string;
  };
}

function loadMapping(): AgentMapping {
  const mapPath = resolve(__dirname, "agent-mapping.json");
  if (!existsSync(mapPath)) {
    console.warn("[relay] agent-mapping.json not found — run seed script first");
    return {};
  }
  return JSON.parse(readFileSync(mapPath, "utf-8"));
}

let agentMapping = loadMapping();

// ── Helpers ─────────────────────────────────────────────────────────
function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString()));
    req.on("error", reject);
  });
}

function json(res: ServerResponse, status: number, data: unknown) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

// ── Paperclip API helpers ───────────────────────────────────────────
async function paperclipGet(path: string): Promise<any> {
  const resp = await fetch(`${PAPERCLIP_API}${path}`);
  if (!resp.ok) throw new Error(`Paperclip GET ${path}: ${resp.status}`);
  return resp.json();
}

async function paperclipPatch(
  path: string,
  body: any,
  runId?: string
): Promise<any> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (runId) headers["X-Paperclip-Run-Id"] = runId;

  const resp = await fetch(`${PAPERCLIP_API}${path}`, {
    method: "PATCH",
    headers,
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`Paperclip PATCH ${path}: ${resp.status}`);
  return resp.json();
}

async function paperclipPost(
  path: string,
  body: any,
  runId?: string
): Promise<any> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (runId) headers["X-Paperclip-Run-Id"] = runId;

  const resp = await fetch(`${PAPERCLIP_API}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Paperclip POST ${path}: ${resp.status} — ${text}`);
  }
  return resp.json();
}

// ── Nanobot call ────────────────────────────────────────────────────
interface NanobotResponse {
  content: string;
  agent: string;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

async function callNanobot(
  agentName: string,
  message: string
): Promise<NanobotResponse> {
  const body = {
    model: "openai-codex/gpt-5.4",
    messages: [
      {
        role: "user",
        content: `@${agentName} ${message}`,
      },
    ],
    stream: false,
  };

  const resp = await fetch(`${NANOBOT_API}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Nanobot API error: ${resp.status} — ${text}`);
  }

  const data = await resp.json();
  const choice = data.choices?.[0];
  const meta = data.nanobot_metadata || {};

  return {
    content: choice?.message?.content || "(no response)",
    agent: meta.responding_agent || agentName,
    usage: data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
}

// ── Report cost to Paperclip ────────────────────────────────────────
async function reportCost(
  companyId: string,
  agentId: string,
  issueId: string | undefined,
  usage: NanobotResponse["usage"]
) {
  // Rough cost estimate: $0.003/1K input, $0.015/1K output (codex pricing)
  const inputCost = (usage.prompt_tokens / 1000) * 0.3; // cents
  const outputCost = (usage.completion_tokens / 1000) * 1.5; // cents
  const costCents = Math.round(inputCost + outputCost);

  try {
    await paperclipPost(`/companies/${companyId}/cost-events`, {
      agentId,
      issueId: issueId || undefined,
      provider: "openai-codex",
      model: "gpt-5.4",
      inputTokens: usage.prompt_tokens,
      outputTokens: usage.completion_tokens,
      costCents,
      occurredAt: new Date().toISOString(),
    });
    console.log(`[relay] cost reported: ${costCents}¢ for agent ${agentId}`);
  } catch (err) {
    console.error("[relay] cost reporting failed:", err);
  }
}

// ── Heartbeat handler ───────────────────────────────────────────────
async function handleHeartbeat(
  res: ServerResponse,
  payload: any
): Promise<void> {
  const { agentId, runId, context } = payload;
  const mapping = agentMapping[agentId];

  if (!mapping) {
    console.warn(`[relay] unknown agent ID: ${agentId}`);
    json(res, 404, { error: `Agent ${agentId} not found in mapping` });
    return;
  }

  console.log(
    `[relay] heartbeat for ${mapping.nanobotName} (${mapping.team}) — run ${runId}`
  );

  try {
    // 1. Get agent info to find companyId
    const agentInfo = await paperclipGet(`/agents/${agentId}`);
    const companyId = agentInfo.companyId;

    // 2. Check for assigned tasks
    const issues = await paperclipGet(
      `/companies/${companyId}/issues?assigneeAgentId=${agentId}&status=todo,in_progress,blocked`
    );

    if (!issues || issues.length === 0) {
      console.log(`[relay] no tasks for ${mapping.nanobotName}, idle heartbeat`);
      json(res, 200, { status: "idle", message: "No tasks assigned" });
      return;
    }

    // 3. Pick highest priority task (in_progress first, then todo)
    const sorted = issues.sort((a: any, b: any) => {
      const statusOrder: Record<string, number> = {
        in_progress: 0,
        todo: 1,
        blocked: 2,
      };
      return (statusOrder[a.status] ?? 9) - (statusOrder[b.status] ?? 9);
    });

    const task = sorted[0];
    console.log(
      `[relay] working on: "${task.title}" (${task.id}) — status: ${task.status}`
    );

    // 4. Checkout if todo
    if (task.status === "todo") {
      try {
        await paperclipPost(
          `/issues/${task.id}/checkout`,
          { agentId, expectedStatuses: ["todo"] },
          runId
        );
      } catch (err: any) {
        if (err.message?.includes("409")) {
          console.log(`[relay] task already checked out, skipping`);
          json(res, 200, { status: "skipped", message: "Task already claimed" });
          return;
        }
        throw err;
      }
    }

    // 5. Build task message for nanobot
    const taskMessage = [
      `Task: ${task.title}`,
      task.description ? `Description: ${task.description}` : "",
      `Priority: ${task.priority || "medium"}`,
      `Status: ${task.status}`,
    ]
      .filter(Boolean)
      .join("\n");

    // 6. Call nanobot agent
    const result = await callNanobot(mapping.nanobotName, taskMessage);

    // 7. Update task with result
    await paperclipPatch(
      `/issues/${task.id}`,
      {
        status: "done",
        comment: result.content.substring(0, 4000), // Paperclip comment limit
      },
      runId
    );

    // 8. Report cost
    await reportCost(companyId, agentId, task.id, result.usage);

    console.log(
      `[relay] completed: "${task.title}" by ${result.agent}`
    );

    json(res, 200, {
      status: "completed",
      task: task.title,
      agent: result.agent,
      tokens: result.usage.total_tokens,
    });
  } catch (err: any) {
    console.error(`[relay] heartbeat error:`, err);
    json(res, 500, { error: err.message || "Heartbeat processing failed" });
  }
}

// ── HTTP Server ─────────────────────────────────────────────────────
const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://localhost:${RELAY_PORT}`);

  // Health check
  if (req.method === "GET" && url.pathname === "/health") {
    json(res, 200, {
      status: "ok",
      service: "paperclip-relay",
      agents: Object.keys(agentMapping).length,
    });
    return;
  }

  // Reload mapping
  if (req.method === "POST" && url.pathname === "/reload") {
    agentMapping = loadMapping();
    json(res, 200, {
      status: "reloaded",
      agents: Object.keys(agentMapping).length,
    });
    return;
  }

  // Heartbeat webhook from Paperclip
  if (req.method === "POST" && url.pathname === "/heartbeat") {
    try {
      const body = JSON.parse(await readBody(req));
      await handleHeartbeat(res, body);
    } catch (err: any) {
      console.error("[relay] parse error:", err);
      json(res, 400, { error: "Invalid request body" });
    }
    return;
  }

  // Agent list
  if (req.method === "GET" && url.pathname === "/agents") {
    json(res, 200, agentMapping);
    return;
  }

  json(res, 404, { error: "Not found" });
});

server.listen(RELAY_PORT, "127.0.0.1", () => {
  console.log(`[relay] Paperclip heartbeat relay listening on http://127.0.0.1:${RELAY_PORT}`);
  console.log(`[relay] ${Object.keys(agentMapping).length} agents mapped`);
  console.log(`[relay] Nanobot API: ${NANOBOT_API}`);
  console.log(`[relay] Paperclip API: ${PAPERCLIP_API}`);
});
