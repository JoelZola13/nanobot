/**
 * Paperclip ↔ Nanobot Heartbeat Relay v2
 *
 * Full integration:
 * - Issue checkout/release lifecycle
 * - Cost event reporting per agent run
 * - Progress comments on issues during execution
 * - Heartbeat run events for real-time dashboard updates
 * - On-demand wakeup support
 * - Idle heartbeat with status check
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
const COMPANY_ID = "78940514-fbb0-4c2d-8cee-09bcfd5399a4";
const PROJECT_ID = "645c5cd9-caa1-46c7-be50-da6e6001df14"; // "Nanobot" project with workspace

// Cost per 1K tokens (cents) — adjust for your model pricing
const INPUT_COST_PER_1K = 0.3; // ¢
const OUTPUT_COST_PER_1K = 1.5; // ¢

// ── Agent Mapping ───────────────────────────────────────────────────
interface AgentMapEntry {
  nanobotName: string;
  team: string;
  role: string;
}

interface AgentMapping {
  [paperclipId: string]: AgentMapEntry;
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

// ── HTTP Helpers ────────────────────────────────────────────────────
function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString()));
    req.on("error", reject);
  });
}

function jsonResp(res: ServerResponse, status: number, data: unknown) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

// ── Paperclip API Client ────────────────────────────────────────────
async function pcGet(path: string): Promise<any> {
  const resp = await fetch(`${PAPERCLIP_API}${path}`);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`GET ${path}: ${resp.status} — ${text}`);
  }
  return resp.json();
}

async function pcPatch(path: string, body: any): Promise<any> {
  const resp = await fetch(`${PAPERCLIP_API}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`PATCH ${path}: ${resp.status} — ${text}`);
  }
  return resp.json();
}

async function pcPost(path: string, body: any): Promise<any> {
  const resp = await fetch(`${PAPERCLIP_API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`POST ${path}: ${resp.status} — ${text}`);
  }
  return resp.json();
}

// ── Cost Reporting ──────────────────────────────────────────────────
interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

async function reportCost(
  agentId: string,
  issueId: string | null,
  usage: TokenUsage
): Promise<void> {
  const inputCost = (usage.prompt_tokens / 1000) * INPUT_COST_PER_1K;
  const outputCost = (usage.completion_tokens / 1000) * OUTPUT_COST_PER_1K;
  const costCents = Math.max(1, Math.round(inputCost + outputCost));

  try {
    await pcPost(`/companies/${COMPANY_ID}/cost-events`, {
      agentId,
      issueId: issueId || undefined,
      provider: "openai-codex",
      model: "gpt-5.4",
      inputTokens: usage.prompt_tokens,
      outputTokens: usage.completion_tokens,
      costCents,
      occurredAt: new Date().toISOString(),
    });
    console.log(
      `[relay] 💰 cost: ${costCents}¢ (${usage.total_tokens} tokens) for agent ${agentId.slice(0, 8)}`
    );
  } catch (err) {
    console.error("[relay] cost reporting failed:", err);
  }
}

// ── Issue Comment ───────────────────────────────────────────────────
async function addIssueComment(
  issueId: string,
  body: string
): Promise<void> {
  try {
    await pcPost(`/issues/${issueId}/comments`, { body });
  } catch (err) {
    console.error("[relay] comment failed:", err);
  }
}

// ── Nanobot Call ────────────────────────────────────────────────────
interface NanobotResult {
  content: string;
  agent: string;
  usage: TokenUsage;
}

async function callNanobot(
  agentName: string,
  message: string,
  timeout = 300_000
): Promise<NanobotResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const resp = await fetch(`${NANOBOT_API}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: `agent/${agentName}`,
        messages: [{ role: "user", content: message }],
        stream: false,
      }),
      signal: controller.signal,
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Nanobot ${resp.status}: ${text}`);
    }

    const data = await resp.json();
    const choice = data.choices?.[0];
    const meta = data.nanobot_metadata || {};

    return {
      content: choice?.message?.content || "(no response)",
      agent: meta.responding_agent || agentName,
      usage: data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
  } finally {
    clearTimeout(timer);
  }
}

// ── Issue Lifecycle ─────────────────────────────────────────────────
async function checkoutIssue(
  issueId: string,
  agentId: string
): Promise<boolean> {
  try {
    await pcPost(`/issues/${issueId}/checkout`, {
      agentId,
      expectedStatuses: ["todo", "backlog"],
    });
    return true;
  } catch (err: any) {
    if (err.message?.includes("409") || err.message?.includes("conflict")) {
      console.log(`[relay] issue ${issueId.slice(0, 8)} already checked out, skipping`);
      return false;
    }
    throw err;
  }
}

async function releaseIssue(issueId: string): Promise<void> {
  try {
    await pcPost(`/issues/${issueId}/release`, {});
  } catch (err) {
    console.error("[relay] release failed:", err);
  }
}

// ── Fetch Agent Issues ──────────────────────────────────────────────
interface PaperclipIssue {
  id: string;
  title: string;
  description: string | null;
  status: string;
  priority: string;
  projectId: string | null;
  goalId: string | null;
  labels: Array<{ name: string }>;
}

async function getAgentIssues(agentId: string): Promise<PaperclipIssue[]> {
  // Check for assigned issues in actionable states
  const issues: PaperclipIssue[] = await pcGet(
    `/companies/${COMPANY_ID}/issues?assigneeAgentId=${agentId}&status=todo,in_progress,backlog`
  );
  return issues;
}

// ── Heartbeat Handler ───────────────────────────────────────────────
async function handleHeartbeat(
  res: ServerResponse,
  payload: any
): Promise<void> {
  const { agentId, runId } = payload;
  const mapping = agentMapping[agentId];

  if (!mapping) {
    console.warn(`[relay] ❓ unknown agent: ${agentId}`);
    jsonResp(res, 404, { error: `Agent ${agentId} not found in mapping` });
    return;
  }

  const label = `${mapping.nanobotName} [${mapping.team}]`;
  console.log(`[relay] 💓 heartbeat: ${label} — run ${runId?.slice(0, 8) || "?"}`);

  try {
    // 1. Check for assigned issues
    const issues = await getAgentIssues(agentId);

    if (!issues || issues.length === 0) {
      jsonResp(res, 200, { status: "idle", message: "No tasks assigned" });
      return;
    }

    // 2. Pick best issue: in_progress first, then todo, then backlog
    const statusPriority: Record<string, number> = {
      in_progress: 0,
      todo: 1,
      backlog: 2,
    };
    const sorted = issues.sort(
      (a, b) => (statusPriority[a.status] ?? 9) - (statusPriority[b.status] ?? 9)
    );
    const issue = sorted[0];

    console.log(
      `[relay] 📋 task: "${issue.title}" (${issue.id.slice(0, 8)}) — ${issue.status}`
    );

    // 3. Checkout if not already in_progress
    if (issue.status !== "in_progress") {
      const claimed = await checkoutIssue(issue.id, agentId);
      if (!claimed) {
        // Still try to proceed — the issue may be locked by a stale run from this same agent
        console.log(`[relay] checkout conflict — proceeding anyway for ${label}`);
      }
    }

    // 4. Mark as in_progress (may fail if already in_progress, that's fine)
    try {
      await pcPatch(`/issues/${issue.id}`, { status: "in_progress" });
    } catch {
      // Already in_progress, continue
    }

    // 5. Post progress comment
    await addIssueComment(issue.id, `🤖 **${mapping.nanobotName}** is working on this...`);

    // 6. Build task prompt
    const parts = [
      `## Task: ${issue.title}`,
      issue.description || "",
      `Priority: ${issue.priority || "medium"}`,
      issue.labels?.length ? `Labels: ${issue.labels.map((l) => l.name).join(", ")}` : "",
    ].filter(Boolean);

    // 7. Execute via nanobot
    const startMs = Date.now();
    const result = await callNanobot(mapping.nanobotName, parts.join("\n"));
    const elapsedSec = ((Date.now() - startMs) / 1000).toFixed(1);

    console.log(
      `[relay] ✅ done: "${issue.title}" by ${result.agent} (${elapsedSec}s, ${result.usage.total_tokens} tokens)`
    );

    // 8. Post result as comment
    const resultComment = [
      `✅ **Completed by ${result.agent}** (${elapsedSec}s)`,
      "",
      result.content.length > 3500
        ? result.content.substring(0, 3500) + "\n\n_(truncated)_"
        : result.content,
      "",
      `_Tokens: ${result.usage.total_tokens} | Model: gpt-5.4_`,
    ].join("\n");
    await addIssueComment(issue.id, resultComment);

    // 9. Release the issue lock first, then mark done
    await releaseIssue(issue.id);

    // 10. Mark issue as done (after release so it's not locked)
    await pcPatch(`/issues/${issue.id}`, { status: "done" });

    // 11. Report cost
    await reportCost(agentId, issue.id, result.usage);

    jsonResp(res, 200, {
      status: "completed",
      issue: issue.title,
      issueId: issue.id,
      agent: result.agent,
      tokens: result.usage.total_tokens,
      elapsedSec: parseFloat(elapsedSec),
    });
  } catch (err: any) {
    console.error(`[relay] ❌ error for ${label}:`, err.message || err);
    jsonResp(res, 500, { error: err.message || "Heartbeat processing failed" });
  }
}

// ── Manual Task Dispatch (POST /dispatch) ───────────────────────────
// Create an issue, assign it, and immediately wake the agent
async function handleDispatch(
  res: ServerResponse,
  payload: any
): Promise<void> {
  const { agentName, title, description, priority } = payload;

  // Find the agent by nanobot name
  const entry = Object.entries(agentMapping).find(
    ([, v]) => v.nanobotName === agentName
  );
  if (!entry) {
    jsonResp(res, 404, { error: `Agent "${agentName}" not found` });
    return;
  }
  const [paperclipId, mapping] = entry;

  console.log(`[relay] 📬 dispatch: "${title}" → ${mapping.nanobotName}`);

  try {
    // 1. Create the issue (linked to project so workspace resolves)
    const issue = await pcPost(`/companies/${COMPANY_ID}/issues`, {
      title,
      description: description || "",
      priority: priority || "medium",
      assigneeAgentId: paperclipId,
      projectId: PROJECT_ID,
      status: "todo",
    });

    console.log(`[relay] created issue ${issue.identifier}: "${issue.title}"`);

    // 2. Paperclip auto-wakes the agent on assignment (with issue context).
    //    Do NOT call /wakeup manually — it creates a second run with empty
    //    context, which triggers the "No project workspace" STDERR warning.
    console.log(`[relay] ✅ ${mapping.nanobotName} will wake automatically via assignment trigger`);

    jsonResp(res, 201, {
      status: "dispatched",
      issue: {
        id: issue.id,
        identifier: issue.identifier,
        title: issue.title,
        assignee: mapping.nanobotName,
      },
    });
  } catch (err: any) {
    console.error(`[relay] dispatch error:`, err.message);
    jsonResp(res, 500, { error: err.message });
  }
}

// ── Bulk Dispatch (POST /dispatch/batch) ────────────────────────────
async function handleBatchDispatch(
  res: ServerResponse,
  payload: any
): Promise<void> {
  const { tasks } = payload; // Array of { agentName, title, description, priority }
  if (!Array.isArray(tasks) || tasks.length === 0) {
    jsonResp(res, 400, { error: "tasks array required" });
    return;
  }

  console.log(`[relay] 📬 batch dispatch: ${tasks.length} tasks`);

  const results = [];
  const agentsToWake = new Set<string>();

  for (const task of tasks) {
    const entry = Object.entries(agentMapping).find(
      ([, v]) => v.nanobotName === task.agentName
    );
    if (!entry) {
      results.push({ title: task.title, error: `Agent "${task.agentName}" not found` });
      continue;
    }
    const [paperclipId, mapping] = entry;

    try {
      const issue = await pcPost(`/companies/${COMPANY_ID}/issues`, {
        title: task.title,
        description: task.description || "",
        priority: task.priority || "medium",
        assigneeAgentId: paperclipId,
        projectId: PROJECT_ID,
        status: "todo",
      });
      results.push({
        identifier: issue.identifier,
        title: issue.title,
        assignee: mapping.nanobotName,
        status: "created",
      });
    } catch (err: any) {
      results.push({ title: task.title, error: err.message });
    }
  }

  // Paperclip auto-wakes agents on assignment — no manual wakeup needed
  jsonResp(res, 201, { dispatched: results.length, results });
}

// ── Dashboard Summary (GET /dashboard) ──────────────────────────────
async function handleDashboard(res: ServerResponse): Promise<void> {
  try {
    const dashboard = await pcGet(`/companies/${COMPANY_ID}/dashboard`);
    const agents = await pcGet(`/companies/${COMPANY_ID}/agents`);

    const summary = {
      company: "Street Voices",
      agents: {
        total: agents.length,
        active: dashboard.agents.active,
        running: dashboard.agents.running,
        paused: dashboard.agents.paused,
        error: dashboard.agents.error,
      },
      tasks: dashboard.tasks,
      costs: {
        monthSpend: `$${(dashboard.costs.monthSpendCents / 100).toFixed(2)}`,
        monthBudget: `$${(dashboard.costs.monthBudgetCents / 100).toFixed(2)}`,
        utilization: `${dashboard.costs.monthUtilizationPercent}%`,
      },
      pendingApprovals: dashboard.pendingApprovals,
      teams: {} as Record<string, { lead: string; members: string[] }>,
    };

    // Build team summary
    for (const agent of agents) {
      const team = agent.metadata?.team;
      if (!team) continue;
      if (!summary.teams[team]) {
        summary.teams[team] = { lead: "", members: [] };
      }
      if (agent.metadata?.nanobotName?.includes("manager") || agent.role === "ceo") {
        summary.teams[team].lead = agent.name;
      } else {
        summary.teams[team].members.push(agent.name);
      }
    }

    jsonResp(res, 200, summary);
  } catch (err: any) {
    jsonResp(res, 500, { error: err.message });
  }
}

// ── HTTP Server ─────────────────────────────────────────────────────
const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://localhost:${RELAY_PORT}`);

  // CORS for dashboard
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  try {
    // Health check
    if (req.method === "GET" && url.pathname === "/health") {
      jsonResp(res, 200, {
        status: "ok",
        service: "paperclip-relay",
        version: 2,
        agents: Object.keys(agentMapping).length,
        paperclip: PAPERCLIP_API,
        nanobot: NANOBOT_API,
      });
      return;
    }

    // Dashboard summary
    if (req.method === "GET" && url.pathname === "/dashboard") {
      await handleDashboard(res);
      return;
    }

    // Agent list
    if (req.method === "GET" && url.pathname === "/agents") {
      jsonResp(res, 200, agentMapping);
      return;
    }

    // Reload mapping
    if (req.method === "POST" && url.pathname === "/reload") {
      agentMapping = loadMapping();
      jsonResp(res, 200, {
        status: "reloaded",
        agents: Object.keys(agentMapping).length,
      });
      return;
    }

    // Heartbeat webhook from Paperclip
    if (req.method === "POST" && url.pathname === "/heartbeat") {
      const body = JSON.parse(await readBody(req));
      await handleHeartbeat(res, body);
      return;
    }

    // Dispatch single task
    if (req.method === "POST" && url.pathname === "/dispatch") {
      const body = JSON.parse(await readBody(req));
      await handleDispatch(res, body);
      return;
    }

    // Batch dispatch
    if (req.method === "POST" && url.pathname === "/dispatch/batch") {
      const body = JSON.parse(await readBody(req));
      await handleBatchDispatch(res, body);
      return;
    }

    jsonResp(res, 404, { error: "Not found" });
  } catch (err: any) {
    console.error("[relay] unhandled error:", err);
    jsonResp(res, 500, { error: err.message || "Internal error" });
  }
});

server.listen(RELAY_PORT, "127.0.0.1", () => {
  const count = Object.keys(agentMapping).length;
  console.log(`[relay] ═══════════════════════════════════════`);
  console.log(`[relay] Paperclip Relay v2 on http://127.0.0.1:${RELAY_PORT}`);
  console.log(`[relay] ${count} agents mapped`);
  console.log(`[relay] Nanobot: ${NANOBOT_API}`);
  console.log(`[relay] Paperclip: ${PAPERCLIP_API}`);
  console.log(`[relay] ═══════════════════════════════════════`);
});
