/**
 * Paperclip ↔ Nanobot Heartbeat Relay v3
 *
 * Full Paperclip integration:
 * - Issue lifecycle: checkout → in_progress → done → release
 * - Sub-issues / subtasks with parentId
 * - Labels for categorization
 * - Milestones with target dates (calendar)
 * - Goals linked to issues
 * - Cost event reporting per agent run
 * - Progress comments on issues during execution
 * - Initiatives for strategic planning
 * - Approval-gated high-cost tasks
 * - On-demand wakeup via assignment trigger
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
const NANOBOT_API = process.env.NANOBOT_API_URL || "http://127.0.0.1:18790";
const PAPERCLIP_API = process.env.PAPERCLIP_API_URL || "http://127.0.0.1:3100/api";
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

function findAgentByName(name: string): [string, AgentMapEntry] | null {
  const entry = Object.entries(agentMapping).find(
    ([, v]) => v.nanobotName === name
  );
  return entry ?? null;
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

async function pcDelete(path: string): Promise<any> {
  const resp = await fetch(`${PAPERCLIP_API}${path}`, { method: "DELETE" });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`DELETE ${path}: ${resp.status} — ${text}`);
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
  parentId: string | null;
  labels: Array<{ name: string }>;
  identifier: string;
}

async function getAgentIssues(agentId: string): Promise<PaperclipIssue[]> {
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
        console.log(`[relay] checkout conflict — proceeding anyway for ${label}`);
      }
    }

    // 4. Mark as in_progress
    try {
      await pcPatch(`/issues/${issue.id}`, { status: "in_progress" });
    } catch {
      // Already in_progress, continue
    }

    // 5. Post progress comment
    await addIssueComment(issue.id, `🤖 **${mapping.nanobotName}** is working on this...`);

    // 6. Build task prompt with full context
    const parts = [
      `## Task: ${issue.title}`,
      issue.description || "",
      `Priority: ${issue.priority || "medium"}`,
      issue.labels?.length ? `Labels: ${issue.labels.map((l) => l.name).join(", ")}` : "",
      issue.parentId ? `Parent issue: ${issue.parentId}` : "",
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

    // 10. Mark issue as done
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

// ═══════════════════════════════════════════════════════════════════
// DISPATCH ENDPOINTS
// ═══════════════════════════════════════════════════════════════════

// ── Single Dispatch (POST /dispatch) ────────────────────────────────
async function handleDispatch(
  res: ServerResponse,
  payload: any
): Promise<void> {
  const { agentName, title, description, priority, parentId, goalId, labelIds, dueDate } = payload;

  const entry = findAgentByName(agentName);
  if (!entry) {
    jsonResp(res, 404, { error: `Agent "${agentName}" not found` });
    return;
  }
  const [paperclipId, mapping] = entry;

  console.log(`[relay] 📬 dispatch: "${title}" → ${mapping.nanobotName}`);

  try {
    const issuePayload: Record<string, any> = {
      title,
      description: description || "",
      priority: priority || "medium",
      assigneeAgentId: paperclipId,
      projectId: PROJECT_ID,
      status: "todo",
    };

    // Optional fields
    if (parentId) issuePayload.parentId = parentId;
    if (goalId) issuePayload.goalId = goalId;
    if (labelIds) issuePayload.labelIds = labelIds;
    if (dueDate) issuePayload.dueDate = dueDate;

    const issue = await pcPost(`/companies/${COMPANY_ID}/issues`, issuePayload);

    console.log(`[relay] created issue ${issue.identifier}: "${issue.title}"`);

    // Paperclip auto-wakes the agent on assignment (with issue context).
    // Do NOT call /wakeup manually — it creates a second run with empty
    // context, which triggers the "No project workspace" STDERR warning.
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

// ── Dispatch with Sub-tasks (POST /dispatch/breakdown) ──────────────
// Creates a parent issue + sub-task issues assigned to different agents
async function handleDispatchBreakdown(
  res: ServerResponse,
  payload: any
): Promise<void> {
  const { title, description, priority, goalId, dueDate, subtasks } = payload;

  if (!Array.isArray(subtasks) || subtasks.length === 0) {
    jsonResp(res, 400, { error: "subtasks array required" });
    return;
  }

  console.log(`[relay] 📬 breakdown dispatch: "${title}" with ${subtasks.length} subtasks`);

  try {
    // 1. Create parent issue (unassigned — it's a container)
    const parent = await pcPost(`/companies/${COMPANY_ID}/issues`, {
      title,
      description: description || "",
      priority: priority || "medium",
      projectId: PROJECT_ID,
      status: "todo",
      ...(goalId ? { goalId } : {}),
      ...(dueDate ? { dueDate } : {}),
    });

    console.log(`[relay] parent: ${parent.identifier} — "${parent.title}"`);

    // 2. Create sub-task issues
    const results = [];
    for (const sub of subtasks) {
      const entry = findAgentByName(sub.agentName);
      if (!entry) {
        results.push({ title: sub.title, error: `Agent "${sub.agentName}" not found` });
        continue;
      }
      const [paperclipId, mapping] = entry;

      try {
        const child = await pcPost(`/companies/${COMPANY_ID}/issues`, {
          title: sub.title,
          description: sub.description || "",
          priority: sub.priority || priority || "medium",
          assigneeAgentId: paperclipId,
          parentId: parent.id,
          projectId: PROJECT_ID,
          status: "todo",
          ...(sub.dueDate ? { dueDate: sub.dueDate } : {}),
          ...(sub.labelIds ? { labelIds: sub.labelIds } : {}),
        });

        results.push({
          identifier: child.identifier,
          title: child.title,
          assignee: mapping.nanobotName,
          status: "created",
        });
      } catch (err: any) {
        results.push({ title: sub.title, error: err.message });
      }
    }

    jsonResp(res, 201, {
      status: "dispatched",
      parent: {
        id: parent.id,
        identifier: parent.identifier,
        title: parent.title,
      },
      subtasks: results,
    });
  } catch (err: any) {
    console.error(`[relay] breakdown error:`, err.message);
    jsonResp(res, 500, { error: err.message });
  }
}

// ── Batch Dispatch (POST /dispatch/batch) ────────────────────────────
async function handleBatchDispatch(
  res: ServerResponse,
  payload: any
): Promise<void> {
  const { tasks } = payload;
  if (!Array.isArray(tasks) || tasks.length === 0) {
    jsonResp(res, 400, { error: "tasks array required" });
    return;
  }

  console.log(`[relay] 📬 batch dispatch: ${tasks.length} tasks`);

  const results = [];
  for (const task of tasks) {
    const entry = findAgentByName(task.agentName);
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
        ...(task.parentId ? { parentId: task.parentId } : {}),
        ...(task.goalId ? { goalId: task.goalId } : {}),
        ...(task.labelIds ? { labelIds: task.labelIds } : {}),
        ...(task.dueDate ? { dueDate: task.dueDate } : {}),
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

// ═══════════════════════════════════════════════════════════════════
// LABELS
// ═══════════════════════════════════════════════════════════════════

async function handleLabels(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL
): Promise<void> {
  if (req.method === "GET") {
    const labels = await pcGet(`/companies/${COMPANY_ID}/labels`);
    jsonResp(res, 200, labels);
    return;
  }

  if (req.method === "POST") {
    const body = JSON.parse(await readBody(req));
    const label = await pcPost(`/companies/${COMPANY_ID}/labels`, {
      name: body.name,
      color: body.color || "#6366f1",
      description: body.description || "",
    });
    jsonResp(res, 201, label);
    return;
  }

  jsonResp(res, 405, { error: "Method not allowed" });
}

// ═══════════════════════════════════════════════════════════════════
// MILESTONES (Calendar)
// ═══════════════════════════════════════════════════════════════════

// Calendar view: project timeline + all active issues grouped by status & priority
async function handleCalendar(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL
): Promise<void> {
  if (req.method === "GET") {
    const [issues, agents, project, goals] = await Promise.all([
      pcGet(`/companies/${COMPANY_ID}/issues`),
      pcGet(`/companies/${COMPANY_ID}/agents`),
      pcGet(`/projects/${PROJECT_ID}`),
      pcGet(`/companies/${COMPANY_ID}/goals`),
    ]);

    const enriched = issues.map((i: any) => ({
      identifier: i.identifier,
      title: i.title,
      status: i.status,
      priority: i.priority,
      createdAt: i.createdAt,
      completedAt: i.completedAt,
      assignee: agents.find((a: any) => a.id === i.assigneeAgentId)?.name || "unassigned",
      team: agents.find((a: any) => a.id === i.assigneeAgentId)?.metadata?.team || "unassigned",
      hasSubtasks: issues.some((c: any) => c.parentId === i.id),
      isSubtask: !!i.parentId,
    }));

    const active = enriched.filter((i: any) =>
      ["todo", "in_progress", "backlog"].includes(i.status)
    );
    const recentDone = enriched
      .filter((i: any) => i.status === "done")
      .sort((a: any, b: any) => new Date(b.completedAt || b.createdAt).getTime() - new Date(a.completedAt || a.createdAt).getTime())
      .slice(0, 10);

    jsonResp(res, 200, {
      project: {
        name: project.name,
        targetDate: project.targetDate,
        status: project.status,
      },
      goals: goals.filter((g: any) => g.status === "active").map((g: any) => ({
        title: g.title,
        level: g.level,
      })),
      active,
      recentlyCompleted: recentDone,
      stats: {
        total: issues.length,
        active: active.length,
        done: enriched.filter((i: any) => i.status === "done").length,
      },
    });
    return;
  }

  jsonResp(res, 405, { error: "Method not allowed" });
}

// ═══════════════════════════════════════════════════════════════════
// GOALS
// ═══════════════════════════════════════════════════════════════════

async function handleGoals(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL
): Promise<void> {
  if (req.method === "GET") {
    const goals = await pcGet(`/companies/${COMPANY_ID}/goals`);
    jsonResp(res, 200, goals);
    return;
  }

  if (req.method === "POST") {
    const body = JSON.parse(await readBody(req));
    const goal = await pcPost(`/companies/${COMPANY_ID}/goals`, {
      title: body.title,
      level: body.level || "team", // company | team | agent | task
      status: body.status || "active",
    });
    jsonResp(res, 201, goal);
    return;
  }

  jsonResp(res, 405, { error: "Method not allowed" });
}

// ═══════════════════════════════════════════════════════════════════
// ISSUES (full CRUD)
// ═══════════════════════════════════════════════════════════════════

async function handleIssues(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL
): Promise<void> {
  const issueId = url.pathname.split("/issues/")[1]?.split("/")[0];

  // GET /issues — list all issues with optional filters
  if (req.method === "GET" && !issueId) {
    const params = url.searchParams.toString();
    const issues = await pcGet(
      `/companies/${COMPANY_ID}/issues${params ? `?${params}` : ""}`
    );
    jsonResp(res, 200, issues);
    return;
  }

  // GET /issues/:id — single issue with children & comments
  if (req.method === "GET" && issueId) {
    const issue = await pcGet(`/companies/${COMPANY_ID}/issues?parentId=${issueId}`);
    // Also get the issue itself
    const allIssues = await pcGet(`/companies/${COMPANY_ID}/issues`);
    const parent = allIssues.find((i: any) => i.id === issueId || i.identifier === issueId);
    const children = allIssues.filter((i: any) => i.parentId === issueId);
    const comments = await pcGet(`/issues/${issueId}/comments`).catch(() => []);

    jsonResp(res, 200, {
      ...parent,
      children,
      comments,
    });
    return;
  }

  // PATCH /issues/:id — update issue
  if (req.method === "PATCH" && issueId) {
    const body = JSON.parse(await readBody(req));
    const updated = await pcPatch(`/issues/${issueId}`, body);
    jsonResp(res, 200, updated);
    return;
  }

  jsonResp(res, 405, { error: "Method not allowed" });
}

// ═══════════════════════════════════════════════════════════════════
// APPROVALS
// ═══════════════════════════════════════════════════════════════════

async function handleApprovals(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL
): Promise<void> {
  const approvalId = url.pathname.split("/approvals/")[1]?.split("/")[0];
  const action = url.pathname.split("/approvals/")[1]?.split("/")[1]; // approve|reject

  // GET /approvals — list pending
  if (req.method === "GET" && !approvalId) {
    const approvals = await pcGet(`/companies/${COMPANY_ID}/approvals`);
    jsonResp(res, 200, approvals);
    return;
  }

  // POST /approvals/:id/approve
  if (req.method === "POST" && approvalId && action === "approve") {
    const result = await pcPost(`/approvals/${approvalId}/approve`, {});
    jsonResp(res, 200, result);
    return;
  }

  // POST /approvals/:id/reject
  if (req.method === "POST" && approvalId && action === "reject") {
    const body = JSON.parse(await readBody(req));
    const result = await pcPost(`/approvals/${approvalId}/reject`, {
      reason: body.reason || "",
    });
    jsonResp(res, 200, result);
    return;
  }

  jsonResp(res, 405, { error: "Method not allowed" });
}

// ═══════════════════════════════════════════════════════════════════
// ACTIVITY FEED
// ═══════════════════════════════════════════════════════════════════

async function handleActivity(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL
): Promise<void> {
  const limit = url.searchParams.get("limit") || "20";
  const activity = await pcGet(`/companies/${COMPANY_ID}/activity?limit=${limit}`);
  jsonResp(res, 200, activity);
}

// ═══════════════════════════════════════════════════════════════════
// COST SUMMARY
// ═══════════════════════════════════════════════════════════════════

async function handleCosts(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL
): Promise<void> {
  const dashboard = await pcGet(`/companies/${COMPANY_ID}/dashboard`);
  const agents = await pcGet(`/companies/${COMPANY_ID}/agents`);

  const agentCosts = agents.map((a: any) => ({
    name: a.name,
    team: a.metadata?.team || "unknown",
    budgetCents: a.budgetMonthlyCents,
    spentCents: a.spentMonthlyCents,
    utilization: a.budgetMonthlyCents > 0
      ? Math.round((a.spentMonthlyCents / a.budgetMonthlyCents) * 100)
      : 0,
  }));

  jsonResp(res, 200, {
    company: {
      monthSpendCents: dashboard.costs.monthSpendCents,
      monthBudgetCents: dashboard.costs.monthBudgetCents,
      utilization: dashboard.costs.monthUtilizationPercent,
    },
    agents: agentCosts.sort((a: any, b: any) => b.spentCents - a.spentCents),
  });
}

// ═══════════════════════════════════════════════════════════════════
// DASHBOARD (enhanced)
// ═══════════════════════════════════════════════════════════════════

async function handleDashboard(res: ServerResponse): Promise<void> {
  try {
    const [dashboard, agents, goals, issues] = await Promise.all([
      pcGet(`/companies/${COMPANY_ID}/dashboard`),
      pcGet(`/companies/${COMPANY_ID}/agents`),
      pcGet(`/companies/${COMPANY_ID}/goals`),
      pcGet(`/companies/${COMPANY_ID}/issues`),
    ]);

    // Build team summary
    const teams: Record<string, { lead: string; members: string[]; issueCount: number }> = {};
    for (const agent of agents) {
      const team = agent.metadata?.team;
      if (!team) continue;
      if (!teams[team]) {
        teams[team] = { lead: "", members: [], issueCount: 0 };
      }
      if (agent.metadata?.nanobotName?.includes("manager") || agent.role === "ceo") {
        teams[team].lead = agent.name;
      } else {
        teams[team].members.push(agent.name);
      }
    }

    // Count issues per team via agent assignments
    for (const issue of issues) {
      if (issue.assigneeAgentId) {
        const agentInfo = agents.find((a: any) => a.id === issue.assigneeAgentId);
        const team = agentInfo?.metadata?.team;
        if (team && teams[team]) {
          teams[team].issueCount++;
        }
      }
    }

    // Issues by status
    const issuesByStatus: Record<string, number> = {};
    for (const issue of issues) {
      issuesByStatus[issue.status] = (issuesByStatus[issue.status] || 0) + 1;
    }

    // Recent issues (last 10)
    const recentIssues = issues
      .sort((a: any, b: any) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, 10)
      .map((i: any) => ({
        identifier: i.identifier,
        title: i.title,
        status: i.status,
        priority: i.priority,
        assignee: agents.find((a: any) => a.id === i.assigneeAgentId)?.name || "unassigned",
      }));

    jsonResp(res, 200, {
      company: "Street Voices",
      agents: {
        total: agents.length,
        active: dashboard.agents.active,
        running: dashboard.agents.running,
        paused: dashboard.agents.paused,
        error: dashboard.agents.error,
      },
      tasks: {
        ...dashboard.tasks,
        byStatus: issuesByStatus,
        total: issues.length,
      },
      costs: {
        monthSpend: `$${(dashboard.costs.monthSpendCents / 100).toFixed(2)}`,
        monthBudget: `$${(dashboard.costs.monthBudgetCents / 100).toFixed(2)}`,
        utilization: `${dashboard.costs.monthUtilizationPercent}%`,
      },
      goals: goals.map((g: any) => ({
        title: g.title,
        level: g.level,
        status: g.status,
      })),
      pendingApprovals: dashboard.pendingApprovals,
      teams,
      recentIssues,
    });
  } catch (err: any) {
    jsonResp(res, 500, { error: err.message });
  }
}

// ═══════════════════════════════════════════════════════════════════
// HTTP SERVER
// ═══════════════════════════════════════════════════════════════════

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://localhost:${RELAY_PORT}`);

  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  try {
    // ── Health ───────────────────────────────────────────────────
    if (req.method === "GET" && url.pathname === "/health") {
      jsonResp(res, 200, {
        status: "ok",
        service: "paperclip-relay",
        version: 3,
        agents: Object.keys(agentMapping).length,
        paperclip: PAPERCLIP_API,
        nanobot: NANOBOT_API,
      });
      return;
    }

    // ── Dashboard ───────────────────────────────────────────────
    if (req.method === "GET" && url.pathname === "/dashboard") {
      await handleDashboard(res);
      return;
    }

    // ── Agents ──────────────────────────────────────────────────
    if (req.method === "GET" && url.pathname === "/agents") {
      jsonResp(res, 200, agentMapping);
      return;
    }

    // ── Reload mapping ──────────────────────────────────────────
    if (req.method === "POST" && url.pathname === "/reload") {
      agentMapping = loadMapping();
      jsonResp(res, 200, { status: "reloaded", agents: Object.keys(agentMapping).length });
      return;
    }

    // ── Heartbeat (from Paperclip) ──────────────────────────────
    if (req.method === "POST" && url.pathname === "/heartbeat") {
      const body = JSON.parse(await readBody(req));
      await handleHeartbeat(res, body);
      return;
    }

    // ── Dispatch ────────────────────────────────────────────────
    if (req.method === "POST" && url.pathname === "/dispatch") {
      const body = JSON.parse(await readBody(req));
      await handleDispatch(res, body);
      return;
    }

    if (req.method === "POST" && url.pathname === "/dispatch/batch") {
      const body = JSON.parse(await readBody(req));
      await handleBatchDispatch(res, body);
      return;
    }

    if (req.method === "POST" && url.pathname === "/dispatch/breakdown") {
      const body = JSON.parse(await readBody(req));
      await handleDispatchBreakdown(res, body);
      return;
    }

    // ── Issues ──────────────────────────────────────────────────
    if (url.pathname.startsWith("/issues")) {
      await handleIssues(req, res, url);
      return;
    }

    // ── Labels ──────────────────────────────────────────────────
    if (url.pathname === "/labels") {
      await handleLabels(req, res, url);
      return;
    }

    // ── Calendar (due dates + project timeline) ────────────────
    if (url.pathname === "/calendar") {
      await handleCalendar(req, res, url);
      return;
    }

    // ── Goals ───────────────────────────────────────────────────
    if (url.pathname === "/goals") {
      await handleGoals(req, res, url);
      return;
    }

    // ── Approvals ───────────────────────────────────────────────
    if (url.pathname.startsWith("/approvals")) {
      await handleApprovals(req, res, url);
      return;
    }

    // ── Activity feed ───────────────────────────────────────────
    if (req.method === "GET" && url.pathname === "/activity") {
      await handleActivity(req, res, url);
      return;
    }

    // ── Cost summary ────────────────────────────────────────────
    if (req.method === "GET" && url.pathname === "/costs") {
      await handleCosts(req, res, url);
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
  console.log(`[relay] Paperclip Relay v3 on http://127.0.0.1:${RELAY_PORT}`);
  console.log(`[relay] ${count} agents mapped`);
  console.log(`[relay] Nanobot: ${NANOBOT_API}`);
  console.log(`[relay] Paperclip: ${PAPERCLIP_API}`);
  console.log(`[relay] ───────────────────────────────────────`);
  console.log(`[relay] Endpoints:`);
  console.log(`[relay]   GET  /dashboard      — full overview`);
  console.log(`[relay]   GET  /agents         — agent mapping`);
  console.log(`[relay]   GET  /issues         — list issues`);
  console.log(`[relay]   GET  /labels         — list labels`);
  console.log(`[relay]   GET  /calendar       — upcoming deadlines`);
  console.log(`[relay]   GET  /goals          — list goals`);
  console.log(`[relay]   GET  /approvals      — pending approvals`);
  console.log(`[relay]   GET  /activity       — activity feed`);
  console.log(`[relay]   GET  /costs          — cost breakdown`);
  console.log(`[relay]   POST /dispatch       — single task`);
  console.log(`[relay]   POST /dispatch/batch — batch tasks`);
  console.log(`[relay]   POST /dispatch/breakdown — parent + subtasks`);
  console.log(`[relay]   POST /heartbeat      — Paperclip webhook`);
  console.log(`[relay]   POST /labels         — create label`);
  console.log(`[relay]   POST /goals          — create goal`);
  console.log(`[relay] ═══════════════════════════════════════`);
});
