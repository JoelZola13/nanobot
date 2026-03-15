#!/usr/bin/env npx tsx
/**
 * Paperclip MCP Server — Exposes Paperclip issue management as MCP tools
 * so nanobot agents can self-manage their work items.
 *
 * Transport: stdio (launched by nanobot's MCP handler)
 * Auth: Uses Paperclip REST API (local trusted mode, no auth needed)
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { readFileSync } from "node:fs";
import { execSync } from "node:child_process";

// ── Config ──────────────────────────────────────────────────────────────────
const PAPERCLIP_API = process.env.PAPERCLIP_API || "http://127.0.0.1:3100/api";
const NANOBOT_API = process.env.NANOBOT_API || "http://127.0.0.1:18790";
const COMPANY_ID = process.env.PAPERCLIP_COMPANY_ID || "78940514-fbb0-4c2d-8cee-09bcfd5399a4";
const PROJECT_ID = process.env.PAPERCLIP_PROJECT_ID || "645c5cd9-caa1-46c7-be50-da6e6001df14";
const CRON_JOBS_PATH = process.env.CRON_JOBS_PATH || `${process.env.HOME}/.nanobot/workspace/cron/jobs.json`;

// ── Helpers ─────────────────────────────────────────────────────────────────
async function api(path: string, opts?: RequestInit) {
  const url = `${PAPERCLIP_API}${path}`;
  const res = await fetch(url, {
    ...opts,
    headers: { "Content-Type": "application/json", ...opts?.headers },
  });
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text, status: res.status };
  }
}

function companyPath(sub: string) {
  return `/companies/${COMPANY_ID}${sub}`;
}

// ── MCP Server ──────────────────────────────────────────────────────────────
const server = new McpServer({
  name: "paperclip",
  version: "1.0.0",
});

// ── Issues ──────────────────────────────────────────────────────────────────

server.tool(
  "list_issues",
  "List issues in the Paperclip project. Filter by status, priority, or assignee.",
  {
    status: z.enum(["backlog", "todo", "in_progress", "done", "cancelled"]).optional().describe("Filter by status"),
    priority: z.enum(["low", "medium", "high", "urgent"]).optional().describe("Filter by priority"),
    assignee_agent_id: z.string().uuid().optional().describe("Filter by assigned agent ID"),
    limit: z.number().min(1).max(100).optional().describe("Max results (default 50)"),
  },
  async ({ status, priority, assignee_agent_id, limit }) => {
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    if (priority) params.set("priority", priority);
    if (assignee_agent_id) params.set("assigneeAgentId", assignee_agent_id);
    params.set("limit", String(limit || 50));
    const qs = params.toString();
    const data = await api(companyPath(`/issues?${qs}`));
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "get_issue",
  "Get a single issue by ID, including comments and sub-issues.",
  {
    issue_id: z.string().describe("Issue UUID or identifier (e.g. PC-12)"),
  },
  async ({ issue_id }) => {
    const issue = await api(companyPath(`/issues/${issue_id}`));
    // Fetch comments
    let comments: any[] = [];
    try {
      comments = await api(companyPath(`/issues/${issue.id || issue_id}/comments`));
    } catch {}
    // Fetch children
    let children: any[] = [];
    try {
      const allIssues = await api(companyPath(`/issues?limit=200`));
      if (Array.isArray(allIssues)) {
        children = allIssues.filter((i: any) => i.parentId === (issue.id || issue_id));
      }
    } catch {}
    const result = { ...issue, comments, children };
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  }
);

server.tool(
  "create_issue",
  "Create a new issue/task in Paperclip. Optionally assign to an agent, set priority, link to parent issue or goal.",
  {
    title: z.string().describe("Issue title"),
    description: z.string().optional().describe("Detailed description"),
    status: z.enum(["backlog", "todo", "in_progress", "done"]).optional().describe("Initial status (default: todo)"),
    priority: z.enum(["low", "medium", "high", "urgent"]).optional().describe("Priority level"),
    assignee_agent_id: z.string().uuid().optional().describe("Agent UUID to assign"),
    parent_id: z.string().uuid().optional().describe("Parent issue ID (makes this a subtask)"),
    goal_id: z.string().uuid().optional().describe("Link to a goal"),
  },
  async ({ title, description, status, priority, assignee_agent_id, parent_id, goal_id }) => {
    const body: any = {
      title,
      projectId: PROJECT_ID,
      status: status || "todo",
    };
    if (description) body.description = description;
    if (priority) body.priority = priority;
    if (assignee_agent_id) body.assigneeAgentId = assignee_agent_id;
    if (parent_id) body.parentId = parent_id;
    if (goal_id) body.goalId = goal_id;

    const data = await api(companyPath("/issues"), {
      method: "POST",
      body: JSON.stringify(body),
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "update_issue",
  "Update an existing issue — change status, priority, assignee, title, or description.",
  {
    issue_id: z.string().uuid().describe("Issue UUID to update"),
    title: z.string().optional().describe("New title"),
    description: z.string().optional().describe("New description"),
    status: z.enum(["backlog", "todo", "in_progress", "done", "cancelled"]).optional().describe("New status"),
    priority: z.enum(["low", "medium", "high", "urgent"]).optional().describe("New priority"),
    assignee_agent_id: z.string().uuid().optional().describe("Reassign to agent UUID (or null to unassign)"),
    comment: z.string().optional().describe("Add a comment with the update"),
  },
  async ({ issue_id, title, description, status, priority, assignee_agent_id, comment }) => {
    const body: any = {};
    if (title) body.title = title;
    if (description) body.description = description;
    if (status) body.status = status;
    if (priority) body.priority = priority;
    if (assignee_agent_id) body.assigneeAgentId = assignee_agent_id;

    const data = await api(companyPath(`/issues/${issue_id}`), {
      method: "PATCH",
      body: JSON.stringify(body),
    });

    // Add comment if provided
    if (comment && data.id) {
      await api(companyPath(`/issues/${data.id}/comments`), {
        method: "POST",
        body: JSON.stringify({ body: comment }),
      });
    }

    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "add_comment",
  "Add a comment to an issue — for progress updates, notes, or blockers.",
  {
    issue_id: z.string().uuid().describe("Issue UUID"),
    body: z.string().describe("Comment text (markdown supported)"),
  },
  async ({ issue_id, body: commentBody }) => {
    const data = await api(companyPath(`/issues/${issue_id}/comments`), {
      method: "POST",
      body: JSON.stringify({ body: commentBody }),
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "create_subtasks",
  "Break down a parent issue into subtasks. Creates child issues linked to the parent.",
  {
    parent_id: z.string().uuid().describe("Parent issue UUID"),
    subtasks: z.array(z.object({
      title: z.string().describe("Subtask title"),
      description: z.string().optional().describe("Subtask description"),
      priority: z.enum(["low", "medium", "high", "urgent"]).optional(),
      assignee_agent_id: z.string().uuid().optional().describe("Agent to assign"),
    })).describe("Array of subtasks to create"),
  },
  async ({ parent_id, subtasks }) => {
    const results = [];
    for (const task of subtasks) {
      const body: any = {
        title: task.title,
        projectId: PROJECT_ID,
        parentId: parent_id,
        status: "todo",
      };
      if (task.description) body.description = task.description;
      if (task.priority) body.priority = task.priority;
      if (task.assignee_agent_id) body.assigneeAgentId = task.assignee_agent_id;

      const data = await api(companyPath("/issues"), {
        method: "POST",
        body: JSON.stringify(body),
      });
      results.push(data);
    }
    return { content: [{ type: "text" as const, text: JSON.stringify(results, null, 2) }] };
  }
);

// ── Agents ──────────────────────────────────────────────────────────────────

server.tool(
  "list_agents",
  "List all agents in the organization with their teams, roles, and current status.",
  {},
  async () => {
    const data = await api(companyPath("/agents"));
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "get_agent",
  "Get details about a specific agent including their team, role, and active issues.",
  {
    agent_id: z.string().uuid().describe("Agent UUID"),
  },
  async ({ agent_id }) => {
    const agent = await api(companyPath(`/agents/${agent_id}`));
    // Get their active issues
    const issues = await api(companyPath(`/issues?assigneeAgentId=${agent_id}&limit=20`));
    const result = { ...agent, activeIssues: Array.isArray(issues) ? issues.filter((i: any) => i.status !== "done") : [] };
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  }
);

// ── Goals ────────────────────────────────────────────────────────────────────

server.tool(
  "list_goals",
  "List all company and team goals.",
  {},
  async () => {
    const data = await api(companyPath("/goals"));
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "create_goal",
  "Create a new goal at company or team level.",
  {
    title: z.string().describe("Goal title"),
    level: z.enum(["company", "team", "agent"]).describe("Goal scope"),
    parent_id: z.string().uuid().optional().describe("Parent goal ID for sub-goals"),
    owner_agent_id: z.string().uuid().optional().describe("Agent who owns this goal"),
  },
  async ({ title, level, parent_id, owner_agent_id }) => {
    const body: any = { title, level };
    if (parent_id) body.parentId = parent_id;
    if (owner_agent_id) body.ownerAgentId = owner_agent_id;
    const data = await api(companyPath("/goals"), {
      method: "POST",
      body: JSON.stringify(body),
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

// ── Approvals ───────────────────────────────────────────────────────────────

server.tool(
  "request_approval",
  "Request approval from a human or manager agent before proceeding with a high-impact action.",
  {
    title: z.string().describe("What needs approval"),
    description: z.string().describe("Details and justification"),
    payload_json: z.string().optional().describe("JSON string of structured data for the approval (e.g. email draft, budget request)"),
  },
  async ({ title, description, payload_json }) => {
    const body: any = { title, description };
    if (payload_json) {
      try { body.payload = JSON.parse(payload_json); } catch { body.payload = payload_json; }
    }
    const data = await api(companyPath("/approvals"), {
      method: "POST",
      body: JSON.stringify(body),
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "list_approvals",
  "List pending and recent approvals.",
  {
    status: z.enum(["pending", "approved", "rejected", "revision_requested"]).optional(),
  },
  async ({ status }) => {
    const qs = status ? `?status=${status}` : "";
    const data = await api(companyPath(`/approvals${qs}`));
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

// ── Activity & Costs ────────────────────────────────────────────────────────

server.tool(
  "activity_feed",
  "Get recent activity across the organization — issue updates, comments, completions.",
  {
    limit: z.number().min(1).max(50).optional().describe("Number of recent events (default 20)"),
  },
  async ({ limit }) => {
    const data = await api(companyPath(`/activity?limit=${limit || 20}`));
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "report_cost",
  "Report token usage / cost for a completed task.",
  {
    agent_id: z.string().uuid().describe("Agent who incurred the cost"),
    issue_id: z.string().uuid().optional().describe("Related issue"),
    input_tokens: z.number().describe("Input tokens used"),
    output_tokens: z.number().describe("Output tokens used"),
    model: z.string().optional().describe("Model name (default: gpt-5.1-codex)"),
  },
  async ({ agent_id, issue_id, input_tokens, output_tokens, model }) => {
    const inputCost = (input_tokens / 1000) * 0.003;
    const outputCost = (output_tokens / 1000) * 0.015;
    const body: any = {
      agentId: agent_id,
      model: model || "gpt-5.1-codex",
      inputTokens: input_tokens,
      outputTokens: output_tokens,
      cost: inputCost + outputCost,
      currency: "USD",
    };
    if (issue_id) body.issueId = issue_id;
    const data = await api(companyPath("/cost-events"), {
      method: "POST",
      body: JSON.stringify(body),
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

// ── Dashboard ───────────────────────────────────────────────────────────────

server.tool(
  "dashboard",
  "Get a high-level dashboard summary: active issues, team workloads, goals progress, recent activity.",
  {},
  async () => {
    const [issues, agents, goals, activity] = await Promise.all([
      api(companyPath("/issues?limit=200")),
      api(companyPath("/agents")),
      api(companyPath("/goals")),
      api(companyPath("/activity?limit=10")),
    ]);

    const issueArr = Array.isArray(issues) ? issues : [];
    const agentArr = Array.isArray(agents) ? agents : [];

    const summary = {
      totalIssues: issueArr.length,
      byStatus: {
        backlog: issueArr.filter((i: any) => i.status === "backlog").length,
        todo: issueArr.filter((i: any) => i.status === "todo").length,
        in_progress: issueArr.filter((i: any) => i.status === "in_progress").length,
        done: issueArr.filter((i: any) => i.status === "done").length,
      },
      byPriority: {
        urgent: issueArr.filter((i: any) => i.priority === "urgent").length,
        high: issueArr.filter((i: any) => i.priority === "high").length,
        medium: issueArr.filter((i: any) => i.priority === "medium").length,
        low: issueArr.filter((i: any) => i.priority === "low").length,
      },
      totalAgents: agentArr.length,
      goals: Array.isArray(goals) ? goals : [],
      recentActivity: Array.isArray(activity) ? activity.slice(0, 5) : [],
    };

    return { content: [{ type: "text" as const, text: JSON.stringify(summary, null, 2) }] };
  }
);

// ── Labels ──────────────────────────────────────────────────────────────────

server.tool(
  "list_labels",
  "List all available labels for categorizing issues.",
  {},
  async () => {
    const data = await api(companyPath("/labels"));
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

// ── Cron / Scheduling ───────────────────────────────────────────────────────

function readCronJobs(): any[] {
  try {
    const data = JSON.parse(readFileSync(CRON_JOBS_PATH, "utf-8"));
    return data.jobs || [];
  } catch {
    return [];
  }
}

server.tool(
  "list_schedules",
  "List all scheduled/recurring agent jobs (cron jobs). Shows what each agent has scheduled, when it runs next, and its last status.",
  {},
  async () => {
    const jobs = readCronJobs();
    const summary = jobs.map((j: any) => ({
      id: j.id,
      name: j.name,
      enabled: j.enabled,
      schedule: j.schedule?.kind === "cron" ? `cron: ${j.schedule.expr} (${j.schedule.tz || "UTC"})` :
               j.schedule?.kind === "every" ? `every ${(j.schedule.everyMs / 1000 / 60).toFixed(0)} min` :
               j.schedule?.kind === "at" ? `once at ${new Date(j.schedule.atMs).toISOString()}` : "unknown",
      message: j.payload?.message?.substring(0, 80) || "",
      channel: j.payload?.channel || "",
      lastRun: j.state?.lastRunAtMs ? new Date(j.state.lastRunAtMs).toISOString() : "never",
      nextRun: j.state?.nextRunAtMs ? new Date(j.state.nextRunAtMs).toISOString() : "not scheduled",
      lastStatus: j.state?.lastStatus || "unknown",
    }));
    return { content: [{ type: "text" as const, text: JSON.stringify(summary, null, 2) }] };
  }
);

server.tool(
  "create_schedule",
  "Create a new scheduled/recurring task for an agent. Use cron expressions for recurring, ISO datetime for one-time.",
  {
    name: z.string().describe("Schedule name (short label)"),
    message: z.string().describe("The task/prompt the agent will execute on each run"),
    cron_expr: z.string().optional().describe("Cron expression like '0 9 * * 1' (9 AM every Monday). Use this for recurring."),
    timezone: z.string().optional().describe("IANA timezone (default: America/Toronto)"),
    at_datetime: z.string().optional().describe("ISO datetime for one-time execution (e.g. '2026-03-15T14:00:00')"),
    every_minutes: z.number().optional().describe("Run every N minutes (for interval-based schedules)"),
    channel: z.string().optional().describe("Delivery channel: 'slack' or 'librechat' (default: slack)"),
    deliver_to: z.string().optional().describe("Channel/DM ID for delivery (default: Joel's Slack DM)"),
  },
  async ({ name, message, cron_expr, timezone, at_datetime, every_minutes, channel, deliver_to }) => {
    // Build the cron tool call via nanobot API
    const cronArgs: any = { action: "add", message };

    if (cron_expr) {
      cronArgs.cron_expr = cron_expr;
      cronArgs.tz = timezone || "America/Toronto";
    } else if (at_datetime) {
      cronArgs.at = at_datetime;
    } else if (every_minutes) {
      cronArgs.every_seconds = every_minutes * 60;
    } else {
      return { content: [{ type: "text" as const, text: "Error: provide cron_expr, at_datetime, or every_minutes" }] };
    }

    // Call nanobot API to create the cron job
    try {
      const res = await fetch(`${NANOBOT_API}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "openai-codex/gpt-5.1-codex",
          messages: [{
            role: "user",
            content: `Use the cron tool to add a scheduled job. Name: "${name}". Details: action=add, message="${message}"${cron_expr ? `, cron_expr="${cron_expr}", tz="${timezone || "America/Toronto"}"` : ""}${at_datetime ? `, at="${at_datetime}"` : ""}${every_minutes ? `, every_seconds=${every_minutes * 60}` : ""}. Just create it and confirm.`
          }],
          stream: false,
        }),
      });
      const data = await res.json();
      const reply = data?.choices?.[0]?.message?.content || "No response";
      return { content: [{ type: "text" as const, text: reply }] };
    } catch (err: any) {
      return { content: [{ type: "text" as const, text: `Error creating schedule: ${err.message}` }] };
    }
  }
);

server.tool(
  "remove_schedule",
  "Remove a scheduled job by its ID.",
  {
    job_id: z.string().describe("The job ID to remove (from list_schedules)"),
  },
  async ({ job_id }) => {
    try {
      const res = await fetch(`${NANOBOT_API}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "openai-codex/gpt-5.1-codex",
          messages: [{ role: "user", content: `Use the cron tool to remove job with id "${job_id}". action=remove, job_id="${job_id}".` }],
          stream: false,
        }),
      });
      const data = await res.json();
      const reply = data?.choices?.[0]?.message?.content || "No response";
      return { content: [{ type: "text" as const, text: reply }] };
    } catch (err: any) {
      return { content: [{ type: "text" as const, text: `Error removing schedule: ${err.message}` }] };
    }
  }
);

// ── Calendar ────────────────────────────────────────────────────────────────

server.tool(
  "calendar",
  "Get a unified calendar view: upcoming cron schedules, active Paperclip issues, goals, and recent completions. Great for seeing what's happening this week.",
  {
    days_ahead: z.number().min(1).max(90).optional().describe("How many days ahead to show (default 7)"),
  },
  async ({ days_ahead }) => {
    const horizon = (days_ahead || 7) * 24 * 60 * 60 * 1000;
    const now = Date.now();
    const cutoff = now + horizon;

    // Fetch everything in parallel
    const [issues, goals, cronJobs] = await Promise.all([
      api(companyPath("/issues?limit=200")),
      api(companyPath("/goals")),
      Promise.resolve(readCronJobs()),
    ]);

    const issueArr = Array.isArray(issues) ? issues : [];

    // Upcoming cron runs
    const upcomingCron = cronJobs
      .filter((j: any) => j.enabled && j.state?.nextRunAtMs && j.state.nextRunAtMs <= cutoff)
      .map((j: any) => ({
        type: "scheduled_job",
        time: new Date(j.state.nextRunAtMs).toISOString(),
        name: j.name,
        id: j.id,
        schedule: j.schedule?.kind === "cron" ? j.schedule.expr : j.schedule?.kind,
        message: j.payload?.message?.substring(0, 60) || "",
      }))
      .sort((a: any, b: any) => a.time.localeCompare(b.time));

    // Active issues (todo + in_progress)
    const activeIssues = issueArr
      .filter((i: any) => i.status === "todo" || i.status === "in_progress")
      .map((i: any) => ({
        type: "active_issue",
        status: i.status,
        priority: i.priority,
        title: i.title,
        id: i.id,
        identifier: i.identifier,
        assignee: i.assigneeAgentId || "unassigned",
        created: i.createdAt,
      }));

    // Recently completed (last N days)
    const recentDone = issueArr
      .filter((i: any) => i.status === "done" && i.updatedAt && (now - new Date(i.updatedAt).getTime()) < horizon)
      .map((i: any) => ({
        type: "completed_issue",
        title: i.title,
        id: i.id,
        identifier: i.identifier,
        completedAt: i.updatedAt,
      }));

    // All cron jobs overview
    const allSchedules = cronJobs
      .filter((j: any) => j.enabled)
      .map((j: any) => ({
        type: "recurring_schedule",
        name: j.name,
        id: j.id,
        schedule: j.schedule?.kind === "cron"
          ? `${j.schedule.expr} ${j.schedule.tz || "UTC"}`
          : j.schedule?.kind === "every"
            ? `every ${(j.schedule.everyMs / 1000 / 60).toFixed(0)}m`
            : `once at ${new Date(j.schedule.atMs).toISOString()}`,
        lastRun: j.state?.lastRunAtMs ? new Date(j.state.lastRunAtMs).toISOString() : null,
        nextRun: j.state?.nextRunAtMs ? new Date(j.state.nextRunAtMs).toISOString() : null,
      }));

    const calendar = {
      period: `${new Date(now).toISOString().split("T")[0]} → ${new Date(cutoff).toISOString().split("T")[0]}`,
      upcomingScheduledJobs: upcomingCron,
      activeIssues,
      recentlyCompleted: recentDone,
      recurringSchedules: allSchedules,
      goals: Array.isArray(goals) ? goals.map((g: any) => ({
        title: g.title,
        level: g.level,
        status: g.status,
      })) : [],
    };

    return { content: [{ type: "text" as const, text: JSON.stringify(calendar, null, 2) }] };
  }
);

// ── Approval Revisions ──────────────────────────────────────────────────────

server.tool(
  "request_revision",
  "Send an approval back for revision with feedback — the agent must resubmit.",
  {
    approval_id: z.string().uuid().describe("Approval UUID"),
    feedback: z.string().describe("What needs to change before re-approval"),
  },
  async ({ approval_id, feedback }) => {
    const data = await api(`/approvals/${approval_id}/request-revision`, {
      method: "POST",
      body: JSON.stringify({ comment: feedback }),
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "resubmit_approval",
  "Resubmit an approval after addressing revision feedback.",
  {
    approval_id: z.string().uuid().describe("Approval UUID"),
    comment: z.string().optional().describe("What was changed"),
    updated_payload_json: z.string().optional().describe("Updated payload JSON if the content changed"),
  },
  async ({ approval_id, comment, updated_payload_json }) => {
    const body: any = {};
    if (comment) body.comment = comment;
    if (updated_payload_json) {
      try { body.payload = JSON.parse(updated_payload_json); } catch { body.payload = updated_payload_json; }
    }
    const data = await api(`/approvals/${approval_id}/resubmit`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

// ── Agent API Keys ──────────────────────────────────────────────────────────

server.tool(
  "create_agent_key",
  "Create an API key for an agent. Returns a token (pcp_...) the agent can use for authenticated Paperclip API calls.",
  {
    agent_id: z.string().uuid().describe("Agent UUID"),
    label: z.string().optional().describe("Key label/name (default: 'default')"),
  },
  async ({ agent_id, label }) => {
    const data = await api(`/agents/${agent_id}/keys`, {
      method: "POST",
      body: JSON.stringify({ label: label || "default" }),
    });
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "list_agent_keys",
  "List all API keys for an agent.",
  {
    agent_id: z.string().uuid().describe("Agent UUID"),
  },
  async ({ agent_id }) => {
    const data = await api(`/agents/${agent_id}/keys`);
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

// ── Company Export / Backup ─────────────────────────────────────────────────

const PAPERCLIP_CLI = process.env.PAPERCLIP_CLI ||
  "/Users/joel/.npm/_npx/43414d9b790239bb/node_modules/.bin/paperclipai";
const BACKUP_DIR = process.env.BACKUP_DIR ||
  "/Users/joel/.nanobot/workspace/backups";

server.tool(
  "export_company",
  "Export the entire company (org chart, agents, config) as a portable backup. Saves to the backups directory.",
  {
    label: z.string().optional().describe("Backup label (appended to folder name, e.g. 'weekly-backup')"),
  },
  async ({ label }) => {
    const ts = new Date().toISOString().split("T")[0];
    const folderName = `street-voices-${ts}${label ? `-${label}` : ""}`;
    const outPath = `${BACKUP_DIR}/${folderName}`;
    try {
      const result = execSync(
        `${PAPERCLIP_CLI} company export ${COMPANY_ID} --out "${outPath}" --json`,
        { timeout: 30000, encoding: "utf-8" }
      );
      return { content: [{ type: "text" as const, text: `Exported to ${outPath}\n${result}` }] };
    } catch (err: any) {
      return { content: [{ type: "text" as const, text: `Export failed: ${err.message}` }] };
    }
  }
);

server.tool(
  "import_company",
  "Import a company backup or template from a local path, URL, or GitHub repo.",
  {
    source: z.string().describe("Path, URL, or GitHub repo to import from"),
    target_mode: z.enum(["new", "existing"]).optional().describe("'new' creates a new company, 'existing' merges into current (default: existing)"),
    collision: z.enum(["rename", "skip", "replace"]).optional().describe("How to handle name collisions (default: rename)"),
    dry_run: z.boolean().optional().describe("Preview without applying (default: false)"),
  },
  async ({ source, target_mode, collision, dry_run }) => {
    const args = [
      `${PAPERCLIP_CLI} company import`,
      `--from "${source}"`,
      `--target ${target_mode || "existing"}`,
    ];
    if (target_mode !== "new") args.push(`-C ${COMPANY_ID}`);
    if (collision) args.push(`--collision ${collision}`);
    if (dry_run) args.push("--dry-run");
    args.push("--json");

    try {
      const result = execSync(args.join(" "), { timeout: 30000, encoding: "utf-8" });
      return { content: [{ type: "text" as const, text: result }] };
    } catch (err: any) {
      return { content: [{ type: "text" as const, text: `Import failed: ${err.message}` }] };
    }
  }
);

// ── Company Config ──────────────────────────────────────────────────────────

server.tool(
  "get_company",
  "Get company details — name, budget, issue prefix, spending.",
  {},
  async () => {
    const data = await api(`/companies/${COMPANY_ID}`);
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

server.tool(
  "get_dashboard_summary",
  "Get the Paperclip native dashboard — agent counts, task stats, cost utilization, pending approvals.",
  {},
  async () => {
    const data = await api(`/companies/${COMPANY_ID}/dashboard`);
    return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
  }
);

// ── Wakeup Agent ────────────────────────────────────────────────────────────

server.tool(
  "wakeup_agent",
  "Dispatch an on-demand task to a specific agent by assigning them an issue. Creates a todo issue and assigns it — Paperclip auto-wakes the agent.",
  {
    agent_id: z.string().uuid().describe("Agent UUID to wake"),
    task: z.string().describe("Task description / prompt"),
    priority: z.enum(["low", "medium", "high", "urgent"]).optional().describe("Task priority"),
    goal_id: z.string().uuid().optional().describe("Link to a goal"),
    parent_id: z.string().uuid().optional().describe("Parent issue (makes this a subtask)"),
  },
  async ({ agent_id, task, priority, goal_id, parent_id }) => {
    const body: any = {
      title: task.substring(0, 80),
      description: task,
      projectId: PROJECT_ID,
      status: "todo",
      assigneeAgentId: agent_id,
    };
    if (priority) body.priority = priority;
    if (goal_id) body.goalId = goal_id;
    if (parent_id) body.parentId = parent_id;

    const issue = await api(companyPath("/issues"), {
      method: "POST",
      body: JSON.stringify(body),
    });
    return { content: [{ type: "text" as const, text: `Dispatched to agent. Issue: ${JSON.stringify(issue, null, 2)}` }] };
  }
);

// ── Start ───────────────────────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`[paperclip-mcp] Connected — ${PAPERCLIP_API} company=${COMPANY_ID}`);
}

main().catch((err) => {
  console.error("[paperclip-mcp] Fatal:", err);
  process.exit(1);
});
