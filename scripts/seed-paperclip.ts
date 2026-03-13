#!/usr/bin/env npx tsx
/**
 * Seed Paperclip with nanobot's 37-agent org structure.
 *
 * Usage: npx tsx scripts/seed-paperclip.ts
 *
 * This script:
 * 1. Finds the existing "Street Voices" company (or creates one)
 * 2. Deletes existing demo agents
 * 3. Creates all 37 agents with correct hierarchy
 * 4. Configures HTTP adapter pointing to the relay at localhost:3050
 * 5. Writes agent-mapping.json for the relay to consume
 */

import { writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const PAPERCLIP_API = "http://127.0.0.1:3100/api";
const RELAY_URL = "http://127.0.0.1:3050/heartbeat";

// ── API helpers ─────────────────────────────────────────────────────
async function api(method: string, path: string, body?: any): Promise<any> {
  const opts: RequestInit = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) opts.body = JSON.stringify(body);

  const resp = await fetch(`${PAPERCLIP_API}${path}`, opts);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${method} ${path}: ${resp.status} — ${text}`);
  }
  return resp.json();
}

const get = (path: string) => api("GET", path);
const post = (path: string, body: any) => api("POST", path, body);
const del = async (path: string) => {
  const resp = await fetch(`${PAPERCLIP_API}${path}`, { method: "DELETE" });
  return resp;
};

// ── Org definition ──────────────────────────────────────────────────
interface AgentDef {
  name: string;
  nanobotName: string;
  role: string;
  title: string;
  icon: string;
  team: string;
  isLead?: boolean;
  members?: AgentDef[];
}

const httpAdapterConfig = {
  url: RELAY_URL,
  method: "POST",
  headers: {},
  timeoutSec: 300,
};

const heartbeatConfig = {
  heartbeat: {
    enabled: true,
    intervalSec: 3600,
    cooldownSec: 30,
    wakeOnDemand: true,
    maxConcurrentRuns: 1,
  },
};

const ORG: AgentDef[] = [
  {
    name: "CEO",
    nanobotName: "ceo",
    role: "ceo",
    title: "Chief Executive Officer",
    icon: "brain",
    team: "executive",
    isLead: true,
    members: [
      {
        name: "Executive Memory",
        nanobotName: "executive_memory",
        role: "researcher",
        title: "Executive Memory Agent",
        icon: "database",
        team: "executive",
      },
      {
        name: "Security & Compliance",
        nanobotName: "security_compliance",
        role: "qa",
        title: "Security & Compliance Officer",
        icon: "shield",
        team: "executive",
      },
    ],
  },
  // ── Communication Team ──
  {
    name: "Communication Manager",
    nanobotName: "communication_manager",
    role: "pm",
    title: "Director of Communications",
    icon: "message-square",
    team: "communication",
    isLead: true,
    members: [
      { name: "Email Agent", nanobotName: "email_agent", role: "general", title: "Email Communications", icon: "mail", team: "communication" },
      { name: "Slack Agent", nanobotName: "slack_agent", role: "general", title: "Slack Communications", icon: "terminal", team: "communication" },
      { name: "WhatsApp Agent", nanobotName: "whatsapp_agent", role: "general", title: "WhatsApp Communications", icon: "message-square", team: "communication" },
      { name: "Calendar Agent", nanobotName: "calendar_agent", role: "general", title: "Calendar & Scheduling", icon: "target", team: "communication" },
      { name: "Communication Memory", nanobotName: "communication_memory", role: "researcher", title: "Communication Memory", icon: "database", team: "communication" },
    ],
  },
  // ── Content Team ──
  {
    name: "Content Manager",
    nanobotName: "content_manager",
    role: "cmo",
    title: "Content Director",
    icon: "file-code",
    team: "content",
    isLead: true,
    members: [
      { name: "Article Researcher", nanobotName: "article_researcher", role: "researcher", title: "Article Researcher", icon: "search", team: "content" },
      { name: "Article Writer", nanobotName: "article_writer", role: "general", title: "Article Writer", icon: "sparkles", team: "content" },
      { name: "Social Media Manager", nanobotName: "social_media_manager", role: "general", title: "Social Media Manager", icon: "globe", team: "content" },
      { name: "Content Memory", nanobotName: "content_memory", role: "researcher", title: "Content Memory", icon: "database", team: "content" },
    ],
  },
  // ── Development Team ──
  {
    name: "Development Manager",
    nanobotName: "development_manager",
    role: "cto",
    title: "VP of Engineering",
    icon: "code",
    team: "development",
    isLead: true,
    members: [
      { name: "Backend Developer", nanobotName: "backend_developer", role: "engineer", title: "Backend Engineer", icon: "terminal", team: "development" },
      { name: "Frontend Developer", nanobotName: "frontend_developer", role: "engineer", title: "Frontend Engineer", icon: "eye", team: "development" },
      { name: "Database Manager", nanobotName: "database_manager", role: "engineer", title: "Database Engineer", icon: "database", team: "development" },
      { name: "DevOps", nanobotName: "devops", role: "devops", title: "DevOps Engineer", icon: "cog", team: "development" },
      { name: "Development Memory", nanobotName: "development_memory", role: "researcher", title: "Development Memory", icon: "database", team: "development" },
    ],
  },
  // ── Finance Team ──
  {
    name: "Finance Manager",
    nanobotName: "finance_manager",
    role: "cfo",
    title: "Director of Finance",
    icon: "gem",
    team: "finance",
    isLead: true,
    members: [
      { name: "Accounting Agent", nanobotName: "accounting_agent", role: "general", title: "Accountant", icon: "wrench", team: "finance" },
      { name: "Crypto Agent", nanobotName: "crypto_agent", role: "general", title: "Crypto Analyst", icon: "atom", team: "finance" },
      { name: "Finance Memory", nanobotName: "finance_memory", role: "researcher", title: "Finance Memory", icon: "database", team: "finance" },
    ],
  },
  // ── Grant Writing Team ──
  {
    name: "Grant Manager",
    nanobotName: "grant_manager",
    role: "pm",
    title: "Director of Grants",
    icon: "file-code",
    team: "grant_writing",
    isLead: true,
    members: [
      { name: "Grant Writer", nanobotName: "grant_writer", role: "general", title: "Grant Writer", icon: "sparkles", team: "grant_writing" },
      { name: "Budget Manager", nanobotName: "budget_manager", role: "general", title: "Grant Budget Manager", icon: "gem", team: "grant_writing" },
      { name: "Project Manager", nanobotName: "project_manager", role: "pm", title: "Grant Project Manager", icon: "target", team: "grant_writing" },
      { name: "Grant Memory", nanobotName: "grant_memory", role: "researcher", title: "Grant Memory", icon: "database", team: "grant_writing" },
    ],
  },
  // ── Research Team ──
  {
    name: "Research Manager",
    nanobotName: "research_manager",
    role: "researcher",
    title: "Director of Research",
    icon: "search",
    team: "research",
    isLead: true,
    members: [
      { name: "Media Platform Researcher", nanobotName: "media_platform_researcher", role: "researcher", title: "Platform Analyst", icon: "globe", team: "research" },
      { name: "Media Program Researcher", nanobotName: "media_program_researcher", role: "researcher", title: "Program Analyst", icon: "telescope", team: "research" },
      { name: "Street Bot Researcher", nanobotName: "street_bot_researcher", role: "researcher", title: "Community Intelligence", icon: "radar", team: "research" },
      { name: "Research Memory", nanobotName: "research_memory", role: "researcher", title: "Research Memory", icon: "database", team: "research" },
    ],
  },
  // ── Scraping Team ──
  {
    name: "Scraping Manager",
    nanobotName: "scraping_manager",
    role: "engineer",
    title: "Director of Data Collection",
    icon: "package",
    team: "scraping",
    isLead: true,
    members: [
      { name: "Scraping Agent", nanobotName: "scraping_agent", role: "engineer", title: "Web Scraper", icon: "globe", team: "scraping" },
      { name: "Scraper Memory", nanobotName: "scraper_memory", role: "researcher", title: "Scraper Memory", icon: "database", team: "scraping" },
    ],
  },
];

// ── Main ────────────────────────────────────────────────────────────
async function main() {
  console.log("🏢 Seeding Paperclip with Street Voices org structure...\n");

  // 1. Find or create company
  const companies = await get("/companies");
  let company = companies.find((c: any) => c.name === "Street Voices");

  if (!company) {
    company = await post("/companies", {
      name: "Street Voices",
      description: "AI-powered media organization for street culture",
    });
    console.log(`✅ Created company: ${company.name} (${company.id})`);
  } else {
    console.log(`📋 Found existing company: ${company.name} (${company.id})`);
  }

  const companyId = company.id;

  // 2. Delete existing agents (clean slate)
  const existingAgents = await get(`/companies/${companyId}/agents`);
  if (existingAgents.length > 0) {
    console.log(`🧹 Removing ${existingAgents.length} existing agents...`);
    for (const agent of existingAgents) {
      try {
        await del(`/agents/${agent.id}`);
        console.log(`   ❌ Deleted: ${agent.name}`);
      } catch (err) {
        console.warn(`   ⚠️  Could not delete ${agent.name}: ${err}`);
      }
    }
  }

  // 3. Create agents with hierarchy
  const mapping: Record<string, { nanobotName: string; team: string; role: string }> = {};
  let totalCreated = 0;

  // Create CEO first
  const ceoDef = ORG[0];
  const ceo = await post(`/companies/${companyId}/agents`, {
    name: ceoDef.name,
    role: ceoDef.role,
    title: ceoDef.title,
    icon: ceoDef.icon,
    adapterType: "http",
    adapterConfig: httpAdapterConfig,
    runtimeConfig: heartbeatConfig,
    budgetMonthlyCents: 5000, // $50/month
    permissions: { canCreateAgents: true },
    metadata: { nanobotName: ceoDef.nanobotName, team: ceoDef.team },
  });
  mapping[ceo.id] = { nanobotName: ceoDef.nanobotName, team: ceoDef.team, role: ceoDef.role };
  totalCreated++;
  console.log(`\n👑 CEO: ${ceo.name} (${ceo.id})`);

  // Create CEO's direct reports (executive_memory, security_compliance)
  if (ceoDef.members) {
    for (const member of ceoDef.members) {
      const agent = await post(`/companies/${companyId}/agents`, {
        name: member.name,
        role: member.role,
        title: member.title,
        icon: member.icon,
        reportsTo: ceo.id,
        adapterType: "http",
        adapterConfig: httpAdapterConfig,
        runtimeConfig: heartbeatConfig,
        budgetMonthlyCents: 1000,
        metadata: { nanobotName: member.nanobotName, team: member.team },
      });
      mapping[agent.id] = { nanobotName: member.nanobotName, team: member.team, role: member.role };
      totalCreated++;
      console.log(`   └─ ${agent.name} (${agent.id})`);
    }
  }

  // Create team leads + members (skip CEO which is ORG[0])
  for (let i = 1; i < ORG.length; i++) {
    const teamLead = ORG[i];

    const lead = await post(`/companies/${companyId}/agents`, {
      name: teamLead.name,
      role: teamLead.role,
      title: teamLead.title,
      icon: teamLead.icon,
      reportsTo: ceo.id,
      adapterType: "http",
      adapterConfig: httpAdapterConfig,
      runtimeConfig: heartbeatConfig,
      budgetMonthlyCents: 3000, // $30/month for leads
      permissions: { canCreateAgents: false },
      metadata: { nanobotName: teamLead.nanobotName, team: teamLead.team },
    });
    mapping[lead.id] = { nanobotName: teamLead.nanobotName, team: teamLead.team, role: teamLead.role };
    totalCreated++;
    console.log(`\n📁 ${teamLead.team.toUpperCase()} — ${lead.name} (${lead.id})`);

    // Create team members
    if (teamLead.members) {
      for (const member of teamLead.members) {
        const agent = await post(`/companies/${companyId}/agents`, {
          name: member.name,
          role: member.role,
          title: member.title,
          icon: member.icon,
          reportsTo: lead.id,
          adapterType: "http",
          adapterConfig: httpAdapterConfig,
          runtimeConfig: heartbeatConfig,
          budgetMonthlyCents: 1000, // $10/month for members
          metadata: { nanobotName: member.nanobotName, team: member.team },
        });
        mapping[agent.id] = { nanobotName: member.nanobotName, team: member.team, role: member.role };
        totalCreated++;
        console.log(`   └─ ${agent.name} (${agent.id})`);
      }
    }
  }

  // 4. Write agent mapping file
  const mappingPath = resolve(__dirname, "../bridge/src/agent-mapping.json");
  writeFileSync(mappingPath, JSON.stringify(mapping, null, 2));
  console.log(`\n📄 Wrote agent-mapping.json (${Object.keys(mapping).length} agents)`);
  console.log(`   Path: ${mappingPath}`);

  // 5. Summary
  console.log(`\n✅ Done! Created ${totalCreated} agents in Paperclip.`);
  console.log(`   Company: ${company.name} (${companyId})`);
  console.log(`   Dashboard: http://127.0.0.1:3100`);
  console.log(`\n   Next steps:`);
  console.log(`   1. Start the relay: pm2 restart paperclip-relay`);
  console.log(`   2. Create tasks in Paperclip dashboard`);
  console.log(`   3. Assign to agents → heartbeat fires → nanobot processes`);
}

main().catch((err) => {
  console.error("❌ Seed failed:", err);
  process.exit(1);
});
