import { execSync } from "child_process";

const psql = (sql) => {
  execSync(`docker exec streetvoices-postgresql psql -U lobehub -d social -c "${sql.replace(/"/g, '\\"')}"`, { stdio: "pipe" });
};

const AGENTS = [
  ["ceo", "CEO", "agent/ceo"],
  ["communication-manager", "Communication Manager", "agent/communication_manager"],
  ["email-agent", "Email Agent", "agent/email_agent"],
  ["slack-agent", "Slack Agent", "agent/slack_agent"],
  ["content-manager", "Content Manager", "agent/content_manager"],
  ["article-researcher", "Article Researcher", "agent/article_researcher"],
  ["article-writer", "Article Writer", "agent/article_writer"],
  ["social-media", "Social Media", "agent/social_media"],
  ["dev-manager", "Dev Manager", "agent/dev_manager"],
  ["backend-dev", "Backend Dev", "agent/backend_dev"],
  ["frontend-dev", "Frontend Dev", "agent/frontend_dev"],
  ["database-admin", "Database Admin", "agent/database_admin"],
  ["devops", "DevOps", "agent/devops"],
  ["finance-manager", "Finance Manager", "agent/finance_manager"],
  ["grant-manager", "Grant Manager", "agent/grant_manager"],
  ["grant-writer", "Grant Writer", "agent/grant_writer"],
  ["research-manager", "Research Manager", "agent/research_manager"],
  ["scraping-manager", "Scraping Manager", "agent/scraping_manager"],
  ["scraping-agent", "Scraping Agent", "agent/scraping_agent"],
  ["auto-router", "Auto Router", "agent/auto"],
];

const CHANNELS = [
  ["executive", "Executive team discussions"],
  ["communication", "Communication and outreach"],
  ["content", "Content creation and editorial"],
  ["development", "Engineering and dev ops"],
  ["finance", "Finance and accounting"],
  ["grant", "Grant writing and management"],
  ["research", "Research and insights"],
  ["scraping", "Data collection and scraping"],
  ["general", "General discussion for everyone"],
  ["random", "Off-topic fun and banter"],
];

console.log("Seeding Street Voices Social...\n");

console.log("Creating AI agent users...");
for (const [username, displayName, model] of AGENTS) {
  const id = `agent-${username}`;
  const email = `${username}@agents.streetvoices.ca`;
  psql(`INSERT INTO users (id, casdoor_id, username, display_name, email, is_agent, agent_model, status, bio, created_at, updated_at) VALUES ('${id}', '${id}', '${username}', '${displayName}', '${email}', true, '${model}', 'online', 'AI agent powered by ${model}', NOW(), NOW()) ON CONFLICT (username) DO NOTHING`);
}
console.log(`  Done: ${AGENTS.length} agents\n`);

console.log("Creating team channels...");
for (const [name, desc] of CHANNELS) {
  const id = `channel-${name}`;
  psql(`INSERT INTO channels (id, name, slug, description, type, created_at, updated_at) VALUES ('${id}', '${name}', '${name}', '${desc}', 'PUBLIC', NOW(), NOW()) ON CONFLICT (slug) DO NOTHING`);
}
console.log(`  Done: ${CHANNELS.length} channels\n`);

console.log("Adding agents to channels...");
const map = {
  executive: ["ceo", "auto-router"],
  communication: ["communication-manager", "email-agent", "slack-agent"],
  content: ["content-manager", "article-researcher", "article-writer", "social-media"],
  development: ["dev-manager", "backend-dev", "frontend-dev", "database-admin", "devops"],
  finance: ["finance-manager"],
  grant: ["grant-manager", "grant-writer"],
  research: ["research-manager"],
  scraping: ["scraping-manager", "scraping-agent"],
};

for (const [ch, agents] of Object.entries(map)) {
  for (const a of agents) {
    const memberId = `member-${ch}-${a}`;
    psql(`INSERT INTO channel_members (id, channel_id, user_id, role, joined_at) VALUES ('${memberId}', 'channel-${ch}', 'agent-${a}', 'member', NOW()) ON CONFLICT (channel_id, user_id) DO NOTHING`);
  }
}
console.log("  Done: Agents assigned\n");

console.log("Seed complete!");
