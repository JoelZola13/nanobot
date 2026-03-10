#!/usr/bin/env python3
"""
Create per-agent memory directories with initialized MEMORY.md and HISTORY.md files.

Each of the 37 nanobot agents gets a persistent memory store:
- MEMORY.md: Long-term facts, decisions, preferences, learned patterns
- HISTORY.md: Chronological event log (searchable)

Run: python scripts/create_agent_memory.py
Output: ~/.nanobot/workspace/memory/agents/{agent_name}/ (37 directories)
"""

import os
from pathlib import Path
from datetime import datetime

# ── Agent Registry (all 37) ─────────────────────────────────────────────────
AGENTS = [
    # Executive
    {"name": "ceo",                      "team": "executive",     "role": "lead",   "display": "CEO"},
    {"name": "security_compliance",      "team": "executive",     "role": "member", "display": "Security & Compliance"},
    {"name": "executive_memory",         "team": "executive",     "role": "memory", "display": "Executive Memory"},
    # Communication
    {"name": "communication_manager",    "team": "communication", "role": "lead",   "display": "Communication Manager"},
    {"name": "email_agent",              "team": "communication", "role": "member", "display": "Email Agent"},
    {"name": "slack_agent",              "team": "communication", "role": "member", "display": "Slack Agent"},
    {"name": "whatsapp_agent",           "team": "communication", "role": "member", "display": "WhatsApp Agent"},
    {"name": "calendar_agent",           "team": "communication", "role": "member", "display": "Calendar Agent"},
    {"name": "communication_memory",     "team": "communication", "role": "memory", "display": "Communication Memory"},
    # Content
    {"name": "content_manager",          "team": "content",       "role": "lead",   "display": "Content Manager"},
    {"name": "article_researcher",       "team": "content",       "role": "member", "display": "Article Researcher"},
    {"name": "article_writer",           "team": "content",       "role": "member", "display": "Article Writer"},
    {"name": "social_media_manager",     "team": "content",       "role": "member", "display": "Social Media Manager"},
    {"name": "content_memory",           "team": "content",       "role": "memory", "display": "Content Memory"},
    # Development
    {"name": "development_manager",      "team": "development",   "role": "lead",   "display": "Development Manager"},
    {"name": "backend_developer",        "team": "development",   "role": "member", "display": "Backend Developer"},
    {"name": "frontend_developer",       "team": "development",   "role": "member", "display": "Frontend Developer"},
    {"name": "database_manager",         "team": "development",   "role": "member", "display": "Database Manager"},
    {"name": "devops",                   "team": "development",   "role": "member", "display": "DevOps Engineer"},
    {"name": "development_memory",       "team": "development",   "role": "memory", "display": "Development Memory"},
    # Finance
    {"name": "finance_manager",          "team": "finance",       "role": "lead",   "display": "Finance Manager"},
    {"name": "accounting_agent",         "team": "finance",       "role": "member", "display": "Accounting Agent"},
    {"name": "crypto_agent",             "team": "finance",       "role": "member", "display": "Crypto Agent"},
    {"name": "finance_memory",           "team": "finance",       "role": "memory", "display": "Finance Memory"},
    # Grant Writing
    {"name": "grant_manager",            "team": "grant_writing", "role": "lead",   "display": "Grant Manager"},
    {"name": "grant_writer",             "team": "grant_writing", "role": "member", "display": "Grant Writer"},
    {"name": "budget_manager",           "team": "grant_writing", "role": "member", "display": "Budget Manager"},
    {"name": "project_manager",          "team": "grant_writing", "role": "member", "display": "Project Manager"},
    {"name": "grant_memory",             "team": "grant_writing", "role": "memory", "display": "Grant Memory"},
    # Research
    {"name": "research_manager",         "team": "research",      "role": "lead",   "display": "Research Manager"},
    {"name": "media_platform_researcher","team": "research",      "role": "member", "display": "Media Platform Researcher"},
    {"name": "media_program_researcher", "team": "research",      "role": "member", "display": "Media Program Researcher"},
    {"name": "street_bot_researcher",    "team": "research",      "role": "member", "display": "StreetBot Researcher"},
    {"name": "research_memory",          "team": "research",      "role": "memory", "display": "Research Memory"},
    # Scraping
    {"name": "scraping_manager",         "team": "scraping",      "role": "lead",   "display": "Scraping Manager"},
    {"name": "scraping_agent",           "team": "scraping",      "role": "member", "display": "Scraping Agent"},
    {"name": "scraper_memory",           "team": "scraping",      "role": "memory", "display": "Scraper Memory"},
]

# ── Team display names ───────────────────────────────────────────────────────
TEAM_DISPLAY = {
    "executive": "Executive",
    "communication": "Communication",
    "content": "Content",
    "development": "Development",
    "finance": "Finance",
    "grant_writing": "Grant Writing",
    "research": "Research",
    "scraping": "Scraping",
}

ROLE_DISPLAY = {
    "lead": "Team Lead",
    "member": "Specialist",
    "memory": "Memory Keeper",
}


def generate_memory_md(agent: dict) -> str:
    """Generate initialized MEMORY.md content for an agent."""
    name = agent["name"]
    display = agent["display"]
    team = TEAM_DISPLAY[agent["team"]]
    role = ROLE_DISPLAY[agent["role"]]
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""# {display} — Long-Term Memory

> Persistent knowledge store for `{name}` | {team} Team | {role}
> Initialized: {today}

---

## Identity

| Field | Value |
|-------|-------|
| Agent | `{name}` |
| Team | {team} |
| Role | {role} |
| Organization | Street Voices |
| Owner | Joel (joel@streetvoices.ca) |

---

## Key Facts

_Facts learned through operation. Updated automatically as the agent works._

<!-- Example format:
- [2026-03-10] Joel prefers email drafts to be concise, max 3 paragraphs
- [2026-03-11] Street Voices fiscal year runs April-March
-->

---

## Decisions & Preferences

_User preferences, standing decisions, and recurring instructions._

<!-- Example format:
- Always CC joel@streetvoices.ca on grant-related emails
- Prefer Canadian English spelling
- Budget reports should use CAD currency
-->

---

## Learned Patterns

_Patterns discovered through repeated interactions and outcomes._

<!-- Example format:
- City council emails typically get responses within 3-5 business days
- Grant applications submitted on Mondays tend to get reviewed faster
- Joel usually reviews content drafts in the morning (9-11am ET)
-->

---

## Contacts & Relationships

_Key people and organizations this agent interacts with._

<!-- Example format:
- Sarah Chen (sarah@example.org) — Grant officer at Ontario Arts Council
- Marcus Williams — City councillor, Ward 13
-->

---

## Domain Knowledge

_Accumulated expertise specific to this agent's function._

<!-- Example format:
- Ontario Arts Council deadlines: March 15 (Operating), June 1 (Project)
- Street Voices' 3 core programs: Community Radio, Youth Media Lab, Digital Archive
-->

---

## Error Log

_Mistakes made and lessons learned to avoid repeating them._

<!-- Example format:
- [2026-03-10] Sent email without Joel's approval — NEVER do this again
- [2026-03-11] Used wrong Airtable base ID — always verify base before querying
-->
"""


def generate_history_md(agent: dict) -> str:
    """Generate initialized HISTORY.md content for an agent."""
    display = agent["display"]
    name = agent["name"]
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""# {display} — Event History

> Chronological log for `{name}` | Searchable by date, topic, or outcome
> Initialized: {today}

---

## Log Format

Each entry follows this structure:
```
### [YYYY-MM-DD HH:MM] Action Type
- **Context**: Why this action was taken
- **Action**: What was done
- **Result**: Outcome or response
- **Follow-up**: Any pending items
```

---

## Event Log

_Events will be recorded here as the agent operates._

<!-- Entries are added chronologically, newest at the bottom -->
"""


def main():
    base_dir = Path.home() / ".nanobot" / "workspace" / "memory" / "agents"

    print(f"Creating {len(AGENTS)} agent memory directories...")
    print(f"Output: {base_dir}/{{agent_name}}/\n")

    created = 0
    existed = 0

    for agent in AGENTS:
        agent_dir = base_dir / agent["name"]
        agent_dir.mkdir(parents=True, exist_ok=True)

        memory_path = agent_dir / "MEMORY.md"
        history_path = agent_dir / "HISTORY.md"

        # Only write if files don't exist (preserve existing data)
        if not memory_path.exists():
            memory_path.write_text(generate_memory_md(agent))
            mem_status = "created"
        else:
            mem_status = "exists"

        if not history_path.exists():
            history_path.write_text(generate_history_md(agent))
            hist_status = "created"
        else:
            hist_status = "exists"

        if mem_status == "created" or hist_status == "created":
            created += 1
        else:
            existed += 1

        role_icon = {"lead": "\u2605", "memory": "\u25ce", "member": "\u25cf"}[agent["role"]]
        team_display = TEAM_DISPLAY[agent["team"]]
        print(f"  {role_icon} {agent['name']:30s} [{team_display:15s}] "
              f"MEMORY:{mem_status:7s} HISTORY:{hist_status}")

    # Summary
    leads = sum(1 for a in AGENTS if a["role"] == "lead")
    memories = sum(1 for a in AGENTS if a["role"] == "memory")
    members = sum(1 for a in AGENTS if a["role"] == "member")

    print(f"\n\u2713 {created} new directories created, {existed} already existed")
    print(f"  Total: {len(AGENTS)} agents ({leads} leads, {members} members, {memories} memory)")
    print(f"  Output: {base_dir}/")


if __name__ == "__main__":
    main()
