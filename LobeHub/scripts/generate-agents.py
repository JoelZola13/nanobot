#!/usr/bin/env python3
"""
Generate LobeHub agent marketplace JSON from Nanobot agent YAML definitions.

Reads:
  - nanobot/agents/teams/*/agents.yaml  (agent definitions)
  - nanobot/agents/teams/*/*.md         (system prompts)
  - LibreChat/librechat.yaml            (descriptions + icons)
  - static/avatars/*.svg                (avatar files)

Outputs:
  - LobeHub/agent-index/index.json      (marketplace index)
  - LobeHub/agent-index/agents/*.json   (individual agent files)
  - LobeHub/agent-index/avatars/*.svg   (copied avatars)
"""

import json
import shutil
import sys
from pathlib import Path

import yaml

# ─── Paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEAMS_DIR = REPO_ROOT / "nanobot" / "agents" / "teams"
LIBRECHAT_YAML = REPO_ROOT / "LibreChat" / "librechat.yaml"
AVATARS_DIR = REPO_ROOT / "static" / "avatars"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "agent-index"
AGENTS_OUT = OUTPUT_DIR  # flat structure — LobeHub expects {baseUrl}/{identifier}.json
AVATARS_OUT = OUTPUT_DIR / "avatars"

# ─── Team metadata ──────────────────────────────────────────────────
TEAM_TAGS = {
    "executive": ["executive", "leadership", "strategy"],
    "communication": ["communication", "messaging", "channels"],
    "content": ["content", "writing", "media"],
    "development": ["development", "engineering", "code"],
    "finance": ["finance", "accounting", "crypto"],
    "grant_writing": ["grants", "proposals", "funding"],
    "research": ["research", "analysis", "intelligence"],
    "scraping": ["scraping", "data-extraction", "web"],
}

TEAM_CATEGORY = {
    "executive": "general",
    "communication": "general",
    "content": "copywriting",
    "development": "programming",
    "finance": "general",
    "grant_writing": "copywriting",
    "research": "general",
    "scraping": "programming",
}

# ─── Opening messages / questions per agent ─────────────────────────
OPENING_DATA = {
    "auto": {
        "message": "Hi! I'm the Auto Router. I'll dispatch your request to the right specialist instantly.",
        "questions": [
            "Run the daily news pipeline",
            "Check my email",
            "What grants are coming up?",
        ],
    },
    "ceo": {
        "message": "Hey Joel. What are we working on today?",
        "questions": [
            "What's the status across all teams?",
            "Run the daily news pipeline",
            "Check my schedule for today",
        ],
    },
    "communication_manager": {
        "message": "I can coordinate across email, Slack, WhatsApp, and calendar. What do you need?",
        "questions": [
            "Check my unread emails",
            "Send a Slack message",
            "What meetings do I have today?",
        ],
    },
    "content_manager": {
        "message": "Ready to run the content pipeline. What stories should we cover today?",
        "questions": [
            "Run the daily news pipeline",
            "What articles are in progress?",
            "Find trending stories about housing policy",
        ],
    },
    "development_manager": {
        "message": "What are we building? I'll coordinate the dev team.",
        "questions": [
            "What's the current sprint status?",
            "Review the API server code",
            "Deploy the latest changes",
        ],
    },
    "finance_manager": {
        "message": "I can handle bookkeeping, crypto tracking, and financial reporting. What do you need?",
        "questions": [
            "What's our current financial position?",
            "Check crypto portfolio",
            "Generate an invoice",
        ],
    },
    "grant_manager": {
        "message": "I'll help with grants from research to submission. What opportunity are we looking at?",
        "questions": [
            "Find upcoming grant deadlines",
            "Start a new grant application",
            "Review our active grants",
        ],
    },
    "research_manager": {
        "message": "What do you need researched? I have specialists for media platforms, programs, and street-level intelligence.",
        "questions": [
            "Research media funding opportunities",
            "Analyze social media trends",
            "Get local community data",
        ],
    },
    "scraping_manager": {
        "message": "I can extract structured data from any website. What do you need scraped?",
        "questions": [
            "Scrape job listings from a site",
            "Extract data from a government page",
            "Monitor a website for changes",
        ],
    },
}


def load_librechat_specs() -> dict:
    """Load LibreChat modelSpecs for descriptions and icons."""
    if not LIBRECHAT_YAML.exists():
        print(f"Warning: {LIBRECHAT_YAML} not found, using defaults")
        return {}

    with open(LIBRECHAT_YAML) as f:
        config = yaml.safe_load(f)

    specs = {}
    for spec in config.get("modelSpecs", {}).get("list", []):
        name = spec.get("name", "")
        specs[name] = {
            "label": spec.get("label", ""),
            "description": spec.get("description", ""),
            "iconURL": spec.get("iconURL", ""),
        }
    return specs


def load_agents() -> list[dict]:
    """Load all agent definitions from YAML files."""
    agents = []

    # Add the auto router (not in YAML, it's synthetic)
    agents.append({
        "name": "auto",
        "team": "executive",
        "description": "Routes requests directly to the right team. Instant keyword-based dispatch with no CEO bottleneck.",
        "system_prompt": "You are the Auto Router. Analyze the user's request and route it to the most appropriate team agent. You dispatch instantly based on keywords and context.",
        "role": "lead",
    })

    for team_dir in sorted(TEAMS_DIR.iterdir()):
        if not team_dir.is_dir():
            continue
        yaml_file = team_dir / "agents.yaml"
        if not yaml_file.exists():
            continue

        team_name = team_dir.name
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        for agent in data.get("agents", []):
            name = agent["name"]
            # Load system prompt
            prompt_file = team_dir / agent.get("system_prompt", f"{name}.md")
            system_prompt = ""
            if prompt_file.exists():
                system_prompt = prompt_file.read_text().strip()

            agents.append({
                "name": name,
                "team": team_name,
                "description": agent.get("description", ""),
                "system_prompt": system_prompt,
                "role": agent.get("role", "member"),
            })

    return agents


def make_identifier(name: str) -> str:
    """Convert agent name to LobeHub identifier."""
    return f"streetvoices-{name.replace('_', '-')}"


# ─── Agent emoji avatars ─────────────────────────────────────────
AGENT_AVATARS = {
    # Executive
    "auto": "⚡",
    "ceo": "👔",
    "executive_memory": "🧠",
    # Communication
    "communication_manager": "📡",
    "email_agent": "📧",
    "calendar_agent": "📅",
    "slack_agent": "💬",
    "whatsapp_agent": "📱",
    "communication_memory": "🗂️",
    # Content
    "content_manager": "📰",
    "article_researcher": "🔍",
    "article_writer": "✍️",
    "social_media_agent": "📣",
    "content_memory": "📚",
    # Development
    "development_manager": "🛠️",
    "backend_developer": "⚙️",
    "frontend_developer": "🎨",
    "database_manager": "🗄️",
    "devops": "🚀",
    "development_memory": "💾",
    # Finance
    "finance_manager": "💰",
    "accounting_agent": "🧾",
    "budget_manager": "📊",
    "crypto_agent": "₿",
    "invoice_agent": "🧮",
    "finance_memory": "🏦",
    # Grant Writing
    "grant_manager": "🏆",
    "grant_writer": "📝",
    "grant_researcher": "🔬",
    "grant_budget": "💵",
    "grant_memory": "📋",
    # Research
    "research_manager": "🧪",
    "media_researcher": "📺",
    "program_researcher": "🎓",
    "street_bot": "🤖",
    "research_memory": "🔎",
    # Scraping
    "scraping_manager": "🕷️",
    "web_scraper": "🌐",
    "scraping_memory": "📦",
}


def get_avatar(name: str) -> str:
    """Get avatar URL for an agent. Falls back to emoji if no SVG exists."""
    svg_path = AVATARS_DIR / (name + ".svg")
    if svg_path.exists():
        return f"http://localhost:8381/a/{name}.svg"
    # Fallback to emoji
    return AGENT_AVATARS.get(name, "🤖")


def build_agent_json(agent: dict, librechat_specs: dict) -> dict:
    """Build a LobeHub agent JSON definition."""
    name = agent["name"]
    team = agent["team"]
    identifier = make_identifier(name)
    model_id = f"agent/{name}"

    # Get LibreChat spec for enrichment
    spec = librechat_specs.get(model_id, {})
    description = spec.get("description") or agent["description"]
    label = spec.get("label") or name.replace("_", " ").title()

    # Opening data
    opening = OPENING_DATA.get(name, {})

    tags = TEAM_TAGS.get(team, [team])
    if agent["role"] == "lead":
        tags = tags + ["team-lead"]

    return {
        "author": "Street Voices",
        "config": {
            "systemRole": agent["system_prompt"],
            "model": model_id,
            "params": {
                "temperature": 0.7,
            },
            **({"openingMessage": opening["message"]} if opening.get("message") else {}),
            **({"openingQuestions": opening["questions"]} if opening.get("questions") else {}),
        },
        "homepage": "https://streetvoices.ca",
        "identifier": identifier,
        "meta": {
            "avatar": get_avatar(name),
            "tags": tags,
            "title": label,
            "description": description,
            "category": TEAM_CATEGORY.get(team, "general"),
        },
        "createdAt": "2026-03-11",
        "schemaVersion": 1,
    }


def build_index(agents_json: list[dict]) -> dict:
    """Build the marketplace index.json."""
    # Collect all unique tags
    all_tags = set()
    for agent in agents_json:
        all_tags.update(agent["meta"]["tags"])

    # Index entries are summary (no systemRole)
    index_agents = []
    for agent in agents_json:
        entry = {
            "author": agent["author"],
            "createdAt": agent["createdAt"],
            "homepage": agent["homepage"],
            "identifier": agent["identifier"],
            "meta": {**agent["meta"]},
            "schemaVersion": agent["schemaVersion"],
        }
        index_agents.append(entry)

    return {
        "schemaVersion": 1,
        "agents": index_agents,
        "tags": sorted(all_tags),
    }


def copy_avatars():
    """Copy SVG avatars to the output directory."""
    AVATARS_OUT.mkdir(parents=True, exist_ok=True)
    if not AVATARS_DIR.exists():
        print(f"Warning: {AVATARS_DIR} not found, skipping avatar copy")
        return 0

    count = 0
    for svg in AVATARS_DIR.glob("*.svg"):
        shutil.copy2(svg, AVATARS_OUT / svg.name)
        count += 1
    return count


def main():
    print("=" * 60)
    print("Street Voices AI — LobeHub Agent Marketplace Generator")
    print("=" * 60)

    # Load data
    librechat_specs = load_librechat_specs()
    agents = load_agents()
    print(f"\nLoaded {len(agents)} agents from {len(TEAM_TAGS)} teams")

    # Generate individual agent JSONs (flat — same dir as index)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agents_json = []

    for agent in agents:
        agent_json = build_agent_json(agent, librechat_specs)
        agents_json.append(agent_json)

        # Write individual file at root level (LobeHub expects {baseUrl}/{id}.json)
        # Also write locale-specific version ({id}.en-US.json) which LobeHub fetches
        for suffix in ("", ".en-US"):
            out_file = AGENTS_OUT / f"{agent_json['identifier']}{suffix}.json"
            with open(out_file, "w") as f:
                json.dump(agent_json, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(agents_json)} agent JSON files")

    # Generate index (both plain and locale-specific)
    index = build_index(agents_json)
    for filename in ("index.json", "index.en-US.json"):
        index_file = OUTPUT_DIR / filename
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Generated index.json + index.en-US.json with {len(index['agents'])} agents and {len(index['tags'])} tags")

    # Copy avatars
    avatar_count = copy_avatars()
    print(f"Copied {avatar_count} avatar SVGs")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
