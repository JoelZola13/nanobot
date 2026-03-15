# LobeHub Street Voices Community — Implementation Plan

## Overview

Deploy LobeHub as a community-facing AI platform for the 37 Nanobot agents on **port 3181**, connected to the existing nanobot API server on port 18790. This runs alongside LibreChat (port 3180) without interference, sharing the same backend.

---

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│  LibreChat (3180)    │     │  LobeHub (3181)      │
│  (existing, no       │     │  (new deployment)    │
│   changes)           │     │                      │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         │   OPENAI_PROXY_URL        │
         └──────────┬───────────────┘
                    │
          ┌─────────▼──────────┐
          │  Nanobot API       │
          │  (port 18790)      │
          │  OpenAI-compatible │
          │  37 agents as      │
          │  /v1/models        │
          └────────────────────┘
```

Both frontends hit the same API. The API is already transferrable — it speaks the OpenAI protocol.

---

## Stack (Docker Compose)

| Service        | Image                      | Port  | Purpose                          |
|----------------|----------------------------|-------|----------------------------------|
| lobehub        | `lobehub/lobehub`          | 3181  | Main LobeHub app                 |
| postgresql     | `pgvector/pgvector:pg17`   | 5433  | Database (PGVector for RAG)      |
| rustfs         | `chrislusf/seaweedfs`      | 8333  | S3-compatible file storage       |
| casdoor        | `casbin/casdoor`           | 8380  | Auth (SSO, login, user mgmt)     |
| redis          | `redis:7-alpine`           | 6380  | Session cache                    |
| agent-index    | `nginx:alpine`             | 8381  | Serves custom agent marketplace  |

All ports chosen to avoid conflicts with existing services (5432, 6379, 8000-8003 already in use).

Container prefix: `streetvoices-` (distinct from LibreChat's `nanobot-` prefix).

---

## Step-by-Step Implementation

### Step 1: Create Docker Compose Stack

**File: `LobeHub/docker-compose.yml`**

Full server-side deployment with:
- PostgreSQL 17 + PGVector (port 5433, not 5432 which is taken)
- RustFS/SeaweedFS for S3 storage (port 8333)
- Redis for sessions (port 6380, not 6379 which is taken)
- Casdoor for auth (port 8380)
- Nginx serving the custom agent marketplace index (port 8381)
- LobeHub connected to all services

### Step 2: Environment Configuration

**File: `LobeHub/.env`**

Critical environment variables:
```env
# ─── API Connection (same backend as LibreChat) ───
OPENAI_PROXY_URL=http://host.docker.internal:18790/v1
OPENAI_API_KEY=nanobot

# Model list — all 37 agents + base models
# Format: +model_id=Display Name
OPENAI_MODEL_LIST=+agent/auto=Auto Router,+agent/ceo=CEO,+agent/executive_memory=Executive Memory,+agent/security_compliance=Security & Compliance,+agent/communication_manager=Communication Manager,+agent/email_agent=Email Agent,+agent/slack_agent=Slack Agent,+agent/whatsapp_agent=WhatsApp Agent,+agent/calendar_agent=Calendar Agent,+agent/communication_memory=Communication Memory,+agent/content_manager=Content Manager,+agent/article_researcher=Article Researcher,+agent/article_writer=Article Writer,+agent/social_media_manager=Social Media Manager,+agent/content_memory=Content Memory,+agent/development_manager=Development Manager,+agent/backend_developer=Backend Developer,+agent/frontend_developer=Frontend Developer,+agent/database_manager=Database Manager,+agent/devops=DevOps Engineer,+agent/development_memory=Development Memory,+agent/finance_manager=Finance Manager,+agent/accounting_agent=Accounting Agent,+agent/crypto_agent=Crypto Agent,+agent/finance_memory=Finance Memory,+agent/grant_manager=Grant Manager,+agent/grant_writer=Grant Writer,+agent/budget_manager=Budget Manager,+agent/project_manager=Project Manager,+agent/grant_memory=Grant Memory,+agent/research_manager=Research Manager,+agent/media_platform_researcher=Media Platform Researcher,+agent/media_program_researcher=Media Program Researcher,+agent/street_bot_researcher=Street Bot Researcher,+agent/research_memory=Research Memory,+agent/scraping_manager=Scraping Manager,+agent/scraping_agent=Scraping Agent,+agent/scraper_memory=Scraper Memory,+gpt-5.4=GPT-5.4,+gpt-5=GPT-5

DEFAULT_AGENT_CONFIG=model=agent/auto

# ─── Database ───
DATABASE_URL=postgresql://lobehub:lobehub_password@postgresql:5432/lobehub

# ─── S3 Storage (RustFS/SeaweedFS) ───
S3_ENDPOINT=http://rustfs:8333
S3_ACCESS_KEY_ID=lobehub
S3_SECRET_ACCESS_KEY=lobehub_s3_secret
S3_BUCKET=lobehub

# ─── Auth (Casdoor) ───
AUTH_CASDOOR_ISSUER=http://casdoor:8000
CASDOOR_ENDPOINT=http://localhost:8380
NEXT_AUTH_SECRET=<32-char-random-secret>
NEXT_AUTH_URL=http://localhost:3181/api/auth

# ─── Custom Agent Marketplace ───
AGENTS_INDEX_URL=http://agent-index:80/index.json

# ─── Feature Flags ───
FEATURE_FLAGS=+market,+plugins,+knowledge_base,+think,+artifacts,+tts

# ─── TTS/STT (reuse nanobot endpoints) ───
OPENAI_TTS_API_KEY=nanobot
OPENAI_STT_API_KEY=nanobot
```

### Step 3: Custom Agent Marketplace (AGENTS_INDEX_URL)

This is the key innovation. LobeHub's `AGENTS_INDEX_URL` env var points to a JSON index of agents that populate the "Discover" marketplace tab. We build a custom index with all 37 agents.

**How it works:**
1. A lightweight nginx container serves static JSON files
2. `AGENTS_INDEX_URL=http://agent-index:80/index.json` points LobeHub at our marketplace
3. Users see all 37 agents organized by team in the Discover tab
4. Users "install" agents to create local copies they can customize

**Agent JSON format** (LobeHub's actual schema from `lobe-chat-agents` repo):

```json
{
  "author": "Street Voices",
  "config": {
    "systemRole": "<full system prompt from .md file>",
    "model": "agent/ceo",
    "params": {
      "temperature": 0.7
    },
    "openingMessage": "Hi, I'm the CEO agent. How can I help coordinate today?",
    "openingQuestions": [
      "What's the status of all teams?",
      "Run the daily news pipeline",
      "Check my schedule for today"
    ]
  },
  "homepage": "https://streetvoices.ca",
  "identifier": "streetvoices-ceo",
  "meta": {
    "avatar": "http://host.docker.internal:18790/avatars/ceo.svg",
    "tags": ["executive", "coordinator", "delegation"],
    "title": "CEO",
    "description": "Central coordinator and entry point. Delegates tasks to the right team manager and handles cross-team coordination.",
    "category": "general"
  },
  "createdAt": "2026-03-10",
  "schemaVersion": 1
}
```

**Index file** (`index.json`) — summary format that LobeHub consumes:

```json
{
  "schemaVersion": 1,
  "agents": [
    {
      "author": "Street Voices",
      "createdAt": "2026-03-10",
      "homepage": "https://streetvoices.ca",
      "identifier": "streetvoices-ceo",
      "meta": {
        "avatar": "http://host.docker.internal:18790/avatars/ceo.svg",
        "title": "CEO",
        "description": "Central coordinator...",
        "tags": ["executive", "coordinator"],
        "category": "general"
      },
      "schemaVersion": 1
    }
  ],
  "tags": ["executive", "communication", "content", "development", "finance", "grant-writing", "research", "scraping"]
}
```

**All 37 agents will be created as individual JSON files** with full `config.systemRole` (pulled from each agent's `.md` system prompt file), organized by team.

### Step 4: Avatar Hosting

Two options (we'll use both for reliability):
1. **Nanobot API** — Already serves SVGs at `http://localhost:18790/avatars/`. Agent JSON references `http://host.docker.internal:18790/avatars/<name>.svg`
2. **Agent index nginx** — Copy SVGs into the same nginx volume, serve at `http://agent-index:80/avatars/<name>.svg`

### Step 5: Casdoor Auth Setup

- Deploy Casdoor with the shared PostgreSQL instance (separate `casdoor` database)
- Configure an OIDC application for LobeHub:
  - Client ID + Client Secret
  - Redirect URI: `http://localhost:3181/api/auth/callback/casdoor`
- Create initial admin user for Joel
- Casdoor init data (`casdoor/init_data.json`) pre-configures the org + app

### Step 6: MCP Tools Integration (Native)

LobeHub has **first-class native MCP support** — this is their primary tool integration path (legacy plugins are being deprecated). Three transport types:

| Transport | Config | Use Case |
|-----------|--------|----------|
| stdio     | `command` + `args` + `env` | Local servers |
| SSE       | `type: "sse"` + `url` | Remote servers |
| Streamable HTTP | `type: "streamable-http"` + `url` | Modern remote |

Configure the same MCP servers currently in LibreChat via the LobeHub UI or JSON import:

```json
{
  "mcpServers": {
    "postiz": {
      "command": "node",
      "args": ["/app/scripts/postiz-mcp.js"],
      "env": {
        "POSTIZ_API_KEY": "...",
        "POSTIZ_API_URL": "http://host.docker.internal:4007/api"
      }
    },
    "google-calendar": {
      "command": "npx",
      "args": ["-y", "@cocal/google-calendar-mcp"]
    },
    "google-drive": {
      "command": "npx",
      "args": ["-y", "@piotr-agier/google-drive-mcp"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "..." }
    },
    "airtable": {
      "command": "npx",
      "args": ["-y", "airtable-mcp-server"],
      "env": { "AIRTABLE_API_KEY": "..." }
    }
  }
}
```

Users can also browse LobeHub's built-in MCP Marketplace (10,000+ servers) for additional tools.

### Step 7: Speech (TTS/STT)

Reuse the same TTS/STT endpoints from the nanobot API. LobeHub supports OpenAI-compatible TTS natively:
- TTS: `http://host.docker.internal:18790/v1/audio/speech`
- STT: `http://host.docker.internal:18790/v1/audio/transcriptions`

### Step 8: Branding & Customization

- Custom title: "Street Voices AI Community"
- Default model: `agent/auto` (Auto Router)
- Feature flags: marketplace, plugins, knowledge base, thinking mode, artifacts, TTS
- Custom agent marketplace with team-organized agents

---

## Files to Create

```
LobeHub/
├── docker-compose.yml              # Full stack
├── .env                            # Environment configuration
├── casdoor/
│   └── init_data.json              # Casdoor app + org config
├── agent-index/                    # Custom agent marketplace (served by nginx)
│   ├── index.json                  # Agent marketplace index (summary)
│   ├── avatars/                    # Copy of static/avatars/ SVGs
│   │   ├── ceo.svg
│   │   ├── email_agent.svg
│   │   └── ... (39 SVG files)
│   └── agents/                     # Full agent definitions
│       ├── streetvoices-auto-router.json
│       ├── streetvoices-ceo.json
│       ├── streetvoices-executive-memory.json
│       ├── streetvoices-security-compliance.json
│       ├── streetvoices-communication-manager.json
│       ├── streetvoices-email-agent.json
│       ├── streetvoices-slack-agent.json
│       ├── streetvoices-whatsapp-agent.json
│       ├── streetvoices-calendar-agent.json
│       ├── streetvoices-communication-memory.json
│       ├── streetvoices-content-manager.json
│       ├── streetvoices-article-researcher.json
│       ├── streetvoices-article-writer.json
│       ├── streetvoices-social-media-manager.json
│       ├── streetvoices-content-memory.json
│       ├── streetvoices-development-manager.json
│       ├── streetvoices-backend-developer.json
│       ├── streetvoices-frontend-developer.json
│       ├── streetvoices-database-manager.json
│       ├── streetvoices-devops.json
│       ├── streetvoices-development-memory.json
│       ├── streetvoices-finance-manager.json
│       ├── streetvoices-accounting-agent.json
│       ├── streetvoices-crypto-agent.json
│       ├── streetvoices-finance-memory.json
│       ├── streetvoices-grant-manager.json
│       ├── streetvoices-grant-writer.json
│       ├── streetvoices-budget-manager.json
│       ├── streetvoices-project-manager.json
│       ├── streetvoices-grant-memory.json
│       ├── streetvoices-research-manager.json
│       ├── streetvoices-media-platform-researcher.json
│       ├── streetvoices-media-program-researcher.json
│       ├── streetvoices-street-bot-researcher.json
│       ├── streetvoices-research-memory.json
│       ├── streetvoices-scraping-manager.json
│       ├── streetvoices-scraping-agent.json
│       └── streetvoices-scraper-memory.json
└── scripts/
    └── generate-agents.py          # Script to convert YAML agents → LobeHub JSON
```

---

## Implementation Script: `generate-agents.py`

To avoid manually creating 37+ JSON files, we'll write a Python script that:

1. Reads all `agents.yaml` files from `nanobot/agents/teams/*/`
2. Reads each agent's `.md` system prompt
3. Reads the LibreChat `modelSpecs` from `librechat.yaml` for descriptions/icons
4. Generates LobeHub-format JSON for each agent
5. Generates the `index.json` marketplace summary
6. Copies SVG avatars to the `agent-index/avatars/` directory

This ensures agents stay in sync — re-run the script after adding new agents.

---

## Key Design Decisions

1. **Server-side mode** (not client-side) — enables multi-user community features, persistent data, proper auth
2. **`AGENTS_INDEX_URL` custom marketplace** — All 37 agents pre-loaded in the Discover tab, users can install and customize them
3. **Casdoor for auth** — LobeHub's recommended self-hosted auth provider
4. **Separate ports for ALL services** — no conflicts with existing LibreChat/MongoDB/Redis stack
5. **Same API backend** — `OPENAI_PROXY_URL` points to the existing nanobot API, zero duplication
6. **MCP-first tool strategy** — LobeHub's native MCP support is the primary path, legacy plugins deprecated
7. **Container name prefix**: `streetvoices-` to distinguish from `nanobot-` LibreChat containers
8. **Agent generation script** — Automated conversion from YAML sources keeps marketplace in sync

---

## Future: Combining LibreChat + LobeHub

Since both use the same nanobot API backend:
- Agent definitions, tools, and handoffs work identically in both
- MCP tools are shared (both support native MCP)
- TTS/STT endpoints are shared
- When ready to combine, can retire one frontend and keep the other
- The API layer (`api_server.py`) doesn't need changes — it already serves any OpenAI-compatible client
- LobeHub's built-in MCP Marketplace (10,000+ tools) gives access to tools beyond what LibreChat offers

---

## Verification Steps

After deployment:
1. `docker compose up -d` in `LobeHub/` directory
2. Visit `http://localhost:3181` — should see LobeHub login
3. Create account via Casdoor (http://localhost:8380)
4. Open Discover tab — verify all 37 agents appear with correct avatars and descriptions
5. Install an agent (e.g., Auto Router) and test a conversation
6. Test agent/auto routing (should dispatch to correct team)
7. Test TTS/STT
8. Configure MCP tools and test
9. Verify LibreChat on port 3180 still works independently
