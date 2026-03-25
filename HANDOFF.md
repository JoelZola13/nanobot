# Nanobot Platform — Claude Handoff Guide

You are helping a teammate set up and develop on the **Street Voices / Nanobot** platform. This document gives you full context.

---

## Architecture Overview

This is a Docker-based platform with these services:

| Service | Container | Port | What it does |
|---------|-----------|------|--------------|
| Nginx | nanobot-nginx | **3180** | Reverse proxy (main entry point) |
| LibreChat | nanobot-librechat | internal | Chat UI + user auth (React) |
| Nanobot API | nanobot-api | 18790 | AI agent engine (Python/FastAPI) |
| Paperclip | nanobot-paperclip | 3100 | Project/task management |
| Paperclip Relay | nanobot-relay | 3050 | Heartbeat + dispatch |
| WhatsApp Bridge | nanobot-whatsapp | internal | WhatsApp messaging |
| MongoDB | nanobot-mongodb | 27018 | Chat history + users |
| Meilisearch | nanobot-meilisearch | internal | Full-text search |
| PostgreSQL | nanobot-vectordb | internal | Vector DB + Paperclip data |

```
Browser → nginx (3180)
            ├── /           → LibreChat (chat UI)
            ├── /sbapi/v1   → nanobot-api (AI agents)
            ├── /STR/       → Paperclip (tasks/projects)
            └── /api/       → LibreChat API + Paperclip
```

## Repo Structure

```
nanobot/                          # Main repo (github.com/JoelZola13/nanobot)
├── LibreChat/                    # Git SUBMODULE (github.com/JoelZola13/LibreChat)
│   ├── docker-compose.yml        # Base compose (from LibreChat upstream)
│   ├── docker-compose.override.yml  # Custom services (copied from deploy/)
│   ├── librechat.yaml            # Agent config (copied from deploy/)
│   ├── .env                      # Secrets (from secrets zip)
│   ├── nginx-unified.conf        # Nginx routing config
│   └── client/src/components/streetbot/  # All custom Street Voices pages
├── nanobot/                      # Python backend
│   ├── agents/teams/             # 40+ agent definitions (markdown files)
│   ├── api_server.py             # FastAPI server
│   ├── providers/                # LLM providers (OpenAI Codex, etc.)
│   └── agent/tools/              # Tool implementations
├── bridge/                       # Node.js WhatsApp bridge + Paperclip relay
├── deploy/                       # Config templates (source of truth)
│   ├── docker-compose.override.yml
│   ├── librechat.yaml
│   ├── .env.nanobot.example
│   ├── librechat.env.example
│   └── config.json.example
├── Dockerfile                    # Builds nanobot-api + whatsapp-bridge
├── Dockerfile.paperclip          # Builds Paperclip service
├── setup.sh                      # One-command setup script
├── login.sh                      # Codex OAuth login script
├── SETUP.md                      # Full setup guide for non-coders
└── .env.nanobot                  # Container env vars (from secrets zip)
```

## Critical: How Auth Works

There are TWO separate auth systems. Do not confuse them:

### 1. User Login (LibreChat accounts)
- Email + password signup at `http://localhost:3180`
- Local to each machine, stored in MongoDB
- NO SSO/Casdoor needed — ignore any OpenID config in the .env
- `OPENID_AUTO_REDIRECT=false` means it just shows normal login

### 2. AI Agent Auth (OpenAI Codex OAuth)
- This is how the AI agents call OpenAI's API (GPT-5/Codex models)
- Uses browser-based ChatGPT login, NOT a traditional API key
- Token stored at `~/.nanobot/codex-token.json`
- Managed by `oauth_cli_kit` library with env var `OAUTH_CLI_KIT_TOKEN_PATH`
- The Docker container reads it via volume mount: `~/.nanobot` → `/root/.nanobot`
- To authenticate: run `./login.sh` from the repo root
- Each teammate can use their own ChatGPT account

### If the bot doesn't respond:
1. Check if `~/.nanobot/codex-token.json` exists
2. If not, run `./login.sh`
3. Then: `cd LibreChat && docker compose restart nanobot-api`

## Setup Flow (What Should Already Be Done)

1. `git clone --recursive https://github.com/JoelZola13/nanobot.git`
2. Secrets zip extracted → `.env.nanobot`, `librechat.env`, `config.json`
3. `./setup.sh` (copies deploy configs, creates dirs, starts Docker)
4. `./login.sh` (Codex OAuth — opens browser for ChatGPT login)
5. Sign up at `http://localhost:3180`

## Config Files (Where Secrets Live)

| File | Location | Purpose |
|------|----------|---------|
| `config.json` | `~/.nanobot/config.json` | MCP servers, API keys (Brave Search, Groq, etc.) |
| `codex-token.json` | `~/.nanobot/codex-token.json` | Codex OAuth token (created by login.sh) |
| `.env.nanobot` | `~/nanobot/.env.nanobot` | Container env vars (Groq key, etc.) |
| `librechat.env` | `~/nanobot/LibreChat/.env` | JWT secrets, MongoDB URI, OpenID config |
| `librechat.yaml` | `~/nanobot/LibreChat/librechat.yaml` | Agent models, endpoints |
| `docker-compose.override.yml` | `~/nanobot/LibreChat/docker-compose.override.yml` | All custom service definitions |

**Source of truth for non-secret configs:** `deploy/` directory in the repo.
**Secrets are NOT in git** — they come from an encrypted zip file Joel sends separately.

## Common Issues

### "Agents don't respond" / "Bot is silent"
→ Missing or expired Codex token. Run `./login.sh`, then restart nanobot-api.

### Missing sidebar icons
→ Icons are in `LibreChat/client/public/public/images/sidebar-icons/`. Make sure submodule is fully cloned: `git submodule update --init --recursive`

### Container keeps restarting
→ Check logs: `cd LibreChat && docker compose logs nanobot-api`

### "Cannot connect to Docker daemon"
→ Docker Desktop isn't running. Open it first.

### Port 3180 already in use
→ Something else is on that port. `lsof -i :3180` to find it.

### LibreChat folder is empty
→ Submodule wasn't cloned. Run: `cd ~/nanobot && git submodule update --init --recursive`

### Casdoor / OpenID errors
→ Ignore. `OPENID_AUTO_REDIRECT=false` means SSO is disabled. Normal email/password login works.

## Making Changes

### Frontend (React)
Files: `LibreChat/client/src/components/streetbot/`
Rebuild: `cd LibreChat && docker compose up -d --build api`

### Backend (Python agents/tools)
Files: `nanobot/agents/teams/`, `nanobot/agent/tools/`, `nanobot/api_server.py`
Rebuild: `cd LibreChat && docker compose build nanobot-api && docker compose up -d nanobot-api`

### Agent definitions
Files: `nanobot/agents/teams/<department>/<agent>.md`
These are markdown files — edit and restart nanobot-api.

### Getting updates from Joel
```bash
cd ~/nanobot
git pull --recurse-submodules
cp deploy/docker-compose.override.yml LibreChat/docker-compose.override.yml
cp deploy/librechat.yaml LibreChat/librechat.yaml
cd LibreChat
docker compose up -d --build
```

## DO NOT

- Do not modify `LibreChat/docker-compose.yml` (that's upstream LibreChat)
- Do not commit secrets to git (`.env.nanobot`, `librechat.env`, `config.json`, `codex-token.json`)
- Do not run `docker compose` from the repo root — always `cd LibreChat` first
- Do not touch the Casdoor/OpenID config — it's unused
- Do not delete `~/.nanobot/` — it has auth tokens, WhatsApp session, and workspace data
