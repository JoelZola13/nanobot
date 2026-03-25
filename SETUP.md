# Nanobot Platform Setup

Everything runs in Docker. One command to start the full stack.

## Prerequisites

- **Docker Desktop** or **OrbStack** (macOS)
- **Git**
- A copy of the nanobot config (`~/.nanobot/config.json`) — get this from the team lead

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/JoelZola13/nanobot.git
cd nanobot

# 2. Copy deployment configs into place
cp deploy/docker-compose.override.yml LibreChat/docker-compose.override.yml
cp deploy/librechat.yaml LibreChat/librechat.yaml

# 3. Set up secrets (get real values from team lead)
#    Option A: Team lead sends you the actual files
cp /path/from/team-lead/.env.nanobot .env.nanobot
cp /path/from/team-lead/librechat.env LibreChat/.env
mkdir -p ~/.nanobot
cp /path/from/team-lead/config.json ~/.nanobot/config.json

#    Option B: Copy examples and fill in values yourself
cp deploy/.env.nanobot.example .env.nanobot
cp deploy/librechat.env.example LibreChat/.env
cp deploy/config.json.example ~/.nanobot/config.json
#    Then edit each file and replace the placeholder values

# 4. Start everything
cd LibreChat
docker compose up -d

# 5. Open the app
open http://localhost:3180
```

## Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| Nginx | nanobot-nginx | **3180** | Unified reverse proxy (main entry point) |
| LibreChat | nanobot-librechat | 3080 (internal) | Chat UI + auth |
| Nanobot API | nanobot-api | 18790 | AI agent engine + MCP tools |
| Paperclip | nanobot-paperclip | 3100 | Project management |
| Paperclip Relay | nanobot-relay | 3050 | Agent dispatch |
| WhatsApp Bridge | nanobot-whatsapp | 3001 (internal) | WhatsApp integration |
| MongoDB | nanobot-mongodb | 27018 | Chat/user data |
| Meilisearch | nanobot-meilisearch | 7700 (internal) | Search index |
| PostgreSQL | nanobot-vectordb | 5432 (internal) | Vector DB + Paperclip DB |
| RAG API | nanobot-rag-api | 8000 (internal) | Retrieval-augmented generation (**disabled by default**) |

## Files You Need

The repo includes example templates in `deploy/`. Three files contain secrets that aren't committed to git:

| File | Example Template | What It Contains |
|------|-----------------|------------------|
| `.env.nanobot` | `deploy/.env.nanobot.example` | Groq API key, Postiz config |
| `LibreChat/.env` | `deploy/librechat.env.example` | JWT secrets, OAuth, MongoDB URI |
| `~/.nanobot/config.json` | `deploy/config.json.example` | LLM provider, email, Slack, MCP servers, Brave Search |

**Easiest path:** Ask Joel for the real files. He'll send them securely.

**DIY path:** Copy the examples, fill in your own API keys.

## Creating Your Account

1. Go to http://localhost:3180
2. Click **Sign up** and create an account with your email
3. You're in — all 40+ agents and 110 tools are shared through the backend

Everyone shares the same nanobot backend. No individual API keys needed.

## LLM Provider (OpenAI Codex OAuth)

The backend uses **OpenAI Codex OAuth** — a browser-based login, no API keys.
The token is stored in `~/.nanobot/config.json` on the host machine.

When the token expires, the team lead re-authenticates:
```bash
cd /path/to/nanobot
.venv/bin/python -m nanobot provider login openai-codex
```
No one else needs to do this.

### `LibreChat/.env` (LibreChat config)

Already in the repo. Contains:
- MongoDB connection
- JWT secrets
- OAuth settings (Google, Facebook)

## Common Commands

```bash
# Start everything
cd LibreChat && docker compose up -d

# Stop everything
docker compose down

# View logs
docker compose logs -f nanobot-api     # API logs
docker compose logs -f nanobot-nginx   # Nginx logs
docker compose logs -f                 # All logs

# Rebuild after code changes
docker compose build nanobot-api       # Rebuild API image
docker compose up -d --force-recreate nanobot-api  # Restart with new image

# Rebuild frontend after client changes
cd client && NODE_OPTIONS='--max-old-space-size=4096' npx vite build --outDir dist
# No container restart needed — dist is bind-mounted

# Check status
docker compose ps
```

## Architecture

```
Browser → nginx (3180)
            ├── / → LibreChat (chat UI)
            ├── /sbapi/v1 → nanobot-api (AI agents)
            ├── /STR/ → Paperclip (project mgmt)
            ├── /api/ → split between LibreChat + Paperclip
            ├── /relay/ → Paperclip Relay
            └── /social → SV Social (external)
```

The nanobot API is the brain — it runs the agent loop, connects to 100+ MCP tools, and streams responses back through LibreChat.

## Troubleshooting

**502 Bad Gateway on `/sbapi/v1/*`**
Nginx cached a stale container IP. Run: `docker exec nanobot-nginx nginx -s reload`

**nanobot-api restarting**
Check logs: `docker compose logs nanobot-api`. Common causes:
- Missing `~/.nanobot/config.json`
- Missing Python deps (rebuild: `docker compose build nanobot-api`)

**nanobot-rag-api restarting (if enabled)**
Needs `OPENAI_API_KEY` in `LibreChat/.env` for embeddings. Disabled by default — see below to enable.

## Enabling RAG API (Optional)

The RAG API (document search/embeddings) is disabled by default because it requires a paid OpenAI API key. To enable:

1. Add `OPENAI_API_KEY=sk-...` to `LibreChat/.env`
2. Edit `LibreChat/docker-compose.override.yml` — remove the `profiles: [rag]` block from the `rag_api` service
3. Add `rag_api` back to the `api` service's `depends_on` list
4. Run:
   ```bash
   docker compose up -d rag_api
   ```

**Port conflicts**
Check: `lsof -iTCP -sTCP:LISTEN -nP | grep <port>`
Key ports: 3180, 18790, 3100, 3050, 27018
