# Street Voices Platform — Complete Setup Guide

This guide walks you through everything from scratch. No coding experience needed.

---

## What You'll Need Before Starting

- A **Mac** (Intel or Apple Silicon) or **Windows 10/11** or **Linux** computer
- At least **16 GB of RAM** (8 GB works but will be slow)
- At least **10 GB of free disk space**
- A decent internet connection (you'll download ~2-3 GB on first setup)
- A free **ChatGPT account** (for AI agent access)

---

## Step 1: Install Docker Desktop

Docker runs the entire platform inside containers on your computer. Think of it as a mini server that lives on your laptop.

### Mac

1. Go to https://www.docker.com/products/docker-desktop/
2. Click **Download for Mac**
   - If you have a newer Mac (2020 or later): choose **Apple Silicon**
   - If you have an older Mac: choose **Intel**
   - Not sure? Click the Apple menu () → **About This Mac** → look for "Chip" (Apple Silicon) or "Processor" (Intel)
3. Open the downloaded `.dmg` file
4. Drag the Docker icon to your **Applications** folder
5. Open **Docker Desktop** from Applications
6. It will ask for your password and permissions — click **Allow** / **OK** for everything
7. Wait until you see a **green whale icon** in your menu bar (top of screen) — this means Docker is running

### Windows

1. Go to https://www.docker.com/products/docker-desktop/
2. Click **Download for Windows**
3. Run the installer — accept all defaults
4. Restart your computer when it asks
5. Open **Docker Desktop** from the Start menu
6. Wait until the Docker icon in your system tray (bottom-right) shows "running"

**Important:** Docker Desktop must be running whenever you use the platform. Leave it open.

---

## Step 2: Install Git

Git is what downloads and manages the project code.

### Mac

1. Press `Cmd + Space`, type **Terminal**, press Enter
2. Type this and press Enter:
   ```
   git --version
   ```
3. If Git is already installed, you'll see a version number — skip to Step 3
4. If not, macOS will pop up asking to install **Command Line Developer Tools** — click **Install**
5. Wait for it to finish (can take a few minutes)
6. Try `git --version` again to confirm it worked

### Windows

1. Go to https://git-scm.com/downloads
2. Click **Download for Windows**
3. Run the installer — accept all defaults (just keep clicking Next)
4. When done, close and reopen your terminal

---

## Step 3: Open Terminal

You'll type all the remaining commands in here. Keep this window open.

- **Mac:** Press `Cmd + Space`, type **Terminal**, press Enter
- **Windows:** Press the Windows key, type **Command Prompt**, press Enter

You'll see a window with a blinking cursor. This is where you paste commands.

**Tip:** To paste in Terminal:
- **Mac:** `Cmd + V`
- **Windows:** Right-click in the window

---

## Step 4: Download and Set Up the Platform

Copy and paste these three commands into Terminal. This downloads everything and builds the platform:

```
git clone --recursive https://github.com/JoelZola13/nanobot.git
cd nanobot
./setup.sh
```

**The first time takes 15-25 minutes.** Docker is building the application from source — compiling the website, installing dependencies, downloading base images. You'll see lots of output scrolling by. This is normal.

When it's done, you'll see:

```
=== Setup Complete ===

  Platform:    http://localhost:3180
  OAuth Admin: http://localhost:8380
```

**If you see an error about `--recursive`** or the `LibreChat/` folder is empty, run:

```
git submodule update --init --recursive
./setup.sh
```

---

## Step 5: Connect the AI Agents

The platform is running, but the AI agents need to connect to GPT-5.4. Run:

```
./login.sh
```

This opens a browser window. **Sign in with any ChatGPT account** (a free account works).

After signing in, the script saves a token and restarts the API automatically. You only need to do this **once** — the token auto-refreshes for weeks.

### Verify agents are working

```
curl http://localhost:18790/health
```

Look for:
- `"agents": 38` — agent definitions loaded
- `"codex_token": "ok"` — AI will respond
- `"status": "ok"` — everything working

---

## Step 6: Open the App

Open your web browser (Chrome, Safari, Firefox — any works) and go to:

**http://localhost:3180**

### Logging in

You have two options:

1. **OAuth login (recommended):** Click **"Sign in with Street Voices"** — this uses the centralized auth system.

2. **Local account:** Click **Sign up**, enter your name, email, and create a password.

---

## Using the Platform

### Chat with AI Agents

The main screen is a chat interface. By default, you're talking to **Street Bot** (the auto-router that picks the best agent for your question).

To talk to a specific agent:
1. Click the **model dropdown** at the top of the chat
2. Choose an agent:
   - **CEO** — strategic decisions, org overview
   - **Grant Manager** — grant research, writing, applications
   - **Communication Manager** — emails, newsletters, social media
   - **Content Manager** — articles, blog posts, media
   - **Research Manager** — web research, reports
   - **Finance Manager** — budgets, financial planning
   - **Development Manager** — technical projects, code
   - **Scraping Manager** — web data collection
   - And 30+ more specialized agents...

### Browse All Agents

Go to **http://localhost:3180/agents** to see the full Agent Marketplace — all 38+ agents organized by team with descriptions.

### Navigate the Platform

Use the left sidebar and top navigation to access:

- **Chat** — the main AI conversation interface
- **Agent Marketplace** — browse and chat with all 38+ agents
- **Messages** — team messaging and presence
- **Social Media** — social media management dashboard
- **Grant Writer** — dedicated grant writing workspace
- **Tasks** — project management board
- **Groups** — team collaboration spaces
- **Gallery** — media and image management
- **News** — content feed and article editor
- **Calendar** — scheduling and events
- **Documents** — file management and sharing
- **Database** — data browser
- **Academy** — learning and training
- **Directory** — services and listings
- **Jobs** — job board and applications

---

## Starting and Stopping

### To start (after restarting your computer):

1. Open **Docker Desktop** — wait for the green status
2. Open **Terminal**
3. Run:
   ```
   cd ~/nanobot/LibreChat
   docker compose up -d
   ```
4. Open http://localhost:3180

### To stop:

```
cd ~/nanobot/LibreChat
docker compose down
```

### To check if it's running:

```
cd ~/nanobot/LibreChat
docker compose ps
```

---

## Getting Updates

When Joel pushes updates to the codebase:

```
cd ~/nanobot
git pull --recurse-submodules
./setup.sh
```

The setup script copies the updated configs and rebuilds any changed services.

---

## Troubleshooting

### "Cannot connect to the Docker daemon"
Docker Desktop isn't running. Open it from Applications and wait for the green whale icon.

### "git: command not found"
Git isn't installed. Go back to Step 2.

### "Port 3180 already in use"
Something else is using that port. Run `docker compose down` first, or restart your computer. Then try again.

### "Port 3100 already in use" (Paperclip)
Another process is on port 3100. Fix it with a clean restart:
```
cd ~/nanobot/LibreChat
docker compose down
docker compose up -d
```

### The clone was fast but LibreChat/ folder is empty
You forgot `--recursive`. Run:
```
cd ~/nanobot
git submodule update --init --recursive
```

### Docker build fails with "out of memory" or is extremely slow
Docker needs more RAM. Open **Docker Desktop** → **Settings** (gear icon) → **Resources** → set **Memory** to at least **4 GB** (6 GB recommended). Click **Apply & Restart**.

### Agents don't respond (most common issue)

This means the Codex OAuth token is missing or expired. Check:

```
curl http://localhost:18790/health
```

If `codex_token` says `"error"` or `"missing"`:

1. Run the login (any ChatGPT account works):
   ```
   cd ~/nanobot
   ./login.sh
   ```
2. Sign in with your ChatGPT account in the browser
3. The script auto-restarts the API
4. Check again: `curl http://localhost:18790/health` — look for `"codex_token": "ok"`

The token auto-refreshes for weeks, so you only need to do this once.

### Agent Marketplace shows "No agents found"
The nanobot-api needs to be running and healthy. Check:
```
cd ~/nanobot/LibreChat
docker compose logs nanobot-api | head -50
```
Look for "Multi-agent system ready: 38 agents across 8 teams". If you see "Multi-agent system not available", pull the latest code and rebuild.

### A container keeps restarting
Check what's wrong:
```
cd ~/nanobot/LibreChat
docker compose logs nanobot-api --tail 20
```
Replace `nanobot-api` with the name of the container that's restarting. Send the last 20 lines to Joel.

### Everything was working but now it's broken
Try a full restart:
```
cd ~/nanobot/LibreChat
docker compose down
docker compose up -d --build
```

### UID/GID warnings in Docker output
You'll see warnings like `The "UID" variable is not set. Defaulting to a blank string.` — **these are harmless**. Ignore them.

### Still stuck?
Message Joel with:
- What you typed in Terminal
- What error you see (a screenshot is ideal)
- Which step you're on

---

## For Developers — Building on the Platform

<details>
<summary>Click to expand (for people writing code)</summary>

### Project Structure

```
nanobot/
├── nanobot/              # Python backend — AI agents, tools, API server
│   ├── agents/           # Agent definitions and orchestration
│   │   └── teams/        # 38 agent configs across 8 teams
│   ├── api_server.py     # Starlette/Uvicorn server (port 18790)
│   └── agent/tools/      # Tool implementations (email, web, browser, etc.)
├── social/               # SV Social — Next.js messaging & presence app
├── bridge/               # Node.js — WhatsApp bridge + Paperclip relay
├── LibreChat/            # Git submodule — customized chat frontend (React)
│   └── client/src/components/streetbot/  # All Street Voices pages (40+)
├── LobeHub/              # Casdoor SSO config + init data
├── deploy/               # Docker config templates (auto-copied by setup.sh)
├── static/               # Agent avatars
├── Dockerfile            # Builds nanobot-api + whatsapp-bridge + relay
├── Dockerfile.paperclip  # Builds Paperclip (project management)
├── login.sh              # Codex OAuth login (any teammate can run this)
└── setup.sh              # One-command setup script
```

### Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| Nginx | nanobot-nginx | **3180** | Reverse proxy (main entry point) |
| LibreChat | nanobot-librechat | internal | Chat UI + user auth (React) |
| Nanobot API | nanobot-api | 18790 | Agent engine + all MCP tools (Python) |
| SV Social | nanobot-social | 3182, 3183 | Real-time messaging + presence (Next.js) |
| Casdoor | nanobot-casdoor | 8380 | OAuth/SSO provider |
| Redis | nanobot-redis | 6380 | Event bus for cross-service messaging |
| Paperclip | nanobot-paperclip | 3100 | Project/task management |
| Paperclip Relay | nanobot-relay | 3050 | Heartbeat + dispatch |
| WhatsApp Bridge | nanobot-whatsapp | internal | WhatsApp messaging (Baileys) |
| MongoDB | nanobot-mongodb | 27018 | Chat history + user database |
| Meilisearch | nanobot-meilisearch | internal | Full-text search |
| PostgreSQL | nanobot-vectordb | internal | Vector DB + Paperclip + Casdoor + Social data |

### Architecture

```
Browser → nginx (3180)
            ├── /           → LibreChat (chat UI)
            ├── /sbapi/v1   → nanobot-api (AI agents)
            ├── /social     → SV Social (messaging)
            ├── /STR/       → Paperclip (tasks/projects)
            └── /api/       → LibreChat API + Paperclip
```

### Config Files

| File | Purpose |
|------|---------|
| `~/.nanobot/config.json` | API keys, MCP servers, LLM provider config (auto-created from defaults) |
| `~/.nanobot/codex-token.json` | Codex OAuth token (created by `./login.sh`) |
| `LibreChat/.env` | LibreChat settings — auth, DB, OAuth (auto-created by `setup.sh`) |
| `LibreChat/librechat.yaml` | Agent models, endpoints, MCP servers (auto-copied by `setup.sh`) |
| `LibreChat/docker-compose.override.yml` | All custom service definitions (auto-copied by `setup.sh`) |
| `.env.nanobot` | Nanobot container environment variables (auto-created by `setup.sh`) |

### Authentication

The platform uses two layers of auth:

1. **User login (Casdoor OpenID):** Teammates log in via "Sign in with Street Voices" which redirects to Casdoor (port 8380). Pre-configured admin: `joel@streetvoices.ca` / `street2020`. Add new users via the Casdoor admin panel.

2. **AI provider (Codex OAuth):** All agents call GPT-5.4 via OpenAI Codex. Any teammate with a ChatGPT account runs `./login.sh` once — the token auto-refreshes for weeks. No need to share token files.

### Making Frontend Changes

The frontend is a React app inside `LibreChat/`. Street Voices custom pages are in `LibreChat/client/src/components/streetbot/`.

To make changes and see them:

```bash
cd ~/nanobot/LibreChat

# Edit files in client/src/components/streetbot/

# Rebuild and restart the frontend container
docker compose up -d --build api
```

The build takes 3-5 minutes (it recompiles the entire React app).

### Making Backend Changes

The backend is Python in `nanobot/`. Changes to agent definitions, tools, or the API server:

```bash
cd ~/nanobot/LibreChat

# Edit files in ../nanobot/

# Rebuild and restart the API container
docker compose build nanobot-api
docker compose up -d nanobot-api
```

### Adding a New Agent

Agent definitions are YAML + markdown files in `nanobot/agents/teams/<department>/`:

1. Add the agent to `agents.yaml` in the team folder
2. Create a `.md` file with the system prompt
3. The agent will be automatically available after API restart
4. Add it to `deploy/librechat.yaml` under `models.default` to show in the UI dropdown

### Useful Commands

```bash
# View logs (most useful for debugging)
docker compose logs -f nanobot-api        # API/agent logs
docker compose logs -f nanobot-librechat  # Frontend logs
docker compose logs -f nanobot-social     # Social/messaging logs
docker compose logs -f                    # All services

# Check agent health + token status
curl http://localhost:18790/health

# Restart a single service
docker compose restart nanobot-api

# Rebuild everything from scratch
docker compose down
docker compose up -d --build

# Check which containers are running
docker compose ps

# Open a shell inside a container (advanced)
docker compose exec nanobot-api bash
```

### LLM Provider

The platform uses **OpenAI Codex OAuth** — a browser-based login that provides API access without a traditional API key. The token is stored in `~/.nanobot/codex-token.json` and **auto-refreshes** for weeks.

Any teammate can authenticate independently:
```bash
cd ~/nanobot
./login.sh
```

This opens a browser, the teammate signs in with any ChatGPT account, and the token is saved and auto-refreshed. No need to share token files between teammates.

### Enabling RAG API (Optional)

The RAG (Retrieval-Augmented Generation) API is disabled by default because it requires a paid OpenAI API key for embeddings.

To enable:
1. Add `OPENAI_API_KEY=sk-...` to `LibreChat/.env`
2. Remove the `profiles: [rag]` block from `rag_api` in `LibreChat/docker-compose.override.yml`
3. Run: `docker compose up -d rag_api`

</details>
