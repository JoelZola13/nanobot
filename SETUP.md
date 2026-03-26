# Street Voices Platform — Complete Setup Guide

This guide walks you through everything from scratch. No coding experience needed.

---

## What You'll Need Before Starting

- A **Mac** (Intel or Apple Silicon) or **Windows 10/11** computer
- At least **16 GB of RAM** (8 GB works but will be slow)
- At least **10 GB of free disk space**
- A decent internet connection (you'll download ~2-3 GB on first setup)
- The **secrets zip file** and **password** from Joel (he'll send these separately)

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

## Step 4: Download the Project

Copy and paste this into Terminal, then press Enter:

```
git clone --recursive https://github.com/JoelZola13/nanobot.git
```

This will take 1-3 minutes. You'll see progress messages.

When it's done, move into the project folder:

```
cd nanobot
```

**If you see an error about `--recursive`** or the download seems too quick (under 10 seconds), run:

```
git submodule update --init --recursive
```

This downloads the chat interface separately. Wait for it to finish.

---

## Step 5: Set Up the Secret Config Files

Joel will send you two things separately (for security):
1. An encrypted zip file called `nanobot-secrets.zip` (via email or Slack)
2. A password to unlock it (via a different channel — text, call, etc.)

### 5a. Download and unzip the secrets

Save `nanobot-secrets.zip` to your **Desktop**.

**Mac — double-click method:**
1. Double-click the zip file on your Desktop
2. Enter the password Joel gave you
3. You should now see 4 files on your Desktop: `.env.nanobot`, `librechat.env`, `config.json`, `codex-token.json`

**Mac — Terminal method (if double-click doesn't ask for password):**
```
cd ~/Desktop
unzip nanobot-secrets.zip
```
Enter the password when prompted.

**Windows:**
1. Right-click the zip file → **Extract All**
2. Enter the password Joel gave you
3. You should see 4 files: `.env.nanobot`, `librechat.env`, `config.json`, `codex-token.json`

**Can't see the files?** Files starting with a dot (`.env.nanobot`) are hidden by default.
- **Mac:** In Finder, press `Cmd + Shift + .` to show hidden files
- **Windows:** In File Explorer, click **View** → check **Hidden items**

### 5b. Put the config files where they belong

Go back to Terminal and paste ALL of these lines at once, then press Enter:

```
cd ~/nanobot

cp deploy/docker-compose.override.yml LibreChat/docker-compose.override.yml
cp deploy/librechat.yaml LibreChat/librechat.yaml

cp ~/Desktop/.env.nanobot .env.nanobot
cp ~/Desktop/librechat.env LibreChat/.env

mkdir -p ~/.nanobot/whatsapp-auth
cp ~/Desktop/config.json ~/.nanobot/config.json
cp ~/Desktop/codex-token.json ~/.nanobot/codex-token.json
```

**No errors?** Great — move to the next step.

> **What is `codex-token.json`?** This is the AI authentication token that lets the agents respond. Without it, you can log in to the app but the bots won't answer. If agents stop responding in the future, Joel will send an updated token file — just replace `~/.nanobot/codex-token.json` and restart: `cd ~/nanobot/LibreChat && docker compose restart nanobot-api`

**If you see "No such file or directory":** Make sure the zip file was saved to your Desktop, and that you unzipped it there. The 3 files must be directly on the Desktop, not inside a subfolder.

---

## Step 6: Build and Start the Platform

You have two options:

### Option A: One-command setup (recommended)

```
cd ~/nanobot
./setup.sh
```

This checks everything and starts the platform automatically.

### Option B: Manual start

```
cd ~/nanobot/LibreChat
docker compose up -d --build
```

**The first time takes 15-25 minutes.** Docker is building the application from source — compiling the website, installing dependencies, downloading base images. You'll see lots of output scrolling by. This is normal.

On future starts (after reboot, etc.), it will be much faster (under 30 seconds).

When it's done, you'll see something like:

```
✔ Container nanobot-mongodb     Started
✔ Container nanobot-vectordb    Started
✔ Container nanobot-api         Started
✔ Container nanobot-librechat   Started
✔ Container nanobot-paperclip   Started
✔ Container nanobot-nginx       Started
...
```

### Verify everything is running

```
cd ~/nanobot/LibreChat
docker compose ps
```

You should see **8-9 containers** with status `Up` or `running`. If any show `restarting` or `exited`, see Troubleshooting below.

---

## Step 7: Open the App

Open your web browser (Chrome, Safari, Firefox — any works) and go to:

**http://localhost:3180**

1. Click **Sign up** (top right)
2. Enter your name, email, and create a password
3. Click **Create Account**
4. You're in!

**Note:** This account is local to your computer only. Everyone creates their own account.

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
   - And more...

### Navigate the Platform

Use the left sidebar and top navigation to access:

- **Chat** — the main AI conversation interface
- **Grant Writer** — dedicated grant writing workspace
- **Tasks** — project management board
- **Groups** — team collaboration spaces
- **Gallery** — media and image management
- **News** — content feed
- **Database** — data browser

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
cp deploy/docker-compose.override.yml LibreChat/docker-compose.override.yml
cp deploy/librechat.yaml LibreChat/librechat.yaml
cd LibreChat
docker compose up -d --build
```

This pulls the latest code, copies the updated configs, and rebuilds any changed services.

---

## Troubleshooting

### "Cannot connect to the Docker daemon"
Docker Desktop isn't running. Open it from Applications and wait for the green whale icon.

### "git: command not found"
Git isn't installed. Go back to Step 2.

### "Port 3180 already in use"
Something else is using that port. Either close it, or restart your computer. Then try again.

### The clone was fast but LibreChat/ folder is empty
You forgot `--recursive`. Run:
```
cd ~/nanobot
git submodule update --init --recursive
```

### "No such file or directory" when copying config files
The secrets files aren't on your Desktop, or they're inside a subfolder. Find the 3 files (`.env.nanobot`, `librechat.env`, `config.json`) and make sure they're directly on your Desktop.

### Docker build fails with "out of memory" or is extremely slow
Docker needs more RAM. Open **Docker Desktop** → **Settings** (gear icon) → **Resources** → set **Memory** to at least **4 GB** (6 GB recommended). Click **Apply & Restart**.

### The page loads but agents don't respond
The AI backend needs a moment after startup. Wait 30 seconds and try again.

If it still doesn't work, the auth token may have expired. Message Joel — only he can refresh it.

### A container keeps restarting
Check what's wrong:
```
cd ~/nanobot/LibreChat
docker compose logs nanobot-api
```
Replace `nanobot-api` with the name of the container that's restarting. Send the last 20 lines to Joel.

### Everything was working but now it's broken
Try a full restart:
```
cd ~/nanobot/LibreChat
docker compose down
docker compose up -d --build
```

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
│   │   └── teams/        # 40+ agent configs (markdown files)
│   ├── api_server.py     # FastAPI server (port 18790)
│   └── agent/tools/      # Tool implementations (email, web, browser, etc.)
├── bridge/               # Node.js — WhatsApp bridge + Paperclip relay
├── LibreChat/            # Git submodule — customized chat frontend (React)
│   └── client/src/components/streetbot/  # All Street Voices pages
├── deploy/               # Docker config templates
├── static/               # Agent avatars
├── Dockerfile            # Builds nanobot-api + whatsapp-bridge + relay
├── Dockerfile.paperclip  # Builds Paperclip (project management)
└── setup.sh              # One-command setup script
```

### Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| Nginx | nanobot-nginx | **3180** | Reverse proxy (main entry point) |
| LibreChat | nanobot-librechat | internal | Chat UI + user auth (React) |
| Nanobot API | nanobot-api | 18790 | Agent engine + all MCP tools (Python/FastAPI) |
| Paperclip | nanobot-paperclip | 3100 | Project/task management |
| Paperclip Relay | nanobot-relay | 3050 | Heartbeat + dispatch |
| WhatsApp Bridge | nanobot-whatsapp | internal | WhatsApp messaging (Baileys) |
| MongoDB | nanobot-mongodb | 27018 | Chat history + user database |
| Meilisearch | nanobot-meilisearch | internal | Full-text search |
| PostgreSQL | nanobot-vectordb | internal | Vector DB + Paperclip data |

### Architecture

```
Browser → nginx (3180)
            ├── /           → LibreChat (chat UI)
            ├── /sbapi/v1   → nanobot-api (AI agents)
            ├── /STR/       → Paperclip (tasks/projects)
            └── /api/       → LibreChat API + Paperclip
```

### Config Files

| File | Purpose |
|------|---------|
| `~/.nanobot/config.json` | API keys, MCP servers, LLM provider token |
| `LibreChat/.env` | LibreChat settings (auth secrets, DB connection) |
| `LibreChat/librechat.yaml` | Agent models, endpoints, MCP servers |
| `LibreChat/docker-compose.override.yml` | All custom service definitions |
| `.env.nanobot` | Nanobot container environment variables |

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

Agent definitions are markdown files in `nanobot/agents/teams/<department>/`:

1. Create a new `.md` file following the pattern of existing agents
2. The agent will be automatically available in the next API restart
3. Add it to `deploy/librechat.yaml` under `models.default` to show in the UI dropdown

### Useful Commands

```bash
# View logs (most useful for debugging)
docker compose logs -f nanobot-api        # API/agent logs
docker compose logs -f nanobot-librechat  # Frontend logs
docker compose logs -f                    # All services

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

The platform uses **OpenAI Codex OAuth** — a browser-based login that provides API access without a traditional API key. The token is stored in `~/.nanobot/codex-token.json`.

When it expires (agents stop responding), Joel re-authenticates:
```bash
.venv/bin/python -m nanobot provider login openai-codex
```

Then sends an updated `config.json` to the team.

### Enabling RAG API (Optional)

The RAG (Retrieval-Augmented Generation) API is disabled by default because it requires a paid OpenAI API key for embeddings.

To enable:
1. Add `OPENAI_API_KEY=sk-...` to `LibreChat/.env`
2. Remove the `profiles: [rag]` block from `rag_api` in `LibreChat/docker-compose.override.yml`
3. Run: `docker compose up -d rag_api`

</details>
