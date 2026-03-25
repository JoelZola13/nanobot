# Street Voices Platform — Setup Guide

This guide assumes you've never used a terminal before. Follow every step exactly.

---

## Step 1: Install Docker Desktop

Docker runs all our services in containers (think of it like a virtual computer inside your computer).

1. Go to https://www.docker.com/products/docker-desktop/
2. Click **Download for Mac** (or Windows)
3. Open the downloaded file and drag Docker to your Applications folder
4. Open **Docker Desktop** from your Applications
5. It will ask for permissions — click **Allow** / **OK** for everything
6. Wait until you see a green "Running" status in the Docker Desktop window

**Leave Docker Desktop running.** It needs to be open whenever you use the platform.

---

## Step 2: Open Terminal

- **Mac:** Press `Cmd + Space`, type `Terminal`, press Enter
- **Windows:** Press `Win + R`, type `cmd`, press Enter

You'll see a window with a blinking cursor. This is where you'll type commands.

---

## Step 3: Download the Project

Copy and paste this into Terminal, then press Enter:

```
git clone --recursive https://github.com/JoelZola13/nanobot.git
```

The `--recursive` part is important — it downloads the chat interface along with the main project.

If you get "git: command not found":
- **Mac:** A popup will ask to install developer tools. Click **Install**, wait, then try again.
- **Windows:** Download Git from https://git-scm.com/downloads and install it, then try again.

If you forgot `--recursive` or the `LibreChat/` folder is empty:
```
cd ~/nanobot
git submodule update --init --recursive
```

---

## Quick Setup (Alternative)

If you already have the secrets files from Joel, you can skip Steps 4-5 and run:

```
cd ~/nanobot
./setup.sh
```

It will check everything and start the platform for you. If something is missing, it will tell you what to do.

---

## Step 4: Set Up the Config Files

Joel will send you an encrypted zip file (`nanobot-secrets.zip`) and a password separately.

### 4a. Unzip the secrets

**Mac:** Double-click the zip file on your Desktop. It will ask for the password Joel sent you.

**Or in Terminal:**
```
cd ~/Desktop
unzip nanobot-secrets.zip
```
Enter the password when it asks.

You should now have 3 files: `.env.nanobot`, `librechat.env`, `config.json`

### 4b. Put the files where they belong

Copy and paste ALL of these lines into Terminal and press Enter:

```
cd ~/nanobot

cp deploy/docker-compose.override.yml LibreChat/docker-compose.override.yml
cp deploy/librechat.yaml LibreChat/librechat.yaml

cp ~/Desktop/.env.nanobot .env.nanobot
cp ~/Desktop/librechat.env LibreChat/.env

mkdir -p ~/.nanobot
cp ~/Desktop/config.json ~/.nanobot/config.json
```

---

## Step 5: Start the Platform

```
cd ~/nanobot/LibreChat
docker compose up -d
```

The first time you run this, it will download a lot of stuff (1-2 GB). This can take 5-15 minutes depending on your internet. You'll see progress bars.

When it's done, you'll see something like:
```
✔ Container nanobot-mongodb     Started
✔ Container nanobot-api         Started
✔ Container nanobot-librechat   Started
✔ Container nanobot-nginx       Started
...
```

---

## Step 6: Open the App

Open your browser and go to:

**http://localhost:3180**

1. Click **Sign up**
2. Create an account with your name and email
3. You're in!

---

## Using the Platform

Once you're logged in, you have access to:

- **40+ AI agents** — Grant Manager, CEO, Communication Manager, and more
- **110 tools** — web search, email, calendar, browser automation, social media
- **Grant Writer** — full grant application workspace
- **Tasks, News, Gallery, Groups** — community management tools

Just start chatting! Select an agent from the model dropdown, or use the default Street Bot.

---

## Starting and Stopping

### To start (every time you restart your computer):

1. Open **Docker Desktop** (wait for green "Running")
2. Open **Terminal**
3. Type:
   ```
   cd ~/nanobot/LibreChat
   docker compose up -d
   ```
4. Go to http://localhost:3180

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

## Troubleshooting

### "Cannot connect to the Docker daemon"
Docker Desktop isn't running. Open it from your Applications and wait for the green status.

### "Port 3180 already in use"
Something else is using that port. Close any other web servers, or restart your computer.

### The page loads but agents don't respond
The AI backend might need a moment to start. Wait 30 seconds and try again. If it still doesn't work, tell Joel — the auth token may have expired (only he can refresh it).

### "command not found: docker"
Docker Desktop isn't installed properly. Reinstall it from Step 1.

### Everything was working but now it's not
Try restarting:
```
cd ~/nanobot/LibreChat
docker compose down
docker compose up -d
```

### Still stuck?
Message Joel. Include what you see on screen (a screenshot helps).

---

## For Developers

<details>
<summary>Click to expand technical details</summary>

### Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| Nginx | nanobot-nginx | **3180** | Reverse proxy (main entry) |
| LibreChat | nanobot-librechat | internal | Chat UI + auth |
| Nanobot API | nanobot-api | 18790 | Agent engine + MCP tools |
| Paperclip | nanobot-paperclip | 3100 | Image generation |
| Paperclip Relay | nanobot-relay | 3050 | Dispatch |
| WhatsApp Bridge | nanobot-whatsapp | internal | WhatsApp messaging |
| MongoDB | nanobot-mongodb | 27018 | Database |
| Meilisearch | nanobot-meilisearch | internal | Search |
| PostgreSQL | nanobot-vectordb | internal | Vector DB |

### Architecture

```
Browser → nginx (3180)
            ├── / → LibreChat (chat UI)
            ├── /sbapi/v1 → nanobot-api (AI agents)
            ├── /STR/ → Paperclip
            └── /api/ → LibreChat + Paperclip
```

### Config Files

| File | Purpose |
|------|---------|
| `~/.nanobot/config.json` | API keys, MCP servers, channels, LLM provider |
| `LibreChat/.env` | LibreChat settings (port, auth, secrets) |
| `LibreChat/librechat.yaml` | Agent models, endpoints |
| `LibreChat/docker-compose.override.yml` | All service definitions |
| `.env.nanobot` | Nanobot container env vars |

### Useful Commands

```bash
docker compose logs -f nanobot-api     # API logs
docker compose logs -f                 # All logs
docker compose build nanobot-api       # Rebuild after code changes
docker compose restart nanobot-api     # Restart just the API
```

### LLM Provider

Uses OpenAI Codex OAuth (browser login, no API key). Token stored in `~/.nanobot/config.json`. When it expires, Joel re-auths:
```bash
.venv/bin/python -m nanobot provider login openai-codex
```

### Enabling RAG API (Optional)

Disabled by default (needs `OPENAI_API_KEY` for embeddings). To enable:

1. Add `OPENAI_API_KEY=sk-...` to `LibreChat/.env`
2. Remove `profiles: [rag]` from `rag_api` in `docker-compose.override.yml`
3. Run: `docker compose up -d rag_api`

</details>
