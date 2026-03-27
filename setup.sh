#!/bin/bash
# Street Voices Platform — One-Click Setup
# Clone the repo, run this script. That's it.
#
# Usage: cd ~/nanobot && ./setup.sh
set -e

echo "=== Street Voices Platform Setup ==="
echo ""

# Check Docker is running
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not running."
  echo "Open Docker Desktop and wait for the green 'Running' status, then try again."
  exit 1
fi

# Initialize LibreChat submodule if not already done
if [ ! -f "LibreChat/docker-compose.yml" ]; then
  echo "→ Downloading LibreChat (this may take a minute)..."
  git submodule update --init --recursive
fi

# ── Auto-create ALL config files from defaults ──
# Teammates don't need to copy anything manually.

echo "→ Setting up configuration..."

# LibreChat override (defines all Street Voices services)
cp deploy/docker-compose.override.yml LibreChat/docker-compose.override.yml

# LibreChat YAML (agent models, endpoints, MCP servers)
cp deploy/librechat.yaml LibreChat/librechat.yaml

# Nginx config (routes for gallery, social, agents)
cp deploy/nginx-unified.conf LibreChat/nginx-unified.conf

# Street Voices frontend patches (gallery submit, etc.)
if [ -d "deploy/streetbot-patches" ]; then
  cp -r deploy/streetbot-patches/gallery/* LibreChat/client/src/components/streetbot/gallery/ 2>/dev/null || true
fi

# LibreChat .env (OAuth, DB, UI settings) — only create if missing
if [ ! -f "LibreChat/.env" ]; then
  cp deploy/librechat.env.example LibreChat/.env
fi

# Nanobot env vars — only create if missing
if [ ! -f ".env.nanobot" ]; then
  cp deploy/.env.nanobot.example .env.nanobot
fi

# Nanobot home directory + config
NANOBOT_DIR="${NANOBOT_HOME:-$HOME/.nanobot}"
mkdir -p "$NANOBOT_DIR/whatsapp-auth"

if [ ! -f "$NANOBOT_DIR/config.json" ]; then
  cp deploy/config.json.example "$NANOBOT_DIR/config.json"
fi

# ── Build and start everything ──
echo "→ Building and starting all services..."
echo "  (First run builds custom frontend + downloads images — may take 15-25 min)"
cd LibreChat
docker compose up -d --build

echo ""
echo "=== Setup Complete ==="
echo ""
echo "  Platform:    http://localhost:3180"
echo "  OAuth Admin: http://localhost:8380  (joel@streetvoices.ca / street2020)"
echo ""
echo "  → Click 'Sign in with Street Voices' to log in."
echo "  → Or click 'Sign up' to create a local account."
echo ""

# ── Check agent health ──
echo "  Checking agent status..."
sleep 5
HEALTH=$(curl -s http://localhost:18790/health 2>/dev/null || echo '{}')
AGENTS=$(echo "$HEALTH" | grep -o '"agents":[0-9]*' | grep -o '[0-9]*' || echo "0")
TOKEN=$(echo "$HEALTH" | grep -o '"codex_token":"[^"]*"' | grep -o ':"[^"]*"' | tr -d ':"' || echo "unknown")

if [ "$AGENTS" -gt 0 ] && [ "$TOKEN" = "ok" ]; then
  echo "  ✓ $AGENTS agents loaded, Codex token valid — agents will respond!"
elif [ "$AGENTS" -gt 0 ]; then
  echo "  ✓ $AGENTS agents loaded"
  echo "  ✗ Codex token: $TOKEN — agents won't respond yet."
  echo ""
  echo "  To fix: get codex-token.json from Joel, then:"
  echo "    cp <path-to>/codex-token.json ~/.nanobot/codex-token.json"
  echo "    cd ~/nanobot/LibreChat && docker compose restart nanobot-api"
else
  echo "  ✗ Agents not loaded yet (API may still be starting — wait 30s and check)"
  echo "    curl http://localhost:18790/health"
fi
echo ""
