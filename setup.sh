#!/bin/bash
# Street Voices Platform — One-Click Setup
# Run this after cloning: ./setup.sh
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

# Auto-create config files from examples if missing
if [ ! -f ".env.nanobot" ]; then
  echo "→ Creating .env.nanobot from example..."
  cp deploy/.env.nanobot.example .env.nanobot
fi

if [ ! -f "LibreChat/.env" ]; then
  echo "→ Creating LibreChat/.env from example..."
  cp deploy/librechat.env.example LibreChat/.env
fi

NANOBOT_DIR="${NANOBOT_HOME:-$HOME/.nanobot}"
mkdir -p "$NANOBOT_DIR/whatsapp-auth"

if [ ! -f "$NANOBOT_DIR/config.json" ]; then
  echo "→ Creating ~/.nanobot/config.json from example..."
  cp deploy/config.json.example "$NANOBOT_DIR/config.json"
fi

# Copy deploy configs into LibreChat
echo "→ Setting up LibreChat configuration..."
cp deploy/docker-compose.override.yml LibreChat/docker-compose.override.yml
cp deploy/librechat.yaml LibreChat/librechat.yaml

# Build and start the platform
echo "→ Building and starting all services..."
echo "  (First run builds custom frontend + downloads images — may take 15-25 min)"
cd LibreChat
docker compose up -d --build

echo ""
echo "=== Setup Complete ==="
echo ""
echo "  Platform:  http://localhost:3180"
echo "  OAuth/SSO: http://localhost:8380  (admin: joel@streetvoices.ca / street2020)"
echo ""
echo "  Click 'Sign in with Street Voices' to log in via OAuth."
echo "  Or click 'Sign up' to create a local account."
echo ""
echo "  To add teammates: open http://localhost:8380, log in as admin,"
echo "  go to Users → Add User, then share their login with them."
