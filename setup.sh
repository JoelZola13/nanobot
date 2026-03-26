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

# Check for required config files
MISSING=0

if [ ! -f ".env.nanobot" ]; then
  echo "MISSING: .env.nanobot"
  MISSING=1
fi

if [ ! -f "LibreChat/.env" ]; then
  echo "MISSING: LibreChat/.env"
  MISSING=1
fi

if [ ! -d "${NANOBOT_HOME:-$HOME/.nanobot}" ] || [ ! -f "${NANOBOT_HOME:-$HOME/.nanobot}/config.json" ]; then
  echo "MISSING: ~/.nanobot/config.json"
  MISSING=1
fi

if [ $MISSING -eq 1 ]; then
  echo ""
  echo "Some config files are missing. Ask Joel for the nanobot-secrets.zip file,"
  echo "then run these commands:"
  echo ""
  echo "  cp <path-to>/.env.nanobot .env.nanobot"
  echo "  cp <path-to>/librechat.env LibreChat/.env"
  echo "  mkdir -p ~/.nanobot"
  echo "  cp <path-to>/config.json ~/.nanobot/config.json"
  echo ""
  echo "Then run this script again."
  exit 1
fi

# Ensure required directories exist
mkdir -p "${NANOBOT_HOME:-$HOME/.nanobot}/whatsapp-auth"

# Copy deploy configs into LibreChat
echo "→ Setting up LibreChat configuration..."
cp deploy/docker-compose.override.yml LibreChat/docker-compose.override.yml
cp deploy/librechat.yaml LibreChat/librechat.yaml

# Build and start the platform
echo "→ Building and starting all services..."
echo "  (First run builds custom frontend + downloads images — may take 10-20 min)"
cd LibreChat
docker compose up -d --build

echo ""
echo "=== Setup Complete ==="
echo "Open http://localhost:3180 in your browser"
echo "Click 'Sign up' to create your account"
