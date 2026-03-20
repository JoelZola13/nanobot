#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Street Voices — Unified Platform Startup
# ═══════════════════════════════════════════════════════════════════
#
# Starts all services:
#   1. OrbStack (Docker runtime)
#   2. Docker Compose (LibreChat, LobeHub, SV Social, Casdoor, DBs)
#   3. PM2 services (nanobot-api, whatsapp-bridge, paperclip, paperclip-relay)
#
# Usage: ./scripts/start-unified.sh
# ═══════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════"
echo "  Street Voices — Unified Platform"
echo "═══════════════════════════════════════════════════"
echo ""

# 1. Ensure OrbStack is running
echo "▸ Starting OrbStack..."
open -a OrbStack 2>/dev/null || true
sleep 2

# 2. Start Docker services
echo "▸ Starting Docker services..."
docker compose -f docker-compose.unified.yml up -d

# 3. Start PM2 services (nanobot-api, paperclip, etc.)
echo "▸ Starting PM2 services..."
pm2 start ecosystem.config.cjs --only nanobot-api 2>/dev/null || pm2 restart nanobot-api
pm2 start ecosystem.config.cjs --only whatsapp-bridge 2>/dev/null || true
pm2 start ecosystem.config.cjs --only paperclip 2>/dev/null || true
pm2 start ecosystem.config.cjs --only paperclip-relay 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All services started!"
echo ""
echo "  Chat:            http://localhost:3180"
echo "  Marketplace:     http://localhost:3181"
echo "  Social:          http://localhost:3182"
echo "  Auth (Casdoor):  http://localhost:8380"
echo "  Nanobot API:     http://localhost:18790"
echo "  Mission Control: Click 'Mission Control' in nav bar"
echo ""
echo "  Login:  joel@streetvoices.ca / street2020"
echo "═══════════════════════════════════════════════════"
