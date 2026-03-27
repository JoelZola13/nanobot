#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/lobehub_app2"

APP_URL=http://localhost:3900 \
LOBE_PORT=3900 \
./node_modules/.bin/next dev -p 3900 --hostname 127.0.0.1 --webpack
