#!/bin/bash
# Login to OpenAI Codex — run this ONCE to authenticate the AI agents.
# Opens a browser window for ChatGPT sign-in.
#
# Usage: cd ~/nanobot && ./login.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Make sure ~/.nanobot exists
mkdir -p "$HOME/.nanobot"

# Store the token where Docker can read it
export OAUTH_CLI_KIT_TOKEN_PATH="$HOME/.nanobot/codex-token.json"

echo "🔑 Logging into OpenAI Codex..."
echo "   A browser window will open — sign in with your ChatGPT account."
echo ""

# Install nanobot temporarily if needed, run login, cleanup
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -m nanobot provider login openai-codex
else
    # No venv — install into a temp venv
    echo "   Setting up (first time only)..."
    python3 -m venv /tmp/nanobot-login-venv
    /tmp/nanobot-login-venv/bin/pip install -q -e .
    /tmp/nanobot-login-venv/bin/python -m nanobot provider login openai-codex
    rm -rf /tmp/nanobot-login-venv
fi

echo ""
if [ -f "$HOME/.nanobot/codex-token.json" ]; then
    echo "✅ Token saved to ~/.nanobot/codex-token.json"
    echo ""
    echo "   Now restart the API to pick it up:"
    echo "   cd ~/nanobot/LibreChat && docker compose restart nanobot-api"
else
    echo "❌ Login may have failed — no token file found."
    echo "   Try again or ask Joel for help."
fi
