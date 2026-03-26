#!/bin/bash
# Login to OpenAI Codex — authenticates the AI agents.
# Opens a browser window for ChatGPT sign-in.
# ANY teammate with a ChatGPT account can run this.
# The token auto-refreshes — you only need to do this once.
#
# Usage: cd ~/nanobot && ./login.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Make sure ~/.nanobot exists
mkdir -p "$HOME/.nanobot"

# Store the token where Docker can read it
export OAUTH_CLI_KIT_TOKEN_PATH="$HOME/.nanobot/codex-token.json"

echo ""
echo "=== Street Voices — AI Agent Login ==="
echo ""
echo "  This connects the AI agents to GPT-5.4 via your ChatGPT account."
echo "  A browser window will open — sign in with any ChatGPT account."
echo "  The token auto-refreshes, so you only need to do this once."
echo ""

# Install nanobot temporarily if needed, run login, cleanup
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -m nanobot provider login openai-codex
else
    # No venv — install into a temp venv
    echo "  Setting up (first time only)..."
    python3 -m venv /tmp/nanobot-login-venv
    /tmp/nanobot-login-venv/bin/pip install -q -e .
    /tmp/nanobot-login-venv/bin/python -m nanobot provider login openai-codex
    rm -rf /tmp/nanobot-login-venv
fi

echo ""
if [ -f "$HOME/.nanobot/codex-token.json" ]; then
    echo "  Token saved to ~/.nanobot/codex-token.json"
    echo ""
    echo "  Restarting the API to pick it up..."
    cd "$SCRIPT_DIR/LibreChat" 2>/dev/null && docker compose restart nanobot-api 2>/dev/null && echo "  Done! Agents should respond now." || echo "  Restart the API manually: cd ~/nanobot/LibreChat && docker compose restart nanobot-api"
else
    echo "  Login may have failed — no token file found."
    echo "  Try again or ask Joel for help."
fi
echo ""
