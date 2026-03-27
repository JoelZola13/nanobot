#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_BASE_URL="http://127.0.0.1:17493"
BASE_URL="${VOICEBOX_API_URL:-$DEFAULT_BASE_URL}"
PYTHON="${PYTHON_BIN:-$ROOT_DIR/backend/venv/bin/python}"
PID=""
LOG_FILE="/tmp/voicebox-opencode-live-smoke.log"

cleanup() {
  if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
}

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for live smoke test" >&2
  exit 1
fi

if [ ! -x "$PYTHON" ]; then
  echo "Python interpreter not found at $PYTHON" >&2
  echo "Set PYTHON_BIN to a venv python path or install dependencies." >&2
  exit 1
fi

if ! "$PYTHON" - <<'PY' >/dev/null 2>&1; then
import importlib.util
spec = importlib.util.find_spec('fastapi')
raise SystemExit(0 if spec is not None else 1)
PY
  echo "FastAPI not installed in $PYTHON" >&2
  exit 1
fi

BASE_HOST_PORT="${BASE_URL#*://}"
PORT="${BASE_HOST_PORT##*:}"
if [ "$PORT" = "$BASE_HOST_PORT" ]; then
  PORT="8000"
fi

RUN_BACKEND="1"
if command -v lsof >/dev/null 2>&1; then
  if lsof -i "TCP:$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    RUN_BACKEND="0"
    echo "Found existing process on port $PORT. Using it for live smoke test."
  fi
fi

if [ "$RUN_BACKEND" = "1" ]; then
  if [ ! -x "$ROOT_DIR/backend/venv/bin/uvicorn" ]; then
    echo "uvicorn executable missing from virtualenv. Using python -m uvicorn fallback."
  fi

  "$PYTHON" -m uvicorn backend.main:app --port "${PORT}" >"$LOG_FILE" 2>&1 &
  PID=$!
  trap cleanup EXIT INT TERM

  for _ in $(seq 1 30); do
    if curl -sSf "$BASE_URL/profiles" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! curl -sSf "$BASE_URL/profiles" >/dev/null 2>&1; then
    echo "Backend did not become ready. Showing log output:" >&2
    cat "$LOG_FILE" >&2
    exit 1
  fi

  echo "Backend started on $BASE_URL (pid $PID)."
else
  echo "Using existing backend on $BASE_URL."
fi

python3 scripts/smoke_opencode_voice_bridge.py --base-url "$BASE_URL"
