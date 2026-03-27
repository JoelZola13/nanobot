#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_CONFIG="$ROOT_DIR/opencode.json"
TARGET_DIR="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}"
TARGET_CONFIG="$TARGET_DIR/opencode.json"
ABS_BRIDGE_PATH="$ROOT_DIR/bin/voicebox-opencode"

if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "Source OpenCode config not found: $SOURCE_CONFIG" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

python3 - "$SOURCE_CONFIG" "$TARGET_CONFIG" "$ABS_BRIDGE_PATH" <<'PY'
import json
import os
import sys

source_path = sys.argv[1]
target_path = sys.argv[2]
bridge_path = sys.argv[3]

with open(source_path, "r", encoding="utf-8") as f:
    source = json.load(f)

if os.path.exists(target_path):
    with open(target_path, "r", encoding="utf-8") as f:
        target = json.load(f)
else:
    target = {}

schema = source.get("$schema")
if schema:
    target["$schema"] = schema

target_mcp = target.get("mcp", {}) if isinstance(target.get("mcp"), dict) else {}
source_mcp = source.get("mcp", {}) if isinstance(source.get("mcp"), dict) else {}
if "voicebox" in source_mcp:
    source_mcp = dict(source_mcp)
    source_mcp["voicebox"] = dict(source_mcp["voicebox"])
    source_mcp["voicebox"]["command"] = [bridge_path, "--mcp"]
target_mcp.update(source_mcp)
target["mcp"] = target_mcp

with open(target_path, "w", encoding="utf-8") as f:
    json.dump(target, f, indent=2)
    f.write("\n")

print(f"Installed OpenCode config to {target_path}")
PY
