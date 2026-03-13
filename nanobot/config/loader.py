"""Configuration loading utilities."""

import json
from pathlib import Path

from nanobot.config.schema import Config


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".nanobot" / "config.json"


def get_data_dir() -> Path:
    """Get the nanobot data directory."""
    from nanobot.utils.helpers import get_data_path
    return get_data_path()


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            data = _migrate_config(data)
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")

    # Seed tools.postiz from existing MCP postiz entry when the dedicated config
    # doesn't exist yet (keeps legacy setups working after introducing the native
    # Postiz tool config.
    if isinstance(tools, dict) and "postiz" not in tools:
        mcp_tools = tools.get("mcpServers") or {}
        postiz_mcp = mcp_tools.get("postiz") if isinstance(mcp_tools, dict) else None
        if isinstance(postiz_mcp, dict) and postiz_mcp.get("url"):
            postiz_url = str(postiz_mcp["url"]).strip()
            if postiz_url:
                parts = postiz_url.split("/api/mcp/", 1)
                base_url = parts[0] or postiz_url
                api_key = parts[1] if len(parts) > 1 else ""
                tools["postiz"] = {
                    "enabled": True,
                    "baseUrl": base_url.rstrip("/"),
                    "apiKey": api_key,
                    "apiKeyHeader": "Authorization",
                    "apiKeyPrefix": "Bearer",
                    "extraHeaders": {},
                    "publishPath": "/api/public/v1/posts",
                    "requestTimeout": 30,
                    "defaultTargetHandle": "streetvoiceswatch",
                    "defaultPlatform": "instagram",
                    "defaultMaxCaptionChars": 2200,
                }

    return data
