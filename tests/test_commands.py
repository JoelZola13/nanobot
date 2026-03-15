import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import Config

runner = CliRunner()


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.utils.helpers.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_task_dispatch_success():
    config = Config()
    config.tools.relay_base_url = "http://relay.test"
    config.tools.relay_token = "secret"

    response = MagicMock()
    response.json.return_value = {"id": "pc-123", "status": "queued"}
    response.raise_for_status.return_value = None

    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = response

    with patch("nanobot.config.loader.load_config", return_value=config), \
         patch("httpx.Client", return_value=client):
        result = runner.invoke(app, ["task", "dev_manager", "Fix the API", "--priority", "low"])

    assert result.exit_code == 0
    client.post.assert_called_once_with(
        "http://relay.test/dispatch",
        json={"agent": "dev_manager", "task": "Fix the API", "priority": "low"},
        headers={"Content-Type": "application/json", "Authorization": "Bearer secret"},
    )
    assert "Created Paperclip task" in result.stdout
    assert "pc-123" in result.stdout


def test_task_dispatch_http_error():
    import httpx

    config = Config()
    config.tools.relay_base_url = "http://relay.test"

    response = MagicMock()
    response.status_code = 500
    response.text = "boom"

    request = httpx.Request("POST", "http://relay.test/dispatch")
    error = httpx.HTTPStatusError("error", request=request, response=response)

    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = error

    with patch("nanobot.config.loader.load_config", return_value=config), \
         patch("httpx.Client", return_value=client):
        result = runner.invoke(app, ["task", "dev_manager", "Fix the API"])

    assert result.exit_code == 1
    assert "Dispatch failed (500): boom" in result.stdout
