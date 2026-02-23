"""YAML loader for agent definitions."""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from nanobot.agents.spec import AgentSpec
from nanobot.agents.registry import AgentRegistry


def load_agents(teams_dir: Path, registry: AgentRegistry | None = None) -> AgentRegistry:
    """
    Load agent definitions from YAML files in the teams directory.

    Expected structure:
        teams_dir/
            executive/
                agents.yaml
                ceo.md
                executive_memory.md
            communication/
                agents.yaml
                ...

    Args:
        teams_dir: Path to the teams directory.
        registry: Optional existing registry to populate.

    Returns:
        Populated AgentRegistry.
    """
    if registry is None:
        registry = AgentRegistry()

    if not teams_dir.exists():
        logger.warning(f"Teams directory not found: {teams_dir}")
        return registry

    for team_dir in sorted(teams_dir.iterdir()):
        if not team_dir.is_dir():
            continue

        agents_file = team_dir / "agents.yaml"
        if not agents_file.exists():
            logger.debug(f"No agents.yaml in {team_dir.name}, skipping")
            continue

        try:
            _load_team(team_dir, agents_file, registry)
        except Exception as e:
            logger.error(f"Failed to load team '{team_dir.name}': {e}")

    logger.info(
        f"Loaded {len(registry)} agents across {len(registry.get_teams())} teams"
    )
    return registry


def _load_team(
    team_dir: Path, agents_file: Path, registry: AgentRegistry
) -> None:
    """Load agents from a single team directory."""
    with open(agents_file) as f:
        data = yaml.safe_load(f)

    if not data or "agents" not in data:
        logger.warning(f"No agents defined in {agents_file}")
        return

    team_name = team_dir.name

    for agent_data in data["agents"]:
        spec = _parse_agent(agent_data, team_name, team_dir)
        registry.register(spec)
        logger.debug(f"Registered agent: {spec.qualified_name}")


def _parse_agent(
    data: dict[str, Any], team_name: str, team_dir: Path
) -> AgentSpec:
    """Parse a single agent definition from YAML data."""
    name = data["name"]

    # Load system prompt from .md file if specified
    system_prompt = ""
    prompt_file = data.get("system_prompt", "")
    if prompt_file:
        prompt_path = team_dir / prompt_file
        if prompt_path.exists():
            system_prompt = prompt_path.read_text().strip()
        else:
            logger.warning(f"System prompt not found: {prompt_path}")

    # Parse tools and handoffs as tuples
    tools = tuple(data.get("tools", []))
    handoffs = tuple(data.get("handoffs", []))

    return AgentSpec(
        name=name,
        team=team_name,
        description=data.get("description", ""),
        role=data.get("role", "member"),
        model=data.get("model", "default"),
        tools=tools,
        handoffs=handoffs,
        system_prompt=system_prompt,
        max_iterations=data.get("max_iterations", 25),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 4096),
    )
