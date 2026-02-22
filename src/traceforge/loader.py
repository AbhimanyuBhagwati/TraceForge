"""YAML scenario loader and validator."""

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from traceforge.models import Scenario


def load_scenarios(path: str | Path, tag_filter: str | None = None) -> list[Scenario]:
    """Load and validate scenarios from a YAML file or directory.

    Args:
        path: Path to a YAML file or directory containing YAML files.
        tag_filter: If set, only return scenarios with this tag.

    Returns:
        List of validated Scenario objects.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If YAML is invalid or fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario path not found: {path}")

    if path.is_file():
        yaml_files = [path]
    else:
        yaml_files = sorted(path.rglob("*.yaml")) + sorted(path.rglob("*.yml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML files found in: {path}")

    scenarios = []
    for yaml_file in yaml_files:
        scenario = _load_single(yaml_file)
        if tag_filter and tag_filter not in scenario.tags:
            continue
        scenarios.append(scenario)

    return scenarios


def _load_single(yaml_file: Path) -> Scenario:
    """Load and validate a single YAML scenario file."""
    try:
        raw = yaml.safe_load(yaml_file.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {yaml_file}: {e}")

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {yaml_file}, got {type(raw).__name__}")

    try:
        scenario = Scenario.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Validation error in {yaml_file}:\n{e}")

    # Resolve system_prompt_file relative to the YAML file
    _resolve_system_prompt(scenario, yaml_file.parent)

    # Resolve mock_response_file for each tool
    _resolve_mock_responses(scenario, yaml_file.parent)

    return scenario


def _resolve_system_prompt(scenario: Scenario, base_dir: Path) -> None:
    """Load system_prompt_file contents into system_prompt."""
    if scenario.agent.system_prompt_file and not scenario.agent.system_prompt:
        prompt_path = base_dir / scenario.agent.system_prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt file not found: {prompt_path} "
                f"(referenced in scenario '{scenario.name}')"
            )
        scenario.agent.system_prompt = prompt_path.read_text().strip()


def _resolve_mock_responses(scenario: Scenario, base_dir: Path) -> None:
    """Load mock_response_file contents into mock_responses."""
    for tool in scenario.agent.tools:
        if tool.mock_response_file and not tool.mock_responses:
            resp_path = base_dir / tool.mock_response_file
            if not resp_path.exists():
                raise FileNotFoundError(
                    f"Mock response file not found: {resp_path} "
                    f"(referenced in tool '{tool.name}', scenario '{scenario.name}')"
                )
            data = json.loads(resp_path.read_text())
            if isinstance(data, list):
                tool.mock_responses = data
            else:
                tool.mock_responses = [data]
