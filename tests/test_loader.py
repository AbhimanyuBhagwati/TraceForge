"""Tests for YAML loading and validation."""

import json
import pytest
from pathlib import Path

from traceforge.loader import load_scenarios


@pytest.fixture
def scenario_dir(tmp_path):
    """Create a temp directory with sample YAML scenarios."""
    # Simple scenario
    simple = tmp_path / "simple.yaml"
    simple.write_text("""\
name: simple_chat
description: Agent responds to greeting
agent:
  model: qwen2.5:7b-instruct
  system_prompt: "You are a helpful assistant."
  seed: 42
steps:
  - user_message: "Hello!"
    expectations:
      - type: response_contains
        values: ["hello", "hi"]
runs: 3
tags: [basic]
""")

    # Scenario with tool
    calc = tmp_path / "calc.yaml"
    calc.write_text("""\
name: calculator
description: Calculator test
agent:
  model: qwen2.5:7b-instruct
  system_prompt: "You are a calculator."
  tools:
    - name: calculate
      description: "Perform math"
      parameters:
        type: object
        properties:
          expression:
            type: string
        required: [expression]
      mock_responses:
        - result: 42
steps:
  - user_message: "What is 6*7?"
    expectations:
      - type: tool_called
        tool: calculate
runs: 5
tags: [math, tools]
""")
    return tmp_path


@pytest.fixture
def prompt_scenario_dir(tmp_path):
    """Create a scenario that references external prompt and mock files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "calc.txt").write_text("You are a calculator assistant.")

    mocks_dir = tmp_path / "mocks"
    mocks_dir.mkdir()
    (mocks_dir / "calc.json").write_text(json.dumps([{"result": 42}, {"result": 100}]))

    scenario = tmp_path / "calc.yaml"
    scenario.write_text("""\
name: calc_external
agent:
  system_prompt_file: ./prompts/calc.txt
  tools:
    - name: calculate
      description: "Math"
      parameters:
        type: object
      mock_response_file: ./mocks/calc.json
steps:
  - user_message: "hi"
""")
    return tmp_path


class TestLoadScenarios:
    def test_load_directory(self, scenario_dir):
        scenarios = load_scenarios(scenario_dir)
        assert len(scenarios) == 2
        names = {s.name for s in scenarios}
        assert names == {"simple_chat", "calculator"}

    def test_load_single_file(self, scenario_dir):
        scenarios = load_scenarios(scenario_dir / "simple.yaml")
        assert len(scenarios) == 1
        assert scenarios[0].name == "simple_chat"

    def test_tag_filter(self, scenario_dir):
        scenarios = load_scenarios(scenario_dir, tag_filter="math")
        assert len(scenarios) == 1
        assert scenarios[0].name == "calculator"

    def test_tag_filter_no_match(self, scenario_dir):
        scenarios = load_scenarios(scenario_dir, tag_filter="nonexistent")
        assert len(scenarios) == 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_scenarios("/nonexistent/path")

    def test_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No YAML files"):
            load_scenarios(tmp_path)

    def test_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : not valid yaml [[[")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_scenarios(bad)

    def test_validation_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: test\n")  # Missing required fields
        with pytest.raises(ValueError, match="Validation error"):
            load_scenarios(bad)

    def test_resolve_system_prompt_file(self, prompt_scenario_dir):
        scenarios = load_scenarios(prompt_scenario_dir)
        assert len(scenarios) == 1
        assert scenarios[0].agent.system_prompt == "You are a calculator assistant."

    def test_resolve_mock_response_file(self, prompt_scenario_dir):
        scenarios = load_scenarios(prompt_scenario_dir)
        tool = scenarios[0].agent.tools[0]
        assert tool.mock_responses == [{"result": 42}, {"result": 100}]

    def test_missing_prompt_file(self, tmp_path):
        scenario = tmp_path / "bad.yaml"
        scenario.write_text("""\
name: bad
agent:
  system_prompt_file: ./nonexistent.txt
steps:
  - user_message: "hi"
""")
        with pytest.raises(FileNotFoundError, match="System prompt file"):
            load_scenarios(scenario)

    def test_missing_mock_response_file(self, tmp_path):
        scenario = tmp_path / "bad.yaml"
        scenario.write_text("""\
name: bad
agent:
  tools:
    - name: calc
      description: math
      parameters:
        type: object
      mock_response_file: ./nonexistent.json
steps:
  - user_message: "hi"
""")
        with pytest.raises(FileNotFoundError, match="Mock response file"):
            load_scenarios(scenario)

    def test_scenario_values(self, scenario_dir):
        scenarios = load_scenarios(scenario_dir / "calc.yaml")
        s = scenarios[0]
        assert s.agent.tools[0].name == "calculate"
        assert s.agent.tools[0].mock_responses == [{"result": 42}]
        assert s.steps[0].expectations[0].type.value == "tool_called"
        assert s.runs == 5

    def test_not_a_mapping(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected a mapping"):
            load_scenarios(bad)
