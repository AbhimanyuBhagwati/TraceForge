"""Tests for execution harness (mocked Ollama)."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

from traceforge.harness import Harness, ModelNotFoundError
from traceforge.models import (
    AgentConfig,
    Expectation,
    ExpectationType,
    Scenario,
    Step,
    ToolDefinition,
)
from traceforge.trace_store import TraceStore


def make_ollama_response(content="Hello!", tool_calls=None):
    """Create a mock Ollama response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    resp = MagicMock()
    resp.message = msg
    resp.model_dump = MagicMock(return_value={
        "model": "qwen2.5:7b-instruct",
        "message": {"content": content, "tool_calls": tool_calls},
    })
    return resp


def make_tool_call(name="calculate", arguments=None):
    """Create a mock tool call object."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments or {"expression": "6 * 7"}
    tc = MagicMock()
    tc.function = fn
    return tc


class TestHarness:
    @patch("traceforge.harness.ollama_client.Client")
    def test_simple_run(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = make_ollama_response("Hello there!")
        mock_client_cls.return_value = mock_client

        scenario = Scenario(
            name="simple",
            agent=AgentConfig(system_prompt="You are helpful."),
            steps=[Step(user_message="Hi")],
            runs=1,
        )

        harness = Harness()
        traces = harness.run_scenario(scenario)

        assert len(traces) == 1
        trace = traces[0]
        assert trace.scenario_name == "simple"
        assert trace.run_number == 1
        assert len(trace.steps) == 1
        assert trace.steps[0].assistant_response == "Hello there!"
        assert trace.trace_id != ""  # finalized

    @patch("traceforge.harness.ollama_client.Client")
    def test_with_tool_calls(self, mock_client_cls):
        mock_client = MagicMock()
        # First call: model returns tool call
        tc = make_tool_call("calculate", {"expression": "6 * 7"})
        resp1 = make_ollama_response("", tool_calls=[tc])
        # Second call: model returns final answer
        resp2 = make_ollama_response("The answer is 42.")
        mock_client.chat.side_effect = [resp1, resp2]
        mock_client_cls.return_value = mock_client

        tool = ToolDefinition(
            name="calculate",
            description="Math",
            parameters={"type": "object"},
            mock_responses=[{"result": 42}],
        )
        scenario = Scenario(
            name="calc",
            agent=AgentConfig(system_prompt="Calculator", tools=[tool]),
            steps=[Step(user_message="What is 6*7?")],
            runs=1,
        )

        harness = Harness()
        traces = harness.run_scenario(scenario)

        assert len(traces) == 1
        step = traces[0].steps[0]
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0].tool_name == "calculate"
        assert step.tool_calls[0].response == {"result": 42}

    @patch("traceforge.harness.ollama_client.Client")
    def test_multiple_runs(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = make_ollama_response("Hi!")
        mock_client_cls.return_value = mock_client

        scenario = Scenario(
            name="multi",
            agent=AgentConfig(system_prompt="Hi"),
            steps=[Step(user_message="Hello")],
            runs=3,
        )

        harness = Harness()
        traces = harness.run_scenario(scenario)
        assert len(traces) == 3
        assert traces[0].run_number == 1
        assert traces[2].run_number == 3

    @patch("traceforge.harness.ollama_client.Client")
    def test_runs_override(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = make_ollama_response("Hi!")
        mock_client_cls.return_value = mock_client

        scenario = Scenario(
            name="override",
            agent=AgentConfig(),
            steps=[Step(user_message="Hello")],
            runs=10,
        )

        harness = Harness()
        traces = harness.run_scenario(scenario, runs=2)
        assert len(traces) == 2

    @patch("traceforge.harness.ollama_client.Client")
    def test_stores_traces(self, mock_client_cls, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.return_value = make_ollama_response("Hi!")
        mock_client_cls.return_value = mock_client

        store = TraceStore(base_dir=str(tmp_path / ".traceforge"))
        scenario = Scenario(
            name="stored",
            agent=AgentConfig(),
            steps=[Step(user_message="Hello")],
            runs=1,
        )

        harness = Harness(trace_store=store)
        traces = harness.run_scenario(scenario)

        stored = store.list_traces()
        assert len(stored) == 1

    @patch("traceforge.harness.ollama_client.Client")
    def test_envelope_captured(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.chat.return_value = make_ollama_response("Hi!")
        mock_client_cls.return_value = mock_client

        scenario = Scenario(
            name="env_test",
            agent=AgentConfig(
                model="llama3:8b",
                system_prompt="Be brief.",
                temperature=0.5,
                seed=123,
                num_ctx=8192,
            ),
            steps=[Step(user_message="Hello")],
            runs=1,
        )

        harness = Harness()
        traces = harness.run_scenario(scenario)
        env = traces[0].envelope
        assert env.model_name == "llama3:8b"
        assert env.temperature == 0.5
        assert env.seed == 123
        assert env.num_ctx == 8192
        assert env.system_prompt == "Be brief."
