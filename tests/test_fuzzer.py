"""Tests for fuzzing orchestration (mocked harness)."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

from traceforge.models import (
    AgentConfig,
    Expectation,
    ExpectationType,
    ExecutionEnvelope,
    MutatorConfig,
    Scenario,
    Step,
    StepRecord,
    ToolCallRecord,
    ToolDefinition,
    TraceIR,
)
from traceforge.evaluator import Evaluator
from traceforge.fuzzer import DifferentialFuzzer
from traceforge.trace_store import TraceStore


def make_trace(passed_response="The answer is 42.", tool_response=None):
    tc = ToolCallRecord(
        tool_name="calc",
        arguments={"expression": "6*7"},
        response=tool_response or {"result": 42},
        latency_ms=5.0,
    )
    return TraceIR(
        trace_id="baseline123",
        scenario_name="calc",
        run_number=1,
        timestamp=datetime.now(timezone.utc),
        envelope=ExecutionEnvelope(
            model_name="qwen2.5:7b-instruct",
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[],
            system_prompt="Calculator",
        ),
        steps=[
            StepRecord(
                step_index=0,
                user_message="What is 6*7?",
                assistant_response=passed_response,
                tool_calls=[tc],
                raw_ollama_response={},
                latency_ms=100.0,
            )
        ],
        total_latency_ms=100.0,
    )


@pytest.fixture
def scenario():
    return Scenario(
        name="calc",
        agent=AgentConfig(
            system_prompt="Calculator",
            tools=[
                ToolDefinition(
                    name="calc",
                    description="Math",
                    parameters={
                        "type": "object",
                        "properties": {
                            "result": {"type": "number"},
                        },
                    },
                    mock_responses=[{"result": 42}],
                )
            ],
        ),
        steps=[
            Step(
                user_message="What is 6*7?",
                expectations=[
                    Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["42"]),
                ],
            )
        ],
        runs=1,
    )


class TestDifferentialFuzzer:
    def test_fuzz_finds_breaks(self, scenario, tmp_path):
        store = TraceStore(base_dir=str(tmp_path / ".traceforge"))

        # Mock harness: baseline passes, mutated runs fail
        mock_harness = MagicMock()
        baseline = make_trace("The answer is 42.", {"result": 42})
        mutated = make_trace("The answer is 0.", {"result": 0})

        mock_harness.run_scenario.side_effect = [
            [baseline],      # baseline run
            [mutated],        # first mutation
            [mutated],        # second mutation
            [mutated],        # etc.
            [mutated],
            [mutated],
        ]

        evaluator = Evaluator()
        config = MutatorConfig(
            enabled=True,
            mutations_per_tool=2,
            mutation_types=["numeric_extreme", "null_injection"],
        )

        fuzzer = DifferentialFuzzer(mock_harness, evaluator, store)
        report = fuzzer.fuzz_scenario(scenario, config)

        assert report.scenario_name == "calc"
        assert report.total_mutations > 0
        assert report.total_breaks > 0
        assert report.robustness_score < 1.0

    def test_fuzz_all_pass(self, scenario, tmp_path):
        store = TraceStore(base_dir=str(tmp_path / ".traceforge"))

        mock_harness = MagicMock()
        passing = make_trace("The answer is 42.", {"result": 42})
        mock_harness.run_scenario.return_value = [passing]

        evaluator = Evaluator()
        config = MutatorConfig(
            enabled=True,
            mutations_per_tool=2,
            mutation_types=["null_injection"],
        )

        fuzzer = DifferentialFuzzer(mock_harness, evaluator, store)
        report = fuzzer.fuzz_scenario(scenario, config)

        # All mutations still pass since response contains "42"
        assert report.robustness_score == 1.0

    def test_fuzz_report_by_type(self, scenario, tmp_path):
        store = TraceStore(base_dir=str(tmp_path / ".traceforge"))

        mock_harness = MagicMock()
        baseline = make_trace("The answer is 42.", {"result": 42})
        failing = make_trace("Error occurred", {"result": None})

        mock_harness.run_scenario.side_effect = [
            [baseline],
            [failing], [failing],  # mutations
            [failing], [failing],
        ]

        evaluator = Evaluator()
        config = MutatorConfig(
            enabled=True, mutations_per_tool=2,
            mutation_types=["numeric_extreme", "type_swap"],
        )

        fuzzer = DifferentialFuzzer(mock_harness, evaluator, store)
        report = fuzzer.fuzz_scenario(scenario, config)

        assert "by_mutation_type" in report.model_dump()
        assert "by_tool" in report.model_dump()
