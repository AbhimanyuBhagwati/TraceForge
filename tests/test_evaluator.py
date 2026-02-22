"""Tests for evaluation logic — all 10 expectation types."""

import pytest
from datetime import datetime, timezone

from traceforge.models import (
    AgentConfig,
    Expectation,
    ExpectationType,
    ExecutionEnvelope,
    Scenario,
    Step,
    StepRecord,
    ToolCallRecord,
    TraceIR,
)
from traceforge.evaluator import Evaluator, calculate_consistency, aggregate_results


def make_step_record(
    response="The answer is 42.",
    tool_calls=None,
    latency_ms=100.0,
    user_message="What is 6*7?",
    step_index=0,
):
    return StepRecord(
        step_index=step_index,
        user_message=user_message,
        assistant_response=response,
        tool_calls=tool_calls or [],
        raw_ollama_response={},
        latency_ms=latency_ms,
    )


def make_trace(steps=None, scenario_name="test"):
    return TraceIR(
        trace_id="abc123",
        scenario_name=scenario_name,
        run_number=1,
        timestamp=datetime.now(timezone.utc),
        envelope=ExecutionEnvelope(
            model_name="qwen2.5:7b-instruct",
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[],
            system_prompt="You are helpful.",
        ),
        steps=steps or [make_step_record()],
        total_latency_ms=100.0,
    )


def make_scenario(expectations, steps=None):
    return Scenario(
        name="test",
        agent=AgentConfig(),
        steps=steps or [Step(user_message="What is 6*7?", expectations=expectations)],
        runs=1,
    )


class TestToolCalled:
    def test_pass(self):
        tc = ToolCallRecord(tool_name="calc", arguments={}, response={"r": 42}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc])])
        scenario = make_scenario([Expectation(type=ExpectationType.TOOL_CALLED, tool="calc")])
        result = Evaluator().evaluate(trace, scenario)
        assert result.passed

    def test_fail(self):
        trace = make_trace([make_step_record(tool_calls=[])])
        scenario = make_scenario([Expectation(type=ExpectationType.TOOL_CALLED, tool="calc")])
        result = Evaluator().evaluate(trace, scenario)
        assert not result.passed


class TestToolNotCalled:
    def test_pass(self):
        trace = make_trace([make_step_record(tool_calls=[])])
        scenario = make_scenario([Expectation(type=ExpectationType.TOOL_NOT_CALLED, tool="calc")])
        result = Evaluator().evaluate(trace, scenario)
        assert result.passed

    def test_fail(self):
        tc = ToolCallRecord(tool_name="calc", arguments={}, response={}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc])])
        scenario = make_scenario([Expectation(type=ExpectationType.TOOL_NOT_CALLED, tool="calc")])
        result = Evaluator().evaluate(trace, scenario)
        assert not result.passed


class TestToolArgsContain:
    def test_pass(self):
        tc = ToolCallRecord(tool_name="calc", arguments={"expression": "6 * 7"}, response={}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc])])
        exp = Expectation(type=ExpectationType.TOOL_ARGS_CONTAIN, tool="calc", args_contain={"expression": "6"})
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_fail(self):
        tc = ToolCallRecord(tool_name="calc", arguments={"expression": "2 + 2"}, response={}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc])])
        exp = Expectation(type=ExpectationType.TOOL_ARGS_CONTAIN, tool="calc", args_contain={"expression": "6"})
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert not result.passed


class TestResponseContains:
    def test_pass_values(self):
        trace = make_trace([make_step_record(response="The answer is 42.")])
        exp = Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["42", "forty-two"])
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_fail_values(self):
        trace = make_trace([make_step_record(response="I don't know.")])
        exp = Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["42"])
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert not result.passed

    def test_case_insensitive(self):
        trace = make_trace([make_step_record(response="HELLO WORLD")])
        exp = Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["hello"])
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_single_value(self):
        trace = make_trace([make_step_record(response="The answer is 42.")])
        exp = Expectation(type=ExpectationType.RESPONSE_CONTAINS, value="42")
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed


class TestResponseNotContains:
    def test_pass(self):
        trace = make_trace([make_step_record(response="The answer is 42.")])
        exp = Expectation(type=ExpectationType.RESPONSE_NOT_CONTAINS, values=["error", "unknown"])
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_fail(self):
        trace = make_trace([make_step_record(response="Error occurred")])
        exp = Expectation(type=ExpectationType.RESPONSE_NOT_CONTAINS, value="error")
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert not result.passed


class TestResponseMatchesRegex:
    def test_pass(self):
        trace = make_trace([make_step_record(response="The result is 42.")])
        exp = Expectation(type=ExpectationType.RESPONSE_MATCHES_REGEX, value=r"\d+")
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_fail(self):
        trace = make_trace([make_step_record(response="No numbers here.")])
        exp = Expectation(type=ExpectationType.RESPONSE_MATCHES_REGEX, value=r"^\d+$")
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert not result.passed


class TestLatencyUnder:
    def test_pass(self):
        trace = make_trace([make_step_record(latency_ms=500.0)])
        exp = Expectation(type=ExpectationType.LATENCY_UNDER, max_ms=1000)
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_fail(self):
        trace = make_trace([make_step_record(latency_ms=2000.0)])
        exp = Expectation(type=ExpectationType.LATENCY_UNDER, max_ms=1000)
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert not result.passed


class TestNoToolErrors:
    def test_pass(self):
        tc = ToolCallRecord(tool_name="calc", arguments={}, response={"result": 42}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc])])
        exp = Expectation(type=ExpectationType.NO_TOOL_ERRORS)
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_fail(self):
        tc = ToolCallRecord(tool_name="calc", arguments={}, response={"error": "boom"}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc])])
        exp = Expectation(type=ExpectationType.NO_TOOL_ERRORS)
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert not result.passed

    def test_no_tool_calls(self):
        trace = make_trace([make_step_record(tool_calls=[])])
        exp = Expectation(type=ExpectationType.NO_TOOL_ERRORS)
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed


class TestToolCallCount:
    def test_eq_pass(self):
        tc = ToolCallRecord(tool_name="calc", arguments={}, response={}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc, tc])])
        exp = Expectation(type=ExpectationType.TOOL_CALL_COUNT, tool="calc", count=2, operator="eq")
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_gte_pass(self):
        tc = ToolCallRecord(tool_name="calc", arguments={}, response={}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc, tc])])
        exp = Expectation(type=ExpectationType.TOOL_CALL_COUNT, tool="calc", count=1, operator="gte")
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed

    def test_lte_fail(self):
        tc = ToolCallRecord(tool_name="calc", arguments={}, response={}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc, tc, tc])])
        exp = Expectation(type=ExpectationType.TOOL_CALL_COUNT, tool="calc", count=2, operator="lte")
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert not result.passed

    def test_no_tool_filter(self):
        tc1 = ToolCallRecord(tool_name="a", arguments={}, response={}, latency_ms=5.0)
        tc2 = ToolCallRecord(tool_name="b", arguments={}, response={}, latency_ms=5.0)
        trace = make_trace([make_step_record(tool_calls=[tc1, tc2])])
        exp = Expectation(type=ExpectationType.TOOL_CALL_COUNT, count=2, operator="eq")
        result = Evaluator().evaluate(trace, make_scenario([exp]))
        assert result.passed


class TestConsistency:
    def test_all_pass(self):
        from traceforge.models import RunResult
        results = [
            RunResult(scenario_name="t", run_number=i, trace_id="x",
                      step_results=[], passed=True, total_latency_ms=100,
                      timestamp=datetime.now(timezone.utc))
            for i in range(5)
        ]
        assert calculate_consistency(results) == 1.0

    def test_all_fail(self):
        from traceforge.models import RunResult
        results = [
            RunResult(scenario_name="t", run_number=i, trace_id="x",
                      step_results=[], passed=False, total_latency_ms=100,
                      timestamp=datetime.now(timezone.utc))
            for i in range(5)
        ]
        assert calculate_consistency(results) == 1.0

    def test_half_and_half(self):
        from traceforge.models import RunResult
        results = [
            RunResult(scenario_name="t", run_number=i, trace_id="x",
                      step_results=[], passed=(i < 2), total_latency_ms=100,
                      timestamp=datetime.now(timezone.utc))
            for i in range(4)
        ]
        assert calculate_consistency(results) == 0.0

    def test_empty(self):
        assert calculate_consistency([]) == 0.0


class TestMultiStep:
    def test_missing_trace_step(self):
        """Trace has fewer steps than scenario."""
        trace = make_trace([make_step_record()])
        scenario = make_scenario(
            [],
            steps=[
                Step(user_message="Q1", expectations=[]),
                Step(user_message="Q2", expectations=[
                    Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["x"])
                ]),
            ],
        )
        result = Evaluator().evaluate(trace, scenario)
        assert not result.passed
        assert not result.step_results[1].all_passed
