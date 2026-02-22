"""Tests for virtual replay — proves identical eval results without model."""

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
from traceforge.evaluator import Evaluator
from traceforge.replay import ReplayEngine
from traceforge.trace_store import TraceStore


def make_trace_and_scenario():
    """Create a matching trace and scenario for replay testing."""
    tool_call = ToolCallRecord(
        tool_name="calculate",
        arguments={"expression": "6 * 7"},
        response={"result": 42},
        latency_ms=5.0,
    )
    trace = TraceIR(
        scenario_name="calc_test",
        run_number=1,
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        envelope=ExecutionEnvelope(
            model_name="qwen2.5:7b-instruct",
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[],
            system_prompt="You are a calculator.",
        ),
        steps=[
            StepRecord(
                step_index=0,
                user_message="What is 6 times 7?",
                assistant_response="The answer is 42.",
                tool_calls=[tool_call],
                raw_ollama_response={"model": "qwen2.5:7b-instruct"},
                latency_ms=500.0,
            ),
        ],
        total_latency_ms=500.0,
    )

    scenario = Scenario(
        name="calc_test",
        agent=AgentConfig(system_prompt="You are a calculator."),
        steps=[
            Step(
                user_message="What is 6 times 7?",
                expectations=[
                    Expectation(type=ExpectationType.TOOL_CALLED, tool="calculate"),
                    Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["42"]),
                    Expectation(type=ExpectationType.LATENCY_UNDER, max_ms=1000),
                    Expectation(type=ExpectationType.NO_TOOL_ERRORS),
                ],
            ),
        ],
        runs=1,
    )
    return trace, scenario


@pytest.fixture
def store(tmp_path):
    return TraceStore(base_dir=str(tmp_path / ".traceforge"))


class TestVirtualReplay:
    def test_produces_identical_results(self, store):
        """Core property: virtual replay produces identical eval results."""
        trace, scenario = make_trace_and_scenario()
        trace_id = store.store(trace, passed=True)

        # Direct evaluation
        evaluator = Evaluator()
        direct_result = evaluator.evaluate(store.load(trace_id), scenario)

        # Virtual replay
        engine = ReplayEngine(store)
        replay_result = engine.virtual_replay(trace_id, scenario)

        # Results must be identical
        assert direct_result.passed == replay_result.passed
        assert len(direct_result.step_results) == len(replay_result.step_results)
        for dr, rr in zip(direct_result.step_results, replay_result.step_results):
            assert dr.all_passed == rr.all_passed
            assert len(dr.results) == len(rr.results)
            for de, re_ in zip(dr.results, rr.results):
                assert de.passed == re_.passed

    def test_replay_with_updated_expectations(self, store):
        """Replay with changed expectations should produce different results."""
        trace, scenario = make_trace_and_scenario()
        trace_id = store.store(trace, passed=True)

        # Original passes
        engine = ReplayEngine(store)
        result1 = engine.virtual_replay(trace_id, scenario)
        assert result1.passed

        # Updated scenario with failing expectation
        new_scenario = scenario.model_copy(
            update={
                "steps": [
                    Step(
                        user_message="What is 6 times 7?",
                        expectations=[
                            Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["999"]),
                        ],
                    ),
                ]
            }
        )
        result2 = engine.virtual_replay(trace_id, new_scenario)
        assert not result2.passed

    def test_replay_determinism(self, store):
        """Multiple replays produce identical results."""
        trace, scenario = make_trace_and_scenario()
        trace_id = store.store(trace, passed=True)

        engine = ReplayEngine(store)
        r1 = engine.virtual_replay(trace_id, scenario)
        r2 = engine.virtual_replay(trace_id, scenario)
        r3 = engine.virtual_replay(trace_id, scenario)

        assert r1.passed == r2.passed == r3.passed
        for s1, s2, s3 in zip(r1.step_results, r2.step_results, r3.step_results):
            assert s1.all_passed == s2.all_passed == s3.all_passed


class TestDiffTraces:
    def test_identical_traces(self, store):
        trace, _ = make_trace_and_scenario()
        trace_id = store.store(trace, passed=True)

        engine = ReplayEngine(store)
        diff = engine.diff_traces(trace_id, trace_id)
        assert diff["trace_1"] == trace_id
        assert diff["trace_2"] == trace_id
        for d in diff["diffs"]:
            assert not d.get("tool_calls_changed", False)
            assert not d.get("response_changed", False)

    def test_different_responses(self, store):
        trace1, _ = make_trace_and_scenario()
        trace1_id = store.store(trace1, passed=True)

        trace2, _ = make_trace_and_scenario()
        trace2.steps[0].assistant_response = "I think it's 43."
        trace2_id = store.store(trace2, passed=False)

        engine = ReplayEngine(store)
        diff = engine.diff_traces(trace1_id, trace2_id)
        assert diff["diffs"][0]["response_changed"] is True

    def test_different_step_counts(self, store):
        trace1, _ = make_trace_and_scenario()
        trace1_id = store.store(trace1, passed=True)

        trace2, _ = make_trace_and_scenario()
        trace2.steps.append(StepRecord(
            step_index=1,
            user_message="Follow up",
            assistant_response="Sure",
            raw_ollama_response={},
            latency_ms=100.0,
        ))
        trace2_id = store.store(trace2, passed=True)

        engine = ReplayEngine(store)
        diff = engine.diff_traces(trace1_id, trace2_id)
        assert len(diff["diffs"]) == 2
        assert diff["diffs"][1].get("missing_in") == "trace_1"
