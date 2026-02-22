"""Tests for delta debugging — ddmin reduces traces correctly."""

from unittest.mock import MagicMock
from datetime import datetime, timezone

import pytest

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
from traceforge.minrepro import MinReproExtractor
from traceforge.trace_store import TraceStore


def make_5step_trace(failing_step=3):
    """Create a 5-step trace where failure is caused by a specific step."""
    steps = []
    for i in range(5):
        response = "OK" if i != failing_step else "ERROR: something went wrong"
        steps.append(StepRecord(
            step_index=i,
            user_message=f"Step {i}",
            assistant_response=response,
            tool_calls=[
                ToolCallRecord(
                    tool_name="tool",
                    arguments={"step": i},
                    response={"ok": True} if i != failing_step else {"error": "fail"},
                    latency_ms=5.0,
                )
            ],
            raw_ollama_response={},
            latency_ms=100.0,
        ))

    return TraceIR(
        scenario_name="multi_step",
        run_number=1,
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        envelope=ExecutionEnvelope(
            model_name="qwen2.5:7b-instruct",
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[],
            system_prompt="Test",
        ),
        steps=steps,
        total_latency_ms=500.0,
    )


def make_5step_scenario(failing_step=3):
    """Create a 5-step scenario where step `failing_step` has a failing expectation."""
    steps = []
    for i in range(5):
        expectations = []
        if i == failing_step:
            expectations.append(
                Expectation(type=ExpectationType.RESPONSE_NOT_CONTAINS, value="error")
            )
        else:
            expectations.append(
                Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["OK"])
            )
        steps.append(Step(user_message=f"Step {i}", expectations=expectations))

    return Scenario(
        name="multi_step",
        agent=AgentConfig(system_prompt="Test"),
        steps=steps,
        runs=1,
    )


@pytest.fixture
def store(tmp_path):
    return TraceStore(base_dir=str(tmp_path / ".traceforge"))


class TestMinRepro:
    def test_reduces_to_failing_step(self, store):
        """5-step trace with failure at step 3 should reduce significantly."""
        trace = make_5step_trace(failing_step=3)
        scenario = make_5step_scenario(failing_step=3)
        trace_id = store.store(trace, passed=False)

        # Mock harness that returns traces matching the subset scenario
        mock_harness = MagicMock()

        def run_scenario_side_effect(subset_scenario, runs=1):
            """Simulate running a subset: reuse original trace steps for matching indices."""
            subset_steps = []
            for i, step in enumerate(subset_scenario.steps):
                # Find the original step index by matching user_message
                orig_idx = int(step.user_message.split()[-1])
                orig_step = trace.steps[orig_idx]
                subset_steps.append(StepRecord(
                    step_index=i,
                    user_message=orig_step.user_message,
                    assistant_response=orig_step.assistant_response,
                    tool_calls=orig_step.tool_calls,
                    raw_ollama_response={},
                    latency_ms=100.0,
                ))
            result_trace = TraceIR(
                scenario_name="multi_step",
                run_number=1,
                timestamp=datetime.now(timezone.utc),
                envelope=trace.envelope,
                steps=subset_steps,
                total_latency_ms=100.0 * len(subset_steps),
            )
            from traceforge.trace_ir import finalize_trace
            return [finalize_trace(result_trace)]

        mock_harness.run_scenario.side_effect = run_scenario_side_effect

        evaluator = Evaluator()
        extractor = MinReproExtractor(mock_harness, evaluator, store)
        result = extractor.minimize(trace_id, scenario)

        assert result.failure_still_reproduces
        assert result.minimized_step_count < result.original_step_count
        assert result.reduction_ratio > 0.0
        assert 3 in result.minimized_steps  # The failing step must remain

    def test_no_reduction_possible(self, store):
        """1-step failing trace can't be reduced."""
        trace = make_5step_trace(failing_step=0)
        # Only 1 step scenario
        scenario = Scenario(
            name="single",
            agent=AgentConfig(system_prompt="Test"),
            steps=[Step(
                user_message="Step 0",
                expectations=[
                    Expectation(type=ExpectationType.RESPONSE_NOT_CONTAINS, value="error"),
                ],
            )],
            runs=1,
        )
        single_trace = TraceIR(
            scenario_name="single",
            run_number=1,
            timestamp=datetime.now(timezone.utc),
            envelope=trace.envelope,
            steps=[trace.steps[0]],
            total_latency_ms=100.0,
        )
        trace_id = store.store(single_trace, passed=False)

        mock_harness = MagicMock()

        def run_scenario_side_effect(subset_scenario, runs=1):
            from traceforge.trace_ir import finalize_trace
            result = TraceIR(
                scenario_name="single",
                run_number=1,
                timestamp=datetime.now(timezone.utc),
                envelope=trace.envelope,
                steps=[trace.steps[0]],
                total_latency_ms=100.0,
            )
            return [finalize_trace(result)]

        mock_harness.run_scenario.side_effect = run_scenario_side_effect

        evaluator = Evaluator()
        extractor = MinReproExtractor(mock_harness, evaluator, store)
        result = extractor.minimize(trace_id, scenario)

        assert result.minimized_step_count == 1

    def test_passing_trace_no_reduction(self, store):
        """A passing trace should report no reduction."""
        # All steps pass
        steps = [
            StepRecord(
                step_index=i,
                user_message=f"Step {i}",
                assistant_response="OK",
                tool_calls=[],
                raw_ollama_response={},
                latency_ms=100.0,
            )
            for i in range(3)
        ]
        trace = TraceIR(
            scenario_name="passing",
            run_number=1,
            timestamp=datetime.now(timezone.utc),
            envelope=ExecutionEnvelope(
                model_name="m", temperature=0.1, seed=42, num_ctx=4096,
                tool_schemas=[], system_prompt="Test",
            ),
            steps=steps,
            total_latency_ms=300.0,
        )
        scenario = Scenario(
            name="passing",
            agent=AgentConfig(system_prompt="Test"),
            steps=[
                Step(user_message=f"Step {i}", expectations=[
                    Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["OK"]),
                ])
                for i in range(3)
            ],
            runs=1,
        )
        trace_id = store.store(trace, passed=True)

        mock_harness = MagicMock()

        def run_scenario_side_effect(subset_scenario, runs=1):
            from traceforge.trace_ir import finalize_trace
            subset_steps = [
                StepRecord(
                    step_index=i, user_message=s.user_message,
                    assistant_response="OK", tool_calls=[], raw_ollama_response={},
                    latency_ms=100.0,
                )
                for i, s in enumerate(subset_scenario.steps)
            ]
            t = TraceIR(
                scenario_name="passing", run_number=1,
                timestamp=datetime.now(timezone.utc),
                envelope=trace.envelope,
                steps=subset_steps, total_latency_ms=100.0,
            )
            return [finalize_trace(t)]

        mock_harness.run_scenario.side_effect = run_scenario_side_effect

        evaluator = Evaluator()
        extractor = MinReproExtractor(mock_harness, evaluator, store)
        result = extractor.minimize(trace_id, scenario)

        assert not result.failure_still_reproduces
        assert result.reduction_ratio == 0.0
