"""Tests for the causal attribution engine."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from traceforge.attribution import InterventionGenerator, CausalAttributionEngine
from traceforge.models import (
    AgentConfig,
    CounterfactualResult,
    Expectation,
    ExpectationType,
    ExecutionEnvelope,
    Intervention,
    InterventionType,
    RunResult,
    Scenario,
    Step,
    StepRecord,
    StepResult,
    ExpectationResult,
    ToolCallRecord,
    ToolDefinition,
    TraceIR,
)


@pytest.fixture
def failing_trace():
    return TraceIR(
        version="1.0.0",
        trace_id="abc123",
        scenario_name="test_calc",
        run_number=1,
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        envelope=ExecutionEnvelope(
            model_name="qwen2.5:7b-instruct",
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[],
            system_prompt="You are a calculator. Always use tools.",
        ),
        steps=[
            StepRecord(
                step_index=0,
                user_message="What is 6 times 7?",
                assistant_response="I think the answer is about 40.",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="calculate",
                        arguments={"expression": "6 * 7"},
                        response={"result": 42},
                        latency_ms=5.0,
                    )
                ],
                raw_ollama_response={},
                latency_ms=1200.0,
            )
        ],
        total_latency_ms=1200.0,
    )


@pytest.fixture
def failing_scenario():
    return Scenario(
        name="test_calc",
        agent=AgentConfig(
            model="qwen2.5:7b-instruct",
            system_prompt="You are a calculator. Always use tools.",
            tools=[
                ToolDefinition(
                    name="calculate",
                    description="Perform calculation",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                    mock_responses=[{"result": 42}],
                )
            ],
        ),
        steps=[
            Step(
                user_message="What is 6 times 7?",
                expectations=[
                    Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["42"]),
                ],
            ),
        ],
    )


class TestInterventionGenerator:
    def test_tool_output_format_numeric_to_string(self, failing_trace):
        gen = InterventionGenerator()
        interventions = gen._tool_output_format_interventions(failing_trace, 0)
        assert len(interventions) >= 1
        fmt = interventions[0]
        assert fmt.intervention_type == InterventionType.TOOL_OUTPUT_FORMAT
        assert fmt.target_tool == "calculate"
        assert fmt.modified_value == "42"

    def test_tool_output_value_interventions(self, failing_trace):
        gen = InterventionGenerator()
        interventions = gen._tool_output_value_interventions(failing_trace, 0)
        # Should get zero, negated, doubled, halved for result=42
        assert len(interventions) == 4
        descriptions = [i.description for i in interventions]
        assert any("zero" in d for d in descriptions)
        assert any("negated" in d for d in descriptions)
        assert any("doubled" in d for d in descriptions)
        assert any("halved" in d for d in descriptions)

    def test_tool_output_field_interventions(self, failing_trace):
        gen = InterventionGenerator()
        interventions = gen._tool_output_field_interventions(failing_trace, 0)
        # Should have removal of 'result' + addition of 'debug_info'
        assert len(interventions) == 2
        assert any(i.modified_value == "__REMOVE__" for i in interventions)
        assert any(i.target_field == "debug_info" for i in interventions)

    def test_system_prompt_clause_interventions(self, failing_scenario):
        gen = InterventionGenerator()
        interventions = gen._system_prompt_clause_interventions(failing_scenario)
        # "You are a calculator. Always use tools." -> 2 sentences
        assert len(interventions) == 2
        for i in interventions:
            assert i.intervention_type == InterventionType.SYSTEM_PROMPT_CLAUSE

    def test_tool_schema_interventions(self, failing_scenario):
        gen = InterventionGenerator()
        interventions = gen._tool_schema_interventions(failing_scenario, 0)
        assert len(interventions) == 1
        assert interventions[0].modified_value == "expression_v2"

    def test_context_truncation_no_prior_steps(self, failing_trace):
        gen = InterventionGenerator()
        interventions = gen._context_truncation_interventions(failing_trace, 0)
        assert len(interventions) == 0  # No prior steps to truncate

    def test_generate_all(self, failing_scenario, failing_trace):
        gen = InterventionGenerator()
        interventions = gen.generate_all(failing_scenario, failing_trace, 0)
        assert len(interventions) > 0
        types = {i.intervention_type for i in interventions}
        assert InterventionType.TOOL_OUTPUT_FORMAT in types
        assert InterventionType.TOOL_OUTPUT_VALUE in types
        assert InterventionType.TOOL_OUTPUT_FIELDS in types
        assert InterventionType.SYSTEM_PROMPT_CLAUSE in types
        assert InterventionType.TOOL_SCHEMA_CHANGE in types

    def test_out_of_range_step(self, failing_trace):
        gen = InterventionGenerator()
        assert gen._tool_output_format_interventions(failing_trace, 99) == []
        assert gen._tool_output_value_interventions(failing_trace, 99) == []
        assert gen._tool_output_field_interventions(failing_trace, 99) == []

    def test_boolean_value_intervention(self):
        trace = TraceIR(
            version="1.0.0",
            trace_id="bool_test",
            scenario_name="test",
            run_number=1,
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            envelope=ExecutionEnvelope(
                model_name="test", temperature=0.1, num_ctx=4096,
                tool_schemas=[], system_prompt="",
            ),
            steps=[StepRecord(
                step_index=0,
                user_message="test",
                assistant_response="test",
                tool_calls=[ToolCallRecord(
                    tool_name="check",
                    arguments={},
                    response={"valid": True},
                    latency_ms=1.0,
                )],
                raw_ollama_response={},
                latency_ms=100.0,
            )],
            total_latency_ms=100.0,
        )
        gen = InterventionGenerator()
        interventions = gen._tool_output_value_interventions(trace, 0)
        assert len(interventions) == 1
        assert interventions[0].modified_value is False


class TestCausalAttributionEngine:
    def test_attribute_with_mocked_harness(self, failing_trace, failing_scenario):
        mock_store = MagicMock()
        mock_store.load.return_value = failing_trace

        mock_harness = MagicMock()
        # Counterfactual runs: some pass, some fail
        passing_trace = failing_trace.model_copy(deep=True)
        passing_trace.steps[0].assistant_response = "The answer is 42."

        call_count = 0

        def mock_run(scenario, runs=1):
            nonlocal call_count
            call_count += 1
            # Every 3rd intervention "fixes" the issue
            if call_count % 3 == 0:
                return [passing_trace]
            return [failing_trace]

        mock_harness.run_scenario.side_effect = mock_run

        from traceforge.evaluator import Evaluator
        evaluator = Evaluator()

        engine = CausalAttributionEngine(mock_harness, evaluator, mock_store)
        report = engine.attribute("abc123", failing_scenario, confirmation_runs=1, max_interventions=10)

        assert report.scenario_name == "test_calc"
        assert report.failing_trace_id == "abc123"
        assert report.failing_step == 0
        assert report.total_interventions <= 10
        assert len(report.causal_factors) > 0
        assert report.summary != ""

    def test_attribute_non_failing_trace_raises(self, failing_scenario):
        passing_trace = TraceIR(
            version="1.0.0",
            trace_id="pass123",
            scenario_name="test_calc",
            run_number=1,
            timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            envelope=ExecutionEnvelope(
                model_name="qwen2.5:7b-instruct", temperature=0.1,
                seed=42, num_ctx=4096, tool_schemas=[],
                system_prompt="You are a calculator. Always use tools.",
            ),
            steps=[StepRecord(
                step_index=0,
                user_message="What is 6 times 7?",
                assistant_response="The answer is 42.",
                tool_calls=[ToolCallRecord(
                    tool_name="calculate",
                    arguments={"expression": "6 * 7"},
                    response={"result": 42},
                    latency_ms=5.0,
                )],
                raw_ollama_response={},
                latency_ms=1200.0,
            )],
            total_latency_ms=1200.0,
        )
        mock_store = MagicMock()
        mock_store.load.return_value = passing_trace
        mock_harness = MagicMock()

        from traceforge.evaluator import Evaluator
        evaluator = Evaluator()

        engine = CausalAttributionEngine(mock_harness, evaluator, mock_store)
        with pytest.raises(ValueError, match="does not appear to fail"):
            engine.attribute("pass123", failing_scenario)

    def test_rank_factors(self):
        engine = CausalAttributionEngine(None, None, None)
        results = [
            CounterfactualResult(
                intervention=Intervention(
                    intervention_type=InterventionType.TOOL_OUTPUT_FORMAT,
                    description="test",
                ),
                original_passed=False,
                counterfactual_passed=True,
                flipped=True,
                trace_id="t1",
                confidence=1.0,
            ),
            CounterfactualResult(
                intervention=Intervention(
                    intervention_type=InterventionType.TOOL_OUTPUT_FORMAT,
                    description="test2",
                ),
                original_passed=False,
                counterfactual_passed=False,
                flipped=False,
                trace_id="t2",
                confidence=1.0,
            ),
            CounterfactualResult(
                intervention=Intervention(
                    intervention_type=InterventionType.SYSTEM_PROMPT_CLAUSE,
                    description="test3",
                ),
                original_passed=False,
                counterfactual_passed=False,
                flipped=False,
                trace_id="t3",
                confidence=1.0,
            ),
        ]
        factors = engine._rank_factors(results)
        assert len(factors) == 2
        assert factors[0]["factor"] == "tool_output_format"
        assert factors[0]["sensitivity"] == 0.5
        assert factors[1]["factor"] == "system_prompt_clause"
        assert factors[1]["sensitivity"] == 0.0

    def test_apply_intervention_system_prompt(self, failing_scenario, failing_trace):
        engine = CausalAttributionEngine(None, None, None)
        intervention = Intervention(
            intervention_type=InterventionType.SYSTEM_PROMPT_CLAUSE,
            description="Remove sentence",
            modified_value="Just use tools.",
        )
        modified = engine._apply_intervention(failing_scenario, failing_trace, intervention)
        assert modified.agent.system_prompt == "Just use tools."
        assert modified.runs == 1

    def test_apply_intervention_tool_schema(self, failing_scenario, failing_trace):
        engine = CausalAttributionEngine(None, None, None)
        intervention = Intervention(
            intervention_type=InterventionType.TOOL_SCHEMA_CHANGE,
            description="Rename expression",
            target_tool="calculate",
            target_field="expression",
            original_value="expression",
            modified_value="expression_v2",
        )
        modified = engine._apply_intervention(failing_scenario, failing_trace, intervention)
        props = modified.agent.tools[0].parameters["properties"]
        assert "expression_v2" in props
        assert "expression" not in props
