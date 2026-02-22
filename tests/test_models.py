"""Tests for Pydantic model serialization/deserialization."""

import json
from datetime import datetime, timezone

from traceforge.models import (
    AgentConfig,
    Expectation,
    ExpectationType,
    ExecutionEnvelope,
    FuzzReport,
    FuzzResult,
    MinReproResult,
    MutationRecord,
    ProbeReport,
    RunResult,
    Scenario,
    ScenarioResult,
    Step,
    StepRecord,
    StepResult,
    ExpectationResult,
    ToolCallRecord,
    ToolDefinition,
    TraceIR,
    TRACE_IR_VERSION,
    JudgeConfig,
    MutatorConfig,
)


class TestToolDefinition:
    def test_basic(self, sample_tool_def):
        assert sample_tool_def.name == "calculate"
        assert len(sample_tool_def.mock_responses) == 2

    def test_no_mock_responses(self):
        t = ToolDefinition(name="foo", description="bar", parameters={"type": "object"})
        assert t.mock_responses is None
        assert t.mock_response_file is None

    def test_roundtrip_json(self, sample_tool_def):
        data = sample_tool_def.model_dump_json()
        restored = ToolDefinition.model_validate_json(data)
        assert restored == sample_tool_def


class TestExpectation:
    def test_all_types_valid(self):
        for et in ExpectationType:
            e = Expectation(type=et)
            assert e.type == et

    def test_tool_called(self):
        e = Expectation(type=ExpectationType.TOOL_CALLED, tool="calculate")
        assert e.tool == "calculate"

    def test_response_contains(self):
        e = Expectation(type=ExpectationType.RESPONSE_CONTAINS, values=["42", "forty-two"])
        assert len(e.values) == 2

    def test_latency_under(self):
        e = Expectation(type=ExpectationType.LATENCY_UNDER, max_ms=5000)
        assert e.max_ms == 5000

    def test_tool_call_count(self):
        e = Expectation(type=ExpectationType.TOOL_CALL_COUNT, tool="calc", count=2, operator="gte")
        assert e.operator == "gte"

    def test_default_operator(self):
        e = Expectation(type=ExpectationType.TOOL_CALL_COUNT, count=1)
        assert e.operator == "eq"


class TestScenario:
    def test_basic(self, sample_scenario):
        assert sample_scenario.name == "test_calculator"
        assert sample_scenario.runs == 3
        assert len(sample_scenario.steps) == 1
        assert len(sample_scenario.agent.tools) == 1

    def test_defaults(self):
        s = Scenario(
            name="minimal",
            agent=AgentConfig(),
            steps=[Step(user_message="hi")],
        )
        assert s.runs == 5
        assert s.tags == []
        assert s.agent.model == "qwen2.5:7b-instruct"
        assert s.agent.temperature == 0.1
        assert s.mutator is None

    def test_roundtrip_json(self, sample_scenario):
        data = sample_scenario.model_dump_json()
        restored = Scenario.model_validate_json(data)
        assert restored == sample_scenario

    def test_with_mutator(self):
        s = Scenario(
            name="fuzzed",
            agent=AgentConfig(),
            steps=[Step(user_message="hi")],
            mutator=MutatorConfig(enabled=True, mutations_per_tool=10),
        )
        assert s.mutator.enabled is True
        assert s.mutator.mutations_per_tool == 10
        assert "numeric_extreme" in s.mutator.mutation_types

    def test_with_judge(self):
        s = Scenario(
            name="judged",
            agent=AgentConfig(),
            judge=JudgeConfig(model="llama3:8b"),
            steps=[Step(user_message="hi")],
        )
        assert s.judge.model == "llama3:8b"
        assert s.judge.seed == 42


class TestTraceIR:
    def test_basic(self, sample_trace):
        assert sample_trace.version == TRACE_IR_VERSION
        assert sample_trace.trace_id == ""
        assert sample_trace.scenario_name == "test_calculator"
        assert len(sample_trace.steps) == 1
        assert len(sample_trace.steps[0].tool_calls) == 1

    def test_roundtrip_json(self, sample_trace):
        data = sample_trace.model_dump_json()
        restored = TraceIR.model_validate_json(data)
        assert restored == sample_trace

    def test_envelope(self, sample_trace):
        env = sample_trace.envelope
        assert env.model_name == "qwen2.5:7b-instruct"
        assert env.seed == 42
        assert env.system_prompt == "You are a calculator assistant."

    def test_metadata_default_empty(self, sample_trace):
        assert sample_trace.metadata == {}

    def test_metadata_extensible(self):
        t = TraceIR(
            scenario_name="x",
            run_number=1,
            timestamp=datetime.now(timezone.utc),
            envelope=ExecutionEnvelope(
                model_name="m", temperature=0.0, num_ctx=2048,
                tool_schemas=[], system_prompt="p",
            ),
            steps=[],
            total_latency_ms=0.0,
            metadata={"gpu": "M3 Max", "ram_gb": 36},
        )
        assert t.metadata["gpu"] == "M3 Max"


class TestResultModels:
    def test_expectation_result(self):
        er = ExpectationResult(
            expectation=Expectation(type=ExpectationType.TOOL_CALLED, tool="calc"),
            passed=True,
            message="Tool 'calc' was called",
        )
        assert er.passed

    def test_step_result(self):
        sr = StepResult(
            step_index=0,
            user_message="hi",
            results=[],
            all_passed=True,
        )
        assert sr.all_passed

    def test_run_result(self):
        rr = RunResult(
            scenario_name="test",
            run_number=1,
            trace_id="abc123",
            step_results=[],
            passed=True,
            total_latency_ms=100.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert rr.trace_id == "abc123"

    def test_scenario_result(self):
        sr = ScenarioResult(
            scenario_name="test",
            total_runs=5,
            passed_runs=4,
            failed_runs=1,
            pass_rate=0.8,
            consistency_score=0.6,
            avg_latency_ms=1000.0,
            min_latency_ms=800.0,
            max_latency_ms=1200.0,
            run_results=[],
            per_step_pass_rates=[1.0, 0.8],
            per_expectation_pass_rates={"tool_called:calc": 1.0},
            tags=["math"],
        )
        assert sr.pass_rate == 0.8

    def test_probe_report(self):
        pr = ProbeReport(
            timestamp=datetime.now(timezone.utc),
            model="qwen2.5:7b-instruct",
            total_scenarios=1,
            total_runs=5,
            overall_pass_rate=0.8,
            scenario_results=[],
        )
        assert pr.regression_warnings == []


class TestFuzzModels:
    def test_mutation_record(self):
        mr = MutationRecord(
            original_args={"expression": "6 * 7"},
            mutated_args={"expression": ""},
            mutation_type="empty_string",
            mutation_description="Set 'expression' to empty string",
        )
        assert mr.mutation_type == "empty_string"

    def test_fuzz_result(self):
        fr = FuzzResult(
            step_index=0,
            tool_name="calc",
            mutation=MutationRecord(
                original_args={}, mutated_args={},
                mutation_type="null_injection",
                mutation_description="Set to null",
            ),
            trace_id="abc",
            original_passed=True,
            mutated_passed=False,
            broke_agent=True,
        )
        assert fr.broke_agent

    def test_fuzz_report(self):
        fr = FuzzReport(
            scenario_name="test",
            total_mutations=10,
            total_breaks=2,
            robustness_score=0.8,
            results=[],
            by_mutation_type={"numeric_extreme": 0.9},
            by_tool={"calc": 0.8},
        )
        assert fr.robustness_score == 0.8


class TestMinReproResult:
    def test_basic(self):
        mr = MinReproResult(
            original_trace_id="aaa",
            original_step_count=5,
            original_tool_call_count=8,
            minimized_trace_id="bbb",
            minimized_step_count=2,
            minimized_tool_call_count=3,
            reduction_ratio=0.6,
            iterations_taken=7,
            failure_still_reproduces=True,
            minimized_steps=[0, 3],
        )
        assert mr.reduction_ratio == 0.6
        assert mr.minimized_steps == [0, 3]

    def test_roundtrip_json(self):
        mr = MinReproResult(
            original_trace_id="aaa",
            original_step_count=5,
            original_tool_call_count=8,
            minimized_trace_id="bbb",
            minimized_step_count=2,
            minimized_tool_call_count=3,
            reduction_ratio=0.6,
            iterations_taken=7,
            failure_still_reproduces=True,
            minimized_steps=[0, 3],
        )
        data = mr.model_dump_json()
        restored = MinReproResult.model_validate_json(data)
        assert restored == mr
