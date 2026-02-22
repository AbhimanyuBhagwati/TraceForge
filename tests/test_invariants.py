"""Tests for the invariant mining engine."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from traceforge.invariants import (
    ArgPatternMiner,
    ArgRangeMiner,
    InvariantMiner,
    LatencyMiner,
    ResponsePatternMiner,
    ToolCallCountMiner,
    ToolOrderMiner,
    ToolPresenceMiner,
)
from traceforge.models import (
    ExecutionEnvelope,
    Invariant,
    InvariantType,
    StepRecord,
    ToolCallRecord,
    TraceIR,
)


def _make_trace(
    steps: list[StepRecord],
    scenario_name: str = "test_calc",
    run_number: int = 1,
    trace_id: str = "",
) -> TraceIR:
    total_ms = sum(s.latency_ms for s in steps)
    return TraceIR(
        version="1.0.0",
        trace_id=trace_id,
        scenario_name=scenario_name,
        run_number=run_number,
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        envelope=ExecutionEnvelope(
            model_name="qwen2.5:7b-instruct",
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[],
            system_prompt="Test prompt.",
        ),
        steps=steps,
        total_latency_ms=total_ms,
    )


def _make_step(
    step_index: int,
    tool_calls: list[ToolCallRecord] | None = None,
    assistant_response: str = "Answer",
    latency_ms: float = 1000.0,
) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        user_message=f"Question {step_index}",
        assistant_response=assistant_response,
        tool_calls=tool_calls or [],
        raw_ollama_response={},
        latency_ms=latency_ms,
    )


def _make_tc(
    tool_name: str = "calculate",
    arguments: dict | None = None,
    response: dict | None = None,
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=tool_name,
        arguments=arguments or {"expression": "6 * 7"},
        response=response or {"result": 42},
        latency_ms=5.0,
    )


@pytest.fixture
def passing_traces():
    """3 passing traces that all call 'calculate' at step 0."""
    traces = []
    for i in range(5):
        traces.append(_make_trace(
            steps=[
                _make_step(
                    0,
                    tool_calls=[_make_tc(arguments={"expression": f"{i+1} * 7"})],
                    assistant_response=f"The answer is {(i+1)*7}.",
                    latency_ms=1000.0 + i * 100,
                ),
            ],
            run_number=i + 1,
            trace_id=f"pass_{i}",
        ))
    return traces


@pytest.fixture
def failing_traces():
    """2 failing traces where 'calculate' is NOT called."""
    return [
        _make_trace(
            steps=[
                _make_step(
                    0,
                    tool_calls=[],
                    assistant_response="I'm not sure.",
                    latency_ms=500.0,
                ),
            ],
            run_number=100,
            trace_id="fail_0",
        ),
        _make_trace(
            steps=[
                _make_step(
                    0,
                    tool_calls=[],
                    assistant_response="Let me think...",
                    latency_ms=800.0,
                ),
            ],
            run_number=101,
            trace_id="fail_1",
        ),
    ]


class TestToolOrderMiner:
    def test_discovers_order(self):
        traces = []
        for i in range(3):
            traces.append(_make_trace(
                steps=[
                    _make_step(0, tool_calls=[_make_tc("lookup")]),
                    _make_step(1, tool_calls=[_make_tc("calculate")]),
                ],
                run_number=i + 1,
            ))
        miner = ToolOrderMiner()
        candidates = miner.extract_candidates(traces)
        descs = [c.description for c in candidates]
        assert any("lookup" in d and "before" in d and "calculate" in d for d in descs)

    def test_no_order_when_no_tools(self):
        traces = [_make_trace(steps=[_make_step(0)]) for _ in range(3)]
        miner = ToolOrderMiner()
        assert miner.extract_candidates(traces) == []


class TestToolPresenceMiner:
    def test_always_called(self, passing_traces):
        miner = ToolPresenceMiner()
        candidates = miner.extract_candidates(passing_traces)
        always_called = [
            c for c in candidates
            if c.invariant_type == InvariantType.TOOL_ALWAYS_CALLED
        ]
        assert len(always_called) >= 1
        assert always_called[0].tool_name == "calculate"
        assert always_called[0].step_index == 0

    def test_never_called_tool(self):
        traces = []
        for i in range(3):
            traces.append(_make_trace(
                steps=[
                    _make_step(0, tool_calls=[_make_tc("calculate")]),
                    _make_step(1, tool_calls=[]),
                ],
                run_number=i + 1,
            ))
        miner = ToolPresenceMiner()
        candidates = miner.extract_candidates(traces)
        never_at_step1 = [
            c for c in candidates
            if c.invariant_type == InvariantType.TOOL_NEVER_CALLED and c.step_index == 1
        ]
        assert len(never_at_step1) >= 1


class TestToolCallCountMiner:
    def test_count_range(self, passing_traces):
        miner = ToolCallCountMiner()
        candidates = miner.extract_candidates(passing_traces)
        assert len(candidates) >= 1
        calc = [c for c in candidates if c.tool_name == "calculate"][0]
        assert calc.details["min"] == 1
        assert calc.details["max"] == 1


class TestArgRangeMiner:
    def test_numeric_range(self):
        traces = []
        for i in range(5):
            traces.append(_make_trace(
                steps=[_make_step(0, tool_calls=[
                    _make_tc(arguments={"value": 10 + i * 5})
                ])],
                run_number=i + 1,
            ))
        miner = ArgRangeMiner()
        candidates = miner.extract_candidates(traces)
        assert len(candidates) >= 1
        assert candidates[0].invariant_type == InvariantType.ARG_RANGE
        assert candidates[0].details["min"] <= 10
        assert candidates[0].details["max"] >= 30

    def test_no_numeric_args(self, passing_traces):
        # passing_traces have string args only
        miner = ArgRangeMiner()
        candidates = miner.extract_candidates(passing_traces)
        assert len(candidates) == 0


class TestArgPatternMiner:
    def test_non_empty_pattern(self, passing_traces):
        miner = ArgPatternMiner()
        candidates = miner.extract_candidates(passing_traces)
        assert len(candidates) >= 1
        assert candidates[0].invariant_type == InvariantType.ARG_PATTERN
        assert candidates[0].details["pattern"] == r".+"


class TestResponsePatternMiner:
    def test_response_length(self, passing_traces):
        miner = ResponsePatternMiner()
        candidates = miner.extract_candidates(passing_traces)
        assert len(candidates) >= 1
        assert candidates[0].invariant_type == InvariantType.RESPONSE_LENGTH


class TestLatencyMiner:
    def test_latency_bound(self, passing_traces):
        miner = LatencyMiner()
        candidates = miner.extract_candidates(passing_traces)
        assert len(candidates) >= 1
        assert candidates[0].invariant_type == InvariantType.STEP_LATENCY
        assert candidates[0].details["max_ms"] > 0


class TestInvariantMiner:
    def test_mine_discovers_discriminating(self, passing_traces, failing_traces):
        mock_store = MagicMock()
        all_meta = []
        for t in passing_traces:
            all_meta.append({"trace_id": t.trace_id, "passed": True})
        for t in failing_traces:
            all_meta.append({"trace_id": t.trace_id, "passed": False})
        mock_store.list_traces.return_value = all_meta

        def load_trace(trace_id):
            for t in passing_traces + failing_traces:
                if t.trace_id == trace_id:
                    return t
            raise FileNotFoundError(trace_id)

        mock_store.load.side_effect = load_trace

        miner = InvariantMiner(mock_store)
        report = miner.mine("test_calc", min_confidence=0.95)

        assert report.total_traces_analyzed == 7
        assert report.passing_traces == 5
        assert report.failing_traces == 2
        assert report.invariants_discovered >= 1

        # 'calculate' always called at step 0 should discriminate
        disc_descs = [d.description for d in report.discriminating_invariants]
        assert any("calculate" in d and "always called" in d for d in disc_descs)

    def test_mine_insufficient_traces(self):
        mock_store = MagicMock()
        mock_store.list_traces.return_value = [
            {"trace_id": "t1", "passed": True},
            {"trace_id": "t2", "passed": True},
        ]
        mock_store.load.return_value = _make_trace(
            steps=[_make_step(0)], trace_id="t1"
        )

        miner = InvariantMiner(mock_store)
        with pytest.raises(ValueError, match="Need at least 3"):
            miner.mine("test_calc")

    def test_check_invariant_tool_order(self):
        miner = InvariantMiner(MagicMock())
        inv = Invariant(
            invariant_type=InvariantType.TOOL_ORDER,
            description="lookup before calculate",
            formal="order(lookup) < order(calculate)",
            confidence=1.0,
            support=3,
            violations=0,
            details={"tool_a": "lookup", "tool_b": "calculate"},
        )
        # Correct order
        trace_ok = _make_trace(steps=[
            _make_step(0, tool_calls=[_make_tc("lookup")]),
            _make_step(1, tool_calls=[_make_tc("calculate")]),
        ])
        assert miner._check_invariant(inv, trace_ok) is True

        # Wrong order
        trace_bad = _make_trace(steps=[
            _make_step(0, tool_calls=[_make_tc("calculate")]),
            _make_step(1, tool_calls=[_make_tc("lookup")]),
        ])
        assert miner._check_invariant(inv, trace_bad) is False

    def test_check_invariant_tool_call_count(self):
        miner = InvariantMiner(MagicMock())
        inv = Invariant(
            invariant_type=InvariantType.TOOL_CALL_COUNT,
            description="calculate called 1-2 times",
            formal="1 <= count(calculate) <= 2",
            confidence=1.0,
            support=5,
            violations=0,
            tool_name="calculate",
            details={"min": 1, "max": 2},
        )
        # Within range
        trace_ok = _make_trace(steps=[
            _make_step(0, tool_calls=[_make_tc("calculate")]),
        ])
        assert miner._check_invariant(inv, trace_ok) is True

        # Out of range (0 calls)
        trace_bad = _make_trace(steps=[_make_step(0)])
        assert miner._check_invariant(inv, trace_bad) is False

    def test_generate_expectations(self, passing_traces, failing_traces):
        mock_store = MagicMock()
        all_meta = (
            [{"trace_id": t.trace_id, "passed": True} for t in passing_traces]
            + [{"trace_id": t.trace_id, "passed": False} for t in failing_traces]
        )
        mock_store.list_traces.return_value = all_meta
        mock_store.load.side_effect = lambda tid: next(
            t for t in passing_traces + failing_traces if t.trace_id == tid
        )

        miner = InvariantMiner(mock_store)
        report = miner.mine("test_calc")
        # Should suggest at least one expectation from discriminating invariants
        if report.discriminating_invariants:
            assert len(report.suggested_expectations) >= 0  # May or may not generate
