"""Tests for canonical serialization and hashing determinism."""

import copy
from datetime import datetime, timezone

from traceforge.models import (
    ExecutionEnvelope,
    StepRecord,
    ToolCallRecord,
    TraceIR,
)
from traceforge.trace_ir import canonical_serialize, compute_trace_id, finalize_trace


def make_trace(**overrides) -> TraceIR:
    defaults = dict(
        scenario_name="test",
        run_number=1,
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        envelope=ExecutionEnvelope(
            model_name="qwen2.5:7b-instruct",
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[],
            system_prompt="You are helpful.",
        ),
        steps=[
            StepRecord(
                step_index=0,
                user_message="Hello",
                assistant_response="Hi there!",
                raw_ollama_response={"model": "qwen2.5:7b-instruct"},
                latency_ms=100.0,
            )
        ],
        total_latency_ms=100.0,
    )
    defaults.update(overrides)
    return TraceIR(**defaults)


class TestCanonicalSerialize:
    def test_determinism(self):
        """Same trace serializes to same string every time."""
        t = make_trace()
        s1 = canonical_serialize(t)
        s2 = canonical_serialize(t)
        assert s1 == s2

    def test_excludes_trace_id(self):
        t1 = make_trace()
        t2 = make_trace()
        t2.trace_id = "some_different_id"
        assert canonical_serialize(t1) == canonical_serialize(t2)

    def test_excludes_metadata(self):
        t1 = make_trace()
        t2 = make_trace(metadata={"gpu": "M3 Max"})
        assert canonical_serialize(t1) == canonical_serialize(t2)

    def test_different_content_different_serialization(self):
        t1 = make_trace(scenario_name="a")
        t2 = make_trace(scenario_name="b")
        assert canonical_serialize(t1) != canonical_serialize(t2)

    def test_sorted_keys(self):
        t = make_trace()
        s = canonical_serialize(t)
        # JSON should have sorted keys, no spaces
        assert '"envelope"' in s
        assert '", "' not in s  # compact separators


class TestComputeTraceId:
    def test_determinism(self):
        """Same trace always gets same hash."""
        t = make_trace()
        h1 = compute_trace_id(t)
        h2 = compute_trace_id(t)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self):
        t1 = make_trace(scenario_name="a")
        t2 = make_trace(scenario_name="b")
        assert compute_trace_id(t1) != compute_trace_id(t2)

    def test_metadata_does_not_affect_hash(self):
        t1 = make_trace()
        t2 = make_trace(metadata={"key": "value"})
        assert compute_trace_id(t1) == compute_trace_id(t2)

    def test_tool_calls_affect_hash(self):
        t1 = make_trace()
        t2 = make_trace()
        t2.steps[0].tool_calls = [
            ToolCallRecord(
                tool_name="calc",
                arguments={"expr": "1+1"},
                response={"result": 2},
                latency_ms=5.0,
            )
        ]
        assert compute_trace_id(t1) != compute_trace_id(t2)


class TestFinalizeTrace:
    def test_sets_trace_id(self):
        t = make_trace()
        assert t.trace_id == ""
        finalized = finalize_trace(t)
        assert finalized.trace_id != ""
        assert len(finalized.trace_id) == 64

    def test_idempotent(self):
        t = make_trace()
        f1 = finalize_trace(t)
        id1 = f1.trace_id
        f2 = finalize_trace(f1)
        assert f2.trace_id == id1

    def test_returns_same_object(self):
        t = make_trace()
        f = finalize_trace(t)
        assert f is t
