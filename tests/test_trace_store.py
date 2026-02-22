"""Tests for content-addressed trace store + SQLite index."""

import pytest
from datetime import datetime, timezone

from traceforge.models import ExecutionEnvelope, StepRecord, ToolCallRecord, TraceIR
from traceforge.trace_ir import compute_trace_id
from traceforge.trace_store import TraceStore


def make_trace(scenario_name="test", run_number=1, **kw) -> TraceIR:
    return TraceIR(
        scenario_name=scenario_name,
        run_number=run_number,
        timestamp=kw.get("timestamp", datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)),
        envelope=ExecutionEnvelope(
            model_name=kw.get("model_name", "qwen2.5:7b-instruct"),
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[],
            system_prompt="You are helpful.",
        ),
        steps=[
            StepRecord(
                step_index=0,
                user_message=kw.get("user_message", "Hello"),
                assistant_response="Hi!",
                tool_calls=kw.get("tool_calls", []),
                raw_ollama_response={"model": "qwen2.5:7b-instruct"},
                latency_ms=100.0,
            )
        ],
        total_latency_ms=100.0,
    )


@pytest.fixture
def store(tmp_path):
    return TraceStore(base_dir=str(tmp_path / ".traceforge"))


class TestTraceStore:
    def test_store_and_load(self, store):
        trace = make_trace()
        trace_id = store.store(trace, passed=True)
        assert len(trace_id) == 64

        loaded = store.load(trace_id)
        assert loaded.scenario_name == "test"
        assert loaded.trace_id == trace_id

    def test_deduplication(self, store):
        trace = make_trace()
        id1 = store.store(trace, passed=True)
        id2 = store.store(trace, passed=True)
        assert id1 == id2

        # Only one file in traces dir
        files = list(store.traces_dir.iterdir())
        assert len(files) == 1

    def test_load_not_found(self, store):
        with pytest.raises(FileNotFoundError, match="Trace not found"):
            store.load("nonexistent_hash")

    def test_list_traces(self, store):
        t1 = make_trace(scenario_name="a")
        t2 = make_trace(scenario_name="b", user_message="different")
        store.store(t1, passed=True)
        store.store(t2, passed=False)

        all_traces = store.list_traces()
        assert len(all_traces) == 2

    def test_list_traces_filter_scenario(self, store):
        store.store(make_trace(scenario_name="a"), passed=True)
        store.store(make_trace(scenario_name="b", user_message="x"), passed=True)

        filtered = store.list_traces(scenario_name="a")
        assert len(filtered) == 1
        assert filtered[0]["scenario_name"] == "a"

    def test_list_traces_filter_passed(self, store):
        store.store(make_trace(scenario_name="a"), passed=True)
        store.store(make_trace(scenario_name="b", user_message="x"), passed=False)

        passed = store.list_traces(passed=True)
        assert len(passed) == 1

        failed = store.list_traces(passed=False)
        assert len(failed) == 1

    def test_get_latest(self, store):
        t1 = make_trace(
            scenario_name="calc",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        t2 = make_trace(
            scenario_name="calc",
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            user_message="later",
        )
        store.store(t1, passed=True)
        store.store(t2, passed=True)

        latest = store.get_latest("calc", "qwen2.5:7b-instruct")
        assert latest is not None
        assert "2026-01-02" in latest["timestamp"]

    def test_get_latest_not_found(self, store):
        result = store.get_latest("nonexistent", "model")
        assert result is None

    def test_index_has_tool_call_count(self, store):
        trace = make_trace(tool_calls=[
            ToolCallRecord(
                tool_name="calc", arguments={}, response={"r": 1}, latency_ms=5.0
            )
        ])
        trace_id = store.store(trace, passed=True)

        traces = store.list_traces()
        assert traces[0]["tool_call_count"] == 1

    def test_compressed_size_tracked(self, store):
        trace = make_trace()
        store.store(trace, passed=True)
        traces = store.list_traces()
        assert traces[0]["compressed_size"] > 0
