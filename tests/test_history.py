"""Tests for SQLite history store."""

import pytest
from datetime import datetime, timezone

from traceforge.models import RunResult, ScenarioResult
from traceforge.history import HistoryStore


def make_scenario_result(name="test", pass_rate=0.8, passed=4, failed=1):
    return ScenarioResult(
        scenario_name=name,
        total_runs=passed + failed,
        passed_runs=passed,
        failed_runs=failed,
        pass_rate=pass_rate,
        consistency_score=0.6,
        avg_latency_ms=1000.0,
        min_latency_ms=800.0,
        max_latency_ms=1200.0,
        run_results=[
            RunResult(
                scenario_name=name,
                run_number=i + 1,
                trace_id=f"trace_{i}",
                step_results=[],
                passed=i < passed,
                total_latency_ms=1000.0,
                timestamp=datetime.now(timezone.utc),
            )
            for i in range(passed + failed)
        ],
        per_step_pass_rates=[pass_rate],
        per_expectation_pass_rates={},
        tags=[],
    )


@pytest.fixture
def store(tmp_path):
    return HistoryStore(base_dir=str(tmp_path / ".traceforge"))


class TestHistoryStore:
    def test_record_and_retrieve(self, store):
        result = make_scenario_result("calc", 1.0)
        store.record(result, "qwen2.5:7b-instruct")

        history = store.get_history()
        assert len(history) == 1
        assert history[0]["scenario_name"] == "calc"
        assert history[0]["pass_rate"] == 1.0

    def test_get_previous(self, store):
        r1 = make_scenario_result("calc", 0.8)
        r2 = make_scenario_result("calc", 0.6)
        store.record(r1, "model")
        store.record(r2, "model")

        prev = store.get_previous("calc", "model")
        assert prev is not None
        assert prev["pass_rate"] == 0.6  # Most recent

    def test_get_previous_none(self, store):
        assert store.get_previous("nonexistent", "model") is None

    def test_check_regression_detected(self, store):
        r1 = make_scenario_result("calc", 0.8)
        store.record(r1, "model")

        r2 = make_scenario_result("calc", 0.4)
        warning = store.check_regression(r2, "model")
        assert warning is not None
        assert "REGRESSION" in warning
        assert "80%" in warning
        assert "40%" in warning

    def test_check_regression_none(self, store):
        r1 = make_scenario_result("calc", 0.8)
        store.record(r1, "model")

        r2 = make_scenario_result("calc", 0.9)
        warning = store.check_regression(r2, "model")
        assert warning is None

    def test_check_regression_no_previous(self, store):
        r = make_scenario_result("calc", 0.8)
        warning = store.check_regression(r, "model")
        assert warning is None

    def test_history_filter_scenario(self, store):
        store.record(make_scenario_result("a"), "model")
        store.record(make_scenario_result("b"), "model")

        history = store.get_history(scenario_name="a")
        assert len(history) == 1
        assert history[0]["scenario_name"] == "a"

    def test_history_limit(self, store):
        for i in range(10):
            store.record(make_scenario_result(f"s{i}"), "model")

        history = store.get_history(limit=3)
        assert len(history) == 3
