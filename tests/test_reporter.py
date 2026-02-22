"""Tests for report generation."""

from datetime import datetime, timezone
from io import StringIO

from rich.console import Console

from traceforge.models import (
    FuzzReport,
    MinReproResult,
    ProbeReport,
    ScenarioResult,
)
from traceforge.reporter import print_report, print_fuzz_report, print_minrepro_result
from traceforge.html_report import generate_html_report


def make_report():
    return ProbeReport(
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        model="qwen2.5:7b-instruct",
        total_scenarios=2,
        total_runs=10,
        overall_pass_rate=0.8,
        scenario_results=[
            ScenarioResult(
                scenario_name="calc_test",
                total_runs=5,
                passed_runs=5,
                failed_runs=0,
                pass_rate=1.0,
                consistency_score=1.0,
                avg_latency_ms=1000.0,
                min_latency_ms=800.0,
                max_latency_ms=1200.0,
                run_results=[],
                per_step_pass_rates=[1.0],
                per_expectation_pass_rates={"tool_called:calc": 1.0},
                tags=["math"],
            ),
            ScenarioResult(
                scenario_name="weather_test",
                total_runs=5,
                passed_runs=3,
                failed_runs=2,
                pass_rate=0.6,
                consistency_score=0.6,
                avg_latency_ms=2000.0,
                min_latency_ms=1500.0,
                max_latency_ms=2500.0,
                run_results=[],
                per_step_pass_rates=[0.6],
                per_expectation_pass_rates={},
                tags=["weather"],
            ),
        ],
    )


class TestReporter:
    def test_print_report_no_crash(self):
        """Printing a report should not crash."""
        report = make_report()
        print_report(report)

    def test_print_report_verbose(self):
        report = make_report()
        print_report(report, verbose=True)

    def test_print_fuzz_report(self):
        report = FuzzReport(
            scenario_name="test",
            total_mutations=10,
            total_breaks=2,
            robustness_score=0.8,
            results=[],
            by_mutation_type={"numeric_extreme": 0.9, "null_injection": 0.7},
            by_tool={"calc": 0.8},
        )
        print_fuzz_report(report)

    def test_print_minrepro(self):
        result = MinReproResult(
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
        print_minrepro_result(result)


class TestHTMLReport:
    def test_generates_html(self, tmp_path):
        report = make_report()
        path = generate_html_report(report, str(tmp_path / "report.html"))
        content = (tmp_path / "report.html").read_text()
        assert "<html" in content
        assert "TraceForge Report" in content
        assert "calc_test" in content
        assert "weather_test" in content

    def test_html_with_fuzz(self, tmp_path):
        report = make_report()
        fuzz = FuzzReport(
            scenario_name="test",
            total_mutations=5,
            total_breaks=1,
            robustness_score=0.8,
            results=[],
            by_mutation_type={"numeric_extreme": 0.9},
            by_tool={"calc": 0.8},
        )
        path = generate_html_report(
            report, str(tmp_path / "report.html"), fuzz_report=fuzz
        )
        content = (tmp_path / "report.html").read_text()
        assert "Fuzzing" in content

    def test_html_with_minrepro(self, tmp_path):
        report = make_report()
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
        path = generate_html_report(
            report, str(tmp_path / "report.html"), minrepro_results=[mr]
        )
        content = (tmp_path / "report.html").read_text()
        assert "Minimal" in content
