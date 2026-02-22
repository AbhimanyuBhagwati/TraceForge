"""Evaluation engine — scores traces against scenario expectations."""

import re
from datetime import datetime, timezone

from traceforge.judge import JudgeClient
from traceforge.models import (
    Expectation,
    ExpectationResult,
    ExpectationType,
    RunResult,
    Scenario,
    ScenarioResult,
    StepResult,
    TraceIR,
)


class Evaluator:
    """Evaluates a TraceIR against a Scenario's expectations."""

    def __init__(self, judge: JudgeClient | None = None):
        self.judge = judge

    def evaluate(self, trace: TraceIR, scenario: Scenario) -> RunResult:
        """Evaluate a trace against a scenario's expectations."""
        step_results = []

        for step_idx, step in enumerate(scenario.steps):
            if step_idx >= len(trace.steps):
                # Trace has fewer steps than scenario
                results = [
                    ExpectationResult(
                        expectation=exp,
                        passed=False,
                        message=f"Step {step_idx} not present in trace",
                    )
                    for exp in step.expectations
                ]
                step_results.append(StepResult(
                    step_index=step_idx,
                    user_message=step.user_message,
                    results=results,
                    all_passed=False,
                ))
                continue

            trace_step = trace.steps[step_idx]
            results = []

            for exp in step.expectations:
                result = self._evaluate_expectation(exp, trace_step, trace)
                results.append(result)

            step_results.append(StepResult(
                step_index=step_idx,
                user_message=step.user_message,
                results=results,
                all_passed=all(r.passed for r in results),
            ))

        all_passed = all(sr.all_passed for sr in step_results)

        return RunResult(
            scenario_name=scenario.name,
            run_number=trace.run_number,
            trace_id=trace.trace_id,
            step_results=step_results,
            passed=all_passed,
            total_latency_ms=trace.total_latency_ms,
            timestamp=trace.timestamp,
        )

    def _evaluate_expectation(self, exp: Expectation, step, trace: TraceIR) -> ExpectationResult:
        """Dispatch to the appropriate evaluation function."""
        handlers = {
            ExpectationType.TOOL_CALLED: self._eval_tool_called,
            ExpectationType.TOOL_NOT_CALLED: self._eval_tool_not_called,
            ExpectationType.TOOL_ARGS_CONTAIN: self._eval_tool_args_contain,
            ExpectationType.RESPONSE_CONTAINS: self._eval_response_contains,
            ExpectationType.RESPONSE_NOT_CONTAINS: self._eval_response_not_contains,
            ExpectationType.RESPONSE_MATCHES_REGEX: self._eval_response_matches_regex,
            ExpectationType.LLM_JUDGE: self._eval_llm_judge,
            ExpectationType.LATENCY_UNDER: self._eval_latency_under,
            ExpectationType.NO_TOOL_ERRORS: self._eval_no_tool_errors,
            ExpectationType.TOOL_CALL_COUNT: self._eval_tool_call_count,
        }
        handler = handlers.get(exp.type)
        if not handler:
            return ExpectationResult(
                expectation=exp,
                passed=False,
                message=f"Unknown expectation type: {exp.type}",
            )
        return handler(exp, step, trace)

    def _eval_tool_called(self, exp, step, trace) -> ExpectationResult:
        tool_names = [tc.tool_name for tc in step.tool_calls]
        passed = exp.tool in tool_names
        return ExpectationResult(
            expectation=exp,
            passed=passed,
            message=f"Tool '{exp.tool}' {'was' if passed else 'was NOT'} called",
            details={"called_tools": tool_names},
        )

    def _eval_tool_not_called(self, exp, step, trace) -> ExpectationResult:
        tool_names = [tc.tool_name for tc in step.tool_calls]
        passed = exp.tool not in tool_names
        return ExpectationResult(
            expectation=exp,
            passed=passed,
            message=f"Tool '{exp.tool}' {'was NOT' if passed else 'WAS'} called",
            details={"called_tools": tool_names},
        )

    def _eval_tool_args_contain(self, exp, step, trace) -> ExpectationResult:
        for tc in step.tool_calls:
            if tc.tool_name == exp.tool:
                if exp.args_contain:
                    all_match = all(
                        str(exp.args_contain[k]).lower() in str(tc.arguments.get(k, "")).lower()
                        for k in exp.args_contain
                    )
                    if all_match:
                        return ExpectationResult(
                            expectation=exp,
                            passed=True,
                            message=f"Tool '{exp.tool}' args contain expected values",
                            details={"actual_args": tc.arguments},
                        )
        return ExpectationResult(
            expectation=exp,
            passed=False,
            message=f"Tool '{exp.tool}' args do not contain expected values",
            details={"expected": exp.args_contain},
        )

    def _eval_response_contains(self, exp, step, trace) -> ExpectationResult:
        response_lower = step.assistant_response.lower()
        if exp.values:
            found = [v for v in exp.values if v.lower() in response_lower]
            passed = len(found) > 0
            return ExpectationResult(
                expectation=exp,
                passed=passed,
                message=f"Response {'contains' if passed else 'does NOT contain'} any of {exp.values}",
                details={"found": found},
            )
        if exp.value:
            passed = exp.value.lower() in response_lower
            return ExpectationResult(
                expectation=exp,
                passed=passed,
                message=f"Response {'contains' if passed else 'does NOT contain'} '{exp.value}'",
            )
        return ExpectationResult(
            expectation=exp, passed=False, message="No value or values specified",
        )

    def _eval_response_not_contains(self, exp, step, trace) -> ExpectationResult:
        response_lower = step.assistant_response.lower()
        if exp.values:
            found = [v for v in exp.values if v.lower() in response_lower]
            passed = len(found) == 0
            return ExpectationResult(
                expectation=exp,
                passed=passed,
                message=f"Response {'does NOT contain' if passed else 'CONTAINS'} {found or exp.values}",
            )
        if exp.value:
            passed = exp.value.lower() not in response_lower
            return ExpectationResult(
                expectation=exp,
                passed=passed,
                message=f"Response {'does NOT contain' if passed else 'CONTAINS'} '{exp.value}'",
            )
        return ExpectationResult(
            expectation=exp, passed=False, message="No value or values specified",
        )

    def _eval_response_matches_regex(self, exp, step, trace) -> ExpectationResult:
        if not exp.value:
            return ExpectationResult(
                expectation=exp, passed=False, message="No regex pattern specified",
            )
        try:
            passed = bool(re.search(exp.value, step.assistant_response))
        except re.error as e:
            return ExpectationResult(
                expectation=exp, passed=False, message=f"Invalid regex: {e}",
            )
        return ExpectationResult(
            expectation=exp,
            passed=passed,
            message=f"Response {'matches' if passed else 'does NOT match'} regex '{exp.value}'",
        )

    def _eval_llm_judge(self, exp, step, trace) -> ExpectationResult:
        if not self.judge:
            return ExpectationResult(
                expectation=exp,
                passed=False,
                message="LLM judge not configured",
            )
        if not exp.criterion:
            return ExpectationResult(
                expectation=exp,
                passed=False,
                message="No criterion specified for LLM judge",
            )
        tool_calls_data = [
            {"tool": tc.tool_name, "args": tc.arguments, "response": tc.response}
            for tc in step.tool_calls
        ]
        result = self.judge.judge(
            criterion=exp.criterion,
            user_message=step.user_message,
            assistant_response=step.assistant_response,
            tool_calls=tool_calls_data if tool_calls_data else None,
        )
        return ExpectationResult(
            expectation=exp,
            passed=result["passed"],
            message=result["reasoning"],
            details={"score": result["score"]},
        )

    def _eval_latency_under(self, exp, step, trace) -> ExpectationResult:
        if exp.max_ms is None:
            return ExpectationResult(
                expectation=exp, passed=False, message="No max_ms specified",
            )
        passed = step.latency_ms <= exp.max_ms
        return ExpectationResult(
            expectation=exp,
            passed=passed,
            message=f"Latency {step.latency_ms:.0f}ms {'<=' if passed else '>'} {exp.max_ms}ms",
        )

    def _eval_no_tool_errors(self, exp, step, trace) -> ExpectationResult:
        errors = [
            tc for tc in step.tool_calls
            if "error" in tc.response
        ]
        passed = len(errors) == 0
        return ExpectationResult(
            expectation=exp,
            passed=passed,
            message=f"{'No' if passed else len(errors)} tool errors found",
            details={"errors": [{"tool": e.tool_name, "error": e.response["error"]} for e in errors]} if errors else None,
        )

    def _eval_tool_call_count(self, exp, step, trace) -> ExpectationResult:
        if exp.count is None:
            return ExpectationResult(
                expectation=exp, passed=False, message="No count specified",
            )
        if exp.tool:
            actual = sum(1 for tc in step.tool_calls if tc.tool_name == exp.tool)
        else:
            actual = len(step.tool_calls)

        op = exp.operator or "eq"
        if op == "eq":
            passed = actual == exp.count
        elif op == "gte":
            passed = actual >= exp.count
        elif op == "lte":
            passed = actual <= exp.count
        else:
            passed = False

        return ExpectationResult(
            expectation=exp,
            passed=passed,
            message=f"Tool call count: {actual} {op} {exp.count} -> {'PASS' if passed else 'FAIL'}",
            details={"actual": actual, "expected": exp.count, "operator": op},
        )


def calculate_consistency(run_results: list[RunResult]) -> float:
    """Calculate consistency score across runs."""
    if not run_results:
        return 0.0
    passed = sum(1 for r in run_results if r.passed)
    failed = len(run_results) - passed
    return 1.0 - (2 * min(passed, failed) / len(run_results))


def aggregate_results(
    scenario: Scenario, run_results: list[RunResult]
) -> ScenarioResult:
    """Aggregate run results into a ScenarioResult."""
    passed_runs = sum(1 for r in run_results if r.passed)
    failed_runs = len(run_results) - passed_runs
    latencies = [r.total_latency_ms for r in run_results]

    # Per-step pass rates
    num_steps = len(scenario.steps)
    per_step_pass_rates = []
    for step_idx in range(num_steps):
        step_passes = sum(
            1 for r in run_results
            if step_idx < len(r.step_results) and r.step_results[step_idx].all_passed
        )
        per_step_pass_rates.append(step_passes / len(run_results) if run_results else 0.0)

    # Per-expectation pass rates
    per_exp_rates: dict[str, list[bool]] = {}
    for r in run_results:
        for sr in r.step_results:
            for er in sr.results:
                key = f"{er.expectation.type.value}"
                if er.expectation.tool:
                    key += f":{er.expectation.tool}"
                per_exp_rates.setdefault(key, []).append(er.passed)

    per_exp_pass_rates = {
        k: sum(v) / len(v) for k, v in per_exp_rates.items()
    }

    return ScenarioResult(
        scenario_name=scenario.name,
        description=scenario.description,
        total_runs=len(run_results),
        passed_runs=passed_runs,
        failed_runs=failed_runs,
        pass_rate=passed_runs / len(run_results) if run_results else 0.0,
        consistency_score=calculate_consistency(run_results),
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        min_latency_ms=min(latencies) if latencies else 0.0,
        max_latency_ms=max(latencies) if latencies else 0.0,
        run_results=run_results,
        per_step_pass_rates=per_step_pass_rates,
        per_expectation_pass_rates=per_exp_pass_rates,
        tags=scenario.tags,
    )
