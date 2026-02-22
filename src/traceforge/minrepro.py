"""Delta debugging minimal failure reproduction."""

from traceforge.evaluator import Evaluator
from traceforge.harness import Harness
from traceforge.models import MinReproResult, Scenario, TraceIR
from traceforge.trace_store import TraceStore


class MinReproExtractor:
    """Applies ddmin to minimize failing traces."""

    def __init__(
        self,
        harness: Harness,
        evaluator: Evaluator,
        trace_store: TraceStore,
    ):
        self.harness = harness
        self.evaluator = evaluator
        self.store = trace_store
        self._iteration_count = 0

    def minimize(self, trace_id: str, scenario: Scenario) -> MinReproResult:
        """Apply delta debugging to minimize a failing trace."""
        original_trace = self.store.load(trace_id)
        n = len(original_trace.steps)
        step_indices = list(range(n))

        orig_tc_count = sum(len(s.tool_calls) for s in original_trace.steps)

        # Verify the original still fails
        if not self._test_fails(scenario, step_indices):
            return MinReproResult(
                original_trace_id=trace_id,
                original_step_count=n,
                original_tool_call_count=orig_tc_count,
                minimized_trace_id=trace_id,
                minimized_step_count=n,
                minimized_tool_call_count=orig_tc_count,
                reduction_ratio=0.0,
                iterations_taken=0,
                failure_still_reproduces=False,
                minimized_steps=step_indices,
            )

        minimized = self._ddmin(scenario, step_indices)
        minimized_trace = self._run_subset(scenario, minimized)

        min_tc_count = sum(len(s.tool_calls) for s in minimized_trace.steps)

        return MinReproResult(
            original_trace_id=trace_id,
            original_step_count=n,
            original_tool_call_count=orig_tc_count,
            minimized_trace_id=minimized_trace.trace_id,
            minimized_step_count=len(minimized),
            minimized_tool_call_count=min_tc_count,
            reduction_ratio=1.0 - (len(minimized) / n) if n > 0 else 0.0,
            iterations_taken=self._iteration_count,
            failure_still_reproduces=True,
            minimized_steps=minimized,
        )

    def _ddmin(self, scenario: Scenario, steps: list[int], granularity: int = 2) -> list[int]:
        """Classic ddmin algorithm adapted for agent steps."""
        self._iteration_count = 0

        while len(steps) > 1:
            chunk_size = max(1, len(steps) // granularity)
            chunks = [steps[i:i + chunk_size] for i in range(0, len(steps), chunk_size)]

            reduced = False
            for chunk in chunks:
                candidate = [s for s in steps if s not in chunk]
                self._iteration_count += 1
                if candidate and self._test_fails(scenario, candidate):
                    steps = candidate
                    granularity = max(granularity - 1, 2)
                    reduced = True
                    break

            if not reduced:
                if granularity >= len(steps):
                    break
                granularity = min(granularity * 2, len(steps))

        return steps

    def _test_fails(self, scenario: Scenario, step_indices: list[int]) -> bool:
        """Run scenario with only the specified steps and check if it still fails."""
        subset_scenario = self._create_subset_scenario(scenario, step_indices)
        traces = self.harness.run_scenario(subset_scenario, runs=1)
        if not traces:
            return False
        result = self.evaluator.evaluate(traces[0], subset_scenario)
        return not result.passed

    def _create_subset_scenario(self, scenario: Scenario, step_indices: list[int]) -> Scenario:
        """Create a new scenario with only the specified steps."""
        subset_steps = [scenario.steps[i] for i in step_indices if i < len(scenario.steps)]
        return scenario.model_copy(update={"steps": subset_steps, "runs": 1})

    def _run_subset(self, scenario: Scenario, step_indices: list[int]) -> TraceIR:
        """Run subset and return the trace."""
        subset = self._create_subset_scenario(scenario, step_indices)
        traces = self.harness.run_scenario(subset, runs=1)
        return traces[0]
