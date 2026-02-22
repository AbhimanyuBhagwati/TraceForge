"""Differential fuzzing orchestrator."""

from traceforge.evaluator import Evaluator
from traceforge.harness import Harness
from traceforge.models import (
    FuzzReport,
    FuzzResult,
    MutatorConfig,
    Scenario,
    ToolDefinition,
)
from traceforge.mutators import generate_mutations
from traceforge.trace_store import TraceStore


class DifferentialFuzzer:
    """Fuzzes tool mock responses and measures agent robustness."""

    def __init__(
        self,
        harness: Harness,
        evaluator: Evaluator,
        trace_store: TraceStore,
    ):
        self.harness = harness
        self.evaluator = evaluator
        self.store = trace_store

    def fuzz_scenario(self, scenario: Scenario, config: MutatorConfig) -> FuzzReport:
        """Run differential fuzzing on a scenario.

        1. Run baseline, get passing traces
        2. For each passing trace's tool calls, mutate the mock responses
        3. Re-run with mutated mocks and check if agent still passes
        4. Compute robustness scores
        """
        # Run baseline
        baseline_traces = self.harness.run_scenario(scenario, runs=1)
        baseline_results = [
            self.evaluator.evaluate(t, scenario) for t in baseline_traces
        ]

        fuzz_results = []
        for trace, result in zip(baseline_traces, baseline_results):
            if not result.passed:
                continue  # Only fuzz passing runs

            for step in trace.steps:
                for tc in step.tool_calls:
                    tool_def = next(
                        (t for t in scenario.agent.tools if t.name == tc.tool_name),
                        None,
                    )
                    if not tool_def:
                        continue

                    mutations = generate_mutations(
                        tool_def,
                        tc.response,
                        config.mutation_types,
                        config.mutations_per_tool,
                    )

                    for mutation in mutations:
                        mutated_scenario = self._apply_mutation(
                            scenario, step.step_index, tc.tool_name, mutation
                        )
                        mutated_traces = self.harness.run_scenario(
                            mutated_scenario, runs=1
                        )
                        if not mutated_traces:
                            continue
                        mutated_result = self.evaluator.evaluate(
                            mutated_traces[0], mutated_scenario
                        )

                        fuzz_results.append(FuzzResult(
                            step_index=step.step_index,
                            tool_name=tc.tool_name,
                            mutation=mutation,
                            trace_id=mutated_traces[0].trace_id,
                            original_passed=True,
                            mutated_passed=mutated_result.passed,
                            broke_agent=not mutated_result.passed,
                        ))

        return self._build_report(scenario.name, fuzz_results)

    def _apply_mutation(self, scenario, step_index, tool_name, mutation):
        """Create a modified scenario with mutated mock responses."""
        new_tools = []
        for tool in scenario.agent.tools:
            if tool.name == tool_name:
                new_tool = tool.model_copy(
                    update={"mock_responses": [mutation.mutated_args]}
                )
                new_tools.append(new_tool)
            else:
                new_tools.append(tool)

        new_agent = scenario.agent.model_copy(update={"tools": new_tools})
        return scenario.model_copy(update={"agent": new_agent, "runs": 1})

    def _build_report(self, scenario_name: str, results: list[FuzzResult]) -> FuzzReport:
        total = len(results)
        breaks = sum(1 for r in results if r.broke_agent)

        # By mutation type
        by_type: dict[str, list[bool]] = {}
        for r in results:
            by_type.setdefault(r.mutation.mutation_type, []).append(not r.broke_agent)
        by_mutation_type = {
            k: sum(v) / len(v) if v else 1.0 for k, v in by_type.items()
        }

        # By tool
        by_t: dict[str, list[bool]] = {}
        for r in results:
            by_t.setdefault(r.tool_name, []).append(not r.broke_agent)
        by_tool = {
            k: sum(v) / len(v) if v else 1.0 for k, v in by_t.items()
        }

        return FuzzReport(
            scenario_name=scenario_name,
            total_mutations=total,
            total_breaks=breaks,
            robustness_score=1.0 - (breaks / total) if total > 0 else 1.0,
            results=results,
            by_mutation_type=by_mutation_type,
            by_tool=by_tool,
        )
