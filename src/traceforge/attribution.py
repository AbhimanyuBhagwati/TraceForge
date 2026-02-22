"""Causal attribution engine — counterfactual replay to find WHY agents fail."""

from collections import defaultdict
from typing import Optional

from traceforge.models import (
    CausalReport,
    CounterfactualResult,
    Intervention,
    InterventionType,
    Scenario,
    TraceIR,
)


class InterventionGenerator:
    """Generates counterfactual interventions for a failing trace."""

    def generate_all(
        self, scenario: Scenario, trace: TraceIR, failing_step: int
    ) -> list[Intervention]:
        interventions = []
        interventions.extend(self._tool_output_format_interventions(trace, failing_step))
        interventions.extend(self._tool_output_value_interventions(trace, failing_step))
        interventions.extend(self._tool_output_field_interventions(trace, failing_step))
        interventions.extend(self._context_truncation_interventions(trace, failing_step))
        interventions.extend(self._system_prompt_clause_interventions(scenario))
        interventions.extend(self._tool_schema_interventions(scenario, failing_step))
        return interventions

    def _tool_output_format_interventions(
        self, trace: TraceIR, step_idx: int
    ) -> list[Intervention]:
        interventions = []
        if step_idx >= len(trace.steps):
            return interventions
        step = trace.steps[step_idx]
        for tc in step.tool_calls:
            for key, value in tc.response.items():
                if isinstance(value, (int, float)):
                    interventions.append(Intervention(
                        intervention_type=InterventionType.TOOL_OUTPUT_FORMAT,
                        description=f"Changed '{key}' from {type(value).__name__} to string",
                        target_step=step_idx,
                        target_tool=tc.tool_name,
                        target_field=key,
                        original_value=value,
                        modified_value=str(value),
                    ))
                elif isinstance(value, str):
                    try:
                        num = float(value)
                        interventions.append(Intervention(
                            intervention_type=InterventionType.TOOL_OUTPUT_FORMAT,
                            description=f"Changed '{key}' from string to number",
                            target_step=step_idx,
                            target_tool=tc.tool_name,
                            target_field=key,
                            original_value=value,
                            modified_value=num,
                        ))
                    except ValueError:
                        pass
        return interventions

    def _tool_output_value_interventions(
        self, trace: TraceIR, step_idx: int
    ) -> list[Intervention]:
        interventions = []
        if step_idx >= len(trace.steps):
            return interventions
        step = trace.steps[step_idx]
        for tc in step.tool_calls:
            for key, value in tc.response.items():
                if isinstance(value, bool):
                    interventions.append(Intervention(
                        intervention_type=InterventionType.TOOL_OUTPUT_VALUE,
                        description=f"Flipped '{key}' from {value} to {not value}",
                        target_step=step_idx,
                        target_tool=tc.tool_name,
                        target_field=key,
                        original_value=value,
                        modified_value=not value,
                    ))
                elif isinstance(value, (int, float)) and value != 0:
                    for new_val, desc in [
                        (0, "zero"),
                        (-value, "negated"),
                        (value * 2, "doubled"),
                        (value / 2, "halved"),
                    ]:
                        interventions.append(Intervention(
                            intervention_type=InterventionType.TOOL_OUTPUT_VALUE,
                            description=f"Changed '{key}' to {desc} ({new_val})",
                            target_step=step_idx,
                            target_tool=tc.tool_name,
                            target_field=key,
                            original_value=value,
                            modified_value=new_val,
                        ))
        return interventions

    def _tool_output_field_interventions(
        self, trace: TraceIR, step_idx: int
    ) -> list[Intervention]:
        interventions = []
        if step_idx >= len(trace.steps):
            return interventions
        step = trace.steps[step_idx]
        for tc in step.tool_calls:
            for key in tc.response:
                interventions.append(Intervention(
                    intervention_type=InterventionType.TOOL_OUTPUT_FIELDS,
                    description=f"Removed field '{key}' from {tc.tool_name} response",
                    target_step=step_idx,
                    target_tool=tc.tool_name,
                    target_field=key,
                    original_value=tc.response[key],
                    modified_value="__REMOVE__",
                ))
            interventions.append(Intervention(
                intervention_type=InterventionType.TOOL_OUTPUT_FIELDS,
                description=f"Added unexpected field 'debug_info' to {tc.tool_name} response",
                target_step=step_idx,
                target_tool=tc.tool_name,
                target_field="debug_info",
                original_value=None,
                modified_value="internal debug data - ignore this",
            ))
        return interventions

    def _context_truncation_interventions(
        self, trace: TraceIR, step_idx: int
    ) -> list[Intervention]:
        interventions = []
        if step_idx > 0:
            for keep_last in [1, 2, max(1, step_idx // 2)]:
                if keep_last < step_idx:
                    interventions.append(Intervention(
                        intervention_type=InterventionType.CONTEXT_TRUNCATION,
                        description=f"Truncated context to last {keep_last} steps (from {step_idx})",
                        target_step=step_idx,
                        original_value=step_idx,
                        modified_value=keep_last,
                    ))
        return interventions

    def _system_prompt_clause_interventions(
        self, scenario: Scenario
    ) -> list[Intervention]:
        interventions = []
        prompt = scenario.agent.system_prompt or ""
        sentences = [s.strip() for s in prompt.split(".") if s.strip()]
        for i, sentence in enumerate(sentences):
            reduced = ". ".join(s for j, s in enumerate(sentences) if j != i) + "."
            interventions.append(Intervention(
                intervention_type=InterventionType.SYSTEM_PROMPT_CLAUSE,
                description=f"Removed prompt sentence {i}: '{sentence[:50]}...'",
                original_value=prompt,
                modified_value=reduced,
            ))
        return interventions

    def _tool_schema_interventions(
        self, scenario: Scenario, step_idx: int
    ) -> list[Intervention]:
        interventions = []
        for tool in scenario.agent.tools:
            props = tool.parameters.get("properties", {})
            for param_name in props:
                interventions.append(Intervention(
                    intervention_type=InterventionType.TOOL_SCHEMA_CHANGE,
                    description=f"Renamed '{param_name}' to '{param_name}_v2' in {tool.name} schema",
                    target_tool=tool.name,
                    target_field=param_name,
                    original_value=param_name,
                    modified_value=f"{param_name}_v2",
                ))
        return interventions


class CausalAttributionEngine:
    """Runs counterfactual experiments to determine WHY an agent fails."""

    def __init__(self, harness, evaluator, trace_store, judge=None):
        self.harness = harness
        self.evaluator = evaluator
        self.store = trace_store
        self.judge = judge
        self.generator = InterventionGenerator()

    def attribute(
        self,
        trace_id: str,
        scenario: Scenario,
        confirmation_runs: int = 3,
        max_interventions: int = 50,
    ) -> CausalReport:
        trace = self.store.load(trace_id)

        # Find the first failing step
        baseline_result = self.evaluator.evaluate(trace, scenario)
        failing_step = self._find_first_failing_step(baseline_result)
        if failing_step is None:
            raise ValueError(f"Trace {trace_id} does not appear to fail any step")

        # Generate interventions
        all_interventions = self.generator.generate_all(scenario, trace, failing_step)
        interventions = all_interventions[:max_interventions]

        # Run counterfactual experiments
        results = []
        for intervention in interventions:
            cf_result = self._run_counterfactual(
                scenario, trace, intervention, failing_step, confirmation_runs
            )
            results.append(cf_result)

        flips = [r for r in results if r.flipped]
        causal_factors = self._rank_factors(results)
        summary = self._generate_summary(scenario, trace, failing_step, causal_factors)

        return CausalReport(
            scenario_name=scenario.name,
            failing_trace_id=trace_id,
            failing_step=failing_step,
            total_interventions=len(results),
            total_flips=len(flips),
            interventions=results,
            causal_factors=causal_factors,
            summary=summary,
        )

    def _run_counterfactual(
        self, scenario, trace, intervention, failing_step, runs
    ) -> CounterfactualResult:
        modified_scenario = self._apply_intervention(scenario, trace, intervention)

        pass_count = 0
        last_trace_id = ""
        for _ in range(runs):
            try:
                traces = self.harness.run_scenario(modified_scenario, runs=1)
                if traces:
                    result = self.evaluator.evaluate(traces[0], modified_scenario)
                    if result.passed:
                        pass_count += 1
                    last_trace_id = traces[0].trace_id
            except Exception:
                pass

        cf_passed = pass_count > runs / 2
        confidence = max(pass_count, runs - pass_count) / runs

        return CounterfactualResult(
            intervention=intervention,
            original_passed=False,
            counterfactual_passed=cf_passed,
            flipped=cf_passed,
            trace_id=last_trace_id,
            confidence=confidence,
        )

    def _apply_intervention(self, scenario, trace, intervention):
        modified = scenario.model_copy(deep=True)

        match intervention.intervention_type:
            case InterventionType.TOOL_OUTPUT_FORMAT | InterventionType.TOOL_OUTPUT_VALUE:
                for tool in modified.agent.tools:
                    if tool.name == intervention.target_tool and tool.mock_responses:
                        for resp in tool.mock_responses:
                            if intervention.target_field in resp:
                                resp[intervention.target_field] = intervention.modified_value

            case InterventionType.TOOL_OUTPUT_FIELDS:
                for tool in modified.agent.tools:
                    if tool.name == intervention.target_tool and tool.mock_responses:
                        for resp in tool.mock_responses:
                            if intervention.modified_value == "__REMOVE__":
                                resp.pop(intervention.target_field, None)
                            else:
                                resp[intervention.target_field] = intervention.modified_value

            case InterventionType.CONTEXT_TRUNCATION:
                keep = intervention.modified_value
                if isinstance(keep, int) and keep < len(modified.steps):
                    modified.steps = modified.steps[-keep:]

            case InterventionType.SYSTEM_PROMPT_CLAUSE:
                modified.agent.system_prompt = intervention.modified_value

            case InterventionType.TOOL_SCHEMA_CHANGE:
                for tool in modified.agent.tools:
                    if tool.name == intervention.target_tool:
                        props = tool.parameters.get("properties", {})
                        if intervention.target_field in props:
                            props[intervention.modified_value] = props.pop(
                                intervention.target_field
                            )
                            required = tool.parameters.get("required", [])
                            if intervention.target_field in required:
                                idx = required.index(intervention.target_field)
                                required[idx] = intervention.modified_value

        modified.runs = 1
        return modified

    def _rank_factors(self, results: list[CounterfactualResult]) -> list[dict]:
        type_counts = defaultdict(lambda: {"total": 0, "flips": 0})

        for r in results:
            t = r.intervention.intervention_type.value
            type_counts[t]["total"] += 1
            if r.flipped:
                type_counts[t]["flips"] += 1

        factors = []
        for factor_type, counts in type_counts.items():
            if counts["total"] > 0:
                sensitivity = counts["flips"] / counts["total"]
                factors.append({
                    "factor": factor_type,
                    "sensitivity": round(sensitivity, 3),
                    "flips": counts["flips"],
                    "total": counts["total"],
                    "description": (
                        f"{counts['flips']}/{counts['total']} interventions of type "
                        f"'{factor_type}' flipped the outcome"
                    ),
                })

        factors.sort(key=lambda f: f["sensitivity"], reverse=True)
        return factors

    def _generate_summary(self, scenario, trace, failing_step, factors) -> str:
        if not factors:
            return "No causal factors identified. The failure may be intrinsic to the model's capabilities."

        top = factors[0]
        lines = [
            f"Causal analysis of '{scenario.name}' failure at step {failing_step}:",
            "",
            f"Primary cause: {top['factor']} (sensitivity: {top['sensitivity']:.0%})",
            f"  {top['description']}",
        ]
        if len(factors) > 1:
            lines.append("")
            lines.append("Secondary factors:")
            for f in factors[1:3]:
                lines.append(f"  - {f['factor']}: {f['sensitivity']:.0%} sensitivity")

        return "\n".join(lines)

    def _find_first_failing_step(self, run_result) -> Optional[int]:
        for sr in run_result.step_results:
            if not sr.all_passed:
                return sr.step_index
        return None
