"""Virtual replay (no model) + seeded rerun + trace diffing."""

from traceforge.evaluator import Evaluator
from traceforge.judge import JudgeClient
from traceforge.models import RunResult, Scenario, TraceIR
from traceforge.trace_store import TraceStore


class ReplayEngine:
    """Replays stored traces for evaluation without calling the model."""

    def __init__(self, trace_store: TraceStore):
        self.store = trace_store

    def virtual_replay(
        self,
        trace_id: str,
        scenario: Scenario,
        judge: JudgeClient | None = None,
    ) -> RunResult:
        """Re-evaluate a stored trace against (possibly updated) expectations.

        Does NOT call Ollama for the agent. May call Ollama for LLM-judge if needed.
        """
        trace = self.store.load(trace_id)
        evaluator = Evaluator(judge=judge)
        return evaluator.evaluate(trace, scenario)

    def diff_traces(self, trace_id_1: str, trace_id_2: str) -> dict:
        """Compare two traces and produce a structured diff."""
        trace1 = self.store.load(trace_id_1)
        trace2 = self.store.load(trace_id_2)

        diffs = []
        max_steps = max(len(trace1.steps), len(trace2.steps))

        for i in range(max_steps):
            s1 = trace1.steps[i] if i < len(trace1.steps) else None
            s2 = trace2.steps[i] if i < len(trace2.steps) else None

            if s1 is None or s2 is None:
                diffs.append({
                    "step": i,
                    "missing_in": "trace_1" if s1 is None else "trace_2",
                })
                continue

            step_diff = {
                "step": i,
                "tool_calls_changed": s1.tool_calls != s2.tool_calls,
                "response_changed": s1.assistant_response != s2.assistant_response,
                "tool_call_details": self._diff_tool_calls(s1.tool_calls, s2.tool_calls),
            }
            if step_diff["response_changed"]:
                step_diff["response_1"] = s1.assistant_response[:200]
                step_diff["response_2"] = s2.assistant_response[:200]
            diffs.append(step_diff)

        return {
            "trace_1": trace_id_1,
            "trace_2": trace_id_2,
            "step_count_1": len(trace1.steps),
            "step_count_2": len(trace2.steps),
            "diffs": diffs,
        }

    def _diff_tool_calls(self, calls1, calls2) -> list[dict]:
        """Produce per-tool-call diffs."""
        diffs = []
        max_calls = max(len(calls1), len(calls2))
        for i in range(max_calls):
            tc1 = calls1[i] if i < len(calls1) else None
            tc2 = calls2[i] if i < len(calls2) else None

            if tc1 is None or tc2 is None:
                diffs.append({
                    "index": i,
                    "missing_in": "trace_1" if tc1 is None else "trace_2",
                })
                continue

            diff = {"index": i}
            if tc1.tool_name != tc2.tool_name:
                diff["tool_name_changed"] = [tc1.tool_name, tc2.tool_name]
            if tc1.arguments != tc2.arguments:
                diff["arguments_changed"] = True
            if tc1.response != tc2.response:
                diff["response_changed"] = True
            if len(diff) > 1:  # more than just index
                diffs.append(diff)

        return diffs
