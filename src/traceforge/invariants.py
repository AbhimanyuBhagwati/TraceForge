"""Invariant mining — Daikon-style behavioral invariant discovery for agent traces."""

import re
from collections import defaultdict
from itertools import combinations

from traceforge.models import (
    Invariant,
    InvariantReport,
    InvariantType,
    TraceIR,
)


class ToolOrderMiner:
    """Discovers tool call ordering invariants."""

    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        all_tools = set()
        for t in traces:
            for s in t.steps:
                for tc in s.tool_calls:
                    all_tools.add(tc.tool_name)

        candidates = []
        for a, b in combinations(sorted(all_tools), 2):
            a_before_b = True
            b_before_a = True
            count = 0
            for t in traces:
                a_idx = None
                b_idx = None
                for i, s in enumerate(t.steps):
                    for tc in s.tool_calls:
                        if tc.tool_name == a and a_idx is None:
                            a_idx = i
                        if tc.tool_name == b and b_idx is None:
                            b_idx = i
                if a_idx is not None and b_idx is not None:
                    count += 1
                    if a_idx > b_idx:
                        a_before_b = False
                    if b_idx > a_idx:
                        b_before_a = False

            if count >= 2 and a_before_b:
                candidates.append(Invariant(
                    invariant_type=InvariantType.TOOL_ORDER,
                    description=f"'{a}' is always called before '{b}'",
                    formal=f"order({a}) < order({b})",
                    confidence=1.0,
                    support=count,
                    violations=0,
                    details={"tool_a": a, "tool_b": b},
                ))
            if count >= 2 and b_before_a:
                candidates.append(Invariant(
                    invariant_type=InvariantType.TOOL_ORDER,
                    description=f"'{b}' is always called before '{a}'",
                    formal=f"order({b}) < order({a})",
                    confidence=1.0,
                    support=count,
                    violations=0,
                    details={"tool_a": b, "tool_b": a},
                ))
        return candidates


class ToolPresenceMiner:
    """Discovers which tools are always/never called at each step."""

    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        candidates = []
        if not traces:
            return candidates

        max_steps = max(len(t.steps) for t in traces)
        all_tools = set()
        for t in traces:
            for s in t.steps:
                for tc in s.tool_calls:
                    all_tools.add(tc.tool_name)

        for step_idx in range(max_steps):
            for tool_name in sorted(all_tools):
                applicable_traces = [t for t in traces if step_idx < len(t.steps)]
                applicable = len(applicable_traces)
                if applicable < 2:
                    continue

                always_called = all(
                    any(tc.tool_name == tool_name for tc in t.steps[step_idx].tool_calls)
                    for t in applicable_traces
                )
                never_called = all(
                    not any(tc.tool_name == tool_name for tc in t.steps[step_idx].tool_calls)
                    for t in applicable_traces
                )

                if always_called:
                    candidates.append(Invariant(
                        invariant_type=InvariantType.TOOL_ALWAYS_CALLED,
                        description=f"'{tool_name}' is always called at step {step_idx}",
                        formal=f"always_called({tool_name}, step={step_idx})",
                        confidence=1.0,
                        support=applicable,
                        violations=0,
                        step_index=step_idx,
                        tool_name=tool_name,
                    ))
                if never_called:
                    candidates.append(Invariant(
                        invariant_type=InvariantType.TOOL_NEVER_CALLED,
                        description=f"'{tool_name}' is never called at step {step_idx}",
                        formal=f"never_called({tool_name}, step={step_idx})",
                        confidence=1.0,
                        support=applicable,
                        violations=0,
                        step_index=step_idx,
                        tool_name=tool_name,
                    ))
        return candidates


class ToolCallCountMiner:
    """Discovers tool call count invariants."""

    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        candidates = []
        all_tools = set()
        for t in traces:
            for s in t.steps:
                for tc in s.tool_calls:
                    all_tools.add(tc.tool_name)

        for tool_name in sorted(all_tools):
            counts = []
            for t in traces:
                count = sum(
                    1 for s in t.steps for tc in s.tool_calls if tc.tool_name == tool_name
                )
                counts.append(count)

            if counts:
                min_c, max_c = min(counts), max(counts)
                candidates.append(Invariant(
                    invariant_type=InvariantType.TOOL_CALL_COUNT,
                    description=f"'{tool_name}' is called {min_c}-{max_c} times per run",
                    formal=f"{min_c} <= count({tool_name}) <= {max_c}",
                    confidence=1.0,
                    support=len(counts),
                    violations=0,
                    tool_name=tool_name,
                    details={"min": min_c, "max": max_c},
                ))
        return candidates


class ArgRangeMiner:
    """Discovers numeric argument range invariants."""

    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        candidates = []
        arg_values: dict[tuple[str, str], list[float]] = defaultdict(list)

        for t in traces:
            for s in t.steps:
                for tc in s.tool_calls:
                    for arg_name, arg_val in tc.arguments.items():
                        if isinstance(arg_val, (int, float)):
                            arg_values[(tc.tool_name, arg_name)].append(float(arg_val))

        for (tool_name, arg_name), values in sorted(arg_values.items()):
            if len(values) >= 3:
                min_v = min(values)
                max_v = max(values)
                margin = (max_v - min_v) * 0.1 if max_v != min_v else abs(min_v) * 0.1 + 1
                candidates.append(Invariant(
                    invariant_type=InvariantType.ARG_RANGE,
                    description=(
                        f"'{tool_name}.{arg_name}' is always in range "
                        f"[{min_v - margin:.2f}, {max_v + margin:.2f}]"
                    ),
                    formal=f"{min_v - margin:.2f} <= {tool_name}.{arg_name} <= {max_v + margin:.2f}",
                    confidence=1.0,
                    support=len(values),
                    violations=0,
                    tool_name=tool_name,
                    details={"arg_name": arg_name, "min": min_v - margin, "max": max_v + margin},
                ))
        return candidates


class ArgPatternMiner:
    """Discovers string argument pattern invariants."""

    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        candidates = []
        arg_values: dict[tuple[str, str], list[str]] = defaultdict(list)

        for t in traces:
            for s in t.steps:
                for tc in s.tool_calls:
                    for arg_name, arg_val in tc.arguments.items():
                        if isinstance(arg_val, str) and len(arg_val) > 0:
                            arg_values[(tc.tool_name, arg_name)].append(arg_val)

        for (tool_name, arg_name), values in sorted(arg_values.items()):
            if len(values) >= 3:
                if all(v.strip() for v in values):
                    candidates.append(Invariant(
                        invariant_type=InvariantType.ARG_PATTERN,
                        description=f"'{tool_name}.{arg_name}' is always non-empty",
                        formal=f"len({tool_name}.{arg_name}) > 0",
                        confidence=1.0,
                        support=len(values),
                        violations=0,
                        tool_name=tool_name,
                        details={"arg_name": arg_name, "pattern": r".+"},
                    ))
        return candidates


class ArgDependencyMiner:
    """Discovers if tool arguments reference previous tool outputs."""

    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        return []


class ResponsePatternMiner:
    """Discovers response content invariants."""

    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        candidates = []
        if not traces:
            return candidates

        max_steps = max(len(t.steps) for t in traces)
        for step_idx in range(max_steps):
            lengths = []
            for t in traces:
                if step_idx < len(t.steps):
                    lengths.append(len(t.steps[step_idx].assistant_response))

            if len(lengths) >= 3:
                min_l, max_l = min(lengths), max(lengths)
                margin = max(50, int((max_l - min_l) * 0.2))
                candidates.append(Invariant(
                    invariant_type=InvariantType.RESPONSE_LENGTH,
                    description=f"Step {step_idx} response length is {min_l}-{max_l} chars",
                    formal=f"{max(0, min_l - margin)} <= len(response[{step_idx}]) <= {max_l + margin}",
                    confidence=1.0,
                    support=len(lengths),
                    violations=0,
                    step_index=step_idx,
                    details={"min": max(0, min_l - margin), "max": max_l + margin},
                ))
        return candidates


class LatencyMiner:
    """Discovers latency invariants per step."""

    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        candidates = []
        if not traces:
            return candidates

        max_steps = max(len(t.steps) for t in traces)
        for step_idx in range(max_steps):
            latencies = []
            for t in traces:
                if step_idx < len(t.steps):
                    latencies.append(t.steps[step_idx].latency_ms)

            if len(latencies) >= 3:
                sorted_lat = sorted(latencies)
                p95_idx = int(len(sorted_lat) * 0.95)
                p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
                candidates.append(Invariant(
                    invariant_type=InvariantType.STEP_LATENCY,
                    description=f"Step {step_idx} latency is under {p95:.0f}ms (p95)",
                    formal=f"latency[{step_idx}] <= {p95:.0f}ms",
                    confidence=0.95,
                    support=len(latencies),
                    violations=0,
                    step_index=step_idx,
                    details={"max_ms": p95},
                ))
        return candidates


class InvariantMiner:
    """Mines behavioral invariants from a corpus of agent execution traces."""

    def __init__(self, trace_store):
        self.store = trace_store
        self.miners = [
            ToolOrderMiner(),
            ToolPresenceMiner(),
            ToolCallCountMiner(),
            ArgRangeMiner(),
            ArgPatternMiner(),
            ArgDependencyMiner(),
            ResponsePatternMiner(),
            LatencyMiner(),
        ]

    def mine(
        self, scenario_name: str, min_confidence: float = 0.95
    ) -> InvariantReport:
        all_trace_meta = self.store.list_traces(scenario_name=scenario_name)
        passing_traces = []
        failing_traces = []
        for meta in all_trace_meta:
            trace = self.store.load(meta["trace_id"])
            if meta.get("passed"):
                passing_traces.append(trace)
            else:
                failing_traces.append(trace)

        if len(passing_traces) < 3:
            raise ValueError(
                f"Need at least 3 passing traces to mine invariants, "
                f"found {len(passing_traces)}. Run more tests first."
            )

        # Phase 1: Extract candidate invariants from passing traces
        candidates = []
        for miner in self.miners:
            candidates.extend(miner.extract_candidates(passing_traces))

        # Phase 2: Validate candidates against ALL passing traces
        validated = []
        for candidate in candidates:
            support = sum(
                1 for t in passing_traces if self._check_invariant(candidate, t)
            )
            confidence = support / len(passing_traces)
            if confidence >= min_confidence:
                candidate.confidence = confidence
                candidate.support = support
                validated.append(candidate)

        # Phase 3: Check violations in failing traces
        discriminating = []
        for inv in validated:
            violations = sum(
                1 for t in failing_traces if not self._check_invariant(inv, t)
            )
            inv.violations = violations
            if violations > 0 and failing_traces:
                discriminating.append(inv)

        discriminating.sort(
            key=lambda i: i.violations / max(len(failing_traces), 1),
            reverse=True,
        )

        suggestions = self._generate_expectations(discriminating[:10])

        return InvariantReport(
            total_traces_analyzed=len(passing_traces) + len(failing_traces),
            passing_traces=len(passing_traces),
            failing_traces=len(failing_traces),
            invariants_discovered=len(validated),
            invariants=validated,
            discriminating_invariants=discriminating,
            suggested_expectations=suggestions,
        )

    def _check_invariant(self, invariant: Invariant, trace: TraceIR) -> bool:
        match invariant.invariant_type:
            case InvariantType.TOOL_ORDER:
                d = invariant.details
                a_idx = self._first_tool_call_step(trace, d["tool_a"])
                b_idx = self._first_tool_call_step(trace, d["tool_b"])
                if a_idx is None or b_idx is None:
                    return True
                return a_idx <= b_idx

            case InvariantType.TOOL_ALWAYS_CALLED:
                if invariant.step_index is None or invariant.step_index >= len(trace.steps):
                    return True
                step = trace.steps[invariant.step_index]
                return any(tc.tool_name == invariant.tool_name for tc in step.tool_calls)

            case InvariantType.TOOL_NEVER_CALLED:
                if invariant.step_index is None or invariant.step_index >= len(trace.steps):
                    return True
                step = trace.steps[invariant.step_index]
                return not any(tc.tool_name == invariant.tool_name for tc in step.tool_calls)

            case InvariantType.TOOL_CALL_COUNT:
                total = sum(
                    1 for s in trace.steps for tc in s.tool_calls
                    if tc.tool_name == invariant.tool_name
                )
                d = invariant.details
                return d["min"] <= total <= d["max"]

            case InvariantType.ARG_RANGE:
                d = invariant.details
                for s in trace.steps:
                    for tc in s.tool_calls:
                        if tc.tool_name == invariant.tool_name:
                            val = tc.arguments.get(d["arg_name"])
                            if val is not None and isinstance(val, (int, float)):
                                if not (d["min"] <= val <= d["max"]):
                                    return False
                return True

            case InvariantType.ARG_PATTERN:
                d = invariant.details
                for s in trace.steps:
                    for tc in s.tool_calls:
                        if tc.tool_name == invariant.tool_name:
                            val = str(tc.arguments.get(d["arg_name"], ""))
                            if val and not re.search(d["pattern"], val):
                                return False
                return True

            case InvariantType.RESPONSE_LENGTH:
                d = invariant.details
                if invariant.step_index is None or invariant.step_index >= len(trace.steps):
                    return True
                length = len(trace.steps[invariant.step_index].assistant_response)
                return d["min"] <= length <= d["max"]

            case InvariantType.STEP_LATENCY:
                d = invariant.details
                if invariant.step_index is None or invariant.step_index >= len(trace.steps):
                    return True
                return trace.steps[invariant.step_index].latency_ms <= d["max_ms"]

            case _:
                return True

    def _first_tool_call_step(self, trace, tool_name):
        for i, step in enumerate(trace.steps):
            if any(tc.tool_name == tool_name for tc in step.tool_calls):
                return i
        return None

    def _generate_expectations(self, invariants: list[Invariant]) -> list[dict]:
        expectations = []
        for inv in invariants:
            match inv.invariant_type:
                case InvariantType.TOOL_ALWAYS_CALLED:
                    expectations.append({
                        "step": inv.step_index,
                        "expectation": {"type": "tool_called", "tool": inv.tool_name},
                        "source": f"Mined from {inv.support} passing traces (confidence: {inv.confidence:.0%})",
                        "discriminating": inv.violations > 0,
                    })
                case InvariantType.ARG_RANGE:
                    d = inv.details
                    expectations.append({
                        "step": inv.step_index,
                        "expectation": {
                            "type": "tool_args_contain",
                            "tool": inv.tool_name,
                            "note": f"Arg '{d['arg_name']}' should be in range [{d['min']}, {d['max']}]",
                        },
                        "source": f"Mined from {inv.support} traces",
                        "discriminating": inv.violations > 0,
                    })
                case InvariantType.STEP_LATENCY:
                    d = inv.details
                    expectations.append({
                        "step": inv.step_index,
                        "expectation": {"type": "latency_under", "max_ms": int(d["max_ms"])},
                        "source": f"95th percentile from {inv.support} traces",
                        "discriminating": inv.violations > 0,
                    })
        return expectations
