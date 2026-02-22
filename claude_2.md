)

The following features are built and working against Qwen 2.5 7B via Ollama on M3 Max:

- YAML scenario DSL with tool definitions and mock responses
- Execution harness with Ollama tool-call loop (multi-turn)
- 10 expectation types (tool_called, tool_not_called, tool_args_contain, response_contains, response_not_contains, response_matches_regex, llm_judge, latency_under, no_tool_errors, tool_call_count)
- Content-addressed Trace IR with SHA-256 hashing
- Trace store (Python implementation with gzip compression + SQLite index)
- Virtual replay (re-evaluate stored traces without calling Ollama)
- Trace diffing (compare two traces structurally)
- Schema-aware differential fuzzing (6 mutation operators)
- Delta debugging minimal reproduction (ddmin over agent# CLAUDE.md — TraceForge v0.2: Causal Attribution, Invariant Mining & Rust Trace Core

> **This is the Phase 2 specification for TraceForge.**
> Phase 1 (core harness, evaluator, trace store, replay, fuzzer, minrepro, reporter) is ALREADY BUILT AND WORKING.
> This spec adds three new capabilities: (1) Causal Attribution Engine, (2) Invariant Mining, (3) Rust Trace Core.
> Do NOT rewrite or break existing working code. Extend it.

---

## Project Identity

- **Name:** TraceForge
- **PyPI name:** `traceforge`
- **Version:** 0.2.0 (upgrading from 0.1.0)
- **Tagline:** "Deterministic replay, causal debugging & behavioral invariant discovery for local AI agents"
- **License:** MIT
- **Author:** Abhimanyu

---

## What Already Works (Phase 1 — DO NOT REWRITE steps)
- Rich CLI reporter with color-coded scorecards
- HTML report generation (dark theme, expandable per-run details)
- SQLite history with regression detection
- CLI commands: `traceforge run`, `traceforge replay`, `traceforge fuzz`, `traceforge minrepro`, `traceforge init`, `traceforge history`, `traceforge traces`

Confirmed working scenarios: calculator_agent (100%), weather_agent (100%), simple_chat (100%), multi_step_math (0% — intentional, designed to test minrepro).

---

## Phase 2 Overview — What We're Adding

### New Capability 1: Causal Attribution Engine
**WHY did the agent fail?** When a step fails, automatically run counterfactual experiments — change one variable at a time and observe which change flips the outcome. Produces a causal attribution report: "Failure is 73% sensitive to tool output format, 27% to context truncation."

### New Capability 2: Invariant Mining (Daikon for Agents)
**What rules does the agent follow?** Automatically discover behavioral invariants from passing traces that break in failing traces. Like: "In all passing traces, the agent calls get_weather before get_forecast" or "calculate is never called with negative arguments when user says 'total'."

### New Capability 3: Rust Trace Core (Performance Layer)
Content-addressed store with mmap-backed segment files, streaming Ollama ingest, and PyO3 bindings. Drop-in replacement for the Python trace store. Enables 10,000+ trace corpora on a laptop.

---

## New Module 1: Causal Attribution Engine

### Concept

When a test fails, ddmin tells you *which* step fails. Causal attribution tells you *WHY*.

The engine works by **counterfactual replay**: take the failing trace, systematically modify one variable at a time (a "do-operation"), re-run the agent, and observe whether the failure persists or resolves. Each variable that flips the outcome is a causal factor.

This is Pearl's do-calculus applied to agent debugging. Nobody has built this for agent testing.

### File: `src/traceforge/attribution.py`

### Counterfactual Variables (the "do-operations")

The engine tests these categories of interventions:

```python
from enum import Enum
from pydantic import BaseModel
from typing import Optional, Any

class InterventionType(str, Enum):
    """Categories of counterfactual interventions."""
    TOOL_OUTPUT_FORMAT = "tool_output_format"      # Change format: string↔number, nested↔flat
    TOOL_OUTPUT_VALUE = "tool_output_value"         # Change specific field values
    TOOL_OUTPUT_FIELDS = "tool_output_fields"       # Add/remove fields from tool response
    CONTEXT_TRUNCATION = "context_truncation"       # Truncate conversation history at various points
    SYSTEM_PROMPT_CLAUSE = "system_prompt_clause"   # Remove individual sentences from system prompt
    TOOL_SCHEMA_CHANGE = "tool_schema_change"       # Rename fields, change types in tool schema
    MESSAGE_ORDER = "message_order"                 # Reorder previous conversation turns
    TOOL_RESPONSE_DELAY = "tool_response_delay"     # Simulate slow tool responses (if latency matters)

class Intervention(BaseModel):
    """A single counterfactual change applied to a scenario."""
    intervention_type: InterventionType
    description: str                                # Human-readable: "Changed 'temperature' from int to string"
    target_step: Optional[int] = None               # Which step is modified (None = global change)
    target_tool: Optional[str] = None               # Which tool's response is modified
    target_field: Optional[str] = None              # Which field in the tool response
    original_value: Optional[Any] = None            # What it was
    modified_value: Optional[Any] = None            # What we changed it to
    
class CounterfactualResult(BaseModel):
    """Result of one counterfactual experiment."""
    intervention: Intervention
    original_passed: bool                           # Did original scenario pass?
    counterfactual_passed: bool                     # Did modified scenario pass?
    flipped: bool                                   # Did the outcome change?
    trace_id: str                                   # Trace from the counterfactual run
    confidence: float                               # How confident (based on repeated runs)

class CausalReport(BaseModel):
    """Full causal attribution report for a failing scenario."""
    scenario_name: str
    failing_trace_id: str
    failing_step: int
    total_interventions: int
    total_flips: int                                # How many interventions changed the outcome
    interventions: list[CounterfactualResult]
    causal_factors: list[dict]                      # Ranked: [{factor, sensitivity, description}]
    summary: str                                    # LLM-generated narrative explanation
```

### Intervention Generators

Each intervention type has a generator that produces concrete modifications:

```python
class InterventionGenerator:
    """Generates counterfactual interventions for a failing trace."""

    def generate_all(self, scenario: Scenario, trace: TraceIR, 
                     failing_step: int) -> list[Intervention]:
        """Generate all counterfactual interventions for the failing step."""
        interventions = []
        interventions.extend(self._tool_output_format_interventions(trace, failing_step))
        interventions.extend(self._tool_output_value_interventions(trace, failing_step))
        interventions.extend(self._tool_output_field_interventions(trace, failing_step))
        interventions.extend(self._context_truncation_interventions(trace, failing_step))
        interventions.extend(self._system_prompt_clause_interventions(scenario))
        interventions.extend(self._tool_schema_interventions(scenario, failing_step))
        return interventions

    def _tool_output_format_interventions(self, trace, step_idx):
        """
        For each tool call in the failing step, generate format changes:
        - If a value is an int, make it a string (and vice versa)
        - If a value is nested JSON, flatten it
        - If a value is a list, make it a single item
        """
        interventions = []
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
                        modified_value=str(value)
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
                            modified_value=num
                        ))
                    except ValueError:
                        pass
        return interventions

    def _tool_output_value_interventions(self, trace, step_idx):
        """
        For each tool output field, generate value changes:
        - Numeric: 0, negative, double, half
        - String: empty, reversed, different plausible value
        - Boolean: flip
        """
        interventions = []
        step = trace.steps[step_idx]
        for tc in step.tool_calls:
            for key, value in tc.response.items():
                if isinstance(value, (int, float)) and value != 0:
                    for new_val, desc in [
                        (0, "zero"), (-value, "negated"), (value * 2, "doubled"), (value / 2, "halved")
                    ]:
                        interventions.append(Intervention(
                            intervention_type=InterventionType.TOOL_OUTPUT_VALUE,
                            description=f"Changed '{key}' to {desc} ({new_val})",
                            target_step=step_idx,
                            target_tool=tc.tool_name,
                            target_field=key,
                            original_value=value,
                            modified_value=new_val
                        ))
                elif isinstance(value, bool):
                    interventions.append(Intervention(
                        intervention_type=InterventionType.TOOL_OUTPUT_VALUE,
                        description=f"Flipped '{key}' from {value} to {not value}",
                        target_step=step_idx,
                        target_tool=tc.tool_name,
                        target_field=key,
                        original_value=value,
                        modified_value=not value
                    ))
        return interventions

    def _tool_output_field_interventions(self, trace, step_idx):
        """
        Add or remove fields from tool responses:
        - Remove each field one at a time
        - Add an unexpected field
        """
        interventions = []
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
                    modified_value="__REMOVE__"
                ))
            # Add unexpected field
            interventions.append(Intervention(
                intervention_type=InterventionType.TOOL_OUTPUT_FIELDS,
                description=f"Added unexpected field 'debug_info' to {tc.tool_name} response",
                target_step=step_idx,
                target_tool=tc.tool_name,
                target_field="debug_info",
                original_value=None,
                modified_value="internal debug data - ignore this"
            ))
        return interventions

    def _context_truncation_interventions(self, trace, step_idx):
        """
        Test context sensitivity by truncating history:
        - Keep only the last N steps of context
        - Remove all previous tool call results
        - Remove all previous assistant responses
        """
        interventions = []
        if step_idx > 0:
            for keep_last in [1, 2, max(1, step_idx // 2)]:
                if keep_last < step_idx:
                    interventions.append(Intervention(
                        intervention_type=InterventionType.CONTEXT_TRUNCATION,
                        description=f"Truncated context to last {keep_last} steps (from {step_idx})",
                        target_step=step_idx,
                        original_value=step_idx,
                        modified_value=keep_last
                    ))
        return interventions

    def _system_prompt_clause_interventions(self, scenario):
        """
        Remove individual sentences from the system prompt.
        Each sentence removal is a separate intervention.
        """
        interventions = []
        prompt = scenario.agent.system_prompt or ""
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        for i, sentence in enumerate(sentences):
            reduced = '. '.join(s for j, s in enumerate(sentences) if j != i) + '.'
            interventions.append(Intervention(
                intervention_type=InterventionType.SYSTEM_PROMPT_CLAUSE,
                description=f"Removed prompt sentence {i}: '{sentence[:50]}...'",
                original_value=prompt,
                modified_value=reduced
            ))
        return interventions

    def _tool_schema_interventions(self, scenario, step_idx):
        """
        Modify tool schemas:
        - Rename a parameter
        - Change a parameter type
        - Remove a required parameter from schema
        """
        interventions = []
        for tool in scenario.agent.tools:
            props = tool.parameters.get("properties", {})
            for param_name, param_schema in props.items():
                # Rename parameter
                interventions.append(Intervention(
                    intervention_type=InterventionType.TOOL_SCHEMA_CHANGE,
                    description=f"Renamed '{param_name}' to '{param_name}_v2' in {tool.name} schema",
                    target_tool=tool.name,
                    target_field=param_name,
                    original_value=param_name,
                    modified_value=f"{param_name}_v2"
                ))
        return interventions
```

### Causal Attribution Engine

```python
class CausalAttributionEngine:
    """
    Runs counterfactual experiments to determine WHY an agent fails.
    
    Algorithm:
    1. Take a failing trace and identify the failing step
    2. Generate all possible interventions
    3. For each intervention, create a modified scenario
    4. Run the modified scenario against Ollama (multiple times for confidence)
    5. Record which interventions flip the outcome (fail → pass)
    6. Rank interventions by causal sensitivity
    7. Generate a narrative explanation
    """

    def __init__(self, harness, evaluator, trace_store, judge=None):
        self.harness = harness
        self.evaluator = evaluator
        self.store = trace_store
        self.judge = judge
        self.generator = InterventionGenerator()

    def attribute(self, trace_id: str, scenario: Scenario,
                  confirmation_runs: int = 3,
                  max_interventions: int = 50) -> CausalReport:
        """
        Run causal attribution on a failing trace.
        
        Args:
            trace_id: ID of the failing trace
            scenario: The scenario that produced the failing trace
            confirmation_runs: Times to re-run each counterfactual for confidence
            max_interventions: Cap on total interventions to test (for speed)
        """
        trace = self.store.load(trace_id)
        
        # Find the first failing step
        baseline_result = self.evaluator.evaluate(trace, scenario)
        failing_step = self._find_first_failing_step(baseline_result)
        if failing_step is None:
            raise ValueError(f"Trace {trace_id} does not appear to fail any step")

        # Generate interventions
        all_interventions = self.generator.generate_all(scenario, trace, failing_step)
        # Cap for speed
        interventions = all_interventions[:max_interventions]

        # Run counterfactual experiments
        results = []
        for intervention in interventions:
            cf_result = self._run_counterfactual(
                scenario, trace, intervention, 
                failing_step, confirmation_runs
            )
            results.append(cf_result)

        # Rank by causal sensitivity
        flips = [r for r in results if r.flipped]
        causal_factors = self._rank_factors(results)

        # Generate narrative summary (using LLM if available)
        summary = self._generate_summary(scenario, trace, failing_step, causal_factors)

        return CausalReport(
            scenario_name=scenario.name,
            failing_trace_id=trace_id,
            failing_step=failing_step,
            total_interventions=len(results),
            total_flips=len(flips),
            interventions=results,
            causal_factors=causal_factors,
            summary=summary
        )

    def _run_counterfactual(self, scenario, trace, intervention, 
                            failing_step, runs) -> CounterfactualResult:
        """Run one counterfactual experiment with confirmation runs."""
        modified_scenario = self._apply_intervention(scenario, trace, intervention)
        
        pass_count = 0
        last_trace_id = ""
        for _ in range(runs):
            traces = self.harness.run_scenario(modified_scenario, runs=1)
            if traces:
                result = self.evaluator.evaluate(traces[0], modified_scenario)
                if result.passed:
                    pass_count += 1
                last_trace_id = traces[0].trace_id

        cf_passed = pass_count > runs / 2  # Majority vote
        confidence = max(pass_count, runs - pass_count) / runs

        return CounterfactualResult(
            intervention=intervention,
            original_passed=False,  # We only attribute failing traces
            counterfactual_passed=cf_passed,
            flipped=cf_passed,      # Original failed, counterfactual passed = flip
            trace_id=last_trace_id,
            confidence=confidence
        )

    def _apply_intervention(self, scenario, trace, intervention):
        """
        Create a modified scenario by applying the intervention.
        Returns a new Scenario object with the modification.
        """
        modified = scenario.model_copy(deep=True)

        match intervention.intervention_type:
            case InterventionType.TOOL_OUTPUT_FORMAT | InterventionType.TOOL_OUTPUT_VALUE:
                # Modify mock responses for the target tool
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
                # Keep only the last N steps
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
                            props[intervention.modified_value] = props.pop(intervention.target_field)
                            required = tool.parameters.get("required", [])
                            if intervention.target_field in required:
                                idx = required.index(intervention.target_field)
                                required[idx] = intervention.modified_value

        modified.runs = 1  # Single run for counterfactuals
        return modified

    def _rank_factors(self, results: list[CounterfactualResult]) -> list[dict]:
        """
        Rank causal factors by sensitivity (what percentage of interventions 
        of each type flipped the outcome).
        """
        from collections import defaultdict
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
                    "description": f"{counts['flips']}/{counts['total']} interventions of type '{factor_type}' flipped the outcome"
                })

        factors.sort(key=lambda f: f["sensitivity"], reverse=True)
        return factors

    def _generate_summary(self, scenario, trace, failing_step, factors) -> str:
        """Generate a human-readable causal narrative."""
        if not factors:
            return "No causal factors identified. The failure may be intrinsic to the model's capabilities."

        top = factors[0]
        lines = [
            f"Causal analysis of '{scenario.name}' failure at step {failing_step}:",
            f"",
            f"Primary cause: {top['factor']} (sensitivity: {top['sensitivity']:.0%})",
            f"  {top['description']}",
        ]
        if len(factors) > 1:
            lines.append(f"")
            lines.append(f"Secondary factors:")
            for f in factors[1:3]:
                lines.append(f"  - {f['factor']}: {f['sensitivity']:.0%} sensitivity")

        if self.judge:
            # Use LLM to generate a richer narrative
            prompt = (
                f"You are analyzing why an AI agent failed a test. "
                f"The test scenario '{scenario.name}' failed at step {failing_step}. "
                f"Counterfactual analysis shows the top causal factors:\n"
            )
            for f in factors[:3]:
                prompt += f"- {f['factor']}: {f['sensitivity']:.0%} of interventions flipped the outcome\n"
            prompt += f"\nWrite a 2-3 sentence explanation of the likely root cause, aimed at a developer trying to fix the agent."

            try:
                import ollama
                resp = ollama.chat(
                    model=self.judge.model if hasattr(self, 'judge') and self.judge else "qwen2.5:7b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0}
                )
                lines.append("")
                lines.append("LLM Analysis:")
                lines.append(resp["message"]["content"])
            except Exception:
                pass

        return "\n".join(lines)

    def _find_first_failing_step(self, run_result) -> Optional[int]:
        """Find the index of the first step that has any failing expectation."""
        for sr in run_result.step_results:
            if not sr.all_passed:
                return sr.step_index
        return None
```

### CLI Command: `traceforge attribute`

```bash
# Attribute a specific failing trace
traceforge attribute <trace-id>

# Attribute with more confirmation runs (higher confidence, slower)
traceforge attribute <trace-id> --confirmation-runs 5

# Attribute with max interventions cap
traceforge attribute <trace-id> --max-interventions 30

# Attribute all failing traces in history
traceforge attribute --all-failures

# Output as JSON
traceforge attribute <trace-id> --json
```

### CLI Output Format

```
╭─────────────── Causal Attribution Report ────────────────╮
│ Scenario: multi_step_math                                │
│ Failing step: 2 | Trace: a7f3b2c1...                    │
│ Interventions tested: 23 | Flips found: 7                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  CAUSAL FACTOR            SENSITIVITY  FLIPS  TESTED    │
│  ──────────────────────────────────────────────────────  │
│  🔴 tool_output_format      73%        3/4    ████████  │
│  🟡 tool_output_value       40%        2/5    █████     │
│  🟢 system_prompt_clause    14%        1/7    ██        │
│  ⚪ context_truncation       0%        0/4              │
│  ⚪ tool_schema_change       0%        0/3              │
│                                                          │
│  Root cause: The agent fails when the calculator tool    │
│  returns results as strings instead of numbers. The      │
│  agent cannot parse "42" (string) and expects 42 (int).  │
╰──────────────────────────────────────────────────────────╯
```

---

## New Module 2: Invariant Mining (Daikon for Agents)

### Concept

Instead of manually writing expectations, **automatically discover behavioral invariants** from execution traces. Run 100+ traces, mine patterns that hold in ALL passing traces, and surface violations in failing traces.

This is the agent equivalent of Daikon (Ernst et al., 2001) — but instead of mining variable invariants from program traces, we mine behavioral invariants from agent tool-call traces.

**Nobody has built this for agent tool-call traces.**

### File: `src/traceforge/invariants.py`

### Invariant Types

```python
from enum import Enum
from pydantic import BaseModel
from typing import Optional, Any

class InvariantType(str, Enum):
    """Categories of discoverable behavioral invariants."""
    TOOL_ORDER = "tool_order"                    # Tool A always called before Tool B
    TOOL_ALWAYS_CALLED = "tool_always_called"    # Tool X is always called in step N
    TOOL_NEVER_CALLED = "tool_never_called"      # Tool X is never called in step N
    TOOL_CALL_COUNT = "tool_call_count"           # Tool X is called exactly N times
    ARG_RANGE = "arg_range"                       # Argument X is always in range [a, b]
    ARG_PATTERN = "arg_pattern"                   # Argument X always matches pattern P
    ARG_DEPENDENCY = "arg_dependency"             # Argument contains value from previous tool output
    RESPONSE_LENGTH = "response_length"           # Response length is always in range [a, b]
    RESPONSE_PATTERN = "response_pattern"         # Response always contains/excludes pattern
    STEP_LATENCY = "step_latency"                 # Step latency is always under N ms
    TOOL_IDEMPOTENCY = "tool_idempotency"         # Same tool args always produce same agent behavior
    CONDITIONAL = "conditional"                   # If condition C, then invariant I holds

class Invariant(BaseModel):
    """A discovered behavioral invariant."""
    invariant_type: InvariantType
    description: str                              # Human-readable
    formal: str                                   # Formal representation for automated checking
    confidence: float                             # What fraction of traces support this (0.0-1.0)
    support: int                                  # Number of traces that support this
    violations: int                               # Number of traces that violate this
    step_index: Optional[int] = None              # Which step this applies to (None = global)
    tool_name: Optional[str] = None
    details: dict = {}                            # Extra info for specific invariant types

class InvariantReport(BaseModel):
    """Report of all mined invariants."""
    total_traces_analyzed: int
    passing_traces: int
    failing_traces: int
    invariants_discovered: int
    invariants: list[Invariant]
    discriminating_invariants: list[Invariant]    # Hold in passing, break in failing
    suggested_expectations: list[dict]            # Auto-generated YAML expectations
```

### Invariant Mining Engine

```python
class InvariantMiner:
    """
    Mines behavioral invariants from a corpus of agent execution traces.
    
    Algorithm:
    1. Collect all traces for a scenario (both passing and failing)
    2. For passing traces, extract candidate invariants using pattern miners
    3. Validate each candidate against ALL passing traces
    4. Check which invariants are violated in failing traces
    5. Rank by discriminating power (holds in passing, breaks in failing)
    6. Generate suggested YAML expectations from top invariants
    """

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

    def mine(self, scenario_name: str, 
             min_confidence: float = 0.95) -> InvariantReport:
        """
        Mine invariants from all stored traces for a scenario.
        
        Args:
            scenario_name: Which scenario to analyze
            min_confidence: Minimum fraction of passing traces that must support an invariant
        """
        # Load all traces
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
                1 for t in passing_traces 
                if self._check_invariant(candidate, t)
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
                1 for t in failing_traces 
                if not self._check_invariant(inv, t)
            )
            inv.violations = violations
            if violations > 0 and failing_traces:
                discriminating.append(inv)

        # Sort by discriminating power
        discriminating.sort(
            key=lambda i: i.violations / max(len(failing_traces), 1),
            reverse=True
        )

        # Generate suggested expectations
        suggestions = self._generate_expectations(discriminating[:10])

        return InvariantReport(
            total_traces_analyzed=len(passing_traces) + len(failing_traces),
            passing_traces=len(passing_traces),
            failing_traces=len(failing_traces),
            invariants_discovered=len(validated),
            invariants=validated,
            discriminating_invariants=discriminating,
            suggested_expectations=suggestions
        )

    def _check_invariant(self, invariant: Invariant, trace: TraceIR) -> bool:
        """Check if a trace satisfies an invariant."""
        match invariant.invariant_type:
            case InvariantType.TOOL_ORDER:
                # Check tool A called before tool B
                d = invariant.details
                a_idx = self._first_tool_call_step(trace, d["tool_a"])
                b_idx = self._first_tool_call_step(trace, d["tool_b"])
                if a_idx is None or b_idx is None:
                    return True  # Vacuously true if tools not present
                return a_idx <= b_idx

            case InvariantType.TOOL_ALWAYS_CALLED:
                step = trace.steps[invariant.step_index] if invariant.step_index < len(trace.steps) else None
                if step is None:
                    return True
                return any(tc.tool_name == invariant.tool_name for tc in step.tool_calls)

            case InvariantType.TOOL_NEVER_CALLED:
                step = trace.steps[invariant.step_index] if invariant.step_index < len(trace.steps) else None
                if step is None:
                    return True
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
                import re
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
                step = trace.steps[invariant.step_index] if invariant.step_index is not None and invariant.step_index < len(trace.steps) else None
                if step is None:
                    return True
                length = len(step.assistant_response)
                return d["min"] <= length <= d["max"]

            case InvariantType.STEP_LATENCY:
                d = invariant.details
                step = trace.steps[invariant.step_index] if invariant.step_index is not None and invariant.step_index < len(trace.steps) else None
                if step is None:
                    return True
                return step.latency_ms <= d["max_ms"]

            case _:
                return True  # Unknown invariant type, assume holds

    def _first_tool_call_step(self, trace, tool_name):
        for i, step in enumerate(trace.steps):
            if any(tc.tool_name == tool_name for tc in step.tool_calls):
                return i
        return None

    def _generate_expectations(self, invariants: list[Invariant]) -> list[dict]:
        """Convert discovered invariants into YAML-compatible expectation dicts."""
        expectations = []
        for inv in invariants:
            match inv.invariant_type:
                case InvariantType.TOOL_ALWAYS_CALLED:
                    expectations.append({
                        "step": inv.step_index,
                        "expectation": {
                            "type": "tool_called",
                            "tool": inv.tool_name
                        },
                        "source": f"Mined from {inv.support} passing traces (confidence: {inv.confidence:.0%})",
                        "discriminating": inv.violations > 0
                    })
                case InvariantType.ARG_RANGE:
                    d = inv.details
                    expectations.append({
                        "step": inv.step_index,
                        "expectation": {
                            "type": "tool_args_contain",
                            "tool": inv.tool_name,
                            "note": f"Arg '{d['arg_name']}' should be in range [{d['min']}, {d['max']}]"
                        },
                        "source": f"Mined from {inv.support} traces",
                        "discriminating": inv.violations > 0
                    })
                case InvariantType.STEP_LATENCY:
                    d = inv.details
                    expectations.append({
                        "step": inv.step_index,
                        "expectation": {
                            "type": "latency_under",
                            "max_ms": int(d["max_ms"])
                        },
                        "source": f"95th percentile from {inv.support} traces",
                        "discriminating": inv.violations > 0
                    })
        return expectations
```

### Pattern Miners (individual invariant extractors)

```python
class ToolOrderMiner:
    """Discovers tool call ordering invariants."""
    
    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        """Find all tool pairs where A is always called before B."""
        from itertools import combinations
        
        # Collect all tool names across all traces
        all_tools = set()
        for t in traces:
            for s in t.steps:
                for tc in s.tool_calls:
                    all_tools.add(tc.tool_name)
        
        candidates = []
        for a, b in combinations(all_tools, 2):
            # Check if A always before B in all traces where both appear
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
                    details={"tool_a": a, "tool_b": b}
                ))
            if count >= 2 and b_before_a:
                candidates.append(Invariant(
                    invariant_type=InvariantType.TOOL_ORDER,
                    description=f"'{b}' is always called before '{a}'",
                    formal=f"order({b}) < order({a})",
                    confidence=1.0,
                    support=count,
                    violations=0,
                    details={"tool_a": b, "tool_b": a}
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
            for tool_name in all_tools:
                always_called = all(
                    any(tc.tool_name == tool_name for tc in t.steps[step_idx].tool_calls)
                    for t in traces if step_idx < len(t.steps)
                )
                never_called = all(
                    not any(tc.tool_name == tool_name for tc in t.steps[step_idx].tool_calls)
                    for t in traces if step_idx < len(t.steps)
                )
                applicable = sum(1 for t in traces if step_idx < len(t.steps))
                
                if applicable >= 2 and always_called:
                    candidates.append(Invariant(
                        invariant_type=InvariantType.TOOL_ALWAYS_CALLED,
                        description=f"'{tool_name}' is always called at step {step_idx}",
                        formal=f"always_called({tool_name}, step={step_idx})",
                        confidence=1.0,
                        support=applicable,
                        violations=0,
                        step_index=step_idx,
                        tool_name=tool_name
                    ))
                if applicable >= 2 and never_called:
                    candidates.append(Invariant(
                        invariant_type=InvariantType.TOOL_NEVER_CALLED,
                        description=f"'{tool_name}' is never called at step {step_idx}",
                        formal=f"never_called({tool_name}, step={step_idx})",
                        confidence=1.0,
                        support=applicable,
                        violations=0,
                        step_index=step_idx,
                        tool_name=tool_name
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
        
        for tool_name in all_tools:
            counts = []
            for t in traces:
                count = sum(1 for s in t.steps for tc in s.tool_calls if tc.tool_name == tool_name)
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
                    details={"min": min_c, "max": max_c}
                ))
        return candidates


class ArgRangeMiner:
    """Discovers numeric argument range invariants."""
    
    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        candidates = []
        # Collect all (tool, arg_name) → [values] across traces
        from collections import defaultdict
        arg_values = defaultdict(list)
        
        for t in traces:
            for s in t.steps:
                for tc in s.tool_calls:
                    for arg_name, arg_val in tc.arguments.items():
                        if isinstance(arg_val, (int, float)):
                            arg_values[(tc.tool_name, arg_name)].append(arg_val)
        
        for (tool_name, arg_name), values in arg_values.items():
            if len(values) >= 3:
                # Use observed range with small margin
                min_v = min(values)
                max_v = max(values)
                margin = (max_v - min_v) * 0.1 if max_v != min_v else abs(min_v) * 0.1 + 1
                candidates.append(Invariant(
                    invariant_type=InvariantType.ARG_RANGE,
                    description=f"'{tool_name}.{arg_name}' is always in range [{min_v - margin:.2f}, {max_v + margin:.2f}]",
                    formal=f"{min_v - margin:.2f} <= {tool_name}.{arg_name} <= {max_v + margin:.2f}",
                    confidence=1.0,
                    support=len(values),
                    violations=0,
                    tool_name=tool_name,
                    details={"arg_name": arg_name, "min": min_v - margin, "max": max_v + margin}
                ))
        return candidates


class ArgPatternMiner:
    """Discovers string argument pattern invariants."""
    
    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        candidates = []
        from collections import defaultdict
        arg_values = defaultdict(list)
        
        for t in traces:
            for s in t.steps:
                for tc in s.tool_calls:
                    for arg_name, arg_val in tc.arguments.items():
                        if isinstance(arg_val, str) and len(arg_val) > 0:
                            arg_values[(tc.tool_name, arg_name)].append(arg_val)
        
        for (tool_name, arg_name), values in arg_values.items():
            if len(values) >= 3:
                # Check if all values are non-empty
                if all(v.strip() for v in values):
                    candidates.append(Invariant(
                        invariant_type=InvariantType.ARG_PATTERN,
                        description=f"'{tool_name}.{arg_name}' is always non-empty",
                        formal=f"len({tool_name}.{arg_name}) > 0",
                        confidence=1.0,
                        support=len(values),
                        violations=0,
                        tool_name=tool_name,
                        details={"arg_name": arg_name, "pattern": r".+"}
                    ))
        return candidates


class ArgDependencyMiner:
    """Discovers if tool arguments reference previous tool outputs."""
    
    def extract_candidates(self, traces: list[TraceIR]) -> list[Invariant]:
        # TODO: Implement cross-step argument dependency detection
        # Check if argument values in step N appear in tool outputs from step N-1
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
                    details={"min": max(0, min_l - margin), "max": max_l + margin}
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
                # Use 95th percentile as upper bound
                sorted_lat = sorted(latencies)
                p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
                candidates.append(Invariant(
                    invariant_type=InvariantType.STEP_LATENCY,
                    description=f"Step {step_idx} latency is under {p95:.0f}ms (p95)",
                    formal=f"latency[{step_idx}] <= {p95:.0f}ms",
                    confidence=0.95,
                    support=len(latencies),
                    violations=0,
                    step_index=step_idx,
                    details={"max_ms": p95}
                ))
        return candidates
```

### CLI Command: `traceforge mine`

```bash
# Mine invariants for a specific scenario
traceforge mine calculator_agent

# Mine with lower confidence threshold (discover more invariants)
traceforge mine calculator_agent --min-confidence 0.9

# Mine and output suggested YAML expectations
traceforge mine calculator_agent --suggest-expectations

# Mine all scenarios
traceforge mine --all

# Output as JSON
traceforge mine calculator_agent --json
```

### CLI Output Format

```
╭──────────────── Invariant Mining Report ─────────────────╮
│ Scenario: calculator_agent                               │
│ Traces analyzed: 30 (27 passing, 3 failing)              │
│ Invariants discovered: 12                                │
│ Discriminating invariants: 4                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  🔍 DISCRIMINATING INVARIANTS (break in failing traces)  │
│  ──────────────────────────────────────────────────────  │
│  1. 'calculate' is always called at step 0               │
│     Confidence: 100% | Violations: 3/3 failing traces    │
│                                                          │
│  2. 'calculate.expression' is always non-empty           │
│     Confidence: 100% | Violations: 2/3 failing traces    │
│                                                          │
│  3. Step 0 response always contains digits               │
│     Confidence: 96% | Violations: 2/3 failing traces     │
│                                                          │
│  4. Step 0 latency under 2500ms (p95)                    │
│     Confidence: 95% | Violations: 1/3 failing traces     │
│                                                          │
│  📝 SUGGESTED EXPECTATIONS (add to YAML):                │
│  ──────────────────────────────────────────────────────  │
│  - type: tool_called                                     │
│    tool: calculate     # Mined from 27 passing traces    │
│  - type: latency_under                                   │
│    max_ms: 2500        # 95th percentile from 27 traces  │
│                                                          │
│  💡 ALL INVARIANTS (12 total):                           │
│     [use -v for details]                                 │
╰──────────────────────────────────────────────────────────╯
```

---

## New Module 3: Rust Trace Core (Performance Layer)

### Overview

The Rust Trace Core is a drop-in replacement for the Python trace store and streaming ingest. It provides:
1. Content-addressed store with mmap-backed segment files (memmap2)
2. BLAKE3 hashing (faster than SHA-256 on Apple Silicon)
3. zstd compression with dictionary training
4. Streaming Ollama ingest via reqwest
5. PyO3 bindings exposed as `traceforge._rust`

### Workspace Layout

```
traceforge/
├── rust/                              # Rust workspace
│   ├── Cargo.toml                     # Workspace root
│   ├── crates/
│   │   ├── core/                      # Pure Rust: CAS store, Trace IR, hashing
│   │   │   ├── Cargo.toml
│   │   │   └── src/
│   │   │       ├── lib.rs
│   │   │       ├── store.rs           # Append-only segment CAS store
│   │   │       ├── trace_ir.rs        # Canonical serialization + hashing
│   │   │       ├── compress.rs        # zstd compression with dict support
│   │   │       └── ollama.rs          # Streaming /api/chat client
│   │   └── python/                    # PyO3 bindings (thin wrapper)
│   │       ├── Cargo.toml
│   │       └── src/
│   │           └── lib.rs             # #[pymodule] traceforge._rust
│   └── .cargo/
│       └── config.toml                # Apple Silicon optimizations
├── python/
│   └── traceforge/
│       ├── __init__.py                # try: from ._rust import ... except: ...
│       └── ...                        # All existing Python modules
└── pyproject.toml                     # maturin build backend
```

### Python Integration Pattern

```python
# src/traceforge/__init__.py (modified)
try:
    from traceforge._rust import (
        RustTraceStore,
        rust_canonical_hash,
        rust_compress,
        rust_decompress,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

def get_trace_store(base_dir=".traceforge"):
    """Factory: returns Rust store if available, Python fallback otherwise."""
    if HAS_RUST:
        return RustTraceStore(base_dir)
    from traceforge.trace_store import TraceStore
    return TraceStore(base_dir)
```

### Rust Crate Dependencies

```toml
# rust/crates/core/Cargo.toml
[package]
name = "traceforge-core"
version = "0.1.0"
edition = "2021"

[dependencies]
memmap2 = "0.9"
blake3 = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
zstd = "0.13"
reqwest = { version = "0.12", features = ["json", "stream"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros", "io-util"] }
futures-util = "0.3"
tokio-util = { version = "0.7", features = ["io"] }

# rust/crates/python/Cargo.toml
[package]
name = "traceforge-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["abi3-py312"] }
traceforge-core = { path = "../core" }
```

### Build Commands

```bash
# Development build
cd traceforge/rust
maturin develop --release

# Build wheel for distribution
maturin build --release

# Run Rust tests
cargo test --workspace
```

### pyproject.toml (updated for maturin)

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "python"
module-name = "traceforge._rust"
manifest-path = "rust/crates/python/Cargo.toml"
strip = true
```

### Apple M3 Max Config

```toml
# rust/.cargo/config.toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=apple-m1"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

---

## Updated CLI Commands (Phase 2)

All existing commands continue to work unchanged. New commands:

```bash
# === Causal Attribution ===
traceforge attribute <trace-id>                    # Why did this trace fail?
traceforge attribute <trace-id> --confirmation-runs 5
traceforge attribute <trace-id> --max-interventions 30
traceforge attribute --all-failures                # Attribute all failing traces
traceforge attribute <trace-id> --json

# === Invariant Mining ===
traceforge mine <scenario-name>                    # Mine invariants for a scenario
traceforge mine <scenario-name> --min-confidence 0.9
traceforge mine <scenario-name> --suggest-expectations
traceforge mine --all                              # Mine all scenarios
traceforge mine <scenario-name> --json

# === System Info ===
traceforge info                                    # Show backend (Python vs Rust), store stats
```

---

## Updated HTML Report

Add new sections to the HTML report:

### Causal Attribution Section
Show the sensitivity bar chart for each failing scenario. Highlight top causal factors.

### Invariant Mining Section
Show discovered invariants with discriminating power. Show suggested expectations with "copy to YAML" code blocks.

---

## Implementation Order (Phase 2)

### Sprint 1: Causal Attribution (Days 1-5)
1. `attribution.py` — Intervention models, generators, causal engine
2. `test_attribution.py` — Test with the existing multi_step_math failing scenario
3. CLI `traceforge attribute` command
4. Reporter update for causal output
5. HTML report update

### Sprint 2: Invariant Mining (Days 6-10)
6. `invariants.py` — All invariant types, miners, mining engine
7. `test_invariants.py` — Test with calculator_agent traces (need 10+ runs first)
8. CLI `traceforge mine` command
9. Reporter update for invariant output
10. HTML report update

### Sprint 3: Rust Trace Core (Days 11-18)
11. Rust workspace setup with maturin
12. `store.rs` — CAS store with mmap segments
13. `trace_ir.rs` — Canonical hash (BLAKE3)
14. `compress.rs` — zstd with dictionary
15. `lib.rs` (python crate) — PyO3 bindings
16. Python integration: `get_trace_store()` factory
17. Benchmark: Python store vs Rust store (RSS + throughput)

### Sprint 4: Polish (Days 19-21)
18. Update README with new features
19. Run full 200-scenario suite for paper data
20. Write paper draft
21. Record demo video

---

## Paper Updates (New Sections)

Add to the paper outline:

### Section 4.4: Causal Attribution via Counterfactual Replay
- Define do-operations for agent traces
- Show sensitivity ranking identifies root causes
- Measure time-to-diagnose vs manual debugging

### Section 4.5: Behavioral Invariant Mining
- Adaptation of Daikon-style invariant detection to agent traces
- 8 invariant type miners
- Discriminating invariant analysis (holds in passing, breaks in failing)
- Auto-generated expectations vs hand-written

### Section 5 Results (new metrics):
- Causal attribution accuracy on synthetic injected faults
- Number of discriminating invariants discovered per scenario
- Quality of auto-generated expectations (do they catch real failures?)
- Rust CAS store: RSS reduction and throughput improvement

---

## Success Criteria (Phase 2)

Phase 2 is done when:

1. `traceforge attribute <failing-trace-id>` produces a ranked causal factor report
2. Causal attribution correctly identifies injected faults (format change, value change, schema change)
3. LLM-generated causal narrative is coherent and actionable
4. `traceforge mine calculator_agent` discovers at least 5 invariants from 10+ passing traces
5. Discriminating invariants correctly identify differences between passing and failing traces
6. `--suggest-expectations` outputs valid YAML that can be copy-pasted into scenario files
7. Rust CAS store passes all tests and is a drop-in replacement
8. `traceforge info` shows "Backend: Rust" when extension is installed
9. Benchmark shows measurable RSS improvement with Rust store
10. HTML report includes causal and invariant sections
11. All existing Phase 1 tests still pass
12. New tests pass with >80% coverage on new modules

---

## Design Principles (Unchanged)

1. **Local-first, always.** No cloud. No telemetry. No internet.
2. **Traces are the source of truth.** Immutable. Content-addressed.
3. **Execution ≠ Evaluation.** Evaluate offline. Attribute offline. Mine offline.
4. **Determinism by default.** Seed everything. Hash everything.
5. **Declarative over imperative.** YAML specs. Auto-generated expectations.
6. **Paper-grade reproducibility.** Full execution envelopes. Immutable trace store.
7. **The tool should be smarter than the developer.** Don't just report failures — explain them, discover patterns, suggest fixes.
