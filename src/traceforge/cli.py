"""Click CLI commands for TraceForge."""

import json
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.table import Table

from traceforge import __version__
from traceforge.evaluator import Evaluator, aggregate_results
from traceforge.harness import Harness
from traceforge.history import HistoryStore
from traceforge.html_report import generate_html_report
from traceforge.judge import JudgeClient
from traceforge.loader import load_scenarios
from traceforge.models import MutatorConfig, ProbeReport
from traceforge.reporter import (
    print_report,
    print_fuzz_report,
    print_minrepro_result,
    print_causal_report,
    print_invariant_report,
)
from traceforge.trace_store import TraceStore

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """TraceForge: Deterministic replay + fuzzing + failure minimization for tool-calling agents."""
    pass


@main.command()
@click.argument("path")
@click.option("--tag", default=None, help="Filter scenarios by tag")
@click.option("--runs", default=None, type=int, help="Override run count")
@click.option("--ollama-host", default="http://localhost:11434", help="Ollama host URL")
@click.option("-v", "--verbose", is_flag=True, help="Show per-step details")
@click.option("--html-report", default=None, help="Path to save HTML report")
def run(path, tag, runs, ollama_host, verbose, html_report):
    """Run test scenarios against Ollama."""
    try:
        scenarios = load_scenarios(path, tag_filter=tag)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    if not scenarios:
        console.print("[yellow]No scenarios found.[/yellow]")
        return

    store = TraceStore()
    history = HistoryStore()
    harness = Harness(trace_store=store, ollama_host=ollama_host)

    all_scenario_results = []
    total_runs = 0
    model = scenarios[0].agent.model

    for scenario in scenarios:
        console.print(f"[blue]Running:[/blue] {scenario.name} ({runs or scenario.runs} runs)")

        judge = None
        if scenario.judge:
            judge = JudgeClient(
                model=scenario.judge.model,
                temperature=scenario.judge.temperature,
                seed=scenario.judge.seed,
                ollama_host=ollama_host,
            )

        evaluator = Evaluator(judge=judge)

        try:
            traces = harness.run_scenario(scenario, runs=runs)
        except Exception as e:
            console.print(f"[red]Error running {scenario.name}:[/red] {e}")
            continue

        run_results = []
        for trace in traces:
            result = evaluator.evaluate(trace, scenario)
            # Update trace store with pass/fail
            store.store(trace, passed=result.passed)
            run_results.append(result)

        scenario_result = aggregate_results(scenario, run_results)
        all_scenario_results.append(scenario_result)
        total_runs += len(run_results)

        # Check regression
        regression = history.check_regression(scenario_result, model)
        history.record(scenario_result, model)
        if regression:
            console.print(f"[yellow]{regression}[/yellow]")

    # Build report
    overall_pass = (
        sum(sr.passed_runs for sr in all_scenario_results) / total_runs
        if total_runs > 0 else 0.0
    )
    regressions = [
        history.check_regression(sr, model)
        for sr in all_scenario_results
    ]
    regressions = [r for r in regressions if r]

    report = ProbeReport(
        timestamp=datetime.now(timezone.utc),
        model=model,
        total_scenarios=len(all_scenario_results),
        total_runs=total_runs,
        overall_pass_rate=overall_pass,
        scenario_results=all_scenario_results,
        regression_warnings=regressions,
    )

    print_report(report, verbose=verbose)

    if html_report:
        path = generate_html_report(report, html_report)
        console.print(f"[green]HTML report saved:[/green] {path}")


@main.command()
@click.argument("trace_id", required=False)
@click.option("--all", "replay_all", is_flag=True, help="Replay all stored traces")
@click.option("--compare", nargs=2, help="Diff two trace IDs")
@click.option("--scenario", default=None, help="Scenario path for expectations")
@click.option("--ollama-host", default="http://localhost:11434")
def replay(trace_id, replay_all, compare, scenario, ollama_host):
    """Virtual replay: re-evaluate traces without calling the model."""
    from traceforge.replay import ReplayEngine

    store = TraceStore()
    engine = ReplayEngine(store)

    if compare:
        diff = engine.diff_traces(compare[0], compare[1])
        console.print_json(json.dumps(diff, indent=2))
        return

    if replay_all:
        traces = store.list_traces()
        if not traces:
            console.print("[yellow]No traces found.[/yellow]")
            return
        console.print(f"[blue]Replaying {len(traces)} traces...[/blue]")
        for t in traces:
            _replay_single(engine, t["trace_id"], scenario, ollama_host)
        return

    if trace_id:
        _replay_single(engine, trace_id, scenario, ollama_host)
    else:
        console.print("[red]Provide a trace ID, --all, or --compare.[/red]")


def _replay_single(engine, trace_id, scenario_path, ollama_host):
    """Replay a single trace."""
    if scenario_path:
        scenarios = load_scenarios(scenario_path)
        trace = engine.store.load(trace_id)
        scenario = next(
            (s for s in scenarios if s.name == trace.scenario_name), None
        )
        if not scenario:
            console.print(f"[yellow]No matching scenario for trace {trace_id[:12]}[/yellow]")
            return
    else:
        # Replay with no expectations (just load and verify)
        trace = engine.store.load(trace_id)
        from traceforge.models import AgentConfig, Scenario, Step
        scenario = Scenario(
            name=trace.scenario_name,
            agent=AgentConfig(),
            steps=[Step(user_message=s.user_message) for s in trace.steps],
        )

    result = engine.virtual_replay(trace_id, scenario)
    icon = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
    console.print(f"  {trace_id[:12]}... {icon}")


@main.command()
@click.argument("path")
@click.option("--mutations", default=None, type=int, help="Mutations per tool")
@click.option("--types", default=None, help="Comma-separated mutation types")
@click.option("--ollama-host", default="http://localhost:11434")
@click.option("--html-report", default=None, help="Path to save HTML report")
def fuzz(path, mutations, types, ollama_host, html_report):
    """Run schema-aware fuzzing on scenarios."""
    from traceforge.fuzzer import DifferentialFuzzer

    scenarios = load_scenarios(path)
    store = TraceStore()
    harness = Harness(trace_store=store, ollama_host=ollama_host)
    evaluator = Evaluator()
    fuzzer = DifferentialFuzzer(harness, evaluator, store)

    for scenario in scenarios:
        config = scenario.mutator or MutatorConfig(enabled=True)
        if mutations:
            config = config.model_copy(update={"mutations_per_tool": mutations})
        if types:
            config = config.model_copy(update={"mutation_types": types.split(",")})

        console.print(f"[blue]Fuzzing:[/blue] {scenario.name}")
        report = fuzzer.fuzz_scenario(scenario, config)
        print_fuzz_report(report)

        if html_report:
            # Generate a minimal probe report wrapper
            probe = ProbeReport(
                timestamp=datetime.now(timezone.utc),
                model=scenario.agent.model,
                total_scenarios=1,
                total_runs=0,
                overall_pass_rate=0.0,
                scenario_results=[],
            )
            path_out = generate_html_report(probe, html_report, fuzz_report=report)
            console.print(f"[green]HTML report saved:[/green] {path_out}")


@main.command()
@click.argument("trace_id", required=False)
@click.option("--all-failures", is_flag=True, help="Minimize all failing traces")
@click.option("--scenario", required=True, help="Scenario path")
@click.option("--ollama-host", default="http://localhost:11434")
def minrepro(trace_id, all_failures, scenario, ollama_host):
    """Minimize a failing trace using delta debugging."""
    from traceforge.minrepro import MinReproExtractor

    store = TraceStore()
    harness = Harness(trace_store=store, ollama_host=ollama_host)
    evaluator = Evaluator()
    extractor = MinReproExtractor(harness, evaluator, store)

    scenarios = load_scenarios(scenario)

    if all_failures:
        failed_traces = store.list_traces(passed=False)
        if not failed_traces:
            console.print("[yellow]No failing traces found.[/yellow]")
            return
        for t in failed_traces:
            s = next((s for s in scenarios if s.name == t["scenario_name"]), None)
            if s:
                result = extractor.minimize(t["trace_id"], s)
                print_minrepro_result(result)
    elif trace_id:
        trace = store.load(trace_id)
        s = next((s for s in scenarios if s.name == trace.scenario_name), None)
        if not s:
            console.print(f"[red]No matching scenario for trace {trace_id[:12]}[/red]")
            return
        result = extractor.minimize(trace_id, s)
        print_minrepro_result(result)
    else:
        console.print("[red]Provide a trace ID or --all-failures.[/red]")


@main.command()
@click.option("--scenario", default=None, help="Filter by scenario name")
@click.option("--limit", default=20, help="Max rows to show")
def history(scenario, limit):
    """View regression history."""
    hist = HistoryStore()
    rows = hist.get_history(scenario_name=scenario, limit=limit)
    if not rows:
        console.print("[yellow]No history found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID")
    table.add_column("Timestamp")
    table.add_column("Scenario")
    table.add_column("Model")
    table.add_column("Rate", justify="right")
    table.add_column("Consist", justify="right")
    table.add_column("Avg MS", justify="right")

    for r in rows:
        table.add_row(
            str(r["id"]),
            r["timestamp"][:19],
            r["scenario_name"],
            r["model"],
            f"{r['pass_rate']:.0%}",
            f"{r['consistency_score']:.2f}",
            f"{r['avg_latency_ms']:.0f}",
        )
    console.print(table)


@main.command()
@click.option("--scenario", default=None, help="Filter by scenario name")
def traces(scenario):
    """List stored traces."""
    store = TraceStore()
    trace_list = store.list_traces(scenario_name=scenario)
    if not trace_list:
        console.print("[yellow]No traces found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Trace ID")
    table.add_column("Scenario")
    table.add_column("Run")
    table.add_column("Passed")
    table.add_column("Steps")
    table.add_column("Tools")
    table.add_column("Latency", justify="right")

    for t in trace_list:
        table.add_row(
            t["trace_id"][:12] + "...",
            t["scenario_name"],
            str(t["run_number"]),
            "[green]Yes[/green]" if t["passed"] else "[red]No[/red]",
            str(t["step_count"]),
            str(t["tool_call_count"]),
            f"{t['total_latency_ms']:.0f}ms",
        )
    console.print(table)


@main.command()
def init():
    """Initialize example scenarios in the current directory."""
    import os
    from pathlib import Path

    examples_dir = Path("examples")
    scenarios_dir = examples_dir / "scenarios"
    prompts_dir = examples_dir / "prompts"
    mocks_dir = examples_dir / "mock_responses"

    for d in [scenarios_dir, prompts_dir, mocks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Calculator prompt
    (prompts_dir / "calculator.txt").write_text(
        "You are a calculator assistant. Use the calculate tool to perform math operations. "
        "Always show the result clearly."
    )

    # Calculator mock responses
    (mocks_dir / "calculator.json").write_text(json.dumps([
        {"result": 42},
        {"result": 3.14159},
        {"result": 100},
    ], indent=2))

    # Calculator scenario
    (scenarios_dir / "calculator_agent.yaml").write_text("""\
name: calculator_agent
description: "Agent correctly uses calculator tool for math questions"
tags: [tools, math, basic]

agent:
  model: qwen2.5:7b-instruct
  system_prompt_file: ../prompts/calculator.txt
  temperature: 0.1
  seed: 42
  tools:
    - name: calculate
      description: "Perform a mathematical calculation"
      parameters:
        type: object
        properties:
          expression:
            type: string
            description: "Math expression to evaluate"
        required: [expression]
      mock_response_file: ../mock_responses/calculator.json

steps:
  - user_message: "What is 6 times 7?"
    expectations:
      - type: tool_called
        tool: calculate
      - type: response_contains
        values: ["42"]
      - type: latency_under
        max_ms: 30000
      - type: no_tool_errors

runs: 3
""")

    # Weather scenario
    (prompts_dir / "weather.txt").write_text(
        "You are a weather assistant. Use the get_weather tool to look up weather information. "
        "Provide clear, helpful weather summaries."
    )

    (mocks_dir / "weather.json").write_text(json.dumps([
        {"temperature": 72, "condition": "sunny", "humidity": 45},
        {"temperature": 55, "condition": "cloudy", "humidity": 80},
    ], indent=2))

    (scenarios_dir / "weather_agent.yaml").write_text("""\
name: weather_agent
description: "Agent looks up weather information"
tags: [tools, weather]

agent:
  model: qwen2.5:7b-instruct
  system_prompt_file: ../prompts/weather.txt
  temperature: 0.1
  seed: 42
  tools:
    - name: get_weather
      description: "Get current weather for a location"
      parameters:
        type: object
        properties:
          location:
            type: string
            description: "City name"
        required: [location]
      mock_response_file: ../mock_responses/weather.json

steps:
  - user_message: "What's the weather in San Francisco?"
    expectations:
      - type: tool_called
        tool: get_weather
      - type: tool_args_contain
        tool: get_weather
        args_contain:
          location: "San Francisco"
      - type: no_tool_errors
      - type: latency_under
        max_ms: 30000

runs: 3
""")

    # Simple chat scenario (no tools)
    (scenarios_dir / "simple_chat.yaml").write_text("""\
name: simple_chat
description: "Agent responds appropriately to a greeting"
tags: [basic, no-tools]

agent:
  model: qwen2.5:7b-instruct
  system_prompt: "You are a helpful, friendly assistant."
  temperature: 0.1
  seed: 42

steps:
  - user_message: "Hello! How are you?"
    expectations:
      - type: response_contains
        values: ["hello", "hi", "hey", "greet"]
      - type: response_not_contains
        value: "error"
      - type: latency_under
        max_ms: 30000

runs: 3
""")

    console.print("[green]Example scenarios created in ./examples/[/green]")
    console.print("Run them with: [bold]traceforge run ./examples/scenarios/[/bold]")


@main.command()
@click.argument("trace_id", required=False)
@click.option("--all-failures", is_flag=True, help="Attribute all failing traces")
@click.option("--scenario", required=True, help="Scenario path")
@click.option("--confirmation-runs", default=3, type=int, help="Runs per counterfactual")
@click.option("--max-interventions", default=50, type=int, help="Max interventions to test")
@click.option("--ollama-host", default="http://localhost:11434")
@click.option("--json-output", "json_out", is_flag=True, help="Output as JSON")
def attribute(trace_id, all_failures, scenario, confirmation_runs, max_interventions, ollama_host, json_out):
    """Causal attribution: find WHY a trace fails."""
    from traceforge.attribution import CausalAttributionEngine

    store = TraceStore()
    harness = Harness(trace_store=store, ollama_host=ollama_host)
    evaluator = Evaluator()
    engine = CausalAttributionEngine(harness, evaluator, store)

    scenarios = load_scenarios(scenario)

    if all_failures:
        failed_traces = store.list_traces(passed=False)
        if not failed_traces:
            console.print("[yellow]No failing traces found.[/yellow]")
            return
        for t in failed_traces:
            s = next((s for s in scenarios if s.name == t["scenario_name"]), None)
            if s:
                try:
                    report = engine.attribute(
                        t["trace_id"], s,
                        confirmation_runs=confirmation_runs,
                        max_interventions=max_interventions,
                    )
                    if json_out:
                        console.print_json(report.model_dump_json())
                    else:
                        print_causal_report(report)
                except ValueError as e:
                    console.print(f"[yellow]{e}[/yellow]")
    elif trace_id:
        trace = store.load(trace_id)
        s = next((s for s in scenarios if s.name == trace.scenario_name), None)
        if not s:
            console.print(f"[red]No matching scenario for trace {trace_id[:12]}[/red]")
            return
        report = engine.attribute(
            trace_id, s,
            confirmation_runs=confirmation_runs,
            max_interventions=max_interventions,
        )
        if json_out:
            console.print_json(report.model_dump_json())
        else:
            print_causal_report(report)
    else:
        console.print("[red]Provide a trace ID or --all-failures.[/red]")


@main.command()
@click.argument("scenario_name", required=False)
@click.option("--all", "mine_all", is_flag=True, help="Mine all scenarios")
@click.option("--min-confidence", default=0.95, type=float, help="Minimum confidence threshold")
@click.option("--suggest-expectations", is_flag=True, help="Show suggested YAML expectations")
@click.option("--json-output", "json_out", is_flag=True, help="Output as JSON")
@click.option("-v", "--verbose", is_flag=True, help="Show all invariants")
def mine(scenario_name, mine_all, min_confidence, suggest_expectations, json_out, verbose):
    """Mine behavioral invariants from stored traces."""
    from traceforge.invariants import InvariantMiner

    store = TraceStore()
    miner = InvariantMiner(store)

    if mine_all:
        all_traces = store.list_traces()
        scenario_names = sorted(set(t["scenario_name"] for t in all_traces))
    elif scenario_name:
        scenario_names = [scenario_name]
    else:
        console.print("[red]Provide a scenario name or --all.[/red]")
        return

    for name in scenario_names:
        try:
            report = miner.mine(name, min_confidence=min_confidence)
            if json_out:
                console.print_json(report.model_dump_json())
            else:
                console.print(f"[blue]Scenario:[/blue] {name}")
                print_invariant_report(report, verbose=verbose)
        except ValueError as e:
            console.print(f"[yellow]{name}: {e}[/yellow]")


@main.command()
def info():
    """Show TraceForge system info and store statistics."""
    store = TraceStore()
    all_traces = store.list_traces()

    console.print(f"[bold]TraceForge v{__version__}[/bold]")
    console.print(f"Backend: Python")
    console.print(f"Store: {store.base_dir}")
    console.print(f"Total traces: {len(all_traces)}")

    if all_traces:
        passing = sum(1 for t in all_traces if t.get("passed"))
        failing = len(all_traces) - passing
        scenarios = set(t["scenario_name"] for t in all_traces)
        console.print(f"Passing: {passing} | Failing: {failing}")
        console.print(f"Scenarios: {', '.join(sorted(scenarios))}")


if __name__ == "__main__":
    main()
