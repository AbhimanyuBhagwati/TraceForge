"""Rich CLI reporter for TraceForge results."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from traceforge.models import (
    CausalReport,
    FuzzReport,
    InvariantReport,
    MinReproResult,
    ProbeReport,
    RunResult,
    ScenarioResult,
)


console = Console()


def print_report(report: ProbeReport, verbose: bool = False):
    """Print the main TraceForge report."""
    # Summary header
    header = Table.grid(padding=1)
    header.add_column(justify="left")
    header.add_column(justify="left")
    header.add_row(
        f"Model: {report.model}",
        f"Date: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    header.add_row(
        f"Scenarios: {report.total_scenarios}",
        f"Total Runs: {report.total_runs}",
    )

    # Scenario table
    table = Table(show_header=True, header_style="bold")
    table.add_column("SCENARIO", style="bold")
    table.add_column("PASS", justify="right")
    table.add_column("FAIL", justify="right")
    table.add_column("RATE", justify="right")
    table.add_column("CONSIST", justify="right")
    table.add_column("AVG MS", justify="right")

    for sr in report.scenario_results:
        icon = _status_icon(sr.pass_rate)
        rate_style = _rate_style(sr.pass_rate)
        table.add_row(
            f"{icon} {sr.scenario_name}",
            f"{sr.passed_runs}/{sr.total_runs}",
            f"{sr.failed_runs}/{sr.total_runs}",
            Text(f"{sr.pass_rate:.0%}", style=rate_style),
            f"{sr.consistency_score:.2f}",
            f"{sr.avg_latency_ms:,.0f}",
        )

    # Footer
    footer_text = (
        f"OVERALL: {report.overall_pass_rate:.1%} pass rate"
    )
    if report.regression_warnings:
        for w in report.regression_warnings:
            footer_text += f"\n  {w}"

    content = Table.grid()
    content.add_row(header)
    content.add_row(table)
    content.add_row(Text(footer_text))

    console.print(Panel(content, title="TraceForge Report", border_style="blue"))

    if verbose:
        for sr in report.scenario_results:
            _print_verbose_scenario(sr)


def _print_verbose_scenario(sr: ScenarioResult):
    """Print per-run details for a scenario."""
    console.print(f"\n[bold]{sr.scenario_name}[/bold]")
    for rr in sr.run_results:
        icon = "[green]PASS[/green]" if rr.passed else "[red]FAIL[/red]"
        console.print(f"  Run {rr.run_number}: {icon} ({rr.total_latency_ms:.0f}ms)")
        for step_r in rr.step_results:
            for er in step_r.results:
                mark = "[green]+[/green]" if er.passed else "[red]-[/red]"
                console.print(f"    {mark} {er.message}")


def print_fuzz_report(report: FuzzReport):
    """Print fuzzing results."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("TOOL")
    table.add_column("MUTATIONS", justify="right")
    table.add_column("BREAKS", justify="right")
    table.add_column("ROBUSTNESS", justify="right")

    for tool_name, score in report.by_tool.items():
        tool_mutations = sum(1 for r in report.results if r.tool_name == tool_name)
        tool_breaks = sum(1 for r in report.results if r.tool_name == tool_name and r.broke_agent)
        warn = "  " if score >= 0.7 else "  "
        table.add_row(tool_name, str(tool_mutations), str(tool_breaks), f"{score:.0%}{warn}")

    # By mutation type
    type_table = Table(show_header=True, header_style="bold", title="By mutation type")
    type_table.add_column("TYPE")
    type_table.add_column("ROBUSTNESS", justify="right")
    for mt, score in report.by_mutation_type.items():
        type_table.add_row(mt, f"{score:.0%}")

    content = Table.grid()
    content.add_row(table)
    content.add_row(type_table)

    console.print(Panel(content, title="Fuzzing Report", border_style="yellow"))


def print_minrepro_result(result: MinReproResult):
    """Print min-repro results."""
    content = (
        f"Original: {result.original_step_count} steps, {result.original_tool_call_count} tool calls\n"
        f"Minimized: {result.minimized_step_count} steps, {result.minimized_tool_call_count} tool calls\n"
        f"Reduction: {result.reduction_ratio:.0%}\n"
        f"Iterations: {result.iterations_taken}\n"
        f"Minimized steps: {result.minimized_steps}"
    )
    console.print(Panel(content, title="Minimal Reproduction", border_style="green"))


def print_causal_report(report: CausalReport):
    """Print causal attribution results."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("CAUSAL FACTOR")
    table.add_column("SENSITIVITY", justify="right")
    table.add_column("FLIPS", justify="right")
    table.add_column("TESTED", justify="right")

    for f in report.causal_factors:
        sensitivity = f["sensitivity"]
        if sensitivity >= 0.5:
            style = "red"
        elif sensitivity > 0:
            style = "yellow"
        else:
            style = "dim"
        table.add_row(
            f["factor"],
            Text(f"{sensitivity:.0%}", style=style),
            f"{f['flips']}/{f['total']}",
            str(f["total"]),
        )

    header_text = (
        f"Scenario: {report.scenario_name}\n"
        f"Failing step: {report.failing_step} | Trace: {report.failing_trace_id[:12]}...\n"
        f"Interventions tested: {report.total_interventions} | Flips found: {report.total_flips}"
    )

    content = Table.grid()
    content.add_row(Text(header_text))
    content.add_row(table)
    content.add_row(Text(report.summary))

    console.print(Panel(content, title="Causal Attribution Report", border_style="red"))


def print_invariant_report(report: InvariantReport, verbose: bool = False):
    """Print invariant mining results."""
    header_text = (
        f"Traces analyzed: {report.total_traces_analyzed} "
        f"({report.passing_traces} passing, {report.failing_traces} failing)\n"
        f"Invariants discovered: {report.invariants_discovered}\n"
        f"Discriminating invariants: {len(report.discriminating_invariants)}"
    )

    content = Table.grid()
    content.add_row(Text(header_text))

    if report.discriminating_invariants:
        content.add_row(Text("\nDISCRIMINATING INVARIANTS (break in failing traces):", style="bold"))
        for i, inv in enumerate(report.discriminating_invariants[:10], 1):
            violation_text = f"Violations: {inv.violations}/{report.failing_traces} failing traces"
            content.add_row(Text(
                f"  {i}. {inv.description}\n"
                f"     Confidence: {inv.confidence:.0%} | {violation_text}"
            ))

    if report.suggested_expectations:
        content.add_row(Text("\nSUGGESTED EXPECTATIONS:", style="bold"))
        for s in report.suggested_expectations:
            exp = s["expectation"]
            content.add_row(Text(f"  - type: {exp['type']}"))
            for k, v in exp.items():
                if k != "type":
                    content.add_row(Text(f"    {k}: {v}"))
            content.add_row(Text(f"    # {s['source']}", style="dim"))

    if verbose and report.invariants:
        content.add_row(Text(f"\nALL INVARIANTS ({len(report.invariants)} total):", style="bold"))
        for inv in report.invariants:
            content.add_row(Text(f"  - {inv.description} (confidence: {inv.confidence:.0%})"))

    console.print(Panel(content, title="Invariant Mining Report", border_style="cyan"))


def _status_icon(pass_rate: float) -> str:
    if pass_rate >= 0.9:
        return "[green]OK[/green]"
    elif pass_rate >= 0.5:
        return "[yellow]!![/yellow]"
    else:
        return "[red]XX[/red]"


def _rate_style(pass_rate: float) -> str:
    if pass_rate >= 0.9:
        return "green"
    elif pass_rate >= 0.5:
        return "yellow"
    return "red"
