"""HTML report generator with Jinja2."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, BaseLoader

from traceforge.models import CausalReport, FuzzReport, InvariantReport, MinReproResult, ProbeReport


# Inline template for self-contained HTML report
REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TraceForge Report</title>
<style>
:root { --bg: #1a1b26; --fg: #c0caf5; --card: #24283b; --border: #414868;
        --green: #9ece6a; --red: #f7768e; --yellow: #e0af68; --blue: #7aa2f7; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'JetBrains Mono', 'Fira Code', monospace; background: var(--bg);
       color: var(--fg); padding: 2rem; line-height: 1.6; }
h1 { color: var(--blue); margin-bottom: 0.5rem; }
h2 { color: var(--fg); margin: 1.5rem 0 0.5rem; font-size: 1.1rem; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 1rem; margin: 1rem 0; }
.stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
        padding: 1rem; text-align: center; }
.stat .value { font-size: 1.8rem; font-weight: bold; }
.stat .label { font-size: 0.8rem; opacity: 0.7; }
.green { color: var(--green); }
.red { color: var(--red); }
.yellow { color: var(--yellow); }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th, td { padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--card); color: var(--blue); }
tr:hover { background: var(--card); }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
        padding: 1rem; margin: 1rem 0; }
details { margin: 0.5rem 0; }
summary { cursor: pointer; padding: 0.3rem 0; }
.bar { height: 16px; border-radius: 4px; background: var(--border); overflow: hidden; display: inline-block; width: 100px; }
.bar-fill { height: 100%; border-radius: 4px; }
.warn { background: var(--card); border-left: 3px solid var(--yellow); padding: 0.8rem; margin: 0.5rem 0; }
footer { margin-top: 2rem; text-align: center; opacity: 0.5; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>TraceForge Report</h1>
<p>{{ report.timestamp.strftime('%Y-%m-%d %H:%M:%S') }} | Model: {{ report.model }}</p>

<div class="summary">
  <div class="stat">
    <div class="value">{{ report.total_scenarios }}</div>
    <div class="label">Scenarios</div>
  </div>
  <div class="stat">
    <div class="value">{{ report.total_runs }}</div>
    <div class="label">Total Runs</div>
  </div>
  <div class="stat">
    <div class="value {{ 'green' if report.overall_pass_rate >= 0.9 else 'yellow' if report.overall_pass_rate >= 0.5 else 'red' }}">
      {{ "%.0f" | format(report.overall_pass_rate * 100) }}%
    </div>
    <div class="label">Pass Rate</div>
  </div>
</div>

{% if report.regression_warnings %}
<h2>Regression Warnings</h2>
{% for w in report.regression_warnings %}
<div class="warn">{{ w }}</div>
{% endfor %}
{% endif %}

<h2>Scenarios</h2>
<table>
  <tr><th>Scenario</th><th>Pass</th><th>Fail</th><th>Rate</th><th>Consistency</th><th>Avg Latency</th></tr>
  {% for sr in report.scenario_results %}
  <tr>
    <td>{{ sr.scenario_name }}</td>
    <td>{{ sr.passed_runs }}/{{ sr.total_runs }}</td>
    <td>{{ sr.failed_runs }}/{{ sr.total_runs }}</td>
    <td class="{{ 'green' if sr.pass_rate >= 0.9 else 'yellow' if sr.pass_rate >= 0.5 else 'red' }}">
      {{ "%.0f" | format(sr.pass_rate * 100) }}%
    </td>
    <td>{{ "%.2f" | format(sr.consistency_score) }}</td>
    <td>{{ "%.0f" | format(sr.avg_latency_ms) }}ms</td>
  </tr>
  {% endfor %}
</table>

{% for sr in report.scenario_results %}
<div class="card">
  <h2>{{ sr.scenario_name }}</h2>
  {% if sr.description %}<p>{{ sr.description }}</p>{% endif %}
  {% for rr in sr.run_results %}
  <details>
    <summary>Run {{ rr.run_number }}: {{ "PASS" if rr.passed else "FAIL" }} ({{ "%.0f" | format(rr.total_latency_ms) }}ms)</summary>
    {% for step_r in rr.step_results %}
    <div style="margin-left: 1rem; margin-top: 0.5rem;">
      <strong>Step {{ step_r.step_index }}: {{ step_r.user_message[:80] }}</strong>
      {% for er in step_r.results %}
      <div style="margin-left: 1rem;">
        <span class="{{ 'green' if er.passed else 'red' }}">{{ "+" if er.passed else "-" }}</span>
        {{ er.message }}
      </div>
      {% endfor %}
    </div>
    {% endfor %}
  </details>
  {% endfor %}
</div>
{% endfor %}

{% if fuzz_report %}
<h2>Fuzzing Results</h2>
<div class="card">
  <p>Robustness: <span class="{{ 'green' if fuzz_report.robustness_score >= 0.8 else 'yellow' if fuzz_report.robustness_score >= 0.5 else 'red' }}">
    {{ "%.0f" | format(fuzz_report.robustness_score * 100) }}%</span>
    ({{ fuzz_report.total_mutations }} mutations, {{ fuzz_report.total_breaks }} breaks)</p>
  <table>
    <tr><th>Tool</th><th>Robustness</th></tr>
    {% for tool, score in fuzz_report.by_tool.items() %}
    <tr>
      <td>{{ tool }}</td>
      <td>
        <div class="bar"><div class="bar-fill" style="width: {{ score * 100 }}%; background: {{ 'var(--green)' if score >= 0.8 else 'var(--yellow)' if score >= 0.5 else 'var(--red)' }};"></div></div>
        {{ "%.0f" | format(score * 100) }}%
      </td>
    </tr>
    {% endfor %}
  </table>
  <table>
    <tr><th>Mutation Type</th><th>Robustness</th></tr>
    {% for mt, score in fuzz_report.by_mutation_type.items() %}
    <tr>
      <td>{{ mt }}</td>
      <td>
        <div class="bar"><div class="bar-fill" style="width: {{ score * 100 }}%; background: {{ 'var(--green)' if score >= 0.8 else 'var(--yellow)' if score >= 0.5 else 'var(--red)' }};"></div></div>
        {{ "%.0f" | format(score * 100) }}%
      </td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

{% if causal_reports %}
<h2>Causal Attribution</h2>
{% for cr in causal_reports %}
<div class="card">
  <h2>{{ cr.scenario_name }} (step {{ cr.failing_step }})</h2>
  <p>Trace: {{ cr.failing_trace_id[:12] }}... | Interventions: {{ cr.total_interventions }} | Flips: {{ cr.total_flips }}</p>
  <table>
    <tr><th>Causal Factor</th><th>Sensitivity</th><th>Flips</th><th>Tested</th></tr>
    {% for f in cr.causal_factors %}
    <tr>
      <td>{{ f.factor }}</td>
      <td class="{{ 'red' if f.sensitivity >= 0.5 else 'yellow' if f.sensitivity > 0 else '' }}">
        {{ "%.0f" | format(f.sensitivity * 100) }}%
      </td>
      <td>{{ f.flips }}/{{ f.total }}</td>
      <td>{{ f.total }}</td>
    </tr>
    {% endfor %}
  </table>
  <div style="margin-top: 1rem; white-space: pre-wrap;">{{ cr.summary }}</div>
</div>
{% endfor %}
{% endif %}

{% if invariant_reports %}
<h2>Invariant Mining</h2>
{% for ir in invariant_reports %}
<div class="card">
  <p>Traces: {{ ir.total_traces_analyzed }} ({{ ir.passing_traces }} passing, {{ ir.failing_traces }} failing) |
     Invariants: {{ ir.invariants_discovered }} | Discriminating: {{ ir.discriminating_invariants | length }}</p>
  {% if ir.discriminating_invariants %}
  <h3 style="color: var(--yellow);">Discriminating Invariants</h3>
  <table>
    <tr><th>#</th><th>Invariant</th><th>Confidence</th><th>Violations</th></tr>
    {% for inv in ir.discriminating_invariants[:10] %}
    <tr>
      <td>{{ loop.index }}</td>
      <td>{{ inv.description }}</td>
      <td>{{ "%.0f" | format(inv.confidence * 100) }}%</td>
      <td>{{ inv.violations }}/{{ ir.failing_traces }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}
  {% if ir.suggested_expectations %}
  <h3>Suggested Expectations</h3>
  <pre style="background: var(--bg); padding: 1rem; border-radius: 4px;">{% for s in ir.suggested_expectations %}
- type: {{ s.expectation.type }}{% for k, v in s.expectation.items() %}{% if k != 'type' %}
  {{ k }}: {{ v }}{% endif %}{% endfor %}
  # {{ s.source }}
{% endfor %}</pre>
  {% endif %}
</div>
{% endfor %}
{% endif %}

{% if minrepro_results %}
<h2>Minimal Reproductions</h2>
{% for mr in minrepro_results %}
<div class="card">
  <p>Original: {{ mr.original_step_count }} steps, {{ mr.original_tool_call_count }} tool calls</p>
  <p>Minimized: {{ mr.minimized_step_count }} steps, {{ mr.minimized_tool_call_count }} tool calls</p>
  <p>Reduction: <span class="green">{{ "%.0f" | format(mr.reduction_ratio * 100) }}%</span> in {{ mr.iterations_taken }} iterations</p>
  <p>Remaining steps: {{ mr.minimized_steps }}</p>
</div>
{% endfor %}
{% endif %}

<footer>Generated by TraceForge v0.2.0</footer>
</body>
</html>"""


def generate_html_report(
    report: ProbeReport,
    output_path: str,
    fuzz_report: FuzzReport | None = None,
    minrepro_results: list[MinReproResult] | None = None,
    causal_reports: list[CausalReport] | None = None,
    invariant_reports: list[InvariantReport] | None = None,
):
    """Generate a self-contained HTML report."""
    env = Environment(loader=BaseLoader())
    template = env.from_string(REPORT_TEMPLATE)

    html = template.render(
        report=report,
        fuzz_report=fuzz_report,
        minrepro_results=minrepro_results or [],
        causal_reports=causal_reports or [],
        invariant_reports=invariant_reports or [],
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return str(output)
