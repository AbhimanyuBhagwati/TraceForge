# TraceForge

A test harness for AI agents that call tools.

If you're building agents with tool-calling (on Ollama, local models, etc.) and you're tired of staring at logs trying to figure out why your agent called the wrong tool or returned garbage — this is for you.

## What it does

You write a YAML file describing what your agent should do. TraceForge runs it, records everything, and then lets you analyze the recordings without re-running the model.

```yaml
name: calculator_agent
agent:
  model: qwen2.5:7b-instruct
  system_prompt: "You are a calculator assistant."
  tools:
    - name: calculate
      description: "Perform a math calculation"
      parameters:
        type: object
        properties:
          expression: { type: string }
        required: [expression]
      mock_responses: [{ result: 42 }]

steps:
  - user_message: "What is 6 times 7?"
    expectations:
      - type: tool_called
        tool: calculate
      - type: response_contains
        values: ["42"]
```

```
$ traceforge run ./scenarios/ --runs 10

╭───────────────────── TraceForge Report ──────────────────────╮
│ SCENARIO            PASS  FAIL  RATE  CONSIST  AVG MS       │
│ OK calculator_agent  10/10 0/10 100%    1.00    1,059       │
│ XX multi_step_math    0/10 10/10  0%    1.00    3,598       │
│ OK simple_chat       10/10 0/10 100%    1.00      898       │
│ OK weather_agent     10/10 0/10 100%    1.00    1,246       │
│                                                              │
│ OVERALL: 75.0% pass rate                                     │
╰──────────────────────────────────────────────────────────────╯
```

## The idea

Running an LLM is expensive and slow. But once you have a recording of what it did, you can re-evaluate it instantly, fuzz it, minimize it, and analyze it — all offline.

TraceForge records every agent run as an immutable, content-addressed trace (SHA-256 hashed). Then it gives you tools to work with those traces:

- **Replay** — re-evaluate a trace with different expectations, no model needed
- **Fuzz** — mutate tool responses (nulls, type swaps, empty strings) and see what breaks your agent
- **MinRepro** — your agent runs 4 steps and fails; delta debugging finds the 1 step that actually matters
- **Mine** — automatically discover behavioral rules from passing traces ("calculate is always called at step 0", "expression is always non-empty")
- **Attribute** — when something fails, run counterfactual experiments to find out why ("the agent is sensitive to tool output values, not format")

## Install

```bash
pip install traceforge
```

Or from source:

```bash
git clone https://github.com/AbhimanyuBhagwati/TraceForge.git
cd TraceForge
pip install -e ".[dev]"
```

You'll need [Ollama](https://ollama.com/) running locally with a model pulled:

```bash
ollama pull qwen2.5:7b-instruct
```

## Quick start

```bash
# Create example scenarios
traceforge init

# Run them
traceforge run ./examples/scenarios/ --runs 5

# See what you've got
traceforge traces
traceforge info

# Replay a trace offline (no model call)
traceforge replay <trace-id>

# Fuzz tool responses
traceforge fuzz ./examples/scenarios/

# Find minimal failing case
traceforge minrepro <failing-trace-id> --scenario ./examples/scenarios/

# Discover behavioral patterns
traceforge mine calculator_agent -v

# Find root cause of failure
traceforge attribute <failing-trace-id> --scenario ./examples/scenarios/
```

## How it works

```
YAML scenario
     |
     v
traceforge run         ->  traces (content-addressed, stored locally)
     |
     v
traceforge replay      ->  re-evaluate offline
traceforge fuzz        ->  break tool responses, find fragility
traceforge minrepro    ->  shrink failing trace to minimal case
traceforge mine        ->  discover behavioral rules from traces
traceforge attribute   ->  counterfactual analysis of failures
     |
     v
CLI output / HTML report / JSON export
```

Everything after `run` works on stored traces. Run the model once, analyze as many times as you want.

## Expectations

10 built-in expectation types you can use in your YAML:

| Type | What it checks |
|------|---------------|
| `tool_called` | Agent called this tool |
| `tool_not_called` | Agent didn't call this tool |
| `tool_args_contain` | Tool was called with these arguments |
| `response_contains` | Agent's response includes these strings |
| `response_not_contains` | Agent's response doesn't include this |
| `response_matches_regex` | Response matches a regex |
| `llm_judge` | Another LLM evaluates the response |
| `latency_under` | Step completed within N ms |
| `no_tool_errors` | No tool calls returned errors |
| `tool_call_count` | Tool was called exactly/at least/at most N times |

## Invariant mining

Instead of writing expectations by hand, let TraceForge figure them out:

```bash
$ traceforge mine calculator_agent -v

╭────────────── Invariant Mining Report ───────────────╮
│ Traces analyzed: 15 (15 passing, 0 failing)          │
│ Invariants discovered: 5                             │
│                                                      │
│   - 'calculate' is always called at step 0           │
│   - 'calculate' is called 1-5 times per run          │
│   - 'calculate.expression' is always non-empty       │
│   - Step 0 response length is 30-48 chars            │
│   - Step 0 latency is under 3916ms                   │
╰──────────────────────────────────────────────────────╯
```

Run enough traces and the miner will find rules that hold in all passing traces but break in failing ones. Those are your bugs.

## Causal attribution

When a trace fails, TraceForge can run counterfactual experiments — change one thing at a time, re-run the agent, and see what flips the outcome.

```bash
$ traceforge attribute <trace-id> --scenario ./scenarios/

╭────────────── Causal Attribution Report ─────────────╮
│ Failing step: 2 | Interventions: 23 | Flips: 7      │
│                                                      │
│  CAUSAL FACTOR          SENSITIVITY                  │
│  tool_output_value         40%                       │
│  tool_output_format         0%                       │
│  system_prompt_clause       0%                       │
╰──────────────────────────────────────────────────────╯
```

"40% of value changes flipped the outcome. Format and prompt don't matter." Now you know where to look.

## Requirements

- Python 3.12+
- Ollama running locally
- A pulled model (tested with `qwen2.5:7b-instruct`)

## Tests

```bash
pytest tests/ -v
```

183 tests, runs in about a second.

## License

MIT
