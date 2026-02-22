# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TraceForge is a local-first testing/debugging framework for LLM tool-calling agents. It records executions as content-addressed traces (Trace IR), replays evaluations without re-running the model, fuzzes tool responses, and minimizes failing traces via delta debugging. Targets Ollama-hosted models (Qwen 2.5 7B).

The full specification is in `claude.md` — follow it precisely for architecture, models, and implementation order.

## Build & Run

```bash
pip install -e .              # Install in dev mode
pip install -e ".[dev]"       # With test dependencies
traceforge run ./examples/scenarios/          # Run all scenarios
traceforge replay <trace-id>                  # Virtual replay (no model call)
traceforge fuzz ./examples/scenarios/calc.yaml  # Schema-aware fuzzing
traceforge minrepro <trace-id>                # Delta-debug a failing trace
```

## Testing

```bash
pytest tests/ -v --cov=traceforge             # Full suite with coverage
pytest tests/test_models.py -v                # Single test file
pytest tests/test_trace_ir.py -k "determinism" # Single test by name
```

## Architecture

Source lives in `src/traceforge/`. Key data flow:

1. **YAML specs** (`loader.py`) → validated `Scenario` Pydantic models (`models.py`)
2. **Execution** (`harness.py`) → calls Ollama, produces `TraceIR` via `trace_ir.py`
3. **Storage** (`trace_store.py`) → gzip-compressed JSON blobs named by SHA-256, SQLite index
4. **Evaluation** (`evaluator.py`) → scores traces against expectations → `RunResult`
5. **Novel modules** operate on stored traces:
   - `replay.py` — re-evaluates traces without calling the model (virtual replay)
   - `fuzzer.py` + `mutators.py` — mutates tool *responses* (not args), reruns agent
   - `minrepro.py` — ddmin algorithm to find minimal failing step subset

All Pydantic models are in `models.py`. The `TraceIR` model is the fundamental unit — everything else operates on it.

## Key Design Constraints

- **Tech stack is fixed:** Python 3.12+, click, ollama, pydantic v2, rich, jinja2, PyYAML. No LangChain/FastAPI/Streamlit.
- **Execution ≠ Evaluation:** Once traces are stored, eval/replay/fuzz/minimize run offline.
- **Content addressing:** `trace_ir.canonical_serialize()` excludes `trace_id` and `metadata`, uses sorted keys and compact JSON. Two identical executions must produce the same hash.
- **Mock tools only** for MVP — no real tool execution.
- **Fuzzer mutates tool responses** (what tools return to the agent), not tool arguments (what the agent sends).
- **Implementation order matters:** Follow the phased order in `claude.md` (models → loader → mock_tools → trace_ir → trace_store → harness → evaluator → judge → replay → mutators → fuzzer → minrepro → reporter → html_report → history → cli).
