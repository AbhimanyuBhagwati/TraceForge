"""Execution harness — runs scenarios against Ollama."""

import json
import time
from datetime import datetime, timezone

import ollama as ollama_client

from traceforge.mock_tools import MockToolRegistry
from traceforge.models import (
    ExecutionEnvelope,
    Scenario,
    StepRecord,
    ToolCallRecord,
    TraceIR,
)
from traceforge.trace_ir import finalize_trace
from traceforge.trace_store import TraceStore


class ModelNotFoundError(Exception):
    pass


class Harness:
    """Runs scenarios against Ollama and produces TraceIR objects."""

    MAX_TOOL_ITERATIONS = 5

    def __init__(
        self,
        trace_store: TraceStore | None = None,
        ollama_host: str = "http://localhost:11434",
    ):
        self.trace_store = trace_store
        self.ollama_host = ollama_host
        self._client = ollama_client.Client(host=ollama_host)

    def run_scenario(self, scenario: Scenario, runs: int | None = None) -> list[TraceIR]:
        """Run a scenario for the specified number of runs. Returns list of TraceIR."""
        num_runs = runs if runs is not None else scenario.runs
        traces = []
        for run_num in range(1, num_runs + 1):
            trace = self._run_single(scenario, run_num)
            if self.trace_store:
                self.trace_store.store(trace, passed=True)  # passed set later by evaluator
            traces.append(trace)
        return traces

    def _run_single(self, scenario: Scenario, run_number: int) -> TraceIR:
        """Execute one run of a scenario."""
        system_prompt = scenario.agent.system_prompt or ""
        mock_registry = MockToolRegistry(scenario.agent.tools)

        tools = self._build_tools(scenario)
        options = self._build_options(scenario)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        envelope = ExecutionEnvelope(
            model_name=scenario.agent.model,
            temperature=scenario.agent.temperature,
            seed=scenario.agent.seed,
            num_ctx=scenario.agent.num_ctx,
            tool_schemas=tools or [],
            system_prompt=system_prompt,
        )

        step_records = []
        total_start = time.perf_counter()

        for step_idx, step in enumerate(scenario.steps):
            messages.append({"role": "user", "content": step.user_message})

            step_start = time.perf_counter()
            tool_call_records = []
            assistant_response = ""
            raw_response = {}

            try:
                for _iteration in range(self.MAX_TOOL_ITERATIONS):
                    response = self._chat(scenario.agent.model, messages, tools, options)
                    raw_response = self._response_to_dict(response)
                    msg = response.get("message", response) if isinstance(response, dict) else response.message

                    # Extract content
                    content = msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
                    tool_calls = msg.get("tool_calls", None) if isinstance(msg, dict) else getattr(msg, "tool_calls", None)

                    if not tool_calls:
                        assistant_response = content
                        messages.append({"role": "assistant", "content": content})
                        break

                    # Process tool calls
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": self._serialize_tool_calls(tool_calls),
                    })

                    for tc in tool_calls:
                        tc_start = time.perf_counter()
                        fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                        fn_name = fn.get("name", "") if isinstance(fn, dict) else fn.name
                        fn_args = fn.get("arguments", {}) if isinstance(fn, dict) else fn.arguments
                        if isinstance(fn_args, str):
                            fn_args = json.loads(fn_args)

                        mock_result = mock_registry.call(fn_name, fn_args)
                        tc_ms = (time.perf_counter() - tc_start) * 1000

                        tool_call_records.append(ToolCallRecord(
                            tool_name=fn_name,
                            arguments=fn_args,
                            response=mock_result,
                            latency_ms=tc_ms,
                        ))

                        messages.append({
                            "role": "tool",
                            "content": json.dumps(mock_result),
                        })

                    assistant_response = content
                else:
                    assistant_response = "[Tool call loop exceeded maximum iterations (5)]"

            except Exception as e:
                assistant_response = f"[Error: {e}]"
                raw_response = {"error": str(e)}

            step_ms = (time.perf_counter() - step_start) * 1000
            step_records.append(StepRecord(
                step_index=step_idx,
                user_message=step.user_message,
                assistant_response=assistant_response,
                tool_calls=tool_call_records,
                raw_ollama_response=raw_response,
                latency_ms=step_ms,
            ))

        total_ms = (time.perf_counter() - total_start) * 1000

        trace = TraceIR(
            scenario_name=scenario.name,
            run_number=run_number,
            timestamp=datetime.now(timezone.utc),
            envelope=envelope,
            steps=step_records,
            total_latency_ms=total_ms,
        )
        return finalize_trace(trace)

    def _chat(self, model, messages, tools, options):
        """Call ollama.chat with error handling."""
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "options": options,
            }
            if tools:
                kwargs["tools"] = tools
            return self._client.chat(**kwargs)
        except ollama_client.ResponseError as e:
            if "not found" in str(e).lower():
                raise ModelNotFoundError(
                    f"Model {model} not found. Pull with: ollama pull {model}"
                )
            raise
        except Exception as e:
            if "connect" in str(e).lower() or "refused" in str(e).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.ollama_host}. Start with: ollama serve"
                )
            raise

    def _build_tools(self, scenario: Scenario) -> list[dict] | None:
        if not scenario.agent.tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in scenario.agent.tools
        ]

    def _build_options(self, scenario: Scenario) -> dict:
        opts = {
            "temperature": scenario.agent.temperature,
            "num_ctx": scenario.agent.num_ctx,
        }
        if scenario.agent.seed is not None:
            opts["seed"] = scenario.agent.seed
        return opts

    def _serialize_tool_calls(self, tool_calls) -> list[dict]:
        """Normalize tool calls to dicts for message history."""
        result = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                result.append(tc)
            else:
                fn = tc.function
                result.append({
                    "function": {
                        "name": fn.name,
                        "arguments": fn.arguments if isinstance(fn.arguments, dict) else json.loads(fn.arguments),
                    }
                })
        return result

    def _response_to_dict(self, response) -> dict:
        """Convert ollama response to a plain dict."""
        if isinstance(response, dict):
            return response
        try:
            return response.model_dump()
        except AttributeError:
            return {"raw": str(response)}
