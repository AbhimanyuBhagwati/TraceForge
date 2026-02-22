"""Shared fixtures for TraceForge tests."""

import pytest
from datetime import datetime, timezone

from traceforge.models import (
    AgentConfig,
    Expectation,
    ExpectationType,
    ExecutionEnvelope,
    Scenario,
    Step,
    StepRecord,
    ToolCallRecord,
    ToolDefinition,
    TraceIR,
)


@pytest.fixture
def sample_tool_def():
    return ToolDefinition(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
            },
            "required": ["expression"],
        },
        mock_responses=[{"result": 42}, {"result": 3.14}],
    )


@pytest.fixture
def sample_scenario(sample_tool_def):
    return Scenario(
        name="test_calculator",
        description="Test calculator agent",
        agent=AgentConfig(
            model="qwen2.5:7b-instruct",
            system_prompt="You are a calculator assistant.",
            temperature=0.1,
            seed=42,
            tools=[sample_tool_def],
        ),
        steps=[
            Step(
                user_message="What is 6 times 7?",
                expectations=[
                    Expectation(type=ExpectationType.TOOL_CALLED, tool="calculate"),
                    Expectation(
                        type=ExpectationType.RESPONSE_CONTAINS, values=["42"]
                    ),
                ],
            ),
        ],
        runs=3,
        tags=["math", "tools"],
    )


@pytest.fixture
def sample_trace(sample_tool_def):
    return TraceIR(
        version="1.0.0",
        scenario_name="test_calculator",
        run_number=1,
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        envelope=ExecutionEnvelope(
            model_name="qwen2.5:7b-instruct",
            temperature=0.1,
            seed=42,
            num_ctx=4096,
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Perform a mathematical calculation",
                        "parameters": sample_tool_def.parameters,
                    },
                }
            ],
            system_prompt="You are a calculator assistant.",
        ),
        steps=[
            StepRecord(
                step_index=0,
                user_message="What is 6 times 7?",
                assistant_response="The answer is 42.",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="calculate",
                        arguments={"expression": "6 * 7"},
                        response={"result": 42},
                        latency_ms=5.0,
                    )
                ],
                raw_ollama_response={"model": "qwen2.5:7b-instruct", "message": {"content": "The answer is 42."}},
                latency_ms=1200.0,
                token_count=50,
            )
        ],
        total_latency_ms=1200.0,
    )
