"""All Pydantic models for TraceForge."""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from enum import Enum
from datetime import datetime


# ============================================================
# SCENARIO MODELS (what the user writes in YAML)
# ============================================================


class ToolDefinition(BaseModel):
    """A tool the agent can call during testing."""
    name: str
    description: str
    parameters: dict  # JSON Schema for tool params
    mock_responses: Optional[list[dict]] = None
    mock_response_file: Optional[str] = None


class ExpectationType(str, Enum):
    TOOL_CALLED = "tool_called"
    TOOL_NOT_CALLED = "tool_not_called"
    TOOL_ARGS_CONTAIN = "tool_args_contain"
    RESPONSE_CONTAINS = "response_contains"
    RESPONSE_NOT_CONTAINS = "response_not_contains"
    RESPONSE_MATCHES_REGEX = "response_matches_regex"
    LLM_JUDGE = "llm_judge"
    LATENCY_UNDER = "latency_under"
    NO_TOOL_ERRORS = "no_tool_errors"
    TOOL_CALL_COUNT = "tool_call_count"


class Expectation(BaseModel):
    """A single assertion about what the agent should do."""
    type: ExpectationType
    tool: Optional[str] = None
    args_contain: Optional[dict[str, str]] = None
    value: Optional[str] = None
    values: Optional[list[str]] = None
    criterion: Optional[str] = None
    max_ms: Optional[int] = None
    count: Optional[int] = None
    operator: Optional[Literal["eq", "gte", "lte"]] = "eq"


class Step(BaseModel):
    """A single turn in a test conversation."""
    user_message: str
    expectations: list[Expectation] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Configuration for the agent under test."""
    model: str = "qwen2.5:7b-instruct"
    system_prompt: Optional[str] = None
    system_prompt_file: Optional[str] = None
    temperature: float = 0.1
    seed: Optional[int] = None
    num_ctx: int = 4096
    tools: list[ToolDefinition] = Field(default_factory=list)


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge."""
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0
    seed: Optional[int] = 42


class MutatorConfig(BaseModel):
    """Configuration for tool argument fuzzing."""
    enabled: bool = False
    mutations_per_tool: int = 5
    mutation_types: list[str] = Field(default_factory=lambda: [
        "numeric_extreme", "missing_required", "type_swap",
        "empty_string", "null_injection", "boundary",
    ])


class Scenario(BaseModel):
    """A complete test scenario."""
    name: str
    description: Optional[str] = None
    agent: AgentConfig
    judge: Optional[JudgeConfig] = None
    steps: list[Step]
    runs: int = 5
    tags: list[str] = Field(default_factory=list)
    mutator: Optional[MutatorConfig] = None


# ============================================================
# TRACE IR MODELS (the reproducibility backbone)
# ============================================================

TRACE_IR_VERSION = "1.0.0"


class ExecutionEnvelope(BaseModel):
    """Everything needed to reproduce an execution."""
    model_name: str
    temperature: float
    seed: Optional[int] = None
    num_ctx: int
    quantization_level: Optional[str] = None
    tool_schemas: list[dict]
    system_prompt: str


class ToolCallRecord(BaseModel):
    """A captured tool invocation within the trace."""
    tool_name: str
    arguments: dict
    response: dict
    latency_ms: float


class StepRecord(BaseModel):
    """One conversation turn fully recorded."""
    step_index: int
    user_message: str
    assistant_response: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    raw_ollama_response: dict
    latency_ms: float
    token_count: Optional[int] = None


class TraceIR(BaseModel):
    """
    The canonical, immutable, content-addressed trace of one agent run.
    This is the fundamental unit of TraceForge — everything else operates on these.
    """
    version: str = TRACE_IR_VERSION
    trace_id: str = ""
    scenario_name: str
    run_number: int
    timestamp: datetime
    envelope: ExecutionEnvelope
    steps: list[StepRecord]
    total_latency_ms: float
    metadata: dict = Field(default_factory=dict)


# ============================================================
# RESULT MODELS (what the evaluator produces)
# ============================================================


class ExpectationResult(BaseModel):
    """Result of evaluating a single expectation."""
    expectation: Expectation
    passed: bool
    message: str
    details: Optional[dict] = None


class StepResult(BaseModel):
    """Results for all expectations in a step."""
    step_index: int
    user_message: str
    results: list[ExpectationResult]
    all_passed: bool


class RunResult(BaseModel):
    """Results for a complete scenario run."""
    scenario_name: str
    run_number: int
    trace_id: str
    step_results: list[StepResult]
    passed: bool
    total_latency_ms: float
    timestamp: datetime


class ScenarioResult(BaseModel):
    """Aggregated results across all runs of a scenario."""
    scenario_name: str
    description: Optional[str] = None
    total_runs: int
    passed_runs: int
    failed_runs: int
    pass_rate: float
    consistency_score: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    run_results: list[RunResult]
    per_step_pass_rates: list[float]
    per_expectation_pass_rates: dict[str, float]
    tags: list[str]


class ProbeReport(BaseModel):
    """The final output of a TraceForge run."""
    timestamp: datetime
    model: str
    total_scenarios: int
    total_runs: int
    overall_pass_rate: float
    scenario_results: list[ScenarioResult]
    regression_warnings: list[str] = Field(default_factory=list)


# ============================================================
# FUZZING RESULT MODELS
# ============================================================


class MutationRecord(BaseModel):
    """Record of a single mutation applied to tool arguments."""
    original_args: dict
    mutated_args: dict
    mutation_type: str
    mutation_description: str


class FuzzResult(BaseModel):
    """Result of fuzzing one tool call in one step."""
    step_index: int
    tool_name: str
    mutation: MutationRecord
    trace_id: str
    original_passed: bool
    mutated_passed: bool
    broke_agent: bool


class FuzzReport(BaseModel):
    """Aggregated fuzzing results for a scenario."""
    scenario_name: str
    total_mutations: int
    total_breaks: int
    robustness_score: float
    results: list[FuzzResult]
    by_mutation_type: dict[str, float]
    by_tool: dict[str, float]


# ============================================================
# MIN-REPRO RESULT MODELS
# ============================================================


class MinReproResult(BaseModel):
    """Result of delta debugging on a failing trace."""
    original_trace_id: str
    original_step_count: int
    original_tool_call_count: int
    minimized_trace_id: str
    minimized_step_count: int
    minimized_tool_call_count: int
    reduction_ratio: float
    iterations_taken: int
    failure_still_reproduces: bool
    minimized_steps: list[int]


# ============================================================
# CAUSAL ATTRIBUTION MODELS
# ============================================================


class InterventionType(str, Enum):
    """Categories of counterfactual interventions."""
    TOOL_OUTPUT_FORMAT = "tool_output_format"
    TOOL_OUTPUT_VALUE = "tool_output_value"
    TOOL_OUTPUT_FIELDS = "tool_output_fields"
    CONTEXT_TRUNCATION = "context_truncation"
    SYSTEM_PROMPT_CLAUSE = "system_prompt_clause"
    TOOL_SCHEMA_CHANGE = "tool_schema_change"
    MESSAGE_ORDER = "message_order"


class Intervention(BaseModel):
    """A single counterfactual change applied to a scenario."""
    intervention_type: InterventionType
    description: str
    target_step: Optional[int] = None
    target_tool: Optional[str] = None
    target_field: Optional[str] = None
    original_value: Optional[Any] = None
    modified_value: Optional[Any] = None


class CounterfactualResult(BaseModel):
    """Result of one counterfactual experiment."""
    intervention: Intervention
    original_passed: bool
    counterfactual_passed: bool
    flipped: bool
    trace_id: str
    confidence: float


class CausalReport(BaseModel):
    """Full causal attribution report for a failing scenario."""
    scenario_name: str
    failing_trace_id: str
    failing_step: int
    total_interventions: int
    total_flips: int
    interventions: list[CounterfactualResult]
    causal_factors: list[dict]
    summary: str


# ============================================================
# INVARIANT MINING MODELS
# ============================================================


class InvariantType(str, Enum):
    """Categories of discoverable behavioral invariants."""
    TOOL_ORDER = "tool_order"
    TOOL_ALWAYS_CALLED = "tool_always_called"
    TOOL_NEVER_CALLED = "tool_never_called"
    TOOL_CALL_COUNT = "tool_call_count"
    ARG_RANGE = "arg_range"
    ARG_PATTERN = "arg_pattern"
    ARG_DEPENDENCY = "arg_dependency"
    RESPONSE_LENGTH = "response_length"
    RESPONSE_PATTERN = "response_pattern"
    STEP_LATENCY = "step_latency"
    TOOL_IDEMPOTENCY = "tool_idempotency"
    CONDITIONAL = "conditional"


class Invariant(BaseModel):
    """A discovered behavioral invariant."""
    invariant_type: InvariantType
    description: str
    formal: str
    confidence: float
    support: int
    violations: int
    step_index: Optional[int] = None
    tool_name: Optional[str] = None
    details: dict = Field(default_factory=dict)


class InvariantReport(BaseModel):
    """Report of all mined invariants."""
    total_traces_analyzed: int
    passing_traces: int
    failing_traces: int
    invariants_discovered: int
    invariants: list[Invariant]
    discriminating_invariants: list[Invariant]
    suggested_expectations: list[dict]
