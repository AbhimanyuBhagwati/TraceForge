"""Trace IR: canonical serialization, versioning, and hashing."""

import hashlib
import json

from traceforge.models import TraceIR


def canonical_serialize(trace: TraceIR) -> str:
    """Produce a canonical JSON string for hashing.

    - Sort all dict keys
    - Exclude trace_id (derived from content)
    - Exclude metadata (allows annotation without changing identity)
    - Compact separators
    - Datetimes as ISO 8601
    """
    data = trace.model_dump(exclude={"trace_id", "metadata"})
    data["timestamp"] = trace.timestamp.isoformat()
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def compute_trace_id(trace: TraceIR) -> str:
    """Compute SHA-256 hash of canonical serialization."""
    canonical = canonical_serialize(trace)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def finalize_trace(trace: TraceIR) -> TraceIR:
    """Set the trace_id based on content hash."""
    trace.trace_id = compute_trace_id(trace)
    return trace
