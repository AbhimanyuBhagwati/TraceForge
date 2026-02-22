"""Content-addressed trace store: SHA-256 objects + SQLite index."""

import gzip
import sqlite3
from pathlib import Path
from typing import Optional

from traceforge.models import TraceIR
from traceforge.trace_ir import finalize_trace


class TraceStore:
    """Content-addressed storage for traces with SQLite metadata index."""

    def __init__(self, base_dir: str = ".traceforge"):
        self.base_dir = Path(base_dir)
        self.traces_dir = self.base_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "history.db"
        self._init_db()

    def _init_db(self):
        """Create SQLite tables for trace index."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trace_index (
                    trace_id TEXT PRIMARY KEY,
                    scenario_name TEXT NOT NULL,
                    run_number INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    passed BOOLEAN,
                    total_latency_ms REAL NOT NULL,
                    step_count INTEGER NOT NULL,
                    tool_call_count INTEGER NOT NULL,
                    compressed_size INTEGER NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trace_scenario ON trace_index(scenario_name)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def store(self, trace: TraceIR, passed: bool) -> str:
        """Store a trace. Returns trace_id. Deduplicates by hash."""
        trace = finalize_trace(trace)
        blob_path = self.traces_dir / f"{trace.trace_id}.json.gz"

        if not blob_path.exists():
            data = trace.model_dump_json()
            with gzip.open(blob_path, "wt") as f:
                f.write(data)

        compressed_size = blob_path.stat().st_size
        tool_call_count = sum(len(s.tool_calls) for s in trace.steps)

        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trace_index
                   (trace_id, scenario_name, run_number, model, timestamp,
                    passed, total_latency_ms, step_count, tool_call_count, compressed_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace.trace_id,
                    trace.scenario_name,
                    trace.run_number,
                    trace.envelope.model_name,
                    trace.timestamp.isoformat(),
                    passed,
                    trace.total_latency_ms,
                    len(trace.steps),
                    tool_call_count,
                    compressed_size,
                ),
            )

        return trace.trace_id

    def resolve_id(self, prefix: str) -> str:
        """Resolve a trace ID prefix to a full ID."""
        if (self.traces_dir / f"{prefix}.json.gz").exists():
            return prefix
        matches = list(self.traces_dir.glob(f"{prefix}*.json.gz"))
        if len(matches) == 1:
            return matches[0].stem.replace(".json", "")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous trace ID prefix '{prefix}' matches {len(matches)} traces"
            )
        raise FileNotFoundError(f"Trace not found: {prefix}")

    def load(self, trace_id: str) -> TraceIR:
        """Load a trace by its ID (or unique prefix)."""
        full_id = self.resolve_id(trace_id)
        blob_path = self.traces_dir / f"{full_id}.json.gz"
        with gzip.open(blob_path, "rt") as f:
            return TraceIR.model_validate_json(f.read())

    def list_traces(
        self,
        scenario_name: Optional[str] = None,
        passed: Optional[bool] = None,
    ) -> list[dict]:
        """Query trace index with optional filters."""
        query = "SELECT * FROM trace_index WHERE 1=1"
        params: list = []
        if scenario_name is not None:
            query += " AND scenario_name = ?"
            params.append(scenario_name)
        if passed is not None:
            query += " AND passed = ?"
            params.append(passed)
        query += " ORDER BY timestamp DESC"

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_latest(self, scenario_name: str, model: str) -> Optional[dict]:
        """Get the most recent trace metadata for a scenario+model combo."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT * FROM trace_index
                   WHERE scenario_name = ? AND model = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (scenario_name, model),
            ).fetchone()
            return dict(row) if row else None
