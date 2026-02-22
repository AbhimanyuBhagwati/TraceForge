"""SQLite history store for regression tracking."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from traceforge.models import ScenarioResult


class HistoryStore:
    """Tracks scenario results over time for regression detection."""

    def __init__(self, base_dir: str = ".traceforge"):
        self.db_path = Path(base_dir) / "history.db"
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    total_runs INTEGER NOT NULL,
                    passed_runs INTEGER NOT NULL,
                    pass_rate REAL NOT NULL,
                    consistency_score REAL NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    trace_ids TEXT NOT NULL,
                    full_report_json TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scenario ON runs(scenario_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def record(self, result: ScenarioResult, model: str):
        """Record a scenario result."""
        trace_ids = [rr.trace_id for rr in result.run_results]
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO runs
                   (timestamp, scenario_name, model, total_runs, passed_runs,
                    pass_rate, consistency_score, avg_latency_ms, trace_ids, full_report_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.run_results[0].timestamp.isoformat() if result.run_results else "",
                    result.scenario_name,
                    model,
                    result.total_runs,
                    result.passed_runs,
                    result.pass_rate,
                    result.consistency_score,
                    result.avg_latency_ms,
                    json.dumps(trace_ids),
                    result.model_dump_json(),
                ),
            )

    def get_previous(self, scenario_name: str, model: str) -> Optional[dict]:
        """Get the most recent recorded result for regression comparison."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT * FROM runs
                   WHERE scenario_name = ? AND model = ?
                   ORDER BY id DESC LIMIT 1""",
                (scenario_name, model),
            ).fetchone()
            return dict(row) if row else None

    def get_history(
        self, scenario_name: Optional[str] = None, limit: int = 20
    ) -> list[dict]:
        """Get run history, optionally filtered by scenario."""
        query = "SELECT id, timestamp, scenario_name, model, total_runs, passed_runs, pass_rate, consistency_score, avg_latency_ms FROM runs"
        params: list = []
        if scenario_name:
            query += " WHERE scenario_name = ?"
            params.append(scenario_name)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def check_regression(
        self, result: ScenarioResult, model: str
    ) -> Optional[str]:
        """Check if the current result is a regression from the previous run."""
        prev = self.get_previous(result.scenario_name, model)
        if not prev:
            return None

        prev_rate = prev["pass_rate"]
        curr_rate = result.pass_rate

        if curr_rate < prev_rate - 0.01:  # Allow 1% tolerance
            return (
                f"REGRESSION: {result.scenario_name} dropped from "
                f"{prev_rate:.0%} -> {curr_rate:.0%}"
            )
        return None
