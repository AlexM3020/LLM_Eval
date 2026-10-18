"""Run / prediction / score persistence.

Plain SQLite via the stdlib for now — zero setup, and the schema is the point.
Day 5+: point DATABASE_URL at Postgres and swap this for SQLAlchemy; runs are
append-only and immutable, which is what makes experiment comparison trustworthy.
"""
from __future__ import annotations

import datetime
import json
import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  job_json TEXT NOT NULL,          -- full EvalJob: model, task, dataset, prompt_version, ...
  metrics_json TEXT NOT NULL,      -- aggregate metrics per evaluator
  total_cost_usd REAL,
  total_latency_ms REAL
);
CREATE TABLE IF NOT EXISTS predictions (
  run_id TEXT NOT NULL,
  example_id TEXT NOT NULL,
  raw_output TEXT,
  parsed TEXT,
  latency_ms REAL,
  cost_usd REAL
);
CREATE TABLE IF NOT EXISTS scores (
  run_id TEXT NOT NULL,
  evaluator TEXT NOT NULL,
  example_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL
);
"""


class Store:
    def __init__(self, url: str = "sqlite:///./eval_service.db"):
        if not url.startswith("sqlite:///"):
            raise ValueError("Day 5: swap in SQLAlchemy/Postgres for non-SQLite URLs")
        self.conn = sqlite3.connect(url.removeprefix("sqlite:///"), check_same_thread=False)
        self.conn.executescript(SCHEMA_SQL)

    def save_run(self, run_record: dict, predictions, reports) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?)",
            (
                run_record["id"],
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                json.dumps(run_record["job"], default=str),
                json.dumps(run_record["metrics"]),
                run_record["total_cost_usd"],
                run_record["total_latency_ms"],
            ),
        )
        for p in predictions:
            cur.execute(
                "INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                (run_record["id"], p.example_id, p.raw_output, p.parsed, p.latency_ms, p.cost_usd),
            )
        for name, report in reports.items():
            for row in report.per_example:
                for metric, value in row.items():
                    if metric == "example_id" or value is None:
                        continue
                    cur.execute(
                        "INSERT INTO scores VALUES (?,?,?,?,?)",
                        (run_record["id"], name, row["example_id"], metric, float(value)),
                    )
        self.conn.commit()

    def list_runs(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT id, created_at, job_json, metrics_json FROM runs ORDER BY created_at DESC"
        )
        return [
            {"id": r[0], "created_at": r[1], "job": json.loads(r[2]), "metrics": json.loads(r[3])}
            for r in cur.fetchall()
        ]

    def get_run(self, run_id: str) -> dict | None:
        cur = self.conn.execute(
            "SELECT id, created_at, job_json, metrics_json, total_cost_usd, total_latency_ms "
            "FROM runs WHERE id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0], "created_at": row[1], "job": json.loads(row[2]),
            "metrics": json.loads(row[3]), "total_cost_usd": row[4], "total_latency_ms": row[5],
        }

    def compare_runs(self, run_a: str, run_b: str) -> dict:
        """Regression report: metric deltas between two runs (e.g. prompt v1 vs v2).

        TODO(Day 6): also return per-example diffs joined on example_id so you can
        see WHICH examples regressed, not just the aggregate.
        """
        a, b = self.get_run(run_a), self.get_run(run_b)
        if not a or not b:
            raise KeyError("unknown run id")
        deltas = {}
        for evaluator, metrics in b["metrics"].items():
            for metric, value in metrics.items():
                baseline = a["metrics"].get(evaluator, {}).get(metric)
                if baseline is not None:
                    deltas[f"{evaluator}.{metric}"] = {"before": baseline, "after": value, "delta": value - baseline}
        return {"run_a": run_a, "run_b": run_b, "deltas": deltas}
