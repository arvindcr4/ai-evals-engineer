"""SQLite metrics store keyed by date and metric name.

Re-running a date replaces that date's rows, so nightly jobs and backfills
are idempotent.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    date TEXT PRIMARY KEY, total INTEGER, sampled INTEGER, created_at REAL
);
CREATE TABLE IF NOT EXISTS metrics (
    date TEXT, metric TEXT, value REAL, PRIMARY KEY (date, metric)
);
CREATE TABLE IF NOT EXISTS dists (
    date TEXT, name TEXT, counts TEXT, PRIMARY KEY (date, name)
);
CREATE TABLE IF NOT EXISTS alerts (
    date TEXT, metric TEXT, method TEXT, severity TEXT, value REAL,
    baseline REAL, score REAL, message TEXT
);
"""


class MetricStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def save_day(
        self,
        day: str,
        total: int,
        sampled: int,
        metrics: dict[str, float],
        dists: dict[str, dict[str, int]],
    ) -> None:
        with self.conn:
            for table in ("metrics", "dists", "alerts"):
                self.conn.execute(f"DELETE FROM {table} WHERE date = ?", (day,))
            self.conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)",
                (day, total, sampled, time.time()),
            )
            self.conn.executemany(
                "INSERT INTO metrics VALUES (?, ?, ?)", [(day, k, v) for k, v in metrics.items()]
            )
            self.conn.executemany(
                "INSERT INTO dists VALUES (?, ?, ?)",
                [(day, k, json.dumps(v)) for k, v in dists.items()],
            )

    def save_alerts(self, day: str, alerts: list) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM alerts WHERE date = ?", (day,))
            self.conn.executemany(
                "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        a.date,
                        a.metric,
                        a.method,
                        a.severity,
                        a.value,
                        a.baseline,
                        a.score,
                        a.message,
                    )
                    for a in alerts
                ],
            )

    def dates(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT date FROM runs ORDER BY date")]

    def run_info(self, day: str) -> tuple[int, int] | None:
        row = self.conn.execute("SELECT total, sampled FROM runs WHERE date = ?", (day,)).fetchone()
        return (row[0], row[1]) if row else None

    def series(self, metric: str, dates: list[str] | None = None) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT date, value FROM metrics WHERE metric = ? ORDER BY date", (metric,)
        )
        out = dict(rows.fetchall())
        return {d: out[d] for d in dates if d in out} if dates is not None else out

    def metric_names(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT metric FROM metrics ORDER BY 1")]

    def dist(self, name: str, day: str) -> dict[str, int] | None:
        row = self.conn.execute(
            "SELECT counts FROM dists WHERE name = ? AND date = ?", (name, day)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def dist_names(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT name FROM dists ORDER BY 1")]

    def alerted_dates(self, severity: str = "critical") -> set[str]:
        rows = self.conn.execute("SELECT DISTINCT date FROM alerts WHERE severity = ?", (severity,))
        return {r[0] for r in rows}

    def alerts(self, day: str | None = None) -> list[dict]:
        q = "SELECT date, metric, method, severity, value, baseline, score, message FROM alerts"
        args: tuple = ()
        if day:
            q, args = q + " WHERE date = ?", (day,)
        cols = ["date", "metric", "method", "severity", "value", "baseline", "score", "message"]
        return [dict(zip(cols, r)) for r in self.conn.execute(q + " ORDER BY date, metric", args)]
