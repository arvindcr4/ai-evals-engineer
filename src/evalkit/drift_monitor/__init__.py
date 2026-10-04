"""Production Drift Monitor.

A nightly job samples a slice of yesterday's live traffic, scores it with
cheap heuristics (and optionally an LLM judge), stores daily metrics in
SQLite and compares them to a rolling baseline with z-score, CUSUM and
Population Stability Index tests, sending alerts before users notice decay.
"""

from evalkit.drift_monitor.alerts import Alert, JsonlSink, StdoutSink, WebhookSink, parse_sink
from evalkit.drift_monitor.detect import DetectorConfig, cusum, detect, psi, zscore
from evalkit.drift_monitor.monitor import DayResult, MonitorConfig, backfill, run_day
from evalkit.drift_monitor.source import JsonlDirSource, hash_sample
from evalkit.drift_monitor.store import MetricStore

__all__ = [
    "Alert",
    "DayResult",
    "DetectorConfig",
    "JsonlDirSource",
    "JsonlSink",
    "MetricStore",
    "MonitorConfig",
    "StdoutSink",
    "WebhookSink",
    "backfill",
    "cusum",
    "detect",
    "hash_sample",
    "parse_sink",
    "psi",
    "run_day",
    "zscore",
]
