"""The nightly job: sample → score → store → detect → alert."""

from __future__ import annotations

from dataclasses import dataclass, field

from evalkit.drift_monitor.alerts import Alert, AlertSink
from evalkit.drift_monitor.detect import DetectorConfig, detect
from evalkit.drift_monitor.scorers import Scorer, aggregate, default_scorers
from evalkit.drift_monitor.source import TrafficSource, date_range, hash_sample
from evalkit.drift_monitor.store import MetricStore


@dataclass
class MonitorConfig:
    rate: float = 0.05
    salt: str = "drift-v1"
    max_samples: int | None = None
    min_samples: int = 20
    detector: DetectorConfig = field(default_factory=DetectorConfig)


@dataclass
class DayResult:
    date: str
    total: int
    sampled: int
    metrics: dict[str, float]
    alerts: list[Alert]
    skipped: str | None = None


def run_day(
    day: str,
    source: TrafficSource,
    store: MetricStore,
    config: MonitorConfig | None = None,
    scorers: list[Scorer] | None = None,
    sinks: list[AlertSink] | None = None,
) -> DayResult:
    """Process one day of traffic and dispatch any alerts to ``sinks``."""
    cfg = config or MonitorConfig()
    sample, total = hash_sample(source.read(day), cfg.rate, cfg.salt, cfg.max_samples)
    if len(sample) < cfg.min_samples:
        return DayResult(
            day,
            total,
            len(sample),
            {},
            [],
            skipped=f"only {len(sample)} samples (< {cfg.min_samples})",
        )
    agg = aggregate(sample, scorers or default_scorers())
    store.save_day(day, total, agg.n, agg.metrics, agg.dists)
    alerts = detect(day, store, cfg.detector)
    store.save_alerts(day, alerts)
    if alerts:
        for sink in sinks or []:
            sink.send(day, alerts)
    return DayResult(day, total, agg.n, agg.metrics, alerts)


def backfill(
    start: str,
    end: str,
    source: TrafficSource,
    store: MetricStore,
    config: MonitorConfig | None = None,
    scorers: list[Scorer] | None = None,
    sinks: list[AlertSink] | None = None,
) -> list[DayResult]:
    """Run ``run_day`` for every date in ``[start, end]`` in order; missing days are skipped."""
    available = set(source.dates())
    results = []
    for day in date_range(start, end):
        if day not in available:
            results.append(DayResult(day, 0, 0, {}, [], skipped="no traffic log"))
            continue
        results.append(run_day(day, source, store, config, scorers, sinks))
    return results
