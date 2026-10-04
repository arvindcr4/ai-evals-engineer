"""`evalkit drift-monitor generate|run|backfill|report`."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from evalkit.core.llm import get_llm
from evalkit.drift_monitor.alerts import parse_sink
from evalkit.drift_monitor.detect import DetectorConfig
from evalkit.drift_monitor.generate import generate
from evalkit.drift_monitor.monitor import DayResult, MonitorConfig, backfill, run_day
from evalkit.drift_monitor.report import render_markdown, write_report
from evalkit.drift_monitor.scorers import JudgeScorer, default_scorers
from evalkit.drift_monitor.source import JsonlDirSource
from evalkit.drift_monitor.store import MetricStore


def _monitor_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--logs", required=True, help="traffic log dir (YYYY-MM-DD.jsonl files)")
    p.add_argument("--db", default="drift.sqlite", help="SQLite metrics store")
    p.add_argument("--rate", type=float, default=0.05, help="sample rate (default 0.05)")
    p.add_argument("--max-samples", type=int, default=None, help="cap samples per day")
    p.add_argument("--baseline-days", type=int, default=14)
    p.add_argument(
        "--sink",
        action="append",
        default=None,
        help="stdout | jsonl:<path> | webhook[:<url>] (repeatable; default stdout)",
    )
    p.add_argument(
        "--judge", default=None, help="add an LLM-judge scorer: 'mock' (offline) or any --llm spec"
    )


def _setup(a: argparse.Namespace):
    scorers = default_scorers()
    if a.judge:
        scorers.append(JudgeScorer() if a.judge == "mock" else JudgeScorer(get_llm(a.judge)))
    cfg = MonitorConfig(
        rate=a.rate,
        max_samples=a.max_samples,
        detector=DetectorConfig(baseline_days=a.baseline_days),
    )
    sinks = [parse_sink(s) for s in (a.sink or ["stdout"])]
    return JsonlDirSource(a.logs), MetricStore(a.db), cfg, scorers, sinks


def _summary(r: DayResult) -> str:
    if r.skipped:
        return f"{r.date}: skipped ({r.skipped})"
    crit = sum(x.severity == "critical" for x in r.alerts)
    status = "OK" if not r.alerts else f"{crit} critical / {len(r.alerts) - crit} warning"
    return f"{r.date}: sampled {r.sampled}/{r.total} -> {status}"


def _generate(a: argparse.Namespace) -> int:
    paths = generate(a.out, a.start, a.days, a.per_day, a.decay_day, a.seed)
    print(
        f"wrote {len(paths)} daily logs ({a.per_day}/day) to {a.out}; decay from day {a.decay_day}"
    )
    return 0


def _run(a: argparse.Namespace) -> int:
    source, store, cfg, scorers, sinks = _setup(a)
    day = a.date or (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    result = run_day(day, source, store, cfg, scorers, sinks)
    print(_summary(result))
    store.close()
    return 1 if a.fail_on_critical and any(x.severity == "critical" for x in result.alerts) else 0


def _backfill(a: argparse.Namespace) -> int:
    source, store, cfg, scorers, sinks = _setup(a)
    dates = source.dates()
    start, end = a.start or dates[0], a.end or dates[-1]
    for r in backfill(start, end, source, store, cfg, scorers, sinks):
        print(_summary(r))
    store.close()
    return 0


def _report(a: argparse.Namespace) -> int:
    store = MetricStore(a.db)
    paths = write_report(store, a.out)
    print(render_markdown(store))
    print("wrote " + ", ".join(str(p) for p in paths.values()))
    store.close()
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("drift-monitor", help="09 production drift monitor")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write synthetic daily traffic logs with a decay")
    g.add_argument("--out", required=True)
    g.add_argument("--start", default="2026-09-01")
    g.add_argument("--days", type=int, default=30)
    g.add_argument("--per-day", type=int, default=2000)
    g.add_argument("--decay-day", type=int, default=20)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=_generate)

    r = sub.add_parser("run", help="nightly job for one date (default: yesterday)")
    _monitor_args(r)
    r.add_argument("--date", default=None, help="YYYY-MM-DD")
    r.add_argument("--fail-on-critical", action="store_true", help="exit 1 on critical alerts")
    r.set_defaults(func=_run)

    b = sub.add_parser("backfill", help="run every date in a range, in order")
    _monitor_args(b)
    b.add_argument("--from", dest="start", default=None, help="first date (default: earliest log)")
    b.add_argument("--to", dest="end", default=None, help="last date (default: latest log)")
    b.set_defaults(func=_backfill)

    rep = sub.add_parser("report", help="Markdown + HTML trend report from the store")
    rep.add_argument("--db", default="drift.sqlite")
    rep.add_argument("--out", default="drift_report")
    rep.set_defaults(func=_report)
