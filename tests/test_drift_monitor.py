import json

import httpx
import pytest

from evalkit.core.llm import MockLLM
from evalkit.drift_monitor import (
    DetectorConfig,
    JsonlDirSource,
    JsonlSink,
    MetricStore,
    MonitorConfig,
    WebhookSink,
    backfill,
    cusum,
    detect,
    hash_sample,
    parse_sink,
    psi,
    run_day,
    zscore,
)
from evalkit.drift_monitor.alerts import Alert, StdoutSink, slack_payload
from evalkit.drift_monitor.generate import generate, severity
from evalkit.drift_monitor.report import write_report
from evalkit.drift_monitor.scorers import (
    JudgeScorer,
    LengthScorer,
    RefusalScorer,
    ToolScorer,
    aggregate,
    default_scorers,
)
from evalkit.drift_monitor.source import date_range


@pytest.fixture(scope="module")
def logs(tmp_path_factory):
    out = tmp_path_factory.mktemp("logs")
    generate(out, start="2026-09-01", days=30, per_day=1500, decay_day=20, seed=3)
    return out


def test_hash_sample_is_deterministic_rate_and_cap():
    recs = [{"request_id": f"r{i}"} for i in range(20000)]
    s1, total = hash_sample(recs, 0.05)
    s2, _ = hash_sample(list(reversed(recs)), 0.05)
    assert total == 20000
    assert {r["request_id"] for r in s1} == {r["request_id"] for r in s2}
    assert 900 < len(s1) < 1100
    capped, _ = hash_sample(recs, 0.05, cap=100)
    assert len(capped) == 100 and capped == s1[:100]
    assert hash_sample(recs, 0.0)[0] == []


def test_scorers_on_handcrafted_records():
    refusal = {"output": "I'm sorry, but I can't help with that.", "tool_calls": []}
    good = {
        "output": "x" * 300,
        "tool_calls": [{"name": "sql", "ok": True}, {"name": "search", "ok": False}],
    }
    assert RefusalScorer().score(refusal)["refusal"] == 1.0
    assert RefusalScorer().score(good)["refusal"] == 0.0
    assert LengthScorer().score({"output": "  "})["empty"] == 1.0
    assert ToolScorer().score(good)["tool_error"] == 0.5
    assert ToolScorer().score(refusal)["tool_error"] is None
    j = JudgeScorer()
    assert j.score(refusal)["judge_score"] == 0.0
    assert j.score({"output": "y" * 400, "tool_calls": []})["judge_score"] == 1.0
    unparseable = JudgeScorer(MockLLM(responder=lambda m: "great answer"))
    assert unparseable.score(good)["judge_score"] is None


def test_aggregate_metrics_and_distributions():
    recs = [
        {"output": "", "tool_calls": [], "latency_s": 1.0, "cost_usd": 0.01},
        {
            "output": "I'm sorry, I can't help",
            "tool_calls": [{"name": "sql", "ok": False}],
            "latency_s": 2.0,
            "cost_usd": 0.01,
        },
        {
            "output": "a" * 600,
            "tool_calls": [{"name": "sql", "ok": True}],
            "latency_s": 3.0,
            "cost_usd": 0.02,
        },
    ]
    agg = aggregate(recs, default_scorers())
    assert agg.n == 3
    assert agg.metrics["empty_rate"] == pytest.approx(1 / 3)
    assert agg.metrics["refusal_rate"] == pytest.approx(1 / 3)
    assert agg.metrics["tool_error_rate"] == pytest.approx(0.5)
    assert agg.metrics["latency_p50"] == 2.0
    assert agg.dists["answer_length"]["empty"] == 1
    assert agg.dists["answer_length"]["500-999"] == 1
    assert agg.dists["tool_mix"] == {"sql": 2}


def test_statistics():
    base = [0.03, 0.031, 0.029, 0.03, 0.032]
    assert zscore(0.03, base) == pytest.approx(0.0, abs=0.5)
    assert zscore(0.2, base) > 10
    assert zscore(0.2, base, abs_floor=0.05) < zscore(0.2, base)
    assert cusum([0.03] * 7, base) == 0.0
    assert cusum([0.035, 0.04, 0.045, 0.05, 0.055], base) > 5
    assert cusum([0.01] * 5, base, direction=1) == 0.0
    same = {"a": 50, "b": 30, "c": 20}
    assert psi(same, same) == pytest.approx(0.0)
    assert psi(same, {"a": 10, "b": 10, "c": 10, "new": 70}) > 1.0
    assert psi({}, same) == 0.0


def test_no_alerts_without_enough_baseline(tmp_path):
    store = MetricStore(tmp_path / "m.sqlite")
    for i, day in enumerate(date_range("2026-01-01", "2026-01-03")):
        store.save_day(day, 100, 50, {"refusal_rate": 0.03 + 0.3 * (i == 2)}, {})
    assert detect("2026-01-03", store) == []


def test_detects_only_bad_direction(tmp_path):
    store = MetricStore(tmp_path / "m.sqlite")
    days = date_range("2026-01-01", "2026-01-10")
    for d in days[:-1]:
        store.save_day(d, 2000, 100, {"refusal_rate": 0.03, "answer_chars_mean": 600.0}, {})
    store.save_day(days[-1], 2000, 100, {"refusal_rate": 0.0, "answer_chars_mean": 900.0}, {})
    assert detect(days[-1], store) == []
    store.save_day(days[-1], 2000, 100, {"refusal_rate": 0.2, "answer_chars_mean": 300.0}, {})
    alerts = detect(days[-1], store)
    assert {a.metric for a in alerts} == {"refusal_rate", "answer_chars_mean"}
    assert all(a.severity == "critical" for a in alerts)


def test_rate_floor_suppresses_tiny_rate_noise(tmp_path):
    store = MetricStore(tmp_path / "m.sqlite")
    days = date_range("2026-01-01", "2026-01-10")
    for d in days[:-1]:
        store.save_day(d, 2000, 100, {"empty_rate": 0.001}, {})
    store.save_day(days[-1], 2000, 100, {"empty_rate": 0.02}, {})
    assert detect(days[-1], store) == []


def test_end_to_end_backfill_flags_decay_not_healthy_days(logs, tmp_path):
    source = JsonlDirSource(logs)
    assert len(source.dates()) == 30
    store = MetricStore(tmp_path / "drift.sqlite")
    sink = JsonlSink(tmp_path / "alerts.jsonl")
    hook = WebhookSink()
    results = backfill(
        "2026-09-01", "2026-09-30", source, store, MonitorConfig(rate=0.05), sinks=[sink, hook]
    )
    assert len(results) == 30
    crit_days = {r.date for r in results if any(a.severity == "critical" for a in r.alerts)}
    healthy = set(date_range("2026-09-01", "2026-09-20"))
    decayed = set(date_range("2026-09-23", "2026-09-30"))
    assert not crit_days & healthy
    assert decayed <= crit_days
    metrics = {a["metric"] for a in store.alerts()}
    assert {"refusal_rate", "tool_error_rate", "answer_chars_mean", "tool_mix"} <= metrics
    lines = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert len(lines) == len(store.alerts())
    assert hook.sent and hook.sent[-1]["blocks"][0]["type"] == "header"
    paths = write_report(store, tmp_path / "rep")
    assert "<svg" in paths["html"].read_text()
    assert "Drift monitor report" in paths["md"].read_text()


def test_run_day_is_idempotent_and_skips_thin_samples(logs, tmp_path):
    source = JsonlDirSource(logs)
    store = MetricStore(tmp_path / "m.sqlite")
    r1 = run_day("2026-09-05", source, store, MonitorConfig(rate=0.05))
    r2 = run_day("2026-09-05", source, store, MonitorConfig(rate=0.05))
    assert r1.metrics == r2.metrics and store.dates() == ["2026-09-05"]
    thin = run_day("2026-09-06", source, store, MonitorConfig(rate=0.001))
    assert thin.skipped and "2026-09-06" not in store.dates()
    with pytest.raises(FileNotFoundError):
        run_day("2027-01-01", source, store)


def test_sinks_and_parsing(capsys, tmp_path, monkeypatch):
    a = Alert("2026-09-21", "refusal_rate", "zscore", "critical", 0.2, 0.03, 9.0, "boom")
    assert isinstance(parse_sink("stdout"), StdoutSink)
    assert isinstance(parse_sink(f"jsonl:{tmp_path}/a.jsonl"), JsonlSink)
    assert parse_sink("webhook:https://hooks.example/x").url == "https://hooks.example/x"
    with pytest.raises(ValueError):
        parse_sink("pagerduty")
    StdoutSink().send(a.date, [a])
    assert "CRITICAL" in capsys.readouterr().out
    payload = slack_payload(a.date, [a])
    assert "1 critical" in payload["text"]

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: client.post(url, **kw))
    WebhookSink("https://hooks.example/x").send(a.date, [a])
    assert seen["body"]["text"] == payload["text"]


def test_generator_severity_ramp():
    assert severity(19, 20) == 0.0
    assert 0 < severity(20, 20) < severity(22, 20) < 1.0
    assert severity(40, 20) == 1.0


def test_detector_config_thresholds(tmp_path):
    store = MetricStore(tmp_path / "m.sqlite")
    days = date_range("2026-01-01", "2026-01-08")
    for i, d in enumerate(days[:-1]):
        store.save_day(d, 2000, 100, {"latency_p50": 1.7 + 0.02 * (i % 3)}, {})
    store.save_day(days[-1], 2000, 100, {"latency_p50": 2.0}, {})
    assert detect(days[-1], store, DetectorConfig(z_warn=3, z_crit=4))
    assert not detect(days[-1], store, DetectorConfig(z_warn=50, z_crit=60, cusum_h=1e9))
