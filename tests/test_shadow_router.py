import json
import threading
import time
import warnings

import pytest

from evalkit.core.llm import Completion, MockLLM
from evalkit.shadow_router import (
    ShadowConfig,
    ShadowRouter,
    build_report,
    create_app,
    sample_key,
    should_shadow,
)
from evalkit.shadow_router.demo import DemoModel, resolve_llm
from evalkit.shadow_router.report import (
    default_judge,
    judge_pair,
    render_html,
    render_markdown,
    token_jaccard,
    wilson,
    write_report,
)
from evalkit.shadow_router.simulate import simulate

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient


def _msgs(q: str) -> list[dict]:
    return [{"role": "user", "content": q}]


def test_sampling_is_deterministic_and_close_to_rate():
    keys = [f"user-{i}" for i in range(20000)]
    picked = [should_shadow(k, 0.05) for k in keys]
    assert picked == [should_shadow(k, 0.05) for k in keys]
    assert 0.04 < sum(picked) / len(keys) < 0.06
    assert not any(should_shadow(k, 0.0) for k in keys[:100])
    assert all(should_shadow(k, 1.0) for k in keys[:100])
    assert [should_shadow(k, 0.05, "other-salt") for k in keys] != picked


def test_sample_key_priority_is_sticky_per_user():
    body = {"user": "alice", "messages": _msgs("x")}
    assert sample_key(body, {"X-User-Id": "bob", "x-request-id": "r1"}) == "bob"
    assert sample_key(body, {"x-request-id": "r1"}) == "alice"
    assert sample_key({"messages": _msgs("x")}, {"x-request-id": "r1"}) == "r1"
    k1, k2 = sample_key({"messages": _msgs("x")}), sample_key({"messages": _msgs("x")})
    assert k1 == k2 and k1.startswith("conv:")


def _router(tmp_path, primary=None, candidate=None, **cfg):
    config = ShadowConfig(log_path=tmp_path / "pairs.jsonl", **{"rate": 1.0, **cfg})
    return ShadowRouter(
        primary or MockLLM(model="prim"), candidate or MockLLM(model="cand"), config
    )


def test_proxy_serves_primary_and_logs_shadow_pair(tmp_path):
    router = _router(tmp_path)
    client = TestClient(create_app(router))
    resp = client.post(
        "/v1/chat/completions", json={"messages": _msgs("hello")}, headers={"x-request-id": "r1"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "prim"
    assert data["choices"][0]["message"]["content"].startswith("[prim]")
    router.flush()
    rows = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["request_id"] == "r1"
    assert rows[0]["primary"]["model"] == "prim" and rows[0]["shadow"]["model"] == "cand"
    assert client.get("/shadow/stats").json()["sampled"] == 1


def test_unsampled_traffic_never_calls_candidate(tmp_path):
    calls = []

    def responder(m):
        calls.append(m)
        return "x"

    router = _router(tmp_path, candidate=MockLLM(model="c", responder=responder), rate=0.0)
    for i in range(20):
        router.handle({"messages": _msgs(f"q{i}")})
    router.close()
    assert calls == []
    assert not (tmp_path / "pairs.jsonl").exists()


class _Boom:
    model = "boom"

    def complete(self, messages, **kw):
        raise RuntimeError("candidate exploded")


class _Slow:
    model = "slow"

    def __init__(self, delay: float):
        self.delay = delay
        self.release = threading.Event()

    def complete(self, messages, **kw):
        self.release.wait(self.delay)
        return Completion(text="late", model=self.model, latency_s=self.delay)


def test_candidate_error_is_swallowed_and_recorded(tmp_path):
    router = _router(tmp_path, candidate=_Boom())
    client = TestClient(create_app(router))
    resp = client.post("/v1/chat/completions", json={"messages": _msgs("hi")})
    assert resp.status_code == 200
    router.flush()
    row = json.loads((tmp_path / "pairs.jsonl").read_text())
    assert "candidate exploded" in row["shadow"]["error"]
    assert router.stats.snapshot()["shadow_errors"] == 1


def test_candidate_never_on_latency_path_and_times_out(tmp_path):
    slow = _Slow(delay=5.0)
    router = _router(tmp_path, candidate=slow, candidate_timeout_s=0.2)
    t0 = time.perf_counter()
    completion, _ = router.handle({"messages": _msgs("hi")})
    assert time.perf_counter() - t0 < 0.15
    assert completion.model == "prim"
    router.flush(timeout=5)
    slow.release.set()
    row = json.loads((tmp_path / "pairs.jsonl").read_text())
    assert row["shadow"]["error"].startswith("timeout")
    router.close()


def test_primary_failure_returns_502_and_skips_shadow(tmp_path):
    router = _router(tmp_path, primary=_Boom())
    client = TestClient(create_app(router))
    resp = client.post("/v1/chat/completions", json={"messages": _msgs("hi")})
    assert resp.status_code == 502
    assert client.post("/v1/chat/completions", json={"nope": 1}).status_code == 400
    router.close()
    assert not (tmp_path / "pairs.jsonl").exists()


def _pair(i, a, b, err=None, ca=1e-3, cb=1e-4, la=1.0, lb=0.4):
    return {
        "request_id": f"r{i}",
        "messages": _msgs("how to tune postgres connection pooling"),
        "primary": {
            "model": "big",
            "text": a,
            "tokens_in": 10,
            "tokens_out": 20,
            "latency_s": la,
            "cost_usd": ca,
            "error": None,
        },
        "shadow": {
            "model": "small",
            "text": b,
            "tokens_in": 10,
            "tokens_out": 18,
            "latency_s": lb,
            "cost_usd": cb,
            "error": err,
        },
    }


def test_report_metrics_on_identical_answers_promote_cheaper_candidate():
    ans = "Tune postgres connection pooling with pgbouncer in transaction mode."
    pairs = [_pair(i, ans, ans) for i in range(30)]
    r = build_report(pairs, judge=default_judge())
    assert r.exact_agreement == 1.0 and r.near_agreement == 1.0
    assert r.cost_delta_pct == pytest.approx(-90.0)
    assert r.latency_p50_delta_s == pytest.approx(-0.6)
    assert r.judge["tie"] == 30 and r.verdict == "PROMOTE"


def test_report_blocks_on_error_rate_and_on_quality_loss():
    good = "Tune postgres connection pooling with pgbouncer in transaction mode and cap pool size."
    errs = [_pair(i, good, "", err="boom") for i in range(5)] + [
        _pair(i, good, good) for i in range(5, 50)
    ]
    r = build_report(errs, judge=None)
    assert r.verdict == "BLOCK" and r.shadow.error_rate == pytest.approx(0.1)
    worse = [_pair(i, good, "Not certain, it depends on your setup.") for i in range(25)]
    r2 = build_report(worse, judge=default_judge())
    assert r2.judge["loss"] == 25 and r2.verdict == "BLOCK"
    assert r2.disagreements and r2.disagreements[0]["judge"] == "loss"


def test_report_requires_comparable_pairs():
    with pytest.raises(ValueError):
        build_report([_pair(0, "a", "", err="x")])


def test_judge_is_position_debiased():
    biased = MockLLM(model="always-a", responder=lambda m: "A")
    assert judge_pair(biased, "q", "one", "two") == "tie"
    assert judge_pair(MockLLM(responder=lambda m: "B"), "q", "x", "y") == "tie"


def test_text_metrics_and_wilson():
    assert token_jaccard("a b c", "a b c") == 1.0
    assert token_jaccard("a b", "c d") == 0.0
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and wilson(0, 0) == (0.0, 1.0)


def test_demo_models_are_deterministic_and_candidate_is_cheaper():
    p, c = resolve_llm("demo:primary"), resolve_llm("demo:candidate")
    assert isinstance(p, DemoModel) and c.model == "demo-small"
    m = _msgs("How do I fix kafka consumer lag?")
    assert p.complete(m).text == p.complete(m).text
    assert c.price_out < p.price_out


def test_simulate_end_to_end_writes_report(tmp_path):
    reqs = tmp_path / "reqs.jsonl"
    reqs.write_text(
        "\n".join(
            json.dumps({"request_id": f"r{i}", "prompt": f"How do I fix topic{i % 13} errors?"})
            for i in range(400)
        )
    )
    cfg = ShadowConfig(rate=0.25, log_path=tmp_path / "pairs.jsonl")
    res = simulate(reqs, DemoModel("primary"), DemoModel("candidate"), cfg)
    assert res.ok == 400 and res.failed == 0
    assert 70 < res.sampled < 130
    pairs = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == res.sampled
    r = build_report(pairs, judge=default_judge())
    paths = write_report(r, tmp_path / "rep")
    assert "Shadow report" in paths["md"].read_text()
    html = paths["html"].read_text()
    assert "<svg" in html and "prefers-color-scheme: dark" in html
    assert render_markdown(r) and render_html(r)
    assert r.cost_delta_pct < -50
