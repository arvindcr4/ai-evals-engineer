"""Regression tests for running the calibrated judge against a real API model (DeepSeek).

The MockLLM responders below mimic replies and failure modes seen or expected from a real
judge: tie verdicts with a high stated confidence, percentage confidences, "Response B"-style
winner strings, prose around the JSON, scale mentions in pointwise replies, and transport
errors that survive the client's retries.
"""

import json
import threading

import pytest

from evalkit.calibrated_judge import (
    PairwiseJudge,
    PointwiseJudge,
    collect_judgments,
    make_anchors,
    parse_pairwise,
    parse_score,
)
from evalkit.calibrated_judge.judge import Usage
from evalkit.cli import main
from evalkit.core.llm import Completion, MockLLM


@pytest.fixture(scope="module")
def anchors():
    return make_anchors(60, seed=7)


@pytest.mark.parametrize("text,expected", [
    # exactly what deepseek-flash returned in the Oct 2026 run
    ('{"winner": "A", "confidence": 0.95}', ("A", 0.95)),
    ('{"winner": "tie", "confidence": 0.85}', ("tie", 0.85)),
    # percentage confidence used to clamp to 1.0
    ('{"winner": "B", "confidence": 85}', ("B", 0.85)),
    ('{"winner": "A", "confidence": "70"}', ("A", 0.7)),
    # winner spelled out
    ('{"winner": "Response B", "confidence": 0.8}', ("B", 0.8)),
    ('{"winner": "a", "confidence": 0.9}', ("A", 0.9)),
    ('{"winner": "both", "confidence": 0.6}', ("tie", 0.6)),
    # prose and an unrelated brace object before the verdict
    ('Criteria: {accuracy, coverage}. Verdict:\n```json\n{"winner": "B", "confidence": 0.9}\n```',
     ("B", 0.9)),
    ('{"notes": "B is longer"} {"winner": "A", "confidence": 0.7}', ("A", 0.7)),
])
def test_parse_pairwise_real_model_variants(text, expected):
    assert parse_pairwise(text) == expected


def test_parse_score_ignores_scale_mentions():
    assert parse_score('{"score": 3}') == 3.0
    assert parse_score('{"score": "6"}') == 6.0
    assert parse_score("On a scale of 1-10 this deserves a 6.") == 6.0
    assert parse_score("Rated from 1 to 10, I give it 8/10.") == 8.0
    assert parse_score("Score: 7") == 7.0
    assert parse_score('{"score": 0}') == 1.0


def test_tie_with_high_confidence_is_neutral():
    j = PairwiseJudge(MockLLM(responder=lambda m: '{"winner": "tie", "confidence": 0.85}'))
    v = j.judge("q", "x", "y")
    assert (v.winner, v.p_a, v.confidence) == ("tie", 0.5, 0.85)


def _fake_api(price_in=0.30, price_out=1.20):
    """A thread-safe LLM returning real-looking verdicts with token usage like the API."""

    class Fake:
        model = "deepseek-flash-fake"

        def complete(self, messages, **kwargs):
            assert kwargs.get("temperature") == 0
            prompt = messages[-1]["content"]
            text = ('{"score": 5}' if "[Response]" in prompt
                    else '{"winner": "A", "confidence": 0.9}')
            return Completion(text, self.model, 300, 14, 0.0, (300 * price_in + 14 * price_out) / 1e6)

    return Fake()


def test_usage_tracks_calls_tokens_and_cost(anchors):
    llm = _fake_api()
    pair, point = PairwiseJudge(llm), PointwiseJudge(llm)
    rows = collect_judgments(anchors[:10], pair, point)
    assert len(rows) == 10
    assert pair.usage.calls == 20 and point.usage.calls == 20
    total = pair.usage.merge(point.usage)
    assert total.tokens_in == 40 * 300 and total.tokens_out == 40 * 14
    assert total.cost_usd == pytest.approx(40 * (300 * 0.30 + 14 * 1.20) / 1e6)
    assert total.to_dict()["errors"] == 0


def test_usage_is_thread_safe():
    u = Usage()
    c = Completion("x", "m", 1, 1, 0.0, 1e-6)
    threads = [threading.Thread(target=lambda: [u.add(c) for _ in range(1000)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert u.calls == 8000 and u.cost_usd == pytest.approx(8000e-6)


def test_concurrent_collection_matches_serial_and_keeps_order(anchors):
    def responder(messages):  # verdict depends on content, not call order
        return ('{"winner": "A", "confidence": 0.8}' if len(messages[-1]["content"]) % 2
                else '{"winner": "B", "confidence": 0.7}')

    serial = collect_judgments(anchors, PairwiseJudge(MockLLM(responder=responder)))
    seen = []
    parallel = collect_judgments(anchors, PairwiseJudge(MockLLM(responder=responder)),
                                 workers=8, progress=lambda d, t: seen.append((d, t)))
    assert [j.to_dict() for j in parallel] == [j.to_dict() for j in serial]
    assert seen[-1] == (len(anchors), len(anchors))


def test_failed_api_calls_drop_the_anchor_instead_of_aborting(anchors, capsys):
    class Flaky:
        model = "flaky"

        def complete(self, messages, **kwargs):
            if anchors[3].response_a in messages[-1]["content"]:
                raise TimeoutError("read timed out after retries")
            return Completion('{"winner": "B", "confidence": 0.9}', "flaky", 10, 5)

    judge = PairwiseJudge(Flaky())
    rows = collect_judgments(anchors[:10], judge, workers=4)
    assert len(rows) == 9 and anchors[3].id not in {r.id for r in rows}
    assert judge.usage.errors == 1
    assert "judge failed on" in capsys.readouterr().err


def test_cli_judge_family_none_and_usage_in_report(tmp_path, monkeypatch):
    import evalkit.calibrated_judge.cli as cj

    monkeypatch.setattr(cj, "get_llm", lambda spec: _fake_api())
    anc, js, out = tmp_path / "a.jsonl", tmp_path / "j.jsonl", tmp_path / "audit.json"
    assert main(["calibrated-judge", "make-anchors", "--n", "60", "--out", str(anc)]) == 0
    assert main(["calibrated-judge", "audit", "--anchors", str(anc), "--llm", "deepseek:x",
                 "--judge-family", "none", "--workers", "4", "--judgments", str(js),
                 "--out-json", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["judge_family"] is None and rep["raw"]["self_preference"]["n"] == 0
    assert rep["usage"]["calls"] == 120 and rep["usage"]["cost_usd"] > 0
    # cached verdicts: calibrate makes no calls and reports no usage
    cal = tmp_path / "cal.json"
    assert main(["calibrated-judge", "calibrate", "--anchors", str(anc), "--llm", "deepseek:x",
                 "--judge-family", "none", "--judgments", str(js), "--out-json",
                 str(cal)]) == 0
    crep = json.loads(cal.read_text())
    assert "usage" not in crep and crep["model"]["judge_family"] is None
