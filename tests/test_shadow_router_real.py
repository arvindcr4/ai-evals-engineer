"""Regression tests for integration bugs found running 02 against real DeepSeek models.

The responders below mimic what deepseek-v4-pro / deepseek-flash actually
returned in the Oct 2026 run (multi-paragraph markdown answers of ~100 words,
single-token judge verdicts, ``finish_reason`` in the raw payload).
"""

import json
import warnings
from dataclasses import dataclass

import pytest

from evalkit.core.llm import Completion, MockLLM
from evalkit.shadow_router import ShadowConfig, ShadowRouter, build_report, create_app
from evalkit.shadow_router.report import (
    judge_pair,
    judge_pair_detailed,
    parse_verdict,
    render_markdown,
    similarity,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

SYSTEM = "You are a senior engineer. Answer in at most 120 words."

# Abridged real outputs for "How do I fix python asyncio cancellation?".
PRO = (
    "To fix asyncio cancellation, ensure your coroutines handle `CancelledError` properly. "
    "Common issues:\n\n1. **Swallowing cancellation**: Avoid broad `except Exception` blocks "
    "that catch `CancelledError`. Use `except asyncio.CancelledError: raise`.\n\n2. **Cleanup "
    "with `finally`**: Use `try/finally` for cleanup, but keep it minimal.\n\n3. **Shielding "
    "critical sections**: Use `asyncio.shield()` to protect tasks that must complete.\n\n4. "
    "**Timeout handling**: Wrap with `asyncio.wait_for()` and catch `TimeoutError`."
)
FLASH = (
    "Handle `asyncio.CancelledError` explicitly and re-raise it after cleanup:\n\n```python\n"
    "async def task():\n    try:\n        await work()\n    except asyncio.CancelledError:\n"
    "        await cleanup()\n        raise\n```\n\nKey points:\n- Never catch `CancelledError` "
    "with bare `except:`.\n- Use `asyncio.shield()` if a critical section must finish.\n- "
    "Cancel via `task.cancel()`, then `await task`.\n- Use `asyncio.timeout()` or `wait_for()`."
)


def _side(model, text, cost, err=None, finish="stop"):
    return {
        "model": model,
        "text": text,
        "tokens_in": 40,
        "tokens_out": 160,
        "latency_s": 2.9 if model == "pro" else 1.4,
        "cost_usd": cost,
        "error": err,
        "finish_reason": finish,
    }


def _pair(i, a=PRO, b=FLASH, **kw):
    return {
        "request_id": f"r{i:03d}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "How do I fix python asyncio cancellation?"},
        ],
        "primary": _side("pro", a, 7e-4),
        "shadow": _side("flash", b, 1.9e-4, **kw),
    }


def test_similarity_does_not_collapse_on_real_length_answers():
    # difflib's autojunk on >200-char strings scored this paraphrase ~0.06.
    assert len(PRO) > 200 and len(FLASH) > 200
    assert similarity(PRO, FLASH) > 0.15
    assert similarity(PRO, PRO) == 1.0
    assert similarity(PRO, "completely unrelated words here") < 0.05


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("B", "B"),
        ("**B**", "B"),
        ("A.", "A"),
        ("TIE", "TIE"),
        ("tie", "TIE"),
        ("Verdict: A", "A"),
        ("B, because it gives a concrete example", "B"),
        ("Answer B", "B"),
        ("A better answer is B", None),
        ("A and B are both fine", None),
        ("", None),
    ],
)
def test_parse_verdict_is_strict_about_ambiguity(reply, expected):
    assert parse_verdict(reply) == expected


def test_ambiguous_judge_reply_is_counted_not_silently_scored_as_a():
    d = judge_pair_detailed(MockLLM(responder=lambda m: "A and B are both fine"), "q", "one", "two")
    assert d["unparsed"] == 2 and d["outcome"] == "tie"


@dataclass
class RecordingJudge:
    """Judge that records kwargs/prompts; prefers whichever slot holds FLASH."""

    model: str = "rec-judge"
    fail: bool = False

    def __post_init__(self):
        self.calls: list[tuple[list, dict]] = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.fail:
            raise RuntimeError("upstream 503 after retries")
        body = messages[-1]["content"]
        a = body.split("Answer A:\n", 1)[1].split("\n\nAnswer B:", 1)[0]
        return Completion(text="A" if a == FLASH else "B", model=self.model, cost_usd=1e-4)


def test_judge_runs_at_temperature_zero_and_sees_system_instructions():
    judge = RecordingJudge()
    r = build_report([_pair(i) for i in range(5)], judge=judge)
    assert r.judge["win"] == 5 and r.judge["calls"] == 10
    assert r.judge["cost_usd"] == pytest.approx(10e-4)
    for messages, kwargs in judge.calls:
        assert kwargs["temperature"] == 0 and kwargs["max_tokens"] <= 32
        assert SYSTEM in messages[-1]["content"]
        assert "system instructions" in messages[-1]["content"]
    # no system prompt -> no instructions block
    judge2 = RecordingJudge()
    judge_pair(judge2, "q", PRO, FLASH)
    assert "System instructions" not in judge2.calls[0][0][-1]["content"]


def test_judge_failure_is_recorded_and_report_still_builds():
    r = build_report([_pair(i) for i in range(4)], judge=RecordingJudge(fail=True))
    assert r.judge["error"] == 4 and r.judge["win"] == r.judge["loss"] == 0
    assert any("not judged" in reason for reason in r.reasons)
    assert "judge errors 4" in render_markdown(r)


def test_position_bias_is_measured():
    always_a = MockLLM(model="always-a", responder=lambda m: "A")
    r = build_report([_pair(i) for i in range(6)], judge=always_a)
    assert r.judge["first_slot_rate"] == 1.0 and r.judge["order_flips"] == 6
    assert r.judge["tie"] == 6


def test_cheaper_candidate_below_non_loss_bar_gets_truthful_hold_reason():
    # (A,B) = shadow loss, (B,A) = shadow win, (A,A) = order flip -> tie.
    # 2 losses, 2 wins, 6 ties: non-loss 80% < 90%, win-rate CI spans 50%.
    outcomes = iter(["A", "B"] * 2 + ["B", "A"] * 2 + ["A", "A"] * 6)
    judge = MockLLM(model="j", responder=lambda m: next(outcomes))
    r = build_report([_pair(i) for i in range(10)], judge=judge)
    assert r.judge["loss"] == 2 and r.judge["win"] == 2 and r.verdict == "HOLD"
    assert r.cost_delta_pct < 0
    assert "cheaper" in r.reasons[0] and "no cost win" not in r.reasons[0]


def test_disagreements_list_judged_losses_first():
    other = PRO.replace("asyncio", "trio")
    pairs = [_pair(0, b=other), _pair(1, b="Not sure.")]
    seq = iter(["A", "B", "B", "A"])  # pair0: shadow loses both orders; pair1: wins
    r = build_report(pairs, judge=MockLLM(model="j", responder=lambda m: next(seq)))
    assert r.disagreements[0]["judge"] == "loss"
    assert r.disagreements[0]["request_id"] == "r000"


@dataclass
class RawLLM:
    """Mimics OpenAICompatibleLLM: finish_reason lives in ``raw``."""

    model: str
    finish: str = "stop"

    def complete(self, messages, **kwargs):
        raw = {"choices": [{"message": {"content": PRO}, "finish_reason": self.finish}]}
        return Completion(text=PRO, model=self.model, tokens_out=400, raw=raw)


def test_finish_reason_passes_through_proxy_and_into_report(tmp_path):
    router = ShadowRouter(
        RawLLM("pro", finish="length"),
        RawLLM("flash"),
        ShadowConfig(rate=1.0, log_path=tmp_path / "pairs.jsonl"),
    )
    resp = TestClient(create_app(router)).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]}
    )
    assert resp.json()["choices"][0]["finish_reason"] == "length"
    router.close()
    rows = [json.loads(x) for x in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert rows[0]["primary"]["finish_reason"] == "length"
    assert rows[0]["shadow"]["finish_reason"] == "stop"
    r = build_report(rows)
    assert r.primary.truncated_rate == 1.0 and r.shadow.truncated_rate == 0.0
    assert "truncated (finish=length) | 100.0%" in render_markdown(r)


def test_mock_models_still_report_stop(tmp_path):
    cfg = ShadowConfig(rate=0.0, log_path=tmp_path / "p.jsonl")
    router = ShadowRouter(MockLLM(model="p"), MockLLM(model="c"), cfg)
    c, _ = router.handle({"messages": [{"role": "user", "content": "x"}]})
    from evalkit.shadow_router.proxy import completion_response

    assert completion_response(c, "r")["choices"][0]["finish_reason"] == "stop"
    router.close()
