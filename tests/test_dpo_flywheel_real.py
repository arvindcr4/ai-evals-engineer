"""Regression tests for bugs found running the DPO flywheel against real DeepSeek models.

The responders below mimic outputs actually seen from deepseek-v4-pro (teacher)
and deepseek-flash (judge) in the Oct 2026 real-model run.
"""

import argparse
import json
import random
import threading

import pytest

from evalkit.core.llm import MockLLM
from evalkit.dpo_flywheel.cli import register
from evalkit.dpo_flywheel.feedback import FeedbackEvent, FeedbackStore
from evalkit.dpo_flywheel.filters import FilterConfig, filter_pairs, leaks_context
from evalkit.dpo_flywheel.llm_cache import MeteredLLM
from evalkit.dpo_flywheel.nightly import DryRunTrainer, MockEvaluator, NightlyConfig, run_nightly
from evalkit.dpo_flywheel.pairs import (
    LLMJudge,
    PairBuilder,
    PreferencePair,
    build_dataset,
    parse_score,
)

CORRECTION = "Go to Settings, choose Security, click Reset password and follow the emailed link."


def ev(cid, prompt, response="Please contact support.", rating="down", correction=None):
    return FeedbackEvent(cid, [{"role": "user", "content": prompt}], response, rating, correction)


def judge_llm(score_for=None):
    """deepseek-flash at T=0 answers the rubric with exactly ``SCORE: <int>``."""

    def responder(msgs):
        body = msgs[-1]["content"]
        reply = body.split("Reply:\n", 1)[1]
        if score_for:
            return f"SCORE: {score_for(reply)}"
        return "SCORE: 2" if "contact support" in reply.lower() else "SCORE: 8"

    return MockLLM(model="judge", responder=responder)


def random_teacher():
    """Sampled at T=0.8: a different (markdown) answer on every call."""
    rng = random.Random()

    def responder(msgs):
        return f"To do that:\n\n1. **Open Settings**\n2. Pick option {rng.randint(0, 10**9)}\n3. Save."

    return MockLLM(model="teacher", responder=responder)


# --- judge parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("SCORE: 2", 2.0),
        ("**SCORE:** 8", 8.0),
        ("Score = 7/10", 7.0),
        ("score: 8.5", 8.5),
        ("The reply is vague (score: 3 for tone).\nSCORE: 9", 9.0),
        ("SCORE: 85", 10.0),
        ("I'd rate it highly.", None),
    ],
)
def test_parse_score_variants(text, expected):
    assert parse_score(text) == expected


def test_judge_parse_failures_are_counted_not_silent():
    j = LLMJudge(MockLLM(responder=lambda m: "Looks fine to me."))
    s = j.score("How do I reset my password?", CORRECTION)
    assert 0 <= s <= 10 and j.parse_failures == 1
    assert LLMJudge(judge_llm()).score("q", CORRECTION) == 8.0


def test_judge_is_called_deterministically_with_a_token_cap():
    seen = {}

    def responder(msgs):
        return "SCORE: 5"

    class Spy(MockLLM):
        def complete(self, messages, **kw):
            seen.update(kw)
            return super().complete(messages, **kw)

    LLMJudge(Spy(responder=responder)).score("q", "r")
    assert seen["temperature"] == 0 and seen["max_tokens"] <= 128


# --- metering + cache ------------------------------------------------------------


def test_metered_llm_counts_cost_and_caches(tmp_path):
    cache = tmp_path / "c.jsonl"
    m = MeteredLLM(random_teacher(), cache_path=cache)
    msgs = [{"role": "user", "content": "hi"}]
    a = m.complete(msgs, temperature=0.8).text
    b = m.complete(msgs, temperature=0.8).text
    assert a == b
    assert m.usage.calls == 1 and m.usage.cache_hits == 1 and m.usage.cost_usd > 0
    assert m.complete(msgs, temperature=0.0).text != a  # kwargs are part of the key
    # persisted: a fresh process gets the same answer for free
    with open(cache, "a") as f:
        f.write('{"key": "torn')  # killed mid-write
    m2 = MeteredLLM(random_teacher(), cache_path=cache)
    assert m2.complete(msgs, temperature=0.8).text == a and m2.usage.calls == 0


def test_metered_llm_counts_errors():
    def boom(msgs):
        raise TimeoutError("read timeout")

    m = MeteredLLM(MockLLM(responder=boom))
    with pytest.raises(TimeoutError):
        m.complete([{"role": "user", "content": "x"}])
    assert m.usage.errors == 1 and m.usage.calls == 0


def _events():
    return [
        ev("c0", "How do I reset my password?", correction=CORRECTION),
        ev("c1", "How do I export invoices as CSV?"),
        ev("c2", "How do I rotate my webhook secret?"),
        ev("c3", "Why is my API key not working?"),
        ev("u1", "How can I change my plan?", "Open Billing, click Change plan and pick one.", "up"),
    ]


def test_cached_teacher_makes_rebuilds_reproducible(tmp_path):
    cache = tmp_path / "cache.jsonl"

    def build(cache_path):
        b = PairBuilder(teacher=MeteredLLM(random_teacher(), cache_path=cache_path),
                        judge=LLMJudge(MeteredLLM(judge_llm(), cache_path=cache_path)))
        return build_dataset(_events(), b, FilterConfig())

    r1, r2 = build(cache), build(cache)
    assert r1.manifest["data_sha256"] == r2.manifest["data_sha256"]
    assert r2.manifest["llm_usage"]["teacher"]["calls"] == 0
    assert r2.manifest["llm_usage"]["total_cost_usd"] == 0
    # without a persistent cache the T=0.8 teacher changes the dataset every rebuild
    assert build(None).manifest["data_sha256"] != build(None).manifest["data_sha256"]


def test_nightly_only_pays_for_new_events(tmp_path):
    calls = []
    lock = threading.Lock()

    def teacher(msgs):
        with lock:
            calls.append(msgs[-1]["content"])
        return "Open Settings, choose Webhooks, click Rotate secret and confirm the change."

    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append(_events())
    cache = tmp_path / "root" / "llm_cache.jsonl"

    def builder():
        return PairBuilder(teacher=MeteredLLM(MockLLM(responder=teacher), cache_path=cache),
                           judge=LLMJudge(MeteredLLM(judge_llm(), cache_path=cache)))

    cfg = NightlyConfig(root=tmp_path / "root", store=store.path, min_new_pairs=1)
    rec = run_nightly(cfg, builder(), DryRunTrainer(), MockEvaluator())
    first = len(calls)
    assert first > 0 and rec["llm_usage"]["teacher"]["calls"] == first
    store.append([ev("c9", "How do I delete a dashboard?")])
    rec2 = run_nightly(cfg, builder(), DryRunTrainer(), MockEvaluator())
    assert len(calls) - first == 3  # n_candidates for the one new thumbs-down only
    assert rec2["llm_usage"]["teacher"]["cache_hits"] == first
    assert "drops" in rec2


# --- concurrency + per-event errors -----------------------------------------------


def test_teacher_error_loses_one_event_not_the_run():
    def flaky(msgs):
        if "webhook" in msgs[-1]["content"]:
            raise TimeoutError("ReadTimeout after retries")
        return "Open Settings, select Developer, and regenerate the key if it was revoked."

    b = PairBuilder(teacher=MockLLM(responder=flaky), judge=LLMJudge(judge_llm()), workers=4)
    pairs, missed = b.build(_events())
    assert missed["teacher_error"] == 1 and missed["no_strategy_applied"] == 1
    assert {p.source_event for p in pairs} == {e.event_id for e in _events()
                                                if e.rating == "down" and "webhook" not in e.prompt_text}


def test_workers_keep_event_order_and_output():
    def teacher(msgs):
        return f"Answer for: {msgs[-1]['content']} Open Settings and follow the steps."

    def run(w):
        b = PairBuilder(teacher=MockLLM(responder=teacher), judge=LLMJudge(judge_llm()), workers=w)
        return [p.to_dict() for p in b.build(_events())[0]]

    assert run(1) == run(8)


# --- teacher grounding, context leaks, PII to the API -----------------------------


def test_teacher_sees_trusted_references_but_not_the_rejected_answer():
    systems = []

    def teacher(msgs):
        systems.append(msgs[0]["content"])
        return "Open Billing, select Invoices, and click Export CSV."

    events = _events()
    b = PairBuilder(teacher=MockLLM(responder=teacher), judge=LLMJudge(judge_llm()), n_candidates=1)
    b.build(events)
    assert systems and all(CORRECTION in s for s in systems)  # correction is a reference
    assert all("Open Billing, click Change plan" in s for s in systems)  # thumbs-up too
    assert all("Please contact support." not in s for s in systems)  # never the bad answer
    assert all("Example conversations" in s for s in systems)


# Real deepseek-v4-pro outputs from the run (prompt v2) and clean ones (prompt v3).
LEAKY = [
    "I don't have a verified answer for two-factor authentication in this product.",
    "I don't have specific steps for rotating a webhook secret in the verified product documentation I've been given.",
    "I don't have specific troubleshooting steps for API key issues in the verified answers I can reference.",
    "I don't have a specific download page for the mobile app in the information available to me.",
    "I don't have specific information about API key troubleshooting in my documentation for this product.",
]
CLEAN = [
    "I can't see your account details here, but a decline usually comes from your bank, not from us.",
    "Go to Settings, choose Security, and follow the two-factor authentication setup steps.",
    "I don't have specific information about rotating webhook secrets for this product. Check your webhook settings.",
    "You can download the mobile app from the official app store for your device.",
    "Verify that your billing address matches; we have provided a refund form in Billing.",
]


@pytest.mark.parametrize("text", LEAKY)
def test_context_leaks_detected(text):
    assert leaks_context(text)


@pytest.mark.parametrize("text", CLEAN)
def test_clean_answers_not_flagged(text):
    assert not leaks_context(text)


def test_leaking_teacher_candidates_are_skipped():
    def teacher(msgs):
        if msgs[0]["content"].startswith("candidate-0"):
            return LEAKY[0]
        return "Go to Settings, choose Security, and turn on Two-factor authentication."

    b = PairBuilder(teacher=MockLLM(responder=teacher), judge=LLMJudge(judge_llm()))
    pairs, _ = b.build([ev("c1", "How do I enable two-factor authentication?")])
    assert len(pairs) == 1 and not leaks_context(pairs[0].chosen)
    assert pairs[0].meta["leaked_candidates"] == 1 and pairs[0].meta["candidates"] == 2


def test_context_leak_filter_applies_to_teacher_pairs_only():
    leak = PreferencePair("q", LEAKY[1], "Contact support.", "teacher", "e1", 5.0)
    user = PreferencePair("q2", "Per the verified answers in our docs, open Billing.", "No.", "correction", "e2", 5.0)
    kept, drops, _ = filter_pairs([leak, user], FilterConfig())
    assert drops["context_leak"] == 1 and kept == [user]


def test_pii_is_scrubbed_before_reaching_teacher_or_judge():
    sent = []
    lock = threading.Lock()

    def record(reply):
        def responder(msgs):
            with lock:
                sent.extend(m["content"] for m in msgs)
            return reply
        return responder

    events = [
        ev("p1", "My email is jo@example.com and card 4111 1111 1111 1111 was charged twice",
           correction="Sorry! Request a refund in Billing or call +1 415 555 0134."),
        ev("p2", "My card 4111 1111 1111 1111 was charged twice, email jo@example.com"),
    ]
    b = PairBuilder(teacher=MockLLM(responder=record("Open Billing and click Request refund on the duplicate.")),
                    judge=LLMJudge(MockLLM(responder=record("SCORE: 7"))), workers=2)
    pairs, _ = b.build(events)
    assert len(pairs) == 2 and sent
    blob = "\n".join(sent)
    for raw in ("jo@example.com", "4111 1111 1111 1111", "415 555 0134"):
        assert raw not in blob
    assert "[EMAIL]" in blob and "[CARD]" in blob


# --- manifest / CLI reporting -------------------------------------------------------


def test_manifest_reports_usage_margins_and_shared_model_once():
    shared = MeteredLLM(MockLLM(responder=lambda m: "SCORE: 6" if "Reply:" in m[-1]["content"]
                                else "Open Settings, choose Developer and regenerate the key."))
    b = PairBuilder(teacher=shared, judge=LLMJudge(shared))
    m = build_dataset(_events(), b, FilterConfig(min_margin=0)).manifest
    u = m["llm_usage"]
    assert u["teacher"] == u["judge"]
    assert u["total_cost_usd"] == pytest.approx(shared.usage.cost_usd, abs=1e-6)
    assert m["judge_parse_failures"] == 0 and m["judge_model"] == shared.model
    assert set(m["margins_raw"]) >= {"correction", "teacher"}
    assert {"n", "min", "median", "mean", "max"} <= set(m["margins_raw"]["teacher"])


def main(argv):
    parser = argparse.ArgumentParser(prog="evalkit")
    register(parser.add_subparsers(dest="system", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def test_cli_nightly_writes_cache_under_root_and_reports_usage(tmp_path, capsys):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append(_events())
    argv = ["dpo-flywheel", "nightly", "--store", str(store.path), "--root", str(tmp_path / "root"),
            "--teacher", "mock", "--judge", "mock", "--min-new-pairs", "1", "--date", "2026-10-04", "--dry-run"]
    assert main(argv) == 0
    rec = json.loads(capsys.readouterr().out)
    assert (tmp_path / "root" / "llm_cache.jsonl").exists()
    assert rec["llm_usage"]["teacher"]["calls"] > 0
    assert main([*argv, "--min-new-pairs", "0"]) == 0
    rec2 = json.loads(capsys.readouterr().out)
    assert rec2["llm_usage"]["teacher"]["calls"] == 0
