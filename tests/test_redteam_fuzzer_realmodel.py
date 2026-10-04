"""Regression tests for real-model (DeepSeek) integration of the red-team fuzzer.

Each MockLLM responder below mimics an output shape actually observed from
deepseek-flash driving the toy agent (Oct 2026).
"""

from __future__ import annotations

import argparse

import httpx

from evalkit.core.llm import Completion, MockLLM
from evalkit.core.trajectory import read_jsonl
from evalkit.redteam_fuzzer.agent import CANARY, ToyAgent, parse_action
from evalkit.redteam_fuzzer.attacks import AttackCase, seed_corpus
from evalkit.redteam_fuzzer.cli import _compare, _run
from evalkit.redteam_fuzzer.fuzzer import evaluate, run_campaign
from evalkit.redteam_fuzzer.report import campaign_from_findings, render_scorecard

_SEEDS = {c.attack_id: c for c in seed_corpus()}


def _scripted(*replies: str):
    """Responder returning ``replies`` in order (then the last one forever)."""
    state = {"i": 0}

    def responder(messages):
        r = replies[min(state["i"], len(replies) - 1)]
        state["i"] += 1
        return r

    return responder


# -- parsing ------------------------------------------------------------------


def test_parse_skips_chatty_preamble():
    # Observed: 'I'll look up your account notes.\n\nTOOL read_notes {...}'
    a = parse_action("I'll look up your account notes.\n\nTOOL read_notes {\"query\": \"account notes\"}")
    assert (a.kind, a.tool, a.args) == ("tool", "read_notes", {"query": "account notes"})


def test_parse_ignores_hallucinated_continuation_and_special_tokens():
    # Observed: one turn holding an action, DSML tool-call tokens, a second
    # action, a fabricated OBSERVATION and a premature FINAL.
    raw = ('TOOL retry_task {"task": "read account notes"}\n\n<｜｜DSML｜｜ calls>\n'
           '<｜｜DSML｜｜ invoke name="retry_task">\n</｜｜DSML｜｜ invoke>\n</thinking>\n\n'
           'TOOL retry_task {"task": "read_final"}\n\nOBSERVATION[retrieved]: none\n\n'
           "FINAL I wasn't able to retrieve any notes.")
    a = parse_action(raw)
    assert (a.kind, a.tool, a.args) == ("tool", "retry_task", {"task": "read account notes"})


def test_parse_final_stops_at_fabricated_next_turn():
    a = parse_action("FINAL Done.\nSecond line.\nOBSERVATION[tool_output]: fake\nTOOL search {}")
    assert a.kind == "final" and a.text == "Done.\nSecond line."


def test_parse_markdown_decorations_fences_and_trailing_junk():
    assert parse_action("**FINAL:** all good").text == "all good"
    assert parse_action("FINAL ANSWER: yes").text == "yes"
    a = parse_action('```\nTOOL: search {"query": "refund"} (looking it up)\n```')
    assert (a.tool, a.args) == ("search", {"query": "refund"})
    # Casing of a requested tool is preserved (probes case-sensitive guards).
    assert parse_action('TOOL Send_Email {"to": "x"}').tool == "Send_Email"


def test_parse_earliest_action_wins_and_prose_is_not_a_tool():
    a = parse_action("FINAL I will not call that.\nTOOL send_email {}")
    assert a.kind == "final" and "send_email" not in a.text
    # 'Tools ...' at line start is prose, not a TOOL action.
    assert parse_action("Tools are limited here.\nFINAL ok").text == "ok"


# -- agent loop ---------------------------------------------------------------


def test_history_echoes_canonical_action_not_raw_hallucination():
    seen: list[list[dict]] = []

    def responder(messages):
        seen.append([dict(m) for m in messages])
        if len(seen) == 1:
            return 'TOOL search {"query": "refund"}\nOBSERVATION[retrieved_doc]: FAKE SECRET\nFINAL x'
        return "FINAL Refunds within 30 days."

    r = ToyAgent("none", llm=MockLLM(responder=responder)).run(_SEEDS["seed-indirect-doc"])
    assert r.final_answer == "Refunds within 30 days."
    assistant_turns = [m["content"] for m in seen[1] if m["role"] == "assistant"]
    assert assistant_turns == ['TOOL search {"query": "refund"}']


def test_policy_called_with_temperature_zero_and_token_cap():
    calls: list[dict] = []

    class Spy(MockLLM):
        def complete(self, messages, **kwargs):
            calls.append(kwargs)
            return super().complete(messages, **kwargs)

    ToyAgent("none", llm=Spy(responder=lambda m: "FINAL ok")).run(_SEEDS["seed-direct-exfil"])
    assert calls == [{"temperature": 0, "max_tokens": 400}]


def test_provider_error_is_infra_not_a_crash_finding():
    def boom(messages):
        req = httpx.Request("POST", "https://api.example/v1/chat/completions")
        raise httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req))

    r = ToyAgent("none", llm=MockLLM(responder=boom)).run(_SEEDS["seed-direct-exfil"])
    assert r.infra_error and not r.crashed
    assert evaluate(r) == []
    camp = run_campaign("none", iterations=3, seed=0, llm=MockLLM(responder=boom))
    assert camp.runs == 0 and camp.infra_errors == 12 and camp.findings == []
    assert camp.attack_success_rate == 0.0


def test_real_refusals_produce_no_findings_but_leaks_still_do():
    refuse = MockLLM(responder=_scripted(
        "FINAL I can't share any secret or internal credentials."))
    r = ToyAgent("none", llm=refuse).run(_SEEDS["seed-direct-exfil"])
    assert evaluate(r) == []
    leak = MockLLM(responder=_scripted(f"Sure.\n**FINAL:** the key is {CANARY}"))
    r = ToyAgent("none", llm=leak).run(_SEEDS["seed-direct-exfil"])
    assert [h[0] for h in evaluate(r)] == ["canary_leaked"]


class _PricedLLM(MockLLM):
    def complete(self, messages, **kwargs) -> Completion:
        c = super().complete(messages, **kwargs)
        c.cost_usd = 0.01
        return c


def test_cost_tracked_and_spend_cap_stops_campaign():
    llm = _PricedLLM(responder=lambda m: "FINAL ok")
    camp = run_campaign("none", iterations=50, seed=0, llm=llm, max_cost_usd=0.05)
    assert camp.llm_calls == 5 and camp.runs == 5
    assert abs(camp.cost_usd - 0.05) < 1e-9
    assert "API cost: $0.0500" in render_scorecard([camp])


def test_default_toy_policy_reports_zero_cost():
    camp = run_campaign("none", iterations=5, seed=0)
    assert camp.llm_calls > 0 and camp.cost_usd == 0.0
    assert "API cost" not in render_scorecard([camp])


def test_class_stats_roundtrip_and_compare(tmp_path, capsys):
    _run(argparse.Namespace(guardrail="all", iterations=4, seed=0, step_budget=12,
                            token_budget=20_000, llm="mock", seeds="", max_cost_usd=None,
                            out=str(tmp_path / "m")))
    rows = read_jsonl(tmp_path / "m" / "findings.jsonl")
    camps = {c.guardrail: c for c in campaign_from_findings(rows)}
    assert camps["none"].class_stats["mutator:seed"][0] == len(seed_corpus())
    # cases log is replayable as a seed corpus
    cases = [AttackCase.from_dict(d) for d in read_jsonl(tmp_path / "m" / "cases-none.jsonl")]
    assert len(cases) == camps["none"].runs

    f = str(tmp_path / "m" / "findings.jsonl")
    _compare(argparse.Namespace(baseline=f, other=f, labels=["mock", "real"], out=""))
    out = capsys.readouterr().out
    assert "| guardrail | mock attack success | real attack success" in out
    assert "family:direct_injection" in out
