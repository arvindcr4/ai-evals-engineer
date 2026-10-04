"""Regression tests for fixes found running 05 rag_adversarial against real DeepSeek models.

Every MockLLM responder here replays a reply shape actually seen from
deepseek-flash / deepseek-v4-pro in the Oct 2026 real run (see
docs/projects/05-rag-adversarial.md, "Real-model run").
"""

from __future__ import annotations

import itertools
import json
import random
import threading
import time
from pathlib import Path

import pytest

from evalkit.cli import main
from evalkit.core.llm import MockLLM
from evalkit.rag_adversarial import (
    ABSTAIN,
    CONFLICT,
    LLMRAG,
    Case,
    Doc,
    build_prompt,
    load_items,
    metrics,
    parse_response,
    perturb,
    rescore,
    run_cases,
    score_case,
    summarize,
    to_markdown,
)

DATA = Path(__file__).resolve().parents[1] / "examples" / "05-rag-adversarial" / "qa.jsonl"
HARD = DATA.with_name("qa_hard.jsonl")


def _case(expected="answer", ids=("d1", "d2")):
    docs = [Doc(ids[0], "The capital of Norvania is Estholm."), Doc(ids[1], "Norvania is cold.")]
    return Case("x::op", "x", "op", "What is the capital of Norvania?", docs, expected,
                "Estholm", [ids[0]], [])


@pytest.fixture(scope="module")
def cases() -> list[Case]:
    return perturb(load_items(DATA))


# --- bare-citation replies (v4-pro returned "[h01-a]" on 23/238 cases) ---------------

@pytest.mark.parametrize("reply", ["[d1]", "[ d1]", "[d1].", "[d1][d2]"])
def test_bare_citation_is_malformed_not_a_lie(reply):
    r = score_case(_case(), reply)
    assert r.malformed and not r.lie and not r.correct and not r.abstained
    assert r.citation_valid is None
    # also on an abstain case: no claim was made, so it is not a lie there either
    r2 = score_case(_case("abstain"), reply)
    assert r2.malformed and not r2.lie and not r2.correct


def test_malformed_counted_separately_from_citation_validity():
    rows = [score_case(_case(), x).to_dict() for x in ("[d1]", "Estholm [d1]")]
    m = metrics(rows)
    assert m["malformed_rate"] == 0.5 and m["lie_rate"] == 0.0
    assert m["citation_validity"] == 1.0  # the bare citation is not an "answered" row
    assert "Malformed" in to_markdown(summarize(rows))


def test_prompt_spells_out_reply_forms():
    system = build_prompt("q?", [Doc("d1", "t")])[0]["content"]
    assert f"\n{ABSTAIN}\n" in system and f"\n{CONFLICT} [doc-id] [doc-id]" in system
    assert "a citation alone is not an answer" in system
    assert "{" not in system  # the f-string placeholders were filled


# --- bracketed sentinel ("[CONFLICTING_EVIDENCE] [D632] [D233]" from v4-pro) ---------

def test_bracketed_sentinel_is_a_flag_not_a_citation():
    p = parse_response(f"[{CONFLICT}] [D632] [D233]")
    assert p.flagged_conflict and p.abstained and p.citations == ["D632", "D233"]
    assert parse_response(f"[{ABSTAIN}]").citations == []
    assert score_case(_case("conflict"), f"[{CONFLICT}] [d1] [d2]").correct


# --- prose abstentions (gold_removal replies from both models) ------------------------

@pytest.mark.parametrize("reply", [
    ("The Lumen Hotel chain is managed day to day by Crane Hospitality [h30-c]. "
     "The documents do not state which company owns the Lumen Hotel chain."),
    ("Peregrine Jet flies two daily routes from Morrow Field, but no document states which "
     "airline operates the hub there. [h23-c] [h23-b]"),
    "The capital is not stated in the documents.",
])
def test_prose_abstention_is_an_abstention(reply):
    assert parse_response(reply).abstained
    r = score_case(_case("abstain"), reply)
    assert r.correct and not r.lie


@pytest.mark.parametrize("reply", [
    "Estholm [d1]",
    "The documents state that the capital is Estholm [d1].",
    "Estholm is the capital, as no other document disputes [d1].",
])
def test_plain_answers_are_not_mistaken_for_abstention(reply):
    r = score_case(_case(), reply)
    assert not r.abstained and r.correct


# --- cost / token accounting (was dropped: CLI never surfaced LLMRAG.cost_usd) --------

def test_usage_recorded_per_case_and_summed(cases):
    rag = LLMRAG(MockLLM(responder=lambda m: ABSTAIN))
    rows = [r.to_dict() for r in run_cases(cases[:8], rag)]
    assert all(r["meta"]["usage"]["cost_usd"] > 0 for r in rows)
    assert rag.calls == 8 and rag.tokens_in > 0 and rag.tokens_out > 0
    m = summarize(rows)[rag.name]["overall"]
    assert m["cost_usd"] == pytest.approx(rag.cost_usd, abs=1e-6)
    assert "$0." in to_markdown(summarize(rows))


# --- concurrency (serial 238 calls took 2m45s against the API) ------------------------

def test_workers_keep_order_and_totals(cases):
    rng, lock = random.Random(0), threading.Lock()

    def slow(messages):
        with lock:
            d = rng.random() / 200
        time.sleep(d)
        q = messages[-1]["content"].rsplit("Question: ", 1)[1]
        return f"{ABSTAIN} {len(q)}"

    serial = run_cases(cases[:24], LLMRAG(MockLLM(responder=slow)))
    rag = LLMRAG(MockLLM(responder=slow))
    par = run_cases(cases[:24], rag, workers=6)
    assert [r.case_id for r in par] == [c.case_id for c in cases[:24]]
    assert [r.response for r in par] == [r.response for r in serial]
    assert rag.calls == 24


# --- non-determinism at temperature 0 (h07/h08 contradiction flipped across reruns) --

def test_repeats_measure_instability():
    flips = itertools.cycle(["Estholm [d1]", f"{CONFLICT} [d1] [d2]"])
    flaky = LLMRAG(MockLLM(responder=lambda m: next(flips)))
    rows = [r.to_dict() for r in run_cases([_case("conflict")], flaky, repeats=4)]
    assert len(rows) == 4 and [r["meta"]["repeat"] for r in rows] == [0, 1, 2, 3]
    m = metrics(rows)
    assert m["unstable_rate"] == 1.0 and m["lie_rate"] == 0.5
    stable = LLMRAG(MockLLM(responder=lambda m: f"{CONFLICT} [d1] [d2]"))
    assert metrics([r.to_dict() for r in run_cases([_case("conflict")], stable, repeats=3)])[
        "unstable_rate"] == 0.0
    assert metrics(rows[:1])["unstable_rate"] is None  # no repeats -> not measured


# --- rescore: apply scorer fixes to paid runs without new calls -----------------------

def test_rescore_regrades_and_keeps_usage():
    old = score_case(_case("abstain"), "Crane [d2]. The documents do not state the capital.")
    row = old.to_dict() | {"abstained": False, "lie": True, "correct": False}
    row["meta"] |= {"usage": {"cost_usd": 0.001}, "repeat": 2}
    (new,) = rescore([_case("abstain")], [row])
    assert new.correct and not new.lie
    assert new.meta["usage"]["cost_usd"] == 0.001 and new.meta["repeat"] == 2


def test_cli_workers_repeats_cost_and_rescore(tmp_path, capsys):
    cases_p, res, res2 = tmp_path / "c.jsonl", tmp_path / "r.jsonl", tmp_path / "r2.jsonl"
    assert main(["rag-adversarial", "perturb", "--dataset", str(DATA), "--out", str(cases_p)]) == 0
    assert main(["rag-adversarial", "run", "--cases", str(cases_p), "--system", "llm",
                 "--llm", "mock", "--workers", "4", "--repeats", "2", "--limit", "8",
                 "--out", str(res)]) == 0
    out = capsys.readouterr().out
    assert "16 cases" in out and "16 calls" in out and "$" in out
    assert main(["rag-adversarial", "rescore", "--cases", str(cases_p), "--results", str(res),
                 "--out", str(res2)]) == 0
    rows = [json.loads(x) for x in res2.read_text().splitlines()]
    assert len(rows) == 16 and all("usage" in r["meta"] for r in rows)
    assert main(["rag-adversarial", "report", str(res2)]) == 0
    assert "| Unstable |" in capsys.readouterr().out


def test_hard_set_perturbs_cleanly():
    items = load_items(HARD)
    assert len(items) == 30
    cs = perturb(items)
    # every operator applies to every item except entity swaps whose near-miss
    # contains the entity (Arrowline -> Arrowline Lite), which the operator skips
    ops = {c.perturbation for c in cs}
    assert len(ops) == 8 and len(cs) >= 30 * 7 + 25
