"""Regressions from running the contamination checker against a real model (DeepSeek).

Responders mimic what real chat models return: verdicts with markdown or labels,
JSON in code fences / after prose, transient API failures, and per-call cost.
"""

import json

import pytest

from evalkit.contamination import ContaminationScanner, EvalIndex
from evalkit.contamination.judge import adjudicate, parse_verdict
from evalkit.contamination.leakgen import (
    KINDS,
    Usage,
    detector_recall,
    extract_json,
    generate_leaks,
    judge_pairs,
    parse_leaks,
    synth_items,
    to_markdown,
)
from evalkit.contamination.report import to_markdown as report_md
from evalkit.contamination.text import Record
from evalkit.core.llm import MockLLM

CAPITAL = (
    "What is the capital city of Australia, and why was it chosen instead of Sydney or "
    "Melbourne? Canberra, purpose-built as a compromise between rival cities."
)
TCP = (
    "In TCP congestion control, what triggers a fast retransmit and how does the sender "
    "adjust its congestion window afterwards? Three duplicate acknowledgements."
)
HASH_Q = (
    "In a hash table that uses linear probing, what happens to lookup performance as the "
    "load factor approaches one, and why do primary clusters form? Probe sequences grow long "
    "because occupied slots form contiguous runs that new keys extend."
)
TRAIN_Q = (
    "A train leaves Pune at 9:40 and travels 312 kilometres at an average speed of 78 "
    "kilometres per hour. At what time does it reach its destination? It arrives at 13:40."
)
EVAL = [("hash", HASH_Q), ("train", TRAIN_Q), ("capital", CAPITAL), ("tcp", TCP)]
PARA = (
    "Many visitors assume Sydney is the national capital, yet the seat of government is "
    "Canberra, a city designed from scratch as a compromise because neither Sydney nor "
    "Melbourne would accept the other as capital of Australia."
)


def suspicious_report():
    idx = EvalIndex.build(EVAL)
    rep = ContaminationScanner(idx).scan([Record("web-1", PARA)])
    assert {r.id: r.status for r in rep.items}["capital"] == "suspicious"
    return idx, rep


# --- judge verdict parsing ---------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ("YES\nThe passage restates it.", "YES"),
        ("NO\nDifferent fact.", "NO"),
        ("**YES** - the passage states the same answer.", "YES"),
        ("**Verdict: NO**\nOnly the topic matches.", "NO"),
        ("Verdict: yes. The answer is given verbatim.", "YES"),
        ("```\nNO\n```\nnot a restatement", "NO"),
        ("Yes, the passage contains the answer.", "YES"),
        ("No.", "NO"),
        ("The passage paraphrases the question and answer.\n\nYES", "YES"),
        ("Nothing conclusive here.", None),
        ("", None),
        ("It could be YES or NO depending on the definition.", None),
    ],
)
def test_parse_verdict_handles_real_reply_shapes(reply, verdict):
    assert parse_verdict(reply) == verdict


def test_bold_yes_upgrades_instead_of_being_ignored():
    # Before the fix only a bare first-line "YES" counted, so "**YES**" was a silent NO.
    idx, rep = suspicious_report()
    n = adjudicate(rep, idx, MockLLM(responder=lambda m: "**YES** - same answer, reworded."))
    assert n == 1
    r = {r.id: r for r in rep.items}["capital"]
    assert r.status == "contaminated" and "judge" in r.methods


def test_unreadable_verdict_stays_suspicious_and_is_counted():
    idx, rep = suspicious_report()
    adjudicate(rep, idx, MockLLM(responder=lambda m: "Hard to say without more context."))
    r = {r.id: r for r in rep.items}["capital"]
    assert r.status == "suspicious"
    assert any("no YES/NO verdict" in x for x in r.reasons)
    assert rep.judge_usage["unparsed"] == 1


def test_judge_api_failure_does_not_abort_scan():
    class Flaky:
        model = "flaky"

        def complete(self, messages, **kw):
            raise TimeoutError("read timed out")

    idx, rep = suspicious_report()
    assert adjudicate(rep, idx, Flaky()) == 0
    r = {r.id: r for r in rep.items}["capital"]
    assert r.status == "suspicious" and r.judge.startswith("error: TimeoutError")
    assert rep.judge_usage["errors"] == 1


def test_judge_cost_and_tokens_are_reported():
    idx, rep = suspicious_report()
    llm = MockLLM(model="paid", responder=lambda m: "NO\nonly the topic matches")
    adjudicate(rep, idx, llm)
    ju = rep.judge_usage
    assert ju["model"] == "paid" and ju["calls"] == 1 and ju["no"] == 1
    assert ju["tokens_in"] > 0 and ju["cost_usd"] > 0
    assert rep.to_dict()["judge_usage"] == ju
    assert "LLM judge `paid`: 1 call(s)" in report_md(rep)
    # no judge -> no usage block in the JSON
    _, plain = suspicious_report()
    assert "judge_usage" not in plain.to_dict()


# --- leak generation ---------------------------------------------------------

VARIANTS = {
    "light_edit": "Study notes. What is the capital of Australia and why not Sydney?",
    "paraphrase": "Q: Which city is Australia's capital? A: Canberra, a compromise.",
    "heavy_paraphrase": "Neither of the two big rival cities won; a new city was built.",
    "answer_only": "Canberra was purpose-built as a compromise between rival cities.",
    "negative": "Sydney Opera House opened in 1973 on Bennelong Point.",
}


def test_extract_json_from_fences_prose_and_trailing_notes():
    body = json.dumps(VARIANTS)
    assert extract_json(body) == VARIANTS
    assert extract_json(f"```json\n{body}\n```") == VARIANTS
    assert extract_json(f"Here are the documents:\n\n{body}\n\nLet me know!") == VARIANTS
    assert extract_json('Sure. [1] is a note.\n```\n[{"question": "q", "answer": "a"}]\n```') == [
        {"question": "q", "answer": "a"}
    ]
    with pytest.raises(ValueError):
        extract_json("I cannot help with that.")


def test_parse_leaks_normalizes_keys_and_nested_values():
    reply = json.dumps({
        "Light Edit": VARIANTS["light_edit"],
        "paraphrase": {"title": "Forum", "text": VARIANTS["paraphrase"]},
        "heavy-paraphrase": VARIANTS["heavy_paraphrase"],
        "ANSWER_ONLY": [VARIANTS["answer_only"]],
        "negative": "",
    })
    out = parse_leaks(reply)
    assert set(out) == {"light_edit", "paraphrase", "heavy_paraphrase", "answer_only"}
    assert out["paraphrase"] == "Forum " + VARIANTS["paraphrase"]


def test_generate_leaks_retries_bad_json_then_records_failures():
    replies = iter(["Sure! The documents are below.", "```json\n" + json.dumps(VARIANTS) + "\n```"])
    usage = Usage()
    rows = generate_leaks([("capital", CAPITAL)], MockLLM(responder=lambda m: next(replies)), usage)
    assert {r["kind"] for r in rows} == set(KINDS)
    assert usage.calls == 2 and not usage.failures and usage.cost_usd > 0

    usage = Usage()
    rows = generate_leaks([("capital", CAPITAL)], MockLLM(responder=lambda m: "nope"), usage)
    assert [r["kind"] for r in rows] == ["verbatim"]
    assert usage.failures and usage.failures[0].startswith("capital: ValueError")


def test_synth_items_accepts_wrapped_array():
    reply = "```json\n" + json.dumps({"items": [
        {"question": "Q one?", "answer": "A one."}, {"question": "", "answer": "x"},
        {"question": "Q two?", "answer": "A two."},
    ]}) + "\n```"
    rows = synth_items(MockLLM(responder=lambda m: reply), 5, Usage())
    assert rows == [
        {"id": "s01", "question": "Q one?", "answer": "A one."},
        {"id": "s02", "question": "Q two?", "answer": "A two."},
    ]


# --- recall scoring ----------------------------------------------------------


def _leaks():
    rows = []
    for k, (eid, text) in enumerate([("capital", CAPITAL), ("tcp", TCP)]):  # 2 of 4 items
        rows.append({"id": f"verbatim-{k}", "eval_id": eid, "kind": "verbatim",
                     "text": f"Practice question. {text} Next one."})
        rows.append({"id": f"negative-{k}", "eval_id": eid, "kind": "negative",
                     "text": "Bread rises because yeast ferments sugar into carbon dioxide."})
    rows.append({"id": "paraphrase-0", "eval_id": "capital", "kind": "paraphrase", "text": PARA})
    return rows


def test_detector_recall_counts_own_doc_and_pipeline_judge():
    idx = EvalIndex.build(EVAL)
    usage = Usage()
    judge = MockLLM(responder=lambda m: "**YES**\nsame answer")
    res = detector_recall(idx, _leaks(), judge=judge, usage=usage)
    assert res["verbatim"]["flag_rate"] == 1.0 and res["verbatim"]["contaminated_rate"] == 1.0
    assert res["negative"]["flag_rate"] == 0.0
    para = res["paraphrase"]
    assert para["items"] == 1 and para["by_method"]["embedding"] == 1
    assert para["contaminated_rate"] == 0.0 and para["pipeline_rate"] == 1.0
    assert para["doc_cosine"]["median"] > 0.3
    assert usage.calls == 1  # only the suspicious paraphrase went to the judge


def test_judge_pairs_and_markdown_flag_label_disagreements():
    idx = EvalIndex.build(EVAL)
    usage = Usage()
    judge = MockLLM(responder=lambda m: "YES\nleak" if "Canberra" in m[-1]["content"].split(
        "TRAINING PASSAGE:")[1] else "NO\nunrelated")
    pairs = judge_pairs(idx, _leaks(), judge, usage)
    assert pairs["verbatim"]["yes"] == 1 and pairs["verbatim"]["no"] == 1
    assert pairs["negative"]["yes_rate"] == 0.0
    result = {"items": 2, "generator": "g", "judge_model": judge.model,
              "detectors": detector_recall(idx, _leaks()), "judge": pairs,
              "usage": usage.to_dict()}
    md = to_markdown(result)
    assert "| verbatim | 2 | 100% |" in md
    assert "`verbatim` tcp: NO unrelated" in md  # a leak the judge rejected is surfaced


def test_cli_leaktest_offline(tmp_path, monkeypatch, capsys):
    from evalkit import cli
    from evalkit.core import llm as core_llm

    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps({"id": "capital", "question": CAPITAL}) + "\n")
    gen = MockLLM(model="gen", responder=lambda m: "Here you go:\n" + json.dumps(VARIANTS))
    monkeypatch.setattr(core_llm, "get_llm", lambda spec=None: gen)
    out = tmp_path / "lt"
    rc = cli.main(["contamination", "leaktest", "--eval", str(ev), "--llm", "x",
                   "--judge", "mock", "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "recall.json").read_text())
    assert res["items"] == 1 and res["detectors"]["verbatim"]["flag_rate"] == 1.0
    assert set(res["judge"]) == set(KINDS)
    assert (out / "leaks.jsonl").read_text().count("\n") == len(KINDS)
    # --reuse re-scores without calling the generator again
    monkeypatch.setattr(core_llm, "get_llm", lambda spec=None: MockLLM(responder=lambda m: "x"))
    assert cli.main(["contamination", "leaktest", "--eval", str(ev), "--llm", "x",
                     "--out", str(out), "--reuse"]) == 0
    assert "Failures" not in (out / "recall.md").read_text()
    assert "# Leak recall test" in capsys.readouterr().out
