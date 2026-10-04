"""Adversarial regression tests for the RAG harness (found in review)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalkit.cli import main
from evalkit.rag_adversarial import QAItem, load_items, parse_response, perturb, score_case
from evalkit.rag_adversarial.data import contains, normalize
from evalkit.rag_adversarial.perturb import alt_answer

DATA = Path(__file__).resolve().parents[1] / "examples" / "05-rag-adversarial" / "qa.jsonl"


def _item(id_: str, q: str, gold: str, docs: list[tuple[str, str]], **kw) -> QAItem:
    return QAItem.from_dict({
        "id": id_, "question": q, "gold_answer": gold,
        "docs": [{"id": i, "text": t, "date": "2020-01-01"} for i, t in docs], **kw,
    })


def test_comma_separated_citations_are_parsed_and_valid():
    cases = {c.case_id: c for c in perturb(load_items(DATA), ["clean"])}
    case = cases["q01::clean"]
    assert parse_response("1987 [q01-a, q01-b]").citations == ["q01-a", "q01-b"]
    assert parse_response("1987 [q01-a; q01-b][q01-a]").citations == ["q01-a", "q01-b"]
    r = score_case(case, "1987 [q01-a, q01-b]")
    assert r.correct and not r.lie and r.citation_valid


def test_contradiction_swap_is_token_bounded():
    it = _item("x", "How many moons does Orvex have?", "7",
               [("x-a", "Orvex, catalogued in 1977, has 7 moons.")], alt_answer="11")
    (c,) = perturb([it], ["contradiction"])
    clash = c.doc("x-contra")
    assert clash is not None
    assert clash.text == "Orvex, catalogued in 1977, has 11 moons."


def test_contradiction_swap_handles_thousands_separator_or_skips():
    it = _item("y", "How many rings does Zed have?", "1,250", [("y-a", "Zed has 1250 rings.")],
               alt_answer="900")
    cases = perturb([it], ["contradiction", "stale"])
    assert len(cases) == 2
    for c in cases:
        planted = [d for d in c.docs if d.id != "y-a"]
        assert planted and all("900" in d.text and "1250" not in d.text for d in planted)


def test_alt_answer_never_equals_gold():
    a = _item("a", "Who founded Foo?", "Ann Lee", [("a1", "Foo was founded by Ann Lee.")])
    b = _item("b", "Who founded Bar?", "ann lee", [("b1", "Bar was founded by Ann Lee.")])
    c = _item("c", "Who founded Baz?", "Bob Ray", [("c1", "Baz was founded by Bob Ray.")])
    assert normalize(alt_answer(a, [a, b, c])) != normalize(a.gold_answer)
    with pytest.raises(ValueError):
        alt_answer(_item("d", "Who?", "Ann", [("d1", "Ann")], alt_answer="ANN"), [])


def test_colliding_doc_ids_across_items_are_disambiguated():
    a = _item("a", "Who founded Foo?", "Ann Lee", [("doc-1", "Foo was founded by Ann Lee.")])
    b = _item("b", "Who founded Bar?", "Bob Ray", [("doc-1", "Bar was founded by Bob Ray.")])
    for c in perturb([a, b], ["distractor", "gold_removal"]):
        ids = [d.id for d in c.docs]
        assert len(ids) == len(set(ids)), ids
    clean = perturb([a, b], ["distractor"])[0]
    assert score_case(clean, "Ann Lee [doc-1]").correct


def test_gold_removal_drops_unlisted_docs_that_still_state_the_answer():
    a = _item("a", "Who founded Foo?", "Ann Lee",
              [("a1", "Foo was founded by Ann Lee."), ("a2", "Ann Lee founded Foo in 1990.")],
              supporting_docs=["a1"])
    b = _item("b", "Who founded Bar?", "Bob Ray", [("b1", "Bar was founded by Bob Ray.")])
    (c,) = [x for x in perturb([a, b], ["gold_removal"]) if x.base_id == "a"]
    assert not any(contains(d.text, normalize("Ann Lee")) for d in c.docs)


def test_entity_swap_skips_when_entity_not_literally_present():
    it = _item("z", "When was the Orvex probe launched?", "2001",
               [("z-a", "The probe was launched in 2001.")], entity="Orvex", near_miss="Orvax")
    assert perturb([it], ["entity_swap"]) == []


def test_report_on_empty_results_fails_cleanly(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert main(["rag-adversarial", "report", str(empty), "--max-lie-rate", "0.05"]) == 1
    assert "no scored results" in capsys.readouterr().out
