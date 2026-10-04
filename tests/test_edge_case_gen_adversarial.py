"""Adversarial regression tests for edge_case_gen (bugs found in review)."""

import argparse
import json

import pytest

from evalkit.core.llm import MockLLM
from evalkit.edge_case_gen import (
    EdgeCase,
    LLMGenerator,
    RuleGenerator,
    Validator,
    coverage_report,
    probe,
    propose_labels,
    spec_from_dict,
    summarize,
)
from evalkit.edge_case_gen.cli import register
from evalkit.edge_case_gen.spec import validate_input

OPTIONAL_SPEC = {
    "name": "optional",
    "description": "Ticket with an optional priority and optional tag.",
    "labels": ["ok", "invalid"],
    "invalid_label": "invalid",
    "fields": [
        {"name": "text", "type": "string", "max_length": 40},
        {"name": "priority", "type": "integer", "required": False, "min": 1, "max": 5},
        {"name": "tag", "type": "enum", "required": False, "choices": ["a", "b"]},
    ],
    "seeds": [
        {"text": "printer is on fire again"},
        {"text": "please reset the vpn token", "tag": "b"},
    ],
}


def test_optional_fields_absent_from_seeds_do_not_crash_generation_or_coverage():
    spec = spec_from_dict(json.loads(json.dumps(OPTIONAL_SPEC)))
    cases = RuleGenerator(spec, seed=3).generate()
    assert {c.axis for c in cases} == {"boundary", "format", "semantic", "adversarial",
                                       "combinatorial"}
    prio = [c for c in cases if c.field == "priority"]
    assert {"min", "max", "below_min", "above_max"} <= {c.category for c in prio}
    rep = coverage_report(spec, Validator(spec).run(cases).kept)
    assert rep["pairwise_coverage"] == 1.0
    combo = [c for c in cases if c.axis == "combinatorial"]
    assert all(validate_input(spec, c.input) == [] for c in combo
               if all(v == "nominal" for v in c.levels.values()))


def test_unlabeled_cases_do_not_dilute_failure_rate():
    cases = [EdgeCase(input={"q": str(i)}, axis="boundary", category="min") for i in range(9)]
    cases.append(EdgeCase(input={"q": "x"}, axis="boundary", category="max", label="good"))
    s = summarize(probe(lambda inp: "bad", cases))
    assert s["overall"]["unlabeled"] == 9 and s["overall"]["scored"] == 1
    assert s["overall"]["failure_rate"] == 1.0
    crash = summarize(probe(lambda inp: 1 / 0, cases[:3]))
    assert crash["overall"]["failure_rate"] == 1.0 and crash["overall"]["crash"] == 3


def test_max_fail_rate_gate_cannot_be_passed_by_an_unlabeled_set(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(OPTIONAL_SPEC))
    sys_path = tmp_path / "sut.py"
    sys_path.write_text("def run(x):\n    return 'wrong'\n")
    rows = [EdgeCase(input={"text": f"t{i}"}, axis="boundary", category="min").to_dict()
            for i in range(20)]
    rows.append(EdgeCase(input={"text": "z"}, axis="boundary", category="max",
                         label="ok").to_dict())
    gold = tmp_path / "g.jsonl"
    gold.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="system"))
    a = parser.parse_args(["edge-case-gen", "probe", "--spec", str(spec_path), "--in", str(gold),
                           "--system", f"{sys_path}:run", "--max-fail-rate", "0.5"])
    assert a.func(a) == 1


def test_help_text_renders_for_every_subcommand(capsys):
    parser = argparse.ArgumentParser(prog="evalkit")
    register(parser.add_subparsers(dest="system"))
    for argv in (["edge-case-gen", "--help"], ["edge-case-gen", "coverage", "--help"]):
        with pytest.raises(SystemExit) as e:
            parser.parse_args(argv)
        assert e.value.code == 0
    assert "pairwise %" in capsys.readouterr().out


def test_distinct_whitespace_perturbations_are_not_near_duplicates():
    spec = spec_from_dict(json.loads(json.dumps(OPTIONAL_SPEC)))
    base = "please reset the vpn token"
    variants = [base.replace(" ", "\r\n", 2), base + "　", base.replace(" ", "\t")]
    cases = [EdgeCase(input={"text": v, "tag": "b"}, axis="format", category=f"ws{i}")
             for i, v in enumerate(variants)]
    cases.append(EdgeCase(input={"text": variants[0], "tag": "b"}, axis="format", category="x"))
    rep = Validator(spec).run(cases)
    assert len(rep.kept) == 3 and [r["reason"] for r in rep.rejected] == ["duplicate"]


def test_llm_bare_array_reply_and_non_list_cases():
    spec = spec_from_dict(json.loads(json.dumps(OPTIONAL_SPEC)))
    arr = json.dumps([{"input": {"text": ""}, "category": "empty"}])
    gen = LLMGenerator(spec, MockLLM(responder=lambda m: f"Here you go:\n{arr}"))
    out = gen.generate(["boundary"])
    assert len(out) == 1 and out[0].category == "empty" and out[0].field == "text"
    gen = LLMGenerator(spec, MockLLM(responder=lambda m: '{"cases": "none today"}'))
    assert gen.generate(["boundary"]) == [] and "not a list" in gen.errors[0]


def test_relabel_drops_stale_label_when_oracle_now_crashes():
    spec = spec_from_dict(json.loads(json.dumps(OPTIONAL_SPEC)))
    c = EdgeCase(input={"text": "hi"}, axis="boundary", category="min", label="ok",
                 label_source="oracle", schema_valid=True)

    def broken(inp):
        raise RuntimeError("down")

    propose_labels(spec, [c], oracle=broken)
    assert c.label is None and c.needs_human_review
    assert any(r.startswith("oracle_error") for r in c.review_reasons)


def test_generation_is_deterministic_per_seed():
    spec = spec_from_dict(json.loads(json.dumps(OPTIONAL_SPEC)))
    a = [c.to_dict() for c in RuleGenerator(spec, seed=1).generate()]
    b = [c.to_dict() for c in RuleGenerator(spec, seed=1).generate()]
    assert a == b
