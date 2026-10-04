import argparse
import json
from pathlib import Path

import pytest

from evalkit.core.llm import MockLLM
from evalkit.core.trajectory import read_jsonl
from evalkit.edge_case_gen import (
    AXES,
    EdgeCase,
    LLMGenerator,
    RuleGenerator,
    Validator,
    coverage_report,
    covering_array,
    field_levels,
    pairwise_coverage,
    probe,
    propose_labels,
    spec_from_dict,
    summarize,
    validate_input,
)
from evalkit.edge_case_gen.cli import register
from evalkit.edge_case_gen.llm_gen import extract_json, llm_label
from evalkit.edge_case_gen.pairwise import all_pairs
from evalkit.edge_case_gen.validate import load_callable

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "12-edge-case-generator"

SPEC = {
    "name": "intent",
    "description": "Classify a banking chat message.",
    "text_field": "message",
    "labels": ["balance", "transfer", "other", "invalid"],
    "invalid_label": "invalid",
    "fields": [
        {"name": "message", "type": "string", "min_length": 1, "max_length": 60},
        {"name": "amount", "type": "integer", "min": 1, "max": 1000},
        {"name": "channel", "type": "enum", "choices": ["app", "web", "sms"]},
        {"name": "sent", "type": "date", "min": "2025-01-01", "max": "2026-12-31"},
        {"name": "verified", "type": "boolean"},
    ],
    "seeds": [
        {"message": "What is my balance today?", "amount": 10, "channel": "app",
         "sent": "2026-02-01", "verified": True},
        {"message": "Send 50 dollars to Priya please", "amount": 50, "channel": "web",
         "sent": "2025-07-15", "verified": False},
    ],
}


@pytest.fixture
def spec():
    return spec_from_dict(json.loads(json.dumps(SPEC)))


def oracle(inp):
    return "transfer" if "send" in inp["message"].lower() else "balance"


def test_spec_rejects_seed_that_violates_schema():
    bad = json.loads(json.dumps(SPEC))
    bad["seeds"][0]["amount"] = 5000
    with pytest.raises(ValueError, match="seed 0"):
        spec_from_dict(bad)


def test_validate_input_catches_each_violation(spec):
    ok = dict(spec.seeds[0])
    assert validate_input(spec, ok) == []
    assert any("missing" in e for e in validate_input(spec, {k: v for k, v in ok.items()
                                                             if k != "amount"}))
    assert validate_input(spec, {**ok, "amount": True})  # bool is not an integer
    assert validate_input(spec, {**ok, "amount": 1.5})
    assert validate_input(spec, {**ok, "sent": "2025-02-30"})
    assert validate_input(spec, {**ok, "sent": "01/02/2026"})
    assert validate_input(spec, {**ok, "channel": "APP"})
    assert validate_input(spec, {**ok, "extra": 1}) == ["unknown field 'extra'"]
    assert validate_input(spec, "not a dict")


def test_field_levels_straddle_boundaries(spec):
    lv = field_levels(spec.field("amount"), 10)
    assert (lv["min"], lv["below_min"], lv["max"], lv["above_max"]) == (1, 0, 1000, 1001)
    s = field_levels(spec.field("message"), "hi there")
    assert len(s["at_max_length"]) == 60 and len(s["over_max_length"]) == 61
    assert "choice=web" in field_levels(spec.field("channel"), "app")


def test_covering_array_hits_every_pair_and_beats_exhaustive():
    sizes = [5, 4, 6, 5, 3]
    rows = covering_array(sizes, seed=3)
    assert pairwise_coverage(sizes, rows) == 1.0
    assert len(rows) < 5 * 4 * 6 * 5 * 3 / 10
    assert len(rows) >= 6 * 5  # lower bound: product of the two largest factors
    assert covering_array(sizes, seed=3) == rows  # deterministic


def test_pairwise_coverage_partial():
    sizes = [2, 2]
    assert len(all_pairs(sizes)) == 4
    assert pairwise_coverage(sizes, [[0, 0], [1, 1]]) == 0.5


def test_rule_generator_covers_all_axes_and_is_deterministic(spec):
    a = RuleGenerator(spec, seed=1).generate()
    b = RuleGenerator(spec, seed=1).generate()
    assert [c.id for c in a] == [c.id for c in b]
    assert {c.axis for c in a} == set(AXES)
    cats = {(c.axis, c.category) for c in a}
    for want in [("boundary", "below_min"), ("boundary", "missing"), ("format", "homoglyph"),
                 ("format", "rtl"), ("semantic", "contradictory"), ("semantic", "out_of_scope"),
                 ("adversarial", "long_input"), ("combinatorial", "pairwise")]:
        assert want in cats
    assert all(c.provenance["generator"].startswith("rule:") for c in a)
    assert all(c.provenance["spec_fingerprint"] == spec.fingerprint for c in a)


def test_boundary_missing_case_drops_the_field(spec):
    cases = RuleGenerator(spec).boundary()
    miss = [c for c in cases if c.category == "missing" and c.field == "amount"]
    assert miss and "amount" not in miss[0].input


def test_semantic_contradiction_mentions_other_seed_value(spec):
    cases = RuleGenerator(spec).semantic()
    contra = next(c for c in cases if c.category == "contradictory")
    assert "actually" in contra.input["message"]


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": [1]}\n```') == {"a": [1]}
    assert extract_json('here: {"b": 2} trailing') == {"b": 2}
    with pytest.raises(ValueError):
        extract_json("no json at all")


def test_llm_generator_with_mock_produces_tagged_cases(spec):
    gen = LLMGenerator(spec, MockLLM(), n_per_axis=3)
    cases = gen.generate(["boundary", "semantic", "combinatorial"])
    assert len(cases) == 6 and not gen.errors
    assert all(c.provenance["generator"] == "llm:mock-1" for c in cases)
    sem = [c for c in cases if c.axis == "semantic"]
    assert all(c.field == "message" and c.levels.get("message") == c.category for c in sem)


def test_llm_generator_survives_garbage_reply(spec):
    gen = LLMGenerator(spec, MockLLM(responder=lambda m: "I cannot do that"), n_per_axis=2)
    assert gen.generate(["format"]) == []
    assert gen.errors and "unparseable" in gen.errors[0]
    gen = LLMGenerator(spec, MockLLM(responder=lambda m: '{"cases": [1, {"input": 3}]}'))
    assert gen.generate(["format"]) == [] and len(gen.errors) == 2


def test_validator_rejects_copies_duplicates_and_unknown_fields(spec):
    seed = dict(spec.seeds[0])
    near = {**seed, "message": "What is my balance today?!", "amount": 999}
    cases = [
        EdgeCase(input=dict(seed), axis="boundary", category="min"),
        EdgeCase(input={**seed, "amount": 999}, axis="boundary", category="max"),
        EdgeCase(input={**seed, "amount": 999}, axis="format", category="casing"),
        EdgeCase(input=near, axis="format", category="emoji"),
        EdgeCase(input={**seed, "bogus": 1}, axis="boundary", category="null"),
        EdgeCase(input={**seed, "amount": 0}, axis="boundary", category="below_min"),
    ]
    rep = Validator(spec, near_dup=0.8).run(cases)
    reasons = [r["reason"] for r in rep.rejected]
    assert reasons == ["copy_of_seed", "duplicate", "near_duplicate", "unknown_fields"]
    assert [c.category for c in rep.kept] == ["max", "below_min"]
    assert rep.kept[0].schema_valid and not rep.kept[1].schema_valid
    assert 0 < rep.kept[0].novelty < 1


def test_near_dup_does_not_merge_valid_and_invalid_boundary(spec):
    lv = field_levels(spec.field("message"), spec.seeds[0]["message"])
    cases = [EdgeCase(input={**spec.seeds[0], "message": lv[k]}, axis="boundary", category=k)
             for k in ("at_max_length", "over_max_length")]
    rep = Validator(spec, near_dup=0.5).run(cases)
    assert len(rep.kept) == 2


def test_min_novelty_filter(spec):
    c = EdgeCase(input={**spec.seeds[0], "amount": 11}, axis="boundary", category="x")
    assert Validator(spec, min_novelty=0.5).run([c]).rejected[0]["reason"] == "low_novelty"


def test_propose_labels_schema_oracle_and_review_flags(spec):
    s0 = spec.seeds[0]
    cases = [
        EdgeCase(input={**s0, "amount": 0}, axis="boundary", category="below_min"),
        EdgeCase(input={**s0, "message": "Send it"}, axis="format", category="casing"),
        EdgeCase(input={**s0, "message": "Send it maybe?"}, axis="semantic",
                 category="ambiguous"),
    ]
    kept = Validator(spec).run(cases).kept
    propose_labels(spec, kept, oracle=oracle)
    assert [c.label for c in kept] == ["invalid", "transfer", "transfer"]
    assert [c.label_source for c in kept] == ["schema", "oracle", "oracle"]
    assert [c.needs_human_review for c in kept] == [False, False, True]


def test_propose_labels_oracle_crash_and_llm_disagreement(spec):
    c1 = EdgeCase(input={**spec.seeds[0], "message": "x1"}, axis="format", category="a")
    c2 = EdgeCase(input={**spec.seeds[0], "message": "x2"}, axis="format", category="b")
    kept = Validator(spec).run([c1, c2]).kept

    def flaky(inp):
        if inp["message"] == "x1":
            raise KeyError("boom")
        return "balance"

    llm = MockLLM(responder=lambda m: '{"label": "other", "confidence": 0.9}')
    propose_labels(spec, kept, oracle=flaky, llm=llm)
    assert kept[0].label == "other" and kept[0].label_source == "llm"
    assert any(r.startswith("oracle_error") for r in kept[0].review_reasons)
    assert kept[1].label == "balance"
    assert kept[1].review_reasons == ["oracle_llm_disagree: llm=other"]


def test_llm_label_rejects_out_of_set_label(spec):
    llm = MockLLM(responder=lambda m: '{"label": "refund", "confidence": 1}')
    assert llm_label(spec, llm, spec.seeds[0]) == (None, 0.0)
    assert llm_label(spec, MockLLM(), spec.seeds[0])[0] in spec.labels


def test_coverage_pairwise_only_complete_with_combinatorial_axis(spec):
    gen = RuleGenerator(spec)
    one_factor = Validator(spec).run(gen.generate(["boundary", "format"])).kept
    rep = coverage_report(spec, one_factor)
    assert rep["pairwise_coverage"] < 0.5
    full = Validator(spec).run(gen.generate()).kept
    rep = coverage_report(spec, full)
    assert rep["pairwise_coverage"] == 1.0
    assert rep["by_axis"]["combinatorial"] > 0
    assert "pairwise_coverage_by_axis" in rep and rep["fields_untouched"] == []


def test_coverage_reports_missing_categories(spec):
    rep = coverage_report(spec, RuleGenerator(spec).semantic())
    assert "boundary" in rep["missing_categories"]
    assert "semantic" not in rep["missing_categories"]


def test_probe_classifies_pass_fail_crash():
    cases = [
        EdgeCase(input={"x": 1}, axis="boundary", category="a", label="1"),
        EdgeCase(input={"x": 2}, axis="boundary", category="a", label="3"),
        EdgeCase(input={"x": None}, axis="format", category="b", label="0"),
        EdgeCase(input={"x": 4}, axis="format", category="b"),
    ]
    res = probe(lambda inp: inp["x"] + 0, cases)
    assert [r.status for r in res] == ["pass", "fail", "crash", "unlabeled"]
    s = summarize(res)
    assert s["overall"]["fail"] == 1 and s["overall"]["crash"] == 1
    assert s["by_axis"]["boundary"]["failure_rate"] == 0.5
    assert "TypeError" in s["failures"][1]["error"]


def test_example_toy_system_breaks_on_edge_cases_but_not_seeds():
    from evalkit.edge_case_gen import load_spec

    spec = load_spec(EXAMPLE / "spec.yaml")
    ref = load_callable("expense_system.py:reference", EXAMPLE)
    naive = load_callable("expense_system.py:naive_triage", EXAMPLE)
    assert all(ref(s) == naive(s) for s in spec.seeds)
    kept = Validator(spec).run(RuleGenerator(spec).generate()).kept
    propose_labels(spec, kept, oracle=ref)
    s = summarize(probe(naive, kept))
    assert s["overall"]["crash"] > 0 and s["overall"]["fail"] > 0
    assert s["by_axis"]["format"]["crash"] > 0  # non-latin-1 text kills the audit log


def test_cli_end_to_end(tmp_path):
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="system"))
    spec_path = str(EXAMPLE / "spec.yaml")
    cand, gold = tmp_path / "c.jsonl", tmp_path / "g.jsonl"

    def run(*argv):
        a = parser.parse_args(["edge-case-gen", *argv])
        return a.func(a)

    assert run("generate", "--spec", spec_path, "--out", str(cand), "--llm", "mock") == 0
    rows = read_jsonl(cand)
    assert any(r["provenance"]["generator"].startswith("llm:") for r in rows)
    assert run("validate", "--spec", spec_path, "--in", str(cand), "--out", str(gold),
               "--oracle", "expense_system.py:reference") == 0
    golden = read_jsonl(gold)
    assert golden and all(r["label"] is not None or r["needs_human_review"] for r in golden)
    assert run("coverage", "--spec", spec_path, "--in", str(gold),
               "--json", str(tmp_path / "cov.json"), "--min-pairwise", "0.99") == 0
    assert json.loads((tmp_path / "cov.json").read_text())["pairwise_coverage"] == 1.0
    assert run("probe", "--spec", spec_path, "--in", str(gold),
               "--system", "expense_system.py:robust_triage", "--max-fail-rate", "0.0") == 0
    assert run("probe", "--spec", spec_path, "--in", str(gold),
               "--system", "expense_system.py:naive_triage", "--max-fail-rate", "0.1") == 1


def test_generate_rejects_unknown_axis(tmp_path):
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="system"))
    a = parser.parse_args(["edge-case-gen", "generate", "--spec", str(EXAMPLE / "spec.yaml"),
                           "--out", str(tmp_path / "x.jsonl"), "--axes", "boundary,vibes"])
    with pytest.raises(SystemExit):
        a.func(a)
