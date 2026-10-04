"""Regression tests for failures seen running edge_case_gen against real DeepSeek output.

Each MockLLM / fake reply below mimics a reply shape deepseek-flash actually
returned (Oct 2026 run, examples/12-edge-case-generator/real_run.sh).
"""

import json
from pathlib import Path

from evalkit.core.llm import Completion, MockLLM
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
from evalkit.edge_case_gen.claims import claim_holds
from evalkit.edge_case_gen.llm_gen import salvage_cases
from evalkit.edge_case_gen.spec import load_spec

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "12-edge-case-generator"


def expense():
    return load_spec(EXAMPLE / "spec.yaml")


SEED0 = {"employee_note": "Team lunch with the Acme client after the quarterly review.",
         "amount": 142.5, "currency": "USD", "expense_date": "2026-03-14",
         "category": "meals", "has_receipt": True}


def case(category, axis="format", desc="", **changes):
    return {"input": {**SEED0, **changes}, "category": category, "description": desc}


def reply(*cases):
    return json.dumps({"cases": list(cases)})


class FakeLLM:
    """Replays canned replies and records every call (messages + kwargs)."""

    def __init__(self, replies, finish=None, model="deepseek-flash"):
        self.replies, self.finish, self.model = list(replies), list(finish or []), model
        self.calls = []

    def complete(self, messages, **kw):
        self.calls.append((messages, kw))
        text = self.replies.pop(0) if self.replies else '{"cases": []}'
        fr = self.finish.pop(0) if self.finish else "stop"
        return Completion(text=text, model=self.model, tokens_in=1000, tokens_out=500,
                          cost_usd=0.001, raw={"choices": [{"finish_reason": fr}]})


def test_phantom_perturbations_are_rejected_by_claim_check():
    # deepseek-flash described invisible characters / exact lengths it never emitted.
    spec = expense()
    cands = [
        case("zero_width", desc="Zero-width space (U+200B) after 'Caf\u00e9'",
             employee_note="Client dinner at Caf\u00e9\u202fM\u00fcnchen \u2014 3 attendees."),
        case("combining_marks", desc="combining acute (U+0301) on e",
             employee_note="Lunch with client after review."),
        case("homoglyph", desc="em-dash homoglyph",
             employee_note="Lunch with the Acme client \u2014 receipt attached"),
        case("at_max_length", axis="boundary", desc="note is exactly 280 characters",
             employee_note="Note with exactly 280 characters. " + "Lorem ipsum " * 37),
        case("instruction_like", axis="adversarial", desc="actually a bad currency",
             employee_note="Taxi ride to the airport.", currency="XEU"),
        # genuine ones survive
        case("zero_width", employee_note="Lunch with the \u200bAcme\u200b client"),
        case("rtl", axis="adversarial",
             employee_note="Receipt attached: \u202ereceipt\u202c please approve."),
    ]
    cases = [EdgeCase(input=c["input"], axis=c.get("axis", "format"), category=c["category"],
                      description=c["description"]) for c in cands]
    for c, raw in zip(cases, cands):
        c.axis = "boundary" if raw["category"] == "at_max_length" else c.axis
    rep = Validator(spec).run(cases)
    reasons = [r["reason"] for r in rep.rejected]
    assert reasons == ["claim_not_in_input"] * 5
    assert [c.category for c in rep.kept] == ["zero_width", "rtl"]
    # the check can be switched off
    assert len(Validator(spec, check_claims=False).run(cases).kept) == 7


def test_every_rule_generated_case_passes_its_own_claim_check():
    spec = expense()
    for seed in range(4):
        for c in RuleGenerator(spec, seed=seed).generate():
            assert claim_holds(spec, c.category, c.input, c.field) is not False, c.category


def test_above_max_claim_holds_even_when_type_is_also_wrong():
    spec = spec_from_dict({
        "name": "t", "fields": [{"name": "n", "type": "integer", "min": 1, "max": 10}],
        "seeds": [{"n": 3}]})
    assert claim_holds(spec, "above_max", {"n": 15.0}) is True
    assert claim_holds(spec, "below_min", {"n": 0.5}) is True
    assert claim_holds(spec, "above_max", {"n": 5}) is False


def test_repair_round_replaces_phantoms_and_drops_resent_cases():
    spec = expense()
    good1 = case("emoji", employee_note="Team lunch \U0001f355 with Acme")
    good2 = case("receipt_threshold_exact", axis="boundary", amount=75)
    phantom = case("date_format", desc="date written as 14/03/2026")  # input == seed
    fixed = case("date_format", desc="date written as 14/03/2026", expense_date="14/03/2026")
    # The real model answered the repair prompt by resending the whole list plus the fix.
    llm = FakeLLM([reply(good1, good2, phantom), reply(good1, good2, fixed)])
    gen = LLMGenerator(spec, llm, n_per_axis=3, temperature=0)
    out = gen.generate_axis("format")
    assert [c.category for c in out] == ["emoji", "receipt_threshold_exact", "date_format",
                                         "date_format"]
    assert out[-1].input["expense_date"] == "14/03/2026" and out[-1].provenance["repaired"]
    assert gen.stats == {"phantom": 1, "repaired": 1, "resent": 2}
    repair_msgs = llm.calls[1][0]
    assert repair_msgs[-2]["role"] == "assistant"
    assert "14/03/2026" in repair_msgs[-1]["content"] and "1 of those" in repair_msgs[-1]["content"]
    assert gen.usage["calls"] == 2 and abs(gen.usage["cost_usd"] - 0.002) < 1e-12
    # temperature and the max-token cap reach the model
    assert llm.calls[0][1] == {"temperature": 0, "max_tokens": 4000}
    # validator then drops the original phantom
    rep = Validator(spec).run(out)
    assert [r["reason"] for r in rep.rejected] == ["copy_of_seed"]


def test_repair_round_stops_when_nothing_is_phantom():
    llm = FakeLLM([reply(case("emoji", employee_note="Lunch \U0001f355"))])
    gen = LLMGenerator(expense(), llm, n_per_axis=1)
    assert len(gen.generate_axis("format")) == 1 and len(llm.calls) == 1


def test_truncated_reply_with_broken_escape_is_salvaged():
    # Degenerate repetition hit the max-token cap mid-\u escape.
    good = [case("fullwidth", employee_note="\uff23lient dinner"),
            case("emoji", employee_note="Lunch \U0001f355")]
    text = reply(*good)[:-2] + ', {"input": {"employee_note": "\u20b9\\u208'
    llm = FakeLLM([text], finish=["length"])
    gen = LLMGenerator(spec := expense(), llm, n_per_axis=3, repair_rounds=0)
    out = gen.generate_axis("format")
    assert [c.category for c in out] == ["fullwidth", "emoji"]
    assert any("truncated" in e for e in gen.errors)
    assert any("salvaged 2" in e for e in gen.errors)
    assert salvage_cases("no json here") == []
    assert len(Validator(spec).run(out).kept) == 2


def test_prompt_asks_for_escapes_and_real_case_edges():
    gen = LLMGenerator(expense(), MockLLM(), n_per_axis=2)
    body = gen.prompt("format")[1]["content"]
    assert "\\u200b" in body and "MUST actually contain" in body
    # an escaped reply decodes to the real character, so the claim holds
    obj = json.loads('{"t": "a\\u200bb"}')
    assert claim_holds(expense(), "zero_width", {**SEED0, "employee_note": obj["t"]})


def test_label_llm_usage_is_accounted_and_fenced_reply_parsed():
    spec = expense()
    llm = MockLLM(responder=lambda m: '```json\n{"label": "approve", "confidence": 0.95}\n```')
    c = EdgeCase(input={**SEED0, "employee_note": "Lunch \U0001f355"}, axis="format",
                 category="emoji", schema_valid=True)
    usage = propose_labels(spec, [c], llm=llm)
    assert c.label == "approve" and c.label_source == "llm" and not c.needs_human_review
    assert usage["calls"] == 1 and usage["cost_usd"] > 0


def test_coverage_reports_llm_novelty_vs_rules_and_new_categories():
    spec = expense()
    rules = RuleGenerator(spec, seed=1).generate(["boundary", "format"])
    llm = [EdgeCase(input={**SEED0, "amount": 75, "has_receipt": False}, axis="boundary",
                    category="receipt_threshold_exact",
                    provenance={"generator": "llm:deepseek-flash"}, novelty=0.4),
           EdgeCase(input={**SEED0, "employee_note": "Lunch \U0001f355"}, axis="format",
                    category="emoji", provenance={"generator": "llm:deepseek-flash"},
                    novelty=0.6)]
    lv = coverage_report(spec, rules + llm)["llm_vs_rules"]
    assert lv["n_llm"] == 2 and lv["n_rule"] == len(rules)
    assert lv["new_categories"] == ["boundary/receipt_threshold_exact"]
    assert lv["llm_cases_in_new_categories"] == 1
    assert 0 < lv["novelty_vs_rules_min"] <= lv["novelty_vs_rules_mean"] < 1
    assert lv["novelty_vs_seeds_mean_llm"] == 0.5
    assert coverage_report(spec, rules)["llm_vs_rules"] is None


def test_probe_splits_failure_rate_by_generator():
    cs = [EdgeCase(input={"x": 1}, axis="a", category="c", label="ok",
                   provenance={"generator": "llm:deepseek-flash"}),
          EdgeCase(input={"x": 2}, axis="a", category="c", label="no",
                   provenance={"generator": "rule:boundary"})]
    s = summarize(probe(lambda inp: "ok", cs))
    assert s["by_generator"]["llm"]["failure_rate"] == 0.0
    assert s["by_generator"]["rule"]["failure_rate"] == 1.0
