import json
from pathlib import Path

import pytest

from evalkit.cli import main
from evalkit.core.llm import MockLLM
from evalkit.regression_gate import (
    CaseResult,
    GatePolicy,
    RunResult,
    Scorer,
    Suite,
    evaluate_gate,
    expand_cases,
    run_suite,
)
from evalkit.regression_gate.suite import Case

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "04-regression-gate"


def _case(expected: dict) -> Case:
    return Case("c", "c", {}, expected)


def test_expand_cases_crosses_param_matrix():
    cases = expand_cases([{"id": "a", "input": {"x": 1}}, {"id": "b", "input": {"x": 2}}],
                         {"tier": ["fast", "slow"], "lang": ["en"]})
    assert [c.case_id for c in cases] == [
        "a[lang=en,tier=fast]", "a[lang=en,tier=slow]",
        "b[lang=en,tier=fast]", "b[lang=en,tier=slow]"]
    assert cases[1].params == {"lang": "en", "tier": "slow"} and cases[1].item_id == "a"
    with pytest.raises(ValueError):
        expand_cases([{"id": "a"}, {"id": "a"}], {})


@pytest.mark.parametrize("scorer,output,expected,ok", [
    (Scorer("exact", "label"), {"label": " Billing "}, {"label": "billing"}, True),
    (Scorer("exact", "label", case_sensitive=True), {"label": "Billing"}, {"label": "billing"},
     False),
    (Scorer("exact", "label"), {"other": 1}, {"label": "x"}, False),
    (Scorer("contains", value=["refund", "order"]), "Refund for ORDER 7", {}, True),
    (Scorer("contains", value=["refund", "order"]), "Refund pending", {}, False),
    (Scorer("regex", pattern=r"^ORD-\d{4}$", field="id"), {"id": "ORD-1234"}, {}, True),
    (Scorer("json_field", "a.b", expected_field="v"), '```json\n{"a": {"b": 3}}\n```',
     {"v": 3}, True),
    (Scorer("json_field", "a.b", expected_field="v"), "not json", {"v": 3}, False),
    (Scorer("numeric", "amt", tolerance=0.01), {"amt": 10.004}, {"amt": 10}, True),
    (Scorer("numeric", "amt", tolerance=0.05, relative=True), {"amt": 11}, {"amt": 10}, False),
    (Scorer("numeric", "amt"), {"amt": None}, {"amt": None}, True),
    (Scorer("numeric", "amt"), {"amt": 3.0}, {"amt": None}, False),
])
def test_scorers(scorer, output, expected, ok):
    assert scorer.score(output, _case(expected))[0] is ok


def test_unknown_scorer_and_gate_key_rejected():
    with pytest.raises(ValueError):
        Scorer("fuzzy")
    with pytest.raises(ValueError):
        GatePolicy.from_dict({"max_drop": 0.1})


def _suite(n: int = 40) -> Suite:
    items = [{"id": f"i{k}", "input": {"x": k}, "expected": {"y": k * 2}} for k in range(n)]
    return Suite.from_dict({"name": "toy", "cases": items, "params": {"p": [0, 1]},
                            "scorers": [{"type": "exact", "field": "y"}]})


def test_crashing_case_fails_but_run_continues():
    def target(inp, p):
        if inp["x"] == 3:
            raise RuntimeError("boom")
        return {"y": inp["x"] * 2}

    res = run_suite(_suite(10), target=target)
    bad = [c for c in res.cases if not c.success]
    assert len(bad) == 2 and all("RuntimeError: boom" in c.error for c in bad)
    assert res.summary()["errors"] == 2 and res.success_rate == pytest.approx(0.9)


def test_llm_target_uses_prompt_template_and_reported_latency():
    suite = Suite.from_dict({"cases": [{"id": "q", "input": {"text": "2+2"},
                                        "expected": {"output": "4"}}],
                             "target": {"prompt": "Compute {text} ({style})"},
                             "params": {"style": ["terse"]},
                             "scorers": [{"type": "contains"}]})
    seen = []

    def responder(messages):
        seen.append(messages[-1]["content"])
        return "The answer is 4."

    res = run_suite(suite, llm=MockLLM(responder=responder, latency_s=0.25))
    assert seen == ["Compute 2+2 (terse)"]
    assert res.cases[0].success and res.cases[0].latency_s == 0.25


def _run(success: list[bool], latency: float = 0.01, items_per: int = 1) -> RunResult:
    return RunResult("s", [CaseResult(f"c{i}", f"i{i // items_per}", {}, s, {"ok": s}, latency)
                           for i, s in enumerate(success)])


def test_gate_blocks_significant_drop_and_reports_new_failures():
    base = _run([True] * 90 + [False] * 10)
    cand = _run([True] * 70 + [False] * 30)
    rep = evaluate_gate(base, cand)
    assert not rep.passed and rep.exit_code == 1
    assert "task success dropped 20.0 pts" in rep.reasons[0]
    assert len(rep.newly_failing) == 20
    md = rep.to_markdown()
    assert "FAIL" in md and "20 newly failing cases" in md and "<!-- evalkit" in md


def test_gate_tolerates_small_or_insignificant_drops():
    base = _run([True] * 45 + [False] * 5)
    cand = _run([True] * 43 + [False] * 7)  # -4 pts on 50 cases: over threshold, but noise
    rep = evaluate_gate(base, cand)
    assert rep.passed and rep.comparison is not None and not rep.comparison.significant
    strict = evaluate_gate(base, cand, GatePolicy(require_significance=False))
    assert not strict.passed
    within = evaluate_gate(_run([True] * 100), _run([True] * 99 + [False]),
                           GatePolicy(require_significance=False))
    assert within.passed  # 1 pt drop is under the 2 pt threshold


def test_gate_clusters_param_variants_of_the_same_item():
    # 8 items regress, each in 4 parameter variants: 32 failing cases but only 8 clusters.
    base = _run([True] * 200, items_per=4)
    cand = _run([False] * 32 + [True] * 168, items_per=4)
    rep = evaluate_gate(base, cand)
    assert rep.comparison.n_clusters == 50 and rep.comparison.test == "cluster-permutation"
    assert not rep.passed


def test_latency_gate_uses_relative_and_absolute_floor():
    ok = [True] * 50
    slow = evaluate_gate(_run(ok, 0.100), _run(ok, 0.150))
    assert not slow.passed and "p95 latency rose +50%" in slow.reasons[0]
    jitter = evaluate_gate(_run(ok, 0.001), _run(ok, 0.002))  # +100% but only 1 ms
    assert jitter.passed


def test_missing_and_added_cases_are_reported():
    base = _run([True] * 30)
    cand = RunResult("s", base.cases[:25] + [CaseResult("new", "new", {}, True, {}, 0.01)])
    rep = evaluate_gate(base, cand)
    assert rep.passed and len(rep.missing_cases) == 5 and rep.added_cases == ["new"]
    assert not evaluate_gate(base, RunResult("s", [])).passed


def test_example_suite_pass_and_fail(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = ["regression-gate", "run", "--suite", str(EXAMPLE / "suite.yaml"),
            "--baseline", str(EXAMPLE / "baseline.json"), "--report", str(tmp_path / "r.md"),
            "--out", str(tmp_path / "res.json")]
    monkeypatch.setenv("SUT_VERSION", "1")
    assert main(args) == 0
    assert "PASS" in summary.read_text()
    monkeypatch.setenv("SUT_VERSION", "2")
    assert main(args) == 1
    report = (tmp_path / "r.md").read_text()
    assert "FAIL" in report and "'shipping' → 'billing'" in report
    saved = json.loads((tmp_path / "res.json").read_text())
    assert saved["summary"]["n_cases"] == 80 and saved["summary"]["success_rate"] < 0.8
    capsys.readouterr()
