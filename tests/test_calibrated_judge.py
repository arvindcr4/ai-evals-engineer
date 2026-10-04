import json
import math

import numpy as np
import pytest

from evalkit.calibrated_judge import (
    CalibratedJudge,
    CalibrationModel,
    PairwiseJudge,
    SimulatedJudge,
    audit,
    calibrate,
    cohen_kappa,
    collect_judgments,
    expected_calibration_error,
    fit_logistic,
    make_anchors,
    parse_pairwise,
    parse_score,
)
from evalkit.calibrated_judge.anchors import family_of
from evalkit.cli import main
from evalkit.core.llm import MockLLM

UNBIASED = {"position_bias": 0.0, "verbosity_bias": 0.0, "self_bias": 0.0, "overconfidence": 1.0}


@pytest.fixture(scope="module")
def anchors():
    return make_anchors(500, seed=3)


@pytest.fixture(scope="module")
def biased_judgments(anchors):
    return collect_judgments(anchors, PairwiseJudge(SimulatedJudge().as_llm()))


def test_anchors_are_seeded_and_length_is_independent_of_quality(anchors):
    again = make_anchors(500, seed=3)
    assert [a.to_dict() for a in anchors[:20]] == [a.to_dict() for a in again[:20]]
    facts = np.array([a.meta["facts_a"] for a in anchors] + [a.meta["facts_b"] for a in anchors])
    words = np.log([len(a.response_a.split()) for a in anchors]
                   + [len(a.response_b.split()) for a in anchors])
    assert abs(np.corrcoef(facts, words)[0, 1]) < 0.15
    better_a = [a for a in anchors if a.meta["facts_a"] >= a.meta["facts_b"] + 2]
    assert np.mean([a.human_pref == "a" for a in better_a]) > 0.95
    assert {family_of(a.model_a) for a in anchors} == {"atlas", "nova", "orion"}
    assert 0.05 < np.mean([a.human_pref == "tie" for a in anchors]) < 0.25


@pytest.mark.parametrize("text,expected", [
    ('{"winner": "B", "confidence": 0.83}', ("B", 0.83)),
    ('Sure.\n```json\n{"winner": "tie", "confidence": 0.6}\n```', ("tie", 0.6)),
    ('{"winner": "A", "confidence": 3}', ("A", 1.0)),
    ("After reflection: [[A]]", ("A", 0.75)),
    ("I cannot decide.", ("tie", 0.5)),
])
def test_parse_pairwise(text, expected):
    assert parse_pairwise(text) == expected


def test_parse_score():
    assert parse_score('{"score": 7}') == 7.0
    assert parse_score("I'd give it 4 out of ten") == 4.0
    assert parse_score("no idea") is None


def test_swap_maps_verdict_back_to_anchor_labels():
    always_first = MockLLM(responder=lambda m: '{"winner": "A", "confidence": 0.9}')
    j = PairwiseJudge(always_first)
    ab, ba = j.judge("q", "x", "y"), j.judge("q", "x", "y", swap=True)
    assert (ab.winner, ab.p_a) == ("a", 0.9)
    assert ba.winner == "b" and ba.p_a == pytest.approx(0.1)


def test_judge_prompt_hides_model_names(anchors):
    seen = []
    PairwiseJudge(MockLLM(responder=lambda m: seen.append(m[-1]["content"]) or "{}")) \
        .judge_anchor(anchors[0])
    assert anchors[0].model_a not in seen[0] and anchors[0].response_a in seen[0]


def test_stats_helpers():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    ece_ok, _ = expected_calibration_error(conf, rng.uniform(size=20000) < conf)
    ece_bad, table = expected_calibration_error(conf, rng.uniform(size=20000) < 0.6)
    assert ece_ok < 0.02 and ece_bad > 0.1 and len(table) == 10
    X = np.column_stack([np.ones(5000), rng.normal(size=5000)])
    y = rng.uniform(size=5000) < 1 / (1 + np.exp(-(0.5 + 2.0 * X[:, 1])))
    assert fit_logistic(X, y.astype(float)) == pytest.approx([0.5, 2.0], abs=0.15)


def test_unbiased_judge_audits_clean(anchors):
    js = collect_judgments(anchors, PairwiseJudge(SimulatedJudge(**UNBIASED).as_llm()))
    rep = audit(anchors, js, "nova")
    assert abs(rep["position"]["position_bias"]) < 0.04
    assert abs(rep["raw"]["verbosity"]["length_logit_coef"]) < 1.0
    assert abs(rep["raw"]["verbosity"]["excess"]) < 0.04
    assert abs(rep["raw"]["self_preference"]["gap"]) < 0.06


def test_audit_recovers_injected_biases(anchors, biased_judgments):
    rep = audit(anchors, biased_judgments, "nova")
    assert rep["position"]["position_bias"] > 0.08
    assert rep["position"]["consistency"] < 0.8
    assert rep["raw"]["verbosity"]["length_logit_coef"] > 2.0
    assert rep["raw"]["verbosity"]["excess"] > 0.05
    assert rep["raw"]["self_preference"]["gap"] > 0.08
    # Self-preference is relative to the judge's family: audited as "atlas" it vanishes.
    other = audit(anchors, biased_judgments, "atlas")["raw"]["self_preference"]["gap"]
    assert other < rep["raw"]["self_preference"]["gap"] - 0.08


def test_verbosity_measure_is_monotone_in_injected_bias(anchors):
    coefs = []
    for vb in (0.0, 1.5, 3.0):
        sim = SimulatedJudge(**{**UNBIASED, "verbosity_bias": vb})
        rep = audit(anchors, collect_judgments(anchors, PairwiseJudge(sim.as_llm())), "nova")
        coefs.append(rep["raw"]["verbosity"]["length_logit_coef"])
    assert coefs[0] < coefs[1] < coefs[2]


def test_calibration_improves_held_out_agreement_and_removes_bias(anchors, biased_judgments):
    model, rep = calibrate(anchors, biased_judgments, "nova", train_frac=0.6, seed=0)
    raw, full = rep["variants"]["raw judge (AB order)"], rep["variants"][
        "swap + length + self (full)"]
    assert rep["n_test"] == 200
    assert full["agreement"] >= raw["agreement"] + 0.04
    assert full["cohen_kappa"] > raw["cohen_kappa"]
    assert abs(full["verbosity"]["length_logit_coef"]) < 0.5 * raw["verbosity"][
        "length_logit_coef"]
    assert abs(full["self_preference"]["gap"]) < raw["self_preference"]["gap"]
    w = dict(zip(model.features, model.weights))
    assert w["judge_logit"] > 0 and w["log_len_ratio"] < 0 and w["self_family"] < 0


def test_calibrated_judge_round_trip(tmp_path, anchors, biased_judgments):
    model, _ = calibrate(anchors, biased_judgments, "nova")
    model.save(tmp_path / "m.json")
    loaded = CalibrationModel.load(tmp_path / "m.json")
    assert loaded == model
    cj = CalibratedJudge(PairwiseJudge(SimulatedJudge().as_llm()), loaded)
    a = anchors[0]
    winner, p = cj.compare(a.prompt, a.response_a, a.response_b, a.model_a, a.model_b)
    assert winner in ("a", "b", "tie") and 0 < p < 1


def test_calibrate_rejects_tiny_anchor_sets(anchors, biased_judgments):
    with pytest.raises(ValueError):
        calibrate(anchors[:30], biased_judgments, "nova")


def test_cli_end_to_end(tmp_path, capsys):
    anc, js = tmp_path / "a.jsonl", tmp_path / "j.jsonl"
    assert main(["calibrated-judge", "make-anchors", "--n", "150", "--out", str(anc)]) == 0
    assert main(["calibrated-judge", "audit", "--anchors", str(anc), "--judgments", str(js),
                 "--pointwise", "--out-json", str(tmp_path / "audit.json")]) == 0
    rep = json.loads((tmp_path / "audit.json").read_text())
    assert rep["n_anchors"] == 150 and rep["pointwise"]["n"] == 300
    assert not math.isnan(rep["raw"]["cohen_kappa"])
    assert main(["calibrated-judge", "calibrate", "--anchors", str(anc), "--judgments",
                 str(js), "--out", str(tmp_path / "m.json")]) == 0
    out = capsys.readouterr().out
    assert "Judge calibration (held-out anchors)" in out
    assert json.loads((tmp_path / "m.json").read_text())["features"][0] == "bias"
