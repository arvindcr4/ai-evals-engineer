"""Turn a biased judge into a calibrated instrument.

Three mitigations, stacked and evaluated on held-out anchors:

1. **swap-and-aggregate** — judge both orders and average P(A wins), which
   cancels position bias by construction;
2. **length control** — a log-length-ratio term in a logistic correction fitted
   on anchors, so length no longer moves the verdict beyond what humans reward
   (the idea behind length-controlled win rates);
3. **self-family control** — a term for "one response is from the judge's own
   model family", absorbing self-preference.

The correction is a logistic regression of the human verdict on the judge's
swap-averaged logit plus those terms; the fitted weights are an auditable JSON
artefact that :class:`CalibratedJudge` applies at inference time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.calibrated_judge.anchors import Anchor
from evalkit.calibrated_judge.audit import (
    Judgment,
    decision_metrics,
    fit_logistic,
    label_from_p,
    position_metrics,
    self_family_diff,
    win_value,
)
from evalkit.calibrated_judge.judge import PairwiseJudge

FEATURES = ("bias", "judge_logit", "log_len_ratio", "self_family")
VARIANTS = {
    "swap + length": ("bias", "judge_logit", "log_len_ratio"),
    "swap + length + self (full)": FEATURES,
}


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.02, 0.98)
    return np.log(p / (1 - p))


def feature_row(ab_p_a: float, ba_p_a: float, log_len_ratio: float, self_diff: int) -> list[float]:
    return [1.0, float((_logit(ab_p_a) + _logit(ba_p_a)) / 2), log_len_ratio, float(self_diff)]


@dataclass
class CalibrationModel:
    weights: list[float]
    features: list[str]
    tie_band: float
    judge_family: str | None
    n_train: int

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        idx = [FEATURES.index(f) for f in self.features]
        z = np.asarray(rows, float)[:, idx] @ np.asarray(self.weights)
        return 1 / (1 + np.exp(-z))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationModel:
        return cls(**json.loads(Path(path).read_text()))


def _design(anchors: Sequence[Anchor], judgments: Sequence[Judgment],
            family: str | None) -> np.ndarray:
    by_id = {j.id: j for j in judgments}
    return np.array([feature_row(by_id[a.id].ab_p_a, by_id[a.id].ba_p_a, a.log_len_ratio,
                                 self_family_diff(a, family)) for a in anchors])


def _best_tie_band(p: np.ndarray, human: np.ndarray) -> float:
    grid = np.round(np.arange(0.0, 0.26, 0.01), 2)
    scores = [np.mean(label_from_p(p, b) == human) for b in grid]
    return float(grid[int(np.argmax(scores))])


def fit_calibration(anchors: Sequence[Anchor], judgments: Sequence[Judgment],
                    family: str | None, features: Sequence[str] = FEATURES,
                    l2: float = 1e-2) -> CalibrationModel:
    X = _design(anchors, judgments, family)
    idx = [FEATURES.index(f) for f in features]
    human = np.array([a.human_pref for a in anchors])
    decisive = human != "tie"
    y = np.array([win_value(h) for h in human])
    w = fit_logistic(X[decisive][:, idx], y[decisive], l2=l2)
    model = CalibrationModel(w.tolist(), list(features), 0.0, family, len(anchors))
    model.tie_band = _best_tie_band(model.predict_proba(X), human)
    return model


def split(anchors: Sequence[Anchor], train_frac: float, seed: int
          ) -> tuple[list[Anchor], list[Anchor]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(anchors))
    k = round(train_frac * len(anchors))
    return [anchors[i] for i in order[:k]], [anchors[i] for i in order[k:]]


def calibrate(anchors: Sequence[Anchor], judgments: Sequence[Judgment], family: str | None,
              train_frac: float = 0.6, seed: int = 0) -> tuple[CalibrationModel, dict[str, Any]]:
    """Fit on a train split, report raw vs mitigated judges on the held-out split."""
    by_id = {j.id: j for j in judgments}
    anchors = [a for a in anchors if a.id in by_id]
    train, test = split(anchors, train_frac, seed)
    if len(train) < 20 or len(test) < 20:
        raise ValueError("need at least 20 anchors in each of train and test")
    test_js = [by_id[a.id] for a in test]
    results: dict[str, Any] = {}
    results["raw judge (AB order)"] = decision_metrics(
        test, [j.ab_winner for j in test_js], [j.ab_p_a for j in test_js], family)
    swap_p = np.array([(j.ab_p_a + j.ba_p_a) / 2 for j in test_js])
    results["swap-aggregated"] = decision_metrics(test, label_from_p(swap_p, 0.02), swap_p,
                                                  family)
    X_test = _design(test, test_js, family)
    final: CalibrationModel | None = None
    for name, feats in VARIANTS.items():
        model = fit_calibration(train, judgments, family, feats)
        p = model.predict_proba(X_test)
        results[name] = decision_metrics(test, label_from_p(p, model.tie_band), p, family)
        results[name]["tie_band"] = model.tie_band
        final = model
    assert final is not None
    report = {"judge_family": family, "n_train": len(train), "n_test": len(test),
              "position_test": position_metrics(test_js),
              "model": asdict(final), "variants": results}
    return final, report


def calibration_markdown(rep: dict[str, Any]) -> str:
    def f(x: Any, pct: bool = False, sign: bool = False) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        if pct:
            return f"{x * 100:+.1f} pts" if sign else f"{x:.1%}"
        return f"{x:+.2f}" if sign else f"{x:.3f}"

    lines = ["# Judge calibration (held-out anchors)", "",
             (f"Fitted on {rep['n_train']} anchors, evaluated on {rep['n_test']} held-out anchors "
             f"· judge family `{rep['judge_family'] or 'n/a'}`"), "",
             "| Judge | Agreement | κ | Verbosity excess | Length coef | Self-pref gap | ECE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in rep["variants"].items():
        v, sp = m["verbosity"], m["self_preference"]
        lines.append(f"| {name} | {f(m['agreement'], True)} | {f(m['cohen_kappa'])} | "
                     f"{f(v['excess'], True, True)} | {f(v['length_logit_coef'], sign=True)} | "
                     f"{f(sp.get('gap'), True, True)} | {f(m['calibration']['ece'])} |")
    mw = rep["model"]
    terms = ", ".join(f"{n}={w:+.2f}" for n, w in zip(mw["features"], mw["weights"]))
    lines += ["", f"Correction: `P(A wins) = σ({terms})`, tie band ±{mw['tie_band']:.2f}.",
              ("Position bias is removed by construction (both orders are averaged); raw "
              f"consistency on the test split was {rep['position_test']['consistency']:.1%}.")]
    return "\n".join(lines) + "\n"


class CalibratedJudge:
    """Production wrapper: both orders + fitted correction → (winner, P(A wins))."""

    def __init__(self, judge: PairwiseJudge, model: CalibrationModel):
        self.judge, self.model = judge, model

    def compare(self, prompt: str, response_a: str, response_b: str,
                model_a: str = "", model_b: str = "") -> tuple[str, float]:
        ab = self.judge.judge(prompt, response_a, response_b, swap=False)
        ba = self.judge.judge(prompt, response_a, response_b, swap=True)
        anchor = Anchor("_", prompt, response_a, response_b, model_a, model_b, "tie")
        row = feature_row(ab.p_a, ba.p_a, anchor.log_len_ratio,
                          self_family_diff(anchor, self.model.judge_family))
        p = float(self.model.predict_proba(np.array([row]))[0])
        return str(label_from_p(np.array([p]), self.model.tie_band)[0]), p
