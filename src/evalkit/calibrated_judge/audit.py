"""Measure a judge against human anchors: agreement and four biases.

Every anchor is judged twice, in the original order (AB) and swapped (BA), so
position effects can be separated from content. :func:`decision_metrics`
scores *any* set of decisions — the raw judge, the swap-aggregated judge or a
calibrated one — with the same yardsticks, which is what lets
:mod:`evalkit.calibrated_judge.calibrate` report before/after numbers.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from evalkit.calibrated_judge.anchors import Anchor, family_of
from evalkit.calibrated_judge.judge import PairwiseJudge, PointwiseJudge
from evalkit.core.trajectory import read_jsonl, write_jsonl

LABELS = ("a", "b", "tie")
LENGTH_BUCKETS = (0.0, 0.2, 0.5, 1.0, math.inf)


@dataclass
class Judgment:
    id: str
    ab_winner: str
    ab_p_a: float
    ab_conf: float
    ba_winner: str
    ba_p_a: float
    ba_conf: float
    score_a: float | None = None
    score_b: float | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _judge_one(a: Anchor, judge: PairwiseJudge, pointwise: PointwiseJudge | None) -> Judgment:
    ab, ba = judge.judge_anchor(a, swap=False), judge.judge_anchor(a, swap=True)
    sa = pointwise.score(a.prompt, a.response_a) if pointwise else None
    sb = pointwise.score(a.prompt, a.response_b) if pointwise else None
    return Judgment(a.id, ab.winner, ab.p_a, ab.confidence, ba.winner, ba.p_a,
                    ba.confidence, sa, sb)


def collect_judgments(anchors: Iterable[Anchor], judge: PairwiseJudge,
                      pointwise: PointwiseJudge | None = None, workers: int = 1,
                      progress: Callable[[int, int], None] | None = None) -> list[Judgment]:
    """Judge every anchor in both orders (plus pointwise scores), in anchor order.

    With ``workers`` > 1 anchors are judged concurrently (API judges are
    latency-bound). An anchor whose judge calls still fail after the client's
    retries is dropped and counted in ``judge.usage.errors`` instead of aborting
    a run that has already paid for hundreds of calls.
    """
    anchors = list(anchors)
    results: list[Judgment | None] = [None] * len(anchors)

    def run(i: int) -> None:
        try:
            results[i] = _judge_one(anchors[i], judge, pointwise)
        except Exception as e:  # noqa: BLE001 — any provider/transport failure
            judge.usage.error()
            print(f"judge failed on {anchors[i].id}: {type(e).__name__}: {e}"[:300],
                  file=sys.stderr)

    if workers <= 1:
        for i in range(len(anchors)):
            run(i)
            if progress:
                progress(i + 1, len(anchors))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, _ in enumerate(as_completed(pool.submit(run, i)
                                                  for i in range(len(anchors))), 1):
                if progress:
                    progress(done, len(anchors))
    return [j for j in results if j is not None]


def save_judgments(path: str, rows: list[Judgment]) -> None:
    write_jsonl(path, rows)


def load_judgments(path: str) -> list[Judgment]:
    return [Judgment(**r) for r in read_jsonl(path)]


# --- statistics ----------------------------------------------------------------


def cohen_kappa(y1: Sequence[str], y2: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    a, b = np.asarray(y1), np.asarray(y2)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == lab)) * float(np.mean(b == lab)) for lab in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton/IRLS; ``y`` may be fractional (ties = 0.5)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.zeros(X.shape[1])
    reg = l2 * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) + reg @ w
        hess = X.T @ (X * (p * (1 - p))[:, None]) + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def expected_calibration_error(conf: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    """ECE over equal-width confidence bins on [0.5, 1] (pairwise confidences)."""
    c, k = np.asarray(conf, float), np.asarray(correct, float)
    if len(c) == 0:
        return float("nan"), []
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1]), 0, n_bins - 1)
    ece, table = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(c[m].mean() - k[m].mean())
        ece += m.mean() * gap
        table.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                      "n": int(m.sum()), "confidence": float(c[m].mean()),
                      "accuracy": float(k[m].mean())})
    return float(ece), table


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def win_value(label: str) -> float:
    return {"a": 1.0, "b": 0.0}.get(label, 0.5)


def label_from_p(p_a: np.ndarray, tie_band: float) -> np.ndarray:
    return np.where(np.abs(p_a - 0.5) <= tie_band, "tie", np.where(p_a > 0.5, "a", "b"))


# --- metrics ---------------------------------------------------------------------


def self_family_diff(anchor: Anchor, family: str | None) -> int:
    """+1 if only A is from the judge's family, −1 if only B, else 0."""
    if not family:
        return 0
    return int(family_of(anchor.model_a) == family) - int(family_of(anchor.model_b) == family)


def decision_metrics(anchors: Sequence[Anchor], winners: Sequence[str],
                     p_a: Sequence[float], family: str | None) -> dict[str, Any]:
    """Agreement, kappa, verbosity, self-preference and calibration of one decision set."""
    human = np.array([a.human_pref for a in anchors])
    w = np.asarray(winners)
    p = np.asarray(p_a, float)
    llr = np.array([a.log_len_ratio for a in anchors])
    selfd = np.array([self_family_diff(a, family) for a in anchors])
    hsign = np.array([win_value(h) * 2 - 1 for h in human])

    decisive = (w != "tie") & (np.abs(llr) > 1e-9)
    longer_is_a = llr > 0
    both = decisive & (human != "tie")
    judge_longer = float(np.mean((w == "a") == longer_is_a, where=decisive)) if decisive.any() \
        else float("nan")
    human_longer = float(np.mean((human == "a") == longer_is_a, where=both)) if both.any() \
        else float("nan")
    buckets = []
    for lo, hi in itertools.pairwise(LENGTH_BUCKETS):
        m = both & (np.abs(llr) >= lo) & (np.abs(llr) < hi)
        if m.sum() == 0:
            continue
        buckets.append({
            "abs_log_len_ratio": f"[{lo:g}, {hi:g})", "n": int(m.sum()),
            "judge_prefers_longer": float(np.mean((w[m] == "a") == longer_is_a[m])),
            "human_prefers_longer": float(np.mean((human[m] == "a") == longer_is_a[m])),
        })
    coef = float("nan")
    mask = w != "tie"
    if mask.sum() > 10:
        X = np.column_stack([np.ones(mask.sum()), hsign[mask], llr[mask], selfd[mask]])
        coef = float(fit_logistic(X, (w[mask] == "a").astype(float), l2=1e-2)[2])

    own = selfd != 0
    sp: dict[str, Any] = {"n": int(own.sum())}
    if own.any():
        own_is_a = selfd[own] > 0
        jw = np.array([win_value(x) for x in w[own]])
        hw = np.array([win_value(x) for x in human[own]])
        jw = np.where(own_is_a, jw, 1 - jw)
        hw = np.where(own_is_a, hw, 1 - hw)
        sp.update(judge_own_win_rate=float(jw.mean()), human_own_win_rate=float(hw.mean()),
                  gap=float(jw.mean() - hw.mean()))

    conf_mask = (w != "tie") & (human != "tie")
    ece, table = expected_calibration_error(np.maximum(p, 1 - p)[conf_mask],
                                            (w == human)[conf_mask])
    return {
        "n": len(anchors),
        "agreement": float(np.mean(w == human)),
        "agreement_decisive": float(np.mean(w[both] == human[both])) if both.any()
        else float("nan"),
        "cohen_kappa": cohen_kappa(human, w),
        "tie_rate": float(np.mean(w == "tie")),
        "verbosity": {"judge_prefers_longer": judge_longer,
                      "human_prefers_longer": human_longer,
                      "excess": judge_longer - human_longer,
                      "length_logit_coef": coef, "buckets": buckets},
        "self_preference": sp,
        "calibration": {"ece": ece, "reliability": table},
    }


def position_metrics(judgments: Sequence[Judgment]) -> dict[str, float]:
    ab = np.array([j.ab_winner for j in judgments])
    ba = np.array([j.ba_winner for j in judgments])
    first = np.concatenate([ab == "a", ba == "b"])  # slot-A picks in each order
    decisive = np.concatenate([ab != "tie", ba != "tie"])
    both = (ab != "tie") & (ba != "tie")
    rate = float(first[decisive].mean()) if decisive.any() else float("nan")
    return {
        "consistency": float(np.mean(ab == ba)),
        "flip_rate": float(np.mean(ab[both] != ba[both])) if both.any() else float("nan"),
        "first_slot_win_rate": rate,
        "position_bias": rate - 0.5,
    }


def pointwise_metrics(anchors: Sequence[Anchor], judgments: Sequence[Judgment]) -> dict | None:
    js, hs, words = [], [], []
    for a, j in zip(anchors, judgments):
        for s, h, r in ((j.score_a, a.human_score_a, a.response_a),
                        (j.score_b, a.human_score_b, a.response_b)):
            if s is not None and h is not None:
                js.append(s), hs.append(h), words.append(len(r.split()))
    if len(js) < 3:
        return None
    j_arr, h_arr = np.array(js, float), np.array(hs, float)
    lw = np.log(np.array(words, float))
    X = np.column_stack([np.ones(len(js)), h_arr, lw])
    beta = np.linalg.lstsq(X, j_arr, rcond=None)[0]
    return {"n": len(js), "pearson": _corr(j_arr, h_arr),
            "spearman": _corr(_rank(j_arr), _rank(h_arr)),
            "mae": float(np.mean(np.abs(j_arr - h_arr))),
            "length_slope_per_log_word": float(beta[2])}


def audit(anchors: Sequence[Anchor], judgments: Sequence[Judgment],
          family: str | None) -> dict[str, Any]:
    """Full bias audit of a judge from its two-order judgments."""
    by_id = {j.id: j for j in judgments}
    pairs = [(a, by_id[a.id]) for a in anchors if a.id in by_id]
    if not pairs:
        raise ValueError("no judgments match the anchors")
    anc = [a for a, _ in pairs]
    js = [j for _, j in pairs]
    raw = decision_metrics(anc, [j.ab_winner for j in js], [j.ab_p_a for j in js], family)
    swap_p = np.array([(j.ab_p_a + j.ba_p_a) / 2 for j in js])
    swap = decision_metrics(anc, label_from_p(swap_p, 0.02), swap_p, family)
    return {"judge_family": family, "n_anchors": len(anc),
            "human_pref_rates": {lab: float(np.mean([a.human_pref == lab for a in anc]))
                                 for lab in LABELS},
            "raw": raw, "position": position_metrics(js), "swap_aggregated": swap,
            "pointwise": pointwise_metrics(anc, js)}


def _f(x: Any, pct: bool = False, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if pct:
        return f"{x * 100:+.1f} pts" if sign else f"{x:.1%}"
    return f"{x:+.2f}" if sign else f"{x:.3f}"


def audit_markdown(rep: dict[str, Any], title: str = "Judge bias audit") -> str:
    raw, pos, swap = rep["raw"], rep["position"], rep["swap_aggregated"]
    v, sp = raw["verbosity"], raw["self_preference"]
    lines = [f"# {title}", "",
             (f"{rep['n_anchors']} human-labelled anchors · judge family "
             f"`{rep['judge_family'] or 'n/a'}`"), "",
             "| Measure | Value | Ideal |", "|---|---:|---:|",
             f"| Agreement with humans (3-way) | {_f(raw['agreement'], True)} | high |",
             f"| Agreement on decisive pairs | {_f(raw['agreement_decisive'], True)} | high |",
             f"| Cohen's κ | {_f(raw['cohen_kappa'])} | → 1 |",
             (f"| Position consistency (same verdict after swap) | "
             f"{_f(pos['consistency'], True)} | 100% |"),
             f"| First-slot win rate | {_f(pos['first_slot_win_rate'], True)} | 50% |",
             (f"| Judge prefers longer | {_f(v['judge_prefers_longer'], True)} | "
             f"= human ({_f(v['human_prefers_longer'], True)}) |"),
             f"| Verbosity excess over humans | {_f(v['excess'], True, True)} | 0 |",
             (f"| Length logit coef (controls for human pref) | "
             f"{_f(v['length_logit_coef'], sign=True)} | 0 |"),
             (f"| Self-preference: own-family win rate, judge vs human | "
             f"{_f(sp.get('judge_own_win_rate'), True)} vs "
             f"{_f(sp.get('human_own_win_rate'), True)} (n={sp['n']}) | equal |"),
             f"| Self-preference gap | {_f(sp.get('gap'), True, True)} | 0 |",
             f"| Confidence ECE | {_f(raw['calibration']['ece'])} | 0 |",
             f"| Agreement after swap-aggregation | {_f(swap['agreement'], True)} | — |"]
    if rep.get("pointwise"):
        pw = rep["pointwise"]
        lines += [(f"| Pointwise score vs human: Pearson / Spearman | {_f(pw['pearson'])} / "
                  f"{_f(pw['spearman'])} | → 1 |"),
                  (f"| Pointwise length slope (points per log-word, human-controlled) | "
                  f"{_f(pw['length_slope_per_log_word'], sign=True)} | 0 |")]
    lines += ["", "## Verbosity by length gap", "",
              "| |log len ratio| | n | Judge prefers longer | Humans prefer longer |",
              "|---|---:|---:|---:|"]
    for b in v["buckets"]:
        lines.append(f"| {b['abs_log_len_ratio']} | {b['n']} | "
                     f"{_f(b['judge_prefers_longer'], True)} | "
                     f"{_f(b['human_prefers_longer'], True)} |")
    lines += ["", "## Confidence reliability", "", "| Confidence bin | n | Stated | Actual |",
              "|---|---:|---:|---:|"]
    for r in raw["calibration"]["reliability"]:
        lines.append(f"| {r['bin_low']:.2f}–{r['bin_high']:.2f} | {r['n']} | "
                     f"{r['confidence']:.1%} | {r['accuracy']:.1%} |")
    return "\n".join(lines) + "\n"
