"""03 — Calibrated LLM-as-a-Judge.

Measures a judge's agreement with human anchors and its position, verbosity
and self-preference biases, then fits and validates corrections. Uncalibrated
judges are expensive vibes; calibrated judges are instruments.
"""

from evalkit.calibrated_judge.anchors import Anchor, load_anchors, make_anchors, save_anchors
from evalkit.calibrated_judge.audit import (
    Judgment,
    audit,
    audit_markdown,
    cohen_kappa,
    collect_judgments,
    decision_metrics,
    expected_calibration_error,
    fit_logistic,
)
from evalkit.calibrated_judge.calibrate import (
    CalibratedJudge,
    CalibrationModel,
    calibrate,
    calibration_markdown,
    fit_calibration,
)
from evalkit.calibrated_judge.judge import (
    PairwiseJudge,
    PointwiseJudge,
    SimulatedJudge,
    parse_pairwise,
    parse_score,
)

__all__ = [
    "Anchor",
    "CalibratedJudge",
    "CalibrationModel",
    "Judgment",
    "PairwiseJudge",
    "PointwiseJudge",
    "SimulatedJudge",
    "audit",
    "audit_markdown",
    "calibrate",
    "calibration_markdown",
    "cohen_kappa",
    "collect_judgments",
    "decision_metrics",
    "expected_calibration_error",
    "fit_calibration",
    "fit_logistic",
    "load_anchors",
    "make_anchors",
    "parse_pairwise",
    "parse_score",
    "save_anchors",
]
