"""04 — CI/CD Regression Gate.

Runs a parameterised eval suite on every PR, compares it with the committed
baseline and blocks the merge when task success drops significantly or p95
latency spikes. Evals that don't block deploys are just dashboards.
"""

from evalkit.regression_gate.gate import GateReport, evaluate_gate, write_step_summary
from evalkit.regression_gate.runner import CaseResult, RunResult, run_suite
from evalkit.regression_gate.suite import Case, GatePolicy, Scorer, Suite, expand_cases

__all__ = [
    "Case",
    "CaseResult",
    "GatePolicy",
    "GateReport",
    "RunResult",
    "Scorer",
    "Suite",
    "evaluate_gate",
    "expand_cases",
    "run_suite",
    "write_step_summary",
]
