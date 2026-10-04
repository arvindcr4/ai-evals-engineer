"""Toy expense-claim triage: a reference oracle and a naive production system.

``reference`` is the policy as written (it only ever sees schema-valid claims;
the validator labels invalid ones). ``naive_triage`` is the kind of code that
passes every seed example yet breaks on edge cases.
"""

from __future__ import annotations

import datetime as dt

USD_RATE = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "INR": 0.012}


def reference(claim: dict) -> str:
    usd = claim["amount"] * USD_RATE[claim["currency"]]
    if usd > 75 and not claim["has_receipt"]:
        return "reject"
    if usd > 1000 or claim["category"] == "other":
        return "escalate"
    return "approve"


def naive_triage(claim: dict) -> str:
    note = claim["employee_note"].strip()
    if not note:
        return "invalid"
    note.encode("latin-1")  # legacy audit log is latin-1 only
    date = dt.date(*map(int, claim["expense_date"].split("-")))
    if date.year < 2025:
        return "invalid"
    amount = float(claim["amount"])
    usd = amount * USD_RATE[claim["currency"].upper()]
    if usd > 75 and claim["has_receipt"] is False:
        return "reject"
    if usd > 1000 or claim["category"] == "other":
        return "escalate"
    return "approve"


def robust_triage(claim: dict) -> str:
    """The fix: validate against the spec first, then apply the policy."""
    from pathlib import Path

    from evalkit.edge_case_gen.spec import load_spec, validate_input

    spec = load_spec(Path(__file__).with_name("spec.yaml"))
    return "invalid" if validate_input(spec, claim) else reference(claim)
