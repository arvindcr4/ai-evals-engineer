"""Toy system under test: a rule-based support-ticket triager.

``triage(input, tier=...)`` returns ``{category, priority, order_id, amount}``.
``SUT_VERSION`` selects a build so the regression gate has something to catch:

* ``1``   — the version on main (baseline).
* ``1.1`` — a harmless refactor that also learns the word "invoice".
* ``2``   — a bad refactor: "refund" now routes to shipping and cents are dropped.
* ``3``   — same answers as ``1`` but three times slower.
"""

from __future__ import annotations

import os
import re
import time

PRIMARY = {
    "billing": ["charged", "charge", "refund", "payment", "billing", "subscription"],
    "shipping": ["delivery", "shipped", "package", "tracking", "courier", "arrive"],
    "account": ["password", "login", "log in", "email address", "username", "2fa"],
    "technical": ["crash", "error", "bug", "not loading", "freezes", "500"],
}
SECONDARY = {
    "billing": ["card", "price", "receipt"],
    "shipping": ["parcel", "address change", "warehouse"],
    "account": ["locked out", "verify", "profile"],
    "technical": ["slow", "blank screen", "timeout"],
}
URGENT = ["urgent", "asap", "immediately", "charged twice", "can't access", "down"]


def _version() -> str:
    return os.environ.get("SUT_VERSION", "1")


def _classify(text: str, tier: str, version: str) -> str:
    primary = {k: list(v) for k, v in PRIMARY.items()}
    if version == "1.1":
        primary["billing"].append("invoice")
    if version == "2":
        primary["billing"].remove("refund")
        primary["shipping"].append("refund")
    tables = [primary] + ([SECONDARY] if tier == "accurate" else [])
    for table in tables:
        hits = {cat: sum(kw in text for kw in kws) for cat, kws in table.items()}
        best = max(hits, key=lambda c: hits[c])
        if hits[best] > 0:
            return best
    return "other"


def triage(input: dict, tier: str = "fast") -> dict:
    version = _version()
    delay = 0.002 if tier == "fast" else 0.004
    time.sleep(delay * (3 if version == "3" else 1))
    text = input["text"].lower()
    order = re.search(r"\b(ord-\d{4,})\b", text)
    amount = re.search(r"\$(\d+(?:\.\d{2})?)", text)
    value = None
    if amount:
        value = float(amount.group(1))
        if version == "2":
            value = float(int(value))
    return {
        "category": _classify(text, tier, version),
        "priority": "high" if any(u in text for u in URGENT) else "normal",
        "order_id": order.group(1).upper() if order else None,
        "amount": value,
    }
