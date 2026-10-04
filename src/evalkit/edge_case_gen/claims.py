"""Does a case actually contain the edge its category claims?

A real model asked for a ``zero_width`` case often writes "contains U+200B" in
the description and then returns plain ASCII text (the invisible character is
lost on the way out), or claims ``at_max_length`` on a 479-character note. Such
a case is labelled and scored as if it tested the claimed edge, which inflates
coverage and hides what was really tested. ``claim_holds`` checks the
categories whose presence can be decided mechanically; ``None`` means the
category is not checkable (semantic phrasing, typos, plain min/max...).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from evalkit.edge_case_gen.spec import TaskSpec, parse_iso_date, validate_input

BIDI_RTL = {"R", "AL", "RLE", "RLO", "RLI", "FSI", "PDF", "PDI", "LRE", "LRO", "LRI"}
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
INSTRUCTION_RE = re.compile(
    r"\b(ignore|disregard|instructions?|system|override|forget|you must|you are|"
    r"assistant|prompt|output|respond|answer|approve|reject|escalate)\b", re.IGNORECASE)
MARKUP_RE = re.compile(r"<[^>]+>|\*\*|__|\[[^\]]*\]\(|^#|```|&\w+;|\{\s*\"", re.MULTILINE)


def _non_ascii_letter(t: str) -> bool:
    return any(ord(ch) > 127 and unicodedata.category(ch).startswith("L") for ch in t)


def _emoji(t: str) -> bool:
    return any(ord(ch) >= 0x1F000 or 0x2600 <= ord(ch) <= 0x27BF for ch in t)


def _casing(t: str) -> bool:
    letters = [ch for ch in t if ch.isalpha()]
    return len(letters) > 1 and (t == t.upper() or t == t.lower()
                                 or bool(re.search(r"[a-z][A-Z]", t)))


TEXT_CHECKS: dict[str, Callable[[str], bool]] = {
    "rtl": lambda t: any(unicodedata.bidirectional(ch) in BIDI_RTL for ch in t),
    "zero_width": lambda t: any(ch in ZERO_WIDTH for ch in t),
    "combining_marks": lambda t: any(unicodedata.combining(ch) for ch in t),
    "fullwidth": lambda t: any(0xFF01 <= ord(ch) <= 0xFF5E or ch == "\u3000" for ch in t),
    "homoglyph": _non_ascii_letter,
    "emoji": _emoji,
    "whitespace": lambda t: bool(re.search(r"[^\S ]|  |^\s|\s$", t)),
    "casing": _casing,
    "markup": lambda t: bool(MARKUP_RE.search(t)),
    "instruction_like": lambda t: bool(INSTRUCTION_RE.search(t)),
    "nested_quoting": lambda t: bool(re.search(r"[\"'`“”‘’\\]", t)),
    "delimiter_collision": lambda t: bool(re.search(r"[|;\t\":{}\[\]=,\n]|---|###", t)),
}


def _errs(spec: TaskSpec, inp: dict, pattern: str) -> bool:
    return any(re.search(pattern, e) for e in validate_input(spec, inp))


def _values(spec: TaskSpec, inp: dict, types: tuple[str, ...]) -> list[Any]:
    return [inp[f.name] for f in spec.fields if f.type in types and f.name in inp]


def _beyond(spec: TaskSpec, inp: dict, sign: int) -> bool:
    """A numeric value past min (sign -1) or max (+1), even if its type is also wrong."""
    for f in spec.fields:
        v = inp.get(f.name)
        if f.type not in ("number", "integer") or isinstance(v, bool) \
                or not isinstance(v, (int, float)):
            continue
        bound = f.max if sign > 0 else f.min
        if bound is not None and (v - bound) * sign > 0:
            return True
    return False


def claim_holds(spec: TaskSpec, category: str, inp: dict, field: str | None = None) -> bool | None:
    """True/False when ``category`` is mechanically checkable on ``inp``, else None."""
    if category in TEXT_CHECKS:
        names = [field] if field and field in inp else []
        names += [spec.text_field] if spec.text_field else []
        texts = [inp.get(n) for n in names]
        texts = [t for t in texts if isinstance(t, str)]
        if not texts:
            return None
        return any(TEXT_CHECKS[category](t) for t in texts)
    if category == "long_input":
        tf = spec.text_field
        if not tf or not isinstance(inp.get(tf), str):
            return None
        mx = spec.field(tf).max_length
        return len(inp[tf]) >= (0.8 * mx if mx else 1000)
    nums = [v for v in _values(spec, inp, ("number", "integer"))
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    strs = [v for v in _values(spec, inp, ("string",)) if isinstance(v, str)]
    checks: dict[str, Callable[[], bool]] = {
        "missing": lambda: _errs(spec, inp, r": missing$"),
        "null": lambda: any(v is None for v in inp.values()),
        "wrong_type": lambda: _errs(spec, inp, r": expected "),
        "invalid_choice": lambda: _errs(spec, inp, r" not in \["),
        "invalid_date": lambda: _errs(spec, inp, r"is not an ISO date"),
        "date_format": lambda: any(isinstance(v, str) and parse_iso_date(v) is None
                                   for v in _values(spec, inp, ("date",))),
        "leap_day": lambda: any(isinstance(v, str) and v.endswith("-02-29")
                                for v in _values(spec, inp, ("date",))),
        "below_min": lambda: _errs(spec, inp, r" < min | before ") or _beyond(spec, inp, -1),
        "above_max": lambda: _errs(spec, inp, r" > max | after ") or _beyond(spec, inp, 1),
        "zero": lambda: any(v == 0 for v in nums),
        "negative": lambda: any(v < 0 for v in nums),
        "empty": lambda: any(v == "" for v in strs),
        "whitespace_only": lambda: any(v and not v.strip() for v in strs),
        "over_max_length": lambda: _errs(spec, inp, r": length \d+ > "),
        "at_max_length": lambda: any(
            f.max_length is not None and isinstance(inp.get(f.name), str)
            and len(inp[f.name]) == f.max_length for f in spec.fields),
    }
    check = checks.get(category)
    return None if check is None else check()
