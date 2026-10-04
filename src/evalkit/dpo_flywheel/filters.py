"""Quality filters that stand between raw feedback pairs and a training set.

Each filter either rewrites a pair (PII scrubbing) or drops it with a named
reason, so the manifest can say exactly why the dataset is the size it is.
Order matters: scrub first (so dedupe and decontamination see the text that
will actually be trained on), then cheap structural checks, then the O(n²)
near-duplicate pass last on whatever survived.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,5}[ .-]?\d{3,4}[ .-]?\d{0,4}(?!\w)"),
}

_IPV4 = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")

REFUSAL_MARKERS = (
    "i can't help with",
    "i cannot help with",
    "i'm sorry, but i can",
    "i am sorry, but i can",
    "i'm unable to",
    "i am unable to",
    "as an ai language model",
    "i won't be able to",
    "i cannot assist",
    "i can't assist",
)


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def norm_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def shingles(text: str, k: int = 3) -> set[str]:
    """Word k-shingles; texts shorter than k fall back to their word set."""
    words = normalize(text).split()
    if len(words) < k:
        return set(words)
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def scrub_pii(text: str) -> tuple[str, Counter]:
    """Replace emails, card numbers (Luhn-checked) and phone numbers with tags."""
    counts: Counter = Counter()

    def _email(m: re.Match) -> str:
        counts["email"] += 1
        return "[EMAIL]"

    def _card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            counts["card"] += 1
            return "[CARD]"
        return m.group()

    def _phone(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if _IPV4.fullmatch(m.group().strip()):
            return m.group()
        if 10 <= len(digits) <= 13:
            counts["phone"] += 1
            return "[PHONE]"
        return m.group()

    text = PII_PATTERNS["email"].sub(_email, text)
    text = PII_PATTERNS["card"].sub(_card, text)
    text = PII_PATTERNS["phone"].sub(_phone, text)
    return text, counts


def is_refusal(text: str) -> bool:
    low = text.lower().replace("’", "'")
    return any(marker in low for marker in REFUSAL_MARKERS)


def ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@dataclass
class Decontaminator:
    """Flags prompts that leak eval items.

    A prompt is contaminated when it normalizes to a golden item, or when the
    word n-gram overlap with some golden item exceeds ``threshold`` measured
    *either way*: as a share of the prompt's n-grams (prompt is mostly eval
    text) or of the golden item's n-grams (eval item pasted inside a longer
    prompt). Golden items of 2..n-1 words match by whole-phrase containment;
    one-word items only by exact match.
    """

    golden: list[str]
    n: int = 5
    threshold: float = 0.5
    _exact: set[str] = field(init=False, repr=False)
    _items: list[set[tuple[str, ...]]] = field(init=False, repr=False)
    _short: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._exact = {norm_hash(g) for g in self.golden}
        self._items, self._short = [], []
        for g in self.golden:
            grams = ngram_set(g, self.n)
            if grams:
                self._items.append(grams)
            elif len(normalize(g).split()) >= 2:
                self._short.append(normalize(g))

    def overlap(self, text: str) -> float:
        if norm_hash(text) in self._exact:
            return 1.0
        padded = f" {normalize(text)} "
        if any(f" {s} " in padded for s in self._short):
            return 1.0
        grams = ngram_set(text, self.n)
        if not grams:
            return 0.0
        best = 0.0
        for item in self._items:
            shared = len(grams & item)
            if shared:
                best = max(best, shared / len(grams), shared / len(item))
        return best

    def is_contaminated(self, text: str) -> bool:
        return self.overlap(text) >= self.threshold


@dataclass
class FilterConfig:
    min_chosen_words: int = 3
    max_chosen_words: int = 800
    max_prompt_words: int = 2000
    min_margin: float = 1.0
    trust_corrections: bool = True
    near_dup_threshold: float = 0.85
    scrub_pii: bool = True
    drop_refusal_chosen: bool = True


def filter_pairs(
    pairs: list,
    cfg: FilterConfig,
    decon: Decontaminator | None = None,
) -> tuple[list, Counter, Counter]:
    """Apply every filter. Returns (kept pairs, drop reasons, PII redactions).

    ``pairs`` are :class:`~evalkit.dpo_flywheel.pairs.PreferencePair` objects;
    they are scrubbed in place.
    """
    drops: Counter = Counter()
    pii: Counter = Counter()
    survivors = []
    for p in pairs:
        if cfg.scrub_pii:
            for attr in ("prompt", "chosen", "rejected"):
                clean, c = scrub_pii(getattr(p, attr))
                setattr(p, attr, clean)
                pii.update(c)
        reason = _structural_reason(p, cfg, decon)
        if reason:
            drops[reason] += 1
        else:
            survivors.append(p)

    kept, seen_hash, kept_shingles = [], set(), []
    for p in survivors:
        key = f"{p.prompt}\n{p.chosen}"
        h = norm_hash(key)
        if h in seen_hash:
            drops["duplicate"] += 1
            continue
        sh = shingles(key)
        if any(jaccard(sh, other) >= cfg.near_dup_threshold for other in kept_shingles):
            drops["near_duplicate"] += 1
            continue
        seen_hash.add(h)
        kept_shingles.append(sh)
        kept.append(p)
    return kept, drops, pii


def _structural_reason(p, cfg: FilterConfig, decon: Decontaminator | None) -> str | None:
    if not p.prompt.strip() or not p.chosen.strip() or not p.rejected.strip():
        return "empty"
    if normalize(p.chosen) == normalize(p.rejected):
        return "identical"
    n_chosen = len(p.chosen.split())
    if n_chosen < cfg.min_chosen_words or n_chosen > cfg.max_chosen_words:
        return "length"
    if len(p.prompt.split()) > cfg.max_prompt_words:
        return "length"
    if cfg.drop_refusal_chosen and is_refusal(p.chosen):
        return "refusal"
    trusted = cfg.trust_corrections and p.strategy == "correction"
    if not trusted and (p.margin is None or p.margin < cfg.min_margin):
        return "low_margin"
    if decon is not None and decon.is_contaminated(p.prompt):
        return "contaminated"
    return None
