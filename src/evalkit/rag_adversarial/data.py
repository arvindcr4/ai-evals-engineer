"""Data model for the RAG adversarial harness.

A :class:`QAItem` is one clean, answerable question with its documents. A
perturbation operator turns it into a :class:`Case`: the question, the exact
document set the RAG system will see, and what a trustworthy system is
expected to do with it (``answer``, ``abstain`` or ``conflict``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from evalkit.core.trajectory import read_jsonl, write_jsonl

Expected = Literal["answer", "abstain", "conflict"]


@dataclass
class Doc:
    id: str
    text: str
    title: str = ""
    date: str | None = None  # ISO date; used to resolve stale-vs-current evidence

    def render(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class QAItem:
    """A clean QA example.

    ``supporting_docs`` are the ids of documents that state ``gold_answer``.
    ``entity`` / ``near_miss`` drive the entity-swap operator and
    ``alt_answer`` is the conflicting value used by contradiction, stale and
    injection operators (derived automatically when absent).
    """

    id: str
    question: str
    gold_answer: str
    docs: list[Doc]
    supporting_docs: list[str]
    entity: str | None = None
    near_miss: str | None = None
    alt_answer: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QAItem:
        d = dict(d)
        d["docs"] = [Doc(**x) for x in d.get("docs", [])]
        if "supporting_docs" not in d:
            gold = normalize(d["gold_answer"])
            d["supporting_docs"] = [x.id for x in d["docs"] if contains(x.text, gold)]
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Case:
    case_id: str
    base_id: str
    perturbation: str
    question: str
    docs: list[Doc]
    expected: Expected
    gold_answer: str
    answer_doc_ids: list[str] = field(default_factory=list)
    forbidden_answers: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        d = dict(d)
        d["docs"] = [Doc(**x) for x in d.get("docs", [])]
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def doc(self, doc_id: str) -> Doc | None:
        return next((x for x in self.docs if x.id == doc_id), None)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation (``1,250`` → ``1250``), collapse whitespace."""
    text = re.sub(r"(?<=\d)[,_](?=\d)", "", text.lower())
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", text).replace(". ", " ").strip(". ").split())


def contains(haystack: str, needle_normalized: str) -> bool:
    """Token-boundary containment of an already-normalized needle."""
    if not needle_normalized:
        return False
    return f" {needle_normalized} " in f" {normalize(haystack)} "


def load_items(path: str | Path) -> list[QAItem]:
    return [QAItem.from_dict(r) for r in read_jsonl(path)]


def load_cases(path: str | Path) -> list[Case]:
    return [Case.from_dict(r) for r in read_jsonl(path)]


def save(path: str | Path, rows: list[Any]) -> None:
    write_jsonl(path, rows)
