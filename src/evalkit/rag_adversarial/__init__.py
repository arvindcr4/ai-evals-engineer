"""05 — RAG Adversarial Harness.

Perturb a clean QA set (distractors, gold-doc removal, contradictions, stale
docs, citation-id shuffles, near-miss entities, prompt injection) and measure
whether a RAG system answers correctly with valid citations or confidently
abstains — and how often it tells a confident, ungrounded lie.
"""

from .data import Case, Doc, QAItem, load_cases, load_items
from .perturb import OPERATORS, PerturbConfig, perturb
from .rag import ABSTAIN, CONFLICT, LLMRAG, BaselineRAG, build_prompt, mock_rag_llm
from .scoring import metrics, parse_response, run_cases, score_case, summarize, to_markdown

__all__ = [
    "ABSTAIN",
    "CONFLICT",
    "LLMRAG",
    "OPERATORS",
    "BaselineRAG",
    "Case",
    "Doc",
    "PerturbConfig",
    "QAItem",
    "build_prompt",
    "load_cases",
    "load_items",
    "metrics",
    "mock_rag_llm",
    "parse_response",
    "perturb",
    "run_cases",
    "score_case",
    "summarize",
    "to_markdown",
]
