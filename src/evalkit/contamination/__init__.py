"""Dataset contamination checker.

Scans fine-tuning corpora for leaks of a golden eval set with three detectors
of increasing recall and decreasing precision: exact word n-gram overlap,
MinHash/LSH near-duplicate containment, and embedding cosine similarity.
"""

from evalkit.contamination.embed import HashedEmbedder, OpenAIEmbedder, get_embedder
from evalkit.contamination.minhash import LSHIndex, MinHasher
from evalkit.contamination.scanner import (
    ContaminationScanner,
    EvalIndex,
    ItemResult,
    ScanConfig,
    ScanReport,
    Thresholds,
    decontaminate,
)
from evalkit.contamination.text import ngram_hashes, normalize, tokenize

__all__ = [
    "ContaminationScanner",
    "EvalIndex",
    "HashedEmbedder",
    "ItemResult",
    "LSHIndex",
    "MinHasher",
    "OpenAIEmbedder",
    "ScanConfig",
    "ScanReport",
    "Thresholds",
    "decontaminate",
    "get_embedder",
    "ngram_hashes",
    "normalize",
    "tokenize",
]
