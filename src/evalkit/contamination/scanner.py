"""Index an eval set, stream a training corpus against it, and grade each item.

Three detectors run per training document:

* **n-gram** — exact hashed word n-grams (13 by default, as in GPT-3's
  decontamination). Per eval item we track the union of its n-grams seen
  anywhere in the corpus (overlap ratio), the longest contiguous common span,
  and the documents that matched.
* **minhash** — training docs are cut into overlapping word windows, each
  window's MinHash signature is looked up in an LSH index of the eval set, and
  candidates are verified as estimated *containment* of the eval item in the
  window. Catches lightly edited copies that break every 13-gram.
* **embedding** — cosine similarity between eval items and the same windows.
  Catches paraphrases; weakest evidence, so it needs a higher bar to convict.

Memory is bounded by the eval set plus one batch of training docs; per-item
match lists are capped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from evalkit.contamination.embed import Embedder, get_embedder
from evalkit.contamination.minhash import LSHIndex, MinHasher, containment, jaccard
from evalkit.contamination.text import (
    DEFAULT_EVAL_FIELDS,
    Record,
    iter_records,
    ngram_hashes,
    token_hashes,
    tokenize,
    windows,
)

STATUSES = ("clean", "suspicious", "contaminated")
METHODS = ("ngram", "minhash", "embedding")
_EMBED_BLOCK = 256


@dataclass
class Thresholds:
    """Evidence levels that turn raw scores into a status."""

    overlap_contaminated: float = 0.5
    min_ngram_hits_suspicious: int = 1
    containment_contaminated: float = 0.8
    containment_suspicious: float = 0.5
    cosine_contaminated: float | None = None
    cosine_suspicious: float | None = None

    def resolve(self, embedder: Embedder) -> Thresholds:
        """Fill unset cosine thresholds from the embedder's calibrated defaults."""
        sus, con = embedder.cosine_thresholds
        return replace(
            self,
            cosine_suspicious=sus if self.cosine_suspicious is None else self.cosine_suspicious,
            cosine_contaminated=con if self.cosine_contaminated is None
            else self.cosine_contaminated,
        )


@dataclass
class ScanConfig:
    n: int = 13
    min_n: int = 8
    shingle: int = 3
    num_perm: int = 128
    bands: int = 64
    rows: int = 2
    window: int = 0
    stride: int = 0
    embedder: str = "hashed"
    methods: tuple[str, ...] = METHODS
    max_matches: int = 20
    batch_docs: int = 64
    seed: int = 1

    def __post_init__(self) -> None:
        self.methods = tuple(self.methods)
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        for name in ("n", "min_n", "shingle", "num_perm", "bands", "rows", "batch_docs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.min_n > self.n:
            raise ValueError("min_n must be <= n")
        if self.bands * self.rows > self.num_perm:
            raise ValueError("bands * rows must be <= num_perm")


@dataclass
class EvalIndex:
    """Everything derived from the eval set that a scan needs."""

    config: ScanConfig
    ids: list[str]
    texts: list[str]
    tokens: list[list[str]] = field(default_factory=list)
    item_n: list[int] = field(default_factory=list)
    item_hashes: list[np.ndarray] = field(default_factory=list)
    hash_to_items: dict[int, list[int]] = field(default_factory=dict)
    shingle_counts: np.ndarray | None = None
    signatures: np.ndarray | None = None
    embeddings: np.ndarray | None = None
    minhasher: MinHasher = field(init=False, repr=False)
    lsh: LSHIndex = field(init=False, repr=False)
    embedder: Embedder = field(init=False, repr=False)
    ngram_sizes: list[int] = field(init=False, repr=False)
    all_hashes: np.ndarray = field(init=False, repr=False)

    @classmethod
    def build(
        cls,
        items: Sequence[tuple[str, str]],
        config: ScanConfig | None = None,
        embedder: Embedder | None = None,
    ) -> EvalIndex:
        config = config or ScanConfig()
        ids = [i for i, _ in items]
        if len(set(ids)) != len(ids):
            raise ValueError("eval item ids must be unique")
        idx = cls(config=config, ids=ids, texts=[t for _, t in items])
        idx.minhasher = MinHasher(config.num_perm, config.shingle, config.seed)
        idx.lsh = LSHIndex(config.bands, config.rows)
        idx.embedder = embedder or get_embedder(config.embedder)
        sigs, counts = [], []
        for i, text in enumerate(idx.texts):
            toks = tokenize(text)
            idx.tokens.append(toks)
            n = min(config.n, len(toks)) if len(toks) >= config.min_n else 0
            hashes = ngram_hashes(toks, n) if n else np.empty(0, dtype=np.uint64)
            idx.item_n.append(n)
            idx.item_hashes.append(hashes)
            for h in set(hashes.tolist()):
                idx.hash_to_items.setdefault(h, []).append(i)
            shingles = idx.minhasher.shingles(toks)
            sig = idx.minhasher.signature_of(shingles)
            idx.lsh.add(i, sig)
            sigs.append(sig)
            counts.append(len(shingles))
        idx.signatures = np.vstack(sigs) if sigs else np.empty((0, config.num_perm), np.uint64)
        idx.shingle_counts = np.asarray(counts, dtype=np.int64)
        idx.ngram_sizes = sorted({n for n in idx.item_n if n})
        idx.all_hashes = np.fromiter(idx.hash_to_items, dtype=np.uint64)
        if "embedding" in config.methods and idx.texts:
            idx.embedder.fit(idx.texts)
            idx.embeddings = idx.embedder.embed(idx.texts)
        return idx

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        fields: Sequence[str] | None = None,
        id_field: str = "id",
        config: ScanConfig | None = None,
        embedder: Embedder | None = None,
    ) -> EvalIndex:
        items = [
            (r.id, r.text)
            for r in iter_records(path, fields, id_field, DEFAULT_EVAL_FIELDS)
            if r.text.strip()
        ]
        return cls.build(items, config, embedder)

    def window_size(self) -> tuple[int, int]:
        """Training window/stride; ``0`` means auto from eval item lengths.

        Auto uses 1.5x the 90th-percentile item length so a verbatim copy fits
        inside one window, and a quarter-window stride.
        """
        window = self.config.window
        if window <= 0:
            lengths = [len(t) for t in self.tokens] or [32]
            window = max(16, int(np.ceil(1.5 * np.percentile(lengths, 90))))
        stride = self.config.stride if self.config.stride > 0 else max(1, window // 4)
        return window, stride

    def to_dict(self) -> dict:
        return {
            "kind": "evalkit.contamination.index",
            "config": asdict(self.config),
            "stats": {
                "items": len(self.ids),
                "ngram_items": sum(1 for n in self.item_n if n),
                "unique_ngrams": len(self.hash_to_items),
                "ngram_sizes": self.ngram_sizes,
                "embedder": self.embedder.name,
            },
            "items": [{"id": i, "text": t} for i, t in zip(self.ids, self.texts)],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(
        cls, path: str | Path, embedder: Embedder | None = None, **overrides: object
    ) -> EvalIndex:
        """Rebuild a saved index; ``overrides`` replace scan-time config such as
        ``methods``, ``window`` or ``batch_docs``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("kind") != "evalkit.contamination.index":
            raise ValueError(f"{path} is not a contamination index")
        config = ScanConfig(**{**data["config"], **overrides})
        return cls.build([(d["id"], d["text"]) for d in data["items"]], config, embedder)


@dataclass
class DocEvidence:
    """What one training document matched, keyed by eval item index."""

    doc_id: str
    ngram_masks: dict[int, np.ndarray] = field(default_factory=dict)
    ngram_runs: dict[int, tuple[int, int]] = field(default_factory=dict)
    containment: dict[int, tuple[float, str]] = field(default_factory=dict)
    cosine: dict[int, tuple[float, str]] = field(default_factory=dict)

    def level(self, index: EvalIndex, th: Thresholds) -> tuple[str, list[str]]:
        """Worst status this doc causes, and the eval ids it touches at that level."""
        hits: dict[str, set[str]] = {"contaminated": set(), "suspicious": set()}
        for i, mask in self.ngram_masks.items():
            if float(mask.mean()) >= th.overlap_contaminated:
                hits["contaminated"].add(index.ids[i])
            elif int(mask.sum()) >= th.min_ngram_hits_suspicious:
                hits["suspicious"].add(index.ids[i])
        for i, (c, _) in self.containment.items():
            if c >= th.containment_contaminated:
                hits["contaminated"].add(index.ids[i])
            elif c >= th.containment_suspicious:
                hits["suspicious"].add(index.ids[i])
        for i, (s, _) in self.cosine.items():
            if s >= th.cosine_contaminated:
                hits["contaminated"].add(index.ids[i])
            elif s >= th.cosine_suspicious:
                hits["suspicious"].add(index.ids[i])
        for level in ("contaminated", "suspicious"):
            if hits[level]:
                return level, sorted(hits[level])
        return "clean", []


@dataclass
class ItemResult:
    id: str
    status: str
    reasons: list[str]
    methods: list[str]
    n: int
    overlap_ratio: float
    longest_common_ngram: int
    longest_common_text: str
    ngram_docs: list[str]
    ngram_doc_count: int
    max_containment: float
    containment_doc: str | None
    max_cosine: float
    cosine_doc: str | None
    evidence: str = ""
    judge: str | None = None


@dataclass
class ScanReport:
    config: dict
    thresholds: dict
    corpus: dict
    items: list[ItemResult]

    @property
    def summary(self) -> dict:
        counts = {s: sum(1 for r in self.items if r.status == s) for s in STATUSES}
        by_method = {
            m: sum(1 for r in self.items if m in r.methods) for m in (*METHODS, "judge")
        }
        total = len(self.items) or 1
        return {
            **counts,
            "items": len(self.items),
            "contamination_rate": round(counts["contaminated"] / total, 4),
            "flag_rate": round((counts["contaminated"] + counts["suspicious"]) / total, 4),
            "flagged_by_method": by_method,
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "corpus": self.corpus,
            "config": self.config,
            "thresholds": self.thresholds,
            "items": [asdict(r) for r in self.items],
        }

    def clean_ids(self) -> list[str]:
        return [r.id for r in self.items if r.status == "clean"]


def _longest_run(item_hashes: np.ndarray, doc_hashes: np.ndarray) -> tuple[int, int]:
    """Longest run of consecutive n-grams shared *in the same order* by item and doc.

    Returns ``(run length in n-grams, start position in the item)``. Matches
    must be consecutive in both sequences, so two separate copies of
    overlapping fragments do not merge into one fictitious long span.
    """
    positions: dict[int, list[int]] = {}
    for k, h in enumerate(item_hashes.tolist()):
        positions.setdefault(h, []).append(k)
    doc_pos = np.flatnonzero(np.isin(doc_hashes, item_hashes)).tolist()
    pairs = sorted((p - k, k) for p in doc_pos for k in positions[int(doc_hashes[p])])
    best = start = run = 0
    prev: tuple[int, int] | None = None
    for diag, k in pairs:
        run = run + 1 if prev == (diag, k - 1) else 1
        prev = (diag, k)
        if run > best:
            best, start = run, k - run + 1
    return best, start


def _batched(records: Iterable[Record], size: int) -> Iterator[list[Record]]:
    batch: list[Record] = []
    for rec in records:
        batch.append(rec)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class ContaminationScanner:
    """Streams training records against an :class:`EvalIndex`."""

    def __init__(self, index: EvalIndex, thresholds: Thresholds | None = None) -> None:
        self.index = index
        self.th = (thresholds or Thresholds()).resolve(index.embedder)
        self.cfg = index.config
        self.window, self.stride = index.window_size()
        self.tokens_seen = 0
        self.windows_seen = 0

    def evidence(self, records: Iterable[Record]) -> Iterator[tuple[Record, DocEvidence]]:
        """Yield ``(record, evidence)`` for every training record, streaming."""
        for batch in _batched(records, self.cfg.batch_docs):
            yield from self._process_batch(batch)

    def _process_batch(self, batch: list[Record]) -> list[tuple[Record, DocEvidence]]:
        idx, cfg = self.index, self.cfg
        out, chunk_texts, chunk_owner = [], [], []
        for rec in batch:
            ev = DocEvidence(rec.id)
            toks = tokenize(rec.text)
            self.tokens_seen += len(toks)
            if "ngram" in cfg.methods and len(idx.all_hashes):
                self._ngram(toks, ev)
            wins = windows(toks, self.window, self.stride)
            self.windows_seen += len(wins)
            if "minhash" in cfg.methods:
                self._minhash(wins, ev)
            if "embedding" in cfg.methods and idx.embeddings is not None:
                for _, w in wins:
                    chunk_texts.append(" ".join(w))
                    chunk_owner.append(len(out))
            out.append((rec, ev))
        floor = self.th.cosine_suspicious * 0.75
        for lo in range(0, len(chunk_texts), _EMBED_BLOCK):
            block = chunk_texts[lo : lo + _EMBED_BLOCK]
            sims = idx.embedder.embed(block) @ idx.embeddings.T
            for r, c in zip(*np.nonzero(sims >= floor)):
                ev = out[chunk_owner[lo + r]][1]
                s = float(sims[r, c])
                if s > ev.cosine.get(int(c), (-1.0, ""))[0]:
                    ev.cosine[int(c)] = (s, block[r])
        return out

    def _ngram(self, toks: list[str], ev: DocEvidence) -> None:
        idx = self.index
        th = token_hashes(toks)
        for n in idx.ngram_sizes:
            doc_hashes = ngram_hashes(th, n)
            if not doc_hashes.size:
                continue
            matched = np.unique(doc_hashes[np.isin(doc_hashes, idx.all_hashes)])
            if not matched.size:
                continue
            items = {i for h in matched.tolist() for i in idx.hash_to_items[h]}
            for i in items:
                if idx.item_n[i] != n:
                    continue
                mask = np.isin(idx.item_hashes[i], matched)
                if mask.any():
                    ev.ngram_masks[i] = mask
                    ev.ngram_runs[i] = _longest_run(idx.item_hashes[i], doc_hashes)

    def _minhash(self, wins: list[tuple[int, list[str]]], ev: DocEvidence) -> None:
        idx = self.index
        for _, w in wins:
            shingles = idx.minhasher.shingles(w)
            sig = idx.minhasher.signature_of(shingles)
            for i in idx.lsh.query(sig):
                j = jaccard(idx.signatures[i], sig)
                c = containment(j, int(idx.shingle_counts[i]), len(shingles))
                if c > ev.containment.get(i, (-1.0, ""))[0]:
                    ev.containment[i] = (c, " ".join(w))

    def scan(self, records: Iterable[Record]) -> ScanReport:
        """Aggregate per-item statistics over a whole corpus."""
        idx, th = self.index, self.th
        t0 = time.perf_counter()
        self.tokens_seen = self.windows_seen = 0
        n_items = len(idx.ids)
        masks = [np.zeros(len(h), dtype=bool) for h in idx.item_hashes]
        longest = [(0, -1, "")] * n_items
        ngram_docs: list[list[str]] = [[] for _ in range(n_items)]
        ngram_doc_counts = [0] * n_items
        best_c: list[tuple[float, str | None, str]] = [(0.0, None, "")] * n_items
        best_s: list[tuple[float, str | None, str]] = [(0.0, None, "")] * n_items
        docs = flagged_docs = 0
        for rec, ev in self.evidence(records):
            docs += 1
            if ev.level(idx, th)[0] != "clean":
                flagged_docs += 1
            for i, mask in ev.ngram_masks.items():
                masks[i] |= mask
                run, start = ev.ngram_runs[i]
                if run > longest[i][0]:
                    longest[i] = (run, start, rec.id)
                ngram_doc_counts[i] += 1
                if len(ngram_docs[i]) < self.cfg.max_matches:
                    ngram_docs[i].append(rec.id)
            for i, (c, text) in ev.containment.items():
                if c > best_c[i][0]:
                    best_c[i] = (c, rec.id, text)
            for i, (s, text) in ev.cosine.items():
                if s > best_s[i][0]:
                    best_s[i] = (s, rec.id, text)
        results = [
            self._grade(
                i, masks[i], longest[i], ngram_docs[i], ngram_doc_counts[i], best_c[i], best_s[i]
            )
            for i in range(n_items)
        ]
        corpus = {
            "documents": docs,
            "tokens": self.tokens_seen,
            "windows": self.windows_seen,
            "flagged_documents": flagged_docs,
            "seconds": round(time.perf_counter() - t0, 3),
        }
        cfg = asdict(self.cfg)
        cfg.update(embedder=idx.embedder.name, window=self.window, stride=self.stride)
        return ScanReport(cfg, asdict(th), corpus, results)

    def _grade(self, i, mask, longest, ngram_docs, doc_count, best_c, best_s) -> ItemResult:
        idx, th = self.index, self.th
        n = idx.item_n[i]
        ratio = float(mask.mean()) if mask.size else 0.0
        run, start, _ = longest
        span = run + n - 1 if run else 0
        span_text = " ".join(idx.tokens[i][start : start + span]) if run else ""
        reasons: list[str] = []
        methods: list[str] = []
        status = "clean"

        def flag(level: str, method: str, reason: str) -> None:
            nonlocal status
            if STATUSES.index(level) > STATUSES.index(status):
                status = level
            if method not in methods:
                methods.append(method)
            reasons.append(f"[{level}] {reason}")

        hits = int(mask.sum())
        if ratio >= th.overlap_contaminated:
            flag("contaminated", "ngram",
                 f"{ratio:.0%} of {n}-grams found in training ({doc_count} doc(s)); "
                 f"longest common span {span} words")
        elif hits and hits >= th.min_ngram_hits_suspicious:
            flag("suspicious", "ngram",
                 f"partial overlap: {hits}/{mask.size} {n}-grams ({ratio:.0%}), "
                 f"longest common span {span} words")
        c, c_doc, c_text = best_c
        if c >= th.containment_contaminated:
            flag("contaminated", "minhash", f"MinHash containment {c:.2f} in {c_doc}")
        elif c >= th.containment_suspicious:
            flag("suspicious", "minhash", f"MinHash containment {c:.2f} in {c_doc}")
        s, s_doc, s_text = best_s
        if s >= th.cosine_contaminated:
            flag("contaminated", "embedding", f"embedding cosine {s:.2f} with {s_doc}")
        elif s >= th.cosine_suspicious:
            flag("suspicious", "embedding", f"embedding cosine {s:.2f} with {s_doc}")
        if not n:
            reasons.append(f"[note] item shorter than {self.cfg.min_n} tokens; n-gram check skipped")
        evidence = (c_text if c >= s else s_text) or span_text
        return ItemResult(
            id=idx.ids[i], status=status, reasons=reasons, methods=methods, n=n,
            overlap_ratio=round(ratio, 4), longest_common_ngram=span,
            longest_common_text=span_text, ngram_docs=ngram_docs,
            ngram_doc_count=doc_count,
            max_containment=round(c, 4), containment_doc=c_doc,
            max_cosine=round(s, 4), cosine_doc=s_doc,
            evidence=evidence[:400] if status != "clean" else "",
        )


def decontaminate(
    scanner: ContaminationScanner,
    records: Iterable[Record],
    out_path: str | Path,
    mode: str = "drop",
    level: str = "contaminated",
) -> dict:
    """Write a filtered copy of the corpus.

    ``mode="drop"`` removes documents whose evidence reaches ``level``;
    ``mode="flag"`` keeps every document and adds a ``_contamination`` field.
    """
    if mode not in ("drop", "flag"):
        raise ValueError("mode must be 'drop' or 'flag'")
    if level not in ("suspicious", "contaminated"):
        raise ValueError("level must be 'suspicious' or 'contaminated'")
    stats = {"documents": 0, "kept": 0, "dropped": 0, "flagged": 0, "dropped_ids": []}
    with Path(out_path).open("w", encoding="utf-8") as fh:
        for rec, ev in scanner.evidence(records):
            stats["documents"] += 1
            doc_level, items = ev.level(scanner.index, scanner.th)
            hit = STATUSES.index(doc_level) >= STATUSES.index(level)
            row = dict(rec.raw) if rec.raw is not None else {"id": rec.id, "text": rec.text}
            if hit and mode == "drop":
                stats["dropped"] += 1
                stats["dropped_ids"].append(rec.id)
                continue
            if hit:
                stats["flagged"] += 1
                row["_contamination"] = {"level": doc_level, "eval_ids": items}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    return stats
