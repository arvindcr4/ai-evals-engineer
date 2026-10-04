"""Adversarial regressions for the contamination checker."""

import json

import numpy as np
import pytest

from evalkit.contamination import ContaminationScanner, EvalIndex, ScanConfig
from evalkit.contamination.text import Record, iter_records

ITEM = (
    "the quick brown fox jumps over the lazy dog near the river bank today at noon "
    "while seven herons watch"
)


def ngram_index(**cfg):
    return EvalIndex.build([("q", ITEM)], ScanConfig(methods=("ngram",), **cfg))


def test_openai_content_parts_are_scanned(tmp_path):
    path = tmp_path / "chat.jsonl"
    row = {"id": "c1", "messages": [
        {"role": "user", "content": [{"type": "text", "text": ITEM}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]}
    path.write_text(json.dumps(row) + "\n")
    rec = next(iter_records(path))
    assert ITEM in rec.text
    rep = ContaminationScanner(ngram_index()).scan([rec])
    assert rep.items[0].status == "contaminated"


def test_non_object_jsonl_rows_do_not_crash(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('[1, 2]\n42\n"plain"\n{"text": "ok"}\n')
    texts = [r.text for r in iter_records(path)]
    assert texts == ["1\n2", "42", "plain", "ok"]


def test_longest_span_requires_contiguity_in_the_doc():
    toks = ITEM.split()
    a, b = " ".join(toks[0:13]), " ".join(toks[1:14])
    rep = ContaminationScanner(ngram_index()).scan([Record("x", f"{a} zzz filler {b}")])
    r = rep.items[0]
    assert r.longest_common_ngram == 13
    contiguous = ContaminationScanner(ngram_index()).scan([Record("y", " ".join(toks[0:14]))])
    assert contiguous.items[0].longest_common_ngram == 14


def test_doc_count_is_not_capped_by_max_matches():
    rep = ContaminationScanner(ngram_index(max_matches=3)).scan(
        [Record(f"d{i}", ITEM) for i in range(10)]
    )
    r = rep.items[0]
    assert len(r.ngram_docs) == 3 and r.ngram_doc_count == 10
    assert "(10 doc(s))" in r.reasons[0]


@pytest.mark.parametrize("bad", [{"n": 0, "min_n": 0}, {"shingle": 0},
                                 {"bands": 100, "rows": 2}, {"batch_docs": 0}])
def test_degenerate_configs_rejected(bad):
    with pytest.raises(ValueError):
        ScanConfig(**bad)


def test_empty_eval_set_scans_cleanly():
    rep = ContaminationScanner(EvalIndex.build([], ScanConfig())).scan([Record("a", ITEM)])
    assert rep.summary["items"] == 0 and rep.summary["contamination_rate"] == 0.0


def test_embedding_blocks_match_unblocked(monkeypatch):
    import evalkit.contamination.scanner as sc

    docs = [Record(f"d{i}", ITEM + f" filler {i} " * 40) for i in range(6)]
    idx = EvalIndex.build([("q", ITEM)], ScanConfig(methods=("embedding",)))
    full = ContaminationScanner(idx).scan(docs).items[0].max_cosine
    monkeypatch.setattr(sc, "_EMBED_BLOCK", 2)
    blocked = ContaminationScanner(idx).scan(docs).items[0].max_cosine
    assert np.isclose(full, blocked) and full > 0.5


def test_decontaminate_refuses_to_overwrite_input(tmp_path):
    from evalkit.cli import main

    ev = tmp_path / "eval.jsonl"
    ev.write_text(json.dumps({"id": "q", "text": ITEM}) + "\n")
    tr = tmp_path / "train.jsonl"
    tr.write_text(json.dumps({"id": "t", "text": ITEM}) + "\n")
    with pytest.raises(SystemExit):
        main(["contamination", "decontaminate", "--eval", str(ev), "--train", str(tr),
              "--out", str(tr)])
    assert ITEM in tr.read_text()
