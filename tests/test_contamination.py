import json

import numpy as np
import pytest

from evalkit.contamination import (
    ContaminationScanner,
    EvalIndex,
    HashedEmbedder,
    LSHIndex,
    MinHasher,
    OpenAIEmbedder,
    ScanConfig,
    Thresholds,
    decontaminate,
    get_embedder,
    ngram_hashes,
    normalize,
    tokenize,
)
from evalkit.contamination.judge import adjudicate, judge_llm
from evalkit.contamination.minhash import containment, jaccard
from evalkit.contamination.report import to_markdown
from evalkit.contamination.text import Record, iter_records, windows
from evalkit.core.llm import MockLLM

HASH_Q = (
    "In a hash table that uses linear probing, what happens to lookup performance as the "
    "load factor approaches one, and why do primary clusters form? Probe sequences grow long "
    "because occupied slots form contiguous runs that new keys extend."
)
TRAIN_Q = (
    "A train leaves Pune at 9:40 and travels 312 kilometres at an average speed of 78 "
    "kilometres per hour. At what time does it reach its destination? It arrives at 13:40."
)
CAPITAL_Q = (
    "What is the capital city of Australia, and why was it chosen instead of Sydney or "
    "Melbourne? Canberra, purpose-built as a compromise between rival cities."
)
TCP_Q = (
    "In TCP congestion control, what triggers a fast retransmit and how does the sender "
    "adjust its congestion window afterwards? Three duplicate acknowledgements."
)
EVAL = [("hash", HASH_Q), ("train", TRAIN_Q), ("capital", CAPITAL_Q), ("tcp", TCP_Q)]

FILLER = "Unrelated preface about cooking lentils with cumin and ginger for a weeknight dinner."
CORPUS = [
    Record("exact", f"{FILLER} {TRAIN_Q.upper()}!!! Next problem follows."),
    Record(
        "edited",
        HASH_Q.replace("performance", "speed").replace("form?", "appear?")
        .replace("because", "since"),
    ),
    Record(
        "para",
        "Many visitors assume Sydney is the national capital, yet the seat of government is "
        "Canberra, a city designed from scratch as a compromise because neither Sydney nor "
        "Melbourne would accept the other as capital of Australia.",
    ),
    Record("noise", "Release notes: the scheduler retries failed jobs with exponential backoff."),
]


def scan(items=EVAL, corpus=CORPUS, **cfg):
    idx = EvalIndex.build(items, ScanConfig(**cfg))
    return idx, ContaminationScanner(idx).scan(iter(corpus))


def by_id(report):
    return {r.id: r for r in report.items}


# --- text ---------------------------------------------------------------


def test_normalize_nfkc_case_and_punctuation():
    assert normalize("Ｆｕｌｌ-width,   TEXT!!\n\tok_go") == "full width text ok go"
    assert tokenize("") == []


def test_ngram_hashes_stable_and_punctuation_insensitive():
    a = ngram_hashes(tokenize("The quick, brown FOX jumps"), 3)
    b = ngram_hashes(tokenize("the quick brown fox -- jumps"), 3)
    assert len(a) == 3 and np.array_equal(a, b)
    assert ngram_hashes(tokenize("too short"), 3).size == 0
    assert len(set(a.tolist())) == 3


def test_windows_cover_tail():
    toks = [str(i) for i in range(10)]
    wins = windows(toks, 4, 3)
    assert wins[0][1] == ["0", "1", "2", "3"] and wins[-1][1][-1] == "9"
    assert windows(toks[:3], 4, 2) == [(0, ["0", "1", "2"])]


def test_iter_records_jsonl_chat_text_and_fallback_ids(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"id": 7, "text": "plain"}) + "\n\n"
        + json.dumps({"messages": [{"role": "user", "content": "hi"},
                                   {"role": "assistant", "content": "yo"}]}) + "\n"
        + json.dumps({"q": "a", "a": "b", "meta": 1}) + "\n"
    )
    recs = list(iter_records(p))
    assert [r.id for r in recs] == ["7", "c.jsonl:3", "c.jsonl:4"]
    assert recs[1].text == "hi\nyo" and recs[2].text == "a\nb"
    assert next(iter_records(p, fields=["text"])).text == "plain"
    t = tmp_path / "c.txt"
    t.write_text("one doc\n\ntwo doc\n")
    assert [r.text for r in iter_records(t)] == ["one doc", "two doc"]


# --- minhash / lsh ------------------------------------------------------


def test_minhash_estimates_jaccard():
    mh = MinHasher(num_perm=256, shingle=1)
    a = [f"w{i}" for i in range(100)]
    b = [f"w{i}" for i in range(50, 150)]
    est = jaccard(mh.signature(a), mh.signature(b))
    assert abs(est - 50 / 150) < 0.1
    assert jaccard(mh.signature(a), mh.signature(a)) == 1.0
    assert jaccard(mh.signature(a), mh.signature([f"z{i}" for i in range(100)])) < 0.05
    assert jaccard(mh.signature([]), mh.signature(a)) == 0.0


def test_containment_formula():
    # A (10) fully inside B (40): J = 10/40 -> containment 1
    assert containment(10 / 40, 10, 40) == pytest.approx(1.0)
    assert containment(0.0, 10, 40) == 0.0
    assert containment(0.5, 0, 10) == 0.0


def test_lsh_finds_near_duplicate_only():
    mh = MinHasher()
    lsh = LSHIndex(bands=64, rows=2)
    lsh.add(0, mh.signature(tokenize(HASH_Q)))
    lsh.add(1, mh.signature(tokenize(TCP_Q)))
    near = tokenize(HASH_Q.replace("performance", "speed"))
    assert lsh.query(mh.signature(near)) == {0}
    assert lsh.query(mh.signature(tokenize("completely different words here today"))) == set()
    assert lsh.candidate_probability(0.8) > lsh.candidate_probability(0.2)
    with pytest.raises(ValueError):
        LSHIndex(bands=200, rows=2).add(0, mh.signature(near))


# --- embeddings ---------------------------------------------------------


def test_hashed_embedder_ranks_paraphrase_above_unrelated():
    emb = HashedEmbedder()
    emb.fit([t for _, t in EVAL])
    m = emb.embed([CAPITAL_Q, CORPUS[2].text, CORPUS[3].text, ""])
    assert m[0] @ m[0] == pytest.approx(1.0, abs=1e-5)
    assert m[0] @ m[1] > 0.4 > m[0] @ m[2]
    assert not m[3].any()


def test_get_embedder_specs_and_openai_hook(monkeypatch):
    assert isinstance(get_embedder("hashed:512"), HashedEmbedder)
    assert get_embedder("hashed:512").dim == 512
    e = get_embedder("http://localhost:8000/v1|bge")
    assert isinstance(e, OpenAIEmbedder) and e.base_url.startswith("http://localhost")
    with pytest.raises(ValueError):
        get_embedder("word2vec")

    calls = []

    class Resp:
        def __init__(self, n):
            self.n = n

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": i, "embedding": [1.0, float(i)]} for i in
                             reversed(range(self.n))]}

    def fake_post(url, headers, json, timeout):
        calls.append((url, json["model"], len(json["input"])))
        return Resp(len(json["input"]))

    monkeypatch.setattr("evalkit.contamination.embed.httpx.post", fake_post)
    out = OpenAIEmbedder(model="m", base_url="http://x/v1", batch_size=2).embed(["a", "b", "c"])
    assert calls == [("http://x/v1/embeddings", "m", 2), ("http://x/v1/embeddings", "m", 1)]
    assert out.shape == (3, 2) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert out[0, 1] == 0.0  # sorted back by index


# --- scanner ------------------------------------------------------------


def test_exact_leak_is_contaminated_despite_case_and_punctuation():
    _, rep = scan()
    r = by_id(rep)["train"]
    assert r.status == "contaminated" and "ngram" in r.methods
    assert r.overlap_ratio == 1.0 and r.ngram_docs == ["exact"]
    assert r.longest_common_ngram == len(tokenize(TRAIN_Q))
    assert r.max_containment > 0.85


def test_light_edit_breaks_ngrams_but_minhash_catches_it():
    _, rep = scan()
    r = by_id(rep)["hash"]
    assert r.overlap_ratio == 0.0
    assert "minhash" in r.methods and r.status in ("suspicious", "contaminated")
    assert r.containment_doc == "edited"


def test_paraphrase_only_caught_by_embedding():
    _, rep = scan()
    r = by_id(rep)["capital"]
    assert r.methods == ["embedding"] and r.status == "suspicious"
    _, rep2 = scan(methods=("ngram", "minhash"))
    assert by_id(rep2)["capital"].status == "clean"


def test_clean_item_and_summary():
    _, rep = scan()
    assert by_id(rep)["tcp"].status == "clean" and by_id(rep)["tcp"].evidence == ""
    s = rep.summary
    assert s["items"] == 4 and s["clean"] == 1 and s["contaminated"] >= 1
    assert rep.corpus["documents"] == 4 and rep.corpus["flagged_documents"] == 3
    assert rep.clean_ids() == ["tcp"]


def test_partial_overlap_is_suspicious_not_contaminated():
    tail = "Probe sequences grow long because occupied slots form contiguous runs that new keys"
    _, rep = scan(corpus=[Record("p", f"{FILLER} {tail} and more unrelated words follow.")],
                  methods=("ngram",))
    r = by_id(rep)["hash"]
    assert r.status == "suspicious" and 0 < r.overlap_ratio < 0.5
    assert r.longest_common_ngram == len(tokenize(tail))


def test_short_item_skips_ngram_and_matches_are_capped():
    items = [("short", "Capital of Peru? Lima."), ("train", TRAIN_Q)]
    corpus = [Record(f"d{i}", TRAIN_Q) for i in range(30)]
    idx, rep = scan(items, corpus, max_matches=5, batch_docs=7)
    assert idx.item_n[0] == 0
    assert any("n-gram check skipped" in x for x in by_id(rep)["short"].reasons)
    assert len(by_id(rep)["train"].ngram_docs) == 5
    assert rep.corpus["documents"] == 30


def test_config_validation_and_duplicate_ids():
    with pytest.raises(ValueError):
        ScanConfig(methods=("bm25",))
    with pytest.raises(ValueError):
        ScanConfig(n=5, min_n=8)
    with pytest.raises(ValueError):
        EvalIndex.build([("a", "x y"), ("a", "z w")])


def test_thresholds_resolve_from_embedder():
    th = Thresholds().resolve(HashedEmbedder())
    assert (th.cosine_suspicious, th.cosine_contaminated) == (0.4, 0.8)
    th2 = Thresholds(cosine_suspicious=0.9).resolve(OpenAIEmbedder())
    assert (th2.cosine_suspicious, th2.cosine_contaminated) == (0.9, 0.95)


def test_raising_thresholds_clears_flags():
    idx = EvalIndex.build(EVAL)
    strict = Thresholds(min_ngram_hits_suspicious=10_000, overlap_contaminated=1.01,
                        containment_contaminated=1.01, containment_suspicious=1.01,
                        cosine_contaminated=1.01, cosine_suspicious=1.01)
    rep = ContaminationScanner(idx, strict).scan(CORPUS)
    assert rep.summary["clean"] == 4


def test_index_roundtrip(tmp_path):
    idx, rep = scan(n=8, min_n=6)
    idx.save(tmp_path / "i.json")
    loaded = EvalIndex.load(tmp_path / "i.json")
    assert loaded.config.n == 8 and loaded.ids == idx.ids
    assert np.array_equal(loaded.signatures, idx.signatures)
    rep2 = ContaminationScanner(loaded).scan(CORPUS)
    assert [r.status for r in rep2.items] == [r.status for r in rep.items]
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError):
        EvalIndex.load(tmp_path / "bad.json")


def test_decontaminate_drop_and_flag(tmp_path):
    idx = EvalIndex.build(EVAL)
    sc = ContaminationScanner(idx)
    out = tmp_path / "o.jsonl"
    stats = decontaminate(sc, CORPUS, out, mode="drop", level="contaminated")
    kept = [json.loads(x)["id"] for x in out.read_text().splitlines()]
    assert "exact" in stats["dropped_ids"] and "exact" not in kept and "noise" in kept
    stats = decontaminate(sc, CORPUS, out, mode="flag", level="suspicious")
    rows = {json.loads(x)["id"]: json.loads(x) for x in out.read_text().splitlines()}
    assert stats["kept"] == 4 and stats["flagged"] == 3
    assert rows["para"]["_contamination"] == {"level": "suspicious", "eval_ids": ["capital"]}
    assert "_contamination" not in rows["noise"]
    with pytest.raises(ValueError):
        decontaminate(sc, CORPUS, out, mode="delete")


def test_markdown_report_lists_flagged_items():
    _, rep = scan()
    md = to_markdown(rep)
    assert "# Contamination report" in md and "| train | **contaminated**" in md
    assert "| tcp |" not in md
    _, clean = scan(corpus=[CORPUS[3]])
    assert "No eval item shows evidence" in to_markdown(clean)


def test_judge_upgrades_confirmed_and_keeps_rejected():
    idx, rep = scan()
    assert by_id(rep)["capital"].status == "suspicious"
    n = adjudicate(rep, idx, judge_llm("mock"))
    assert by_id(rep)["capital"].judge.startswith(("YES", "NO"))
    assert n == sum(1 for r in rep.items if "judge" in r.methods)
    rep_no = scan()[1]
    adjudicate(rep_no, idx, MockLLM(responder=lambda m: "NO\nnot a restatement"))
    r = by_id(rep_no)["capital"]
    assert r.status == "suspicious" and r.judge == "NO not a restatement"


# --- cli ----------------------------------------------------------------


def test_cli_end_to_end(tmp_path, capsys):
    from evalkit.cli import main

    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps({"id": i, "question": t}) + "\n" for i, t in EVAL))
    tr = tmp_path / "train.jsonl"
    tr.write_text("".join(json.dumps({"id": r.id, "text": r.text}) + "\n" for r in CORPUS))
    out = tmp_path / "out"
    assert main(["contamination", "index", "--eval", str(ev), "--out", str(tmp_path / "i.json")]) == 0
    rc = main(["contamination", "scan", "--index", str(tmp_path / "i.json"), "--train", str(tr),
               "--out", str(out), "--clean-eval", str(tmp_path / "clean.jsonl")])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["items"] == 4 and (out / "report.md").exists()
    assert [json.loads(x)["id"] for x in (tmp_path / "clean.jsonl").read_text().splitlines()] == [
        "tcp"
    ]
    assert main(["contamination", "scan", "--eval", str(ev), "--train", str(tr), "--out",
                 str(out), "--fail-on", "contaminated"]) == 1
    assert main(["contamination", "decontaminate", "--eval", str(ev), "--train", str(tr),
                 "--out", str(tmp_path / "d.jsonl")]) == 0
    assert "contaminated" in capsys.readouterr().out


def test_index_load_overrides_scan_time_methods(tmp_path):
    idx, _ = scan()
    idx.save(tmp_path / "i.json")
    loaded = EvalIndex.load(tmp_path / "i.json", methods=("ngram",))
    assert loaded.embeddings is None
    rep = ContaminationScanner(loaded).scan(CORPUS)
    statuses = {r.id: r.status for r in rep.items}
    assert statuses == {"hash": "clean", "train": "contaminated", "capital": "clean",
                        "tcp": "clean"}
