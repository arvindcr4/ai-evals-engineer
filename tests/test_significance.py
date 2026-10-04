import json
import math

import numpy as np
import pytest

from evalkit.cli import main
from evalkit.core.trajectory import write_jsonl
from evalkit.significance import (
    adjust_pvalues,
    bootstrap_mean,
    compare,
    leaderboard,
    load_scores,
    mcnemar_exact,
    minimum_detectable_effect,
    paired_permutation_test,
    power_two_proportions,
    required_sample_size,
    simulate_results,
)
from evalkit.significance.stats import norm_cdf, norm_ppf


def test_norm_ppf_inverts_cdf():
    for p in (1e-6, 0.01, 0.025, 0.3, 0.5, 0.8, 0.975, 0.999):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, rel=1e-8)
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)


@pytest.mark.parametrize("method", ["bca", "percentile"])
def test_bootstrap_ci_covers_true_delta_at_nominal_rate(method):
    rng = np.random.default_rng(1)
    true_delta, covered, trials = 0.3, 0, 200
    for t in range(trials):
        diffs = rng.exponential(1.0, 80) - 1.0 + true_delta  # skewed, mean = true_delta
        ci = bootstrap_mean(diffs, n_boot=1000, method=method, seed=t)
        covered += ci.low <= true_delta <= ci.high
    assert 0.89 <= covered / trials <= 0.99


def test_permutation_false_positive_rate_is_about_alpha():
    rng = np.random.default_rng(7)
    trials, rejections = 400, 0
    for t in range(trials):
        a = rng.normal(0.6, 0.2, 60)
        b = a + rng.normal(0, 0.1, 60)  # no true difference, correlated with a
        rejections += paired_permutation_test(b - a, n_perm=500, seed=t) < 0.05
    assert 0.02 <= rejections / trials <= 0.085


def test_mcnemar_false_positive_rate_and_power():
    null_rej = sum(
        compare(load_scores(simulate_results({"a": 0.8, "b": 0.8}, 300, seed=s)), "a", "b",
                n_boot=300).significant
        for s in range(150)
    )
    assert null_rej / 150 <= 0.08
    big = compare(load_scores(simulate_results({"a": 0.70, "b": 0.85}, 500, seed=3)), "a", "b",
                  n_boot=2000)
    assert big.significant and big.winner == "b" and big.test == "mcnemar-exact"
    assert big.ci.low < 0.15 < big.ci.high


def test_mcnemar_exact_known_values():
    a = [1] * 10 + [0] * 2 + [1] * 50
    b = [0] * 10 + [1] * 2 + [1] * 50
    p, only_a, only_b = mcnemar_exact(a, b)
    assert (only_a, only_b) == (10, 2)
    expected = 2 * sum(math.comb(12, k) for k in range(3)) / 2**12
    assert p == pytest.approx(expected)
    assert mcnemar_exact([1, 0], [1, 0])[0] == 1.0


def test_adjust_pvalues_holm_and_bh():
    p = [0.01, 0.04, 0.03, 0.005]
    assert adjust_pvalues(p, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert adjust_pvalues(p, "bh") == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert adjust_pvalues(p, "bonferroni") == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert adjust_pvalues(p, "none") == p


def test_power_matches_textbook_sample_size():
    # Classic two-proportion example: 50% vs 60%, alpha .05, power .8 -> ~385-388 per arm.
    n = required_sample_size(0.5, 0.1, power=0.8)
    assert 380 <= n <= 392
    assert power_two_proportions(n, 0.5, 0.6) >= 0.8 > power_two_proportions(n - 5, 0.5, 0.6)
    # Pairing with low discordance needs far fewer items.
    paired = required_sample_size(0.8, 0.02, paired=True, discordance=0.06)
    assert paired < required_sample_size(0.8, 0.02) / 3
    mde = minimum_detectable_effect(n, 0.5)
    assert mde == pytest.approx(0.1, abs=0.003)


def test_two_points_on_fifty_samples_is_noise():
    rows = []
    for i in range(50):
        rows.append({"item_id": i, "model": "A", "score": int(i % 5 != 0)})
        rows.append({"item_id": i, "model": "B", "score": int(i % 5 != 0 or i == 0)})
    c = compare(load_scores(rows), "A", "B", n_boot=2000)
    assert c.delta == pytest.approx(0.02)
    assert not c.significant
    assert c.verdict().startswith("No significant difference between A and B")
    assert "n=50 paired" in c.verdict()


def test_clustered_bootstrap_widens_ci_for_correlated_items():
    rows = simulate_results({"a": 0.7, "b": 0.72}, 600, n_clusters=20, cluster_sd=1.5, seed=4)
    for r in rows:
        if r["model"] == "b" and int(r["cluster"][1:]) < 8:
            r["score"] = 1  # cluster-level shift: a few conversations all flip together
    table = load_scores(rows)
    clustered = compare(table, "a", "b", n_boot=2000)
    naive = compare(table, "a", "b", n_boot=2000, clustered=False)
    assert clustered.test == "cluster-permutation" and clustered.n_clusters == 20
    assert clustered.ci.half_width > 1.3 * naive.ci.half_width
    assert clustered.p_value > naive.p_value


def test_unpaired_fallback_and_verdict_format():
    rng = np.random.default_rng(0)
    rows = [{"item_id": f"a{i}", "model": "A", "score": float(rng.normal(5, 1))}
            for i in range(200)]
    rows += [{"item_id": f"b{i}", "model": "B", "score": float(rng.normal(5.6, 1))}
             for i in range(200)]
    c = compare(load_scores(rows), "A", "B", n_boot=2000)
    assert not c.paired and c.test == "unpaired-permutation"
    assert c.significant and c.unit == "" and c.scale == 1.0
    assert "B beats A by +0." in c.verdict() and "unpaired" in c.verdict()


def test_load_scores_long_format_and_missing_model():
    rows = [{"item_id": 1, "model": "A", "metric": "f1", "score": 0.5},
            {"item_id": 1, "model": "A", "metric": "em", "score": 0.0},
            {"item_id": 1, "model": "A", "metric": "f1", "score": 0.7}]
    t = load_scores(rows, "f1")
    assert t["A"]["1"][0] == pytest.approx(0.6)
    with pytest.raises(KeyError):
        compare(t, "A", "Z")


def test_leaderboard_tiers_and_correction():
    rows = simulate_results({"big": 0.90, "big2": 0.895, "small": 0.70}, 800, seed=2)
    lb, pairs, adj = leaderboard(load_scores(rows), n_boot=1000)
    assert [r.model for r in lb][2] == "small"
    assert lb[0].tier == lb[1].tier == 1 and lb[2].tier == 2
    assert len(pairs) == 3 and all(p2 >= p1 for p1, p2 in zip([c.p_value for c in pairs], adj))


def test_cli_compare_and_power(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    write_jsonl(path, simulate_results({"A": 0.7, "B": 0.8}, 400, seed=5))
    assert main(["significance", "compare", "--results", str(path), "--a", "A", "--b", "B",
                 "--n-boot", "1000", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["winner"] == "B" and out[0]["verdict"].startswith("B beats A by +")
    assert main(["significance", "power", "--baseline", "0.8", "--delta", "0.02",
                 "--json"]) == 0
    text = capsys.readouterr().out
    assert json.loads(text[text.index("{"):])["required_n"] > 5000
