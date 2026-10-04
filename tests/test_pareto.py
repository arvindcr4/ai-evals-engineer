import json
import math

import pytest

from evalkit.cli import main
from evalkit.pareto import (
    apply_price_factor,
    frontier_diff,
    pareto_frontier,
    simulate_router,
    summarize,
    wilson,
)
from evalkit.pareto.analysis import mark_router_frontier, recommend
from evalkit.pareto.dashboard import auto_router_pair, best_router, build_views, write_outputs
from evalkit.pareto.generate import generate_rows


def _rows(tenant, config, outcomes, cost, conf=None):
    return [
        {
            "tenant": tenant,
            "config": config,
            "task_id": f"t{i}",
            "success": bool(o),
            "cost_usd": cost,
            "latency_s": 1.0,
            **({"confidence": conf[i]} if conf else {}),
        }
        for i, o in enumerate(outcomes)
    ]


def test_frontier_basic_and_ties():
    pts = [(1.0, 0.5), (2.0, 0.4), (3.0, 0.9), (0.5, 0.2), (3.0, 0.8), (4.0, 0.9)]
    assert pareto_frontier(pts) == [3, 0, 2]
    assert pareto_frontier([(1.0, 0.5), (1.0, 0.5)]) == [0]
    assert pareto_frontier([]) == []


def test_frontier_matches_brute_force_on_random_points():
    import random

    rng = random.Random(0)
    for _ in range(50):
        pts = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(12)]
        brute = {
            i
            for i, (c, s) in enumerate(pts)
            if not any(
                (c2 <= c and s2 >= s) and (c2 < c or s2 > s)
                for j, (c2, s2) in enumerate(pts)
                if j != i
            )
        }
        assert set(pareto_frontier(pts)) == brute


def test_wilson_interval():
    lo, hi = wilson(45, 60)
    assert lo < 0.75 < hi and 0.6 < lo and hi < 0.86
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_summarize_stats_and_dominance():
    rows = (
        _rows("t", "cheap", [1, 0, 1, 0], 0.001)
        + _rows("t", "pricey-bad", [1, 0, 0, 0], 0.01)
        + _rows("t", "pricey-good", [1, 1, 1, 0], 0.02)
    )
    stats = {s.config: s for s in summarize(rows)["t"]}
    assert stats["cheap"].success_rate == 0.5
    assert stats["cheap"].cost_per_success == pytest.approx(0.002)
    assert stats["pricey-bad"].on_frontier is False
    assert stats["cheap"].on_frontier and stats["pricey-good"].on_frontier
    never = summarize(_rows("t", "zero", [0, 0], 0.01))["t"][0]
    assert math.isinf(never.cost_per_success)


def test_router_endpoints_and_cascade_cost():
    cheap = _rows("t", "s1", [1, 0, 1, 0], 0.001, conf=[0.9, 0.2, 0.8, 0.4])
    exp = _rows("t", "s2", [1, 1, 1, 1], 0.01)
    rows = cheap + exp
    pts = simulate_router(rows, "t", "s1", "s2", thresholds=[0.0, 0.5, 1.01])
    never, mid, always = pts
    assert never.escalation_rate == 0 and never.success_rate == 0.5
    assert never.mean_cost == pytest.approx(0.001)
    assert mid.escalation_rate == 0.5 and mid.success_rate == 1.0
    assert mid.mean_cost == pytest.approx((0.001 * 2 + 0.011 * 2) / 4)
    assert always.success_rate == 1.0 and always.mean_cost == pytest.approx(0.011)
    pre = simulate_router(rows, "t", "s1", "s2", thresholds=[1.01], cascade=False)[0]
    assert pre.mean_cost == pytest.approx(0.01)
    stats = summarize(rows)["t"]
    mark_router_frontier(stats, pts)
    assert mid.on_frontier and not always.on_frontier


def test_router_errors_and_confidence_fallback():
    rows = _rows("t", "a", [1, 0] * 20, 0.001) + _rows("t", "b", [1] * 40, 0.01)
    with pytest.raises(KeyError):
        simulate_router(rows, "t", "a", "missing")
    pts = simulate_router(rows, "t", "a", "b", thresholds=[0.5])
    assert 0.2 < pts[0].escalation_rate < 0.8


def test_price_whatif_moves_frontier():
    rows = (
        _rows("t", "mid", [1, 1, 0, 0], 0.005)
        + _rows("t", "big", [1, 1, 1, 0], 0.02)
        + _rows("t", "best", [1, 1, 1, 1], 0.08)
    )
    cheaper = apply_price_factor(rows, "best", 0.1)
    assert rows[-1]["cost_usd"] == 0.08 and cheaper[-1]["cost_usd"] == pytest.approx(0.008)
    diff = frontier_diff(summarize(rows), summarize(cheaper))["t"]
    assert diff["joined"] == [] and diff["left"] == ["big"]
    assert diff["cost_per_success"]["best"] == pytest.approx((0.08, 0.008))
    assert apply_price_factor(rows, "b*", 2.0)[-1]["cost_usd"] == pytest.approx(0.16)
    with pytest.raises(ValueError):
        apply_price_factor(rows, "best", -1)


def test_recommend_value_pick():
    rows = _rows("t", "a", [1] * 9 + [0], 0.001) + _rows("t", "b", [1] * 10, 0.1)
    rec = recommend(summarize(rows)["t"], tolerance=0.15)
    assert rec["best_quality"] == "b" and rec["value_pick"] == "a"
    assert rec["savings_vs_best"] == pytest.approx(0.99)


def test_generator_is_deterministic_and_realistic():
    a, b = generate_rows(40, seed=5), generate_rows(40, seed=5)
    assert a == b and len(a) == 4 * 7 * 40
    summary = summarize(a)
    for stats in summary.values():
        by = {s.config: s for s in stats}
        assert by["s2-large+reasoning"].success_rate > by["s1-nano"].success_rate
        assert by["s2-frontier"].mean_cost > 10 * by["s1-mini"].mean_cost
        assert not by["s2-legacy"].on_frontier


def test_dashboard_outputs(tmp_path):
    rows = generate_rows(40, seed=2)
    views = build_views(rows, ("s1-mini+rag", "s2-large+reasoning"))
    assert len(views) == 4 and all(v.router for v in views)
    assert auto_router_pair(views[0].stats)[1] in {s.config for s in views[0].stats}
    assert best_router(views[0]) is not None
    paths = write_outputs(views, tmp_path, "T", "sub")
    page = paths["html"].read_text()
    assert page.count('<section class="tenant"') == 4
    assert "prefers-color-scheme: dark" in page and "<svg" in page and "<select" in page
    data = json.loads(paths["json"].read_text())
    assert set(data["tenants"]) == {v.tenant for v in views}
    assert "| config |" in paths["md"].read_text()


def test_cli_generate_build_whatif(tmp_path):
    res = tmp_path / "r.jsonl"
    assert main(["pareto", "generate", "--out", str(res), "--tasks", "30"]) == 0
    assert main(["pareto", "build", str(res), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "pareto.html").exists()
    assert (
        main(
            [
                "pareto",
                "whatif",
                str(res),
                "--config",
                "s2-*",
                "--factor",
                "0.5",
                "--out",
                str(tmp_path / "w"),
            ]
        )
        == 0
    )
    data = json.loads((tmp_path / "w" / "whatif.json").read_text())
    assert data["whatif"]["factor"] == 0.5
    with pytest.raises(SystemExit):
        main(["pareto", "whatif", str(res), "--config", "nope", "--factor", "0.5"])
