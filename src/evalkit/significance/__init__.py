"""07 — Statistical Significance Engine.

Turns "B scored 81.4 vs A's 79.2" into "B beats A by +2.2 pts ± 1.1 (95% CI
[1.1, 3.3], p=0.002, n=500 paired)" — or tells you the gap is noise.
"""

from evalkit.significance.compare import (
    Comparison,
    LeaderboardRow,
    compare,
    leaderboard,
    leaderboard_markdown,
    load_scores,
    simulate_results,
)
from evalkit.significance.stats import (
    BootstrapCI,
    adjust_pvalues,
    bootstrap_mean,
    bootstrap_unpaired,
    mcnemar_exact,
    minimum_detectable_effect,
    paired_permutation_test,
    power_two_proportions,
    required_sample_size,
)

__all__ = [
    "BootstrapCI",
    "Comparison",
    "LeaderboardRow",
    "adjust_pvalues",
    "bootstrap_mean",
    "bootstrap_unpaired",
    "compare",
    "leaderboard",
    "leaderboard_markdown",
    "load_scores",
    "mcnemar_exact",
    "minimum_detectable_effect",
    "paired_permutation_test",
    "power_two_proportions",
    "required_sample_size",
    "simulate_results",
]
