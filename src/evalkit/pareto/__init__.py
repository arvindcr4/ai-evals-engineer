"""Cost-Quality Pareto Dashboard.

Per tenant, maps every model/router configuration onto cost-per-task vs task
success (with Wilson intervals), finds the non-dominated frontier, simulates
System-One → System-Two escalation routers along a confidence threshold, and
answers "what if this model's price dropped by X%?".
"""

from evalkit.pareto.analysis import (
    ConfigStats,
    RouterPoint,
    apply_price_factor,
    frontier_diff,
    pareto_frontier,
    simulate_router,
    summarize,
    wilson,
)

__all__ = [
    "ConfigStats",
    "RouterPoint",
    "apply_price_factor",
    "frontier_diff",
    "pareto_frontier",
    "simulate_router",
    "summarize",
    "wilson",
]
