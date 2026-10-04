"""Shadow Routing Comparator.

An OpenAI-compatible proxy that always answers from the primary model while
mirroring a sticky, hash-sampled slice of traffic to a candidate model in the
background, logging both sides so a report can diff quality, cost and latency
without any user ever seeing the candidate's output.
"""

from evalkit.shadow_router.proxy import (
    PairLogger,
    ShadowConfig,
    ShadowRouter,
    create_app,
    sample_key,
    should_shadow,
)
from evalkit.shadow_router.report import ShadowReport, build_report

__all__ = [
    "PairLogger",
    "ShadowConfig",
    "ShadowReport",
    "ShadowRouter",
    "build_report",
    "create_app",
    "sample_key",
    "should_shadow",
]
