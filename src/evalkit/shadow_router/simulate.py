"""Replay a JSONL of production requests through the proxy in-process.

Each row is ``{"request_id"?, "user"?, "messages": [...]}`` (or ``"prompt"``
as shorthand for a single user message). Requests go through the real FastAPI
app via ``TestClient`` so the demo exercises the same code path as ``serve``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from evalkit.core.llm import LLM
from evalkit.core.trajectory import read_jsonl
from evalkit.shadow_router.proxy import ShadowConfig, ShadowRouter, create_app


@dataclass
class SimulationResult:
    requests: int
    ok: int
    failed: int
    sampled: int
    shadow_errors: int
    pairs_path: Path
    wall_s: float


def _to_body(row: dict) -> dict:
    body = {k: v for k, v in row.items() if k not in ("request_id", "prompt")}
    if "messages" not in body:
        body["messages"] = [{"role": "user", "content": row.get("prompt", "")}]
    return body


def simulate(
    requests_path: str | Path,
    primary: LLM,
    candidate: LLM,
    config: ShadowConfig,
    fresh: bool = True,
) -> SimulationResult:
    """Send every request through the proxy, wait for shadows, return counts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

    log_path = Path(config.log_path)
    if fresh and log_path.exists():
        log_path.unlink()
    rows = read_jsonl(requests_path)
    router = ShadowRouter(primary, candidate, config)
    client = TestClient(create_app(router))
    ok = failed = 0
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        headers = {"x-request-id": str(row.get("request_id", f"sim-{i:05d}"))}
        resp = client.post("/v1/chat/completions", json=_to_body(row), headers=headers)
        if resp.status_code == 200:
            ok += 1
        else:
            failed += 1
    router.close()
    stats = router.stats.snapshot()
    return SimulationResult(
        requests=len(rows),
        ok=ok,
        failed=failed,
        sampled=stats["sampled"],
        shadow_errors=stats["shadow_errors"],
        pairs_path=log_path,
        wall_s=time.perf_counter() - t0,
    )
