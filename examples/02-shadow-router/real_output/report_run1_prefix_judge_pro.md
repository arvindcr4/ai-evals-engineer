# Shadow report: `deepseek-flash` vs `deepseek-v4-pro`

**Verdict: HOLD** — no significant quality difference and no cost win; collect more pairs

47 shadow pairs logged, 47 comparable (both sides succeeded).

| metric | primary | shadow | delta |
|---|---:|---:|---:|
| mean cost / request | $0.000687 | $0.000184 | -73.2% |
| latency p50 | 2.876s | 1.401s | -1.475s |
| latency p95 | 3.353s | 1.789s | -1.564s |
| mean output tokens | 160.0 | 142.9 | |
| error rate | 0.0% | 0.0% | |

## Agreement

- exact (normalised): **0.0%**
- near (token Jaccard ≥ 0.6): **0.0%**
- mean token Jaccard: 0.284; mean similarity: 0.064

## Judge (`deepseek-v4-pro`, both orders)

- shadow wins 12, losses 7, ties 28
- win rate (decided pairs) **63.2%** (95% CI 41.0%–80.9%)
- non-loss rate 85.1%

## Largest disagreements

| request | sim | judge | primary | shadow |
|---|---:|---|---|---|
| r00074 | 0.00 | tie | When resolving rebase conflicts:  1. **Resolve per-commit**, not all at once — `git rebas… | Resolve conflicts as they appear, then `git add <file>` and `git rebase --continue`. Key … |
| r00058 | 0.01 | tie | The best practice is **cache-aside with explicit invalidation**, not TTL alone.  1. **Inv… | Use **cache-aside with explicit invalidation on writes**:  1. On read: check cache; on mi… |
| r00030 | 0.01 | tie | First, verify your backend configuration supports locking (e.g., S3 with DynamoDB, GCS, A… | Check the backend configuration first: confirm the lock table (DynamoDB for S3) or lock r… |
| r00011 | 0.01 | tie | For FastAPI dependency injection, follow these best practices:  1. **Use `Depends` for re… | Use `Depends()` with small, composable callables, and prefer `Annotated` for clarity:  ``… |
| r00001 | 0.01 | tie | For a small team, prioritize simplicity and security:  1. **Use a managed auth provider**… | Store the refresh token server-side (encrypted), never in the browser. On login, issue yo… |
| r00002 | 0.02 | win | Vector index recall degrades under load primarily due to resource contention and algorith… | Recall usually drops under load because the index is being searched with fewer candidates… |
| r00066 | 0.02 | tie | For a small team, handle rebase conflicts locally and collaboratively:  1. **Rebase often… | Rebase one branch at a time, and resolve conflicts commit-by-commit:  1. `git rebase main… |
| r00052 | 0.03 | loss | First, check the test logs for **timing or ordering dependencies**: look for hardcoded sl… | Check the failure pattern first: is it the same test, same step, or random? Then verify: … |
