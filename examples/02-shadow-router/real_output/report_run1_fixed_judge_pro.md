# Shadow report: `deepseek-flash` vs `deepseek-v4-pro`

**Verdict: PROMOTE** — judge says candidate wins significantly

47 shadow pairs logged, 47 comparable (both sides succeeded).

| metric | primary | shadow | delta |
|---|---:|---:|---:|
| mean cost / request | $0.000687 | $0.000184 | -73.2% |
| latency p50 | 2.876s | 1.401s | -1.475s |
| latency p95 | 3.353s | 1.789s | -1.564s |
| mean output tokens | 160.0 | 142.9 | |
| error rate | 0.0% | 0.0% | |
| truncated (finish=length) | 0.0% | 0.0% | |

## Agreement

- exact (normalised): **0.0%**
- near (token Jaccard ≥ 0.6): **0.0%**
- mean token Jaccard: 0.284; mean similarity: 0.233

## Judge (`deepseek-v4-pro`, both orders)

- shadow wins 21, losses 4, ties 22
- win rate (decided pairs) **84.0%** (95% CI 65.3%–93.6%)
- non-loss rate 91.5%
- order flips (verdict changed with A/B order, scored tie) 22; unparsed verdicts 0; judge errors 0
- slot-A share of decisive single-order verdicts 67% (0.5 = no position bias)
- judge spend: 94 calls, $0.0511

## Largest disagreements

| request | sim | judge | primary | shadow |
|---|---:|---|---|---|
| r00058 | 0.25 | loss | The best practice is **cache-aside with explicit invalidation**, not TTL alone.  1. **Inv… | Use **cache-aside with explicit invalidation on writes**:  1. On read: check cache; on mi… |
| r00016 | 0.31 | loss | A CUDA out-of-memory (OOM) error occurs when your program requests more GPU memory (VRAM)… | CUDA out of memory means the GPU couldn't allocate enough VRAM for a requested tensor or … |
| r00044 | 0.37 | loss | The Rust borrow checker enforces memory safety at compile time without a garbage collecto… | The borrow checker enforces Rust's ownership rules at compile time:  - Each value has one… |
| r00030 | 0.37 | loss | First, verify your backend configuration supports locking (e.g., S3 with DynamoDB, GCS, A… | Check the backend configuration first: confirm the lock table (DynamoDB for S3) or lock r… |
| r00055 | 0.08 | win | First, verify your **embedding model matches the one used at index time** — a mismatch is… | Check these in order:  1. **Ground truth quality** — verify your "expected" neighbors are… |
| r00025 | 0.09 | win | Feature flag cleanup often breaks under load due to race conditions and resource contenti… | Feature flag cleanup usually breaks under load because the flag isn't just a boolean—it's… |
| r00040 | 0.09 | win | Nginx rate limiting typically "breaks" under load due to the **burst** and **nodelay** pa… | Nginx `limit_req` is per-worker, not global. Under load, each worker keeps its own leaky … |
| r00002 | 0.10 | win | Vector index recall degrades under load primarily due to resource contention and algorith… | Recall usually drops under load because the index is being searched with fewer candidates… |
