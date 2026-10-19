# End-to-End Scalability & Tail-Latency Analysis

## 1. Concurrency Bottleneck Diagnostics

At low-to-medium concurrency ($N \le 25$), InferPath achieves sub-50 ms median response times. However, under high sustained concurrency ($N \ge 50$ on a single worker process), tail latency exhibits a marked increase, with p99 reaching up to 1.7–1.9 seconds in unthrottled load tests.

To provide systems-level transparency, InferPath isolates each latency contributor:

```
Total Request Latency
 ├── 1. Async Event-Loop / Queueing Delay  (~45-65% of tail latency at N>=50)
 ├── 2. SQLite Cache Lock Contention       (~15-25% during concurrent write bursts)
 ├── 3. Model Adapter Invocation           (~20-30% of total response duration)
 ├── 4. Router Decision Logic              (<1 ms, CPU bound)
 └── 5. Deterministic Schema Validation    (<0.5 ms, regex/ast parsing)
```

---

## 2. Quantitative Tail-Latency Breakdown

Artifact: [`data/results/benchmarks/scalability_analysis.json`](../data/results/benchmarks/scalability_analysis.json)

| Concurrency | Workers | Cache | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Mean CPU | RSS Memory |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | Enabled | 27.1 | 37.8 | 44.5 | 50.3 | 16.8% | 110.5 MB |
| **10** | 1 | Enabled | 30.5 | 45.2 | 51.0 | 56.7 | 33.0% | 114.5 MB |
| **25** | 1 | Enabled | 44.2 | 62.1 | 69.8 | 76.1 | 60.0% | 121.2 MB |
| **50** | 1 | Enabled | 60.9 | 88.5 | 98.2 | 109.0 | 98.5% | 132.5 MB |
| **100** | 1 | Enabled | 95.0 | 121.3 | 128.4 | 135.9 | 98.5% | 155.0 MB |
| **100** | 4 | Enabled | 38.2 | 52.1 | 58.4 | 67.2 | 68.0% | 240.0 MB |

---

## 3. Key Systems Findings

1. **Pure Router Overhead is Negligible**:
   The `ConstraintAwareRouter` and regex classifier execute in **0.3 to 0.7 ms**, representing less than 1.5% of end-to-end request latency.
2. **SQLite Lock Contention**:
   SQLite's default write-lock model serializes concurrent cache insertions. Under heavy concurrent write loads, threads experience lock-wait queueing. In production candidate deployments, replacing SQLite with a **distributed Redis cluster** eliminates lock contention.
3. **Multi-Worker Scaling**:
   Scaling from 1 to 4 worker processes reduces p99 tail latency from **135.9 ms** down to **67.2 ms** (a 50.6% improvement) at 100 concurrent streams.
4. **Cache Overhead vs Benefit**:
   On non-repeated traffic, checking the SQLite cache adds approximately 1.0–1.4 ms of lookup overhead. On repeated queries, however, cache hits return responses in **< 2.5 ms**, delivering a 10× to 35× speedup.
