# Claims and Evidence Matrix

To prevent misleading claims, this matrix maps every empirical assertion to its exact methodological source, current verification status, and documented limitations.

---

## 1. Summary Claims & Evidence Matrix

| Research Claim | Methodological Evidence | Status | Documented Caveats & Limitations |
| :--- | :--- | :--- | :--- |
| **Multi-tier routing reduces inference costs by ~85%** | Deterministic offline benchmark on 160-item `dev` split with dated pricing (`data/results/simulated/benchmark_summary.json`) | **Supported in Simulation** | Validated under simulated token counts and dated pricing tables; not yet established across unconstrained enterprise workloads. |
| **InferPath maintains quality parity with frontier models** | Evaluated on standard benchmark tasks (`task_family_breakdown`) | **Not Supported as Universal Parity** | Quality pass rate decreases by 6.3% (from 69.4% to 63.1%) relative to Always-Large. InferPath explores a constrained trade-off space rather than asserting strict quality parity. |
| **Autonomous privacy layer blocks sensitive leaks** | Adversarial DLP evaluation with 20 probes (`data/results/privacy_evaluation.json`) | **Supported as Defense-in-Depth** | Achieves 100% precision on benign queries, but exhibits a 38.5% False Negative Rate under adversarial obfuscation. Callers must declare classifications. |
| **Cache delivers 10× to 35× latency speedups** | Workload trace simulation across 7 profiles (`data/results/caching_workload_analysis.json`) | **Qualified by Workload Entropy** | Highly effective on repetitive enterprise queries (84.6% hit rate); introduces 1.2 ms uncompensated lookup overhead on unique long-tail queries. |
| **Multi-rater human agreement confirms output usability** | 120-item blind study across 3 annotators + lead adjudicator (`data/results/human_study_expanded.json`) | **Supported in Expanded Pilot** | High human-human quadratic agreement ($\kappa_w \approx 0.80$); moderate automated judge agreement ($\kappa_w \approx 0.47$) confirms judges should not serve as sole ground truth. |
| **Pure routing compute overhead is negligible** | End-to-end latency profiling across concurrency levels ($N \le 100$) (`data/results/benchmarks/scalability_analysis.json`) | **Supported by Microbenchmarks** | Router decision logic executes in $<1.0\text{ ms}$; however, at $N \ge 50$, SQLite lock contention and queueing drive tail latency unless Redis is enabled. |
| **Router reduces energy and carbon footprint** | Environmental physics model based on Joules/token calculations (`environmental.py`) | **Estimated / Modeled** | Derived from published average datacenter PUE and GPU wattages, not directly measured through hardware power meters. |
| **Production viability at horizontal enterprise scale** | Architectural boundaries across 3 operational modes (`docs/production_readiness.md`) | **Candidate Prototype** | Core repository provides local and Docker Compose implementations; multi-node Redis clusters, OTel collectors, and K8s orchestration remain deployment-layer responsibilities. |

---

## 2. Standardized Provenance & Evidence Badges

All metrics throughout documentation and results reporting are labeled with one of the following badges:

- `[MEASURED LIVE API]`: Directly observed during live OpenRouter HTTP network transactions.
- `[DETERMINISTIC SIMULATION]`: Produced by reproducible offline mock adapters under deterministic seeds.
- `[CALCULATED / DATED PRICING]`: Derived mathematically from token counts and frozen pricing tables.
- `[ESTIMATED / MODELED]`: Derived from external scientific models (e.g., carbon emissions, datacenter PUE).
- `[HUMAN STUDY (EXPANDED PILOT)]`: Scored by human annotators under blind 5-point Likert protocols.
- `[COUNTERFACTUAL ORACLE]`: Theoretical post-hoc optimal bound, not executable in real-time.
