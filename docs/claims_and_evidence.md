# Claims and Methodological Evidence Matrix

To prevent misleading claims, this matrix maps every empirical assertion to its exact methodological source, current verification status, reproducible command, committed artifact, and documented limitations.

---

## 1. Feature Capability & Validation Status Taxonomy

To maintain strict scientific and engineering rigor, capabilities in InferPath are classified into six explicit validation tiers:

| Status Tier | Definition | Criteria for Assignment |
| :--- | :--- | :--- |
| **`Implemented`** | Code written and merged into the core repository | Module exists with interfaces defined. |
| **`Unit-tested`** | Hermetic automated tests verify syntax and logic | Passes automated `pytest` suites in isolation. |
| **`Offline-evaluated`** | Benchmarked on simulated or synthetic datasets | Empirically measured across multiple trials under mock/simulated adapters. |
| **`Externally reproduced`** | Fully reproducible via independent runner | All artifacts generated deterministically via `scripts/reproduce_all.py`. |
| **`Real-provider validated`** | Measured against live commercial LLM endpoints | Evaluated via OpenRouter/commercial APIs with live tokens and costs. |
| **`Production-observed`** | Validated under real-world production traffic | Operating continuously under multi-tenant enterprise workload telemetry. |

### Component Status Mapping Matrix

| Feature / Subsystem | Validation Status | Evidence Base & Operational Scope |
| :--- | :--- | :--- |
| **Explicit Rule & Constraint Router** | `Externally reproduced` | Evaluated across 3 trials on 160 `dev` items; verified via `scripts/reproduce_all.py`. |
| **Stage 1 Declared Privacy Interception** | `Externally reproduced` | 100% interception of `confidential_private` and `internal` prompts (HTTP 403 / local). |
| **Adversarial Pattern DLP Detection** | `Offline-evaluated` | 20 adversarial probes: 100% precision, 38.5% FNR under obfuscation. Documented as defense-in-depth only. |
| **Inference Cost Reduction (~85%)** | `Offline-evaluated (Simulated)` | Calibrated on dated OpenRouter pricing models and synthetic token counts on `dev` split. |
| **Real-World Multi-Provider Cost Reduction** | `Not validated` | Live commercial multi-provider evaluation is ongoing; no qualifying full real split published yet. |
| **Exact Caching & TTL Purge** | `Externally reproduced` | SQLite storage, TTL expiration, purge routine, and 7 workload traces verified in unit & trace tests. |
| **Distributed Rate Limiting** | `Implemented & Unit-tested` | In-memory token bucket verified in CI; multi-node Redis architecture documented for Mode 3 production. |
| **Total Cost of Ownership (TCO) Model** | `Modeled & Sensitivity-tested` | Router compute, validation, and serialization amortized against token counts from 10 to 4,000 tokens. |
| **Environmental Footprint (Carbon/Water)** | `Estimated (Not physically measured)` | Derived from modeled PUE (1.2) and WUE (1.8 L/kWh) regional coefficients, not datacenter meters. |
| **Provider Resilience & Bounded Fallback** | `Unit-tested & Offline-evaluated` | Single-tier fallback (Tier C $\to$ Tier D) tested on output validator failures; network outage tests pending. |

---

## 2. Empirical Verification Manifest: Claims, Artifacts, Commands & Tests

Every empirical claim in the documentation and research reports is directly verifiable through a committed artifact, an executable script, and an automated regression test:

| Empirical Claim & Analysis | Committed Result Artifact | Reproducible Command | Validation Script / Test | Evidence Label | Key Empirical Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Adversarial Privacy Interception** | [`data/results/privacy_evaluation.json`](../data/results/privacy_evaluation.json) | `python scripts/evaluate_privacy.py` | [`tests/unit/test_privacy_eval.py`](../tests/unit/test_privacy_eval.py) | `[MEASURED OFFLINE]` | 100% precision on benign code; 38.5% FNR under adversarial obfuscation. |
| **Workload Caching Performance** | [`data/results/caching_workload_analysis.json`](../data/results/caching_workload_analysis.json) | `python scripts/evaluate_caching_workloads.py` | [`tests/unit/test_caching_workloads.py`](../tests/unit/test_caching_workloads.py) | `[SIMULATED TRACE]` | 10x-35x latency speedup on repetitive Zipfian traffic; 1.2ms miss penalty on long-tail. |
| **Expanded Multi-Rater Human Study** | [`data/results/human_study_expanded.json`](../data/results/human_study_expanded.json) | `python scripts/run_expanded_human_study.py` | [`tests/unit/test_human_review_expanded.py`](../tests/unit/test_human_review_expanded.py) | `[EMPIRICAL STUDY]` | Fleiss' $\kappa = 0.8037$ across 3 raters; $\kappa_w = 0.4721$ automated judge agreement. |
| **Distribution Shift & Robustness** | [`data/results/distribution_shift_evaluation.json`](../data/results/distribution_shift_evaluation.json) | `python scripts/evaluate_distribution_shift.py` | [`tests/unit/test_distribution_shift.py`](../tests/unit/test_distribution_shift.py) | `[SIMULATED STRESS]` | 10 OOD shift dimensions evaluated; bounded fallback rescues edge cases. |
| **Simulator Fidelity & Calibration** | [`data/results/simulator_calibration.json`](../data/results/simulator_calibration.json) | `python scripts/evaluate_simulator_calibration.py` | [`tests/unit/test_simulator_calibration.py`](../tests/unit/test_simulator_calibration.py) | `[CALCULATED]` | KS test stat 0.075 ($p=0.6305$), ECE 0.0581, cosine similarity 0.9948. |
| **Total Cost of Ownership (TCO)** | [`data/results/tco_analysis.json`](../data/results/tco_analysis.json) | `python scripts/evaluate_tco.py` | [`tests/unit/test_tco_model.py`](../tests/unit/test_tco_model.py) | `[MODELED]` | Router overhead drops from 10.1% (<25 tokens) to <0.3% (>2,000 tokens). |
| **One-Command Master Reproduction** | [`docs/reproduction_report.md`](reproduction_report.md) | `python scripts/reproduce_all.py` | [`tests/unit/test_reproducibility.py`](../tests/unit/test_reproducibility.py) | `[HERMETIC PIPELINE]` | 12 verification stages execute in ~89s with SHA-256 provenance manifest. |

---

## 3. Methodological Evidence & Documented Limitations

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

## 4. Standardized Provenance & Evidence Badges

All metrics throughout documentation and results reporting are labeled with one of the following badges:

- `[MEASURED LIVE API]`: Directly observed during live OpenRouter HTTP network transactions.
- `[DETERMINISTIC SIMULATION]`: Produced by reproducible offline mock adapters under deterministic seeds.
- `[CALCULATED / DATED PRICING]`: Derived mathematically from token counts and frozen pricing tables.
- `[ESTIMATED / MODELED]`: Derived from external scientific models (e.g., carbon emissions, datacenter PUE).
- `[HUMAN STUDY (EXPANDED PILOT)]`: Scored by human annotators under blind 5-point Likert protocols.
- `[COUNTERFACTUAL ORACLE]`: Theoretical post-hoc optimal bound, not executable in real-time.
