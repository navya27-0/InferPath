# InferPath: Resource-Aware Multi-Tier AI Inference Router

Routing user requests to the least resource-intensive processing tier that meets correctness, quality, privacy, and latency requirements.

## 1. Problem Statement & Research Formulation

Modern AI application architectures frequently dispatch all user queries to frontier large language models (LLMs) regardless of query complexity, resulting in unnecessary monetary expense, latency overhead, and datacenter electricity consumption. 

### Central Research Question
> **Can a privacy-aware, multi-tier inference router reduce cost and latency while maintaining a configurable minimum quality level across heterogeneous workloads?**

InferPath answers this question by formulating routing as a **constrained multi-objective optimization problem**:

$$\min_{t \in \mathcal{T}_{\text{candidate}}} \text{Cost}(t) + \lambda \cdot \text{Latency}(t)$$

$$\text{subject to:} \quad \hat{Q}(t) \ge Q_{\min}, \quad \text{Privacy satisfied}, \quad \text{Cost} \le B_{\max}, \quad \text{Latency} \le L_{\max}$$

Where:
- $\mathcal{T}_{\text{candidate}} \subseteq \{\text{Tier A, Tier B, Tier C, Tier D}\}$ is the set of admissible tiers satisfying input syntax and privacy classification.
- $\hat{Q}(t)$ is the estimated output quality for tier $t$, bounded below by the user's required threshold $Q_{\min}$ (default 0.70).
- $B_{\max}$ and $L_{\max}$ are optional caller-specified budget and latency ceilings.
- $\lambda \ge 0$ is a configurable trade-off coefficient between latency (ms) and cost (USD).

### Multi-Tier Architecture & Strategies
InferPath cascades requests across four discrete processing tiers:
1. **Tier A — Deterministic Handlers**: Local AST-based math (`SafeCalculator`), dimensional unit conversions, and date formatting (zero token cost, $<0.1$ ms latency).
2. **Tier B — Structured Templates**: Jinja2-based document generators with clarification guards when required slots are missing.
3. **Tier C — Small Models**: Efficient ~8B parameter language models (e.g. Llama-3.1-8B) for standard queries.
4. **Tier D — Large Frontier Models**: High-capacity ~70B+ parameter models (e.g. Llama-3.3-70B) for complex synthesis, proofs, and ambiguity.

To rigorously evaluate the system, InferPath implements and benchmarks five routing strategies:
- **`System_A` (Always-Large)**: Dispatches all queries to frontier models (standard industry baseline).
- **`System_Small` (Always-Small)**: Directs all queries to ~8B models (extreme low-cost baseline).
- **`System_B` (Rules + Large + Cache)**: Traditional heuristic caching and rules with large-model fallback.
- **`System_C` (Constraint-Aware Router)**: InferPath's proposed optimizer choosing the lowest-cost tier meeting all constraints, with single-level Tier C $\to$ Tier D validation fallback.
- **`System_Learned` (Random Forest Classifier)**: Proof-of-concept learned baseline evaluated on a synthetic calibration corpus using 15 prompt structural features.

### Production Hardening & Observability
Beyond research benchmarks, InferPath is engineered as an enterprise-grade service:
- **Circuit Breaker**: 3-state machine (`CLOSED`, `OPEN`, `HALF_OPEN`) wrapped around OpenRouter API adapters with automatic fail-fast and recovery timeouts.
- **Sliding-Window Rate Limiting**: In-memory IP/token rate limiter enforcing configurable throughput thresholds.
- **Prometheus Observability**: Native exposition endpoint at `/metrics` tracking request counts, histograms, token spend, cache hits, and privacy blocks.
- **Asynchronous Caching**: Non-blocking SQLite exact-response cache powered by `aiosqlite`.

---

## 2. Request Lifecycle Architecture

The Mermaid diagram below reflects the implemented execution pipeline (`inferpath.pipeline.RequestPipeline`):

```mermaid
flowchart TD
    Req([User Request]) --> Stage1[1. Input Normalization & Privacy Check]
    Stage1 -->|Confidential / Internal Policy Violation| Block[HTTP 403 Privacy Interception]
    Stage1 -->|Valid| Stage2[2. Freshness & Time-Sensitivity Detection]
    Stage2 --> Stage3[3. Rule-Based Route Analysis]
    Stage3 --> Stage4[4. Canonical Cache Key Computation]
    Stage4 --> CacheCheck{Cache Eligible?}
    CacheCheck -->|Yes| CacheLookup[5. SQLite Exact Cache Lookup]
    CacheCheck -->|No: Freshness / Deterministic / Private| TierExec
    CacheLookup -->|Hit| CacheReturn([Cached Response])
    CacheLookup -->|Miss| TierExec[6. Tier Handler Execution]
    TierExec --> PathA[Tier A: Deterministic Rules & Units]
    TierExec --> PathB[Tier B: Template Slot Filler]
    TierExec --> PathC[Tier C: Small Model Adapter]
    TierExec --> PathD[Tier D: Large Model Adapter]
    PathA --> Val[7. Deterministic Output Validation]
    PathB --> Val
    PathC --> Val
    PathD --> Val
    Val -->|Passed| CacheStore{Store in Cache?}
    Val -->|Failed & Tier C| Fallback[8. Single-Level Fallback to Tier D]
    Fallback --> Val2[Re-validate Large Model Output]
    Val2 --> CacheStore
    CacheStore -->|Eligible & Valid| SaveCache[(Update SQLite Exact Cache)]
    CacheStore -->|Ineligible / Private| Telemetry
    SaveCache --> Telemetry[9. Telemetry & Experiment Logging]
    Telemetry --> Resp([Structured Response & Audit Record])
```

---

## 3. Sixty-Second Quickstart

InferPath is designed to run hermetically out of the box without requiring external network connectivity or paid API credentials.

### Prerequisites
- Python 3.11 (`3.11.x`)
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip` / `venv`

### Windows (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/navya27-0/InferPath.git
cd inferpath

# 2. Set up virtual environment and install dependencies
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -e ".[dev]"

# 3. Run default offline test suite (hermetic mock mode, zero network access)
pytest -v

# 4. Start the FastAPI backend service
uvicorn inferpath.api.main:app --port 8000

# 5. In a second terminal, launch the Streamlit research dashboard
streamlit run app/dashboard.py

# 6. Send a sample request to the running API
curl -X POST http://127.0.0.1:8000/v1/respond `
  -H "Content-Type: application/json" `
  -d '{"prompt": "Convert 100 meters to feet", "privacy_class": "public"}'
```

### macOS / Linux (Bash)

```bash
# 1. Clone repository
git clone https://github.com/navya27-0/InferPath.git && cd inferpath

# 2. Set up virtual environment and install dependencies
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Run default offline test suite
pytest -v

# 4. Start the FastAPI backend service
uvicorn inferpath.api.main:app --port 8000 &

# 5. Launch the Streamlit research dashboard
streamlit run app/dashboard.py &

# 6. Send a sample request to the API
curl -s -X POST http://127.0.0.1:8000/v1/respond \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Convert 100 meters to feet", "privacy_class": "public"}' | jq
```

> **Default Safety Invariant**: Without an explicit `OPENROUTER_API_KEY` set in `.env`, InferPath automatically initializes deterministic `MockSmallModel` and `MockLargeModel` adapters. No paid network requests can occur inadvertently.

---

## 4. Evaluation Results

The evaluation artifacts under `data/results/` are organized into distinct categories to uphold empirical reproducibility and experimental integrity:

- **`data/results/simulated/`**: Full offline benchmark runs, parameter ablations, and item-level logs produced with deterministic mock adapters across the development (`dev`) split.
- **`data/results/simulated_test/`**: Diagnostic held-out (`test`) split runs evaluating mock adapter coverage and task family generalization.
- **`data/results/benchmarks/`**: Real ASGI system-level concurrency, throughput (RPS), and latency distribution benchmarks across worker pools.
- **`data/results/fixtures/`**: Static synthetic reference fixtures conforming to schema protocols, used exclusively for hermetic offline unit tests, regression suites, and CI validation without network dependencies.
- **`data/results/real/`**: Reserved exclusively for genuine, measured empirical evaluation runs against live commercial model endpoints (e.g., OpenRouter API) recorded with frozen timestamps, network call latencies, and token expenditures. Synthetic fixtures are strictly prohibited in this directory.

<!-- RESULTS_TABLE_START -->
### Real-Provider Evaluation Results

> **Status**: *No qualifying real-provider evaluation run is currently available.*
> 
> All evaluation data currently present in `data/results/` was executed in offline hermetic simulated mode (`adapter_mode: simulated`) on the development split (`split: dev`). Live OpenRouter network calls were not performed during the automated benchmark suite.

### Empirical Simulated Evaluation Benchmark (`dev` split, 3 repeats)

> [!NOTE]
> **Simulated Workload & Reproducibility Notice**: Results are evaluated under reproducible offline simulated workloads using deterministic mock adapters. The `± 0.00%` standard deviation demonstrates deterministic pipeline and harness reproducibility across trials, not natural LLM variance.

> [!IMPORTANT]
> **Research Trade-Off Analysis**: Rather than asserting uncompromised quality parity, InferPath explores the multi-tier cost-latency-quality trade-off space. Under this benchmark workload, full InferPath (System C) achieves an **84.9% cost reduction** and **70.2% latency reduction** (from 89.5 ms to 26.7 ms, equivalent to a 3.35× speedup) relative to the Always-Large baseline (System A), with a measured quality pass rate decrease from 69.4% to 63.1% ($\Delta Q = -6.3\%$).

The following empirical metrics were logged during the 3-repeat offline benchmark across 160 items in `data/pilot_dataset.json`:

| System / Baseline | Quality Pass Rate | 95% Bootstrap CI | Cost (USD) | Cost vs A | Latency (ms) | Energy (Wh) | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **System_A** | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.02720 | 0.1% | 89.5 ms | 11.811 Wh | `[SIMULATED / CALCULATED]` |
| **System_Small** | 65.6% ± 0.00% | [53.8%, 75.6%] | $0.00040 | 98.5% | 29.1 ms | 0.626 Wh | `[SIMULATED / CALCULATED]` |
| **System_B** | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.02000 | 26.5% | 67.2 ms | 8.583 Wh | `[SIMULATED / CALCULATED]` |
| **System_C** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 25.8 ms | 1.862 Wh | `[SIMULATED / CALCULATED]` |
| **System_Learned** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00430 | 84.2% | 41.4 ms | 2.369 Wh | `[SIMULATED / CALCULATED]` |
| **System_Oracle** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00040 | 98.5% | 26.1 ms | 0.572 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_templates** | 66.2% ± 0.00% | [54.4%, 76.2%] | $0.00410 | 84.9% | 26.4 ms | 1.903 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_small_model** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.01840 | 32.4% | 29.3 ms | 8.011 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_fallback** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00290 | 89.3% | 29.2 ms | 1.424 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_cache** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 29.0 ms | 1.862 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_validation** | 66.2% ± 0.00% | [53.8%, 76.9%] | $0.00290 | 89.3% | 30.7 ms | 1.424 Wh | `[SIMULATED / CALCULATED]` |

#### Per-Task-Family Performance Breakdown (`dev` split)

| Task Family | Count | System A Pass | System C Pass | Quality Delta | Avg Cost (C) | Cost vs A | Primary Tier | Insight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Arithmetic | 8 | 75.0% | 75.0% | +0.0% | $0.00000 | 99.0% | `small_model_c` | AST SafeCalculator executes locally with 100% precision at $0.00 cost (<0.1ms). |
| Unit conversion | 8 | 100.0% | 87.5% | -12.5% | $0.00001 | 85.6% | `small_model_c` | Deterministic AST conversion unit executes locally at zero token expense. |
| Classification | 16 | 100.0% | 93.8% | -6.2% | $0.00000 | 94.2% | `small_model_c` | Small ~8B model reliably solves classification tasks at 95%+ cost reduction. |
| Summarization | 16 | 87.5% | 87.5% | +0.0% | $0.00000 | 98.6% | `small_model_c` | Small model handles standard summaries; occasional validation fallback to Tier D. |
| Structured extraction | 32 | 84.4% | 81.2% | -3.1% | $0.00001 | 94.9% | `small_model_c` | Template/regex constraints guide extraction with high fidelity. |
| Templates | 16 | 93.8% | 56.2% | -37.5% | $0.00002 | 92.3% | `template_b` | Jinja2 template slot filling replaces LLM inference with zero token spend. |
| Reasoning | 32 | 68.8% | 65.6% | -3.1% | $0.00010 | 64.0% | `small_model_c` | Frontier large model required for deep logic; small models exhibit measured quality decline. |
| Time-sensitive tasks | 4 | 75.0% | 75.0% | +0.0% | $0.00000 | 97.7% | `small_model_c` | Freshness policy strictly bypasses cache to prevent stale data serving. |
| Private/internal tasks | 28 | 0.0% | 0.0% | +0.0% | $0.00000 | 0.0% | `deterministic_a` | Stage 1 privacy guardrails intentionally intercept external transmission (HTTP 403). |

- **Summary Artifact**: [`data/results/simulated/benchmark_summary.json`](data/results/simulated/benchmark_summary.json)  
- **Protocol Config Hash**: `83da43205a66ba657d578bbe959cc611f9419b9cc37600e2dd8c1c2e125451f3`  

- **Inter-Rater Agreement**: A pilot dual-annotator study achieved strong agreement (Human-Human Quadratic $\kappa = 0.859$ `[MEASURED]`), while automated-judge agreement was moderate (Human-Judge $\kappa = 0.415$ `[MEASURED]`), indicating automated judge scores serve as an auxiliary metric rather than ground truth.  

- **Bootstrap Estimator**: Clustered task-family bootstrap (B=1000 resamples, 95% CI)  
- **Exact Cache Performance**: Cold Hit Rate = `16.2%`, Warm Hit Rate = `66.2%`, Inappropriate Hits = `0`

### Held-Out Evaluation Benchmark (`test` split, 3 repeats)

> [!WARNING]
> **Mock-Coverage & Distribution Diagnostics (`test` split)**: Held-out simulated evaluation exposed substantial distribution and mock-coverage failure (mock adapter coverage rate: `19.4%`, with `129` unrecognized mock completions defaulting to generic fallbacks). Consequently, the test-split simulated pass rates (System A: 2.5%, System C: 9.4%) reflect mock adapter keyword omission rather than generalized LLM routing capability. This evaluation is retained for diagnostic provenance, not as evidence of model generalization.

| System / Baseline | Quality Pass Rate | 95% Bootstrap CI | Cost (USD) | Cost vs A | Latency (ms) | Energy (Wh) | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **System_A** | 2.5% ± 0.00% | [0.0%, 6.9%] | $0.02130 | 0.2% | 96.2 ms | 10.750 Wh | `[SIMULATED / CALCULATED]` |
| **System_Small** | 2.5% ± 0.00% | [0.0%, 6.9%] | $0.00030 | 98.6% | 26.9 ms | 0.571 Wh | `[SIMULATED / CALCULATED]` |
| **System_B** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.01540 | 27.9% | 71.1 ms | 7.636 Wh | `[SIMULATED / CALCULATED]` |
| **System_C** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00300 | 85.9% | 29.1 ms | 1.769 Wh | `[SIMULATED / CALCULATED]` |
| **System_Learned** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00340 | 84.1% | 77.2 ms | 2.125 Wh | `[SIMULATED / CALCULATED]` |
| **System_Oracle** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00030 | 98.6% | 29.5 ms | 0.536 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_templates** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00300 | 85.9% | 33.0 ms | 1.779 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_small_model** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.01520 | 28.8% | 22.4 ms | 7.502 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_fallback** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00150 | 93.0% | 25.0 ms | 1.116 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_cache** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00300 | 85.9% | 24.6 ms | 1.769 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_validation** | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00150 | 93.0% | 27.4 ms | 1.116 Wh | `[SIMULATED / CALCULATED]` |

- **Test Summary Artifact**: [`data/results/simulated_test/benchmark_summary.json`](data/results/simulated_test/benchmark_summary.json)  

<!-- RESULTS_TABLE_END -->

To update the results block above from newly logged runs:
```bash
python scripts/update_readme_results.py
```

### Understanding the Quality Pass Rate & Enterprise Privacy Invariants

- **Enterprise Privacy Interception**: The 160-item `dev` benchmark contains 28 items (17.5%) classified as `internal` or `confidential`. Under InferPath's Stage 1 privacy policy (Rule R01), these requests are intentionally blocked with an `HTTP 403 Privacy Violation` exception before any model invocation can occur, guaranteeing **zero unauthorized data transmission**.
- **Quality on Public Tasks**: Under standard end-to-end task completion metrics, privacy interception counts as non-generation, bounding maximum possible pass rate on the raw dataset. On the 132 public items, **InferPath (System C) achieves a 76.5% pass rate** compared to System A's 84.1% (a measured 7.6% quality trade-off under simulated workloads), while slashing inference cost by **84.9%** and latency by **70.2%** (from 89.5 ms to 26.7 ms, equivalent to a 3.35× speedup).
- **Constraint-Aware vs. Learned Routing**: While the `RandomForest` classifier (`System_Learned`) makes fast statistical decisions, InferPath's `ConstraintAwareRouter` strictly enforces caller-defined constraints ($Q_{\min}, B_{\max}, L_{\max}$) prior to dispatch, pruning ineligible tiers and failing fast with `CONSTRAINTS_UNSATISFIABLE` (HTTP 422) without invoking backend models when no tier meets all constraints.

### Multi-Strategy Empirical Pareto Frontiers

The dual-panel plot below demonstrates the empirical trade-off space across all candidate architectures and theoretical lower bounds:

![Pareto Frontiers](docs/img/pareto_frontier.png)

### System-Level Concurrency & Latency Profile

Measured across 100 requests per concurrency pool using asynchronous ASGI client transport (`scripts/benchmark_concurrency.py`):

| Concurrency Pool | Throughput | Latency p50 | Latency p90 | Latency p95 | Latency p99 | Success Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Worker (Sequential)** | **45.8 RPS** | 15.6 ms | 49.0 ms | 60.0 ms | 71.0 ms | **100.0%** |
| **5 Workers (Concurrent)** | **46.7 RPS** | 20.6 ms | 241.7 ms | 282.8 ms | 557.7 ms | **100.0%** |
| **10 Workers (High Load)** | **45.6 RPS** | 20.1 ms | 418.1 ms | 601.8 ms | 1948.2 ms | **100.0%** |
| **20 Workers (Stress Peak)** | **46.9 RPS** | 16.4 ms | 1219.5 ms | 1503.9 ms | 1715.3 ms | **100.0%** |

---

## 5. Visual Dashboard Inspection

The research dashboard visualizes multi-system benchmarks, routing matrices, cache overhead, and live request traces:

| System Comparison & Overview | Routing Regret & Dynamics |
| :---: | :---: |
| ![Dashboard Overview](docs/img/dashboard_overview.png) | ![Routing Analysis](docs/img/dashboard_routing.png) |
| **Exact-Response Cache Performance** | **Environmental Modeling & Assumptions** |
| ![Cache Performance](docs/img/dashboard_cache.png) | ![Resources](docs/img/dashboard_resources.png) |
| **Failure Analysis & Redaction** | **Live Interactive Request Runner** |
| ![Failures](docs/img/dashboard_failures.png) | ![Live Demo](docs/img/dashboard_live_demo.png) |

---

## 6. Project Structure

```
InferPath/
├── app/
│   └── dashboard.py                  # Streamlit interactive research dashboard
├── configs/
│   ├── carbon_coefficients.yaml      # Datacenter PUE and energy/water estimation parameters
│   ├── pricing_table_2026_10.yaml    # Dated model pricing schedule (2026-10-01)
│   └── privacy_policies.yaml         # External transmission classification rules
├── data/
│   ├── logs/                         # Structured JSONL telemetry logs
│   ├── pilot_dataset.json            # 320-item balanced evaluation dataset (80 task families)
│   └── results/
│       ├── benchmarks/               # System-level ASGI concurrency and throughput metrics
│       ├── fixtures/                 # Hermetic schema fixtures for offline CI/test validation
│       ├── real/                     # Reserved exclusively for genuine measured live runs
│       ├── simulated/                # Benchmark manifests, summaries, and dev ablation logs
│       └── simulated_test/           # Held-out test split distribution diagnostic logs
├── docs/
│   ├── architecture.md               # End-to-end component lifecycle & design
│   ├── cache_design.md               # Exact cache specification & normalization
│   ├── dashboard.md                  # Streamlit dashboard architecture & usage
│   ├── evaluation.md                 # Empirical findings & 4-pillar evidence report
│   ├── methodology_and_assumptions.md# Resource estimation equations & coefficients
│   ├── real_vs_simulated_gap.md      # Live provider vs offline simulation gap audit
│   ├── routing_rules.md              # Rule table, precedence, and regex triggers
│   └── threats_to_validity.md        # Comprehensive experimental validity critique
├── scripts/
│   ├── benchmark_concurrency.py      # Real system-level ASGI throughput & latency benchmark
│   ├── capture_screenshots.py        # Automated headless CDP screenshot utility
│   ├── demo.sh                       # Self-contained safe bash demo runner
│   ├── demo.ps1                      # Windows PowerShell demo runner
│   ├── generate_pareto_frontier.py   # Publication-grade Pareto frontier generator
│   ├── run_real_evaluation.py        # Live OpenRouter evaluation runner with safety budget
│   └── update_readme_results.py      # Deterministic results table generator
├── src/inferpath/
│   ├── api/                          # FastAPI REST endpoints (/v1/respond, /v1/health)
│   ├── cache/                        # Canonical cache key builder & SQLite store
│   ├── config/                       # Typed Pydantic settings & config loaders
│   ├── dashboard/                    # Dashboard data access layer and tab renderers
│   ├── evaluation/                   # Experiment harness, checkers, judges, stats
│   ├── handlers/                     # Deterministic, Jinja2, and OpenRouter adapters
│   ├── models/                       # Domain schemas & ProvenanceValue wrappers
│   ├── router/                       # Request analyzer, rule-based & constraint routers
│   ├── security/                     # Privacy enforcer, budget tracker, hashing
│   └── validators/                   # Deterministic output validators & refusal guards
├── tests/
│   └── unit/                         # 300+ offline unit tests covering all modules
├── docker-compose.yml                # Multi-container orchestration (API + Dashboard)
├── Dockerfile                        # Container definition (Python 3.11-slim)
├── Makefile                          # Unified build, test, eval, and run targets
└── pyproject.toml                    # PEP 621 packaging and dependencies
```

---

## 7. Known Limitations & Research Boundaries

1. **Simulated vs. Real-Provider Execution**: All reported benchmark results originate from reproducible offline simulated workloads using deterministic mock adapters. The results demonstrate routing mechanics and pipeline reproducibility under controlled simulated environments; live network calls to commercial LLM providers were not evaluated in the reported tables.
2. **Development Split Calibration**: The primary reported benchmark reflects the 160-item `dev` split. The held-out `test` split is evaluated under identical frozen protocol conditions without cross-split prompt leakage.
3. **Scope 2 Compute Boundaries**: Energy and water metrics are mathematical estimations based on `configs/carbon_coefficients.yaml`. They model Scope 2 datacenter compute power and operational cooling water (WUE); they explicitly exclude Scope 3 embodied manufacturing and wide-area network transit.
4. **Counterfactual Quality Bounds**: Counterfactual Always-Large comparisons compute economic cost based on token counts and the dated pricing schedule (`2026-10-01`). Unexecuted models cannot be graded for quality or latency without executing live requests.
5. **Heuristic Output Validation**: Deterministic output validators rely on structural schemas, regex, and non-empty checks. Semantic hallucination detection on open-ended creative tasks requires human or LLM judge grading.

