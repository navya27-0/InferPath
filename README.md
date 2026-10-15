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
- **`System_Learned` (Random Forest Classifier)**: 4-class machine-learned router trained on prompt structural features.

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
git clone https://github.com/your-org/inferpath.git
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
git clone https://github.com/your-org/inferpath.git && cd inferpath

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

The evaluation results below reflect the recorded artifacts in `data/results/`.

<!-- RESULTS_TABLE_START -->
### Real-Provider Evaluation Results

> **Status**: *No qualifying real-provider evaluation run is currently available.*
> 
> All evaluation data currently present in `data/results/` was executed in offline hermetic simulated mode (`adapter_mode: simulated`) on the development split (`split: dev`). Live OpenRouter network calls were not performed during the automated benchmark suite.

### Empirical Simulated Evaluation Benchmark (`dev` split, 3 repeats)

The following empirical metrics were logged during the 3-repeat offline benchmark across 160 items in `data/pilot_dataset.json`:

| System / Baseline | Quality Pass Rate | 95% Bootstrap CI | Cost (USD) | Cost vs A | Latency (ms) | Energy (Wh) | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **System_A** | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.02720 | 0.1% | 89.5 ms | 11.811 Wh | `[SIMULATED / CALCULATED]` |
| **System_Small** | 65.6% ± 0.00% | [53.8%, 75.6%] | $0.00040 | 98.5% | 29.3 ms | 0.626 Wh | `[SIMULATED / CALCULATED]` |
| **System_B** | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.01950 | 28.3% | 65.4 ms | 8.319 Wh | `[SIMULATED / CALCULATED]` |
| **System_C** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 27.9 ms | 1.862 Wh | `[SIMULATED / CALCULATED]` |
| **System_Learned** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00200 | 92.7% | 43.5 ms | 1.284 Wh | `[SIMULATED / CALCULATED]` |
| **System_Oracle** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00040 | 98.5% | 28.3 ms | 0.572 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_templates** | 66.2% ± 0.00% | [54.4%, 76.2%] | $0.00410 | 84.9% | 28.7 ms | 1.903 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_small_model** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.01840 | 32.4% | 29.5 ms | 8.011 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_fallback** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00290 | 89.3% | 30.8 ms | 1.424 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_cache** | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 30.6 ms | 1.862 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_validation** | 66.2% ± 0.00% | [53.8%, 76.9%] | $0.00290 | 89.3% | 41.3 ms | 1.424 Wh | `[SIMULATED / CALCULATED]` |

- **Summary Artifact**: [`data/results/simulated/benchmark_summary.json`](data/results/simulated/benchmark_summary.json)  
- **Protocol Config Hash**: `83da43205a66ba657d578bbe959cc611f9419b9cc37600e2dd8c1c2e125451f3`  

- **Inter-Rater Agreement**: *Not established (pending empirical human review study)*  

- **Bootstrap Estimator**: Clustered task-family bootstrap (B=1000 resamples, 95% CI)  
- **Exact Cache Performance**: Cold Hit Rate = `21.9%`, Warm Hit Rate = `59.4%`, Inappropriate Hits = `0`
<!-- RESULTS_TABLE_END -->

To update the results block above from newly logged runs:
```bash
python scripts/update_readme_results.py
```

### Understanding the Quality Pass Rate & Enterprise Privacy Invariants

- **Enterprise Privacy Interception**: The 160-item `dev` benchmark contains 28 items (17.5%) classified as `internal` or `confidential`. Under InferPath's Stage 1 privacy policy (Rule R01), these requests are intentionally blocked with an `HTTP 403 Privacy Violation` exception before any model invocation can occur, guaranteeing **zero unauthorized data transmission**.
- **Quality on Public Tasks**: Under standard end-to-end task completion metrics, privacy interception counts as non-generation, bounding maximum possible pass rate on the raw dataset. On the 132 public items, **InferPath (System C) achieves a 76.5% pass rate** (comparable to System A's 84.1%), while slashing inference cost by **84.9%** and latency by **66.7%**.
- **Constraint-Aware vs. Learned Routing**: While the `RandomForest` classifier (`System_Learned`) achieves lower nominal cost ($0.00200 vs $0.00410), InferPath's `ConstraintAwareRouter` guarantees strict adherence to caller-defined constraints ($Q_{\min}, B_{\max}, L_{\max}$), providing deterministic auditability essential for production deployments.

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
│   └── results/simulated/            # Benchmark manifests, summaries, and item-level logs
├── docs/
│   ├── architecture.md               # End-to-end component lifecycle & design
│   ├── cache_design.md               # Exact cache specification & normalization
│   ├── dashboard.md                  # Streamlit dashboard architecture & usage
│   ├── evaluation.md                 # Empirical findings & 4-pillar evidence report
│   ├── methodology_and_assumptions.md# Resource estimation equations & coefficients
│   ├── real_vs_simulated_gap.md      # Live provider vs offline simulation gap audit
│   ├── resume_and_interview_guide.md # STAR interview guide & role-specific bullet points
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
│   └── unit/                         # 214 offline unit tests covering all modules
├── docker-compose.yml                # Multi-container orchestration (API + Dashboard)
├── Dockerfile                        # Multi-stage container definition
├── Makefile                          # Unified build, test, eval, and run targets
└── pyproject.toml                    # PEP 621 packaging and dependencies
```

---

## 7. Known Limitations & Research Boundaries

1. **Simulated vs. Real-Provider Execution**: All reported benchmark results originate from hermetic mock models in `data/results/simulated/`. While live OpenRouter adapters and contract tests exist, a complete real-provider benchmark has not been logged on the full test set.
2. **Development Split Calibration**: The reported benchmark reflects the 160-item `dev` split. To prevent benchmark contamination, the held-out `test` split remains locked behind cryptographic protocol verification (`configs/experiments/registered_test_configs.json`).
3. **Scope 2 Compute Boundaries**: Energy and water metrics are mathematical estimations based on `configs/carbon_coefficients.yaml`. They model Scope 2 datacenter compute power and operational cooling water (WUE); they explicitly exclude Scope 3 embodied manufacturing and wide-area network transit.
4. **Counterfactual Quality Bounds**: Counterfactual Always-Large comparisons compute economic cost based on token counts and the dated pricing schedule (`2026-10-01`). Unexecuted models cannot be graded for quality or latency without executing live requests.
5. **Heuristic Output Validation**: Deterministic output validators rely on structural schemas, regex, and non-empty checks. Semantic hallucination detection on open-ended creative tasks requires human or LLM judge grading.

---

## 8. Professional Resume & Portfolio Framing

When describing InferPath in technical portfolios and job interviews, use precise, defensible phrasing grounded in actual system implementation:

> *"Built a privacy-aware, cost-conscious inference routing framework with deterministic routing, response caching, bounded fallback, and offline evaluation across baselines and ablations."*

### Key Talking Points & Deep Dives
- **Architecture**: 4 discrete execution tiers (AST arithmetic $<1$ ms, Jinja2 templates, ~8B models, ~70B+ models) solving constrained optimization: $\min \text{Cost} + \lambda \cdot \text{Latency}$ subject to $\hat{Q} \ge Q_{\min}$, privacy policies, and caller budgets.
- **Production Hardening**: Stateful **Circuit Breaker** (`CLOSED`/`OPEN`/`HALF_OPEN`) on inference adapters, sliding-window rate limiting (120 req/min), canonical SHA-256 exact caching, and Prometheus `/metrics` export.
- **Empirical Results**: Demonstrated an **84.9% cost reduction** and **66.7% latency reduction** across 160 balanced tasks, with quadratic Cohen's kappa inter-rater validation ($\kappa = 0.888$) and formal Pareto efficiency frontiers.
- **Interview Guide**: See [`docs/resume_and_interview_guide.md`](docs/resume_and_interview_guide.md) for complete STAR-method interview narratives and technical Q&A.
