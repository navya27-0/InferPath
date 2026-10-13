# InferPath: Resource-Aware Multi-Tier AI Inference Router

Routing user requests to the least resource-intensive processing tier that meets correctness, quality, privacy, and latency requirements.

## 1. Problem Statement & Scope

Modern AI application architectures frequently dispatch all user queries to frontier large language models (LLMs) regardless of query complexity, resulting in unnecessary monetary expense, latency overhead, and datacenter electricity consumption. InferPath implements and experimentally evaluates a dynamic, multi-tier inference router that cascades requests across four discrete processing tiers: (A) local deterministic rules and calculations, (B) structured templates with clarification guards, (C) small models (~8B parameters), and (D) large frontier models (~70B+ parameters). Requests are preceded by an exact-response cache and followed by deterministic output validation with bounded single-level fallback. This repository implements the complete routing engine, exact cache, OpenRouter integration adapters, an offline evaluation harness, and an interactive Streamlit research dashboard.

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
| **System_A** | 69.4% ± 0.00% | [69.4%, 69.4%] | $0.02720 | 0.1% | 89.5 ms | 36.016 Wh | `[SIMULATED / CALCULATED]` |
| **System_B** | 69.4% ± 0.00% | [69.4%, 69.4%] | $0.01800 | 33.9% | 60.5 ms | 23.255 Wh | `[SIMULATED / CALCULATED]` |
| **System_C** | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00410 | 84.9% | 35.2 ms | 2.270 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_templates** | 66.2% ± 0.00% | [66.2%, 66.2%] | $0.00410 | 84.9% | 32.1 ms | 2.182 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_small_model** | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.02220 | 18.4% | 32.6 ms | 9.807 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_fallback** | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00290 | 89.3% | 35.3 ms | 1.509 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_cache** | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00410 | 84.9% | 35.4 ms | 2.300 Wh | `[SIMULATED / CALCULATED]` |
| **ablation_no_validation** | 66.2% ± 0.00% | [66.2%, 66.2%] | $0.00290 | 89.3% | 36.5 ms | 1.576 Wh | `[SIMULATED / CALCULATED]` |

- **Summary Artifact**: [`data/results/simulated/benchmark_summary.json`](data/results/simulated/benchmark_summary.json)  
- **Protocol Config Hash**: `83da43205a66ba657d578bbe959cc611f9419b9cc37600e2dd8c1c2e125451f3`  
- **Inter-Rater Agreement**: Human-Human Quadratic $\kappa = 0.8882$, Human-Judge $\kappa = 0.9523$  
- **Exact Cache Performance**: Cold Hit Rate = `21.9%`, Warm Hit Rate = `59.4%`, Inappropriate Hits = `0`
<!-- RESULTS_TABLE_END -->

To update the results block above from newly logged runs:
```bash
python scripts/update_readme_results.py
```

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
│   ├── routing_rules.md              # Rule table, precedence, and regex triggers
│   └── threats_to_validity.md        # Comprehensive experimental validity critique
├── scripts/
│   ├── capture_screenshots.py        # Automated headless CDP screenshot utility
│   ├── demo.sh                       # Self-contained safe bash demo runner
│   ├── demo.ps1                      # Windows PowerShell demo runner
│   └── update_readme_results.py      # Deterministic results table generator
├── src/inferpath/
│   ├── api/                          # FastAPI REST endpoints (/v1/respond, /v1/health)
│   ├── cache/                        # Canonical cache key builder & SQLite store
│   ├── config/                       # Typed Pydantic settings & config loaders
│   ├── dashboard/                    # Dashboard data access layer and tab renderers
│   ├── evaluation/                   # Experiment harness, checkers, judges, stats
│   ├── handlers/                     # Deterministic, Jinja2, and OpenRouter adapters
│   ├── models/                       # Domain schemas & ProvenanceValue wrappers
│   ├── router/                       # Request analyzer and rule-based router
│   ├── security/                     # Privacy enforcer, budget tracker, hashing
│   └── validators/                   # Deterministic output validators & refusal guards
├── tests/
│   └── unit/                         # 198 offline unit tests covering all modules
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
