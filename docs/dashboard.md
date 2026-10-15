# InferPath — Research Dashboard and Live Demo Specification

**Document Version**: 1.0.0  
**Effective Date**: 2026-10-13  
**Status**: Implemented & Verified  
**Entrypoint**: `app/dashboard.py`  
**Launch Command**: `make run-dashboard` (or `streamlit run app/dashboard.py`)

---

## 1. Overview & Architectural Principles

The InferPath Research Dashboard provides a clean, research-grade Streamlit application designed for empirical evaluation of multi-tier resource-aware inference routing:
- **Tier A**: Deterministic Rules & Unit Conversions (Zero LLM cost)
- **Tier B**: Structured Jinja2 Templates with Slot Clarification
- **Tier C**: Small Model (~8B parameter class)
- **Tier D**: Large Frontier Model (~70B+ parameter class)

Preceded by a SHA-256 canonical exact-response cache and followed by deterministic output validation with bounded single-level fallback.

### Empirical Non-Negotiable Standards
1. **Zero Metric Fabrication**: Every KPI, chart series, and table entry is parsed directly from recorded execution artifacts (`data/results/` and `data/logs/`). Missing metrics are rendered as `N/A` or explicit unavailable states—never synthetic zeroes or fabricated values.
2. **Strict Provenance Labeling**: Every metric is explicitly tagged with exactly one of four provenance categories:
   - `[MEASURED]`: Physical observations (token counts from API, observed wall-clock latency).
   - `[CALCULATED]`: Deterministic formula derivations (token cost calculated using the dated pricing schedule: `2026-10-01`).
   - `[ESTIMATED]`: Modeled parameters with low, central, and high ranges (Scope 2 datacenter compute energy and operational cooling water).
   - `[SIMULATED]`: Synthetic numbers produced during offline mock model execution.
3. **Read-Only Data Access**: The dashboard operates over an immutable data access layer (`inferpath.dashboard.data_loader`), preventing any mutation or overwrite of research artifacts.

---

## 2. Dashboard Structure & Tab Guide

```
app/
└── dashboard.py                  # Main Streamlit application entrypoint
src/inferpath/dashboard/
├── __init__.py                   # Package exports
├── data_loader.py                # Read-only artifact discovery & validation layer
├── components.py                 # Accessible design components & provenance badges
└── tabs/
    ├── overview.py               # Tab 1: Systems A, B, C comparison & scatter plots
    ├── routing.py                # Tab 2: Routing distribution, regret & oracle matrix
    ├── cache.py                  # Tab 3: Cold/warm hit rates, overhead & bypass audit
    ├── resources.py              # Tab 4: Energy/water bounds & carbon assumptions
    ├── failures.py               # Tab 5: Worst 20 failure triage with privacy redaction
    └── live_demo.py              # Tab 6: Live FastAPI runner & counterfactual projection
```

### Tab 1: Overview
- **Side-by-Side Baselines**: Direct comparison of **System A** (Always-Large), **System B** (Cache + Rules + Large), and **System C** (Full InferPath).
- **Core KPIs**: Quality Pass Rate ($Q \ge Q_{\min}$), Total USD Cost, Mean Latency, P95 Latency, and Estimated Energy/Water.
- **Clustered Bootstrap Confidence Intervals**: Resampling over 80 distinct `task_family_id` clusters to preserve semantic correlation structure.
- **Quality vs. Cost Scatter**: Individual item evaluation records colored by difficulty tier.
- **Difficulty Tier Breakdown**: Latency, cost, and quality aggregated across Easy, Moderate, and Difficult tiers.

### Tab 2: Routing & Oracle Calibration
- **Routing Regret**: Cost difference against the oracle optimal route ($Cost_{actual} - Cost_{oracle}$ for items meeting $Q_{\min}$).
- **Over-Escalation & Under-Escalation**: Percentage of queries routed to higher-than-needed tiers vs. under-routed causing quality failure.
- **Confusion Matrix**: Selected route tier versus the oracle optimal tier.
- **Category Breakdown**: Filterable table showing performance across 8 distinct task categories.

### Tab 3: Exact-Response Cache
- **Isolated Workloads**: Cold cache (16.2% overall hit rate, 20.3% eligible from intra-batch repeats) vs. warm cache (66.2% overall hit rate, 82.8% eligible).
- **Denominator Transparency**: Strict separation of overall hit rate ($Hits / Total$) vs. eligible hit rate ($Hits / Eligible$).
- **Storage Overhead**: Physical SQLite file size in bytes and entries count on disk (`data/cache.db`).
- **Bypass Breakdown**: Deterministic local bypass, time-sensitive freshness bypass, and privacy policy interception.
- **Inappropriate-Hit Audit**: Audited verification of zero cross-tenant, private, or stale cache hits.

### Tab 4: Resources & Environmental Assumptions
- **Mandatory Disclaimer Banner**: Prominently warns that energy and water metrics are mathematical estimates, not measured provider telemetry.
- **Assumption Ledger**: Full parameter table loaded directly from `configs/carbon_coefficients.yaml` displaying Low, Central, and High bounds with source citations and verification statuses.
- **System Boundaries**: Explicit inclusion of Scope 2 compute energy and operational cooling water (WUE); transparent exclusion of Scope 3 hardware manufacturing and network transit transit.

### Tab 5: Failure Analysis & Triage
- **Ranked Edge Cases**: Top 20 worst cases ranked by quality score ascending ($Q < Q_{\min}$ or validation rejection).
- **Privacy Redaction**: Confidential or internal request prompts are automatically hashed and redacted (`[REDACTED: SENSITIVE/PRIVATE PROMPT]`) to prevent credential or secret leakage in UI logs.
- **Deep-Dive Cards**: Individual failure inspections detailing triggered rule IDs, oracle targets, and fallback histories.

### Tab 6: Live Interactive Demo & Counterfactual Projection
- **Direct FastAPI Dispatch**: Connects via HTTP to `POST /v1/respond` on the active InferPath backend (default: `http://localhost:8000`).
- **Full Trace Telemetry**: Displays the complete step-by-step decision path, cache status, measured latency boundary, token usage, and calculated cost.
- **Always-Large Counterfactual**: Projects the counterfactual cost of routing the identical prompt to `anthropic/claude-3.7-sonnet` using the dated price schedule (`2026-10-01`).
- **Explicit Counterfactual Boundaries**: Labels unexecuted large-model quality and latency as `Unavailable (Unexecuted Counterfactual)` to prevent unsubstantiated claims.
- **Offline / Error Resilience**: Displays clear guidance when the FastAPI server is offline, requests time out, or confidential requests are blocked by privacy policies.

---

## 3. Visual Artifacts

Genuine screenshots captured from the running application:
- **Overview**: [`docs/img/dashboard_overview.png`](file:///C:/InferPath/docs/img/dashboard_overview.png)
- **Routing**: [`docs/img/dashboard_routing.png`](file:///C:/InferPath/docs/img/dashboard_routing.png)
- **Cache Engine**: [`docs/img/dashboard_cache.png`](file:///C:/InferPath/docs/img/dashboard_cache.png)
- **Resources**: [`docs/img/dashboard_resources.png`](file:///C:/InferPath/docs/img/dashboard_resources.png)
- **Failure Analysis**: [`docs/img/dashboard_failures.png`](file:///C:/InferPath/docs/img/dashboard_failures.png)
- **Live Demo**: [`docs/img/dashboard_live_demo.png`](file:///C:/InferPath/docs/img/dashboard_live_demo.png)
- **Live Demo Result**: [`docs/img/dashboard_live_demo_result.png`](file:///C:/InferPath/docs/img/dashboard_live_demo_result.png)

---

## 4. Known Limitations & Research Boundaries

1. **Simulated vs. Real Provider Runs**: Currently, all benchmark runs in `data/results/simulated/` were conducted using deterministic mock model adapters. No live OpenRouter runs have been logged on the full test set. The dashboard explicitly indicates this state via the `[SIMULATED]` provenance badge and split markers.
2. **Provider Telemetry Gap**: Commercial model APIs do not expose real-time energy or cooling water consumption. All environmental figures reflect static Scope 2 models from `configs/carbon_coefficients.yaml`.
3. **Counterfactual Quality**: Counterfactual comparisons estimate cost only. An unexecuted model cannot be graded for quality or latency without running live inference.
