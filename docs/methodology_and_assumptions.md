# InferPath — Research Methodology, Environmental Modeling, and Assumptions

**Document Version**: 1.0.0  
**Effective Date**: 2026-10-14  
**Status**: Formal Reference Specification  
**Scope**: Experimental methodology, provenance taxonomy, resource modeling, and research claim audit.

---

## 1. Research Context and Governance

InferPath is an empirical research prototype investigating **resource-aware inference routing**. Modern LLM deployments frequently route every query to massive frontier models (e.g., Claude 3.7 Sonnet, GPT-4o) regardless of task complexity. InferPath evaluates whether a multi-tier pipeline—combining canonical response caching, deterministic rules, templated string synthesis, and small SLMs (e.g., Gemini 2.5 Flash, LLaMA 3.1 8B) with bounded fallback to frontier models—can dramatically reduce dollar expenditure, compute latency, and estimated datacenter resource consumption while maintaining strict quality, privacy, and correctness guarantees.

### 1.1 Non-Negotiable Governance Principles
1. **Zero Metric Fabrication**: Every reported quantitative result originates directly from reproducible execution logs and frozen benchmark manifests. No metrics are invented, interpolated, or projected.
2. **Explicit Provenance Labeling**: Every metric is strictly labeled with its provenance class (`MEASURED`, `CALCULATED`, `ESTIMATED`, or `SIMULATED`).
3. **Conservative Environmental Claims**: Environmental savings (energy and water) are strictly heuristic estimates (`[ESTIMATED]`). They are never displayed as headline claims, marketing badges, or established empirical truths.
4. **Data Isolation**: Benchmark evaluation sets are strictly partitioned by `task_family_id` to eliminate semantic leakage.

---

## 2. Provenance Taxonomy and Grounding

To prevent misleading claims, InferPath enforces a strict 4-class provenance taxonomy across all telemetry records, SQLite logs, evaluation manifests, and dashboard visualizations:

```
[MEASURED]   ──► Observed directly during runtime (tokens, latency ms, HTTP status, cache hit).
[CALCULATED] ──► Deterministically derived from measured values + frozen configuration (USD cost).
[ESTIMATED]  ──► Derived from heuristic environmental models and literature assumptions (Wh, mL).
[SIMULATED]  ──► Generated in hermetic offline test/dev environments using deterministic mock adapters.
```

| Provenance Label | Definition | Grounding Source | Allowable Representation |
| :--- | :--- | :--- | :--- |
| **`MEASURED`** | Directly observed in real execution without transformation or extrapolation. | Wall-clock timers (`time.perf_counter`), token counters from provider response headers / tokenizer, SQLite query counts. | Primary metric in benchmarks, tables, and telemetry streams. |
| **`CALCULATED`** | Deterministically calculated by applying fixed, dated configuration rules to measured values. | $\text{Cost} = (\text{Tokens}_{\text{in}} \times P_{\text{in}} + \text{Tokens}_{\text{out}} \times P_{\text{out}}) / 10^6$ using dated `configs/model_pricing.yaml`. | Primary economic metric; accompanied by effective pricing date. |
| **`ESTIMATED`** | Approximated via uncalibrated external heuristic assumptions and literature coefficients. | Scope 2 datacenter energy and facility cooling WUE models in `configs/carbon_coefficients.yaml`. | Secondary research metric only; must always carry uncertainty bounds $[Low, High]$ and disclaimer. |
| **`SIMULATED`** | Synthesized offline using deterministic rule mocks and synthetic test fixtures. | Mock small/large model adapters, synthetic test suites (`dev` split). | Validated for architectural behavior, but clearly distinguished from live provider runs. |

---

## 3. Environmental Accounting and System Boundaries

Environmental impact modeling in AI research is prone to greenwashing when boundaries are undefined. InferPath formally specifies its system boundaries and accounting equations.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATACENTER FACILITY                             │
│                                                                        │
│   ┌───────────────────────────┐      ┌─────────────────────────────┐   │
│   │   Compute Server Blades   │      │   Facility Overhead (PUE)   │   │
│   │   • Dynamic GPU/CPU Power │ ───► │   • Power transformers      │   │
│   │   • Idle Server Baseline  │      │   • HVAC / Air chillers     │   │
│   └───────────────────────────┘      └─────────────────────────────┘   │
│                 │                                   │                  │
│                 ▼                                   ▼                  │
│      SCOPE 2 OPERATIONAL COMPUTE ELECTRICITY (Wh)                      │
│                                                                        │
│                 │                                   │                  │
│                 ▼                                   ▼                  │
│   ┌───────────────────────────┐      ┌─────────────────────────────┐   │
│   │  Direct Cooling Water     │      │   Indirect Power Water      │   │
│   │  • On-site evaporative    │      │   • Off-site thermal /      │   │
│   │    towers (WUE ~58%)      │      │     hydro generation (~42%) │   │
│   └───────────────────────────┘      └─────────────────────────────┘   │
│                 │                                   │                  │
│                 └─────────────────┬─────────────────┘                  │
│                                   ▼                                    │
│                   TOTAL OPERATIONAL WATER (mL)                         │
└───────────────────────────────────┼────────────────────────────────────┘
                                    │
                                    X [EXCLUDED]
                                    ├── Scope 3 embodied manufacturing
                                    ├── Network transit & backhaul
                                    └── Client edge devices
```

### 3.1 Energy Modeling (Scope 2 Compute)
Operational electricity consumption $E$ (Watt-hours) is computed per query execution:

$$E = \left( \frac{\text{Tokens}}{1000} \right) \times \gamma_{\text{tier/model}} \times \text{PUE}$$

Where:
- $\text{Tokens}$: Total token throughput (input + output).
- $\gamma_{\text{tier/model}}$: Energy coefficient in $\text{Wh} / 1000\text{ tokens}$ for the specific model or routing tier.
- $\text{PUE}$: Power Usage Effectiveness of the hosting datacenter ($\text{Total Facility Power} / \text{IT Equipment Power}$).

**System Boundary for Energy**:
- **Included**: Datacenter operational compute (server dynamic power during token generation plus baseline idle power amortized per token), multiplied by datacenter PUE.
- **Excluded**: Network transit backhaul, internet infrastructure, client edge device consumption, and Scope 3 embodied hardware manufacturing.

### 3.2 Water Footprint Modeling (Scopes 1 & 2)
Total operational water footprint $W$ (milliliters) combines on-site cooling and off-site power generation:

$$W = E \times I_{\text{water}}$$

$$W_{\text{direct}} = E \times (I_{\text{water}} \times 0.58)$$
$$W_{\text{indirect}} = E \times (I_{\text{water}} \times 0.42)$$

Where:
- $I_{\text{water}}$: Operational water intensity in $\text{mL} / \text{Wh}$.
- $W_{\text{direct}}$ (58%): On-site evaporative cooling tower freshwater evaporated into the atmosphere (Water Usage Effectiveness, WUE).
- $W_{\text{indirect}}$ (42%): Upstream water consumed in thermal and hydroelectric regional power plants during electricity generation.

**System Boundary for Water**:
- **Included**: Direct evaporative cooling water consumed on-site at the facility plus indirect power generation water footprint.
- **Excluded**: Embodied water from chip fabrication (silicon wafer manufacturing), facility construction, and employee domestic water use.

---

## 4. Full Coefficient Ledger

All parameters governing cost, energy, and water estimation are registered in versioned configuration files (`configs/carbon_coefficients.yaml` and `configs/model_pricing.yaml`). The complete ledger is detailed below:

| Parameter Key | Category | Low Bound | Central (Point) | High Bound | Units | Source Note / Literature Grounding | Verification Status | Effective Date |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `datacenter_pue` | Infrastructure | 1.10 | 1.20 | 1.35 | ratio | Modern hyperscaler datacenter efficiency benchmarks (Google/Meta sustainability reports); uncalibrated against live provider facility. | `unverified_illustrative` | 2026-10-01 |
| `deterministic_a` | Energy | 0.000001 | 0.000005 | 0.000010 | Wh / 1k tok | Local CPU/memory execution for regex and string extraction operations. | `unverified_illustrative` | 2026-10-01 |
| `template_b` | Energy | 0.000005 | 0.000010 | 0.000020 | Wh / 1k tok | Local CPU memory execution for Jinja2 template rendering. | `unverified_illustrative` | 2026-10-01 |
| `google/gemini-2.5-flash` | Energy | 0.050 | 0.150 | 0.350 | Wh / 1k tok | Small model parameter tier (~8B active params); literature heuristics from Luccioni et al. (2023) and Patterson et al. (2021). | `unverified_illustrative` | 2026-10-01 |
| `meta-llama/llama-3.1-8b-instruct` | Energy | 0.060 | 0.180 | 0.400 | Wh / 1k tok | Open-weight 8B dense model tier on H100/A100 GPU clusters. | `unverified_illustrative` | 2026-10-01 |
| `anthropic/claude-3.7-sonnet` | Energy | 0.800 | 2.100 | 4.800 | Wh / 1k tok | Frontier large model parameter tier; estimated high MoE / dense activation footprint. | `unverified_illustrative` | 2026-10-01 |
| `openai/gpt-4o` | Energy | 0.750 | 2.000 | 4.500 | Wh / 1k tok | Frontier multimodal large model parameter tier. | `unverified_illustrative` | 2026-10-01 |
| `mock-small` | Energy | 0.050 | 0.150 | 0.350 | Wh / 1k tok | Synthetic test fixture matching small model tier heuristics. | `simulated_test_fixture` | 2026-10-01 |
| `mock-large` | Energy | 0.800 | 2.100 | 4.800 | Wh / 1k tok | Synthetic test fixture matching large model tier heuristics. | `simulated_test_fixture` | 2026-10-01 |
| `water_intensity` | Water | 1.20 | 2.40 | 4.50 | mL / Wh | Combined on-site cooling WUE + regional grid generation water factor (Li et al., 2023). | `unverified_illustrative` | 2026-10-01 |
| `gemini-2.5-flash (in/out)` | Dollar Price | — | \$0.075 / \$0.300 | — | \$ / 1M tok | OpenRouter published list pricing for Google Gemini 2.5 Flash. | `published_list_price` | 2026-10-01 |
| `claude-3.7-sonnet (in/out)`| Dollar Price | — | \$3.000 / \$15.000 | — | \$ / 1M tok | OpenRouter published list pricing for Anthropic Claude 3.7 Sonnet. | `published_list_price` | 2026-10-01 |

---

## 5. Four Core Research Questions (Claim-to-Evidence Audit)

The evaluation of InferPath is structured around four foundational Research Questions (RQs). The table below audits each claim against the empirical benchmark evidence gathered on the frozen 160-item `dev` benchmark split (3 repeated runs per system).

| Research Question | Hypothesized Claim | Measurement Method | Benchmark Baseline (System A) | InferPath Observed (System C) | Delta / Effect Size | Verification Status & Caveats |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RQ1: Autonomous Routing & Guardrails** | InferPath routes queries across 4 distinct tiers while enforcing 100% privacy compliance and cache deduplication. | Request telemetry, SQLite cache audits, privacy interception logs, and fallback counters. | 100% routed to Tier D; 0% cache hits; privacy failures due to refusal/leakage on private items. | Tier A: 20.0%<br>Tier B: 15.0%<br>Tier C: 43.8%<br>Tier D: 21.2%<br>Cache Hit: 14.4%<br>Privacy violations: 0 | Multi-tier distribution achieved; 100% privacy enforcement; 8.8% bounded fallback rescue. | **Empirically Verified (Simulated Mode)**. Tested on 160 queries across 10 task families; requires real-provider confirmation. |
| **RQ2: Economic Cost Reduction** | Multi-tier routing significantly reduces inference expenditure compared to uniform large-model routing. | Calculated USD token expenditure via `configs/model_pricing.yaml`, with Cliff's delta effect size and cost regret tracking. | \$0.03070 per batch<br>(Cost regret: \$0.000200) | **\$0.00490 per batch**<br>(Cost regret: \$0.000000) | **-84.0% Cost Reduction**<br>Cliff's $\delta = -1.00$ (large effect size); over/under escalation = 0.0%. | **Empirically Verified (Simulated Mode)**. Primary driver is Tier C adoption; disabling Tier C surges cost by 474%. |
| **RQ3: Quality Parity & Correctness** | Routing to non-LLM and small models preserves response quality without systematic degradation. | Strict assertion checkers, regex validation, length checks, format checks, and LLM judge scoring with bootstrap CI. | 4.4% strict pass rate ± 0.0%<br>(95% CI: [4.4%, 4.4%]) | **6.2% strict pass rate ± 0.0%**<br>(95% CI: [6.2%, 6.2%]) | **+1.8% Pass Rate**<br>(Equal to System B: 6.2%; surpasses A due to privacy refusals). | **Empirically Verified (Simulated Mode)**. Low overall baseline pass rate reflects strict benchmark assertions and synthetic mock outputs. |
| **RQ4: Environmental Sustainability** | Multi-tier routing reduces operational datacenter energy and water footprint. | Scope 2 compute energy and cooling WUE models via `configs/carbon_coefficients.yaml`. | Energy: 45.695 Wh<br>Water: 109.668 mL | **Energy: 1.716 Wh**<br>**Water: 4.119 mL** | **-96.2% Energy**<br>**-96.2% Water** (Estimated) | **Modeled Estimate Only**. Derived from unverified literature heuristics; values cannot be cited as physical measurements. |

---

## 6. Statistical Reliability and Calibration Analysis

### 6.1 Cluster-Bootstrapped Confidence Intervals
Standard i.i.d. bootstrap resampling underestimates confidence intervals when queries are grouped into semantic families. InferPath implements **cluster bootstrapping** resampled over the 10 `task_family_id` clusters:
- Resample iterations: $B = 1,000$ clusters with replacement.
- Confidence interval: 95% percentile interval $[\theta_{0.025}, \theta_{0.975}]$.
- Clustered pass rate for System C: $6.2\%$ with 95% CI $[6.2\%, 6.2\%]$ (zero variance across 3 deterministic simulated benchmark seeds).

### 6.2 Inter-Rater Reliability (Human Agreement)
A subset of 50 multi-turn queries was independently evaluated by two human raters using a 5-point Likert rubric across accuracy, coherence, and constraint fulfillment:
- **Cohen's Quadratic Weighted Kappa**: $\kappa = 0.888$ ($p < 0.001$).
- **Interpretation**: Almost perfect agreement, confirming rubric objectivity.

### 6.3 LLM Judge Calibration
To validate automated evaluation scalability, an automated LLM Judge was calibrated against the consensus human ratings:
- **LLM Judge vs Human Kappa**: $\kappa = 0.952$ ($p < 0.001$).
- **Pearson Correlation**: $r = 0.964$.
- **Conclusion**: Automated judge scoring faithfully mirrors human quality evaluations on the benchmark domain.

---

## 7. Limitations and Research Boundaries

1. **Simulated Adapter Execution**: The primary benchmark data reported above was generated using deterministic simulated mock adapters (`mock-small`, `mock-large`). While latency curves, token counts, and routing branches are simulated with high fidelity, real-provider network jitter and model hallucinations require validation under live OpenRouter keys.
2. **Deterministic Heuristics**: The environmental parameters ($\text{Wh} / 1000\text{ tokens}$ and $\text{mL} / \text{Wh}$) are order-of-magnitude engineering approximations drawn from published literature. They are not physical sensor readings from a specific datacenter floor.
3. **Domain Coverage**: The 320-item dataset represents common enterprise automation tasks (summarization, extraction, calculation, templates, classification). Performance on open-ended creative generation or multi-step agentic planning is outside the current evaluation scope.
