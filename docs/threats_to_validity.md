# InferPath — Threats to Validity

**Document Version**: 1.0.0  
**Effective Date**: 2026-10-14  
**Status**: Formal Methodological Review  
**Scope**: Scientific limitations, empirical boundaries, and threats to internal, external, construct, and statistical conclusion validity.

---

## 1. Overview and Scientific Stance

Empirical evaluation of AI inference routing systems is subject to numerous threats to validity. Rather than minimizing these limitations, InferPath explicitly documents the methodological risks, boundary conditions, and evaluation constraints governing its results.

This document identifies threats across four standard scientific dimensions:
1. **Internal Validity**: Flaws or confounding factors in the experimental design and execution environment.
2. **External Validity**: Limits on the generalizability of the findings to broader production workloads.
3. **Construct Validity**: Mismatches between theoretical constructs (e.g., "quality", "sustainability") and the metrics used to quantify them.
4. **Statistical Conclusion Validity**: Risks of erroneous inferences arising from sample size, distribution clustering, or estimator variance.

---

## 2. Internal Validity

Internal validity addresses whether the observed effects (cost reduction, latency improvement, quality preservation) were genuinely caused by InferPath's multi-tier routing architecture rather than experimental artifacts.

### 2.1 Simulation Fidelity vs. Real-Provider Behavior
- **Risk**: The primary benchmark results reported in `docs/evaluation.md` and `README.md` were collected using deterministic simulated mock adapters (`mock-small`, `mock-large`).
- **Confounding Factors**:
  - *Network Jitter and Latency Tails*: Mock adapters return responses within parameterized artificial delays (15–35 ms for small, 70–160 ms for large). Real-provider API calls through OpenRouter incur variable internet transit latency, queueing times, and long-tail latency spikes (P99 > 1,000 ms).
  - *Provider Errors and Rate Limits*: Real APIs intermittently throw HTTP 429 (Rate Limit Exceeded), HTTP 500 (Internal Server Error), and socket read timeouts. While InferPath implements exponential backoff and bounded fallback, these dynamic failure cascades are not fully exercised during simulated mock runs.
  - *Response Length Variability*: Synthetic mocks produce outputs of predictable token lengths. Live models generate variable token distributions that can perturb calculated costs.
- **Mitigation**: The codebase includes an OpenRouter live adapter (`OpenRouterModelAdapter`) and a dedicated live benchmark flag (`--mode real`), requiring explicit credentials and live execution before production claims are finalized.

### 2.2 Cache Hit Sensitivity and Workload Ordering
- **Risk**: Cache performance is highly sensitive to the order and repetition structure of the input query stream.
- **Confounding Factors**:
  - The evaluated 160-item `dev` split contains structured query repeats within specific task families, yielding a 16.2% cold hit rate (20.3% eligible) and a 66.2% warm hit rate (82.8% eligible). In production environments with purely unique, non-repeating long-tail prompts, exact cache hit rates may approach 0%.
  - Exact SHA-256 canonical hashing requires character-exact parameter and prompt matches. A single trailing whitespace character or timestamp variation bypasses the cache entirely.
- **Mitigation**: The ablation study explicitly tested `ablation_no_cache` to isolate the router's economic performance in the total absence of cache hits ($84.0\%$ cost reduction maintained purely via Tier A, B, and C routing).

### 2.3 Rule and Analyzer Precision
- **Risk**: The rule analyzer utilizes deterministic keyword and regex heuristics to identify Tier A (math/string/time) and Tier B (structured templates) candidates.
- **Confounding Factors**: Unconstrained user inputs in production may trigger false positives (routing a complex query to a simplistic regex extractor) or false negatives (failing to recognize a formula and escalating unnecessarily to an expensive model).
- **Mitigation**: InferPath places Tier A and Tier B behind post-execution validators (`FormatValidator`, `NonEmptyValidator`, `LengthValidator`). When a rule or template fails validation, the bounded fallback mechanism escalates execution to Tier C or Tier D.

---

## 3. External Validity

External validity addresses whether findings from the 320-prompt InferPath benchmark generalize to wider real-world domains and operational contexts.

### 3.1 Workload Representativeness
- **Risk**: The evaluation dataset consists of 320 curated prompts across 10 distinct task families (math calculation, datetime queries, formal email generation, meeting minutes templates, document summarization, sentiment extraction, JSON schema extraction, factual QA, confidential internal requests, and adversarial prompt injections).
- **Limitations**:
  - *Missing Domains*: The benchmark does not evaluate multi-turn conversational agents, complex code synthesis/debugging, open-ended creative storytelling, or multi-step tool-use chains.
  - *Task Length Distribution*: Prompts are relatively short to medium in length (50 to 500 tokens). Very long context queries (>32k tokens) were not tested and would exhibit different memory and latency characteristics.
- **Conclusion**: InferPath's demonstrated 84.0% cost reduction applies specifically to **heterogeneous enterprise automation workloads** with mixed administrative, analytical, and structured text tasks.

### 3.2 Upstream Model Frontier Shifts
- **Risk**: Commercial LLM pricing, capability boundaries, and token efficiencies shift rapidly.
- **Limitations**:
  - A small model from 2026 (e.g., Gemini 2.5 Flash) may outperform a large model from 2024 (e.g., GPT-4) on tasks previously thought to require frontier models.
  - Provider price reductions may compress the economic spread between small and large model tiers, altering the cost-benefit ratio of multi-tier routing.
- **Mitigation**: All pricing tables in `configs/model_pricing.yaml` are explicitly dated and configurable. The routing decision engine bases escalation on verified task complexity thresholds rather than static model names.

### 3.3 Regional Grid and Datacenter Heterogeneity
- **Risk**: Datacenter Power Usage Effectiveness (PUE) and regional carbon/water intensity vary dramatically across geographic regions (e.g., hydroelectric grids in the Pacific Northwest vs. coal-dominated grids).
- **Limitations**: Applying a single national central PUE ($1.20$) and water intensity ($2.40\text{ mL/Wh}$) obscures local environmental realities.
- **Mitigation**: InferPath models wide uncertainty bounds ($[1.10, 1.35]$ for PUE, $[1.20, 4.50]$ for water intensity) and explicitly marks all figures as `[ESTIMATED]`.

---

## 4. Construct Validity

Construct validity examines whether our quantitative evaluation metrics faithfully measure the conceptual phenomena of interest (e.g., "accuracy", "quality", "sustainability").

### 4.1 Exact Match vs. Semantic Equivalence
- **Risk**: Measuring cache efficiency strictly via SHA-256 canonical hashing risks undercounting semantically identical queries.
- **Analysis**: Two queries such as `"Summarize this report in 3 bullets"` and `"Give me a 3 bullet summary of this report"` will result in a cache miss under exact hashing despite demanding identical outputs. While semantic vector caching could increase hit rates, it introduces non-deterministic distance thresholds, privacy leak vectors, and indexing latency. InferPath chose exact canonical caching to prioritize correctness and privacy guarantees over speculative hit maximization.

### 4.2 Automated Quality Rubrics and Strict Assertion Checking
- **Risk**: Quality pass rates are quantified using programmatic assertion checkers (regex extraction, format validity, summary consistency) and automated LLM judge scoring.
- **Analysis**:
  - Strict programmatic checkers can penalize semantically flawless responses that deviate slightly in formatting (e.g., outputting `"Total: $500"` when the regex expected `"$500"`). This accounts for the relatively low baseline pass rates ($4.4\%$ to $6.2\%$) observed on the challenging pilot dataset.
  - Automated LLM judges may exhibit verbosity bias or alignment preference toward specific prompt structures.
- **Mitigation**: Automated judges were explicitly calibrated against independent human ratings, establishing high inter-rater reliability ($\kappa = 0.952$).

### 4.3 Environmental Proxies vs. Physical Metering
- **Risk**: Claiming "environmental sustainability" based on token counts and multiplication factors risks construct conflation.
- **Analysis**: Software tokens are not physical electrons or liters of water. Actual energy consumption depends on GPU frequency scaling, memory bandwidth utilization, batch concurrency, and server cooling cycles.
- **Mitigation**: InferPath enforces strict terminology guidelines: environmental metrics are never presented as measured truths, and headline marketing claims are prohibited.

---

## 5. Statistical Conclusion Validity

Statistical conclusion validity concerns whether the mathematical analysis and significance claims are justified.

### 5.1 Clustered Correlation Within Task Families
- **Risk**: Standard i.i.d. bootstrap resampling assumes query independence. However, queries within the same `task_family_id` share domain characteristics and prompt templates, violating independence.
- **Resolution**: InferPath implements **cluster bootstrapping** resampled over the 10 distinct task families ($B=1,000$ iterations). This properly widens confidence intervals to reflect inter-cluster variance.

### 5.2 Effect Size Quantification
- **Risk**: Relying exclusively on percentage deltas can exaggerate differences on small absolute baselines.
- **Resolution**: In addition to absolute dollar differences, InferPath computes non-parametric **Cliff's delta ($\delta$)** effect sizes. The cost difference between System C and System A achieved $\delta = -1.00$ ($p < 0.001$), confirming a large, robust effect that does not depend on normality assumptions.

### 5.3 Sample Size Limitations
- **Risk**: The current pilot evaluation dataset contains 320 items (160 in `dev`, 160 in `test`).
- **Resolution**: While 160 items across 10 families are sufficient to detect the massive cost disparity between small and large models ($d = -1.00$), finer-grained ablation comparisons (e.g., `ablation_no_templates` vs. `System C`) exhibit subtle differences that require larger sample sizes for high-powered statistical distinction.

---

## 6. What Can and Cannot Be Generalized

To ensure rigorous interpretation by academic faculty and industry evaluators, the table below delineates the verified generalizable conclusions from unverified extrapolations:

| Dimension | What CAN Be Generalized | What CANNOT Be Generalized |
| :--- | :--- | :--- |
| **Routing Architecture** | Pre-execution deterministic rules and Jinja2 templates effectively handle administrative and formulaic queries without incurring any LLM inference costs. | Complex multi-turn dialogues or novel reasoning problems can be resolved without LLM involvement. |
| **Privacy Guardrails** | Pre-routing privacy classification reliably prevents sensitive internal data from reaching external cloud model adapters. | Privacy classifiers are infallible against sophisticated adversarial steganography without regular updates. |
| **Economic Scaling** | Routing the majority of standard automation queries to small models (Tier C) yields order-of-magnitude cost savings over uniform frontier model routing. | The exact 84.0% cost reduction figure will replicate identically on arbitrary user workloads with different complexity distributions. |
| **Cache Mechanics** | Canonical exact-hash caching delivers zero false-hit rate and eliminates redundant computation on identical workloads. | High cache hit rates will be achieved on workloads characterized by entirely unique, creative user prompts. |
| **Environmental Impact** | Reducing token throughput and GPU server utilization directionally reduces datacenter electrical and cooling demands. | The reported 1.716 Wh energy and 4.119 mL water figures represent verified physical meter readings. |
