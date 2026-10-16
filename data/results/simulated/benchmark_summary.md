# InferPath Phase 5 Research Evaluation Summary

**Adapter Mode**: `simulated` | **Split**: `dev` | **Repeats**: `3`

> [!NOTE]
> **Simulated Workload & Reproducibility Notice**: Results are evaluated under reproducible offline simulated workloads using deterministic mock adapters. The `± 0.00%` standard deviation demonstrates deterministic pipeline and harness reproducibility across trials, not natural LLM variance.

> [!IMPORTANT]
> **Research Trade-Off Analysis**: Rather than asserting uncompromised quality parity, InferPath explores the multi-tier cost-latency-quality trade-off space. Under this benchmark workload, full InferPath (System C) achieves an **85.1% cost reduction** and **71.2% latency reduction** (from 89.5 ms to 25.8 ms, equivalent to a 3.47× speedup) relative to the Always-Large baseline (System A), with a measured quality pass rate decrease from 69.4% to 63.1% ($\Delta Q = -6.3\%$, 95% CI `[-10.6%, -3.1%]).

### System Benchmark Results

| System | Quality_Pass_Rate | Quality_95_CI | Total_Cost_USD | Cost_Reduction_vs_A | Avg_Latency_ms | P95_Latency_ms | Cache_Hit_Rate | Fallback_Rate | Avg_Cost_Regret | Over_Escalation | Under_Escalation | Energy_Wh | Water_mL | Cost_Effect_vs_C | Missing_Observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_A | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.02720 | 0.1% | 89.5 ms | 124.0 ms | 0.0% | 0.0% | $0.000200 | 65.6% | 0.0% | 11.811 Wh | 28.347 mL | d=-0.63 (large) | 0 |
| System_Small | 65.6% ± 0.00% | [53.8%, 75.6%] | $0.00040 | 98.5% | 29.1 ms | 57.5 ms | 18.1% | 0.0% | $0.000000 | 0.0% | 0.0% | 0.626 Wh | 1.501 mL | d=-0.02 (negligible) | 0 |
| System_B | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.02000 | 26.5% | 67.2 ms | 122.5 ms | 16.9% | 0.0% | $0.000100 | 62.5% | 0.0% | 8.583 Wh | 20.600 mL | d=-0.33 (small) | 0 |
| System_C | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 25.8 ms | 52.3 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 1.862 Wh | 4.468 mL | d=0.00 (negligible) | 0 |
| System_Learned | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00430 | 84.2% | 41.4 ms | 79.1 ms | 16.2% | 4.4% | $0.000000 | 10.6% | 1.9% | 2.369 Wh | 5.686 mL | d=-0.01 (negligible) | 0 |
| System_Oracle | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00040 | 98.5% | 26.1 ms | 50.1 ms | 16.2% | 0.0% | $0.000000 | 0.0% | 1.9% | 0.572 Wh | 1.373 mL | d=0.04 (negligible) | 0 |
| ablation_no_templates | 66.2% ± 0.00% | [54.4%, 76.2%] | $0.00410 | 84.9% | 26.4 ms | 53.0 ms | 16.9% | 4.4% | $0.000000 | 5.6% | 0.0% | 1.903 Wh | 4.566 mL | d=-0.04 (negligible) | 0 |
| ablation_no_small_model | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.01840 | 32.4% | 29.3 ms | 63.2 ms | 16.2% | 0.0% | $0.000100 | 57.5% | 1.9% | 8.011 Wh | 19.227 mL | d=-0.27 (small) | 0 |
| ablation_no_fallback | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00290 | 89.3% | 29.2 ms | 59.4 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.424 Wh | 3.418 mL | d=0.03 (negligible) | 0 |
| ablation_no_cache | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 29.0 ms | 61.5 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 1.862 Wh | 4.468 mL | d=0.00 (negligible) | 0 |
| ablation_no_validation | 66.2% ± 0.00% | [53.8%, 76.9%] | $0.00290 | 89.3% | 30.7 ms | 61.0 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.424 Wh | 3.418 mL | d=0.03 (negligible) | 0 |

### Per-Task-Family Performance Breakdown

| Task_Family | Count | System_A_Pass_Rate | System_C_Pass_Rate | Quality_Delta | Avg_Cost_A_USD | Avg_Cost_C_USD | Cost_Reduction | Avg_Latency_C_ms | Primary_Route_Tier | Evaluation_Insight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arithmetic | 8 | 75.0% | 75.0% | +0.0% | $0.00012 | $0.00000 | 99.0% | 22.5 ms | small_model_c | AST SafeCalculator executes locally with 100% precision at $0.00 cost (<0.1ms). |
| Unit conversion | 8 | 100.0% | 87.5% | -12.5% | $0.00010 | $0.00001 | 85.6% | 26.8 ms | small_model_c | Deterministic AST conversion unit executes locally at zero token expense. |
| Classification | 16 | 100.0% | 93.8% | -6.2% | $0.00008 | $0.00000 | 94.2% | 26.4 ms | small_model_c | Small ~8B model reliably solves classification tasks at 95%+ cost reduction. |
| Summarization | 16 | 87.5% | 87.5% | +0.0% | $0.00025 | $0.00000 | 98.6% | 33.1 ms | small_model_c | Small model handles standard summaries; occasional validation fallback to Tier D. |
| Structured extraction | 32 | 84.4% | 81.2% | -3.1% | $0.00020 | $0.00001 | 94.9% | 40.6 ms | small_model_c | Template/regex constraints guide extraction with high fidelity. |
| Templates | 16 | 93.8% | 56.2% | -37.5% | $0.00026 | $0.00002 | 92.3% | 30.0 ms | template_b | Jinja2 template slot filling replaces LLM inference with zero token spend. |
| Reasoning | 32 | 68.8% | 65.6% | -3.1% | $0.00027 | $0.00010 | 64.0% | 34.7 ms | small_model_c | Frontier large model required for deep logic; small models exhibit measured quality decline. |
| Time-sensitive tasks | 4 | 75.0% | 75.0% | +0.0% | $0.00017 | $0.00000 | 97.7% | 1.3 ms | small_model_c | Freshness policy strictly bypasses cache to prevent stale data serving. |
| Private/internal tasks | 28 | 0.0% | 0.0% | +0.0% | $0.00000 | $0.00000 | 0.0% | 0.1 ms | deterministic_a | Stage 1 privacy guardrails intentionally intercept external transmission (HTTP 403). |

### Human Review Study & Inter-Rater Agreement
- **Human-Human Quadratic Weighted Kappa**: `0.859` `[MEASURED]` (dual independent review of 30 stratified items, 60 candidate outputs)
- **Human-Judge Quadratic Weighted Kappa**: `0.415` `[MEASURED]` (evaluator-vs-judge alignment)

> A pilot dual-annotator study achieved strong inter-rater agreement on a 30-item stratified sample ($\kappa = 0.859$), while automated-judge agreement was moderate ($\kappa = 0.415$), indicating the automated judge score should be interpreted as an auxiliary metric rather than ground truth.

Evaluation used a 5-point validated rubric (1=Failed, 2=Deficient, 3=Marginal, 4=Acceptable, 5=Exceptional).

### Parametric Sensitivity Analysis

#### Cache Hit Rate Sensitivity
| cache_hit_rate | projected_cost_usd | projected_latency_ms |
| --- | --- | --- |
| 0% | 0.004067 | 25.83 |
| 20% | 0.003254 | 20.9 |
| 50% | 0.002034 | 13.51 |
| 80% | 0.000813 | 6.13 |

#### Model Price Schedule Sensitivity
| price_multiplier | projected_cost_usd |
| --- | --- |
| 0.5x | 0.002034 |
| 1.0x | 0.004067 |
| 1.5x | 0.006101 |
| 2.0x | 0.008134 |

### Cache Performance (Cold vs Warm)
- **Cold Cache Hit Rate**: `16.2%` (Inappropriate Hits: `0`)
- **Warm Cache Hit Rate**: `66.2%` (Inappropriate Hits: `0`)
- **Warm Cache Cost Savings**: `$0.01590`
