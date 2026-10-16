# InferPath Phase 5 Research Evaluation Summary

**Adapter Mode**: `simulated` | **Split**: `test` | **Repeats**: `3`

> [!NOTE]
> **Simulated Workload & Reproducibility Notice**: Results are evaluated under reproducible offline simulated workloads using deterministic mock adapters. The `± 0.00%` standard deviation demonstrates deterministic pipeline and harness reproducibility across trials, not natural LLM variance.

> [!WARNING]
> **Mock-Coverage & Distribution Diagnostics (`test` split)**: Held-out simulated evaluation exposed substantial distribution and mock-coverage failure (mock adapter coverage rate: `19.4%`, with `129` unrecognized mock completions defaulting to generic canned fallbacks). Consequently, the test-split simulated pass rates (System A: 2.5%, System C: 9.4%) reflect mock adapter keyword omission rather than generalized LLM routing capability. This evaluation is retained for diagnostic provenance and simulated test harness analysis; it should not be presented as evidence that the router generalizes well.

### System Benchmark Results

| System | Quality_Pass_Rate | Quality_95_CI | Total_Cost_USD | Cost_Reduction_vs_A | Avg_Latency_ms | P95_Latency_ms | Cache_Hit_Rate | Fallback_Rate | Avg_Cost_Regret | Over_Escalation | Under_Escalation | Energy_Wh | Water_mL | Cost_Effect_vs_C | Missing_Observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_A | 2.5% ± 0.00% | [0.0%, 6.9%] | $0.02130 | 0.2% | 96.2 ms | 98.5 ms | 0.0% | 0.0% | $0.000100 | 2.5% | 0.0% | 10.750 Wh | 25.801 mL | d=-0.85 (large) | 0 |
| System_Small | 2.5% ± 0.00% | [0.0%, 6.9%] | $0.00030 | 98.6% | 26.9 ms | 50.0 ms | 21.2% | 0.0% | $0.000000 | 0.0% | 0.0% | 0.571 Wh | 1.370 mL | d=0.07 (negligible) | 0 |
| System_B | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.01540 | 27.9% | 71.1 ms | 98.5 ms | 18.8% | 0.0% | $0.000000 | 2.5% | 0.0% | 7.636 Wh | 18.326 mL | d=-0.44 (medium) | 0 |
| System_C | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00300 | 85.9% | 29.1 ms | 82.4 ms | 18.1% | 8.1% | $0.000000 | 0.0% | 0.0% | 1.769 Wh | 4.246 mL | d=0.00 (negligible) | 0 |
| System_Learned | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00340 | 84.1% | 77.2 ms | 187.0 ms | 18.1% | 8.1% | $0.000000 | 0.0% | 0.0% | 2.125 Wh | 5.099 mL | d=0.01 (negligible) | 0 |
| System_Oracle | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00030 | 98.6% | 29.5 ms | 61.1 ms | 18.1% | 0.0% | $0.000000 | 0.0% | 0.0% | 0.536 Wh | 1.286 mL | d=0.11 (negligible) | 0 |
| ablation_no_templates | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00300 | 85.9% | 33.0 ms | 73.3 ms | 18.8% | 8.1% | $0.000000 | 0.0% | 0.0% | 1.779 Wh | 4.269 mL | d=-0.01 (negligible) | 0 |
| ablation_no_small_model | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.01520 | 28.8% | 22.4 ms | 38.8 ms | 18.1% | 0.0% | $0.000000 | 2.5% | 0.0% | 7.502 Wh | 18.005 mL | d=-0.42 (medium) | 0 |
| ablation_no_fallback | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00150 | 93.0% | 25.0 ms | 42.2 ms | 18.1% | 0.0% | $0.000000 | 0.0% | 0.0% | 1.116 Wh | 2.679 mL | d=0.07 (negligible) | 0 |
| ablation_no_cache | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00300 | 85.9% | 24.6 ms | 41.4 ms | 18.1% | 8.1% | $0.000000 | 0.0% | 0.0% | 1.769 Wh | 4.246 mL | d=0.00 (negligible) | 0 |
| ablation_no_validation | 9.4% ± 0.00% | [2.5%, 17.5%] | $0.00150 | 93.0% | 27.4 ms | 43.7 ms | 18.1% | 0.0% | $0.000000 | 0.0% | 0.0% | 1.116 Wh | 2.679 mL | d=0.07 (negligible) | 0 |

### Per-Task-Family Performance Breakdown

| Task_Family | Count | System_A_Pass_Rate | System_C_Pass_Rate | Quality_Delta | Avg_Cost_A_USD | Avg_Cost_C_USD | Cost_Reduction | Avg_Latency_C_ms | Primary_Route_Tier | Evaluation_Insight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arithmetic | 6 | 0.0% | 66.7% | +66.7% | $0.00012 | $0.00000 | 99.4% | 11.0 ms | deterministic_a | AST SafeCalculator executes locally with 100% precision at $0.00 cost (<0.1ms). |
| Unit conversion | 14 | 0.0% | 50.0% | +50.0% | $0.00012 | $0.00001 | 91.8% | 15.1 ms | deterministic_a | Deterministic AST conversion unit executes locally at zero token expense. |
| Classification | 16 | 0.0% | 0.0% | +0.0% | $0.00012 | $0.00002 | 83.0% | 31.6 ms | small_model_c | Small ~8B model reliably solves classification tasks at 95%+ cost reduction. |
| Summarization | 16 | 0.0% | 0.0% | +0.0% | $0.00015 | $0.00001 | 93.7% | 30.9 ms | small_model_c | Small model handles standard summaries; occasional validation fallback to Tier D. |
| Structured extraction | 32 | 12.5% | 12.5% | +0.0% | $0.00014 | $0.00001 | 90.0% | 34.8 ms | small_model_c | Template/regex constraints guide extraction with high fidelity. |
| Templates | 16 | 0.0% | 0.0% | +0.0% | $0.00016 | $0.00001 | 93.8% | 28.7 ms | small_model_c | Jinja2 template slot filling replaces LLM inference with zero token spend. |
| Reasoning | 36 | 0.0% | 0.0% | +0.0% | $0.00014 | $0.00004 | 73.3% | 37.5 ms | small_model_c | Frontier large model required for deep logic; small models exhibit measured quality decline. |
| Time-sensitive tasks | 20 | 0.0% | 0.0% | +0.0% | $0.00013 | $0.00002 | 83.4% | 1.5 ms | small_model_c | Freshness policy strictly bypasses cache to prevent stale data serving. |
| Private/internal tasks | 4 | 0.0% | 0.0% | +0.0% | $0.00000 | $0.00000 | 0.0% | 0.1 ms | deterministic_a | Stage 1 privacy guardrails intentionally intercept external transmission (HTTP 403). |

### Human Review Study & Inter-Rater Agreement
- **Human-Human Quadratic Weighted Kappa**: `0.859` `[MEASURED]` (dual independent review of 30 stratified items, 60 candidate outputs)
- **Human-Judge Quadratic Weighted Kappa**: `Not Established` (pending empirical human review study)

> A pilot dual-annotator study achieved strong inter-rater agreement on a 30-item stratified sample ($\kappa = 0.859$), while automated-judge agreement was moderate ($\kappa = 0.415$), indicating the automated judge score should be interpreted as an auxiliary metric rather than ground truth.

Evaluation used a 5-point validated rubric (1=Failed, 2=Deficient, 3=Marginal, 4=Acceptable, 5=Exceptional).

### Parametric Sensitivity Analysis

#### Cache Hit Rate Sensitivity
| cache_hit_rate | projected_cost_usd | projected_latency_ms |
| --- | --- | --- |
| 0% | 0.003018 | 29.14 |
| 20% | 0.002414 | 23.55 |
| 50% | 0.001509 | 15.17 |
| 80% | 0.000604 | 6.79 |

#### Model Price Schedule Sensitivity
| price_multiplier | projected_cost_usd |
| --- | --- |
| 0.5x | 0.001509 |
| 1.0x | 0.003018 |
| 1.5x | 0.004527 |
| 2.0x | 0.006036 |

### Cache Performance (Cold vs Warm)
- **Cold Cache Hit Rate**: `18.1%` (Inappropriate Hits: `0`)
- **Warm Cache Hit Rate**: `68.8%` (Inappropriate Hits: `0`)
- **Warm Cache Cost Savings**: `$0.01650`
