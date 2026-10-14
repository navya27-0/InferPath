# InferPath Phase 5 Research Evaluation Summary

**Adapter Mode**: `simulated` | **Split**: `dev` | **Repeats**: `3`

| System | Quality_Pass_Rate | Quality_95_CI | Total_Cost_USD | Cost_Reduction_vs_A | Avg_Latency_ms | P95_Latency_ms | Cache_Hit_Rate | Fallback_Rate | Avg_Cost_Regret | Over_Escalation | Under_Escalation | Energy_Wh | Water_mL | Cost_Effect_vs_C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_A | 69.4% ± 0.00% | [69.4%, 69.4%] | $0.02720 | 0.1% | 89.5 ms | 124.0 ms | 0.0% | 0.0% | $0.000200 | 65.6% | 0.0% | 11.811 Wh | 28.347 mL | d=-1.00 (large) |
| System_Small | 65.6% ± 0.00% | [65.6%, 65.6%] | $0.00040 | 98.5% | 30.8 ms | 62.6 ms | 18.1% | 0.0% | $0.000000 | 0.0% | 0.0% | 0.626 Wh | 1.501 mL | d=1.00 (large) |
| System_B | 69.4% ± 0.00% | [69.4%, 69.4%] | $0.01950 | 28.3% | 65.4 ms | 122.5 ms | 20.0% | 0.0% | $0.000100 | 62.5% | 0.0% | 8.319 Wh | 19.965 mL | d=-1.00 (large) |
| System_C | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00410 | 84.9% | 29.8 ms | 57.8 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 1.862 Wh | 4.468 mL | d=0.00 (negligible) |
| System_Learned | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00200 | 92.7% | 41.7 ms | 92.2 ms | 16.2% | 4.4% | $0.000000 | 2.5% | 1.9% | 1.284 Wh | 3.081 mL | d=1.00 (large) |
| ablation_no_templates | 66.2% ± 0.00% | [66.2%, 66.2%] | $0.00410 | 84.9% | 26.6 ms | 48.4 ms | 16.9% | 4.4% | $0.000000 | 5.6% | 0.0% | 1.903 Wh | 4.566 mL | d=-1.00 (large) |
| ablation_no_small_model | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.02220 | 18.4% | 27.5 ms | 57.7 ms | 16.2% | 0.0% | $0.000100 | 57.5% | 1.9% | 9.856 Wh | 23.654 mL | d=-1.00 (large) |
| ablation_no_fallback | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00290 | 89.3% | 26.5 ms | 48.8 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.424 Wh | 3.418 mL | d=1.00 (large) |
| ablation_no_cache | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00410 | 84.9% | 26.5 ms | 50.7 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 1.862 Wh | 4.468 mL | d=0.00 (negligible) |
| ablation_no_validation | 66.2% ± 0.00% | [66.2%, 66.2%] | $0.00290 | 89.3% | 29.7 ms | 59.5 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.424 Wh | 3.418 mL | d=1.00 (large) |

### Inter-Rater Agreement
- **Human-Human Quadratic Weighted Kappa**: `0.888`
- **Human-Judge Quadratic Weighted Kappa**: `0.952`

### Cache Performance (Cold vs Warm)
- **Cold Cache Hit Rate**: `66.2%` (Inappropriate Hits: `0`)
- **Warm Cache Hit Rate**: `66.2%` (Inappropriate Hits: `0`)
- **Warm Cache Cost Savings**: `$0.01590`
