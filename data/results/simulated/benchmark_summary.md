# InferPath Phase 5 Research Evaluation Summary

**Adapter Mode**: `simulated` | **Split**: `dev` | **Repeats**: `3`

| System | Quality_Pass_Rate | Quality_95_CI | Total_Cost_USD | Cost_Reduction_vs_A | Avg_Latency_ms | P95_Latency_ms | Cache_Hit_Rate | Fallback_Rate | Avg_Cost_Regret | Over_Escalation | Under_Escalation | Energy_Wh | Water_mL | Cost_Effect_vs_C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_A | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.02720 | 0.1% | 89.5 ms | 124.0 ms | 0.0% | 0.0% | $0.000200 | 65.6% | 0.0% | 11.811 Wh | 28.347 mL | d=-1.00 (large) |
| System_Small | 65.6% ± 0.00% | [53.8%, 75.6%] | $0.00040 | 98.5% | 29.3 ms | 49.2 ms | 18.1% | 0.0% | $0.000000 | 0.0% | 0.0% | 0.626 Wh | 1.501 mL | d=1.00 (large) |
| System_B | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.01950 | 28.3% | 65.4 ms | 122.5 ms | 20.0% | 0.0% | $0.000100 | 62.5% | 0.0% | 8.319 Wh | 19.965 mL | d=-1.00 (large) |
| System_C | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 27.9 ms | 49.7 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 1.862 Wh | 4.468 mL | d=0.00 (negligible) |
| System_Learned | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00200 | 92.7% | 43.5 ms | 83.7 ms | 16.2% | 4.4% | $0.000000 | 2.5% | 1.9% | 1.284 Wh | 3.081 mL | d=1.00 (large) |
| System_Oracle | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00040 | 98.5% | 28.3 ms | 50.7 ms | 16.2% | 0.0% | $0.000000 | 0.0% | 1.9% | 0.572 Wh | 1.373 mL | d=1.00 (large) |
| ablation_no_templates | 66.2% ± 0.00% | [54.4%, 76.2%] | $0.00410 | 84.9% | 28.7 ms | 49.8 ms | 16.9% | 4.4% | $0.000000 | 5.6% | 0.0% | 1.903 Wh | 4.566 mL | d=-1.00 (large) |
| ablation_no_small_model | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.01840 | 32.4% | 29.5 ms | 55.8 ms | 16.2% | 0.0% | $0.000100 | 57.5% | 1.9% | 8.011 Wh | 19.227 mL | d=-1.00 (large) |
| ablation_no_fallback | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00290 | 89.3% | 30.8 ms | 54.5 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.424 Wh | 3.418 mL | d=1.00 (large) |
| ablation_no_cache | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 30.6 ms | 62.2 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 1.862 Wh | 4.468 mL | d=0.00 (negligible) |
| ablation_no_validation | 66.2% ± 0.00% | [53.8%, 76.9%] | $0.00290 | 89.3% | 41.3 ms | 176.4 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.424 Wh | 3.418 mL | d=1.00 (large) |

### Inter-Rater Agreement
- **Human-Human Quadratic Weighted Kappa**: `Not Established` (pending empirical human review study)
- **Human-Judge Quadratic Weighted Kappa**: `Not Established` (pending empirical human review study)

### Cache Performance (Cold vs Warm)
- **Cold Cache Hit Rate**: `16.2%` (Inappropriate Hits: `0`)
- **Warm Cache Hit Rate**: `66.2%` (Inappropriate Hits: `0`)
- **Warm Cache Cost Savings**: `$0.01590`
