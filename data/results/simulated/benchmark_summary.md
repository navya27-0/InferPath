# InferPath Phase 5 Research Evaluation Summary

**Adapter Mode**: `simulated` | **Split**: `dev` | **Repeats**: `3`

| System | Quality_Pass_Rate | Quality_95_CI | Total_Cost_USD | Cost_Reduction_vs_A | Avg_Latency_ms | P95_Latency_ms | Cache_Hit_Rate | Fallback_Rate | Avg_Cost_Regret | Over_Escalation | Under_Escalation | Energy_Wh | Water_mL | Cost_Effect_vs_C | Missing_Observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_A | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.02720 | 0.1% | 89.5 ms | 124.0 ms | 0.0% | 0.0% | $0.000200 | 65.6% | 0.0% | 11.811 Wh | 28.347 mL | d=-0.63 (large) | 0 |
| System_Small | 65.6% ± 0.00% | [53.8%, 75.6%] | $0.00040 | 98.5% | 31.3 ms | 68.2 ms | 18.1% | 0.0% | $0.000000 | 0.0% | 0.0% | 0.626 Wh | 1.501 mL | d=-0.02 (negligible) | 0 |
| System_B | 69.4% ± 0.00% | [56.2%, 80.0%] | $0.02000 | 26.5% | 67.2 ms | 122.5 ms | 16.9% | 0.0% | $0.000100 | 62.5% | 0.0% | 8.583 Wh | 20.600 mL | d=-0.33 (small) | 0 |
| System_C | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 51.1 ms | 150.3 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 1.862 Wh | 4.468 mL | d=0.00 (negligible) | 0 |
| System_Learned | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00200 | 92.7% | 48.4 ms | 96.7 ms | 16.2% | 4.4% | $0.000000 | 2.5% | 1.9% | 1.284 Wh | 3.081 mL | d=-0.00 (negligible) | 0 |
| System_Oracle | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00040 | 98.5% | 32.6 ms | 68.0 ms | 16.2% | 0.0% | $0.000000 | 0.0% | 1.9% | 0.572 Wh | 1.373 mL | d=0.04 (negligible) | 0 |
| ablation_no_templates | 66.2% ± 0.00% | [54.4%, 76.2%] | $0.00410 | 84.9% | 41.8 ms | 86.0 ms | 16.9% | 4.4% | $0.000000 | 5.6% | 0.0% | 1.903 Wh | 4.566 mL | d=-0.04 (negligible) | 0 |
| ablation_no_small_model | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.01840 | 32.4% | 34.7 ms | 72.2 ms | 16.2% | 0.0% | $0.000100 | 57.5% | 1.9% | 8.011 Wh | 19.227 mL | d=-0.27 (small) | 0 |
| ablation_no_fallback | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00290 | 89.3% | 39.1 ms | 81.0 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.424 Wh | 3.418 mL | d=0.03 (negligible) | 0 |
| ablation_no_cache | 63.1% ± 0.00% | [51.2%, 73.8%] | $0.00410 | 84.9% | 36.5 ms | 73.5 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 1.862 Wh | 4.468 mL | d=0.00 (negligible) | 0 |
| ablation_no_validation | 66.2% ± 0.00% | [53.8%, 76.9%] | $0.00290 | 89.3% | 38.2 ms | 71.1 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.424 Wh | 3.418 mL | d=0.03 (negligible) | 0 |

### Inter-Rater Agreement
- **Human-Human Quadratic Weighted Kappa**: `Not Established` (pending empirical human review study)
- **Human-Judge Quadratic Weighted Kappa**: `Not Established` (pending empirical human review study)

### Cache Performance (Cold vs Warm)
- **Cold Cache Hit Rate**: `16.2%` (Inappropriate Hits: `0`)
- **Warm Cache Hit Rate**: `66.2%` (Inappropriate Hits: `0`)
- **Warm Cache Cost Savings**: `$0.01590`
