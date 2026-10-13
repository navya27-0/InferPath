# InferPath Phase 5 Research Evaluation Summary

**Adapter Mode**: `simulated` | **Split**: `dev` | **Repeats**: `3`

| System | Quality_Pass_Rate | Quality_95_CI | Total_Cost_USD | Cost_Reduction_vs_A | Avg_Latency_ms | P95_Latency_ms | Cache_Hit_Rate | Fallback_Rate | Avg_Cost_Regret | Over_Escalation | Under_Escalation | Energy_Wh | Water_mL | Cost_Effect_vs_C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_A | 4.4% ± 0.00% | [4.4%, 4.4%] | $0.03070 | 0.0% | 113.4 ms | 141.5 ms | 0.0% | 0.0% | $0.000200 | 7.5% | 0.0% | 45.695 Wh | 109.668 mL | d=-1.00 (large) |
| System_B | 6.2% ± 0.00% | [6.2%, 6.2%] | $0.01960 | 36.2% | 76.1 ms | 141.5 ms | 20.0% | 0.0% | $0.000100 | 3.1% | 0.0% | 29.209 Wh | 70.102 mL | d=-1.00 (large) |
| System_C | 6.2% ± 0.00% | [6.2%, 6.2%] | $0.00490 | 84.0% | 20.5 ms | 39.9 ms | 14.4% | 8.8% | $0.000000 | 0.0% | 0.0% | 1.716 Wh | 4.119 mL | d=0.00 (negligible) |
| ablation_no_templates | 6.2% ± 0.00% | [6.2%, 6.2%] | $0.00490 | 84.0% | 20.2 ms | 36.8 ms | 16.2% | 8.8% | $0.000000 | 0.0% | 0.0% | 1.644 Wh | 3.945 mL | d=-1.00 (large) |
| ablation_no_small_model | 6.2% ± 0.00% | [6.2%, 6.2%] | $0.02810 | 8.5% | 23.4 ms | 52.0 ms | 14.4% | 0.0% | $0.000100 | 2.5% | 0.0% | 6.864 Wh | 16.473 mL | d=-1.00 (large) |
| ablation_no_fallback | 6.2% ± 0.00% | [6.2%, 6.2%] | $0.00190 | 93.8% | 25.0 ms | 50.4 ms | 14.4% | 0.0% | $0.000000 | 0.0% | 0.0% | 1.119 Wh | 2.686 mL | d=1.00 (large) |
| ablation_no_cache | 6.2% ± 0.00% | [6.2%, 6.2%] | $0.00490 | 84.0% | 25.5 ms | 53.1 ms | 14.4% | 8.8% | $0.000000 | 0.0% | 0.0% | 2.209 Wh | 5.301 mL | d=0.00 (negligible) |
| ablation_no_validation | 6.2% ± 0.00% | [6.2%, 6.2%] | $0.00190 | 93.8% | 25.9 ms | 48.0 ms | 15.0% | 0.0% | $0.000000 | 0.0% | 0.0% | 1.077 Wh | 2.585 mL | d=1.00 (large) |

### Inter-Rater Agreement
- **Human-Human Quadratic Weighted Kappa**: `0.888`
- **Human-Judge Quadratic Weighted Kappa**: `0.952`

### Cache Performance (Cold vs Warm)
- **Cold Cache Hit Rate**: `21.9%` (Inappropriate Hits: `0`)
- **Warm Cache Hit Rate**: `59.4%` (Inappropriate Hits: `0`)
- **Warm Cache Cost Savings**: `$0.01425`
