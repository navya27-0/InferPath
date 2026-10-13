# InferPath Phase 5 Research Evaluation Summary

**Adapter Mode**: `simulated` | **Split**: `dev` | **Repeats**: `3`

| System | Quality_Pass_Rate | Quality_95_CI | Total_Cost_USD | Cost_Reduction_vs_A | Avg_Latency_ms | P95_Latency_ms | Cache_Hit_Rate | Fallback_Rate | Avg_Cost_Regret | Over_Escalation | Under_Escalation | Energy_Wh | Water_mL | Cost_Effect_vs_C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_A | 69.4% ± 0.00% | [69.4%, 69.4%] | $0.02720 | 0.1% | 89.5 ms | 124.0 ms | 0.0% | 0.0% | $0.000200 | 66.2% | 0.0% | 36.016 Wh | 86.438 mL | d=-1.00 (large) |
| System_B | 69.4% ± 0.00% | [69.4%, 69.4%] | $0.01800 | 33.9% | 60.5 ms | 122.5 ms | 20.0% | 0.0% | $0.000100 | 56.2% | 0.0% | 23.255 Wh | 55.812 mL | d=-1.00 (large) |
| System_C | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00410 | 84.9% | 35.2 ms | 69.2 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 2.270 Wh | 5.447 mL | d=0.00 (negligible) |
| ablation_no_templates | 66.2% ± 0.00% | [66.2%, 66.2%] | $0.00410 | 84.9% | 32.1 ms | 57.1 ms | 16.9% | 4.4% | $0.000000 | 5.6% | 0.0% | 2.182 Wh | 5.238 mL | d=-1.00 (large) |
| ablation_no_small_model | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.02220 | 18.4% | 32.6 ms | 66.5 ms | 16.2% | 0.0% | $0.000100 | 57.5% | 1.9% | 9.807 Wh | 23.537 mL | d=-1.00 (large) |
| ablation_no_fallback | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00290 | 89.3% | 35.3 ms | 66.3 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.509 Wh | 3.621 mL | d=1.00 (large) |
| ablation_no_cache | 63.1% ± 0.00% | [63.1%, 63.1%] | $0.00410 | 84.9% | 35.4 ms | 70.7 ms | 16.2% | 4.4% | $0.000000 | 5.6% | 1.9% | 2.300 Wh | 5.521 mL | d=0.00 (negligible) |
| ablation_no_validation | 66.2% ± 0.00% | [66.2%, 66.2%] | $0.00290 | 89.3% | 36.5 ms | 66.0 ms | 16.2% | 0.0% | $0.000000 | 5.6% | 1.9% | 1.576 Wh | 3.783 mL | d=1.00 (large) |

### Inter-Rater Agreement
- **Human-Human Quadratic Weighted Kappa**: `0.888`
- **Human-Judge Quadratic Weighted Kappa**: `0.952`

### Cache Performance (Cold vs Warm)
- **Cold Cache Hit Rate**: `16.2%` (Inappropriate Hits: `0`)
- **Warm Cache Hit Rate**: `66.2%` (Inappropriate Hits: `0`)
- **Warm Cache Cost Savings**: `$0.01590`
