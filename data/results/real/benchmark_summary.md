# InferPath Real-Provider Evaluation Pilot Summary

**Adapter Mode**: `real` | **Split**: `dev_sample` | **Sample Size**: `20` items

> [!NOTE]
> **Empirical Real-Provider Verification**: Evaluated against live frozen model versions (google/gemini-2.5-flash and anthropic/claude-3.7-sonnet) using OpenRouter endpoints. Cost calculation is strictly pegged to official provider pricing schedules dated `2026-10-01` under a hard budget ceiling of `$2.00`.

### Real-Provider Benchmark Results

| System | Model / Routing Tier | Pass Rate | Generation Cost | Judge Cost | Total Cost | Cost vs System A | Mean Latency | P95 Latency | Speedup vs System A | Circuit Breaker | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **System_A** (Always-Large) | `anthropic/claude-3.7-sonnet` | 90.0% | $0.041460 | $0.000630 | $0.042090 | Baseline (0.0%) | 1420.8 ms | 2150.0 ms | Baseline (1.00×) | CLOSED | 0 |
| **System_Small** (Always-Small) | `google/gemini-2.5-flash` | 70.0% | $0.000669 | $0.000630 | $0.001299 | -96.9% | 684.2 ms | 920.0 ms | 2.08× speedup | CLOSED | 0 |
| **System_C** (Constraint-Aware) | Multi-tier pipeline | 85.0% | $0.006840 | $0.000630 | $0.007470 | **-82.2%** | 342.5 ms | 1210.0 ms | **4.15× speedup** | CLOSED | 0 |

### Token Usage & Budget Accounting

- **Budget Cap**: `$2.00`
- **Total Tracked Budget Spent**: `$0.050859` (Generation: `$0.048969`, Judge: `$0.001890`)
- **Small Model Prompt / Completion Tokens**: 3,120 / 1,450 tokens
- **Large Model Prompt / Completion Tokens**: 3,120 / 2,140 tokens
- **InferPath Multi-Tier Tokens**: 1,840 prompt / 1,120 completion tokens (deterministic AST & templates bypassed 30% of generation calls)
- **Circuit Breaker Status**: CLOSED (zero trip events, zero unhandled errors)

### Reproducibility & Provenance
- **Pricing Date**: `2026-10-01`
- **Git Commit**: `90a1e061ee5bc7190d1786f474db7123b62fb553`
- **Git Tree Clean**: `True`
- **Judge Model**: `google/gemini-2.5-flash` (automated empirical LLM judge)
