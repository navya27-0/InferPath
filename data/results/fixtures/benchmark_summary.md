# InferPath Offline Pilot Fixture (Synthetic Demonstration)

**Adapter Mode**: `fixture` | **Split**: `dev` | **Sample Size**: `20` items

> [!NOTE]
> **Synthetic Demonstration Notice**: This artifact is an offline demonstration fixture designed for CI regression and schema validation. **Live provider network calls were NOT performed.** All results are explicitly marked with `adapter_mode: fixture`, `is_fixture: true`, and `live_network_calls: false`. Genuine real-provider evaluations require an explicit `OPENROUTER_API_KEY` and record outputs directly into `data/results/real/`.

### Offline Fixture Model Results

| System | Model / Routing Tier | Pass Rate | Generation Cost | Judge Cost | Total Cost | Cost vs System A | Mean Latency | P95 Latency | Speedup vs System A | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **System_A** (Always-Large) | `anthropic/claude-3.7-sonnet` | 90.0% | $0.041460 | $0.000630 | $0.042090 | Baseline | 1420.8 ms | 2150.0 ms | 1.00× | `[FIXTURE]` |
| **System_Small** (Always-Small) | `google/gemini-2.5-flash` | 70.0% | $0.000669 | $0.000630 | $0.001299 | -96.9% | 684.2 ms | 920.0 ms | 2.08× speedup | `[FIXTURE]` |
| **System_C** (Constraint-Aware) | Multi-tier pipeline | 85.0% | $0.006840 | $0.000630 | $0.007470 | **-82.2%** | 342.5 ms | 1210.0 ms | **4.15× speedup** | `[FIXTURE]` |

### Provenance & Safeguards
- **Fixture Mode**: Active (`is_fixture: true`, `live_network_calls: false`)
- **Pricing Schedule**: `2026-10-01`
- **Storage Location**: `data/results/fixtures/` (completely isolated from live `data/results/real/`)
