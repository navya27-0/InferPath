# InferPath Offline Pilot Fixture (Synthetic Demonstration)

**Adapter Mode**: `fixture` | **Split**: `dev` | **Sample Size**: `20` items | **Repeats**: `2`

> [!NOTE]
> **Synthetic Demonstration Notice**: This artifact is an offline demonstration fixture designed for CI regression and schema validation. **Live provider network calls were NOT performed.** All results are explicitly marked with `adapter_mode: fixture`, `is_fixture: true`, and `live_network_calls: false`. Genuine real-provider evaluations require an explicit `OPENROUTER_API_KEY` and record outputs directly into `data/results/real/`.

### Offline Fixture Model Results

| System | Model / Routing Tier | Repeats | Pass Rate | Generation Cost | Judge Cost | Total Cost | Mean Latency | P95 Latency | Std Latency | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **System_A** (Always-Large) | `anthropic/claude-3.7-sonnet` | 2 | 90.0% | $0.082920 | $0.001260 | $0.084180 | 1420.8 ms | 2150.0 ms | 265.4 ms | `[FIXTURE]` |
| **System_Small** (Always-Small) | `google/gemini-2.5-flash` | 2 | 70.0% | $0.001338 | $0.001260 | $0.002598 | 684.2 ms | 920.0 ms | 118.5 ms | `[FIXTURE]` |
| **System_C** (Constraint-Aware) | Multi-tier pipeline | 2 | 85.0% | $0.013680 | $0.001260 | $0.014940 | 342.5 ms | 1210.0 ms | 385.2 ms | `[FIXTURE]` |

### Provenance & Safeguards
- **Fixture Mode**: Active (`is_fixture: true`, `live_network_calls: false`)
- **Pricing Schedule**: `2026-10-01`
- **Provider**: `openrouter`
- **Storage Location**: `data/results/fixtures/` (completely isolated from live `data/results/real/`)
