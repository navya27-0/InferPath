# Model Registry & Provider Drift Evaluation

## 1. Motivation: Maintaining Routing Viability Over Time

Static model routing rules degrade rapidly as commercial AI providers update models, drop prices, or introduce new tiers.

To prevent benchmark obsolescence, InferPath decouples the routing engine from hard-coded model identifiers by utilizing a **dynamic Model Registry** ([`src/inferpath/models/registry.py`](../src/inferpath/models/registry.py)) that tracks:
- Provider, model ID, version tag, release date, and pricing date.
- Context windows and capability vectors (reasoning, math, coding, multilingual).
- Calibrated empirical latency (p50, p95) and quality prior distributions.
- Model lifecycle status (active, deprecated, replacement redirection).

---

## 2. Market Drift Scenarios & Empirical Adaptation

Artifact: [`data/results/drift_evaluation.json`](../data/results/drift_evaluation.json)

The system was evaluated under five realistic market drift scenarios:

| Scenario | Market Event | Router Adaptation | Cost Impact |
| :--- | :--- | :--- | :---: |
| **Baseline** | Nominal October 2026 commercial pricing | Tier C preferred for routine tasks | 84.9% savings vs Always-Large |
| **A: Price War** | 50% price reduction on Tier C (Flash) | Small model score improves; increases routine share | 92.4% savings vs Always-Large |
| **B: Inflation** | 35% price increase on Tier D frontier models | Escalation threshold becomes more selective | 88.8% savings vs Always-Large |
| **C: Congestion** | 2.5× latency degradation on Tier C | Router dynamically routes SLA-critical tasks to Tier A/D | Preserves user latency limits |
| **D: Capability Leap**| Small model v2 quality jumps from 0.72 to 0.89 | Router routes 85%+ of general tasks to small model | 94.2% savings vs Always-Large |
| **E: Deprecation**| Legacy model ID deprecated by provider | Automatic pointer redirection to successor endpoint | Zero client-side downtime |

### Key Architectural Benefit
By decoupling routing decisions into a scalarized objective $J(m, x)$ evaluated over registry priors, operators can update prices or swap models simply by updating YAML configurations or registry entries, without rewriting routing heuristics.
