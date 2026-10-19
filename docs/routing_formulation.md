# Formal Routing Optimization Formulation vs Implementation

## 1. Problem Definition & Mathematical Formulation

InferPath models inference routing as a **two-stage constrained multi-objective optimization problem with execution-time confidence escalation**.

### 1.1 Decision Space
Let $\mathcal{M} = \{m_1, m_2, m_3, m_4\}$ represent the available execution tiers:
- $m_1 = \text{Deterministic Rules / Sympy AST / Regex}$ (`deterministic_a`)
- $m_2 = \text{Strict Template Engine}$ (`template_b`)
- $m_3 = \text{Low-Resource Tier (e.g. Gemini 2.5 Flash / Llama 8B)}$ (`small_model_c`)
- $m_4 = \text{Frontier Large Tier (e.g. Claude 3.7 Sonnet / GPT-4o)}$ (`large_model_d`)

For an incoming request $x \in \mathcal{X}$, the caller declares utility constraints:
- Maximum cost budget: $B_{\max}(x)$
- Maximum latency ceiling: $L_{\max}(x)$
- Minimum acceptable quality threshold: $Q_{\min}(x)$
- Privacy classification: $P(x) \in \{\text{Public}, \text{Internal}, \text{Confidential}\}$

---

## 2. Two-Stage Decision Architecture

```
                  Incoming Request x
                          │
                          ▼
        ┌───────────────────────────────────┐
        │   Stage 1: Hard Feasibility Gate  │
        │   - Privacy boundary filtering    │
        │   - Budget & Latency ceilings     │
        │   - Quality threshold check       │
        └─────────────────┬─────────────────┘
                          │ Feasible Set M_feasible(x)
                          ▼
        ┌───────────────────────────────────┐
        │   Stage 2: Scalarized Optimization│
        │   min J(m, x) = Cost + λ·Lat - β·Q│
        └─────────────────┬─────────────────┘
                          │ Candidate Route m*
                          ▼
        ┌───────────────────────────────────┐
        │   Stage 3: Verification & Fallback│
        │   - Schema / AST verification     │
        │   - Confidence escalation policy  │
        └───────────────────────────────────┘
```

### 2.1 Stage 1: Hard Feasibility Filtering
The set of feasible candidate tiers $\mathcal{M}_{\text{feasible}}(x) \subseteq \mathcal{M}$ is defined as:
$$\mathcal{M}_{\text{feasible}}(x) = \left\{ m \in \mathcal{M} \;\middle|\; \begin{aligned} &\text{PrivacyAllowed}(m, P(x)) \\ \land\; &\widehat{\text{Cost}}(m, x) \le B_{\max}(x) \\ \land\; &\widehat{\text{Latency}}(m, x) \le L_{\max}(x) \\ \land\; &\widehat{Q}(m, x) \ge Q_{\min}(x) \end{aligned} \right\}$$

- **Privacy Invariant**: If $P(x) \neq \text{Public}$, $\mathcal{M}_{\text{feasible}}(x) \subseteq \{m_1, m_2\}$. External API calls to commercial endpoints are strictly forbidden.
- **Feasibility Guarantee**: If $\mathcal{M}_{\text{feasible}}(x) = \emptyset$, InferPath falls back to the highest-capability privacy-compliant tier with an audit notification.

### 2.2 Stage 2: Scalarized Objective Minimization
Over the feasible set $\mathcal{M}_{\text{feasible}}(x)$, the router computes a scalarized objective:
$$m^* = \arg\min_{m \in \mathcal{M}_{\text{feasible}}(x)} J(m, x)$$
where:
$$J(m, x) = \widehat{\text{Cost}}(m, x) + \lambda \cdot \widehat{\text{Latency}}(m, x) - \beta \cdot \widehat{Q}(m, x)$$

- **$\lambda$ Selection**: $\lambda$ represents the caller's marginal willingness to pay for latency reduction ($\$ / \text{ms}$).
  Default: $\lambda = 1.0 \times 10^{-4} \;\$/\text{ms}$ ($10\text{ cents per second of latency saved}$).
- **$\beta$ Selection**: $\beta$ scales the quality prior penalty ($\beta = 0.005$).

### 2.3 Stage 3: Verification & Confidence Escalation
If candidate $m^*$ produces an output that fails deterministic schema parsing or triggers a confidence alarm ($\text{Conf}(m^*, x) < \theta$), the pipeline initiates **confidence escalation**:
$$m_{\text{escalate}} = \arg\max_{m \in \mathcal{M}_{\text{feasible}}(x)} \widehat{Q}(m, x)$$

---

## 3. Explicit Implementation Mapping

| Theoretical Concept | Mathematical Symbol | Code Implementation |
| :--- | :---: | :--- |
| Feasible Tier Candidate Set | $\mathcal{M}_{\text{feasible}}(x)$ | `ConstrainedOptimizationPolicy.determine_feasible_set()` |
| Pre-execution Cost Estimate | $\widehat{\text{Cost}}(m, x)$ | `TierPerformancePrior.estimated_cost_per_1k` |
| Pre-execution Latency Estimate | $\widehat{\text{Latency}}(m, x)$ | `TierPerformancePrior.estimated_latency_ms` |
| Quality Prior | $\widehat{Q}(m, x)$ | `TierPerformancePrior.quality_prior` |
| Tradeoff Parameter | $\lambda$ | `ConstrainedOptimizationPolicy.lambda_tradeoff` |
| Scalarized Objective Loss | $J(m, x)$ | `ConstrainedOptimizationPolicy.compute_objective_score()` |
| Escalation Fallback Policy | $m_{\text{escalate}}$ | `RequestPipeline._handle_failure()` / `ConfidenceEscalationRouter` |

### Answers to Methodological Review Questions:
1. **Which parts are true optimization?**
   Stage 2 explicitly evaluates the scalarized objective $J(m, x)$ across the feasible candidates and selects the cost-optimal tier.
2. **Which parts are rule-based filtering?**
   Privacy boundary checks, hard budget boundaries, and hard latency ceilings act as zero-tolerance feasibility cuts before scoring.
3. **Are costs and latencies predicted prior to execution?**
   Yes. Estimates are calculated from tokenizer length heuristics and dated pricing tables (`configs/pricing_table_2026_10.yaml`).
4. **Is the selected tier globally optimal?**
   The decision is globally optimal within the constrained hypothesis space $\mathcal{M}_{\text{feasible}}(x)$ with respect to scalarized objective $J(m, x)$.
