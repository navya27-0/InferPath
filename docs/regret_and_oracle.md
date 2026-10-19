# Rigorous Routing Regret & Oracle Formulation

## 1. Information Boundaries & Oracle Definition

A major source of ambiguity in LLM router benchmarks is what an "Oracle" knows and when it knows it.

### 1.1 Pre-Execution Information Set ($\mathcal{I}_0$)
Available strictly at query arrival ($t = t_0$):
- Prompt string $x$, token length $|x|$, and detected syntax features.
- Caller-declared preferences: budget $B_{\max}$, latency ceiling $L_{\max}$, minimum quality $Q_{\min}$, and privacy class $P(x)$.
- Static model registry: dated token pricing, context limits, and historical latency/quality priors.
- Local cache index status (hit/miss).

### 1.2 Counterfactual Post-Hoc Oracle Set ($\mathcal{I}_{\text{oracle}}$)
The Oracle operates with **counterfactual omniscience**:
- For every candidate tier $m \in \mathcal{M}$, the Oracle knows the true post-hoc execution outcome:
  $$Y(m, x) = (\text{Success}(m, x), \text{QualityScore}(m, x), \text{ExactTokens}(m, x), \text{LatencyMs}(m, x))$$
- **Oracle Decision Rule**:
  $$m^*_{\text{oracle}} = \arg\min_{m \in \mathcal{M}} \text{Cost}(m, x) \quad \text{s.t.} \quad \begin{cases} \text{QualityScore}(m, x) \ge Q_{\min}(x) \\ \text{PrivacyAllowed}(m, P(x)) \end{cases}$$
- The Oracle represents a theoretical **greatest lower bound (GLB) on cost** among quality-satisfying tiers, not an executable pre-routing policy.

---

## 2. Formal Regret Accounting

### 2.1 Cost Regret ($R_{\text{cost}}$)
$$R_{\text{cost}}(x) = \text{Cost}_{\text{actual}}(x) - \text{Cost}(m^*_{\text{oracle}}, x)$$
where:
$$\text{Cost}_{\text{actual}}(x) = \begin{cases} 0 & \text{if cache hit} \\ \text{Cost}(m_{\text{selected}}, x) + \mathbb{I}(\text{fallback}) \cdot \text{Cost}(m_{\text{fallback}}, x) & \text{otherwise} \end{cases}$$

- **Fallback Accounting**: If the selected tier $m_{\text{selected}}$ fails quality verification, the cost of invoking the escalation model is **fully added** to actual cost.
- **Validation Overhead**: Deterministic verification CPU overhead is factored into pipeline accounting.

### 2.2 Quality Regret ($R_{\text{quality}}$)
$$R_{\text{quality}}(x) = \max\left(0, \;\text{Quality}(m^*_{\text{oracle}}, x) - \text{Quality}_{\text{final}}(x)\right)$$

### 2.3 Escalation Classification Rates
- **Over-Escalation Rate**: The fraction of queries where the router selected a large frontier model when a smaller tier or deterministic handler would have successfully satisfied $Q_{\min}$.
  $$\text{OverEscalation} = \mathbb{I}(m_{\text{selected}} = \text{Large} \;\land\; m^*_{\text{oracle}} \neq \text{Large})$$
- **Under-Escalation Rate**: The fraction of queries where the router selected a small or deterministic tier that failed validation, requiring costly fallback escalation.
  $$\text{UnderEscalation} = \mathbb{I}(m_{\text{selected}} \neq \text{Large} \;\land\; m^*_{\text{oracle}} = \text{Large})$$

---

## 3. Boundary & Edge Case Handling

1. **Unsatisfiable Constraints**:
   If no tier satisfies $Q_{\min}(x)$ within $B_{\max}(x)$, the Oracle selects the maximum-quality privacy-compliant tier, and $R_{\text{cost}} = 0$ while quality shortfall is recorded.
2. **Exact Cache Hits**:
   Cache hits have $\text{Cost} = \$0.00$ and $\text{Latency} \le 2\text{ ms}$. If an exact match exists in cache, the Oracle also receives $\text{Cost} = \$0.00$.
3. **Privacy Invariant**:
   The Oracle is strictly bound by data governance policies: it cannot select external cloud LLMs for `internal` or `confidential` queries, regardless of potential quality gains.
