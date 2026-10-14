# InferPath — Real-Provider vs. Simulated Evaluation Analysis

**Document Version**: 1.0.0  
**Effective Date**: 2026-10-14  
**Status**: Research Methodology & Gap Audit  
**Scope**: Empirical divergence between offline deterministic simulation and live API telemetry

---

## 1. Executive Summary & Purpose

A critical vulnerability in contemporary AI inference routing research is the conflation of **offline simulated benchmarks** with **live multi-tenant cloud execution**. In simulated benchmarks, mock models produce instant responses with deterministic token lengths and idealized zero-latency network channels. In production cloud environments, requests face non-deterministic wide-area network (WAN) transport jitter, provider-side GPU batching queues, variable token generation dynamics, and transient upstream outages (HTTP 429, 502, 503).

This document provides a rigorous, transparent comparative analysis between InferPath's offline hermetic simulated benchmark suite and live OpenRouter execution.

---

## 2. Comparative Matrix: Simulation vs. Real-Provider Execution

| Dimension | Offline Simulated Benchmark (`adapter_mode: simulated`) | Live OpenRouter Execution (`adapter_mode: real`) | Impact on Research Findings |
| :--- | :--- | :--- | :--- |
| **Model Backends** | `MockSmallModel`, `MockLargeModel` (deterministic string returns) | `google/gemini-2.5-flash`, `anthropic/claude-3.7-sonnet` via OpenRouter | Live generation has stochastic linguistic variety and nuanced semantic nuance. |
| **Latency Characteristics** | Local Python pipeline execution overhead only (**15–90 ms**) | WAN TLS handshake + Provider server queue + Autoregressive token decoding (**300–2500 ms**) | Real-world latency savings from Tier A (local math $<1$ ms) and Tier B (templates $<2$ ms) are orders of magnitude more dramatic in live environments ($>100\times$). |
| **Token Usage** | Approximate deterministic token counts calculated from text length | Genuine provider-reported prompt & completion tokens via response headers | Live tokens fluctuate by temperature and prompt structure; pricing calculations match actual bills. |
| **Economic Accounting** | Static snapshot pricing table (`pricing_table_2026_10.yaml`) | Dynamic provider billing matching exact OpenRouter credit deductions | Preserves reproducibility against provider price changes while verifying billing accuracy. |
| **Failure Modes** | Synthetic injected errors via reproducible random seed | Real-world HTTP 429 (rate limits), 502/503 (upstream downtime), and timeouts | Requires production resilience: circuit breakers, bounded retries, and rate limiters. |
| **Data Privacy** | Local memory evaluation, zero external socket transmission | External TLS transmission for public queries; Stage 1 local interception for private queries | Stage 1 privacy enforcer guarantees zero bytes of confidential data reach external networks. |

---

## 3. The Real-World Latency Paradox

In simulated testing, the relative latency difference between Tier C (Small Model) and Tier D (Large Model) is modest (approx. 27 ms vs 89 ms) because mock models do not perform matrix multiplications or token-by-token autoregression.

In live production deployments:
1. **Time-to-First-Token (TTFT)**: Frontier models (~70B+) typically have TTFT of 400–800 ms, whereas lightweight models (~8B) deliver TTFT of 120–250 ms.
2. **Generation Throughput**: 8B models generate tokens at 80–140 tokens/sec; 70B models generate at 25–45 tokens/sec.
3. **Deterministic Handlers (Tier A & B)**: Local execution requires $<1$ ms and zero external network transit. For arithmetic, unit conversions, and templated emails, bypassing the LLM yields a **$500\times$ latency reduction** ($2$ ms vs $1000$ ms) that cannot be fully captured in mock-only benchmarks.

---

## 4. Production Fault Tolerance: The Circuit Breaker Invariant

In live cloud environments, external inference providers inevitably degrade or experience outages. Without circuit breakers, a multi-tier router can cascade failures across services.

InferPath implements a 3-state **`CircuitBreaker`** state machine wrapped around the `OpenRouterAdapter`:
- **`CLOSED`**: Standard operation. All requests pass to OpenRouter.
- **`OPEN`**: Tripped after 3 consecutive failures (HTTP 429, 5xx, or network timeout). Immediately rejects further external requests with `CircuitBreakerOpenError` during a 60-second recovery window, preventing request queue buildup and thread exhaustion.
- **`HALF_OPEN`**: After recovery expiration, allows a single probe request to test provider recovery. If successful, transitions to `CLOSED`; if failed, resets the 60-second backoff timer.

---

## 5. Live-Provider Execution Protocol

To execute a genuine live-provider benchmark on a curated sample:

```bash
# 1. Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-..."      # macOS / Linux
$env:OPENROUTER_API_KEY = "sk-or-v1-..."     # Windows PowerShell

# 2. Run the curated real-provider benchmark (20 items, $2.00 safety budget ceiling)
python scripts/run_real_evaluation.py --sample-size 20 --max-budget 2.0
```

All outputs and telemetry records are saved strictly with `[MEASURED]` provenance tags to `data/results/real/`.
