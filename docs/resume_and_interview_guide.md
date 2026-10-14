# InferPath — Resume, Portfolio & Interview Guide

This guide provides precise, high-impact resume phrasing, STAR-method interview narratives, and deep technical talking points grounded in the actual codebase and empirical findings.

---

## 1. Resume Bullet Points (Tailored by Role)

### Option A: AI Systems & Inference Infrastructure Engineer (Recommended)
- **Engineered InferPath**, an asynchronous multi-tier AI inference router (FastAPI, Python 3.11, aiosqlite) that dynamically cascades user requests across 4 execution tiers: deterministic AST math ($<1$ ms), Jinja2 templates, lightweight models (~8B), and frontier LLMs (~70B+).
- **Formulated a constrained multi-objective optimizer** ($\min \text{Cost} + \lambda \cdot \text{Latency}$ s.t. $\hat{Q} \ge Q_{\min}$) coupled with an ex-post **Oracle baseline**, achieving **84.9% cost reduction** and **66.7% latency reduction** while bounding quality loss to 6.3% on public tasks.
- **Architected enterprise guardrails**: Zero-leakage Stage 1 privacy enforcer (HTTP 403), stateful **Circuit Breaker** (CLOSED/OPEN/HALF_OPEN), sliding-window rate limiter (120 req/min), canonical SHA-256 exact-response cache, and native Prometheus `/metrics` exposition.
- **Built an empirical research evaluation harness** benchmarking 5 routing architectures and 5 ablations across 160 balanced tasks with clustered bootstrap confidence intervals and quadratic Cohen's kappa inter-rater validation ($\kappa = 0.888$).

### Option B: ML Systems & Backend Platform Engineer
- Developed a high-throughput AI gateway and routing framework processing 46+ concurrent requests per second with $<20$ ms median latency and 100% reliability under concurrent stress testing.
- Integrated OpenRouter API adapters with non-blocking asynchronous I/O, bounded single-level fallback (Tier C $\to$ Tier D), and AST-based deterministic execution for arithmetic, unit conversions, and date formatting.
- Designed a non-blocking canonical exact-response cache backed by `aiosqlite`, achieving a 59.4% warm hit rate and zero inappropriate cache hits for volatile or private queries.
- Deployed a full-stack research dashboard in Streamlit visualizing real-time routing regret, Pareto efficiency frontiers, failure taxonomy, and Scope 2 environmental footprint estimations.

### Option C: Research Engineer / Applied AI Scientist
- Designed and benchmarked a resource-aware AI inference cascade investigating trade-offs between frontier model capabilities and resource consumption under formal privacy and latency constraints.
- Established the empirical Pareto frontier across rule-based, constraint-aware, machine-learned (RandomForest), and ex-post Oracle routing strategies.
- Enforced strict statistical integrity: clustered bootstrap confidence intervals on semantic task families, Cliff's delta effect sizes ($d = -1.00$), and blind human-vs-judge rating calibration ($\kappa = 0.952$).

---

## 2. STAR-Method Interview Narratives

### Narrative 1: The Core Architecture & Trade-Offs
- **Situation**: Modern AI products routinely send all user queries directly to expensive frontier models (GPT-4, Claude 3.7 Sonnet) regardless of whether the prompt is complex reasoning or simple unit conversion, resulting in 80%+ unnecessary expense, latency overhead, and datacenter power draw.
- **Task**: Design an autonomous routing framework that directs each query to the least resource-intensive processing tier while strictly meeting privacy, latency, and minimum quality constraints.
- **Action**: I formulated routing as a constrained optimization problem and built an 11-step pipeline in Python/FastAPI:
  1. Input normalization and Stage 1 privacy enforcement (intercepting internal/confidential data with HTTP 403).
  2. Freshness detection to bypass caching on time-sensitive queries.
  3. Pluggable routing strategies (Rule-based, Constraint-Aware, Learned Random Forest, and Oracle).
  4. Non-blocking SQLite exact-response cache keyed by canonical SHA-256 digests.
  5. Deterministic output validation with bounded single-level fallback from small model (Tier C) to large model (Tier D).
  6. Comprehensive telemetry encapsulating every metric in typed `ProvenanceValue` wrappers (`[MEASURED]`, `[CALCULATED]`, `[ESTIMATED]`, `[SIMULATED]`).
- **Result**: In our 3-repeat empirical benchmark across 160 heterogeneous prompts, InferPath achieved an **84.9% cost reduction** and **66.7% latency reduction** with a 76.5% quality pass rate on public tasks, matching the empirical Pareto-optimal frontier.

---

### Narrative 2: Engineering Reliability & Fault Tolerance
- **Question**: *"How did you handle external provider instability and system bottlenecks?"*
- **Answer**: 
  "Third-party inference APIs are notorious for transient rate limits (HTTP 429) and upstream server timeouts. I engineered three layers of defense:
  1. **Circuit Breaker Pattern**: Wrapped around the OpenRouter client with a 3-state machine (`CLOSED`, `OPEN`, `HALF_OPEN`). If 3 consecutive requests fail, the circuit trips open for 60 seconds, failing fast with `CircuitBreakerOpenError` instead of letting worker threads pile up.
  2. **Sliding-Window Rate Limiting**: Implemented an in-memory sliding-window limiter (120 req/min per IP) to guard our FastAPI gateway against burst spikes.
  3. **Concurrency Benchmarking**: Profiled the ASGI stack under concurrent loads from 1 to 20 workers, verifying 46+ RPS throughput with $<20$ ms median latency and 100% reliability."

---

## 3. Tough Interview Questions & Precise Answers

### Q1: *"Why did you use exact canonical hashing instead of semantic embedding similarity for the cache?"*
> **Answer**:  
> "Semantic similarity caching using embeddings is popular, but in high-stakes enterprise applications it introduces dangerous failure modes: false positive cache hits where subtle prompt differences (e.g., 'not guilty' vs 'guilty', or differing date ranges) return cached answers with high cosine similarity but inverted semantic meaning.  
> InferPath's canonical exact cache sorts JSON keys, includes task parameters, user scope, and prompt text, hashing the normalized structure with SHA-256. This guarantees zero inappropriate cache hits (`inappropriate_hits = 0`), sub-millisecond lookup latency, and deterministic auditability."

### Q2: *"Why is overall quality pass rate 63.1% in the benchmark table if System A is 69.4%?"*
> **Answer**:  
> "This reflects our strict adherence to scientific honesty and security invariants:  
> Our 160-item development dataset contains 28 items (17.5%) classified as `internal` or `confidential`. InferPath's Stage 1 privacy policy unconditionally intercepts these requests with an HTTP 403 error before any model call can occur.  
> Under standard end-to-end task completion metrics, privacy interception counts as non-generation, bounding the maximum possible score. On the 132 public items, InferPath achieves a **76.5% quality pass rate** (compared to System A's 84.1%), representing an acceptable 7.6% quality trade-off in exchange for an **84.9% reduction in inference cost** and **66.7% reduction in latency**."

### Q3: *"What is the difference between your Constraint-Aware Router and the Learned Router?"*
> **Answer**:  
> "The `LearnedRouter` uses a 4-class `RandomForestClassifier` trained on prompt structural features (character length, token count, digit ratios, JSON tokens). While it achieved lower nominal cost ($0.00200 vs $0.00410), it is a statistical model whose decisions can be opaque.  
> The `ConstraintAwareRouter` is an explicit optimizer: it models the Lagrangian objective $\text{Cost}(t) + \lambda \cdot \text{Latency}(t)$ subject to user-defined $Q_{\min}$, $B_{\max}$, and $L_{\max}$ ceilings. In production, this provides deterministic SLA compliance and explainable routing traces, which enterprises value over purely black-box routing."
