# InferPath — Architecture Specification & Component Lifecycle

**Document Version**: 1.0.0  
**Effective Date**: 2026-10-14  
**Status**: Implemented & Tested  
**Primary Implementation**: `src/inferpath/pipeline.py` (`inferpath.pipeline.RequestPipeline`)

---

## 1. Architectural Philosophy

InferPath is built around a single foundational premise: **an inference router must pick the least resource-intensive processing tier that meets user constraints without sacrificing correctness or leaking private data.**

The architecture is designed as a deterministic 11-step pipeline enforcing strict separation between deterministic local processing (Tiers A & B) and neural language model invocation (Tiers C & D), preceded by exact caching and followed by output validation with bounded fallback.

```mermaid
flowchart TD
    subgraph Client["Client Tier"]
        Req[UserRequest]
        Resp[RespondResponse]
    end

    subgraph PreExecution["Pre-Execution Guardrails"]
        S1[Step 1: Request Schema & Parameter Validation]
        S2[Step 2: Privacy Policy Classification & Interception]
        S3[Step 3: Freshness & Time-Sensitivity Analysis]
        S4[Step 4: Rule-Based Intent & Route Selection]
        S5[Step 5: Canonical SHA-256 Cache Key & Eligibility]
    end

    subgraph Caching["Exact-Response Cache"]
        CacheCheck{Step 5: Cache Eligible?}
        CacheLookup[Step 6: SQLite Exact Cache Lookup]
        CacheStore[(Step 10: SQLite Cache Store)]
    end

    subgraph Execution["Multi-Tier Execution Engine"]
        TierA[Tier A: Safe Math & Unit Conversion]
        TierB[Tier B: Jinja2 Template Slot Filler]
        TierC[Tier C: Small Model Adapter]
        TierD[Tier D: Large Model Adapter]
    end

    subgraph PostExecution["Post-Execution Guardrails"]
        Val[Step 8: Deterministic Output Validation]
        FB{Step 9: Fallback Required?}
        Escalate[Step 9: Bounded Escalation C -> D]
        Telemetry[Step 11: Audit Logging & Telemetry]
    end

    Req --> S1
    S1 --> S2
    S2 -->|Confidential Policy Violation| Block[HTTP 403 Privacy Exception]
    S2 -->|Allowed| S3
    S3 --> S4
    S4 --> S5
    S5 --> CacheCheck
    CacheCheck -->|Eligible| CacheLookup
    CacheCheck -->|Ineligible: Freshness / Deterministic / Private| Execution
    CacheLookup -->|Hit| Telemetry
    CacheLookup -->|Miss| Execution
    Execution --> TierA
    Execution --> TierB
    Execution --> TierC
    Execution --> TierD
    TierA --> Val
    TierB --> Val
    TierC --> Val
    TierD --> Val
    Val --> FB
    FB -->|No| CacheStore
    FB -->|Yes & Tier C| Escalate
    Escalate --> TierD
    CacheStore --> Telemetry
    Telemetry --> Resp
```

---

## 2. Component Breakdown & Class Responsibilities

### 2.1 Pre-Execution Guardrails
1. **Privacy Policy Enforcer** (`inferpath.security.privacy.PrivacyPolicyEnforcer`):
   - Categorizes incoming queries into `public`, `internal`, or `confidential_private`.
   - Strictly intercepts confidential queries before any external network transmission can occur (Rule R01).
   - Enforces zero API key leakage in error traces using `scrub_api_keys()`.
2. **Freshness Detector** (`inferpath.router.analyzer.FreshnessDetector`):
   - Evaluates linguistic cues for temporal volatility (e.g., "current stock price", "weather right now").
   - Distinguishes genuine time-sensitive queries from timeless rhetorical constructs (e.g., "Today we know that the Earth orbits the Sun").
   - Automatically marks volatile queries as cache-ineligible (Rule R02).
3. **Rule-Based Router** (`inferpath.router.router.RuleBasedRouter`):
   - Executes ordered pattern matchers:
     - Rule R03/R04: Exact arithmetic and unit conversions $\to$ Tier A.
     - Rule R05: Structured business templates $\to$ Tier B.
     - Rule R06/R07: JSON schema validation and date formatting $\to$ Tier A.
     - Rule R08: Complex synthesis and proofs $\to$ Tier D.
     - Rule R09: Standard text classification and summarization $\to$ Tier C.
     - Rule R10: Ambiguous or high-quality preference $\to$ Tier D.

### 2.2 Exact-Response Cache Engine
- **Canonical Key Builder** (`inferpath.cache.key_builder.CanonicalKeyBuilder`):
  - Builds canonical JSON with sorted keys containing: `normalized_prompt`, `task_parameters`, `context_hash`, `user_scope`, `selected_route`, `handler_version`, `template_version`, and `privacy_class`.
  - Computes a deterministic SHA-256 digest ensuring identical canonical inputs yield identical keys.
- **Cache Storage** (`inferpath.cache.exact_cache.ExactCache`):
  - Persistent SQLite backing store with TTL management and transactional writes.
  - Skips cache lookups for Tier A (which executes locally in $<0.1$ ms).

### 2.3 Tier Handlers & Model Adapters
- **Tier A Handlers** (`inferpath.handlers`):
  - `SafeCalculator`: AST-parsed arithmetic without Python `eval()`.
  - `UnitConverter`: Dimensional analysis with floating precision.
  - `DateFormatter`: ISO 8601 formatting.
- **Tier B Template Engine** (`inferpath.templates.engine.TemplateEngine`):
  - Jinja2 in-memory slot filler.
  - Emits structured `ClarificationRequest` when required slots are missing, preventing ungrounded generation.
- **Tier C & D Model Adapters** (`inferpath.handlers`):
  - `OpenRouterAdapter`: Asynchronous HTTP client communicating with `https://openrouter.ai/api/v1` using OpenAI-compatible payload structures.
  - `MockSmallModel` and `MockLargeModel`: Hermetic, deterministic offline mock adapters for offline test and benchmark suites.

### 2.4 Output Validation & Fallback State Machine
- **Composite Validator** (`inferpath.validators`):
  - `NonEmptyValidator`: Rejects blank or whitespace-only outputs.
  - `LengthValidator`: Validates token/character length boundaries.
  - `FormatValidator`: Validates JSON parseability or required markdown headings.
  - `RefusalDetector`: Detects canned model refusals (e.g., "I cannot fulfill this request").
- **Bounded Fallback** (`inferpath.pipeline.RequestPipeline`):
  - If Tier C small model output fails validation, automatically escalates to Tier D large model.
  - Maximum fallback depth is strictly bounded to 1 to prevent unbounded latency or billing cycles.

---

## 3. Provenance & Telemetry Invariants

Every metric generated during execution is encapsulated in a `ProvenanceValue[T]` model:
```python
class ProvenanceValue(BaseModel, Generic[T]):
    value: T
    provenance: ProvenanceLabel  # MEASURED, CALCULATED, ESTIMATED, or SIMULATED
    unit: str
    system_boundary: BoundaryDefinition | None = None
    range_bounds: RangeBounds | None = None
```

- **Measured**: Direct physical or wall-clock observations (`latency_ms`, token counts from API headers).
- **Calculated**: Deterministically derived from measured tokens and the dated pricing table (`cost_usd`).
- **Estimated**: Modeled parameters from documented engineering literature (`energy_wh`, `water_ml`).
- **Simulated**: Synthetic metrics produced during mock model execution.
