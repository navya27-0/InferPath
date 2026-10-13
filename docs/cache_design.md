# InferPath Exact Cache Engine Design Specification

## 1. Overview & Architectural Role

The **InferPath Exact Cache Engine** sits between Stage 3 (Routing Decision) and Stage 5 (Handler / Model Execution) of the 9-stage inference routing pipeline. Its goal is to eliminate redundant compute, latency, and cost for previously validated deterministic, template, or model outputs without risking stale, invalid, or private data leakage.

```
Request ──> [Stage 1: Privacy] ──> [Stage 2: Freshness] ──> [Stage 3: Route Selection]
                                                                     │
               ┌─────────────────────────────────────────────────────┘
               ▼
[Stage 4: Cache Key Build & Lookup] ──(Hit)──> Bypass execution ──> [Stage 9: Log]
               │ (Miss / Bypassed)
               ▼
[Stage 5: Execute Handler / Model]
               │
[Stage 6: Output Validation]
               │
[Stage 7: Fallback if needed]
               │
[Stage 8: Cache Storage (if eligible & validated)]
               │
[Stage 9: Telemetry & Audit Log]
```

---

## 2. Canonical Cache-Key Structure & Rationale

The cache key is computed as a **SHA-256 hex digest** over a canonical JSON dictionary. Because routing dictates the execution tier and version, the key is constructed **after route selection** in Stage 4.

### 2.1 Canonical Payload Fields

| Field Name | Type | Description & Design Rationale |
| :--- | :--- | :--- |
| `normalized_prompt` | `string` | The conservatively normalized prompt text. Prevents redundant misses from whitespace or Unicode differences. |
| `task_parameters` | `object` | Explicit task parameters: `audience`, `expected_format`, `language`, `max_latency_ms`, `max_tokens`, `quality_preference`. Guarantees queries with identical prompts but different target formats (e.g., text vs JSON) or constraints produce distinct keys. |
| `context_hash` | `string` | SHA-256 digest of additional context/documents. Ensures same prompt against different documents does not hit identical cached answers. |
| `user_scope` | `string` | Tenant, organization, or user permission boundary. Prevents cross-tenant or cross-role cache contamination. |
| `route` | `string` | The selected processing tier (`template_b`, `small_model_c`, `large_model_d`). Ensures responses are tied to the chosen tier. |
| `handler_version` | `string` | The version string of the executing model or handler. Invalidates cached answers when model weights or handler logic change. |
| `template_version` | `string` | Template version identifier for Tier B requests. Invalidates cache when template templates are updated. |
| `privacy_class` | `string` | Data classification level (`public`, `internal`, `confidential_private`). Invariant: only `public` requests ever enter cache lookup or storage. |

### 2.2 Serialization Invariants
* Keys are sorted lexicographically (`sort_keys=True`).
* Compact separators are enforced (`,`, `:` with zero extra whitespace).
* Strict UTF-8 character encoding with `ensure_ascii=False`.
* Resulting JSON is hashed via `hashlib.sha256()`.

---

## 3. Conservative Normalization Rules

Prompt normalization is strictly conservative to avoid semantic drift or accidental collision:

1. **Unicode NFC Normalization**: Converts all Unicode input to Normalization Form C (`unicodedata.normalize("NFC", text)`), resolving precomposed vs decomposed character differences (e.g., `café`).
2. **Standardized Linebreaks**: Normalizes Windows (`\r\n`) and legacy Mac (`\r`) linebreaks into standard Unix linebreaks (`\n`).
3. **Inconsequential Horizontal Whitespace Collapse**: Sequences of multiple spaces or tabs (`[ \t]+`) within a line are collapsed into a single space.
4. **Leading & Trailing Whitespace Stripping**: Trims leading and trailing whitespace per line and overall string.

### Non-Negotiable Normalization Safeguards
* **No Indiscriminate Lowercasing**: Casing is strictly preserved. Lowercasing can alter acronyms, proper nouns, and code snippets.
* **No Punctuation Stripping**: Mathematical operators, commas, periods, and quotation marks are strictly preserved.
* **No Number Alteration**: Numeric constraints (`"summarize in 50 words"` vs `"summarize in 500 words"`) remain distinct strings.

---

## 4. Cache Eligibility & Freshness Policies

Cache eligibility is managed by `CacheEligibilityPolicy` with explicit rejection codes:

| Condition | Policy Action | Policy Code | Rationale |
| :--- | :--- | :--- | :--- |
| `privacy_class != public` | **Bypass Lookup & Storage** | `BYPASS_PRIVATE` | Confidential or private data must never enter persistent cache tables. |
| `is_time_sensitive == True` | **Bypass Lookup & Storage** | `BYPASS_TIME_SENSITIVE` | Explicit or detected temporal queries must never serve stale answers. |
| Temporal keywords detected | **Bypass Lookup & Storage** | `BYPASS_TIME_SENSITIVE` | Rule-based detection of tokens like `today`, `now`, `weather`, `stock price`, `latest`. |
| `route == deterministic_a` | **Bypass Lookup** | `BYPASS_DETERMINISTIC` | Direct math/API execution is cheaper and faster than SQLite cache lookup overhead. |
| `output.needs_clarification` | **Bypass Storage** | `BYPASS_CLARIFICATION` | Clarification requests indicate incomplete inputs, not final answers. |
| `not output.validated` | **Bypass Storage** | `BYPASS_UNVALIDATED` | Output must pass validation before storage. |
| `validation_result.is_valid == False` | **Bypass Storage** | `BYPASS_INVALID` | Defective outputs (malformed schema, empty, etc.) are never persisted. |

---

## 5. Storage Schema & Lifecycle Semantics

Cache entries are persisted in SQLite within the `exact_cache` table:

```sql
CREATE TABLE IF NOT EXISTS exact_cache (
    cache_key TEXT PRIMARY KEY,
    normalized_prompt TEXT NOT NULL,
    output_text TEXT NOT NULL,
    tier TEXT NOT NULL,
    response_payload TEXT NOT NULL,
    requirements_snapshot TEXT NOT NULL,
    version TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exact_cache_expires ON exact_cache(expires_at);
```

### 5.1 Lookup & Hit Semantics
1. Exact key lookup: `SELECT response_payload, expires_at FROM exact_cache WHERE cache_key = ?`.
2. Expiration check: If `current_utc_time >= expires_at`, the entry is treated strictly as a **cache miss** (`return None`). An expired entry is **never returned**.
3. Hit count tracking: When a valid hit occurs, `hit_count` is incremented atomically (`UPDATE exact_cache SET hit_count = hit_count + 1 WHERE cache_key = ?`).
4. Safe deserialization: The stored JSON payload is deserialized into `HandlerOutput` with `cache_hit = True`. If corrupted, the error is caught and returns `None` safely.

### 5.2 TTL & Purge Semantics
* Route-specific TTLs are configured in `configs/settings.yaml` under `routing.cache_ttls`:
  * `deterministic_a`: 0 seconds (never stored).
  * `template_b`: 86,400 seconds (1 day).
  * `small_model_c`: 86,400 seconds (1 day).
  * `large_model_d`: 86,400 seconds (1 day).
* Purge routine: `ExactCache.purge_expired()` executes `DELETE FROM exact_cache WHERE expires_at <= CURRENT_TIMESTAMP`, returning the count of deleted rows.

---

## 6. Privacy & Failure Handling

* **Zero Secret Leakage**: Parameterized queries prevent SQL injection. Diagnostic tools (`get_entry_metadata`) scrub any API key patterns.
* **No Private Prompts in Cache**: Requests marked `internal` or `confidential_private` are rejected before key construction or storage.
* **Graceful Degradation**: If SQLite encounters disk locks or corrupted JSON records, the cache fails open (returns `None`), allowing the router pipeline to continue without crashing.

---

## 7. Known Limitations

1. **Exact-Match Only**: This engine does not perform semantic or embedding-based approximate matching. Slight phrasing variations that alter non-whitespace tokens result in a cache miss.
2. **Local SQLite Bound**: SQLite is single-node; distributed multi-process scaling requires WAL mode and per-process connections.
3. **No Measured Performance Claims**: Resource and latency savings cannot be claimed until empirical benchmarks (cold vs warm cache) are executed in Phase 5.
