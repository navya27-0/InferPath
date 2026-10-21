# InferPath Operational Architecture & Production Readiness Boundaries

To prevent reviewers or operators from conflating research simulation with enterprise production infrastructure, InferPath explicitly demarcates **three distinct operational modes**:

```
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│   1. Research Prototype │      │ 2. Demonstration Deploy │      │ 3. Production Candidate │
│                         │      │                         │      │                         │
│ • Deterministic Sim     │ ───► │ • FastAPI REST API      │ ───► │ • Distributed Redis     │
│ • Offline Dev/Test Sets │      │ • Streamlit Playground  │      │ • OpenTelemetry Tracing │
│ • Local SQLite Cache    │      │ • Docker Compose        │      │ • Vault / GCP Secrets   │
│ • Mock Adapters         │      │ • Bearer Auth Tokens    │      │ • Provider Failover     │
│ • Static Rule Engine    │      │ • In-Memory Rate Limit  │      │ • Production SLOs       │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

---

## 1. The Three Operational Modes

### Mode 1: Research Prototype (Evaluation & Ablation)
- **Primary Objective**: Controlled, reproducible, deterministic evaluation of multi-tier routing policies.
- **Model Execution**: Mock adapters with calibrated difficulty solvers and token delays (`adapter_mode: simulated`).
- **Data Persistence**: Local SQLite exact cache (`data/storage/cache.db`).
- **Rate Limiting**: Single-process in-memory token bucket.
- **Privacy Enforcement**: Heuristic regex and keyword token matching (defense-in-depth).
- **Environment**: Single process, local terminal, hermetic offline CI.

### Mode 2: Demonstration Deployment (Interactive Testing & Staging)
- **Primary Objective**: Developer evaluation, interactive prompting, and prototype integration.
- **Service Interfaces**:
  - FastAPI HTTP server ([`src/inferpath/api/main.py`](../src/inferpath/api/main.py)).
  - Streamlit dashboard UI ([`src/inferpath/dashboard/`](../src/inferpath/dashboard/)).
- **Containerization**: Multi-container stack via `docker-compose.yml`.
- **Authentication**: Bearer API tokens verified against environment configuration.
- **Monitoring**: Local `/health` and `/metrics` telemetry endpoints.

### Mode 3: Production Candidate (Enterprise Infrastructure)
- **Primary Objective**: Highly available, horizontally scalable multi-tenant production gateway.
- **Requirements & Boundaries**:
  - **Distributed Caching**: Redis-backed distributed cache replacing local SQLite to prevent multi-worker lock contention.
  - **Distributed Rate Limiting**: Redis token-bucket limiter across horizontal replicas ([`src/inferpath/services/rate_limiter.py`](../src/inferpath/services/rate_limiter.py)).
  - **Secret Management**: External secret management (AWS Secrets Manager, Google Cloud Secret Manager, or HashiCorp Vault) via [`SecretManagerProvider`](../src/inferpath/security/secret_manager.py).
  - **Observability**: Structured JSON logging and OpenTelemetry-compliant trace propagation ([`src/inferpath/telemetry/`](../src/inferpath/telemetry/)).
  - **Provider Failover & Circuit Breaking**: Automatic retry with exponential backoff and circuit breaking (`CircuitBreakerState`) across primary and fallback endpoints.
  - **Deployment Profiles**: Explicit configuration enforcement (`development`, `staging`, `production`) in [`src/inferpath/config/deployment_profiles.py`](../src/inferpath/config/deployment_profiles.py).

---

## 2. Feature Matrix across Operational Modes

| Component / Subsystem | Mode 1: Research Prototype | Mode 2: Demonstration | Mode 3: Production Candidate |
| :--- | :--- | :--- | :--- |
| **Model Invocations** | Deterministic mock adapter | OpenRouter API / Live | Multi-provider failover pool |
| **Cache Store** | SQLite on disk | SQLite / Memory | Distributed Redis cluster |
| **Rate Limiter** | In-memory token bucket | In-memory token bucket | Distributed Redis limiter |
| **Secret Loading** | Plain environment vars | Environment / `.env` | Secret Manager SDK |
| **Telemetry** | Local JSON experiment log | Console / `/metrics` | OTel Collector / Datadog |
| **Privacy Layer** | Regex & pattern detector | Pattern detector | Defense-in-depth + enterprise DLP |
| **Scale Target** | 1 process ($N \le 25$) | Docker container ($N \le 50$) | Multi-replica K8s ($N \ge 500$) |

---

## 3. Operational Hardening & Production Pre-Flight Checklist

When transitioning from Mode 2 (Demonstration) to Mode 3 (Production Candidate), operators must verify the following security and infrastructure configurations:

### 3.1 Multi-Worker Distributed Rate Limiting
Under multi-worker Uvicorn configurations (`--workers 4`) or multi-replica Kubernetes deployments, independent in-memory rate limiters permit per-worker request bursts that multiply the effective traffic threshold by worker count.

To enforce global cluster-wide rate limits:
1. Provide a managed Redis connection string via `INFERPATH_REDIS_URL` (e.g. `redis://:auth_token@redis-cluster.internal:6379/0`).
2. Enforce Redis strict mode by setting `INFERPATH_REQUIRE_REDIS=true`. This ensures the application halts during startup if Redis cannot be reached, rather than silently degrading to local in-memory limiting.
3. Configure fail-closed semantics with `INFERPATH_RATE_LIMIT_FAIL_CLOSED=true` to reject incoming traffic with HTTP 429 when the Redis backend experiences transient outages.

```bash
export INFERPATH_PROFILE=production
export INFERPATH_REDIS_URL=redis://:strong_password@redis-master.infra:6379/0
export INFERPATH_REQUIRE_REDIS=true
export INFERPATH_RATE_LIMIT_FAIL_CLOSED=true
```

### 3.2 Secret Management & Authentication Verification
- **API Key Requirement**: Production profile (`ENVIRONMENT=production` or `INFERPATH_PROFILE=production`) halts startup immediately if `INFERPATH_API_KEY` is empty. Never deploy production endpoints without setting a strong secret key.
- **Provider API Keys**: When running in real-provider mode without mock fallback (`allow_mock_fallback=False`), `OPENROUTER_API_KEY` must be configured.
- **Secret Manager Integration**: Configure cloud secret providers using [`GLOBAL_SECRET_MANAGER.register_provider()`](../src/inferpath/security/secret_manager.py) to resolve secrets dynamically from AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault rather than static plaintext files.

### 3.3 Network Interface Binding Safeguards
- **Loopback Default**: InferPath defaults to `127.0.0.1` (`INFERPATH_HOST_BIND=127.0.0.1`).
- **Fail-Closed Binding Guard**: Binding to public interfaces (`0.0.0.0` or any non-loopback IP) is strictly rejected by the application startup guard under unauthenticated profiles (`local-demo`). Public exposure requires an authenticated profile (`production` or `staging`) and an explicit `INFERPATH_API_KEY`.
- **Authoritative CLI Check**: The startup guard validates both environment variables and CLI arguments (`--host 0.0.0.0`), preventing configuration mismatch bypasses.
- **Reverse Proxy Headers**: When deploying behind an ingress proxy (e.g., NGINX, Envoy, Cloudflare), configure trusted proxy CIDRs via `INFERPATH_TRUSTED_PROXIES` (e.g. `10.0.0.0/8,172.16.0.0/12`) so `X-Forwarded-For` client IP addresses are validated accurately for rate limiting.

### 3.4 Diagnostic Telemetry Protection
- **Readiness Probes**: `/health/live` and `/health/ready` serve as unauthenticated Kubernetes probes returning minimal liveness/readiness signals (`status`, `service`, `version`). Detailed internals (active profile, rate limiter status) are restricted to authenticated operators.
- **Diagnostics & Metrics**: `/health/dependencies` and `/metrics` require valid API key authentication (`X-API-Key` or `Authorization: Bearer <key>`) under protected deployment profiles.
