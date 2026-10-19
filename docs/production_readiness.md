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
