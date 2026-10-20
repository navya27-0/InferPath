"""Comprehensive regression test suite for InferPath October 2026 Final Audit Remediations.

Validates all 7 release verification items:
1. P1 - Privacy override verified redaction: confidential content in prompt, context, and metadata
   is never transmitted raw externally.
2. P1 - Pre-dispatch unpriced model validation: unknown models fail closed before any budget
   reservation or HTTP request.
3. P1 - Deployment safety: public host bindings (0.0.0.0) strictly require authentication; Compose
   passes binding environment and protects dashboard.
4. Dashboard protection: Streamlit dashboard cannot be reached by unauthenticated public clients.
5. P2 - Distributed rate limiting fail-closed policy: Redis outages fail closed and expose degradation
   in health telemetry.
6. Legacy SQLite cache auto-migration: constructor automatically migrates older schemas without
   manual intervention and preserves cached entries.
7. Environment configuration: every documented .env.example variable modifies the effective runtime settings.
8. Provider failure paths: timeouts, malformed JSON, 429, and 500 responses safely release budget reservations.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from app.dashboard import check_dashboard_auth
from inferpath.api.main import create_app
from inferpath.cache.exact_cache import CURRENT_SCHEMA_VERSION, ExactCache
from inferpath.config.deployment_profiles import (
    get_deployment_profile,
)
from inferpath.config.settings import PricingTable, load_settings
from inferpath.handlers.openrouter_adapter import (
    CircuitBreaker,
    OpenRouterAdapter,
    OpenRouterProviderError,
    OpenRouterRateLimitError,
    OpenRouterResponseError,
    OpenRouterTimeoutError,
    UnknownModelPricingError,
)
from inferpath.models.domain import HandlerOutput, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.pipeline import RequestPipeline
from inferpath.security.budget import RunBudgetTracker
from inferpath.security.privacy import PrivacyPolicyEnforcer, PrivacyViolationError
from inferpath.services.rate_limiter import (
    DistributedRateLimiter,
    RateLimiterBackendError,
)

# =============================================================================
# 1. P1: Privacy Override Verified Redaction (Prompt, Context, Metadata)
# =============================================================================


class PromptRecordingAdapter:
    """Mock adapter that records the exact prompt and kwargs received during dispatch."""

    def __init__(self, model_id: str = "mock-small"):
        self.model_id = model_id
        self.recorded_prompts: list[str] = []
        self.recorded_kwargs: list[dict] = []

    def generate(self, prompt: str, **kwargs) -> MagicMock:
        self.recorded_prompts.append(prompt)
        self.recorded_kwargs.append(kwargs)
        res = MagicMock()
        res.text = "Model completion for: " + prompt
        res.input_tokens = 10
        res.output_tokens = 20
        res.latency_ms = 45.0
        res.cost_usd_calculated = 0.0001
        res.source = "simulated"
        res.model_id = self.model_id
        return res

    async def generate_async(self, prompt: str, **kwargs) -> MagicMock:
        return self.generate(prompt, **kwargs)


def test_private_request_without_override_strictly_blocked():
    """Confidential request without approved override is blocked from external transmission."""
    pipeline = RequestPipeline(small_adapter=PromptRecordingAdapter())

    req = UserRequest(
        request_id="req_confidential_block",
        prompt="Confidential financial report for Q3 with internal trade_secret.",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        allow_external_transmission=False,
    )

    with pytest.raises(PrivacyViolationError, match="strictly prohibits external transmission"):
        pipeline._execute_tier(req, MagicMock(), RouteTier.SMALL_MODEL_C)


def test_private_request_with_override_transmits_only_redacted_prompt_context_and_metadata():
    """When override is enabled, prompt, context, and metadata are verified and redacted.

    No raw sensitive tokens can reach external model adapters.
    """
    enforcer = PrivacyPolicyEnforcer()
    enforcer.approved_exceptions["allow_redacted_transmission_override"] = True

    mock_adapter = PromptRecordingAdapter()
    pipeline = RequestPipeline(small_adapter=mock_adapter)
    pipeline.privacy_enforcer = enforcer

    secret_raw = "sk-supersecretkey123456789012345"
    ssn_raw = "123-45-6789"
    bearer_raw = "Bearer topsecretbearertoken123456"

    raw_prompt = (
        f"Confidential analysis for account: {secret_raw} SSN: {ssn_raw} with s e c r e t info."
    )
    raw_context = f"Internal user context with auth: {bearer_raw}"
    raw_metadata = {"api_key_ref": secret_raw, "system_id": "sys_internal"}

    req = UserRequest(
        request_id="req_override_tx",
        prompt=raw_prompt,
        context=raw_context,
        metadata=raw_metadata,
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        allow_external_transmission=True,
    )

    # Execute tier
    pipeline._execute_tier(req, MagicMock(), RouteTier.SMALL_MODEL_C)

    # Verify adapter received payload
    assert len(mock_adapter.recorded_prompts) == 1
    received_prompt = mock_adapter.recorded_prompts[0]
    received_kwargs = mock_adapter.recorded_kwargs[0]

    # Invariants: raw secret, raw SSN, and raw bearer must NEVER appear anywhere
    assert secret_raw not in received_prompt
    assert ssn_raw not in received_prompt
    assert "s e c r e t" not in received_prompt

    # Redacted replacement tokens MUST be present in prompt
    assert "[REDACTED_API_KEY]" in received_prompt or "[REDACTED_KEY]" in received_prompt
    assert "[REDACTED_SSN]" in received_prompt
    assert "[REDACTED_SECRET]" in received_prompt

    # Context must also be sanitized
    received_context = received_kwargs.get("context", "")
    assert bearer_raw not in received_context
    assert (
        "[REDACTED_TOKEN]" in received_context
        or "[REDACTED_BEARER]" in received_context
        or "[REDACTED_API_KEY]" in received_context
    )

    # Metadata must also be sanitized
    received_meta = received_kwargs.get("metadata", {})
    assert secret_raw not in received_meta.get("api_key_ref", "")
    assert "[REDACTED" in received_meta.get("api_key_ref", "")

    # Verified clean by policy enforcer
    assert enforcer.verify_redacted(received_prompt) is True
    assert enforcer.verify_redacted(received_context) is True
    assert enforcer.verify_redacted(str(received_meta)) is True


# =============================================================================
# 2. P1: Pre-Dispatch Unpriced Model Validation
# =============================================================================


def test_unpriced_model_rejected_before_http_dispatch():
    """Unpriced model ID is rejected immediately prior to dispatch; HTTP client is NEVER called."""
    table = PricingTable(
        effective_date="2026-10-01",
        source="test",
        rates_per_1m={"meta-llama/llama-3.1-8b-instruct": {"prompt": 0.075, "completion": 0.30}},
    )
    mock_http_client = MagicMock()
    adapter = OpenRouterAdapter(
        model_id="unregistered/unknown-model:v1",
        pricing_table=table,
        client=mock_http_client,
    )

    with pytest.raises(UnknownModelPricingError, match="is unpriced in active pricing table"):
        adapter.generate("Test prompt")

    # Invariant: HTTP client must NEVER be called
    mock_http_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_unpriced_model_rejected_before_async_dispatch():
    """Unpriced model ID is rejected immediately prior to async dispatch."""
    table = PricingTable(
        effective_date="2026-10-01",
        source="test",
        rates_per_1m={"meta-llama/llama-3.1-8b-instruct": {"prompt": 0.075, "completion": 0.30}},
    )
    mock_async_client = AsyncMock()
    adapter = OpenRouterAdapter(
        model_id="unregistered/unknown-model:v1",
        pricing_table=table,
        async_client=mock_async_client,
    )

    with pytest.raises(UnknownModelPricingError, match="is unpriced in active pricing table"):
        await adapter.generate_async("Test prompt")

    mock_async_client.post.assert_not_called()


def test_estimate_call_cost_rejects_unpriced_model():
    """RunBudgetTracker.estimate_call_cost raises UnknownModelPricingError on unpriced models."""
    table = PricingTable(
        effective_date="2026-10-01",
        source="test",
        rates_per_1m={"known-model": {"prompt": 0.10, "completion": 0.20}},
    )
    tracker = RunBudgetTracker(max_usd=1.0, pricing_table=table)

    with pytest.raises(UnknownModelPricingError, match="is unpriced in active pricing table"):
        tracker.estimate_call_cost("unknown-model", "Hello world", max_tokens=50)


def test_openrouter_circuit_breaker_and_timeout_config():
    """OpenRouterAdapter initializes circuit breaker and timeout correctly."""
    table = PricingTable(
        effective_date="2026-10-01",
        source="test",
        rates_per_1m={"test-model": {"prompt": 0.10, "completion": 0.20}},
    )
    cb = CircuitBreaker(failure_threshold=4, recovery_timeout_seconds=30.0)
    adapter = OpenRouterAdapter(
        model_id="test-model",
        pricing_table=table,
        timeout_seconds=45.0,
        max_retries=3,
        circuit_breaker=cb,
    )

    assert adapter.timeout_seconds == 45.0
    assert adapter.max_retries == 3
    assert adapter.circuit_breaker.failure_threshold == 4
    assert adapter.circuit_breaker.recovery_timeout_seconds == 30.0
    assert adapter.circuit_breaker.state == "CLOSED"


# =============================================================================
# 3. P1 & Checklist: Deployment Profiles & Docker Compose Passing
# =============================================================================


def test_unauthenticated_public_interface_binding_rejected(monkeypatch):
    """Binding to 0.0.0.0 without authentication is strictly prohibited and halts startup."""
    monkeypatch.setenv("INFERPATH_HOST_BIND", "0.0.0.0")
    monkeypatch.setenv("INFERPATH_DEPLOYMENT_MODE", "local-demo")
    monkeypatch.delenv("INFERPATH_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError, match="Binding to public interface '0.0.0.0' is strictly prohibited"
    ):
        create_app()


def test_production_environment_with_local_demo_rejected(monkeypatch):
    """ENVIRONMENT=production cannot be paired with unauthenticated local-demo profile."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("INFERPATH_DEPLOYMENT_MODE", "local-demo")

    with pytest.raises(
        ValueError, match="cannot be combined with unauthenticated 'local-demo' profile"
    ):
        get_deployment_profile()


def test_docker_compose_defaults_and_environment_passing():
    """docker-compose.yml must default to production mode, localhost bindings, and pass host bind to containers."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists()

    with open(compose_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. API container environment must receive INFERPATH_HOST_BIND
    api_env = cfg["services"]["api"]["environment"]
    api_env_dict = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in api_env}

    assert api_env_dict["INFERPATH_DEPLOYMENT_MODE"] == "${INFERPATH_DEPLOYMENT_MODE:-production}"
    assert "INFERPATH_HOST_BIND" in api_env_dict
    assert api_env_dict["INFERPATH_HOST_BIND"] == "${INFERPATH_HOST_BIND:-127.0.0.1}"

    # 2. API ports must default to 127.0.0.1
    api_ports = cfg["services"]["api"]["ports"]
    assert any("127.0.0.1" in p for p in api_ports)

    # 3. Dashboard container environment must receive INFERPATH_HOST_BIND and INFERPATH_DASHBOARD_BIND
    dash_env = cfg["services"]["dashboard"]["environment"]
    dash_env_dict = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in dash_env}
    assert "INFERPATH_HOST_BIND" in dash_env_dict
    assert "INFERPATH_DASHBOARD_BIND" in dash_env_dict

    # 4. Dashboard ports must default to 127.0.0.1
    dash_ports = cfg["services"]["dashboard"]["ports"]
    assert any("${INFERPATH_DASHBOARD_BIND:-127.0.0.1}:8501:8501" in p for p in dash_ports)


# =============================================================================
# 4. Checklist: Dashboard Protection Against Unauthenticated Public Clients
# =============================================================================


def test_dashboard_auth_halts_on_unauthenticated_public_exposure(monkeypatch):
    """When dashboard is configured on a public interface without an API key, access is halted."""
    monkeypatch.setenv("INFERPATH_HOST_BIND", "0.0.0.0")
    monkeypatch.setenv("INFERPATH_DASHBOARD_BIND", "0.0.0.0")
    monkeypatch.delenv("INFERPATH_API_KEY", raising=False)

    with patch("streamlit.error") as mock_err, patch("streamlit.stop") as mock_stop:
        result = check_dashboard_auth()
        assert result is False
        mock_err.assert_called_once()
        assert "Unauthenticated public access is strictly prohibited" in str(mock_err.call_args)
        mock_stop.assert_called_once()


def test_dashboard_auth_enforces_key_when_configured(monkeypatch):
    """When INFERPATH_API_KEY is configured, unauthenticated sessions are blocked."""
    monkeypatch.setenv("INFERPATH_HOST_BIND", "127.0.0.1")
    monkeypatch.setenv("INFERPATH_API_KEY", "super_secret_dashboard_key_2026")

    with (
        patch("streamlit.session_state", {}),
        patch("streamlit.warning") as mock_warn,
        patch("streamlit.stop") as mock_stop,
    ):
        result = check_dashboard_auth()
        assert result is False
        mock_warn.assert_called_once()
        mock_stop.assert_called_once()


def test_dashboard_auth_permits_authenticated_session(monkeypatch):
    """When user has authenticated in session state, dashboard access is granted."""
    monkeypatch.setenv("INFERPATH_API_KEY", "super_secret_dashboard_key_2026")

    with patch("streamlit.session_state", {"authenticated": True}):
        result = check_dashboard_auth()
        assert result is True


def test_dashboard_auth_permits_localhost_in_demo_mode(monkeypatch):
    """Localhost bindings without an API key are permitted for local development/demo."""
    monkeypatch.setenv("INFERPATH_HOST_BIND", "127.0.0.1")
    monkeypatch.setenv("INFERPATH_DASHBOARD_BIND", "127.0.0.1")
    monkeypatch.delenv("INFERPATH_API_KEY", raising=False)

    result = check_dashboard_auth()
    assert result is True


# =============================================================================
# 5. P2: Distributed Rate Limiting Fail-Closed Policy
# =============================================================================


def test_distributed_rate_limiter_fail_closed_on_redis_outage():
    """When fail_closed=True, Redis exceptions raise RateLimiterBackendError and record degraded state."""
    failing_redis = MagicMock()
    failing_redis.eval.side_effect = ConnectionError("Redis cluster connection refused")

    limiter = DistributedRateLimiter(
        redis_client=failing_redis,
        requests_per_minute=60,
        fail_closed=True,
    )

    with pytest.raises(RateLimiterBackendError, match="Redis cluster connection refused"):
        limiter.is_allowed("client-ip-123")

    assert limiter.is_degraded is True
    assert limiter.failure_count == 1


def test_api_returns_429_on_rate_limiter_backend_failure():
    """API responds with HTTP 429 and RATE_LIMIT_BACKEND_UNAVAILABLE on Redis failure in fail-closed mode."""
    failing_redis = MagicMock()
    failing_redis.eval.side_effect = ConnectionError("Simulated Redis outage")

    limiter = DistributedRateLimiter(
        redis_client=failing_redis,
        requests_per_minute=60,
        fail_closed=True,
    )

    app = create_app(rate_limiter=limiter, deployment_mode="local-demo")
    client = TestClient(app)

    res = client.post("/v1/respond", json={"prompt": "Ping query"})
    assert res.status_code == 429
    body = res.json()
    assert body["code"] == "RATE_LIMIT_BACKEND_UNAVAILABLE"

    # Health endpoint must expose degradation
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "degraded"
    assert health_res.json()["rate_limiter_degraded"] is True

    # Dependency check exposes backend status
    deps_res = client.get("/health/dependencies")
    assert deps_res.status_code == 200
    deps = deps_res.json()["dependencies"]["rate_limiter"]
    assert deps["degraded"] is True
    assert deps["fail_closed"] is True


# =============================================================================
# 6. P2 & Checklist: Legacy SQLite Cache Automatic Migration on Startup
# =============================================================================


def test_legacy_sqlite_cache_auto_migrates_on_constructor_instantiation(tmp_path: Path):
    """Instantiating ExactCache against an older database automatically upgrades columns without manual migrate call."""
    legacy_db = tmp_path / "legacy_cache_auto.db"

    # 1. Seed legacy table without modern columns or schema_version table
    with sqlite3.connect(legacy_db) as conn:
        conn.execute(
            """
            CREATE TABLE exact_cache (
                cache_key TEXT PRIMARY KEY,
                normalized_prompt TEXT NOT NULL,
                output_text TEXT NOT NULL,
                tier TEXT NOT NULL,
                response_payload TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            );
            """
        )
        from inferpath.models.enums import ProvenanceLabel
        from inferpath.models.telemetry import ProvenanceValue

        seed_payload = HandlerOutput(
            tier=RouteTier.TEMPLATE_B,
            output_text="Legacy cached response",
            input_tokens=ProvenanceValue(
                value=10, provenance=ProvenanceLabel.CALCULATED, unit="tokens"
            ),
            output_tokens=ProvenanceValue(
                value=20, provenance=ProvenanceLabel.CALCULATED, unit="tokens"
            ),
            latency_ms=ProvenanceValue(
                value=15.0, provenance=ProvenanceLabel.CALCULATED, unit="ms"
            ),
            model_version="template-v1",
        ).model_dump_json()

        conn.execute(
            """
            INSERT INTO exact_cache (cache_key, normalized_prompt, output_text, tier, response_payload, version, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now', '+1 hour'));
            """,
            (
                "legacy_key_123",
                "prompt 1",
                "Legacy cached response",
                "template_b",
                seed_payload,
                "0.1.0",
            ),
        )
        conn.commit()

    # 2. Instantiate modern ExactCache normally (DO NOT call migrate_if_needed manually)
    cache = ExactCache(db_path=legacy_db)

    # 3. Verify schema version was updated automatically by constructor
    assert cache.get_schema_version() == CURRENT_SCHEMA_VERSION

    # 4. Verify columns were added automatically
    with sqlite3.connect(legacy_db) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(exact_cache);")
        columns = {row[1] for row in cursor.fetchall()}
        assert "hit_count" in columns
        assert "validation_status" in columns
        assert "requirements_snapshot" in columns
        assert "input_tokens" in columns
        assert "output_tokens" in columns
        assert "latency_ms" in columns

    # 5. Read legacy cached item immediately
    cached_output = cache.lookup("legacy_key_123")
    assert cached_output is not None
    assert cached_output.text == "Legacy cached response"
    assert cached_output.tier == RouteTier.TEMPLATE_B

    # 6. Write a new entry to verify write path works seamlessly
    store_req = UserRequest(request_id="req_test_write", prompt="prompt 2")
    new_output = HandlerOutput(
        tier=RouteTier.DETERMINISTIC_A,
        output_text="New response post-migration",
        input_tokens=ProvenanceValue(value=5, provenance=ProvenanceLabel.CALCULATED, unit="tokens"),
        output_tokens=ProvenanceValue(
            value=12, provenance=ProvenanceLabel.CALCULATED, unit="tokens"
        ),
        latency_ms=ProvenanceValue(value=2.0, provenance=ProvenanceLabel.CALCULATED, unit="ms"),
        model_version="rules-v1",
    )
    cache.store("new_key_456", store_req, new_output, route=RouteTier.DETERMINISTIC_A)

    retrieved = cache.lookup("new_key_456")
    assert retrieved is not None
    assert retrieved.text == "New response post-migration"


# =============================================================================
# 7. Checklist: Environment Configuration Matches .env.example
# =============================================================================


def test_env_example_variables_modify_settings(monkeypatch, tmp_path: Path):
    """Every variable documented in .env.example modifies effective runtime Settings when loaded."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-customkey-999888")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.custom-router.ai/api/v1")
    monkeypatch.setenv("INFERPATH_ENV", "staging-test")
    monkeypatch.setenv("INFERPATH_DB_PATH", str(tmp_path / "env_test.db"))
    monkeypatch.setenv("INFERPATH_SMALL_MODEL", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setenv("INFERPATH_LARGE_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("INFERPATH_STRICT_PRIVACY", "false")
    monkeypatch.setenv("INFERPATH_ENABLE_CACHE", "false")
    monkeypatch.setenv("INFERPATH_ENABLE_FALLBACK", "false")
    monkeypatch.setenv("INFERPATH_LOG_LEVEL", "DEBUG")

    settings = load_settings()

    assert settings.openrouter.api_key == "sk-or-v1-customkey-999888"
    assert settings.openrouter.base_url == "https://proxy.custom-router.ai/api/v1"
    assert settings.environment == "staging-test"
    assert settings.database_path == str(tmp_path / "env_test.db")
    assert settings.openrouter.small_model == "meta-llama/llama-3.1-8b-instruct"
    assert settings.openrouter.large_model == "openai/gpt-4o"
    assert settings.strict_privacy is False
    assert settings.routing.enable_cache is False
    assert settings.routing.enable_fallback is False
    assert settings.log_level == "DEBUG"


# =============================================================================
# 8. Checklist: Provider Failure Paths & Budget Reservation Release
# =============================================================================


def test_provider_failure_paths_release_budget_reservation():
    """Timeouts, malformed JSON, 429, and 500 errors release reserved budget tokens."""
    table = PricingTable(
        effective_date="2026-10-01",
        source="test",
        rates_per_1m={"provider-test-model": {"prompt": 0.50, "completion": 1.50}},
    )
    budget = RunBudgetTracker(max_usd=0.10, pricing_table=table)
    initial_remaining = budget.remaining_budget

    # 1. Timeout failure releases reservation
    timeout_http = MagicMock()
    timeout_http.post.side_effect = httpx.TimeoutException("Read timed out")
    adapter_timeout = OpenRouterAdapter(
        model_id="provider-test-model",
        pricing_table=table,
        client=timeout_http,
        budget_tracker=budget,
    )

    with pytest.raises(OpenRouterTimeoutError):
        adapter_timeout.generate("Timeout test")
    assert budget.remaining_budget == initial_remaining
    assert budget.total_spent_usd == 0.0

    # 2. Malformed JSON failure releases reservation
    malformed_http = MagicMock()
    malformed_resp = MagicMock()
    malformed_resp.status_code = 200
    malformed_resp.json.return_value = {"invalid_structure": True}  # Missing choices
    malformed_http.post.return_value = malformed_resp
    adapter_malformed = OpenRouterAdapter(
        model_id="provider-test-model",
        pricing_table=table,
        client=malformed_http,
        budget_tracker=budget,
    )

    with pytest.raises(OpenRouterResponseError):
        adapter_malformed.generate("Malformed test")
    assert budget.remaining_budget == initial_remaining
    assert budget.total_spent_usd == 0.0

    # 3. HTTP 429 Rate Limit failure releases reservation
    rate_http = MagicMock()
    rate_resp = MagicMock()
    rate_resp.status_code = 429
    rate_resp.headers = {"Retry-After": "30"}
    rate_resp.text = '{"error": {"message": "Rate limit exceeded"}}'
    rate_resp.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    rate_http.post.return_value = rate_resp
    adapter_429 = OpenRouterAdapter(
        model_id="provider-test-model",
        pricing_table=table,
        client=rate_http,
        budget_tracker=budget,
    )

    with pytest.raises(OpenRouterRateLimitError):
        adapter_429.generate("429 test")
    assert budget.remaining_budget == initial_remaining
    assert budget.total_spent_usd == 0.0

    # 4. HTTP 500 Internal Error failure releases reservation
    server_err_http = MagicMock()
    server_err_resp = MagicMock()
    server_err_resp.status_code = 500
    server_err_resp.text = '{"error": {"message": "Internal error"}}'
    server_err_resp.json.return_value = {"error": {"message": "Internal error"}}
    server_err_http.post.return_value = server_err_resp
    adapter_500 = OpenRouterAdapter(
        model_id="provider-test-model",
        pricing_table=table,
        client=server_err_http,
        budget_tracker=budget,
    )

    with pytest.raises(OpenRouterProviderError):
        adapter_500.generate("500 test")
    assert budget.remaining_budget == initial_remaining
    assert budget.total_spent_usd == 0.0
