"""Comprehensive regression test suite for InferPath October 2026 Final Audit Remediations.

Validates:
1. P1 - Privacy override verified redaction: confidential content never transmitted externally.
2. P1 - Pre-dispatch unpriced model validation: unknown models fail closed before any HTTP request.
3. P1 - Deployment safety: public host bindings (0.0.0.0) strictly require authentication; Compose defaults consistent.
4. P2 - Distributed rate limiting fail-closed policy: Redis outages fail closed and expose degradation in health telemetry.
5. Legacy SQLite cache migration: safely upgrades older schemas, creates schema_version, and preserves cached entries.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

from inferpath.api.main import create_app
from inferpath.cache.exact_cache import CURRENT_SCHEMA_VERSION, ExactCache
from inferpath.config.deployment_profiles import (
    get_deployment_profile,
)
from inferpath.config.settings import PricingTable
from inferpath.handlers.openrouter_adapter import (
    OpenRouterAdapter,
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
# 1. P1: Privacy Override Verified Redaction
# =============================================================================


class PromptRecordingAdapter:
    """Mock adapter that records the exact prompt received during dispatch."""

    def __init__(self, model_id: str = "mock-small"):
        self.model_id = model_id
        self.recorded_prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> MagicMock:
        self.recorded_prompts.append(prompt)
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


def test_private_request_with_override_transmits_only_redacted_prompt():
    """When override is enabled, the model adapter receives ONLY verified redacted content."""
    enforcer = PrivacyPolicyEnforcer()
    enforcer.approved_exceptions["allow_redacted_transmission_override"] = True

    mock_adapter = PromptRecordingAdapter()
    pipeline = RequestPipeline(small_adapter=mock_adapter)
    pipeline.privacy_enforcer = enforcer

    secret_raw = "sk-supersecretkey123456789012345"
    ssn_raw = "123-45-6789"
    raw_prompt = (
        f"Confidential analysis for account: {secret_raw} SSN: {ssn_raw} with s e c r e t info."
    )

    req = UserRequest(
        request_id="req_override_tx",
        prompt=raw_prompt,
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        allow_external_transmission=True,
    )

    # Execute tier
    pipeline._execute_tier(req, MagicMock(), RouteTier.SMALL_MODEL_C)

    # Verify adapter received a payload
    assert len(mock_adapter.recorded_prompts) == 1
    received_prompt = mock_adapter.recorded_prompts[0]

    # Invariants: raw secret and raw SSN must NEVER appear in outbound request
    assert secret_raw not in received_prompt
    assert ssn_raw not in received_prompt
    assert "s e c r e t" not in received_prompt

    # Redacted replacement tokens MUST be present
    assert "[REDACTED_API_KEY]" in received_prompt or "[REDACTED_KEY]" in received_prompt
    assert "[REDACTED_SSN]" in received_prompt
    assert "[REDACTED_SECRET]" in received_prompt

    # Verified clean
    assert enforcer.verify_redacted(received_prompt) is True


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
        model_id="unpriced/futuristic-frontier-model",
        pricing_table=table,
        api_key="sk-or-valid-key",
        client=mock_http_client,
    )

    # Synchronous dispatch
    with pytest.raises(UnknownModelPricingError, match="is unpriced in active pricing table"):
        adapter.generate("Test query")

    # Critical invariant: HTTP client post was NEVER called
    assert mock_http_client.post.call_count == 0


@pytest.mark.asyncio
async def test_unpriced_model_rejected_before_async_http_dispatch():
    """Asynchronous unpriced model dispatch fails before async HTTP request."""
    table = PricingTable(
        effective_date="2026-10-01",
        source="test",
        rates_per_1m={},
    )
    mock_async_client = AsyncMock()

    adapter = OpenRouterAdapter(
        model_id="unpriced/unknown-model",
        pricing_table=table,
        api_key="sk-or-valid-key",
        async_client=mock_async_client,
    )

    with pytest.raises(UnknownModelPricingError, match="is unpriced in active pricing table"):
        await adapter.generate_async("Test query")

    assert mock_async_client.post.call_count == 0


def test_budget_tracker_fails_fast_on_unpriced_model():
    """RunBudgetTracker.estimate_call_cost fails closed when model is missing from pricing table."""
    table = PricingTable(
        effective_date="2026-10-01",
        source="test",
        rates_per_1m={"known-model": {"prompt": 1.0, "completion": 2.0}},
    )
    tracker = RunBudgetTracker(max_usd=5.0, pricing_table=table)

    with pytest.raises(UnknownModelPricingError, match="is unpriced in active pricing table"):
        tracker.estimate_call_cost("unknown-model", "test prompt")


# =============================================================================
# 3. P1: Deployment Safety & Host Binding Safeguards
# =============================================================================


def test_public_binding_rejected_under_unauthenticated_profile(monkeypatch):
    """Binding to 0.0.0.0 without API key authentication is strictly rejected at startup."""
    monkeypatch.setenv("INFERPATH_HOST_BIND", "0.0.0.0")
    monkeypatch.delenv("INFERPATH_API_KEY", raising=False)
    monkeypatch.setenv("INFERPATH_DEPLOYMENT_PROFILE", "local-demo")

    with pytest.raises(
        RuntimeError, match="Binding to public interface '0.0.0.0' is strictly prohibited"
    ):
        create_app()


def test_conflicting_production_environment_with_local_demo_rejected(monkeypatch):
    """ENVIRONMENT=production cannot be combined with unauthenticated local-demo profile."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("INFERPATH_DEPLOYMENT_MODE", "local-demo")

    with pytest.raises(
        ValueError, match="cannot be combined with unauthenticated 'local-demo' profile"
    ):
        get_deployment_profile()


def test_docker_compose_defaults_are_consistent_with_production():
    """docker-compose.yml must default to production mode and localhost port bindings."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists()

    with open(compose_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    api_env = cfg["services"]["api"]["environment"]
    env_dict = {}
    for item in api_env:
        k, v = item.split("=", 1)
        env_dict[k] = v

    # Deployment mode in Compose must default to production
    assert env_dict["INFERPATH_DEPLOYMENT_MODE"] == "${INFERPATH_DEPLOYMENT_MODE:-production}"
    # Ports must default to 127.0.0.1
    api_ports = cfg["services"]["api"]["ports"]
    assert any("127.0.0.1" in p for p in api_ports)


# =============================================================================
# 4. P2: Distributed Rate Limiting Fail-Closed Policy
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
# 5. Legacy SQLite Cache Migration Test
# =============================================================================


def test_legacy_sqlite_cache_migration_and_data_preservation(tmp_path: Path):
    """Legacy SQLite cache tables without schema_version or modern columns migrate safely."""
    legacy_db = tmp_path / "legacy_cache.db"

    # 1. Manually create an older schema lacking schema_version and newer columns
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
        # Seed an older cached item
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

    # 2. Instantiate modern ExactCache pointing to the legacy database
    cache = ExactCache(db_path=legacy_db)
    new_version = cache.migrate_if_needed()

    assert new_version == CURRENT_SCHEMA_VERSION
    assert cache.get_schema_version() == CURRENT_SCHEMA_VERSION

    # 3. Verify columns were added
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

    # 4. Verify existing cached entry is preserved and can be retrieved
    cached_output = cache.lookup("legacy_key_123")
    assert cached_output is not None
    assert cached_output.text == "Legacy cached response"
    assert cached_output.tier == RouteTier.TEMPLATE_B
