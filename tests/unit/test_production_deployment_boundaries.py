"""Unit tests for production deployment boundaries: rate limiting, structured logging, tracing, secrets, and profiles."""

import json
import logging
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inferpath.api.main import create_app
from inferpath.config.deployment_profiles import (
    ProfileName,
    get_deployment_profile,
)
from inferpath.security.secret_manager import (
    EnvironmentSecretProvider,
    FileSecretProvider,
    MockSecretProvider,
    SecretManager,
)
from inferpath.services.rate_limiter import DistributedRateLimiter, RateLimiter
from inferpath.telemetry.structured_logging import (
    StructuredJsonFormatter,
    configure_structured_logging,
)
from inferpath.telemetry.tracing import Tracer


def test_rate_limiter_service_boundaries():
    """Verify rate limiter operates with sliding window and bounded client memory."""
    limiter = RateLimiter(requests_per_minute=2, max_clients=3)

    # Allow 2 requests, block 3rd
    assert limiter.is_allowed("client_a") is True
    assert limiter.is_allowed("client_a") is True
    assert limiter.is_allowed("client_a") is False

    # Different client gets own quota
    assert limiter.is_allowed("client_b") is True
    assert limiter.is_allowed("client_c") is True

    # Capacity bounding
    stats = limiter.get_stats()
    assert stats["max_clients"] == 3
    assert stats["requests_per_minute"] == 2

    # Reset clears history
    limiter.reset()
    assert limiter.is_allowed("client_a") is True


def test_structured_json_logging_and_redaction():
    """Verify structured logger formats JSON with UTC timestamps and scrubs secrets."""
    formatter = StructuredJsonFormatter(service_name="inferpath", environment="test")

    record = logging.LogRecord(
        name="inferpath.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Connecting with key sk-or-v1-abcdef1234567890 to backend.",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["service"] == "inferpath"
    assert parsed["environment"] == "test"
    assert "sk-or-v1-abcdef1234567890" not in parsed["message"]
    assert "[REDACTED_API_KEY]" in parsed["message"]
    assert "timestamp" in parsed

    configured = configure_structured_logging(service_name="inferpath_test", log_level="DEBUG")
    assert configured.name == "inferpath"


def test_opentelemetry_tracing_and_w3c_context():
    """Verify tracer correctly generates, records, and parses W3C Trace Context."""
    tracer = Tracer(service_name="test-service")

    valid_header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    parsed = tracer.parse_w3c_traceparent(valid_header)
    assert parsed is not None
    trace_id, parent_span_id = parsed
    assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert parent_span_id == "00f067aa0ba902b7"

    # Context manager span
    with tracer.start_as_current_span(
        "test_operation", trace_id=trace_id, parent_span_id=parent_span_id
    ) as span:
        span.set_attribute("task.category", "arithmetic")
        span.add_event("cache_check", {"status": "miss"})
        assert span.trace_id == trace_id
        assert span.duration_ms >= 0.0

    assert span.status == "OK"
    assert span.end_time_ns is not None
    traceparent_out = span.export_w3c_traceparent()
    assert traceparent_out.startswith("00-")
    assert trace_id in traceparent_out


def test_secret_manager_hierarchy_and_masking(tmp_path: Path):
    """Verify SecretManager hierarchy: file -> environment -> mock fallback with masking."""
    # 1. File provider
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    (secret_dir / "api_token").write_text("file-secret-value-12345", encoding="utf-8")

    file_prov = FileSecretProvider(secrets_dir=secret_dir)
    os.environ["TEST_ENV_SECRET_KEY"] = "env-secret-val-444"
    env_prov = EnvironmentSecretProvider(prefix="TEST_")
    mock_prov = MockSecretProvider({"backup_key": "mock-secret-99999"})
    mgr = SecretManager(providers=[file_prov, env_prov, mock_prov])

    assert mgr.get("api_token") == "file-secret-value-12345"
    assert mgr.get("ENV_SECRET_KEY") == "env-secret-val-444"
    assert mgr.get("backup_key") == "mock-secret-99999"
    assert mgr.get("non_existent") is None

    # Masking
    masked = mgr.mask_secret("api_token", show_prefix=4, show_suffix=4)
    assert masked == "file...2345"
    assert "secret-value" not in masked


def test_deployment_profiles():
    """Verify canonical deployment profiles have appropriate operational guardrails."""
    local_cfg = get_deployment_profile("local-demo")
    assert local_cfg.name == ProfileName.LOCAL_DEMO
    assert local_cfg.require_api_key is False
    assert local_cfg.default_bind_host == "127.0.0.1"

    prod_cfg = get_deployment_profile("production")
    assert prod_cfg.name == ProfileName.PRODUCTION
    assert prod_cfg.require_api_key is True
    assert prod_cfg.adapter_mode == "real"

    staging_cfg = get_deployment_profile("staging")
    assert staging_cfg.name == ProfileName.STAGING
    assert staging_cfg.require_api_key is True

    offline_cfg = get_deployment_profile("benchmark-offline")
    assert offline_cfg.name == ProfileName.BENCHMARK_OFFLINE
    assert offline_cfg.rate_limit_rpm == 10000


def test_api_tracing_headers_in_response():
    """Verify /v1/respond returns X-Trace-Id and traceparent in response headers."""
    app = create_app(deployment_mode="local-demo")
    client = TestClient(app)

    response = client.post("/v1/respond", json={"prompt": "125 * 8"})
    assert response.status_code == 200
    assert "X-Trace-Id" in response.headers
    assert len(response.headers["X-Trace-Id"]) == 32
    assert "traceparent" in response.headers
    assert response.headers["traceparent"].startswith("00-")

    # App state inspection
    assert app.state.deployment_profile.name == ProfileName.LOCAL_DEMO
    assert app.state.default_bind_host == "127.0.0.1"


def test_production_profile_fails_startup_without_real_credentials(monkeypatch):
    """Production profile disallowing mock fallback must fail startup when OPENROUTER_API_KEY is missing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="enforces real adapter mode"):
        create_app(deployment_mode="production")


def test_production_profile_strictly_requires_api_key_authentication():
    """In production mode with no API key configured, all requests must be rejected with 401."""
    # When injected with a mock pipeline, app initializes but rejects unauthenticated calls
    from inferpath.config.settings import load_pricing_table
    from inferpath.pipeline import RequestPipeline

    pipeline = RequestPipeline(pricing_table=load_pricing_table())
    app = create_app(pipeline=pipeline, deployment_mode="production", api_key="")
    client = TestClient(app)

    res = client.post("/v1/respond", json={"prompt": "125 * 8"})
    assert res.status_code == 401
    assert res.json()["code"] == "UNAUTHORIZED"
    assert "requires valid API key authentication" in res.json()["detail"]


def test_tracing_disabled_profile_omits_trace_headers():
    """When a profile has enable_tracing=False, response headers must not include trace IDs."""
    from inferpath.config.settings import load_pricing_table
    from inferpath.pipeline import RequestPipeline

    pipeline = RequestPipeline(pricing_table=load_pricing_table())
    app = create_app(pipeline=pipeline, deployment_mode="benchmark-offline")
    client = TestClient(app)

    res = client.post("/v1/respond", json={"prompt": "125 * 8"})
    assert res.status_code == 200
    assert "X-Trace-Id" not in res.headers
    assert "traceparent" not in res.headers


def test_production_secret_loading_end_to_end(tmp_path: Path, monkeypatch):
    """Verify that credentials supplied through mounted secret files propagate into RequestPipeline."""
    from inferpath.handlers.openrouter_adapter import OpenRouterAdapter
    from inferpath.pipeline import RequestPipeline
    from inferpath.security.secret_manager import GLOBAL_SECRET_MANAGER

    # Ensure no environment variables contaminate secret resolution
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("INFERPATH_OPENROUTER_API_KEY", raising=False)

    # Set up mounted secret file
    secret_dir = tmp_path / "k8s_secrets"
    secret_dir.mkdir()
    secret_file = secret_dir / "openrouter_api_key"
    secret_file.write_text("sk-or-v1-mounted-secret-test-key", encoding="utf-8")

    # Reconfigure GLOBAL_SECRET_MANAGER with the file provider
    orig_providers = GLOBAL_SECRET_MANAGER.providers
    GLOBAL_SECRET_MANAGER.clear_cache()
    GLOBAL_SECRET_MANAGER.providers = [
        FileSecretProvider(secrets_dir=secret_dir),
        EnvironmentSecretProvider(),
    ]

    try:
        # Pipeline in real mode should read secret from GLOBAL_SECRET_MANAGER
        pipeline = RequestPipeline(adapter_mode="real", allow_mock_fallback=False)
        assert pipeline.settings.openrouter.api_key == "sk-or-v1-mounted-secret-test-key"
        assert isinstance(pipeline.small_adapter, OpenRouterAdapter)
        assert isinstance(pipeline.large_adapter, OpenRouterAdapter)
        assert pipeline.small_adapter.api_key == "sk-or-v1-mounted-secret-test-key"

        # FastAPI app creation under production profile succeeds without error
        app = create_app(deployment_mode="production", api_key="test-api-key")
        assert app.state.deployment_profile.name == ProfileName.PRODUCTION
    finally:
        GLOBAL_SECRET_MANAGER.providers = orig_providers
        GLOBAL_SECRET_MANAGER.clear_cache()


def test_adapter_mode_strict_selection_rules(monkeypatch):
    """Verify adapter mode selection strictly adheres to mode, credentials, and fallback settings."""
    from inferpath.handlers.mock_adapter import MockLargeModel, MockSmallModel
    from inferpath.pipeline import PipelineExecutionError, RequestPipeline
    from inferpath.security.secret_manager import GLOBAL_SECRET_MANAGER

    # 1. simulated mode: ALWAYS mock even if API key exists
    pipeline_sim = RequestPipeline(adapter_mode="simulated")
    pipeline_sim.settings.openrouter.api_key = "sk-or-dummy-key"
    adapter_s = pipeline_sim._create_small_adapter()
    adapter_l = pipeline_sim._create_large_adapter()
    assert isinstance(adapter_s, MockSmallModel)
    assert isinstance(adapter_l, MockLargeModel)

    # 2. fixture mode: ALWAYS mock even if API key exists
    pipeline_fix = RequestPipeline(adapter_mode="fixture")
    pipeline_fix.settings.openrouter.api_key = "sk-or-dummy-key"
    assert isinstance(pipeline_fix._create_small_adapter(), MockSmallModel)
    assert isinstance(pipeline_fix._create_large_adapter(), MockLargeModel)

    # 3. real mode without credentials and allow_mock_fallback=False raises error
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    GLOBAL_SECRET_MANAGER.clear_cache()
    with pytest.raises(PipelineExecutionError, match="requires OPENROUTER_API_KEY"):
        RequestPipeline(adapter_mode="real", allow_mock_fallback=False)

    # 4. real mode without credentials but allow_mock_fallback=True uses mock
    pipeline_real_fallback = RequestPipeline(adapter_mode="real", allow_mock_fallback=True)
    pipeline_real_fallback.settings.openrouter.api_key = ""
    assert isinstance(pipeline_real_fallback._create_small_adapter(), MockSmallModel)
    assert isinstance(pipeline_real_fallback._create_large_adapter(), MockLargeModel)


def test_distributed_rate_limiter_contract():
    """Verify DistributedRateLimiter fallback and mocked redis pipeline behaviors."""
    # Fallback to local in-memory when redis_client is None
    dist_limiter = DistributedRateLimiter(requests_per_minute=2)
    assert dist_limiter.is_allowed("client_x") is True
    assert dist_limiter.is_allowed("client_x") is True
    assert dist_limiter.is_allowed("client_x") is False

    dist_limiter.reset()
    assert dist_limiter.is_allowed("client_x") is True

    # Mock Redis client execution
    class MockRedisPipeline:
        def __init__(self, parent):
            self.parent = parent
            self._ops = []

        def zremrangebyscore(self, key, min_s, max_s):
            self._ops.append(("rem", key, min_s, max_s))
            return self

        def zcard(self, key):
            self._ops.append(("card", key))
            return self

        def zadd(self, key, mapping):
            self._ops.append(("add", key, mapping))
            return self

        def expire(self, key, seconds):
            self._ops.append(("expire", key, seconds))
            return self

        def execute(self):
            # Return tuple: (rem_result, card_count, add_result, expire_result)
            return (0, self.parent.simulated_count, 1, True)

    class MockRedisClient:
        def __init__(self):
            self.simulated_count = 1

        def pipeline(self):
            return MockRedisPipeline(self)

        def keys(self, pattern):
            return ["inferpath:ratelimit:client_y"]

        def delete(self, *keys):
            return len(keys)

    mock_client = MockRedisClient()
    redis_limiter = DistributedRateLimiter(redis_client=mock_client, requests_per_minute=5)
    # Count 1 < 5 -> allowed
    assert redis_limiter.is_allowed("client_y") is True

    # Count 5 >= 5 -> rejected
    mock_client.simulated_count = 5
    assert redis_limiter.is_allowed("client_y") is False

    # Reset calls delete
    redis_limiter.reset()


def test_adapter_factory_isolation():
    """Verify AdapterFactory creates mock and real adapters with isolated logic."""
    from inferpath.config.settings import load_pricing_table, load_settings
    from inferpath.handlers.factory import AdapterFactory
    from inferpath.handlers.mock_adapter import MockLargeModel, MockSmallModel
    from inferpath.handlers.openrouter_adapter import OpenRouterAdapter

    settings = load_settings()
    pricing = load_pricing_table()

    # 1. simulated mode
    s_adapter = AdapterFactory.create_small_adapter(settings, pricing, adapter_mode="simulated")
    l_adapter = AdapterFactory.create_large_adapter(settings, pricing, adapter_mode="simulated")
    assert isinstance(s_adapter, MockSmallModel)
    assert isinstance(l_adapter, MockLargeModel)

    # 2. real mode with key
    settings.openrouter.api_key = "sk-or-test-factory-key"
    s_real = AdapterFactory.create_small_adapter(settings, pricing, adapter_mode="real")
    l_real = AdapterFactory.create_large_adapter(settings, pricing, adapter_mode="real")
    assert isinstance(s_real, OpenRouterAdapter)
    assert isinstance(l_real, OpenRouterAdapter)


def test_create_app_with_distributed_rate_limiter():
    """Verify create_app integrates seamlessly with DistributedRateLimiter."""
    dist_limiter = DistributedRateLimiter(requests_per_minute=1)
    app = create_app(rate_limiter=dist_limiter, deployment_mode="local-demo")
    client = TestClient(app)

    res1 = client.post("/v1/respond", json={"prompt": "125 * 8"})
    assert res1.status_code == 200

    # Second call should exceed quota
    res2 = client.post("/v1/respond", json={"prompt": "125 * 8"})
    assert res2.status_code == 429
    assert res2.json()["code"] == "RATE_LIMIT_EXCEEDED"


def test_pipeline_and_api_adapter_factory_injection():
    """Verify adapter_factory can be injected into RequestPipeline and create_app."""
    from unittest.mock import MagicMock

    from inferpath import __version__
    from inferpath.handlers.mock_adapter import MockLargeModel, MockSmallModel
    from inferpath.pipeline import RequestPipeline

    custom_factory = MagicMock()
    custom_factory.create_small_adapter.return_value = MockSmallModel(model_id="mock-custom-s")
    custom_factory.create_large_adapter.return_value = MockLargeModel(model_id="mock-custom-l")

    # Injected into RequestPipeline
    pipeline = RequestPipeline(adapter_factory=custom_factory)
    assert pipeline.adapter_factory == custom_factory
    assert custom_factory.create_small_adapter.called
    assert custom_factory.create_large_adapter.called

    # Injected into create_app
    app = create_app(deployment_mode="local-demo", adapter_factory=custom_factory)
    client = TestClient(app)
    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["version"] == __version__
    assert health.json()["version"] == "0.2.0"
