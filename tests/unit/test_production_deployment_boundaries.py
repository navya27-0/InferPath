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
    """Verify canonical deployment profiles have appropriate operational guardrails matching documentation."""
    local_cfg = get_deployment_profile("local-demo")
    assert local_cfg.name == ProfileName.LOCAL_DEMO
    assert local_cfg.require_api_key is False
    assert local_cfg.default_bind_host == "127.0.0.1"
    assert local_cfg.rate_limit_rpm == 60
    assert local_cfg.adapter_mode == "simulated"
    assert local_cfg.allow_mock_fallback is True

    staging_cfg = get_deployment_profile("staging")
    assert staging_cfg.name == ProfileName.STAGING
    assert staging_cfg.require_api_key is True
    assert staging_cfg.default_bind_host == "0.0.0.0"
    assert staging_cfg.rate_limit_rpm == 120
    assert staging_cfg.adapter_mode == "real"
    assert staging_cfg.allow_mock_fallback is True

    prod_cfg = get_deployment_profile("production")
    assert prod_cfg.name == ProfileName.PRODUCTION
    assert prod_cfg.require_api_key is True
    assert prod_cfg.default_bind_host == "0.0.0.0"
    assert prod_cfg.rate_limit_rpm == 240
    assert prod_cfg.adapter_mode == "real"
    assert prod_cfg.allow_mock_fallback is False

    offline_cfg = get_deployment_profile("benchmark-offline")
    assert offline_cfg.name == ProfileName.BENCHMARK_OFFLINE
    assert offline_cfg.require_api_key is False
    assert offline_cfg.default_bind_host == "127.0.0.1"
    assert offline_cfg.rate_limit_rpm == 10000
    assert offline_cfg.adapter_mode == "simulated"
    assert offline_cfg.allow_mock_fallback is True


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
    assert health.json()["version"] == "0.3.0"


def test_redis_rate_limiter_health_reporting_and_degradation(monkeypatch):
    """Verify health endpoint reports rate limiter status and handles graceful degradation."""
    # 1. Default in-memory
    monkeypatch.delenv("INFERPATH_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    app_default = create_app(deployment_mode="local-demo")
    client_default = TestClient(app_default)
    health_default = client_default.get("/v1/health").json()
    assert health_default["rate_limiter"] == "in_memory"

    # 2. Injected DistributedRateLimiter
    dist_limiter = DistributedRateLimiter(requests_per_minute=60)
    app_dist = create_app(rate_limiter=dist_limiter, deployment_mode="local-demo")
    client_dist = TestClient(app_dist)
    health_dist = client_dist.get("/v1/health").json()
    assert health_dist["rate_limiter"] == "distributed"

    # 3. Graceful degradation when redis connection fails and require_redis is False
    monkeypatch.setenv("INFERPATH_REDIS_URL", "redis://invalid-host-cannot-connect:6379/0")
    monkeypatch.setenv("INFERPATH_REQUIRE_REDIS", "false")
    app_degraded = create_app(deployment_mode="local-demo")
    client_degraded = TestClient(app_degraded)
    health_degraded = client_degraded.get("/v1/health").json()
    assert health_degraded["rate_limiter"] == "in_memory (degraded_fallback)"

    # 4. Strict failure when require_redis is True
    monkeypatch.setenv("INFERPATH_REQUIRE_REDIS", "true")
    with pytest.raises(RuntimeError, match="Failed to initialize distributed rate limiter"):
        create_app(deployment_mode="local-demo")


def test_production_profile_enforces_redis_requirement(monkeypatch):
    """Verify production profile refuses to silently fall back if REDIS_URL is provided but broken."""
    monkeypatch.setenv("REDIS_URL", "redis://invalid-host-cannot-connect:6379/0")
    monkeypatch.delenv("INFERPATH_REQUIRE_REDIS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-prod-key")
    with pytest.raises(RuntimeError, match="Failed to initialize distributed rate limiter"):
        create_app(deployment_mode="production")


# =============================================================================
# Security & Concurrency Hardening: Redis Timestamp Collisions, Extra Sanitization, URL Masking
# =============================================================================


class InMemoryFakeRedis:
    """In-memory Redis fake with sorted set and optimistic locking support for rate limiter testing."""

    def __init__(self):
        self._zsets = {}
        self._versions = {}

    def get_version(self, key: str) -> int:
        return self._versions.get(key, 0)

    def zremrangebyscore(self, key: str, min_score: float | str, max_score: float | str) -> int:
        z = self._zsets.get(key, {})
        max_val = float(max_score)
        to_del = [m for m, s in z.items() if s <= max_val]
        for m in to_del:
            del z[m]
        if to_del:
            self._versions[key] = self._versions.get(key, 0) + 1
        return len(to_del)

    def zcard(self, key: str) -> int:
        return len(self._zsets.get(key, {}))

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if key not in self._zsets:
            self._zsets[key] = {}
        added = 0
        for m, s in mapping.items():
            if m not in self._zsets[key]:
                added += 1
            self._zsets[key][m] = s
        if added > 0:
            self._versions[key] = self._versions.get(key, 0) + 1
        return added

    def expire(self, key: str, seconds: int) -> bool:
        return True

    def eval(
        self,
        script: str,
        numkeys: int,
        key: str,
        now: float,
        cutoff: float,
        limit: int,
        ttl: int,
        member: str,
    ) -> int:
        self.zremrangebyscore(key, "-inf", cutoff)
        if self.zcard(key) < limit:
            self.zadd(key, {member: float(now)})
            return 1
        return 0

    def pipeline(self, transaction: bool = True):
        return FakeRedisPipeline(self)


class FakeRedisPipeline:
    def __init__(self, client: InMemoryFakeRedis):
        self.client = client
        self.calls = []
        self._watched = None
        self._watched_version = None
        self._in_multi = False

    def watch(self, key):
        self._watched = key
        self._watched_version = self.client.get_version(key)
        return self

    def unwatch(self):
        self._watched = None
        self._watched_version = None
        return self

    def multi(self):
        self._in_multi = True
        return self

    def zremrangebyscore(self, key, min_s, max_s):
        # In redis-py, if not in multi block and watching, command executes immediately
        if self._watched is not None and not self._in_multi:
            return self.client.zremrangebyscore(key, min_s, max_s)
        self.calls.append(("zrem", key, min_s, max_s))
        return self

    def zcard(self, key):
        # In redis-py, if not in multi block and watching, command executes immediately
        if self._watched is not None and not self._in_multi:
            return self.client.zcard(key)
        self.calls.append(("zcard", key))
        return self

    def zadd(self, key, mapping):
        self.calls.append(("zadd", key, mapping))
        return self

    def expire(self, key, ttl):
        self.calls.append(("expire", key, ttl))
        return self

    def execute(self):
        if (
            self._in_multi
            and self._watched is not None
            and self.client.get_version(self._watched) != self._watched_version
        ):
            self.calls = []
            self._in_multi = False
            raise RuntimeError("WatchError: watched key was modified during transaction")

        results = []
        for call in self.calls:
            if call[0] == "zrem":
                results.append(self.client.zremrangebyscore(call[1], call[2], call[3]))
            elif call[0] == "zcard":
                results.append(self.client.zcard(call[1]))
            elif call[0] == "zadd":
                results.append(self.client.zadd(call[1], call[2]))
            elif call[0] == "expire":
                results.append(self.client.expire(call[1], call[2]))
        self.calls = []
        self._in_multi = False
        return results


def test_redis_rate_limiter_identical_timestamp_member_collision(monkeypatch):
    """Verify Redis rate limiter generates unique sorted-set members under identical timestamps."""
    import time

    from inferpath.services.rate_limiter import DistributedRateLimiter

    fake_redis = InMemoryFakeRedis()
    limiter = DistributedRateLimiter(redis_client=fake_redis, requests_per_minute=2)

    # Freeze time so all requests arrive at identical float timestamp
    frozen_time = 1700000000.0
    monkeypatch.setattr(time, "time", lambda: frozen_time)

    # Request 1 & 2 allowed
    assert limiter.is_allowed("client_lua") is True
    assert limiter.is_allowed("client_lua") is True

    # Card count in redis must be 2, NOT 1
    assert fake_redis.zcard("inferpath:ratelimit:client_lua") == 2

    # Request 3 must be blocked under identical timestamp
    assert limiter.is_allowed("client_lua") is False

    # Now verify fallback pipeline path (without eval)
    fake_redis_pipe = InMemoryFakeRedis()
    # Remove eval to trigger pipeline fallback path
    fake_redis_pipe.eval = None
    limiter_pipe = DistributedRateLimiter(redis_client=fake_redis_pipe, requests_per_minute=2)

    assert limiter_pipe.is_allowed("client_pipe") is True
    assert limiter_pipe.is_allowed("client_pipe") is True
    assert fake_redis_pipe.zcard("inferpath:ratelimit:client_pipe") == 2
    assert limiter_pipe.is_allowed("client_pipe") is False


def test_redis_rate_limiter_concurrency_stress(monkeypatch):
    """Verify concurrent requests with colliding timestamps strictly enforce rate quotas."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from inferpath.services.rate_limiter import DistributedRateLimiter

    fake_redis = InMemoryFakeRedis()
    limiter = DistributedRateLimiter(redis_client=fake_redis, requests_per_minute=5)
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(limiter.is_allowed, "concurrent_client") for _ in range(20)]
        for f in futures:
            results.append(f.result())

    # Exactly 5 requests allowed, 15 rejected
    assert results.count(True) == 5
    assert results.count(False) == 15
    assert fake_redis.zcard("inferpath:ratelimit:concurrent_client") == 5


def test_redis_rate_limiter_fallback_concurrency_stress(monkeypatch):
    """Verify concurrent requests through non-Lua fallback paths strictly enforce quotas without races."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from inferpath.services.rate_limiter import DistributedRateLimiter

    monkeypatch.setattr(time, "time", lambda: 1700000000.0)

    # 1. Fallback path with WATCH/MULTI/EXEC optimistic locking (eval = None)
    fake_redis_watch = InMemoryFakeRedis()
    fake_redis_watch.eval = None
    limiter_watch = DistributedRateLimiter(redis_client=fake_redis_watch, requests_per_minute=4)

    results_watch = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(limiter_watch.is_allowed, "concurrent_fallback_watch")
            for _ in range(20)
        ]
        for f in futures:
            results_watch.append(f.result())

    # Exactly 4 allowed, 16 rejected
    assert results_watch.count(True) == 4
    assert results_watch.count(False) == 16
    assert fake_redis_watch.zcard("inferpath:ratelimit:concurrent_fallback_watch") == 4

    # 2. Fallback path without WATCH support (serialized under self._fallback_lock)
    class FakeRedisPipelineNoWatch(FakeRedisPipeline):
        watch = None
        unwatch = None
        multi = None

    class FakeRedisNoWatch(InMemoryFakeRedis):
        def pipeline(self, transaction: bool = True):
            return FakeRedisPipelineNoWatch(self)

    fake_redis_nowatch = FakeRedisNoWatch()
    fake_redis_nowatch.eval = None
    limiter_nowatch = DistributedRateLimiter(redis_client=fake_redis_nowatch, requests_per_minute=3)

    results_nowatch = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(limiter_nowatch.is_allowed, "concurrent_fallback_nowatch")
            for _ in range(20)
        ]
        for f in futures:
            results_nowatch.append(f.result())

    # Exactly 3 allowed, 17 rejected
    assert results_nowatch.count(True) == 3
    assert results_nowatch.count(False) == 17
    assert fake_redis_nowatch.zcard("inferpath:ratelimit:concurrent_fallback_nowatch") == 3


def test_structured_logging_recursive_sanitization():
    """Verify structured logger recursively scrubs API keys and sensitive tokens in extra fields."""
    import json
    import logging

    from inferpath.telemetry.structured_logging import StructuredJsonFormatter

    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Sample log message with key sk-or-v1-supersecretkey1234567890",
        args=(),
        exc_info=None,
    )
    # Deeply nested extra with sensitive tokens in dicts, lists, tuples, and keys
    record.auth_payload = {
        "nested": {
            "token": "sk-123456789012345678",
            "bearer": "Bearer topsecrettokenvalue123456",
        },
        "credentials_list": [
            "sk-or-v1-nestedinlist1234567890",
            {"key": "secret: mysecrettoken123456"},
        ],
        "credentials_tuple": ("sk-or-v1-inatuple1234567890",),
    }

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    # Assert no sensitive raw keys exist anywhere in JSON string
    assert "sk-or-v1-supersecretkey1234567890" not in formatted
    assert "sk-123456789012345678" not in formatted
    assert "topsecrettokenvalue123456" not in formatted
    assert "sk-or-v1-nestedinlist1234567890" not in formatted
    assert "mysecrettoken123456" not in formatted
    assert "sk-or-v1-inatuple1234567890" not in formatted

    # Assert redaction token is present in nested structures
    assert parsed["extra"]["auth_payload"]["nested"]["token"] == "[REDACTED_API_KEY]"
    assert parsed["extra"]["auth_payload"]["credentials_list"][0] == "[REDACTED_API_KEY]"
    assert parsed["extra"]["auth_payload"]["credentials_tuple"][0] == "[REDACTED_API_KEY]"


def test_redis_connection_credential_masking(monkeypatch, caplog):
    """Verify Redis connection errors and logs mask embedded passwords in redis_url."""
    import logging

    from inferpath.api.main import _sanitize_redis_url, create_app

    # 1. URL sanitizer unit behavior
    assert (
        _sanitize_redis_url("redis://:supersecretpass@localhost:6379/0")
        == "redis://:***@localhost:6379/0"
    )
    assert (
        _sanitize_redis_url("redis://admin:supersecretpass@host:6379/1")
        == "redis://admin:***@host:6379/1"
    )
    assert (
        _sanitize_redis_url("rediss://myuser:secret12345@cache.internal:6380")
        == "rediss://myuser:***@cache.internal:6380"
    )
    assert _sanitize_redis_url("redis://localhost:6379/0") == "redis://localhost:6379/0"

    # 2. Exception message masking when fail_closed or require_redis is True
    monkeypatch.setenv(
        "INFERPATH_REDIS_URL", "redis://:very_secret_redis_password_999@127.0.0.1:6379/0"
    )
    monkeypatch.setenv("INFERPATH_REQUIRE_REDIS", "true")

    with pytest.raises(RuntimeError) as exc_info:
        create_app(deployment_mode="local-demo")

    exc_msg = str(exc_info.value)
    assert "very_secret_redis_password_999" not in exc_msg
    assert "redis://:***@127.0.0.1:6379/0" in exc_msg

    # 3. Log warning masking when non-failing mode degrades to in-memory
    monkeypatch.setenv("INFERPATH_REQUIRE_REDIS", "false")
    monkeypatch.setenv("INFERPATH_RATE_LIMIT_FAIL_CLOSED", "false")
    with caplog.at_level(logging.WARNING):
        app = create_app(deployment_mode="local-demo")
        assert app.state.rate_limiter_status == "in_memory (degraded_fallback)"

    log_text = caplog.text
    assert "very_secret_redis_password_999" not in log_text
