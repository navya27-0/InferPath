"""Unit tests for production deployment boundaries: rate limiting, structured logging, tracing, secrets, and profiles."""

import json
import logging
import os
from pathlib import Path

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
from inferpath.services.rate_limiter import RateLimiter
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
