"""Unit tests for Prometheus metrics export, rate limiting, and API authentication."""

from fastapi.testclient import TestClient

from inferpath.api.main import RateLimiter, create_app
from inferpath.telemetry.metrics import MetricsCollector


def test_metrics_collector_export():
    metrics = MetricsCollector()
    metrics.record_request(
        tier="MODEL_SMALL",
        status="success",
        latency_ms=120.5,
        cost_usd=0.000045,
        was_cached=False,
        fallback_used=False,
    )
    metrics.record_request(
        tier="DETERMINISTIC_A",
        status="success",
        latency_ms=1.2,
        cost_usd=0.0,
        was_cached=True,
        fallback_used=False,
    )
    metrics.record_privacy_block()

    output = metrics.export_prometheus()
    assert "inferpath_requests_total" in output
    assert 'inferpath_requests_total{tier="MODEL_SMALL",status="success"} 1' in output
    assert 'inferpath_requests_total{tier="DETERMINISTIC_A",status="success"} 1' in output
    assert "inferpath_cache_hits_total 1" in output
    assert "inferpath_privacy_blocked_total 1" in output
    assert "inferpath_cost_usd_total 0.000045" in output


def test_rate_limiter():
    limiter = RateLimiter(requests_per_minute=3)
    client_id = "test_client_1"

    assert limiter.is_allowed(client_id) is True
    assert limiter.is_allowed(client_id) is True
    assert limiter.is_allowed(client_id) is True
    # 4th request within 60s window must be rejected
    assert limiter.is_allowed(client_id) is False


def test_api_metrics_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "inferpath_requests_total" in response.text


def test_api_auth_enforcement():
    # App configured with required API key
    app = create_app(api_key="secret-token-123")
    client = TestClient(app)

    payload = {"prompt": "Calculate: 10 + 20"}

    # Missing API key -> 401
    res_no_key = client.post("/v1/respond", json=payload)
    assert res_no_key.status_code == 401

    # Invalid API key -> 401
    res_bad_key = client.post("/v1/respond", json=payload, headers={"X-API-Key": "wrong-token"})
    assert res_bad_key.status_code == 401

    # Valid API key -> 200
    res_good_key = client.post(
        "/v1/respond", json=payload, headers={"X-API-Key": "secret-token-123"}
    )
    assert res_good_key.status_code == 200
    assert res_good_key.json()["route"] == "deterministic_a"
