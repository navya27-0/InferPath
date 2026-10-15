"""Unit tests for API security, authentication modes, proxy boundaries, and payload validation."""

from fastapi.testclient import TestClient

from inferpath.api.main import RateLimiter, create_app


def test_protected_mode_rejects_unauthenticated():
    """Protected mode must reject requests lacking a valid API key with HTTP 401."""
    app = create_app(api_key="secret-master-key", deployment_mode="protected")
    client = TestClient(app)

    # 1. Missing header -> 401
    res1 = client.post("/v1/respond", json={"prompt": "Calculate: 1 + 1"})
    assert res1.status_code == 401
    assert res1.json()["code"] == "UNAUTHORIZED"

    # 2. Invalid header -> 401
    res2 = client.post(
        "/v1/respond",
        json={"prompt": "Calculate: 1 + 1"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert res2.status_code == 401
    assert res2.json()["code"] == "UNAUTHORIZED"

    # 3. Valid Bearer token -> 200
    res3 = client.post(
        "/v1/respond",
        json={"prompt": "Calculate: 1 + 1"},
        headers={"Authorization": "Bearer secret-master-key"},
    )
    assert res3.status_code == 200


def test_local_demo_mode_allows_unauthenticated_by_default():
    """Local demonstration mode allows requests without credentials when none configured."""
    app = create_app(api_key="", deployment_mode="local-demo")
    client = TestClient(app)

    res = client.post("/v1/respond", json={"prompt": "Convert 100 F to C"})
    assert res.status_code == 200


def test_proxy_header_not_trusted_by_default():
    """X-Forwarded-For must be ignored unless client IP matches configured trusted_proxies."""
    limiter = RateLimiter(requests_per_minute=2)
    # App without trusted proxies
    app = create_app(rate_limiter=limiter, deployment_mode="local-demo")
    client = TestClient(app)

    # Send 2 requests with spoofed X-Forwarded-For headers
    res1 = client.post(
        "/v1/respond",
        json={"prompt": "Calculate: 2 + 2"},
        headers={"X-Forwarded-For": "198.51.100.1"},
    )
    assert res1.status_code == 200

    res2 = client.post(
        "/v1/respond",
        json={"prompt": "Calculate: 2 + 2"},
        headers={"X-Forwarded-For": "198.51.100.2"},
    )
    assert res2.status_code == 200

    # 3rd request from same client (testclient default) should be rate limited despite spoofed IP
    res3 = client.post(
        "/v1/respond",
        json={"prompt": "Calculate: 2 + 2"},
        headers={"X-Forwarded-For": "198.51.100.3"},
    )
    assert res3.status_code == 429
    assert res3.json()["code"] == "RATE_LIMIT_EXCEEDED"


def test_rate_limiter_bounded_capacity():
    """RateLimiter must prune expired client entries and bound dictionary memory size."""
    limiter = RateLimiter(requests_per_minute=10, max_clients=5)

    # Insert 5 clients
    for i in range(5):
        assert limiter.is_allowed(f"client_{i}") is True

    assert len(limiter._history) == 5

    # Insert 6th client -> must trigger capacity bound check
    assert limiter.is_allowed("client_new") is True
    # Capacity must not exceed max_clients
    assert len(limiter._history) <= 5


def test_payload_validation_prompt_too_large():
    """Prompt exceeding 32KB must be rejected with HTTP 400 PAYLOAD_TOO_LARGE."""
    app = create_app(deployment_mode="local-demo")
    client = TestClient(app)

    oversized_prompt = "A" * 35_000
    res = client.post("/v1/respond", json={"prompt": oversized_prompt})
    assert res.status_code == 400
    assert res.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_payload_validation_invalid_constraints():
    """Invalid constraint ranges must be rejected with HTTP 400 INVALID_CONSTRAINTS."""
    app = create_app(deployment_mode="local-demo")
    client = TestClient(app)

    # Quality > 1.0
    res1 = client.post(
        "/v1/respond",
        json={"prompt": "Calculate 10*10", "minimum_quality": 1.5},
    )
    assert res1.status_code == 400
    assert res1.json()["code"] == "INVALID_CONSTRAINTS"

    # Negative budget
    res2 = client.post(
        "/v1/respond",
        json={"prompt": "Calculate 10*10", "maximum_cost_usd": -0.50},
    )
    assert res2.status_code == 400
    assert res2.json()["code"] == "INVALID_CONSTRAINTS"

    # Negative latency
    res3 = client.post(
        "/v1/respond",
        json={"prompt": "Calculate 10*10", "maximum_latency_ms": -100.0},
    )
    assert res3.status_code == 400
    assert res3.json()["code"] == "INVALID_CONSTRAINTS"
