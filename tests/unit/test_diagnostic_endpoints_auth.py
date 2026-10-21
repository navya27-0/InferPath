"""Tests for diagnostic endpoints (/health/dependencies, /metrics) authentication and host binding detection."""

import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from inferpath.api.main import create_app
from inferpath.config.deployment_profiles import get_deployment_profile
from inferpath.pipeline import RequestPipeline


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock(spec=RequestPipeline)
    pipeline.cache = MagicMock()
    pipeline.cache.get_schema_version.return_value = 2
    return pipeline


def test_protected_profile_requires_auth_for_diagnostics_and_metrics(mock_pipeline):
    """Under protected profiles, /health/dependencies and /metrics require valid API keys."""
    app = create_app(
        pipeline=mock_pipeline,
        api_key="operator_secret_key_123",
        deployment_mode="production",
        deployment_profile=get_deployment_profile("production"),
    )
    client = TestClient(app)

    # 1. Minimal public liveness is accessible without credentials
    live_resp = client.get("/health/live")
    assert live_resp.status_code == 200
    assert live_resp.json()["status"] == "alive"

    # 2. Public health check without credentials returns minimal status (no leaked internals)
    health_anon = client.get("/health")
    assert health_anon.status_code == 200
    anon_data = health_anon.json()
    assert "status" in anon_data
    assert "service" in anon_data
    assert "rate_limiter" not in anon_data
    assert "profile" not in anon_data

    # 3. Authenticated health check returns complete operational details
    health_auth = client.get("/health", headers={"X-API-Key": "operator_secret_key_123"})
    assert health_auth.status_code == 200
    auth_data = health_auth.json()
    assert "rate_limiter" in auth_data
    assert "profile" in auth_data

    # 4. /health/dependencies without credentials rejected with 401
    deps_anon = client.get("/health/dependencies")
    assert deps_anon.status_code == 401
    assert deps_anon.json()["code"] == "UNAUTHORIZED"

    # 5. /health/dependencies with valid credentials succeeds
    deps_auth = client.get("/health/dependencies", headers={"X-API-Key": "operator_secret_key_123"})
    assert deps_auth.status_code == 200
    assert deps_auth.json()["status"] == "healthy"

    # 6. /metrics without credentials rejected with 401
    metrics_anon = client.get("/metrics")
    assert metrics_anon.status_code == 401
    assert metrics_anon.json()["code"] == "UNAUTHORIZED"

    # 7. /metrics with Bearer token succeeds
    metrics_auth = client.get(
        "/metrics", headers={"Authorization": "Bearer operator_secret_key_123"}
    )
    assert metrics_auth.status_code == 200
    assert "inferpath_" in metrics_auth.text


def test_local_demo_profile_allows_unauthenticated_operator_diagnostics(mock_pipeline):
    """Under local-demo profile without configured key, local diagnostic inspection is permitted."""
    app = create_app(
        pipeline=mock_pipeline,
        api_key="",
        deployment_mode="local-demo",
        deployment_profile=get_deployment_profile("local-demo"),
    )
    client = TestClient(app)

    deps_resp = client.get("/health/dependencies")
    assert deps_resp.status_code == 200
    assert deps_resp.json()["status"] == "healthy"

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200


def test_cli_host_detection_rejects_unauthenticated_public_binding(monkeypatch):
    """Passing --host 0.0.0.0 on CLI halts startup under unauthenticated profiles."""
    monkeypatch.delenv("INFERPATH_HOST_BIND", raising=False)
    monkeypatch.delenv("HOST", raising=False)

    # Simulate sys.argv containing --host 0.0.0.0
    monkeypatch.setattr(
        sys,
        "argv",
        ["uvicorn", "inferpath.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )

    with pytest.raises(
        RuntimeError, match="Binding to public interface '0.0.0.0' is strictly prohibited"
    ):
        create_app(deployment_mode="local-demo", api_key="")


def test_cli_host_detection_allows_loopback_binding(monkeypatch):
    """Passing --host 127.0.0.1 on CLI allows unauthenticated local demo."""
    monkeypatch.delenv("INFERPATH_HOST_BIND", raising=False)
    monkeypatch.delenv("HOST", raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["uvicorn", "inferpath.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
    )

    app = create_app(deployment_mode="local-demo", api_key="")
    assert app is not None
