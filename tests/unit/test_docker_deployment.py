"""Regression tests for Docker container deployment, host binding, and mode security."""

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from inferpath.api.main import create_app
from inferpath.config.settings import load_pricing_table
from inferpath.pipeline import RequestPipeline


def test_dockerfile_syntax_and_security_configuration():
    """Verify Dockerfile uses Python 3.11, exposes required ports, and sets safe defaults."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"

    content = dockerfile_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    # Base image verification
    assert any("python:3.11" in line for line in lines), "Dockerfile must use Python 3.11 base"

    # Port exposure
    assert any("8000" in line for line in lines), "Dockerfile must expose port 8000 (API)"
    assert any("8501" in line for line in lines), "Dockerfile must expose port 8501 (Dashboard)"

    # Safe default deployment mode and loopback host binding
    assert any("INFERPATH_DEPLOYMENT_MODE=local-demo" in line for line in lines), (
        "Dockerfile must default to local-demo mode"
    )
    assert any("INFERPATH_HOST_BIND=127.0.0.1" in line for line in lines), (
        "Dockerfile must default to loopback host bind (127.0.0.1)"
    )
    assert any("INFERPATH_CONTAINER_DEPLOYMENT=true" in line for line in lines), (
        "Dockerfile must default to INFERPATH_CONTAINER_DEPLOYMENT=true"
    )
    assert any("uvicorn" in line and "--port" in line for line in lines if "CMD" in line), (
        "Dockerfile default command must launch Uvicorn"
    )


def test_docker_compose_host_binding_localhost_default():
    """docker-compose.yml must bind to 127.0.0.1 by default to prevent broad network exposure."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml must exist at repository root"

    with open(compose_path, encoding="utf-8") as f:
        compose_cfg = yaml.safe_load(f)

    assert "services" in compose_cfg
    services = compose_cfg["services"]

    # Verify API service
    assert "api" in services
    api_ports = services["api"]["ports"]
    assert any("127.0.0.1" in port_mapping for port_mapping in api_ports), (
        "API service must bind to 127.0.0.1 by default"
    )

    # Verify Dashboard service
    assert "dashboard" in services
    dashboard_ports = services["dashboard"]["ports"]
    assert any("127.0.0.1" in port_mapping for port_mapping in dashboard_ports), (
        "Dashboard service must bind to 127.0.0.1 by default"
    )

    # Verify healthchecks are defined
    assert "healthcheck" in services["api"]
    assert services["api"]["healthcheck"]["test"] in (
        ["CMD", "curl", "-f", "http://localhost:8000/health"],
        ["CMD", "curl", "-f", "http://localhost:8000/health/live"],
    )


def test_api_deployment_mode_protected_enforces_auth():
    """In protected deployment mode, API rejects requests lacking valid API key with HTTP 401."""
    pricing = load_pricing_table()
    pipeline = RequestPipeline(pricing_table=pricing)

    # Initialize app in protected mode
    app = create_app(
        pipeline=pipeline,
        deployment_mode="protected",
        api_key="secret-prod-key-12345",
    )
    client = TestClient(app)

    # Missing API key -> 401
    res_no_key = client.post("/v1/respond", json={"prompt": "test query"})
    assert res_no_key.status_code == 401
    assert res_no_key.json()["code"] == "UNAUTHORIZED"

    # Wrong API key -> 401
    res_wrong_key = client.post(
        "/v1/respond",
        json={"prompt": "test query"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert res_wrong_key.status_code == 401

    # Valid API key -> allowed
    res_valid_key = client.post(
        "/v1/respond",
        json={"prompt": "test query"},
        headers={"X-API-Key": "secret-prod-key-12345"},
    )
    assert res_valid_key.status_code == 200


def test_api_deployment_mode_local_demo_allows_unauthenticated():
    """In local-demo deployment mode, requests without API key succeed."""
    pricing = load_pricing_table()
    pipeline = RequestPipeline(pricing_table=pricing)

    app = create_app(
        pipeline=pipeline,
        deployment_mode="local-demo",
        api_key="",
    )
    client = TestClient(app)

    res = client.post("/v1/respond", json={"prompt": "Calculate 15 * 4"})
    assert res.status_code == 200
    assert res.json()["route"] == "deterministic_a"


def test_docker_compose_local_demo_configuration():
    """docker-compose.local-demo.yml must be self-starting with 127.0.0.1 binding and simulated adapters."""
    compose_path = Path("docker-compose.local-demo.yml")
    assert compose_path.exists(), "docker-compose.local-demo.yml must exist"

    with open(compose_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "services" in cfg
    services = cfg["services"]

    # API verification
    assert "api" in services
    api_ports = services["api"]["ports"]
    assert any("127.0.0.1:8000" in p for p in api_ports)
    api_env = services["api"]["environment"]
    assert "INFERPATH_DEPLOYMENT_MODE=local-demo" in api_env
    assert "ADAPTER_MODE=simulated" in api_env
    assert "INFERPATH_HOST_BIND=127.0.0.1" in api_env
    assert "INFERPATH_CONTAINER_DEPLOYMENT=true" in api_env

    # Dashboard verification
    assert "dashboard" in services
    dash_ports = services["dashboard"]["ports"]
    assert any("127.0.0.1:8501" in p for p in dash_ports)
    dash_env = services["dashboard"]["environment"]
    assert "INFERPATH_DEPLOYMENT_MODE=local-demo" in dash_env
    assert "INFERPATH_HOST_BIND=127.0.0.1" in dash_env
    assert "INFERPATH_CONTAINER_DEPLOYMENT=true" in dash_env


def test_local_demo_container_deployment_startup_and_health(monkeypatch):
    """Under container deployment, internal 0.0.0.0 bind is accepted when host bind is 127.0.0.1."""
    import sys

    monkeypatch.setenv("INFERPATH_CONTAINER_DEPLOYMENT", "true")
    monkeypatch.setenv("INFERPATH_HOST_BIND", "127.0.0.1")
    monkeypatch.setenv("INFERPATH_DEPLOYMENT_MODE", "local-demo")
    monkeypatch.delenv("INFERPATH_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["uvicorn", "inferpath.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )

    app = create_app(deployment_mode="local-demo", api_key="")
    client = TestClient(app)
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_local_demo_container_deployment_public_bind_fails_closed(monkeypatch):
    """Under container deployment, publishing to 0.0.0.0 without API key halts startup."""
    import sys

    import pytest

    monkeypatch.setenv("INFERPATH_CONTAINER_DEPLOYMENT", "true")
    monkeypatch.setenv("INFERPATH_HOST_BIND", "0.0.0.0")
    monkeypatch.setenv("INFERPATH_DEPLOYMENT_MODE", "local-demo")
    monkeypatch.delenv("INFERPATH_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["uvicorn", "inferpath.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )

    with pytest.raises(
        RuntimeError, match="Binding to public interface '0.0.0.0' is strictly prohibited"
    ):
        create_app(deployment_mode="local-demo", api_key="")
