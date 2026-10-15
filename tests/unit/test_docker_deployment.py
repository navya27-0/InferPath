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

    # Safe default deployment mode
    assert any("INFERPATH_DEPLOYMENT_MODE=local-demo" in line for line in lines), (
        "Dockerfile must default to local-demo mode"
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
    assert services["api"]["healthcheck"]["test"] == [
        "CMD",
        "curl",
        "-f",
        "http://localhost:8000/health",
    ]


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
