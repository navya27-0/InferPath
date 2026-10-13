"""Unit tests for FastAPI REST endpoints (/v1/respond, /v1/health)."""

import pytest
from fastapi.testclient import TestClient

from inferpath.api.main import create_app
from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable, Settings
from inferpath.handlers.mock_adapter import MockLargeModel, MockSmallModel
from inferpath.pipeline import RequestPipeline
from inferpath.security.budget import RunBudgetTracker


@pytest.fixture
def test_client(tmp_path):
    db_path = str(tmp_path / "api_test.db")
    settings = Settings()
    settings.database_path = db_path

    pricing_table = PricingTable(
        effective_date="2026-10-01",
        source="Test Pricing Table",
        rates_per_1m={
            "google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
            "anthropic/claude-3.7-sonnet": {"prompt": 3.0, "completion": 15.0},
        },
    )

    small_adapter = MockSmallModel(
        model_id="google/gemini-2.5-flash",
        prompt_price_per_1m=0.075,
        completion_price_per_1m=0.30,
        simulated_base_latency_ms=40.0,
    )
    large_adapter = MockLargeModel(
        model_id="anthropic/claude-3.7-sonnet",
        prompt_price_per_1m=3.0,
        completion_price_per_1m=15.0,
        simulated_base_latency_ms=120.0,
    )

    pipeline = RequestPipeline(
        settings=settings,
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
        small_adapter=small_adapter,
        large_adapter=large_adapter,
    )

    app = create_app(pipeline=pipeline)
    return TestClient(app)


def test_health_check(test_client):
    response = test_client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_respond_successful_deterministic(test_client):
    payload = {
        "request_id": "api_req_1",
        "prompt": "Convert 25°C to Fahrenheit",
        "privacy_class": "public",
    }
    response = test_client.post("/v1/respond", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["request_id"] == "api_req_1"
    assert data["text"] == "77°F"
    assert data["route"] == "deterministic_a"
    assert data["rule_id"] == "R04"
    assert data["validated"] is True
    assert data["cost_usd"] == 0.0
    assert data["cost_provenance"] == "calculated"
    assert data["latency_provenance"] == "measured"
    assert isinstance(data["route_trace"], list)


def test_respond_template_clarification(test_client):
    payload = {
        "request_id": "api_req_2",
        "prompt": "Draft a leave application for Bob for medical leave due to dental surgery",
        "privacy_class": "public",
    }
    response = test_client.post("/v1/respond", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["needs_clarification"] is True
    assert "start_date" in data["text"]
    assert "end_date" in data["text"]


def test_respond_privacy_rejection(test_client):
    payload = {
        "request_id": "api_req_3",
        "prompt": "Summarize confidential employee contracts",
        "privacy_class": "confidential_private",
    }
    response = test_client.post("/v1/respond", json=payload)
    assert response.status_code == 403
    assert response.json()["code"] == "PRIVACY_BLOCKED"


def test_respond_empty_prompt_bad_request(test_client):
    payload = {
        "request_id": "api_req_4",
        "prompt": "   ",
        "privacy_class": "public",
    }
    response = test_client.post("/v1/respond", json=payload)
    assert response.status_code == 400


def test_respond_budget_cap_exceeded(tmp_path):
    """Budget cap violation returns HTTP 402."""
    db_path = str(tmp_path / "budget_api.db")
    settings = Settings()
    settings.database_path = db_path

    pricing_table = PricingTable(
        effective_date="2026-10-01",
        source="Test Pricing Table",
        rates_per_1m={
            "google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
            "anthropic/claude-3.7-sonnet": {"prompt": 3.0, "completion": 15.0},
        },
    )

    # Budget tracker with zero remaining budget
    zero_budget = RunBudgetTracker(max_usd=0.000001, pricing_table=pricing_table)
    # Deplete budget
    zero_budget.spent_usd = 0.000001

    pipeline = RequestPipeline(
        settings=settings,
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
        budget_tracker=zero_budget,
    )

    client = TestClient(create_app(pipeline=pipeline))
    payload = {
        "request_id": "api_req_budget",
        "prompt": "Synthesize deep mathematical concepts",
        "privacy_class": "public",
    }
    response = client.post("/v1/respond", json=payload)
    assert response.status_code == 402
    assert response.json()["code"] == "BUDGET_EXCEEDED"
