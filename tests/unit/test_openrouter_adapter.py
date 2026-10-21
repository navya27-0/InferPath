"""Unit tests for OpenRouterAdapter: serialization, parsing, retries, cost, and budget cap."""

import json

import httpx
import pytest

from inferpath.config.settings import PricingTable
from inferpath.handlers.mock_adapter import MockSmallModel, ModelAdapter
from inferpath.handlers.openrouter_adapter import (
    OpenRouterAdapter,
    OpenRouterAuthError,
    OpenRouterRateLimitError,
)
from inferpath.models.enums import ProvenanceLabel
from inferpath.security.budget import BudgetExceededError, RunBudgetTracker


@pytest.fixture
def sample_pricing_table():
    return PricingTable(
        effective_date="2026-10-01",
        source="Test Pricing Table",
        rates_per_1m={
            "google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
            "anthropic/claude-3.7-sonnet": {"prompt": 3.0, "completion": 15.0},
        },
    )


def test_adapter_interchangeability(sample_pricing_table):
    """Confirm both mock and real adapters conform to ModelAdapter protocol."""
    mock_adapter = MockSmallModel(
        model_id="google/gemini-2.5-flash",
        prompt_price_per_1m=0.075,
        completion_price_per_1m=0.30,
        simulated_base_latency_ms=100.0,
    )
    real_adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        api_key="sk-test-key",
    )

    assert isinstance(mock_adapter, ModelAdapter)
    assert isinstance(real_adapter, ModelAdapter)


def test_openrouter_successful_generation(sample_pricing_table):
    """Tests successful OpenRouter completion parsing, latency measurement, and calculated cost."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "google/gemini-2.5-flash"
        assert body["messages"][0]["content"] == "Hello OpenRouter"

        # Verify Authorization header is sent
        assert request.headers["Authorization"] == "Bearer sk-test-key"

        mock_data = {
            "id": "gen-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello human!"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": 0.0000225,
            },
        }
        return httpx.Response(200, json=mock_data)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        api_key="sk-test-key",
        client=client,
    )

    result = adapter.generate("Hello OpenRouter")

    assert result.text == "Hello human!"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.source == ProvenanceLabel.MEASURED
    assert result.latency_ms > 0
    assert result.price_table_date == "2026-10-01"
    # Cost calculation: (100/1M)*0.075 + (50/1M)*0.30 = 0.0000075 + 0.000015 = 0.0000225
    assert result.cost_usd_calculated == pytest.approx(0.000023, abs=1e-5)
    assert result.provider_reported_cost_usd == 0.0000225


def test_openrouter_missing_usage_metadata(sample_pricing_table):
    """Provider response omitting usage metadata should not fabricate values or crash."""

    def handler(request: httpx.Request) -> httpx.Response:
        mock_data = {
            "choices": [{"message": {"role": "assistant", "content": "Output without usage"}}]
        }
        return httpx.Response(200, json=mock_data)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        api_key="sk-test-key",
        client=client,
    )

    result = adapter.generate("Test prompt")
    assert result.text == "Output without usage"
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.cost_usd_calculated is None
    assert result.usage_available is False
    assert result.provider_reported_cost_usd is None


def test_openrouter_authentication_error(sample_pricing_table):
    """HTTP 401 should immediately raise OpenRouterAuthError without retries."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, json={"error": "Invalid API Key"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        api_key="sk-bad-key",
        client=client,
        max_retries=2,
    )

    with pytest.raises(OpenRouterAuthError, match="Authentication failure"):
        adapter.generate("Test")

    # Verify no retries on auth failure
    assert call_count == 1


def test_openrouter_transient_retry_success(sample_pricing_table):
    """HTTP 502 should retry and succeed on subsequent attempt."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Recovered!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        api_key="sk-test-key",
        client=client,
        max_retries=2,
    )

    result = adapter.generate("Test retry")
    assert result.text == "Recovered!"
    assert call_count == 2


def test_openrouter_rate_limit_exhausted(sample_pricing_table):
    """Persistent HTTP 429 should exhaust retries and raise OpenRouterRateLimitError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "Rate limit reached"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        api_key="sk-test-key",
        client=client,
        max_retries=1,
    )

    with pytest.raises(OpenRouterRateLimitError):
        adapter.generate("Test rate limit")


def test_budget_cap_prevents_dispatch(sample_pricing_table):
    """Budget tracker prevents model dispatch when estimated call cost exceeds budget."""
    # Budget of $0.00001 (very small)
    tracker = RunBudgetTracker(max_usd=0.00001, pricing_table=sample_pricing_table)

    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    client = httpx.Client(transport=transport)

    # Large model estimated cost will easily exceed $0.00001
    adapter = OpenRouterAdapter(
        model_id="anthropic/claude-3.7-sonnet",
        pricing_table=sample_pricing_table,
        api_key="sk-test-key",
        client=client,
        budget_tracker=tracker,
    )

    with pytest.raises(BudgetExceededError, match="exceeds remaining budget"):
        adapter.generate("A complex prompt that would require substantial tokens")


@pytest.mark.asyncio
async def test_pipeline_openrouter_missing_usage_sync_and_async(sample_pricing_table):
    """Pipeline and cost accumulator gracefully handle missing provider usage without raising TypeError."""
    from inferpath.models.domain import UserRequest
    from inferpath.models.enums import PrivacyClass
    from inferpath.pipeline import RequestPipeline

    def handler(request: httpx.Request) -> httpx.Response:
        mock_data = {
            "id": "gen-missing-usage",
            "choices": [
                {"message": {"role": "assistant", "content": "Analysis of quantum state vector."}}
            ],
            # usage metadata intentionally omitted by provider
        }
        return httpx.Response(200, json=mock_data)

    transport = httpx.MockTransport(handler)
    sync_client = httpx.Client(transport=transport)
    async_client = httpx.AsyncClient(transport=transport)

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        api_key="sk-test-key",
        client=sync_client,
        async_client=async_client,
    )

    pipeline = RequestPipeline(
        pricing_table=sample_pricing_table,
        small_adapter=adapter,
        large_adapter=adapter,
    )

    req_sync = UserRequest(
        request_id="req_missing_usage_sync",
        prompt="Explain quantum entanglement in detail",
        privacy_class=PrivacyClass.PUBLIC,
    )

    # 1. Synchronous execution
    res_sync = pipeline.execute(req_sync)
    assert res_sync.final_output == "Analysis of quantum state vector."
    assert res_sync.cost_usd.value is None
    assert res_sync.cost_usd.provenance == ProvenanceLabel.UNAVAILABLE
    assert res_sync.actual_calculated_cost_usd.value is None
    assert res_sync.actual_calculated_cost_usd.provenance == ProvenanceLabel.UNAVAILABLE
    assert res_sync.input_tokens.value is None
    assert res_sync.input_tokens.provenance == ProvenanceLabel.UNAVAILABLE
    assert res_sync.output_tokens.value is None
    assert res_sync.output_tokens.provenance == ProvenanceLabel.UNAVAILABLE
    assert res_sync.energy_wh.value is None
    assert res_sync.energy_wh.provenance == ProvenanceLabel.UNAVAILABLE
    assert res_sync.water_ml.value is None
    assert res_sync.water_ml.provenance == ProvenanceLabel.UNAVAILABLE

    # 2. Asynchronous execution
    req_async = UserRequest(
        request_id="req_missing_usage_async",
        prompt="Explain quantum entanglement in detail",
        privacy_class=PrivacyClass.PUBLIC,
    )
    res_async = await pipeline.execute_async(req_async)
    assert res_async.final_output == "Analysis of quantum state vector."
    assert res_async.cost_usd.value is None
    assert res_async.cost_usd.provenance == ProvenanceLabel.UNAVAILABLE
    assert res_async.actual_calculated_cost_usd.value is None
    assert res_async.actual_calculated_cost_usd.provenance == ProvenanceLabel.UNAVAILABLE
    assert res_async.input_tokens.value is None
    assert res_async.input_tokens.provenance == ProvenanceLabel.UNAVAILABLE
    assert res_async.output_tokens.value is None
    assert res_async.output_tokens.provenance == ProvenanceLabel.UNAVAILABLE


def test_api_openrouter_missing_usage_returns_200_with_null_cost(sample_pricing_table):
    """FastAPI endpoint /v1/respond returns HTTP 200 with null cost and unavailable provenance."""
    from fastapi.testclient import TestClient

    from inferpath.api.main import create_app
    from inferpath.pipeline import RequestPipeline

    def handler(request: httpx.Request) -> httpx.Response:
        mock_data = {
            "id": "gen-missing-usage",
            "choices": [
                {"message": {"role": "assistant", "content": "Analysis of quantum state vector."}}
            ],
        }
        return httpx.Response(200, json=mock_data)

    transport = httpx.MockTransport(handler)
    sync_client = httpx.Client(transport=transport)
    async_client = httpx.AsyncClient(transport=transport)

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        api_key="sk-test-key",
        client=sync_client,
        async_client=async_client,
    )

    pipeline = RequestPipeline(
        pricing_table=sample_pricing_table,
        small_adapter=adapter,
        large_adapter=adapter,
    )

    app = create_app(pipeline=pipeline, deployment_mode="local-demo")
    client = TestClient(app)

    response = client.post(
        "/v1/respond",
        json={
            "prompt": "Explain quantum entanglement in detail",
            "privacy_class": "public",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Analysis of quantum state vector."
    assert body["cost_usd"] is None
    assert body["cost_provenance"] == "unavailable"
    assert body["actual_calculated_cost_usd"] is None
    assert body["input_tokens"] is None
    assert body["input_tokens_provenance"] == "unavailable"
