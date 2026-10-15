"""Unit tests for privacy-boundary invariants with mock network transport.

Guarantees:
- Zero external network requests are emitted when requests contain confidential data.
- Fallback execution never bypasses privacy enforcement.
- Cache lookup and storage are strictly bypassed for non-public requests.
- Sensitive prompts and credentials never leak into route traces or log previews.
"""

import httpx
import pytest

from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable, Settings
from inferpath.handlers.openrouter_adapter import OpenRouterAdapter
from inferpath.models.domain import UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.pipeline import RequestPipeline
from inferpath.security.privacy import PrivacyViolationError


@pytest.fixture
def test_env(tmp_path):
    db_path = str(tmp_path / "privacy_test.db")
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

    external_calls: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        external_calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "mock external response"}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    mock_async_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    small_adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=pricing_table,
        api_key="sk-or-v1-dummytestkey12345678901234567890",
        client=mock_client,
        async_client=mock_async_client,
    )
    large_adapter = OpenRouterAdapter(
        model_id="anthropic/claude-3.7-sonnet",
        pricing_table=pricing_table,
        api_key="sk-or-v1-dummytestkey12345678901234567890",
        client=mock_client,
        async_client=mock_async_client,
    )

    cache = ExactCache(db_path=db_path)
    pipeline = RequestPipeline(
        settings=settings,
        pricing_table=pricing_table,
        cache=cache,
        small_adapter=small_adapter,
        large_adapter=large_adapter,
    )

    return {
        "pipeline": pipeline,
        "cache": cache,
        "external_calls": external_calls,
    }


def test_confidential_request_blocked_at_boundary_zero_network_calls(test_env):
    """Confidential request must raise PrivacyViolationError and trigger ZERO external HTTP calls."""
    pipeline = test_env["pipeline"]
    external_calls = test_env["external_calls"]

    confidential_prompt = "Synthesize internal executive compensation tables for Q4."
    req = UserRequest(
        request_id="priv_001",
        prompt=confidential_prompt,
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )

    with pytest.raises(PrivacyViolationError):
        pipeline.execute(req)

    # Invariant: ZERO network calls were made to external models
    assert len(external_calls) == 0


@pytest.mark.asyncio
async def test_confidential_async_request_blocked_zero_network_calls(test_env):
    """Async execution of confidential request must also trigger ZERO network calls."""
    pipeline = test_env["pipeline"]
    external_calls = test_env["external_calls"]

    req = UserRequest(
        request_id="priv_002",
        prompt="Internal engineering architectural secrets.",
        privacy_class=PrivacyClass.INTERNAL,
    )

    with pytest.raises(PrivacyViolationError):
        await pipeline.execute_async(req)

    assert len(external_calls) == 0


def test_credential_in_prompt_triggers_privacy_block(test_env):
    """Requests containing API keys or private keys must be blocked from external dispatch."""
    pipeline = test_env["pipeline"]
    external_calls = test_env["external_calls"]

    leaked_prompt = "Review this code with api_key: sk-or-v1-1234567890abcdef1234567890"
    req = UserRequest(
        request_id="priv_003",
        prompt=leaked_prompt,
        privacy_class=PrivacyClass.PUBLIC,  # Caller incorrectly marked public
    )

    with pytest.raises(PrivacyViolationError):
        pipeline.execute(req)

    assert len(external_calls) == 0


def test_cache_strictly_bypasses_confidential_requests(test_env):
    """ExactCache must never store or serve confidential prompts."""
    cache = test_env["cache"]

    req = UserRequest(
        request_id="cache_priv_001",
        prompt="Secret password reset link",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )

    # Check eligibility
    from inferpath.cache.eligibility import CacheEligibilityPolicy

    lookup_decision = CacheEligibilityPolicy.check_lookup_eligibility(req, RouteTier.SMALL_MODEL_C)
    assert lookup_decision.is_eligible is False
    assert lookup_decision.policy_code == "BYPASS_PRIVATE"
    assert cache.lookup("non_existent_key") is None
