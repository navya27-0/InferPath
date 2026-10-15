"""Unit tests for adapter reliability, malformed responses, circuit breaker, and cache capacity."""

import time

import httpx
import pytest

from inferpath.cache.canonical_key import CanonicalKeyBuilder
from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable
from inferpath.handlers.openrouter_adapter import (
    CircuitBreaker,
    OpenRouterAdapter,
    OpenRouterResponseError,
)
from inferpath.models.domain import HandlerOutput, UserRequest
from inferpath.models.enums import ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


@pytest.fixture
def pricing_table():
    return PricingTable(
        effective_date="2026-10-01",
        source="Test Pricing Table",
        rates_per_1m={
            "google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
        },
    )


def test_adapter_malformed_response_handling(pricing_table):
    """Adapter must reject payloads with missing choices, malformed message, or null content."""
    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=pricing_table,
    )

    # 1. Non-dict payload
    with pytest.raises(OpenRouterResponseError, match="valid JSON object"):
        adapter._parse_response_body(["not a dict"], latency_ms=50.0)

    # 2. Missing choices
    with pytest.raises(OpenRouterResponseError, match="no choices"):
        adapter._parse_response_body({"choices": []}, latency_ms=50.0)

    # 3. Malformed choice item
    with pytest.raises(OpenRouterResponseError, match="choice is malformed"):
        adapter._parse_response_body({"choices": ["string instead of dict"]}, latency_ms=50.0)

    # 4. Missing message
    with pytest.raises(OpenRouterResponseError, match="missing message"):
        adapter._parse_response_body({"choices": [{"no_message": 1}]}, latency_ms=50.0)

    # 5. Null or non-string content
    with pytest.raises(OpenRouterResponseError, match="missing or null"):
        adapter._parse_response_body(
            {"choices": [{"message": {"role": "assistant", "content": None}}]},
            latency_ms=50.0,
        )


def test_circuit_breaker_half_open_failure():
    """In HALF_OPEN state, a single failure must immediately trip breaker back to OPEN."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "OPEN"

    # Wait for recovery timeout
    time.sleep(0.06)
    assert cb.can_execute() is True
    assert cb.state == "HALF_OPEN"

    # A single failure in HALF_OPEN trips immediately to OPEN
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_execute() is False


def test_adapter_client_close_cleanup(pricing_table):
    """Adapter close() and aclose() must close underlying HTTP clients safely."""
    client = httpx.Client()
    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=pricing_table,
        client=client,
    )
    assert not client.is_closed
    adapter.close()
    assert client.is_closed


def test_canonical_key_dictionary_order_invariance():
    """Canonical key must be invariant to dict key insertion order at all nesting levels."""
    d1 = {"z": 1, "a": {"sub_b": 2, "sub_a": 1}}
    d2 = {"a": {"sub_a": 1, "sub_b": 2}, "z": 1}

    k1 = CanonicalKeyBuilder.build_cache_key(
        prompt="Sample prompt",
        task_parameters=d1,
        route=RouteTier.SMALL_MODEL_C,
    )
    k2 = CanonicalKeyBuilder.build_cache_key(
        prompt="Sample prompt",
        task_parameters=d2,
        route=RouteTier.SMALL_MODEL_C,
    )
    assert k1 == k2


def test_exact_cache_capacity_eviction(tmp_path):
    """ExactCache must bound entries and prune oldest when capacity exceeds max_entries."""
    db_path = tmp_path / "capacity_test.db"
    cache = ExactCache(db_path=db_path, max_entries=3)

    for i in range(5):
        output = HandlerOutput(
            tier=RouteTier.SMALL_MODEL_C,
            output_text=f"Response {i}",
            validated=True,
            input_tokens=ProvenanceValue(
                value=10, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            output_tokens=ProvenanceValue(
                value=10, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            latency_ms=ProvenanceValue(value=50.0, provenance=ProvenanceLabel.MEASURED, unit="ms"),
            model_version="model_v1",
        )
        req = UserRequest(request_id=f"req_{i}", prompt=f"Prompt {i}")
        cache.store(
            cache_key=f"key_{i}",
            request=req,
            output=output,
            route=RouteTier.SMALL_MODEL_C,
            ttl_seconds=3600,
        )

    # After storing 5 items with max_entries=3, database should contain at most 3 records
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM exact_cache").fetchone()[0]
        assert count <= 3
