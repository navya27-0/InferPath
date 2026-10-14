"""Unit tests for OpenRouter Circuit Breaker fail-fast behavior."""

import time

import pytest

from inferpath.config.settings import PricingTable
from inferpath.handlers.openrouter_adapter import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    OpenRouterAdapter,
)


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


def test_circuit_breaker_state_transitions():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=0.1)

    assert cb.state == "CLOSED"
    assert cb.can_execute() is True

    # 1 failure -> still CLOSED
    cb.record_failure()
    assert cb.state == "CLOSED"
    assert cb.can_execute() is True

    # 2 failures -> still CLOSED
    cb.record_failure()
    assert cb.state == "CLOSED"

    # 3 failures -> trips to OPEN
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_execute() is False

    # Wait for recovery timeout -> transitions to HALF_OPEN
    time.sleep(0.15)
    assert cb.can_execute() is True
    assert cb.state == "HALF_OPEN"

    # Success in HALF_OPEN resets to CLOSED
    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.consecutive_failures == 0


def test_adapter_fail_fast_when_circuit_open(sample_pricing_table):
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=10.0)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "OPEN"

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=sample_pricing_table,
        circuit_breaker=cb,
    )

    # Dispatch must immediately fail fast without touching network
    with pytest.raises(CircuitBreakerOpenError, match="Circuit breaker is OPEN"):
        adapter.generate("Hello world")
