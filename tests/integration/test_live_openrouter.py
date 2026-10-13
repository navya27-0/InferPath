"""Opt-in live integration test against OpenRouter API.

NOTE: This test is marked 'live' and requires both OPENROUTER_API_KEY and INFERPATH_LIVE_TESTS=1.
It is strictly skipped in CI and offline test suites.
"""

import os

import pytest

from inferpath.config.settings import load_pricing_table, load_settings
from inferpath.handlers.openrouter_adapter import OpenRouterAdapter
from inferpath.models.enums import ProvenanceLabel


@pytest.mark.live
def test_live_openrouter_completion():
    """Live billable test verifying OpenRouter API connectivity with strict timeout and cost bounds."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    opt_in = os.getenv("INFERPATH_LIVE_TESTS", "").strip() in ("1", "true", "yes")

    if not api_key or not opt_in:
        pytest.skip(
            "Live integration test skipped: requires OPENROUTER_API_KEY and INFERPATH_LIVE_TESTS=1."
        )

    settings = load_settings()
    pricing = load_pricing_table()

    # Use small model and short prompt to minimize cost
    adapter = OpenRouterAdapter(
        model_id=settings.openrouter.small_model,
        pricing_table=pricing,
        api_key=api_key,
        base_url=settings.openrouter.base_url,
        timeout_seconds=15.0,
        max_retries=1,
    )

    result = adapter.generate("Reply with one word: hello", max_tokens=5)

    assert result.text and len(result.text.strip()) > 0
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms > 0
    assert result.source == ProvenanceLabel.MEASURED
    assert result.cost_usd_calculated > 0.0
