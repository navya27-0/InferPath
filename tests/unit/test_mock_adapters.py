"""Unit tests verifying deterministic mock model adapters and simulated provenance."""

import pytest

from inferpath.handlers.mock_adapter import (
    MockLargeModel,
    MockSmallModel,
    ModelAdapter,
)
from inferpath.models.domain import ModelResult
from inferpath.models.enums import ProvenanceLabel


def test_mock_adapters_implement_protocol():
    """Confirms MockSmallModel and MockLargeModel satisfy the ModelAdapter protocol."""
    small = MockSmallModel()
    large = MockLargeModel()

    assert isinstance(small, ModelAdapter)
    assert isinstance(large, ModelAdapter)
    assert small.model_id == "mock-small"
    assert large.model_id == "mock-large"


def test_mock_small_model_deterministic_generation():
    """Verifies that MockSmallModel produces deterministic canned outputs labelled simulated."""
    model = MockSmallModel(price_table_date="2026-10-01")
    prompt = "Convert 25 Celsius to Fahrenheit"

    res1 = model.generate(prompt)
    res2 = model.generate(prompt)

    # Determinism
    assert res1.text == res2.text
    assert "77°F" in res1.text
    assert res1.input_tokens == res2.input_tokens
    assert res1.output_tokens == res2.output_tokens
    assert res1.latency_ms == res2.latency_ms
    assert res1.cost_usd_calculated == res2.cost_usd_calculated

    # Strict provenance: source must be SIMULATED
    assert res1.source == ProvenanceLabel.SIMULATED
    assert res1.provider_reported_cost_usd is None
    assert res1.price_table_date == "2026-10-01"


def test_mock_large_model_deterministic_generation():
    """Verifies that MockLargeModel produces deterministic outputs with simulated provenance."""
    model = MockLargeModel(price_table_date="2026-10-01")
    prompt = "Summarize the key advantages of multi-tier inference routing"

    res = model.generate(prompt)
    assert isinstance(res, ModelResult)
    assert res.source == ProvenanceLabel.SIMULATED
    assert res.model_id == "mock-large"
    assert res.cost_usd_calculated > 0.0
    assert res.latency_ms > 0.0


def test_mock_large_model_higher_cost_and_latency_than_small():
    """Large model must simulate higher latency and calculated cost than small model."""
    small = MockSmallModel()
    large = MockLargeModel()
    prompt = "Explain quantum entanglement in simple terms"

    res_small = small.generate(prompt)
    res_large = large.generate(prompt)

    assert res_large.cost_usd_calculated > res_small.cost_usd_calculated
    assert res_large.latency_ms > res_small.latency_ms
    assert res_small.source == ProvenanceLabel.SIMULATED
    assert res_large.source == ProvenanceLabel.SIMULATED


@pytest.mark.asyncio
async def test_mock_model_async_execution():
    """Verifies asynchronous generation wrapper operates deterministically."""
    model = MockSmallModel()
    prompt = "What is the capital of France?"

    res = await model.generate_async(prompt)
    assert "Paris" in res.text
    assert res.source == ProvenanceLabel.SIMULATED
