"""Unit tests for model registry and drift tracking."""

from inferpath.models.enums import ProvenanceLabel, RouteTier
from inferpath.models.registry import ModelRegistry


def test_model_registry_initialization():
    registry = ModelRegistry()
    flash = registry.get_model("google/gemini-2.5-flash")
    assert flash is not None
    assert flash.tier_mapping == RouteTier.SMALL_MODEL_C
    assert flash.is_deprecated is False
    assert flash.input_cost_per_1m == 0.075
    assert flash.output_cost_per_1m == 0.30
    assert flash.price_provenance == ProvenanceLabel.CALCULATED
    assert flash.latency_provenance == ProvenanceLabel.ESTIMATED

    sonnet = registry.get_model("anthropic/claude-3.7-sonnet")
    assert sonnet is not None
    assert sonnet.tier_mapping == RouteTier.LARGE_MODEL_D
    assert sonnet.input_cost_per_1m == 3.00
    assert sonnet.output_cost_per_1m == 15.00
    assert sonnet.price_provenance == ProvenanceLabel.CALCULATED

    mock_s = registry.get_model("mock-small")
    assert mock_s is not None
    assert mock_s.price_provenance == ProvenanceLabel.SIMULATED


def test_model_deprecation():
    registry = ModelRegistry()
    registry.deprecate_model("google/gemini-2.5-flash", "google/gemini-3.0-flash")
    model = registry.get_model("google/gemini-2.5-flash")
    assert model.is_deprecated is True
    assert model.replacement_model_id == "google/gemini-3.0-flash"


def test_model_price_drift():
    registry = ModelRegistry()
    initial_cost = registry.get_model("google/gemini-2.5-flash").input_cost_per_1m
    registry.simulate_price_drift("google/gemini-2.5-flash", 0.5)
    new_cost = registry.get_model("google/gemini-2.5-flash").input_cost_per_1m
    assert new_cost == initial_cost * 0.5
