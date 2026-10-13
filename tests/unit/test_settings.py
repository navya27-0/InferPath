"""Unit tests for configuration loading, dated pricing tables, and carbon bounds."""

from inferpath.config.settings import (
    CarbonBoundConfig,
    PricingTable,
    Settings,
)


def test_settings_load_successfully(real_settings: Settings):
    """Verifies that base settings load with valid defaults, cache TTLs, and thresholds."""
    assert real_settings.environment == "research"
    assert real_settings.openrouter.base_url == "https://openrouter.ai/api/v1"
    assert real_settings.openrouter.small_model == "google/gemini-2.5-flash"
    assert real_settings.openrouter.large_model == "anthropic/claude-3.7-sonnet"
    assert real_settings.routing.enable_cache is True
    assert real_settings.routing.enable_fallback is True

    # Cache TTLs and thresholds validation
    assert "deterministic_a" in real_settings.routing.cache_ttls
    assert real_settings.routing.cache_ttls["deterministic_a"] == 0
    assert "confidence_threshold" in real_settings.routing.thresholds
    assert real_settings.routing.thresholds["confidence_threshold"] >= 0.5


def test_pricing_table_load(real_pricing: PricingTable):
    """Verifies that dated pricing table loads dated reference rates and honest status."""
    assert real_pricing.effective_date == "2026-10-01"
    assert real_pricing.verification_status == "unverified_illustrative"
    assert "google/gemini-2.5-flash" in real_pricing.rates_per_1m
    assert "anthropic/claude-3.7-sonnet" in real_pricing.rates_per_1m

    # Tier A & B cost $0
    assert real_pricing.rates_per_1m["deterministic_a"]["prompt"] == 0.0
    assert real_pricing.rates_per_1m["template_b"]["prompt"] == 0.0

    # Large model is strictly more expensive than small model
    small_prompt = real_pricing.rates_per_1m["google/gemini-2.5-flash"]["prompt"]
    large_prompt = real_pricing.rates_per_1m["anthropic/claude-3.7-sonnet"]["prompt"]
    assert large_prompt > small_prompt


def test_carbon_bounds_load(real_carbon_bounds: CarbonBoundConfig):
    """Verifies system boundary definitions, model coefficients, units, and source notes."""
    assert real_carbon_bounds.effective_date == "2026-10-01"
    assert real_carbon_bounds.verification_status == "unverified_illustrative"
    assert "Scope 2" in real_carbon_bounds.system_boundary_energy.included
    assert "Scope 3" in real_carbon_bounds.system_boundary_energy.excluded
    assert (
        real_carbon_bounds.system_boundary_water.notes
        == "Water is never a headline claim; reported as secondary metric only."
    )

    # PUE assumption validation
    pue = real_carbon_bounds.datacenter_pue
    assert pue.unit == "ratio"
    assert pue.low <= pue.central <= pue.high
    assert pue.source_note != ""
    assert pue.verification_status == "unverified_illustrative"

    # Coefficients validation
    coeffs = real_carbon_bounds.coefficients_per_1000_tokens
    assert "google/gemini-2.5-flash" in coeffs
    assert "anthropic/claude-3.7-sonnet" in coeffs
    assert coeffs["google/gemini-2.5-flash"].unit == "Wh / 1000 tokens"
    assert coeffs["google/gemini-2.5-flash"].source_note != ""
    assert coeffs["google/gemini-2.5-flash"].central < coeffs["anthropic/claude-3.7-sonnet"].central

    # Water intensity assumption validation
    water = real_carbon_bounds.water_intensity
    assert water.unit == "mL / Wh"
    assert water.low <= water.central <= water.high
    assert water.source_note != ""
