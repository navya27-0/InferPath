"""Unit tests verifying Prompt 1 schema requirements and field exposures."""

from inferpath.models.domain import (
    ModelResult,
    Request,
    Response,
    RouteDecision,
    RunRecord,
    TaskParameters,
)
from inferpath.models.enums import PrivacyClass, ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


def test_request_schema_exposes_prompt1_fields():
    """Confirms Request exposes quality, format, language, latency, privacy_class, and preferences."""
    params = TaskParameters(
        expected_format="json",
        language="en",
        quality_preference="high",
        max_latency_ms=250.0,
    )
    req = Request(
        request_id="req-p1-001",
        prompt="Synthesize research data",
        context="Academic corpus",
        task_parameters=params,
        privacy_class=PrivacyClass.PUBLIC,
    )

    assert req.prompt == "Synthesize research data"
    assert req.context == "Academic corpus"
    assert req.quality == "high"
    assert req.format == "json"
    assert req.language == "en"
    assert req.latency == 250.0
    assert req.privacy_class == PrivacyClass.PUBLIC

    prefs = req.preferences
    assert prefs["quality"] == "high"
    assert prefs["format"] == "json"
    assert prefs["language"] == "en"
    assert prefs["latency"] == 250.0
    assert prefs["privacy_class"] == "public"


def test_route_decision_exposes_prompt1_fields():
    """Confirms RouteDecision exposes route, reason, rule_id, and confidence."""
    decision = RouteDecision(
        selected_tier=RouteTier.SMALL_MODEL_C,
        route=RouteTier.SMALL_MODEL_C,
        rule_id="R03",
        reason="Moderate complexity summarization task",
        confidence=0.88,
    )

    assert decision.route == RouteTier.SMALL_MODEL_C
    assert decision.selected_tier == RouteTier.SMALL_MODEL_C
    assert decision.rule_id == "R03"
    assert decision.reason == "Moderate complexity summarization task"
    assert decision.confidence == 0.88


def test_response_and_run_record_expose_prompt1_fields():
    """Confirms Response and RunRecord expose text, route, cache_hit, validated, fallback_used."""
    resp = Response(
        tier=RouteTier.TEMPLATE_B,
        output_text="Formal Leave Application Form",
        input_tokens=ProvenanceValue(value=10, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        output_tokens=ProvenanceValue(value=30, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        latency_ms=ProvenanceValue(value=1.5, provenance=ProvenanceLabel.MEASURED, unit="ms"),
        model_version="template-v1",
        cache_hit=False,
        validated=True,
        fallback_used=False,
    )

    assert resp.text == "Formal Leave Application Form"
    assert resp.route == RouteTier.TEMPLATE_B
    assert resp.cache_hit is False
    assert resp.validated is True
    assert resp.fallback_used is False

    record = RunRecord(
        request_id="req-rec-001",
        prompt_hash="abc123hash",
        final_output="Final Output Content",
        initial_tier=RouteTier.SMALL_MODEL_C,
        final_tier=RouteTier.LARGE_MODEL_D,
        was_cached=False,
        fallback_occurred=True,
        total_latency_ms=ProvenanceValue(
            value=150.0, provenance=ProvenanceLabel.MEASURED, unit="ms"
        ),
        cost_usd=ProvenanceValue(value=0.002, provenance=ProvenanceLabel.CALCULATED, unit="usd"),
        energy_wh=ProvenanceValue(
            value=0.5,
            provenance=ProvenanceLabel.ESTIMATED,
            unit="Wh",
            system_boundary={"included": "Scope 2", "excluded": "Scope 3"},
        ),
        water_ml=ProvenanceValue(
            value=1.2,
            provenance=ProvenanceLabel.ESTIMATED,
            unit="mL",
            system_boundary={"included": "cooling", "excluded": "supply"},
        ),
        privacy_class=PrivacyClass.PUBLIC,
    )

    assert record.text == "Final Output Content"
    assert record.route == RouteTier.LARGE_MODEL_D
    assert record.cache_hit is False
    assert record.fallback_used is True


def test_model_result_schema_fields():
    """Confirms ModelResult strictly exposes all 8 required fields."""
    res = ModelResult(
        text="Sample output text",
        input_tokens=15,
        output_tokens=25,
        latency_ms=45.2,
        cost_usd_calculated=0.00005,
        provider_reported_cost_usd=None,
        price_table_date="2026-10-01",
        source=ProvenanceLabel.SIMULATED,
        model_id="mock-small",
    )

    assert res.text == "Sample output text"
    assert res.input_tokens == 15
    assert res.output_tokens == 25
    assert res.latency_ms == 45.2
    assert res.cost_usd_calculated == 0.00005
    assert res.provider_reported_cost_usd is None
    assert res.price_table_date == "2026-10-01"
    assert res.source == ProvenanceLabel.SIMULATED
    assert res.model_id == "mock-small"
