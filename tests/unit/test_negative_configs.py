"""Negative tests for malformed configurations, invalid provenance, and privacy leak prevention."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from inferpath.config.settings import load_pricing_table
from inferpath.models.domain import UserRequest
from inferpath.models.enums import PrivacyClass, ProvenanceLabel
from inferpath.models.telemetry import (
    ProvenanceValue,
    RangeBounds,
)
from inferpath.security.privacy import PrivacyPolicyEnforcer


def test_negative_invalid_range_bounds_inverted():
    """RangeBounds rejects inverted bounds where low > high or central > high."""
    with pytest.raises(ValidationError, match="low <= central <= high"):
        RangeBounds(low=100.0, central=50.0, high=10.0)

    with pytest.raises(ValidationError, match="low <= central <= high"):
        RangeBounds(low=10.0, central=50.0, high=25.0)


def test_negative_measured_with_range_bounds():
    """Measured metrics are direct observations and cannot carry estimated ranges."""
    with pytest.raises(ValidationError, match="cannot carry estimated range bounds"):
        ProvenanceValue[int](
            value=250,
            provenance=ProvenanceLabel.MEASURED,
            unit="tokens",
            range_bounds=RangeBounds(low=200, central=250, high=300),
        )


def test_negative_estimated_missing_boundary():
    """Estimated metrics must state an explicit system boundary."""
    with pytest.raises(ValidationError, match="must state an explicit system boundary"):
        ProvenanceValue[float](
            value=1.5,
            provenance=ProvenanceLabel.ESTIMATED,
            unit="Wh",
            system_boundary=None,
        )


def test_negative_malformed_pricing_yaml_missing_models(tmp_path: Path):
    """Malformed pricing YAML without models dictionary raises validation or handles safely."""
    bad_yaml = tmp_path / "bad_pricing.yaml"
    bad_yaml.write_text("table_metadata:\n  date: '2026-10-01'\n", encoding="utf-8")

    # Should load with empty models dictionary without crash
    pt = load_pricing_table(bad_yaml)
    assert pt.rates_per_1m == {}
    assert pt.verification_status == "unverified_illustrative"


def test_privacy_metadata_scrubs_smuggled_private_input(privacy_enforcer: PrivacyPolicyEnforcer):
    """
    Negative test: If a caller attempts to leak raw private prompt text inside metadata
    under a confidential/private request, sanitize_for_audit must scrub or redact it.
    """
    private_secret = "Confidential Patient Medical Diagnosis 12345"
    req = UserRequest(
        request_id="req-leak-test",
        prompt=private_secret,
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        metadata={
            "raw_prompt_copy": private_secret,
            "patient_notes": f"Note regarding {private_secret}",
            "regular_tag": "medical_department",
        },
    )

    sanitized = privacy_enforcer.sanitize_for_audit(req)

    # Raw prompt sample must be strictly None
    assert sanitized["prompt_sample"] is None

    # Metadata must NOT contain the raw private secret
    meta = sanitized["metadata"]
    assert private_secret not in str(meta)
    assert "[REDACTED_PRIVATE_PROMPT]" in meta["patient_notes"]
    assert "hash:" in meta["raw_prompt_copy"]
