"""Unit tests verifying explicit cache eligibility gates and freshness policies."""

from inferpath.cache.eligibility import CacheEligibilityPolicy
from inferpath.models.domain import (
    HandlerOutput,
    UserRequest,
    ValidationResult,
)
from inferpath.models.enums import PrivacyClass, ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


def _make_dummy_output(validated: bool = True, needs_clarification: bool = False) -> HandlerOutput:
    return HandlerOutput(
        tier=RouteTier.SMALL_MODEL_C,
        output_text="Test response",
        validated=validated,
        needs_clarification=needs_clarification,
        input_tokens=ProvenanceValue(value=10, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        output_tokens=ProvenanceValue(value=20, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        latency_ms=ProvenanceValue(value=50.0, provenance=ProvenanceLabel.MEASURED, unit="ms"),
        model_version="test-model-v1",
    )


def test_private_requests_bypass_cache_lookup_and_storage():
    """Private and confidential requests strictly bypass lookup and storage."""
    req_internal = UserRequest(
        request_id="req-priv-1",
        prompt="Internal quarterly revenue draft",
        privacy_class=PrivacyClass.INTERNAL,
    )
    req_conf = UserRequest(
        request_id="req-priv-2",
        prompt="Patient blood test results",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )

    # Lookup checks
    d_int = CacheEligibilityPolicy.check_lookup_eligibility(req_internal, RouteTier.SMALL_MODEL_C)
    assert d_int.is_eligible is False
    assert d_int.policy_code == "BYPASS_PRIVATE"

    d_conf = CacheEligibilityPolicy.check_lookup_eligibility(req_conf, RouteTier.LARGE_MODEL_D)
    assert d_conf.is_eligible is False
    assert d_conf.policy_code == "BYPASS_PRIVATE"

    # Storage checks
    output = _make_dummy_output()
    s_int = CacheEligibilityPolicy.check_storage_eligibility(
        req_internal, RouteTier.SMALL_MODEL_C, output
    )
    assert s_int.is_eligible is False
    assert s_int.policy_code == "BYPASS_PRIVATE"


def test_time_sensitive_requests_bypass_cache():
    """Time-sensitive queries bypass lookup and storage even when prompts are identical."""
    # Explicit time-sensitivity flag
    req_flagged = UserRequest(
        request_id="req-ts-1",
        prompt="Summarize today's headlines",
        is_time_sensitive=True,
    )
    d1 = CacheEligibilityPolicy.check_lookup_eligibility(req_flagged, RouteTier.SMALL_MODEL_C)
    assert d1.is_eligible is False
    assert d1.policy_code == "BYPASS_TIME_SENSITIVE"

    # Detected temporal keywords
    req_weather = UserRequest(
        request_id="req-ts-2",
        prompt="What is the weather today in Boston?",
    )
    d2 = CacheEligibilityPolicy.check_lookup_eligibility(req_weather, RouteTier.SMALL_MODEL_C)
    assert d2.is_eligible is False
    assert d2.policy_code == "BYPASS_TIME_SENSITIVE"

    # Storage bypass
    output = _make_dummy_output()
    s_ts = CacheEligibilityPolicy.check_storage_eligibility(
        req_weather, RouteTier.SMALL_MODEL_C, output
    )
    assert s_ts.is_eligible is False
    assert s_ts.policy_code == "BYPASS_TIME_SENSITIVE"


def test_deterministic_tier_a_bypasses_cache_lookup():
    """Deterministic rules (Tier A) bypass lookup overhead for direct execution."""
    req_math = UserRequest(
        request_id="req-math-1",
        prompt="Calculate 25 * 40",
    )
    decision = CacheEligibilityPolicy.check_lookup_eligibility(req_math, RouteTier.DETERMINISTIC_A)
    assert decision.is_eligible is False
    assert decision.policy_code == "BYPASS_DETERMINISTIC"


def test_unvalidated_or_failed_results_never_stored():
    """Unvalidated, invalid, or clarification-seeking responses are never stored."""
    req = UserRequest(
        request_id="req-pub-1",
        prompt="Explain quantum teleportation",
    )

    # 1. Output marked unvalidated
    out_unval = _make_dummy_output(validated=False)
    s1 = CacheEligibilityPolicy.check_storage_eligibility(req, RouteTier.SMALL_MODEL_C, out_unval)
    assert s1.is_eligible is False
    assert s1.policy_code == "BYPASS_UNVALIDATED"

    # 2. Output seeking clarification
    out_clar = _make_dummy_output(needs_clarification=True)
    s2 = CacheEligibilityPolicy.check_storage_eligibility(req, RouteTier.SMALL_MODEL_C, out_clar)
    assert s2.is_eligible is False
    assert s2.policy_code == "BYPASS_CLARIFICATION"

    # 3. Explicit validation failure
    out_valid = _make_dummy_output(validated=True)
    val_fail = ValidationResult(is_valid=False, rejection_reason="Output did not match JSON schema")
    s3 = CacheEligibilityPolicy.check_storage_eligibility(
        req, RouteTier.SMALL_MODEL_C, out_valid, val_fail
    )
    assert s3.is_eligible is False
    assert s3.policy_code == "BYPASS_INVALID"


def test_eligible_validated_public_request_passes():
    """Valid, non-temporal public request passes both lookup and storage eligibility."""
    req = UserRequest(
        request_id="req-ok-1",
        prompt="Explain the nitrogen cycle in agriculture",
        privacy_class=PrivacyClass.PUBLIC,
    )
    lookup_dec = CacheEligibilityPolicy.check_lookup_eligibility(req, RouteTier.SMALL_MODEL_C)
    assert lookup_dec.is_eligible is True
    assert lookup_dec.policy_code == "ELIGIBLE"

    output = _make_dummy_output(validated=True)
    val_ok = ValidationResult(is_valid=True)
    storage_dec = CacheEligibilityPolicy.check_storage_eligibility(
        req, RouteTier.SMALL_MODEL_C, output, val_ok
    )
    assert storage_dec.is_eligible is True
    assert storage_dec.policy_code == "ELIGIBLE"
