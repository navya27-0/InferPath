"""Unit tests verifying non-negotiable privacy boundaries and log redaction."""

from inferpath.models.domain import UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.security.hashing import hash_prompt, scrub_api_keys
from inferpath.security.privacy import PrivacyPolicyEnforcer


def test_public_request_permitted_external_and_cached(privacy_enforcer: PrivacyPolicyEnforcer):
    """Public requests can be sent externally and cached."""
    req = UserRequest(
        request_id="req-1",
        prompt="What is the boiling point of nitrogen?",
        privacy_class=PrivacyClass.PUBLIC,
    )
    can_transmit, reason = privacy_enforcer.can_transmit_externally(req, RouteTier.LARGE_MODEL_D)
    assert can_transmit is True
    assert reason is None
    assert privacy_enforcer.is_cache_permitted(req) is True


def test_internal_request_blocked_from_external_transmission(
    privacy_enforcer: PrivacyPolicyEnforcer,
):
    """Internal requests are blocked from external LLMs by default."""
    req = UserRequest(
        request_id="req-2",
        prompt="Internal project budget details",
        privacy_class=PrivacyClass.INTERNAL,
    )
    can_transmit, reason = privacy_enforcer.can_transmit_externally(req, RouteTier.SMALL_MODEL_C)
    assert can_transmit is False
    assert "strictly prohibits external transmission" in reason


def test_confidential_private_blocked_from_cache(privacy_enforcer: PrivacyPolicyEnforcer):
    """Confidential or private requests are never cached."""
    req = UserRequest(
        request_id="req-3",
        prompt="Patient health record details",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )
    assert privacy_enforcer.is_cache_permitted(req) is False


def test_confidential_private_allowed_deterministic_local(privacy_enforcer: PrivacyPolicyEnforcer):
    """Confidential requests are permitted to execute on local deterministic tier (Path A)."""
    req = UserRequest(
        request_id="req-4",
        prompt="250 * 40",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )
    can_transmit, reason = privacy_enforcer.can_transmit_externally(req, RouteTier.DETERMINISTIC_A)
    assert can_transmit is True
    assert reason is None


def test_sanitize_for_audit_never_logs_private_prompt_text(privacy_enforcer: PrivacyPolicyEnforcer):
    """Private prompts must strictly record prompt_sample as None and preserve only SHA-256."""
    secret_text = "Highly confidential salary information"
    req = UserRequest(
        request_id="req-5",
        prompt=secret_text,
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )
    sanitized = privacy_enforcer.sanitize_for_audit(req)

    assert sanitized["prompt_sha256"] == hash_prompt(secret_text)
    assert sanitized["prompt_sample"] is None
    assert secret_text not in str(sanitized)


def test_sanitize_for_audit_allows_public_sample(privacy_enforcer: PrivacyPolicyEnforcer):
    """Public prompts may record a truncated prompt sample."""
    public_text = "What is the capital of France?"
    req = UserRequest(
        request_id="req-6",
        prompt=public_text,
        privacy_class=PrivacyClass.PUBLIC,
    )
    sanitized = privacy_enforcer.sanitize_for_audit(req)

    assert sanitized["prompt_sample"] == public_text
    assert sanitized["prompt_sha256"] == hash_prompt(public_text)


def test_scrub_api_keys_redacts_credentials():
    """Scrubber removes OpenRouter and generic API keys from logs and messages."""
    raw = "Failed call with key sk-or-v1-abcdef0123456789abcdef0123456789 and sk-1234567890abcdef1234567890abcdef"
    scrubbed = scrub_api_keys(raw)
    assert "sk-or-v1" not in scrubbed
    assert "sk-12345" not in scrubbed
    assert "[REDACTED_API_KEY]" in scrubbed
