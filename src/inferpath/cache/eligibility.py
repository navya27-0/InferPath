"""Explicit cache eligibility and freshness policy enforcement."""

from pydantic import BaseModel

from inferpath.cache.freshness import FreshnessClassifier
from inferpath.models.domain import HandlerOutput, UserRequest, ValidationResult
from inferpath.models.enums import PrivacyClass, RouteTier


class CacheEligibilityDecision(BaseModel):
    """Explicit decision describing whether an operation may use the exact cache."""

    is_eligible: bool
    bypass_reason: str | None = None
    policy_code: str  # e.g., "ELIGIBLE", "BYPASS_PRIVATE", "BYPASS_TIME_SENSITIVE", "BYPASS_DETERMINISTIC", etc.


class CacheEligibilityPolicy:
    """
    Non-negotiable cache eligibility gatekeeper:
    - Never caches private or confidential data.
    - Never caches time-sensitive queries.
    - Bypasses lookup for deterministic Tier A operations.
    - Bypasses storage for unvalidated, invalid, or clarification-seeking responses.
    """

    @classmethod
    def check_lookup_eligibility(
        cls,
        request: UserRequest,
        route: RouteTier | str,
    ) -> CacheEligibilityDecision:
        """
        Determines if a request is eligible to perform an exact cache lookup.
        Executed in Stage 4 of the routing pipeline.
        """
        # 1. Privacy gate: private or internal requests never touch cache
        if request.privacy_class != PrivacyClass.PUBLIC:
            return CacheEligibilityDecision(
                is_eligible=False,
                bypass_reason=f"Privacy class '{request.privacy_class.value}' strictly prohibits cache lookup and storage.",
                policy_code="BYPASS_PRIVATE",
            )

        # 2. Freshness gate: explicit time-sensitivity or detected temporal keywords
        if request.is_time_sensitive is True:
            return CacheEligibilityDecision(
                is_eligible=False,
                bypass_reason="Request is flagged time-sensitive; cache lookup bypassed.",
                policy_code="BYPASS_TIME_SENSITIVE",
            )

        is_temporal, temp_reason = FreshnessClassifier.is_time_sensitive(request.prompt)
        if is_temporal:
            return CacheEligibilityDecision(
                is_eligible=False,
                bypass_reason=temp_reason,
                policy_code="BYPASS_TIME_SENSITIVE",
            )

        # 3. Deterministic Tier A bypass: direct execution is cheaper and faster than SQLite lookup
        route_str = route.value if isinstance(route, RouteTier) else str(route)
        if route_str in (RouteTier.DETERMINISTIC_A.value, "deterministic_a"):
            return CacheEligibilityDecision(
                is_eligible=False,
                bypass_reason="Deterministic rules/APIs execute directly; cache lookup bypassed.",
                policy_code="BYPASS_DETERMINISTIC",
            )

        return CacheEligibilityDecision(
            is_eligible=True,
            bypass_reason=None,
            policy_code="ELIGIBLE",
        )

    @classmethod
    def check_storage_eligibility(
        cls,
        request: UserRequest,
        route: RouteTier | str,
        output: HandlerOutput,
        validation_result: ValidationResult | None = None,
    ) -> CacheEligibilityDecision:
        """
        Determines if a validated result is eligible to be written into the exact cache.
        Executed in Stage 8 of the routing pipeline.
        """
        # Base lookup restrictions also prevent writing
        lookup_check = cls.check_lookup_eligibility(request, route)
        if not lookup_check.is_eligible:
            return lookup_check

        # Missing template fields / clarification requests are not final answers
        if output.needs_clarification:
            return CacheEligibilityDecision(
                is_eligible=False,
                bypass_reason="Clarification request cannot be stored as a cached answer.",
                policy_code="BYPASS_CLARIFICATION",
            )

        # Output must be validated
        if not output.validated:
            return CacheEligibilityDecision(
                is_eligible=False,
                bypass_reason="Unvalidated response cannot be stored in exact cache.",
                policy_code="BYPASS_UNVALIDATED",
            )

        # Output validation result failure
        if validation_result is not None and not validation_result.is_valid:
            rejection = validation_result.rejection_reason or "validation failed"
            return CacheEligibilityDecision(
                is_eligible=False,
                bypass_reason=f"Output failed validation checks: {rejection}",
                policy_code="BYPASS_INVALID",
            )

        return CacheEligibilityDecision(
            is_eligible=True,
            bypass_reason=None,
            policy_code="ELIGIBLE",
        )
