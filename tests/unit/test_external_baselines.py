"""Unit tests verifying external baseline router strategies."""

from __future__ import annotations

import pytest

from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.router.strategies import (
    ConfidenceEscalationRouter,
    CostOnlyRouter,
    LatencyOnlyRouter,
    PromptLengthRouter,
    QualityOnlyRouter,
    RandomConstrainedRouter,
    SemanticSimilarityRouter,
)


@pytest.fixture
def base_request() -> UserRequest:
    return UserRequest(
        request_id="req_base_001",
        prompt="Explain the difference between TCP and UDP.",
        privacy_class=PrivacyClass.PUBLIC,
    )


@pytest.fixture
def private_request() -> UserRequest:
    return UserRequest(
        request_id="req_priv_001",
        prompt="User SSN is 000-11-2222 and salary is $150,000.",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )


@pytest.fixture
def math_request() -> UserRequest:
    return UserRequest(
        request_id="req_math_001",
        prompt="Calculate 125 * 8 + 450",
        privacy_class=PrivacyClass.PUBLIC,
    )


class TestExternalBaselines:
    def test_random_constrained_router_respects_privacy(self, private_request: UserRequest):
        router = RandomConstrainedRouter(seed=42)
        dec = router.route(private_request)
        assert dec.rule_id == "R01"
        assert dec.eligible_tiers == []

    def test_random_constrained_router_reproducible(self, base_request: UserRequest):
        r1 = RandomConstrainedRouter(seed=42)
        r2 = RandomConstrainedRouter(seed=42)
        assert r1.route(base_request).selected_tier == r2.route(base_request).selected_tier

    def test_confidence_escalation_router(
        self, base_request: UserRequest, math_request: UserRequest, private_request: UserRequest
    ):
        router = ConfidenceEscalationRouter(escalation_threshold=0.70)

        # Privacy
        assert router.route(private_request).rule_id == "R01"

        # Deterministic
        assert router.route(math_request).selected_tier == RouteTier.DETERMINISTIC_A

        # Standard question -> passes confidence
        dec_std = router.route(base_request)
        assert dec_std.selected_tier == RouteTier.SMALL_MODEL_C

        # Highly complex ambiguous question -> escalates
        complex_req = UserRequest(
            request_id="req_complex_001",
            prompt="Maybe perhaps it could be argued to compare and contrast the epistemological frameworks of Kant and Hegel, but clarify what is meant by pure reason?",
            task_parameters=TaskParameters(quality_preference="high", minimum_quality=0.90),
        )
        dec_complex = router.route(complex_req)
        assert dec_complex.selected_tier == RouteTier.LARGE_MODEL_D
        assert dec_complex.rule_id == "CONFIDENCE_ESCALATED_D"

    def test_prompt_length_router(self, private_request: UserRequest, math_request: UserRequest):
        router = PromptLengthRouter(length_threshold_words=20)

        # Privacy
        assert router.route(private_request).rule_id == "R01"

        # Deterministic
        assert router.route(math_request).selected_tier == RouteTier.DETERMINISTIC_A

        # Short prompt (< 20 words)
        short_req = UserRequest(request_id="req_short_001", prompt="What is photosynthesis?")
        assert router.route(short_req).selected_tier == RouteTier.SMALL_MODEL_C
        assert router.route(short_req).rule_id == "PROMPT_LENGTH_SHORT_C"

        # Long prompt (>= 20 words)
        long_words = "word " * 25
        long_req = UserRequest(
            request_id="req_long_001",
            prompt=f"Explain in detail the following concepts: {long_words}",
        )
        assert router.route(long_req).selected_tier == RouteTier.LARGE_MODEL_D
        assert router.route(long_req).rule_id == "PROMPT_LENGTH_LONG_D"

    def test_quality_only_router(
        self, base_request: UserRequest, math_request: UserRequest, private_request: UserRequest
    ):
        router = QualityOnlyRouter()
        assert router.route(private_request).rule_id == "R01"
        assert router.route(math_request).selected_tier == RouteTier.DETERMINISTIC_A
        assert router.route(base_request).selected_tier == RouteTier.LARGE_MODEL_D
        assert router.route(base_request).rule_id == "QUALITY_MAX_D"

    def test_cost_only_router(
        self, base_request: UserRequest, math_request: UserRequest, private_request: UserRequest
    ):
        router = CostOnlyRouter()
        assert router.route(private_request).rule_id == "R01"
        assert router.route(math_request).selected_tier == RouteTier.DETERMINISTIC_A
        assert router.route(base_request).selected_tier == RouteTier.SMALL_MODEL_C
        assert router.route(base_request).rule_id == "COST_MIN_C"

    def test_latency_only_router(
        self, base_request: UserRequest, math_request: UserRequest, private_request: UserRequest
    ):
        router = LatencyOnlyRouter()
        assert router.route(private_request).rule_id == "R01"
        assert router.route(math_request).selected_tier == RouteTier.DETERMINISTIC_A
        assert router.route(base_request).selected_tier == RouteTier.SMALL_MODEL_C
        assert router.route(base_request).rule_id == "LATENCY_MIN_C"

    def test_semantic_similarity_router(
        self, base_request: UserRequest, math_request: UserRequest, private_request: UserRequest
    ):
        router = SemanticSimilarityRouter()
        assert router.route(private_request).rule_id == "R01"
        assert router.route(math_request).selected_tier == RouteTier.DETERMINISTIC_A

        proof_req = UserRequest(
            request_id="req_proof_001",
            prompt="Prove mathematically the formal derivation of the state machine.",
        )
        assert router.route(proof_req).selected_tier == RouteTier.LARGE_MODEL_D
