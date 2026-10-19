"""Unit tests for formal routing optimization and candidate selection policy."""

from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.router.formulation import ConstrainedOptimizationPolicy


def test_public_unconstrained_request_routes_to_optimal():
    policy = ConstrainedOptimizationPolicy()
    req = UserRequest(
        request_id="req_01",
        prompt="Explain quantum entanglement.",
        privacy_class=PrivacyClass.PUBLIC,
        allow_external_transmission=True,
        task_parameters=TaskParameters(quality_preference="high", minimum_quality=0.85),
    )
    feasible = policy.determine_feasible_set(req)
    assert RouteTier.LARGE_MODEL_D in feasible
    best_tier, _, _ = policy.select_optimal_tier(req)
    assert best_tier == RouteTier.LARGE_MODEL_D


def test_sensitive_request_excludes_external_tiers():
    policy = ConstrainedOptimizationPolicy()
    req = UserRequest(
        request_id="req_priv",
        prompt="Customer SSN: 000-11-2222",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        allow_external_transmission=False,
    )
    feasible = policy.determine_feasible_set(req)
    assert RouteTier.SMALL_MODEL_C not in feasible
    assert RouteTier.LARGE_MODEL_D not in feasible
    assert RouteTier.DETERMINISTIC_A in feasible or RouteTier.TEMPLATE_B in feasible
    best_tier, _, _ = policy.select_optimal_tier(req)
    assert best_tier in (RouteTier.DETERMINISTIC_A, RouteTier.TEMPLATE_B)


def test_budget_constraint_filtering():
    policy = ConstrainedOptimizationPolicy()
    # Budget of $0.0005 excludes Large Model ($0.0035)
    req = UserRequest(
        request_id="req_budget",
        prompt="Extract names from text.",
        privacy_class=PrivacyClass.PUBLIC,
        allow_external_transmission=True,
        task_parameters=TaskParameters(maximum_cost_usd=0.0005, minimum_quality=0.60),
    )
    feasible = policy.determine_feasible_set(req)
    assert RouteTier.LARGE_MODEL_D not in feasible
    assert RouteTier.SMALL_MODEL_C in feasible
    best_tier, _, _ = policy.select_optimal_tier(req)
    assert best_tier == RouteTier.SMALL_MODEL_C
