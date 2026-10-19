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


def test_privacy_external_transmission_combinations():
    policy = ConstrainedOptimizationPolicy()

    # (a) Public with default external transmission
    req_public_default = UserRequest(
        request_id="req_pub_def",
        prompt="What is the capital of France?",
        privacy_class=PrivacyClass.PUBLIC,
        task_parameters=TaskParameters(minimum_quality=0.60),
    )
    assert req_public_default.allow_external_transmission is True
    feasible_pub = policy.determine_feasible_set(req_public_default)
    assert RouteTier.SMALL_MODEL_C in feasible_pub
    assert RouteTier.LARGE_MODEL_D in feasible_pub

    # (b) Public with explicit local-only override (allow_external_transmission=False)
    req_public_local = UserRequest(
        request_id="req_pub_local",
        prompt="What is the capital of France?",
        privacy_class=PrivacyClass.PUBLIC,
        allow_external_transmission=False,
        task_parameters=TaskParameters(minimum_quality=0.60),
    )
    assert req_public_local.allow_external_transmission is False
    feasible_pub_local = policy.determine_feasible_set(req_public_local)
    assert RouteTier.SMALL_MODEL_C not in feasible_pub_local
    assert RouteTier.LARGE_MODEL_D not in feasible_pub_local

    # (c) Internal (defaults to allow_external_transmission=False)
    req_internal = UserRequest(
        request_id="req_internal",
        prompt="Internal documentation query.",
        privacy_class=PrivacyClass.INTERNAL,
    )
    assert req_internal.allow_external_transmission is False
    feasible_internal = policy.determine_feasible_set(req_internal)
    assert RouteTier.SMALL_MODEL_C not in feasible_internal
    assert RouteTier.LARGE_MODEL_D not in feasible_internal

    # (d) Confidential (defaults to allow_external_transmission=False)
    req_confidential = UserRequest(
        request_id="req_confidential",
        prompt="Secret customer record.",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )
    assert req_confidential.allow_external_transmission is False
    feasible_conf = policy.determine_feasible_set(req_confidential)
    assert RouteTier.SMALL_MODEL_C not in feasible_conf
    assert RouteTier.LARGE_MODEL_D not in feasible_conf


def test_local_only_metadata_excludes_external_tiers():
    policy = ConstrainedOptimizationPolicy()
    req = UserRequest(
        request_id="req_local_meta",
        prompt="What is the capital of France?",
        privacy_class=PrivacyClass.PUBLIC,
        metadata={"local_only": True},
        task_parameters=TaskParameters(minimum_quality=0.60),
    )
    feasible = policy.determine_feasible_set(req)
    assert RouteTier.SMALL_MODEL_C not in feasible
    assert RouteTier.LARGE_MODEL_D not in feasible


def test_unsatisfiable_constraints_raises_error():
    import pytest

    from inferpath.router.strategies import ConstraintsUnsatisfiableError

    policy = ConstrainedOptimizationPolicy()
    req = UserRequest(
        request_id="req_impossible",
        prompt="Explain quantum gravity in detail.",
        privacy_class=PrivacyClass.PUBLIC,
        task_parameters=TaskParameters(
            minimum_quality=0.999,
            maximum_cost_usd=0.0000001,
            maximum_latency_ms=0.1,
        ),
    )
    feasible = policy.determine_feasible_set(req)
    assert feasible == []

    with pytest.raises(ConstraintsUnsatisfiableError) as exc_info:
        policy.select_optimal_tier(req)
    assert "quality" in exc_info.value.unsatisfied_constraints
