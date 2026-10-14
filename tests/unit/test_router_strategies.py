"""Unit tests for InferPath Multi-Strategy Routing Architecture."""

import pytest

from inferpath.config.settings import PricingTable
from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.router import (
    AlwaysLargeRouter,
    AlwaysSmallRouter,
    ConstraintAwareRouter,
    LearnedRouter,
)


@pytest.fixture
def pricing_table():
    return PricingTable(
        effective_date="2026-10-01",
        source="Test Pricing Table",
        rates_per_1m={
            "google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
            "anthropic/claude-3.7-sonnet": {"prompt": 3.0, "completion": 15.0},
        },
    )


def test_always_large_router(pricing_table):
    router = AlwaysLargeRouter(pricing_table=pricing_table)

    # Public request should route to Large Tier D
    req_public = UserRequest(
        request_id="req_pub",
        prompt="Tell me a simple joke.",
        privacy_class=PrivacyClass.PUBLIC,
    )
    dec_pub = router.route(req_public)
    assert dec_pub.selected_tier == RouteTier.LARGE_MODEL_D
    assert dec_pub.rule_id == "ALWAYS_LARGE"

    # Confidential request without external transmission MUST be blocked by R01
    req_conf = UserRequest(
        request_id="req_conf",
        prompt="Confidential internal roadmap details.",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        allow_external_transmission=False,
    )
    dec_conf = router.route(req_conf)
    assert dec_conf.rule_id == "R01"


def test_always_small_router(pricing_table):
    router = AlwaysSmallRouter(pricing_table=pricing_table)

    # Public request should route to Small Tier C
    req_pub = UserRequest(
        request_id="req_pub",
        prompt="Draft an email acknowledging receipt of application.",
        privacy_class=PrivacyClass.PUBLIC,
    )
    dec_pub = router.route(req_pub)
    assert dec_pub.selected_tier == RouteTier.SMALL_MODEL_C
    assert dec_pub.rule_id == "ALWAYS_SMALL"

    # Confidential request MUST trigger R01
    req_conf = UserRequest(
        request_id="req_conf",
        prompt="Internal financial forecast for Q3.",
        privacy_class=PrivacyClass.INTERNAL,
        allow_external_transmission=False,
    )
    dec_conf = router.route(req_conf)
    assert dec_conf.rule_id == "R01"


def test_constraint_aware_router_deterministic(pricing_table):
    router = ConstraintAwareRouter(pricing_table=pricing_table)

    # Arithmetic -> Tier A (Cost=$0, Q=1.0)
    req_math = UserRequest(
        request_id="math_1",
        prompt="Calculate: 45 * 12 + 100",
        task_parameters=TaskParameters(minimum_quality=0.95),
    )
    dec_math = router.route(req_math)
    assert dec_math.selected_tier == RouteTier.DETERMINISTIC_A
    assert dec_math.rule_id == "R03"

    # Unit conversion -> Tier A
    req_unit = UserRequest(
        request_id="unit_1",
        prompt="Convert 100 miles to km",
        task_parameters=TaskParameters(minimum_quality=0.90),
    )
    dec_unit = router.route(req_unit)
    assert dec_unit.selected_tier == RouteTier.DETERMINISTIC_A
    assert dec_unit.rule_id == "R04"


def test_constraint_aware_router_quality_pruning(pricing_table):
    router = ConstraintAwareRouter(pricing_table=pricing_table)

    # Complex multi-step reasoning task with high Q_min (0.85):
    # Tier C (estimated ~0.58) should be PRUNED; Tier D selected
    complex_prompt = (
        "Analyze the causal relationship between macroeconomic interest rate shifts and "
        "venture capital deployment, step-by-step with proofs and multi-step deduction."
    )
    req = UserRequest(
        request_id="req_complex",
        prompt=complex_prompt,
        task_parameters=TaskParameters(minimum_quality=0.85),
    )
    dec = router.route(req)
    assert dec.selected_tier == RouteTier.LARGE_MODEL_D
    assert "PRUNED" in dec.rule_id or "CONSTRAINT_OPT_D" in dec.rule_id


def test_constraint_aware_router_cost_optimization(pricing_table):
    router = ConstraintAwareRouter(pricing_table=pricing_table)

    # Standard drafting task with standard Q_min (0.70):
    # Both Tier C (Q_hat=0.88) and Tier D (Q_hat=0.95) qualify, but Tier C minimizes Cost!
    req = UserRequest(
        request_id="req_draft",
        prompt="Summarize this brief update for our team: The deployment completed smoothly at 2 PM.",
        task_parameters=TaskParameters(minimum_quality=0.70),
    )
    dec = router.route(req)
    assert dec.selected_tier == RouteTier.SMALL_MODEL_C
    assert "C" in dec.rule_id


def test_constraint_aware_router_budget_pruning(pricing_table):
    router = ConstraintAwareRouter(pricing_table=pricing_table)

    # If user specifies a budget cap lower than Tier D's estimated cost, Tier D is pruned
    req = UserRequest(
        request_id="req_budget",
        prompt="Explain quantum entanglement simply.",
        task_parameters=TaskParameters(
            minimum_quality=0.50,
            maximum_cost_usd=0.001,  # Too small for Tier D ($3.00/1M prompt, ~$0.0038), fits Tier C (~$0.00008)
        ),
    )
    dec = router.route(req)
    assert dec.selected_tier == RouteTier.SMALL_MODEL_C


def test_learned_router(pricing_table):
    router = LearnedRouter(pricing_table=pricing_table)

    # Math query should predict Tier A
    req_math = UserRequest(
        request_id="math_ml",
        prompt="Calculate: 25 * 4 + 10",
        task_parameters=TaskParameters(minimum_quality=0.70),
    )
    dec_math = router.route(req_math)
    assert dec_math.selected_tier == RouteTier.DETERMINISTIC_A
    assert dec_math.rule_id == "LEARNED_CLASSIFIER"

    # Confidential request MUST be blocked by R01 regardless of ML model prediction
    req_conf = UserRequest(
        request_id="conf_ml",
        prompt="Internal confidential payroll breakdown",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        allow_external_transmission=False,
    )
    dec_conf = router.route(req_conf)
    assert dec_conf.rule_id == "R01"
