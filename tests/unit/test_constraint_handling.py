"""Unit tests for multi-constraint handling and fallback budget/latency/quality boundaries.

Verifies:
1. Cost limit plus latency limit evaluated independently (not shadowed by elif).
2. Quality plus cost limit.
3. Quality plus latency limit.
4. All three constraints simultaneously.
5. Impossible constraints (fail fast with CONSTRAINTS_UNSATISFIABLE).
6. Fallback under constrained budgets.
"""

import pytest

from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable, Settings
from inferpath.handlers.mock_adapter import BaseMockModel, MockLargeModel, MockSmallModel
from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import RouteTier
from inferpath.pipeline import RequestPipeline
from inferpath.router.strategies import ConstraintsUnsatisfiableError
from inferpath.security.budget import RunBudgetTracker
from inferpath.validators.base import CompositeValidator
from inferpath.validators.format import FormatValidator


class FailingSmallModel(BaseMockModel):
    """Mock small model that always outputs invalid format to trigger fallback."""

    def _determine_output_text(self, prompt: str) -> str:
        return "Plain text that is not JSON."


class SucceedingLargeModel(BaseMockModel):
    """Mock large model that produces valid JSON."""

    def _determine_output_text(self, prompt: str) -> str:
        return '{"status": "ok", "tier": "large"}'


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


@pytest.fixture
def constraint_pipeline(tmp_path, pricing_table):
    db_path = str(tmp_path / "constraint_test.db")
    settings = Settings()
    settings.database_path = db_path

    small_adapter = FailingSmallModel(
        model_id="google/gemini-2.5-flash",
        prompt_price_per_1m=0.075,
        completion_price_per_1m=0.30,
        simulated_base_latency_ms=80.0,
    )
    large_adapter = SucceedingLargeModel(
        model_id="anthropic/claude-3.7-sonnet",
        prompt_price_per_1m=3.0,
        completion_price_per_1m=15.0,
        simulated_base_latency_ms=450.0,
    )

    validator = CompositeValidator(validators=[FormatValidator()])
    budget_tracker = RunBudgetTracker(max_usd=1.0, pricing_table=pricing_table)

    return RequestPipeline(
        settings=settings,
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
        validator=validator,
        small_adapter=small_adapter,
        large_adapter=large_adapter,
        budget_tracker=budget_tracker,
    )


def test_cost_limit_satisfied_but_latency_limit_exceeded_blocks_fallback(constraint_pipeline):
    """When cost limit is generous but latency limit is tight, latency constraint must block fallback.

    This directly tests the bug where an 'elif' chain would skip latency evaluation
    if maximum_cost_usd was present and satisfied.
    """
    req = UserRequest(
        request_id="req_cost_ok_lat_tight",
        prompt="Format output as JSON: summary of system logs",
        task_parameters=TaskParameters(
            expected_format="json",
            maximum_cost_usd=0.50,  # Generous budget: Tier D cost (~$0.005) is well within $0.50
            maximum_latency_ms=100.0,  # Tight latency: Tier D takes ~450ms, so 100ms MUST block fallback!
        ),
    )

    result = constraint_pipeline.execute(req)

    # Fallback should be blocked because latency constraint is violated by Tier D
    assert result.fallback_occurred is False
    assert result.validation_passed is False
    trace_steps = [s.get("step") for s in result.route_trace]
    assert "fallback_blocked" in trace_steps
    blocked_reasons = [
        s.get("reason") for s in result.route_trace if s.get("step") == "fallback_blocked"
    ]
    assert "user_latency_constraint" in blocked_reasons


def test_latency_limit_satisfied_but_cost_limit_exceeded_blocks_fallback(constraint_pipeline):
    """When latency limit is generous but cost limit is tight, cost constraint must block fallback."""
    req = UserRequest(
        request_id="req_lat_ok_cost_tight",
        prompt="Format output as JSON: summary of system logs",
        task_parameters=TaskParameters(
            expected_format="json",
            maximum_cost_usd=0.0001,  # Tight budget: Tier D cost (~$0.005) exceeds $0.0001
            maximum_latency_ms=2000.0,  # Generous latency
        ),
    )

    result = constraint_pipeline.execute(req)

    assert result.fallback_occurred is False
    assert result.validation_passed is False
    blocked_reasons = [
        s.get("reason") for s in result.route_trace if s.get("step") == "fallback_blocked"
    ]
    assert "user_budget_constraint" in blocked_reasons


def test_both_cost_and_latency_satisfied_allows_fallback(constraint_pipeline):
    """When both cost and latency limits accommodate Tier D, fallback successfully executes."""
    req = UserRequest(
        request_id="req_both_satisfied",
        prompt="Format output as JSON: summary of system logs",
        task_parameters=TaskParameters(
            expected_format="json",
            maximum_cost_usd=0.50,  # Generous budget
            maximum_latency_ms=1000.0,  # Generous latency
        ),
    )

    result = constraint_pipeline.execute(req)

    assert result.fallback_occurred is True
    assert result.final_tier == RouteTier.LARGE_MODEL_D
    assert result.validation_passed is True
    assert "fallback_success" in result.final_output or "status" in result.final_output


def test_quality_plus_cost_limit_fallback_check(constraint_pipeline):
    """Quality limit above Tier D capability (>0.96) raises ConstraintsUnsatisfiableError."""
    req = UserRequest(
        request_id="req_qual_too_high",
        prompt="Format output as JSON: critical medical diagnostics summary",
        task_parameters=TaskParameters(
            expected_format="json",
            minimum_quality=0.98,  # Beyond Tier D's estimated ceiling of 0.96
            maximum_cost_usd=0.50,
        ),
    )

    with pytest.raises(ConstraintsUnsatisfiableError) as exc_info:
        constraint_pipeline.execute(req)
    assert "quality" in exc_info.value.unsatisfied_constraints or "quality" in str(exc_info.value)


def test_quality_plus_latency_limit_fallback_check(constraint_pipeline):
    """Quality limit ok (0.90) but latency too tight (<250ms) blocks fallback."""
    req = UserRequest(
        request_id="req_qual_ok_lat_tight",
        prompt="Format output as JSON: fast summary",
        task_parameters=TaskParameters(
            expected_format="json",
            minimum_quality=0.90,
            maximum_latency_ms=150.0,
        ),
    )

    result = constraint_pipeline.execute(req)

    assert result.fallback_occurred is False
    blocked_reasons = [
        s.get("reason") for s in result.route_trace if s.get("step") == "fallback_blocked"
    ]
    assert "user_latency_constraint" in blocked_reasons


def test_all_three_constraints_simultaneously_satisfied(constraint_pipeline):
    """Quality (0.85), cost ($0.50), and latency (1000ms) all satisfied -> fallback executes."""
    req = UserRequest(
        request_id="req_all_three_ok",
        prompt="Format output as JSON: standard extraction",
        task_parameters=TaskParameters(
            expected_format="json",
            minimum_quality=0.85,
            maximum_cost_usd=0.50,
            maximum_latency_ms=1000.0,
        ),
    )

    result = constraint_pipeline.execute(req)

    assert result.fallback_occurred is True
    assert result.final_tier == RouteTier.LARGE_MODEL_D
    assert result.validation_passed is True


def test_impossible_constraints_fail_fast():
    """Impossible constraints (e.g. Q_min=0.99 or budget=$0.0000001) fail fast before model dispatch."""
    settings = Settings()
    pricing = PricingTable(
        effective_date="2026-10-01",
        source="Test",
        rates_per_1m={"google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30}},
    )

    small_adapter = MockSmallModel(
        model_id="google/gemini-2.5-flash",
        prompt_price_per_1m=0.075,
        completion_price_per_1m=0.30,
    )
    large_adapter = MockLargeModel(
        model_id="anthropic/claude-3.7-sonnet",
        prompt_price_per_1m=3.0,
        completion_price_per_1m=15.0,
    )

    pipeline = RequestPipeline(
        settings=settings,
        pricing_table=pricing,
        small_adapter=small_adapter,
        large_adapter=large_adapter,
    )

    # Impossible quality constraint
    req_impossible_q = UserRequest(
        request_id="req_imp_q",
        prompt="Explain quantum entanglement",
        task_parameters=TaskParameters(minimum_quality=0.99),
    )
    with pytest.raises(ConstraintsUnsatisfiableError) as exc_q:
        pipeline.execute(req_impossible_q)
    assert "quality" in exc_q.value.unsatisfied_constraints or "quality" in str(exc_q.value)

    # Impossible budget constraint
    req_impossible_b = UserRequest(
        request_id="req_imp_b",
        prompt="Explain quantum entanglement",
        task_parameters=TaskParameters(maximum_cost_usd=0.000001),
    )
    with pytest.raises(ConstraintsUnsatisfiableError) as exc_b:
        pipeline.execute(req_impossible_b)
    assert "budget" in exc_b.value.unsatisfied_constraints or "budget" in str(exc_b.value)

    # Impossible latency constraint
    req_impossible_l = UserRequest(
        request_id="req_imp_l",
        prompt="Explain quantum entanglement",
        task_parameters=TaskParameters(maximum_latency_ms=5.0),
    )
    with pytest.raises(ConstraintsUnsatisfiableError) as exc_l:
        pipeline.execute(req_impossible_l)
    assert "latency" in exc_l.value.unsatisfied_constraints or "latency" in str(exc_l.value)


@pytest.mark.asyncio
async def test_async_cost_limit_satisfied_but_latency_limit_exceeded(constraint_pipeline):
    """execute_async also independently evaluates latency constraint when cost is satisfied."""
    req = UserRequest(
        request_id="async_cost_ok_lat_tight",
        prompt="Format output as JSON: async summary",
        task_parameters=TaskParameters(
            expected_format="json",
            maximum_cost_usd=0.50,
            maximum_latency_ms=100.0,
        ),
    )

    result = await constraint_pipeline.execute_async(req)

    assert result.fallback_occurred is False
    assert result.validation_passed is False
    blocked_reasons = [
        s.get("reason") for s in result.route_trace if s.get("step") == "fallback_blocked"
    ]
    assert "user_latency_constraint" in blocked_reasons
