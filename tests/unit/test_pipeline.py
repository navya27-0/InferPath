"""Unit tests for single orchestration RequestPipeline and one-level fallback policy."""

import pytest

from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable, Settings
from inferpath.handlers.mock_adapter import BaseMockModel, MockLargeModel, MockSmallModel
from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.pipeline import RequestPipeline
from inferpath.security.budget import RunBudgetTracker
from inferpath.security.privacy import PrivacyViolationError
from inferpath.telemetry.experiment_logger import ExperimentLogger
from inferpath.validators.base import CompositeValidator
from inferpath.validators.format import FormatValidator
from inferpath.validators.non_empty import NonEmptyValidator


class FailingValidationMockModel(BaseMockModel):
    """Mock model that intentionally produces invalid JSON to test fallback."""

    def _determine_output_text(self, prompt: str) -> str:
        return "This is plain text, not valid JSON."


class SecondChanceMockLargeModel(BaseMockModel):
    """Mock large model that produces valid JSON upon fallback."""

    def _determine_output_text(self, prompt: str) -> str:
        return '{"fallback_success": true, "recovered": true}'


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
def in_memory_pipeline(tmp_path, pricing_table):
    db_path = str(tmp_path / "test_pipeline.db")
    log_path = str(tmp_path / "experiments.jsonl")

    settings = Settings()
    settings.database_path = db_path
    cache = ExactCache(db_path=db_path)
    logger = ExperimentLogger(log_path=log_path)

    small_adapter = MockSmallModel(
        model_id="google/gemini-2.5-flash",
        prompt_price_per_1m=0.075,
        completion_price_per_1m=0.30,
        simulated_base_latency_ms=50.0,
    )
    large_adapter = MockLargeModel(
        model_id="anthropic/claude-3.7-sonnet",
        prompt_price_per_1m=3.0,
        completion_price_per_1m=15.0,
        simulated_base_latency_ms=150.0,
    )

    return RequestPipeline(
        settings=settings,
        pricing_table=pricing_table,
        cache=cache,
        logger=logger,
        small_adapter=small_adapter,
        large_adapter=large_adapter,
    )


def test_pipeline_deterministic_execution(in_memory_pipeline):
    """Tier A (math) executes deterministically, skips cache lookup and write."""
    req = UserRequest(request_id="p_det_1", prompt="Calculate 125 * 8")
    result = in_memory_pipeline.execute(req)

    assert result.final_output == "1000"
    assert result.final_tier == RouteTier.DETERMINISTIC_A
    assert result.rule_id == "R03"
    assert result.was_cached is False
    assert result.cost_usd.value == 0.0

    # Verify deterministic calculation did NOT write to cache
    lookup_again = in_memory_pipeline.execute(req)
    assert lookup_again.was_cached is False  # Deterministic never cached


def test_pipeline_template_clarification(in_memory_pipeline):
    """Tier B (template) missing dates returns ClarificationRequest, not cached."""
    req = UserRequest(
        request_id="p_tpl_1",
        prompt="Draft a leave application for Bob for medical leave due to dental surgery",
    )
    result = in_memory_pipeline.execute(req)

    assert result.needs_clarification is True
    assert "start_date" in result.final_output
    assert "end_date" in result.final_output
    assert result.was_cached is False


def test_pipeline_cache_hit_on_repeat(in_memory_pipeline):
    """Standard small model call executes on miss, writes to cache, and hits on repeat."""
    req = UserRequest(
        request_id="p_cache_1",
        prompt="What is the capital of France?",
        privacy_class=PrivacyClass.PUBLIC,
    )

    # First call: cache miss
    result1 = in_memory_pipeline.execute(req)
    assert result1.was_cached is False
    assert result1.final_tier == RouteTier.SMALL_MODEL_C
    assert "Paris" in result1.final_output

    # Second call with identical request: cache hit!
    req2 = UserRequest(
        request_id="p_cache_2",
        prompt="What is the capital of France?",
        privacy_class=PrivacyClass.PUBLIC,
    )
    result2 = in_memory_pipeline.execute(req2)
    assert result2.was_cached is True
    assert result2.final_tier == RouteTier.SMALL_MODEL_C
    assert "Paris" in result2.final_output
    assert result2.cost_usd.value == 0.0  # Cache hit incurs 0 model cost


def test_pipeline_fallback_small_to_large_on_validation_failure(tmp_path, pricing_table):
    """Validation failure on small model triggers exactly one fallback to large model."""
    db_path = str(tmp_path / "fallback.db")
    settings = Settings()
    settings.database_path = db_path
    cache = ExactCache(db_path=db_path)

    # Small model fails JSON validation
    bad_small = FailingValidationMockModel(
        model_id="google/gemini-2.5-flash",
        prompt_price_per_1m=0.075,
        completion_price_per_1m=0.30,
        simulated_base_latency_ms=50.0,
    )
    # Large model succeeds JSON validation
    good_large = SecondChanceMockLargeModel(
        model_id="anthropic/claude-3.7-sonnet",
        prompt_price_per_1m=3.0,
        completion_price_per_1m=15.0,
        simulated_base_latency_ms=150.0,
    )

    validator = CompositeValidator(validators=[NonEmptyValidator(), FormatValidator()])

    pipeline = RequestPipeline(
        settings=settings,
        pricing_table=pricing_table,
        cache=cache,
        validator=validator,
        small_adapter=bad_small,
        large_adapter=good_large,
    )

    req = UserRequest(
        request_id="fb_1",
        prompt="Output user profile in JSON",
        task_parameters=TaskParameters(expected_format="json"),
    )

    result = pipeline.execute(req)

    assert result.fallback_occurred is True
    assert len(result.fallback_history) == 1
    fb_entry = result.fallback_history[0]
    assert fb_entry.from_tier == RouteTier.SMALL_MODEL_C
    assert fb_entry.to_tier == RouteTier.LARGE_MODEL_D
    assert "valid JSON" in fb_entry.rejection_reason
    assert result.final_tier == RouteTier.LARGE_MODEL_D
    assert result.validation_passed is True
    assert "fallback_success" in result.final_output

    # Cost Accumulator verifies both attempts contributed to total cost & environmental footprints
    # Small model cost (~0.000003) + Large model cost (~0.000117)
    assert result.cost_usd.value > 0.000117
    assert result.input_tokens.value == 18  # 9 + 9
    assert result.output_tokens.value == 15  # 9 + 6


def test_pipeline_fallback_depth_capped_at_one(tmp_path, pricing_table):
    """If large model also fails validation, pipeline terminates without further recursion."""
    db_path = str(tmp_path / "fail_twice.db")
    settings = Settings()
    settings.database_path = db_path

    bad_small = FailingValidationMockModel(
        model_id="google/gemini-2.5-flash",
        prompt_price_per_1m=0.075,
        completion_price_per_1m=0.30,
        simulated_base_latency_ms=50.0,
    )
    bad_large = FailingValidationMockModel(
        model_id="anthropic/claude-3.7-sonnet",
        prompt_price_per_1m=3.0,
        completion_price_per_1m=15.0,
        simulated_base_latency_ms=150.0,
    )

    validator = CompositeValidator(validators=[NonEmptyValidator(), FormatValidator()])

    pipeline = RequestPipeline(
        settings=settings,
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
        validator=validator,
        small_adapter=bad_small,
        large_adapter=bad_large,
    )

    req = UserRequest(
        request_id="fb_fail",
        prompt="Generate JSON",
        task_parameters=TaskParameters(expected_format="json"),
    )

    result = pipeline.execute(req)

    # Fallback was attempted once, but large model also failed
    assert result.fallback_occurred is True
    assert len(result.fallback_history) == 1
    assert result.validation_passed is False


def test_pipeline_privacy_blocks_external_transmission(in_memory_pipeline):
    """Confidential request requiring external model is rejected immediately by policy."""
    req = UserRequest(
        request_id="p_priv_1",
        prompt="Summarize confidential employee contracts",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )

    with pytest.raises(PrivacyViolationError, match="prohibits external transmission"):
        in_memory_pipeline.execute(req)


def test_pipeline_budget_cap_blocks_fallback(tmp_path, pricing_table):
    """Fallback to large model is blocked if remaining budget cannot cover estimated cost."""
    db_path = str(tmp_path / "budget_fb.db")
    settings = Settings()
    settings.database_path = db_path

    bad_small = FailingValidationMockModel(
        model_id="google/gemini-2.5-flash",
        prompt_price_per_1m=0.075,
        completion_price_per_1m=0.30,
        simulated_base_latency_ms=50.0,
    )
    good_large = SecondChanceMockLargeModel(
        model_id="anthropic/claude-3.7-sonnet",
        prompt_price_per_1m=3.0,
        completion_price_per_1m=15.0,
        simulated_base_latency_ms=150.0,
    )

    # Tiny budget sufficient for small model but not large model
    tiny_budget = RunBudgetTracker(max_usd=0.0005, pricing_table=pricing_table)

    pipeline = RequestPipeline(
        settings=settings,
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
        validator=CompositeValidator(validators=[FormatValidator()]),
        small_adapter=bad_small,
        large_adapter=good_large,
        budget_tracker=tiny_budget,
    )

    req = UserRequest(
        request_id="b_fb_1",
        prompt="Generate JSON report",
        task_parameters=TaskParameters(expected_format="json"),
    )

    result = pipeline.execute(req)

    # Fallback was blocked due to budget exhaustion
    assert result.fallback_occurred is False
    assert result.validation_passed is False
    trace_steps = [s.get("step") for s in result.route_trace]
    assert "fallback_blocked" in trace_steps


@pytest.mark.asyncio
async def test_pipeline_execute_async_deterministic(in_memory_pipeline):
    """execute_async handles Tier A requests and produces identical deterministic results."""
    req = UserRequest(request_id="p_async_det_1", prompt="Calculate 125 * 8")
    result = await in_memory_pipeline.execute_async(req)

    assert result.final_output == "1000"
    assert result.final_tier == RouteTier.DETERMINISTIC_A
    assert result.rule_id == "R03"
    assert result.was_cached is False
    assert result.cost_usd.value == 0.0


@pytest.mark.asyncio
async def test_pipeline_execute_async_model_and_caching(in_memory_pipeline):
    """execute_async completes small model requests asynchronously and caches results."""
    req = UserRequest(
        request_id="p_async_model_1",
        prompt="Classify this review: The camera quality was superb and battery lasted two days.",
    )
    result = await in_memory_pipeline.execute_async(req)

    assert result.final_tier == RouteTier.SMALL_MODEL_C
    assert result.was_cached is False
    assert result.validation_passed is True
    assert result.energy_wh.provenance == result.energy_wh.provenance.ESTIMATED
    assert result.water_ml.provenance == result.water_ml.provenance.ESTIMATED
    assert result.energy_wh.range_bounds is not None
    assert result.water_ml.range_bounds is not None

    # Second execution should hit exact cache
    cached_res = await in_memory_pipeline.execute_async(req)
    assert cached_res.was_cached is True
    assert cached_res.final_output == result.final_output
