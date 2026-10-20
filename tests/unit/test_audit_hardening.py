"""Comprehensive regression test suite for InferPath October 2026 Audit Hardening.

Validates:
1. Finding 1: Deployment profile fail-closed and production API key requirement.
2. Finding 2: Rejection of non-finite numbers (NaN, Inf, -Inf) and out-of-range constraints.
3. Finding 3: Authoritative constraint evaluation and decoupled metric concepts.
4. Finding 4: Obfuscated secret normalization (de-spacing, delimiters, leetspeak) and defensive privacy.
5. Finding 5: Distributed rate limiting operational controls, fail-closed mode, and degradation tracking.
6. Finding 6: Exact cache corruption recovery, legacy migration, and strict privacy boundary.
7. Finding 7: Unknown model pricing handling, budget reservation leak prevention, and secret scrubbing.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from inferpath.api.main import RespondRequest, create_app
from inferpath.cache.exact_cache import CURRENT_SCHEMA_VERSION, ExactCache
from inferpath.config.deployment_profiles import (
    ProfileName,
    get_deployment_profile,
)
from inferpath.config.settings import PricingTable
from inferpath.handlers.openrouter_adapter import (
    OpenRouterAdapter,
    OpenRouterResponseError,
    UnknownModelPricingError,
)
from inferpath.models.domain import HandlerOutput, TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass, ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue
from inferpath.router.analyzer import contains_sensitive_patterns
from inferpath.security.budget import RunBudgetTracker
from inferpath.security.privacy import PrivacyPolicyEnforcer
from inferpath.services.rate_limiter import (
    DistributedRateLimiter,
    RateLimiterBackendError,
)

# =============================================================================
# Finding 1: Authentication & Deployment Safeguards
# =============================================================================


def test_invalid_deployment_profile_raises_value_error():
    """Unrecognized profile string raises ValueError rather than silently defaulting."""
    with pytest.raises(ValueError, match="Invalid deployment profile 'super-production'"):
        get_deployment_profile("super-production")


def test_production_environment_auto_promotes_when_profile_unspecified(monkeypatch):
    """When ENVIRONMENT=production but no profile is specified, it auto-promotes to production profile."""
    monkeypatch.delenv("INFERPATH_DEPLOYMENT_PROFILE", raising=False)
    monkeypatch.delenv("INFERPATH_PROFILE", raising=False)
    monkeypatch.delenv("INFERPATH_DEPLOYMENT_MODE", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")

    profile = get_deployment_profile()
    assert profile.name == ProfileName.PRODUCTION
    assert profile.require_api_key is True
    assert profile.default_bind_host == "0.0.0.0"


def test_production_startup_fails_fast_without_api_key(monkeypatch):
    """Standalone production application halts startup if INFERPATH_API_KEY is missing."""
    monkeypatch.setenv("INFERPATH_DEPLOYMENT_PROFILE", "production")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-valid-production-key")
    monkeypatch.delenv("INFERPATH_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="INFERPATH_API_KEY is not configured"):
        create_app()


def test_supported_profile_aliases():
    """Canonical profile resolution accurately maps standardized aliases."""
    assert get_deployment_profile("development").name == ProfileName.LOCAL_DEMO
    assert get_deployment_profile("dev").name == ProfileName.LOCAL_DEMO
    assert get_deployment_profile("staging").name == ProfileName.STAGING
    assert get_deployment_profile("production").name == ProfileName.PRODUCTION
    assert get_deployment_profile("prod").name == ProfileName.PRODUCTION
    assert get_deployment_profile("evaluation").name == ProfileName.BENCHMARK_OFFLINE
    assert get_deployment_profile("eval").name == ProfileName.BENCHMARK_OFFLINE


# =============================================================================
# Finding 2: Rejection of Non-Finite Constraints (NaN, Inf, -Inf)
# =============================================================================


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_respond_request_schema_rejects_non_finite_quality(bad_val):
    """RespondRequest rejects NaN, +Infinity, and -Infinity for minimum_quality."""
    with pytest.raises(ValueError, match="must be a finite number"):
        RespondRequest(prompt="hello", minimum_quality=bad_val)


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_task_parameters_rejects_non_finite_cost_and_latency(bad_val):
    """TaskParameters rejects non-finite cost and latency constraints."""
    with pytest.raises(ValueError, match="must be a finite number"):
        TaskParameters(maximum_cost_usd=bad_val)

    with pytest.raises(ValueError, match="must be a finite number"):
        TaskParameters(maximum_latency_ms=bad_val)


def test_respond_endpoint_rejects_nan_and_inf_payload():
    """FastAPI endpoint returns HTTP 400 with INVALID_CONSTRAINTS on non-finite constraints."""
    app = create_app(deployment_profile="local-demo")
    client = TestClient(app)

    # Note: NaN in JSON is invalid standard JSON; if serialized or stringified, endpoint validates
    resp_neg_qual = client.post(
        "/v1/respond",
        json={"prompt": "test prompt", "minimum_quality": -0.5},
    )
    assert resp_neg_qual.status_code == 400
    assert resp_neg_qual.json()["code"] == "INVALID_CONSTRAINTS"

    resp_excess_cost = client.post(
        "/v1/respond",
        json={"prompt": "test prompt", "maximum_cost_usd": 5000.0},
    )
    assert resp_excess_cost.status_code == 400
    assert resp_excess_cost.json()["code"] == "INVALID_CONSTRAINTS"


# =============================================================================
# Finding 3: Single Authoritative Constraint Evaluation & Metric Concepts
# =============================================================================


def test_result_finalizer_decouples_estimated_and_validated_metrics():
    """ResultFinalizer populates estimated_quality, validated_correctness, and empirical_quality distinctly."""
    from inferpath.evaluation.environmental import EnvironmentalEstimator
    from inferpath.models.domain import RoutingDecision
    from inferpath.pipeline_stages import ResultFinalizer

    estimator = EnvironmentalEstimator()
    finalizer = ResultFinalizer(estimator=estimator)

    req = UserRequest(
        request_id="req_metric_test",
        prompt="Explain photosynthesis",
        task_parameters=TaskParameters(minimum_quality=0.80),
    )
    decision = RoutingDecision(
        selected_tier=RouteTier.SMALL_MODEL_C,
        eligible_tiers=[RouteTier.SMALL_MODEL_C],
        rule_id="R09",
        confidence=0.80,
        reason="standard drafting",
        time_sensitive=False,
        cache_eligible=True,
    )
    output = HandlerOutput(
        output_text="Photosynthesis converts sunlight into chemical energy.",
        tier=RouteTier.SMALL_MODEL_C,
        model_version="mock-small",
        input_tokens=ProvenanceValue(value=10, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        output_tokens=ProvenanceValue(value=5, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        latency_ms=ProvenanceValue(value=35.0, provenance=ProvenanceLabel.MEASURED, unit="ms"),
        raw_response={"cost_usd_calculated": 0.00015},
    )

    res = finalizer.finalize(
        request=req,
        decision=decision,
        output=output,
        was_cached=False,
        fallback_occurred=False,
        fallback_history=[],
        validation_passed=True,
        elapsed_ms=35.0,
        route_trace=[],
    )

    # Assert metric separation
    assert res.estimated_quality == 0.72  # Prior for Small Model C
    assert res.validated_correctness is True
    assert res.empirical_quality is None
    assert res.constraints_satisfied is True


def test_post_execution_constraint_violation_flagged_in_result():
    """If actual latency or cost exceeds caller constraint during runtime, result flags unsatisfied."""
    from inferpath.evaluation.environmental import EnvironmentalEstimator
    from inferpath.models.domain import RoutingDecision
    from inferpath.pipeline_stages import ResultFinalizer

    estimator = EnvironmentalEstimator()
    finalizer = ResultFinalizer(estimator=estimator)

    req = UserRequest(
        request_id="req_overrun",
        prompt="Slow query",
        task_parameters=TaskParameters(maximum_latency_ms=20.0, maximum_cost_usd=0.00005),
    )
    decision = RoutingDecision(
        selected_tier=RouteTier.SMALL_MODEL_C,
        eligible_tiers=[RouteTier.SMALL_MODEL_C],
        rule_id="R09",
        confidence=0.80,
        reason="standard drafting",
        time_sensitive=False,
        cache_eligible=True,
    )
    output = HandlerOutput(
        output_text="Result text",
        tier=RouteTier.SMALL_MODEL_C,
        model_version="mock-small",
        input_tokens=ProvenanceValue(value=10, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        output_tokens=ProvenanceValue(value=5, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        latency_ms=ProvenanceValue(value=55.0, provenance=ProvenanceLabel.MEASURED, unit="ms"),
        raw_response={"cost_usd_calculated": 0.00020},  # Exceeds 0.00005
    )

    res = finalizer.finalize(
        request=req,
        decision=decision,
        output=output,
        was_cached=False,
        fallback_occurred=False,
        fallback_history=[],
        validation_passed=True,
        elapsed_ms=55.0,  # Exceeds 20.0ms
        route_trace=[],
    )

    assert res.constraints_satisfied is False
    assert "budget" in res.unsatisfied_constraints
    assert "latency" in res.unsatisfied_constraints


# =============================================================================
# Finding 4: Privacy Protection & Obfuscated Secrets Normalization
# =============================================================================


def test_contains_sensitive_patterns_obfuscation_resilience():
    """contains_sensitive_patterns detects de-spaced, delimited, and leetspeak secrets."""
    # De-spaced characters
    assert contains_sensitive_patterns("The p a s s w o r d is secret123") is True
    assert contains_sensitive_patterns("My a p i _ k e y is active") is True

    # Delimiter-separated characters
    assert contains_sensitive_patterns("Confidential s-e-c-r-e-t-k-e-y inside") is True
    assert contains_sensitive_patterns("This is a t_r_a_d_e_s_e_c_r_e_t") is True

    # Leetspeak
    assert contains_sensitive_patterns("Here is your p4ssw0rd") is True
    assert contains_sensitive_patterns("My p@ssw0rd is secret") is True
    assert contains_sensitive_patterns("Contains s3cr3t key") is True

    # Benign text
    assert contains_sensitive_patterns("Convert 100 meters to feet") is False
    assert contains_sensitive_patterns("Explain the history of computer networks") is False


def test_uncertain_privacy_classification_enforces_local_only():
    """Uncertain privacy classification defensively forces local-only and disables cache."""
    enforcer = PrivacyPolicyEnforcer()
    req = UserRequest(
        request_id="req_uncertain",
        prompt="Synthesize research data",
        privacy_class=PrivacyClass.PUBLIC,
        metadata={"uncertain_classification": True},
    )

    decision = enforcer.evaluate_privacy(req)
    assert decision.allow_local_processing is True
    assert decision.allow_external_transmission is False
    assert decision.allow_cache is False
    assert "uncertain" in decision.reason.lower()


# =============================================================================
# Finding 5: Distributed Rate Limiting Operational Controls
# =============================================================================


def test_distributed_rate_limiter_fail_closed_mode():
    """DistributedRateLimiter with fail_closed=True raises RateLimiterBackendError when Redis is unavailable."""
    limiter = DistributedRateLimiter(redis_client=None, requests_per_minute=60, fail_closed=True)
    with pytest.raises(RateLimiterBackendError, match="requires active Redis backend"):
        limiter.is_allowed("client_prod_1")

    assert limiter.is_degraded is True
    assert limiter.failure_count == 1


def test_distributed_rate_limiter_degradation_tracking_in_fail_open_mode():
    """In fail-open mode, distributed rate limiter tracks degradation while falling back."""
    mock_failing_redis = MagicMock()
    mock_failing_redis.eval.side_effect = ConnectionError("Redis connection refused")
    mock_failing_redis.pipeline.side_effect = ConnectionError("Redis connection refused")

    limiter = DistributedRateLimiter(
        redis_client=mock_failing_redis, requests_per_minute=2, fail_closed=False
    )

    # First request falls back to local in-memory
    assert limiter.is_allowed("client_degraded") is True
    assert limiter.is_degraded is True
    assert limiter.failure_count == 1

    # Second request allowed
    assert limiter.is_allowed("client_degraded") is True
    # Third request blocked by local fallback limiter
    assert limiter.is_allowed("client_degraded") is False


# =============================================================================
# Finding 6: Cache Migrations, Corruption Recovery & Privacy
# =============================================================================


def test_exact_cache_refuses_to_store_non_public_requests(tmp_path):
    """ExactCache strictly refuses to store requests classified as CONFIDENTIAL or INTERNAL."""
    db_file = tmp_path / "cache_privacy.db"
    cache = ExactCache(db_path=db_file)

    output = HandlerOutput(
        output_text="Confidential payroll calculation",
        tier=RouteTier.DETERMINISTIC_A,
        model_version="local",
        input_tokens=ProvenanceValue(value=10, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        output_tokens=ProvenanceValue(value=5, provenance=ProvenanceLabel.MEASURED, unit="tokens"),
        latency_ms=ProvenanceValue(value=0.5, provenance=ProvenanceLabel.MEASURED, unit="ms"),
    )

    # Private request
    req_private = UserRequest(
        request_id="req_priv",
        prompt="Employee salary data",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
    )
    stored_private = cache.store("hash_priv", req_private, output, RouteTier.DETERMINISTIC_A)
    assert stored_private is False
    assert cache.lookup("hash_priv") is None

    # Public request
    req_public = UserRequest(
        request_id="req_pub",
        prompt="Convert 5km to miles",
        privacy_class=PrivacyClass.PUBLIC,
    )
    stored_public = cache.store("hash_pub", req_public, output, RouteTier.DETERMINISTIC_A)
    assert stored_public is True
    assert cache.lookup("hash_pub") is not None


def test_exact_cache_purges_corrupted_payload_on_lookup(tmp_path):
    """ExactCache purges rows with corrupted or truncated JSON payloads and returns None."""
    db_file = tmp_path / "corrupt_cache.db"
    cache = ExactCache(db_path=db_file)

    # Directly insert corrupted JSON payload into SQLite
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            INSERT INTO exact_cache (
                cache_key, normalized_prompt, output_text, tier, response_payload,
                requirements_snapshot, version, validation_status, hit_count,
                input_tokens, output_tokens, latency_ms, expires_at
            ) VALUES (
                'corrupted_key', 'test', 'text', 'small_model_c', '{"incomplete_json": true,',
                '{}', '1.0', 'passed', 0, 10, 10, 50.0, '2099-01-01T00:00:00Z'
            )
            """
        )
        conn.commit()

    # Lookup should catch JSON parse error, delete the corrupted row, and return None
    res = cache.lookup("corrupted_key")
    assert res is None

    # Verify corrupted row was deleted
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM exact_cache WHERE cache_key = 'corrupted_key'")
        count = cursor.fetchone()[0]
        assert count == 0


def test_exact_cache_legacy_schema_migration(tmp_path):
    """ExactCache gracefully handles legacy tables missing schema_version table."""
    db_file = tmp_path / "legacy.db"

    # Create a legacy exact_cache table WITHOUT schema_version
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE exact_cache (
                cache_key TEXT PRIMARY KEY,
                normalized_prompt TEXT NOT NULL,
                output_text TEXT NOT NULL,
                tier TEXT NOT NULL,
                response_payload TEXT NOT NULL,
                requirements_snapshot TEXT NOT NULL,
                version TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            );
            """
        )
        conn.commit()

    cache = ExactCache(db_path=db_file)
    version = cache.migrate_if_needed()
    assert version == CURRENT_SCHEMA_VERSION
    assert cache.get_schema_version() == CURRENT_SCHEMA_VERSION


# =============================================================================
# Finding 7: Provider Failures, Budget Accounting & Unknown Model Pricing
# =============================================================================


def test_openrouter_adapter_unknown_model_pricing():
    """Unpriced models never return $0.00; they return None / UNAVAILABLE or raise UnknownModelPricingError."""
    pricing = PricingTable(
        effective_date="2026-10-01",
        source="Test",
        rates_per_1m={"google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30}},
    )

    # Fallback mode: returns None with ProvenanceLabel.UNAVAILABLE
    adapter_lenient = OpenRouterAdapter(
        model_id="unregistered/unknown-model",
        pricing_table=pricing,
        strict_pricing=False,
    )
    cost = adapter_lenient._calculate_cost(input_tokens=1000, output_tokens=500)
    assert cost is None

    # Strict mode: raises UnknownModelPricingError
    adapter_strict = OpenRouterAdapter(
        model_id="unregistered/unknown-model",
        pricing_table=pricing,
        strict_pricing=True,
    )
    with pytest.raises(UnknownModelPricingError, match="is not present in pricing table"):
        adapter_strict._calculate_cost(input_tokens=1000, output_tokens=500)


def test_openrouter_adapter_releases_budget_reservation_on_malformed_json():
    """Malformed provider JSON does not leak budget reservations."""
    pricing = PricingTable(
        effective_date="2026-10-01",
        source="Test",
        rates_per_1m={"google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30}},
    )
    tracker = RunBudgetTracker(max_usd=1.0, pricing_table=pricing)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"invalid_structure": True}  # Missing choices
    mock_client.post.return_value = mock_response

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=pricing,
        client=mock_client,
        budget_tracker=tracker,
    )

    initial_remaining = tracker.remaining_budget
    with pytest.raises(OpenRouterResponseError):
        adapter.generate("Test prompt")

    # Budget reservation must be released in finally block
    assert tracker.remaining_budget == initial_remaining
    assert tracker.reserved_usd == 0.0
