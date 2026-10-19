"""Unit tests for routing regret and oracle metrics."""

from inferpath.evaluation.regret import (
    RouteTier,
    TierExecutionRealization,
    aggregate_regret_metrics,
    compute_oracle_tier,
    compute_routing_regret,
)


def _sample_realizations():
    return {
        RouteTier.DETERMINISTIC_A: TierExecutionRealization(
            tier=RouteTier.DETERMINISTIC_A,
            quality_score=0.0,
            cost_usd=0.000001,
            latency_ms=1.0,
            is_success=False,
        ),
        RouteTier.TEMPLATE_B: TierExecutionRealization(
            tier=RouteTier.TEMPLATE_B,
            quality_score=0.0,
            cost_usd=0.000005,
            latency_ms=2.0,
            is_success=False,
        ),
        RouteTier.SMALL_MODEL_C: TierExecutionRealization(
            tier=RouteTier.SMALL_MODEL_C,
            quality_score=0.85,
            cost_usd=0.00015,
            latency_ms=25.0,
            is_success=True,
        ),
        RouteTier.LARGE_MODEL_D: TierExecutionRealization(
            tier=RouteTier.LARGE_MODEL_D,
            quality_score=0.95,
            cost_usd=0.00350,
            latency_ms=85.0,
            is_success=True,
        ),
    }


def test_oracle_tier_selection():
    realizations = _sample_realizations()
    # When Q_min = 0.70, Small model satisfies it and is cheapest
    oracle = compute_oracle_tier(realizations, minimum_quality=0.70)
    assert oracle == RouteTier.SMALL_MODEL_C

    # When Q_min = 0.90, only Large model satisfies it
    oracle_high = compute_oracle_tier(realizations, minimum_quality=0.90)
    assert oracle_high == RouteTier.LARGE_MODEL_D


def test_routing_regret_with_fallback():
    realizations = _sample_realizations()
    # Router selected Deterministic (failed), fell back to Large
    rec = compute_routing_regret(
        item_id="item_01",
        selected_tier=RouteTier.DETERMINISTIC_A,
        realizations=realizations,
        had_fallback=True,
        fallback_tier=RouteTier.LARGE_MODEL_D,
        minimum_quality=0.70,
    )
    # Oracle is Small Model ($0.00015)
    # Actual cost = Det ($0.000001) + Large ($0.0035) = $0.003501
    assert rec.is_under_escalation is True
    assert rec.cost_regret_usd > 0.003
    assert rec.quality_regret == 0.0  # Fallback large quality (0.95) >= oracle (0.85)


def test_routing_regret_cache_hit():
    realizations = _sample_realizations()
    rec = compute_routing_regret(
        item_id="item_cache",
        selected_tier=RouteTier.SMALL_MODEL_C,
        realizations=realizations,
        is_cache_hit=True,
    )
    assert rec.actual_cost_usd == 0.0
    assert rec.cost_regret_usd == 0.0


def test_aggregate_regret_metrics():
    realizations = _sample_realizations()
    r1 = compute_routing_regret("1", RouteTier.SMALL_MODEL_C, realizations, minimum_quality=0.70)
    r2 = compute_routing_regret("2", RouteTier.LARGE_MODEL_D, realizations, minimum_quality=0.70)
    summary = aggregate_regret_metrics([r1, r2])
    assert summary["total_items"] == 2
    assert summary["over_escalation_rate"] == 0.5
    assert summary["under_escalation_rate"] == 0.0
