"""Formal routing regret and oracle baseline evaluation framework.

Defines information boundaries, counterfactual oracle selection, and regret accounting:

1. Routing-Time Information (I_0):
   Prompt text, character length, detected keywords, privacy class, caller constraints,
   and prior performance estimates. Pre-execution outcome is unknown.

2. Counterfactual Oracle Information (I_oracle):
   Post-hoc omniscience. Knows true execution success, exact token counts, and true
   latencies for all candidate tiers m in M.

3. Formal Regret Metrics:
   - Cost Regret:
     R_cost = ActualCost(m_selected) + I(fallback)*ActualCost(m_fallback) + Cost_overhead - Cost(m_oracle)
   - Quality Regret:
     R_quality = max(0, Q(m_oracle) - Q_final)
   - Over-Escalation Rate:
     Fraction where router routed to large model when small model/template satisfied Q_min.
   - Under-Escalation Rate:
     Fraction where router routed to small model/deterministic and failed, triggering fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inferpath.models.enums import RouteTier


@dataclass
class TierExecutionRealization:
    """True post-hoc realization of executing prompt x on tier m."""

    tier: RouteTier
    quality_score: float  # [0.0, 1.0]
    cost_usd: float
    latency_ms: float
    is_success: bool


@dataclass
class RoutingRegretRecord:
    item_id: str
    selected_tier: RouteTier
    oracle_tier: RouteTier
    actual_cost_usd: float
    oracle_cost_usd: float
    cost_regret_usd: float
    actual_quality: float
    oracle_quality: float
    quality_regret: float
    had_fallback: bool
    is_over_escalation: bool
    is_under_escalation: bool
    is_cache_hit: bool


def compute_oracle_tier(
    realizations: dict[RouteTier, TierExecutionRealization],
    minimum_quality: float = 0.70,
    is_sensitive: bool = False,
) -> RouteTier:
    """Finds the lowest-cost tier that satisfies privacy and minimum quality in post-hoc realization."""
    eligible_tiers = []
    for tier, real in realizations.items():
        if is_sensitive and tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D):
            continue
        if real.quality_score >= minimum_quality:
            eligible_tiers.append(real)

    if not eligible_tiers:
        # Fallback to highest quality tier available
        best_real = max(
            realizations.values(),
            key=lambda r: (
                0
                if is_sensitive and r.tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D)
                else r.quality_score
            ),
        )
        return best_real.tier

    # Oracle selects the minimal-cost feasible tier
    cheapest = min(eligible_tiers, key=lambda r: r.cost_usd)
    return cheapest.tier


def compute_routing_regret(
    item_id: str,
    selected_tier: RouteTier,
    realizations: dict[RouteTier, TierExecutionRealization],
    had_fallback: bool = False,
    fallback_tier: RouteTier = RouteTier.LARGE_MODEL_D,
    validation_overhead_ms: float = 0.5,
    is_cache_hit: bool = False,
    minimum_quality: float = 0.70,
    is_sensitive: bool = False,
) -> RoutingRegretRecord:
    """Computes comprehensive regret metrics for a single query."""
    oracle_tier = compute_oracle_tier(
        realizations, minimum_quality=minimum_quality, is_sensitive=is_sensitive
    )

    oracle_real = realizations[oracle_tier]
    oracle_cost = 0.0 if is_cache_hit else oracle_real.cost_usd
    oracle_qual = oracle_real.quality_score

    if is_cache_hit:
        actual_cost = 0.0
        actual_qual = oracle_qual
    else:
        selected_real = realizations[selected_tier]
        actual_cost = selected_real.cost_usd
        actual_qual = selected_real.quality_score
        if had_fallback:
            fallback_real = realizations.get(fallback_tier, realizations[RouteTier.LARGE_MODEL_D])
            actual_cost += fallback_real.cost_usd
            actual_qual = fallback_real.quality_score

    cost_regret = max(0.0, actual_cost - oracle_cost)
    quality_regret = max(0.0, oracle_qual - actual_qual)

    # Over-escalation: router selected large tier when small tier would have satisfied Q_min
    is_over = selected_tier == RouteTier.LARGE_MODEL_D and oracle_tier in (
        RouteTier.SMALL_MODEL_C,
        RouteTier.DETERMINISTIC_A,
        RouteTier.TEMPLATE_B,
    )

    # Under-escalation: router selected small/deterministic, but failed, needing fallback
    is_under = had_fallback or (
        selected_tier in (RouteTier.SMALL_MODEL_C, RouteTier.DETERMINISTIC_A, RouteTier.TEMPLATE_B)
        and oracle_tier == RouteTier.LARGE_MODEL_D
    )

    return RoutingRegretRecord(
        item_id=item_id,
        selected_tier=selected_tier,
        oracle_tier=oracle_tier,
        actual_cost_usd=round(actual_cost, 6),
        oracle_cost_usd=round(oracle_cost, 6),
        cost_regret_usd=round(cost_regret, 6),
        actual_quality=round(actual_qual, 4),
        oracle_quality=round(oracle_qual, 4),
        quality_regret=round(quality_regret, 4),
        had_fallback=had_fallback,
        is_over_escalation=is_over,
        is_under_escalation=is_under,
        is_cache_hit=is_cache_hit,
    )


def aggregate_regret_metrics(records: list[RoutingRegretRecord]) -> dict[str, Any]:
    """Aggregates individual regret records into population summary statistics."""
    if not records:
        return {}

    n = len(records)
    total_cost_regret = sum(r.cost_regret_usd for r in records)
    total_quality_regret = sum(r.quality_regret for r in records)
    over_count = sum(1 for r in records if r.is_over_escalation)
    under_count = sum(1 for r in records if r.is_under_escalation)
    fallback_count = sum(1 for r in records if r.had_fallback)

    return {
        "total_items": n,
        "mean_cost_regret_usd": round(total_cost_regret / n, 6),
        "total_cost_regret_usd": round(total_cost_regret, 4),
        "mean_quality_regret": round(total_quality_regret / n, 4),
        "over_escalation_rate": round(over_count / n, 4),
        "under_escalation_rate": round(under_count / n, 4),
        "fallback_rate": round(fallback_count / n, 4),
    }
