"""Mathematical formulation and candidate selection specification for InferPath routing.

Formalizes the two-stage constrained candidate selection and confidence escalation policy:

Stage 1: Hard Feasibility Filtering
  M_feasible(x) = { m in M :
      PrivacyAllowed(m, x)
      and EstimatedCost(m, x) <= Budget_max
      and EstimatedLatency(m, x) <= Latency_max
      and QualityPrior(m, x) >= Quality_min
  }

Stage 2: Constrained Scalarized Objective Optimization
  m* = argmin_{m in M_feasible(x)} [ EstimatedCost(m, x) + lambda * EstimatedLatency(m, x) - beta * QualityPrior(m, x) ]

Stage 3: Verification & Execution-Time Confidence Escalation
  If SelectedTier(x) fails deterministic validation or confidence threshold theta:
      Trigger fallback to m_fallback in M_feasible(x) with minimal risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferpath.models.domain import UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier


@dataclass
class TierPerformancePrior:
    tier: RouteTier
    estimated_cost_per_1k: float
    estimated_latency_ms: float
    quality_prior: float  # [0.0, 1.0]
    privacy_compliant: bool = True


DEFAULT_TIER_PRIORS: dict[RouteTier, TierPerformancePrior] = {
    RouteTier.DETERMINISTIC_A: TierPerformancePrior(
        tier=RouteTier.DETERMINISTIC_A,
        estimated_cost_per_1k=0.000001,
        estimated_latency_ms=1.5,
        quality_prior=0.99,  # High deterministic accuracy when template/regex matches
        privacy_compliant=True,
    ),
    RouteTier.TEMPLATE_B: TierPerformancePrior(
        tier=RouteTier.TEMPLATE_B,
        estimated_cost_per_1k=0.000005,
        estimated_latency_ms=2.5,
        quality_prior=0.95,
        privacy_compliant=True,
    ),
    RouteTier.SMALL_MODEL_C: TierPerformancePrior(
        tier=RouteTier.SMALL_MODEL_C,
        estimated_cost_per_1k=0.00015,
        estimated_latency_ms=25.0,
        quality_prior=0.72,
        privacy_compliant=True,
    ),
    RouteTier.LARGE_MODEL_D: TierPerformancePrior(
        tier=RouteTier.LARGE_MODEL_D,
        estimated_cost_per_1k=0.00350,
        estimated_latency_ms=85.0,
        quality_prior=0.92,
        privacy_compliant=True,
    ),
}


class ConstrainedOptimizationPolicy:
    """Computes exact feasible sets and evaluates scalarized cost-latency-quality objectives."""

    def __init__(
        self,
        lambda_tradeoff: float = 1e-4,  # $/ms tradeoff parameter
        beta_quality_weight: float = 0.005,  # Quality reward scale
        priors: dict[RouteTier, TierPerformancePrior] | None = None,
    ):
        self.lambda_tradeoff = lambda_tradeoff
        self.beta_quality_weight = beta_quality_weight
        self.priors = priors or DEFAULT_TIER_PRIORS

    def determine_feasible_set(self, request: UserRequest) -> list[RouteTier]:
        """Stage 1: Hard rule-based feasibility filtering.

        Filters candidates based on:
        - Deterministic / template eligibility (general open prompts cannot use regex/templates)
        - Privacy constraints (external transmission prohibited for CONFIDENTIAL/INTERNAL)
        - Caller budget ceilings
        - Caller latency deadlines
        - Minimum acceptable quality prior
        """
        is_sensitive = (
            request.privacy_class != PrivacyClass.PUBLIC or not request.allow_external_transmission
        )

        prompt_lower = request.prompt.lower()
        has_math_or_regex = any(
            kw in prompt_lower
            for kw in ["calculate", "evaluate", "convert", "date", "time", "+", "-", "*", "/"]
        )
        is_complex = any(
            kw in prompt_lower
            for kw in [
                "explain",
                "analyze",
                "why",
                "prove",
                "quantum",
                "differential",
                "summarize",
                "implications",
            ]
        )

        feasible: list[RouteTier] = []
        max_cost = request.maximum_cost_usd
        max_lat = request.maximum_latency_ms
        min_qual = request.minimum_quality

        for tier, prior in self.priors.items():
            qual = prior.quality_prior
            # Deterministic and template tiers cannot handle open-ended text unless pattern matches
            if tier in (RouteTier.DETERMINISTIC_A, RouteTier.TEMPLATE_B):
                if not has_math_or_regex and not is_sensitive:
                    qual = 0.0
            elif tier == RouteTier.SMALL_MODEL_C and is_complex:
                qual = 0.65

            # Privacy barrier: external LLM tiers excluded if request is sensitive
            if is_sensitive and tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D):
                continue

            # Hard budget barrier
            if max_cost is not None and prior.estimated_cost_per_1k > max_cost:
                continue

            # Hard latency barrier
            if max_lat is not None and prior.estimated_latency_ms > max_lat:
                continue

            # Quality prior barrier
            if qual < min_qual:
                continue

            feasible.append(tier)

        # Fallback guarantee: if feasible set is empty due to over-constrained parameters,
        # return highest-capability privacy-compliant tier
        if not feasible:
            feasible = [RouteTier.DETERMINISTIC_A] if is_sensitive else [RouteTier.LARGE_MODEL_D]

        return feasible

    def compute_objective_score(self, tier: RouteTier) -> float:
        """Stage 2: Scalarized objective function J(m, x).

        J(m, x) = Cost(m, x) + lambda * Latency(m, x) - beta * Quality(m, x)
        Lower is better.
        """
        prior = self.priors.get(tier, DEFAULT_TIER_PRIORS[RouteTier.LARGE_MODEL_D])
        cost = prior.estimated_cost_per_1k
        lat = prior.estimated_latency_ms
        qual = prior.quality_prior

        loss = cost + (self.lambda_tradeoff * lat) - (self.beta_quality_weight * qual)
        return loss

    def select_optimal_tier(self, request: UserRequest) -> tuple[RouteTier, list[RouteTier], float]:
        """Finds tier m* in Feasible(x) that minimizes J(m, x)."""
        feasible = self.determine_feasible_set(request)
        best_tier = feasible[0]
        best_score = float("inf")

        for tier in feasible:
            score = self.compute_objective_score(tier)
            if score < best_score:
                best_score = score
                best_tier = tier

        return best_tier, feasible, best_score
