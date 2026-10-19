"""Total Cost of Ownership (TCO) model for InferPath.

Accounts for:
1. Model inference cost (tokens).
2. Router compute cost (CPU time on AWS Lambda / Cloud Run @ $0.0000166667 / vCPU-sec).
3. Output validation & parsing compute.
4. Fallback invocation costs.
5. Cache infrastructure (SQLite disk / Redis cluster @ $0.025 / GB-hour).
6. Observability & logging (@ $0.50 / GB ingested).
7. Network egress (@ $0.09 / GB).

Evaluates overhead percentage across prompt sizes:
- Ultra-short prompts (10-25 tokens)
- Medium queries (100-250 tokens)
- Long context queries (1000-4000 tokens)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TCOCostBreakdown:
    prompt_tokens: int
    completion_tokens: int
    inference_cost_usd: float
    router_compute_cost_usd: float
    validation_cost_usd: float
    fallback_cost_usd: float
    cache_infrastructure_cost_usd: float
    observability_cost_usd: float
    network_egress_cost_usd: float
    total_cost_usd: float
    overhead_percentage: float


class TotalCostOfOwnershipEstimator:
    """Calculates comprehensive end-to-end system cost per query and per million queries."""

    # Industry cloud cost benchmarks (AWS / GCP 2026)
    VCPU_SEC_COST_USD = 0.0000166667  # AWS Lambda / Google Cloud Run per vCPU-sec
    LOGGING_COST_PER_MB_USD = 0.00050  # $0.50 per GB CloudWatch / Datadog
    EGRESS_COST_PER_MB_USD = 0.00009  # $0.09 per GB
    REDIS_COST_PER_QUERY_USD = 0.0000005  # Amortized Redis cluster cost

    def estimate_single_query_tco(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        inference_cost_usd: float,
        router_cpu_ms: float = 0.5,
        validation_cpu_ms: float = 0.3,
        had_fallback: bool = False,
        fallback_cost_usd: float = 0.0035,
    ) -> TCOCostBreakdown:
        # 1. Router compute cost
        router_compute = (router_cpu_ms / 1000.0) * self.VCPU_SEC_COST_USD
        # 2. Validation compute cost
        val_compute = (validation_cpu_ms / 1000.0) * self.VCPU_SEC_COST_USD
        # 3. Fallback cost
        fb_cost = fallback_cost_usd if had_fallback else 0.0
        # 4. Cache infrastructure
        cache_cost = self.REDIS_COST_PER_QUERY_USD
        # 5. Observability (estimated ~1 KB structured JSON log + span)
        log_cost = (1.0 / 1024.0) * self.LOGGING_COST_PER_MB_USD
        # 6. Egress (prompt + completion tokens * 4 bytes)
        payload_mb = ((prompt_tokens + completion_tokens) * 4) / (1024.0 * 1024.0)
        egress_cost = payload_mb * self.EGRESS_COST_PER_MB_USD

        overhead = router_compute + val_compute + cache_cost + log_cost + egress_cost
        total = inference_cost_usd + fb_cost + overhead
        overhead_pct = (overhead / total * 100.0) if total > 0 else 0.0

        return TCOCostBreakdown(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            inference_cost_usd=round(inference_cost_usd, 7),
            router_compute_cost_usd=round(router_compute, 7),
            validation_cost_usd=round(val_compute, 7),
            fallback_cost_usd=round(fb_cost, 7),
            cache_infrastructure_cost_usd=round(cache_cost, 7),
            observability_cost_usd=round(log_cost, 7),
            network_egress_cost_usd=round(egress_cost, 7),
            total_cost_usd=round(total, 7),
            overhead_percentage=round(overhead_pct, 3),
        )

    def evaluate_workload_tco(
        self, prompt_lengths: list[int] | None = None
    ) -> list[dict[str, Any]]:
        lengths = prompt_lengths or [15, 50, 200, 1000, 4000]
        results = []
        for pl in lengths:
            comp_len = max(20, pl // 2)
            # Simulated inference cost for small tier (Tier C): $0.10/M in, $0.40/M out
            inf_cost = (pl * 0.00000010) + (comp_len * 0.00000040)
            res = self.estimate_single_query_tco(
                prompt_tokens=pl,
                completion_tokens=comp_len,
                inference_cost_usd=inf_cost,
            )
            results.append(
                {
                    "prompt_tokens": pl,
                    "completion_tokens": comp_len,
                    "inference_cost_usd": res.inference_cost_usd,
                    "router_overhead_usd": round(
                        res.router_compute_cost_usd
                        + res.validation_cost_usd
                        + res.cache_infrastructure_cost_usd
                        + res.observability_cost_usd,
                        7,
                    ),
                    "total_tco_usd": res.total_cost_usd,
                    "router_overhead_percentage": res.overhead_percentage,
                    "breakeven_assessment": "Overhead negligible (<2%)"
                    if res.overhead_percentage < 2.0
                    else "Meaningful overhead on short prompts",
                }
            )
        return results
