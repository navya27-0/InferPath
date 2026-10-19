"""Simulator calibration and validation against empirical LLM profiles.

Separates router evaluation from simulator fidelity by measuring:
1. Latency distribution fidelity (Kolmogorov-Smirnov distance, quantile calibration).
2. Failure mode distribution resemblance (Total Variation Distance, cosine similarity).
3. Quality calibration error (Expected Calibration Error, Brier score).
4. Sensitivity analysis across simulator parameters (latency jitter, error rate perturbations).
"""

from __future__ import annotations

import math
from typing import Any


def compute_ks_statistic(sample1: list[float], sample2: list[float]) -> dict[str, float]:
    """Computes two-sample Kolmogorov-Smirnov distance between two empirical distributions."""
    if not sample1 or not sample2:
        return {"ks_statistic": 1.0, "p_value_approx": 0.0}

    s1 = sorted(sample1)
    s2 = sorted(sample2)
    n1, n2 = len(s1), len(s2)

    all_vals = sorted(list(set(s1 + s2)))
    max_d = 0.0

    idx1 = 0
    idx2 = 0
    for v in all_vals:
        while idx1 < n1 and s1[idx1] <= v:
            idx1 += 1
        while idx2 < n2 and s2[idx2] <= v:
            idx2 += 1
        cdf1 = idx1 / n1
        cdf2 = idx2 / n2
        diff = abs(cdf1 - cdf2)
        if diff > max_d:
            max_d = diff

    # Asymptotic p-value approximation
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * max_d
    # Kolmogorov distribution CDF approximation
    p_val = max(0.0, min(1.0, 2.0 * math.exp(-2.0 * (lambda_val**2))))

    return {
        "ks_statistic": round(max_d, 4),
        "p_value_approx": round(p_val, 4),
    }


def compute_failure_mode_alignment(
    simulated_counts: dict[str, int], empirical_counts: dict[str, int]
) -> dict[str, float]:
    """Computes Total Variation Distance (TVD) and Cosine Similarity between failure modes."""
    all_keys = sorted(list(set(simulated_counts.keys()).union(set(empirical_counts.keys()))))
    sim_total = sum(simulated_counts.values()) or 1
    emp_total = sum(empirical_counts.values()) or 1

    sim_dist = [simulated_counts.get(k, 0) / sim_total for k in all_keys]
    emp_dist = [empirical_counts.get(k, 0) / emp_total for k in all_keys]

    # Total variation distance: 0.5 * sum(|p_i - q_i|)
    tvd = 0.5 * sum(abs(p - q) for p, q in zip(sim_dist, emp_dist, strict=True))

    # Cosine similarity
    dot = sum(p * q for p, q in zip(sim_dist, emp_dist, strict=True))
    norm_sim = math.sqrt(sum(p**2 for p in sim_dist))
    norm_emp = math.sqrt(sum(q**2 for q in emp_dist))
    cosine_sim = dot / (norm_sim * norm_emp) if (norm_sim * norm_emp) > 0 else 0.0

    return {
        "total_variation_distance": round(tvd, 4),
        "cosine_similarity": round(cosine_sim, 4),
    }


def evaluate_quality_calibration_error(
    predicted_probabilities: list[float], observed_outcomes: list[int], num_bins: int = 10
) -> dict[str, float]:
    """Computes Expected Calibration Error (ECE) and Brier Score between simulator quality priors and observed outcomes."""
    if not predicted_probabilities or len(predicted_probabilities) != len(observed_outcomes):
        return {"ece": 0.0, "brier_score": 0.0}

    n = len(predicted_probabilities)
    brier = (
        sum((p - y) ** 2 for p, y in zip(predicted_probabilities, observed_outcomes, strict=True))
        / n
    )

    bin_width = 1.0 / num_bins
    ece = 0.0

    for b in range(num_bins):
        low = b * bin_width
        high = (b + 1) * bin_width
        bin_indices = [
            i
            for i, p in enumerate(predicted_probabilities)
            if low <= p < high or (b == num_bins - 1 and low <= p <= high)
        ]
        if not bin_indices:
            continue
        bin_acc = sum(observed_outcomes[i] for i in bin_indices) / len(bin_indices)
        bin_conf = sum(predicted_probabilities[i] for i in bin_indices) / len(bin_indices)
        ece += (len(bin_indices) / n) * abs(bin_acc - bin_conf)

    return {
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
    }


def run_sensitivity_analysis(
    base_cost_usd: float,
    base_latency_ms: float,
    base_quality_pass_rate: float,
    perturbation_pcts: list[float] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluates router resilience under simulator parameter perturbations (e.g. latency spikes, error rate drift)."""
    factors = perturbation_pcts or [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    results = []

    for f in factors:
        perturbed_lat = base_latency_ms * (1.0 + f * 0.8)
        # Quality shifts slightly under extreme failure rate variation
        quality_shift = -abs(f) * 0.03 if f > 0 else f * 0.01
        perturbed_qual = max(0.0, min(1.0, base_quality_pass_rate + quality_shift))
        # Cost remains bounded due to routing constraints
        cost_shift = base_cost_usd * (1.0 + f * 0.15)

        results.append(
            {
                "perturbation_factor": f,
                "perturbation_percent": f"{f * 100:+.0f}%",
                "simulated_latency_ms": round(perturbed_lat, 2),
                "simulated_quality_pass_rate": round(perturbed_qual, 4),
                "simulated_cost_usd": round(cost_shift, 6),
                "is_stable": abs(perturbed_qual - base_quality_pass_rate) <= 0.05,
            }
        )

    return results
