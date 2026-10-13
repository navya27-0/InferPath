"""Statistical utilities for clustered bootstrap confidence intervals, effect sizes, and run aggregations."""

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel


class BootstrapCI(BaseModel):
    """95% confidence interval produced via bootstrap resampling."""

    mean: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_resamples: int = 1000


class RunSummaryStats(BaseModel):
    """Aggregated statistics across repeated experiment runs."""

    mean: float
    std: float
    median: float
    iqr: float
    min_val: float
    max_val: float
    sample_size: int


def compute_clustered_bootstrap_ci(
    cluster_values: dict[str, list[float]],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    num_bootstraps: int | None = None,
) -> BootstrapCI:
    """
    Computes a bootstrap confidence interval respecting task-family clustering.
    Resamples clusters (task families) with replacement rather than individual points.
    """
    if num_bootstraps is not None:
        n_resamples = num_bootstraps
    rng = random.Random(seed)
    cluster_keys = list(cluster_values.keys())
    k = len(cluster_keys)

    if k == 0:
        return BootstrapCI(mean=0.0, ci_lower=0.0, ci_upper=0.0)

    # Flatten all points for nominal mean
    all_points = [val for vals in cluster_values.values() for val in vals]
    if not all_points:
        return BootstrapCI(mean=0.0, ci_lower=0.0, ci_upper=0.0)

    nominal_mean = sum(all_points) / len(all_points)

    resampled_means: list[float] = []
    for _ in range(n_resamples):
        sampled_clusters = rng.choices(cluster_keys, k=k)
        sampled_vals = [v for c in sampled_clusters for v in cluster_values[c]]
        if sampled_vals:
            resampled_means.append(sum(sampled_vals) / len(sampled_vals))

    if not resampled_means:
        return BootstrapCI(
            mean=round(nominal_mean, 4),
            ci_lower=round(nominal_mean, 4),
            ci_upper=round(nominal_mean, 4),
        )

    resampled_means.sort()
    alpha = (1.0 - confidence_level) / 2.0
    idx_lower = int(math.floor(alpha * len(resampled_means)))
    idx_upper = int(math.ceil((1.0 - alpha) * len(resampled_means))) - 1

    idx_lower = max(0, min(idx_lower, len(resampled_means) - 1))
    idx_upper = max(0, min(idx_upper, len(resampled_means) - 1))

    return BootstrapCI(
        mean=round(nominal_mean, 4),
        ci_lower=round(resampled_means[idx_lower], 4),
        ci_upper=round(resampled_means[idx_upper], 4),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )


def compute_cliffs_delta(sample_a: Sequence[float], sample_b: Sequence[float]) -> tuple[float, str]:
    """
    Computes Cliff's delta non-parametric effect size for ordinal or continuous comparisons.
    Interpretation thresholds: |d| < 0.147 (negligible), < 0.33 (small), < 0.474 (medium), >= 0.474 (large).
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a == 0 or n_b == 0:
        return 0.0, "negligible"

    greater = 0
    lesser = 0
    for a in sample_a:
        for b in sample_b:
            if a > b:
                greater += 1
            elif a < b:
                lesser += 1

    d = (greater - lesser) / (n_a * n_b)
    abs_d = abs(d)

    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.330:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"

    return round(float(d), 4), interp


def compute_repeated_run_summary(values: Sequence[float]) -> RunSummaryStats:
    """Computes mean, std, median, IQR, min, max across repeated runs."""
    n = len(values)
    if n == 0:
        return RunSummaryStats(
            mean=0.0, std=0.0, median=0.0, iqr=0.0, min_val=0.0, max_val=0.0, sample_size=0
        )

    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(variance)

    sorted_vals = sorted(values)
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[(n // 2) - 1] + sorted_vals[n // 2]) / 2.0

    # IQR calculation
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]
    iqr_val = q3 - q1

    return RunSummaryStats(
        mean=round(mean_val, 4),
        std=round(std_val, 4),
        median=round(median_val, 4),
        iqr=round(iqr_val, 4),
        min_val=round(min(values), 4),
        max_val=round(max(values), 4),
        sample_size=n,
    )
