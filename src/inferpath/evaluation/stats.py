"""Statistical utilities for clustered bootstrap confidence intervals, effect sizes, and run aggregations."""

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel


class BootstrapCI(BaseModel):
    """Confidence interval produced via bootstrap resampling."""

    mean: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_resamples: int = 1000
    estimator: str = "clustered_task_family_bootstrap"
    seed: int = 42

    @property
    def point_estimate(self) -> float:
        return self.mean


def compute_hedges_g(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Computes Hedges' g standardized mean difference with small-sample correction."""
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = sum(sample_a) / n_a
    mean_b = sum(sample_b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)

    s_pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if s_pooled == 0.0:
        return 0.0

    d = (mean_a - mean_b) / s_pooled
    df = n_a + n_b - 2
    j_factor = 1.0 - (3.0 / (4.0 * df - 1)) if df > 1 else 1.0
    return round(float(d * j_factor), 4)


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
    """Computes a bootstrap confidence interval respecting task-family clustering.

    Resamples clusters (task families) with replacement rather than individual points.
    Preserves intra-cluster correlation across items within the same task family.

    Raises:
        ValueError: If confidence_level is not strictly in (0.0, 1.0) or n_resamples < 1.
    """
    if num_bootstraps is not None:
        n_resamples = num_bootstraps

    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError(
            f"confidence_level must be strictly between 0.0 and 1.0, got {confidence_level}"
        )
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")

    # Clean cluster values to only include finite numeric observations
    cleaned_clusters: dict[str, list[float]] = {}
    for k_id, vals in cluster_values.items():
        if vals is not None:
            clean_vals = [
                float(v) for v in vals if v is not None and not math.isnan(v) and not math.isinf(v)
            ]
            if clean_vals:
                cleaned_clusters[k_id] = clean_vals

    cluster_keys = list(cleaned_clusters.keys())
    k = len(cluster_keys)

    if k == 0:
        return BootstrapCI(
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            estimator="clustered_task_family_bootstrap",
            seed=seed,
        )

    # Flatten all points for nominal mean
    all_points = [val for vals in cleaned_clusters.values() for val in vals]
    if not all_points:
        return BootstrapCI(
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            estimator="clustered_task_family_bootstrap",
            seed=seed,
        )

    nominal_mean = sum(all_points) / len(all_points)

    rng = random.Random(seed)
    resampled_means: list[float] = []
    for _ in range(n_resamples):
        sampled_clusters = rng.choices(cluster_keys, k=k)
        sampled_vals = [v for c in sampled_clusters for v in cleaned_clusters[c]]
        if sampled_vals:
            resampled_means.append(sum(sampled_vals) / len(sampled_vals))

    if not resampled_means:
        return BootstrapCI(
            mean=round(nominal_mean, 4),
            ci_lower=round(nominal_mean, 4),
            ci_upper=round(nominal_mean, 4),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            estimator="clustered_task_family_bootstrap",
            seed=seed,
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
        estimator="clustered_task_family_bootstrap",
        seed=seed,
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
