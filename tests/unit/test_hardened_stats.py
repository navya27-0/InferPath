"""Unit tests for hardened statistical procedures and human review provenance.

All fixtures and datasets in this file are SYNTHETIC and explicitly marked as such.
"""

import numpy as np
import pytest

from inferpath.evaluation.human_review import (
    HumanReviewRecord,
    compute_inter_rater_agreement,
    compute_weighted_cohen_kappa,
    validate_ratings,
)
from inferpath.evaluation.stats import (
    compute_clustered_bootstrap_ci,
    compute_hedges_g,
)


def test_clustered_bootstrap_deterministic_with_seed():
    """Clustered bootstrap must be 100% reproducible given the same seed."""
    # Synthetic family clusters
    synthetic_clusters = {
        "fam_1": [1.0, 1.0, 0.0],
        "fam_2": [0.0, 0.0, 1.0],
        "fam_3": [1.0, 1.0, 1.0],
        "fam_4": [0.0, 0.0, 0.0],
    }
    ci_1 = compute_clustered_bootstrap_ci(
        synthetic_clusters, n_resamples=500, confidence_level=0.95, seed=42
    )
    ci_2 = compute_clustered_bootstrap_ci(
        synthetic_clusters, n_resamples=500, confidence_level=0.95, seed=42
    )
    assert ci_1 == ci_2
    assert ci_1.estimator == "clustered_task_family_bootstrap"
    assert ci_1.seed == 42
    assert ci_1.n_resamples == 500
    assert ci_1.confidence_level == 0.95


def test_clustered_bootstrap_known_homogeneous_distribution():
    """Synthetic all-pass distribution must yield exact [1.0, 1.0] interval."""
    all_pass_clusters = {f"fam_{i}": [1.0, 1.0, 1.0] for i in range(10)}
    ci = compute_clustered_bootstrap_ci(all_pass_clusters, n_resamples=200, seed=123)
    assert ci.point_estimate == 1.0
    assert ci.ci_lower == 1.0
    assert ci.ci_upper == 1.0

    all_fail_clusters = {f"fam_{i}": [0.0, 0.0] for i in range(10)}
    ci_fail = compute_clustered_bootstrap_ci(all_fail_clusters, n_resamples=200, seed=123)
    assert ci_fail.point_estimate == 0.0
    assert ci_fail.ci_lower == 0.0
    assert ci_fail.ci_upper == 0.0


def test_clustered_bootstrap_invalid_parameters_rejected():
    """Invalid statistical parameters must raise ValueError."""
    synthetic_clusters = {"fam_1": [1.0, 0.0]}

    # Confidence level <= 0 or >= 1
    with pytest.raises(ValueError, match="confidence_level"):
        compute_clustered_bootstrap_ci(synthetic_clusters, confidence_level=0.0)
    with pytest.raises(ValueError, match="confidence_level"):
        compute_clustered_bootstrap_ci(synthetic_clusters, confidence_level=1.0)
    with pytest.raises(ValueError, match="confidence_level"):
        compute_clustered_bootstrap_ci(synthetic_clusters, confidence_level=1.2)

    # Resamples <= 0
    with pytest.raises(ValueError, match="n_resamples"):
        compute_clustered_bootstrap_ci(synthetic_clusters, n_resamples=0)
    with pytest.raises(ValueError, match="n_resamples"):
        compute_clustered_bootstrap_ci(synthetic_clusters, n_resamples=-10)


def test_clustered_bootstrap_empty_and_nan_handling():
    """Empty and malformed clusters must be handled gracefully."""
    ci_empty = compute_clustered_bootstrap_ci({})
    assert ci_empty.point_estimate == 0.0
    assert ci_empty.ci_lower == 0.0
    assert ci_empty.ci_upper == 0.0

    # Cluster with NaN / inf values
    clusters_with_nan = {
        "fam_1": [1.0, float("nan"), 0.0],
        "fam_2": [float("inf"), 1.0],
    }
    ci_nan = compute_clustered_bootstrap_ci(clusters_with_nan, n_resamples=100, seed=42)
    assert np.isfinite(ci_nan.point_estimate)
    assert np.isfinite(ci_nan.ci_lower)
    assert np.isfinite(ci_nan.ci_upper)


def test_hedges_g_identical_distributions():
    """Hedges' g on identical samples must be zero."""
    s1 = [0.8, 0.9, 0.85, 0.95]
    s2 = [0.8, 0.9, 0.85, 0.95]
    g = compute_hedges_g(s1, s2)
    assert g == pytest.approx(0.0)


def test_human_review_provenance_unestablished_when_empty():
    """When no genuine ratings exist, agreement must be reported as not established."""
    result = compute_inter_rater_agreement([])
    assert result["status"] == "not_established"
    assert result["human_human_kappa"] is None
    assert result["human_judge_kappa"] is None
    assert "No empirical human review records" in result["message"]


def test_human_review_record_validation():
    """Validates individual review records and catches corrupt/out-of-range ratings."""
    valid_record = HumanReviewRecord(
        review_id="REV_001_A",
        item_id="item_001",
        reviewer_id="rater_A",
        candidate_id="candidate_a",
        score=4,
        rubric_version="1.0.0",
        provenance="synthetic_unit_test",
    )
    assert valid_record.item_id == "item_001"

    # Out of range rating rejected by pydantic
    with pytest.raises(ValueError):
        HumanReviewRecord(
            review_id="REV_002_A",
            item_id="item_002",
            reviewer_id="rater_B",
            candidate_id="candidate_a",
            score=10,  # invalid for scale 1-5
            rubric_version="1.0.0",
            provenance="synthetic_unit_test",
        )

    # validate_ratings returns False on out-of-range rating
    assert validate_ratings([1, 2, 3, 4, 5]) is True
    assert validate_ratings([1, 2, 10], min_val=1, max_val=5) is False


def test_synthetic_cohens_kappa_calculation():
    """Unit test for weighted Cohen's Kappa using explicit synthetic ratings."""
    r1 = [1, 2, 3, 4, 5]
    r2 = [1, 2, 3, 4, 5]
    assert compute_weighted_cohen_kappa(r1, r2) == 1.0

    r3 = [1, 1, 1, 1, 1]
    r4 = [2, 2, 2, 2, 2]
    k = compute_weighted_cohen_kappa(r3, r4)
    assert k <= 0.0
