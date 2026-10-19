"""Unit tests for expanded human review evaluation methods."""

from inferpath.evaluation.human_review import (
    adjudicate_ratings,
    bootstrap_human_score_ci,
    compute_fleiss_kappa,
)


def test_compute_fleiss_kappa():
    # Perfect agreement
    matrix_perfect = [[5, 5, 5], [4, 4, 4], [3, 3, 3], [2, 2, 2]]
    k_perfect = compute_fleiss_kappa(matrix_perfect, categories=[1, 2, 3, 4, 5])
    assert k_perfect == 1.0

    # Variable ratings
    matrix_var = [[5, 4, 5], [4, 4, 3], [3, 2, 3], [1, 1, 2]]
    k_var = compute_fleiss_kappa(matrix_var, categories=[1, 2, 3, 4, 5])
    assert -1.0 <= k_var <= 1.0


def test_adjudicate_ratings():
    # Within threshold (no adjudication needed)
    score, was_adj = adjudicate_ratings([4, 4, 5], threshold_diff=1)
    assert was_adj is False
    assert score == 4

    # Exceeds threshold (adjudication needed)
    score_adj, was_adj2 = adjudicate_ratings([2, 5, 5], threshold_diff=1, adjudicator_score=4)
    assert was_adj2 is True
    assert score_adj == 4


def test_bootstrap_human_score_ci():
    scores = [4, 4, 5, 4, 5, 3, 4, 5]
    mean, ci_low, ci_high = bootstrap_human_score_ci(scores, resamples=500)
    assert ci_low <= mean <= ci_high
    assert 3.0 <= ci_low <= 5.0
