"""Unit tests for simulator calibration and validation."""

from inferpath.evaluation.simulator_calibration import (
    compute_failure_mode_alignment,
    compute_ks_statistic,
    evaluate_quality_calibration_error,
    run_sensitivity_analysis,
)


def test_ks_statistic_identical():
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = compute_ks_statistic(sample, sample)
    assert res["ks_statistic"] == 0.0
    assert res["p_value_approx"] == 1.0


def test_ks_statistic_different():
    s1 = [1.0, 2.0, 3.0]
    s2 = [10.0, 20.0, 30.0]
    res = compute_ks_statistic(s1, s2)
    assert res["ks_statistic"] == 1.0


def test_failure_mode_alignment():
    f1 = {"timeout": 10, "format_error": 5}
    f2 = {"timeout": 10, "format_error": 5}
    res = compute_failure_mode_alignment(f1, f2)
    assert res["total_variation_distance"] == 0.0
    assert res["cosine_similarity"] == 1.0


def test_quality_calibration_error():
    probs = [0.8, 0.9, 0.2, 0.1]
    outcomes = [1, 1, 0, 0]
    res = evaluate_quality_calibration_error(probs, outcomes)
    assert res["ece"] >= 0.0
    assert res["brier_score"] <= 0.05


def test_sensitivity_analysis():
    res = run_sensitivity_analysis(0.004, 25.0, 0.65)
    assert len(res) == 7
    assert all("is_stable" in r for r in res)
