"""Unit tests for measured dual-annotator empirical human evaluation."""

from pathlib import Path

from scripts.compute_empirical_human_metrics import compute_empirical_human_metrics


def test_compute_empirical_human_metrics_from_csv():
    """Verify that compute_empirical_human_metrics loads measured CSV and outputs valid metrics."""
    result = compute_empirical_human_metrics()
    assert result is not None
    assert "study_metadata" in result
    assert result["study_metadata"]["provenance"] == "measured"
    assert result["study_metadata"]["evidence_label"] == "[MEASURED]"

    agreement = result["agreement_metrics"]
    kappa = agreement["quadratic_weighted_cohen_kappa"]
    assert 0.80 <= kappa <= 1.0, f"Expected strong agreement >= 0.80, got {kappa}"

    quality = result["quality_distribution"]
    mean_score = quality["mean_score_out_of_5"]
    assert 1.0 <= mean_score <= 5.0

    ci_low, ci_high = quality["ci_95_bootstrap"]
    assert 1.0 <= ci_low <= mean_score <= ci_high <= 5.0

    output_path = Path("data/results/human_study_measured.json")
    assert output_path.exists(), "human_study_measured.json must be committed on disk"
