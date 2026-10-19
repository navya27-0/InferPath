"""Unit tests for privacy evaluation module."""

from inferpath.evaluation.privacy_eval import (
    build_privacy_probe_suite,
    detect_sensitive_content,
    evaluate_privacy_classifier,
)


def test_detect_sensitive_content():
    # Direct secret
    assert detect_sensitive_content("My secret password is 12345") is True
    # SSN
    assert detect_sensitive_content("SSN: 000-11-2222") is True
    # Context leak
    assert detect_sensitive_content("Hello", context="bearer token secret_val") is True
    # Benign
    assert detect_sensitive_content("Calculate 2 + 2") is False


def test_evaluate_privacy_classifier():
    probes = build_privacy_probe_suite()
    assert len(probes) >= 15
    res = evaluate_privacy_classifier(probes)
    assert res["total_probes"] == len(probes)
    assert "confusion_matrix" in res
    assert res["metrics"]["precision"] >= 0.90
    assert 0.0 <= res["metrics"]["recall_sensitivity"] <= 1.0
