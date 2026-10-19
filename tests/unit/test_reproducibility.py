"""Unit tests for reproducibility and public manifest module."""

from inferpath.evaluation.reproducibility import (
    compute_dict_sha256,
    compute_file_sha256,
    sanitize_run_manifest,
)


def test_compute_file_sha256():
    # Test on pyproject.toml
    h = compute_file_sha256("pyproject.toml")
    assert len(h) == 64
    assert h != "file_not_found"

    # Non-existent file
    h_none = compute_file_sha256("non_existent_file_xyz.txt")
    assert h_none == "file_not_found"


def test_compute_dict_sha256():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    # Deterministic canonical serialization
    assert compute_dict_sha256(d1) == compute_dict_sha256(d2)


def test_sanitize_run_manifest():
    raw = {
        "run_id": "run_test_01",
        "timestamp": "2026-10-19T00:00:00Z",
        "model_id": "test/model",
        "summary_metrics": {"total_items": 10},
    }
    sanitized = sanitize_run_manifest(raw)
    assert sanitized["run_id"] == "run_test_01"
    assert "network_and_transport" in sanitized
    assert "hashes" in sanitized
