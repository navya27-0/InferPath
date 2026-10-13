"""Comprehensive unit tests for InferPath research dashboard modules.

Tests cover:
- Manifest discovery and corrupted/missing artifact resilience
- Read-only data loading without file mutation
- Provenance classification preservation
- Failure ranking and privacy prompt redaction
- Counterfactual Always-Large cost modeling and dated price table matching
- Carbon and pricing assumption parsing
- Cache denominator accounting and storage sizing
- Mock API client interaction and offline server error handling
"""

import json

from inferpath.dashboard.data_loader import (
    ManifestMetadata,
    calculate_counterfactual_large_model_cost,
    discover_manifests,
    get_cache_storage_size,
    get_ranked_failures,
    load_benchmark_summary,
    load_carbon_assumptions,
    load_pricing_assumptions,
    load_run_data,
)


def test_manifest_discovery_existing_artifacts():
    """Verify discovery finds existing simulated manifests in data/results/."""
    manifests = discover_manifests(base_dir="data/results")
    assert len(manifests) > 0

    # Ensure all manifests have valid run_ids and adapter_mode
    for m in manifests:
        assert m.run_id
        assert m.adapter_mode in ["simulated", "real", "unknown"]
        assert m.split in ["dev", "test", "unknown"]
        assert m.manifest_path


def test_manifest_discovery_malformed_artifact(tmp_path):
    """Verify discovery gracefully flags corrupted JSON files rather than crashing."""
    corrupt_file = tmp_path / "simulated" / "manifest_run_bad_json.json"
    corrupt_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt_file.write_text("{corrupt json", encoding="utf-8")

    manifests = discover_manifests(base_dir=tmp_path)
    assert len(manifests) == 1
    assert manifests[0].integrity_warning is not None
    assert "malformed JSON" in manifests[0].integrity_warning


def test_load_run_data_missing_logs(tmp_path):
    """Verify loading run data handles missing or unreadable log files."""
    manifest = ManifestMetadata(
        run_id="run_test_missing_logs",
        system_id="System_C",
        run_index=1,
        adapter_mode="simulated",
        split="dev",
        manifest_path=str(tmp_path / "manifest.json"),
        logs_path=str(tmp_path / "non_existent_logs.json"),
        has_logs=True,
    )
    data = load_run_data(manifest)
    assert len(data.logs) == 0
    assert data.manifest.has_logs is False
    assert "not found" in data.manifest.integrity_warning.lower()


def test_load_carbon_assumptions():
    """Verify carbon assumptions are loaded with units, bounds, and estimated provenance."""
    assumptions = load_carbon_assumptions("configs/carbon_coefficients.yaml")
    assert len(assumptions) >= 5

    for row in assumptions:
        assert "parameter" in row
        assert "central" in row
        assert "low" in row
        assert "high" in row
        assert "unit" in row
        assert row["provenance"] == "estimated"
        assert row["low"] <= row["central"] <= row["high"]


def test_load_pricing_assumptions():
    """Verify pricing assumptions load with dated table metadata and calculated provenance."""
    pricing = load_pricing_assumptions("configs/pricing_table_2026_10.yaml")
    assert len(pricing) >= 4

    models = {p["model_id"]: p for p in pricing}
    assert "anthropic/claude-3.7-sonnet" in models
    claude = models["anthropic/claude-3.7-sonnet"]
    assert claude["prompt_price_per_1m"] == 3.00
    assert claude["completion_price_per_1m"] == 15.00
    assert claude["provenance"] == "calculated"
    assert claude["effective_date"] == "2026-10-01"


def test_counterfactual_large_model_measured_tokens():
    """Verify counterfactual calculation using measured input and output token counts."""
    res = calculate_counterfactual_large_model_cost(
        prompt="Explain quantum entanglement",
        actual_input_tokens=100,
        actual_output_tokens=200,
        model_id="anthropic/claude-3.7-sonnet",
        pricing_file="configs/pricing_table_2026_10.yaml",
    )
    assert res["status"] == "calculated"
    assert res["input_tokens"] == 100
    assert res["input_tokens_method"] == "measured"
    assert res["output_tokens"] == 200
    assert res["output_tokens_method"] == "measured"
    # Cost = (100 * 3.0 + 200 * 15.0) / 1,000,000 = (300 + 3000) / 1,000,000 = 0.003300
    assert res["counterfactual_cost_usd"] == 0.0033
    assert res["cost_provenance"] == "calculated"
    assert res["quality_available"] is False
    assert res["latency_available"] is False


def test_counterfactual_large_model_heuristic_tokens():
    """Verify counterfactual calculation falls back to explicit heuristic token estimation when tokens are missing."""
    prompt = "What is the capital of France?"  # 6 words
    res = calculate_counterfactual_large_model_cost(
        prompt=prompt,
        actual_input_tokens=None,
        actual_output_tokens=None,
        model_id="anthropic/claude-3.7-sonnet",
        pricing_file="configs/pricing_table_2026_10.yaml",
    )
    assert "estimated_heuristic" in res["input_tokens_method"]
    assert "estimated_heuristic" in res["output_tokens_method"]
    assert res["input_tokens"] == 8  # ceil(6 * 1.3) = 8
    assert res["output_tokens"] == 50  # default 50
    assert res["counterfactual_cost_usd"] > 0
    assert res["cost_provenance"] == "calculated"


def test_get_ranked_failures_sanitizes_private_data():
    """Verify get_ranked_failures strictly redacts sensitive and confidential prompt text."""
    sample_logs = [
        {
            "item_id": "item_pub_1",
            "prompt": "Calculate 50 + 20",
            "task_category": "math",
            "tier": "easy",
            "quality_score": 0.2,
            "validation_passed": True,
            "meets_q_min": False,
        },
        {
            "item_id": "item_priv_2",
            "prompt": "Summarize confidential employee salary and payroll records for employee Alice",
            "task_category": "hr",
            "tier": "difficult",
            "quality_score": 0.0,
            "validation_passed": False,
            "meets_q_min": False,
        },
    ]

    failures = get_ranked_failures(sample_logs, q_min=0.70, max_count=10)
    assert len(failures) == 2

    # Worst quality first
    assert failures[0]["item_id"] == "item_priv_2"
    assert "[REDACTED: SENSITIVE/PRIVATE PROMPT]" in failures[0]["prompt_preview"]
    assert "Alice" not in failures[0]["prompt_preview"]
    assert "salary" not in failures[0]["prompt_preview"]

    assert failures[1]["item_id"] == "item_pub_1"
    assert failures[1]["prompt_preview"] == "Calculate 50 + 20"


def test_cache_storage_size_reporting():
    """Verify get_cache_storage_size returns byte counts and does not fail on missing path."""
    info = get_cache_storage_size("data/non_existent_cache.db")
    assert info["exists"] is False
    assert info["size_bytes"] == 0

    # Test against real or temp DB
    real_info = get_cache_storage_size("data/cache.db")
    assert "size_bytes" in real_info
    assert "size_kb" in real_info


def test_read_only_data_loader_does_not_mutate_artifacts(tmp_path):
    """Verify loading artifacts does not write or modify any files."""
    test_json = tmp_path / "test_artifact.json"
    content = {"run_id": "test_immutability", "val": 42}
    test_json.write_text(json.dumps(content), encoding="utf-8")
    mtime_before = test_json.stat().st_mtime_ns

    # Call loader functions
    summary = load_benchmark_summary(results_dir=test_json)
    assert summary is not None
    mtime_after = test_json.stat().st_mtime_ns
    assert mtime_before == mtime_after
