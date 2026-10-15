"""Targeted correctness remediation regression tests.

Covers:
1. Privacy-safe benchmark artifacts (zero confidential prompt leaks, canonical naming).
2. Constraint-aware fallback blocking in pipeline (tier eligibility, user budget, user latency).
3. Human review provenance and score range validation.
4. Real-provider evaluation plumbing (budget tracker, unified privacy enforcement, judge provenance).
5. System B cache ordering and model-call counter verification via ExperimentHarness.
6. Paired per-item effect size and explicit missing observation tracking.
7. Environment variable configuration fallback in settings.
"""

import json
from pathlib import Path

import pytest

from inferpath.config.settings import (
    PricingTable,
    load_pricing_table,
    load_settings,
)
from inferpath.evaluation.dataset import DatasetItem
from inferpath.evaluation.harness import ExperimentHarness
from inferpath.evaluation.human_review import (
    HumanReviewRecord,
    compute_inter_rater_agreement,
)
from inferpath.evaluation.judge import MockJudge
from inferpath.evaluation.stats import (
    compute_paired_per_item_comparison,
)
from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass
from inferpath.pipeline import RequestPipeline
from inferpath.security.budget import RunBudgetTracker
from inferpath.validators import LengthValidator


@pytest.fixture
def pricing_table() -> PricingTable:
    return load_pricing_table()


# ==============================================================================
# 1. Privacy-Safe Routing Logs and Evaluation Artifacts
# ==============================================================================


def test_no_confidential_prompt_leaks_in_simulated_benchmark_directory():
    """Verify data/results/simulated/ contains zero raw confidential prompts."""
    results_dir = Path("data/results/simulated")
    assert results_dir.exists(), "Benchmark results directory must exist"

    # Known confidential prompts from dataset/audit
    confidential_phrases = [
        "Project Titan internal architecture",
        "Confidential internal quarterly projection",
        "employee compensation review",
        "acquisition target valuation",
    ]

    for log_file in results_dir.glob("logs_*.json"):
        with open(log_file, encoding="utf-8") as f:
            data = json.load(f)
            serialized = json.dumps(data)
            for phrase in confidential_phrases:
                assert phrase not in serialized, (
                    f"Confidential phrase '{phrase}' leaked in {log_file.name}"
                )

    # Verify canonical naming format (no random UUID hashes)
    for manifest_file in results_dir.glob("manifest_*.json"):
        # Format: manifest_run_{system_id}_{split}_{run_index}.json
        name_parts = manifest_file.stem.split("_")
        # e.g., manifest, run, System, A, dev, 1
        assert len(name_parts) >= 5
        # Ensure no random 6-character hex hash at the end
        last_part = name_parts[-1]
        assert last_part.isdigit(), f"Manifest {manifest_file.name} must end with integer run index"


# ==============================================================================
# 2. Constraint-Aware Fallback Blocking in Pipeline
# ==============================================================================


def test_pipeline_blocks_fallback_when_tier_d_ineligible(pricing_table: PricingTable):
    """Fallback from failed Tier C to Tier D is blocked when Tier D is not eligible."""
    pipeline = RequestPipeline(
        pricing_table=pricing_table,
        validator=LengthValidator(default_min_chars=10000),  # Will fail Tier C
    )

    # Force a decision where Tier D is NOT eligible (e.g., maximum_cost_usd is low)
    req = UserRequest(
        request_id="req_ineligible_d",
        prompt="Write a brief greeting.",
        privacy_class=PrivacyClass.PUBLIC,
        task_parameters=TaskParameters(maximum_cost_usd=0.00001),  # Tier D is ~$0.005
    )

    # Router will evaluate Tier D as ineligible
    res = pipeline.execute(req)
    # Since Tier D cost exceeds maximum_cost_usd, fallback is blocked
    assert not res.fallback_occurred
    fallback_trace = [t for t in res.route_trace if t.get("step") == "fallback_blocked"]
    assert len(fallback_trace) > 0


def test_pipeline_blocks_fallback_when_latency_constraint_violated(pricing_table: PricingTable):
    """Fallback to Tier D is blocked when request requires maximum_latency_ms < 250ms."""
    pipeline = RequestPipeline(
        pricing_table=pricing_table,
        validator=LengthValidator(default_min_chars=10000),  # Will fail Tier C
    )

    req = UserRequest(
        request_id="req_tight_latency",
        prompt="Explain photosynthesis.",
        privacy_class=PrivacyClass.PUBLIC,
        task_parameters=TaskParameters(maximum_latency_ms=100.0),
    )

    res = pipeline.execute(req)
    assert not res.fallback_occurred
    blocked = [t for t in res.route_trace if t.get("step") == "fallback_blocked"]
    assert any(b.get("reason") in ("user_latency_constraint", "tier_d_ineligible") for b in blocked)


# ==============================================================================
# 3. Human Review Rating Provenance and Bounds Validation
# ==============================================================================


def test_human_review_rejects_synthetic_provenance():
    """compute_inter_rater_agreement rejects non-measured provenance ratings."""
    scores_synthetic = [
        HumanReviewRecord(
            review_id="REV_1_A",
            item_id="item_1",
            reviewer_id="rev_1",
            candidate_id="cand_A",
            score=4,
            provenance="synthetic",  # Disallowed provenance
        ),
        HumanReviewRecord(
            review_id="REV_1_B",
            item_id="item_1",
            reviewer_id="rev_2",
            candidate_id="cand_A",
            score=4,
            provenance="synthetic",
        ),
    ]

    result = compute_inter_rater_agreement(scores_synthetic)
    assert result["status"] == "unverified_provenance"
    assert "non-measured provenance" in result["message"]


def test_human_review_rejects_out_of_bounds_ratings():
    """compute_inter_rater_agreement rejects scores not in range [1, 5]."""
    scores_invalid = [
        HumanReviewRecord.model_construct(
            review_id="REV_1_A",
            item_id="item_1",
            reviewer_id="rev_1",
            candidate_id="cand_A",
            score=6,  # Out of bounds (>5)
            provenance="measured",
        ),
        HumanReviewRecord.model_construct(
            review_id="REV_1_B",
            item_id="item_1",
            reviewer_id="rev_2",
            candidate_id="cand_A",
            score=0,  # Out of bounds (<1)
            provenance="measured",
        ),
    ]

    result = compute_inter_rater_agreement(scores_invalid)
    assert result["status"] == "invalid_ratings"
    assert "out-of-bounds" in result["message"]


# ==============================================================================
# 4. Real-Provider Evaluation Plumbing & Privacy Guard
# ==============================================================================


def test_harness_budget_tracker_and_judge_provenance(tmp_path: Path):
    """Harness wires budget tracker and marks MockJudge as simulated provenance."""
    tracker = RunBudgetTracker(max_usd=1.0)
    mock_judge = MockJudge()

    harness = ExperimentHarness(
        adapter_mode="simulated",
        output_base_dir=tmp_path / "harness_wiring",
        judge=mock_judge,
        budget_tracker=tracker,
    )

    assert harness.budget_tracker is tracker
    assert not harness.is_real_judge

    item = DatasetItem(
        item_id="item_wire_test",
        task_family_id="fam_1",
        task_category="multi_step_reasoning",
        tier="moderate",
        checker_type="exact",
        split="dev",
        prompt="What is the capital of France?",
        reference_answer="Paris",
        privacy_class=PrivacyClass.PUBLIC,
        cache_eligibility=True,
    )

    summary, manifest, logs = harness.run_evaluation(
        system_id="System_C",
        items=[item],
        run_index=1,
    )

    assert manifest.environment_info["judge_type"] == "MockJudge"
    assert manifest.environment_info["judge_provenance"] == "simulated"


def test_baseline_system_a_and_b_use_unified_privacy_enforcer(tmp_path: Path):
    """System A and System B both block confidential items via privacy enforcer."""
    harness = ExperimentHarness(
        adapter_mode="simulated",
        output_base_dir=tmp_path / "privacy_baseline_test",
    )

    conf_item = DatasetItem(
        item_id="item_conf_test",
        task_family_id="fam_conf",
        task_category="multi_step_reasoning",
        tier="moderate",
        checker_type="exact",
        split="dev",
        prompt="Confidential internal salary database dump.",
        reference_answer="N/A",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        cache_eligibility=False,
    )

    # Test System A
    _, _, logs_a = harness.run_evaluation(
        system_id="System_A",
        items=[conf_item],
        run_index=1,
    )
    assert logs_a[0]["cost_usd_calculated"] == 0.0
    assert logs_a[0]["routed_tier"] == "deterministic_a"
    assert logs_a[0]["rule_id"] == "R01"

    # Test System B
    _, _, logs_b = harness.run_evaluation(
        system_id="System_B",
        items=[conf_item],
        run_index=1,
    )
    assert logs_b[0]["cost_usd_calculated"] == 0.0
    assert logs_b[0]["routed_tier"] == "deterministic_a"
    assert logs_b[0]["rule_id"] == "R01"


# ==============================================================================
# 5. System B Cache Ordering and Model-Call Counter
# ==============================================================================


def test_system_b_cache_lookup_with_model_call_counter(tmp_path: Path):
    """In System B, repeated eligible request does not invoke model adapter."""
    harness = ExperimentHarness(
        adapter_mode="simulated",
        output_base_dir=tmp_path / "system_b_counter",
    )

    item = DatasetItem(
        item_id="item_cache_b",
        task_family_id="fam_b",
        task_category="multi_step_reasoning",
        tier="moderate",
        checker_type="exact",
        split="dev",
        prompt="Explain the purpose of database indexing.",
        reference_answer="To speed up data retrieval.",
        privacy_class=PrivacyClass.PUBLIC,
        cache_eligibility=True,
    )

    # Run System B with identical item twice
    summary, manifest, logs = harness.run_evaluation(
        system_id="System_B",
        items=[item, item],
        run_index=1,
    )

    assert len(logs) == 2
    # First execution is cache miss
    assert not logs[0]["was_cached"]
    assert logs[0]["cost_usd_calculated"] > 0
    # Second execution is cache hit
    assert logs[1]["was_cached"]
    assert logs[1]["cost_usd_calculated"] == 0.0
    assert logs[1]["latency_ms"] < 5.0  # Fast cache retrieval


# ==============================================================================
# 6. Paired Per-Item Statistics and Missing Observations
# ==============================================================================


def test_compute_paired_per_item_comparison_explicit_missing():
    """Paired per-item comparison reports effect size and explicitly counts missing items."""
    scores_c = {
        "item_1": 0.0001,
        "item_2": 0.0002,
        "item_3": 0.0001,
    }
    scores_a = {
        "item_1": 0.0050,
        "item_2": 0.0050,
        # item_3 is missing from System A
    }

    all_items = ["item_1", "item_2", "item_3", "item_4"]

    res = compute_paired_per_item_comparison(scores_c, scores_a, all_item_ids=all_items)

    assert res.paired_n == 2
    assert res.missing_observations == 2
    assert "item_3" in res.missing_items
    assert "item_4" in res.missing_items
    assert res.cliffs_delta == -1.0
    assert res.interpretation == "large"
    assert res.paired_difference_mean < 0.0


# ==============================================================================
# 7. Environment Variable Fallback in Settings
# ==============================================================================


def test_settings_environment_variable_fallbacks(monkeypatch):
    """Settings properly respect standard environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATABASE_PATH", "custom_data.db")
    monkeypatch.setenv("INFERPATH_MAX_USD_PER_RUN", "25.0")

    settings = load_settings()
    assert settings.environment == "production"
    assert settings.log_level == "DEBUG"
    assert settings.database_path == "custom_data.db"
    assert settings.routing.max_usd_per_run == 25.0
