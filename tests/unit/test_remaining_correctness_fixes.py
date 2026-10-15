"""Regression tests for the four remaining correctness fixes:
1. Prompt privacy: sensitive-pattern checks on router decision logger and human-review exporter.
2. Judge calls: budget cap accounting, reservation, and total evaluation cost reflection.
3. Human-rating validation: rejection of fractional ratings and defaulting missing provenance to unknown.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferpath.evaluation.dataset import DatasetItem
from inferpath.evaluation.human_review import (
    compute_inter_rater_agreement,
    generate_blind_human_review_csv,
    load_human_review_csv,
)
from inferpath.evaluation.judge import OpenRouterJudge
from inferpath.models.domain import UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.router.analyzer import contains_sensitive_patterns
from inferpath.router.router import RuleBasedRouter
from inferpath.router.strategies import BaseRouter, RoutingDecision
from inferpath.security.budget import BudgetExceededError, RunBudgetTracker

# ==============================================================================
# 1. Close remaining prompt-privacy paths
# ==============================================================================


def test_contains_sensitive_patterns():
    """contains_sensitive_patterns accurately detects credentials and sensitive keywords."""
    assert contains_sensitive_patterns("My api_key is secret123") is True
    assert contains_sensitive_patterns("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9") is True
    assert contains_sensitive_patterns("Social security number: 000-12-3456") is True
    assert contains_sensitive_patterns("Employee salary and payroll data") is True
    assert contains_sensitive_patterns("What is the capital of France?") is False
    assert contains_sensitive_patterns("") is False
    assert contains_sensitive_patterns(None) is False


def test_router_decision_logger_redacts_sensitive_public_prompt():
    """Router decision logger must NOT record prompt_preview if sensitive pattern is present."""
    router = RuleBasedRouter()
    logger_mock = MagicMock()
    router.logger = logger_mock

    # Public request containing sensitive pattern
    req = UserRequest(
        request_id="req_test_1",
        prompt="Please reset password for user admin with token bearer abc123def456",
        privacy_class=PrivacyClass.PUBLIC,
    )
    router.route(req)

    assert logger_mock.log_event.called
    metadata = logger_mock.log_event.call_args[0][1]
    assert "prompt_preview" not in metadata, (
        "Sensitive prompt leaked into decision logger metadata!"
    )


def test_base_router_decision_logger_redacts_sensitive_public_prompt():
    """BaseRouter._log_decision must NOT record prompt_preview if sensitive pattern is present."""

    class DummyRouter(BaseRouter):
        def route(self, request: UserRequest) -> RoutingDecision:
            return RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                rule_id="R_DUMMY",
                reason="Dummy test",
                confidence=1.0,
            )

    router = DummyRouter()
    logger_mock = MagicMock()
    router.logger = logger_mock

    req = UserRequest(
        request_id="req_test_2",
        prompt="confidential internal payroll data for Q3",
        privacy_class=PrivacyClass.PUBLIC,
    )
    decision = router.route(req)
    router._log_decision_safely(req, decision)

    assert logger_mock.log_event.called
    metadata = logger_mock.log_event.call_args[0][1]
    assert "prompt_preview" not in metadata, "Sensitive prompt leaked into BaseRouter metadata!"


def test_human_review_exporter_excludes_sensitive_pattern_items(tmp_path: Path):
    """Human review CSV exporter must exclude public-classified items containing sensitive patterns."""
    items = [
        DatasetItem(
            item_id="item_safe",
            prompt="What is photosynthesis?",
            ground_truth="Process of plants converting light to energy",
            task_family_id="tf_biology",
            task_category="short_explanations",
            tier="easy",
            checker_type="exact",
            split="dev",
            privacy_class=PrivacyClass.PUBLIC,
            quality_criteria=["Mentions sunlight"],
        ),
        DatasetItem(
            item_id="item_sensitive",
            prompt="My password is secret123 and api_key is sk-12345678901234567890",
            ground_truth="N/A",
            task_family_id="tf_sec",
            task_category="short_explanations",
            tier="easy",
            checker_type="exact",
            split="dev",
            privacy_class=PrivacyClass.PUBLIC,  # Misclassified as public
            quality_criteria=["None"],
        ),
    ]

    system_outputs = {
        "item_safe": {"System_A": "Output A", "System_C": "Output C"},
        "item_sensitive": {"System_A": "Secret A", "System_C": "Secret C"},
    }

    out_csv = tmp_path / "review.csv"
    key_json = tmp_path / "key.json"

    generate_blind_human_review_csv(
        test_items=items,
        system_outputs=system_outputs,
        output_path=out_csv,
        key_path=key_json,
        sample_ratio=1.0,
    )

    content = out_csv.read_text(encoding="utf-8")
    assert "item_safe" in content
    assert "item_sensitive" not in content, "Sensitive prompt was exported to human review CSV!"
    assert "secret123" not in content


# ==============================================================================
# 2. Include judge calls in the live budget cap
# ==============================================================================


def test_openrouter_judge_enforces_budget_cap():
    """OpenRouterJudge aborts with BudgetExceededError when budget is insufficient."""
    tracker = RunBudgetTracker(max_usd=0.000001)  # Tiny budget
    judge = OpenRouterJudge(
        judge_model_id="openai/gpt-4o-mini",
        api_key="mock-test-key",
        budget_tracker=tracker,
    )

    item = DatasetItem(
        item_id="it_1",
        prompt="Explain quantum entanglement",
        ground_truth="Physics explanation",
        task_family_id="tf_phys",
        task_category="short_explanations",
        tier="difficult",
        checker_type="exact",
        split="dev",
        privacy_class=PrivacyClass.PUBLIC,
        quality_criteria=["Clear explanation"],
    )

    with pytest.raises(BudgetExceededError):
        judge.evaluate(item, "Candidate response about quantum physics")


def test_openrouter_judge_reserves_and_reconciles_budget():
    """OpenRouterJudge reserves and reconciles budget on successful API call."""
    tracker = RunBudgetTracker(max_usd=1.0)
    judge = OpenRouterJudge(
        judge_model_id="openai/gpt-4o-mini",
        api_key="mock-test-key",
        budget_tracker=tracker,
    )

    item = DatasetItem(
        item_id="it_1",
        prompt="Test prompt",
        ground_truth="Test ground truth",
        task_family_id="tf_test",
        task_category="short_explanations",
        tier="difficult",
        checker_type="exact",
        split="dev",
        privacy_class=PrivacyClass.PUBLIC,
        quality_criteria=["Accurate"],
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.85, "rationale": "Well written"}'}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "cost": 0.000045},
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        res = judge.evaluate(item, "Sample candidate response")
        assert res.score == 0.85
        assert res.pass_threshold is True
        assert res.judge_cost_usd == 0.000045

    assert tracker.total_spent_usd > 0
    assert tracker.total_spent_usd == pytest.approx(0.000045, rel=1e-5)


# ==============================================================================
# 4. Tighten human-rating validation
# ==============================================================================


def test_load_human_review_csv_rejects_fractional_ratings(tmp_path: Path):
    """load_human_review_csv rejects fractional ratings (e.g. 3.7) instead of truncating to 3."""
    csv_file = tmp_path / "ratings_fractional.csv"
    csv_file.write_text(
        "review_id,item_id,reviewer_id,score_a_1_to_5,score_b_1_to_5,provenance\n"
        "REV_0001,item_1,rev_1,3.7,4.0,measured\n",
        encoding="utf-8",
    )

    records = load_human_review_csv(csv_file)
    # Candidate A had 3.7 -> rejected! Candidate B had 4.0 (integer) -> accepted!
    assert len(records) == 1
    assert records[0].candidate_id == "candidate_b"
    assert records[0].score == 4


def test_load_human_review_csv_defaults_missing_provenance_to_unknown(tmp_path: Path):
    """load_human_review_csv defaults missing or empty provenance to 'unknown', not 'measured'."""
    csv_file = tmp_path / "ratings_no_prov.csv"
    csv_file.write_text(
        "review_id,item_id,reviewer_id,score_a_1_to_5,score_b_1_to_5,provenance\n"
        "REV_0001,item_1,rev_1,4,5,\n",  # empty provenance
        encoding="utf-8",
    )

    records = load_human_review_csv(csv_file)
    assert len(records) == 2
    for r in records:
        assert r.provenance == "unknown", f"Expected 'unknown' provenance, got '{r.provenance}'"

    # Inter-rater agreement should flag unverified provenance
    agreement = compute_inter_rater_agreement(records)
    assert agreement["status"] == "unverified_provenance"
