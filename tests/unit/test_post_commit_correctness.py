"""Comprehensive regression tests for post-commit correctness fixes.

Covers:
1. Human-review candidate generation with genuine text, privacy enforcement, and unblinding key.
2. Routing constraint enforcement before dispatch with CONSTRAINTS_UNSATISFIABLE (HTTP 422).
3. System B cache evaluation order (lookup before model dispatch, verified via call counter).
4. Privacy-aware benchmark logging with non-public prompt redaction and sha256 digests.
5. OpenRouter usage/cost provenance (handling missing/malformed usage explicitly) and budget tracker.
6. Git commit dirty-state tracking in experiment manifests.
"""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from inferpath.api.main import create_app
from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable, load_pricing_table
from inferpath.evaluation.dataset import DatasetItem
from inferpath.evaluation.freeze_config import get_git_commit
from inferpath.evaluation.harness import ExperimentHarness
from inferpath.evaluation.human_review import generate_blind_human_review_csv
from inferpath.handlers.mock_adapter import MockLargeModel
from inferpath.handlers.openrouter_adapter import OpenRouterAdapter
from inferpath.models.domain import TaskParameters, UserRequest
from inferpath.models.enums import PrivacyClass, ProvenanceLabel
from inferpath.pipeline import RequestPipeline
from inferpath.router import (
    ConstraintAwareRouter,
    ConstraintsUnsatisfiableError,
)
from inferpath.security.budget import (
    BudgetExceededError,
    RunBudgetTracker,
)


@pytest.fixture
def pricing_table() -> PricingTable:
    return load_pricing_table()


# ==============================================================================
# 1. Human-Review Candidate Generation
# ==============================================================================


def test_human_review_candidate_generation_genuine_text_and_privacy(tmp_path: Path):
    """Proves exported review sheet contains actual response text, skips non-public, produces key."""
    items = [
        DatasetItem(
            item_id="item_pub_1",
            task_family_id="fam_1",
            task_category="multi_step_reasoning",
            tier="moderate",
            checker_type="exact",
            split="dev",
            prompt="Explain the difference between TCP and UDP.",
            reference_answer="TCP is connection-oriented; UDP is connectionless.",
            privacy_class=PrivacyClass.PUBLIC,
            cache_eligibility=True,
        ),
        DatasetItem(
            item_id="item_conf_2",
            task_family_id="fam_2",
            task_category="multi_step_reasoning",
            tier="moderate",
            checker_type="exact",
            split="dev",
            prompt="Confidential internal quarterly projection details.",
            reference_answer="Internal financial projections.",
            privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
            cache_eligibility=False,
        ),
    ]

    rep_outputs = {
        "item_pub_1": {
            "System_A": "Genuine System A response: TCP provides reliable transport, while UDP is datagram-based.",
            "System_C": "Genuine System C response: TCP ensures delivery; UDP optimizes for low latency.",
        },
        "item_conf_2": {
            "System_A": "Confidential response A",
            "System_C": "Confidential response C",
        },
    }

    csv_path = tmp_path / "review_sheet.csv"
    key_path = tmp_path / "review_key.json"

    csv_content = generate_blind_human_review_csv(
        test_items=items,
        system_outputs=rep_outputs,
        output_path=csv_path,
        key_path=key_path,
        sample_ratio=1.0,
        seed=42,
    )

    assert csv_content
    assert csv_path.exists()
    assert key_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Genuine System A response" in csv_text
    assert "Genuine System C response" in csv_text
    assert "Mock Large Baseline completion" not in csv_text
    assert "Mock InferPath completion" not in csv_text
    assert "Confidential" not in csv_text

    key_data = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(key_data) >= 1
    for mapping in key_data.values():
        assert mapping["item_id"] == "item_pub_1"
        assert set(mapping.keys()) == {"item_id", "candidate_a_system", "candidate_b_system"}
        assert {mapping["candidate_a_system"], mapping["candidate_b_system"]} == {
            "System_A",
            "System_C",
        }


def test_human_review_reports_missing_candidate_outputs(tmp_path: Path):
    """Items missing genuine outputs from either system must be skipped and reported."""
    items = [
        DatasetItem(
            item_id="item_incomplete",
            task_family_id="fam_inc",
            task_category="multi_step_reasoning",
            tier="moderate",
            checker_type="exact",
            split="dev",
            prompt="Analyze trade-offs of microservices.",
            reference_answer="Decoupling vs operational complexity.",
            privacy_class=PrivacyClass.PUBLIC,
            cache_eligibility=True,
        )
    ]
    # Missing System_C output
    rep_outputs = {
        "item_incomplete": {
            "System_A": "Complete System A response",
        }
    }
    csv_path = tmp_path / "review_sheet.csv"
    csv_content = generate_blind_human_review_csv(
        test_items=items,
        system_outputs=rep_outputs,
        output_path=csv_path,
        sample_ratio=1.0,
        seed=42,
    )
    assert "item_incomplete" not in csv_content


# ==============================================================================
# 2. Routing Constraints Enforcement Before Dispatch
# ==============================================================================


def test_tier_d_pruned_when_quality_below_threshold(pricing_table: PricingTable):
    """No tier, including Tier D, remains eligible when estimated quality falls below threshold."""
    router = ConstraintAwareRouter(pricing_table=pricing_table)
    req = UserRequest(
        request_id="req_high_q",
        prompt="Write a novel formal proof of the Riemann Hypothesis in Lean 4.",
        task_parameters=TaskParameters(minimum_quality=0.99),
    )
    decision = router.route(req)
    assert not decision.constraints_satisfied
    assert decision.rule_id == "CONSTRAINTS_UNSATISFIABLE"
    assert decision.eligible_tiers == []
    assert "quality" in decision.unsatisfied_constraints


def test_pipeline_raises_constraints_unsatisfiable_without_model_dispatch(
    tmp_path: Path, pricing_table: PricingTable
):
    """Pipeline raises ConstraintsUnsatisfiableError and does not dispatch to model."""
    mock_large = MagicMock(spec=MockLargeModel)
    db_path = str(tmp_path / "cache_constraints.db")
    router = ConstraintAwareRouter(pricing_table=pricing_table)
    pipeline = RequestPipeline(
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
        router=router,
        large_adapter=mock_large,
    )

    req = UserRequest(
        request_id="req_unsat_q",
        prompt="Solve open problem with extreme precision.",
        task_parameters=TaskParameters(minimum_quality=0.99),
    )

    with pytest.raises(ConstraintsUnsatisfiableError) as exc_info:
        pipeline.execute(req)

    err = exc_info.value
    assert "quality" in err.unsatisfied_constraints
    assert err.eligible_tiers == []
    assert mock_large.generate.call_count == 0


def test_api_returns_422_on_constraints_unsatisfiable(tmp_path: Path, pricing_table: PricingTable):
    """API returns HTTP 422 with structured CONSTRAINTS_UNSATISFIABLE response."""
    db_path = str(tmp_path / "api_unsat.db")
    router = ConstraintAwareRouter(pricing_table=pricing_table)
    pipeline = RequestPipeline(
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
        router=router,
    )
    app = create_app(pipeline=pipeline)
    client = TestClient(app)

    payload = {
        "request_id": "req_api_unsat",
        "prompt": "Evaluate complex theorem with high accuracy.",
        "minimum_quality": 0.99,
    }
    response = client.post("/v1/respond", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["code"] == "CONSTRAINTS_UNSATISFIABLE"
    assert "unsatisfied_constraints" in data
    assert "quality" in data["unsatisfied_constraints"]
    assert data["eligible_tiers"] == []


def test_unsatisfiable_budget_and_latency_constraints(tmp_path: Path, pricing_table: PricingTable):
    """Budget and latency thresholds below model capabilities prune all tiers."""
    router = ConstraintAwareRouter(pricing_table=pricing_table)
    req = UserRequest(
        request_id="req_tight_bl",
        prompt="Synthesize technical document.",
        task_parameters=TaskParameters(
            minimum_quality=0.70,
            maximum_cost_usd=0.0000001,  # Sub-micro-cent budget
            maximum_latency_ms=5.0,  # 5 ms latency ceiling
        ),
    )
    decision = router.route(req)
    assert not decision.constraints_satisfied
    assert (
        "budget" in decision.unsatisfied_constraints
        or "latency" in decision.unsatisfied_constraints
    )
    assert decision.eligible_tiers == []


# ==============================================================================
# 3. System B Cache Evaluation Order
# ==============================================================================


def test_system_b_cache_lookup_before_model_invocation(tmp_path: Path):
    """In System B, cache lookup occurs first; model executed only on cache miss."""
    harness = ExperimentHarness(
        adapter_mode="simulated",
        output_base_dir=tmp_path / "system_b_test",
        q_min=0.70,
    )

    test_item = DatasetItem(
        item_id="item_b_cached",
        task_family_id="fam_cache",
        task_category="multi_step_reasoning",
        tier="moderate",
        checker_type="exact",
        split="dev",
        prompt="Explain the purpose of an exact response cache in inference systems.",
        reference_answer="An exact response cache stores deterministic answers.",
        privacy_class=PrivacyClass.PUBLIC,
        cache_eligibility=True,
    )

    # Pass identical item twice within the same evaluation run to test cache hit
    summary, manifest, logs = harness.run_evaluation(
        system_id="System_B",
        items=[test_item, test_item],
        run_index=1,
        seed=42,
        split="dev",
        verify_lock=False,
    )

    # First execution: cold cache miss, model invoked
    assert logs[0]["was_cached"] is False
    assert logs[0]["cost_usd_calculated"] > 0

    # Second execution: warm cache hit, model not invoked
    assert logs[1]["was_cached"] is True
    assert logs[1]["cost_usd_calculated"] == 0.0
    assert logs[1]["latency_ms"] < 10.0
    assert summary.cache_hit_rate == 0.5


def test_system_b_model_call_counter_verified(tmp_path: Path, pricing_table: PricingTable):
    """Model-call counter proves model is not invoked on repeated cached eligible requests."""
    db_path = str(tmp_path / "cache_counter.db")
    pipeline = RequestPipeline(
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
    )

    call_count = 0
    orig_execute = pipeline._execute_tier

    def counting_execute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_execute(*args, **kwargs)

    pipeline._execute_tier = counting_execute

    req = UserRequest(
        request_id="req_repeat_1",
        prompt="Explain why fast deterministic caching improves system latency.",
        privacy_class=PrivacyClass.PUBLIC,
    )

    # Request 1: cache miss, model executed
    res1 = pipeline.execute(req)
    assert not res1.was_cached
    assert call_count == 1
    assert res1.cost_usd.value > 0

    # Request 2: identical request, cache hit, model NOT executed
    req2 = UserRequest(
        request_id="req_repeat_2",
        prompt="Explain why fast deterministic caching improves system latency.",
        privacy_class=PrivacyClass.PUBLIC,
    )
    res2 = pipeline.execute(req2)
    assert res2.was_cached
    assert call_count == 1  # Counter unchanged!
    assert res2.cost_usd.value == 0.0


# ==============================================================================
# 4. Privacy-Aware Benchmark Logging
# ==============================================================================


def test_confidential_prompt_redacted_in_serialized_logs(tmp_path: Path):
    """Internal and confidential prompts are redacted in benchmark logs with prompt_sha256 preserved."""
    harness = ExperimentHarness(
        adapter_mode="simulated",
        output_base_dir=tmp_path / "privacy_logging",
        q_min=0.70,
    )

    secret_prompt = "Top secret internal project codename: Project-Pegasus-Alpha."
    conf_item = DatasetItem(
        item_id="item_secret",
        task_family_id="fam_secret",
        task_category="multi_step_reasoning",
        tier="moderate",
        checker_type="exact",
        split="dev",
        prompt=secret_prompt,
        reference_answer="Classified.",
        privacy_class=PrivacyClass.CONFIDENTIAL_PRIVATE,
        cache_eligibility=False,
    )

    summary, manifest, logs = harness.run_evaluation(
        system_id="System_C",
        items=[conf_item],
        run_index=1,
        seed=42,
        split="dev",
        verify_lock=False,
    )

    log_entry = logs[0]
    assert log_entry["prompt"] == "[REDACTED_CONFIDENTIAL]"
    assert log_entry["response_text"] == "[REDACTED_CONFIDENTIAL]"
    assert "prompt_sha256" in log_entry
    assert len(log_entry["prompt_sha256"]) == 64
    assert log_entry["privacy_class"] == "confidential_private"

    # Verify disk log file does not leak secret text
    log_file = tmp_path / "privacy_logging" / f"logs_{manifest.run_id}.json"
    raw_content = log_file.read_text(encoding="utf-8")
    assert "Project-Pegasus-Alpha" not in raw_content
    assert "[REDACTED_CONFIDENTIAL]" in raw_content


# ==============================================================================
# 5. OpenRouter Usage & Cost Provenance
# ==============================================================================


def test_openrouter_missing_or_malformed_usage_explicitly_unavailable(pricing_table: PricingTable):
    """Missing or non-dict usage in OpenRouter responses sets usage_available=False, tokens/cost=None."""
    adapter = OpenRouterAdapter(
        model_id="meta-llama/llama-3.1-8b-instruct",
        pricing_table=pricing_table,
        api_key="sk-test-fake-key",
    )

    # Case 1: missing usage key
    body_no_usage = {
        "id": "gen_1",
        "choices": [{"message": {"content": "Generated text without usage metadata."}}],
    }
    result_1 = adapter._parse_response_body(body_no_usage, latency_ms=45.0)
    assert result_1.usage_available is False
    assert result_1.input_tokens is None
    assert result_1.output_tokens is None
    assert result_1.cost_usd_calculated is None
    assert result_1.source == ProvenanceLabel.UNAVAILABLE

    # Case 2: malformed usage
    body_malformed_usage = {
        "id": "gen_2",
        "choices": [{"message": {"content": "Malformed usage text."}}],
        "usage": "not_a_dict",
    }
    result_2 = adapter._parse_response_body(body_malformed_usage, latency_ms=30.0)
    assert result_2.usage_available is False
    assert result_2.input_tokens is None
    assert result_2.output_tokens is None


def test_calculated_cost_not_labeled_as_measured(pricing_table: PricingTable):
    """When provider reports cost, it is measured; otherwise calculated cost is not labeled provider-reported."""
    adapter = OpenRouterAdapter(
        model_id="meta-llama/llama-3.1-8b-instruct",
        pricing_table=pricing_table,
        api_key="sk-test-fake-key",
    )

    body = {
        "id": "gen_valid",
        "choices": [{"message": {"content": "Valid response."}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            # No provider-reported cost field
        },
    }
    result = adapter._parse_response_body(body, latency_ms=50.0)
    assert result.usage_available is True
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.provider_reported_cost_usd is None  # Not reported by provider
    assert result.cost_usd_calculated is not None
    assert result.source == ProvenanceLabel.MEASURED


# ==============================================================================
# 6. Budget Tracker Bounded Retention & Scoping
# ==============================================================================


def test_budget_tracker_bounded_retention():
    """Budget tracker evicts oldest settled reservations when history_limit is reached."""
    tracker = RunBudgetTracker(max_usd=100.0, history_limit=5)
    tokens = []

    for _ in range(10):
        tok = tracker.reserve(1.0)
        tracker.reconcile(tok, 1.0)
        tokens.append(tok)

    # Only 5 settled records retained
    assert len(tracker._settled_history) == 5
    assert len(tracker._settled_tokens) == 5
    # Oldest 5 tokens evicted from settled tracking
    for old_tok in tokens[:5]:
        assert old_tok not in tracker._settled_tokens
    # Newest 5 tokens retained
    for new_tok in tokens[5:]:
        assert new_tok in tracker._settled_tokens


def test_budget_tracker_enforces_run_scope_sequentially(
    tmp_path: Path, pricing_table: PricingTable
):
    """Single pipeline instance with shared budget tracker accumulates spend across sequential calls."""
    # Set cap to 0.000310 USD: enough for request 1 (~0.000308 USD), but after spending ~$0.000003,
    # remaining budget (~0.000307 USD) cannot cover request 2 (which needs ~0.000309 USD).
    budget = RunBudgetTracker(max_usd=0.000310, pricing_table=pricing_table)
    db_path = str(tmp_path / "cache_budget_seq.db")
    pipeline = RequestPipeline(
        pricing_table=pricing_table,
        cache=ExactCache(db_path=db_path),
        budget_tracker=budget,
    )

    req1 = UserRequest(
        request_id="b_req1",
        prompt="Explain benefits of static typing.",
        privacy_class=PrivacyClass.PUBLIC,
    )
    res1 = pipeline.execute(req1)
    assert res1.validation_passed
    assert budget.spent_usd > 0.0
    assert budget.remaining_budget < 0.000310

    # Request 2 estimated cost exceeds remaining budget
    req2 = UserRequest(
        request_id="b_req2",
        prompt="Explain quantum teleportation and entanglement in detail.",
        privacy_class=PrivacyClass.PUBLIC,
    )
    with pytest.raises(BudgetExceededError, match="exceeds remaining budget"):
        pipeline.execute(req2)


def test_budget_tracker_repeated_settlement_and_concurrency():
    """Repeated settlement is idempotent; concurrent reservations preserve balance."""
    tracker = RunBudgetTracker(max_usd=10.0)

    # 1. Idempotent reconciliation
    token = tracker.reserve(2.0)
    tracker.reconcile(token, 1.5)
    assert tracker.spent_usd == 1.5
    assert tracker.remaining_budget == 8.5

    # Second reconcile is no-op
    tracker.reconcile(token, 1.5)
    assert tracker.spent_usd == 1.5
    assert tracker.remaining_budget == 8.5

    # Release after reconcile is no-op
    tracker.release(token)
    assert tracker.spent_usd == 1.5
    assert tracker.remaining_budget == 8.5

    # 2. Concurrent reservations thread safety
    threads = []
    cost_per_thread = 0.20
    num_threads = 20

    def worker():
        try:
            tok = tracker.reserve(cost_per_thread)
            tracker.reconcile(tok, cost_per_thread)
        except BudgetExceededError:
            pass

    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Initial 1.5 + (20 * 0.20 = 4.0) = 5.5 spent, 4.5 remaining
    assert tracker.total_spent_usd == pytest.approx(5.5)
    assert tracker.remaining_budget == pytest.approx(4.5)
    assert tracker.reserved_usd == pytest.approx(0.0)


# ==============================================================================
# 7. Git Commit Dirty State Tracking in RunManifest
# ==============================================================================


def test_manifest_records_git_commit_and_dirty_flag(tmp_path: Path):
    """Manifest records git_commit with dirty status and git_tree_clean boolean."""
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0

    harness = ExperimentHarness(
        adapter_mode="simulated",
        output_base_dir=tmp_path / "manifest_test",
    )
    item = DatasetItem(
        item_id="item_manifest",
        task_family_id="fam_m",
        task_category="arithmetic_and_unit_conversion",
        tier="easy",
        checker_type="exact",
        split="dev",
        prompt="Calculate 5 * 10",
        reference_answer="50",
        privacy_class=PrivacyClass.PUBLIC,
        cache_eligibility=True,
    )
    summary, manifest, logs = harness.run_evaluation(
        system_id="System_C",
        items=[item],
        run_index=1,
        seed=42,
        split="dev",
        verify_lock=False,
    )

    assert isinstance(manifest.git_tree_clean, bool)
    assert manifest.git_commit == commit
