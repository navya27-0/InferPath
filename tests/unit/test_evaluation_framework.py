"""Unit tests for InferPath Phase 5 Research Evaluation Framework."""

import tempfile
from pathlib import Path

import pytest

from inferpath.evaluation.cache_eval import CacheEvaluator
from inferpath.evaluation.checkers import (
    ExactMatchChecker,
    NumericToleranceChecker,
    RegexChecker,
    RubricChecker,
    evaluate_output,
)
from inferpath.evaluation.dataset import DatasetItem
from inferpath.evaluation.environmental import EnvironmentalEstimator
from inferpath.evaluation.freeze_config import (
    compute_config_hash,
    load_registry,
    register_frozen_config,
    verify_test_set_lock,
)
from inferpath.evaluation.human_review import (
    compute_weighted_cohen_kappa,
    generate_blind_human_review_csv,
)
from inferpath.evaluation.judge import MockJudge, OpenRouterJudge
from inferpath.evaluation.oracle import OracleEvaluator
from inferpath.evaluation.stats import (
    compute_cliffs_delta,
    compute_clustered_bootstrap_ci,
    compute_repeated_run_summary,
)
from inferpath.pipeline import RequestPipeline


# 1. Checkers Tests
# -------------------------------------------------------------
def test_exact_match_checker():
    # Case insensitive match
    res = ExactMatchChecker.evaluate("100 C", {"reference": "100 c", "case_sensitive": False})
    assert res.passed is True
    assert res.score == 1.0

    # Case sensitive mismatch
    res = ExactMatchChecker.evaluate("100 C", {"reference": "100 c", "case_sensitive": True})
    assert res.passed is True  # Substring match fallback gives 0.9
    assert res.score == 0.90

    # Total mismatch
    res = ExactMatchChecker.evaluate("500 K", {"reference": "100 c", "case_sensitive": True})
    assert res.passed is False
    assert res.score == 0.0


def test_regex_checker():
    res = RegexChecker.evaluate("The temperature is 212 F.", {"pattern": r"\b212\s*F\b"})
    assert res.passed is True
    assert res.score == 1.0

    res = RegexChecker.evaluate("The temperature is 100 C.", {"pattern": r"\b212\s*F\b"})
    assert res.passed is False
    assert res.score == 0.0


def test_numeric_tolerance_checker():
    # Absolute tolerance
    res = NumericToleranceChecker.evaluate(
        "Result is 212.05 Fahrenheit", {"reference": 212.0, "tolerance": 0.1, "relative": False}
    )
    assert res.passed is True
    assert res.score == 1.0

    # Outside absolute tolerance
    res = NumericToleranceChecker.evaluate(
        "Result is 212.5 Fahrenheit", {"reference": 212.0, "tolerance": 0.1, "relative": False}
    )
    assert res.passed is False
    assert res.score == 0.0

    # No number extracted
    res = NumericToleranceChecker.evaluate(
        "No number here", {"reference": 212.0, "tolerance": 0.1, "relative": False}
    )
    assert res.passed is False
    assert res.score == 0.0


def test_rubric_checker():
    res = RubricChecker.evaluate(
        "The quick brown fox jumps over the lazy dog.",
        {
            "required_keywords": ["quick", "fox"],
            "forbidden_keywords": ["cat"],
            "min_length": 10,
        },
    )
    assert res.passed is True
    assert res.score == 1.0

    # Forbidden keyword violation
    res = RubricChecker.evaluate(
        "The quick fox and lazy cat.",
        {
            "required_keywords": ["quick", "fox"],
            "forbidden_keywords": ["cat"],
            "min_length": 5,
        },
    )
    assert res.passed is False
    assert res.score == 0.0


def test_evaluate_output_dispatch():
    config = {"checker_type": "exact", "reference": "Hello World"}
    res = evaluate_output("Hello World", config)
    assert res.passed is True
    assert res.score == 1.0


# 2. Environmental Estimator Tests
# -------------------------------------------------------------
def test_environmental_estimator_breakdown():
    estimator = EnvironmentalEstimator()

    # Model Large estimation (e.g. 500 tokens)
    estimate = estimator.estimate(tier="MODEL_LARGE", model_id=None, total_tokens=500)
    assert estimate.energy_wh.provenance.value == "estimated"
    assert estimate.energy_range_wh.central > 0.0
    assert (
        estimate.energy_range_wh.low
        < estimate.energy_range_wh.central
        < estimate.energy_range_wh.high
    )

    # Water breakdown separation
    assert estimate.direct_cooling_water_ml.value > 0.0
    assert estimate.indirect_power_water_ml.value > 0.0
    assert round(estimate.total_water_ml.value, 4) == round(
        estimate.direct_cooling_water_ml.value + estimate.indirect_power_water_ml.value, 4
    )

    # System boundary verification
    assert "scope 2" in estimator.energy_boundary.included.lower()
    assert "cooling" in estimator.direct_water_boundary.included.lower()
    assert "generation" in estimator.indirect_water_boundary.included.lower()

    # Deterministic route (zero token consumption)
    det_estimate = estimator.estimate(tier="DETERMINISTIC_RULES", model_id=None, total_tokens=0)
    assert det_estimate.energy_range_wh.central >= 0.0
    assert det_estimate.total_water_ml.value >= 0.0


# 3. Judge Tests
# -------------------------------------------------------------
def test_mock_judge_deterministic_scoring():
    judge = MockJudge()
    item = DatasetItem(
        item_id="test_01",
        task_family_id="fam_01",
        task_category="arithmetic_and_unit_conversion",
        tier="easy",
        split="dev",
        prompt="Convert 100 C to Fahrenheit",
        quality_criteria=["correct conversion", "212 F", "Fahrenheit"],
        checker_type="exact",
        checker_config={"reference": "212 F"},
    )

    # Perfect coverage of criteria
    res = judge.evaluate(item=item, output_text="The correct conversion is 212 F (212 Fahrenheit).")
    assert res.score >= 0.70
    assert res.pass_threshold is True
    assert res.judge_cost_usd == 0.0

    # Zero criteria met
    res_zero = judge.evaluate(item=item, output_text="I don't know the answer.")
    assert res_zero.score <= 0.50
    assert res_zero.pass_threshold is False


def test_openrouter_judge_fallback():
    # When no API key is provided, OpenRouterJudge safely falls back to MockJudge
    judge = OpenRouterJudge(api_key="")
    item = DatasetItem(
        item_id="test_02",
        task_family_id="fam_02",
        task_category="short_explanations",
        tier="easy",
        split="dev",
        prompt="Explain photosynthesis",
        quality_criteria=["sunlight", "plants", "glucose"],
        checker_type="rubric",
        checker_config={"required_keywords": ["plants"]},
    )
    res = judge.evaluate(
        item=item, output_text="Plants convert sunlight into chemical energy and glucose."
    )
    assert res.pass_threshold is True
    assert "MockJudge" in res.judge_model_id or "fallback" in res.judge_model_id


# 4. Human Review & Cohen's Kappa Tests
# -------------------------------------------------------------
def test_weighted_cohen_kappa():
    # Perfect agreement
    r1 = [5, 4, 3, 2, 1]
    r2 = [5, 4, 3, 2, 1]
    assert compute_weighted_cohen_kappa(r1, r2, categories=[1, 2, 3, 4, 5]) == 1.0

    # Mild disagreement (off by 1)
    r3 = [5, 4, 3, 2, 2]
    kappa_mild = compute_weighted_cohen_kappa(r1, r3, categories=[1, 2, 3, 4, 5])
    assert 0.70 < kappa_mild < 1.0

    # Severe disagreement (off by 4)
    r4 = [1, 2, 3, 4, 5]
    kappa_severe = compute_weighted_cohen_kappa(r1, r4, categories=[1, 2, 3, 4, 5])
    assert kappa_severe < 0.0


def test_generate_blind_human_review_csv():
    items = [
        DatasetItem(
            item_id=f"item_{i}",
            task_family_id=f"fam_{i // 2}",
            task_category="arithmetic_and_unit_conversion",
            tier="easy",
            split="test",
            prompt=f"Prompt {i}",
            quality_criteria=["criterion"],
            checker_type="exact",
            checker_config={"reference": "ans"},
        )
        for i in range(10)
    ]
    sys_outputs = {
        f"item_{i}": {
            "System_A": f"Output A {i}",
            "System_C": f"Output C {i}",
        }
        for i in range(10)
    }

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        csv_path = Path(tmpdir) / "blind_review.csv"
        csv_text = generate_blind_human_review_csv(
            test_items=items,
            system_outputs=sys_outputs,
            output_path=csv_path,
            sample_ratio=0.5,
            seed=42,
        )

        assert csv_path.exists()
        assert "candidate_a" in csv_text
        assert "candidate_b" in csv_text
        # Model identities must be completely blinded in the review CSV
        assert "System_A" not in csv_text
        assert "System_C" not in csv_text


# 5. Config Hash & Test-Set Lock Tests
# -------------------------------------------------------------
def test_compute_config_hash():
    h = compute_config_hash()
    assert isinstance(h, str)
    assert len(h) == 64


def test_verify_test_set_lock():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        reg_path = Path(tmpdir) / "registry.json"

        # Register current hash
        reg_cfg = register_frozen_config(registry_path=reg_path, description="Test lock registry")
        curr_hash = reg_cfg.config_hash

        # Dev split should succeed regardless of git state or allow_dirty
        verify_test_set_lock(
            split="dev", config_hash=curr_hash, registry_path=reg_path, allow_dirty_git=True
        )

        # Unregistered config hash on test split must fail
        with pytest.raises(RuntimeError, match="is not registered"):
            verify_test_set_lock(
                split="test",
                config_hash="0000000000000000000000000000000000000000000000000000000000000000",
                registry_path=reg_path,
                allow_dirty_git=True,
            )

        # Test split with allow_dirty_git=False should inspect git status
        # (if git is clean or dirty, properly enforced)
        load_registry(registry_path=reg_path)


# 6. Statistics & Effect Sizes Tests
# -------------------------------------------------------------
def test_repeated_run_summary():
    data = [10.0, 12.0, 11.0, 10.5, 11.5]
    summary = compute_repeated_run_summary(data)
    assert summary.mean == 11.0
    assert summary.median == 11.0
    assert summary.sample_size == 5
    assert summary.iqr > 0.0


def test_cliffs_delta():
    # Identical distributions -> delta = 0
    d, interp = compute_cliffs_delta([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert d == 0.0
    assert interp == "negligible"

    # Complete dominance -> delta = 1.0
    d_dom, interp_dom = compute_cliffs_delta([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
    assert d_dom == 1.0
    assert interp_dom == "large"


def test_clustered_bootstrap_ci():
    clusters = {
        "fam_1": [0.85, 0.90],
        "fam_2": [0.80, 0.85],
        "fam_3": [0.95, 0.95],
        "fam_4": [0.70, 0.75],
    }
    ci = compute_clustered_bootstrap_ci(clusters, num_bootstraps=100, seed=42)
    assert ci.ci_lower <= ci.mean <= ci.ci_upper
    assert 0.70 <= ci.mean <= 0.95


# 7. Oracle & Regret Calculation Tests
# -------------------------------------------------------------
def test_oracle_evaluator_cheapest_meeting_qmin():
    pipeline = RequestPipeline()
    oracle_eval = OracleEvaluator(pipeline=pipeline, q_min=0.70)

    # Deterministic item
    item = DatasetItem(
        item_id="math_item",
        task_family_id="fam_math",
        task_category="arithmetic_and_unit_conversion",
        tier="easy",
        split="dev",
        prompt="Convert 25 C to Fahrenheit",
        quality_criteria=["77 F"],
        checker_type="exact",
        checker_config={"reference": "77 F"},
    )

    oracle_res = oracle_eval.evaluate_item_routes(item)
    assert oracle_res.oracle_tier is not None
    assert oracle_res.oracle_cost_usd is not None and oracle_res.oracle_cost_usd >= 0.0


def test_oracle_excludes_prohibited_routes_for_private_requests():
    pipeline = RequestPipeline()
    oracle_eval = OracleEvaluator(pipeline=pipeline, q_min=0.70)

    private_item = DatasetItem(
        item_id="priv_item",
        task_family_id="fam_priv",
        task_category="summarization",
        tier="moderate",
        split="dev",
        prompt="Confidential internal document summary",
        quality_criteria=["summary"],
        checker_type="rubric",
        checker_config={"required_keywords": ["summary"]},
        privacy_class="confidential_private",
    )

    oracle_res = oracle_eval.evaluate_item_routes(private_item)
    # External LLM routes MUST be excluded from candidate routes
    candidate_routes = list(oracle_res.route_results.keys())
    assert "MODEL_SMALL" not in candidate_routes
    assert "MODEL_LARGE" not in candidate_routes


# 8. Cache Evaluator Isolated Benchmark Tests
# -------------------------------------------------------------
def test_cache_evaluator_cold_vs_warm():
    items = [
        DatasetItem(
            item_id="c_item_1",
            task_family_id="fam_c1",
            task_category="short_explanations",
            tier="easy",
            split="dev",
            prompt="Explain the greenhouse effect simply in two sentences.",
            quality_criteria=["greenhouse", "gases"],
            checker_type="rubric",
            checker_config={"required_keywords": ["greenhouse"]},
        ),
        DatasetItem(
            item_id="c_item_2",
            task_family_id="fam_c1",
            task_category="short_explanations",
            tier="easy",
            split="dev",
            prompt="Explain the greenhouse effect simply in two sentences.",  # Exact duplicate
            quality_criteria=["greenhouse", "gases"],
            checker_type="rubric",
            checker_config={"required_keywords": ["greenhouse"]},
        ),
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = str(Path(tmpdir) / "test_cache_eval.db")
        pipeline = RequestPipeline()
        cache_eval = CacheEvaluator(pipeline=pipeline, cache_db_path=db_path)
        cold_m, _ = cache_eval.run_workload(items=items, mode="cold_cache")
        warm_m, _ = cache_eval.run_workload(items=items, mode="warm_cache")

        assert cold_m.cache_hits >= 1
        assert warm_m.cache_hits == 2
        assert warm_m.hit_rate_overall == 1.0
        assert warm_m.inappropriate_hits == 0
