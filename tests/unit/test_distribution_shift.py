"""Unit tests for distribution shift evaluation suite."""

from inferpath.evaluation.distribution_shift import (
    evaluate_router_under_distribution_shift,
    generate_distribution_shift_dataset,
)


def test_generate_distribution_shift_dataset():
    dataset = generate_distribution_shift_dataset()
    assert len(dataset) >= 10
    dimensions = {item.dimension for item in dataset}
    assert len(dimensions) == 10
    assert "paraphrased_prompts" in dimensions
    assert "unseen_task_families" in dimensions
    assert "multilingual" in dimensions


def test_evaluate_router_under_distribution_shift():
    results = evaluate_router_under_distribution_shift()
    assert results["total_shift_items"] >= 10
    assert 0.0 <= results["overall_pass_rate"] <= 1.0
    assert results["total_cost_usd"] > 0.0
    assert len(results["dimension_breakdown"]) == 10
