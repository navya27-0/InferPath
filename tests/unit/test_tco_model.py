"""Unit tests for Total Cost of Ownership (TCO) model."""

from inferpath.evaluation.tco_model import TotalCostOfOwnershipEstimator


def test_tco_estimator_single_query():
    estimator = TotalCostOfOwnershipEstimator()
    res = estimator.estimate_single_query_tco(
        prompt_tokens=100,
        completion_tokens=50,
        inference_cost_usd=0.0001,
        had_fallback=False,
    )
    assert res.total_cost_usd > res.inference_cost_usd
    assert 0.0 < res.overhead_percentage < 10.0


def test_tco_workload_sweep():
    estimator = TotalCostOfOwnershipEstimator()
    sweep = estimator.evaluate_workload_tco([10, 100, 1000])
    assert len(sweep) == 3
    # Shorter prompt should have higher percentage overhead
    assert sweep[0]["router_overhead_percentage"] > sweep[2]["router_overhead_percentage"]
