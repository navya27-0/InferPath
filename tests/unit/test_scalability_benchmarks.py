import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_scalability import simulate_concurrent_load  # noqa: E402


def test_simulate_concurrent_load():
    res = simulate_concurrent_load(concurrency=5, num_requests=20, workers=1)
    assert res["concurrency"] == 5
    assert res["total_requests"] == 20
    assert "latency_percentiles_ms" in res
    assert res["latency_percentiles_ms"]["p50"] <= res["latency_percentiles_ms"]["p99"]
    assert "component_breakdown_avg_ms" in res
    assert res["component_breakdown_avg_ms"]["router_decision_compute"] < 5.0
