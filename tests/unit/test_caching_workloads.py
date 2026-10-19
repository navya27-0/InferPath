"""Unit tests for caching workload evaluation."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_caching_workloads import simulate_trace_workload  # noqa: E402


def test_simulate_trace_workload():
    rep = simulate_trace_workload("enterprise_repetitive", num_requests=50)
    assert rep["hit_rate_pct"] > 50.0
    assert rep["cost_savings_pct"] > 50.0

    unique = simulate_trace_workload("unique_long_tail", num_requests=50)
    assert unique["hit_rate_pct"] == 0.0

    priv = simulate_trace_workload("privacy_restricted", num_requests=50)
    assert priv["hit_rate_pct"] == 0.0
