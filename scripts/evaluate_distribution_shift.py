"""Script to evaluate router robustness across 10 distribution shift dimensions."""

from __future__ import annotations

import json
from pathlib import Path

from inferpath.evaluation.distribution_shift import evaluate_router_under_distribution_shift

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "data" / "results" / "distribution_shift_evaluation.json"


def main() -> int:
    results = evaluate_router_under_distribution_shift()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Distribution shift evaluation saved to {OUTPUT_FILE}")
    print(f"  Total items: {results['total_shift_items']}")
    print(f"  Overall pass rate: {results['overall_pass_rate'] * 100:.1f}%")
    print(f"  Tier distribution: {results['tier_distribution']}")
    print(f"  Tested dimensions: {list(results['dimension_breakdown'].keys())}")
    return 0


if __name__ == "__main__":
    main()
