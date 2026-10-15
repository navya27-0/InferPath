"""Curated real-provider evaluation script executing live models via OpenRouter.

Runs a curated sample of 20 benchmark items across System A (Always-Large),
System Small (Always-Small), and System C (Constraint-Aware InferPath) against
live OpenRouter endpoints.

Strictly requires OPENROUTER_API_KEY in the environment or `.env`.
Never falls back to mock models silently. Logs all results with [MEASURED] provenance
to `data/results/real/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from inferpath.config.settings import load_settings
from inferpath.evaluation.dataset import load_pilot_dataset
from inferpath.evaluation.harness import ExperimentHarness
from inferpath.security.budget import RunBudgetTracker

load_dotenv()

OUTPUT_REAL_DIR = Path("data/results/real")


def run_real_evaluation(sample_size: int = 20, max_budget_usd: float = 2.0) -> int:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key.startswith("your_") or api_key == "":
        print("\n" + "=" * 70)
        print("ERROR: OPENROUTER_API_KEY is not set or contains a placeholder.")
        print("=" * 70)
        print("InferPath strictly adheres to empirical research validity standards:")
        print("Real-provider runs require a valid OpenRouter API key.")
        print("\nTo execute a live-provider evaluation:")
        print("  1. Obtain an API key from https://openrouter.ai/keys")
        print("  2. Set the key in your environment or .env file:")
        print("     $env:OPENROUTER_API_KEY = 'sk-or-v1-...'   (PowerShell)")
        print("     export OPENROUTER_API_KEY='sk-or-v1-...'   (Bash)")
        print("  3. Re-run this script: python scripts/run_real_evaluation.py --sample-size 20")
        print("=" * 70 + "\n")
        return 1

    settings = load_settings()
    settings.openrouter.api_key = api_key

    dataset = load_pilot_dataset()
    dev_items = dataset.get_split("dev")
    sample_items = dev_items[:sample_size]

    print("=== InferPath Live OpenRouter Real-Provider Benchmark ===")
    print(f"Sample Size: {len(sample_items)} items | Max Budget: ${max_budget_usd:.2f}")
    print(f"Small Model: {settings.openrouter.small_model}")
    print(f"Large Model: {settings.openrouter.large_model}")
    print(f"Output Directory: {OUTPUT_REAL_DIR}")

    OUTPUT_REAL_DIR.mkdir(parents=True, exist_ok=True)
    budget_tracker = RunBudgetTracker(max_budget_usd=max_budget_usd)

    harness = ExperimentHarness(
        adapter_mode="real",
        output_base_dir=OUTPUT_REAL_DIR,
        q_min=0.70,
        budget_tracker=budget_tracker,
    )

    systems_to_test = ["System_Small", "System_C", "System_A"]
    summary_results = []

    for sys_id in systems_to_test:
        print(f"\n--> Executing live run for: {sys_id} ({len(sample_items)} items)...")
        try:
            summary, manifest, logs = harness.run_evaluation(
                system_id=sys_id,
                items=sample_items,
                run_index=1,
                seed=42,
                split="dev_sample",
                config_hash="live_evaluation_pilot",
                verify_lock=False,
            )
            summary_results.append(
                {
                    "system": sys_id,
                    "quality_pass_rate": summary.quality_pass_rate,
                    "total_cost_usd": summary.total_cost_usd_calculated,
                    "avg_latency_ms": summary.avg_latency_ms,
                    "p95_latency_ms": summary.p95_latency_ms,
                    "cache_hit_rate": summary.cache_hit_rate,
                    "fallback_rate": summary.fallback_rate,
                }
            )
            print(
                f"    Completed {sys_id}: Quality={summary.quality_pass_rate * 100:.1f}%, "
                f"Cost=${summary.total_cost_usd_calculated:.5f}, "
                f"Latency={summary.avg_latency_ms:.1f}ms, "
                f"Fallback={summary.fallback_rate * 100:.1f}%"
            )
        except Exception as e:
            print(f"    Failed execution for {sys_id}: {e}")

    summary_file = OUTPUT_REAL_DIR / "real_pilot_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sample_size": len(sample_items),
                "total_dev_items": len(dev_items),
                "is_partial_comparison": len(sample_items) < len(dev_items),
                "adapter_mode": "real",
                "results": summary_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved live-provider summary to {summary_file}")
    print(
        f"Safety budget total tracked: ${budget_tracker.total_spent_usd:.5f} (limit: ${max_budget_usd:.2f})"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run curated live OpenRouter benchmark.")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of items to sample.")
    parser.add_argument(
        "--max-budget", type=float, default=2.0, help="Maximum budget ceiling in USD."
    )
    args = parser.parse_args()

    sys.exit(run_real_evaluation(sample_size=args.sample_size, max_budget_usd=args.max_budget))
