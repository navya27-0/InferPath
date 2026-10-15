"""Main evaluation benchmark runner executing Systems A, B, C, ablations, and generating evaluation summaries."""

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from inferpath.evaluation.cache_eval import CacheEvaluator
from inferpath.evaluation.dataset import load_pilot_dataset
from inferpath.evaluation.freeze_config import (
    load_registry,
    register_frozen_config,
)
from inferpath.evaluation.harness import (
    ExperimentHarness,
    SystemExecutionSummary,
)
from inferpath.evaluation.human_review import (
    generate_blind_human_review_csv,
)
from inferpath.evaluation.stats import (
    compute_clustered_bootstrap_ci,
    compute_paired_per_item_comparison,
    compute_repeated_run_summary,
)


def run_full_benchmark_suite(
    adapter_mode: str = "simulated",
    split: str = "dev",
    repeats: int = 3,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Executes the complete Phase 5 evaluation protocol:
    1. Loads dataset and registers frozen configuration.
    2. Runs Systems A, B, C and 5 ablations for N repeated trials.
    3. Runs cold-cache and warm-cache benchmark.
    4. Generates blind human review CSV and computes agreement metrics.
    5. Computes clustered bootstrap CIs and effect sizes.
    6. Formats markdown summary table and writes results to disk.
    """
    print("=== InferPath Phase 5 Benchmark Suite ===")
    print(f"Adapter Mode: {adapter_mode} | Split: {split} | Repeats: {repeats}")

    dataset = load_pilot_dataset()
    items = dataset.get_split(split)
    print(
        f"Loaded {len(items)} items for split '{split}' (Total dataset: {dataset.total_count} items)."
    )

    # Register/ensure frozen config
    registry = load_registry()
    if not registry.registered_configs:
        cfg = register_frozen_config(q_min=0.70)
        config_hash = cfg.config_hash
    else:
        config_hash = list(registry.registered_configs.keys())[0]

    harness = ExperimentHarness(adapter_mode=adapter_mode, output_base_dir=output_dir, q_min=0.70)

    systems = [
        "System_A",  # Large-only baseline
        "System_Small",  # Small-only baseline
        "System_B",  # Exact cache + deterministic rules + large model
        "System_C",  # Full InferPath (Constraint-aware)
        "System_Learned",  # Supervised ML router
        "System_Oracle",  # Theoretical ex-post Pareto optimal lower bound
        "ablation_no_templates",
        "ablation_no_small_model",
        "ablation_no_fallback",
        "ablation_no_cache",
        "ablation_no_validation",
    ]

    all_summaries: dict[str, list[SystemExecutionSummary]] = {}
    all_system_logs: dict[str, list[list[dict[str, Any]]]] = {}
    representative_outputs: dict[str, dict[str, str]] = {}  # {item_id: {sys_id: text}}

    for sys_id in systems:
        print(f"\n--> Running system: {sys_id} ({repeats} runs)...")
        all_summaries[sys_id] = []
        all_system_logs[sys_id] = []
        for r in range(1, repeats + 1):
            seed = 42 + r * 10
            summary, manifest, logs = harness.run_evaluation(
                system_id=sys_id,
                items=items,
                run_index=r,
                seed=seed,
                split=split,
                config_hash=config_hash,
                verify_lock=False,  # Unlocked for dev/mock runs
            )
            all_summaries[sys_id].append(summary)
            all_system_logs[sys_id].append(logs)

            # Store representative outputs for human review
            if r == 1:
                for log in logs:
                    representative_outputs.setdefault(log["item_id"], {})[sys_id] = log.get(
                        "response_text", ""
                    )

            print(
                f"    Run {r}: Quality={summary.quality_pass_rate * 100:.1f}%, "
                f"Cost=${summary.total_cost_usd_calculated:.5f}, "
                f"Latency={summary.avg_latency_ms:.1f}ms, "
                f"CacheHit={summary.cache_hit_rate * 100:.1f}%, "
                f"Fallback={summary.fallback_rate * 100:.1f}%"
            )

    # 3. Cache Benchmark (Cold vs Warm)
    print("\n--> Running Cache Cold vs Warm Benchmark...")
    bench_cache_file = harness.base_dir / "bench_cache.db"
    if bench_cache_file.exists():
        with contextlib.suppress(OSError):
            bench_cache_file.unlink()
    cache_evaluator = CacheEvaluator(
        pipeline=harness._create_pipeline_for_system("System_C", str(bench_cache_file)),
        cache_db_path=str(bench_cache_file),
    )
    cold_metrics, _ = cache_evaluator.run_workload(items, mode="cold_cache")
    warm_metrics, _ = cache_evaluator.run_workload(items, mode="warm_cache")

    print(
        f"    Cold Cache: Hit Rate = {cold_metrics.hit_rate_overall * 100:.1f}%, "
        f"Inappropriate Hits = {cold_metrics.inappropriate_hits}"
    )
    print(
        f"    Warm Cache: Hit Rate = {warm_metrics.hit_rate_overall * 100:.1f}%, "
        f"Inappropriate Hits = {warm_metrics.inappropriate_hits}, "
        f"Savings = ${warm_metrics.cost_savings_usd:.5f}"
    )

    # 4. Generate Blind Human Review CSV & Ingest Genuine Ratings (if available)
    print("\n--> Generating Blind Human Review CSV...")
    review_csv_path = harness.base_dir / "human_review_blind.csv"
    key_csv_path = harness.base_dir / "human_review_key.json"
    generate_blind_human_review_csv(
        test_items=items,
        system_outputs=representative_outputs,
        output_path=review_csv_path,
        key_path=key_csv_path,
        sample_ratio=0.20,
    )
    print(f"    Saved blind review CSV to {review_csv_path}")

    # Inspect whether genuine empirical human review ratings are provided
    ratings_csv_path = harness.base_dir / "human_ratings.csv"
    if not ratings_csv_path.exists():
        ratings_csv_path = Path("data") / "human_ratings.csv"

    from inferpath.evaluation.human_review import (
        compute_inter_rater_agreement,
        load_human_review_csv,
    )

    if ratings_csv_path.exists():
        records = load_human_review_csv(ratings_csv_path)
        agreement_res = compute_inter_rater_agreement(records)
        human_human_kappa = agreement_res.get("human_human_kappa")
        human_judge_kappa = agreement_res.get("human_judge_kappa")
        agreement_status = agreement_res.get("status", "measured")
        print(f"    Loaded {len(records)} human review records from {ratings_csv_path}.")
    else:
        human_human_kappa = None
        human_judge_kappa = None
        agreement_status = "not_established"
        print("    Inter-rater agreement: Not established (pending empirical human review study).")

    # 5. Statistical Aggregations across runs
    print("\n--> Computing Statistical Aggregations & Effect Sizes...")
    table_rows: list[dict[str, Any]] = []
    # Precompute per-item average costs for reference System C
    sys_c_logs = all_system_logs.get("System_C", [])
    item_costs_c: dict[str, float] = {}
    item_cost_lists_c: dict[str, list[float]] = {}
    for run_logs in sys_c_logs:
        for log_entry in run_logs:
            item_cost_lists_c.setdefault(log_entry["item_id"], []).append(
                log_entry.get("cost_usd_calculated", 0.0)
            )
    for itm_id, c_list in item_cost_lists_c.items():
        item_costs_c[itm_id] = sum(c_list) / len(c_list) if c_list else 0.0

    all_item_ids = [itm.item_id for itm in items]

    for sys_id, summaries in all_summaries.items():
        cost_stats = compute_repeated_run_summary([s.total_cost_usd_calculated for s in summaries])
        qual_stats = compute_repeated_run_summary([s.quality_pass_rate for s in summaries])
        lat_stats = compute_repeated_run_summary([s.avg_latency_ms for s in summaries])
        p95_stats = compute_repeated_run_summary([s.p95_latency_ms for s in summaries])
        regret_stats = compute_repeated_run_summary([s.avg_cost_regret for s in summaries])
        over_esc_stats = compute_repeated_run_summary([s.over_escalation_rate for s in summaries])
        under_esc_stats = compute_repeated_run_summary([s.under_escalation_rate for s in summaries])
        cache_stats = compute_repeated_run_summary([s.cache_hit_rate for s in summaries])
        fb_stats = compute_repeated_run_summary([s.fallback_rate for s in summaries])
        energy_stats = compute_repeated_run_summary([s.total_energy_wh for s in summaries])
        water_stats = compute_repeated_run_summary([s.total_water_ml for s in summaries])

        # Clustered bootstrap CI for quality based on genuine item-level outcomes
        # Extract item-level pass outcomes across repeat evaluations
        system_runs_logs = all_system_logs.get(sys_id, [])
        item_outcomes: dict[str, list[float]] = {}
        for run_logs in system_runs_logs:
            for log_entry in run_logs:
                itm_id = log_entry["item_id"]
                pass_val = 1.0 if log_entry.get("meets_q_min", False) else 0.0
                item_outcomes.setdefault(itm_id, []).append(pass_val)

        family_clusters: dict[str, list[float]] = {}
        missing_outcomes = 0
        for itm in items:
            passes = item_outcomes.get(itm.item_id, [])
            if passes:
                avg_item_pass = sum(passes) / len(passes)
                family_clusters.setdefault(itm.task_family_id, []).append(avg_item_pass)
            else:
                missing_outcomes += 1

        bootstrap_ci = compute_clustered_bootstrap_ci(
            family_clusters, n_resamples=1000, confidence_level=0.95, seed=42
        )

        # Paired per-item cost effect size relative to full InferPath (System C)
        item_costs_sys: dict[str, float] = {}
        item_cost_lists_sys: dict[str, list[float]] = {}
        for run_logs in system_runs_logs:
            for log_entry in run_logs:
                item_cost_lists_sys.setdefault(log_entry["item_id"], []).append(
                    log_entry.get("cost_usd_calculated", 0.0)
                )
        for itm_id, c_list in item_cost_lists_sys.items():
            item_costs_sys[itm_id] = sum(c_list) / len(c_list) if c_list else 0.0

        paired_res = compute_paired_per_item_comparison(
            item_costs_c, item_costs_sys, all_item_ids=all_item_ids
        )
        cliffs_d = paired_res.cliffs_delta
        effect_interp = paired_res.interpretation

        # Cost savings relative to System A
        sys_a_cost = sum(s.total_cost_usd_calculated for s in all_summaries["System_A"]) / len(
            all_summaries["System_A"]
        )
        cost_reduction_pct = (
            ((sys_a_cost - cost_stats.mean) / sys_a_cost) * 100.0 if sys_a_cost > 0 else 0.0
        )

        row = {
            "System": sys_id,
            "Quality_Pass_Rate": f"{qual_stats.mean * 100:.1f}% ± {qual_stats.std * 100:.2f}%",
            "Quality_95_CI": f"[{bootstrap_ci.ci_lower * 100:.1f}%, {bootstrap_ci.ci_upper * 100:.1f}%]",
            "Total_Cost_USD": f"${cost_stats.mean:.5f}",
            "Cost_Reduction_vs_A": f"{cost_reduction_pct:.1f}%",
            "Avg_Latency_ms": f"{lat_stats.mean:.1f} ms",
            "P95_Latency_ms": f"{p95_stats.mean:.1f} ms",
            "Cache_Hit_Rate": f"{cache_stats.mean * 100:.1f}%",
            "Fallback_Rate": f"{fb_stats.mean * 100:.1f}%",
            "Avg_Cost_Regret": f"${regret_stats.mean:.6f}",
            "Over_Escalation": f"{over_esc_stats.mean * 100:.1f}%",
            "Under_Escalation": f"{under_esc_stats.mean * 100:.1f}%",
            "Energy_Wh": f"{energy_stats.mean:.3f} Wh",
            "Water_mL": f"{water_stats.mean:.3f} mL",
            "Cost_Effect_vs_C": f"d={cliffs_d:.2f} ({effect_interp})"
            if paired_res.paired_n > 0
            else "N/A",
            "Missing_Observations": paired_res.missing_observations,
        }
        table_rows.append(row)

    # 6. Format Markdown Table
    md_table = format_markdown_table(table_rows)

    benchmark_bundle = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": harness.git_commit,
            "git_tree_clean": harness.git_tree_clean,
            "adapter_mode": adapter_mode,
            "split": split,
            "repeats": repeats,
            "config_hash": config_hash,
            "dataset_version": dataset.version,
            "dataset_count": len(items),
            "human_agreement_status": agreement_status,
            "human_human_kappa": human_human_kappa,
            "human_judge_kappa": human_judge_kappa,
            "bootstrap_estimator": "clustered_family_bootstrap_percentile",
            "bootstrap_seed": 42,
            "bootstrap_resamples": 1000,
            "confidence_level": 0.95,
        },
        "table_rows": table_rows,
        "markdown_table": md_table,
        "cold_cache_metrics": cold_metrics.model_dump(mode="json"),
        "warm_cache_metrics": warm_metrics.model_dump(mode="json"),
    }

    # Save to JSON and Markdown
    results_json_path = harness.base_dir / "benchmark_summary.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_bundle, f, indent=2, ensure_ascii=False)

    results_md_path = harness.base_dir / "benchmark_summary.md"
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write("# InferPath Phase 5 Research Evaluation Summary\n\n")
        f.write(
            f"**Adapter Mode**: `{adapter_mode}` | **Split**: `{split}` | **Repeats**: `{repeats}`\n\n"
        )
        f.write(md_table)
        f.write("\n\n### Inter-Rater Agreement\n")
        if human_human_kappa is not None:
            f.write(
                f"- **Human-Human Quadratic Weighted Kappa**: `{human_human_kappa:.3f}` `[MEASURED]`\n"
            )
        else:
            f.write(
                "- **Human-Human Quadratic Weighted Kappa**: `Not Established` (pending empirical human review study)\n"
            )
        if human_judge_kappa is not None:
            f.write(
                f"- **Human-Judge Quadratic Weighted Kappa**: `{human_judge_kappa:.3f}` `[MEASURED]`\n\n"
            )
        else:
            f.write(
                "- **Human-Judge Quadratic Weighted Kappa**: `Not Established` (pending empirical human review study)\n\n"
            )
        f.write("### Cache Performance (Cold vs Warm)\n")
        f.write(
            f"- **Cold Cache Hit Rate**: `{cold_metrics.hit_rate_overall * 100:.1f}%` (Inappropriate Hits: `{cold_metrics.inappropriate_hits}`)\n"
        )
        f.write(
            f"- **Warm Cache Hit Rate**: `{warm_metrics.hit_rate_overall * 100:.1f}%` (Inappropriate Hits: `{warm_metrics.inappropriate_hits}`)\n"
        )
        f.write(f"- **Warm Cache Cost Savings**: `${warm_metrics.cost_savings_usd:.5f}`\n")

    print(f"\nSaved benchmark outputs to:\n  - {results_json_path}\n  - {results_md_path}")
    print("\n" + md_table)
    return benchmark_bundle


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run InferPath evaluation benchmark suite.")
    parser.add_argument(
        "--mode",
        choices=["simulated", "real"],
        default="simulated",
        help="Adapter mode (simulated or real)",
    )
    parser.add_argument(
        "--split", choices=["dev", "test"], default="dev", help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of repeated runs per system/ablation"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    run_full_benchmark_suite(
        adapter_mode=args.mode,
        split=args.split,
        repeats=args.repeats,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
