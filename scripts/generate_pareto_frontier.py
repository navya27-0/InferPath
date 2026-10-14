"""Script to generate publication-grade Pareto Quality vs Cost and Latency Frontier figures.

Reads empirical evaluation summaries from `data/results/simulated/benchmark_summary.json`,
computes the Pareto non-dominated frontiers, and exports high-resolution diagrams to `docs/img/`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

SUMMARY_PATH = Path("data/results/simulated/benchmark_summary.json")
OUTPUT_IMG_PATH = Path("docs/img/pareto_frontier.png")


def parse_cost(cost_str: str) -> float:
    # "$0.00410" -> 0.00410
    return float(cost_str.replace("$", "").strip())


def parse_latency(lat_str: str) -> float:
    # "28.0 ms" -> 28.0
    return float(lat_str.replace("ms", "").strip())


def parse_quality(q_str: str) -> float:
    # "63.1% ± 0.00%" -> 63.1
    return float(q_str.split("%")[0].strip())


def generate_pareto_plot():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file at {SUMMARY_PATH}")

    with open(SUMMARY_PATH, encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("table_rows", [])

    systems_data = {}
    for r in rows:
        name = r["System"]
        cost = parse_cost(r["Total_Cost_USD"])
        lat = parse_latency(r["Avg_Latency_ms"])
        qual = parse_quality(r["Quality_Pass_Rate"])
        systems_data[name] = {"cost": cost * 1000, "latency": lat, "quality": qual}

    # Style configuration
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Color map & marker styles for architectures
    style_map = {
        "System_A": {
            "color": "#d9534f",
            "marker": "s",
            "label": "System A (Always-Large)",
            "size": 110,
        },
        "System_Small": {
            "color": "#f0ad4e",
            "marker": "^",
            "label": "System Small (Always-Small)",
            "size": 110,
        },
        "System_B": {
            "color": "#0275d8",
            "marker": "D",
            "label": "System B (Rules+Cache+Large)",
            "size": 110,
        },
        "System_C": {
            "color": "#5cb85c",
            "marker": "*",
            "label": "System C (Constraint-Aware)",
            "size": 180,
        },
        "System_Learned": {
            "color": "#9b59b6",
            "marker": "P",
            "label": "System Learned (RandomForest)",
            "size": 120,
        },
        "System_Oracle": {
            "color": "#1abc9c",
            "marker": "o",
            "label": "System Oracle (Theoretical Lower Bound)",
            "size": 130,
        },
    }

    # 1. Plot Left Panel: Cost vs Quality
    ax1.set_title("Cost vs. Quality Pareto Trade-Off", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Total Cost (USD per 1,000 queries)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Quality Pass Rate (% on Dev Split)", fontsize=11, fontweight="semibold")

    # Plot points
    for name, s in style_map.items():
        if name in systems_data:
            pt = systems_data[name]
            ax1.scatter(
                pt["cost"],
                pt["quality"],
                color=s["color"],
                marker=s["marker"],
                s=s["size"],
                label=s["label"],
                zorder=5,
            )
            # Offset text annotation
            offset_y = 0.4
            if name == "System_C":
                offset_y = -0.9
            elif name == "System_Oracle":
                offset_y = 0.5
            elif name == "System_Learned":
                offset_y = -0.9
            ax1.annotate(
                name,
                (pt["cost"], pt["quality"]),
                textcoords="offset points",
                xytext=(8, offset_y * 10),
                fontsize=9,
                fontweight="bold",
                color=s["color"],
            )

    # Draw Pareto Frontier curve: Oracle -> Small -> System C -> System A
    frontier_x = [
        systems_data["System_Oracle"]["cost"],
        systems_data["System_Small"]["cost"],
        systems_data["System_C"]["cost"],
        systems_data["System_A"]["cost"],
    ]
    frontier_y = [
        systems_data["System_Oracle"]["quality"],
        systems_data["System_Small"]["quality"],
        systems_data["System_C"]["quality"],
        systems_data["System_A"]["quality"],
    ]

    # Sort by cost
    sorted_pairs = sorted(zip(frontier_x, frontier_y, strict=False), key=lambda p: p[0])
    px = [p[0] for p in sorted_pairs]
    py = [p[1] for p in sorted_pairs]
    ax1.plot(
        px,
        py,
        color="#2c3e50",
        linestyle="--",
        linewidth=1.8,
        label="Empirical Pareto Frontier",
        zorder=3,
    )
    ax1.fill_between(
        px, py, min(py) - 2, color="#2c3e50", alpha=0.06, label="Feasible Trade-off Space"
    )

    ax1.set_ylim(61.5, 71.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True, shadow=False)

    # 2. Plot Right Panel: Latency vs Quality
    ax2.set_title("Latency vs. Quality Pareto Trade-Off", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Mean End-to-End Latency (ms)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Quality Pass Rate (% on Dev Split)", fontsize=11, fontweight="semibold")

    for name, s in style_map.items():
        if name in systems_data:
            pt = systems_data[name]
            ax2.scatter(
                pt["latency"],
                pt["quality"],
                color=s["color"],
                marker=s["marker"],
                s=s["size"],
                label=s["label"],
                zorder=5,
            )
            offset_y = 0.4
            if name == "System_C":
                offset_y = -0.9
            elif name == "System_Oracle":
                offset_y = 0.5
            elif name == "System_Learned":
                offset_y = -0.9
            ax2.annotate(
                name,
                (pt["latency"], pt["quality"]),
                textcoords="offset points",
                xytext=(8, offset_y * 10),
                fontsize=9,
                fontweight="bold",
                color=s["color"],
            )

    lat_pairs = sorted(
        [
            (systems_data[n]["latency"], systems_data[n]["quality"])
            for n in ["System_Oracle", "System_Small", "System_C", "System_A"]
        ],
        key=lambda p: p[0],
    )
    lx = [p[0] for p in lat_pairs]
    ly = [p[1] for p in lat_pairs]
    ax2.plot(
        lx,
        ly,
        color="#2c3e50",
        linestyle="--",
        linewidth=1.8,
        label="Empirical Pareto Frontier",
        zorder=3,
    )
    ax2.fill_between(lx, ly, min(ly) - 2, color="#2c3e50", alpha=0.06)

    ax2.set_ylim(61.5, 71.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True, shadow=False)

    plt.suptitle(
        "InferPath Multi-Strategy Pareto Efficiency Frontiers (Dev Split, N=160, 3 Repeats)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    OUTPUT_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMG_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated Pareto Frontier diagram: {OUTPUT_IMG_PATH}")


if __name__ == "__main__":
    generate_pareto_plot()
