"""Overview Tab for InferPath Research Dashboard.

Displays:
- Side-by-side KPI comparison of System A, System B, and System C
- Provenance-badged metrics (measured latency, calculated cost, estimated energy/water)
- Clustered bootstrap confidence intervals with methodology and sample size
- Quality vs. Cost scatter plot
- Difficulty tier breakdown (Easy, Moderate, Difficult)
"""

import altair as alt
import pandas as pd
import streamlit as st

from inferpath.dashboard.components import metric_card
from inferpath.dashboard.data_loader import ManifestMetadata, RunData


def render_overview_tab(
    selected_run: RunData,
    benchmark_summary: dict | None,
    all_manifests: list[ManifestMetadata],
):
    st.subheader("System Comparison & Core Benchmarks")

    manifest = selected_run.manifest
    logs = selected_run.logs

    # Provenance context banner
    mode_text = (
        "Simulated (Synthetic/Mock Model Adapters)"
        if manifest.adapter_mode == "simulated"
        else "Real Provider Execution (Live API)"
    )
    st.caption(
        f"Selected Run: **{manifest.run_id}** | Mode: **{mode_text}** | Split: **{manifest.split.upper()}** | "
        f"Items: **{len(logs) if logs else manifest.summary_metrics.get('total_items', 'N/A')}**"
    )

    # 1. Side-by-Side Comparison of Systems A, B, C from Benchmark Summary
    st.markdown("### Primary Architectures Side-by-Side")
    st.caption(
        "Baseline comparison across Systems A, B, and C. Metrics update dynamically from evaluation summary."
    )

    if benchmark_summary and "table_rows" in benchmark_summary:
        rows = benchmark_summary["table_rows"]
        # Filter primary systems
        primary_rows = [r for r in rows if r["System"] in ["System_A", "System_B", "System_C"]]
        if primary_rows:
            cols = st.columns(len(primary_rows))
            for i, r in enumerate(primary_rows):
                sys_id = r["System"]
                title_map = {
                    "System_A": "System A: Large Model Only",
                    "System_B": "System B: Cache + Rules + Large",
                    "System_C": "System C: Full InferPath",
                }
                disp_title = title_map.get(sys_id, sys_id)
                with cols[i]:
                    st.markdown(f"**{disp_title}**")
                    metric_card(
                        "Quality Pass Rate",
                        r.get("Quality_Pass_Rate", "N/A"),
                        "measured" if manifest.adapter_mode == "real" else "simulated",
                        subtitle=f"95% CI: {r.get('Quality_95_CI', 'N/A')}",
                    )
                    metric_card(
                        "Total Cost (USD)",
                        r.get("Total_Cost_USD", "N/A"),
                        "calculated",
                        subtitle=f"Reduction vs A: {r.get('Cost_Reduction_vs_A', '0.0%')}",
                    )
                    metric_card(
                        "Mean Latency",
                        r.get("Avg_Latency_ms", "N/A"),
                        "measured" if manifest.adapter_mode == "real" else "simulated",
                        subtitle=f"P95: {r.get('P95_Latency_ms', 'N/A')}",
                    )
                    metric_card(
                        "Energy (Compute)",
                        r.get("Energy_Wh", "N/A"),
                        "estimated",
                        subtitle="Scope 2 Compute",
                    )
                    metric_card(
                        "Water Footprint",
                        r.get("Water_mL", "N/A"),
                        "estimated",
                        subtitle="Direct WUE + Indirect",
                    )

            # Comparative table
            st.markdown("#### Full Systems and Ablations Evaluation Matrix")
            df_summary = pd.DataFrame(rows)
            st.dataframe(df_summary, use_container_width=True)
            st.caption(
                "Source Provenance: Latency/Quality = [SIMULATED] | Cost = [CALCULATED] | Energy/Water = [ESTIMATED] (Dated Table: 2026-10-01)"
            )
    else:
        st.info(
            "No multi-run benchmark summary found. Displaying single-run metrics for selected manifest."
        )

    st.markdown("---")

    # 2. Selected Run Deep-Dive
    st.markdown(f"### Selected Run Detail: `{manifest.system_id}` (Run #{manifest.run_index})")

    metrics = manifest.summary_metrics
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pass_rate = metrics.get("quality_pass_rate", None)
            val_str = f"{pass_rate * 100:.1f}%" if pass_rate is not None else "N/A"
            metric_card(
                "Quality Pass Rate",
                val_str,
                "simulated" if manifest.adapter_mode == "simulated" else "measured",
                subtitle=f"Q >= {manifest.q_min}",
            )
        with c2:
            cost_val = metrics.get("total_cost_usd_calculated", None)
            val_str = f"${cost_val:.5f}" if cost_val is not None else "N/A"
            metric_card(
                "Batch Cost", val_str, "calculated", unit="USD", subtitle="Price table: 2026-10-01"
            )
        with c3:
            lat_val = metrics.get("avg_latency_ms", None)
            val_str = f"{lat_val:.1f}" if lat_val is not None else "N/A"
            metric_card(
                "Mean Latency",
                val_str,
                "simulated" if manifest.adapter_mode == "simulated" else "measured",
                unit="ms",
                subtitle=f"P95: {metrics.get('p95_latency_ms', 'N/A')} ms",
            )
        with c4:
            hit_val = metrics.get("cache_hit_rate", None)
            val_str = f"{hit_val * 100:.1f}%" if hit_val is not None else "N/A"
            metric_card(
                "Cache Hit Rate",
                val_str,
                "measured" if manifest.adapter_mode == "real" else "simulated",
                subtitle=f"Fallback: {metrics.get('fallback_rate', 0.0) * 100:.1f}%",
            )

    if not logs:
        st.warning("Item-level logs are not available for this run to render distribution charts.")
        return

    # Convert logs to DataFrame for charts
    df_logs = pd.DataFrame(logs)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Quality vs. Cost per Item")
        st.caption(
            "Scatter of individual evaluated requests colored by difficulty tier [CALCULATED / SIMULATED]"
        )
        if "quality_score" in df_logs.columns and "cost_usd_calculated" in df_logs.columns:
            scatter_chart = (
                alt.Chart(df_logs)
                .mark_circle(size=70, opacity=0.7)
                .encode(
                    x=alt.X(
                        "cost_usd_calculated:Q",
                        title="Cost (USD, Calculated)",
                        scale=alt.Scale(zero=True),
                    ),
                    y=alt.Y(
                        "quality_score:Q",
                        title="Quality Score [0.0 - 1.0]",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    color=alt.Color(
                        "tier:N", title="Difficulty Tier", scale=alt.Scale(scheme="tealblues")
                    ),
                    tooltip=[
                        "item_id",
                        "tier",
                        "routed_tier",
                        "quality_score",
                        "cost_usd_calculated",
                        "latency_ms",
                    ],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(scatter_chart, use_container_width=True)

    with col_chart2:
        st.markdown("#### Performance by Difficulty Tier")
        st.caption("Mean latency and cost grouped by prompt difficulty [MEASURED / CALCULATED]")
        if "tier" in df_logs.columns:
            tier_summary = (
                df_logs.groupby("tier")
                .agg(
                    count=("item_id", "count"),
                    avg_quality=("quality_score", "mean"),
                    avg_cost=("cost_usd_calculated", "mean"),
                    avg_latency=("latency_ms", "mean"),
                    meets_q_min=("meets_q_min", lambda x: sum(x) / len(x) if len(x) > 0 else 0),
                )
                .reset_index()
            )

            bar_chart = (
                alt.Chart(tier_summary)
                .mark_bar(color="#0D9488")
                .encode(
                    x=alt.X("tier:N", title="Difficulty Tier"),
                    y=alt.Y("avg_latency:Q", title="Mean Latency (ms)"),
                    tooltip=["tier", "count", "avg_latency", "avg_cost", "avg_quality"],
                )
                .properties(height=320)
            )
            st.altair_chart(bar_chart, use_container_width=True)

    # 3. Clustered Bootstrap Confidence Interval Disclosure
    st.markdown("#### Confidence Interval Methodology & Grounding")
    st.info(
        "**Statistical Design**: Confidence intervals are computed using clustered bootstrap resampling over "
        "80 distinct `task_family_id` semantic groups (1,000 iterations). Because queries within the same family "
        "exhibit correlated semantic structure, standard independent-observation assumptions are invalid. "
        "All reported intervals preserve cluster boundaries to guard against artificially narrow error bands."
    )
