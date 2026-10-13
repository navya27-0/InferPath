"""Routing & Oracle Analysis Tab for InferPath Research Dashboard.

Displays:
- Route distribution by tier and task category
- Quality-threshold satisfaction rate (with explicit denominators)
- Routing regret against the cheapest eligible route meeting Q_min
- Over-escalation and under-escalation rates
- Fallback frequency and associated cost
- Confusion matrix comparing Selected Route vs. Oracle Reference Route
- Filterable task category breakdown table with complete provenance tagging
"""

import altair as alt
import pandas as pd
import streamlit as st

from inferpath.dashboard.components import metric_card


def render_routing_tab(selected_run, benchmark_summary):
    st.subheader("Routing Dynamics & Oracle Calibration")

    manifest = selected_run.manifest
    logs = selected_run.logs

    if not logs:
        st.warning("No item-level execution logs available for this run.")
        return

    df = pd.DataFrame(logs)
    total_items = len(df)

    # 1. Routing KPIs
    st.markdown("### Routing Efficiency & Oracle Regret KPIs")
    c1, c2, c3, c4 = st.columns(4)

    # Regret
    regret_items = df[df["cost_regret"].notna()]
    mean_regret = regret_items["cost_regret"].mean() if not regret_items.empty else None
    with c1:
        regret_str = f"${mean_regret:.6f}" if mean_regret is not None else "N/A"
        metric_card(
            "Routing Cost Regret",
            regret_str,
            "calculated",
            subtitle=f"Denominator: {len(regret_items)}/{total_items} items with oracle",
        )

    # Over-escalation
    over_esc_count = df["is_over_escalation"].sum() if "is_over_escalation" in df else 0
    with c2:
        over_pct = (over_esc_count / total_items) * 100.0 if total_items > 0 else 0.0
        metric_card(
            "Over-Escalation Rate",
            f"{over_pct:.1f}%",
            "calculated",
            subtitle=f"{over_esc_count}/{total_items} items routed higher than needed",
        )

    # Under-escalation
    under_esc_count = df["is_under_escalation"].sum() if "is_under_escalation" in df else 0
    with c3:
        under_pct = (under_esc_count / total_items) * 100.0 if total_items > 0 else 0.0
        metric_card(
            "Under-Escalation Rate",
            f"{under_pct:.1f}%",
            "calculated",
            subtitle=f"{under_esc_count}/{total_items} items failed quality due to under-routing",
        )

    # Fallback Rate & Cost
    fb_count = df["fallback_occurred"].sum() if "fallback_occurred" in df else 0
    fb_cost = (
        df[df["fallback_occurred"]]["cost_usd_calculated"].sum()
        if "fallback_occurred" in df
        else 0.0
    )
    with c4:
        fb_pct = (fb_count / total_items) * 100.0 if total_items > 0 else 0.0
        metric_card(
            "Fallback Trigger Rate",
            f"{fb_pct:.1f}%",
            "measured" if manifest.adapter_mode == "real" else "simulated",
            subtitle=f"Cost of fallback: ${fb_cost:.5f}",
        )

    st.markdown("---")

    # 2. Route Distribution & Confusion Matrix
    col_dist, col_matrix = st.columns([1, 1])

    with col_dist:
        st.markdown("#### Route Tier Distribution")
        st.caption("Proportion of requests routed to each tier [MEASURED/SIMULATED]")
        tier_counts = df["routed_tier"].value_counts().reset_index()
        tier_counts.columns = ["routed_tier", "count"]

        chart_dist = (
            alt.Chart(tier_counts)
            .mark_bar(color="#0D9488")
            .encode(
                x=alt.X("count:Q", title="Number of Requests"),
                y=alt.Y("routed_tier:N", title="Route Tier", sort="-x"),
                tooltip=["routed_tier", "count"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_dist, use_container_width=True)

    with col_matrix:
        st.markdown("#### Route Selection vs. Oracle Reference")
        st.caption("Comparison between router choice and cheapest eligible route meeting Q_min")

        valid_oracle = df[df["oracle_tier"].notna()]
        if not valid_oracle.empty:
            crosstab = (
                pd.crosstab(valid_oracle["routed_tier"], valid_oracle["oracle_tier"])
                .reset_index()
                .melt(id_vars="routed_tier", var_name="oracle_tier", value_name="count")
            )

            chart_matrix = (
                alt.Chart(crosstab)
                .mark_rect()
                .encode(
                    x=alt.X("oracle_tier:N", title="Oracle Optimal Tier"),
                    y=alt.Y("routed_tier:N", title="InferPath Selected Tier"),
                    color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="blues")),
                    tooltip=["routed_tier", "oracle_tier", "count"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart_matrix, use_container_width=True)
        else:
            st.info("No recorded oracle counterfactual targets available for confusion matrix.")

    st.markdown("---")

    # 3. Category Breakdown Table with Filters
    st.markdown("### Performance Breakdown by Task Category")
    st.caption("Evaluated performance metrics stratified across distinct task families.")

    categories = sorted(df["task_category"].unique().tolist())
    selected_cats = st.multiselect(
        "Filter Categories",
        options=categories,
        default=categories[:5] if len(categories) > 5 else categories,
    )

    df_filtered = df[df["task_category"].isin(selected_cats)] if selected_cats else df

    category_summary = (
        df_filtered.groupby("task_category")
        .agg(
            observations=("item_id", "count"),
            avg_quality=("quality_score", lambda x: f"{x.mean():.3f}"),
            satisfaction_rate=(
                "meets_q_min",
                lambda x: f"{(sum(x) / len(x) * 100):.1f}% ({sum(x)}/{len(x)})",
            ),
            total_cost=("cost_usd_calculated", lambda x: f"${x.sum():.5f}"),
            avg_latency_ms=("latency_ms", lambda x: f"{x.mean():.1f} ms"),
            fallback_rate=(
                "fallback_occurred",
                lambda x: f"{(sum(x) / len(x) * 100):.1f}% ({sum(x)}/{len(x)})",
            ),
        )
        .reset_index()
    )
    category_summary["provenance"] = "[CALCULATED / SIMULATED]"
    st.dataframe(category_summary, use_container_width=True)
