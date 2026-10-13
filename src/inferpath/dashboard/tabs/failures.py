"""Failures & Edge-Case Analysis Tab for InferPath Research Dashboard.

Displays:
- Ranked worst 20 failure cases per selected system
- Filterable by Category, Difficulty Tier, Routed Tier, and Failure Classification
- Safe redaction of private or sensitive prompt text (zero privacy leakage)
- Detailed inspection cards with rule IDs, fallback status, and quality scores
- Clear reporting of available failure count without synthetic fabrication
"""

import pandas as pd
import streamlit as st

from inferpath.dashboard.components import render_banner
from inferpath.dashboard.data_loader import get_ranked_failures


def render_failures_tab(selected_run):
    st.subheader("Failure Analysis & Worst-Case Triage")

    manifest = selected_run.manifest
    logs = selected_run.logs

    if not logs:
        st.warning("No item-level execution logs available for failure inspection.")
        return

    # Extract ranked failures using data_loader
    failures = get_ranked_failures(logs, q_min=manifest.q_min, max_count=20)
    total_failures_available = len(failures)

    if total_failures_available == 0:
        render_banner(
            f"✓ ZERO FAILURES RECORDED: All {len(logs)} items in run `{manifest.run_id}` satisfied the quality threshold (Q >= {manifest.q_min}) and passed all deterministic validators.",
            level="info",
        )
        return

    st.caption(
        f"Showing top {total_failures_available} ranked edge cases from `{manifest.run_id}` "
        f"(Ranked by quality score ascending; maximum 20 cases displayed without artificial padding)."
    )

    df_fail = pd.DataFrame(failures)

    # 1. Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        cat_options = ["All"] + sorted(df_fail["task_category"].unique().tolist())
        selected_cat = st.selectbox("Filter by Category", cat_options)
    with col_f2:
        tier_options = ["All"] + sorted(df_fail["tier"].unique().tolist())
        selected_tier = st.selectbox("Filter by Difficulty Tier", tier_options)
    with col_f3:
        class_options = ["All"] + sorted(df_fail["failure_classification"].unique().tolist())
        selected_class = st.selectbox("Filter by Failure Type", class_options)

    # Apply filters
    filtered = df_fail.copy()
    if selected_cat != "All":
        filtered = filtered[filtered["task_category"] == selected_cat]
    if selected_tier != "All":
        filtered = filtered[filtered["tier"] == selected_tier]
    if selected_class != "All":
        filtered = filtered[filtered["failure_classification"] == selected_class]

    st.markdown(f"**Displaying {len(filtered)} matching failure cases:**")

    # 2. Summary Table
    table_cols = [
        "item_id",
        "task_category",
        "tier",
        "prompt_preview",
        "routed_tier",
        "rule_id",
        "quality_score",
        "validation_passed",
        "fallback_occurred",
        "failure_classification",
    ]
    st.dataframe(filtered[table_cols], use_container_width=True)

    st.markdown("---")

    # 3. Individual Detailed Case Cards
    st.markdown("### Deep-Dive Case Inspector")
    for idx, case in filtered.iterrows():
        with st.expander(
            f"#{idx + 1} | [{case['item_id']}] {case['task_category']} (Q = {case['quality_score']:.2f}) - {case['failure_classification']}"
        ):
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.markdown(f"**Sanitized Prompt Preview:** `{case['prompt_preview']}`")
                st.markdown(f"**Failure Classification:** `{case['failure_classification']}`")
                st.markdown(
                    f"**Selected Route Tier:** `{case['routed_tier']}` (Rule: `{case['rule_id']}`)"
                )
                st.markdown(f"**Oracle Reference Tier:** `{case['oracle_tier']}`")
            with c_right:
                st.markdown(f"**Quality Score:** `{case['quality_score']:.4f}`")
                st.markdown(f"**Validation Passed:** `{case['validation_passed']}`")
                st.markdown(f"**Fallback Occurred:** `{case['fallback_occurred']}`")
                st.markdown(f"**Item Cost (USD):** `${case['cost_usd']:.6f}`")
                st.markdown(f"**Latency:** `{case['latency_ms']:.1f} ms`")
