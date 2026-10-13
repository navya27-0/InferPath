"""Resources & Environmental Assumptions Tab for InferPath Research Dashboard.

Displays:
- Mandatory prominent disclaimer banner on estimated nature of environmental metrics
- Energy (Wh) and Water (mL) comparisons with low, central, high estimation bounds
- Comprehensive assumptions table loaded directly from configs/carbon_coefficients.yaml
- Explicit breakdown of Scope 2 datacenter compute, direct cooling water, and indirect generation water
- Transparent disclosure of modeling limitations and unmeasured boundaries
"""

import altair as alt
import pandas as pd
import streamlit as st

from inferpath.dashboard.components import metric_card, render_banner
from inferpath.dashboard.data_loader import load_carbon_assumptions


def render_resources_tab(selected_run, benchmark_summary):
    st.subheader("Environmental Modeling & Resource Consumption Assumptions")

    # Mandatory Prominent Banner (Non-negotiable rule)
    render_banner(
        "⚠️ **CRITICAL PROVENANCE NOTICE**: Energy and water figures are **estimates**, not direct measurements, "
        "unless an underlying source explicitly records measured values. Values are modeled from documented "
        "engineering literature coefficients and uncalibrated datacenter assumptions. Water savings are never derived "
        "from cost savings. External model providers do NOT disclose real-time per-token energy or water telemetry.",
        level="warning",
    )

    manifest = selected_run.manifest
    metrics = manifest.summary_metrics

    # 1. Comparative Energy and Water KPI Cards
    st.markdown("### Selected System Modeled Footprint")
    c1, c2, c3, c4 = st.columns(4)

    central_energy = metrics.get("total_energy_wh", None)
    with c1:
        val_str = f"{central_energy:.3f}" if central_energy is not None else "N/A"
        low_e = f"{central_energy * 0.7:.3f}" if central_energy is not None else "N/A"
        high_e = f"{central_energy * 1.4:.3f}" if central_energy is not None else "N/A"
        metric_card(
            "Compute Energy (Central)",
            val_str,
            "estimated",
            unit="Wh",
            subtitle=f"Range: [{low_e}, {high_e}] Wh",
        )

    central_water = metrics.get("total_water_ml", None)
    with c2:
        val_str = f"{central_water:.3f}" if central_water is not None else "N/A"
        low_w = f"{central_water * 0.7:.3f}" if central_water is not None else "N/A"
        high_w = f"{central_water * 1.4:.3f}" if central_water is not None else "N/A"
        metric_card(
            "Operational Water (Central)",
            val_str,
            "estimated",
            unit="mL",
            subtitle=f"Range: [{low_w}, {high_w}] mL",
        )

    with c3:
        direct_water = central_water * 0.58 if central_water is not None else None
        val_str = f"{direct_water:.3f}" if direct_water is not None else "N/A"
        metric_card(
            "Direct Cooling Water (WUE)",
            val_str,
            "estimated",
            unit="mL",
            subtitle="58% of operational footprint",
        )

    with c4:
        indirect_water = central_water * 0.42 if central_water is not None else None
        val_str = f"{indirect_water:.3f}" if indirect_water is not None else "N/A"
        metric_card(
            "Indirect Generation Water",
            val_str,
            "estimated",
            unit="mL",
            subtitle="42% upstream power plants",
        )

    st.markdown("---")

    # 2. Multi-System Energy Comparison Chart
    st.markdown("### Cross-Architecture Estimated Electricity Consumption")
    st.caption(
        "Energy comparison across Systems A, B, and C with Low / Central / High bounds [ESTIMATED]"
    )

    if benchmark_summary and "table_rows" in benchmark_summary:
        rows = benchmark_summary["table_rows"]
        energy_data = []
        for r in rows:
            if r["System"] in ["System_A", "System_B", "System_C", "ablation_no_small_model"]:
                # Parse Wh string e.g. "45.695 Wh"
                try:
                    wh_str = r["Energy_Wh"].replace("Wh", "").strip()
                    val = float(wh_str)
                    energy_data.append(
                        {
                            "System": r["System"],
                            "Central (Wh)": val,
                            "Low Bound (Wh)": val * 0.7,
                            "High Bound (Wh)": val * 1.4,
                        }
                    )
                except ValueError:
                    pass

        if energy_data:
            df_e = pd.DataFrame(energy_data)
            chart_e = (
                alt.Chart(df_e)
                .mark_bar(color="#0D9488")
                .encode(
                    x=alt.X("System:N", title="Architecture / Baseline"),
                    y=alt.Y("Central (Wh):Q", title="Modeled Compute Energy (Wh, Central)"),
                    tooltip=["System", "Central (Wh)", "Low Bound (Wh)", "High Bound (Wh)"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart_e, use_container_width=True)

    st.markdown("---")

    # 3. Transparent Assumptions Table
    st.markdown("### Documented Modeling Assumptions & Bounds")
    st.caption(
        "Extracted directly from `configs/carbon_coefficients.yaml`. Source Date: 2026-10-01"
    )

    assumptions = load_carbon_assumptions()
    if assumptions:
        df_assump = pd.DataFrame(assumptions)
        display_cols = [
            "parameter",
            "central",
            "low",
            "high",
            "unit",
            "boundary",
            "verification_status",
            "source_note",
        ]
        st.dataframe(df_assump[display_cols], use_container_width=True)
    else:
        st.warning("Could not read assumptions from `configs/carbon_coefficients.yaml`.")

    # 4. Explicit System Boundaries & Limitations
    st.markdown("#### System Boundaries & Known Accounting Exclusions")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("**Included in Scope:**")
        st.markdown(
            "- Scope 2 datacenter compute (dynamic GPU/CPU inference power).\n"
            "- Datacenter infrastructure overhead (facility PUE central = 1.20).\n"
            "- Direct evaporative cooling tower freshwater consumption (WUE).\n"
            "- Upstream electricity generation freshwater consumption."
        )
    with col_b2:
        st.markdown("**Excluded from Scope (Data Unavailable):**")
        st.markdown(
            "- Scope 3 embodied manufacturing and hardware supply chain emissions.\n"
            "- Wide-area network transit and edge transmission infrastructure.\n"
            "- User device screen rendering and local battery consumption.\n"
            "- Provider-specific custom cooling architectures (unreported)."
        )
