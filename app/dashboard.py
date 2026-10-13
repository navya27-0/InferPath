"""InferPath Research Dashboard and Live Demo.

Streamlit application providing empirical visualization of evaluation benchmarks,
run manifests, exact-response caching, environmental modeling, failure analysis,
and an interactive live request demo connected to FastAPI.
"""

import streamlit as st

from inferpath.dashboard.components import inject_custom_css
from inferpath.dashboard.data_loader import (
    discover_manifests,
    load_benchmark_summary,
    load_run_data,
)
from inferpath.dashboard.tabs.cache import render_cache_tab
from inferpath.dashboard.tabs.failures import render_failures_tab
from inferpath.dashboard.tabs.live_demo import render_live_demo_tab
from inferpath.dashboard.tabs.overview import render_overview_tab
from inferpath.dashboard.tabs.resources import render_resources_tab
from inferpath.dashboard.tabs.routing import render_routing_tab

# Page setup
st.set_page_config(
    page_title="InferPath — Research Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


def main():
    st.title("🧭 InferPath — Research Evaluation & Live Demo")
    st.caption(
        "Empirical evaluation platform for multi-tier resource-aware inference routing: "
        "(A) Deterministic Rules, (B) Templates, (C) Small Model, (D) Large Model, with Exact Caching and Fallback."
    )

    # Sidebar: Artifact Discovery & Run Selector
    st.sidebar.title("🔬 Run Selector & Filters")

    # Refresh control
    if st.sidebar.button("🔄 Refresh Artifacts", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    manifests = discover_manifests()

    if not manifests:
        st.error(
            "No evaluation manifests found in `data/results/`. Run `make eval` to generate benchmark data."
        )
        st.stop()

    # Mode Filter
    mode_options = ["All", "simulated", "real"]
    selected_mode = st.sidebar.selectbox("Filter Execution Mode", mode_options)

    # Split Filter
    split_options = ["All", "dev", "test"]
    selected_split = st.sidebar.selectbox("Filter Split", split_options)

    # Filter manifests
    filtered_manifests = manifests
    if selected_mode != "All":
        filtered_manifests = [m for m in filtered_manifests if m.adapter_mode == selected_mode]
    if selected_split != "All":
        filtered_manifests = [m for m in filtered_manifests if m.split == selected_split]

    if not filtered_manifests:
        if selected_mode == "real":
            st.sidebar.warning(
                "⚠️ No real-provider (live API) runs found. All current benchmarks are simulated."
            )
        else:
            st.sidebar.warning("No runs matching the selected filters.")
        filtered_manifests = manifests

    # Manifest Selection Box
    manifest_display_names = [
        f"[{m.adapter_mode.upper()} - {m.split.upper()}] {m.system_id} (Run #{m.run_index})"
        for m in filtered_manifests
    ]
    selected_idx = st.sidebar.selectbox(
        "Select Active Manifest / Run",
        range(len(filtered_manifests)),
        format_func=lambda i: manifest_display_names[i],
    )
    selected_manifest = filtered_manifests[selected_idx]

    # Display Manifest Metadata in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Manifest Provenance")
    st.sidebar.markdown(f"**Run ID**: `{selected_manifest.run_id}`")
    st.sidebar.markdown(f"**System**: `{selected_manifest.system_id}`")
    st.sidebar.markdown(f"**Adapter Mode**: `{selected_manifest.adapter_mode.upper()}`")
    st.sidebar.markdown(f"**Dataset Split**: `{selected_manifest.split.upper()}`")
    if selected_manifest.split == "dev":
        st.sidebar.caption("📌 *Development partition (used for calibration; not held-out test)*")
    else:
        st.sidebar.caption("🔒 *Held-out test partition (registered test lock)*")

    st.sidebar.markdown(
        f"**Config Hash**: `{selected_manifest.config_hash[:12]}...`"
        if selected_manifest.config_hash
        else "**Config Hash**: `N/A`"
    )
    st.sidebar.markdown(
        f"**Git Commit**: `{selected_manifest.git_commit[:8]}...`"
        if selected_manifest.git_commit
        else "**Git Commit**: `N/A`"
    )
    st.sidebar.markdown(f"**Price Table**: `{selected_manifest.price_table_date}`")

    if selected_manifest.integrity_warning:
        st.sidebar.warning(f"⚠️ {selected_manifest.integrity_warning}")

    # Load active RunData & Benchmark Summary
    selected_run = load_run_data(selected_manifest)
    benchmark_summary = load_benchmark_summary()

    # Main Tabs
    tab_overview, tab_routing, tab_cache, tab_resources, tab_failures, tab_demo = st.tabs(
        [
            "📊 Overview",
            "🔀 Routing & Oracle",
            "⚡ Exact Cache",
            "🌿 Resources & Eco Assumptions",
            "⚠️ Failure Analysis",
            "🚀 Live Demo & Counterfactual",
        ]
    )

    with tab_overview:
        render_overview_tab(selected_run, benchmark_summary, manifests)

    with tab_routing:
        render_routing_tab(selected_run, benchmark_summary)

    with tab_cache:
        render_cache_tab(selected_run, benchmark_summary)

    with tab_resources:
        render_resources_tab(selected_run, benchmark_summary)

    with tab_failures:
        render_failures_tab(selected_run)

    with tab_demo:
        render_live_demo_tab()


if __name__ == "__main__":
    main()
