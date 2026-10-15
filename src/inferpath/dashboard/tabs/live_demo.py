"""Live Interactive Demo & Counterfactual Inspector for InferPath Research Dashboard.

Calls the existing FastAPI `POST /v1/respond` endpoint (never reimplements pipeline logic).
Displays:
- Full output text, selected tier, rule ID, and step-by-step route trace
- End-to-end measured latency, calculated cost, and estimated energy/water
- Input/output token counts with strict provenance labeling
- Validation and fallback outcomes
- Always-Large baseline counterfactual projection with dated pricing table (2026-10-01)
- Clear handling for offline API, timeouts, privacy blocks, and budget caps
"""

import os

import httpx
import pandas as pd
import streamlit as st

from inferpath.dashboard.components import metric_card, render_banner, render_provenance_badge
from inferpath.dashboard.data_loader import calculate_counterfactual_large_model_cost


def render_live_demo_tab():
    st.subheader("Live Interactive Request Runner & Counterfactual Comparison")

    render_banner(
        "ℹ️ **INTERACTIVE DEMO NOTICE**: This interactive runner sends live requests to the active FastAPI "
        "service at `/v1/respond`. It serves as a real-time behavioral demonstration and is separate from the "
        "frozen, multi-run research benchmark evaluation.",
        level="info",
    )

    # API Configuration
    default_url = os.environ.get("INFERPATH_API_URL", "http://127.0.0.1:8000")
    col_api_cfg, col_mode_cfg = st.columns([3, 1])
    with col_api_cfg:
        api_url = st.text_input("FastAPI Service Base URL", value=default_url)
    with col_mode_cfg:
        privacy_choice = st.selectbox(
            "Privacy Classification",
            ["public", "internal", "confidential_private"],
            help="Confidential and Internal requests trigger strict local routing or privacy interception.",
        )

    # Prompt Inputs and Sample Presets
    st.markdown("#### Input Prompt")
    preset = st.selectbox(
        "Load Example Prompt (Optional)",
        [
            "Custom Prompt",
            "Deterministic Rule A: Convert 100 meters to feet",
            "Deterministic Rule A: Calculate: (15 * 8) + (240 / 6)",
            "Template Rule B: Draft a leave application for Alice for annual leave from 2026-11-01 to 2026-11-05",
            "Small Model Tier C: Summarize the key benefits of exercise in three bullet points",
            "Large Model Tier D: Provide a multi-step proof that the square root of 2 is irrational",
            "Time-Sensitive (Bypass Cache): What is the current stock price of Apple right now?",
            "Privacy Block: Summarize employee salary and payroll records for employee Bob",
        ],
    )

    default_prompt = "Convert 100 meters to feet"
    if preset != "Custom Prompt":
        default_prompt = preset.split(": ", 1)[-1]

    prompt_text = st.text_area("Prompt Text", value=default_prompt, height=100)

    # Submission
    submit_clicked = st.button("🚀 Route & Execute Request", type="primary")

    if submit_clicked:
        if not prompt_text.strip():
            st.error("Prompt cannot be empty.")
            return

        payload = {
            "prompt": prompt_text.strip(),
            "privacy_class": privacy_choice,
            "task_parameters": {
                "expected_format": "text",
                "language": "en",
            },
        }

        # Support protected deployment authentication
        api_key = os.environ.get("INFERPATH_API_KEY", "").strip()
        headers = {"X-API-Key": api_key} if api_key else {}

        with st.spinner("Dispatching through InferPath pipeline..."):
            try:
                resp = httpx.post(
                    f"{api_url}/v1/respond", json=payload, headers=headers, timeout=12.0
                )
            except httpx.ConnectError:
                st.error(
                    f"⚠️ **FastAPI Server Offline**: Could not connect to InferPath service at `{api_url}`. "
                    "Please start the service in your terminal: `uvicorn inferpath.api.main:app --port 8000` "
                    "or verify the `INFERPATH_API_URL` environment variable."
                )
                return
            except httpx.TimeoutException:
                st.error(
                    f"⏱️ **Request Timeout**: Inference request to `{api_url}` exceeded 12.0s limit."
                )
                return
            except Exception as err:
                st.error(f"❌ **Connection Error**: {err}")
                return

        if resp.status_code == 401:
            err_data = resp.json() if resp.headers.get("content-type") == "application/json" else {}
            render_banner(
                f"🔒 **AUTHENTICATION REQUIRED (HTTP 401)**: {err_data.get('detail', 'Unauthorized')}. "
                "The InferPath API is operating in protected mode. Set `INFERPATH_API_KEY` in your environment.",
                level="warning",
            )
            return

        if resp.status_code == 403:
            err_data = resp.json()
            render_banner(
                f"🛑 **PRIVACY INTERCEPTION (HTTP 403)**: {err_data.get('detail', 'Privacy policy violation')}. "
                "The pipeline intercepted the confidential payload and blocked transmission to external model tiers.",
                level="warning",
            )
            return

        if resp.status_code == 402:
            err_data = resp.json()
            render_banner(
                f"💰 **BUDGET EXCEEDED (HTTP 402)**: {err_data.get('detail', 'Run budget cap exceeded')}.",
                level="warning",
            )
            return

        if resp.status_code == 422:
            err_data = resp.json() if resp.headers.get("content-type") == "application/json" else {}
            unsatisfied = err_data.get("unsatisfied_constraints", [])
            render_banner(
                f"⚠️ **CONSTRAINTS UNSATISFIABLE (HTTP 422)**: {err_data.get('error', 'Constraints could not be satisfied')}. "
                f"Unsatisfied: {', '.join(unsatisfied) if unsatisfied else 'None qualifying'}. No model was executed.",
                level="warning",
            )
            return

        if resp.status_code != 200:
            st.error(f"Server returned HTTP {resp.status_code}: {resp.text}")
            return

        data = resp.json()

        # Display Live Result
        st.success(f"Execution Completed in {data.get('latency_ms', 0):.1f} ms")

        # 1. KPI Cards
        st.markdown("### Execution Telemetry")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tier_val = data.get("route", "N/A")
            rule_val = data.get("rule_id", "N/A")
            metric_card(
                "Selected Route Tier", tier_val, "measured", subtitle=f"Triggered Rule: {rule_val}"
            )
        with c2:
            lat_val = data.get("latency_ms", 0.0)
            lat_prov = data.get("latency_provenance", "measured")
            metric_card(
                "Measured Latency",
                f"{lat_val:.1f}",
                lat_prov,
                unit="ms",
                subtitle="End-to-end execution",
            )
        with c3:
            cost_val = data.get("cost_usd", 0.0)
            cost_prov = data.get("cost_provenance", "calculated")
            metric_card(
                "Calculated Cost",
                f"${cost_val:.6f}",
                cost_prov,
                unit="USD",
                subtitle="Price table: 2026-10-01",
            )
        with c4:
            hit_val = data.get("cache_hit", False)
            fb_val = data.get("fallback_used", False)
            metric_card(
                "Cache Status",
                "HIT" if hit_val else "MISS",
                "measured",
                subtitle=f"Fallback: {'YES' if fb_val else 'NO'}",
            )

        # 2. Output & Clarification
        st.markdown("### Pipeline Output")
        if data.get("needs_clarification"):
            st.warning("⚠️ **Template Clarification Required**: Missing slots detected.")
            st.code(data.get("clarification_prompt", data.get("text", "")))
        else:
            st.markdown(f"```\n{data.get('text', '')}\n```")

        # 3. Route Trace & Validation Details
        st.markdown("### Route Trace & Decision Path")
        trace = data.get("route_trace", [])
        if trace:
            df_trace = pd.DataFrame(trace)
            st.dataframe(df_trace, use_container_width=True)

        st.markdown("---")

        # 4. Large-Model Baseline Counterfactual Comparison
        st.markdown("### Always-Large Baseline Counterfactual Projection")
        st.caption(
            "Economic comparison against the standard unrouted enterprise baseline (Always-Large Model). "
            "Counterfactual cost is derived from identical prompt token usage and dated price schedules."
        )

        counterfactual = calculate_counterfactual_large_model_cost(
            prompt=prompt_text,
            output_text=data.get("text"),
            actual_input_tokens=data.get("input_tokens"),
            actual_output_tokens=data.get("output_tokens"),
            model_id="anthropic/claude-3.7-sonnet",
        )

        col_cf1, col_cf2 = st.columns(2)
        with col_cf1:
            st.markdown("**InferPath Multi-Tier Route**")
            st.markdown(f"- **Tier Assigned**: `{data.get('route')}`")
            st.markdown(
                f"- **Inference Cost**: `${data.get('cost_usd', 0.0):.6f}` {render_provenance_badge(data.get('cost_provenance', 'calculated'))}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"- **Observed Latency**: `{data.get('latency_ms', 0.0):.1f} ms` {render_provenance_badge(data.get('latency_provenance', 'measured'))}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"- **Input Tokens**: `{data.get('input_tokens', 'N/A')}` | **Output Tokens**: `{data.get('output_tokens', 'N/A')}`"
            )

        with col_cf2:
            st.markdown("**Counterfactual: Always-Large Baseline (`anthropic/claude-3.7-sonnet`)**")
            cf_cost = counterfactual["counterfactual_cost_usd"]
            st.markdown(
                f"- **Projected Cost**: `${cf_cost:.6f}` {render_provenance_badge('calculated')}",
                unsafe_allow_html=True,
            )
            st.markdown("- **Quality**: `Unavailable (Unexecuted Counterfactual)`")
            st.markdown("- **Latency**: `Unavailable (Unexecuted Counterfactual)`")
            st.markdown(
                f"- **Input Tokens**: `{counterfactual['input_tokens']}` ({counterfactual['input_tokens_method']}) | "
                f"**Output Tokens**: `{counterfactual['output_tokens']}` ({counterfactual['output_tokens_method']})"
            )
            st.markdown(f"- **Price Schedule Date**: `{counterfactual['price_table_date']}`")

        # Cost savings computation
        if cf_cost > 0:
            diff_usd = cf_cost - data.get("cost_usd", 0.0)
            reduction_pct = (diff_usd / cf_cost) * 100.0 if diff_usd > 0 else 0.0
            st.info(
                f"💡 **Calculated Cost Reduction**: InferPath saved **${diff_usd:.6f} ({reduction_pct:.1f}%)** "
                f"relative to the Always-Large counterfactual for this request."
            )
