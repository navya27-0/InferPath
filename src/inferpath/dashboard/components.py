"""Reusable UI components and styles for the InferPath research dashboard.

Maintains a restrained, research-grade design aesthetic:
- Navy (#0F172A, #1E293B) for structural anchors
- Teal (#0D9488, #14B8A6) for secondary emphasis
- Gold (#D97706, #F59E0B) for limited highlight
- Consistent provenance badges and source classification
"""

import streamlit as st

# Color palette
COLOR_NAVY_DARK = "#0F172A"
COLOR_NAVY_LIGHT = "#1E293B"
COLOR_TEAL_DARK = "#0D9488"
COLOR_TEAL_LIGHT = "#14B8A6"
COLOR_GOLD = "#D97706"
COLOR_BG_CARD = "#F8FAFC"
COLOR_BORDER = "#E2E8F0"

PROVENANCE_COLORS = {
    "measured": {"bg": "#ECFDF5", "border": "#10B981", "text": "#065F46"},
    "calculated": {"bg": "#EFF6FF", "border": "#3B82F6", "text": "#1E40AF"},
    "estimated": {"bg": "#FEF3C7", "border": "#F59E0B", "text": "#92400E"},
    "simulated": {"bg": "#F1F5F9", "border": "#94A3B8", "text": "#334155"},
}


def inject_custom_css():
    """Injects minimal, accessible CSS styling for research dashboard aesthetics."""
    st.markdown(
        f"""
        <style>
        /* General layout and typography */
        .main .block-container {{
            max-width: 1280px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }}
        h1, h2, h3 {{
            color: {COLOR_NAVY_DARK};
            font-weight: 600;
            letter-spacing: -0.02em;
        }}
        /* Metric Card styling */
        .kpi-card {{
            background: {COLOR_BG_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 6px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }}
        .kpi-label {{
            font-size: 0.82rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {COLOR_NAVY_DARK};
            line-height: 1.2;
        }}
        .kpi-unit {{
            font-size: 0.95rem;
            color: #475569;
            font-weight: 500;
            margin-left: 4px;
        }}
        .kpi-footer {{
            margin-top: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        /* Provenance Pill */
        .prov-pill {{
            display: inline-block;
            font-size: 0.70rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid;
        }}
        /* Callout Banners */
        .banner-warning {{
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            color: #92400E;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 16px;
            font-size: 0.90rem;
        }}
        .banner-info {{
            background-color: #F0FDFA;
            border-left: 4px solid #0D9488;
            color: #115E59;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 16px;
            font-size: 0.90rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_provenance_badge(provenance: str) -> str:
    """Renders an inline HTML provenance badge."""
    p_norm = provenance.lower().strip()
    cfg = PROVENANCE_COLORS.get(p_norm, {"bg": "#F8FAFC", "border": "#CBD5E1", "text": "#475569"})
    return (
        f'<span class="prov-pill" style="background:{cfg["bg"]}; border-color:{cfg["border"]}; color:{cfg["text"]};">'
        f"[{p_norm.upper()}]</span>"
    )


def metric_card(
    label: str,
    value: str,
    provenance: str,
    unit: str = "",
    subtitle: str | None = None,
):
    """Renders a standardized KPI metric card with mandatory provenance classification."""
    badge_html = render_provenance_badge(provenance)
    sub_html = (
        f'<div style="font-size: 0.75rem; color:#64748B;">{subtitle}</div>' if subtitle else ""
    )
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
        <div class="kpi-footer">
            <div>{badge_html}</div>
            {sub_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_banner(text: str, level: str = "warning"):
    """Displays a prominent banner."""
    css_class = "banner-warning" if level == "warning" else "banner-info"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)
