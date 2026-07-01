"""Theme and global CSS for the medical workstation UI."""

from __future__ import annotations

import streamlit as st

THEMES = {
    "dark": {
        "bg": "#0e1117",
        "panel": "#1a1f2e",
        "panel_border": "#2d3748",
        "text": "#e8eaed",
        "muted": "#9aa0a6",
        "accent": "#4cc9f0",
        "accent2": "#4361ee",
        "success": "#06d6a0",
        "warning": "#ffd166",
        "danger": "#ef476f",
        "card_shadow": "0 4px 24px rgba(0,0,0,0.35)",
    },
    "light": {
        "bg": "#f4f7fb",
        "panel": "#ffffff",
        "panel_border": "#d0d7e2",
        "text": "#1a202c",
        "muted": "#64748b",
        "accent": "#2563eb",
        "accent2": "#0ea5e9",
        "success": "#059669",
        "warning": "#d97706",
        "danger": "#dc2626",
        "card_shadow": "0 4px 20px rgba(15,23,42,0.08)",
    },
}


def get_theme() -> dict:
    mode = st.session_state.get("ui_theme", "dark")
    return THEMES.get(mode, THEMES["dark"])


def apply_theme() -> None:
    t = get_theme()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background: {t['bg']};
            color: {t['text']};
        }}
        section[data-testid="stSidebar"] {{
            background: {t['panel']};
            border-right: 1px solid {t['panel_border']};
        }}
        .cortex-hero {{
            background: linear-gradient(135deg, {t['accent2']}22, {t['accent']}11);
            border: 1px solid {t['panel_border']};
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
        }}
        .cortex-hero h1 {{
            font-size: 2.4rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, {t['accent']}, {t['accent2']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .cortex-card {{
            background: {t['panel']};
            border: 1px solid {t['panel_border']};
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            box-shadow: {t['card_shadow']};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .cortex-card:hover {{
            transform: translateY(-2px);
        }}
        .cortex-metric-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: {t['accent']};
        }}
        .cortex-metric-label {{
            font-size: 0.85rem;
            color: {t['muted']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .cortex-pipeline-step {{
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            margin: 0.25rem 0;
            border-left: 3px solid {t['panel_border']};
        }}
        .cortex-pipeline-active {{
            border-left-color: {t['accent']};
            background: {t['accent']}15;
        }}
        .cortex-pipeline-done {{
            border-left-color: {t['success']};
        }}
        div[data-testid="stMetric"] {{
            background: {t['panel']};
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid {t['panel_border']};
        }}
        .stButton > button {{
            border-radius: 8px;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
