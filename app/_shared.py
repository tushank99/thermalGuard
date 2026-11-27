"""Shared loaders + chart styling for the ThermalGuard Streamlit app."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts"

# Streamlit Community Cloud: secrets (e.g. GROQ_API_KEY) -> env vars, so the package reads them like .env locally.
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:  # no secrets file locally
    pass

# Palette: warm paper + ink, one teal accent. Categorical order is fixed (validated for colour-blind separation);
# status colours are reserved for health states only.
SERIES = ["#009683", "#C4672B", "#7A4CB0", "#8A8F2A"]
STATUS_COLOR = {"healthy": "#3E8E5E", "warning": "#C98A12", "critical": "#B8322A"}
STATUS_LABEL = {"healthy": "● Healthy", "warning": "● Warning", "critical": "● Critical"}
INK, MUTED, GRID, AXIS, SURFACE, ACCENT = "#1E1D1A", "#7A756A", "#E3DDD1", "#CFC8B9", "#F4F1EA", "#0B6E64"

CSS = """<style>
[data-testid="stDecoration"]{display:none}
.block-container{padding-top:2.4rem;max-width:1280px}
[data-testid="stHeading"] h3{font-size:1.05rem;font-weight:600;letter-spacing:-.005em;margin-top:.4rem}
.tg-eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#0B6E64;font-weight:600}
.tg-title{font-size:2.05rem;font-weight:600;line-height:1.15;letter-spacing:-.02em;margin:.15rem 0 .4rem;color:#1E1D1A}
.tg-sub{color:#6B675E;font-size:.95rem;line-height:1.5;max-width:780px;margin-bottom:1.5rem}
.tg-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:0 0 1.6rem}
.tg-kpi{background:#FBF9F4;border:1px solid #DDD6C8;border-radius:12px;padding:14px 16px}
.tg-kpi .l{font-size:.78rem;color:#6B675E;display:flex;align-items:center;gap:7px}
.tg-kpi .v{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:1.75rem;font-weight:500;margin-top:6px;
  color:#1E1D1A;letter-spacing:-.02em}
.tg-kpi .s{font-size:.74rem;color:#8A857A;margin-top:2px}
.tg-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.tg-pill{display:inline-flex;align-items:center;gap:6px;padding:3px 11px;border-radius:999px;font-size:.8rem;
  font-weight:600;background:#FBF9F4;border:1px solid #DDD6C8}
.tg-foot{color:#8A857A;font-size:.78rem;margin-top:2.2rem;border-top:1px solid #DDD6C8;padding-top:.8rem}
.tg-side{font-size:.75rem;color:#A9A396;line-height:1.5;margin-top:.5rem}
</style>"""

SENSOR_NAMES = {
    1: "T2 fan inlet temp", 2: "T24 LPC outlet temp", 3: "T30 HPC outlet temp", 4: "T50 LPT outlet temp",
    5: "P2 fan inlet pressure", 6: "P15 bypass duct pressure", 7: "P30 HPC outlet pressure", 8: "Nf fan speed",
    9: "Nc core speed", 10: "epr pressure ratio", 11: "Ps30 HPC static pressure", 12: "phi fuel flow / Ps30",
    13: "NRf corrected fan speed", 14: "NRc corrected core speed", 15: "BPR bypass ratio",
    16: "farB fuel-air ratio", 17: "htBleed bleed enthalpy", 18: "Nf_dmd", 19: "PCNfR_dmd",
    20: "W31 HPT coolant bleed", 21: "W32 LPT coolant bleed",
}
INFORMATIVE = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


def sensor_label(i: int) -> str:
    return f"sensor_{i} · {SENSOR_NAMES[i]}"


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="tg-eyebrow">{eyebrow}</div><div class="tg-title">{title}</div>'
                f'<div class="tg-sub">{subtitle}</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="tg-side">ThermalGuard · predictive maintenance<br>'
                        'Data: NASA C-MAPSS turbofan simulation</div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple]) -> None:
    """items: (label, value, sub-text, dot colour or None)."""
    cards = "".join(
        f'<div class="tg-kpi"><div class="l">'
        + (f'<span class="tg-dot" style="background:{dot}"></span>' if dot else "")
        + f'{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'
        for label, value, sub, dot in items)
    st.markdown(f'<div class="tg-kpis">{cards}</div>', unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="tg-pill"><span class="tg-dot" style="background:{color}"></span>{text}</span>'


def style_status(df: pd.DataFrame, col: str = "Status"):
    """Colour the status column text by health state (table keeps its own column config)."""
    lookup = {v: STATUS_COLOR[k] for k, v in STATUS_LABEL.items()}
    return df.style.map(lambda v: f"color:{lookup.get(v, INK)};font-weight:600", subset=[col])


def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=40, b=8),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, font=dict(color=MUTED)),
        font=dict(family="Instrument Sans, sans-serif", size=13, color=INK),
        title_font=dict(size=14, color=INK),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color=SURFACE, family="Instrument Sans")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=2))
    return fig


def need(path: Path, how: str) -> bool:
    if not path.exists():
        st.warning(f"Missing `{path.relative_to(ROOT)}`. Run: `{how}`")
        return False
    return True


@st.cache_data(show_spinner=False)
def fleet_status() -> pd.DataFrame:
    return pd.read_parquet(ART / "fleet/status.parquet")


@st.cache_data(show_spinner=False)
def trajectory() -> pd.DataFrame:
    return pd.read_parquet(ART / "fleet/trajectory.parquet")


SENSORS_PATH = ART / "dashboard/sensors.parquet"


@st.cache_data(show_spinner="Loading sensor data…")
def sensors(subset: str) -> pd.DataFrame:
    """Slim export of gold (scripts/export_dashboard_data.py): raw informative sensors per cycle."""
    return pd.read_parquet(SENSORS_PATH, filters=[("subset", "==", subset)])


def add_rolling(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    """Rolling mean/std per engine; same definition as the gold table (trailing window, std of 1 row = 0)."""
    g = df.groupby("asset_id")
    out = df.copy()
    for c in cols:
        out[f"{c}_mean_{window}"] = g[c].transform(lambda s: s.rolling(window, min_periods=1).mean())
        out[f"{c}_std_{window}"] = g[c].transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0.0)
    return out


@st.cache_data(show_spinner=False)
def report(name: str) -> dict | None:
    p = ROOT / "reports" / name
    return json.loads(p.read_text()) if p.exists() else None
