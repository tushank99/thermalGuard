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

# Reference palette (dataviz skill): categorical in fixed order, status colours reserved.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS_COLOR = {"healthy": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
STATUS_ICON = {"healthy": "🟢 Healthy", "warning": "🟠 Warning", "critical": "🔴 Critical"}
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"

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


def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        font=dict(size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED))
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
