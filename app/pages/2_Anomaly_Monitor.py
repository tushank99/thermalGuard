"""Unsupervised anomaly monitor: Isolation Forest + autoencoder scores per engine against their thresholds."""
import json

import plotly.graph_objects as go
import streamlit as st

from _shared import ART, MUTED, SERIES, STATUS_COLOR, need, report, style, trajectory

st.set_page_config(page_title="Anomaly Monitor · ThermalGuard", page_icon="📈", layout="wide")
st.title("📈 Anomaly monitor")
st.caption("Models trained only on healthy early-life cycles. An alert fires when the score stays above the "
           "threshold (99.5th percentile of healthy scores) for 3 consecutive cycles.")

if not (need(ART / "fleet/trajectory.parquet", "python -m thermalguard.fleet")
        and need(ART / "anomaly/thresholds.json", "python -m thermalguard.anomaly")):
    st.stop()

rep = report("anomaly.json")
if rep:
    st.subheader("How early do alerts come? (held-out run-to-failure engines)")
    cols = st.columns(len(rep["results"]))
    for c, (name, m) in zip(cols, rep["results"].items()):
        c.metric(f"{'Isolation Forest' if name == 'iforest' else 'Autoencoder'} · median lead time",
                 f"{m['median_lead_time_cycles']:.0f} cycles before failure")
        c.caption(f"Detected {m['detection_rate']:.0%} of engines · lead time P10–P90 "
                  f"{m['p10_lead_time_cycles']:.0f}–{m['p90_lead_time_cycles']:.0f} cycles · "
                  f"early-alarm rate {m['early_alarm_rate']:.0%}")

thr = json.loads((ART / "anomaly/thresholds.json").read_text())
t = trajectory()
split = st.radio("Engines", ["test", "train"], horizontal=True,
                 format_func=lambda s: "In service (test)" if s == "test" else "Run-to-failure (train)")
engines = sorted(t[t.asset_id.str.contains(f"_{split}_")].asset_id.unique())
engine = st.selectbox("Engine", engines)
g = t[t.asset_id == engine]

grid = st.columns(2)
for k, (col, name) in enumerate([("iforest_score", "Isolation Forest"), ("autoencoder_score", "Autoencoder")]):
    key = "iforest" if col == "iforest_score" else "autoencoder"
    above = (g[col] > thr[key]).astype(int)
    run = above.groupby((above == 0).cumsum()).cumsum()
    first = g[run >= thr["persist_cycles"]].head(1)
    fig = go.Figure()
    fig.add_scatter(x=g.cycle, y=g[col], name="Anomaly score", line=dict(color=SERIES[k]))
    fig.add_hline(y=thr[key], line=dict(color=MUTED, width=1.5, dash="dash"),
                  annotation_text="threshold", annotation_font_color=MUTED)
    if len(first):
        r = first.iloc[0]
        fig.add_scatter(x=[r.cycle], y=[r[col]], mode="markers", name="First alert",
                        marker=dict(size=11, color=STATUS_COLOR["critical"], line=dict(color="white", width=2)),
                        hovertemplate=f"First alert at cycle {int(r.cycle)}<br>Actual RUL then: {int(r.rul_true)}"
                                      "<extra></extra>")
        msg = f"first alert at cycle **{int(r.cycle)}** (actual RUL then: {int(r.rul_true)} cycles)"
    else:
        msg = "no alert raised"
    fig.update_layout(title=dict(text=name, font=dict(size=14)), xaxis_title="Operating cycle",
                      yaxis_title="Score (higher = more anomalous)")
    with grid[k]:
        st.plotly_chart(style(fig, 340, legend=False), width="stretch")
        st.markdown(f"{name}: {msg}")
