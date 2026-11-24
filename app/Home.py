"""ThermalGuard operator dashboard: fleet overview.   Run: streamlit run app/Home.py"""
import plotly.graph_objects as go
import streamlit as st

from _shared import (ART, MUTED, SERIES, STATUS_COLOR, STATUS_ICON, fleet_status, need, report, style,
                     trajectory)

st.set_page_config(page_title="ThermalGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ ThermalGuard · Fleet health")
st.caption("Remaining useful life (RUL) predicted by the served LSTM model, with unsupervised anomaly alerts. "
           "Fleet = in-service engines (NASA C-MAPSS FD001 test set). Decision support only.")

if not need(ART / "fleet/status.parquet", "python -m thermalguard.fleet"):
    st.stop()

status = fleet_status()
fleet = status[status.split == "test"].copy()
counts = fleet.status.value_counts()
alerts = int(fleet["first_anomaly_cycle"].notna().sum()) if "first_anomaly_cycle" in fleet else 0

c = st.columns(5)
c[0].metric("Engines in service", len(fleet))
c[1].metric("🔴 Critical (RUL < 30)", int(counts.get("critical", 0)))
c[2].metric("🟠 Warning (30–59)", int(counts.get("warning", 0)))
c[3].metric("🟢 Healthy (≥ 60)", int(counts.get("healthy", 0)))
c[4].metric("Anomaly alerts raised", alerts)

left, right = st.columns([3, 2])
with left:
    st.subheader("Maintenance priority")
    view = fleet.assign(Status=fleet.status.map(STATUS_ICON))[
        ["asset_id", "Status", "rul_pred", "last_cycle"] + (["first_anomaly_cycle"] if "first_anomaly_cycle" in fleet else [])]
    st.dataframe(
        view, hide_index=True, width="stretch", height=380,
        column_config={
            "asset_id": "Engine",
            "rul_pred": st.column_config.ProgressColumn("Predicted RUL (cycles)", min_value=0, max_value=125,
                                                        format="%.0f"),
            "last_cycle": st.column_config.NumberColumn("Cycles run"),
            "first_anomaly_cycle": st.column_config.NumberColumn("First anomaly alert (cycle)"),
        })
with right:
    st.subheader("Predicted RUL distribution")
    fig = go.Figure(go.Histogram(x=fleet.rul_pred, xbins=dict(start=0, end=130, size=10),
                                 marker=dict(color=SERIES[0], line=dict(color="white", width=2)),
                                 hovertemplate="RUL %{x}: %{y} engines<extra></extra>"))
    for x, s in [(30, "critical"), (60, "warning")]:
        fig.add_vline(x=x, line=dict(color=STATUS_COLOR[s], width=1.5, dash="dash"),
                      annotation_text=f"{x}", annotation_font_color=MUTED)
    fig.update_layout(xaxis_title="Predicted RUL (cycles)", yaxis_title="Engines", hovermode="closest")
    st.plotly_chart(style(fig, 380, legend=False), width="stretch")

st.subheader("Engine drill-down")
engine = st.selectbox("Engine", fleet.asset_id.tolist(), index=0)
t = trajectory()
t = t[t.asset_id == engine]
row = fleet[fleet.asset_id == engine].iloc[0]
st.markdown(f"**{engine}** · {STATUS_ICON[row.status]} · predicted RUL **{row.rul_pred:.0f}** cycles after "
            f"{int(row.last_cycle)} cycles")
fig = go.Figure()
fig.add_scatter(x=t.cycle, y=t.rul_pred, name="Predicted RUL", line=dict(color=SERIES[0]))
fig.add_scatter(x=t.cycle, y=t.rul_true.clip(upper=125), name="Actual RUL (capped at 125)",
                line=dict(color=SERIES[1], dash="dot"))
for y, s in [(30, "critical"), (60, "warning")]:
    fig.add_hline(y=y, line=dict(color=STATUS_COLOR[s], width=1, dash="dash"))
fig.update_layout(xaxis_title="Operating cycle", yaxis_title="RUL (cycles)")
st.plotly_chart(style(fig), width="stretch")

m = report("metrics_lstm.json")
if m:
    st.caption(f"Model: PyTorch LSTM → ONNX · test RMSE {m['lstm']['test']['rmse']:.1f} cycles "
               f"(see governance/model_cards/rul-lstm.md)")
