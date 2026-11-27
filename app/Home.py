"""ThermalGuard operator dashboard: fleet overview.   Run: streamlit run app/Home.py"""
import plotly.graph_objects as go
import streamlit as st

from _shared import (ART, INK, MUTED, SERIES, STATUS_COLOR, STATUS_LABEL, SURFACE, fleet_status, kpi_row, need,
                     page_header, pill, report, style, style_status, trajectory)

st.set_page_config(page_title="ThermalGuard · Fleet health", page_icon=":material/shield:", layout="wide")
page_header("Fleet overview", "Fleet health",
            "Remaining useful life (RUL) predicted by the LSTM model, with unsupervised anomaly alerts, for the "
            "in-service engines of NASA C-MAPSS FD001. Decision support, not a removal order.")

if not need(ART / "fleet/status.parquet", "python -m thermalguard.fleet"):
    st.stop()

status = fleet_status()
fleet = status[status.split == "test"].copy()
counts = fleet.status.value_counts()
alerts = int(fleet["first_anomaly_cycle"].notna().sum()) if "first_anomaly_cycle" in fleet else 0

kpi_row([
    ("Engines in service", len(fleet), "FD001 test fleet", None),
    ("Critical", int(counts.get("critical", 0)), "RUL under 30 cycles", STATUS_COLOR["critical"]),
    ("Warning", int(counts.get("warning", 0)), "RUL 30 to 59 cycles", STATUS_COLOR["warning"]),
    ("Healthy", int(counts.get("healthy", 0)), "RUL 60 cycles or more", STATUS_COLOR["healthy"]),
    ("Anomaly alerts", alerts, "engines flagged by the detectors", None),
])

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Maintenance priority")
    view = fleet.assign(Status=fleet.status.map(STATUS_LABEL))[
        ["asset_id", "Status", "rul_pred", "last_cycle"] + (["first_anomaly_cycle"] if "first_anomaly_cycle" in fleet else [])]
    st.dataframe(
        style_status(view), hide_index=True, width="stretch", height=390,
        column_config={
            "asset_id": "Engine",
            "rul_pred": st.column_config.ProgressColumn("Predicted RUL", min_value=0, max_value=125, format="%.0f"),
            "last_cycle": st.column_config.NumberColumn("Cycles run"),
            "first_anomaly_cycle": st.column_config.NumberColumn("First alert (cycle)", format="%.0f"),
        })
with right:
    st.subheader("Predicted RUL across the fleet")
    with st.container(border=True):
        fig = go.Figure(go.Histogram(x=fleet.rul_pred, xbins=dict(start=0, end=130, size=10),
                                     marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
                                     hovertemplate="RUL %{x}: %{y} engines<extra></extra>"))
        for x, s in [(30, "critical"), (60, "warning")]:
            fig.add_vline(x=x, line=dict(color=STATUS_COLOR[s], width=1.5, dash="dot"),
                          annotation_text=s, annotation_font_color=STATUS_COLOR[s], annotation_position="top")
        fig.update_layout(xaxis_title="Predicted RUL (cycles)", yaxis_title="Engines", hovermode="closest",
                          bargap=0.06)
        st.plotly_chart(style(fig, 350, legend=False), width="stretch")

st.subheader("Engine drill-down")
engine = st.selectbox("Engine", fleet.asset_id.tolist(), index=0, label_visibility="collapsed")
t = trajectory()
t = t[t.asset_id == engine]
row = fleet[fleet.asset_id == engine].iloc[0]
st.markdown(f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin:.2rem 0 .6rem'>"
            f"{pill(engine, INK)}{pill(STATUS_LABEL[row.status][2:], STATUS_COLOR[row.status])}"
            f"{pill(f'Predicted RUL {row.rul_pred:.0f} cycles', SERIES[0])}"
            f"{pill(f'{int(row.last_cycle)} cycles run', MUTED)}</div>", unsafe_allow_html=True)
with st.container(border=True):
    fig = go.Figure()
    fig.add_scatter(x=t.cycle, y=t.rul_pred, name="Predicted RUL", line=dict(color=SERIES[0]))
    fig.add_scatter(x=t.cycle, y=t.rul_true.clip(upper=125), name="Actual RUL (capped at 125)",
                    line=dict(color=INK, dash="dot", width=1.5))
    for y, s in [(30, "critical"), (60, "warning")]:
        fig.add_hline(y=y, line=dict(color=STATUS_COLOR[s], width=1, dash="dot"))
    fig.update_layout(xaxis_title="Operating cycle", yaxis_title="RUL (cycles)")
    st.plotly_chart(style(fig), width="stretch")

m = report("metrics_lstm.json")
if m:
    st.markdown(f"<div class='tg-foot'>Model: PyTorch LSTM, served as ONNX · test RMSE "
                f"{m['lstm']['test']['rmse']:.1f} cycles · model card: governance/model_cards/rul-lstm.md</div>",
                unsafe_allow_html=True)
