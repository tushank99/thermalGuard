"""Self-serve analytics: explore the lakehouse gold table by engine and sensor without writing SQL."""
import plotly.graph_objects as go
import streamlit as st

from _shared import INFORMATIVE, SENSORS_PATH, SERIES, add_rolling, need, sensor_label, sensors, style

st.set_page_config(page_title="Engine Explorer · ThermalGuard", page_icon="🔎", layout="wide")
st.title("🔎 Engine explorer")
st.caption("Self-serve view of the lakehouse gold table (PySpark + Delta). Pick engines and sensors; no SQL needed.")

if not need(SENSORS_PATH, "python scripts/export_dashboard_data.py"):
    st.stop()

f = st.columns([1, 1, 3, 1.4, 1])
subset = f[0].selectbox("Dataset", ["FD001", "FD002", "FD003", "FD004"])
df = sensors(subset)
split = f[1].selectbox("Split", ["test", "train"], help="train = run to failure; test = in service")
df = df[df.split == split]
engines_all = sorted(df.asset_id.unique())
engines = f[2].multiselect("Engines (up to 4)", engines_all, default=engines_all[:2], max_selections=4)
view = f[3].selectbox("Feature", ["Raw value", "Rolling mean", "Rolling std"])
window = f[4].selectbox("Window", [5, 10, 30], index=1, disabled=view == "Raw value")

sensors = st.multiselect("Sensors (one chart each)", INFORMATIVE, default=[3, 11], format_func=sensor_label,
                         max_selections=4)
if not engines or not sensors:
    st.info("Select at least one engine and one sensor.")
    st.stop()

sel = df[df.asset_id.isin(engines)]
lo, hi = int(sel.cycle.min()), int(sel.cycle.max())
cyc = st.slider("Cycle range", lo, hi, (lo, hi))
if view != "Raw value":  # rolling first, then trim to the cycle range (windows see earlier cycles)
    sel = add_rolling(sel, [f"sensor_{s}" for s in sensors], window)
sel = sel[sel.cycle.between(*cyc)]


def col(i: int) -> str:
    return f"sensor_{i}" if view == "Raw value" else f"sensor_{i}_{'mean' if view == 'Rolling mean' else 'std'}_{window}"


# Colour follows the engine: each selected engine keeps its slot while others are added or removed.
cmap = {e: c for e, c in st.session_state.get("engine_colors", {}).items() if e in engines}
for e in engines:
    if e not in cmap:
        cmap[e] = next(c for c in SERIES if c not in cmap.values())
st.session_state.engine_colors = cmap

grid = st.columns(2)
for k, s in enumerate(sensors):
    fig = go.Figure()
    for e in engines:
        g = sel[sel.asset_id == e]
        fig.add_scatter(x=g.cycle, y=g[col(s)], name=e, line=dict(color=cmap[e]))
    fig.update_layout(title=dict(text=f"{sensor_label(s)} · {view.lower()}"
                                 + ("" if view == "Raw value" else f" (w={window})"), font=dict(size=14)),
                      xaxis_title="Operating cycle")
    grid[k % 2].plotly_chart(style(fig, 320, legend=len(engines) > 1), width="stretch")

cols = ["asset_id", "cycle", "rul", *[col(s) for s in sensors]]
st.subheader("Data")
st.dataframe(sel[cols], hide_index=True, width="stretch", height=260)
c1, c2 = st.columns([1, 3])
c1.download_button("⬇ Download CSV", sel[cols].to_csv(index=False), file_name=f"thermalguard_{subset}_{split}.csv",
                   mime="text/csv")
with c2.expander("Summary statistics"):
    st.dataframe(sel.groupby("asset_id")[[col(s) for s in sensors]].describe().T, width="stretch")
