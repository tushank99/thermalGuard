"""Fleet status: per-cycle RUL predictions (served ONNX LSTM) + anomaly alerts for every engine.

Writes artifacts/fleet/{trajectory,status}.parquet, used by the Streamlit dashboard and the assistant.
Usage:  python -m thermalguard.fleet [--subsets FD001]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort
import pandas as pd

from thermalguard import config
from thermalguard.data import load_gold

LSTM_DIR = config.ARTIFACTS_DIR / "lstm"
ANOM_DIR = config.ARTIFACTS_DIR / "anomaly"
OUT = config.ARTIFACTS_DIR / "fleet"


def health_status(rul: float) -> str:
    return "critical" if rul < 30 else "warning" if rul < 60 else "healthy"


def predict_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    pre = json.loads((LSTM_DIR / "preprocess.json").read_text())
    sess = ort.InferenceSession(str(LSTM_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    lo, hi, L = np.array(pre["min"]), np.array(pre["max"]), pre["seq_len"]
    out = []
    for asset, g in df.groupby("asset_id", sort=False):
        v = ((g[pre["sensors"]].to_numpy() - lo) / (hi - lo + 1e-9)).astype(np.float32)
        pad = np.vstack([np.repeat(v[:1], L - 1, axis=0), v])
        windows = np.stack([pad[i:i + L] for i in range(len(v))])
        pred = np.clip(sess.run(None, {"window": windows})[0], 0, pre["rul_cap"])
        out.append(pd.DataFrame({"asset_id": asset, "cycle": g.cycle.to_numpy(), "rul_pred": pred.round(1),
                                 "rul_true": g.rul.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD001"])
    args = ap.parse_args()

    df = load_gold(args.subsets)
    traj = predict_trajectories(df)

    if (ANOM_DIR / "scores.parquet").exists():
        sc = pd.read_parquet(ANOM_DIR / "scores.parquet")[["asset_id", "cycle", "iforest_score", "autoencoder_score"]]
        thr = json.loads((ANOM_DIR / "thresholds.json").read_text())
        traj = traj.merge(sc, on=["asset_id", "cycle"], how="left")
        above = (traj["iforest_score"] > thr["iforest"]).astype(int)
        run = above.groupby([traj["asset_id"], (above == 0).cumsum()]).cumsum()
        traj["anomaly_alert"] = run >= thr["persist_cycles"]

    last = traj.sort_values("cycle").groupby("asset_id").tail(1).copy()
    last["status"] = last["rul_pred"].map(health_status)
    if "anomaly_alert" in traj:
        first = traj[traj.anomaly_alert].groupby("asset_id").cycle.min().rename("first_anomaly_cycle")
        last = last.merge(first, on="asset_id", how="left")
    last = last.rename(columns={"cycle": "last_cycle"}).sort_values("rul_pred")
    last["split"] = last.asset_id.str.split("_").str[1]

    OUT.mkdir(parents=True, exist_ok=True)
    traj.to_parquet(OUT / "trajectory.parquet", index=False)
    last.to_parquet(OUT / "status.parquet", index=False)
    print(f"fleet: {last.asset_id.nunique()} engines | status counts: {last.status.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
