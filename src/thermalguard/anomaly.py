"""Unsupervised anomaly detection on healthy-cycle sensor data: Isolation Forest + PyTorch autoencoder.

Train ONLY on healthy cycles (RUL > 125, early life) of training engines. Threshold = 99.5th percentile
of the healthy-cycle score. An alert fires when the score stays above threshold for 3 consecutive cycles.
Key metric: lead time = cycles before failure at the first alert, measured on held-out run-to-failure engines.

Usage:  python -m thermalguard.anomaly [--subsets FD001]
"""
from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.models import infer_signature
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn

from thermalguard import config
from thermalguard.data import load_features, train_val_split

FEATURES = [f"sensor_{i}_mean_10" for i in config.INFORMATIVE_SENSORS]  # 10-cycle smoothed sensors
HEALTHY_RUL = 125
THRESHOLD_Q = 0.995
PERSIST = 3
OUT = config.ARTIFACTS_DIR / "anomaly"


class AutoEncoder(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, 8), nn.ReLU(), nn.Linear(8, 3))
        self.dec = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, n))

    def forward(self, x):
        return self.dec(self.enc(x))


def ae_score(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    """Reconstruction error + latent drift.

    Degradation moves correlated sensors together, which an AE can still reconstruct, so reconstruction error
    alone misses it. We add the squared, standardised distance of the latent code from the healthy latent
    centre (statistics stored on the model after training).
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32))
        z = model.enc(x)
        recon = ((model.dec(z) - x) ** 2).mean(dim=1)
        drift = (((z - model.z_mu) / model.z_sd) ** 2).mean(dim=1)
        return (recon + drift).numpy()


def train_ae(X: np.ndarray, epochs: int = 60) -> AutoEncoder:
    torch.manual_seed(42)
    model = AutoEncoder(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.from_numpy(X.astype(np.float32))
    for _ in range(epochs):
        perm = torch.randperm(len(data))
        for i in range(0, len(data), 256):
            xb = data[perm[i:i + 256]]
            opt.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():  # healthy latent centre/spread for the drift term
        z = model.enc(data)
        model.register_buffer("z_mu", z.mean(dim=0))
        model.register_buffer("z_sd", z.std(dim=0) + 1e-6)
    return model


def first_alert(df: pd.DataFrame, score_col: str, thr: float) -> pd.DataFrame:
    """Per engine: first cycle where the score has been above threshold for PERSIST consecutive cycles."""
    rows = []
    for asset, g in df.groupby("asset_id", sort=False):
        above = (g[score_col].to_numpy() > thr).astype(int)
        run = pd.Series(above).groupby((above == 0).cumsum()).cumsum().to_numpy()
        hit = np.flatnonzero(run >= PERSIST)
        if len(hit):
            r = g.iloc[hit[0]]
            rows.append({"asset_id": asset, "alert_cycle": int(r.cycle), "lead_time": int(r.rul)})
        else:
            rows.append({"asset_id": asset, "alert_cycle": None, "lead_time": None})
    return pd.DataFrame(rows)


def lead_time_metrics(alerts: pd.DataFrame) -> dict:
    lt = alerts["lead_time"].dropna()
    return {
        "engines": len(alerts),
        "detection_rate": round(len(lt) / len(alerts), 3),
        "median_lead_time_cycles": float(lt.median()) if len(lt) else None,
        "p10_lead_time_cycles": float(lt.quantile(0.1)) if len(lt) else None,
        "p90_lead_time_cycles": float(lt.quantile(0.9)) if len(lt) else None,
        # Alert while the engine is still in its healthy phase = early/false alarm.
        "early_alarm_rate": round(float((lt > HEALTHY_RUL).sum() / len(alerts)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD001"])
    ap.add_argument("--source", default=config.FEATURE_SOURCE, choices=["lakehouse"])
    args = ap.parse_args()

    df = load_features(args.subsets, source=args.source)
    train_df, val_df = train_val_split(df[df.split == "train"])
    healthy = train_df[train_df.rul > HEALTHY_RUL]
    scaler = StandardScaler().fit(healthy[FEATURES])
    X_h = scaler.transform(healthy[FEATURES])
    print(f"healthy training cycles={len(X_h):,} from {healthy.asset_id.nunique()} engines; "
          f"validation engines={val_df.asset_id.nunique()}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("thermalguard-anomaly")
    OUT.mkdir(parents=True, exist_ok=True)
    results, thresholds = {}, {}

    # ---- Isolation Forest
    iforest = IsolationForest(n_estimators=300, max_samples=512, random_state=42, n_jobs=-1).fit(X_h)
    if_score = lambda X: -iforest.score_samples(X)  # noqa: E731  higher = more anomalous
    # ---- Autoencoder
    ae = train_ae(X_h)

    scorers = {"iforest": if_score, "autoencoder": lambda X: ae_score(ae, X)}
    for name, scorer in scorers.items():
        with mlflow.start_run(run_name=f"{name}-{'-'.join(args.subsets)}"):
            thr = float(np.quantile(scorer(X_h), THRESHOLD_Q))
            v = val_df.copy()
            v["score"] = scorer(scaler.transform(v[FEATURES]))
            m = lead_time_metrics(first_alert(v, "score", thr))
            mlflow.set_tags({"model_family": name, "task": "anomaly_detection"})
            mlflow.log_params({"features": "14 sensors, 10-cycle rolling mean",
                               "ae_score": "recon_mse + latent_drift", "healthy_rul_gt": HEALTHY_RUL,
                               "threshold_quantile": THRESHOLD_Q, "persist_cycles": PERSIST})
            mlflow.log_metrics({"threshold": thr, **{k: v_ for k, v_ in m.items() if v_ is not None}})
            sample = X_h[:5].astype(np.float32)
            if name == "iforest":
                mlflow.sklearn.log_model(iforest, name="model", registered_model_name="anomaly-iforest",
                                         skops_trusted_types=["sklearn.tree._tree.Tree"],  # our own model
                                         signature=infer_signature(sample, if_score(sample)))
            else:
                mlflow.pytorch.log_model(ae, name="model", registered_model_name="anomaly-autoencoder",
                                         input_example=sample)
            results[name], thresholds[name] = m, thr
            print(f"{name:12s} thr={thr:.4f} {m}")

    # ---- Persist scores for the dashboard (all engines of the chosen subsets).
    X_all = scaler.transform(df[FEATURES])
    scores = df[["asset_id", "subset", "split", "cycle", "rul"]].copy()
    scores["iforest_score"] = if_score(X_all)
    scores["autoencoder_score"] = ae_score(ae, X_all)
    scores.to_parquet(OUT / "scores.parquet", index=False)
    joblib.dump({"scaler": scaler, "iforest": iforest, "features": FEATURES}, OUT / "iforest.joblib")
    torch.save(ae.state_dict(), OUT / "autoencoder.pt")
    (OUT / "thresholds.json").write_text(json.dumps({"persist_cycles": PERSIST, **thresholds}, indent=2))

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "anomaly.json").write_text(json.dumps(
        {"definition": f"first cycle with score > q{THRESHOLD_Q} of healthy scores for {PERSIST} consecutive "
                       f"cycles; lead time = RUL at that cycle; held-out run-to-failure engines",
         "results": results, "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()
