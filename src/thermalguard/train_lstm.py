"""PyTorch LSTM for RUL on sliding windows of sensor history, tracked in MLflow, exported to ONNX.

Usage:  python -m thermalguard.train_lstm [--subsets FD001] [--epochs 40]
"""
from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from thermalguard import config
from thermalguard.data import evaluate, load_features, train_val_split

SEQ_LEN = 30
SENSORS = [f"sensor_{i}" for i in config.INFORMATIVE_SENSORS]


class RULLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):  # x: (batch, seq, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_windows(df: pd.DataFrame, lo: np.ndarray, hi: np.ndarray, last_only: bool = False):
    """Return (X, y): one window per cycle (or only the last), front-padded with the first row."""
    X, y = [], []
    for _, g in df.groupby("asset_id", sort=False):
        v = ((g[SENSORS].to_numpy() - lo) / (hi - lo + 1e-9)).astype(np.float32)
        if len(v) < SEQ_LEN:
            v = np.vstack([np.repeat(v[:1], SEQ_LEN - len(v), axis=0), v])
        target = g["rul"].to_numpy() if last_only else g["rul_capped"].to_numpy()
        target = np.concatenate([np.full(len(v) - len(target), target[0]), target])
        ends = [len(v)] if last_only else range(SEQ_LEN, len(v) + 1)
        for e in ends:
            X.append(v[e - SEQ_LEN:e])
            y.append(target[e - 1])
    return np.stack(X), np.asarray(y, dtype=np.float32)


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD001"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--source", default=config.FEATURE_SOURCE, choices=["lakehouse", "dbt"])
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_features(args.subsets, source=args.source)
    train_df, val_df = train_val_split(df[df.split == "train"])
    test_df = df[df.split == "test"]
    lo, hi = train_df[SENSORS].min().to_numpy(), train_df[SENSORS].max().to_numpy()

    X_tr, y_tr = make_windows(train_df, lo, hi)
    X_va, y_va = make_windows(val_df, lo, hi)
    X_te, y_te = make_windows(test_df, lo, hi, last_only=True)
    print(f"windows train={len(X_tr):,} val={len(X_va):,} test_engines={len(X_te)}")

    model = RULLSTM(len(SENSORS))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=256, shuffle=True)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("thermalguard-rul")
    with mlflow.start_run(run_name=f"lstm-{'-'.join(args.subsets)}"):
        mlflow.set_tags({"model_family": "lstm", "subsets": ",".join(args.subsets)})
        mlflow.log_params({"seq_len": SEQ_LEN, "hidden": 64, "layers": 2, "dropout": 0.2, "lr": args.lr,
                           "epochs": args.epochs, "rul_cap": config.RUL_CAP, "n_features": len(SENSORS)})
        best, best_state, patience = float("inf"), None, 0
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            val_rmse = evaluate(y_va, predict(model, X_va))["rmse"]
            sched.step(val_rmse)
            mlflow.log_metric("val_rmse_epoch", val_rmse, step=epoch)
            print(f"epoch {epoch:02d} val_rmse={val_rmse:.2f}")
            if val_rmse < best - 0.05:
                best, best_state, patience = val_rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 8:
                    break
        model.load_state_dict(best_state)

        val_m = evaluate(y_va, predict(model, X_va))
        test_m = evaluate(y_te, predict(model, X_te))
        mlflow.log_metrics({**{f"val_{k}": v for k, v in val_m.items()},
                            **{f"test_{k}": v for k, v in test_m.items()}})

        # Serving artefacts: ONNX graph + normalisation params (onnxruntime only in Lambda, no torch).
        out = config.ARTIFACTS_DIR / "lstm"
        out.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model.eval(), torch.from_numpy(X_te[:1]), out / "model.onnx",
                          input_names=["window"], output_names=["rul"],
                          dynamic_axes={"window": {0: "batch"}, "rul": {0: "batch"}}, dynamo=False)
        (out / "preprocess.json").write_text(json.dumps(
            {"sensors": SENSORS, "seq_len": SEQ_LEN, "min": lo.tolist(), "max": hi.tolist(),
             "rul_cap": config.RUL_CAP}, indent=2))
        torch.save(model.state_dict(), out / "state_dict.pt.data")
        print(f"lstm val_rmse={val_m['rmse']:.2f} test_rmse={test_m['rmse']:.2f} nasa={test_m['nasa_score']:.0f}")
        config.REPORTS_DIR.mkdir(exist_ok=True)
        (config.REPORTS_DIR / "metrics_lstm.json").write_text(
            json.dumps({"lstm": {"val": val_m, "test": test_m}}, indent=2))

        mlflow.log_artifacts(str(out), artifact_path="serving")
        mlflow.pytorch.log_model(model, name="model", registered_model_name="rul-lstm",
                                 input_example=X_te[:1])


if __name__ == "__main__":
    main()
