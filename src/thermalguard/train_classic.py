"""Train the scikit-learn baseline (Ridge) and XGBoost RUL models, track in MLflow, register them.

Usage:  python -m thermalguard.train_classic [--subsets FD001]
"""
from __future__ import annotations

import argparse
import json

import mlflow
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thermalguard import config
from thermalguard.data import evaluate, feature_columns, load_features, train_val_split

EXPERIMENT = "thermalguard-rul"


def build_models(seed: int = 42):
    return {
        "ridge": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), {"alpha": 1.0}),
        "xgboost": (
            xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, early_stopping_rounds=50,
                             random_state=seed, n_jobs=-1),
            {"n_estimators": 1000, "learning_rate": 0.03, "max_depth": 5, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 5},
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD001"])
    ap.add_argument("--source", default=config.FEATURE_SOURCE, choices=["lakehouse", "dbt"])
    args = ap.parse_args()

    df = load_features(args.subsets, source=args.source)
    feats = feature_columns(df)
    train_df, val_df = train_val_split(df[df.split == "train"])
    test_df = df[(df.split == "test") & df.is_last_cycle]
    print(f"train={len(train_df):,} val={len(val_df):,} test_engines={len(test_df)} features={len(feats)}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    results = {}

    for name, (model, params) in build_models().items():
        with mlflow.start_run(run_name=f"{name}-{'-'.join(args.subsets)}"):
            mlflow.set_tags({"model_family": name, "subsets": ",".join(args.subsets),
                             "data_source": "delta:gold/rul_features" if args.source == "lakehouse" else f"{config.TARGET}:marts.fct_rul_features"})
            mlflow.log_params({**params, "rul_cap": config.RUL_CAP, "n_features": len(feats)})

            X_tr, y_tr = train_df[feats], train_df.rul_capped
            if name == "xgboost":
                model.fit(X_tr, y_tr, eval_set=[(val_df[feats], val_df.rul_capped)], verbose=False)
                mlflow.log_metric("best_iteration", model.best_iteration)
            else:
                model.fit(X_tr, y_tr)

            val_m = evaluate(val_df.rul_capped, model.predict(val_df[feats]))
            test_m = evaluate(test_df.rul, model.predict(test_df[feats]))
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

            sig = infer_signature(X_tr.head(50), model.predict(X_tr.head(50)))
            flavor = mlflow.xgboost if name == "xgboost" else mlflow.sklearn
            flavor.log_model(model, name="model", signature=sig,
                             registered_model_name=f"rul-{name}", input_example=X_tr.head(3))
            results[name] = {"val": val_m, "test": test_m}
            print(f"{name:8s} val_rmse={val_m['rmse']:.2f} test_rmse={test_m['rmse']:.2f} "
                  f"nasa={test_m['nasa_score']:.0f}")

    base = results["ridge"]["test"]["rmse"]
    for name, r in results.items():
        r["test"]["rmse_reduction_vs_baseline_pct"] = round(100 * (base - r["test"]["rmse"]) / base, 1)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "metrics_classic.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v["test"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
