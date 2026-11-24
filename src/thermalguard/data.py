"""Feature loading, splits and RUL metrics shared by all trainers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from thermalguard import config
from thermalguard.warehouse import read_sql


GOLD_PATH = config.ROOT / "data" / "lakehouse" / "gold" / "rul_features"


def load_gold(subsets: list[str] | None = None) -> pd.DataFrame:
    """Read the lakehouse gold Delta table with delta-rs (no Spark/JVM needed)."""
    from deltalake import DeltaTable

    filters = [("subset", "in", subsets)] if subsets else None
    df = DeltaTable(str(GOLD_PATH)).to_pandas(filters=filters)
    return df.sort_values(["asset_id", "cycle"]).reset_index(drop=True)


def load_features(subsets: list[str], source: str | None = None) -> pd.DataFrame:
    """source='lakehouse' -> gold Delta table (PySpark pipeline); 'dbt' -> warehouse mart (dbt pipeline)."""
    source = source or config.FEATURE_SOURCE
    if source == "lakehouse":
        return load_gold(subsets)
    in_list = ", ".join(f"'{s}'" for s in subsets)
    return read_sql(f"select * from marts.fct_rul_features where subset in ({in_list}) order by asset_id, cycle")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def train_val_split(train_df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Split by asset so no engine leaks across train/val."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["asset_id"]))
    return train_df.iloc[tr_idx], train_df.iloc[va_idx]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions (over-estimating RUL) are penalised harder."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(y_true, y_pred) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, config.RUL_CAP)
    y_true = np.minimum(np.asarray(y_true), config.RUL_CAP)
    return {"rmse": rmse(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred)}
