"""Load raw NASA C-MAPSS text files into the RAW schema of the warehouse.

Usage:  python scripts/ingest.py [--target duckdb|snowflake] [--subsets FD001 FD002 ...]
Each load gets a batch_id + loaded_at for lineage; dbt tests downstream decide if the batch is usable.
"""
from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone

import pandas as pd

from thermalguard import config
from thermalguard.warehouse import write_table


def read_split(subset: str, split: str) -> pd.DataFrame:
    path = config.RAW_DIR / f"{split}_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=config.RAW_COLS, engine="python")
    df.insert(0, "subset", subset)
    return df


def read_rul(subset: str) -> pd.DataFrame:
    rul = pd.read_csv(config.RAW_DIR / f"RUL_{subset}.txt", header=None, names=["rul_true"])
    rul.insert(0, "unit_id", range(1, len(rul) + 1))
    rul.insert(0, "subset", subset)
    return rul


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=config.TARGET, choices=["duckdb", "snowflake"])
    ap.add_argument("--subsets", nargs="+", default=config.SUBSETS)
    args = ap.parse_args()

    batch_id = uuid.uuid4().hex[:12]
    loaded_at = datetime.now(timezone.utc).replace(tzinfo=None)

    frames = {
        "cmapss_train": pd.concat([read_split(s, "train") for s in args.subsets], ignore_index=True),
        "cmapss_test": pd.concat([read_split(s, "test") for s in args.subsets], ignore_index=True),
        "cmapss_rul": pd.concat([read_rul(s) for s in args.subsets], ignore_index=True),
    }
    for name, df in frames.items():
        df["batch_id"] = batch_id
        df["loaded_at"] = loaded_at
        write_table(df, "raw", name, target=args.target)
        print(f"[{args.target}] raw.{name}: {len(df):,} rows")

    n_cycles = len(frames["cmapss_train"]) + len(frames["cmapss_test"])
    n_assets = frames["cmapss_train"].groupby("subset").unit_id.nunique().sum() + \
        frames["cmapss_test"].groupby("subset").unit_id.nunique().sum()
    print(f"batch {batch_id}: {n_cycles:,} cycles x {len(config.SENSOR_COLS)} sensors = "
          f"{n_cycles * len(config.SENSOR_COLS):,} sensor readings from {n_assets:,} assets")


if __name__ == "__main__":
    main()
