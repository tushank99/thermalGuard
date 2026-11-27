"""Export a slim copy of the gold table for the hosted dashboard (Streamlit Community Cloud).

The full Delta gold table is 100+ MB; the explorer only needs the 14 informative raw sensors per cycle and
recomputes rolling mean/std for the engines a user selects (same definition as gold).
Usage:  python scripts/export_dashboard_data.py   ->  artifacts/dashboard/sensors.parquet
"""
from __future__ import annotations

from thermalguard import config
from thermalguard.data import load_gold

OUT = config.ARTIFACTS_DIR / "dashboard" / "sensors.parquet"
SENSORS = [f"sensor_{i}" for i in config.INFORMATIVE_SENSORS]


def main() -> None:
    df = load_gold(["FD001", "FD002", "FD003", "FD004"])[["asset_id", "subset", "split", "cycle", "rul", *SENSORS]]
    df = df.astype({s: "float32" for s in SENSORS} | {"cycle": "int16", "rul": "int16"})
    df = df.sort_values(["asset_id", "cycle"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"wrote {OUT} ({len(df):,} rows, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
