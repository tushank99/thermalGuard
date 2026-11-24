"""Central config: paths, column names, modelling constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw" / "cmapss"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
DBT_DIR = ROOT / "dbt"

TARGET = os.getenv("TG_TARGET", "duckdb")
# Where training reads features from: "lakehouse" (Delta gold table) or "dbt" (warehouse mart).
FEATURE_SOURCE = os.getenv("TG_FEATURE_SOURCE", "lakehouse")
DUCKDB_PATH = ROOT / os.getenv("DUCKDB_PATH", "data/warehouse.duckdb")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLS = ["unit_id", "cycle", *OP_COLS, *SENSOR_COLS]

# Sensors that carry degradation signal in FD001/FD003 (the rest are near-constant).
INFORMATIVE_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

# Piecewise-linear RUL target: early life is treated as "healthy" at a capped value.
RUL_CAP = 125
