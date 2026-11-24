"""Shared lakehouse settings: table paths, Spark session, sensor lists (no thermalguard.config import,
so this runs inside the slim Spark container without python-dotenv)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_DIRS = [ROOT / "data/raw/cmapss", ROOT / "data/raw/injected"]
LAKE = ROOT / "data/lakehouse"


def table(layer: str, name: str) -> str:
    return str(LAKE / layer / name)


BRONZE_LINES = table("bronze", "cmapss_lines")
SILVER_CYCLES = table("silver", "cycles")
SILVER_RUL = table("silver", "rul_truth")
SILVER_QUARANTINE = table("silver", "quarantine")
GOLD_FEATURES = table("gold", "rul_features")

OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
INFORMATIVE = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
WINDOWS = (5, 10, 30)
RUL_CAP = 125

# Batch plan: two incremental loads so Delta time travel has something real to show.
BATCHES = {
    1: ["FD001", "FD002"],
    2: ["FD003", "FD004", "injected"],
}


def spark_session(app: str = "thermalguard-lakehouse", driver_memory: str = "4g"):
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
