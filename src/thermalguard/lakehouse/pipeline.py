"""Medallion lakehouse on PySpark + Delta Lake.

bronze  raw lines as-is + source_file, line_number, batch_id, ingest_ts      (append-only, 2 batches)
silver  parsed + typed with an explicit schema; bad rows -> silver.quarantine with a dq_reason
gold    RUL label + rolling mean/std over 5/10/30 cycles for 14 sensors     (partitioned by subset)

Run inside the Spark container:
  docker run --rm -v "${PWD}:/app" thermalguard-spark python -m thermalguard.lakehouse.pipeline
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from thermalguard.lakehouse.common import (
    BATCHES, BRONZE_LINES, GOLD_FEATURES, INFORMATIVE, LAKE, OP_COLS, RAW_DIRS, ROOT, RUL_CAP,
    SENSOR_COLS, SILVER_CYCLES, SILVER_QUARANTINE, SILVER_RUL, WINDOWS, spark_session,
)

BRONZE_SCHEMA = T.StructType([
    T.StructField("value", T.StringType(), False),
    T.StructField("line_number", T.LongType(), False),
    T.StructField("source_file", T.StringType(), False),
    T.StructField("batch_id", T.IntegerType(), False),
    T.StructField("ingest_ts", T.TimestampType(), False),
])

SILVER_SCHEMA = T.StructType(
    [T.StructField("asset_id", T.StringType(), False),
     T.StructField("subset", T.StringType(), False),
     T.StructField("split", T.StringType(), False),
     T.StructField("unit_id", T.IntegerType(), False),
     T.StructField("cycle", T.IntegerType(), False)]
    + [T.StructField(c, T.DoubleType(), False) for c in OP_COLS + SENSOR_COLS]
    + [T.StructField("source_file", T.StringType(), False),
       T.StructField("batch_id", T.IntegerType(), False),
       T.StructField("ingest_ts", T.TimestampType(), False)]
)


# --------------------------------------------------------------------------------------------- bronze
def _files_for(subsets: list[str]):
    for s in subsets:
        if s == "injected":
            yield from sorted(RAW_DIRS[1].glob("*.txt"))
        else:
            for kind in ("train", "test", "RUL"):
                yield RAW_DIRS[0] / f"{kind}_{s}.txt"


def bronze(spark: SparkSession, batch_id: int) -> int:
    """Ingest raw files untouched. Batch 1 creates the table, later batches append."""
    frames = []
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    for path in _files_for(BATCHES[batch_id]):
        rdd = spark.read.text(str(path)).rdd.zipWithIndex().map(
            lambda r, name=path.name: (r[0].value, r[1] + 1, name, batch_id, ts))
        frames.append(spark.createDataFrame(rdd, BRONZE_SCHEMA))
    df = frames[0]
    for f in frames[1:]:
        df = df.unionByName(f)
    df.write.format("delta").mode("overwrite" if batch_id == 1 else "append").save(BRONZE_LINES)
    n = df.count()
    print(f"bronze batch {batch_id}: +{n:,} lines from {len(frames)} files")
    return n


# --------------------------------------------------------------------------------------------- silver
def silver(spark: SparkSession) -> dict:
    b = (spark.read.format("delta").load(BRONZE_LINES)
         .withColumn("kind", F.regexp_extract("source_file", r"^(train|test|RUL)_", 1))
         .withColumn("subset", F.regexp_extract("source_file", r"(FD00\d)", 1)))

    # Ground-truth RUL files: one integer per line, line number = test unit id.
    rul = (b.filter(F.col("kind") == "RUL")
           .select(F.concat_ws("_", "subset", F.lit("test"), F.lpad(F.col("line_number").cast("string"), 3, "0"))
                   .alias("asset_id"),
                   F.trim("value").cast("int").alias("rul_true")))
    rul.write.format("delta").mode("overwrite").save(SILVER_RUL)

    tok = F.split(F.trim("value"), r"\s+")
    cols = ["unit_id", "cycle", *OP_COLS, *SENSOR_COLS]
    parsed = b.filter(F.col("kind").isin("train", "test")).withColumn("n_tok", F.size(tok))
    for i, c in enumerate(cols):
        parsed = parsed.withColumn(c, tok.getItem(i).cast("int" if i < 2 else "double"))
    parsed = (parsed.withColumnRenamed("kind", "split")
              .withColumn("asset_id", F.concat_ws("_", "subset", "split",
                                                  F.lpad(F.col("unit_id").cast("string"), 3, "0"))))

    any_null = F.lit(False)
    for c in cols:
        any_null = any_null | F.col(c).isNull()
    sensor_bad = F.lit(False)
    for c in SENSOR_COLS:
        sensor_bad = sensor_bad | (F.col(c) < 0) | (F.col(c) > 10000)

    checked = parsed.withColumn(
        "dq_reason",
        F.when(F.col("n_tok") != len(cols), "bad_token_count")
         .when(any_null, "unparseable_value")
         .when((F.col("unit_id") < 1) | (F.col("cycle") < 1), "invalid_key")
         .when(~F.col("op_setting_3").between(60, 100), "op_setting_out_of_range")
         .when(sensor_bad, "sensor_out_of_range"))

    # Among valid rows, keep the first arrival of each (asset, cycle); later copies are duplicates.
    w = Window.partitionBy("asset_id", "cycle").orderBy("batch_id", "source_file", "line_number")
    checked = checked.withColumn(
        "dq_reason",
        F.when(F.col("dq_reason").isNull() & (F.row_number().over(w) > 1), "duplicate_key")
         .otherwise(F.col("dq_reason")))
    checked.cache()

    good = checked.filter(F.col("dq_reason").isNull()).select(*[f.name for f in SILVER_SCHEMA.fields])
    spark.createDataFrame(good.rdd, SILVER_SCHEMA).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").save(SILVER_CYCLES)

    bad = checked.filter(F.col("dq_reason").isNotNull()).select(
        "value", "source_file", "line_number", "batch_id", "dq_reason", "ingest_ts",
        F.current_timestamp().alias("quarantined_at"))
    bad.write.format("delta").mode("overwrite").save(SILVER_QUARANTINE)

    reasons = {r["dq_reason"]: r["count"] for r in bad.groupBy("dq_reason").count().collect()}
    checked.unpersist()
    print(f"silver: quarantined {sum(reasons.values())} rows {reasons}")
    return reasons


# ----------------------------------------------------------------------------------------------- gold
def gold_transform(cycles: DataFrame, truth: DataFrame) -> DataFrame:
    """Shared by the pipeline and the PySpark-vs-pandas benchmark."""
    by_asset = Window.partitionBy("asset_id")
    df = (cycles.withColumn("last_cycle", F.max("cycle").over(by_asset))
          .join(truth, "asset_id", "left")
          .withColumn("rul", F.when(F.col("split") == "train", F.col("last_cycle") - F.col("cycle"))
                      .otherwise(F.col("rul_true") + F.col("last_cycle") - F.col("cycle")))
          .withColumn("rul_capped", F.least(F.col("rul"), F.lit(RUL_CAP)))
          .withColumn("is_last_cycle", F.col("cycle") == F.col("last_cycle")))
    feats = []
    for wsize in WINDOWS:
        win = Window.partitionBy("asset_id").orderBy("cycle").rowsBetween(-(wsize - 1), 0)
        for s in INFORMATIVE:
            feats.append(F.avg(s).over(win).alias(f"{s}_mean_{wsize}"))
            feats.append(F.coalesce(F.stddev_samp(s).over(win), F.lit(0.0)).alias(f"{s}_std_{wsize}"))
    keep = ["asset_id", "subset", "split", "unit_id", "cycle", "is_last_cycle", "rul", "rul_capped",
            *OP_COLS, *INFORMATIVE]
    return df.select(*keep, *feats)


def gold(spark: SparkSession) -> int:
    cycles = spark.read.format("delta").load(SILVER_CYCLES)
    truth = spark.read.format("delta").load(SILVER_RUL)
    g = gold_transform(cycles, truth)
    g.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .partitionBy("subset").save(GOLD_FEATURES)
    n = spark.read.format("delta").load(GOLD_FEATURES).count()
    print(f"gold: {n:,} feature rows, {len(g.columns)} columns")
    return n


# ------------------------------------------------------------------------------------- delta features
def demo_schema_enforcement(spark: SparkSession) -> dict:
    """Try to append a row with a wrong type + an unexpected column. Delta must reject it."""
    rogue = spark.read.format("delta").load(SILVER_CYCLES).limit(1) \
        .withColumn("sensor_2", F.lit("not-a-number")).withColumn("rogue_col", F.lit(1))
    try:
        rogue.write.format("delta").mode("append").save(SILVER_CYCLES)
        return {"rejected": False}
    except Exception as e:  # AnalysisException
        msg = str(e).strip().splitlines()[0][:300]
        print(f"schema enforcement: rejected bad append -> {type(e).__name__}")
        return {"rejected": True, "error_type": type(e).__name__, "message": msg}


def demo_time_travel(spark: SparkSession) -> dict:
    from delta.tables import DeltaTable

    v0 = spark.read.format("delta").option("versionAsOf", 0).load(BRONZE_LINES).count()
    latest = spark.read.format("delta").load(BRONZE_LINES).count()
    hist = [
        {"version": r["version"], "operation": r["operation"],
         "rows_written": (r["operationMetrics"] or {}).get("numOutputRows")}
        for r in DeltaTable.forPath(spark, BRONZE_LINES).history()
        .select("version", "operation", "operationMetrics").orderBy("version").collect()
    ]
    print(f"time travel: bronze v0={v0:,} rows, latest={latest:,} rows")
    return {"bronze_rows_version_0": v0, "bronze_rows_latest": latest, "bronze_history": hist}


# ------------------------------------------------------------------------------------------- report
def write_report(spark: SparkSession, reasons: dict, schema_demo: dict, tt: dict) -> dict:
    count = lambda p: spark.read.format("delta").load(p).count()  # noqa: E731
    gold_df = spark.read.format("delta").load(GOLD_FEATURES)
    rep = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {
            "bronze_lines": count(BRONZE_LINES),
            "silver_cycles": count(SILVER_CYCLES),
            "silver_rul_truth": count(SILVER_RUL),
            "silver_quarantine": count(SILVER_QUARANTINE),
            "gold_features": gold_df.count(),
        },
        "assets": gold_df.select("asset_id").distinct().count(),
        "gold_columns": len(gold_df.columns),
        "quarantine_by_reason": reasons,
        "schema_enforcement": schema_demo,
        "time_travel": tt,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/lakehouse_quality.json").write_text(json.dumps(rep, indent=2, default=str))

    r = rep["rows"]
    md = [
        "# Data-Quality Report: Lakehouse",
        f"_Generated {rep['generated_at']} by `thermalguard.lakehouse.pipeline`._\n",
        "| Layer | Table | Rows |", "|---|---|---|",
        f"| Bronze | `bronze/cmapss_lines` | {r['bronze_lines']:,} |",
        f"| Silver | `silver/cycles` | {r['silver_cycles']:,} |",
        f"| Silver | `silver/rul_truth` | {r['silver_rul_truth']:,} |",
        f"| Silver | `silver/quarantine` | {r['silver_quarantine']:,} |",
        f"| Gold | `gold/rul_features` ({rep['gold_columns']} cols, {rep['assets']:,} assets) | {r['gold_features']:,} |",
        "",
        "Reconciliation: bronze lines = silver cycles + silver RUL rows + quarantined rows → "
        f"{r['bronze_lines']:,} = {r['silver_cycles'] + r['silver_rul_truth'] + r['silver_quarantine']:,}",
        "",
        "## Quarantined rows by reason",
        "All quarantined rows come from the **synthetic fault-injection batch** "
        "(`scripts/make_fault_batch.py`). The NASA source files themselves are clean.\n",
        "| Reason | Rows |", "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in sorted(reasons.items())],
        "",
        "## Delta Lake features",
        f"- **Schema enforcement:** appending a row with `sensor_2` as a string plus an extra column was "
        f"{'**rejected**' if schema_demo.get('rejected') else 'NOT rejected'} "
        f"(`{schema_demo.get('error_type', '')}`).",
        f"- **Time travel:** `bronze/cmapss_lines` version 0 has {tt['bronze_rows_version_0']:,} rows; the latest "
        f"version has {tt['bronze_rows_latest']:,} (after the batch-2 append).",
        "",
        "| Version | Operation | Rows written |", "|---|---|---|",
        *[f"| {h['version']} | {h['operation']} | {h['rows_written']} |" for h in tt["bronze_history"]],
    ]
    (ROOT / "governance/data_quality_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rep


def main() -> None:
    shutil.rmtree(LAKE, ignore_errors=True)  # full, reproducible rebuild
    spark = spark_session()
    for batch_id in sorted(BATCHES):
        bronze(spark, batch_id)
    reasons = silver(spark)
    gold(spark)
    rep = write_report(spark, reasons, demo_schema_enforcement(spark), demo_time_travel(spark))
    print(json.dumps(rep["rows"], indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
