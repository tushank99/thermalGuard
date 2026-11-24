"""PySpark vs pandas on the SAME gold transform (RUL label + rolling mean/std over 5/10/30 for 14 sensors).

Both engines read the same Parquet input and fully materialise the result, in the same container.
Scales: 1x (real data) and 5x (fleet replicated with new asset ids).
  docker run --rm -v "${PWD}:/app" thermalguard-spark python -m thermalguard.lakehouse.benchmark
"""
from __future__ import annotations

import json
import os
import shutil
import time

import pandas as pd
from pyspark.sql import functions as F

from thermalguard.lakehouse.common import (
    INFORMATIVE, LAKE, ROOT, RUL_CAP, SILVER_CYCLES, SILVER_RUL, WINDOWS, spark_session,
)
from thermalguard.lakehouse.pipeline import gold_transform

TMP = LAKE / "_bench"


def pandas_transform(cycles: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    df = cycles.sort_values(["asset_id", "cycle"]).reset_index(drop=True)
    df["last_cycle"] = df.groupby("asset_id")["cycle"].transform("max")
    df = df.merge(truth, on="asset_id", how="left")
    df["rul"] = (df["last_cycle"] - df["cycle"]).where(df["split"] == "train",
                                                      df["rul_true"] + df["last_cycle"] - df["cycle"])
    df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    df["is_last_cycle"] = df["cycle"] == df["last_cycle"]
    g = df.groupby("asset_id", sort=False)[INFORMATIVE]
    feats = {}
    for w in WINDOWS:
        r = g.rolling(w, min_periods=1)
        mean, std = r.mean().reset_index(level=0, drop=True), r.std().reset_index(level=0, drop=True).fillna(0.0)
        for s in INFORMATIVE:
            feats[f"{s}_mean_{w}"], feats[f"{s}_std_{w}"] = mean[s], std[s]
    return pd.concat([df, pd.DataFrame(feats)], axis=1)


def prepare(spark, factor: int) -> tuple[str, str]:
    cyc = spark.read.format("delta").load(SILVER_CYCLES)
    tru = spark.read.format("delta").load(SILVER_RUL)
    reps = spark.range(factor).withColumnRenamed("id", "rep")
    newid = F.concat("asset_id", F.lit("_r"), F.col("rep").cast("string"))
    cyc = cyc.crossJoin(reps).withColumn("asset_id", newid).drop("rep")
    tru = tru.crossJoin(reps).withColumn("asset_id", newid).drop("rep")
    cp, tp = str(TMP / f"cycles_{factor}x"), str(TMP / f"truth_{factor}x")
    cyc.write.mode("overwrite").parquet(cp)
    tru.write.mode("overwrite").parquet(tp)
    return cp, tp


def main() -> None:
    spark = spark_session("thermalguard-benchmark", driver_memory="2g")  # leave RAM for pandas
    results = []
    for factor in (1, 5):
        cp, tp = prepare(spark, factor)

        t0 = time.perf_counter()
        out = gold_transform(spark.read.parquet(cp), spark.read.parquet(tp))
        out.write.format("noop").mode("overwrite").save()  # forces full computation, no write cost
        spark_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pdf = pandas_transform(pd.read_parquet(cp), pd.read_parquet(tp))
        pandas_s = time.perf_counter() - t0

        rows = len(pdf)
        results.append({"scale": f"{factor}x", "rows": rows, "pyspark_s": round(spark_s, 2),
                        "pandas_s": round(pandas_s, 2), "speedup_spark_vs_pandas": round(pandas_s / spark_s, 2)})
        print(results[-1], flush=True)
        del pdf

    spark.stop()
    shutil.rmtree(TMP, ignore_errors=True)
    out = {"cpu_cores": os.cpu_count(), "transform": "RUL label + rolling mean/std (w=5,10,30) x 14 sensors",
           "results": results}
    (ROOT / "reports/benchmark_spark_vs_pandas.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
