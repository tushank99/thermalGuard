"""Thin warehouse abstraction so the same code reads/writes DuckDB (dev) or Snowflake (showcase)."""
from __future__ import annotations

import os

import pandas as pd

from thermalguard import config


def _snowflake_conn():
    import snowflake.connector  # optional dependency: requirements-snowflake.txt

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "THERMALGUARD"),
    )


def write_table(df: pd.DataFrame, schema: str, table: str, target: str | None = None) -> None:
    target = target or config.TARGET
    if target == "duckdb":
        import duckdb

        config.DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(config.DUCKDB_PATH)) as con:
            con.execute(f"create schema if not exists {schema}")
            con.register("df_view", df)
            con.execute(f"create or replace table {schema}.{table} as select * from df_view")
    elif target == "snowflake":
        from snowflake.connector.pandas_tools import write_pandas

        with _snowflake_conn() as con:
            con.cursor().execute(f"create schema if not exists {schema.upper()}")
            df = df.copy()
            df.columns = [c.upper() for c in df.columns]
            write_pandas(con, df, table.upper(), schema=schema.upper(),
                         auto_create_table=True, overwrite=True)
    else:
        raise ValueError(f"unknown target {target}")


def read_sql(sql: str, target: str | None = None) -> pd.DataFrame:
    target = target or config.TARGET
    if target == "duckdb":
        import duckdb

        with duckdb.connect(str(config.DUCKDB_PATH), read_only=True) as con:
            return con.execute(sql).df()
    with _snowflake_conn() as con:
        df = con.cursor().execute(sql).fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df
