"""End-to-end pipeline with a quality gate: ingest -> dbt build (models + tests) -> train.

If any dbt data test fails, the batch is rejected and training does NOT run.
Usage:  python scripts/run_pipeline.py [--target duckdb|snowflake] [--skip-ingest] [--skip-train]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from thermalguard import config


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if subprocess.call(cmd, cwd=config.ROOT) != 0:
        sys.exit(f"step failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=config.TARGET, choices=["duckdb", "snowflake"])
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    os.environ["TG_TARGET"] = args.target
    py = sys.executable

    if not args.skip_ingest:
        run([py, "scripts/ingest.py", "--target", args.target])

    dbt = [str(Path(py).parent / "dbt")]  # dbt executable from the active venv
    print("\n$ dbt build (quality gate)", flush=True)
    rc = subprocess.call(dbt + ["build", "--project-dir", "dbt", "--profiles-dir", "dbt",
                                "--target", args.target], cwd=config.ROOT)
    if rc != 0:
        sys.exit("QUALITY GATE FAILED: dbt tests rejected this batch, so training is blocked.")

    if not args.skip_train:
        run([py, "-m", "thermalguard.train_classic"])


if __name__ == "__main__":
    main()
