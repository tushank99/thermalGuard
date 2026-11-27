"""Data-drift monitor (Evidently): compare a new batch of sensor data against the training reference.

reference = half of the FD001 run-to-failure engines (split by engine)
batches   = FD001_new: the other half (same fleet and regime: expected stable)
            FD002: run-to-failure engines under six operating regimes (expected drift)
Writes reports/drift.json and reports/drift_<batch>.html. The retrain workflow retrains when drift is detected.
Usage:  python -m thermalguard.drift [--batch FD001_new|FD002]   (default: both)
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from thermalguard import config
from thermalguard.data import load_features, train_val_split

DRIFT_SHARE = 0.5  # dataset drift = at least half of the monitored sensors drifted
# Normed Wasserstein threshold, calibrated on a same-fleet holdout: healthy engine-to-engine variation scores up to
# ~0.16 (Evidently's generic 0.1 default flags it), a new operating regime scores 4+. 0.3 = ~2x the healthy max.
THRESHOLD = 0.3
BATCHES = ["FD001_new", "FD002"]
COLS = [f"sensor_{i}" for i in config.INFORMATIVE_SENSORS]


def check(reference: pd.DataFrame, current: pd.DataFrame, cols: list[str], html_path) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    preset = DataDriftPreset(drift_share=DRIFT_SHARE, num_method="wasserstein", num_threshold=THRESHOLD)
    snap = Report([preset]).run(current_data=current[cols],
                                                                   reference_data=reference[cols])
    snap.save_html(str(html_path))
    out = {"rows_reference": len(reference), "rows_current": len(current), "columns": len(cols), "drifted": []}
    for m in snap.dict()["metrics"]:
        cfg, val = m["config"], m["value"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            out["drifted_columns"], out["drifted_share"] = int(val["count"]), round(float(val["share"]), 3)
        elif cfg["type"].endswith("ValueDrift"):
            out.setdefault("tests", {})[cfg["column"]] = {"method": cfg["method"], "score": round(float(val), 4),
                                                         "threshold": cfg["threshold"]}
    out["drift_detected"] = out["drifted_share"] >= DRIFT_SHARE
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=BATCHES, action="append")
    args = ap.parse_args()

    df = load_features(["FD001", "FD002"])
    df = df[df.split == "train"]  # full run-to-failure histories, so life stages are comparable
    ref, fd001_new = train_val_split(df[df.subset == "FD001"], val_frac=0.5)
    current = {"FD001_new": fd001_new, "FD002": df[df.subset == "FD002"]}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    results = {}
    for name in args.batch or BATCHES:
        results[name] = check(ref, current[name], COLS, config.REPORTS_DIR / f"drift_{name}.html")
        r = results[name]
        print(f"{name}: {r['drifted_columns']}/{r['columns']} columns drifted -> drift_detected={r['drift_detected']}")

    (config.REPORTS_DIR / "drift.json").write_text(json.dumps(
        {"reference": f"FD001 run-to-failure, {ref.asset_id.nunique()} engines", "method": f"Evidently DataDriftPreset, normed Wasserstein per sensor (threshold {THRESHOLD}, "
         f"calibrated on a same-fleet holdout); dataset drift if >= {DRIFT_SHARE:.0%} of sensors drift", "batches": results}, indent=2))
    if gh := os.getenv("GITHUB_OUTPUT"):  # retrain workflow reads this
        with open(gh, "a") as f:
            f.write(f"drift_detected={str(any(r['drift_detected'] for r in results.values())).lower()}\n")


if __name__ == "__main__":
    main()
