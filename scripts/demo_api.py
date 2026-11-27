"""Call the RUL API with real FD001 test engines and compare against NASA's true RUL.

Usage:  python scripts/demo_api.py <api_url> [engine ...]     (default engines: 31 34 50 81)
Stdlib only, so it runs anywhere.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cmapss"
SENSORS = json.loads((ROOT / "artifacts/lstm/preprocess.json").read_text())["sensors"]


def engine_readings(unit: int) -> list[dict[str, float]]:
    rows = [line.split() for line in (RAW / "test_FD001.txt").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if int(r[0]) == unit]
    # columns: unit, cycle, 3 op settings, sensor_1..sensor_21
    return [{s: float(r[4 + int(s.split("_")[1])]) for s in SENSORS} for r in rows]


def post(url: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r), (time.perf_counter() - t) * 1000


def main() -> None:
    base = sys.argv[1].rstrip("/")
    units = [int(u) for u in sys.argv[2:]] or [31, 34, 50, 81]
    truth = [int(x) for x in (RAW / "RUL_FD001.txt").read_text().split()]
    with urllib.request.urlopen(f"{base}/health", timeout=30) as r:
        print("health:", json.load(r))
    print(f"{'engine':<16}{'predicted':>10}{'true':>7}  {'status':<10}{'latency':>9}")
    for u in units:
        res, ms = post(f"{base}/predict", {"asset_id": f"FD001_test_{u:03d}", "readings": engine_readings(u)})
        print(f"{res['asset_id']:<16}{res['rul_cycles']:>10.1f}{min(truth[u - 1], 125):>7}  {res['status']:<10}{ms:>7.0f}ms")


if __name__ == "__main__":
    main()
