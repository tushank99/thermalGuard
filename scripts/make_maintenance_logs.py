"""Generate SYNTHETIC, pseudonymised maintenance logs for FD001 test engines, consistent with their true RUL.

Output: docs/knowledge/maintenance_logs.jsonl (one entry per event). Technicians appear only as role
tokens (TECH_xx), in line with governance/gdpr_data_handling.md.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cmapss"
OUT = ROOT / "docs/knowledge/maintenance_logs.jsonl"

ROUTINE = [
    "Routine visual inspection completed. No findings.",
    "Oil sample taken; results within limits.",
    "Fan blades inspected; no FOD damage observed.",
]
EARLY = [
    "Slight upward trend in HPC outlet temperature (T30) noted on trend review; continue monitoring.",
    "Ps30 trending above fleet median; scheduled compressor wash.",
    "Compressor water wash performed; T30 margin partially recovered.",
]
LATE = [
    "Borescope HPC stages 3/5/7: moderate leading-edge erosion on stage 5; re-inspect in 25 cycles.",
    "ThermalGuard warning status acknowledged; HPC blade kit ordered.",
    "Borescope HPC: severe erosion stage 5 and 7, tip clearance increased; recommend removal.",
    "Engine flagged critical by ThermalGuard; removal planned at next maintenance window.",
]


def main(seed: int = 11) -> None:
    rng = random.Random(seed)
    test = pd.read_csv(RAW / "test_FD001.txt", sep=r"\s+", header=None, usecols=[0, 1], names=["unit", "cycle"])
    rul = pd.read_csv(RAW / "RUL_FD001.txt", header=None, names=["rul"])
    last = test.groupby("unit").cycle.max()
    start = date(2026, 1, 5)
    entries = []
    for unit in rng.sample(list(last.index), 60):
        asset = f"FD001_test_{unit:03d}"
        last_cycle, rul_now = int(last[unit]), int(rul.rul[unit - 1])
        for cyc in sorted(rng.sample(range(10, last_cycle + 1), k=min(4, last_cycle - 10))):
            rul_at = rul_now + (last_cycle - cyc)
            pool = LATE if rul_at < 50 else EARLY if rul_at < 110 else ROUTINE
            entries.append({
                "date": str(start + timedelta(days=cyc)),
                "asset_id": asset,
                "cycle": cyc,
                "technician": f"TECH_{rng.randint(1, 12):02d}",
                "note": rng.choice(pool),
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(e) for e in sorted(entries, key=lambda e: (e["asset_id"], e["cycle"]))) + "\n")
    print(f"wrote {len(entries)} synthetic log entries for 60 engines -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
