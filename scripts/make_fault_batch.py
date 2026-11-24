"""Create a SYNTHETIC corrupted batch to exercise the silver-layer quarantine.

The NASA files are clean, so without this the quarantine table would always be empty. The batch simulates
real field failures: sensor dropouts (missing tokens), garbage values, negative readings, spikes and
duplicate re-sends. Output: data/raw/injected/train_FD001_injected.txt (fixed seed, reproducible).
"""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/cmapss/train_FD001.txt"
OUT = ROOT / "data/raw/injected/train_FD001_injected.txt"


def main(n: int = 400, seed: int = 7) -> None:
    rng = random.Random(seed)
    lines = SRC.read_text().splitlines()
    out = []
    for i, line in enumerate(rng.sample(lines, n)):
        tok = line.split()
        kind = i % 5
        if kind == 0:   # sensor dropout -> fewer columns
            del tok[rng.randrange(5, len(tok))]
        elif kind == 1:  # garbage token from a flaky gateway
            tok[rng.randrange(5, len(tok))] = "ERR#"
        elif kind == 2:  # negative reading (impossible physically)
            tok[rng.randrange(5, len(tok))] = "-12.5"
        elif kind == 3:  # spike
            tok[rng.randrange(5, len(tok))] = "99999.0"
        # kind == 4: exact duplicate re-send of a valid row
        out.append(" ".join(tok))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {len(out)} synthetic rows -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
