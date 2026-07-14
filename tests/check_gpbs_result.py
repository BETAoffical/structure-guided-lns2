from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f"GPBS statistics file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("GPBS statistics file has no result rows")
    cost = int(float(rows[-1]["solution cost"]))
    if cost < 0:
        raise SystemExit(f"GPBS smoke did not find a solution: cost={cost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
