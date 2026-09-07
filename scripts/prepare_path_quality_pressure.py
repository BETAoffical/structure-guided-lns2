"""Generate and audit pressure Pilot tasks only; no reset or timing entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lns2_selector.evaluation.path_quality_preflight import contained
from lns2_selector.evaluation.path_quality_pressure import prepare_pressure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/path_quality_pressure_pilot_v1.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    report = prepare_pressure(ROOT, contained(ROOT, args.config), verify=args.verify)
    print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "output_sha256"}}, indent=2))
    return int(report["generation_errors"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
