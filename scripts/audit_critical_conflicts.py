from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.critical_conflict_audit import run_critical_conflict_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit critical-conflict seed retention without rerunning the solver."
    )
    parser.add_argument(
        "--source", default="build/initlns-v3-pilot-v1"
    )
    parser.add_argument(
        "--output", default="build/initlns-v2-critical-conflict-audit-v1"
    )
    arguments = parser.parse_args()
    report = run_critical_conflict_audit(arguments.source, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
